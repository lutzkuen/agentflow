from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


SCHEMA = "agentflow.instruction_dedup_impact.v1"
ROLLBACK_ACTION_SCHEMA = "agentflow.instruction_dedup_rollback_action.v1"
TOKEN_CHARS = 4


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


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_instruction_text_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "terminal_output_included": False,
        "raw_terminal_text_included": False,
        "tool_payloads_included": False,
        "raw_tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "tenant_ids_included": False,
        "policy_file_contents_included": False,
        "instruction_section_fingerprints_included": False,
        "secrets_included": False,
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
        "saved_chars": 0,
        "saved_tokens_est": 0,
        "projected_saved_usd": 0.0,
        "net_savings_usd": 0.0,
        "negative_net_savings_count": 0,
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
        "saved_chars": _as_int(raw.get("saved_chars")),
        "saved_tokens_est": _as_int(raw.get("saved_tokens_est")),
        "projected_saved_usd": round(_as_float(raw.get("projected_saved_usd")), 8),
        "net_savings_usd": round(_as_float(raw.get("net_savings_usd")), 8),
        "net_savings_per_call_usd": round(_as_float(raw.get("net_savings_usd")) / count, 8) if count else 0.0,
        "negative_net_savings_count": _as_int(raw.get("negative_net_savings_count")),
        "negative_net_savings_rate": round(_as_int(raw.get("negative_net_savings_count")) / count, 6) if count else 0.0,
        "status_breakdown": _counter_rows(status_counts),
        "reason_breakdown": _counter_rows(reason_counts),
    }


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    where = "where created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since, capped) if since else (capped,)
    rows = store_obj.conn.execute(
        f"""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint, requested_model, routed_model,
               requested_model_family, routed_model_family, stream, status_code,
               latency_ms, cost_est_usd, cost_baseline_usd, category, retry_count,
               actual_input_tokens, input_tokens_est, crunch_json, routing_json
        from calls
        {where}
        order by created_at desc
        limit ?
        """,
        params,
    ).fetchall()
    return list(reversed([dict(row) for row in rows]))


def _dedup_meta(row: dict[str, Any]) -> dict[str, Any]:
    crunch = _json_obj(row.get("crunch_json"))
    meta = crunch.get("instruction_section_deduplication")
    return meta if isinstance(meta, dict) else {}


def _cohort(meta: dict[str, Any]) -> str:
    status = str(meta.get("status") or "").strip().lower()
    reason = str(meta.get("reason") or "").strip().lower()
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    canary_status = str(canary.get("status") or canary.get("cohort") or "").strip().lower()
    if status == "applied" or bool(meta.get("applied")) or canary_status == "applied":
        return "applied"
    if status == "holdout" or canary_status == "holdout" or reason == "instruction-dedup-holdout":
        return "holdout"
    if status == "safety_stopped" or "safety-stop" in reason or meta.get("safety_stop"):
        return "safety_stop"
    if status in {"suppressed", "blocked"} or reason == "coordinator-conflict":
        return "blocked"
    return "skipped"


def _coordinator_status(meta: dict[str, Any]) -> str:
    compatibility = meta.get("coordinator_compatibility")
    if not isinstance(compatibility, dict):
        return "unknown"
    return public_label(compatibility.get("status") or "unknown", "unknown")


def _group_parts(row: dict[str, Any], routing: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    provider = str(meta.get("provider") or row.get("provider") or "anthropic").lower()
    source_surface = meta.get("source_surface") or row.get("source_surface") or routing.get("source_surface") or "unknown"
    requested_family = row.get("requested_model_family") or "unknown"
    routed_family = row.get("routed_model_family") or requested_family
    rule_id = public_label(meta.get("selected_rule_id") or "instruction-section-dedup-policy", "unknown")
    candidate = public_id(meta.get("candidate_id") or rule_id, prefix="instruction-dedup-candidate", fallback=rule_id)
    return {
        "provider": public_label(provider, "unknown"),
        "source_surface": public_label(source_surface, "unknown"),
        "endpoint": public_label(meta.get("endpoint") or row.get("endpoint") or "messages", "unknown"),
        "category": public_label(meta.get("category") or row.get("category") or routing.get("category") or "unknown", "unknown"),
        "workflow_phase": public_label(meta.get("workflow_phase") or routing.get("workflow_phase") or row.get("category") or "unknown", "unknown"),
        "requested_model_family": public_label(requested_family, "unknown"),
        "routed_model_family": public_label(routed_family, "unknown"),
        "stream": bool(_as_int(row.get("stream"))),
        "policy_source": public_label(meta.get("policy_source") or "unknown", "unknown"),
        "rule_id": rule_id,
        "candidate_id": candidate or rule_id,
    }


def _new_aggregate(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_key": "|".join(str(parts.get(key) or "unknown") for key in (
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
            "blocked": _empty_cohort(),
            "skipped": _empty_cohort(),
            "safety_stop": _empty_cohort(),
        },
        "coordinator_conflict_count": 0,
        "first_observed_at": None,
        "last_observed_at": None,
    }


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
    cost = _as_float(row.get("cost_est_usd"))
    baseline = _as_float(row.get("cost_baseline_usd"))
    projected = _as_float(meta.get("projected_saved_usd"))
    saved_chars = _as_int(meta.get("saved_chars"))
    saved_tokens = _as_int(meta.get("tokens_saved_est")) or max(0, saved_chars // TOKEN_CHARS)
    cohort["cost_est_usd"] += cost
    cohort["cost_baseline_usd"] += baseline
    cohort["status_counts"][_status_bucket(status_code)] += 1
    for reason in meta.get("reason_codes") or [meta.get("reason") or "unknown"]:
        cohort["reason_counts"][public_label(reason, "unknown")] += 1
    if cohort_name in {"applied", "holdout"}:
        cohort["saved_chars"] += saved_chars
        cohort["saved_tokens_est"] += saved_tokens
        cohort["projected_saved_usd"] += projected
    if cohort_name == "applied":
        net = projected
        if baseline > 0 or cost > 0:
            net = (baseline - cost) + projected
        cohort["net_savings_usd"] += net
        if net <= 0:
            cohort["negative_net_savings_count"] += 1
    if _coordinator_status(meta) == "conflict":
        aggregate["coordinator_conflict_count"] += 1

    observed_at = _parse_time(row.get("created_at"))
    if observed_at is not None:
        first = aggregate.get("first_observed_at")
        last = aggregate.get("last_observed_at")
        aggregate["first_observed_at"] = observed_at if first is None or observed_at < first else first
        aggregate["last_observed_at"] = observed_at if last is None or observed_at > last else last


def _delta(applied: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    latency_delta = None
    if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
        latency_delta = round(float(applied["latency_avg_ms"]) - float(holdout["latency_avg_ms"]), 2)
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
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return {
        "schema": ROLLBACK_ACTION_SCHEMA,
        "action_id": f"instruction-dedup-rollback:{digest}",
        "action_type": "rollback-local-instruction-section-dedup-canary",
        "review_only": True,
        "wrote_local_files": False,
        "target_policy_family": "crunch",
        "target_rule_id": candidate.get("rule_id"),
        "target_candidate_id": candidate.get("candidate_id"),
        "recommended_local_policy_patch": {
            "instruction_section_deduplication": {
                "enabled": True,
                "canary": {"fraction": 0.0, "holdout_fraction": 1.0},
            }
        },
        "reason_codes": reason_codes,
        "trigger_metrics": candidate.get("deltas", {}),
        "privacy": _privacy_summary(),
    }


def _next_action(candidate: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    applied = candidate["cohorts"]["applied"]
    holdout = candidate["cohorts"]["holdout"]
    safety = candidate["cohorts"]["safety_stop"]
    deltas = candidate["deltas"]
    if safety["count"] > 0:
        return "rollback", ["safety-stop-observed"], [_rollback_action(candidate, ["safety-stop-observed"])]
    if applied["count"] < thresholds["min_applied_samples"]:
        return "collect_more_samples", ["insufficient-applied-samples"], []
    if holdout["count"] < thresholds["min_holdout_samples"]:
        return "collect_more_samples", ["insufficient-holdout-samples"], []

    rollback_reasons: list[str] = []
    if applied["error_rate"] >= thresholds["rollback_error_rate"]:
        rollback_reasons.append("rollback-absolute-error-rate")
    if deltas["error_rate_delta"] >= thresholds["max_error_rate_delta"]:
        rollback_reasons.append("rollback-error-rate-delta")
    if deltas["retry_rate_delta"] >= thresholds["max_retry_rate_delta"]:
        rollback_reasons.append("rollback-retry-rate-delta")
    if rollback_reasons:
        return "rollback", rollback_reasons, [_rollback_action(candidate, rollback_reasons)]

    hold_reasons: list[str] = []
    if applied["error_rate"] > thresholds["max_error_rate"]:
        hold_reasons.append("hold-applied-error-rate")
    latency_delta = deltas.get("latency_avg_ms_delta")
    if latency_delta is not None and latency_delta > thresholds["max_latency_regression_ms"]:
        hold_reasons.append("hold-latency-regression")
    if applied["net_savings_usd"] < thresholds["min_net_savings_usd"]:
        hold_reasons.append("hold-minimum-net-savings")
    if applied["negative_net_savings_rate"] > thresholds["max_negative_net_savings_rate"]:
        hold_reasons.append("hold-negative-net-savings")
    if hold_reasons:
        return "hold", hold_reasons, []
    return "widen", ["impact-positive"], []


def _finalize_candidate(aggregate: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    cohorts = {name: _finalize_cohort(raw) for name, raw in aggregate["cohorts"].items()}
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
        "coordinator_conflict_count": _as_int(aggregate.get("coordinator_conflict_count")),
        "cohorts": cohorts,
        "deltas": _delta(cohorts["applied"], cohorts["holdout"]),
        "net_savings_usd": cohorts["applied"]["net_savings_usd"],
        "projected_holdout_savings_usd": cohorts["holdout"]["projected_saved_usd"],
        "privacy": _privacy_summary(),
    }
    next_action, reason_codes, rollback_actions = _next_action(candidate, thresholds)
    candidate["next_action"] = next_action
    candidate["reason_codes"] = reason_codes
    candidate["rollback_actions"] = rollback_actions
    return candidate


def build_instruction_dedup_impact_report(
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
    max_negative_net_savings_rate: float = 0.0,
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
        "max_negative_net_savings_rate": round(float(max_negative_net_savings_rate), 6),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
    }
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    aggregates: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    cohort_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    coordinator_counts: Counter[str] = Counter()

    for row in rows:
        meta = _dedup_meta(row)
        if not meta:
            continue
        routing = _json_obj(row.get("routing_json"))
        parts = _group_parts(row, routing, meta)
        group_key = "|".join(str(parts.get(key) or "unknown") for key in (
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
        cohort_name = _cohort(meta)
        _add_row(aggregate, row, meta, cohort_name)
        observed_rows += 1
        cohort_counts[cohort_name] += 1
        status_counts[_status_bucket(row.get("status_code"))] += 1
        coordinator_counts[_coordinator_status(meta)] += 1

    candidates = [_finalize_candidate(aggregate, thresholds) for aggregate in aggregates.values()]
    candidates.sort(
        key=lambda item: (
            {"rollback": 0, "hold": 1, "collect_more_samples": 2, "widen": 3}.get(str(item.get("next_action")), 4),
            str(item.get("candidate_id") or ""),
        )
    )
    action_counts = Counter(str(item.get("next_action") or "unknown") for item in candidates)
    reason_counts: Counter[str] = Counter()
    rollback_actions: list[dict[str, Any]] = []
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        rollback_actions.extend(action for action in item.get("rollback_actions") or [] if isinstance(action, dict))

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
        "status": "matched" if observed_rows else "no-instruction-dedup-canary-metadata",
        "summary": {
            "sampled_call_count": len(rows),
            "observed_instruction_dedup_metadata_row_count": observed_rows,
            "candidate_group_count": len(candidates),
            "applied_count": cohort_counts.get("applied", 0),
            "holdout_count": cohort_counts.get("holdout", 0),
            "blocked_count": cohort_counts.get("blocked", 0),
            "skipped_count": cohort_counts.get("skipped", 0),
            "safety_stop_count": cohort_counts.get("safety_stop", 0),
            "saved_chars": sum(_as_int(item["cohorts"]["applied"].get("saved_chars")) for item in candidates),
            "saved_tokens_est": sum(_as_int(item["cohorts"]["applied"].get("saved_tokens_est")) for item in candidates),
            "projected_saved_usd": round(sum(_as_float(item["cohorts"]["applied"].get("projected_saved_usd")) for item in candidates), 8),
            "net_savings_usd": round(sum(_as_float(item.get("net_savings_usd")) for item in candidates), 8),
            "projected_holdout_savings_usd": round(sum(_as_float(item.get("projected_holdout_savings_usd")) for item in candidates), 8),
            "coordinator_conflict_count": sum(_as_int(item.get("coordinator_conflict_count")) for item in candidates),
            "rollback_action_count": len(rollback_actions),
            "next_action_counts": _counter_rows(action_counts),
            "reason_code_counts": _counter_rows(reason_counts),
            "status_breakdown": _counter_rows(status_counts),
            "cohort_breakdown": _counter_rows(cohort_counts),
            "coordinator_breakdown": _counter_rows(coordinator_counts),
        },
        "dashboard_rows": [
            {
                "candidate_id": item["candidate_id"],
                "rule_id": item["rule_id"],
                "source_surface": item["source_surface"],
                "category": item["category"],
                "workflow_phase": item["workflow_phase"],
                "requested_model_family": item["requested_model_family"],
                "applied_count": item["cohorts"]["applied"]["count"],
                "holdout_count": item["cohorts"]["holdout"]["count"],
                "saved_tokens_est": item["cohorts"]["applied"]["saved_tokens_est"],
                "net_savings_usd": item["net_savings_usd"],
                "error_rate_delta": item["deltas"]["error_rate_delta"],
                "retry_rate_delta": item["deltas"]["retry_rate_delta"],
                "next_action": item["next_action"],
                "reason_codes": item["reason_codes"],
            }
            for item in candidates
        ],
        "rollback_actions": rollback_actions,
        "candidates": candidates,
        "privacy": _privacy_summary(),
    }
