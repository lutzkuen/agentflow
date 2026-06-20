from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.limiter import model_tier
from tokenclaw.repeated_scaffold_impact import (
    TOKEN_CHARS,
    _as_float,
    _as_int,
    _counter_rows,
    _json_obj,
    _privacy_summary,
    _reason_code,
)
from tokenclaw.repeated_scaffold_feedback import SOURCE_SURFACE as LIFECYCLE_SOURCE_SURFACE
from tokenclaw.store import utc_now


SCHEMA = "agentflow.repeated_scaffold_activation.v1"
ANTHROPIC_SOURCE_SURFACE = "anthropic_messages"
FEEDBACK_SOURCE_SURFACES = {ANTHROPIC_SOURCE_SURFACE, LIFECYCLE_SOURCE_SURFACE}


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    where_since = "and created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since, capped) if since else (capped,)
    rows = store_obj.conn.execute(
        f"""
        select created_at, path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint, requested_model, routed_model,
               requested_model_family, routed_model_family, stream, status_code,
               latency_ms, input_tokens_est, actual_input_tokens,
               cache_read_input_tokens, retry_count, category, routing_json, crunch_json
        from calls
        where (coalesce(provider, 'anthropic') = 'anthropic' or path = '/v1/messages')
          {where_since}
        order by created_at desc
        limit ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _feedback_rows(store_obj: Any) -> list[dict[str, Any]]:
    if not hasattr(store_obj, "managed_outcome_feedback_rows"):
        return []
    try:
        rows = store_obj.managed_outcome_feedback_rows(limit=10_000)
    except Exception:
        return []
    return [
        row
        for row in rows
        if str(row.get("source_surface") or "") in FEEDBACK_SOURCE_SURFACES
    ]


def _safe_surface(row: dict[str, Any], routing: dict[str, Any], managed: dict[str, Any]) -> str:
    return _reason_code(
        row.get("source_surface")
        or managed.get("source_surface")
        or routing.get("source_surface")
        or ANTHROPIC_SOURCE_SURFACE,
        ANTHROPIC_SOURCE_SURFACE,
    )


def _workflow_phase(routing: dict[str, Any]) -> str:
    phase = routing.get("workflow_phase")
    if not phase and isinstance(routing.get("phase"), str):
        phase = routing.get("phase")
    return _reason_code(phase, "unknown")


def _provider_meta(crunch: dict[str, Any]) -> dict[str, Any]:
    meta = crunch.get("repeated_provider_scaffolding")
    return meta if isinstance(meta, dict) else {}


def _managed_repeated_provider_scaffolding(managed: dict[str, Any]) -> dict[str, Any]:
    crunch = managed.get("crunch")
    if not isinstance(crunch, dict):
        return {}
    repeated = crunch.get("repeated_provider_scaffolding")
    return repeated if isinstance(repeated, dict) else {}


def _cohort(provider_meta: dict[str, Any]) -> str:
    for rule in provider_meta.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        canary = rule.get("canary")
        if isinstance(canary, dict):
            value = canary.get("cohort") or canary.get("status")
            if isinstance(value, str) and value:
                return _reason_code(value, "unknown")
    canary = provider_meta.get("canary")
    if isinstance(canary, dict):
        value = canary.get("cohort") or canary.get("status")
        if isinstance(value, str) and value:
            return _reason_code(value, "unknown")
    reason = str(provider_meta.get("reason") or "").lower()
    if reason == "canary_holdout":
        return "canary-holdout"
    if _as_int(provider_meta.get("applied_count")) > 0 or provider_meta.get("status") == "applied":
        return "canary-applied"
    return "none"


def _safety_stop_reason(provider_meta: dict[str, Any]) -> str | None:
    status = str(provider_meta.get("status") or "").lower()
    reason = str(provider_meta.get("reason") or "").lower()
    if "safety" in status or "safety" in reason:
        return _reason_code(reason or status, "safety-stop")
    for rule in provider_meta.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        for skip in rule.get("skip_reasons") or []:
            if not isinstance(skip, dict):
                continue
            skip_reason = str(skip.get("reason") or "").lower()
            if "safety" in skip_reason:
                return _reason_code(skip_reason, "safety-stop")
    return None


def _is_applied(provider_meta: dict[str, Any], crunch: dict[str, Any]) -> bool:
    if not provider_meta:
        return False
    return (
        provider_meta.get("status") == "applied"
        or bool(crunch.get("changed"))
        or _as_int(provider_meta.get("applied_count")) > 0
        or _as_int(provider_meta.get("tokens_saved_est")) > 0
        or _as_int(provider_meta.get("saved_chars")) > 0
    )


def _is_holdout(provider_meta: dict[str, Any]) -> bool:
    if _cohort(provider_meta) in {"canary-holdout", "holdout"}:
        return True
    reason = str(provider_meta.get("reason") or "").lower()
    status = str(provider_meta.get("status") or "").lower()
    return reason == "canary_holdout" or "holdout" in reason or status == "holdout"


def _activation_state(managed: dict[str, Any], provider_meta: dict[str, Any], crunch: dict[str, Any]) -> str:
    if not managed:
        return "historical-no-managed-policy-metadata"
    status = _reason_code(managed.get("status"), "missing")
    reason = _reason_code(managed.get("reason"), "unknown")
    if managed.get("enabled") is False or (status == "skipped" and reason in {"disabled", "recommendations-disabled"}):
        return "preflight-disabled"
    if status == "error":
        if reason in {"timeout", "unreachable", "request-failed"}:
            return "server-unreachable"
        return "server-error"
    if status == "invalid":
        return "server-invalid-response"
    safety_reason = _safety_stop_reason(provider_meta)
    if safety_reason:
        return "safety-stopped"
    repeated_recommended = bool(_managed_repeated_provider_scaffolding(managed))
    if _is_applied(provider_meta, crunch) and repeated_recommended:
        return "applied-repeated-scaffold-profile"
    if _is_applied(provider_meta, crunch):
        return "local-repeated-scaffold-profile-applied"
    if _is_holdout(provider_meta):
        return "recommended-holdout" if repeated_recommended else "local-repeated-scaffold-holdout"
    if repeated_recommended:
        if provider_meta:
            return "recommended-not-applied"
        return "recommended-missing-local-crunch-metadata"
    if status == "received":
        return "baseline-no-repeated-scaffold-policy"
    if status == "skipped":
        return f"skipped-{reason}"
    return status


def _feedback_status(managed: dict[str, Any]) -> str:
    feedback = managed.get("outcome_feedback")
    if not isinstance(feedback, dict) or not feedback:
        return "missing"
    return _reason_code(feedback.get("status"), "unknown")


def _feedback_reason(managed: dict[str, Any]) -> str:
    feedback = managed.get("outcome_feedback")
    if not isinstance(feedback, dict) or not feedback:
        return "missing"
    return _reason_code(feedback.get("reason"), "unknown")


def _add_group(
    groups: dict[tuple[str, ...], dict[str, Any]],
    *,
    state: str,
    row: dict[str, Any],
    routing: dict[str, Any],
    managed: dict[str, Any],
    saved_chars: int,
    saved_tokens: int,
) -> None:
    provider = _reason_code(row.get("provider"), "anthropic")
    requested_model = str(row.get("requested_model") or "")
    routed_model = str(row.get("routed_model") or "")
    key = (
        state,
        _safe_surface(row, routing, managed),
        provider,
        _reason_code(model_tier(requested_model), "unknown"),
        _reason_code(model_tier(routed_model), "unknown"),
        _reason_code(row.get("category") or routing.get("category"), "unknown"),
        _workflow_phase(routing),
    )
    group = groups.setdefault(
        key,
        {
            "activation_state": key[0],
            "source_surface": key[1],
            "provider": key[2],
            "requested_model_tier": key[3],
            "routed_model_tier": key[4],
            "category": key[5],
            "workflow_phase": key[6],
            "call_count": 0,
            "error_count": 0,
            "estimated_saved_chars": 0,
            "estimated_saved_tokens": 0,
        },
    )
    group["call_count"] += 1
    if _as_int(row.get("status_code")) >= 400:
        group["error_count"] += 1
    group["estimated_saved_chars"] += saved_chars
    group["estimated_saved_tokens"] += saved_tokens


def _explanation(summary: dict[str, Any], state_counts: Counter[str]) -> str:
    if summary["applied_count"] and summary["repeated_scaffold_recommended_count"]:
        return "managed policy-decision activation is applying repeated-scaffold crunches locally"
    if summary["applied_count"]:
        return "local repeated-scaffold crunches are applying, but sampled managed policy decisions did not include a repeated-scaffold crunch recommendation"
    if state_counts.get("recommended-holdout"):
        return "managed repeated-scaffold recommendations are present but current sampled calls are in holdout"
    if state_counts.get("safety-stopped"):
        return "managed repeated-scaffold activation is safety-stopped by local canary evidence"
    if state_counts.get("baseline-no-repeated-scaffold-policy"):
        return "managed policy-decision calls are returning baseline decisions without a repeated-scaffold crunch profile"
    if state_counts.get("server-error") or state_counts.get("server-unreachable"):
        return "managed policy-decision activation is blocked by managed server errors or reachability failures"
    if state_counts.get("preflight-disabled"):
        return "managed policy-decision preflight is disabled, so local crunch policy remains authoritative"
    return "no repeated-scaffold policy-decision activation metadata was observed in the sampled Anthropic calls"


def build_repeated_scaffold_activation_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    feedback_rows = _feedback_rows(store_obj)

    state_counts: Counter[str] = Counter()
    managed_status_counts: Counter[str] = Counter()
    managed_reason_counts: Counter[str] = Counter()
    apply_reason_counts: Counter[str] = Counter()
    optimization_unit_counts: Counter[str] = Counter()
    crunch_status_counts: Counter[str] = Counter()
    crunch_reason_counts: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    safety_reason_counts: Counter[str] = Counter()
    call_feedback_status_counts: Counter[str] = Counter()
    call_feedback_reason_counts: Counter[str] = Counter()
    source_surface_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    requested_tier_counts: Counter[str] = Counter()
    routed_tier_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    queue_status_counts: Counter[str] = Counter()
    queue_source_surface_counts: Counter[str] = Counter()
    groups: dict[tuple[str, ...], dict[str, Any]] = {}

    managed_rows = 0
    recommended_count = 0
    applied_count = 0
    holdout_count = 0
    safety_stop_count = 0
    saved_chars_total = 0
    saved_tokens_total = 0

    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        managed = routing.get("managed_recommendation")
        managed = managed if isinstance(managed, dict) else {}
        provider_meta = _provider_meta(crunch)
        state = _activation_state(managed, provider_meta, crunch)
        state_counts[state] += 1
        if managed:
            managed_rows += 1
            managed_status_counts[_reason_code(managed.get("status"), "missing")] += 1
            managed_reason_counts[_reason_code(managed.get("reason"), "unknown")] += 1
            apply_reason_counts[_reason_code(managed.get("apply_reason"), "none")] += 1
            optimization_unit_counts["present" if managed.get("optimization_unit_id") is not None else "missing"] += 1
            call_feedback_status_counts[_feedback_status(managed)] += 1
            call_feedback_reason_counts[_feedback_reason(managed)] += 1
            if _managed_repeated_provider_scaffolding(managed):
                recommended_count += 1
        else:
            managed_status_counts["missing"] += 1
            managed_reason_counts["historical-null"] += 1
            apply_reason_counts["none"] += 1
            optimization_unit_counts["missing"] += 1
            call_feedback_status_counts["missing"] += 1
            call_feedback_reason_counts["missing"] += 1

        crunch_status_counts[_reason_code(provider_meta.get("status"), "missing")] += 1
        crunch_reason_counts[_reason_code(provider_meta.get("reason"), "missing")] += 1
        cohort = _cohort(provider_meta)
        cohort_counts[cohort] += 1
        safety_reason = _safety_stop_reason(provider_meta)
        if safety_reason:
            safety_reason_counts[safety_reason] += 1
            safety_stop_count += 1
        if _is_applied(provider_meta, crunch):
            applied_count += 1
        if _is_holdout(provider_meta):
            holdout_count += 1

        saved_chars = _as_int(provider_meta.get("saved_chars"))
        saved_tokens = _as_int(provider_meta.get("tokens_saved_est"))
        if saved_tokens <= 0 and saved_chars > 0:
            saved_tokens = saved_chars // TOKEN_CHARS
        saved_chars_total += saved_chars
        saved_tokens_total += saved_tokens

        source_surface = _safe_surface(row, routing, managed)
        provider = _reason_code(row.get("provider"), "anthropic")
        requested_tier = _reason_code(model_tier(str(row.get("requested_model") or "")), "unknown")
        routed_tier = _reason_code(model_tier(str(row.get("routed_model") or "")), "unknown")
        category = _reason_code(row.get("category") or routing.get("category"), "unknown")
        phase = _workflow_phase(routing)
        source_surface_counts[source_surface] += 1
        provider_counts[provider] += 1
        requested_tier_counts[requested_tier] += 1
        routed_tier_counts[routed_tier] += 1
        category_counts[category] += 1
        phase_counts[phase] += 1
        _add_group(
            groups,
            state=state,
            row=row,
            routing=routing,
            managed=managed,
            saved_chars=saved_chars,
            saved_tokens=saved_tokens,
        )

    for row in feedback_rows:
        queue_status_counts[_reason_code(row.get("status"), "unknown")] += 1
        queue_source_surface_counts[_reason_code(row.get("source_surface"), "unknown")] += 1

    summary = {
        "sampled_call_count": len(rows),
        "managed_recommendation_rows": managed_rows,
        "historical_null_rows": len(rows) - managed_rows,
        "repeated_scaffold_recommended_count": recommended_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "safety_stop_count": safety_stop_count,
        "server_error_count": state_counts.get("server-error", 0),
        "server_unreachable_count": state_counts.get("server-unreachable", 0),
        "baseline_count": state_counts.get("baseline-no-repeated-scaffold-policy", 0),
        "preflight_disabled_count": state_counts.get("preflight-disabled", 0),
        "optimization_unit_present_count": optimization_unit_counts.get("present", 0),
        "optimization_unit_missing_count": optimization_unit_counts.get("missing", 0),
        "feedback_sent_count": call_feedback_status_counts.get("sent", 0) + queue_status_counts.get("sent", 0),
        "feedback_queued_count": call_feedback_status_counts.get("queued", 0) + queue_status_counts.get("queued", 0),
        "feedback_retryable_error_count": queue_status_counts.get("retryable-error", 0),
        "feedback_queue_total": sum(queue_status_counts.values()),
        "estimated_saved_chars": saved_chars_total,
        "estimated_saved_tokens": saved_tokens_total,
        "activation_group_count": len(groups),
    }
    summary["explanation"] = _explanation(summary, state_counts)

    activation_groups = sorted(
        groups.values(),
        key=lambda item: (
            -_as_int(item.get("call_count")),
            str(item.get("activation_state")),
            str(item.get("source_surface")),
            str(item.get("category")),
        ),
    )[:100]
    for group in activation_groups:
        group["error_rate"] = round(
            _as_float(group.get("error_count")) / max(1, _as_int(group.get("call_count"))),
            6,
        )

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
        "status": "matched" if managed_rows or any(_provider_meta(_json_obj(row.get("crunch_json"))) for row in rows) else "no-activation-metadata",
        "summary": summary,
        "activation_state_counts": _counter_rows(state_counts),
        "managed_status_counts": _counter_rows(managed_status_counts),
        "managed_reason_counts": _counter_rows(managed_reason_counts),
        "managed_apply_reason_counts": _counter_rows(apply_reason_counts),
        "optimization_unit_presence_counts": _counter_rows(optimization_unit_counts),
        "crunch_profile_status_counts": _counter_rows(crunch_status_counts),
        "crunch_profile_reason_counts": _counter_rows(crunch_reason_counts),
        "canary_cohort_counts": _counter_rows(cohort_counts),
        "safety_stop_reason_counts": _counter_rows(safety_reason_counts),
        "call_feedback_status_counts": _counter_rows(call_feedback_status_counts),
        "call_feedback_reason_counts": _counter_rows(call_feedback_reason_counts),
        "feedback_queue_status_counts": _counter_rows(queue_status_counts),
        "feedback_queue_source_surface_counts": _counter_rows(queue_source_surface_counts),
        "source_surface_counts": _counter_rows(source_surface_counts),
        "provider_counts": _counter_rows(provider_counts),
        "requested_model_tier_counts": _counter_rows(requested_tier_counts),
        "routed_model_tier_counts": _counter_rows(routed_tier_counts),
        "category_counts": _counter_rows(category_counts),
        "workflow_phase_counts": _counter_rows(phase_counts),
        "activation_groups": activation_groups,
        "privacy": {
            **_privacy_summary(),
            "basis": "local calls.routing_json, calls.crunch_json, and managed feedback queue metadata only",
            "optimization_unit_ids_included": False,
            "feedback_payloads_included": False,
        },
    }
