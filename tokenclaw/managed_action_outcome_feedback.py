"""Local feedback recorder for managed ``ActionExecutor`` decisions.

The local ``tokenclaw`` client sends normalized, metadata-only feedback after
every managed decision so the managed optimizer server can learn from what the
local executor actually did: applied, held out, vetoed by a local opt-out,
unsupported by the local executor, or skipped because of a fallback.

This module deliberately consumes only the public ``ActionExecutor`` result
shape (see :mod:`tokenclaw.action_executor`) plus a small whitelist of numeric
outcome metrics. It never forwards raw prompts, raw responses, provider bodies,
file paths, cache keys, or secrets. Sending reuses the existing managed policy
event queue (bounded retry/backoff) so a send failure can never block provider
forwarding.
"""

from __future__ import annotations

from typing import Any

from tokenclaw.action_executor import DEFAULT_ACTION_FAMILIES
from tokenclaw.managed_egress import (
    ManagedEgressBlocked,
    assert_managed_egress_safe,
    managed_egress_blocked_meta,
)
from tokenclaw.policy_files import utc_now


MANAGED_ACTION_OUTCOME_FEEDBACK_SCHEMA = "tokenclaw.managed_action_outcome_feedback.v1"
MANAGED_ACTION_OUTCOME_EVENT_TYPE = "managed_action_outcome"
MANAGED_ACTION_OUTCOME_SOURCE_SURFACE = "managed_action_outcome"
MANAGED_ACTION_OUTCOME_ENDPOINT = "/v1/policy-events"

# Local results that the executor can produce for a managed decision. Each is a
# fixed enum value, never free-form text derived from a request.
LOCAL_RESULTS = ("applied", "held", "heldout", "vetoed", "unsupported", "fallback", "noop")

# Numeric outcome metrics the recorder is allowed to forward. Anything not in
# this whitelist is dropped, keeping the payload metadata-only. Note that keys
# such as ``input``/``output``/``token`` are intentionally avoided because the
# managed egress guard treats them as raw-like.
_INT_METRIC_KEYS = (
    "status_code",
    "retry_count",
    "fallback_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
_FLOAT_METRIC_KEYS = (
    "latency_ms",
    "estimated_cost_usd",
    "estimated_baseline_usd",
)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (ValueError, AttributeError):
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None
    return None


def _normalize_metrics(outcome_metrics: Any) -> dict[str, Any]:
    if not isinstance(outcome_metrics, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key in _INT_METRIC_KEYS:
        coerced = _coerce_int(outcome_metrics.get(key))
        if coerced is not None:
            metrics[key] = coerced
    for key in _FLOAT_METRIC_KEYS:
        coerced = _coerce_float(outcome_metrics.get(key))
        if coerced is not None:
            metrics[key] = coerced
    return metrics


def _family_record(result: dict[str, Any], family: str) -> dict[str, Any]:
    section = result.get(family)
    if not isinstance(section, dict):
        section = {}
    status = str(section.get("status") or "not-present")
    reason = section.get("veto_reason") or section.get("apply_reason")
    return {
        "family": family,
        "status": status,
        "applied": bool(section.get("applied")),
        "reason": str(reason) if reason else None,
    }


def _capability_reason_codes(result: dict[str, Any]) -> list[str]:
    """Collect bounded reason codes for actions the local executor could not run.

    Only fixed ``reason`` enum strings are read from each unsupported entry; the
    arbitrary ``section``/``type`` values are never forwarded.
    """
    codes: set[str] = set()
    for entry in result.get("unsupported_actions") or []:
        if isinstance(entry, dict):
            reason = entry.get("reason")
            if reason:
                codes.add(str(reason))
        elif entry is not None:
            codes.add("unsupported-action")
    return sorted(codes)


def _veto_reason_codes(result: dict[str, Any], actions: list[dict[str, Any]]) -> list[str]:
    codes: set[str] = set()
    for record in actions:
        if record.get("status") == "vetoed" and record.get("reason"):
            codes.add(str(record["reason"]))
    if str(result.get("status")) == "vetoed":
        reason = result.get("apply_reason")
        if reason:
            codes.add(str(reason))
    return sorted(codes)


def _families_with_status(actions: list[dict[str, Any]], status: str) -> list[str]:
    return sorted(
        record["family"]
        for record in actions
        if record.get("status") == status
    )


def build_managed_action_feedback(
    result: dict[str, Any],
    *,
    source_surface: str | None = None,
    app_family: str | None = None,
    contract_id: str | None = None,
    provider: str | None = None,
    outcome_metrics: Any = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Build a metadata-only managed feedback event from an executor result.

    Every managed decision yields feedback, even when no action is applied.
    Locally vetoed actions are reported as vetoes (``local_result`` and the
    per-family ``status``), not as policy failures.
    """
    result = result if isinstance(result, dict) else {}
    actions = [_family_record(result, family) for family in DEFAULT_ACTION_FAMILIES]
    local_result = str(result.get("status") or "held")
    if local_result not in LOCAL_RESULTS:
        local_result = "held"

    product_mode = result.get("product_mode")
    event: dict[str, Any] = {
        "schema": MANAGED_ACTION_OUTCOME_FEEDBACK_SCHEMA,
        "event_type": MANAGED_ACTION_OUTCOME_EVENT_TYPE,
        "generated_at": now or utc_now(),
        "provider": str(provider or result.get("provider") or "unknown"),
        "source_surface": str(source_surface or "unknown"),
        "app_family": str(app_family or "unknown"),
        "contract_id": str(contract_id) if contract_id else None,
        "decision_id": result.get("decision_id"),
        "policy_id": result.get("policy_id"),
        "policy_source": result.get("policy_source"),
        "local_result": local_result,
        "apply_reason": result.get("apply_reason"),
        "applied": bool(result.get("applied")),
        "fallback": result.get("fallback"),
        "shadow_only": bool(result.get("shadow_only")),
        "server_traffic_treatment": result.get("server_traffic_treatment"),
        "server_route_selected": result.get("server_route_selected"),
        "canary_fraction": result.get("canary_fraction"),
        "holdout_fraction": result.get("holdout_fraction"),
        "enabled": bool(result.get("enabled")),
        "application_enabled": bool(result.get("application_enabled")),
        "product_mode_enforced": bool(result.get("product_mode_enforced")),
        "product_mode": product_mode if isinstance(product_mode, dict) else None,
        "actions": actions,
        "applied_families": sorted(result.get("applied_families") or []),
        "vetoed_families": _families_with_status(actions, "vetoed"),
        "held_families": _families_with_status(actions, "held"),
        "heldout_families": _families_with_status(actions, "heldout"),
        "unsupported_action_count": len(result.get("unsupported_actions") or []),
        "veto_reason_codes": _veto_reason_codes(result, actions),
        "capability_reason_codes": _capability_reason_codes(result),
        "supported_local_action_families": sorted(result.get("supported_local_action_families") or []),
        "enabled_local_action_families": sorted(result.get("enabled_local_action_families") or []),
        "privacy_summary": {
            "schema": "tokenclaw.managed_action_outcome_privacy.v1",
            "metadata_only": True,
            "raw_payload_included": False,
            "raw_prompt_included": False,
            "raw_response_included": False,
            "provider_body_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "secrets_included": False,
            "api_key_value_included": False,
        },
    }

    metrics = _normalize_metrics(outcome_metrics)
    if metrics:
        metrics["schema"] = "tokenclaw.managed_action_outcome_metrics.v1"
        event["outcome_metrics"] = metrics

    return {key: value for key, value in event.items() if value is not None}


async def record_managed_action_feedback(
    store_obj: Any,
    result: dict[str, Any],
    *,
    source_surface: str | None = None,
    app_family: str | None = None,
    contract_id: str | None = None,
    provider: str | None = None,
    outcome_metrics: Any = None,
    flush_immediately: bool = True,
) -> dict[str, Any]:
    """Record managed action feedback, queueing locally with bounded retry.

    The event is enqueued through the managed policy event queue (even when
    managed mode is disabled) so feedback survives transient send failures and
    never blocks provider forwarding. Any unexpected error is swallowed and
    surfaced as metadata so callers can treat this as best-effort.
    """
    event = build_managed_action_feedback(
        result,
        source_surface=source_surface,
        app_family=app_family,
        contract_id=contract_id,
        provider=provider,
        outcome_metrics=outcome_metrics,
    )

    try:
        assert_managed_egress_safe(event)
    except ManagedEgressBlocked as exc:
        meta = managed_egress_blocked_meta(
            endpoint=MANAGED_ACTION_OUTCOME_ENDPOINT,
            violations=exc.violations,
        )
        meta.update({
            "enabled": False,
            "source_surface": source_surface or MANAGED_ACTION_OUTCOME_SOURCE_SURFACE,
            "payload_included": False,
        })
        return meta

    try:
        from tokenclaw.recommendations import queue_policy_event_feedback

        feedback_meta = await queue_policy_event_feedback(
            store_obj,
            event,
            source_surface=source_surface or MANAGED_ACTION_OUTCOME_SOURCE_SURFACE,
            queue_when_disabled=True,
            flush_immediately=flush_immediately,
        )
    except Exception as exc:  # best-effort: never block forwarding
        return {
            "enabled": True,
            "status": "error",
            "reason": "queue-failed",
            "endpoint": MANAGED_ACTION_OUTCOME_ENDPOINT,
            "source_surface": source_surface or MANAGED_ACTION_OUTCOME_SOURCE_SURFACE,
            "error": repr(exc),
            "payload_included": False,
        }

    return {
        "enabled": bool(feedback_meta.get("enabled")),
        "status": feedback_meta.get("status"),
        "reason": feedback_meta.get("reason"),
        "endpoint": feedback_meta.get("endpoint") or MANAGED_ACTION_OUTCOME_ENDPOINT,
        "queue_id": feedback_meta.get("queue_id"),
        "attempts": feedback_meta.get("attempts"),
        "status_code": feedback_meta.get("status_code"),
        "latency_ms": feedback_meta.get("latency_ms"),
        "source_surface": source_surface or MANAGED_ACTION_OUTCOME_SOURCE_SURFACE,
        "local_result": event.get("local_result"),
        "payload_included": False,
    }
