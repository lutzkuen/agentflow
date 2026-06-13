from __future__ import annotations

from typing import Any


SCHEMA = "agentflow.savings_report.v1"

OPPORTUNITY_FAMILY_ACTIVATION = "activation"
OPPORTUNITY_FAMILY_CACHE_REPLAY = "cache-replay"
OPPORTUNITY_FAMILY_MODEL_ROUTING = "model-routing"
OPPORTUNITY_FAMILY_CRUNCH = "crunch"
OPPORTUNITY_FAMILY_POLICY_RELOAD = "policy-reload"
OPPORTUNITY_FAMILY_SAFETY_STOP = "safety-stop"

SAVINGS_BUCKET_NONE = "none"
SAVINGS_BUCKET_LOW = "low"
SAVINGS_BUCKET_MEDIUM = "medium"
SAVINGS_BUCKET_HIGH = "high"
SAVINGS_BUCKET_UNKNOWN = "unknown"

_BUCKET_ORDER = {
    SAVINGS_BUCKET_HIGH: 0,
    SAVINGS_BUCKET_MEDIUM: 1,
    SAVINGS_BUCKET_LOW: 2,
    SAVINGS_BUCKET_UNKNOWN: 3,
    SAVINGS_BUCKET_NONE: 4,
}

_FAMILY_ORDER = {
    OPPORTUNITY_FAMILY_MODEL_ROUTING: 0,
    OPPORTUNITY_FAMILY_CACHE_REPLAY: 1,
    OPPORTUNITY_FAMILY_CRUNCH: 2,
    OPPORTUNITY_FAMILY_POLICY_RELOAD: 3,
    OPPORTUNITY_FAMILY_SAFETY_STOP: 4,
    OPPORTUNITY_FAMILY_ACTIVATION: 5,
}


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "filesystem_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _savings_bucket(usd: float | None) -> str:
    if usd is None:
        return SAVINGS_BUCKET_UNKNOWN
    if usd <= 0.0:
        return SAVINGS_BUCKET_NONE
    if usd < 0.10:
        return SAVINGS_BUCKET_LOW
    if usd < 1.00:
        return SAVINGS_BUCKET_MEDIUM
    return SAVINGS_BUCKET_HIGH


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _opportunity_sort_key(opp: dict[str, Any]) -> tuple[int, int]:
    family = str(opp.get("opportunity_family") or "")
    bucket = str(opp.get("projected_savings_bucket") or SAVINGS_BUCKET_UNKNOWN)
    return (_FAMILY_ORDER.get(family, 99), _BUCKET_ORDER.get(bucket, 3))


def _activation_opportunity(target: str, provider: str) -> dict[str, Any]:
    return {
        "target": target,
        "provider": provider,
        "source_surface": target,
        "opportunity_family": OPPORTUNITY_FAMILY_ACTIVATION,
        "blocker_codes": ["target-not-configured"],
        "projected_savings_bucket": SAVINGS_BUCKET_UNKNOWN,
        "evidence_window": None,
        "suggested_command": f"agentflow activate {target}",
    }


def _top_breakdown_value(breakdown: Any, fallback: str = "unknown") -> str:
    if not isinstance(breakdown, list) or not breakdown:
        return fallback
    first = breakdown[0]
    if isinstance(first, dict):
        return str(first.get("value") or fallback)
    return fallback


def _breakdown_blocker_codes(breakdown: Any) -> list[str]:
    if not isinstance(breakdown, list):
        return []
    return [str(item["value"]) for item in breakdown if isinstance(item, dict) and item.get("value")]


def _routing_opportunity(target: str, provider: str, report: dict[str, Any]) -> dict[str, Any] | None:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    call_count = _as_int(summary.get("openai_call_count"))
    if call_count == 0:
        return None

    projected = _as_float(summary.get("projected_savings_usd"))
    top_surface = _top_breakdown_value(report.get("source_surface_breakdown"))
    blockers = _breakdown_blocker_codes(report.get("blocker_reason_breakdown"))
    if not blockers:
        candidate_count = _as_int(summary.get("candidate_count"))
        blockers = ["no-routing-candidates"] if candidate_count == 0 else ["no-routing-blockers"]

    suggested = "agentflow-openai-routing-report" if target == "openai" else None
    return {
        "target": target,
        "provider": provider,
        "source_surface": top_surface,
        "opportunity_family": OPPORTUNITY_FAMILY_MODEL_ROUTING,
        "blocker_codes": blockers,
        "projected_savings_bucket": _savings_bucket(projected),
        "projected_savings_usd": round(projected, 6),
        "evidence_window": {"calls": call_count},
        "suggested_command": suggested,
    }


def _cache_replay_opportunity(
    target: str,
    provider: str,
    report: dict[str, Any],
    *,
    blocker_ladder: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    call_count = _as_int(summary.get("openai_call_count"))
    if call_count == 0:
        return None

    projected = _as_float(summary.get("projected_savings_usd"))
    cache_hits = _as_int(summary.get("already_cache_hit_count"))
    top_surface = _top_breakdown_value(report.get("endpoint_breakdown"))
    blockers = _breakdown_blocker_codes(report.get("blocker_reason_breakdown"))

    if cache_hits == 0 and "no-cache-hits" not in blockers:
        blockers = ["no-cache-hits"] + blockers
    if isinstance(blocker_ladder, dict):
        ladder = blocker_ladder.get("ladder") if isinstance(blocker_ladder.get("ladder"), list) else []
        ladder_codes = [
            str(row.get("blocker_code"))
            for row in ladder
            if isinstance(row, dict)
            and row.get("blocker_code")
            and row.get("blocker_code") != "cache-hit-observed"
        ]
        for code in ladder_codes[:5]:
            if code not in blockers:
                blockers.append(code)
    if not blockers:
        blockers = ["cache-replay-ready"]

    suggested = "agentflow-openai-cache-replay-report" if target == "openai" else None
    opportunity = {
        "target": target,
        "provider": provider,
        "source_surface": top_surface,
        "opportunity_family": OPPORTUNITY_FAMILY_CACHE_REPLAY,
        "blocker_codes": blockers,
        "projected_savings_bucket": _savings_bucket(projected),
        "projected_savings_usd": round(projected, 6),
        "evidence_window": {"calls": call_count, "cache_hits": cache_hits},
        "suggested_command": suggested,
    }
    if isinstance(blocker_ladder, dict):
        ladder_summary = blocker_ladder.get("summary") if isinstance(blocker_ladder.get("summary"), dict) else {}
        ladder_rows = blocker_ladder.get("ladder") if isinstance(blocker_ladder.get("ladder"), list) else []
        opportunity["cache_blocker_ladder_summary"] = {
            "scan_limit": _as_int(ladder_summary.get("scan_limit")),
            "scanned_rows": _as_int(ladder_summary.get("scanned_rows")),
            "bounded_recent_window": bool(ladder_summary.get("bounded_recent_window")),
            "zero_hit_window": bool(ladder_summary.get("zero_hit_window")),
            "top_blocker_code": ladder_summary.get("top_blocker_code"),
            "top_next_action_family": ladder_summary.get("top_next_action_family"),
        }
        opportunity["cache_blocker_ladder"] = [row for row in ladder_rows[:10] if isinstance(row, dict)]
        opportunity["evidence_window"]["cache_blocker_scan_rows"] = _as_int(ladder_summary.get("scanned_rows"))
    return opportunity


def _cache_blocker_ladder_for_store(store: Any, *, provider: str, limit: int) -> dict[str, Any] | None:
    try:
        from agentflow_proxy.stats import _cache_zero_hit_blocker_ladder
    except Exception:
        return None
    capped = max(1, min(int(limit or 1000), 5000))
    conn = store.conn
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                select created_at, stream, cache_hit, status_code, cache_json, routing_json,
                       path, coalesce(provider, 'anthropic') as provider,
                       source_surface, endpoint
                from calls
                where coalesce(provider, 'anthropic') = ?
                order by created_at desc
                limit ?
                """,
                (provider, capped),
            ).fetchall()
        ]
    except Exception:
        return None
    return _cache_zero_hit_blocker_ladder(rows, scan_limit=capped)


def build_savings_report(
    activation_config: dict[str, Any],
    *,
    store: Any = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Build a ranked savings opportunity report from local activation config and an optional DB store.

    No provider calls or managed-server calls are made.
    """
    from agentflow_proxy.store import utc_now

    targets_raw = activation_config.get("targets")
    targets = targets_raw if isinstance(targets_raw, dict) else {}

    openai_profile = targets.get("openai") if isinstance(targets.get("openai"), dict) else {}
    claude_profile = targets.get("claude") if isinstance(targets.get("claude"), dict) else {}
    openai_configured = bool(openai_profile.get("configured"))
    claude_configured = bool(claude_profile.get("configured"))

    opportunities: list[dict[str, Any]] = []

    if not openai_configured:
        opportunities.append(_activation_opportunity("openai", "openai"))
    if not claude_configured:
        opportunities.append(_activation_opportunity("claude", "anthropic"))

    if store is not None:
        if openai_configured:
            try:
                from agentflow_proxy.openai_routing_report import build_openai_routing_report
                routing_report = build_openai_routing_report(store, limit=limit)
                opp = _routing_opportunity("openai", "openai", routing_report)
                if opp is not None:
                    opportunities.append(opp)
            except Exception:
                pass

            try:
                from agentflow_proxy.openai_cache_replay_report import build_openai_cache_replay_report
                cache_report = build_openai_cache_replay_report(store, limit=limit)
                blocker_ladder = _cache_blocker_ladder_for_store(store, provider="openai", limit=limit)
                opp = _cache_replay_opportunity("openai", "openai", cache_report, blocker_ladder=blocker_ladder)
                if opp is not None:
                    opportunities.append(opp)
            except Exception:
                pass

    opportunities.sort(key=_opportunity_sort_key)

    lifecycle_feedback = None
    if store is not None:
        try:
            from agentflow_proxy.activation_lifecycle_feedback import activation_lifecycle_feedback_summary

            lifecycle_feedback = activation_lifecycle_feedback_summary(store, limit=limit)
        except Exception:
            lifecycle_feedback = None

    return {
        "schema": SCHEMA,
        "ok": True,
        "generated_at": utc_now(),
        "privacy": _privacy_summary(),
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "activation_lifecycle_feedback": lifecycle_feedback
        or {
            "schema": "agentflow.activation_staged_lifecycle_feedback_summary.v1",
            "queue_rows": 0,
            "family_event_count": 0,
            "state_breakdown": [],
            "event_phase_breakdown": [],
            "cohort_breakdown": [],
            "family_state_breakdown": [],
            "candidate_id_breakdown": [],
            "payload_json_included": False,
            "privacy": _privacy_summary(),
        },
    }
