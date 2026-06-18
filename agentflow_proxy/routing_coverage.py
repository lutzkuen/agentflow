from __future__ import annotations

import json
from typing import Any


SCHEMA = "agentflow.routing_coverage_report.v1"
ROW_SCHEMA = "agentflow.routing_coverage_row.v1"


def _json_obj(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or default


def _bool_count(value: int) -> bool:
    return bool(value and value > 0)


def _counter_breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _empty_surface(surface: str, label: str) -> dict[str, Any]:
    return {
        "surface": surface,
        "label": label,
        "sample_count": 0,
        "routed_count": 0,
        "holdout_count": 0,
        "applied_count": 0,
        "safety_stop_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "source_surface_counts": {},
        "provider_counts": {},
        "category_counts": {},
    }


def _surface_state(rows: list[dict[str, Any]], codex_app_event_count: int) -> dict[str, dict[str, Any]]:
    surfaces = {
        "openai_api": _empty_surface("openai_api", "OpenAI-compatible API apps"),
        "codex_vscode_or_cli": _empty_surface("codex_vscode_or_cli", "Codex VS Code / CLI through OpenAI-compatible proxy"),
        "codex_app_server_telemetry": _empty_surface("codex_app_server_telemetry", "Codex app-server telemetry relay"),
        "anthropic_api": _empty_surface("anthropic_api", "Anthropic-compatible API apps"),
        "claude_vscode_or_claude_code": _empty_surface("claude_vscode_or_claude_code", "Claude VS Code / Claude Code"),
        "unsupported_or_unknown": _empty_surface("unsupported_or_unknown", "Unsupported or unknown surface"),
    }
    surfaces["codex_app_server_telemetry"]["sample_count"] = max(0, codex_app_event_count)

    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        provider = _label(row.get("provider") or routing.get("provider"), "unknown")
        source_surface = _label(row.get("source_surface") or routing.get("source_surface"), "unknown")
        endpoint = _label(row.get("endpoint") or row.get("path"), "unknown")
        category = _label(row.get("category") or routing.get("category"), "unknown")
        requested = str(row.get("requested_model") or routing.get("requested_model") or "")
        routed = str(row.get("routed_model") or routing.get("routed_model") or requested)
        app_family = _label(routing.get("app_family") or routing.get("client_family"), "")

        if provider == "openai" and (source_surface.startswith("codex") or "codex" in app_family):
            keys = ["codex_vscode_or_cli"]
        elif provider == "openai":
            keys = ["openai_api"]
        elif provider == "anthropic" and (
            "claude" in app_family or category.startswith("tool") or source_surface == "anthropic_messages"
        ):
            keys = ["anthropic_api", "claude_vscode_or_claude_code"]
        elif provider == "anthropic":
            keys = ["anthropic_api"]
        else:
            keys = ["unsupported_or_unknown"]

        for key in keys:
            surface = surfaces[key]
            surface["sample_count"] += 1
            if requested and routed and requested != routed:
                surface["routed_count"] += 1
            if _as_int(row.get("status_code")) >= 400:
                surface["error_count"] += 1
            surface["retry_count"] += _as_int(row.get("retry_count"))
            surface["source_surface_counts"][source_surface] = surface["source_surface_counts"].get(source_surface, 0) + 1
            surface["provider_counts"][provider] = surface["provider_counts"].get(provider, 0) + 1
            surface["category_counts"][category] = surface["category_counts"].get(category, 0) + 1

            openai_canary = routing.get("openai_canary") if isinstance(routing.get("openai_canary"), dict) else {}
            phase_canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
            canary_status = _label(
                openai_canary.get("status")
                or openai_canary.get("decision")
                or openai_canary.get("cohort")
                or phase_canary.get("status")
                or phase_canary.get("cohort"),
                "",
            )
            if "holdout" in canary_status:
                surface["holdout_count"] += 1
            if "applied" in canary_status or canary_status in {"routed", "canary"}:
                surface["applied_count"] += 1
            if "safety" in canary_status or _label(phase_canary.get("reason"), "").startswith("safety"):
                surface["safety_stop_count"] += 1
            safety = openai_canary.get("safety_stop") if isinstance(openai_canary.get("safety_stop"), dict) else {}
            if safety.get("triggered"):
                surface["safety_stop_count"] += 1

    return surfaces


def _row(
    state: dict[str, Any],
    *,
    routing_supported: bool,
    local_mutation_possible: bool,
    holdout_available: bool,
    outcome_feedback_available: bool,
    top_blocker_reason: str,
    next_action: str,
    routing_active: bool | None = None,
    telemetry_only: bool = False,
    expansion_candidate: bool = False,
    explanation: str,
) -> dict[str, Any]:
    active = _bool_count(state["routed_count"] + state["applied_count"]) if routing_active is None else routing_active
    return {
        "schema": ROW_SCHEMA,
        "surface": state["surface"],
        "label": state["label"],
        "traffic_seen": _bool_count(state["sample_count"]),
        "sample_count": state["sample_count"],
        "routing_supported": routing_supported,
        "routing_active": active,
        "local_mutation_possible": local_mutation_possible,
        "telemetry_only": telemetry_only,
        "holdout_available": holdout_available,
        "outcome_feedback_available": outcome_feedback_available,
        "top_blocker_reason": top_blocker_reason,
        "next_action": next_action,
        "expansion_candidate": expansion_candidate,
        "routed_count": state["routed_count"],
        "applied_count": state["applied_count"],
        "holdout_count": state["holdout_count"],
        "safety_stop_count": state["safety_stop_count"],
        "error_count": state["error_count"],
        "retry_count": state["retry_count"],
        "source_surface_breakdown": _counter_breakdown(state["source_surface_counts"]),
        "category_breakdown": _counter_breakdown(state["category_counts"]),
        "explanation": explanation,
        "privacy": _privacy(),
    }


def _privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_responses_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _report_rows(surfaces: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    openai = surfaces["openai_api"]
    openai_active = _bool_count(openai["routed_count"] + openai["applied_count"])
    openai_holdout = _bool_count(openai["holdout_count"])
    return [
        _row(
            openai,
            routing_supported=True,
            routing_active=openai_active,
            local_mutation_possible=True,
            holdout_available=openai_holdout,
            outcome_feedback_available=_bool_count(openai["sample_count"]),
            top_blocker_reason="none-routing-active" if openai_active else "openai-routing-not-observed-in-window",
            next_action="measure-openai-routing-rule-outcomes" if openai_active else "stage-openai-routing-canary",
            explanation=(
                "OpenAI-compatible API traffic is the only surface where AgentFlow currently has a local "
                "request-mutation executor, canary/holdout metadata, and observed outcome counters."
            ),
        ),
        _row(
            surfaces["codex_vscode_or_cli"],
            routing_supported=True,
            local_mutation_possible=True,
            holdout_available=False,
            outcome_feedback_available=False,
            top_blocker_reason="requires-openai-compatible-local-proxy-attribution",
            next_action="tag-codex-openai-proxy-traffic-and-reuse-openai-routing-gates",
            expansion_candidate=True,
            explanation=(
                "Codex VS Code / CLI is the next plausible expansion when it is configured through "
                "the local OpenAI-compatible proxy: mutation is possible, but traffic needs distinct "
                "Codex attribution and the same OpenAI canary evidence gates."
            ),
        ),
        _row(
            surfaces["codex_app_server_telemetry"],
            routing_supported=False,
            routing_active=False,
            local_mutation_possible=False,
            holdout_available=False,
            outcome_feedback_available=False,
            top_blocker_reason="telemetry-only-no-provider-request-mutation",
            next_action="keep-codex-app-server-routing-telemetry-only",
            telemetry_only=True,
            explanation=(
                "The Codex app-server relay sees telemetry events, not provider requests that the local "
                "proxy can safely mutate, so it can inform policy but cannot execute routing."
            ),
        ),
        _row(
            surfaces["anthropic_api"],
            routing_supported=True,
            routing_active=False,
            local_mutation_possible=True,
            holdout_available=_bool_count(surfaces["anthropic_api"]["holdout_count"]),
            outcome_feedback_available=_bool_count(surfaces["anthropic_api"]["sample_count"]),
            top_blocker_reason=(
                "anthropic-routing-safety-stop-active"
                if surfaces["anthropic_api"]["safety_stop_count"]
                else "missing-anthropic-applied-holdout-routing-evidence"
            ),
            next_action="collect-anthropic-applied-holdout-coverage-before-routing",
            expansion_candidate=True,
            explanation=(
                "Anthropic-compatible traffic can be mutated by the local Claude proxy, but routing is "
                "held until safety-stop and applied/holdout lifecycle evidence are clean."
            ),
        ),
        _row(
            surfaces["claude_vscode_or_claude_code"],
            routing_supported=True,
            routing_active=False,
            local_mutation_possible=True,
            holdout_available=_bool_count(surfaces["claude_vscode_or_claude_code"]["holdout_count"]),
            outcome_feedback_available=_bool_count(surfaces["claude_vscode_or_claude_code"]["sample_count"]),
            top_blocker_reason="claude-code-tool-thinking-safety-evidence-needed",
            next_action="clear-claude-tool-thinking-routing-safety-gates",
            expansion_candidate=True,
            explanation=(
                "Claude Code / Claude VS Code traffic is high-value but tool and thinking mechanics make "
                "downgrades riskier than the current OpenAI API rule."
            ),
        ),
        _row(
            surfaces["unsupported_or_unknown"],
            routing_supported=False,
            routing_active=False,
            local_mutation_possible=False,
            holdout_available=False,
            outcome_feedback_available=False,
            top_blocker_reason="unsupported-or-unknown-provider-surface",
            next_action="classify-source-surface-before-routing",
            explanation="Unknown surfaces are not routed until the provider, endpoint, and local mutation path are explicit.",
        ),
    ]


def _next_expansion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priority = ["codex_vscode_or_cli", "anthropic_api", "claude_vscode_or_claude_code"]
    by_surface = {row["surface"]: row for row in rows}
    for surface in priority:
        row = by_surface.get(surface)
        if row and row.get("routing_supported") and row.get("local_mutation_possible") and not row.get("routing_active"):
            return {
                "surface": row["surface"],
                "label": row["label"],
                "reason": row["top_blocker_reason"],
                "next_action": row["next_action"],
                "traffic_seen": row["traffic_seen"],
                "why_plausible": row["explanation"],
            }
    return {
        "surface": None,
        "label": None,
        "reason": "no-eligible-expansion-surface",
        "next_action": "keep-measuring-openai-routing",
        "traffic_seen": False,
        "why_plausible": "No non-active surface currently has both support and local mutation capability.",
    }


def build_routing_coverage_report(store_obj: Any, limit: int = 5000) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 5000), 25_000))
    call_rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model, status_code,
                   retry_count, category, routing_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    try:
        codex_row = store_obj.conn.execute(
            """
            select count(*) as count
            from (
              select id from codex_app_events
              order by created_at desc
              limit ?
            )
            """,
            (capped_limit,),
        ).fetchone()
        codex_app_event_count = _as_int(codex_row["count"] if codex_row else 0)
    except Exception:
        codex_app_event_count = 0

    surfaces = _surface_state(call_rows, codex_app_event_count)
    rows = _report_rows(surfaces)
    active_surfaces = [row["surface"] for row in rows if row["routing_active"]]
    telemetry_only = [row["surface"] for row in rows if row["telemetry_only"] and row["traffic_seen"]]
    next_expansion = _next_expansion(rows)
    return {
        "schema": SCHEMA,
        "status": "reported",
        "summary": {
            "routing_currently_active_only_for": active_surfaces or [],
            "openai_api_routing_active": "openai_api" in active_surfaces,
            "active_surface_count": len(active_surfaces),
            "metadata_only": True,
            "sample_count": len(call_rows),
            "codex_app_event_count": codex_app_event_count,
            "telemetry_only_surfaces_with_traffic": telemetry_only,
            "next_expansion_surface": next_expansion,
        },
        "rows": rows,
        "privacy": _privacy(),
    }
