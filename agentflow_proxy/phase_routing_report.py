from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from agentflow_proxy.pricing import estimate_cost

SCHEMA = "agentflow.phase_routing_opportunity.v1"
SAFE_DOWNGRADE_PHASES = {"tool-execution", "summary"}
TOKEN_CHARS = 4


def _json_obj(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _round_usd(value: float) -> float:
    return round(float(value or 0.0), 6)


def _model_tier(model: Any) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    return "unknown"


def _target_for_requested(model: Any) -> tuple[str | None, str | None]:
    tier = _model_tier(model)
    if tier == "sonnet":
        return "haiku", "claude-haiku-4-5-20251001"
    if tier == "opus":
        return "sonnet", "claude-sonnet-4-6"
    return None, None


def _text_bucket(chars: int) -> str:
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _token_bucket(tokens: int) -> str:
    if tokens < 1_000:
        return "lt_1k_tokens"
    if tokens < 4_000:
        return "1k_4k_tokens"
    if tokens < 16_000:
        return "4k_16k_tokens"
    if tokens < 64_000:
        return "16k_64k_tokens"
    return "gte_64k_tokens"


def _error_bucket(row: dict[str, Any]) -> str:
    status = _as_int(row.get("status_code"))
    if status <= 0:
        return "unknown"
    if status < 400:
        return "ok"
    if status == 429:
        return "rate_limited"
    if status == 529:
        return "overloaded"
    if status == 401:
        return "auth_error"
    if status == 400:
        return "bad_request"
    if status >= 500:
        return "server_error"
    return f"http_{status}"


def _derive_phase(row: dict[str, Any], routing: dict[str, Any]) -> tuple[str, str]:
    explicit = str(routing.get("workflow_phase") or "").strip().lower()
    explicit_map = {
        "tool-result": "tool-execution",
        "tool_execution": "tool-execution",
        "tool-execution": "tool-execution",
        "summary": "summary",
        "planning": "planning",
        "verification": "verification",
        "thinking": "thinking",
        "chat": "unknown",
    }
    if explicit:
        return explicit_map.get(explicit, explicit), "routing_json.workflow_phase"

    reason = str(routing.get("reason") or "").lower()
    category = str(row.get("category") or routing.get("category") or "").lower()
    if "thinking" in reason:
        return "thinking", "routing_reason"
    if category == "tool-result":
        return "tool-execution", "category"
    if category in {"short-completion", "summary"}:
        return "summary", "category"
    if category == "code-gen":
        return "verification", "category"
    if category in {"tool-heavy", "tool-light"}:
        return "verification", "category"
    if category == "long-context":
        return "planning", "category"
    return "unknown", "fallback"


def _blockers(row: dict[str, Any], routing: dict[str, Any], phase: str, target_tier: str | None) -> list[str]:
    blockers: list[str] = []
    requested_tier = _model_tier(row.get("requested_model"))
    routed_tier = _model_tier(row.get("routed_model"))
    reason = str(routing.get("reason") or "").lower()
    if target_tier is None:
        blockers.append("non_candidate_model")
    elif routed_tier == target_tier:
        blockers.append("already_routed")
    if phase == "thinking" or "thinking" in reason:
        blockers.append("thinking")
    if phase not in SAFE_DOWNGRADE_PHASES:
        blockers.append(f"phase_{phase}")
    if _error_bucket(row) != "ok":
        blockers.append("historical_error")
    if _as_int(row.get("retry_count")) > 0:
        blockers.append("retried")
    if routing.get("fallback_reason"):
        blockers.append("fallback")
    if requested_tier == "unknown":
        blockers.append("unknown_requested_tier")
    return blockers


def _estimated_input_tokens(row: dict[str, Any], routing: dict[str, Any]) -> int:
    actual = _as_int(row.get("actual_input_tokens"))
    if actual:
        return actual
    estimate = _as_int(row.get("input_tokens_est"))
    if estimate:
        return estimate
    return max(0, _as_int(routing.get("text_chars")) // TOKEN_CHARS)


def _estimated_output_tokens(row: dict[str, Any]) -> int:
    return _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))


def _load_recent_rows(store: Any, *, limit: int) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 10_000))
    rows = store.conn.execute(
        f"""
        select created_at, requested_model, routed_model, stream, status_code,
               input_tokens_est, output_tokens_est, actual_input_tokens, actual_output_tokens,
               cache_creation_input_tokens, cache_read_input_tokens, cost_est_usd,
               cost_baseline_usd, crunch_json, routing_json, cache_json, category,
               retry_count, thinking_output_tokens, provider, session_id
        from calls
        where coalesce(provider, 'anthropic') = 'anthropic'
        order by created_at desc
        limit {limit}
        """
    ).fetchall()
    return [dict(row) for row in rows]


def build_phase_routing_report(store: Any, *, limit: int = 1000) -> dict[str, Any]:
    rows = _load_recent_rows(store, limit=limit)
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    phase_counts: dict[str, int] = defaultdict(int)
    blocker_totals: dict[str, int] = defaultdict(int)
    risk_totals: dict[str, int] = defaultdict(int)
    text_bucket_counts: dict[str, int] = defaultdict(int)
    token_bucket_counts: dict[str, int] = defaultdict(int)
    session_hashes: set[str] = set()
    unknown_sessions = 0
    window_start: str | None = None
    window_end: str | None = None
    total_current_cost = 0.0
    total_baseline_cost = 0.0

    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        phase, phase_source = _derive_phase(row, routing)
        phase_counts[phase] += 1
        target_tier, target_model = _target_for_requested(row.get("requested_model"))
        requested_tier = _model_tier(row.get("requested_model"))
        routed_tier = _model_tier(row.get("routed_model"))
        model_pair = f"{requested_tier}_to_{target_tier}" if target_tier else f"{requested_tier}_to_unknown"
        key = (phase, model_pair, target_model or "none")
        if key not in groups:
            groups[key] = {
                "phase": phase,
                "model_pair": model_pair,
                "target_model": target_model,
                "sample_count": 0,
                "current_routed_count": 0,
                "blocked_count": 0,
                "blocked_count_by_reason": defaultdict(int),
                "risk_exclusions": defaultdict(int),
                "projected_candidate_count": 0,
                "current_cost_usd": 0.0,
                "projected_target_cost_usd": 0.0,
                "projected_savings_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "prompt_cache_savings_usd": 0.0,
                "crunch_savings_tokens_est": 0,
                "phase_source_breakdown": defaultdict(int),
                "text_bucket_breakdown": defaultdict(int),
                "token_bucket_breakdown": defaultdict(int),
                "status_bucket_breakdown": defaultdict(int),
            }
        group = groups[key]
        group["sample_count"] += 1
        group["phase_source_breakdown"][phase_source] += 1

        input_tokens = _estimated_input_tokens(row, routing)
        output_tokens = _estimated_output_tokens(row)
        text_chars = _as_int(routing.get("text_chars")) or input_tokens * TOKEN_CHARS
        text_bucket = _text_bucket(text_chars)
        token_bucket = _token_bucket(input_tokens)
        status_bucket = _error_bucket(row)
        group["text_bucket_breakdown"][text_bucket] += 1
        group["token_bucket_breakdown"][token_bucket] += 1
        group["status_bucket_breakdown"][status_bucket] += 1
        text_bucket_counts[text_bucket] += 1
        token_bucket_counts[token_bucket] += 1

        current_cost = _as_float(row.get("cost_est_usd"))
        baseline_cost = _as_float(row.get("cost_baseline_usd"))
        total_current_cost += current_cost
        total_baseline_cost += baseline_cost
        group["current_cost_usd"] += current_cost
        group["baseline_cost_usd"] += baseline_cost
        group["prompt_cache_savings_usd"] += max(0.0, baseline_cost - current_cost)
        group["crunch_savings_tokens_est"] += _as_int(crunch.get("tokens_saved_est"))

        if target_tier and routed_tier == target_tier:
            group["current_routed_count"] += 1

        blockers = _blockers(row, routing, phase, target_tier)
        if blockers:
            group["blocked_count"] += 1
        for blocker in blockers:
            blocker_totals[blocker] += 1
            group["blocked_count_by_reason"][blocker] += 1
            if blocker.startswith("phase_") or blocker in {"thinking", "historical_error", "retried", "fallback"}:
                risk_totals[blocker] += 1
                group["risk_exclusions"][blocker] += 1
        candidate = not blockers and target_model is not None
        if candidate:
            target_cost = estimate_cost(
                target_model,
                input_tokens,
                output_tokens,
                _as_int(row.get("cache_creation_input_tokens")),
                _as_int(row.get("cache_read_input_tokens")),
                provider="anthropic",
            )
            if target_cost is None:
                group["blocked_count_by_reason"]["target_price_unknown"] += 1
                blocker_totals["target_price_unknown"] += 1
            else:
                group["projected_candidate_count"] += 1
                group["projected_target_cost_usd"] += target_cost
                group["projected_savings_usd"] += max(0.0, current_cost - target_cost)

        sid = row.get("session_id")
        if sid:
            session_hashes.add(str(sid))
        else:
            unknown_sessions += 1
        created_at = row.get("created_at")
        if created_at:
            created = str(created_at)
            window_start = created if window_start is None else min(window_start, created)
            window_end = created if window_end is None else max(window_end, created)

    opportunities = []
    for group in groups.values():
        for key in (
            "blocked_count_by_reason",
            "risk_exclusions",
            "phase_source_breakdown",
            "text_bucket_breakdown",
            "token_bucket_breakdown",
            "status_bucket_breakdown",
        ):
            group[key] = [
                {"value": item_key, "count": item_count}
                for item_key, item_count in sorted(group[key].items(), key=lambda item: (-item[1], item[0]))
            ]
        for key in (
            "current_cost_usd",
            "projected_target_cost_usd",
            "projected_savings_usd",
            "baseline_cost_usd",
            "prompt_cache_savings_usd",
        ):
            group[key] = _round_usd(group[key])
        opportunities.append(group)

    opportunities.sort(
        key=lambda item: (
            -float(item["projected_savings_usd"]),
            -int(item["sample_count"]),
            str(item["phase"]),
            str(item["model_pair"]),
        )
    )

    return {
        "schema": SCHEMA,
        "sampled_call_count": len(rows),
        "limit": max(1, min(int(limit), 10_000)),
        "window": {"start": window_start, "end": window_end},
        "summary": {
            "candidate_count": sum(int(item["projected_candidate_count"]) for item in opportunities),
            "current_routed_count": sum(int(item["current_routed_count"]) for item in opportunities),
            "projected_savings_usd": _round_usd(sum(float(item["projected_savings_usd"]) for item in opportunities)),
            "current_cost_usd": _round_usd(total_current_cost),
            "baseline_cost_usd": _round_usd(total_baseline_cost),
            "unique_session_count": len(session_hashes),
            "unknown_session_count": unknown_sessions,
        },
        "phase_breakdown": [
            {"phase": key, "count": value}
            for key, value in sorted(phase_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "blocked_count_by_reason": [
            {"reason": key, "count": value}
            for key, value in sorted(blocker_totals.items(), key=lambda item: (-item[1], item[0]))
        ],
        "risk_exclusions": [
            {"reason": key, "count": value}
            for key, value in sorted(risk_totals.items(), key=lambda item: (-item[1], item[0]))
        ],
        "text_bucket_breakdown": [
            {"bucket": key, "count": value}
            for key, value in sorted(text_bucket_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "token_bucket_breakdown": [
            {"bucket": key, "count": value}
            for key, value in sorted(token_bucket_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "opportunities": opportunities,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "session_ids_included": False,
            "request_ids_included": False,
            "file_paths_included": False,
            "error_samples_included": False,
        },
    }
