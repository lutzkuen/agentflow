from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

import yaml

from agentflow_proxy.pricing import estimate_cost

SCHEMA = "agentflow.phase_routing_opportunity.v1"
DRY_RUN_SCHEMA = "agentflow.phase_routing_dry_run.v1"
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


def _resolve_route_to(route_to: Any) -> str:
    value = str(route_to or "").strip()
    if value == "haiku":
        return os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
    if value == "sonnet":
        return os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4-6")
    if value == "opus":
        return os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4-5")
    return value


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


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _row_features(row: dict[str, Any]) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    phase, phase_source = _derive_phase(row, routing)
    input_tokens = _estimated_input_tokens(row, routing)
    output_tokens = _estimated_output_tokens(row)
    text_chars = _as_int(routing.get("text_chars")) or input_tokens * TOKEN_CHARS
    category = row.get("category") or routing.get("category")
    has_tools = routing.get("has_tools")
    if has_tools is None:
        has_tools = str(category or "").startswith("tool-")
    thinking = (
        phase == "thinking"
        or _as_int(row.get("thinking_output_tokens")) > 0
        or "thinking" in str(routing.get("reason") or "").lower()
    )
    return {
        "created_at": row.get("created_at"),
        "provider": row.get("provider") or "anthropic",
        "requested_model": row.get("requested_model"),
        "routed_model": row.get("routed_model") or row.get("requested_model"),
        "requested_tier": _model_tier(row.get("requested_model")),
        "routed_tier": _model_tier(row.get("routed_model")),
        "workflow_phase": phase,
        "workflow_phase_source": phase_source,
        "category": category,
        "has_tools": bool(has_tools),
        "stream": bool(row.get("stream")),
        "status_bucket": _error_bucket(row),
        "status_code": _as_int(row.get("status_code")),
        "retry_count": _as_int(row.get("retry_count")),
        "fallback": bool(routing.get("fallback_reason")),
        "thinking": bool(thinking),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": _as_int(row.get("cache_creation_input_tokens")),
        "cache_read_input_tokens": _as_int(row.get("cache_read_input_tokens")),
        "text_chars": text_chars,
        "text_bucket": _text_bucket(text_chars),
        "token_bucket": _token_bucket(input_tokens),
        "current_cost_usd": _as_float(row.get("cost_est_usd")),
        "baseline_cost_usd": _as_float(row.get("cost_baseline_usd")),
        "routing": routing,
    }


def _value_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _condition_matches(item: dict[str, Any], conditions: dict[str, Any]) -> bool:
    requested = str(item.get("requested_model") or "").lower()
    model_pattern = conditions.get("model_pattern")
    if model_pattern and str(model_pattern).lower() not in requested:
        return False
    env_flag = conditions.get("env_flag")
    if env_flag:
        raw = os.getenv(str(env_flag), "0").strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            return False
    for key, feature_key in (
        ("category", "category"),
        ("workflow_phase", "workflow_phase"),
        ("text_bucket", "text_bucket"),
        ("token_bucket", "token_bucket"),
    ):
        if key in conditions and str(item.get(feature_key) or "") not in set(_value_list(conditions.get(key))):
            return False
    if "category_not_in" in conditions:
        excluded = set(_value_list(conditions.get("category_not_in")))
        if str(item.get("category") or "") in excluded:
            return False
    if "has_tools" in conditions and bool(conditions["has_tools"]) != bool(item.get("has_tools")):
        return False
    text_chars = _as_int(item.get("text_chars"))
    if "text_chars_lt" in conditions and not (text_chars < _as_int(conditions["text_chars_lt"])):
        return False
    if "text_chars_gt" in conditions and not (text_chars > _as_int(conditions["text_chars_gt"])):
        return False
    if "text_chars_lte" in conditions and not (text_chars <= _as_int(conditions["text_chars_lte"])):
        return False
    if "text_chars_gte" in conditions and not (text_chars >= _as_int(conditions["text_chars_gte"])):
        return False
    if "min_text_chars" in conditions and not (text_chars >= _as_int(conditions["min_text_chars"])):
        return False
    if "max_text_chars" in conditions and not (text_chars <= _as_int(conditions["max_text_chars"])):
        return False
    return True


def _rule_entries(proposed: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(proposed, list):
        routing = {"rules": proposed}
        policy_source = None
    elif isinstance(proposed, dict) and isinstance(proposed.get("policies"), dict):
        routing = proposed.get("policies", {}).get("routing")
        if not isinstance(routing, dict):
            routing = {}
        policy_source = routing.get("policy_source") or (proposed.get("recommendation") or {}).get("policy_source")
    elif isinstance(proposed, dict):
        routing = proposed
        policy_source = routing.get("policy_source")
    else:
        routing = {}
        policy_source = None

    entries: list[dict[str, Any]] = []
    rules = routing.get("rules") if isinstance(routing.get("rules"), list) else []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        managed = rule.get("managed_recommendation") if isinstance(rule.get("managed_recommendation"), dict) else {}
        entries.append({
            "path": f"$.policies.routing.rules[{index}]",
            "source": "rules",
            "rule_id": rule.get("id") or rule.get("rule_id") or managed.get("policy_id"),
            "candidate_id": rule.get("candidate_id") or managed.get("candidate_id") or managed.get("recommendation_id"),
            "policy_source": rule.get("policy_source") or managed.get("policy_source") or policy_source,
            "conditions": rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {},
            "action": rule.get("action") if isinstance(rule.get("action"), dict) else {},
        })

    canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
    if canary:
        conditions: dict[str, Any] = {}
        if canary.get("model_pattern"):
            conditions["model_pattern"] = canary.get("model_pattern")
        phases = _value_list(canary.get("eligible_workflow_phases"))
        if phases:
            conditions["workflow_phase"] = phases
        excluded_categories = _value_list(canary.get("excluded_categories"))
        if excluded_categories:
            conditions["category_not_in"] = excluded_categories
        if canary.get("min_text_chars") is not None:
            conditions["min_text_chars"] = canary.get("min_text_chars")
        if canary.get("max_text_chars") is not None:
            conditions["max_text_chars"] = canary.get("max_text_chars")
        target_model = canary.get("target_model") or "haiku"
        entries.append({
            "path": "$.policies.routing.phase_canary",
            "source": "phase_canary",
            "rule_id": canary.get("policy_id"),
            "candidate_id": canary.get("candidate_id") or canary.get("recommendation_id"),
            "policy_source": canary.get("policy_source") or policy_source,
            "conditions": conditions,
            "action": {"route_to": target_model, "reason": "phase canary dry-run"},
        })
    return entries, policy_source


def load_phase_routing_policy(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if isinstance(payload, list):
        return {"policy_source": "local-manual", "rules": payload}
    if isinstance(payload, dict) and "policies" not in payload and "policy_source" not in payload:
        payload = dict(payload)
        payload["policy_source"] = "local-manual"
    return payload


def _risk_list(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"reason": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _add_count(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def build_phase_routing_dry_run(
    store: Any,
    proposed_policy: Any,
    *,
    limit: int = 1000,
    min_samples: int = 1,
    max_error_rate: float = 0.05,
    stale_hours: int = 168,
    require_shadow_support: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    rows = _load_recent_rows(store, limit=limit)
    features = [_row_features(row) for row in rows]
    entries, policy_source = _rule_entries(proposed_policy)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stale_before = now - timedelta(hours=max(1, int(stale_hours))) if stale_hours > 0 else None

    summaries: list[dict[str, Any]] = []
    total_excluded: dict[str, int] = {}
    warnings: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        conditions = entry.get("conditions") or {}
        action = entry.get("action") or {}
        target_model = _resolve_route_to(action.get("route_to"))
        matched = [item for item in features if _condition_matches(item, conditions)]
        excluded: dict[str, int] = {}
        projected: list[dict[str, Any]] = []
        current_cost = 0.0
        projected_target_cost = 0.0
        target_cost_known = 0

        for item in matched:
            item_current_cost = float(item.get("current_cost_usd") or 0.0)
            current_cost += item_current_cost
            reasons: list[str] = []
            if item.get("workflow_phase") in {"", "unknown", None}:
                reasons.append("missing_phase")
            if item.get("thinking"):
                reasons.append("thinking")
            if item.get("status_code", 0) >= 400:
                reasons.append("high_error_rate")
            if item.get("retry_count", 0) > 0:
                reasons.append("retry_history")
            if item.get("fallback"):
                reasons.append("fallback_history")
            if require_shadow_support and item.get("stream"):
                reasons.append("streaming_shadow_unsupported")
            created_at = _parse_dt(item.get("created_at"))
            if stale_before and (created_at is None or created_at < stale_before):
                reasons.append("stale_evidence")
            if target_model and _model_tier(item.get("routed_model")) == _model_tier(target_model):
                reasons.append("already_routed")
            if not target_model:
                reasons.append("no_baseline_support")
                target_cost = None
            else:
                target_cost = estimate_cost(
                    target_model,
                    _as_int(item.get("input_tokens")),
                    _as_int(item.get("output_tokens")),
                    _as_int(item.get("cache_creation_input_tokens")),
                    _as_int(item.get("cache_read_input_tokens")),
                    provider="anthropic",
                )
                if target_cost is None or item_current_cost <= 0:
                    reasons.append("no_baseline_support")
            if reasons:
                for reason in sorted(set(reasons)):
                    _add_count(excluded, reason)
                    _add_count(total_excluded, reason)
                continue
            projected.append(item)
            projected_target_cost += float(target_cost or 0.0)
            target_cost_known += 1

        if len(projected) < max(1, int(min_samples)):
            if projected:
                excluded["insufficient_samples"] = excluded.get("insufficient_samples", 0) + len(projected)
                total_excluded["insufficient_samples"] = total_excluded.get("insufficient_samples", 0) + len(projected)
            projected_target_cost = 0.0
            target_cost_known = 0
            projected = []

        matched_errors = sum(1 for item in matched if _as_int(item.get("status_code")) >= 400)
        matched_retries = sum(1 for item in matched if _as_int(item.get("retry_count")) > 0)
        historical_error_rate = (matched_errors / len(matched)) if matched else 0.0
        historical_retry_rate = (matched_retries / len(matched)) if matched else 0.0
        rule_warnings: list[dict[str, Any]] = []
        path = str(entry.get("path") or f"$.policies.routing.rules[{index}]")
        if historical_error_rate > max_error_rate:
            rule_warnings.append({
                "code": "high-error-rate",
                "path": path,
                "severity": "warning",
                "message": f"matched historical calls had {historical_error_rate:.1%} error rate",
            })
        if not any(str(conditions.get(key) or "") for key in ("workflow_phase",)):
            rule_warnings.append({
                "code": "missing-workflow-phase-condition",
                "path": path,
                "severity": "warning",
                "message": "phase-routing dry-run rule has no workflow_phase condition",
            })
        warnings.extend(rule_warnings)
        projected_current_cost = sum(float(item.get("current_cost_usd") or 0.0) for item in projected)
        summaries.append({
            "path": path,
            "source": entry.get("source"),
            "rule_id": entry.get("rule_id") or f"routing-rule-{index}",
            "candidate_id": entry.get("candidate_id"),
            "policy_source": entry.get("policy_source") or policy_source,
            "conditions": conditions,
            "action": {
                "route_to": action.get("route_to"),
                "target_model": target_model,
                "reason": action.get("reason"),
            },
            "matched_count": len(matched),
            "projected_candidate_count": len(projected),
            "excluded_count": sum(excluded.values()),
            "excluded_count_by_reason": _risk_list(excluded),
            "current_cost_usd": _round_usd(current_cost),
            "projected_current_cost_usd": _round_usd(projected_current_cost),
            "projected_target_cost_usd": _round_usd(projected_target_cost),
            "projected_savings_usd": _round_usd(max(0.0, projected_current_cost - projected_target_cost)),
            "cost_estimate_count": target_cost_known,
            "historical_error_rate": round(historical_error_rate, 6),
            "historical_retry_rate": round(historical_retry_rate, 6),
            "risk_warnings": rule_warnings,
        })

    return {
        "schema": DRY_RUN_SCHEMA,
        "ok": True,
        "dry_run": True,
        "wrote_local_files": False,
        "altered_provider_routing": False,
        "sampled_call_count": len(features),
        "limit": max(1, min(int(limit), 10_000)),
        "policy_source": policy_source,
        "rule_count": len(summaries),
        "summary": {
            "matched_count": sum(item["matched_count"] for item in summaries),
            "projected_candidate_count": sum(item["projected_candidate_count"] for item in summaries),
            "excluded_count": sum(item["excluded_count"] for item in summaries),
            "excluded_count_by_reason": _risk_list(total_excluded),
            "projected_savings_usd": _round_usd(sum(float(item["projected_savings_usd"]) for item in summaries)),
            "projected_target_cost_usd": _round_usd(sum(float(item["projected_target_cost_usd"]) for item in summaries)),
            "risk_warning_count": len(warnings),
            "candidate_rule_ids": [
                str(item.get("candidate_id") or item.get("rule_id"))
                for item in summaries
                if item.get("candidate_id") or item.get("rule_id")
            ],
        },
        "rules": summaries,
        "risk_warnings": warnings,
        "settings": {
            "min_samples": max(1, int(min_samples)),
            "max_error_rate": float(max_error_rate),
            "stale_hours": int(stale_hours),
            "require_shadow_support": bool(require_shadow_support),
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "raw_body_columns_read": False,
            "session_ids_included": False,
            "request_ids_included": False,
            "file_paths_included": False,
            "error_samples_included": False,
        },
    }


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
