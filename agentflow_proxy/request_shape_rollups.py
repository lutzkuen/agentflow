from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentflow_proxy.pricing import pricing_basis
from agentflow_proxy.public_metadata import public_label
from agentflow_proxy.store import stable_json, utc_now


SCHEMA = "agentflow.request_shape_rollups.v1"
ROLLUP_ROW_SCHEMA = "agentflow.request_shape_rollup_row.v1"
REPLAYABILITY_DRY_RUN_SCHEMA = "agentflow.request_shape_cache_replayability_dry_run.v1"
CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA = "agentflow.request_shape_crunch_opportunity_dry_run.v1"
CRUNCH_CANARY_ACTION_SCHEMA = "agentflow.request_shape_crunch_canary_action.v1"
CRUNCH_CANARY_APPLY_SCHEMA = "agentflow.request_shape_crunch_canary_apply.v1"
CRUNCH_CANARY_LIFECYCLE_SCHEMA = "agentflow.request_shape_crunch_canary_lifecycle.v1"
CRUNCH_CANARY_IMPACT_SCHEMA = "agentflow.request_shape_crunch_canary_impact.v1"
REPEATED_CONTEXT_TEXT_BUCKETS = {"8k_32k_chars", "32k_128k_chars", "gte_128k_chars"}
REPLAY_SUPPORTED_ENDPOINTS = {"messages", "responses", "chat_completions", "chat"}
REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE = 0.05
DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS = 72.0


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


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    label = public_label(key, "unknown")
    counter[label] = counter.get(label, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _provider_family(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "").strip().lower()
    if provider:
        return public_label(provider, "unknown")
    path = str(row.get("path") or "")
    if "responses" in path or "chat/completions" in path:
        return "openai"
    if "messages" in path:
        return "anthropic"
    return "unknown"


def _endpoint(row: dict[str, Any]) -> str:
    endpoint = row.get("endpoint")
    if endpoint:
        return public_label(endpoint, "unknown")
    path = str(row.get("path") or "")
    if "chat/completions" in path:
        return "chat_completions"
    if "responses" in path:
        return "responses"
    if "messages" in path:
        return "messages"
    return "unknown"


def _source_surface(row: dict[str, Any], provider: str, endpoint: str) -> str:
    source = row.get("source_surface")
    if source:
        return public_label(source, "unknown")
    if provider == "openai":
        return f"openai_{endpoint}"
    if provider == "anthropic":
        return "anthropic_messages"
    return "unknown"


def _model_family(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    if "claude" in text:
        for family in ("haiku", "sonnet", "opus"):
            if family in text:
                return f"claude-{family}"
        return "claude"
    if text.startswith("gpt-5"):
        return "gpt-5"
    if text.startswith("gpt-4"):
        return "gpt-4"
    return public_label(text, fallback)


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
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
    if tokens <= 0:
        return "unknown"
    if tokens < 500:
        return "lt_500_tokens"
    if tokens < 2_000:
        return "500_2k_tokens"
    if tokens < 8_000:
        return "2k_8k_tokens"
    if tokens < 32_000:
        return "8k_32k_tokens"
    return "gte_32k_tokens"


def _cost_bucket(cost: float) -> str:
    if cost <= 0:
        return "unknown"
    if cost < 0.001:
        return "lt_0_001_usd"
    if cost < 0.01:
        return "0_001_0_01_usd"
    if cost < 0.05:
        return "0_01_0_05_usd"
    if cost < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


def _savings_bucket(savings: float) -> str:
    if savings <= 0:
        return "none"
    if savings < 0.001:
        return "lt_0_001_usd"
    if savings < 0.01:
        return "0_001_0_01_usd"
    if savings < 0.05:
        return "0_01_0_05_usd"
    if savings < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


def _input_savings_usd(tokens: int, *, provider: str, model: str, fallback_cost: float = 0.0, fallback_tokens: int = 0) -> float:
    if tokens <= 0:
        return 0.0
    basis = pricing_basis(model, provider)
    price = _as_float(basis.get("input_usd_per_million"))
    if price > 0:
        return (tokens / 1_000_000.0) * price
    if fallback_cost > 0 and fallback_tokens > 0:
        return fallback_cost * (tokens / float(fallback_tokens))
    return 0.0


def _crunch_saved_tokens(crunch: dict[str, Any]) -> int:
    for key in ("tokens_saved_est", "saved_tokens", "tokens_saved", "crunch_tokens_saved"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    saved_chars = _crunch_saved_chars(crunch)
    return saved_chars // 4


def _crunch_saved_chars(crunch: dict[str, Any]) -> int:
    for key in ("saved_chars", "chars_saved", "crunch_chars_saved"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    before = _as_int(crunch.get("before_chars") or crunch.get("original_chars"))
    after = _as_int(crunch.get("after_chars") or crunch.get("result_chars"))
    if before > after > 0:
        return before - after
    return 0


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


def _retry_bucket(retries: int) -> str:
    if retries <= 0:
        return "0"
    if retries == 1:
        return "1"
    if retries <= 3:
        return "2_3"
    return "4_plus"


def _cache_status(row: dict[str, Any], cache: dict[str, Any]) -> str:
    status = str(cache.get("status") or "").strip().lower()
    if status:
        return public_label(status, "unknown")
    return "hit" if _as_int(row.get("cache_hit")) else "missing"


def _routing_status(row: dict[str, Any], routing: dict[str, Any]) -> str:
    requested = str(row.get("requested_model") or routing.get("requested_model") or "")
    routed = str(row.get("routed_model") or routing.get("routed_model") or requested)
    if requested and routed and requested != routed:
        return "routed"
    if routing.get("enabled") is False:
        return "disabled"
    return "passthrough"


def _has_tools(row: dict[str, Any], routing: dict[str, Any], cache: dict[str, Any]) -> bool:
    if routing.get("has_tools") is not None:
        return bool(routing.get("has_tools"))
    tool_features = routing.get("tool_features") if isinstance(routing.get("tool_features"), dict) else {}
    if tool_features.get("has_tools") is not None:
        return bool(tool_features.get("has_tools"))
    if cache.get("has_tools") is not None:
        return bool(cache.get("has_tools"))
    category = str(row.get("category") or routing.get("category") or "").lower()
    reason = str(cache.get("reason") or "").lower()
    return category.startswith("tool") or "tool" in reason


def _workflow_phase(row: dict[str, Any], routing: dict[str, Any]) -> str:
    for key in ("workflow_phase", "phase", "category"):
        value = routing.get(key)
        if value:
            return public_label(value, "unknown")
    return public_label(row.get("category"), "unknown")


def _blocker_codes(
    *,
    row: dict[str, Any],
    cache: dict[str, Any],
    routing: dict[str, Any],
    cache_status: str,
    routing_status: str,
    stream: bool,
    has_tools: bool,
) -> list[str]:
    blockers: set[str] = set()
    reason = str(cache.get("reason") or "").lower()
    routing_reason = str(routing.get("reason") or "").lower()
    status_bucket = _status_bucket(row.get("status_code"))
    if stream:
        blockers.add("unsupported-streaming-shape")
    if has_tools and ("tools-disabled" in reason or "tool" in reason and "disabled" in reason):
        blockers.add("tool-call-cache-disabled")
    if "semantic" in reason and "disabled" in reason:
        blockers.add("semantic-cache-disabled")
    if cache_status in {"miss", "missing"} or "exact-miss" in reason:
        blockers.add("exact-cache-miss")
    if cache_status == "skipped" and not blockers:
        blockers.add("cache-skipped")
    if cache_status == "hit":
        blockers.add("already-cache-hit")
    if "thinking" in routing_reason:
        blockers.add("thinking-routing-guard")
    if "rate" in routing_reason or status_bucket in {"4xx", "5xx"} and _as_int(row.get("retry_count")) > 0:
        blockers.add("rate-or-error-pressure")
    if routing_status == "passthrough" and not blockers:
        blockers.add("routing-rule-required")
    return sorted(public_label(code, "unknown") for code in blockers if code)


def _candidate_families(
    *,
    cache_status: str,
    routing_status: str,
    blockers: list[str],
    observed_savings: float,
    cost: float,
) -> list[str]:
    families: set[str] = set()
    if cache_status != "hit":
        families.add("cache_replay")
    if any(
        code.startswith("exact-cache")
        or code.startswith("cache-")
        or code in {"unsupported-streaming-shape", "tool-call-cache-disabled", "semantic-cache-disabled"}
        for code in blockers
    ):
        families.add("cache_blocker")
    if routing_status == "passthrough" and cost > 0:
        families.add("routing_candidate")
    if routing_status == "routed" or observed_savings > 0:
        families.add("routing_evidence")
    return sorted(families or {"observability"})


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = stable_json(basis)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    provider = str(basis.get("provider_family") or "unknown").replace("_", "-")
    endpoint = str(basis.get("endpoint") or "unknown").replace("_", "-")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"request-shape:{provider}:{endpoint}:{category}:{digest}"


def _crunch_canary_cohort_id(row: dict[str, Any]) -> str:
    basis = {
        "provider_family": row.get("provider_family"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "category": row.get("category"),
        "workflow_phase": row.get("workflow_phase"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": row.get("text_bucket"),
        "token_bucket": row.get("token_bucket"),
        "cache_status": row.get("cache_status"),
        "routing_status": row.get("routing_status"),
    }
    digest = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:16]
    provider = str(row.get("provider_family") or "unknown").replace("_", "-")
    endpoint = str(row.get("endpoint") or "unknown").replace("_", "-")
    category = str(row.get("category") or "unknown").replace("_", "-")
    return f"request-shape-crunch:{provider}:{endpoint}:{category}:{digest}"


def _crunch_canary_policy_id(cohort_id: str) -> str:
    digest = hashlib.sha256(cohort_id.encode("utf-8")).hexdigest()[:12]
    return f"local-repeated-context-crunch-canary-{digest}"


def _crunch_canary_lifecycle_from_meta(crunch: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "request_shape_repeated_context_canary",
        "repeated_context_crunch_canary",
        "request_shape_crunch_canary",
        "crunch_canary",
    ):
        meta = crunch.get(key)
        if isinstance(meta, dict):
            status = public_label(meta.get("status") or meta.get("lifecycle") or meta.get("cohort"), "unknown")
            cohort = public_label(meta.get("cohort") or status, "unknown")
            if status in {"canary-applied", "canary_applied"}:
                status = "applied"
            elif status in {"canary-holdout", "canary_holdout"}:
                status = "holdout"
            if cohort == "canary-applied":
                cohort = "canary_applied"
            elif cohort == "canary-holdout":
                cohort = "canary_holdout"
            return {
                "status": status,
                "cohort": cohort,
                "policy_id": public_label(meta.get("policy_id"), "unknown"),
                "cohort_id": public_label(meta.get("cohort_id"), "unknown"),
                "safety_stop": bool(meta.get("safety_stop")) or status == "safety-stopped",
            }
    return None


def _crunch_canary_cohort_name(lifecycle: dict[str, Any]) -> str:
    status = str(lifecycle.get("status") or "").strip().lower().replace("_", "-")
    cohort = str(lifecycle.get("cohort") or "").strip().lower().replace("-", "_")
    if status == "applied" or cohort == "canary_applied":
        return "canary_applied"
    if status == "holdout" or cohort == "canary_holdout":
        return "canary_holdout"
    if bool(lifecycle.get("safety_stop")) or status in {"safety-stopped", "safety-stop"} or cohort in {"safety_stopped", "safety_stop"}:
        return "safety_stopped"
    if status in {"fallback", "fallback-applied"} or cohort in {"fallback", "fallback_applied"}:
        return "fallback"
    if status in {"rollback", "rollback-required"} or cohort in {"rollback", "rollback_required"}:
        return "rollback"
    if status in {"skipped", "disabled", "ineligible"} or cohort in {"skipped", "bypassed_or_disabled"}:
        return "skipped"
    return "unknown"


def _crunch_before_chars(crunch: dict[str, Any], *, text_chars: int, input_tokens: int) -> int:
    for key in ("before_chars", "original_chars", "input_chars", "text_chars_before"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    saved = _crunch_saved_chars(crunch)
    for key in ("after_chars", "result_chars", "output_chars", "text_chars_after"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value + saved
    return max(text_chars, input_tokens * 4 if input_tokens > 0 else 0, saved)


def _crunch_after_chars(crunch: dict[str, Any], before_chars: int) -> int:
    for key in ("after_chars", "result_chars", "output_chars", "text_chars_after"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    saved = _crunch_saved_chars(crunch)
    return max(0, before_chars - saved)


def _crunch_savings_usd(
    crunch: dict[str, Any],
    *,
    tokens_saved: int,
    provider: str,
    model: str,
    fallback_cost: float,
    fallback_tokens: int,
) -> float:
    for key in ("savings_usd", "saved_usd", "crunch_savings_usd", "estimated_savings_usd"):
        value = _as_float(crunch.get(key))
        if value > 0:
            return value
    return _input_savings_usd(
        tokens_saved,
        provider=provider,
        model=model,
        fallback_cost=fallback_cost,
        fallback_tokens=fallback_tokens,
    )


def _empty_crunch_impact_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "before_chars": 0,
        "after_chars": 0,
        "saved_chars": 0,
        "saved_tokens": 0,
        "saved_usd": 0.0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "safety_stop_count": 0,
        "rollback_count": 0,
    }


def _empty_crunch_impact_candidate(policy_id: str, cohort_id: str, row: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "cohort_metadata": {
            "provider_family": row.get("provider_family"),
            "source_surface": row.get("source_surface"),
            "endpoint": row.get("endpoint"),
            "category": row.get("category"),
            "workflow_phase": row.get("workflow_phase"),
            "stream": bool(row.get("stream")),
            "has_tools": bool(row.get("has_tools")),
            "text_bucket": row.get("text_bucket"),
            "token_bucket": row.get("token_bucket"),
            "cache_status": row.get("cache_status"),
            "routing_status": row.get("routing_status"),
        },
        "policy_source": public_label(lifecycle.get("policy_source") or "local-manual", "local-manual"),
        "cohorts": {
            "canary_applied": _empty_crunch_impact_cohort(),
            "canary_holdout": _empty_crunch_impact_cohort(),
            "safety_stopped": _empty_crunch_impact_cohort(),
            "fallback": _empty_crunch_impact_cohort(),
            "rollback": _empty_crunch_impact_cohort(),
            "skipped": _empty_crunch_impact_cohort(),
            "unknown": _empty_crunch_impact_cohort(),
        },
        "status_counts": {},
        "reason_counts": {},
        "first_observed_at": None,
        "latest_observed_at": None,
    }


def _add_crunch_impact_row(candidate: dict[str, Any], row: dict[str, Any], crunch: dict[str, Any], lifecycle: dict[str, Any]) -> None:
    cohort_name = _crunch_canary_cohort_name(lifecycle)
    cohort = candidate["cohorts"].setdefault(cohort_name, _empty_crunch_impact_cohort())
    input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    text_chars = _as_int(row.get("text_chars"))
    before_chars = _crunch_before_chars(crunch, text_chars=text_chars, input_tokens=input_tokens)
    after_chars = _crunch_after_chars(crunch, before_chars)
    saved_chars = _crunch_saved_chars(crunch)
    saved_tokens = _crunch_saved_tokens(crunch)
    model = str(row.get("routed_model") or row.get("requested_model") or "")
    saved_usd = _crunch_savings_usd(
        crunch,
        tokens_saved=saved_tokens,
        provider=str(row.get("provider_family") or "unknown"),
        model=model,
        fallback_cost=_as_float(row.get("cost_est_usd")),
        fallback_tokens=input_tokens,
    )
    cohort["count"] += 1
    cohort["before_chars"] += before_chars
    cohort["after_chars"] += after_chars
    cohort["saved_chars"] += saved_chars
    cohort["saved_tokens"] += saved_tokens
    cohort["saved_usd"] += saved_usd
    if _status_bucket(row.get("status_code")) in {"4xx", "5xx"}:
        cohort["error_count"] += 1
    cohort["retry_count"] += _as_int(row.get("retry_count"))
    if cohort_name == "fallback":
        cohort["fallback_count"] += 1
    if cohort_name == "safety_stopped":
        cohort["safety_stop_count"] += 1
    if cohort_name == "rollback":
        cohort["rollback_count"] += 1
    _increment(candidate["status_counts"], cohort_name)
    _increment(candidate["reason_counts"], lifecycle.get("reason") or cohort_name)
    created_at = str(row.get("created_at") or "")
    if created_at:
        first = candidate.get("first_observed_at")
        latest = candidate.get("latest_observed_at")
        candidate["first_observed_at"] = created_at if first is None else min(str(first), created_at)
        candidate["latest_observed_at"] = created_at if latest is None else max(str(latest), created_at)


def _finalize_crunch_impact_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    errors = _as_int(raw.get("error_count"))
    retries = _as_int(raw.get("retry_count"))
    return {
        "count": count,
        "before_chars": _as_int(raw.get("before_chars")),
        "after_chars": _as_int(raw.get("after_chars")),
        "saved_chars": _as_int(raw.get("saved_chars")),
        "saved_tokens": _as_int(raw.get("saved_tokens")),
        "saved_usd": round(_as_float(raw.get("saved_usd")), 8),
        "error_count": errors,
        "retry_count": retries,
        "fallback_count": _as_int(raw.get("fallback_count")),
        "safety_stop_count": _as_int(raw.get("safety_stop_count")),
        "rollback_count": _as_int(raw.get("rollback_count")),
        "error_rate": round(errors / count, 6) if count else 0.0,
        "retry_rate": round(retries / count, 6) if count else 0.0,
        "avg_saved_tokens": round(_as_int(raw.get("saved_tokens")) / count, 2) if count else 0.0,
        "avg_saved_usd": round(_as_float(raw.get("saved_usd")) / count, 8) if count else 0.0,
    }


def _crunch_impact_stale(latest_observed_at: str | None, *, max_age_hours: float) -> bool:
    latest = _parse_utc(latest_observed_at)
    if latest is None:
        return False
    age_hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600.0
    return age_hours > max_age_hours


def _crunch_impact_verdict(
    *,
    applied: dict[str, Any],
    holdout: dict[str, Any],
    safety: dict[str, Any],
    fallback: dict[str, Any],
    rollback: dict[str, Any],
    stale: bool,
) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    if _as_int(rollback.get("count")) or _as_int(rollback.get("rollback_count")):
        reasons.append("rollback-observed")
    if _as_int(safety.get("count")) or _as_int(safety.get("safety_stop_count")):
        reasons.append("canary-safety-stopped")
    if stale:
        reasons.append("stale-canary-impact-evidence")
    if _as_int(applied.get("count")) <= 0:
        reasons.append("missing-applied-coverage")
    if _as_int(holdout.get("count")) <= 0:
        reasons.append("missing-holdout-coverage")
    if _as_int(applied.get("count")) > 0 and _as_int(applied.get("saved_tokens")) <= 0 and _as_float(applied.get("saved_usd")) <= 0:
        reasons.append("no-applied-savings")
    if _as_float(applied.get("error_rate")) > _as_float(holdout.get("error_rate")):
        reasons.append("error-rate-regression")
    if _as_float(applied.get("retry_rate")) > _as_float(holdout.get("retry_rate")):
        reasons.append("retry-rate-regression")
    if _as_int(fallback.get("count")) or _as_int(fallback.get("fallback_count")):
        reasons.append("fallback-observed")
    if reasons:
        return "no-widen", sorted(set(reasons), key=reasons.index), reasons[0]
    return "widen-ready", ["applied-savings-with-holdout-no-regression"], "ready-to-widen-repeated-context-crunch-canary"


def _crunch_impact_activation_lifecycle_feedback(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    family_state_counts: dict[str, int] = {}
    metadata: list[dict[str, Any]] = []
    for candidate in candidates:
        verdict = str(candidate.get("verdict") or "unknown")
        state = "healthy_canary" if verdict == "widen-ready" else "suppressed"
        _increment(state_counts, state)
        _increment(family_state_counts, f"crunch:{state}")
        cohorts = candidate.get("cohorts") if isinstance(candidate.get("cohorts"), dict) else {}
        for name, cohort in cohorts.items():
            count = _as_int(cohort.get("count")) if isinstance(cohort, dict) else 0
            if count:
                _increment(cohort_counts, name, count)
        metadata.append(
            {
                "policy_ref": public_label(candidate.get("policy_id"), "unknown"),
                "cohort_label": "canary",
                "action_family": "crunch",
                "event_count": _as_int(candidate.get("observed_count")),
                "applied_count": _as_int(candidate.get("applied_count")),
                "holdout_count": _as_int(candidate.get("holdout_count")),
                "fallback_count": _as_int(candidate.get("fallback_count")),
                "error_count": _as_int(candidate.get("applied_error_count")),
                "retry_count": _as_int(candidate.get("applied_retry_count")),
                "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
                "savings_estimate_usd": round(_as_float(candidate.get("saved_usd")), 8),
                "reason_codes": candidate.get("reason_codes") or [],
                "blocker_reason_breakdown": [
                    {"value": reason, "count": 1}
                    for reason in candidate.get("reason_codes") or []
                ],
            }
        )
    return {
        "schema": "agentflow.activation_staged_lifecycle_feedback_summary.v1",
        "queue_rows": 0,
        "family_event_count": sum(_as_int(item.get("observed_count")) for item in candidates),
        "state_breakdown": _breakdown(state_counts),
        "event_phase_breakdown": [{"value": "impact", "count": len(candidates)}] if candidates else [],
        "cohort_breakdown": _breakdown(cohort_counts),
        "family_state_breakdown": _breakdown(family_state_counts),
        "candidate_id_breakdown": [],
        "cohort_lifecycle_metadata": metadata[:50],
        "payload_json_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_canary_impact_report(
    rows: list[dict[str, Any]],
    *,
    max_evidence_age_hours: float = DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS,
) -> dict[str, Any]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    observed_rows = 0
    for row in rows:
        crunch = _json_obj(row.get("crunch_json"))
        lifecycle = _crunch_canary_lifecycle_from_meta(crunch)
        if lifecycle is None:
            continue
        policy_id = public_label(lifecycle.get("policy_id"), "unknown")
        cohort_id = public_label(lifecycle.get("cohort_id"), "unknown")
        key = (policy_id, cohort_id)
        candidate = candidates.setdefault(key, _empty_crunch_impact_candidate(policy_id, cohort_id, row, lifecycle))
        _add_crunch_impact_row(candidate, row, crunch, lifecycle)
        observed_rows += 1

    finalized: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for raw in candidates.values():
        cohorts = {
            key: _finalize_crunch_impact_cohort(value)
            for key, value in raw["cohorts"].items()
        }
        applied = cohorts["canary_applied"]
        holdout = cohorts["canary_holdout"]
        fallback = cohorts["fallback"]
        safety = cohorts["safety_stopped"]
        rollback = cohorts["rollback"]
        stale = _crunch_impact_stale(
            raw.get("latest_observed_at"),
            max_age_hours=max(0.0, _as_float(max_evidence_age_hours, DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS)),
        )
        verdict, reasons, top_blocker = _crunch_impact_verdict(
            applied=applied,
            holdout=holdout,
            safety=safety,
            fallback=fallback,
            rollback=rollback,
            stale=stale,
        )
        _increment(verdict_counts, verdict)
        if verdict != "widen-ready":
            for reason in reasons:
                _increment(blocker_counts, reason)
        finalized.append(
            {
                "schema": "agentflow.request_shape_crunch_canary_impact_candidate.v1",
                "policy_id": raw["policy_id"],
                "cohort_id": raw["cohort_id"],
                "cohort_metadata": raw["cohort_metadata"],
                "policy_source": raw["policy_source"],
                "observed_count": sum(_as_int(cohort.get("count")) for cohort in cohorts.values()),
                "applied_count": _as_int(applied.get("count")),
                "holdout_count": _as_int(holdout.get("count")),
                "fallback_count": _as_int(fallback.get("count")),
                "safety_stop_count": _as_int(safety.get("count")),
                "rollback_count": _as_int(rollback.get("count")),
                "saved_chars": _as_int(applied.get("saved_chars")),
                "saved_tokens": _as_int(applied.get("saved_tokens")),
                "saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                "applied_error_count": _as_int(applied.get("error_count")),
                "holdout_error_count": _as_int(holdout.get("error_count")),
                "applied_retry_count": _as_int(applied.get("retry_count")),
                "holdout_retry_count": _as_int(holdout.get("retry_count")),
                "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
                "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
                "fallback_rate_delta": round(
                    (_as_int(fallback.get("count")) / max(1, _as_int(applied.get("count"))))
                    - 0.0,
                    6,
                ),
                "stale_evidence": {
                    "stale": stale,
                    "max_age_hours": round(max(0.0, _as_float(max_evidence_age_hours, DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS)), 6),
                },
                "first_observed_at": raw.get("first_observed_at"),
                "latest_observed_at": raw.get("latest_observed_at"),
                "verdict": verdict,
                "top_blocker": top_blocker if verdict != "widen-ready" else None,
                "reason_codes": reasons,
                "status_breakdown": _breakdown(raw.get("status_counts", {})),
                "reason_breakdown": _breakdown(raw.get("reason_counts", {})),
                "cohorts": cohorts,
                "privacy": _crunch_opportunity_privacy(),
            }
        )

    finalized.sort(
        key=lambda item: (
            item.get("verdict") == "widen-ready",
            _as_float(item.get("saved_usd")),
            _as_int(item.get("saved_tokens")),
            _as_int(item.get("observed_count")),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(finalized, start=1):
        candidate["rank"] = rank
    blocker_breakdown = _breakdown(blocker_counts)
    status = "no-crunch-canary-impact-metadata"
    if finalized:
        status = "widen-ready" if any(item.get("verdict") == "widen-ready" for item in finalized) else "no-widen"
    return {
        "schema": CRUNCH_CANARY_IMPACT_SCHEMA,
        "status": status,
        "summary": {
            "candidate_count": len(finalized),
            "observed_canary_metadata_row_count": observed_rows,
            "applied_count": sum(_as_int(item.get("applied_count")) for item in finalized),
            "holdout_count": sum(_as_int(item.get("holdout_count")) for item in finalized),
            "saved_chars": sum(_as_int(item.get("saved_chars")) for item in finalized),
            "saved_tokens": sum(_as_int(item.get("saved_tokens")) for item in finalized),
            "saved_usd": round(sum(_as_float(item.get("saved_usd")) for item in finalized), 8),
            "projected_saved_chars": sum(_as_int(item.get("saved_chars")) for item in finalized),
            "projected_saved_tokens": sum(_as_int(item.get("saved_tokens")) for item in finalized),
            "projected_saved_usd": round(sum(_as_float(item.get("saved_usd")) for item in finalized), 8),
            "error_rate_delta": round(max((_as_float(item.get("error_rate_delta")) for item in finalized), default=0.0), 6),
            "retry_rate_delta": round(max((_as_float(item.get("retry_rate_delta")) for item in finalized), default=0.0), 6),
            "fallback_count": sum(_as_int(item.get("fallback_count")) for item in finalized),
            "safety_stop_count": sum(_as_int(item.get("safety_stop_count")) for item in finalized),
            "rollback_count": sum(_as_int(item.get("rollback_count")) for item in finalized),
            "widen_ready_count": sum(1 for item in finalized if item.get("verdict") == "widen-ready"),
            "no_widen_count": sum(1 for item in finalized if item.get("verdict") != "widen-ready"),
            "top_blocker_code": blocker_breakdown[0]["value"] if blocker_breakdown else None,
            "next_action": "widen-repeated-context-crunch-canary" if status == "widen-ready" else "review-repeated-context-crunch-canary-impact-blocker",
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "verdict_breakdown": _breakdown(verdict_counts),
        "blocker_reason_breakdown": blocker_breakdown,
        "candidates": finalized,
        "activation_lifecycle_feedback": _crunch_impact_activation_lifecycle_feedback(finalized),
        "privacy": _crunch_opportunity_privacy(),
    }


def _new_group(basis: dict[str, Any], *, candidate_id: str, rollup_key: str) -> dict[str, Any]:
    return {
        "schema": ROLLUP_ROW_SCHEMA,
        "rollup_key": rollup_key,
        "candidate_id": candidate_id,
        **basis,
        "row_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "cache_hit_count": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "observed_savings_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "successful_input_tokens": 0,
        "input_token_cost_usd": 0.0,
        "current_crunch_tokens_saved": 0,
        "current_crunch_chars_saved": 0,
        "current_crunch_savings_usd": 0.0,
        "candidate_family_counts": {},
        "blocker_counts": {},
        "status_counts": {},
        "retry_bucket_counts": {},
        "cost_bucket_counts": {},
        "savings_bucket_counts": {},
        "cache_reason_counts": {},
        "crunch_canary_lifecycle_counts": {},
        "crunch_canary_policy_counts": {},
    }


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    candidate_family_counts = group.pop("candidate_family_counts", {})
    blocker_counts = group.pop("blocker_counts", {})
    crunch_canary_lifecycle_counts = group.pop("crunch_canary_lifecycle_counts", {})
    crunch_canary_policy_counts = group.pop("crunch_canary_policy_counts", {})
    candidate_families = sorted(candidate_family_counts)
    blocker_codes = sorted(blocker_counts)
    candidate_classes = _candidate_work_classes(
        row_count=_as_int(group.get("row_count")),
        text_bucket=str(group.get("text_bucket") or "unknown"),
        token_bucket=str(group.get("token_bucket") or "unknown"),
        candidate_families=candidate_families,
        blocker_codes=blocker_codes,
        routing_status=str(group.get("routing_status") or "unknown"),
        observed_savings=_as_float(group.get("observed_savings_usd")),
    )
    metadata = {
        "schema": "agentflow.request_shape_rollup_metadata.v1",
        "status_breakdown": _breakdown(group.pop("status_counts", {})),
        "retry_bucket_breakdown": _breakdown(group.pop("retry_bucket_counts", {})),
        "cost_bucket_breakdown": _breakdown(group.pop("cost_bucket_counts", {})),
        "savings_bucket_breakdown": _breakdown(group.pop("savings_bucket_counts", {})),
        "cache_reason_breakdown": _breakdown(group.pop("cache_reason_counts", {})),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "crunch_canary_lifecycle_breakdown": _breakdown(crunch_canary_lifecycle_counts),
        "crunch_canary_policy_breakdown": _breakdown(crunch_canary_policy_counts),
        "candidate_class_breakdown": [{"value": value, "count": _as_int(group.get("row_count"))} for value in candidate_classes],
        "raw_body_required": False,
        "aggregate_only": True,
    }
    group["candidate_families"] = candidate_families
    group["candidate_work_classes"] = candidate_classes
    group["blocker_codes"] = blocker_codes
    group["cost_est_usd"] = round(_as_float(group.get("cost_est_usd")), 6)
    group["baseline_cost_usd"] = round(_as_float(group.get("baseline_cost_usd")), 6)
    group["observed_savings_usd"] = round(_as_float(group.get("observed_savings_usd")), 6)
    group["input_token_cost_usd"] = round(_as_float(group.get("input_token_cost_usd")), 6)
    group["current_crunch_savings_usd"] = round(_as_float(group.get("current_crunch_savings_usd")), 6)
    repeated_weight = 0.0
    row_count = _as_int(group.get("row_count"))
    if row_count > 1:
        repeated_weight = (row_count - 1) / float(row_count)
    if "repeated_context" in candidate_classes and "crunch" in candidate_classes:
        projected_tokens = int(
            _as_int(group.get("successful_input_tokens"))
            * REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE
            * repeated_weight
        )
        projected_savings = (
            _as_float(group.get("input_token_cost_usd"))
            * REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE
            * repeated_weight
        )
    else:
        projected_tokens = 0
        projected_savings = 0.0
    group["projected_crunch_tokens_saved"] = max(0, projected_tokens)
    group["projected_crunch_chars_saved"] = max(0, projected_tokens * 4)
    group["projected_crunch_savings_usd"] = round(max(0.0, projected_savings), 6)
    group["crunch_canary_lifecycle"] = {
        "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
        "cohort_id": _crunch_canary_cohort_id(group),
        "policy_id": _crunch_canary_policy_id(_crunch_canary_cohort_id(group)),
        "applied_count": _as_int(crunch_canary_lifecycle_counts.get("applied"))
        + _as_int(crunch_canary_lifecycle_counts.get("canary_applied")),
        "holdout_count": _as_int(crunch_canary_lifecycle_counts.get("holdout"))
        + _as_int(crunch_canary_lifecycle_counts.get("canary_holdout")),
        "skipped_count": _as_int(crunch_canary_lifecycle_counts.get("skipped")),
        "safety_stopped_count": _as_int(crunch_canary_lifecycle_counts.get("safety-stopped"))
        + _as_int(crunch_canary_lifecycle_counts.get("safety_stop")),
        "fallback_count": _as_int(crunch_canary_lifecycle_counts.get("fallback")),
        "rollback_count": _as_int(crunch_canary_lifecycle_counts.get("rollback")),
        "status_breakdown": _breakdown(crunch_canary_lifecycle_counts),
        "policy_breakdown": _breakdown(crunch_canary_policy_counts),
        "metadata_only": True,
        "aggregate_only": True,
    }
    group["metadata"] = metadata
    group["privacy"] = {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
    }
    return group


def _replayability_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _crunch_opportunity_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _shape_crunch_decision(row: dict[str, Any]) -> dict[str, Any]:
    row_count = _as_int(row.get("row_count") or row.get("count"))
    classes = {str(item) for item in row.get("candidate_work_classes") or []}
    text_bucket = str(row.get("text_bucket") or "unknown")
    token_bucket = str(row.get("token_bucket") or "unknown")
    projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    observed_tokens = _as_int(row.get("current_crunch_tokens_saved"))
    lifecycle = row.get("crunch_canary_lifecycle") if isinstance(row.get("crunch_canary_lifecycle"), dict) else {}
    applied_count = _as_int(lifecycle.get("applied_count"))
    holdout_count = _as_int(lifecycle.get("holdout_count"))
    safety_stopped_count = _as_int(lifecycle.get("safety_stopped_count"))
    blockers: set[str] = set()

    if safety_stopped_count > 0:
        return {
            "readiness": "canary-safety-stopped",
            "reason": "repeated-context-crunch-canary-safety-stopped",
            "blockers": ["canary-safety-stopped"],
        }
    if applied_count > 0 and holdout_count > 0:
        return {
            "readiness": "canary-staged",
            "reason": "repeated-context-crunch-canary-applied-and-holdout",
            "blockers": [],
        }
    if applied_count > 0:
        return {
            "readiness": "canary-applied",
            "reason": "repeated-context-crunch-canary-applied",
            "blockers": [],
        }
    if holdout_count > 0:
        return {
            "readiness": "canary-holdout",
            "reason": "repeated-context-crunch-canary-holdout",
            "blockers": [],
        }

    if row_count < 2:
        blockers.add("insufficient-repeat-evidence")
    if text_bucket not in REPEATED_CONTEXT_TEXT_BUCKETS and token_bucket not in {"8k_32k_tokens", "gte_32k_tokens"}:
        blockers.add("not-large-context")
    if "crunch" not in classes and "repeated_context" not in classes:
        blockers.add("not-crunch-work-class")
    if _as_int(row.get("successful_input_tokens") or row.get("input_tokens")) <= 0:
        blockers.add("missing-token-metadata")
    if projected_tokens <= 0 and observed_tokens <= 0:
        blockers.add("non-positive-projection")

    if not blockers:
        return {
            "readiness": "measurement-ready",
            "reason": "repeated-context-crunch-opportunity",
            "blockers": [],
        }

    reason_priority = (
        "missing-token-metadata",
        "insufficient-repeat-evidence",
        "not-large-context",
        "not-crunch-work-class",
        "non-positive-projection",
    )
    return {
        "readiness": "skipped",
        "reason": next((item for item in reason_priority if item in blockers), sorted(blockers)[0]),
        "blockers": sorted(blockers),
    }


def _bounded_fraction(value: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _request_shape_crunch_canary_action(
    cohort: dict[str, Any],
    *,
    candidate_count: int,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    cohort_id = str(cohort.get("cohort_id") or _crunch_canary_cohort_id(cohort))
    policy_id = _crunch_canary_policy_id(cohort_id)
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    return {
        "schema": CRUNCH_CANARY_ACTION_SCHEMA,
        "action_type": "stage-local-repeated-context-crunch-canary",
        "target_local_policy": "crunch_rules",
        "policy_section": "crunch",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "candidate_rule": cohort.get("candidate_rule"),
        "candidate_count": candidate_count,
        "cohort_row_count": _as_int(cohort.get("row_count")),
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "canary_fraction": round(rollout, 6),
        "policy_source": "local-manual",
        "conditions": {
            "provider_family": cohort.get("provider_family"),
            "source_surface": cohort.get("source_surface"),
            "endpoint": cohort.get("endpoint"),
            "category": cohort.get("category"),
            "workflow_phase": cohort.get("workflow_phase"),
            "stream": bool(cohort.get("stream")),
            "has_tools": bool(cohort.get("has_tools")),
            "text_bucket": cohort.get("text_bucket"),
            "token_bucket": cohort.get("token_bucket"),
            "cache_status": cohort.get("cache_status"),
            "routing_status": cohort.get("routing_status"),
        },
        "projected_saved_chars": _as_int(cohort.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(cohort.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(cohort.get("projected_saved_usd")), 6),
        "safety_gates": {
            "metadata_only": True,
            "aggregate_only": True,
            "local_file_backed": True,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "holdout_required": holdout > 0,
            "max_rollout_fraction": DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
            "records_applied_holdout_skipped_safety_stopped_fallback_rollback": True,
        },
        "next_action": "apply-local-crunch-canary-after-review",
        "privacy": _crunch_opportunity_privacy(),
    }


def request_shape_crunch_canary_lifecycle(action: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    conditions = action.get("conditions") if isinstance(action.get("conditions"), dict) else {}
    cohort_id = str(action.get("cohort_id") or _crunch_canary_cohort_id(conditions))
    policy_id = str(action.get("policy_id") or _crunch_canary_policy_id(cohort_id))
    mismatch = [
        key
        for key, expected in conditions.items()
        if expected is not None and features.get(key) is not None and features.get(key) != expected
    ]
    base = {
        "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "metadata_only": True,
    }
    if mismatch:
        return {
            **base,
            "status": "skipped",
            "cohort": "skipped",
            "reason": "cohort-mismatch",
            "mismatched_conditions": sorted(public_label(item, "unknown") for item in mismatch),
        }

    rollout = _bounded_fraction(action.get("rollout_fraction", action.get("canary_fraction")), DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(action.get("holdout_fraction"), DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    unit = str(features.get("request_fingerprint") or features.get("cohort_sample_id") or stable_json({
        key: features.get(key)
        for key in sorted(conditions)
        if features.get(key) is not None
    }))
    material = stable_json({"policy_id": policy_id, "cohort_id": cohort_id, "unit": unit})
    bucket = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if bucket < rollout:
        status = "applied"
        cohort = "canary_applied"
        reason = "selected-canary"
    elif bucket < rollout + holdout:
        status = "holdout"
        cohort = "canary_holdout"
        reason = "selected-holdout"
    else:
        status = "skipped"
        cohort = "skipped"
        reason = "outside-canary-and-holdout"
    return {
        **base,
        "status": status,
        "cohort": cohort,
        "reason": reason,
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "bucket": round(bucket, 8),
        "cohort_key_hash": "sha256:" + hashlib.sha256(stable_json({
            "policy_id": policy_id,
            "cohort_id": cohort_id,
            "unit": unit,
        }).encode("utf-8")).hexdigest(),
    }


def apply_request_shape_crunch_canary_action(
    action: dict[str, Any],
    *,
    rules_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if action.get("schema") != CRUNCH_CANARY_ACTION_SCHEMA:
        errors.append({"path": "$.schema", "message": "unsupported request-shape crunch canary action schema"})
    if action.get("action_type") != "stage-local-repeated-context-crunch-canary":
        errors.append({"path": "$.action_type", "message": "unsupported request-shape crunch canary action type"})
    if action.get("target_local_policy") != "crunch_rules":
        errors.append({"path": "$.target_local_policy", "message": "request-shape crunch canary must target crunch_rules"})
    privacy = action.get("privacy") if isinstance(action.get("privacy"), dict) else {}
    safety = action.get("safety_gates") if isinstance(action.get("safety_gates"), dict) else {}
    for key in ("raw_prompts_included", "provider_bodies_included", "request_ids_included", "session_ids_included"):
        if privacy.get(key) or safety.get(key):
            errors.append({"path": f"$.privacy.{key}", "message": "request-shape crunch canary action is not metadata-only"})
    if _as_float(action.get("projected_saved_tokens")) <= 0 and _as_float(action.get("projected_saved_chars")) <= 0:
        errors.append({"path": "$.projected_saved_tokens", "message": "request-shape crunch canary needs positive projected savings"})

    path = Path(rules_path)
    existing: dict[str, Any] = {}
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing = loaded if isinstance(loaded, dict) else {}
    if errors:
        return {
            "schema": CRUNCH_CANARY_APPLY_SCHEMA,
            "ok": False,
            "dry_run": bool(dry_run),
            "wrote_policy_files": False,
            "rules_path_included": False,
            "errors": errors,
            "privacy": _crunch_opportunity_privacy(),
        }

    policy_id = str(action.get("policy_id") or _crunch_canary_policy_id(str(action.get("cohort_id") or "")))
    canary_rule = {
        "id": policy_id,
        "enabled": True,
        "policy_source": "local-manual",
        "cohort_id": action.get("cohort_id"),
        "conditions": action.get("conditions") if isinstance(action.get("conditions"), dict) else {},
        "rollout": {
            "schema": "agentflow.request_shape_crunch_canary_rollout.v1",
            "canary_enabled": True,
            "canary_fraction": _bounded_fraction(action.get("rollout_fraction"), DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION),
            "holdout_fraction": _bounded_fraction(action.get("holdout_fraction"), DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION),
            "canary_salt": policy_id,
            "canary_unit": "request_shape_cohort",
        },
        "projected_saved_chars": _as_int(action.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(action.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(action.get("projected_saved_usd")), 6),
        "safety_gates": action.get("safety_gates") if isinstance(action.get("safety_gates"), dict) else {},
        "staged_at": utc_now(),
    }
    updated = dict(existing)
    section = updated.get("request_shape_repeated_context_canaries")
    if not isinstance(section, dict):
        section = {}
    rules = section.get("rules") if isinstance(section.get("rules"), list) else []
    kept = [rule for rule in rules if not (isinstance(rule, dict) and rule.get("id") == policy_id)]
    section.update({
        "enabled": True,
        "schema": "agentflow.request_shape_repeated_context_canaries.v1",
        "rules": kept + [canary_rule],
    })
    updated["request_shape_repeated_context_canaries"] = section
    if not dry_run:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")

    return {
        "schema": CRUNCH_CANARY_APPLY_SCHEMA,
        "ok": True,
        "dry_run": bool(dry_run),
        "wrote_policy_files": not dry_run,
        "target_local_policy": "crunch_rules",
        "policy_id": policy_id,
        "cohort_id": action.get("cohort_id"),
        "canary_fraction": canary_rule["rollout"]["canary_fraction"],
        "holdout_fraction": canary_rule["rollout"]["holdout_fraction"],
        "rules_path_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_opportunity_dry_run(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 25,
    rollout_fraction: float = DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
    holdout_fraction: float = DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    crunch_rows = [
        row
        for row in rollups
        if isinstance(row, dict)
        and (
            "crunch" in {str(item) for item in row.get("candidate_work_classes") or []}
            or "repeated_context" in {str(item) for item in row.get("candidate_work_classes") or []}
        )
    ]
    cohorts: list[dict[str, Any]] = []
    readiness_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    work_class_counts: dict[str, int] = {}
    projected_tokens = 0
    projected_chars = 0
    projected_savings = 0.0
    observed_tokens = 0
    observed_chars = 0
    observed_savings = 0.0
    canary_applied_rows = 0
    canary_holdout_rows = 0
    canary_safety_stopped_rows = 0

    for row in crunch_rows:
        decision = _shape_crunch_decision(row)
        row_count = _as_int(row.get("row_count") or row.get("count"))
        classes = sorted({public_label(item, "unknown") for item in row.get("candidate_work_classes") or []})
        readiness = str(decision["readiness"])
        reason = str(decision["reason"])
        row_projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
        row_projected_chars = _as_int(row.get("projected_crunch_chars_saved"))
        row_projected_savings = _as_float(row.get("projected_crunch_savings_usd"))
        row_observed_tokens = _as_int(row.get("current_crunch_tokens_saved"))
        row_observed_chars = _as_int(row.get("current_crunch_chars_saved"))
        row_observed_savings = _as_float(row.get("current_crunch_savings_usd"))
        lifecycle = row.get("crunch_canary_lifecycle") if isinstance(row.get("crunch_canary_lifecycle"), dict) else {}
        cohort_id = str(lifecycle.get("cohort_id") or _crunch_canary_cohort_id(row))
        policy_id = str(lifecycle.get("policy_id") or _crunch_canary_policy_id(cohort_id))
        applied_count = _as_int(lifecycle.get("applied_count"))
        holdout_count = _as_int(lifecycle.get("holdout_count"))
        safety_stopped_count = _as_int(lifecycle.get("safety_stopped_count"))

        _increment(readiness_counts, readiness)
        _increment(reason_counts, reason)
        for blocker in decision.get("blockers") or []:
            _increment(blocker_counts, blocker)
        for work_class in classes:
            _increment(work_class_counts, work_class, row_count)
        if readiness in {"measurement-ready", "canary-staged", "canary-applied", "canary-holdout"}:
            projected_tokens += row_projected_tokens
            projected_chars += row_projected_chars
            projected_savings += row_projected_savings
            observed_tokens += row_observed_tokens
            observed_chars += row_observed_chars
            observed_savings += row_observed_savings
        canary_applied_rows += applied_count
        canary_holdout_rows += holdout_count
        canary_safety_stopped_rows += safety_stopped_count

        cohorts.append(
            {
                "schema": "agentflow.request_shape_crunch_opportunity_cohort.v1",
                "cohort_id": cohort_id,
                "policy_id": policy_id,
                "readiness": readiness,
                "reason": reason,
                "blockers": decision.get("blockers") or [],
                "provider_family": row.get("provider_family"),
                "source_surface": row.get("source_surface"),
                "endpoint": row.get("endpoint"),
                "category": row.get("category"),
                "workflow_phase": row.get("workflow_phase"),
                "stream": bool(row.get("stream")),
                "has_tools": bool(row.get("has_tools")),
                "cache_status": row.get("cache_status"),
                "routing_status": row.get("routing_status"),
                "text_bucket": row.get("text_bucket"),
                "token_bucket": row.get("token_bucket"),
                "row_count": row_count,
                "work_classes": classes,
                "current_conservative_tokens_saved": row_observed_tokens,
                "current_conservative_chars_saved": row_observed_chars,
                "current_conservative_savings_usd": round(row_observed_savings, 6),
                "projected_saved_tokens": row_projected_tokens,
                "projected_saved_chars": row_projected_chars,
                "projected_saved_usd": round(row_projected_savings, 6),
                "crunch_canary_lifecycle": lifecycle,
                "candidate_rule": "repeated-context-conservative-dry-run",
                "estimate_basis": (
                    "metadata-only projection using aggregate input tokens, repeated-shape row count, "
                    f"and {REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE:.0%} conservative input-token reduction"
                ),
                "aggregate_only": True,
                "privacy": _crunch_opportunity_privacy(),
            }
        )

    cohorts.sort(
        key=lambda item: (
            item.get("readiness") == "measurement-ready",
            _as_float(item.get("projected_saved_usd")) + _as_float(item.get("current_conservative_savings_usd")),
            _as_int(item.get("projected_saved_tokens")) + _as_int(item.get("current_conservative_tokens_saved")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(cohorts, start=1):
        row["rank"] = rank

    recommended_actions = [
        _request_shape_crunch_canary_action(
            cohort,
            candidate_count=len(cohorts),
            rollout_fraction=rollout_fraction,
            holdout_fraction=holdout_fraction,
        )
        for cohort in cohorts[:1]
        if cohort.get("readiness") == "measurement-ready"
    ]
    blocker_breakdown = _breakdown(blocker_counts)
    top_blocker = blocker_breakdown[0]["value"] if blocker_breakdown else None
    status = "ranked" if projected_tokens > 0 or observed_tokens > 0 or projected_savings > 0 or observed_savings > 0 else "no-positive-crunch-opportunity"
    if canary_safety_stopped_rows:
        status = "canary-safety-stopped"
    elif canary_applied_rows or canary_holdout_rows:
        status = "canary-staged"
    if not cohorts:
        status = "no-repeated-context-crunch-cohorts"
    missing = []
    if not cohorts:
        missing.append("repeated-context-crunch-cohorts")
    if projected_tokens <= 0 and observed_tokens <= 0 and projected_savings <= 0 and observed_savings <= 0:
        missing.append("positive-observed-or-projected-savings")

    return {
        "schema": CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        "status": status,
        "summary": {
            "candidate_count": len(cohorts),
            "matched_count": sum(_as_int(row.get("row_count") or row.get("count")) for row in crunch_rows),
            "rows_considered": sum(_as_int(row.get("row_count") or row.get("count")) for row in crunch_rows),
            "measurement_ready_cohort_count": readiness_counts.get("measurement-ready", 0),
            "canary_staged_cohort_count": readiness_counts.get("canary-staged", 0),
            "canary_applied_cohort_count": readiness_counts.get("canary-applied", 0),
            "canary_holdout_cohort_count": readiness_counts.get("canary-holdout", 0),
            "canary_safety_stopped_cohort_count": readiness_counts.get("canary-safety-stopped", 0),
            "canary_applied_rows": canary_applied_rows,
            "canary_holdout_rows": canary_holdout_rows,
            "canary_safety_stopped_rows": canary_safety_stopped_rows,
            "recommended_action_count": len(recommended_actions),
            "skipped_cohort_count": readiness_counts.get("skipped", 0),
            "current_conservative_tokens_saved": observed_tokens,
            "current_conservative_chars_saved": observed_chars,
            "current_conservative_savings_usd": round(observed_savings, 6),
            "projected_saved_tokens": projected_tokens,
            "projected_saved_chars": projected_chars,
            "projected_saved_usd": round(projected_savings, 6),
            "top_blocker_code": top_blocker,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "recommended_actions": recommended_actions,
        "readiness_breakdown": _breakdown(readiness_counts),
        "reason_breakdown": _breakdown(reason_counts),
        "blocker_reason_breakdown": blocker_breakdown,
        "work_class_breakdown": _breakdown(work_class_counts),
        "cohorts": cohorts[: max(1, min(_as_int(limit) or 25, 1000))],
        "missing_measurements": missing,
        "privacy": _crunch_opportunity_privacy(),
    }


def _shape_replayability_decision(row: dict[str, Any]) -> dict[str, Any]:
    row_count = _as_int(row.get("row_count") or row.get("count"))
    endpoint = str(row.get("endpoint") or "unknown")
    cache_status = str(row.get("cache_status") or "unknown")
    stream = bool(row.get("stream"))
    has_tools = bool(row.get("has_tools"))
    blockers: set[str] = set()

    if endpoint not in REPLAY_SUPPORTED_ENDPOINTS:
        blockers.add("unsupported-endpoint")
    if cache_status == "hit":
        blockers.add("already-cache-hit")
    if row_count < 2:
        blockers.add("insufficient-repeat-evidence")
    if stream:
        blockers.add("streaming-replay-not-supported")
    if has_tools:
        blockers.add("tools-present")
        blockers.add("invalidation-evidence-missing")
        blockers.add("unsafe-tool-calls-without-invalidation")

    if not blockers:
        return {
            "readiness": "replay-ready",
            "reason": "replay-ready-exact-non-tool-shape",
            "blockers": [],
            "projected_hits": max(0, row_count - 1),
        }

    reason_priority = (
        "unsupported-endpoint",
        "already-cache-hit",
        "streaming-replay-not-supported",
        "invalidation-evidence-missing",
        "unsafe-tool-calls-without-invalidation",
        "tools-present",
        "insufficient-repeat-evidence",
    )
    reason = next((item for item in reason_priority if item in blockers), sorted(blockers)[0])
    return {
        "readiness": "skipped",
        "reason": reason,
        "blockers": sorted(blockers),
        "projected_hits": 0,
    }


def build_request_shape_cache_replayability_dry_run(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 25,
) -> dict[str, Any]:
    replay_rows = [
        row
        for row in rollups
        if isinstance(row, dict)
        and (
            "replayability" in {str(item) for item in row.get("candidate_work_classes") or []}
            or "cache_replay" in {str(item) for item in row.get("candidate_families") or []}
            or "cache_blocker" in {str(item) for item in row.get("candidate_families") or []}
        )
    ]
    cohorts: list[dict[str, Any]] = []
    readiness_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    replay_ready_rows = 0
    skipped_rows = 0
    projected_hits = 0
    projected_savings = 0.0

    for row in replay_rows:
        decision = _shape_replayability_decision(row)
        row_count = _as_int(row.get("row_count") or row.get("count"))
        cost = _as_float(row.get("cost_est_usd"))
        hits = _as_int(decision.get("projected_hits"))
        saved = 0.0
        if hits and row_count:
            avg_cost = cost / row_count
            saved = max(0.0, cost - avg_cost)
        readiness = str(decision["readiness"])
        reason = str(decision["reason"])
        _increment(readiness_counts, readiness)
        _increment(reason_counts, reason)
        for blocker in decision.get("blockers") or []:
            _increment(blocker_counts, blocker)
        if readiness == "replay-ready":
            replay_ready_rows += row_count
            projected_hits += hits
            projected_savings += saved
        else:
            skipped_rows += row_count
        cohorts.append(
            {
                "schema": "agentflow.request_shape_cache_replayability_cohort.v1",
                "readiness": readiness,
                "reason": reason,
                "blockers": decision.get("blockers") or [],
                "provider_family": row.get("provider_family"),
                "source_surface": row.get("source_surface"),
                "endpoint": row.get("endpoint"),
                "category": row.get("category"),
                "workflow_phase": row.get("workflow_phase"),
                "stream": bool(row.get("stream")),
                "has_tools": bool(row.get("has_tools")),
                "cache_status": row.get("cache_status"),
                "routing_status": row.get("routing_status"),
                "text_bucket": row.get("text_bucket"),
                "token_bucket": row.get("token_bucket"),
                "row_count": row_count,
                "projected_hits": hits,
                "projected_savings_usd": round(saved, 6),
                "aggregate_only": True,
                "privacy": _replayability_privacy(),
            }
        )

    cohorts.sort(
        key=lambda item: (
            item.get("readiness") == "replay-ready",
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("projected_hits")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(cohorts, start=1):
        row["rank"] = rank

    top_blocker = None
    blocker_breakdown = _breakdown(blocker_counts)
    if blocker_breakdown:
        top_blocker = blocker_breakdown[0]["value"]
    return {
        "schema": REPLAYABILITY_DRY_RUN_SCHEMA,
        "status": "ranked" if cohorts else "no-replayability-cohorts",
        "summary": {
            "cohort_count": len(cohorts),
            "rows_considered": sum(_as_int(row.get("row_count") or row.get("count")) for row in replay_rows),
            "replay_ready_cohort_count": readiness_counts.get("replay-ready", 0),
            "replay_ready_rows": replay_ready_rows,
            "skipped_cohort_count": readiness_counts.get("skipped", 0),
            "skipped_rows": skipped_rows,
            "projected_hits": projected_hits,
            "projected_savings_usd": round(projected_savings, 6),
            "top_blocker_code": top_blocker,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "readiness_breakdown": _breakdown(readiness_counts),
        "skipped_reason_breakdown": _breakdown(reason_counts),
        "blocker_breakdown": blocker_breakdown,
        "cohorts": cohorts[: max(1, min(_as_int(limit) or 25, 1000))],
        "privacy": _replayability_privacy(),
    }


def _candidate_work_classes(
    *,
    row_count: int,
    text_bucket: str,
    token_bucket: str,
    candidate_families: list[str],
    blocker_codes: list[str],
    routing_status: str,
    observed_savings: float,
) -> list[str]:
    classes: set[str] = set()
    repeated_large_context = row_count >= 2 and text_bucket in REPEATED_CONTEXT_TEXT_BUCKETS
    token_heavy_context = token_bucket in {"8k_32k_tokens", "gte_32k_tokens"}
    if repeated_large_context or (row_count >= 2 and token_heavy_context):
        classes.add("repeated_context")
        classes.add("crunch")
    if any(family in {"cache_replay", "cache_blocker"} for family in candidate_families) or any(
        code
        in {
            "unsupported-streaming-shape",
            "tool-call-cache-disabled",
            "semantic-cache-disabled",
            "exact-cache-miss",
            "cache-skipped",
        }
        for code in blocker_codes
    ):
        classes.add("replayability")
    if "routing_candidate" in candidate_families or routing_status == "passthrough":
        classes.add("routing")
    if "routing_evidence" in candidate_families or observed_savings > 0:
        classes.add("routing_evidence")
    return sorted(classes or {"observability"})


def _persistable_row(
    *,
    run_id: str,
    generated_at: str,
    window_start: str | None,
    window_end: str | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{run_id}:{row['rollup_key']}",
        "run_id": run_id,
        "generated_at": generated_at,
        "window_start": window_start,
        "window_end": window_end,
        "rollup_key": row["rollup_key"],
        "candidate_id": row["candidate_id"],
        "source_surface": row["source_surface"],
        "endpoint": row["endpoint"],
        "provider_family": row["provider_family"],
        "requested_model_family": row["requested_model_family"],
        "routed_model_family": row["routed_model_family"],
        "category": row["category"],
        "workflow_phase": row["workflow_phase"],
        "stream": 1 if row["stream"] else 0,
        "has_tools": 1 if row["has_tools"] else 0,
        "text_bucket": row["text_bucket"],
        "token_bucket": row["token_bucket"],
        "cache_status": row["cache_status"],
        "routing_status": row["routing_status"],
        "candidate_families_json": stable_json(row["candidate_families"]),
        "blocker_codes_json": stable_json(row["blocker_codes"]),
        "row_count": row["row_count"],
        "error_count": row["error_count"],
        "retry_count": row["retry_count"],
        "cache_hit_count": row["cache_hit_count"],
        "cost_est_usd": row["cost_est_usd"],
        "baseline_cost_usd": row["baseline_cost_usd"],
        "observed_savings_usd": row["observed_savings_usd"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "metadata_json": stable_json(row["metadata"]),
    }


def build_request_shape_rollups_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    persist: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    generated_at = utc_now()
    run_id = run_id or f"shape-rollups-{uuid4().hex[:12]}"
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, crunch_json,
                   routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[str, dict[str, Any]] = {}
    provider_counts: dict[str, int] = {}
    candidate_family_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    body_rows_read = 0
    window_start: str | None = None
    window_end: str | None = None
    impact_rows: list[dict[str, Any]] = []

    for row in rows:
        created_at = str(row.get("created_at") or "")
        if created_at:
            window_start = created_at if window_start is None else min(window_start, created_at)
            window_end = created_at if window_end is None else max(window_end, created_at)
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        crunch = _json_obj(row.get("crunch_json"))
        provider = _provider_family(row)
        endpoint = _endpoint(row)
        source_surface = _source_surface(row, provider, endpoint)
        requested_family = public_label(row.get("requested_model_family"), "") or _model_family(row.get("requested_model"))
        routed_family = public_label(row.get("routed_model_family"), "") or _model_family(
            row.get("routed_model"),
            requested_family,
        )
        category = public_label(row.get("category") or routing.get("category"), "unknown")
        workflow_phase = _workflow_phase(row, routing)
        stream = bool(_as_int(row.get("stream")))
        has_tools = _has_tools(row, routing, cache)
        text_chars = _as_int(routing.get("text_chars"))
        input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        if text_chars <= 0 and input_tokens > 0:
            text_chars = input_tokens * 4
        projection_input_tokens = max(input_tokens, text_chars // 4 if text_chars > 0 else 0)
        output_tokens = _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))
        cost = _as_float(row.get("cost_est_usd"))
        baseline = _as_float(row.get("cost_baseline_usd"))
        observed_savings = max(0.0, baseline - cost)
        status_bucket = _status_bucket(row.get("status_code"))
        current_crunch_tokens = _crunch_saved_tokens(crunch)
        current_crunch_chars = _crunch_saved_chars(crunch)
        crunch_canary_lifecycle = _crunch_canary_lifecycle_from_meta(crunch)
        crunch_model = str(row.get("routed_model") or row.get("requested_model") or "")
        current_crunch_savings = _input_savings_usd(
            current_crunch_tokens,
            provider=provider,
            model=crunch_model,
            fallback_cost=cost,
            fallback_tokens=input_tokens,
        )
        input_token_cost = _input_savings_usd(
            projection_input_tokens,
            provider=provider,
            model=crunch_model,
            fallback_cost=cost,
            fallback_tokens=input_tokens or projection_input_tokens,
        )
        cache_status = _cache_status(row, cache)
        cache_reason = public_label(cache.get("reason"), "unknown")
        routing_status = _routing_status(row, routing)
        blockers = _blocker_codes(
            row=row,
            cache=cache,
            routing=routing,
            cache_status=cache_status,
            routing_status=routing_status,
            stream=stream,
            has_tools=has_tools,
        )
        candidate_families = _candidate_families(
            cache_status=cache_status,
            routing_status=routing_status,
            blockers=blockers,
            observed_savings=observed_savings,
            cost=cost,
        )
        basis = {
            "source_surface": source_surface,
            "endpoint": endpoint,
            "provider_family": provider,
            "requested_model_family": requested_family,
            "routed_model_family": routed_family,
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": _text_bucket(text_chars),
            "token_bucket": _token_bucket(input_tokens),
            "cache_status": cache_status,
            "routing_status": routing_status,
        }
        impact_rows.append(
            {
                **row,
                **basis,
                "text_chars": text_chars,
                "input_tokens": input_tokens,
            }
        )
        rollup_key = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
        candidate_id = _candidate_id(basis)
        group = groups.setdefault(rollup_key, _new_group(basis, candidate_id=candidate_id, rollup_key=rollup_key))
        group["row_count"] += 1
        group["error_count"] += int(_status_bucket(row.get("status_code")) in {"4xx", "5xx"})
        group["retry_count"] += _as_int(row.get("retry_count"))
        group["cache_hit_count"] += int(_as_int(row.get("cache_hit")) > 0 or cache_status == "hit")
        group["cost_est_usd"] += cost
        group["baseline_cost_usd"] += baseline
        group["observed_savings_usd"] += observed_savings
        group["input_tokens"] += input_tokens
        group["output_tokens"] += output_tokens
        if status_bucket in {"2xx", "3xx"}:
            group["successful_input_tokens"] += projection_input_tokens
            group["input_token_cost_usd"] += input_token_cost
        group["current_crunch_tokens_saved"] += current_crunch_tokens
        group["current_crunch_chars_saved"] += current_crunch_chars
        group["current_crunch_savings_usd"] += current_crunch_savings
        _increment(provider_counts, provider)
        _increment(group["status_counts"], status_bucket)
        _increment(group["retry_bucket_counts"], _retry_bucket(_as_int(row.get("retry_count"))))
        _increment(group["cost_bucket_counts"], _cost_bucket(cost))
        _increment(group["savings_bucket_counts"], _savings_bucket(observed_savings))
        _increment(group["cache_reason_counts"], cache_reason)
        for family in candidate_families:
            _increment(candidate_family_counts, family)
            _increment(group["candidate_family_counts"], family)
        for blocker in blockers:
            _increment(blocker_counts, blocker)
            _increment(group["blocker_counts"], blocker)
        if crunch_canary_lifecycle:
            status = str(crunch_canary_lifecycle.get("status") or "unknown")
            _increment(group["crunch_canary_lifecycle_counts"], status)
            policy_id = str(crunch_canary_lifecycle.get("policy_id") or "unknown")
            _increment(group["crunch_canary_policy_counts"], policy_id)

    rollups = [_finalize_group(group) for group in groups.values()]
    rollups.sort(
        key=lambda item: (
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("cost_est_usd")),
            _as_int(item.get("row_count")),
            item.get("candidate_id") or "",
        ),
        reverse=True,
    )
    persistable = [
        _persistable_row(
            run_id=run_id,
            generated_at=generated_at,
            window_start=window_start,
            window_end=window_end,
            row=row,
        )
        for row in rollups
    ]
    persisted_count = 0
    if persist and hasattr(store_obj, "persist_request_shape_rollups"):
        persisted_count = store_obj.persist_request_shape_rollups(
            run_id=run_id,
            generated_at=generated_at,
            rows=persistable,
        )

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "run_id": run_id,
        "limit": capped_limit,
        "persisted": bool(persisted_count),
        "persisted_count": persisted_count,
        "window": {
            "start": window_start,
            "end": window_end,
            "source": "recent-local-call-metadata",
        },
        "summary": {
            "rows_considered": len(rows),
            "rollup_count": len(rollups),
            "collapsed_rows": max(0, len(rows) - len(rollups)),
            "total_cost_est_usd": round(sum(_as_float(row.get("cost_est_usd")) for row in rollups), 6),
            "total_baseline_cost_usd": round(sum(_as_float(row.get("baseline_cost_usd")) for row in rollups), 6),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in rollups), 6),
            "body_rows_read": body_rows_read,
        },
        "provider_breakdown": _breakdown(provider_counts),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_code_breakdown": _breakdown(blocker_counts),
        "cache_replayability_dry_run": build_request_shape_cache_replayability_dry_run(rollups, limit=25),
        "crunch_opportunity_dry_run": build_request_shape_crunch_opportunity_dry_run(rollups, limit=25),
        "crunch_canary_impact": build_request_shape_crunch_canary_impact_report(impact_rows),
        "rollups": rollups,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "tenant_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
