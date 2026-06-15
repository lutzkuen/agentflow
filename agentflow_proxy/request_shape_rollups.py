from __future__ import annotations

import hashlib
import json
import math
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
REPLAY_BLOCKER_CLASSIFICATION_SCHEMA = "agentflow.request_shape_cache_replay_blocker_classification.v1"
REPLAY_BLOCKER_CLASSIFICATION_ROW_SCHEMA = "agentflow.request_shape_cache_replay_blocker_classification_row.v1"
REPLAY_CACHE_CANARY_ACTION_SCHEMA = "agentflow.request_shape_cache_replay_canary_action.v1"
REPLAY_CACHE_CANARY_STAGE_SCHEMA = "agentflow.request_shape_cache_replay_canary_stage.v1"
CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA = "agentflow.request_shape_crunch_opportunity_dry_run.v1"
CRUNCH_CANARY_ACTION_SCHEMA = "agentflow.request_shape_crunch_canary_action.v1"
CRUNCH_CANARY_STAGE_SCHEMA = "agentflow.request_shape_repeated_context_crunch_canary_stage.v1"
CRUNCH_CANARY_APPLY_SCHEMA = "agentflow.request_shape_crunch_canary_apply.v1"
CRUNCH_CANARY_LIFECYCLE_SCHEMA = "agentflow.request_shape_crunch_canary_lifecycle.v1"
CRUNCH_CANARY_IMPACT_SCHEMA = "agentflow.request_shape_crunch_canary_impact.v1"
FOLLOW_UP_CANDIDATES_SCHEMA = "agentflow.request_shape_follow_up_candidates.v1"
FOLLOW_UP_BLOCKER_COHORT_SCHEMA = "agentflow.request_shape_blocker_cohort.v1"
REPEATED_CONTEXT_TEXT_BUCKETS = {"8k_32k_chars", "32k_128k_chars", "gte_128k_chars"}
REPLAY_SUPPORTED_ENDPOINTS = {"messages", "responses", "chat_completions", "chat"}
REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE = 0.05
DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS = 72.0
DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION = 0.10
DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION = 0.10


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


def _public_label_list(values: Any) -> list[str]:
    if not isinstance(values, list | tuple | set):
        return []
    return sorted(
        {
            public_label(item, "unknown")
            for item in values
            if public_label(item, "unknown") != "unknown"
        }
    )


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
        "latency_ms_total": 0,
        "latency_sample_count": 0,
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
    latency_ms = _as_int(row.get("latency_ms"), -1)
    if latency_ms >= 0:
        cohort["latency_ms_total"] += latency_ms
        cohort["latency_sample_count"] += 1
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
    latency_samples = _as_int(raw.get("latency_sample_count"))
    return {
        "count": count,
        "before_chars": _as_int(raw.get("before_chars")),
        "after_chars": _as_int(raw.get("after_chars")),
        "saved_chars": _as_int(raw.get("saved_chars")),
        "saved_tokens": _as_int(raw.get("saved_tokens")),
        "saved_usd": round(_as_float(raw.get("saved_usd")), 8),
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
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


def _crunch_impact_recommendation(
    *,
    verdict: str,
    reasons: list[str],
    applied: dict[str, Any],
    holdout: dict[str, Any],
) -> tuple[str, str]:
    reason_set = set(reasons)
    if verdict == "widen-ready":
        return "promotion-ready", "widen-repeated-context-crunch-canary"
    if reason_set & {
        "rollback-observed",
        "canary-safety-stopped",
        "error-rate-regression",
        "retry-rate-regression",
        "fallback-observed",
    }:
        return "rollback", "rollback-repeated-context-crunch-canary"
    if reason_set & {"missing-applied-coverage", "missing-holdout-coverage", "stale-canary-impact-evidence"}:
        return "collect-more-evidence", "collect-repeated-context-crunch-canary-impact-evidence"
    if _as_int(applied.get("count")) > 0 and _as_int(holdout.get("count")) > 0:
        return "keep-blocked", "keep-repeated-context-crunch-canary-blocked"
    return "collect-more-evidence", "collect-repeated-context-crunch-canary-impact-evidence"


def _crunch_impact_next_action(
    *,
    impact_recommendation: str | None,
    applied_count: int,
    holdout_count: int,
    reason_codes: list[str],
) -> str:
    if impact_recommendation == "promotion-ready":
        return "widen"
    if impact_recommendation == "rollback":
        return "rollback"
    if applied_count <= 0 or "missing-applied-coverage" in reason_codes:
        return "stage-canary-first"
    return "keep-observing"


def _crunch_impact_coverage(
    *,
    applied_count: int,
    holdout_count: int,
    skipped_count: int = 0,
    fallback_count: int = 0,
    safety_stop_count: int = 0,
    rollback_count: int = 0,
    unknown_count: int = 0,
) -> dict[str, Any]:
    observed_count = (
        applied_count
        + holdout_count
        + skipped_count
        + fallback_count
        + safety_stop_count
        + rollback_count
        + unknown_count
    )
    return {
        "schema": "agentflow.request_shape_crunch_canary_coverage.v1",
        "observed_count": observed_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "fallback_count": fallback_count,
        "safety_stop_count": safety_stop_count,
        "rollback_count": rollback_count,
        "unknown_count": unknown_count,
        "has_applied_coverage": applied_count > 0,
        "has_holdout_coverage": holdout_count > 0,
        "applied_coverage_rate": round(applied_count / observed_count, 6) if observed_count else 0.0,
        "holdout_coverage_rate": round(holdout_count / observed_count, 6) if observed_count else 0.0,
        "applied_to_holdout_ratio": round(applied_count / holdout_count, 6) if holdout_count else None,
        "aggregate_only": True,
        "metadata_only": True,
    }


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
        impact_recommendation, recommended_next_action = _crunch_impact_recommendation(
            verdict=verdict,
            reasons=reasons,
            applied=applied,
            holdout=holdout,
        )
        public_next_action = _crunch_impact_next_action(
            impact_recommendation=impact_recommendation,
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            reason_codes=reasons,
        )
        _increment(verdict_counts, verdict)
        if verdict != "widen-ready":
            for reason in reasons:
                _increment(blocker_counts, reason)
        latency_delta = None
        if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
            latency_delta = round(_as_float(applied.get("latency_avg_ms")) - _as_float(holdout.get("latency_avg_ms")), 2)
        observed_count = sum(_as_int(cohort.get("count")) for cohort in cohorts.values())
        coverage = _crunch_impact_coverage(
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            skipped_count=_as_int(cohorts["skipped"].get("count")),
            fallback_count=_as_int(fallback.get("count")),
            safety_stop_count=_as_int(safety.get("count")),
            rollback_count=_as_int(rollback.get("count")),
            unknown_count=_as_int(cohorts["unknown"].get("count")),
        )
        finalized.append(
            {
                "schema": "agentflow.request_shape_crunch_canary_impact_candidate.v1",
                "policy_id": raw["policy_id"],
                "cohort_id": raw["cohort_id"],
                "cohort_metadata": raw["cohort_metadata"],
                "policy_source": raw["policy_source"],
                "observed_count": observed_count,
                "applied_count": _as_int(applied.get("count")),
                "holdout_count": _as_int(holdout.get("count")),
                "fallback_count": _as_int(fallback.get("count")),
                "safety_stop_count": _as_int(safety.get("count")),
                "rollback_count": _as_int(rollback.get("count")),
                "saved_chars": _as_int(applied.get("saved_chars")),
                "saved_tokens": _as_int(applied.get("saved_tokens")),
                "saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                "estimated_saved_chars": _as_int(applied.get("saved_chars")),
                "estimated_saved_tokens": _as_int(applied.get("saved_tokens")),
                "estimated_saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                "applied_error_count": _as_int(applied.get("error_count")),
                "holdout_error_count": _as_int(holdout.get("error_count")),
                "applied_retry_count": _as_int(applied.get("retry_count")),
                "holdout_retry_count": _as_int(holdout.get("retry_count")),
                "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
                "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
                "latency_avg_delta_ms": latency_delta,
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
                "impact_recommendation": impact_recommendation,
                "promotion_recommendation": impact_recommendation,
                "recommended_next_action": recommended_next_action,
                "next_action": public_next_action,
                "top_blocker": top_blocker if verdict != "widen-ready" else None,
                "reason_codes": reasons,
                "coverage": coverage,
                "applied_vs_holdout_coverage": coverage,
                "promotion_metadata": {
                    "schema": "agentflow.request_shape_crunch_canary_promotion_recommendation.v1",
                    "action_family": "crunch",
                    "local_action_family": "crunch",
                    "target_local_policy": "crunch_rules",
                    "impact_recommendation": impact_recommendation,
                    "recommended_next_action": recommended_next_action,
                    "next_action": public_next_action,
                    "reason_codes": reasons,
                    "applied_count": _as_int(applied.get("count")),
                    "holdout_count": _as_int(holdout.get("count")),
                    "safety_stop_count": _as_int(safety.get("count")),
                    "fallback_count": _as_int(fallback.get("count")),
                    "rollback_count": _as_int(rollback.get("count")),
                    "observed_saved_tokens": _as_int(applied.get("saved_tokens")),
                    "observed_saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                    "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
                    "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
                    "latency_avg_delta_ms": latency_delta,
                    "privacy": _crunch_opportunity_privacy(),
                },
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
    recommendation_counts: dict[str, int] = {}
    for item in finalized:
        _increment(recommendation_counts, item.get("impact_recommendation") or "unknown")
    recommendation_breakdown = _breakdown(recommendation_counts)
    top_recommendation = recommendation_breakdown[0]["value"] if recommendation_breakdown else None
    top_next_action = None
    if finalized:
        top_next_action = str(finalized[0].get("next_action") or "")
    total_applied = sum(_as_int(item.get("applied_count")) for item in finalized)
    total_holdout = sum(_as_int(item.get("holdout_count")) for item in finalized)
    total_skipped = sum(_as_int((item.get("cohorts") or {}).get("skipped", {}).get("count")) for item in finalized)
    total_fallback = sum(_as_int(item.get("fallback_count")) for item in finalized)
    total_safety = sum(_as_int(item.get("safety_stop_count")) for item in finalized)
    total_rollback = sum(_as_int(item.get("rollback_count")) for item in finalized)
    total_unknown = sum(_as_int((item.get("cohorts") or {}).get("unknown", {}).get("count")) for item in finalized)
    coverage = _crunch_impact_coverage(
        applied_count=total_applied,
        holdout_count=total_holdout,
        skipped_count=total_skipped,
        fallback_count=total_fallback,
        safety_stop_count=total_safety,
        rollback_count=total_rollback,
        unknown_count=total_unknown,
    )
    if finalized and total_applied <= 0 and total_safety <= 0 and total_rollback <= 0:
        status = "no-applied-coverage"
    if not finalized:
        status = "no-applied-coverage"
        top_next_action = "stage-canary-first"
    missing_measurements = []
    if total_applied <= 0:
        missing_measurements.append("applied-crunch-canary-coverage")
    if finalized and total_holdout <= 0:
        missing_measurements.append("holdout-crunch-canary-coverage")
    if not finalized:
        missing_measurements.append("crunch-canary-lifecycle-metadata")
    return {
        "schema": CRUNCH_CANARY_IMPACT_SCHEMA,
        "status": status,
        "ok": True,
        "read_only": True,
        "next_action": top_next_action or "stage-canary-first",
        "recommended_next_action": str(finalized[0].get("recommended_next_action") or "") if finalized else "stage-repeated-context-crunch-canary",
        "missing_measurements": missing_measurements,
        "summary": {
            "candidate_count": len(finalized),
            "observed_canary_metadata_row_count": observed_rows,
            "applied_count": total_applied,
            "holdout_count": total_holdout,
            "saved_chars": sum(_as_int(item.get("saved_chars")) for item in finalized),
            "saved_tokens": sum(_as_int(item.get("saved_tokens")) for item in finalized),
            "saved_usd": round(sum(_as_float(item.get("saved_usd")) for item in finalized), 8),
            "estimated_saved_chars": sum(_as_int(item.get("estimated_saved_chars")) for item in finalized),
            "estimated_saved_tokens": sum(_as_int(item.get("estimated_saved_tokens")) for item in finalized),
            "estimated_saved_usd": round(sum(_as_float(item.get("estimated_saved_usd")) for item in finalized), 8),
            "projected_saved_chars": sum(_as_int(item.get("saved_chars")) for item in finalized),
            "projected_saved_tokens": sum(_as_int(item.get("saved_tokens")) for item in finalized),
            "projected_saved_usd": round(sum(_as_float(item.get("saved_usd")) for item in finalized), 8),
            "error_rate_delta": round(max((_as_float(item.get("error_rate_delta")) for item in finalized), default=0.0), 6),
            "retry_rate_delta": round(max((_as_float(item.get("retry_rate_delta")) for item in finalized), default=0.0), 6),
            "latency_avg_delta_ms": max(
                (
                    _as_float(item.get("latency_avg_delta_ms"))
                    for item in finalized
                    if item.get("latency_avg_delta_ms") is not None
                ),
                default=None,
            ),
            "fallback_count": total_fallback,
            "safety_stop_count": total_safety,
            "rollback_count": total_rollback,
            "widen_ready_count": sum(1 for item in finalized if item.get("verdict") == "widen-ready"),
            "no_widen_count": sum(1 for item in finalized if item.get("verdict") != "widen-ready"),
            "promotion_ready_count": sum(1 for item in finalized if item.get("impact_recommendation") == "promotion-ready"),
            "rollback_recommended_count": sum(1 for item in finalized if item.get("impact_recommendation") == "rollback"),
            "keep_blocked_count": sum(1 for item in finalized if item.get("impact_recommendation") == "keep-blocked"),
            "collect_more_evidence_count": sum(
                1 for item in finalized if item.get("impact_recommendation") == "collect-more-evidence"
            ),
            "top_impact_recommendation": top_recommendation,
            "top_blocker_code": blocker_breakdown[0]["value"] if blocker_breakdown else None,
            "next_action": top_next_action or "stage-canary-first",
            "top_next_action": top_next_action or "stage-canary-first",
            "recommended_next_action": str(finalized[0].get("recommended_next_action") or "") if finalized else "stage-repeated-context-crunch-canary",
            "coverage": coverage,
            "applied_vs_holdout_coverage": coverage,
            "cohort_counts": {
                "canary_applied": total_applied,
                "canary_holdout": total_holdout,
                "skipped": total_skipped,
                "fallback": total_fallback,
                "safety_stopped": total_safety,
                "rollback": total_rollback,
                "unknown": total_unknown,
            },
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "verdict_breakdown": _breakdown(verdict_counts),
        "impact_recommendation_breakdown": recommendation_breakdown,
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


def _scaled_projection(total: int, selected: int, matched: int) -> int:
    if total <= 0 or selected <= 0 or matched <= 0:
        return 0
    return max(0, int(round(total * (selected / float(matched)))))


def _request_shape_crunch_canary_lifecycle_projection(
    cohort: dict[str, Any],
    *,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    matched = _as_int(cohort.get("row_count") or cohort.get("cohort_row_count"))
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)

    readiness = public_label(cohort.get("readiness"), "unknown")
    reason = public_label(cohort.get("reason"), "unknown")
    blockers = [
        public_label(item, "unknown")
        for item in cohort.get("blockers") or []
        if public_label(item, "unknown") != "unknown"
    ]
    evidence_blockers = _public_label_list(cohort.get("evidence_blocker_codes") or cohort.get("blocker_codes"))
    applied = holdout_count = skipped = safety_stopped = 0
    lifecycle_status = "skipped"
    explicit_reason = reason

    if readiness == "measurement-ready" and matched > 0:
        holdout_count = min(matched, int(math.ceil(matched * holdout))) if holdout > 0 else 0
        remaining = max(0, matched - holdout_count)
        applied = min(remaining, int(math.ceil(matched * rollout))) if rollout > 0 else 0
        skipped = max(0, matched - applied - holdout_count)
        lifecycle_status = "projected-applied-holdout" if applied > 0 and holdout_count > 0 else "projected-partial"
        explicit_reason = "projected-canary-applied-and-holdout" if applied > 0 and holdout_count > 0 else "projected-canary-partial"
    elif readiness == "canary-safety-stopped":
        safety_stopped = matched
        lifecycle_status = "safety-stopped"
        explicit_reason = reason or "repeated-context-crunch-canary-safety-stopped"
    else:
        skipped = matched
        lifecycle_status = "skipped"
        explicit_reason = reason or (blockers[0] if blockers else "not-stageable")

    projected_tokens = _as_int(cohort.get("projected_saved_tokens"))
    projected_chars = _as_int(cohort.get("projected_saved_chars"))
    projected_usd = _as_float(cohort.get("projected_saved_usd"))

    return {
        "schema": "agentflow.request_shape_crunch_canary_projected_lifecycle.v1",
        "status": lifecycle_status,
        "reason": explicit_reason,
        "readiness": readiness,
        "matched_count": matched,
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "projected_canary_applied_count": applied,
        "projected_canary_holdout_count": holdout_count,
        "projected_skipped_count": skipped,
        "projected_safety_stopped_count": safety_stopped,
        "projected_fallback_count": 0,
        "projected_rollback_count": 0,
        "projected_saved_tokens": projected_tokens,
        "projected_saved_chars": projected_chars,
        "projected_saved_usd": round(projected_usd, 6),
        "projected_applied_saved_tokens": _scaled_projection(projected_tokens, applied, matched),
        "projected_applied_saved_chars": _scaled_projection(projected_chars, applied, matched),
        "projected_applied_saved_usd": round(projected_usd * (applied / float(matched)), 6) if matched and applied else 0.0,
        "projected_holdout_saved_tokens": _scaled_projection(projected_tokens, holdout_count, matched),
        "projected_holdout_saved_chars": _scaled_projection(projected_chars, holdout_count, matched),
        "projected_holdout_saved_usd": round(projected_usd * (holdout_count / float(matched)), 6) if matched and holdout_count else 0.0,
        "blocker_reasons": blockers,
        "evidence_blocker_codes": evidence_blockers,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }


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
    lifecycle_projection = _request_shape_crunch_canary_lifecycle_projection(
        cohort,
        rollout_fraction=rollout,
        holdout_fraction=holdout,
    )
    evidence_blockers = _public_label_list(cohort.get("evidence_blocker_codes") or cohort.get("blocker_codes"))
    return {
        "schema": CRUNCH_CANARY_ACTION_SCHEMA,
        "action_type": "stage-local-repeated-context-crunch-canary",
        "target_local_policy": "crunch_rules",
        "policy_section": "crunch",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "source_evidence_schema": cohort.get("source_evidence_schema") or CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        "source_evidence_schemas": [
            FOLLOW_UP_CANDIDATES_SCHEMA,
            CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        ],
        "local_only_reason": "file-backed-local-policy-no-managed-dependency",
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
        "evidence_blocker_codes": evidence_blockers,
        "projected_lifecycle": lifecycle_projection,
        "safety_gates": {
            "metadata_only": True,
            "aggregate_only": True,
            "local_file_backed": True,
            "local_only": True,
            "tool_call_cache_enabled": False,
            "tool_call_cache_enablement_allowed": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "tool_payloads_included": False,
            "holdout_required": holdout > 0,
            "max_rollout_fraction": DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
            "records_applied_holdout_skipped_safety_stopped_fallback_rollback": True,
        },
        "lifecycle_metadata": {
            "schema": "agentflow.request_shape_crunch_canary_stage_lifecycle_metadata.v1",
            "emits_applied": True,
            "emits_holdout": True,
            "emits_skipped": True,
            "emits_safety_stopped": True,
            "emits_fallback": True,
            "emits_rollback": True,
            "projected_canary_applied_count": lifecycle_projection["projected_canary_applied_count"],
            "projected_canary_holdout_count": lifecycle_projection["projected_canary_holdout_count"],
            "projected_skipped_count": lifecycle_projection["projected_skipped_count"],
            "projected_safety_stopped_count": lifecycle_projection["projected_safety_stopped_count"],
            "evidence_blocker_codes": evidence_blockers,
            "impact_report": CRUNCH_CANARY_IMPACT_SCHEMA,
            "lifecycle_schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
            "metadata_only": True,
            "aggregate_only": True,
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
        "source_evidence_schema": action.get("source_evidence_schema"),
        "source_evidence_schemas": _public_label_list(action.get("source_evidence_schemas")),
        "local_only_reason": public_label(action.get("local_only_reason"), "file-backed-local-policy-no-managed-dependency"),
        "evidence_blocker_codes": _public_label_list(action.get("evidence_blocker_codes")),
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
        "lifecycle_metadata": action.get("lifecycle_metadata") if isinstance(action.get("lifecycle_metadata"), dict) else {},
        "privacy": _crunch_opportunity_privacy(),
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


def _request_shape_crunch_follow_up(
    *,
    status: str,
    report_key: str,
    evidence_schema: str,
    candidate_count: int,
    matched_count: int,
    rows_considered: int,
    recommended_action_count: int,
    canary_applied_rows: int,
    canary_holdout_rows: int,
    canary_safety_stopped_rows: int,
    projected_saved_chars: int,
    projected_saved_tokens: int,
    projected_saved_usd: float,
    top_blocker: str | None,
    missing_measurements: list[str],
) -> dict[str, Any]:
    if canary_safety_stopped_rows > 0:
        activation_state = "blocked"
        next_action = "review-repeated-context-crunch-canary-safety-stop"
        activation_mode = "review-required"
        blocker = "canary-safety-stopped"
        no_op_reason = "matching-repeated-context-crunch-canary-safety-stopped"
    elif canary_applied_rows > 0 or canary_holdout_rows > 0:
        activation_state = "measurement-required"
        next_action = "measure-repeated-context-crunch-canary-impact"
        activation_mode = "staged-canary-measurement"
        blocker = "missing-crunch-canary-impact-measurement"
        no_op_reason = "matching-repeated-context-crunch-canary-already-staged"
    elif recommended_action_count > 0:
        activation_state = "activation-ready"
        next_action = "stage-repeated-context-crunch-canary"
        activation_mode = "canary-candidate"
        blocker = None
        no_op_reason = None
    elif status == "no-repeated-context-crunch-cohorts":
        activation_state = "missing-evidence"
        next_action = "rank-repeated-context-crunch-dry-run"
        activation_mode = "evidence-required"
        blocker = "repeated-context-crunch-cohorts"
        no_op_reason = blocker
    elif missing_measurements:
        activation_state = "missing-measurement"
        next_action = "inspect-crunch-coverage-and-projection"
        activation_mode = "evidence-required"
        blocker = missing_measurements[0]
        no_op_reason = blocker
    else:
        activation_state = "ranked"
        next_action = "rank-crunch-opportunity-follow-up"
        activation_mode = "review-required"
        blocker = top_blocker
        no_op_reason = blocker

    follow_up_missing = list(dict.fromkeys(str(item) for item in missing_measurements if str(item or "").strip()))
    if activation_state == "measurement-required" and blocker not in follow_up_missing:
        follow_up_missing.append(blocker)
    if activation_state == "blocked" and blocker not in follow_up_missing:
        follow_up_missing.append(blocker)
    canary_already_staged = canary_applied_rows > 0 or canary_holdout_rows > 0
    savings_status = (
        "projected-savings-ranked"
        if projected_saved_chars > 0 or projected_saved_tokens > 0 or projected_saved_usd > 0
        else "no-positive-projection"
    )

    return {
        "schema": "agentflow.request_shape_crunch_activation_follow_up.v1",
        "status": status,
        "savings_status": savings_status,
        "report_key": report_key,
        "evidence_schema": evidence_schema,
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "rows_considered": rows_considered,
        "activation_state": activation_state,
        "activation_mode": activation_mode,
        "next_action": next_action,
        "target_local_policy": "crunch_rules",
        "policy_section": "crunch",
        "local_file_backed": True,
        "projected_saved_chars": projected_saved_chars,
        "projected_saved_tokens": projected_saved_tokens,
        "projected_saved_usd": round(projected_saved_usd, 6),
        "recommended_action_count": recommended_action_count,
        "canary_applied_rows": canary_applied_rows,
        "canary_holdout_rows": canary_holdout_rows,
        "canary_safety_stopped_rows": canary_safety_stopped_rows,
        "canary_already_staged": canary_already_staged,
        "canary_already_applied": canary_applied_rows > 0,
        "no_op_reason": no_op_reason,
        "duplicate_suppression": {
            "schema": "agentflow.request_shape_crunch_follow_up_duplicate_suppression.v1",
            "suppresses_new_stage_action": recommended_action_count == 0 and (canary_already_staged or canary_safety_stopped_rows > 0),
            "reason": no_op_reason,
            "matching_local_policy": "crunch_rules" if canary_already_staged or canary_safety_stopped_rows > 0 else None,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "top_blocker": blocker or top_blocker,
        "missing_measurements": follow_up_missing,
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
                "source_evidence_schema": row.get("source_schema") or row.get("schema") or ROLLUP_ROW_SCHEMA,
                "readiness": readiness,
                "reason": reason,
                "blockers": decision.get("blockers") or [],
                "evidence_blocker_codes": _public_label_list(row.get("blocker_codes")),
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
    matched_count = sum(_as_int(row.get("row_count") or row.get("count")) for row in crunch_rows)
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
    activation_follow_up = _request_shape_crunch_follow_up(
        status=status,
        report_key="request_shape_crunch_opportunity",
        evidence_schema=CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        candidate_count=len(cohorts),
        matched_count=matched_count,
        rows_considered=matched_count,
        recommended_action_count=len(recommended_actions),
        canary_applied_rows=canary_applied_rows,
        canary_holdout_rows=canary_holdout_rows,
        canary_safety_stopped_rows=canary_safety_stopped_rows,
        projected_saved_chars=projected_chars,
        projected_saved_tokens=projected_tokens,
        projected_saved_usd=projected_savings,
        top_blocker=top_blocker,
        missing_measurements=missing,
    )

    return {
        "schema": CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        "status": status,
        "summary": {
            "candidate_count": len(cohorts),
            "matched_count": matched_count,
            "rows_considered": matched_count,
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
            "activation_state": activation_follow_up["activation_state"],
            "top_next_action": activation_follow_up["next_action"],
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "activation_follow_up": activation_follow_up,
        "recommended_actions": recommended_actions,
        "readiness_breakdown": _breakdown(readiness_counts),
        "reason_breakdown": _breakdown(reason_counts),
        "blocker_reason_breakdown": blocker_breakdown,
        "work_class_breakdown": _breakdown(work_class_counts),
        "cohorts": cohorts[: max(1, min(_as_int(limit) or 25, 1000))],
        "missing_measurements": missing,
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_canary_stage_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    run_id: str | None = None,
    persist_rollups: bool = False,
    rollout_fraction: float = DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
    holdout_fraction: float = DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    rollup_report = build_request_shape_rollups_report(
        store_obj,
        limit=limit,
        persist=persist_rollups,
        run_id=run_id,
    )
    dry_run = build_request_shape_crunch_opportunity_dry_run(
        [row for row in rollup_report.get("rollups") or [] if isinstance(row, dict)],
        limit=25,
        rollout_fraction=rollout_fraction,
        holdout_fraction=holdout_fraction,
    )
    actions = [
        action
        for action in dry_run.get("recommended_actions") or []
        if isinstance(action, dict) and action.get("schema") == CRUNCH_CANARY_ACTION_SCHEMA
    ]
    cohorts = [cohort for cohort in dry_run.get("cohorts") or [] if isinstance(cohort, dict)]
    top_action = actions[0] if actions else None
    top_cohort = cohorts[0] if cohorts else None
    lifecycle_projections = [
        _request_shape_crunch_canary_lifecycle_projection(
            cohort,
            rollout_fraction=rollout_fraction,
            holdout_fraction=holdout_fraction,
        )
        for cohort in cohorts
    ]
    skipped_or_safety_reasons: dict[str, int] = {}
    for item in lifecycle_projections:
        skipped_or_safety = _as_int(item.get("projected_skipped_count")) + _as_int(item.get("projected_safety_stopped_count"))
        if skipped_or_safety:
            _increment(skipped_or_safety_reasons, item.get("reason") or "unknown", skipped_or_safety)
    stage_lifecycle_projection = {
        "schema": "agentflow.request_shape_crunch_canary_stage_lifecycle_projection.v1",
        "cohort_count": len(lifecycle_projections),
        "matched_count": sum(_as_int(item.get("matched_count")) for item in lifecycle_projections),
        "projected_canary_applied_count": sum(_as_int(item.get("projected_canary_applied_count")) for item in lifecycle_projections),
        "projected_canary_holdout_count": sum(_as_int(item.get("projected_canary_holdout_count")) for item in lifecycle_projections),
        "projected_skipped_count": sum(_as_int(item.get("projected_skipped_count")) for item in lifecycle_projections),
        "projected_safety_stopped_count": sum(_as_int(item.get("projected_safety_stopped_count")) for item in lifecycle_projections),
        "projected_fallback_count": sum(_as_int(item.get("projected_fallback_count")) for item in lifecycle_projections),
        "projected_rollback_count": sum(_as_int(item.get("projected_rollback_count")) for item in lifecycle_projections),
        "projected_applied_saved_tokens": sum(_as_int(item.get("projected_applied_saved_tokens")) for item in lifecycle_projections),
        "projected_applied_saved_chars": sum(_as_int(item.get("projected_applied_saved_chars")) for item in lifecycle_projections),
        "projected_applied_saved_usd": round(sum(_as_float(item.get("projected_applied_saved_usd")) for item in lifecycle_projections), 6),
        "skipped_or_safety_reasons": _breakdown(skipped_or_safety_reasons),
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }
    if actions:
        status = "staged"
        next_action = "apply-local-crunch-canary-after-review"
        reason = "staged-repeated-context-crunch-canary"
    else:
        status = "no-stageable-cohort"
        next_action = (dry_run.get("activation_follow_up") or {}).get("next_action") or "rank-repeated-context-crunch-dry-run"
        reason = (dry_run.get("activation_follow_up") or {}).get("top_blocker") or dry_run.get("status") or "no-stageable-cohort"
    return {
        "schema": CRUNCH_CANARY_STAGE_SCHEMA,
        "status": status,
        "ok": bool(actions),
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "run_id": rollup_report.get("run_id"),
        "reason": reason,
        "next_action": next_action,
        "staged_canary_count": len(actions),
        "stage_actions": actions,
        "top_stage_action": top_action,
        "top_cohort": top_cohort,
        "stage_lifecycle_projection": stage_lifecycle_projection,
        "cohort_lifecycle_projections": lifecycle_projections[:25],
        "source_report": {
            "schema": rollup_report.get("schema"),
            "window": rollup_report.get("window"),
            "summary": {
                "rows_considered": (rollup_report.get("summary") or {}).get("rows_considered"),
                "rollup_count": (rollup_report.get("summary") or {}).get("rollup_count"),
                "top_next_action": (rollup_report.get("summary") or {}).get("top_next_action"),
                "body_rows_read": (rollup_report.get("summary") or {}).get("body_rows_read"),
            },
            "crunch_opportunity_summary": dry_run.get("summary"),
            "activation_follow_up": dry_run.get("activation_follow_up"),
        },
        "acceptance": {
            "stages_one_repeated_context_crunch_canary": len(actions) == 1,
            "has_projected_tokens": bool(top_action and _as_int(top_action.get("projected_saved_tokens")) > 0),
            "has_projected_savings": bool(top_action and _as_float(top_action.get("projected_saved_usd")) > 0),
            "has_holdout_metadata": bool(top_action and _as_float(top_action.get("holdout_fraction")) > 0),
            "has_projected_lifecycle_split": bool(
                stage_lifecycle_projection["projected_canary_applied_count"] > 0
                and stage_lifecycle_projection["projected_canary_holdout_count"] > 0
            ),
            "has_safety_stop_metadata": bool(
                top_action
                and isinstance(top_action.get("lifecycle_metadata"), dict)
                and bool(top_action["lifecycle_metadata"].get("emits_safety_stopped"))
            ),
            "unsafe_or_stale_cohorts_remain_skipped": all(
                item.get("readiness") == "measurement-ready"
                or _as_int(item.get("projected_skipped_count")) > 0
                or _as_int(item.get("projected_safety_stopped_count")) > 0
                for item in lifecycle_projections
            ),
        },
        "privacy": _crunch_opportunity_privacy(),
    }


def _cache_replay_canary_cohort_id(cohort: dict[str, Any]) -> str:
    basis = {
        "schema": "agentflow.request_shape_cache_replay_canary_cohort_id_basis.v1",
        "provider_family": cohort.get("provider_family"),
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "stream": bool(cohort.get("stream")),
        "has_tools": bool(cohort.get("has_tools")),
        "cache_status": cohort.get("cache_status"),
        "routing_status": cohort.get("routing_status"),
        "text_bucket": cohort.get("text_bucket"),
        "token_bucket": cohort.get("token_bucket"),
    }
    digest = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:16]
    endpoint = public_label(cohort.get("endpoint"), "unknown").replace("_", "-")
    category = public_label(cohort.get("category"), "unknown").replace("_", "-")
    return f"request-shape-cache-replay:{endpoint}:{category}:{digest}"


def _cache_replay_canary_policy_id(cohort_id: str) -> str:
    digest = hashlib.sha256(cohort_id.encode("utf-8")).hexdigest()[:16]
    return f"local-openai-cache-replay-canary:{digest}"


def _request_shape_cache_replay_canary_lifecycle_projection(
    cohort: dict[str, Any],
    *,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    matched = _as_int(cohort.get("row_count") or cohort.get("cohort_row_count"))
    projected_hits = _as_int(cohort.get("projected_hits"))
    projected_savings = _as_float(cohort.get("projected_savings_usd"))
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)

    readiness = public_label(cohort.get("readiness"), "unknown")
    reason = public_label(cohort.get("reason"), "unknown")
    blockers = [
        public_label(item, "unknown")
        for item in cohort.get("blockers") or []
        if public_label(item, "unknown") != "unknown"
    ]
    applied = holdout_count = skipped = 0
    lifecycle_status = "skipped"
    explicit_reason = reason

    if readiness == "replay-ready" and matched > 0:
        holdout_count = min(matched, int(math.ceil(matched * holdout))) if holdout > 0 else 0
        remaining = max(0, matched - holdout_count)
        applied = min(remaining, int(math.ceil(matched * rollout))) if rollout > 0 else 0
        skipped = max(0, matched - applied - holdout_count)
        lifecycle_status = "projected-applied-holdout" if applied > 0 and holdout_count > 0 else "projected-partial"
        explicit_reason = "projected-cache-replay-applied-and-holdout" if applied > 0 and holdout_count > 0 else "projected-cache-replay-partial"
    else:
        skipped = matched
        lifecycle_status = "skipped"
        explicit_reason = reason or (blockers[0] if blockers else "not-stageable")

    return {
        "schema": "agentflow.request_shape_cache_replay_canary_projected_lifecycle.v1",
        "status": lifecycle_status,
        "reason": explicit_reason,
        "readiness": readiness,
        "matched_count": matched,
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "canary_applied_eligible": applied > 0,
        "canary_holdout_eligible": holdout_count > 0,
        "projected_canary_applied_count": applied,
        "projected_canary_holdout_count": holdout_count,
        "projected_skipped_count": skipped,
        "projected_invalidated_count": 0,
        "projected_bypassed_count": 0,
        "projected_hits": projected_hits,
        "projected_savings_usd": round(projected_savings, 6),
        "projected_applied_hits": _scaled_projection(projected_hits, applied, matched),
        "projected_applied_savings_usd": round(projected_savings * (applied / float(matched)), 6) if matched and applied else 0.0,
        "projected_holdout_hits": _scaled_projection(projected_hits, holdout_count, matched),
        "projected_holdout_savings_usd": round(projected_savings * (holdout_count / float(matched)), 6) if matched and holdout_count else 0.0,
        "blocker_reasons": blockers,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _replayability_privacy(),
    }


def _request_shape_cache_replay_canary_action(
    cohort: dict[str, Any],
    *,
    candidate_count: int,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    cohort_id = str(cohort.get("cohort_id") or _cache_replay_canary_cohort_id(cohort))
    policy_id = _cache_replay_canary_policy_id(cohort_id)
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    lifecycle_projection = _request_shape_cache_replay_canary_lifecycle_projection(
        cohort,
        rollout_fraction=rollout,
        holdout_fraction=holdout,
    )
    return {
        "schema": REPLAY_CACHE_CANARY_ACTION_SCHEMA,
        "action_type": "stage-local-openai-cache-replay-canary",
        "target_local_policy": "cache_canary_policy",
        "policy_section": "cache",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
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
            "cache_status": cohort.get("cache_status"),
            "routing_status": cohort.get("routing_status"),
            "text_bucket": cohort.get("text_bucket"),
            "token_bucket": cohort.get("token_bucket"),
            "readiness": cohort.get("readiness"),
            "reason": cohort.get("reason"),
        },
        "projected_hits": _as_int(cohort.get("projected_hits")),
        "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
        "projected_lifecycle": lifecycle_projection,
        "canary_applied_eligible": lifecycle_projection["canary_applied_eligible"],
        "canary_holdout_eligible": lifecycle_projection["canary_holdout_eligible"],
        "safety_gates": {
            "metadata_only": True,
            "aggregate_only": True,
            "local_file_backed": True,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "cache_entries_written": False,
            "policy_files_written": False,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "openai_responses_only": cohort.get("source_surface") == "openai_responses" and cohort.get("endpoint") == "responses",
            "exact_non_tool_only": not bool(cohort.get("has_tools")) and not bool(cohort.get("stream")),
            "tool_call_cache_enabled": False,
            "streaming_replay_enabled": False,
            "holdout_required": holdout > 0,
            "records_applied_holdout_skipped_invalidation_blocked": True,
            "records_applied_holdout_skipped_bypassed_invalidated_hit_miss": True,
            "records_applied_holdout_hit_miss_bypass_invalidation_blocked_stale_risk": True,
        },
        "cache_decision_metadata": {
            "schema": "agentflow.request_shape_cache_replay_decision_metadata.v1",
            "cache_json_field": "cache_replay_canary",
            "emits_statuses": [
                "applied",
                "holdout",
                "skipped",
                "bypass",
                "bypassed",
                "invalidated",
                "invalidation_blocked",
                "stale_risk",
                "cache_hit",
                "cache_miss",
            ],
            "records_applied": True,
            "records_holdout": True,
            "records_skipped": True,
            "records_bypass": True,
            "records_bypassed": True,
            "records_invalidated": True,
            "records_invalidation_blocked": True,
            "records_stale_risk": True,
            "records_cache_hit": True,
            "records_cache_miss": True,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "lifecycle_metadata": {
            "schema": "agentflow.request_shape_cache_replay_canary_stage_lifecycle_metadata.v1",
            "emits_applied": True,
            "emits_holdout": True,
            "emits_skipped": True,
            "emits_bypass": True,
            "emits_bypassed": True,
            "emits_invalidated": True,
            "emits_invalidation_blocked": True,
            "emits_stale_risk": True,
            "emits_cache_hit": True,
            "emits_cache_miss": True,
            "canary_applied_eligible": lifecycle_projection["canary_applied_eligible"],
            "canary_holdout_eligible": lifecycle_projection["canary_holdout_eligible"],
            "projected_canary_applied_count": lifecycle_projection["projected_canary_applied_count"],
            "projected_canary_holdout_count": lifecycle_projection["projected_canary_holdout_count"],
            "projected_skipped_count": lifecycle_projection["projected_skipped_count"],
            "projected_invalidated_count": lifecycle_projection["projected_invalidated_count"],
            "projected_bypassed_count": lifecycle_projection["projected_bypassed_count"],
            "impact_report": "agentflow.openai_cache_replay_impact.v1",
            "lifecycle_feedback_schema": "agentflow.openai_cache_replay_lifecycle_feedback.v1",
            "metadata_only": True,
            "aggregate_only": True,
        },
        "next_action": "apply-local-cache-replay-canary-after-review",
        "privacy": _replayability_privacy(),
    }


def _cache_replay_stage_skipped_guard_summary(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    skipped = [
        cohort
        for cohort in cohorts
        if isinstance(cohort, dict) and cohort.get("readiness") == "skipped"
    ]
    blocker_counts: dict[str, int] = {}
    skipped_rows = 0
    tool_count = 0
    streaming_count = 0
    invalidation_missing_count = 0
    stale_risk_count = 0
    examples: list[dict[str, Any]] = []

    for cohort in skipped:
        row_count = _as_int(cohort.get("row_count"))
        skipped_rows += row_count
        blockers = [public_label(item, "unknown") for item in cohort.get("blockers") or [] if item]
        reason = public_label(cohort.get("reason"), "unknown")
        if not blockers and reason != "unknown":
            blockers = [reason]
        for blocker in blockers:
            _increment(blocker_counts, blocker)
        has_tools = bool(cohort.get("has_tools")) or any("tool" in blocker for blocker in blockers)
        is_streaming = bool(cohort.get("stream")) or "streaming-replay-not-supported" in blockers
        invalidation_missing = "invalidation-evidence-missing" in blockers
        stale_risk = any("stale" in blocker for blocker in blockers)
        if has_tools:
            tool_count += 1
        if is_streaming:
            streaming_count += 1
        if invalidation_missing:
            invalidation_missing_count += 1
        if stale_risk:
            stale_risk_count += 1
        if len(examples) < 5:
            examples.append(
                {
                    "rank": _as_int(cohort.get("rank")),
                    "reason": reason,
                    "blockers": blockers,
                    "provider_family": cohort.get("provider_family"),
                    "source_surface": cohort.get("source_surface"),
                    "endpoint": cohort.get("endpoint"),
                    "category": cohort.get("category"),
                    "stream": bool(cohort.get("stream")),
                    "has_tools": bool(cohort.get("has_tools")),
                    "row_count": row_count,
                    "projected_hits": _as_int(cohort.get("projected_hits")),
                }
            )

    return {
        "schema": "agentflow.request_shape_cache_replay_canary_skipped_guards.v1",
        "skipped_cohort_count": len(skipped),
        "skipped_rows": skipped_rows,
        "tool_cohort_count": tool_count,
        "streaming_cohort_count": streaming_count,
        "invalidation_missing_cohort_count": invalidation_missing_count,
        "stale_risk_cohort_count": stale_risk_count,
        "blocker_breakdown": _breakdown(blocker_counts),
        "examples": examples,
        "tool_streaming_and_invalidation_missing_remain_skipped": (
            all(cohort.get("readiness") == "skipped" for cohort in skipped)
            and tool_count > 0
            and streaming_count > 0
            and invalidation_missing_count > 0
        ),
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
    }


def build_request_shape_cache_replay_canary_stage_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    run_id: str | None = None,
    persist_rollups: bool = False,
    rollout_fraction: float = DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION,
    holdout_fraction: float = DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    rollup_report = build_request_shape_rollups_report(
        store_obj,
        limit=limit,
        persist=persist_rollups,
        run_id=run_id,
    )
    dry_run = (
        rollup_report.get("cache_replayability_dry_run")
        if isinstance(rollup_report.get("cache_replayability_dry_run"), dict)
        else {}
    )
    cohorts = [
        cohort
        for cohort in dry_run.get("cohorts") or []
        if isinstance(cohort, dict)
        and cohort.get("readiness") == "replay-ready"
        and cohort.get("provider_family") == "openai"
        and cohort.get("source_surface") == "openai_responses"
        and cohort.get("endpoint") == "responses"
        and cohort.get("category") == "chat"
        and not bool(cohort.get("has_tools"))
        and not bool(cohort.get("stream"))
        and _as_int(cohort.get("projected_hits")) > 0
        and _as_float(cohort.get("projected_savings_usd")) > 0
    ]
    actions = [
        _request_shape_cache_replay_canary_action(
            cohort,
            candidate_count=len(cohorts),
            rollout_fraction=rollout_fraction,
            holdout_fraction=holdout_fraction,
        )
        for cohort in cohorts
    ]
    skipped_guards = _cache_replay_stage_skipped_guard_summary(
        [cohort for cohort in dry_run.get("cohorts") or [] if isinstance(cohort, dict)]
    )
    top_action = actions[0] if actions else None
    top_cohort = cohorts[0] if cohorts else None
    if actions:
        status = "staged"
        next_action = "apply-local-cache-replay-canary-after-review"
        reason = "staged-openai-responses-cache-replay-canary"
    else:
        status = "no-stageable-cohort"
        next_action = "rank-request-shape-cache-replayability"
        reason = (dry_run.get("summary") or {}).get("top_blocker_code") or dry_run.get("status") or "no-stageable-cohort"
    return {
        "schema": REPLAY_CACHE_CANARY_STAGE_SCHEMA,
        "status": status,
        "ok": bool(actions),
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "run_id": rollup_report.get("run_id"),
        "reason": reason,
        "next_action": next_action,
        "staged_canary_count": len(actions),
        "stage_actions": actions,
        "top_stage_action": top_action,
        "top_cohort": top_cohort,
        "skipped_cohort_guards": skipped_guards,
        "source_report": {
            "schema": rollup_report.get("schema"),
            "window": rollup_report.get("window"),
            "summary": {
                "rows_considered": (rollup_report.get("summary") or {}).get("rows_considered"),
                "rollup_count": (rollup_report.get("summary") or {}).get("rollup_count"),
                "top_next_action": (rollup_report.get("summary") or {}).get("top_next_action"),
                "body_rows_read": (rollup_report.get("summary") or {}).get("body_rows_read"),
            },
            "cache_replayability_summary": dry_run.get("summary"),
            "readiness_breakdown": dry_run.get("readiness_breakdown"),
            "blocker_breakdown": dry_run.get("blocker_breakdown"),
        },
        "acceptance": {
            "has_replay_ready_openai_responses_cohort": bool(actions),
            "has_projected_hits": bool(top_action and _as_int(top_action.get("projected_hits")) > 0),
            "has_projected_savings": bool(top_action and _as_float(top_action.get("projected_savings_usd")) > 0),
            "writes_no_provider_bodies": bool(top_action and not top_action["safety_gates"]["provider_bodies_included"]),
            "writes_no_cache_entries": bool(top_action and not top_action["safety_gates"]["cache_entries_written"]),
            "has_holdout_metadata": bool(top_action and _as_float(top_action.get("holdout_fraction")) > 0),
            "has_lifecycle_metadata": bool(
                top_action
                and isinstance(top_action.get("lifecycle_metadata"), dict)
                and bool(top_action["lifecycle_metadata"].get("emits_applied"))
                and bool(top_action["lifecycle_metadata"].get("emits_holdout"))
                and bool(top_action["lifecycle_metadata"].get("emits_skipped"))
                and bool(top_action["lifecycle_metadata"].get("emits_bypass"))
                and bool(top_action["lifecycle_metadata"].get("emits_invalidated"))
                and bool(top_action["lifecycle_metadata"].get("emits_invalidation_blocked"))
                and bool(top_action["lifecycle_metadata"].get("emits_stale_risk"))
            ),
            "has_applied_and_holdout_eligibility": bool(
                top_action
                and bool(top_action.get("canary_applied_eligible"))
                and bool(top_action.get("canary_holdout_eligible"))
                and isinstance(top_action.get("projected_lifecycle"), dict)
                and _as_int(top_action["projected_lifecycle"].get("projected_canary_applied_count")) > 0
                and _as_int(top_action["projected_lifecycle"].get("projected_canary_holdout_count")) > 0
            ),
            "records_hit_miss_bypass_invalidation_and_stale_risk": bool(
                top_action
                and isinstance(top_action.get("cache_decision_metadata"), dict)
                and bool(top_action["cache_decision_metadata"].get("records_cache_hit"))
                and bool(top_action["cache_decision_metadata"].get("records_cache_miss"))
                and bool(top_action["cache_decision_metadata"].get("records_bypass"))
                and bool(top_action["cache_decision_metadata"].get("records_invalidated"))
                and bool(top_action["cache_decision_metadata"].get("records_invalidation_blocked"))
                and bool(top_action["cache_decision_metadata"].get("records_stale_risk"))
            ),
            "preserves_tool_and_streaming_guards": all(
                not bool(action.get("conditions", {}).get("has_tools")) and not bool(action.get("conditions", {}).get("stream"))
                for action in actions
            ),
            "stages_only_openai_responses_chat": all(
                action.get("conditions", {}).get("provider_family") == "openai"
                and action.get("conditions", {}).get("source_surface") == "openai_responses"
                and action.get("conditions", {}).get("endpoint") == "responses"
                and action.get("conditions", {}).get("category") == "chat"
                for action in actions
            ),
            "tool_streaming_and_invalidation_missing_cohorts_skipped": bool(
                skipped_guards.get("tool_streaming_and_invalidation_missing_remain_skipped")
            ),
        },
        "privacy": _replayability_privacy(),
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


def _cache_replay_blocker_classification(blocker: str) -> tuple[str, str, str, str]:
    code = public_label(blocker, "unknown")
    if code in {"invalidation-evidence-missing", "unsafe-tool-calls-without-invalidation"}:
        return (
            "collect-invalidation-evidence",
            "collect-cache-invalidation-evidence",
            "blocked",
            "cache",
        )
    if code in {"tools-present", "tool-call-cache-disabled"}:
        return (
            "keep-tool-cache-disabled",
            "keep-tool-cache-disabled",
            "blocked",
            "cache",
        )
    if code in {"streaming-replay-not-supported", "unsupported-streaming-shape"}:
        return (
            "streaming-replay-support-needed",
            "design-streaming-cache-replay-support",
            "blocked",
            "cache",
        )
    if code == "insufficient-repeat-evidence":
        return (
            "insufficient-repeat-evidence",
            "collect-more-repeat-evidence",
            "needs-evidence",
            "cache",
        )
    return (
        "unsupported-safety-shape",
        "keep-cache-replay-noop",
        "blocked",
        "cache",
    )


def build_request_shape_cache_replay_blocker_classification_report(
    cache_replayability_dry_run: dict[str, Any],
    *,
    limit: int = 25,
) -> dict[str, Any]:
    cohorts = [
        cohort
        for cohort in cache_replayability_dry_run.get("cohorts") or []
        if isinstance(cohort, dict) and cohort.get("readiness") == "skipped"
    ]
    class_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for cohort in cohorts:
        row_count = _as_int(cohort.get("row_count"))
        blockers = [
            public_label(item, "unknown")
            for item in cohort.get("blockers") or []
            if public_label(item, "unknown") != "unknown"
        ]
        reason = public_label(cohort.get("reason"), "unknown")
        if not blockers and reason != "unknown":
            blockers = [reason]
        classes: dict[str, dict[str, Any]] = {}
        for blocker in blockers:
            blocker_class, next_action, status, family = _cache_replay_blocker_classification(blocker)
            _increment(blocker_counts, blocker, row_count)
            _increment(class_counts, blocker_class, row_count)
            _increment(next_action_counts, next_action, row_count)
            _increment(status_counts, status, row_count)
            _increment(family_counts, family, row_count)
            classified = classes.setdefault(
                blocker_class,
                {
                    "class": blocker_class,
                    "next_action": next_action,
                    "status": status,
                    "local_action_family": family,
                    "blockers": [],
                },
            )
            classified["blockers"].append(blocker)

        if not classes:
            blocker_class, next_action, status, family = _cache_replay_blocker_classification(reason)
            _increment(class_counts, blocker_class, row_count)
            _increment(next_action_counts, next_action, row_count)
            _increment(status_counts, status, row_count)
            _increment(family_counts, family, row_count)
            classes[blocker_class] = {
                "class": blocker_class,
                "next_action": next_action,
                "status": status,
                "local_action_family": family,
                "blockers": [reason],
            }

        primary = sorted(
            classes.values(),
            key=lambda item: (
                {
                    "collect-invalidation-evidence": 0,
                    "keep-tool-cache-disabled": 1,
                    "streaming-replay-support-needed": 2,
                    "insufficient-repeat-evidence": 3,
                    "unsupported-safety-shape": 4,
                }.get(str(item.get("class")), 99),
                str(item.get("class")),
            ),
        )[0]
        classified_blockers = [
            {
                **item,
                "blockers": sorted(set(public_label(blocker, "unknown") for blocker in item.get("blockers") or [])),
                "emits_cache_apply_action": False,
                "requires_explicit_invalidation_safety_evidence": item.get("class") == "collect-invalidation-evidence",
            }
            for item in sorted(classes.values(), key=lambda item: str(item.get("class")))
        ]
        rows.append(
            {
                "schema": REPLAY_BLOCKER_CLASSIFICATION_ROW_SCHEMA,
                "rank": 0,
                "provider_family": public_label(cohort.get("provider_family"), "unknown"),
                "source_surface": public_label(cohort.get("source_surface"), "unknown"),
                "endpoint": public_label(cohort.get("endpoint"), "unknown"),
                "category": public_label(cohort.get("category"), "unknown"),
                "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
                "stream": bool(cohort.get("stream")),
                "has_tools": bool(cohort.get("has_tools")),
                "cache_status": public_label(cohort.get("cache_status"), "unknown"),
                "routing_status": public_label(cohort.get("routing_status"), "unknown"),
                "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
                "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
                "row_count": row_count,
                "projected_hits": _as_int(cohort.get("projected_hits")),
                "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
                "readiness": "blocked",
                "reason": reason,
                "blocker_codes": sorted(set(blockers)),
                "blocker_class": primary.get("class"),
                "next_action": primary.get("next_action"),
                "local_action_family": primary.get("local_action_family"),
                "classified_blockers": classified_blockers,
                "emits_cache_apply_action": False,
                "requires_explicit_invalidation_safety_evidence": any(
                    item.get("requires_explicit_invalidation_safety_evidence")
                    for item in classified_blockers
                ),
                "aggregate_only": True,
                "privacy": _replayability_privacy(),
            }
        )

    rows.sort(
        key=lambda item: (
            {
                "collect-invalidation-evidence": 5,
                "keep-tool-cache-disabled": 4,
                "streaming-replay-support-needed": 3,
                "insufficient-repeat-evidence": 2,
                "unsupported-safety-shape": 1,
            }.get(str(item.get("blocker_class")), 0),
            _as_int(item.get("row_count")),
            str(item.get("endpoint")),
            str(item.get("category")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    action_rows = [row for row in rows if bool(row.get("emits_cache_apply_action"))]
    unsafe_apply_rows = [
        row
        for row in action_rows
        if not bool(row.get("requires_explicit_invalidation_safety_evidence"))
    ]
    class_breakdown = _breakdown(class_counts)
    top_class = class_breakdown[0]["value"] if class_breakdown else None
    next_action_breakdown = _breakdown(next_action_counts)
    top_next_action = next_action_breakdown[0]["value"] if next_action_breakdown else None
    return {
        "schema": REPLAY_BLOCKER_CLASSIFICATION_SCHEMA,
        "status": "classified" if rows else "no-skipped-cache-replay-cohorts",
        "summary": {
            "skipped_cohort_count": len(rows),
            "skipped_rows": sum(_as_int(row.get("row_count")) for row in rows),
            "classified_blocker_count": sum(
                len(row.get("classified_blockers") or [])
                for row in rows
            ),
            "collect_invalidation_evidence_rows": class_counts.get("collect-invalidation-evidence", 0),
            "keep_tool_cache_disabled_rows": class_counts.get("keep-tool-cache-disabled", 0),
            "streaming_replay_support_needed_rows": class_counts.get("streaming-replay-support-needed", 0),
            "insufficient_repeat_evidence_rows": class_counts.get("insufficient-repeat-evidence", 0),
            "unsupported_safety_shape_rows": class_counts.get("unsupported-safety-shape", 0),
            "top_blocker_class": top_class,
            "top_next_action": top_next_action,
            "cache_apply_action_count": len(action_rows),
            "unsafe_cache_apply_action_count": len(unsafe_apply_rows),
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "class_breakdown": class_breakdown,
        "next_action_breakdown": next_action_breakdown,
        "status_breakdown": _breakdown(status_counts),
        "local_action_family_breakdown": _breakdown(family_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "classifications": rows[:capped_limit],
        "acceptance": {
            "has_tool_blocker_class": class_counts.get("keep-tool-cache-disabled", 0) > 0,
            "has_invalidation_evidence_class": class_counts.get("collect-invalidation-evidence", 0) > 0,
            "has_streaming_support_class": class_counts.get("streaming-replay-support-needed", 0) > 0,
            "has_insufficient_repeat_class": class_counts.get("insufficient-repeat-evidence", 0) > 0,
            "has_unsupported_safety_shape_class": class_counts.get("unsupported-safety-shape", 0) > 0,
            "no_cache_apply_without_invalidation_safety_evidence": len(unsafe_apply_rows) == 0,
            "emits_no_cache_apply_actions": len(action_rows) == 0,
            "metadata_only": True,
            "aggregate_only": True,
        },
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


def _shape_next_action(classes: list[str], blockers: list[str]) -> str:
    class_set = set(classes)
    blocker_set = set(blockers)
    if "repeated_context" in class_set and "replayability" in class_set:
        return "rank-repeated-context-replayability-cohort"
    if "repeated_context" in class_set and "crunch" in class_set:
        return "rank-repeated-context-crunch-dry-run"
    if "routing" in class_set:
        return "stage-routing-lifecycle-evidence"
    if blocker_set:
        return "classify-request-shape-blocker"
    return "keep-observability-only"


def _shape_local_action_family(next_action: str, classes: list[str]) -> str:
    if next_action in {"stage-repeated-context-crunch-canary", "measure-repeated-context-crunch-canary-impact"}:
        return "crunch"
    if next_action == "stage-cache-replay-canary":
        return "cache"
    if next_action in {"collect-thinking-routing-lifecycle-evidence", "stage-routing-lifecycle-evidence"}:
        return "routing"
    if next_action == "rank-repeated-context-replayability-cohort":
        return "cache"
    if next_action == "rank-repeated-context-crunch-dry-run":
        return "crunch"
    if next_action == "stage-routing-lifecycle-evidence":
        return "routing"
    if "replayability" in classes:
        return "cache"
    if "crunch" in classes:
        return "crunch"
    if "routing" in classes:
        return "routing"
    return "cohort-ranking"


def _shape_follow_up_privacy() -> dict[str, Any]:
    privacy = _replayability_privacy()
    privacy["policy_files_written"] = False
    return privacy


def _estimated_cache_replay_savings(row: dict[str, Any], row_count: int) -> tuple[int, float]:
    projected_hits = max(0, row_count - 1)
    if projected_hits <= 0:
        return 0, 0.0
    cost = _as_float(row.get("cost_est_usd"))
    return projected_hits, cost * (projected_hits / float(row_count)) if row_count > 0 else 0.0


def _shape_activation_decision(row: dict[str, Any], classes: list[str], blockers: list[str]) -> dict[str, Any]:
    class_set = set(classes)
    blocker_set = set(blockers)
    row_count = _as_int(row.get("row_count") or row.get("count"))
    projected_hits = 0
    projected_savings = 0.0
    projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    projected_crunch_savings = _as_float(row.get("projected_crunch_savings_usd"))

    if "crunch" in class_set or "repeated_context" in class_set:
        crunch_decision = _shape_crunch_decision(row)
        crunch_readiness = str(crunch_decision.get("readiness") or "unknown")
        if crunch_readiness == "measurement-ready":
            return {
                "readiness_state": "activation-ready",
                "next_action": "stage-repeated-context-crunch-canary",
                "local_action_family": "crunch",
                "actionability_reason": str(crunch_decision.get("reason") or "repeated-context-crunch-opportunity"),
                "projected_hits": 0,
                "projected_saved_tokens": projected_tokens,
                "projected_savings_usd": projected_crunch_savings,
                "blocker_codes": blockers,
            }
        if crunch_readiness in {"canary-staged", "canary-applied", "canary-holdout"}:
            return {
                "readiness_state": "measurement-required",
                "next_action": "measure-repeated-context-crunch-canary-impact",
                "local_action_family": "crunch",
                "actionability_reason": str(crunch_decision.get("reason") or "missing-crunch-canary-impact-measurement"),
                "projected_hits": 0,
                "projected_saved_tokens": projected_tokens,
                "projected_savings_usd": projected_crunch_savings,
                "blocker_codes": list(crunch_decision.get("blockers") or blockers),
            }
        if crunch_readiness == "canary-safety-stopped":
            return {
                "readiness_state": "blocked",
                "next_action": "review-repeated-context-crunch-canary-safety-stop",
                "local_action_family": "crunch",
                "actionability_reason": str(crunch_decision.get("reason") or "canary-safety-stopped"),
                "projected_hits": 0,
                "projected_saved_tokens": projected_tokens,
                "projected_savings_usd": projected_crunch_savings,
                "blocker_codes": list(crunch_decision.get("blockers") or blockers),
            }

    cache_status = public_label(row.get("cache_status"), "unknown")
    stream = bool(row.get("stream"))
    has_tools = bool(row.get("has_tools"))
    if "replayability" in class_set and not stream and not has_tools and cache_status in {"miss", "missing"} and row_count >= 2:
        projected_hits, projected_savings = _estimated_cache_replay_savings(row, row_count)
        return {
            "readiness_state": "activation-ready",
            "next_action": "stage-cache-replay-canary",
            "local_action_family": "cache",
            "actionability_reason": "replay-ready-exact-non-tool-shape",
            "projected_hits": projected_hits,
            "projected_saved_tokens": 0,
            "projected_savings_usd": projected_savings,
            "blocker_codes": [blocker for blocker in blockers if blocker != "exact-cache-miss"],
        }
    if "tool-call-cache-disabled" in blocker_set:
        return {
            "readiness_state": "blocked",
            "next_action": "collect-tool-call-cache-invalidation-evidence",
            "local_action_family": "cache",
            "actionability_reason": "tool-call-cache-needs-invalidation-evidence",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }
    if "unsupported-streaming-shape" in blocker_set and "replayability" in class_set:
        return {
            "readiness_state": "blocked",
            "next_action": "add-streaming-cache-replay-support",
            "local_action_family": "cache",
            "actionability_reason": "streaming-cache-replay-not-supported",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }
    if "thinking-routing-guard" in blocker_set:
        return {
            "readiness_state": "needs-lifecycle-evidence",
            "next_action": "collect-thinking-routing-lifecycle-evidence",
            "local_action_family": "routing",
            "actionability_reason": "thinking-routing-guard-needs-lifecycle-evidence",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }
    if "routing" in class_set:
        return {
            "readiness_state": "needs-lifecycle-evidence",
            "next_action": "stage-routing-lifecycle-evidence",
            "local_action_family": "routing",
            "actionability_reason": "routing-candidate-needs-lifecycle-evidence",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }

    next_action = _shape_next_action(classes, blockers)
    return {
        "readiness_state": "needs-classification" if blockers else "observability-only",
        "next_action": next_action,
        "local_action_family": _shape_local_action_family(next_action, classes),
        "actionability_reason": blockers[0] if blockers else "no-actionable-blocker",
        "projected_hits": 0,
        "projected_saved_tokens": projected_tokens,
        "projected_savings_usd": projected_crunch_savings,
        "blocker_codes": blockers,
    }


def _shape_follow_up_candidate(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    classes = sorted(public_label(item, "unknown") for item in row.get("candidate_work_classes") or [])
    families = sorted(public_label(item, "unknown") for item in row.get("candidate_families") or [])
    blockers = sorted(public_label(item, "unknown") for item in row.get("blocker_codes") or [])
    decision = _shape_activation_decision(row, classes, blockers)
    next_action = str(decision["next_action"])
    provider = public_label(row.get("provider_family"), "unknown")
    source_surface = public_label(row.get("source_surface"), "unknown")
    endpoint = public_label(row.get("endpoint"), "unknown")
    row_count = _as_int(row.get("row_count") or row.get("count"))
    cost = _as_float(row.get("cost_est_usd"))
    observed_savings = _as_float(row.get("observed_savings_usd"))
    error_count = _as_int(row.get("error_count"))
    retry_count = _as_int(row.get("retry_count"))
    projected_crunch_savings = _as_float(row.get("projected_crunch_savings_usd"))
    projected_crunch_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    readiness = str(decision.get("readiness_state") or "unknown")
    readiness_weight = {
        "activation-ready": 500.0,
        "measurement-required": 350.0,
        "needs-lifecycle-evidence": 150.0,
        "blocked": 50.0,
    }.get(readiness, 0.0)
    replay_weight = 100.0 if "replayability" in classes else 0.0
    repeated_weight = 150.0 if "repeated_context" in classes else 0.0
    routing_weight = 75.0 if "routing" in classes else 0.0
    crunch_weight = 75.0 if "crunch" in classes else 0.0
    score = (
        row_count
        + cost * 1000.0
        + observed_savings * 2000.0
        + projected_crunch_savings * 2500.0
        + repeated_weight
        + replay_weight
        + routing_weight
        + crunch_weight
        + readiness_weight
        - error_count * 5.0
        - retry_count * 0.5
    )
    return {
        "schema": FOLLOW_UP_BLOCKER_COHORT_SCHEMA,
        "rank": rank,
        "provider_surface_bucket": "/".join(part for part in (provider, source_surface, endpoint) if part) or "mixed",
        "provider_family": provider,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "requested_model_family": public_label(row.get("requested_model_family"), "unknown"),
        "routed_model_family": public_label(row.get("routed_model_family"), "unknown"),
        "category": public_label(row.get("category"), "unknown"),
        "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": public_label(row.get("text_bucket"), "unknown"),
        "token_bucket": public_label(row.get("token_bucket"), "unknown"),
        "cache_status": public_label(row.get("cache_status"), "unknown"),
        "routing_status": public_label(row.get("routing_status"), "unknown"),
        "row_count": row_count,
        "sample_count": row_count,
        "error_count": error_count,
        "retry_count": retry_count,
        "cost_est_usd": round(cost, 6),
        "observed_savings_usd": round(observed_savings, 6),
        "projected_hits": _as_int(decision.get("projected_hits")),
        "projected_crunch_tokens_saved": projected_crunch_tokens,
        "projected_crunch_savings_usd": round(projected_crunch_savings, 6),
        "projected_saved_tokens": _as_int(decision.get("projected_saved_tokens")),
        "projected_savings_usd": round(_as_float(decision.get("projected_savings_usd")), 6),
        "candidate_work_classes": classes,
        "candidate_families": families,
        "blocker_codes": sorted(public_label(item, "unknown") for item in decision.get("blocker_codes") or []),
        "readiness_state": readiness,
        "actionability_reason": public_label(decision.get("actionability_reason"), "unknown"),
        "next_action": next_action,
        "local_action_family": public_label(decision.get("local_action_family"), "cohort-ranking"),
        "aggregate_only": True,
        "privacy": _shape_follow_up_privacy(),
        "_score": score,
    }


def build_request_shape_follow_up_candidates(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    relevant_classes = {"repeated_context", "replayability", "routing", "crunch"}
    candidates = [
        _shape_follow_up_candidate(row, rank=index)
        for index, row in enumerate(rollups, start=1)
        if isinstance(row, dict)
        and relevant_classes.intersection({str(item) for item in row.get("candidate_work_classes") or []})
    ]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("_score")),
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("cost_est_usd")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )

    class_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    clean: list[dict[str, Any]] = []
    capped_limit = max(1, min(_as_int(limit, 10), 100))
    for rank, candidate in enumerate(candidates[:capped_limit], start=1):
        candidate["rank"] = rank
        row_count = _as_int(candidate.get("row_count"))
        for work_class in candidate.get("candidate_work_classes") or []:
            _increment(class_counts, work_class, row_count)
        for blocker in candidate.get("blocker_codes") or []:
            _increment(blocker_counts, blocker, row_count)
        _increment(action_counts, candidate.get("next_action"), row_count)
        _increment(family_counts, candidate.get("local_action_family"), row_count)
        _increment(readiness_counts, candidate.get("readiness_state"), row_count)
        item = dict(candidate)
        item.pop("_score", None)
        clean.append(item)

    status = "candidates-ranked" if clean else "no-request-shape-follow-up-candidates"
    top = clean[0] if clean else None
    missing = [] if clean else ["request_shape_follow_up_candidates"]
    return {
        "schema": FOLLOW_UP_CANDIDATES_SCHEMA,
        "status": status,
        "summary": {
            "rows_considered": sum(_as_int(row.get("row_count") or row.get("count")) for row in rollups if isinstance(row, dict)),
            "rollup_count": len([row for row in rollups if isinstance(row, dict)]),
            "ranked_candidate_count": len(clean),
            "top_next_action": top.get("next_action") if top else None,
            "top_local_action_family": top.get("local_action_family") if top else None,
            "top_readiness_state": top.get("readiness_state") if top else None,
            "activation_ready_count": sum(1 for item in clean if item.get("readiness_state") == "activation-ready"),
            "class_breakdown": _breakdown(class_counts),
            "blocker_breakdown": _breakdown(blocker_counts),
            "readiness_breakdown": _breakdown(readiness_counts),
            "next_action_breakdown": _breakdown(action_counts),
            "local_action_family_breakdown": _breakdown(family_counts),
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "top_candidate": top,
        "top_blocker_cohort": top,
        "candidates": clean,
        "blocker_cohorts": clean,
        "missing_measurements": missing,
        "privacy": _shape_follow_up_privacy(),
    }


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
    max_crunch_canary_evidence_age_hours: float = DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS,
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
    cache_replayability_dry_run = build_request_shape_cache_replayability_dry_run(rollups, limit=25)
    cache_replay_blocker_classification = build_request_shape_cache_replay_blocker_classification_report(
        cache_replayability_dry_run,
        limit=25,
    )
    crunch_opportunity_dry_run = build_request_shape_crunch_opportunity_dry_run(rollups, limit=25)
    crunch_canary_impact = build_request_shape_crunch_canary_impact_report(
        impact_rows,
        max_evidence_age_hours=max_crunch_canary_evidence_age_hours,
    )
    follow_up_candidates = build_request_shape_follow_up_candidates(rollups, limit=10)

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
            "follow_up_candidate_count": follow_up_candidates["summary"]["ranked_candidate_count"],
            "top_next_action": follow_up_candidates["summary"]["top_next_action"],
            "top_local_action_family": follow_up_candidates["summary"]["top_local_action_family"],
        },
        "provider_breakdown": _breakdown(provider_counts),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_code_breakdown": _breakdown(blocker_counts),
        "follow_up_candidates": follow_up_candidates,
        "cache_replayability_dry_run": cache_replayability_dry_run,
        "cache_replay_blocker_classification": cache_replay_blocker_classification,
        "crunch_opportunity_dry_run": crunch_opportunity_dry_run,
        "crunch_canary_impact": crunch_canary_impact,
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
