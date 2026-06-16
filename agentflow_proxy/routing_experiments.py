from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from agentflow_proxy.crunch import build_embedding, sha256_text
from agentflow_proxy.paths import agentflow_config_path
from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.store import cosine_similarity, stable_json

_PUBLIC_LABEL_RAW_MARKERS = (
    "raw",
    "provider_body",
    "provider body",
    "request_id",
    "request id",
    "session_id",
    "session id",
    "tenant_id",
    "tenant id",
    "account_id",
    "account id",
    "thread_id",
    "thread id",
    "cache_key",
    "cache key",
    "file_path",
    "file path",
    "tool_payload",
    "tool payload",
    "authorization",
    "api_key",
    "api key",
    "secret",
    "transcript",
    "/tmp/",
    "/home/",
    "sk-",
)


def _public_label(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    lowered = text.lower()
    if len(text) > 160 or any(marker in lowered for marker in _PUBLIC_LABEL_RAW_MARKERS):
        return "redacted-metadata-label"
    return text


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _default_experiment_policy() -> dict[str, Any]:
    return {
        "profile_id": "first-safe-openai-codex-claude-shadow-pass-through-v1",
        "mode": "shadow_candidate_pass_through",
        "enabled": True,
        "kill_switch": False,
        "sample_rate": 0.10,
        "daily_budget_usd": 10.0,
        "min_text_chars": 0,
        "max_text_chars": 8000,
        "providers": ["anthropic", "openai"],
        "source_surfaces": ["anthropic_messages", "openai_responses", "openai_chat", "codex_turn"],
        "streaming_shadow_source_surfaces": ["anthropic_messages"],
        "model_pairs": [
            {"requested_model": "claude-sonnet-4-6", "routed_model": "claude-haiku-4-5-20251001"},
            {"requested_model": "claude-opus-4-5", "routed_model": "claude-sonnet-4-6"},
            {"requested_model": "gpt-5-codex", "routed_model": "gpt-5-mini"},
            {"requested_model": "gpt-5.4", "routed_model": "gpt-5.4-mini"},
            {"requested_model": "gpt-5.5", "routed_model": "gpt-5-mini"},
        ],
        "workflow_phases": [],
        "categories": ["chat", "short-completion", "codex-turn"],
        "similarity_threshold": 0.86,
        "min_samples_for_confidence": 20,
        "store_response_bodies": False,
        "eligibility_overrides": [],
    }


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(agentflow_config_path(filename))
    return candidates


def _override_scope(override: dict[str, Any]) -> str:
    requested = str(override.get("scope") or "").strip().replace("_", "-")
    if requested in {"global", "provider", "source-surface", "category"}:
        return requested
    if override.get("category") not in (None, ""):
        return "category"
    if override.get("source_surface") not in (None, ""):
        return "source-surface"
    if override.get("provider") not in (None, ""):
        return "provider"
    return "global"


def _scope_rank(scope: str) -> int:
    return {
        "global": 0,
        "provider": 1,
        "source-surface": 2,
        "category": 3,
    }.get(scope, 0)


def _apply_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    if data.get("profile_id") not in (None, ""):
        policy["profile_id"] = str(data["profile_id"])
    if data.get("mode") not in (None, ""):
        mode = str(data["mode"]).strip()
        if mode in {"applied_routed_down", "shadow_candidate_pass_through"}:
            policy["mode"] = mode
    policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
    policy["kill_switch"] = _as_bool(data.get("kill_switch"), policy["kill_switch"])
    if data.get("sample_rate") is not None:
        policy["sample_rate"] = max(0.0, min(1.0, float(data["sample_rate"])))
    if data.get("daily_budget_usd") is not None:
        policy["daily_budget_usd"] = max(0.0, float(data["daily_budget_usd"]))
    if data.get("min_text_chars") is not None:
        policy["min_text_chars"] = int(data["min_text_chars"])
    if data.get("max_text_chars") is not None:
        policy["max_text_chars"] = int(data["max_text_chars"])
    for key in ("providers", "source_surfaces", "streaming_shadow_source_surfaces", "workflow_phases", "categories"):
        values = data.get(key)
        if isinstance(values, list):
            policy[key] = [str(c) for c in values if c is not None]
    pairs = data.get("model_pairs")
    if isinstance(pairs, list):
        clean_pairs: list[dict[str, str]] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            requested = str(pair.get("requested_model") or pair.get("requested") or "").strip()
            routed = str(pair.get("routed_model") or pair.get("routed") or "").strip()
            if requested and routed:
                clean_pairs.append({"requested_model": requested, "routed_model": routed})
        policy["model_pairs"] = clean_pairs
    elif data.get("requested_model") is not None and data.get("routed_model") is not None:
        policy["model_pairs"] = [
            {
                "requested_model": str(data["requested_model"]),
                "routed_model": str(data["routed_model"]),
            }
        ]
    categories = data.get("categories")
    if isinstance(categories, list):
        policy["categories"] = [str(c) for c in categories if c is not None]
    if data.get("similarity_threshold") is not None:
        policy["similarity_threshold"] = max(0.0, min(1.0, float(data["similarity_threshold"])))
    if data.get("min_samples_for_confidence") is not None:
        policy["min_samples_for_confidence"] = max(1, int(data["min_samples_for_confidence"]))
    policy["store_response_bodies"] = _as_bool(
        data.get("store_response_bodies"),
        policy["store_response_bodies"],
    )
    overrides = data.get("eligibility_overrides")
    if isinstance(overrides, list):
        clean_overrides: list[dict[str, Any]] = []
        for raw in overrides:
            if not isinstance(raw, dict):
                continue
            item: dict[str, Any] = {}
            for key in ("scope", "provider", "source_surface", "category", "workflow_phase", "label"):
                if raw.get(key) not in (None, ""):
                    item[key] = str(raw[key])
            if raw.get("stream") is not None:
                item["stream"] = _as_bool(raw.get("stream"), False)
            if raw.get("min_text_chars") is not None:
                item["min_text_chars"] = max(0, int(raw["min_text_chars"]))
            if raw.get("max_text_chars") is not None:
                item["max_text_chars"] = max(0, int(raw["max_text_chars"]))
            if raw.get("sample_rate") is not None:
                item["sample_rate"] = max(0.0, min(1.0, float(raw["sample_rate"])))
            if raw.get("daily_budget_usd") is not None:
                item["daily_budget_usd"] = max(0.0, float(raw["daily_budget_usd"]))
            if any(key in item for key in ("min_text_chars", "max_text_chars", "sample_rate", "daily_budget_usd")):
                item["scope"] = _override_scope(item)
                clean_overrides.append(item)
        policy["eligibility_overrides"] = clean_overrides
    return policy


def _load_experiment_policy() -> tuple[dict[str, Any], str, str]:
    for path in _manual_rule_candidates("routing_experiments.yaml", "AGENTFLOW_ROUTING_EXPERIMENTS"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return _apply_policy_yaml(_default_experiment_policy(), data), "local-manual", str(path)

    defaults_path = Path(__file__).parent / "routing_experiments.yaml"
    policy = _default_experiment_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _apply_policy_yaml(policy, data)

    policy["enabled"] = os.getenv("AGENTFLOW_ROUTING_EXPERIMENTS_ENABLED", "1" if policy["enabled"] else "0") != "0"
    policy["kill_switch"] = os.getenv(
        "AGENTFLOW_ROUTING_EXPERIMENT_KILL_SWITCH",
        "1" if policy.get("kill_switch") else "0",
    ) != "0"
    policy["sample_rate"] = max(
        0.0,
        min(1.0, float(os.getenv("AGENTFLOW_ROUTING_EXPERIMENT_SAMPLE_RATE", str(policy["sample_rate"])))),
    )
    policy["daily_budget_usd"] = max(
        0.0,
        float(os.getenv("AGENTFLOW_ROUTING_EXPERIMENT_DAILY_BUDGET_USD", str(policy["daily_budget_usd"]))),
    )
    policy["similarity_threshold"] = max(
        0.0,
        min(1.0, float(os.getenv("AGENTFLOW_ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD", str(policy["similarity_threshold"])))),
    )
    return policy, "local-default", str(defaults_path)


ROUTING_EXPERIMENT_POLICY, ROUTING_EXPERIMENT_POLICY_SOURCE, ROUTING_EXPERIMENT_RULES_PATH = _load_experiment_policy()
ROUTING_EXPERIMENT_RULES_LOADED_AT = utc_now()
ROUTING_EXPERIMENT_RULES_LOADED_FILE = policy_file_snapshot(ROUTING_EXPERIMENT_RULES_PATH)
ROUTING_EXPERIMENT_ENABLED = bool(ROUTING_EXPERIMENT_POLICY["enabled"])
ROUTING_EXPERIMENT_MODE = str(ROUTING_EXPERIMENT_POLICY.get("mode") or "applied_routed_down")
ROUTING_EXPERIMENT_SAMPLE_RATE = float(ROUTING_EXPERIMENT_POLICY["sample_rate"])
ROUTING_EXPERIMENT_DAILY_BUDGET_USD = float(ROUTING_EXPERIMENT_POLICY["daily_budget_usd"])
ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD = float(ROUTING_EXPERIMENT_POLICY["similarity_threshold"])
ROUTING_EXPERIMENT_MIN_SAMPLES = int(ROUTING_EXPERIMENT_POLICY["min_samples_for_confidence"])
ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES = bool(ROUTING_EXPERIMENT_POLICY["store_response_bodies"])
ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE = "routing_experiment_outcome"
ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS = 168
ROUTING_PROMOTION_MIN_COMPARED_COVERAGE = 0.80
ROUTING_PROMOTION_MIN_PASS_RATE = 0.90
ROUTING_PROMOTION_MAX_SHADOW_ERROR_RATE = 0.05
ROUTING_PROMOTION_MAX_PRIMARY_ERROR_RATE = 0.05
ROUTING_PROMOTION_SCHEMA = "agentflow.routing_experiment_promotion_verdict.v1"


def _override_matches(
    override: dict[str, Any],
    *,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
) -> bool:
    checks = {
        "provider": provider,
        "source_surface": source_surface,
        "category": category,
        "workflow_phase": workflow_phase,
    }
    for key, actual in checks.items():
        expected = override.get(key)
        if expected not in (None, "") and str(expected) != str(actual):
            return False
    if "stream" in override and bool(override.get("stream")) != bool(stream):
        return False
    return True


def _effective_experiment_controls(
    *,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
) -> dict[str, Any]:
    controls: dict[str, Any] = {
        "min_text_chars": int(ROUTING_EXPERIMENT_POLICY.get("min_text_chars") or 0),
        "min_text_chars_scope": "global",
        "max_text_chars": int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0),
        "max_text_chars_scope": "global",
        "sample_rate": ROUTING_EXPERIMENT_SAMPLE_RATE,
        "sample_rate_scope": "global",
        "daily_budget_usd": ROUTING_EXPERIMENT_DAILY_BUDGET_USD,
        "daily_budget_scope": "global",
        "budget_filter": {},
        "applied_overrides": [],
    }
    overrides = [
        dict(item)
        for item in ROUTING_EXPERIMENT_POLICY.get("eligibility_overrides") or []
        if isinstance(item, dict)
        and _override_matches(
            item,
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
        )
    ]
    overrides.sort(key=lambda item: (_scope_rank(_override_scope(item)), str(item.get("label") or "")))
    for override in overrides:
        scope = _override_scope(override)
        clean_override = {
            "scope": scope,
            "provider": override.get("provider"),
            "source_surface": override.get("source_surface"),
            "category": override.get("category"),
            "workflow_phase": override.get("workflow_phase"),
            "stream": override.get("stream") if "stream" in override else None,
            "label": _public_label(override.get("label"), fallback=f"{scope}-override"),
        }
        for key in ("min_text_chars", "max_text_chars", "sample_rate", "daily_budget_usd"):
            if key in override:
                clean_override[key] = override[key]
        controls["applied_overrides"].append(clean_override)
        if "min_text_chars" in override:
            controls["min_text_chars"] = int(override["min_text_chars"])
            controls["min_text_chars_scope"] = scope
        if "max_text_chars" in override:
            controls["max_text_chars"] = int(override["max_text_chars"])
            controls["max_text_chars_scope"] = scope
        if "sample_rate" in override:
            controls["sample_rate"] = float(override["sample_rate"])
            controls["sample_rate_scope"] = scope
        if "daily_budget_usd" in override:
            controls["daily_budget_usd"] = float(override["daily_budget_usd"])
            controls["daily_budget_scope"] = scope
            controls["budget_filter"] = {
                key: override.get(key)
                for key in ("provider", "source_surface", "category")
                if override.get(key) not in (None, "")
            }
    return controls


def _today_shadow_spend_usd(
    store_obj: Any | None,
    *,
    provider: str | None = None,
    source_surface: str | None = None,
    category: str | None = None,
) -> float:
    if store_obj is None or not hasattr(store_obj, "conn"):
        return 0.0
    clauses = ["date(created_at) = date('now')"]
    params: list[Any] = []
    if provider:
        clauses.append("coalesce(provider, 'anthropic') = ?")
        params.append(provider)
    if source_surface:
        clauses.append("coalesce(source_surface, 'anthropic_messages') = ?")
        params.append(source_surface)
    if category:
        clauses.append("coalesce(category, 'unknown') = ?")
        params.append(category)
    try:
        row = store_obj.conn.execute(
            f"""
            select coalesce(sum(coalesce(shadow_cost_est_usd, 0)), 0) as shadow_spend_usd
            from routing_experiments
            where {' and '.join(clauses)}
            """,
            tuple(params),
        ).fetchone()
    except Exception:
        return 0.0
    return float(row["shadow_spend_usd"] or 0.0) if row else 0.0


def _model_pair_allowed(requested: str, routed: str) -> bool:
    pairs = ROUTING_EXPERIMENT_POLICY.get("model_pairs") or []
    if not pairs:
        return True
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        if str(pair.get("requested_model") or "") == requested and str(pair.get("routed_model") or "") == routed:
            return True
    return False


def _route_down_candidate_for_requested(requested: str) -> str | None:
    for pair in ROUTING_EXPERIMENT_POLICY.get("model_pairs") or []:
        if not isinstance(pair, dict):
            continue
        if str(pair.get("requested_model") or "") != requested:
            continue
        routed = str(pair.get("routed_model") or "").strip()
        if routed and routed != requested:
            return routed
    return None


def _value_allowed(value: str, configured: Any) -> bool:
    values = {str(item) for item in configured or []}
    return not values or value in values


def routing_experiment_decision(
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    *,
    stream: bool,
    provider: str = "anthropic",
    source_surface: str = "anthropic_messages",
    store_obj: Any | None = None,
    random_value: Callable[[], float] = random.random,
) -> dict[str, Any]:
    requested = str(routing_meta.get("requested_model") or body.get("model") or "")
    routed = str(routing_meta.get("routed_model") or body.get("model") or "")
    mode = ROUTING_EXPERIMENT_MODE
    openai_canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
    managed = routing_meta.get("managed_recommendation") if isinstance(routing_meta.get("managed_recommendation"), dict) else {}
    forced_openai_canary_shadow = (
        provider == "openai"
        and not stream
        and requested == routed
        and (
            openai_canary.get("status") == "applied"
            or managed.get("selected_for_shadow_evaluation") is True
        )
    )
    if forced_openai_canary_shadow:
        mode = "shadow_candidate_pass_through"
    category = str(routing_meta.get("category") or "")
    workflow_phase = str(routing_meta.get("workflow_phase") or "")
    text_chars = int(routing_meta.get("text_chars") or 0)
    categories = set(str(c) for c in ROUTING_EXPERIMENT_POLICY.get("categories") or [])
    controls = _effective_experiment_controls(
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
    )
    budget_limit = float(controls["daily_budget_usd"])
    budget_filter = controls.get("budget_filter") or {"provider": provider, "source_surface": source_surface}
    budget_spent = _today_shadow_spend_usd(
        store_obj,
        provider=budget_filter.get("provider"),
        source_surface=budget_filter.get("source_surface"),
        category=budget_filter.get("category"),
    )
    budget_remaining = max(0.0, budget_limit - budget_spent)

    meta = {
        "schema": "agentflow.routing_experiment_decision.v1",
        "enabled": ROUTING_EXPERIMENT_ENABLED,
        "mode": mode,
        "kill_switch": bool(ROUTING_EXPERIMENT_POLICY.get("kill_switch")),
        "status": "skipped",
        "sampled": False,
        "reason": "disabled",
        "counterfactual": False,
        "shadow_only": False,
        "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
        "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
        "sample_rate": round(float(controls["sample_rate"]), 6),
        "sample_rate_scope": controls["sample_rate_scope"],
        "daily_budget_usd": round(budget_limit, 6),
        "daily_budget_scope": controls["daily_budget_scope"],
        "profile_id": str(ROUTING_EXPERIMENT_POLICY.get("profile_id") or ""),
        "budget_spent_usd": round(budget_spent, 6),
        "budget_remaining_usd": round(budget_remaining, 6),
        "budget_exhausted": budget_limit <= 0 or budget_spent >= budget_limit,
        "budget_cap_scope": controls["daily_budget_scope"],
        "similarity_threshold": ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD,
        "min_samples_for_confidence": ROUTING_EXPERIMENT_MIN_SAMPLES,
        "provider": provider,
        "source_surface": source_surface,
        "stream": bool(stream),
        "streaming_shadow_supported": (
            bool(stream)
            and mode == "shadow_candidate_pass_through"
            and _value_allowed(source_surface, ROUTING_EXPERIMENT_POLICY.get("streaming_shadow_source_surfaces"))
        ),
        "requested_model": requested,
        "routed_model": routed,
        "shadow_model": requested,
        "primary_model": routed,
        "user_visible_model": routed,
        "category": category,
        "workflow_phase": workflow_phase,
        "text_chars": text_chars,
        "min_text_chars": int(controls["min_text_chars"]),
        "min_text_chars_scope": controls["min_text_chars_scope"],
        "max_text_chars": int(controls["max_text_chars"]),
        "max_text_chars_scope": controls["max_text_chars_scope"],
        "eligibility_overrides_applied": controls["applied_overrides"],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": bool(ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES),
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
        },
    }
    if not ROUTING_EXPERIMENT_ENABLED:
        if not forced_openai_canary_shadow:
            return meta
    if meta["kill_switch"]:
        meta["reason"] = "kill-switch"
        return meta
    if not _value_allowed(provider, ROUTING_EXPERIMENT_POLICY.get("providers")):
        meta["reason"] = "provider-not-enabled"
        return meta
    if not _value_allowed(source_surface, ROUTING_EXPERIMENT_POLICY.get("source_surfaces")):
        meta["reason"] = "source-surface-not-enabled"
        return meta
    if stream and not meta["streaming_shadow_supported"]:
        meta["reason"] = "streaming-shadow-unsupported"
        return meta
    if not requested:
        meta["reason"] = "missing-requested-model"
        return meta
    if mode == "shadow_candidate_pass_through":
        if requested != routed:
            meta["reason"] = "already-routed-down"
            return meta
        candidate = None
        force_shadow = False
        if provider == "openai" and openai_canary.get("status") == "applied":
            candidate = str(openai_canary.get("shadow_model") or openai_canary.get("target_model") or "").strip() or None
            force_shadow = candidate is not None
            meta["trigger"] = "openai-local-routing-canary"
            meta["canary_policy_id"] = openai_canary.get("policy_id")
            meta["canary_cohort"] = openai_canary.get("cohort")
        elif provider == "openai" and managed.get("selected_for_shadow_evaluation") is True:
            candidate = str(managed.get("shadow_model") or managed.get("would_route_model") or "").strip() or None
            force_shadow = candidate is not None
            meta["trigger"] = "managed-policy-routing-canary"
            meta["managed_policy_id"] = managed.get("policy_id")
            meta["canary_cohort"] = (managed.get("local_canary") or {}).get("cohort") if isinstance(managed.get("local_canary"), dict) else None
        if candidate is None:
            candidate = _route_down_candidate_for_requested(requested)
        if not candidate:
            meta["reason"] = "model-pair-not-enabled"
            return meta
        meta["routed_model"] = candidate
        meta["shadow_model"] = candidate
        meta["primary_model"] = requested
        meta["user_visible_model"] = requested
        meta["counterfactual"] = True
        meta["shadow_only"] = True
    elif mode == "applied_routed_down":
        force_shadow = False
        if requested == routed:
            meta["reason"] = "not-routed-down"
            return meta
        if not _model_pair_allowed(requested, routed):
            meta["reason"] = "model-pair-not-enabled"
            return meta
    else:
        meta["reason"] = "unsupported-mode"
        return meta
    if routing_meta.get("fallback_reason"):
        meta["reason"] = "fallback-used"
        return meta
    min_chars = int(controls["min_text_chars"])
    max_chars = int(controls["max_text_chars"])
    if text_chars < min_chars:
        scope = str(controls["min_text_chars_scope"])
        meta["skip_diagnostic"] = f"{scope}-min-text-chars-not-met"
        meta["reason"] = "request-too-small" if scope == "global" else meta["skip_diagnostic"]
        return meta
    if max_chars > 0 and text_chars > max_chars:
        scope = str(controls["max_text_chars_scope"])
        meta["skip_diagnostic"] = f"{scope}-max-text-chars-exceeded"
        meta["reason"] = "request-too-large" if scope == "global" else meta["skip_diagnostic"]
        return meta
    if categories and category not in categories:
        meta["reason"] = "category-not-enabled"
        return meta
    workflow_phases = set(str(c) for c in ROUTING_EXPERIMENT_POLICY.get("workflow_phases") or [])
    if workflow_phases and workflow_phase not in workflow_phases:
        meta["reason"] = "workflow-phase-not-enabled"
        return meta
    if budget_limit <= 0 and not force_shadow:
        meta["reason"] = "daily-budget-zero"
        return meta
    if budget_spent >= budget_limit and not force_shadow:
        meta["reason"] = "daily-budget-exhausted"
        return meta
    sample_rate = float(controls["sample_rate"])
    if sample_rate <= 0 and not force_shadow:
        meta["reason"] = "sample-rate-zero"
        return meta
    if not force_shadow and random_value() >= sample_rate:
        meta["reason"] = "streaming-shadow-not-sampled" if stream else "sample-rate-not-selected"
        return meta

    meta["status"] = "selected"
    meta["sampled"] = True
    if force_shadow:
        meta["sampled_by_canary"] = True
        meta["sample_rate"] = 1.0
        meta["sample_rate_scope"] = "canary-selected"
    if stream and mode == "shadow_candidate_pass_through":
        meta["reason"] = "streaming-shadow-sampled"
    else:
        meta["reason"] = (
            "sampled-shadow-candidate-pass-through"
            if mode == "shadow_candidate_pass_through"
            else "sampled-routed-down-call"
        )
    return meta


def response_output_text(resp: dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return ""
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts: list[str] = []
    for block in resp.get("content") or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block.get("content"), str):
                parts.append(block["content"])
    for choice in resp.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            parts.append(message["content"])
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            parts.append(delta["content"])
    return "\n".join(parts)


def compare_response_outputs(primary_response: dict[str, Any] | None, shadow_response: dict[str, Any] | None) -> dict[str, Any]:
    primary_text = response_output_text(primary_response or {})
    shadow_text = response_output_text(shadow_response or {})
    if primary_text or shadow_text:
        similarity = cosine_similarity(build_embedding(primary_text), build_embedding(shadow_text))
    else:
        similarity = 1.0 if stable_json(primary_response or {}) == stable_json(shadow_response or {}) else 0.0
    return {
        "primary_output_chars": len(primary_text),
        "shadow_output_chars": len(shadow_text),
        "primary_output_sha256": sha256_text(primary_text),
        "shadow_output_sha256": sha256_text(shadow_text),
        "output_similarity": round(float(similarity), 6),
        "passed_threshold": float(similarity) >= ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD,
    }


def _text_chars_bucket(text_chars: Any) -> str:
    try:
        chars = int(text_chars or 0)
    except (TypeError, ValueError):
        chars = 0
    if chars < 2000:
        return "lt-2k"
    if chars < 8000:
        return "2k-8k"
    if chars < 30000:
        return "8k-30k"
    return "gte-30k"


def _model_family(model: Any) -> str | None:
    if not model:
        return None
    model_l = str(model).lower()
    for family in ("haiku", "sonnet", "opus", "codex", "gpt-5", "gpt-4", "gpt-3"):
        if family in model_l:
            return family
    return "other"


def _app_family(provider: Any, source_surface: Any, requested_model: Any) -> str:
    provider_l = str(provider or "").lower()
    surface_l = str(source_surface or "").lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" or surface_l == "anthropic_messages":
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def _status_class(status_code: Any) -> str:
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        return "missing"
    if code < 200:
        return "other"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _hash_identifier(value: Any) -> str | None:
    if not value:
        return None
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def routing_experiment_feedback_features(
    *,
    experiment_id: str,
    experiment_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    comparison: dict[str, Any],
    primary_model: str,
    shadow_model: str,
    primary_status_code: int | None,
    shadow_status_code: int | None,
    primary_latency_ms: int | None,
    shadow_latency_ms: int | None,
    primary_cost_est_usd: float | None,
    shadow_cost_est_usd: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    category = str(routing_meta.get("category") or experiment_meta.get("category") or "unknown")
    requested_model = str(experiment_meta.get("requested_model") or routing_meta.get("requested_model") or "")
    routed_model = str(experiment_meta.get("routed_model") or routing_meta.get("routed_model") or "")
    shadow_preflight = experiment_meta.get("shadow_request_preflight") if isinstance(experiment_meta, dict) else None
    unsupported_shape_reason = None
    if isinstance(shadow_preflight, dict) and shadow_preflight.get("status") == "unsupported":
        unsupported_shape_reason = _public_label(shadow_preflight.get("reason"), fallback="unsupported-shape")
    unavailable_reason = None
    if isinstance(shadow_preflight, dict) and shadow_preflight.get("status") == "unavailable":
        unavailable_reason = _public_label(shadow_preflight.get("reason"), fallback="shadow-unavailable")
    compared = (
        primary_status_code is not None
        and primary_status_code < 400
        and shadow_status_code is not None
        and shadow_status_code < 400
        and comparison.get("output_similarity") is not None
    )
    status = "compared" if compared else "shadow-unavailable"
    if unsupported_shape_reason:
        status = "shadow-unsupported-shape"
    elif unavailable_reason:
        status = "shadow-unavailable"
    elif shadow_status_code == 400:
        status = "shadow-http-400"
    if error:
        if unsupported_shape_reason:
            status = "shadow-unsupported-shape"
        elif unavailable_reason:
            status = "shadow-unavailable"
        elif shadow_status_code == 400:
            status = "shadow-http-400"
        else:
            status = "shadow-error"
    reason_codes: list[str] = []
    if primary_status_code is not None and primary_status_code >= 400:
        reason_codes.append("primary-error")
    if unsupported_shape_reason:
        reason_codes.append("shadow-unsupported-shape")
        reason_codes.append(f"unsupported-shadow-shape-{unsupported_shape_reason}")
    if unavailable_reason:
        reason_codes.append("shadow-unavailable")
        reason_codes.append(f"shadow-unavailable-{unavailable_reason}")
    if shadow_status_code is None:
        if not unsupported_shape_reason and not unavailable_reason:
            reason_codes.append("shadow-missing")
    elif shadow_status_code >= 400:
        reason_codes.append("shadow-http-400" if shadow_status_code == 400 else "shadow-error")
    if error and not unsupported_shape_reason and not unavailable_reason and shadow_status_code != 400:
        reason_codes.append("shadow-exception")
    if compared and not comparison.get("passed_threshold"):
        reason_codes.append("below-similarity-threshold")
    if compared and comparison.get("passed_threshold"):
        reason_codes.append("passed")
    return {
        "schema": "agentflow.routing_experiment_feedback.v1",
        "experiment_id": experiment_id,
        "sampled": bool(experiment_meta.get("sampled")),
        "mode": experiment_meta.get("mode") or "applied_routed_down",
        "counterfactual": bool(experiment_meta.get("counterfactual")),
        "shadow_only": bool(experiment_meta.get("shadow_only")),
        "status": status,
        "provider": experiment_meta.get("provider") or "anthropic",
        "source_surface": experiment_meta.get("source_surface") or "anthropic_messages",
        "requested_model": requested_model,
        "routed_model": routed_model,
        "primary_model": primary_model,
        "shadow_model": shadow_model,
        "category": category,
        "workflow_phase": routing_meta.get("workflow_phase") or experiment_meta.get("workflow_phase"),
        "candidate_bucket": f"{category}:{requested_model}->{routed_model}",
        "text_chars_bucket": _text_chars_bucket(experiment_meta.get("text_chars") or routing_meta.get("text_chars")),
        "routing_reason": routing_meta.get("reason"),
        "primary_status_code": primary_status_code,
        "shadow_status_code": shadow_status_code,
        "primary_latency_ms": primary_latency_ms,
        "shadow_latency_ms": shadow_latency_ms,
        "primary_cost_est_usd": primary_cost_est_usd,
        "shadow_cost_est_usd": shadow_cost_est_usd,
        "primary_output_chars": comparison.get("primary_output_chars"),
        "shadow_output_chars": comparison.get("shadow_output_chars"),
        "primary_output_sha256": comparison.get("primary_output_sha256"),
        "shadow_output_sha256": comparison.get("shadow_output_sha256"),
        "output_similarity": comparison.get("output_similarity"),
        "similarity_threshold": experiment_meta.get("similarity_threshold"),
        "passed_threshold": bool(comparison.get("passed_threshold")),
        "reason_codes": reason_codes,
        "shadow_unsupported_shape_reason": unsupported_shape_reason,
        "shadow_unavailable_reason": unavailable_reason,
        "shadow_error_class": _public_label(error, fallback="none") if error else None,
        "cost_delta_usd": (
            round(float(primary_cost_est_usd or 0.0) - float(shadow_cost_est_usd or 0.0), 6)
            if primary_cost_est_usd is not None or shadow_cost_est_usd is not None else None
        ),
        "latency_delta_ms": (
            int(primary_latency_ms or 0) - int(shadow_latency_ms or 0)
            if primary_latency_ms is not None or shadow_latency_ms is not None else None
        ),
        "privacy": experiment_meta.get("privacy") or {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
        },
        "error_present": bool(error),
    }


def routing_experiment_outcome_event(feedback_features: dict[str, Any]) -> dict[str, Any]:
    """Build a metadata-only policy event for managed A/B evidence ingestion."""
    provider = feedback_features.get("provider") or "anthropic"
    source_surface = feedback_features.get("source_surface") or "anthropic_messages"
    requested_model = feedback_features.get("requested_model")
    routed_model = feedback_features.get("routed_model")
    shadow_model = feedback_features.get("shadow_model")
    requested_family = _model_family(requested_model)
    routed_family = _model_family(routed_model)
    shadow_family = _model_family(shadow_model)
    primary_status_class = _status_class(feedback_features.get("primary_status_code"))
    shadow_status_class = _status_class(feedback_features.get("shadow_status_code"))
    compared = feedback_features.get("status") == "compared"
    mode = str(feedback_features.get("mode") or "applied_routed_down")
    counterfactual = bool(feedback_features.get("counterfactual"))
    shadow_only = bool(feedback_features.get("shadow_only"))
    reason_codes = [
        str(item)
        for item in feedback_features.get("reason_codes") or []
        if item is not None
    ]
    event = {
        "schema": "agentflow.routing_experiment_outcome_event.v1",
        "event_type": "routing_experiment_outcome",
        "generated_at": utc_now(),
        "source_surface": source_surface,
        "app_family": _app_family(provider, source_surface, requested_model),
        "provider": provider,
        "workflow_phase": feedback_features.get("workflow_phase") or "unknown",
        "category": feedback_features.get("category") or "unknown",
        "candidate": {
            "schema": "agentflow.routing_experiment_candidate.v1",
            "mode": mode,
            "counterfactual": counterfactual,
            "shadow_only": shadow_only,
            "candidate_bucket": (
                f"{feedback_features.get('category') or 'unknown'}:{requested_family or 'unknown'}->{routed_family or 'unknown'}"
            ),
            "text_chars_bucket": feedback_features.get("text_chars_bucket"),
            "requested_model_family": requested_family,
            "routed_model_family": routed_family,
            "shadow_model_family": shadow_family,
        },
        "outcome": {
            "schema": "agentflow.routing_experiment_outcome_summary.v1",
            "mode": mode,
            "counterfactual": counterfactual,
            "shadow_only": shadow_only,
            "sampled": bool(feedback_features.get("sampled")),
            "status": feedback_features.get("status"),
            "compared": compared,
            "passed_threshold": bool(feedback_features.get("passed_threshold")),
            "primary_status_class": primary_status_class,
            "shadow_status_class": shadow_status_class,
            "status_class_pair": f"{primary_status_class}:{shadow_status_class}",
            "output_similarity": feedback_features.get("output_similarity"),
            "similarity_threshold": feedback_features.get("similarity_threshold"),
            "latency_delta_ms": feedback_features.get("latency_delta_ms"),
            "cost_delta_usd": feedback_features.get("cost_delta_usd"),
            "primary_output_sha256": feedback_features.get("primary_output_sha256"),
            "shadow_output_sha256": feedback_features.get("shadow_output_sha256"),
            "error_present": bool(feedback_features.get("error_present")),
        },
        "reason_codes": reason_codes,
        "routing": {
            "schema": "agentflow.routing_experiment_routing_basis.v1",
            "routing_reason": feedback_features.get("routing_reason"),
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_safe": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "tenant_ids_included": False,
            "secrets_included": False,
        },
    }
    experiment_hash = _hash_identifier(feedback_features.get("experiment_id"))
    if experiment_hash:
        event["experiment_hash"] = experiment_hash
    return event


def _parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    try:
        parsed = yaml.safe_load(value) or {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_hours(value: Any) -> float | None:
    parsed = _parse_utc(value)
    now = _parse_utc(utc_now())
    if parsed is None or now is None:
        return None
    return max(0.0, (now - parsed).total_seconds() / 3600.0)


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _sample_mode_from_experiment(experiment: dict[str, Any]) -> str:
    mode = str(experiment.get("mode") or "").strip()
    if mode:
        return _public_label(mode)
    if experiment.get("shadow_only") or experiment.get("counterfactual"):
        return "shadow_candidate_pass_through"
    return "applied_routed_down"


def _workflow_phase_from_payloads(experiment: dict[str, Any], routing: dict[str, Any]) -> str:
    for source in (experiment, routing):
        value = source.get("workflow_phase") if isinstance(source, dict) else None
        if value not in (None, ""):
            return _public_label(value)
    feedback = experiment.get("managed_feedback") if isinstance(experiment, dict) else None
    if isinstance(feedback, dict) and feedback.get("workflow_phase") not in (None, ""):
        return _public_label(feedback["workflow_phase"])
    return "unknown"


def _promotion_scope_for_mode(mode: str) -> str:
    if mode == "shadow_candidate_pass_through":
        return "stage_local_canary_from_shadow"
    if mode == "applied_routed_down":
        return "widen_applied_canary"
    return "review_only"


def _score_routing_promotion_candidate(
    item: dict[str, Any],
    *,
    policy: dict[str, Any],
    today_spend_usd: float,
    budget_limit_usd: float,
) -> dict[str, Any]:
    samples = int(item.get("samples") or 0)
    compared = int(item.get("compared_samples") or 0)
    pass_rate = item.get("pass_rate")
    shadow_error_rate = float(item.get("shadow_error_rate") or 0.0)
    primary_error_rate = float(item.get("primary_error_rate") or 0.0)
    cost_delta = float(item.get("cost_delta_usd") or 0.0)
    last_sample_age_hours = item.get("last_sample_age_hours")
    min_samples = int(policy.get("min_samples_for_confidence") or ROUTING_EXPERIMENT_MIN_SAMPLES)
    compared_coverage = round(compared / samples, 4) if samples else 0.0
    budget_exhausted = bool(ROUTING_EXPERIMENT_ENABLED and (budget_limit_usd <= 0 or today_spend_usd >= budget_limit_usd))
    stale = last_sample_age_hours is None or float(last_sample_age_hours) > ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS

    reason_codes: list[str] = []
    verdict = "promote"
    if samples < min_samples:
        verdict = "needs_more_samples"
        reason_codes.append("insufficient-samples")
    if compared < min_samples:
        verdict = "needs_more_samples"
        reason_codes.append("insufficient-compared-samples")
    if compared_coverage < ROUTING_PROMOTION_MIN_COMPARED_COVERAGE:
        verdict = "needs_more_samples"
        reason_codes.append("insufficient-compared-coverage")
    if stale:
        if verdict == "promote":
            verdict = "hold"
        reason_codes.append("stale-evidence")
    if budget_exhausted:
        if verdict == "promote":
            verdict = "hold"
        reason_codes.append("daily-budget-exhausted")
    if primary_error_rate > ROUTING_PROMOTION_MAX_PRIMARY_ERROR_RATE:
        if verdict == "promote":
            verdict = "hold"
        reason_codes.append("primary-error-rate-high")
    if shadow_error_rate > ROUTING_PROMOTION_MAX_SHADOW_ERROR_RATE:
        verdict = "reject"
        reason_codes.append("shadow-error-rate-high")
    if int(item.get("shadow_http_400_samples") or 0) > 0:
        if verdict == "promote":
            verdict = "hold"
        reason_codes.append("shadow-http-400-observed")
    if int(item.get("shadow_unsupported_shape_samples") or 0) > 0:
        if verdict == "promote":
            verdict = "hold"
        reason_codes.append("shadow-unsupported-shape-observed")
    if int(item.get("shadow_unavailable_samples") or 0) > 0:
        if verdict == "promote":
            verdict = "needs_more_samples"
        reason_codes.append("shadow-unavailable-observed")
    if pass_rate is None:
        if verdict == "promote":
            verdict = "needs_more_samples"
        reason_codes.append("missing-similarity-pass-rate")
    elif float(pass_rate) < ROUTING_PROMOTION_MIN_PASS_RATE:
        verdict = "reject"
        reason_codes.append("below-similarity-pass-rate")
    if cost_delta < 0:
        verdict = "reject"
        reason_codes.append("shadow-more-expensive")
    if int(item.get("fallback_or_retry_count") or 0) > 0:
        if verdict == "promote":
            verdict = "hold"
        reason_codes.append("fallback-or-retry-observed")
    if not reason_codes:
        reason_codes.append("promotion-thresholds-met")

    mode = str(item.get("mode") or "unknown")
    return {
        "schema": ROUTING_PROMOTION_SCHEMA,
        "verdict": verdict,
        "promotion_ready": verdict == "promote",
        "reason_codes": sorted(set(reason_codes)),
        "evidence_kind": "shadow_pass_through" if mode == "shadow_candidate_pass_through" else "applied_canary",
        "promotion_scope": _promotion_scope_for_mode(mode),
        "canary_evidence": mode == "applied_routed_down",
        "shadow_only": mode == "shadow_candidate_pass_through",
        "thresholds": {
            "min_samples": min_samples,
            "min_compared_coverage": ROUTING_PROMOTION_MIN_COMPARED_COVERAGE,
            "min_similarity_pass_rate": ROUTING_PROMOTION_MIN_PASS_RATE,
            "max_shadow_error_rate": ROUTING_PROMOTION_MAX_SHADOW_ERROR_RATE,
            "max_primary_error_rate": ROUTING_PROMOTION_MAX_PRIMARY_ERROR_RATE,
            "freshness_max_age_hours": ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS,
        },
        "coverage": {
            "samples": samples,
            "compared_samples": compared,
            "sample_coverage": round(min(1.0, samples / min_samples), 4) if min_samples else 0.0,
            "compared_coverage": compared_coverage,
        },
        "budget": {
            "daily_budget_usd": round(budget_limit_usd, 6),
            "today_shadow_spend_usd": round(today_spend_usd, 6),
            "daily_budget_exhausted": budget_exhausted,
        },
    }


def _text_chars_from_routing(routing: dict[str, Any]) -> int:
    for source in (routing.get("routing_experiment"), routing):
        if not isinstance(source, dict):
            continue
        value = source.get("text_chars")
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
    return 0


def _build_shadow_eligibility_projection(conn: Any) -> dict[str, Any]:
    global_min = int(ROUTING_EXPERIMENT_POLICY.get("min_text_chars") or 0)
    global_max = int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0)
    try:
        rows = conn.execute(
            """
            select coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   coalesce(category, 'unknown') as category,
                   coalesce(stream, 0) as stream,
                   requested_model,
                   routed_model,
                   routing_json
            from calls
            where routing_json is not null
            order by created_at desc
            limit 5000
            """
        ).fetchall()
    except Exception:
        rows = []

    grouped: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    for row in rows:
        routing = _parse_jsonish(row["routing_json"])
        if not routing:
            continue
        provider = str(row["provider"] or "unknown")
        source_surface = str(row["source_surface"] or "unknown")
        category = str(row["category"] or routing.get("category") or "unknown")
        stream = bool(row["stream"])
        text_chars = _text_chars_from_routing(routing)
        workflow_phase = str(routing.get("workflow_phase") or "")
        controls = _effective_experiment_controls(
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
        )
        effective_min = int(controls["min_text_chars"])
        effective_max = int(controls["max_text_chars"])
        globally_cap_eligible = text_chars >= global_min and (global_max <= 0 or text_chars <= global_max)
        effectively_cap_eligible = text_chars >= effective_min and (effective_max <= 0 or text_chars <= effective_max)
        key = (provider, source_surface, category, stream)
        item = grouped.setdefault(
            key,
            {
                "provider": _public_label(provider),
                "source_surface": _public_label(source_surface),
                "category": _public_label(category),
                "stream": stream,
                "observed_call_count": 0,
                "global_cap_eligible_count": 0,
                "effective_cap_eligible_count": 0,
                "newly_eligible_call_count": 0,
                "blocked_by_global_cap_count": 0,
                "global_min_text_chars": global_min,
                "global_max_text_chars": global_max,
                "effective_min_text_chars": effective_min,
                "effective_max_text_chars": effective_max,
                "effective_max_text_chars_scope": controls["max_text_chars_scope"],
                "effective_sample_rate": round(float(controls["sample_rate"]), 6),
                "effective_sample_rate_scope": controls["sample_rate_scope"],
                "effective_daily_budget_usd": round(float(controls["daily_budget_usd"]), 6),
                "effective_daily_budget_scope": controls["daily_budget_scope"],
                "applied_overrides": controls["applied_overrides"],
            },
        )
        item["observed_call_count"] += 1
        if globally_cap_eligible:
            item["global_cap_eligible_count"] += 1
        elif text_chars >= global_min and global_max > 0 and text_chars > global_max:
            item["blocked_by_global_cap_count"] += 1
        if effectively_cap_eligible:
            item["effective_cap_eligible_count"] += 1
        if not globally_cap_eligible and effectively_cap_eligible:
            item["newly_eligible_call_count"] += 1

    rows_out = sorted(
        grouped.values(),
        key=lambda item: (
            item["provider"] != "anthropic",
            not bool(item["stream"]),
            -int(item["newly_eligible_call_count"]),
            -int(item["observed_call_count"]),
            str(item["category"]),
        ),
    )
    claude_streaming = [
        item for item in rows_out
        if item["provider"] == "anthropic" and item["source_surface"] == "anthropic_messages" and item["stream"]
    ]
    return {
        "schema": "agentflow.routing_experiment_eligibility_projection.v1",
        "observed_window_limit": 5000,
        "global_min_text_chars": global_min,
        "global_max_text_chars": global_max,
        "claude_streaming": claude_streaming,
        "rows": rows_out,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
        },
    }


def _increment_count(target: dict[str, int], key: Any, *, fallback: str = "unknown", amount: int = 1) -> None:
    label = _public_label(key, fallback=fallback)
    target[label] = int(target.get(label, 0)) + int(amount)


def _count_rows(mapping: dict[str, int], *, key_name: str = "reason") -> list[dict[str, Any]]:
    return [
        {key_name: key, "count": count}
        for key, count in sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
    ]


def _since_cutoff_iso(*, since: str | None = None, window_hours: float | None = 24.0) -> str | None:
    if since:
        parsed = _parse_utc(since)
        if parsed is not None:
            return parsed.isoformat()
        return str(since)
    if window_hours is None:
        return None
    try:
        hours = float(window_hours)
    except (TypeError, ValueError):
        hours = 24.0
    if hours <= 0:
        return None
    now = _parse_utc(utc_now()) or datetime.now(timezone.utc)
    return (now - timedelta(hours=hours)).isoformat()


def _experiment_from_routing_json(value: Any) -> dict[str, Any]:
    routing = _parse_jsonish(value)
    experiment = routing.get("routing_experiment") if isinstance(routing, dict) else None
    return experiment if isinstance(experiment, dict) else {}


def _comparison_blocker(row: Any) -> str:
    experiment = _parse_jsonish(row["experiment_json"]) if "experiment_json" in row.keys() else {}
    feedback = experiment.get("optimization_feedback") if isinstance(experiment, dict) else {}
    if isinstance(feedback, dict):
        status = str(feedback.get("status") or "")
        reason_codes = {str(item) for item in feedback.get("reason_codes") or []}
        if status == "shadow-unavailable" or "shadow-unavailable" in reason_codes:
            return "shadow-unavailable"
        if status == "shadow-unsupported-shape" or "shadow-unsupported-shape" in reason_codes:
            return "shadow-unsupported-shape"
        if status == "shadow-http-400" or "shadow-http-400" in reason_codes:
            return "shadow-http-400"
    primary_status = row["primary_status_code"]
    shadow_status = row["shadow_status_code"]
    if primary_status is None:
        return "primary-status-missing"
    if int(primary_status) >= 400:
        return "primary-error"
    if shadow_status is None:
        return "shadow-status-missing"
    if int(shadow_status) >= 400:
        return "shadow-error"
    if row["output_similarity"] is None:
        return "similarity-missing"
    return "compared"


def _feedback_status_and_reason(row: Any) -> tuple[str, str]:
    experiment = _parse_jsonish(row["experiment_json"]) if "experiment_json" in row.keys() else {}
    feedback = experiment.get("optimization_feedback") if isinstance(experiment, dict) else {}
    if isinstance(feedback, dict):
        status = _public_label(feedback.get("status"), fallback="")
        reason_codes = [
            _public_label(reason, fallback="")
            for reason in feedback.get("reason_codes") or []
            if reason not in (None, "")
        ]
        if status:
            reason = reason_codes[0] if reason_codes else _comparison_blocker(row)
            return status, reason
    blocker = _comparison_blocker(row)
    if blocker == "compared":
        return "compared", "passed" if row["passed_threshold"] else "below-similarity-threshold"
    return blocker, blocker


_NOT_SAMPLED_REASONS = {
    "disabled",
    "kill-switch",
    "daily-budget-zero",
    "daily-budget-exhausted",
    "sample-rate-zero",
    "sample-rate-not-selected",
    "streaming-shadow-not-sampled",
}
_OUT_OF_SCOPE_REASONS = {
    "provider-not-enabled",
    "source-surface-not-enabled",
    "source-surface-not-canonical",
    "codex-app-event-not-turn-start",
}


def _coverage_class_for_decision(experiment: dict[str, Any] | None) -> str:
    if not isinstance(experiment, dict):
        return "metadata-missing"
    explicit = str(experiment.get("coverage_class") or "").strip().replace("_", "-")
    if explicit in {"sampled", "compared", "blocked", "not-sampled", "out-of-scope", "metadata-missing"}:
        return explicit
    status = str(experiment.get("status") or "").strip()
    reason = str(experiment.get("reason") or "").strip()
    if status == "compared":
        return "compared"
    if status == "selected" or bool(experiment.get("sampled")):
        return "sampled"
    if status == "out-of-scope" or reason in _OUT_OF_SCOPE_REASONS:
        return "out-of-scope"
    if reason in _NOT_SAMPLED_REASONS:
        return "not-sampled"
    if status.startswith("shadow-") or reason.startswith("shadow-") or "shadow-" in reason:
        return "blocked"
    if status == "skipped":
        return "blocked"
    return "metadata-missing" if not status and not reason else "blocked"


def _post_fix_shadow_yield_candidate_key(
    *,
    provider: Any,
    source_surface: Any,
    category: Any,
    requested_model: Any,
    shadow_model: Any,
) -> tuple[str, str, str, str, str]:
    return (
        _public_label(provider),
        _public_label(source_surface),
        _public_label(category),
        _public_label(requested_model, fallback=""),
        _public_label(shadow_model, fallback="none"),
    )


def _new_post_fix_shadow_yield_row(key: tuple[str, str, str, str, str]) -> dict[str, Any]:
    return {
        "provider": key[0],
        "source_surface": key[1],
        "category": key[2],
        "requested_model": key[3],
        "shadow_model": key[4],
        "sample_count": 0,
        "compared_count": 0,
        "uncompared_count": 0,
        "clean_yield": 0.0,
        "selected_decision_count": 0,
        "skipped_decision_count": 0,
        "eligible_unsampled_count": 0,
        "passed_threshold_count": 0,
        "shadow_cost_usd": 0.0,
        "reason_counts": {},
        "status_counts": {},
        "decision_reason_counts": {},
        "managed_feedback_status_counts": {},
        "last_sample_at": None,
        "last_decision_at": None,
    }


def _finalize_post_fix_shadow_yield_row(row: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(row.get("sample_count") or 0)
    compared_count = int(row.get("compared_count") or 0)
    row["clean_yield"] = round(compared_count / sample_count, 4) if sample_count else 0.0
    row["pass_rate"] = (
        round(int(row.get("passed_threshold_count") or 0) / compared_count, 4)
        if compared_count else None
    )
    row["shadow_cost_usd"] = round(float(row.get("shadow_cost_usd") or 0.0), 6)
    row["reason_counts"] = _count_rows(row.get("reason_counts") or {})
    row["status_counts"] = _count_rows(row.get("status_counts") or {}, key_name="status")
    row["decision_reason_counts"] = _count_rows(row.get("decision_reason_counts") or {})
    row["managed_feedback_status_counts"] = _count_rows(
        row.get("managed_feedback_status_counts") or {},
        key_name="status",
    )
    return row


def build_post_fix_shadow_yield_report(
    store_obj: Any,
    *,
    since: str | None = None,
    window_hours: float | None = 24.0,
    limit: int = 50,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 50), 1000))
    cutoff = _since_cutoff_iso(since=since, window_hours=window_hours)
    conn = store_obj.conn
    where_samples = ""
    sample_params: list[Any] = []
    if cutoff:
        where_samples = "where created_at >= ?"
        sample_params.append(cutoff)
    try:
        sample_rows = conn.execute(
            f"""
            select created_at,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   requested_model,
                   routed_model,
                   primary_model,
                   shadow_model,
                   coalesce(category, 'unknown') as category,
                   primary_status_code,
                   shadow_status_code,
                   output_similarity,
                   passed_threshold,
                   shadow_cost_est_usd,
                   error,
                   routing_json,
                   experiment_json
            from routing_experiments
            {where_samples}
            order by created_at desc
            limit 50000
            """,
            tuple(sample_params),
        ).fetchall()
    except Exception:
        sample_rows = []

    rows_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    managed_feedback_status_counts: dict[str, int] = {}
    compared_count = 0
    total_shadow_cost = 0.0
    last_sample_at = None
    for raw_row in sample_rows:
        row = dict(raw_row)
        status, reason = _feedback_status_and_reason(raw_row)
        key = _post_fix_shadow_yield_candidate_key(
            provider=row.get("provider"),
            source_surface=row.get("source_surface"),
            category=row.get("category"),
            requested_model=row.get("requested_model"),
            shadow_model=row.get("shadow_model") or row.get("routed_model"),
        )
        item = rows_by_key.setdefault(key, _new_post_fix_shadow_yield_row(key))
        item["sample_count"] += 1
        item["last_sample_at"] = max(
            [value for value in (item.get("last_sample_at"), row.get("created_at")) if value],
            default=None,
        )
        if status == "compared":
            item["compared_count"] += 1
            compared_count += 1
        else:
            item["uncompared_count"] += 1
        if row.get("passed_threshold"):
            item["passed_threshold_count"] += 1
        shadow_cost = float(row.get("shadow_cost_est_usd") or 0.0)
        item["shadow_cost_usd"] += shadow_cost
        total_shadow_cost += shadow_cost
        _increment_count(item["status_counts"], status, fallback="unknown")
        _increment_count(item["reason_counts"], reason, fallback="unknown")
        _increment_count(status_counts, status, fallback="unknown")
        _increment_count(reason_counts, reason, fallback="unknown")
        experiment = _parse_jsonish(row.get("experiment_json"))
        managed = experiment.get("managed_feedback") if isinstance(experiment, dict) else {}
        managed_status = _public_label((managed or {}).get("status"), fallback="not-exported") if isinstance(managed, dict) else "not-exported"
        _increment_count(item["managed_feedback_status_counts"], managed_status, fallback="not-exported")
        _increment_count(managed_feedback_status_counts, managed_status, fallback="not-exported")
        if last_sample_at is None or str(row.get("created_at")) > str(last_sample_at):
            last_sample_at = row.get("created_at")

    where_decisions = "routing_json is not null and routing_json like '%\"routing_experiment\"%'"
    decision_params: list[Any] = []
    if cutoff:
        where_decisions += " and created_at >= ?"
        decision_params.append(cutoff)
    decision_rows: list[Any] = []
    try:
        decision_rows.extend(
            conn.execute(
                f"""
                select created_at,
                       coalesce(provider, 'anthropic') as provider,
                       coalesce(source_surface, 'anthropic_messages') as source_surface,
                       coalesce(category, 'unknown') as category,
                       requested_model,
                       routed_model,
                       routing_json
                from calls
                where {where_decisions}
                order by created_at desc
                limit 50000
                """,
                tuple(decision_params),
            ).fetchall()
        )
    except Exception:
        pass
    codex_params: list[Any] = []
    codex_where = "direction = 'client_to_server' and method = 'turn/start' and routing_json is not null and routing_json like '%\"routing_experiment\"%'"
    if cutoff:
        codex_where += " and created_at >= ?"
        codex_params.append(cutoff)
    try:
        decision_rows.extend(
            conn.execute(
                f"""
                select created_at,
                       'openai' as provider,
                       'codex_turn' as source_surface,
                       null as category,
                       null as requested_model,
                       null as routed_model,
                       routing_json
                from codex_app_events
                where {codex_where}
                order by created_at desc
                limit 50000
                """,
                tuple(codex_params),
            ).fetchall()
        )
    except Exception:
        pass

    decision_status_counts: dict[str, int] = {}
    decision_reason_counts: dict[str, int] = {}
    selected_decisions = 0
    skipped_decisions = 0
    eligible_unsampled_decisions = 0
    last_decision_at = None
    for raw_row in decision_rows:
        row = dict(raw_row)
        routing = _parse_jsonish(row.get("routing_json"))
        experiment = routing.get("routing_experiment") if isinstance(routing, dict) else None
        if not isinstance(experiment, dict):
            continue
        status = _public_label(experiment.get("status"), fallback="unknown")
        reason = _public_label(experiment.get("reason"), fallback="unknown")
        requested = experiment.get("requested_model") or row.get("requested_model")
        shadow = experiment.get("shadow_model") or experiment.get("routed_model") or row.get("routed_model")
        category = experiment.get("category") or row.get("category") or "unknown"
        key = _post_fix_shadow_yield_candidate_key(
            provider=experiment.get("provider") or row.get("provider"),
            source_surface=experiment.get("source_surface") or row.get("source_surface"),
            category=category,
            requested_model=requested,
            shadow_model=shadow,
        )
        item = rows_by_key.setdefault(key, _new_post_fix_shadow_yield_row(key))
        if status == "selected":
            selected_decisions += 1
            item["selected_decision_count"] += 1
        else:
            skipped_decisions += 1
            item["skipped_decision_count"] += 1
            if reason == "sample-rate-not-selected" or reason == "streaming-shadow-not-sampled":
                eligible_unsampled_decisions += 1
                item["eligible_unsampled_count"] += 1
        item["last_decision_at"] = max(
            [value for value in (item.get("last_decision_at"), row.get("created_at")) if value],
            default=None,
        )
        _increment_count(item["decision_reason_counts"], reason, fallback="unknown")
        _increment_count(decision_status_counts, status, fallback="unknown")
        _increment_count(decision_reason_counts, reason, fallback="unknown")
        if last_decision_at is None or str(row.get("created_at")) > str(last_decision_at):
            last_decision_at = row.get("created_at")

    rows = [_finalize_post_fix_shadow_yield_row(row) for row in rows_by_key.values()]
    rows.sort(
        key=lambda item: (
            -int(item.get("sample_count") or 0),
            -int(item.get("selected_decision_count") or 0),
            str(item.get("provider") or ""),
            str(item.get("source_surface") or ""),
            str(item.get("category") or ""),
            str(item.get("requested_model") or ""),
            str(item.get("shadow_model") or ""),
        )
    )
    return {
        "schema": "agentflow.post_fix_shadow_yield.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "since": cutoff,
        "window_hours": None if cutoff is None or since else float(window_hours or 0),
        "limit": capped_limit,
        "summary": {
            "sample_count": len(sample_rows),
            "compared_count": compared_count,
            "uncompared_count": max(0, len(sample_rows) - compared_count),
            "clean_yield": round(compared_count / len(sample_rows), 4) if sample_rows else 0.0,
            "decision_count": selected_decisions + skipped_decisions,
            "selected_decision_count": selected_decisions,
            "skipped_decision_count": skipped_decisions,
            "eligible_unsampled_count": eligible_unsampled_decisions,
            "shadow_cost_usd": round(total_shadow_cost, 6),
            "last_sample_at": last_sample_at,
            "last_decision_at": last_decision_at,
        },
        "status_counts": _count_rows(status_counts, key_name="status"),
        "reason_counts": _count_rows(reason_counts),
        "decision_status_counts": _count_rows(decision_status_counts, key_name="status"),
        "decision_reason_counts": _count_rows(decision_reason_counts),
        "managed_feedback_status_counts": _count_rows(managed_feedback_status_counts, key_name="status"),
        "candidates": rows[:capped_limit],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "api_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local routing experiment metadata after the configured cutoff",
        },
    }


def _build_claude_shadow_yield_report(
    conn: Any,
    *,
    candidates: list[dict[str, Any]],
    observed_limit: int = 5000,
) -> dict[str, Any]:
    provider = "anthropic"
    source_surface = "anthropic_messages"
    try:
        call_rows = conn.execute(
            """
            select created_at,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   coalesce(category, 'unknown') as category,
                   coalesce(stream, 0) as stream,
                   requested_model,
                   routed_model,
                   routing_json,
                   cost_est_usd,
                   cost_baseline_usd
            from calls
            where coalesce(provider, 'anthropic') = ?
              and coalesce(source_surface, 'anthropic_messages') = ?
            order by created_at desc
            limit ?
            """,
            (provider, source_surface, max(1, int(observed_limit))),
        ).fetchall()
    except Exception:
        call_rows = []

    observed_groups: dict[tuple[str, bool, str, str], dict[str, Any]] = {}
    decision_reason_counts: dict[str, int] = {}
    decision_status_counts: dict[str, int] = {}
    sampled_reason_counts: dict[str, int] = {}
    skipped_reason_counts: dict[str, int] = {}
    cap_block_reason_counts: dict[str, int] = {}
    effective_cap_rows: dict[tuple[str, bool], dict[str, Any]] = {}
    eligible_count = 0
    ineligible_count = 0
    selected_count = 0
    skipped_count = 0
    cap_unlimited_candidate_count = 0
    expected_samples_at_current_rate = 0.0
    expected_samples_at_full_rate = 0.0
    expected_samples_if_category_caps_removed = 0.0
    observed_cost_sum = 0.0
    observed_baseline_sum = 0.0

    for row in call_rows:
        experiment = _experiment_from_routing_json(row["routing_json"])
        routing = _parse_jsonish(row["routing_json"])
        requested = _public_label(
            experiment.get("requested_model") or row["requested_model"],
            fallback="",
        )
        candidate = _public_label(
            experiment.get("shadow_model")
            or experiment.get("routed_model")
            or _route_down_candidate_for_requested(str(row["requested_model"] or "")),
            fallback="none",
        )
        category = _public_label(experiment.get("category") or row["category"])
        stream = bool(row["stream"])
        workflow_phase = str(experiment.get("workflow_phase") or routing.get("workflow_phase") or "")
        text_chars = _text_chars_from_routing(routing)
        controls = _effective_experiment_controls(
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
        )
        decision_status = _public_label(experiment.get("status"), fallback="not-evaluated")
        decision_reason = _public_label(experiment.get("reason"), fallback="not-evaluated")
        cap_blocked = (
            decision_reason.endswith("max-text-chars-exceeded")
            or decision_reason.endswith("min-text-chars-not-met")
            or decision_reason in {"request-too-large", "request-too-small"}
        )
        effectively_cap_eligible = text_chars >= int(controls["min_text_chars"]) and (
            int(controls["max_text_chars"]) <= 0 or text_chars <= int(controls["max_text_chars"])
        )
        categories = set(str(c) for c in ROUTING_EXPERIMENT_POLICY.get("categories") or [])
        workflow_phases = set(str(c) for c in ROUTING_EXPERIMENT_POLICY.get("workflow_phases") or [])
        current_policy_eligible = (
            ROUTING_EXPERIMENT_ENABLED
            and not bool(ROUTING_EXPERIMENT_POLICY.get("kill_switch"))
            and _value_allowed(provider, ROUTING_EXPERIMENT_POLICY.get("providers"))
            and _value_allowed(source_surface, ROUTING_EXPERIMENT_POLICY.get("source_surfaces"))
            and (not stream or _value_allowed(source_surface, ROUTING_EXPERIMENT_POLICY.get("streaming_shadow_source_surfaces")))
            and bool(requested)
            and candidate != "none"
            and (ROUTING_EXPERIMENT_MODE != "shadow_candidate_pass_through" or str(row["requested_model"] or "") == str(row["routed_model"] or ""))
            and (not categories or category in categories)
            and (not workflow_phases or workflow_phase in workflow_phases)
            and effectively_cap_eligible
        )
        category_cap_removed_eligible = text_chars >= int(controls["min_text_chars"]) and (
            int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0) <= 0
            or text_chars <= max(
                int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0),
                int(controls["max_text_chars"]),
                text_chars if cap_blocked and category in {"tool-result", "tool-heavy"} else 0,
            )
        ) and (
            ROUTING_EXPERIMENT_ENABLED
            and not bool(ROUTING_EXPERIMENT_POLICY.get("kill_switch"))
            and _value_allowed(provider, ROUTING_EXPERIMENT_POLICY.get("providers"))
            and _value_allowed(source_surface, ROUTING_EXPERIMENT_POLICY.get("source_surfaces"))
            and (not stream or _value_allowed(source_surface, ROUTING_EXPERIMENT_POLICY.get("streaming_shadow_source_surfaces")))
            and bool(requested)
            and candidate != "none"
            and (ROUTING_EXPERIMENT_MODE != "shadow_candidate_pass_through" or str(row["requested_model"] or "") == str(row["routed_model"] or ""))
            and (not categories or category in categories)
            and (not workflow_phases or workflow_phase in workflow_phases)
        )
        if decision_status == "selected":
            selected_count += 1
            eligible_count += 1
            _increment_count(sampled_reason_counts, decision_reason)
        else:
            skipped_count += 1
            _increment_count(skipped_reason_counts, decision_reason)
            if cap_blocked:
                _increment_count(cap_block_reason_counts, decision_reason)
            if current_policy_eligible:
                eligible_count += 1
            else:
                ineligible_count += 1
        _increment_count(decision_reason_counts, decision_reason)
        _increment_count(decision_status_counts, decision_status)

        sample_rate = float(controls["sample_rate"])
        if current_policy_eligible:
            expected_samples_at_current_rate += sample_rate
            expected_samples_at_full_rate += 1.0
        if category_cap_removed_eligible:
            expected_samples_if_category_caps_removed += sample_rate
            if not effectively_cap_eligible:
                cap_unlimited_candidate_count += 1

        observed_cost_sum += float(row["cost_est_usd"] or 0.0)
        observed_baseline_sum += float(row["cost_baseline_usd"] or 0.0)
        group_key = (category, stream, requested, candidate)
        group = observed_groups.setdefault(
            group_key,
            {
                "category": category,
                "stream": stream,
                "requested_model": requested,
                "candidate_target_model": candidate,
                "observed_call_count": 0,
                "selected_count": 0,
                "skipped_count": 0,
                "eligible_count": 0,
                "ineligible_count": 0,
                "effective_min_text_chars": int(controls["min_text_chars"]),
                "effective_min_text_chars_scope": controls["min_text_chars_scope"],
                "effective_max_text_chars": int(controls["max_text_chars"]),
                "effective_max_text_chars_scope": controls["max_text_chars_scope"],
                "effective_sample_rate": round(sample_rate, 6),
                "effective_sample_rate_scope": controls["sample_rate_scope"],
                "effective_daily_budget_usd": round(float(controls["daily_budget_usd"]), 6),
                "effective_daily_budget_scope": controls["daily_budget_scope"],
                "decision_reasons": {},
            },
        )
        group["observed_call_count"] += 1
        group["selected_count"] += 1 if decision_status == "selected" else 0
        group["skipped_count"] += 0 if decision_status == "selected" else 1
        if decision_status == "selected" or current_policy_eligible:
            group["eligible_count"] += 1
        else:
            group["ineligible_count"] += 1
        group["decision_reasons"][decision_reason] = group["decision_reasons"].get(decision_reason, 0) + 1

        cap_key = (category, stream)
        cap_row = effective_cap_rows.setdefault(
            cap_key,
            {
                "category": category,
                "stream": stream,
                "min_text_chars": int(controls["min_text_chars"]),
                "min_text_chars_scope": controls["min_text_chars_scope"],
                "max_text_chars": int(controls["max_text_chars"]),
                "max_text_chars_scope": controls["max_text_chars_scope"],
                "sample_rate": round(sample_rate, 6),
                "sample_rate_scope": controls["sample_rate_scope"],
                "daily_budget_usd": round(float(controls["daily_budget_usd"]), 6),
                "daily_budget_scope": controls["daily_budget_scope"],
                "observed_call_count": 0,
            },
        )
        cap_row["observed_call_count"] += 1

    observed = list(observed_groups.values())
    for group in observed:
        group["decision_reasons"] = _count_rows(group["decision_reasons"])
        group["projected_samples_current_rate"] = round(
            float(group["eligible_count"]) * float(group["effective_sample_rate"]),
            3,
        )
        group["projected_samples_sample_rate_100pct"] = int(group["eligible_count"])
    observed.sort(
        key=lambda item: (
            not bool(item["stream"]),
            -int(item["observed_call_count"]),
            item["category"],
            item["requested_model"],
            item["candidate_target_model"],
        )
    )

    try:
        sample_rows = conn.execute(
            """
            select created_at,
                   requested_model,
                   routed_model,
                   primary_model,
                   shadow_model,
                   coalesce(category, 'unknown') as category,
                   coalesce(routing_reason, 'unknown') as routing_reason,
                   primary_status_code,
                   shadow_status_code,
                   output_similarity,
                   experiment_json,
                   shadow_cost_est_usd
            from routing_experiments
            where coalesce(provider, 'anthropic') = ?
              and coalesce(source_surface, 'anthropic_messages') = ?
            order by created_at desc
            limit 50000
            """,
            (provider, source_surface),
        ).fetchall()
    except Exception:
        sample_rows = []

    compared_count = 0
    uncompared_count = 0
    comparison_blocker_counts: dict[str, int] = {}
    shadow_costs: list[float] = []
    for row in sample_rows:
        blocker = _comparison_blocker(row)
        if blocker == "compared":
            compared_count += 1
        else:
            uncompared_count += 1
            _increment_count(comparison_blocker_counts, blocker)
        if row["shadow_cost_est_usd"] is not None and float(row["shadow_cost_est_usd"] or 0.0) > 0:
            shadow_costs.append(float(row["shadow_cost_est_usd"]))

    avg_shadow_cost = _mean(shadow_costs)
    try:
        today_spend_row = conn.execute(
            """
            select coalesce(sum(coalesce(shadow_cost_est_usd, 0)), 0) as shadow_spend_usd
            from routing_experiments
            where date(created_at) = date('now')
              and coalesce(provider, 'anthropic') = ?
              and coalesce(source_surface, 'anthropic_messages') = ?
            """,
            (provider, source_surface),
        ).fetchone()
        today_shadow_spend = float(today_spend_row["shadow_spend_usd"] or 0.0) if today_spend_row else 0.0
    except Exception:
        today_shadow_spend = 0.0
    policy_budget_remaining = max(0.0, ROUTING_EXPERIMENT_DAILY_BUDGET_USD - today_shadow_spend)
    budget_limited_projected = expected_samples_at_current_rate
    if avg_shadow_cost and avg_shadow_cost > 0:
        budget_limited_projected = min(expected_samples_at_current_rate, policy_budget_remaining / avg_shadow_cost)

    promotion_verdict_counts: dict[str, int] = {}
    promotion_freshness_counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.get("provider") != "anthropic" or candidate.get("source_surface") != "anthropic_messages":
            continue
        verdict = candidate.get("promotion_verdict") or "unknown"
        _increment_count(promotion_verdict_counts, verdict)
        age = candidate.get("last_sample_age_hours")
        if age is None:
            freshness = "unknown"
        elif float(age) <= ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS:
            freshness = "fresh"
        else:
            freshness = "stale"
        _increment_count(promotion_freshness_counts, freshness, fallback="unknown")

    return {
        "schema": "agentflow.claude_shadow_routing_yield.v1",
        "provider": provider,
        "source_surface": source_surface,
        "observed_window_limit": max(1, int(observed_limit)),
        "summary": {
            "observed_call_count": len(call_rows),
            "eligible_count": eligible_count,
            "ineligible_count": ineligible_count,
            "selected_count": selected_count,
            "skipped_count": skipped_count,
            "sampled_count": len(sample_rows),
            "compared_count": compared_count,
            "uncompared_count": uncompared_count,
            "observed_cost_usd": round(observed_cost_sum, 6),
            "observed_baseline_usd": round(observed_baseline_sum, 6),
        },
        "decision_status_counts": _count_rows(decision_status_counts, key_name="status"),
        "decision_reason_counts": _count_rows(decision_reason_counts),
        "sampled_reason_counts": _count_rows(sampled_reason_counts),
        "skipped_reason_counts": _count_rows(skipped_reason_counts),
        "cap_block_reason_counts": _count_rows(cap_block_reason_counts),
        "comparison_blocker_counts": _count_rows(comparison_blocker_counts, key_name="blocker"),
        "effective_caps": sorted(
            effective_cap_rows.values(),
            key=lambda item: (not bool(item["stream"]), item["category"]),
        ),
        "observed": observed,
        "projection": {
            "projected_samples_current_sample_rate": round(expected_samples_at_current_rate, 3),
            "projected_samples_sample_rate_100pct": round(expected_samples_at_full_rate, 3),
            "projected_samples_budget_unlimited": round(expected_samples_at_current_rate, 3),
            "projected_samples_budget_limited": round(budget_limited_projected, 3),
            "avg_shadow_sample_cost_usd": round(float(avg_shadow_cost), 6) if avg_shadow_cost is not None else None,
            "projected_samples_if_category_caps_removed": round(expected_samples_if_category_caps_removed, 3),
            "additional_cap_unlimited_candidate_count": cap_unlimited_candidate_count,
        },
        "promotion_verdict_counts": _count_rows(promotion_verdict_counts, key_name="verdict"),
        "promotion_freshness_counts": _count_rows(promotion_freshness_counts, key_name="freshness"),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "api_keys_included": False,
            "rule_paths_included": False,
        },
    }


def build_routing_experiment_report(
    store_obj: Any,
    *,
    limit: int = 20,
    since: str | None = None,
    window_hours: float | None = 24.0,
) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 1000))
    conn = store_obj.conn
    rows = conn.execute(
        """
        select created_at,
               coalesce(provider, 'anthropic') as provider,
               coalesce(source_surface, 'anthropic_messages') as source_surface,
               coalesce(stream, 0) as stream,
               requested_model,
               routed_model,
               primary_model,
               shadow_model,
               coalesce(category, 'unknown') as category,
               coalesce(routing_reason, 'unknown') as routing_reason,
               primary_status_code,
               shadow_status_code,
               primary_latency_ms,
               shadow_latency_ms,
               output_similarity,
               passed_threshold,
               primary_cost_est_usd,
               shadow_cost_est_usd,
               error,
               routing_json,
               experiment_json
        from routing_experiments
        order by created_at desc
        limit 50000
        """
    ).fetchall()
    grouped: dict[tuple[str, str, bool, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        experiment = _parse_jsonish(row["experiment_json"])
        routing = _parse_jsonish(row["routing_json"])
        mode = _sample_mode_from_experiment(experiment)
        workflow_phase = _workflow_phase_from_payloads(experiment, routing)
        key = (
            _public_label(row["provider"]),
            _public_label(row["source_surface"]),
            bool(row["stream"]),
            _public_label(row["requested_model"], fallback=""),
            _public_label(row["routed_model"], fallback=""),
            _public_label(row["category"]),
            workflow_phase,
            mode,
        )
        item = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "source_surface": key[1],
                "stream": key[2],
                "requested_model": key[3],
                "routed_model": key[4],
                "category": key[5],
                "workflow_phase": key[6],
                "mode": key[7],
                "routing_reasons": {},
                "mode_composition": {},
                "samples": 0,
                "compared_samples": 0,
                "primary_error_samples": 0,
                "shadow_error_samples": 0,
                "shadow_unavailable_samples": 0,
                "shadow_http_400_samples": 0,
                "shadow_unsupported_shape_samples": 0,
                "fallback_or_retry_count": 0,
                "primary_cost_usd": 0.0,
                "shadow_cost_usd": 0.0,
                "similarities": [],
                "passed": [],
                "latency_deltas": [],
                "last_sample_at": None,
            },
        )
        item["samples"] += 1
        item["mode_composition"][mode] = item["mode_composition"].get(mode, 0) + 1
        reason = _public_label(row["routing_reason"])
        item["routing_reasons"][reason] = item["routing_reasons"].get(reason, 0) + 1
        primary_status = row["primary_status_code"]
        shadow_status = row["shadow_status_code"]
        feedback = experiment.get("optimization_feedback") if isinstance(experiment, dict) else {}
        feedback_status = str(feedback.get("status") or "") if isinstance(feedback, dict) else ""
        feedback_reason_codes = {str(item) for item in (feedback.get("reason_codes") if isinstance(feedback, dict) else []) or []}
        primary_ok = primary_status is not None and int(primary_status) < 400
        shadow_ok = shadow_status is not None and int(shadow_status) < 400
        if not primary_ok:
            item["primary_error_samples"] += 1
        shadow_unsupported_shape = feedback_status == "shadow-unsupported-shape" or "shadow-unsupported-shape" in feedback_reason_codes
        shadow_unavailable = feedback_status == "shadow-unavailable" or "shadow-unavailable" in feedback_reason_codes
        shadow_http_400 = feedback_status == "shadow-http-400" or "shadow-http-400" in feedback_reason_codes or shadow_status == 400
        if shadow_http_400:
            item["shadow_http_400_samples"] += 1
        if shadow_unsupported_shape:
            item["shadow_unsupported_shape_samples"] += 1
        if shadow_unavailable:
            item["shadow_unavailable_samples"] += 1
        if (not shadow_ok or row["error"]) and not shadow_unsupported_shape and not shadow_unavailable:
            item["shadow_error_samples"] += 1
        if primary_ok and shadow_ok and row["output_similarity"] is not None:
            item["compared_samples"] += 1
            item["similarities"].append(float(row["output_similarity"]))
            item["passed"].append(1.0 if row["passed_threshold"] else 0.0)
        if row["primary_latency_ms"] is not None and row["shadow_latency_ms"] is not None:
            item["latency_deltas"].append(float(row["primary_latency_ms"]) - float(row["shadow_latency_ms"]))
        item["primary_cost_usd"] += float(row["primary_cost_est_usd"] or 0.0)
        item["shadow_cost_usd"] += float(row["shadow_cost_est_usd"] or 0.0)
        if routing.get("fallback_reason") or routing.get("retry_count") or experiment.get("fallback_reason") or experiment.get("retry_count"):
            item["fallback_or_retry_count"] += 1
        if item["last_sample_at"] is None or str(row["created_at"]) > str(item["last_sample_at"]):
            item["last_sample_at"] = row["created_at"]

    today_spend = _today_shadow_spend_usd(store_obj)
    budget_limit = ROUTING_EXPERIMENT_DAILY_BUDGET_USD
    candidates: list[dict[str, Any]] = []
    for item in grouped.values():
        compared_samples = int(item.get("compared_samples") or 0)
        avg_similarity = _mean(item.pop("similarities"))
        pass_rate = _mean(item.pop("passed"))
        avg_latency_delta = _mean(item.pop("latency_deltas"))
        confidence_score = 0.0
        if avg_similarity is not None and compared_samples > 0:
            confidence_score = float(avg_similarity) * min(1.0, compared_samples / ROUTING_EXPERIMENT_MIN_SAMPLES)
        item["compared_samples"] = compared_samples
        item["avg_similarity"] = round(float(avg_similarity), 6) if avg_similarity is not None else None
        item["pass_rate"] = round(float(pass_rate), 4) if pass_rate is not None else None
        item["compared_coverage"] = round(compared_samples / int(item["samples"]), 4) if item["samples"] else 0.0
        item["primary_error_rate"] = round(int(item["primary_error_samples"]) / int(item["samples"]), 4) if item["samples"] else 0.0
        item["shadow_error_rate"] = round(int(item["shadow_error_samples"]) / int(item["samples"]), 4) if item["samples"] else 0.0
        item["primary_cost_usd"] = round(float(item["primary_cost_usd"]), 6)
        item["shadow_cost_usd"] = round(float(item["shadow_cost_usd"]), 6)
        item["cost_delta_usd"] = round(float(item["primary_cost_usd"]) - float(item["shadow_cost_usd"]), 6)
        item["avg_latency_delta_ms"] = round(float(avg_latency_delta), 2) if avg_latency_delta is not None else None
        last_age = _age_hours(item.get("last_sample_at"))
        item["last_sample_age_hours"] = round(last_age, 2) if last_age is not None else None
        item["confidence_score"] = round(confidence_score, 6)
        item["min_samples_for_confidence"] = ROUTING_EXPERIMENT_MIN_SAMPLES
        item["promotion"] = _score_routing_promotion_candidate(
            item,
            policy=ROUTING_EXPERIMENT_POLICY,
            today_spend_usd=today_spend,
            budget_limit_usd=budget_limit,
        )
        item["promotion_verdict"] = item["promotion"]["verdict"]
        item["promotion_reason_codes"] = item["promotion"]["reason_codes"]
        routing_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(item["routing_reasons"].items(), key=lambda entry: (-entry[1], entry[0]))
        ]
        item["routing_reasons"] = routing_reasons
        item["routing_reason"] = routing_reasons[0]["reason"] if routing_reasons else "unknown"
        candidates.append(item)
    promotion_verdict_counts: dict[str, int] = {}
    promotion_reason_counts: dict[str, int] = {}
    for item in candidates:
        verdict = str(item.get("promotion_verdict") or "unknown")
        promotion_verdict_counts[verdict] = promotion_verdict_counts.get(verdict, 0) + 1
        for reason in item.get("promotion_reason_codes") or []:
            reason = str(reason)
            promotion_reason_counts[reason] = promotion_reason_counts.get(reason, 0) + 1
    candidates.sort(key=lambda item: (int(item["samples"]), str(item.get("last_sample_at") or "")), reverse=True)
    candidates = candidates[:capped]

    summary_row = conn.execute(
        """
        select count(*) as samples,
               sum(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then 1 else 0 end) as compared_samples,
               avg(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then output_similarity else null end) as avg_similarity,
               avg(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then passed_threshold else null end) as pass_rate,
               round(sum(coalesce(primary_cost_est_usd, 0)), 6) as primary_cost_usd,
               round(sum(coalesce(shadow_cost_est_usd, 0)), 6) as shadow_cost_usd,
               round(sum(coalesce(primary_cost_est_usd, 0) - coalesce(shadow_cost_est_usd, 0)), 6) as cost_delta_usd,
               round(avg(case when primary_latency_ms is not null and shadow_latency_ms is not null
                              then primary_latency_ms - shadow_latency_ms else null end), 2) as avg_latency_delta_ms
        from routing_experiments
        """
    ).fetchone()
    feedback_status_counts: dict[str, int] = {}
    sample_mode_counts: dict[str, int] = {}
    for row in conn.execute("select experiment_json from routing_experiments where experiment_json is not null").fetchall():
        experiment = _parse_jsonish(row["experiment_json"])
        if not experiment:
            status = "invalid-json"
            mode = "invalid-json"
        else:
            feedback = experiment.get("managed_feedback") if isinstance(experiment, dict) else None
            status = _public_label((feedback or {}).get("status"), fallback="not-exported") if isinstance(feedback, dict) else "not-exported"
            mode = _sample_mode_from_experiment(experiment) if isinstance(experiment, dict) else "unknown"
        feedback_status_counts[status] = feedback_status_counts.get(status, 0) + 1
        sample_mode_counts[mode] = sample_mode_counts.get(mode, 0) + 1

    decision_reason_counts: dict[tuple[str, str, str, str], int] = {}
    decision_status_counts: dict[str, int] = {}
    decision_surface_counts: dict[tuple[str, str, str], int] = {}
    decision_coverage_counts: dict[str, int] = {
        "sampled": 0,
        "compared": 0,
        "blocked": 0,
        "not-sampled": 0,
        "out-of-scope": 0,
        "metadata-missing": 0,
    }

    def record_decision(provider_value: Any, source_surface_value: Any, routing_json: Any) -> None:
        routing = _parse_jsonish(routing_json)
        if not routing:
            provider_label = _public_label(provider_value)
            source_surface_label = _public_label(source_surface_value)
            status = "unknown"
            reason = "invalid-routing-json"
            coverage_class = "metadata-missing"
        else:
            experiment = routing.get("routing_experiment") if isinstance(routing, dict) else None
            if not isinstance(experiment, dict):
                provider_label = _public_label(provider_value)
                source_surface_label = _public_label(source_surface_value)
                reason = "routing-experiment-metadata-missing"
                status = "out-of-scope"
                coverage_class = "metadata-missing"
            else:
                provider_label = _public_label(experiment.get("provider") or provider_value)
                source_surface_label = _public_label(experiment.get("source_surface") or source_surface_value)
                reason = _public_label(experiment.get("reason"))
                status = _public_label(experiment.get("status"))
                coverage_class = _coverage_class_for_decision(experiment)
        key = (provider_label, source_surface_label, status, reason)
        decision_reason_counts[key] = decision_reason_counts.get(key, 0) + 1
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1
        surface_key = (provider_label, source_surface_label, status)
        decision_surface_counts[surface_key] = decision_surface_counts.get(surface_key, 0) + 1
        decision_coverage_counts[coverage_class] = int(decision_coverage_counts.get(coverage_class, 0)) + 1

    try:
        decision_rows = conn.execute(
            """
            select coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   routing_json
            from calls
            where routing_json is not null
            order by created_at desc
            limit 5000
            """
        ).fetchall()
    except Exception:
        decision_rows = []
    for row in decision_rows:
        record_decision(row["provider"], row["source_surface"], row["routing_json"])
    try:
        codex_decision_rows = conn.execute(
            """
            select 'openai' as provider,
                   'codex_turn' as source_surface,
                   routing_json
            from codex_app_events
            where direction = 'client_to_server'
              and method = 'turn/start'
              and routing_json is not null
            order by created_at desc
            limit 5000
            """
        ).fetchall()
    except Exception:
        codex_decision_rows = []
    for row in codex_decision_rows:
        record_decision(row["provider"], row["source_surface"], row["routing_json"])
    decision_reasons = [
        {
            "provider": provider,
            "source_surface": source_surface,
            "status": status,
            "reason": reason,
            "count": count,
        }
        for (provider, source_surface, status, reason), count in sorted(
            decision_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    decision_surfaces = [
        {
            "provider": provider,
            "source_surface": source_surface,
            "status": status,
            "count": count,
        }
        for (provider, source_surface, status), count in sorted(
            decision_surface_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    compared_total = int((summary_row or {}).get("compared_samples") or 0)
    avg_similarity_total = (summary_row or {}).get("avg_similarity")
    pass_rate_total = (summary_row or {}).get("pass_rate")
    decision_denominator_count = int(sum(decision_status_counts.values()))
    metadata_missing_count = int(decision_coverage_counts.get("metadata-missing", 0))
    out_of_scope_count = int(decision_coverage_counts.get("out-of-scope", 0))
    decision_eligible_count = max(0, decision_denominator_count - metadata_missing_count - out_of_scope_count)
    metadata_coverage_rate = (
        round((decision_denominator_count - metadata_missing_count) / decision_denominator_count, 4)
        if decision_denominator_count else 1.0
    )
    eligibility_projection = _build_shadow_eligibility_projection(conn)
    claude_shadow_yield = _build_claude_shadow_yield_report(conn, candidates=candidates)
    post_fix_shadow_yield = build_post_fix_shadow_yield_report(
        store_obj,
        since=since,
        window_hours=window_hours,
        limit=limit,
    )
    return {
        "schema": "agentflow.routing_experiment_report.v1",
        "generated_at": utc_now(),
        "policy": {
            "profile_id": str(ROUTING_EXPERIMENT_POLICY.get("profile_id") or ""),
            "mode": ROUTING_EXPERIMENT_MODE,
            "enabled": ROUTING_EXPERIMENT_ENABLED,
            "kill_switch": bool(ROUTING_EXPERIMENT_POLICY.get("kill_switch")),
            "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
            "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
            "sample_rate": ROUTING_EXPERIMENT_SAMPLE_RATE,
            "daily_budget_usd": round(budget_limit, 6),
            "today_shadow_spend_usd": round(today_spend, 6),
            "today_budget_remaining_usd": round(max(0.0, budget_limit - today_spend), 6),
            "daily_budget_exhausted": bool(ROUTING_EXPERIMENT_ENABLED and (budget_limit <= 0 or today_spend >= budget_limit)),
            "providers": list(ROUTING_EXPERIMENT_POLICY.get("providers") or []),
            "source_surfaces": list(ROUTING_EXPERIMENT_POLICY.get("source_surfaces") or []),
            "streaming_shadow_source_surfaces": list(ROUTING_EXPERIMENT_POLICY.get("streaming_shadow_source_surfaces") or []),
            "model_pairs": list(ROUTING_EXPERIMENT_POLICY.get("model_pairs") or []),
            "categories": list(ROUTING_EXPERIMENT_POLICY.get("categories") or []),
            "workflow_phases": list(ROUTING_EXPERIMENT_POLICY.get("workflow_phases") or []),
            "min_text_chars": int(ROUTING_EXPERIMENT_POLICY.get("min_text_chars") or 0),
            "max_text_chars": int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0),
            "eligibility_overrides": list(ROUTING_EXPERIMENT_POLICY.get("eligibility_overrides") or []),
        },
        "summary": {
            "sample_count": int((summary_row or {}).get("samples") or 0),
            "comparison_count": compared_total,
            "pass_rate": round(float(pass_rate_total), 4) if pass_rate_total is not None else None,
            "avg_similarity": round(float(avg_similarity_total), 6) if avg_similarity_total is not None else None,
            "primary_cost_usd": float((summary_row or {}).get("primary_cost_usd") or 0.0),
            "shadow_cost_usd": float((summary_row or {}).get("shadow_cost_usd") or 0.0),
            "cost_delta_usd": float((summary_row or {}).get("cost_delta_usd") or 0.0),
            "avg_latency_delta_ms": (summary_row or {}).get("avg_latency_delta_ms"),
            "feedback_status_counts": feedback_status_counts,
            "sample_mode_counts": sample_mode_counts,
            "decision_count": int(sum(decision_status_counts.values())),
            "routing_experiment_denominator_count": decision_denominator_count,
            "routing_experiment_metadata_count": max(0, decision_denominator_count - metadata_missing_count),
            "routing_experiment_metadata_coverage_rate": metadata_coverage_rate,
            "eligible_count": decision_eligible_count,
            "sampled_count": int((summary_row or {}).get("samples") or 0),
            "compared_count": compared_total,
            "blocked_count": int(decision_coverage_counts.get("blocked", 0)),
            "not_sampled_count": int(decision_coverage_counts.get("not-sampled", 0)),
            "out_of_scope_count": out_of_scope_count,
            "metadata_missing_count": metadata_missing_count,
            "decision_coverage_counts": dict(decision_coverage_counts),
            "decision_status_counts": decision_status_counts,
            "applied_routed_down_samples": int(sample_mode_counts.get("applied_routed_down", 0)),
            "shadow_candidate_pass_through_samples": int(sample_mode_counts.get("shadow_candidate_pass_through", 0)),
            "promotion_verdict_counts": promotion_verdict_counts,
            "promotion_reason_counts": promotion_reason_counts,
            "promotion_ready_candidates": int(promotion_verdict_counts.get("promote", 0)),
        },
        "decision_reasons": decision_reasons,
        "decision_surfaces": decision_surfaces,
        "eligibility_projection": eligibility_projection,
        "claude_shadow_yield": claude_shadow_yield,
        "post_fix_shadow_yield": post_fix_shadow_yield,
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included_by_default": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
        },
    }
