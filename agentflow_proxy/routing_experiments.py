from __future__ import annotations

import hashlib
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from agentflow_proxy.crunch import build_embedding, sha256_text
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
    candidates.append(Path.home() / ".agentflow" / filename)
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
    if budget_limit <= 0:
        meta["reason"] = "daily-budget-zero"
        return meta
    if budget_spent >= budget_limit:
        meta["reason"] = "daily-budget-exhausted"
        return meta
    sample_rate = float(controls["sample_rate"])
    if sample_rate <= 0:
        meta["reason"] = "sample-rate-zero"
        return meta
    if random_value() >= sample_rate:
        meta["reason"] = "streaming-shadow-not-sampled" if stream else "sample-rate-not-selected"
        return meta

    meta["status"] = "selected"
    meta["sampled"] = True
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
    compared = (
        primary_status_code is not None
        and primary_status_code < 400
        and shadow_status_code is not None
        and shadow_status_code < 400
        and comparison.get("output_similarity") is not None
    )
    status = "compared" if compared else "shadow-unavailable"
    if error:
        status = "shadow-error"
    reason_codes: list[str] = []
    if primary_status_code is not None and primary_status_code >= 400:
        reason_codes.append("primary-error")
    if shadow_status_code is None:
        reason_codes.append("shadow-missing")
    elif shadow_status_code >= 400:
        reason_codes.append("shadow-error")
    if error:
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
        try:
            routing = yaml.safe_load(row["routing_json"]) or {}
        except Exception:
            continue
        if not isinstance(routing, dict):
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


def build_routing_experiment_report(store_obj: Any, *, limit: int = 20) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 1000))
    conn = store_obj.conn
    rows = conn.execute(
        """
        select created_at,
               coalesce(provider, 'anthropic') as provider,
               coalesce(source_surface, 'anthropic_messages') as source_surface,
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
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        experiment = _parse_jsonish(row["experiment_json"])
        routing = _parse_jsonish(row["routing_json"])
        mode = _sample_mode_from_experiment(experiment)
        workflow_phase = _workflow_phase_from_payloads(experiment, routing)
        key = (
            _public_label(row["provider"]),
            _public_label(row["source_surface"]),
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
                "requested_model": key[2],
                "routed_model": key[3],
                "category": key[4],
                "workflow_phase": key[5],
                "mode": key[6],
                "routing_reasons": {},
                "mode_composition": {},
                "samples": 0,
                "compared_samples": 0,
                "primary_error_samples": 0,
                "shadow_error_samples": 0,
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
        primary_ok = primary_status is not None and int(primary_status) < 400
        shadow_ok = shadow_status is not None and int(shadow_status) < 400
        if not primary_ok:
            item["primary_error_samples"] += 1
        if not shadow_ok or row["error"]:
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
        try:
            experiment = yaml.safe_load(row["experiment_json"]) or {}
        except Exception:
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

    def record_decision(provider_value: Any, source_surface_value: Any, routing_json: Any) -> None:
        try:
            routing = yaml.safe_load(routing_json) or {}
        except Exception:
            provider_label = _public_label(provider_value)
            source_surface_label = _public_label(source_surface_value)
            status = "unknown"
            reason = "invalid-routing-json"
        else:
            experiment = routing.get("routing_experiment") if isinstance(routing, dict) else None
            if not isinstance(experiment, dict):
                return
            provider_label = _public_label(experiment.get("provider") or provider_value)
            source_surface_label = _public_label(experiment.get("source_surface") or source_surface_value)
            reason = _public_label(experiment.get("reason"))
            status = _public_label(experiment.get("status"))
        key = (provider_label, source_surface_label, status, reason)
        decision_reason_counts[key] = decision_reason_counts.get(key, 0) + 1
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1
        surface_key = (provider_label, source_surface_label, status)
        decision_surface_counts[surface_key] = decision_surface_counts.get(surface_key, 0) + 1

    try:
        decision_rows = conn.execute(
            """
            select coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   routing_json
            from calls
            where routing_json is not null
              and routing_json like '%"routing_experiment"%'
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
              and routing_json like '%"routing_experiment"%'
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
    eligibility_projection = _build_shadow_eligibility_projection(conn)
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
