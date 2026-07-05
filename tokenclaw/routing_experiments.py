from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from tokenclaw.crunch import build_embedding, sha256_text
from tokenclaw.paths import tokenclaw_config_path
from tokenclaw.policy_files import policy_file_snapshot, utc_now
from tokenclaw.store import cosine_similarity, stable_json

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


def _env_bool(name: str, default: bool = False) -> bool:
    if name not in os.environ:
        return default
    return _as_bool(os.environ.get(name), default)


def _managed_policy_decisions_configured() -> bool:
    recommendations_enabled = (
        _env_bool("TOKENCLAW_RECOMMENDATIONS_ENABLED", False)
        if "TOKENCLAW_RECOMMENDATIONS_ENABLED" in os.environ
        else _env_bool("TOKENCLAW_RECOMMENDATION_ENABLED", False)
    )
    policy_decisions_enabled = (
        _env_bool("TOKENCLAW_POLICY_DECISIONS_ENABLED", False)
        if "TOKENCLAW_POLICY_DECISIONS_ENABLED" in os.environ
        else _env_bool("TOKENCLAW_POLICY_DECISION_ENABLED", False)
    )
    server_url = str(os.environ.get("TOKENCLAW_RECOMMENDATION_SERVER_URL") or "").strip()
    return bool(recommendations_enabled and policy_decisions_enabled and server_url)


def _as_non_negative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _as_non_negative_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _safe_candidate_id(value: Any, *, fallback: str) -> str:
    public = _public_label(value, fallback=fallback)
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", public).strip("-")
    if not cleaned or cleaned == "redacted-metadata-label":
        return fallback
    return cleaned[:96]


def _default_experiment_policy() -> dict[str, Any]:
    return {
        "profile_id": "first-safe-openai-codex-claude-shadow-pass-through-v1",
        "mode": "shadow_candidate_pass_through",
        "enabled": True,
        "kill_switch": False,
        "thin_client_routing": True,
        "sample_rate": 0.10,
        "daily_budget_usd": 10.0,
        "min_text_chars": 0,
        "max_text_chars": 8000,
        # Local canary origination is disabled (see the "Backed or off" gate in
        # routing_experiment_decision): the managed server decides anthropic canaries,
        # and server-directed anthropic shadows are executed via
        # _managed_shadow_experiment_decision, not this policy. Only the server-forced
        # OpenAI/codex shadow path still flows through this local policy, so only its
        # OpenAI/codex candidates remain. Anthropic candidates, model pairs, fallback
        # routes, the anthropic streaming-shadow surface, and the anthropic text-size
        # eligibility caps (incl. the 128k tool-result cap) were removed as dead code.
        "providers": ["openai"],
        "source_surfaces": ["openai_responses", "openai_chat", "codex_turn"],
        "streaming_shadow_source_surfaces": [],
        "blocklist": [],
        "preferred_pathways": [],
        "fallback_routes": [
            {"requested_model": "gpt-5.5", "routed_model": "gpt-5.4"},
            {"requested_model": "gpt-5.4", "routed_model": "gpt-5.3"},
            {"requested_model": "gpt-5.3-codex", "routed_model": "gpt-5-codex"},
            {"requested_model": "gpt-5-codex", "routed_model": "gpt-5-mini"},
            {"requested_model": "gpt-5.3", "routed_model": "gpt-5-mini"},
            {"requested_model": "gpt-5-mini", "routed_model": "gpt-5-nano"},
        ],
        "model_pairs": [
            {"requested_model": "gpt-5.5", "routed_model": "gpt-5.4"},
            {"requested_model": "gpt-5.4", "routed_model": "gpt-5.4-mini"},
            {"requested_model": "gpt-5.3-codex", "routed_model": "gpt-5-codex"},
            {"requested_model": "gpt-5-codex", "routed_model": "gpt-5-mini"},
            {"requested_model": "gpt-5.3", "routed_model": "gpt-5-mini"},
            {"requested_model": "gpt-5-mini", "routed_model": "gpt-5-nano"},
        ],
        "routing_candidates": [
            {
                "candidate_id": "codex-gpt55-to-gpt53-codex-summary",
                "requested_model": "gpt-5.5",
                "routed_model": "gpt-5.3-codex",
                "provider": "openai",
                "source_surface": "codex_turn",
                "app_family": "codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "max_text_chars": 8000,
                "sample_weight": 4.0,
            },
            {
                "candidate_id": "codex-gpt55-to-gpt53-codex-unknown-phase",
                "requested_model": "gpt-5.5",
                "routed_model": "gpt-5.3-codex",
                "provider": "openai",
                "source_surface": "codex_turn",
                "app_family": "codex",
                "category": "codex-turn",
                "workflow_phase": "unknown",
                "max_text_chars": 8000,
            },
            {
                "candidate_id": "codex-gpt53-codex-to-gpt5-codex-summary",
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5-codex",
                "provider": "openai",
                "source_surface": "codex_turn",
                "app_family": "codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "max_text_chars": 8000,
            },
            {
                "candidate_id": "codex-gpt5-codex-to-mini-short-summary",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-mini",
                "provider": "openai",
                "source_surface": "codex_turn",
                "app_family": "codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "max_text_chars": 2000,
                "sample_rate": 0.05,
            },
            {
                "candidate_id": "generic-gpt55-to-gpt54-chat",
                "requested_model": "gpt-5.5",
                "routed_model": "gpt-5.4",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "generic_openai",
                "category": "chat",
                "max_text_chars": 8000,
                "sample_weight": 4.0,
            },
            {
                "candidate_id": "generic-gpt55-to-mini-short-exploratory",
                "requested_model": "gpt-5.5",
                "routed_model": "gpt-5-mini",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "generic_openai",
                "category": "short-completion",
                "max_text_chars": 2000,
                "sample_rate": 0.02,
            },
            {
                "candidate_id": "generic-gpt54-to-gpt53-chat",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.3",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "generic_openai",
                "category": "chat",
                "max_text_chars": 8000,
            },
            {
                "candidate_id": "generic-gpt54-to-gpt54-mini-tool-light",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "generic_openai",
                "category": "tool-light",
                "max_text_chars": 8000,
            },
            {
                "candidate_id": "generic-gpt53-to-mini-short",
                "requested_model": "gpt-5.3",
                "routed_model": "gpt-5-mini",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "generic_openai",
                "category": "short-completion",
                "max_text_chars": 4000,
            },
            {
                "candidate_id": "generic-gpt5-mini-to-nano-summary",
                "requested_model": "gpt-5-mini",
                "routed_model": "gpt-5-nano",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "generic_openai",
                "category": "short-completion",
                "workflow_phase": "summary",
                "max_text_chars": 1000,
                "sample_rate": 0.02,
            },
        ],
        "workflow_phases": [],
        "categories": ["chat", "short-completion", "tool-light", "tool-result", "tool-heavy", "codex-turn"],
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
        if os.getenv(f"{env_name}_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return candidates
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(tokenclaw_config_path(filename))
    return candidates


def _writable_experiment_config_path() -> Path:
    return tokenclaw_config_path("routing_experiments.yaml")


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


def _candidate_id(candidate: dict[str, Any], index: int, *, shape: str) -> str:
    explicit = (
        candidate.get("candidate_id")
        or candidate.get("policy_id")
        or candidate.get("recommendation_id")
        or candidate.get("id")
    )
    requested = str(candidate.get("requested_model") or "requested").strip() or "requested"
    routed = str(candidate.get("routed_model") or "routed").strip() or "routed"
    fallback = f"{shape}:{requested}->{routed}:{index + 1}"
    return _safe_candidate_id(explicit, fallback=fallback)


def _clean_routing_candidate(raw: dict[str, Any], index: int, *, shape: str) -> dict[str, Any] | None:
    requested = str(raw.get("requested_model") or raw.get("requested") or "").strip()
    routed = str(
        raw.get("routed_model")
        or raw.get("routed")
        or raw.get("target_model")
        or raw.get("candidate_target_model")
        or ""
    ).strip()
    if not requested or not routed:
        return None
    candidate: dict[str, Any] = {
        "candidate_id": _candidate_id(raw, index, shape=shape),
        "requested_model": requested,
        "routed_model": routed,
        "policy_shape": shape,
    }
    for key in ("provider", "source_surface", "app_family", "category", "workflow_phase", "candidate_source"):
        if raw.get(key) not in (None, ""):
            candidate[key] = str(raw[key])
    if raw.get("stream") is not None:
        candidate["stream"] = _as_bool(raw.get("stream"), False)
    for key in ("min_text_chars", "max_text_chars", "min_input_tokens", "max_input_tokens"):
        parsed = _as_non_negative_int(raw.get(key))
        if parsed is not None:
            candidate[key] = parsed
    sample_weight = _as_non_negative_float(raw.get("sample_weight"))
    if sample_weight is not None:
        candidate["sample_weight"] = sample_weight
    sample_rate = _as_non_negative_float(raw.get("sample_rate"))
    if sample_rate is not None:
        candidate["sample_rate"] = min(1.0, sample_rate)
    label = _public_label(raw.get("label"), fallback="")
    if label:
        candidate["label"] = label
    return candidate


def _clean_routing_candidates(raw_candidates: Any, *, shape: str) -> list[dict[str, Any]]:
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            continue
        candidate = _clean_routing_candidate(raw, index, shape=shape)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _clean_fallback_routes(raw_routes: Any) -> list[dict[str, Any]]:
    if isinstance(raw_routes, dict):
        raw_list = [
            {"requested_model": requested, "routed_model": routed}
            for requested, routed in raw_routes.items()
        ]
    else:
        raw_list = raw_routes
    return _clean_routing_candidates(raw_list, shape="fallback_routes")


def _clean_blocklist(raw_blocklist: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_blocklist, list):
        return []
    blocked: list[dict[str, Any]] = []
    for raw in raw_blocklist:
        if isinstance(raw, str):
            model = raw.strip()
            if model:
                blocked.append({"model": model})
            continue
        if not isinstance(raw, dict):
            continue
        item: dict[str, Any] = {}
        for source, target in (
            ("model", "model"),
            ("requested_model", "requested_model"),
            ("requested", "requested_model"),
            ("routed_model", "routed_model"),
            ("target_model", "routed_model"),
            ("route_to", "routed_model"),
            ("provider", "provider"),
            ("source_surface", "source_surface"),
            ("app_family", "app_family"),
            ("category", "category"),
            ("workflow_phase", "workflow_phase"),
            ("pathway_id", "pathway_id"),
        ):
            value = raw.get(source)
            if value not in (None, ""):
                item[target] = str(value).strip()
        if item:
            blocked.append(item)
    return blocked


def distill_thin_routing_policy(data: dict[str, Any]) -> dict[str, Any]:
    """Build the thin routing policy form from a legacy experiment policy."""

    base = _default_experiment_policy()
    policy = _apply_policy_yaml(base, data if isinstance(data, dict) else {})
    preferred: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in (policy.get("preferred_pathways") or []):
        if not isinstance(candidate, dict):
            continue
        key = (str(candidate.get("requested_model") or ""), str(candidate.get("routed_model") or ""))
        if key[0] and key[1]:
            preferred[key] = dict(candidate)
    for candidate in (policy.get("routing_candidates") or policy.get("model_pairs") or []):
        if not isinstance(candidate, dict):
            continue
        key = (str(candidate.get("requested_model") or ""), str(candidate.get("routed_model") or ""))
        if key[0] and key[1]:
            preferred.setdefault(key, dict(candidate))

    return {
        "profile_id": str(policy.get("profile_id") or "thin-routing-policy-v1"),
        "mode": str(policy.get("mode") or "shadow_candidate_pass_through"),
        "enabled": bool(policy.get("enabled", True)),
        "kill_switch": bool(policy.get("kill_switch", False)),
        "sample_rate": float(policy.get("sample_rate") or 0.0),
        "daily_budget_usd": float(policy.get("daily_budget_usd") or 0.0),
        "min_text_chars": int(policy.get("min_text_chars") or 0),
        "max_text_chars": int(policy.get("max_text_chars") or 0),
        "providers": list(policy.get("providers") or []),
        "source_surfaces": list(policy.get("source_surfaces") or []),
        "streaming_shadow_source_surfaces": list(policy.get("streaming_shadow_source_surfaces") or []),
        "blocklist": list(policy.get("blocklist") or []),
        "preferred_pathways": list(preferred.values()),
        "fallback_routes": [
            {
                "requested_model": item.get("requested_model"),
                "routed_model": item.get("routed_model"),
                **({"provider": item["provider"]} if item.get("provider") else {}),
                **({"source_surface": item["source_surface"]} if item.get("source_surface") else {}),
            }
            for item in (policy.get("fallback_routes") or [])
            if isinstance(item, dict)
        ],
        "eligibility_overrides": list(policy.get("eligibility_overrides") or []),
        "similarity_threshold": float(policy.get("similarity_threshold") or 0.0),
        "min_samples_for_confidence": int(policy.get("min_samples_for_confidence") or 1),
        "store_response_bodies": bool(policy.get("store_response_bodies", False)),
    }


def _routing_candidate_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(candidate.get(key) for key in (
        "requested_model",
        "routed_model",
        "provider",
        "source_surface",
        "app_family",
        "category",
        "workflow_phase",
        "stream",
        "min_text_chars",
        "max_text_chars",
        "min_input_tokens",
        "max_input_tokens",
    ))


def _candidate_id_for_dashboard(payload: dict[str, Any]) -> str:
    material = stable_json({
        key: payload.get(key)
        for key in (
            "requested_model",
            "routed_model",
            "provider",
            "source_surface",
            "app_family",
            "category",
            "workflow_phase",
            "stream",
        )
        if payload.get(key) not in (None, "")
    })
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    provider = _public_label(payload.get("provider"), fallback="provider")
    requested = _public_label(payload.get("requested_model"), fallback="requested")
    routed = _public_label(payload.get("routed_model"), fallback="routed")
    category = _public_label(payload.get("category"), fallback="shape")
    return _safe_candidate_id(
        f"dashboard-{provider}-{requested}-to-{routed}-{category}-{digest}",
        fallback=f"dashboard-routing-candidate:{digest}",
    )


def _suggest_adjacent_routed_model(requested: str) -> str:
    model_l = requested.lower()
    if "fable" in model_l or "mythos" in model_l:
        return "claude-opus-4-8"
    if "opus" in model_l:
        return "claude-sonnet-4-6"
    if "sonnet" in model_l:
        return "claude-haiku-4-5-20251001"
    if "gpt-5.5" in model_l or "gpt-5-5" in model_l:
        return "gpt-5.4"
    if "gpt-5.4" in model_l or "gpt-5-4" in model_l:
        return "gpt-5.3"
    if "gpt-5.3-codex" in model_l or "gpt-5-3-codex" in model_l:
        return "gpt-5-codex"
    if "gpt-5-codex" in model_l:
        return "gpt-5-mini"
    if "gpt-5.3" in model_l or "gpt-5-3" in model_l:
        return "gpt-5-mini"
    if "gpt-5-mini" in model_l:
        return "gpt-5-nano"
    return requested


def _fallback_route_for_requested(requested: str) -> str | None:
    for route in ROUTING_EXPERIMENT_POLICY.get("fallback_routes") or []:
        if not isinstance(route, dict):
            continue
        if str(route.get("requested_model") or "") != requested:
            continue
        routed = str(route.get("routed_model") or "").strip()
        if routed and routed != requested:
            return routed
    suggested = _suggest_adjacent_routed_model(requested)
    return suggested if suggested and suggested != requested else None


def _blocklist_matches(
    item: dict[str, Any],
    *,
    requested: str,
    routed: str,
    provider: str = "",
    source_surface: str = "",
    app_family: str = "",
    category: str = "",
    workflow_phase: str = "",
    pathway_id: str = "",
) -> bool:
    model = str(item.get("model") or "").strip()
    if model and model not in {requested, routed}:
        return False
    exact_fields = {
        "requested_model": requested,
        "routed_model": routed,
        "provider": provider,
        "source_surface": source_surface,
        "app_family": app_family,
        "category": category,
        "workflow_phase": workflow_phase,
        "pathway_id": pathway_id,
    }
    for key, actual in exact_fields.items():
        expected = str(item.get(key) or "").strip()
        if expected and expected != str(actual):
            return False
    return bool(model or any(str(item.get(key) or "").strip() for key in exact_fields))


def _blocked_pathway(
    *,
    requested: str,
    routed: str,
    provider: str = "",
    source_surface: str = "",
    app_family: str = "",
    category: str = "",
    workflow_phase: str = "",
    pathway_id: str = "",
) -> dict[str, Any] | None:
    for item in ROUTING_EXPERIMENT_POLICY.get("blocklist") or []:
        if not isinstance(item, dict):
            continue
        if _blocklist_matches(
            item,
            requested=requested,
            routed=routed,
            provider=provider,
            source_surface=source_surface,
            app_family=app_family,
            category=category,
            workflow_phase=workflow_phase,
            pathway_id=pathway_id,
        ):
            return dict(item)
    return None


def _preferred_pathway_for_requested(
    requested: str,
    *,
    provider: str = "",
    source_surface: str = "",
    category: str = "",
    workflow_phase: str = "",
    stream: bool = False,
    text_chars: int = 0,
    input_tokens: int = 0,
) -> dict[str, Any] | None:
    app_family = _app_family(provider, source_surface, requested)
    for candidate in ROUTING_EXPERIMENT_POLICY.get("preferred_pathways") or []:
        if not isinstance(candidate, dict):
            continue
        if _candidate_matches(
            candidate,
            requested=requested,
            provider=provider,
            source_surface=source_surface,
            app_family=app_family,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
            text_chars=text_chars,
            input_tokens=input_tokens,
        ):
            routed = str(candidate.get("routed_model") or "").strip()
            if routed and not _blocked_pathway(
                requested=requested,
                routed=routed,
                provider=provider,
                source_surface=source_surface,
                app_family=app_family,
                category=category,
                workflow_phase=workflow_phase,
                pathway_id=str(candidate.get("candidate_id") or ""),
            ):
                return dict(candidate)
    return None


def _thin_candidate_for_requested(
    requested: str,
    *,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_chars: int,
    input_tokens: int,
    target_model: str | None = None,
    candidate_source: str = "deterministic-fallback",
) -> dict[str, Any] | None:
    if not bool(ROUTING_EXPERIMENT_POLICY.get("thin_client_routing")):
        return None
    app_family = _app_family(provider, source_surface, requested)
    preferred = _preferred_pathway_for_requested(
        requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    )
    if preferred is not None:
        preferred["policy_shape"] = "preferred_pathways"
        preferred["candidate_source"] = preferred.get("candidate_source") or "client-preferred-pathway"
        return preferred

    if candidate_source == "deterministic-fallback" and str(category or "").startswith("tool"):
        return None

    routed = str(target_model or "").strip() or _fallback_route_for_requested(requested)
    if not routed or routed == requested:
        return None
    if _blocked_pathway(
        requested=requested,
        routed=routed,
        provider=provider,
        source_surface=source_surface,
        app_family=app_family,
        category=category,
        workflow_phase=workflow_phase,
    ):
        return None
    material = {
        "requested_model": requested,
        "routed_model": routed,
        "provider": provider,
        "source_surface": source_surface,
        "category": category,
        "workflow_phase": workflow_phase,
        "stream": bool(stream),
        "candidate_source": candidate_source,
    }
    digest = hashlib.sha256(stable_json(material).encode("utf-8")).hexdigest()[:12]
    return {
        "candidate_id": f"thin-{candidate_source}:{digest}",
        "requested_model": requested,
        "routed_model": routed,
        "provider": provider,
        "source_surface": source_surface,
        "app_family": app_family,
        "category": category,
        "workflow_phase": workflow_phase,
        "stream": bool(stream),
        "policy_shape": "fallback_routes" if candidate_source == "deterministic-fallback" else candidate_source,
        "candidate_source": candidate_source,
    }


def routing_pathway_policy_decision(
    *,
    provider: str,
    requested_model: str,
    current_model: str,
    target_model: str | None,
    source_surface: str = "",
    category: str = "",
    workflow_phase: str = "",
    stream: bool = False,
    pathway_id: str = "",
) -> dict[str, Any]:
    refresh_experiment_policy_if_changed()
    requested = str(requested_model or current_model or "").strip()
    current = str(current_model or requested).strip()
    target = str(target_model or "").strip()
    app_family = _app_family(provider, source_surface, requested)
    preferred = _preferred_pathway_for_requested(
        requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
    )
    preferred_target = str((preferred or {}).get("routed_model") or "").strip()
    decision = "allow"
    reason = "managed-recommendation-allowed"
    route_source = "managed-recommended"
    if preferred_target and preferred_target != target:
        target = preferred_target
        decision = "preferred-override"
        reason = "client-preferred-pathway"
        route_source = "client-preferred-pathway"
    blocked = _blocked_pathway(
        requested=requested,
        routed=target,
        provider=provider,
        source_surface=source_surface,
        app_family=app_family,
        category=category,
        workflow_phase=workflow_phase,
        pathway_id=pathway_id,
    )
    if blocked is not None:
        decision = "blocked"
        reason = "client-routing-blocklist"
        route_source = "client-blocklist"
    return {
        "schema": "tokenclaw.routing_pathway_policy_decision.v1",
        "decision": decision,
        "allowed": decision != "blocked",
        "reason": reason,
        "route_source": route_source,
        "requested_model": requested,
        "current_model": current,
        "original_target_model": str(target_model or "").strip() or None,
        "target_model": target or None,
        "preferred_pathway": _candidate_public_metadata(preferred) if preferred is not None else None,
        "blocklist_match": blocked,
        "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
        "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }


def _apply_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    if data.get("profile_id") not in (None, ""):
        policy["profile_id"] = str(data["profile_id"])
    if data.get("mode") not in (None, ""):
        mode = str(data["mode"]).strip()
        if mode in {"applied_routed_down", "shadow_candidate_pass_through"}:
            policy["mode"] = mode
    policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
    policy["kill_switch"] = _as_bool(data.get("kill_switch"), policy["kill_switch"])
    policy["thin_client_routing"] = _as_bool(
        data.get("thin_client_routing"),
        bool(policy.get("thin_client_routing", True)),
    )
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
        policy["model_pairs"] = _clean_routing_candidates(pairs, shape="model_pairs")
        if "routing_candidates" not in data and "candidates" not in data:
            policy["routing_candidates"] = []
    elif data.get("requested_model") is not None and data.get("routed_model") is not None:
        policy["model_pairs"] = [
            _clean_routing_candidate(data, 0, shape="model_pairs")
        ]
        policy["model_pairs"] = [item for item in policy["model_pairs"] if item is not None]
    routing_candidates = data.get("routing_candidates")
    if routing_candidates is None:
        routing_candidates = data.get("candidates")
    if isinstance(routing_candidates, list):
        policy["routing_candidates"] = _clean_routing_candidates(routing_candidates, shape="routing_candidates")
    if isinstance(data.get("preferred_pathways"), list):
        policy["preferred_pathways"] = _clean_routing_candidates(data["preferred_pathways"], shape="preferred_pathways")
    if data.get("fallback_routes") is not None:
        policy["fallback_routes"] = _clean_fallback_routes(data.get("fallback_routes"))
    if isinstance(data.get("blocklist"), list):
        policy["blocklist"] = _clean_blocklist(data.get("blocklist"))
    if (
        any(key in data for key in ("blocklist", "preferred_pathways", "fallback_routes"))
        and "model_pairs" not in data
        and "routing_candidates" not in data
        and "candidates" not in data
    ):
        policy["thin_client_routing"] = True
        policy["model_pairs"] = []
        policy["routing_candidates"] = []
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
    manual_layers: list[tuple[Path, dict[str, Any]]] = []
    for path in _manual_rule_candidates("routing_experiments.yaml", "TOKENCLAW_ROUTING_EXPERIMENTS"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            manual_layers.append((path, data))
    if manual_layers:
        primary_path, primary_data = manual_layers[0]
        policy = _apply_policy_yaml(_default_experiment_policy(), primary_data)
        primary_uses_model_pairs_only = (
            isinstance(primary_data.get("model_pairs"), list)
            and primary_data.get("routing_candidates") is None
            and primary_data.get("candidates") is None
        )
        primary_uses_thin_only = (
            any(key in primary_data for key in ("blocklist", "preferred_pathways", "fallback_routes"))
            and primary_data.get("model_pairs") is None
            and primary_data.get("routing_candidates") is None
            and primary_data.get("candidates") is None
        )
        if primary_uses_model_pairs_only:
            policy["thin_client_routing"] = False
            return policy, "local-manual", str(primary_path)
        if primary_uses_thin_only:
            policy["thin_client_routing"] = True
            return policy, "local-manual", str(primary_path)
        existing_identities = {
            _routing_candidate_identity(candidate)
            for candidate in _clean_routing_candidates(
                policy.get("routing_candidates") or [],
                shape="routing_candidates",
            )
        }
        for _, layer_data in manual_layers[1:]:
            raw_candidates = layer_data.get("routing_candidates")
            if raw_candidates is None:
                raw_candidates = layer_data.get("candidates")
            for candidate in _clean_routing_candidates(raw_candidates, shape="routing_candidates"):
                identity = _routing_candidate_identity(candidate)
                if identity in existing_identities:
                    continue
                policy.setdefault("routing_candidates", []).append(candidate)
                existing_identities.add(identity)
        return policy, "local-manual", str(primary_path)

    defaults_path = Path(__file__).parent / "routing_experiments.yaml"
    policy = _default_experiment_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _apply_policy_yaml(policy, data)

    policy["enabled"] = os.getenv("TOKENCLAW_ROUTING_EXPERIMENTS_ENABLED", "1" if policy["enabled"] else "0") != "0"
    policy["kill_switch"] = os.getenv(
        "TOKENCLAW_ROUTING_EXPERIMENT_KILL_SWITCH",
        "1" if policy.get("kill_switch") else "0",
    ) != "0"
    policy["sample_rate"] = max(
        0.0,
        min(1.0, float(os.getenv("TOKENCLAW_ROUTING_EXPERIMENT_SAMPLE_RATE", str(policy["sample_rate"])))),
    )
    policy["daily_budget_usd"] = max(
        0.0,
        float(os.getenv("TOKENCLAW_ROUTING_EXPERIMENT_DAILY_BUDGET_USD", str(policy["daily_budget_usd"]))),
    )
    policy["similarity_threshold"] = max(
        0.0,
        min(1.0, float(os.getenv("TOKENCLAW_ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD", str(policy["similarity_threshold"])))),
    )
    return policy, "local-default", str(defaults_path)


def _experiment_policy_file_paths() -> list[Path]:
    paths = _manual_rule_candidates("routing_experiments.yaml", "TOKENCLAW_ROUTING_EXPERIMENTS")
    paths.append(Path(__file__).parent / "routing_experiments.yaml")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _experiment_policy_file_snapshots() -> dict[str, dict[str, Any]]:
    return {str(path): policy_file_snapshot(path) for path in _experiment_policy_file_paths()}


def _experiment_policy_snapshot_key(snapshots: dict[str, dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                path,
                bool(snapshot.get("exists")),
                bool(snapshot.get("is_file")),
                snapshot.get("size"),
                snapshot.get("mtime_ns"),
                snapshot.get("sha256"),
            )
            for path, snapshot in snapshots.items()
        )
    )


ROUTING_EXPERIMENT_POLICY, ROUTING_EXPERIMENT_POLICY_SOURCE, ROUTING_EXPERIMENT_RULES_PATH = _load_experiment_policy()
ROUTING_EXPERIMENT_RULES_LOADED_AT = utc_now()
ROUTING_EXPERIMENT_RULES_LOADED_FILE = policy_file_snapshot(ROUTING_EXPERIMENT_RULES_PATH)
ROUTING_EXPERIMENT_RULES_LOADED_FILES = _experiment_policy_file_snapshots()
ROUTING_EXPERIMENT_ENABLED = bool(ROUTING_EXPERIMENT_POLICY["enabled"])
ROUTING_EXPERIMENT_MODE = str(ROUTING_EXPERIMENT_POLICY.get("mode") or "applied_routed_down")
ROUTING_EXPERIMENT_SAMPLE_RATE = float(ROUTING_EXPERIMENT_POLICY["sample_rate"])
ROUTING_EXPERIMENT_DAILY_BUDGET_USD = float(ROUTING_EXPERIMENT_POLICY["daily_budget_usd"])
ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD = float(ROUTING_EXPERIMENT_POLICY["similarity_threshold"])
ROUTING_EXPERIMENT_MIN_SAMPLES = int(ROUTING_EXPERIMENT_POLICY["min_samples_for_confidence"])
ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES = bool(ROUTING_EXPERIMENT_POLICY["store_response_bodies"])
ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE = "routing_experiment_outcome"
ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS = 168
# Promotion evidence is scored over a recent window so that pre-fix shadow failures
# (e.g. the system-role / clear_thinking 400s and text-only-0.0 scores collected
# before those fixes deployed) age out and stop permanently poisoning a candidate's
# 400-count and pass-rate. Older rows still appear in the raw table; they just no
# longer gate promotion. 0 / unset disables the window (score all history).
try:
    ROUTING_PROMOTION_EVIDENCE_WINDOW_HOURS = float(
        os.getenv("TOKENCLAW_ROUTING_PROMOTION_EVIDENCE_WINDOW_HOURS", "0")
    )
except (TypeError, ValueError):
    ROUTING_PROMOTION_EVIDENCE_WINDOW_HOURS = 0.0
ROUTING_PROMOTION_MIN_COMPARED_COVERAGE = 0.80
ROUTING_PROMOTION_MIN_PASS_RATE = 0.90
ROUTING_PROMOTION_MAX_SHADOW_ERROR_RATE = 0.05
ROUTING_PROMOTION_MAX_PRIMARY_ERROR_RATE = 0.05
ROUTING_PROMOTION_SCHEMA = "tokenclaw.routing_experiment_promotion_verdict.v1"


def refresh_experiment_policy_if_changed(*, force: bool = False) -> bool:
    global ROUTING_EXPERIMENT_POLICY
    global ROUTING_EXPERIMENT_POLICY_SOURCE
    global ROUTING_EXPERIMENT_RULES_PATH
    global ROUTING_EXPERIMENT_RULES_LOADED_AT
    global ROUTING_EXPERIMENT_RULES_LOADED_FILE
    global ROUTING_EXPERIMENT_RULES_LOADED_FILES
    global ROUTING_EXPERIMENT_ENABLED
    global ROUTING_EXPERIMENT_MODE
    global ROUTING_EXPERIMENT_SAMPLE_RATE
    global ROUTING_EXPERIMENT_DAILY_BUDGET_USD
    global ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD
    global ROUTING_EXPERIMENT_MIN_SAMPLES
    global ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES

    current_snapshots = _experiment_policy_file_snapshots()
    if not force and _experiment_policy_snapshot_key(current_snapshots) == _experiment_policy_snapshot_key(
        ROUTING_EXPERIMENT_RULES_LOADED_FILES
    ):
        return False

    policy, source, rules_path = _load_experiment_policy()
    ROUTING_EXPERIMENT_POLICY = policy
    ROUTING_EXPERIMENT_POLICY_SOURCE = source
    ROUTING_EXPERIMENT_RULES_PATH = rules_path
    ROUTING_EXPERIMENT_RULES_LOADED_AT = utc_now()
    ROUTING_EXPERIMENT_RULES_LOADED_FILE = policy_file_snapshot(rules_path)
    ROUTING_EXPERIMENT_RULES_LOADED_FILES = _experiment_policy_file_snapshots()
    ROUTING_EXPERIMENT_ENABLED = bool(policy["enabled"])
    ROUTING_EXPERIMENT_MODE = str(policy.get("mode") or "applied_routed_down")
    ROUTING_EXPERIMENT_SAMPLE_RATE = float(policy["sample_rate"])
    ROUTING_EXPERIMENT_DAILY_BUDGET_USD = float(policy["daily_budget_usd"])
    ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD = float(policy["similarity_threshold"])
    ROUTING_EXPERIMENT_MIN_SAMPLES = int(policy["min_samples_for_confidence"])
    ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES = bool(policy["store_response_bodies"])
    return True


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
    candidates = _all_routing_candidates()
    if not candidates:
        return True
    for candidate in candidates:
        if str(candidate.get("requested_model") or "") == requested and str(candidate.get("routed_model") or "") == routed:
            return True
    return False


def _all_routing_candidates() -> list[dict[str, Any]]:
    refresh_experiment_policy_if_changed()
    configured = [
        dict(item)
        for item in ROUTING_EXPERIMENT_POLICY.get("routing_candidates") or []
        if isinstance(item, dict)
    ]
    if configured:
        return configured
    return [
        dict(item)
        for item in ROUTING_EXPERIMENT_POLICY.get("model_pairs") or []
        if isinstance(item, dict)
    ]


def _input_tokens_from_routing(routing_meta: dict[str, Any], body: dict[str, Any]) -> int:
    for source in (routing_meta, body):
        for key in ("input_tokens_est", "input_tokens", "actual_input_tokens"):
            value = source.get(key) if isinstance(source, dict) else None
            parsed = _as_non_negative_int(value)
            if parsed is not None:
                return parsed
    return 0


def _candidate_matches(
    candidate: dict[str, Any],
    *,
    requested: str,
    provider: str,
    source_surface: str,
    app_family: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_chars: int,
    input_tokens: int,
) -> bool:
    if str(candidate.get("requested_model") or "") != requested:
        return False
    routed = str(candidate.get("routed_model") or "").strip()
    if not routed or routed == requested:
        return False
    exact_fields = {
        "provider": provider,
        "source_surface": source_surface,
        "app_family": app_family,
        "category": category,
        "workflow_phase": workflow_phase,
    }
    for key, actual in exact_fields.items():
        expected = candidate.get(key)
        if expected not in (None, "") and str(expected) != str(actual):
            return False
    if "stream" in candidate and bool(candidate.get("stream")) != bool(stream):
        return False
    min_text = _as_non_negative_int(candidate.get("min_text_chars"))
    if min_text is not None and text_chars < min_text:
        return False
    max_text = _as_non_negative_int(candidate.get("max_text_chars"))
    if max_text is not None and max_text > 0 and text_chars > max_text:
        return False
    min_tokens = _as_non_negative_int(candidate.get("min_input_tokens"))
    if min_tokens is not None and input_tokens < min_tokens:
        return False
    max_tokens = _as_non_negative_int(candidate.get("max_input_tokens"))
    if max_tokens is not None and max_tokens > 0 and input_tokens > max_tokens:
        return False
    return True


def _eligible_routing_candidates(
    *,
    requested: str,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_chars: int,
    input_tokens: int,
) -> list[dict[str, Any]]:
    app_family = _app_family(provider, source_surface, requested)
    return [
        candidate
        for candidate in _all_routing_candidates()
        if _candidate_matches(
            candidate,
            requested=requested,
            provider=provider,
            source_surface=source_surface,
            app_family=app_family,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
            text_chars=text_chars,
            input_tokens=input_tokens,
        )
    ]


def _candidate_selector_basis(
    *,
    requested: str,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_chars: int,
    input_tokens: int,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "source_surface": source_surface,
        "app_family": _app_family(provider, source_surface, requested),
        "requested_model": requested,
        "category": category,
        "workflow_phase": workflow_phase,
        "stream": bool(stream),
        "text_chars_bucket": _text_chars_bucket(text_chars),
        "input_tokens_bucket": _text_chars_bucket(input_tokens * 4),
    }


def _select_weighted_candidate(
    candidates: list[dict[str, Any]],
    *,
    selector_basis: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if not candidates:
        return None, "none"
    if len(candidates) == 1:
        return candidates[0], "single-eligible-candidate"
    if all(str(item.get("policy_shape") or "") == "model_pairs" for item in candidates):
        return candidates[0], "legacy-model-pairs-first-match"
    weighted: list[tuple[dict[str, Any], float]] = []
    for candidate in candidates:
        weight = _as_non_negative_float(candidate.get("sample_weight"))
        if weight is None:
            weight = 1.0
        if weight > 0:
            weighted.append((candidate, weight))
    if not weighted:
        return None, "all-candidate-weights-zero"
    total = sum(weight for _, weight in weighted)
    digest = hashlib.sha256(stable_json(selector_basis).encode("utf-8")).hexdigest()
    slot = (int(digest[:16], 16) / float(16**16)) * total
    cumulative = 0.0
    for candidate, weight in weighted:
        cumulative += weight
        if slot <= cumulative:
            return candidate, "weighted-metadata-hash"
    return weighted[-1][0], "weighted-metadata-hash"


def _route_down_candidate_for_requested(
    requested: str,
    *,
    provider: str = "",
    source_surface: str = "",
    category: str = "",
    workflow_phase: str = "",
    stream: bool = False,
    text_chars: int = 0,
    input_tokens: int = 0,
) -> str | None:
    eligible = _eligible_routing_candidates(
        requested=requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    )
    selector_basis = _candidate_selector_basis(
        requested=requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    )
    selected, _ = _select_weighted_candidate(eligible, selector_basis=selector_basis)
    if selected is None:
        return None
    return str(selected.get("routed_model") or "").strip() or None


def _applied_candidate_for_pair(
    requested: str,
    routed: str,
    *,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_chars: int,
    input_tokens: int,
) -> dict[str, Any] | None:
    for candidate in _eligible_routing_candidates(
        requested=requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    ):
        if str(candidate.get("routed_model") or "") == routed:
            return candidate
    return None


def _candidate_public_metadata(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    public = {
        "candidate_id": candidate.get("candidate_id"),
        "requested_model": candidate.get("requested_model"),
        "routed_model": candidate.get("routed_model"),
        "policy_shape": candidate.get("policy_shape"),
        "sample_weight": candidate.get("sample_weight", 1.0),
    }
    for key in ("provider", "source_surface", "app_family", "category", "workflow_phase", "candidate_source", "stream", "sample_rate"):
        if key in candidate:
            public[key] = candidate[key]
    return public


def _route_down_candidate_selection(
    requested: str,
    *,
    provider: str,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_chars: int,
    input_tokens: int,
) -> dict[str, Any]:
    eligible = _eligible_routing_candidates(
        requested=requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    )
    basis = _candidate_selector_basis(
        requested=requested,
        provider=provider,
        source_surface=source_surface,
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    )
    selected, selector = _select_weighted_candidate(eligible, selector_basis=basis)
    policy_shape = "routing_candidates" if ROUTING_EXPERIMENT_POLICY.get("routing_candidates") else "model_pairs"
    return {
        "selected": selected,
        "eligible_candidate_count": len(eligible),
        "eligible_candidate_ids": [candidate.get("candidate_id") for candidate in eligible],
        "candidate_selector": selector,
        "candidate_selector_basis": basis,
        "candidate_policy_shape": policy_shape,
    }


def routing_candidate_coverage(
    *,
    requested_model: Any,
    provider: Any = "",
    source_surface: Any = "",
    category: Any = "",
    workflow_phase: Any = "",
    stream: Any = False,
    text_chars: Any = 0,
    input_tokens: Any = 0,
) -> dict[str, Any]:
    refresh_experiment_policy_if_changed()
    requested = str(requested_model or "").strip()
    provider_label = str(provider or "").strip() or "unknown"
    surface_label = str(source_surface or "").strip() or "unknown"
    category_label = str(category or "").strip() or "unknown"
    phase_label = str(workflow_phase or "").strip()
    stream_bool = bool(stream)
    text_count = _as_non_negative_int(text_chars) or 0
    token_count = _as_non_negative_int(input_tokens) or 0
    app_family = _app_family(provider_label, surface_label, requested)
    if not requested:
        return {
            "schema": "tokenclaw.routing_candidate_coverage.v1",
            "status": "missing-requested-model",
            "covered": False,
            "actionable": False,
            "reason": "requested-model-missing",
            "eligible_candidate_count": 0,
            "eligible_candidate_ids": [],
            "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
            "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
            "add_payload": None,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }

    if str(ROUTING_EXPERIMENT_POLICY_SOURCE).startswith("local-"):
        return {
            "schema": "tokenclaw.routing_candidate_coverage.v1",
            "status": "routing-off",
            "covered": False,
            "actionable": False,
            "reason": "no-backed-routing",
            "eligible_candidate_count": 0,
            "eligible_candidate_ids": [],
            "selected_candidate_id": None,
            "suggested_routed_model": None,
            "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
            "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
            "add_payload": None,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "file_paths_included": False,
                "cache_keys_included": False,
            },
        }

    selection = _route_down_candidate_selection(
        requested,
        provider=provider_label,
        source_surface=surface_label,
        category=category_label,
        workflow_phase=phase_label,
        stream=stream_bool,
        text_chars=text_count,
        input_tokens=token_count,
    )
    selected = selection.get("selected")
    suggested = str((selected or {}).get("routed_model") or "").strip()
    covered = selected is not None
    payload: dict[str, Any] | None = None
    if covered and suggested:
        payload = {
            "requested_model": requested,
            "routed_model": suggested,
            "provider": provider_label,
            "source_surface": surface_label,
            "app_family": app_family,
            "category": category_label,
            "stream": stream_bool,
        }
        if phase_label:
            payload["workflow_phase"] = phase_label
        if text_count > 0:
            payload["max_text_chars"] = max(8000, text_count)
    return {
        "schema": "tokenclaw.routing_candidate_coverage.v1",
        "status": "covered" if covered else "uncovered",
        "covered": covered,
        "actionable": False,
        "reason": "matched-routing-candidate" if covered else "no-routing-candidate",
        "eligible_candidate_count": int(selection.get("eligible_candidate_count") or 0),
        "eligible_candidate_ids": list(selection.get("eligible_candidate_ids") or []),
        "selected_candidate_id": (selected or {}).get("candidate_id") if isinstance(selected, dict) else None,
        "suggested_routed_model": suggested or None,
        "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
        "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
        "add_payload": payload,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
        },
    }


def append_dashboard_routing_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    requested = str(payload.get("requested_model") or "").strip()
    routed = str(payload.get("routed_model") or "").strip()
    if not requested or not routed:
        return {
            "schema": "tokenclaw.routing_candidate_append.v1",
            "ok": False,
            "status": "blocked",
            "error": {"type": "invalid_payload", "message": "requested_model and routed_model are required"},
            "wrote_active_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        }
    if requested == routed:
        return {
            "schema": "tokenclaw.routing_candidate_append.v1",
            "ok": False,
            "status": "blocked",
            "error": {"type": "invalid_payload", "message": "routed_model must differ from requested_model"},
            "wrote_active_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        }

    raw_candidate = {
        key: payload.get(key)
        for key in (
            "requested_model",
            "routed_model",
            "provider",
            "source_surface",
            "app_family",
            "category",
            "workflow_phase",
            "candidate_source",
            "stream",
            "min_text_chars",
            "max_text_chars",
            "min_input_tokens",
            "max_input_tokens",
            "sample_rate",
            "sample_weight",
        )
        if payload.get(key) not in (None, "")
    }
    raw_candidate.setdefault("candidate_id", payload.get("candidate_id") or _candidate_id_for_dashboard(raw_candidate))
    raw_candidate.setdefault("candidate_source", "dashboard-recent-call")
    raw_candidate.setdefault("sample_rate", 0.05)
    candidate = _clean_routing_candidate(raw_candidate, 0, shape="routing_candidates")
    if candidate is None:
        return {
            "schema": "tokenclaw.routing_candidate_append.v1",
            "ok": False,
            "status": "blocked",
            "error": {"type": "invalid_payload", "message": "candidate could not be normalized"},
            "wrote_active_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        }

    path = _writable_experiment_config_path()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            data = {}
    else:
        data = dict(ROUTING_EXPERIMENT_POLICY)
        data["policy_source"] = "local-manual"
    candidates = data.get("routing_candidates")
    if not isinstance(candidates, list):
        candidates = []
    clean_existing = _clean_routing_candidates(candidates, shape="routing_candidates")
    target_identity = _routing_candidate_identity(candidate)
    for existing in clean_existing:
        if _routing_candidate_identity(existing) == target_identity:
            return {
                "schema": "tokenclaw.routing_candidate_append.v1",
                "ok": True,
                "status": "already-present",
                "candidate_id": existing.get("candidate_id"),
                "candidate": _candidate_public_metadata(existing),
                "target_file": str(path),
                "wrote_active_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
            }

    data["routing_candidates"] = candidates + [candidate]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return {
        "schema": "tokenclaw.routing_candidate_append.v1",
        "ok": True,
        "status": "appended",
        "candidate_id": candidate.get("candidate_id"),
        "candidate": _candidate_public_metadata(candidate),
        "target_file": str(path),
        "wrote_active_policy_files": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _legacy_route_down_candidate_for_requested(requested: str) -> str | None:
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
    refresh_experiment_policy_if_changed()
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
    input_tokens = _input_tokens_from_routing(routing_meta, body)
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
        "schema": "tokenclaw.routing_experiment_decision.v1",
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
        "input_tokens_est": input_tokens,
        "min_text_chars": int(controls["min_text_chars"]),
        "min_text_chars_scope": controls["min_text_chars_scope"],
        "max_text_chars": int(controls["max_text_chars"]),
        "max_text_chars_scope": controls["max_text_chars_scope"],
        "eligibility_overrides_applied": controls["applied_overrides"],
        "candidate_id": None,
        "selected_candidate": None,
        "eligible_candidate_count": 0,
        "eligible_candidate_ids": [],
        "candidate_selector": "not-evaluated",
        "candidate_policy_shape": "routing_candidates" if ROUTING_EXPERIMENT_POLICY.get("routing_candidates") else "model_pairs",
        "candidate_selector_basis": _candidate_selector_basis(
            requested=requested,
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
            text_chars=text_chars,
            input_tokens=input_tokens,
        ),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": bool(ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES),
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
        },
    }
    if not ROUTING_EXPERIMENT_ENABLED:
        if not forced_openai_canary_shadow:
            return meta
    # Backed or off: local policy (the bundled default *or* the user's manual YAML)
    # is evidence-collection scaffolding, not a license to mint canaries. Canaries are
    # a server responsibility (ARCHITECTURE.md "canary-driven routing lives in
    # tokenclaw_server"); when the server is unavailable or not backing this call there
    # is no point running a canary, so there is no local fallback. Stay off for any
    # local-originated policy source unless the server explicitly forced this shadow
    # (forced_openai_canary_shadow is itself a server signal). Server-directed anthropic
    # shadows arrive via _managed_shadow_experiment_decision and bypass this path.
    if (
        str(ROUTING_EXPERIMENT_POLICY_SOURCE).startswith("local-")
        and not forced_openai_canary_shadow
    ):
        meta["reason"] = "no-backed-routing"
        meta["backing_reason"] = "local-policy-without-managed-backing"
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
    if mode == "shadow_candidate_pass_through":
        if requested != routed:
            meta["reason"] = "already-routed-down"
            return meta
        candidate = None
        selected_candidate: dict[str, Any] | None = None
        force_shadow = False
        if provider == "openai" and managed.get("selected_for_shadow_evaluation") is True:
            candidate = str(managed.get("shadow_model") or managed.get("would_route_model") or "").strip() or None
            force_shadow = candidate is not None
            meta["trigger"] = "managed-policy-routing-canary"
            meta["managed_policy_id"] = managed.get("policy_id")
            meta["canary_cohort"] = (managed.get("local_canary") or {}).get("cohort") if isinstance(managed.get("local_canary"), dict) else None
            if force_shadow:
                meta["candidate_selector"] = "forced-managed-policy-routing-canary"
                meta["candidate_policy_shape"] = "managed-shadow"
                meta["candidate_id"] = _safe_candidate_id(
                    managed.get("policy_id") or managed.get("recommendation_id") or "managed-openai-shadow",
                    fallback="managed-openai-shadow",
                )
                meta["policy_source"] = managed.get("policy_source") or "managed-recommended"
            else:
                meta["reason"] = "managed-shadow-target-missing"
                return meta
        elif provider == "openai":
            meta["reason"] = "openai-shadow-requires-managed-target"
            if openai_canary.get("status") == "applied":
                meta["retired_local_trigger"] = "openai-local-routing-canary"
            return meta
        if candidate is None:
            selection = _route_down_candidate_selection(
                requested,
                provider=provider,
                source_surface=source_surface,
                category=category,
                workflow_phase=workflow_phase,
                stream=stream,
                text_chars=text_chars,
                input_tokens=input_tokens,
            )
            selected_candidate = selection["selected"]
            candidate = str((selected_candidate or {}).get("routed_model") or "").strip() or None
            meta["eligible_candidate_count"] = selection["eligible_candidate_count"]
            meta["eligible_candidate_ids"] = selection["eligible_candidate_ids"]
            meta["candidate_selector"] = selection["candidate_selector"]
            meta["candidate_selector_basis"] = selection["candidate_selector_basis"]
            meta["candidate_policy_shape"] = selection["candidate_policy_shape"]
            if selected_candidate is None:
                selected_candidate = _thin_candidate_for_requested(
                    requested,
                    provider=provider,
                    source_surface=source_surface,
                    category=category,
                    workflow_phase=workflow_phase,
                    stream=stream,
                    text_chars=text_chars,
                    input_tokens=input_tokens,
                )
                candidate = str((selected_candidate or {}).get("routed_model") or "").strip() or None
                if selected_candidate is not None:
                    meta["candidate_selector"] = "thin-client-fallback"
                    meta["candidate_policy_shape"] = selected_candidate.get("policy_shape")
            blocked = _blocked_pathway(
                requested=requested,
                routed=str(candidate or ""),
                provider=provider,
                source_surface=source_surface,
                app_family=_app_family(provider, source_surface, requested),
                category=category,
                workflow_phase=workflow_phase,
                pathway_id=str((selected_candidate or {}).get("candidate_id") or ""),
            )
            if blocked is not None:
                meta["reason"] = "model-pair-blocked"
                meta["blocklist_match"] = blocked
                return meta
        if not candidate:
            fallback_block_target = _fallback_route_for_requested(requested)
            if fallback_block_target:
                blocked = _blocked_pathway(
                    requested=requested,
                    routed=fallback_block_target,
                    provider=provider,
                    source_surface=source_surface,
                    app_family=_app_family(provider, source_surface, requested),
                    category=category,
                    workflow_phase=workflow_phase,
                )
                if blocked is not None:
                    meta["reason"] = "model-pair-blocked"
                    meta["blocklist_match"] = blocked
                    meta["blocked_routed_model"] = fallback_block_target
                    return meta
            meta["reason"] = "model-pair-not-enabled"
            return meta
        if selected_candidate is not None:
            meta["candidate_id"] = selected_candidate.get("candidate_id")
            meta["selected_candidate"] = _candidate_public_metadata(selected_candidate)
            if selected_candidate.get("sample_rate") is not None and not force_shadow:
                meta["sample_rate"] = round(float(selected_candidate["sample_rate"]), 6)
                meta["sample_rate_scope"] = "candidate"
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
        applied_candidate = _applied_candidate_for_pair(
            requested,
            routed,
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
            text_chars=text_chars,
            input_tokens=input_tokens,
        )
        if applied_candidate is None:
            applied_candidate = _thin_candidate_for_requested(
                requested,
                provider=provider,
                source_surface=source_surface,
                category=category,
                workflow_phase=workflow_phase,
                stream=stream,
                text_chars=text_chars,
                input_tokens=input_tokens,
                target_model=routed,
                candidate_source="applied-or-managed-route",
            )
        if applied_candidate is None:
            meta["reason"] = "model-pair-not-enabled"
            return meta
        selection = _route_down_candidate_selection(
            requested,
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
            text_chars=text_chars,
            input_tokens=input_tokens,
        )
        meta["eligible_candidate_count"] = selection["eligible_candidate_count"]
        meta["eligible_candidate_ids"] = selection["eligible_candidate_ids"]
        meta["candidate_selector"] = "applied-routed-pair-match"
        meta["candidate_selector_basis"] = selection["candidate_selector_basis"]
        meta["candidate_policy_shape"] = selection["candidate_policy_shape"]
        meta["candidate_id"] = applied_candidate.get("candidate_id")
        meta["selected_candidate"] = _candidate_public_metadata(applied_candidate)
        if applied_candidate.get("sample_rate") is not None:
            meta["sample_rate"] = round(float(applied_candidate["sample_rate"]), 6)
            meta["sample_rate_scope"] = "candidate"
    else:
        meta["reason"] = "unsupported-mode"
        return meta
    if routing_meta.get("fallback_reason"):
        meta["reason"] = "fallback-used"
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
    sample_rate = float(meta["sample_rate"])
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


def _response_tool_calls(resp: dict[str, Any]) -> list[str]:
    """Canonical per-tool-call strings (name + sorted-key JSON args) from a response.

    Tool-execution turns answer with tool_use/tool_call blocks and little or no
    prose, so a text-only similarity is ~0 even when two models choose the very same
    action. Comparing the tool calls is what makes the quality gate meaningful for
    the dominant tool-execution traffic. Computed locally for scoring only; the raw
    strings are never persisted (only the count, a hash, and the float similarity).
    """
    if not isinstance(resp, dict):
        return []
    calls: list[str] = []

    def add(name: Any, args: Any) -> None:
        calls.append("tool_use:" + str(name or "") + ":" + stable_json(args))

    for block in resp.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            add(block.get("name"), block.get("input"))
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("function_call", "tool_call"):
            add(item.get("name"), item.get("arguments"))
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                add(block.get("name"), block.get("input"))
    for choice in resp.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                add(function.get("name"), function.get("arguments"))
    return calls


def response_output_signature(resp: dict[str, Any]) -> str:
    """Text output plus tool-call signature — the basis for routing-quality similarity."""
    text = response_output_text(resp)
    tools = "\n".join(_response_tool_calls(resp))
    if text and tools:
        return text + "\n" + tools
    return text or tools


def _tool_call_set_similarity(primary_calls: list[str], shadow_calls: list[str]) -> float | None:
    """Jaccard agreement between two canonical tool-call sets, or None if no calls."""
    primary_set, shadow_set = set(primary_calls), set(shadow_calls)
    union = primary_set | shadow_set
    if not union:
        return None
    return len(primary_set & shadow_set) / len(union)


def compare_response_outputs(primary_response: dict[str, Any] | None, shadow_response: dict[str, Any] | None) -> dict[str, Any]:
    primary_text = response_output_text(primary_response or {})
    shadow_text = response_output_text(shadow_response or {})
    primary_signature = response_output_signature(primary_response or {})
    shadow_signature = response_output_signature(shadow_response or {})
    primary_calls = _response_tool_calls(primary_response or {})
    shadow_calls = _response_tool_calls(shadow_response or {})
    if primary_signature or shadow_signature:
        text_similarity = cosine_similarity(
            build_embedding(primary_signature), build_embedding(shadow_signature)
        )
    else:
        text_similarity = 1.0 if stable_json(primary_response or {}) == stable_json(shadow_response or {}) else 0.0
    tool_call_similarity = _tool_call_set_similarity(primary_calls, shadow_calls)
    # On a tool-execution turn the actionable output is the tool call, so route
    # equivalence is tool-call agreement — not prose overlap. A thinking primary and
    # a (forced) non-thinking shadow always differ in wording around an identical
    # action, which dragged the diluted text+tool cosine to ~0.2 and made every
    # tool-execution canary fail quality, blocking promotion on the dominant traffic.
    # Score those turns on tool-call agreement; same tool+args -> equivalent action.
    if primary_calls:
        similarity = tool_call_similarity if tool_call_similarity is not None else text_similarity
    else:
        similarity = text_similarity
    return {
        "primary_output_chars": len(primary_text),
        "shadow_output_chars": len(shadow_text),
        "primary_tool_call_count": len(primary_calls),
        "shadow_tool_call_count": len(shadow_calls),
        "primary_output_sha256": sha256_text(primary_signature),
        "shadow_output_sha256": sha256_text(shadow_signature),
        "text_similarity": round(float(text_similarity), 6),
        "tool_call_similarity": round(float(tool_call_similarity), 6) if tool_call_similarity is not None else None,
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
    for family in ("haiku", "sonnet", "opus", "fable", "mythos", "codex", "gpt-5", "gpt-4", "gpt-3"):
        if family in model_l:
            return family
    return "other"


def _app_family(provider: Any, source_surface: Any, requested_model: Any) -> str:
    provider_l = str(provider or "").lower()
    surface_l = str(source_surface or "").lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" or surface_l == "anthropic_messages":
        return "claude_code"
    if provider_l == "openai" and (surface_l == "codex_turn" or "codex" in model_l):
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
    shadow_http_error_detail = (
        experiment_meta.get("shadow_http_error_detail")
        if isinstance(experiment_meta.get("shadow_http_error_detail"), dict)
        else None
    )
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
        "schema": "tokenclaw.routing_experiment_feedback.v1",
        "experiment_id": experiment_id,
        "sampled": bool(experiment_meta.get("sampled")),
        "mode": experiment_meta.get("mode") or "applied_routed_down",
        "counterfactual": bool(experiment_meta.get("counterfactual")),
        "shadow_only": bool(experiment_meta.get("shadow_only")),
        "status": status,
        "provider": experiment_meta.get("provider") or "anthropic",
        "source_surface": experiment_meta.get("source_surface") or "anthropic_messages",
        "policy_source": experiment_meta.get("policy_source"),
        "requested_model": requested_model,
        "routed_model": routed_model,
        "primary_model": primary_model,
        "shadow_model": shadow_model,
        "category": category,
        "workflow_phase": routing_meta.get("workflow_phase") or experiment_meta.get("workflow_phase"),
        "candidate_id": experiment_meta.get("candidate_id"),
        "eligible_candidate_count": experiment_meta.get("eligible_candidate_count"),
        "candidate_selector": experiment_meta.get("candidate_selector"),
        "candidate_policy_shape": experiment_meta.get("candidate_policy_shape"),
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
        "shadow_http_error_detail": shadow_http_error_detail,
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
        "schema": "tokenclaw.routing_experiment_outcome_event.v1",
        "event_type": "routing_experiment_outcome",
        "generated_at": utc_now(),
        "source_surface": source_surface,
        "app_family": _app_family(provider, source_surface, requested_model),
        "provider": provider,
        "workflow_phase": feedback_features.get("workflow_phase") or "unknown",
        "category": feedback_features.get("category") or "unknown",
        "candidate": {
            "schema": "tokenclaw.routing_experiment_candidate.v1",
            "mode": mode,
            "counterfactual": counterfactual,
            "shadow_only": shadow_only,
            "candidate_bucket": (
                f"{feedback_features.get('category') or 'unknown'}:{requested_family or 'unknown'}->{routed_family or 'unknown'}"
            ),
            "candidate_id": feedback_features.get("candidate_id"),
            "eligible_candidate_count": feedback_features.get("eligible_candidate_count"),
            "candidate_selector": feedback_features.get("candidate_selector"),
            "candidate_policy_shape": feedback_features.get("candidate_policy_shape"),
            "policy_source": feedback_features.get("policy_source"),
            "provider": provider,
            "source_surface": source_surface,
            "text_chars_bucket": feedback_features.get("text_chars_bucket"),
            "requested_model_family": requested_family,
            "routed_model_family": routed_family,
            "shadow_model_family": shadow_family,
        },
        "outcome": {
            "schema": "tokenclaw.routing_experiment_outcome_summary.v1",
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
            "shadow_http_error_detail": feedback_features.get("shadow_http_error_detail"),
        },
        "reason_codes": reason_codes,
        "routing": {
            "schema": "tokenclaw.routing_experiment_routing_basis.v1",
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
    if provider == "openai":
        event["candidate"]["requested_model"] = requested_model
        event["candidate"]["routed_model"] = routed_model
        event["candidate"]["shadow_model"] = shadow_model
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


def _configured_candidate_key(candidate: dict[str, Any]) -> tuple[str, str, bool, str, str, str, str]:
    return (
        _public_label(candidate.get("provider")),
        _public_label(candidate.get("source_surface")),
        bool(candidate.get("stream")),
        _public_label(candidate.get("requested_model"), fallback=""),
        _public_label(candidate.get("routed_model"), fallback=""),
        _public_label(candidate.get("category")),
        _public_label(candidate.get("workflow_phase")),
    )


def _report_candidate_key(candidate: dict[str, Any]) -> tuple[str, str, bool, str, str, str, str]:
    return (
        _public_label(candidate.get("provider")),
        _public_label(candidate.get("source_surface")),
        bool(candidate.get("stream")),
        _public_label(candidate.get("requested_model"), fallback=""),
        _public_label(candidate.get("routed_model"), fallback=""),
        _public_label(candidate.get("category")),
        _public_label(candidate.get("workflow_phase")),
    )


def _scoreboard_candidate_id(candidate: dict[str, Any]) -> str:
    explicit = candidate.get("candidate_id") or candidate.get("configured_candidate_id")
    if explicit:
        return _public_label(explicit, fallback="unknown-candidate")
    parts = list(_report_candidate_key(candidate))
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"shadow-route-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _new_readiness_scoreboard_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.routing_candidate_readiness_scoreboard_row.v1",
        "candidate_id": _scoreboard_candidate_id(candidate),
        "label": _public_label(candidate.get("label"), fallback=""),
        "provider": _public_label(candidate.get("provider")),
        "source_surface": _public_label(candidate.get("source_surface")),
        "stream": bool(candidate.get("stream")),
        "requested_model": _public_label(candidate.get("requested_model"), fallback=""),
        "routed_model": _public_label(candidate.get("routed_model"), fallback=""),
        "category": _public_label(candidate.get("category")),
        "workflow_phase": _public_label(candidate.get("workflow_phase")),
        "sample_count": 0,
        "compared_count": 0,
        "passed_count": 0.0,
        "holdout_count": 0,
        "applied_count": 0,
        "baseline_cost_usd": 0.0,
        "candidate_cost_usd": 0.0,
        "holdout_baseline_cost_usd": 0.0,
        "applied_candidate_cost_usd": 0.0,
        "candidate_error_count": 0.0,
        "primary_error_count": 0.0,
        "fallback_or_retry_count": 0,
        "similarity_total": 0.0,
        "similarity_count": 0,
        "latest_evidence_at": None,
        "mode_counts": {},
        "reason_codes": [],
    }


def _merge_readiness_observed_row(row: dict[str, Any], observed: dict[str, Any]) -> None:
    samples = int(observed.get("samples") or 0)
    compared = int(observed.get("compared_samples") or 0)
    mode = str(observed.get("mode") or "unknown")
    primary_cost = float(observed.get("primary_cost_usd") or 0.0)
    shadow_cost = float(observed.get("shadow_cost_usd") or 0.0)
    pass_rate = observed.get("pass_rate")
    avg_similarity = observed.get("avg_similarity")

    row["sample_count"] += samples
    row["compared_count"] += compared
    if pass_rate is not None:
        row["passed_count"] += float(pass_rate) * compared
    if avg_similarity is not None:
        row["similarity_total"] += float(avg_similarity) * compared
        row["similarity_count"] += compared
    row["fallback_or_retry_count"] += int(observed.get("fallback_or_retry_count") or 0)
    row["primary_error_count"] += float(observed.get("primary_error_rate") or 0.0) * samples
    row["mode_counts"][mode] = int(row["mode_counts"].get(mode, 0)) + samples

    if mode == "applied_routed_down":
        candidate_cost = primary_cost
        baseline_cost = shadow_cost
        row["applied_count"] += samples
        row["applied_candidate_cost_usd"] += candidate_cost
        row["candidate_error_count"] += float(observed.get("primary_error_rate") or 0.0) * samples
    else:
        baseline_cost = primary_cost
        candidate_cost = shadow_cost
        row["holdout_count"] += samples
        row["holdout_baseline_cost_usd"] += baseline_cost
        row["candidate_error_count"] += float(observed.get("shadow_error_rate") or 0.0) * samples

    row["baseline_cost_usd"] += baseline_cost
    row["candidate_cost_usd"] += candidate_cost
    for reason in observed.get("promotion_reason_codes") or []:
        if reason:
            row["reason_codes"].append(str(reason))
    latest = observed.get("last_sample_at")
    if latest and (row.get("latest_evidence_at") is None or str(latest) > str(row["latest_evidence_at"])):
        row["latest_evidence_at"] = latest


def _finalize_readiness_scoreboard_row(row: dict[str, Any], *, min_samples: int) -> dict[str, Any]:
    samples = int(row.get("sample_count") or 0)
    compared = int(row.get("compared_count") or 0)
    applied_count = int(row.get("applied_count") or 0)
    holdout_count = int(row.get("holdout_count") or 0)
    pass_rate = (float(row.get("passed_count") or 0.0) / compared) if compared else None
    avg_similarity = (
        float(row.get("similarity_total") or 0.0) / int(row.get("similarity_count") or 0)
        if int(row.get("similarity_count") or 0)
        else None
    )
    candidate_error_rate = float(row.get("candidate_error_count") or 0.0) / samples if samples else 0.0
    fallback_rate = int(row.get("fallback_or_retry_count") or 0) / samples if samples else 0.0
    realized_vs_baseline = float(row.get("baseline_cost_usd") or 0.0) - float(row.get("candidate_cost_usd") or 0.0)
    if applied_count and holdout_count:
        holdout_avg = float(row.get("holdout_baseline_cost_usd") or 0.0) / holdout_count
        realized_vs_holdout = (holdout_avg * applied_count) - float(row.get("applied_candidate_cost_usd") or 0.0)
    else:
        realized_vs_holdout = realized_vs_baseline

    reasons = sorted(set(str(reason) for reason in row.get("reason_codes") or [] if reason))
    status = "ready"
    if samples <= 0:
        status = "insufficient-evidence"
        reasons.append("no-source-traffic")
    if samples < min_samples:
        status = "insufficient-evidence"
        reasons.append("insufficient-samples")
    if compared < min_samples:
        status = "insufficient-evidence"
        reasons.append("insufficient-compared-samples")
    if pass_rate is not None and pass_rate < ROUTING_PROMOTION_MIN_PASS_RATE:
        status = "regressing"
        reasons.append("below-similarity-pass-rate")
    if candidate_error_rate > ROUTING_PROMOTION_MAX_SHADOW_ERROR_RATE:
        status = "regressing"
        reasons.append("candidate-error-rate-high")
    if realized_vs_baseline < 0 or realized_vs_holdout < 0:
        status = "regressing"
        reasons.append("candidate-more-expensive")
    if int(row.get("fallback_or_retry_count") or 0) > 0 and status == "ready":
        status = "regressing"
        reasons.append("fallback-or-retry-observed")
    if not reasons and status == "ready":
        reasons.append("readiness-thresholds-met")

    mode_counts = [
        {"value": key, "count": count}
        for key, count in sorted(row.get("mode_counts", {}).items(), key=lambda item: (-item[1], item[0]))
    ]
    finalized = dict(row)
    finalized.update(
        {
            "readiness_status": status,
            "ready": status == "ready",
            "min_sample_gate": {
                "min_samples": min_samples,
                "sample_count": samples,
                "compared_count": compared,
                "sample_gate_passed": samples >= min_samples,
                "compared_gate_passed": compared >= min_samples,
            },
            "sample_count": samples,
            "compared_count": compared,
            "holdout_count": holdout_count,
            "applied_count": applied_count,
            "pass_rate": round(pass_rate, 6) if pass_rate is not None else None,
            "avg_similarity": round(avg_similarity, 6) if avg_similarity is not None else None,
            "candidate_error_rate": round(candidate_error_rate, 6),
            "fallback_or_retry_rate": round(fallback_rate, 6),
            "baseline_cost_usd": round(float(row.get("baseline_cost_usd") or 0.0), 8),
            "candidate_cost_usd": round(float(row.get("candidate_cost_usd") or 0.0), 8),
            "realized_cost_delta_vs_baseline_usd": round(realized_vs_baseline, 8),
            "realized_cost_delta_vs_holdout_usd": round(realized_vs_holdout, 8),
            "mode_counts": mode_counts,
            "reason_codes": sorted(set(reasons)),
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
                "cache_keys_included": False,
            },
        }
    )
    for key in ("passed_count", "similarity_total", "similarity_count", "candidate_error_count", "primary_error_count", "holdout_baseline_cost_usd", "applied_candidate_cost_usd"):
        finalized.pop(key, None)
    return finalized


def _build_routing_candidate_readiness_scoreboard(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    min_samples = ROUTING_EXPERIMENT_MIN_SAMPLES
    configured = _all_routing_candidates()
    rows_by_key: dict[tuple[str, str, bool, str, str, str, str], dict[str, Any]] = {}
    if configured:
        for candidate in configured:
            rows_by_key[_configured_candidate_key(candidate)] = _new_readiness_scoreboard_row(candidate)

    for observed in candidates:
        key = _report_candidate_key(observed)
        row = rows_by_key.setdefault(key, _new_readiness_scoreboard_row(observed))
        if observed.get("candidate_id") and not row.get("candidate_id"):
            row["candidate_id"] = _scoreboard_candidate_id(observed)
        _merge_readiness_observed_row(row, observed)

    rows = [_finalize_readiness_scoreboard_row(row, min_samples=min_samples) for row in rows_by_key.values()]
    rows.sort(
        key=lambda item: (
            {"ready": 0, "regressing": 1, "insufficient-evidence": 2}.get(str(item.get("readiness_status")), 3),
            -int(item.get("sample_count") or 0),
            str(item.get("candidate_id") or ""),
        )
    )
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("readiness_status") or "unknown")] = status_counts.get(str(row.get("readiness_status") or "unknown"), 0) + 1
        for reason in row.get("reason_codes") or []:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "schema": "tokenclaw.routing_candidate_readiness_scoreboard.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "summary": {
            "candidate_count": len(rows),
            "configured_candidate_count": len(configured),
            "sample_count": sum(int(row.get("sample_count") or 0) for row in rows),
            "compared_count": sum(int(row.get("compared_count") or 0) for row in rows),
            "ready_count": status_counts.get("ready", 0),
            "insufficient_evidence_count": status_counts.get("insufficient-evidence", 0),
            "regressing_count": status_counts.get("regressing", 0),
            "total_realized_cost_delta_vs_baseline_usd": round(sum(float(row.get("realized_cost_delta_vs_baseline_usd") or 0.0) for row in rows), 8),
            "total_realized_cost_delta_vs_holdout_usd": round(sum(float(row.get("realized_cost_delta_vs_holdout_usd") or 0.0) for row in rows), 8),
            "min_samples": min_samples,
        },
        "readiness_counts": _count_rows(status_counts, key_name="status"),
        "reason_counts": _count_rows(reason_counts),
        "candidates": rows,
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
            "cache_keys_included": False,
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
        "schema": "tokenclaw.routing_experiment_eligibility_projection.v1",
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


def _is_dashboard_routing_candidate(candidate: dict[str, Any]) -> bool:
    candidate_id = str(candidate.get("candidate_id") or "")
    source = str(candidate.get("candidate_source") or candidate.get("source") or "").replace("_", "-")
    return candidate_id.startswith("dashboard-") or source in {
        "dashboard-recent-call",
        "dashboard-added",
        "dashboard",
    }


def _routing_candidate_fingerprint(candidate: dict[str, Any]) -> str:
    material = {
        key: candidate.get(key)
        for key in (
            "candidate_id",
            "requested_model",
            "routed_model",
            "provider",
            "source_surface",
            "app_family",
            "category",
            "workflow_phase",
            "stream",
            "min_text_chars",
            "max_text_chars",
            "min_input_tokens",
            "max_input_tokens",
            "candidate_source",
        )
        if candidate.get(key) not in (None, "")
    }
    digest = hashlib.sha256(stable_json(material).encode("utf-8")).hexdigest()[:16]
    return f"routing-candidate:{digest}"


def _public_counterfactual_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": _routing_candidate_fingerprint(candidate),
        "candidate_source": _public_label(candidate.get("candidate_source"), fallback="dashboard-recent-call"),
        "provider": _public_label(candidate.get("provider"), fallback="unknown"),
        "source_surface": _public_label(candidate.get("source_surface"), fallback="unknown"),
        "app_family": _public_label(candidate.get("app_family"), fallback="unknown"),
        "requested_model": _public_label(candidate.get("requested_model"), fallback=""),
        "routed_model": _public_label(candidate.get("routed_model"), fallback=""),
        "category": _public_label(candidate.get("category"), fallback="unknown"),
        "workflow_phase": _public_label(candidate.get("workflow_phase"), fallback="unknown"),
        "stream": bool(candidate.get("stream")) if candidate.get("stream") is not None else None,
        "text_chars_bucket": _text_chars_bucket(candidate.get("max_text_chars") or 0),
    }


def _nested_dicts(*values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            result.append(value)
            for nested in value.values():
                if isinstance(nested, dict):
                    result.append(nested)
    return result


def _routing_lifecycle_status(
    *,
    candidate: dict[str, Any],
    row: dict[str, Any],
    routing: dict[str, Any],
    experiment: dict[str, Any],
) -> str:
    labels: list[str] = []
    for source in _nested_dicts(
        routing,
        experiment,
        routing.get("routing_experiment"),
        routing.get("openai_canary"),
        routing.get("anthropic_canary"),
        routing.get("canary"),
        routing.get("managed_recommendation"),
    ):
        for key in (
            "lifecycle_event",
            "lifecycle_status",
            "canary_cohort",
            "cohort",
            "status",
            "reason",
            "fallback_reason",
            "outcome_bucket",
            "routing_outcome_label",
        ):
            value = source.get(key)
            if value not in (None, ""):
                labels.append(str(value).lower().replace("_", "-"))
    label_text = " ".join(labels)
    if "safety-stop" in label_text or "safety-stopped" in label_text:
        return "safety_stopped"
    if "rollback" in label_text:
        return "rollback"
    if "holdout" in label_text:
        return "canary_holdout"
    if "canary-applied" in label_text or "selected-canary" in label_text:
        return "canary_applied"
    if experiment.get("shadow_only") or experiment.get("counterfactual"):
        return "canary_holdout"
    routed = str(row.get("routed_model") or "")
    if routed and routed == str(candidate.get("routed_model") or ""):
        return "canary_applied"
    requested = str(row.get("requested_model") or "")
    if routed and requested and routed == requested:
        return "canary_holdout"
    return "unknown"


def _routing_row_matches_candidate(
    candidate: dict[str, Any],
    row: dict[str, Any],
    *,
    routing: dict[str, Any],
    experiment: dict[str, Any],
) -> bool:
    candidate_id = str(candidate.get("candidate_id") or "")
    selected_candidate = experiment.get("selected_candidate")
    selected_candidate_id = selected_candidate.get("candidate_id") if isinstance(selected_candidate, dict) else ""
    observed_candidate_id = str(
        experiment.get("candidate_id")
        or selected_candidate_id
        or ""
    )
    if candidate_id and observed_candidate_id == candidate_id:
        return True
    requested = str(row.get("requested_model") or experiment.get("requested_model") or "").strip()
    provider = str(row.get("provider") or experiment.get("provider") or candidate.get("provider") or "unknown")
    source_surface = str(
        row.get("source_surface")
        or experiment.get("source_surface")
        or candidate.get("source_surface")
        or "unknown"
    )
    category = str(row.get("category") or experiment.get("category") or routing.get("category") or "unknown")
    workflow_phase = _workflow_phase_from_payloads(experiment, routing)
    stream = bool(row.get("stream"))
    text_chars = _as_non_negative_int(
        experiment.get("text_chars")
        or routing.get("text_chars")
        or row.get("text_chars")
        or 0
    ) or 0
    input_tokens = _as_non_negative_int(
        row.get("actual_input_tokens")
        or row.get("input_tokens_est")
        or experiment.get("input_tokens_est")
        or routing.get("input_tokens_est")
        or 0
    ) or 0
    return _candidate_matches(
        candidate,
        requested=requested,
        provider=provider,
        source_surface=source_surface,
        app_family=_app_family(provider, source_surface, requested),
        category=category,
        workflow_phase=workflow_phase,
        stream=stream,
        text_chars=text_chars,
        input_tokens=input_tokens,
    )


def _new_lifecycle_bucket(candidate: dict[str, Any]) -> dict[str, Any]:
    public = _public_counterfactual_candidate(candidate)
    return {
        "schema": "tokenclaw.routing_experiment_lifecycle_outcome.v1",
        **public,
        "status": "no-local-traffic",
        "matched_count": 0,
        "observed_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "safety_stop_count": 0,
        "rollback_count": 0,
        "skipped_count": 0,
        "unknown_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "projected_saved_usd": 0.0,
        "observed_saved_usd": 0.0,
        "projected_savings_per_1000_calls_usd": 0.0,
        "observed_savings_per_1000_calls_usd": 0.0,
        "latest_observed_at": None,
        "freshness": {
            "schema": "tokenclaw.routing_experiment_lifecycle_freshness.v1",
            "latest_observed_age_hours": None,
            "max_age_hours": ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS,
            "stale": True,
            "reason": "no-local-traffic",
        },
        "cohort_counts": {
            "canary_applied": 0,
            "canary_holdout": 0,
            "safety_stopped": 0,
            "rollback": 0,
            "skipped": 0,
            "unknown": 0,
        },
        "coverage": {
            "schema": "tokenclaw.routing_experiment_lifecycle_coverage.v1",
            "matched_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "has_applied_coverage": False,
            "has_holdout_coverage": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "regression_deltas": {
            "schema": "tokenclaw.routing_experiment_lifecycle_regression_deltas.v1",
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "fallback_rate_delta": 0.0,
            "applied_error_rate": 0.0,
            "holdout_error_rate": 0.0,
            "applied_retry_rate": 0.0,
            "holdout_retry_rate": 0.0,
            "applied_fallback_rate": 0.0,
            "holdout_fallback_rate": 0.0,
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "raw_response_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "managed_server_calls_made": False,
            "provider_calls_made": False,
            "policy_files_written": False,
        },
        "_applied_errors": 0,
        "_holdout_errors": 0,
        "_applied_retries": 0,
        "_holdout_retries": 0,
        "_applied_fallbacks": 0,
        "_holdout_fallbacks": 0,
    }


def build_routing_experiment_lifecycle_outcomes(
    store_obj: Any,
    *,
    limit: int = 50,
    since: str | None = None,
    window_hours: float | None = 168.0,
) -> dict[str, Any]:
    """Summarize local lifecycle evidence for dashboard-added routing candidates."""
    dashboard_candidates = [
        dict(candidate)
        for candidate in _all_routing_candidates()
        if isinstance(candidate, dict) and _is_dashboard_routing_candidate(candidate)
    ]
    capped = max(1, min(int(limit or 1), 1000))
    buckets = {
        _routing_candidate_fingerprint(candidate): _new_lifecycle_bucket(candidate)
        for candidate in dashboard_candidates
    }
    cutoff = _since_cutoff_iso(since=since, window_hours=window_hours)
    params: list[Any] = []
    where = ""
    if cutoff:
        where = "where created_at >= ?"
        params.append(cutoff)
    try:
        rows = store_obj.conn.execute(
            f"""
            select id, created_at,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   coalesce(stream, 0) as stream,
                   requested_model,
                   routed_model,
                   coalesce(category, 'unknown') as category,
                   status_code,
                   latency_ms,
                   input_tokens_est,
                   actual_input_tokens,
                   cost_est_usd,
                   cost_baseline_usd,
                   retry_count,
                   routing_outcome_label,
                   error,
                   routing_json
            from calls
            {where}
            order by created_at desc
            limit 50000
            """,
            tuple(params),
        ).fetchall()
    except Exception:
        rows = []

    for raw_row in rows:
        row = dict(raw_row)
        routing = _parse_jsonish(row.get("routing_json"))
        experiment = routing.get("routing_experiment") if isinstance(routing.get("routing_experiment"), dict) else {}
        if row.get("routing_outcome_label") not in (None, ""):
            routing["routing_outcome_label"] = row.get("routing_outcome_label")
        for candidate in dashboard_candidates:
            if not _routing_row_matches_candidate(candidate, row, routing=routing, experiment=experiment):
                continue
            bucket = buckets[_routing_candidate_fingerprint(candidate)]
            lifecycle = _routing_lifecycle_status(candidate=candidate, row=row, routing=routing, experiment=experiment)
            bucket["matched_count"] += 1
            bucket["observed_count"] += 1
            bucket["status"] = "observed"
            if lifecycle == "canary_applied":
                bucket["applied_count"] += 1
                bucket["cohort_counts"]["canary_applied"] += 1
            elif lifecycle == "canary_holdout":
                bucket["holdout_count"] += 1
                bucket["cohort_counts"]["canary_holdout"] += 1
            elif lifecycle == "safety_stopped":
                bucket["safety_stop_count"] += 1
                bucket["cohort_counts"]["safety_stopped"] += 1
            elif lifecycle == "rollback":
                bucket["rollback_count"] += 1
                bucket["cohort_counts"]["rollback"] += 1
            elif lifecycle == "skipped":
                bucket["skipped_count"] += 1
                bucket["cohort_counts"]["skipped"] += 1
            else:
                bucket["unknown_count"] += 1
                bucket["cohort_counts"]["unknown"] += 1
            status_code = _as_non_negative_int(row.get("status_code"))
            retry_count = _as_non_negative_int(row.get("retry_count")) or 0
            has_error = bool(row.get("error")) or (status_code is not None and status_code >= 400)
            has_fallback = bool(routing.get("fallback_reason") or experiment.get("fallback_reason"))
            if has_error:
                bucket["error_count"] += 1
            if retry_count:
                bucket["retry_count"] += retry_count
            if has_fallback:
                bucket["fallback_count"] += 1
            if lifecycle == "canary_applied":
                bucket["_applied_errors"] += 1 if has_error else 0
                bucket["_applied_retries"] += retry_count
                bucket["_applied_fallbacks"] += 1 if has_fallback else 0
                baseline = float(row.get("cost_baseline_usd") or 0.0)
                actual = float(row.get("cost_est_usd") or 0.0)
                if baseline > actual:
                    bucket["observed_saved_usd"] += baseline - actual
            elif lifecycle == "canary_holdout":
                bucket["_holdout_errors"] += 1 if has_error else 0
                bucket["_holdout_retries"] += retry_count
                bucket["_holdout_fallbacks"] += 1 if has_fallback else 0
            if bucket["latest_observed_at"] is None or str(row.get("created_at")) > str(bucket["latest_observed_at"]):
                bucket["latest_observed_at"] = row.get("created_at")

    try:
        experiment_rows = store_obj.conn.execute(
            f"""
            select created_at,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(source_surface, 'anthropic_messages') as source_surface,
                   coalesce(stream, 0) as stream,
                   requested_model,
                   routed_model,
                   coalesce(category, 'unknown') as category,
                   primary_cost_est_usd,
                   shadow_cost_est_usd,
                   experiment_json,
                   routing_json
            from routing_experiments
            {where}
            order by created_at desc
            limit 50000
            """,
            tuple(params),
        ).fetchall()
    except Exception:
        experiment_rows = []
    for raw_row in experiment_rows:
        row = dict(raw_row)
        experiment = _parse_jsonish(row.get("experiment_json"))
        routing = _parse_jsonish(row.get("routing_json"))
        for candidate in dashboard_candidates:
            if not _routing_row_matches_candidate(candidate, row, routing=routing, experiment=experiment):
                continue
            bucket = buckets[_routing_candidate_fingerprint(candidate)]
            primary = float(row.get("primary_cost_est_usd") or 0.0)
            shadow = float(row.get("shadow_cost_est_usd") or 0.0)
            if primary > shadow:
                bucket["projected_saved_usd"] += primary - shadow
            if bucket["latest_observed_at"] is None or str(row.get("created_at")) > str(bucket["latest_observed_at"]):
                bucket["latest_observed_at"] = row.get("created_at")

    outcomes: list[dict[str, Any]] = []
    for bucket in buckets.values():
        matched = int(bucket["matched_count"])
        applied = int(bucket["applied_count"])
        holdout = int(bucket["holdout_count"])
        projected = round(float(bucket["projected_saved_usd"]), 6)
        observed = round(float(bucket["observed_saved_usd"]), 6)
        bucket["projected_saved_usd"] = projected
        bucket["observed_saved_usd"] = observed
        bucket["projected_savings_per_1000_calls_usd"] = round((projected / matched) * 1000, 6) if matched else 0.0
        bucket["observed_savings_per_1000_calls_usd"] = round((observed / applied) * 1000, 6) if applied else 0.0
        bucket["coverage"] = {
            **bucket["coverage"],
            "matched_count": matched,
            "applied_count": applied,
            "holdout_count": holdout,
            "has_applied_coverage": applied > 0,
            "has_holdout_coverage": holdout > 0,
        }
        applied_error_rate = bucket["_applied_errors"] / applied if applied else 0.0
        holdout_error_rate = bucket["_holdout_errors"] / holdout if holdout else 0.0
        applied_retry_rate = bucket["_applied_retries"] / applied if applied else 0.0
        holdout_retry_rate = bucket["_holdout_retries"] / holdout if holdout else 0.0
        applied_fallback_rate = bucket["_applied_fallbacks"] / applied if applied else 0.0
        holdout_fallback_rate = bucket["_holdout_fallbacks"] / holdout if holdout else 0.0
        bucket["regression_deltas"] = {
            **bucket["regression_deltas"],
            "error_rate_delta": round(applied_error_rate - holdout_error_rate, 6),
            "retry_rate_delta": round(applied_retry_rate - holdout_retry_rate, 6),
            "fallback_rate_delta": round(applied_fallback_rate - holdout_fallback_rate, 6),
            "applied_error_rate": round(applied_error_rate, 6),
            "holdout_error_rate": round(holdout_error_rate, 6),
            "applied_retry_rate": round(applied_retry_rate, 6),
            "holdout_retry_rate": round(holdout_retry_rate, 6),
            "applied_fallback_rate": round(applied_fallback_rate, 6),
            "holdout_fallback_rate": round(holdout_fallback_rate, 6),
        }
        age = _age_hours(bucket.get("latest_observed_at"))
        stale = age is None or age > ROUTING_PROMOTION_FRESHNESS_MAX_AGE_HOURS
        bucket["freshness"] = {
            **bucket["freshness"],
            "latest_observed_age_hours": round(age, 3) if age is not None else None,
            "stale": bool(stale),
            "reason": "stale-lifecycle-evidence" if stale and age is not None else "fresh" if not stale else "no-local-traffic",
        }
        if applied > 0 and holdout > 0 and not bucket["safety_stop_count"] and not bucket["rollback_count"]:
            bucket["status"] = "coverage-ready"
        elif matched > 0:
            bucket["status"] = "insufficient-lifecycle-coverage"
        for key in list(bucket):
            if key.startswith("_"):
                bucket.pop(key, None)
        outcomes.append(bucket)
    outcomes.sort(
        key=lambda item: (
            int(item.get("applied_count") or 0) + int(item.get("holdout_count") or 0),
            str(item.get("latest_observed_at") or ""),
        ),
        reverse=True,
    )
    outcomes = outcomes[:capped]
    status_counts: dict[str, int] = {}
    for outcome in outcomes:
        _increment_count(status_counts, outcome.get("status"), fallback="unknown")
    return {
        "schema": "tokenclaw.routing_experiment_lifecycle_outcomes.v1",
        "generated_at": utc_now(),
        "status": "tracked" if dashboard_candidates else "no-dashboard-routing-candidates",
        "candidate_source": "dashboard-recent-call",
        "dashboard_candidate_count": len(dashboard_candidates),
        "outcome_count": len(outcomes),
        "summary": {
            "matched_count": sum(int(item.get("matched_count") or 0) for item in outcomes),
            "applied_count": sum(int(item.get("applied_count") or 0) for item in outcomes),
            "holdout_count": sum(int(item.get("holdout_count") or 0) for item in outcomes),
            "safety_stop_count": sum(int(item.get("safety_stop_count") or 0) for item in outcomes),
            "rollback_count": sum(int(item.get("rollback_count") or 0) for item in outcomes),
            "observed_saved_usd": round(sum(float(item.get("observed_saved_usd") or 0.0) for item in outcomes), 6),
            "projected_saved_usd": round(sum(float(item.get("projected_saved_usd") or 0.0) for item in outcomes), 6),
            "status_counts": _count_rows(status_counts, key_name="status"),
        },
        "outcomes": outcomes,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "raw_response_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "managed_server_calls_made": False,
            "provider_calls_made": False,
            "policy_files_written": False,
        },
    }


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
        "schema": "tokenclaw.post_fix_shadow_yield.v1",
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
        "schema": "tokenclaw.claude_shadow_routing_yield.v1",
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
    # Score promotion evidence over a recent window so pre-fix shadow failures age
    # out instead of permanently poisoning a candidate's 400-count and pass-rate.
    promotion_cutoff = (
        _since_cutoff_iso(window_hours=ROUTING_PROMOTION_EVIDENCE_WINDOW_HOURS)
        if ROUTING_PROMOTION_EVIDENCE_WINDOW_HOURS and ROUTING_PROMOTION_EVIDENCE_WINDOW_HOURS > 0
        else None
    )
    rows_select = """
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
               coalesce(shadow_routed_cost_est_usd, shadow_cost_est_usd) as shadow_routed_cost_est_usd,
               error,
               routing_json,
               experiment_json
        from routing_experiments
        {where}
        order by created_at desc
        limit 50000
    """
    if promotion_cutoff:
        rows = conn.execute(rows_select.format(where="where created_at >= ?"), (promotion_cutoff,)).fetchall()
    else:
        rows = conn.execute(rows_select.format(where=""), ()).fetchall()
    grouped: dict[tuple[str, str, bool, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        experiment = _parse_jsonish(row["experiment_json"])
        routing = _parse_jsonish(row["routing_json"])
        mode = _sample_mode_from_experiment(experiment)
        workflow_phase = _workflow_phase_from_payloads(experiment, routing)
        candidate_id = _public_label(
            experiment.get("candidate_id") or (experiment.get("selected_candidate") or {}).get("candidate_id"),
            fallback="",
        )
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
                "candidate_id": candidate_id,
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
                "shadow_routed_cost_usd": 0.0,
                "similarities": [],
                "passed": [],
                "latency_deltas": [],
                "last_sample_at": None,
            },
        )
        if candidate_id and not item.get("candidate_id"):
            item["candidate_id"] = candidate_id
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
        item["shadow_routed_cost_usd"] += float(
            row["shadow_routed_cost_est_usd"] if row["shadow_routed_cost_est_usd"] is not None else (row["shadow_cost_est_usd"] or 0.0)
        )
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
        item["shadow_routed_cost_usd"] = round(float(item["shadow_routed_cost_usd"]), 6)
        # Routing economics use the counterfactual routed cost (what the cheaper model
        # would cost on the same cached token profile), not the uncached shadow probe.
        item["cost_delta_usd"] = round(float(item["primary_cost_usd"]) - float(item["shadow_routed_cost_usd"]), 6)
        item["shadow_probe_cost_usd"] = item["shadow_cost_usd"]
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
               round(sum(coalesce(primary_cost_est_usd, 0) - coalesce(shadow_routed_cost_est_usd, shadow_cost_est_usd, 0)), 6) as cost_delta_usd,
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
    readiness_scoreboard = _build_routing_candidate_readiness_scoreboard(candidates)
    lifecycle_outcomes = build_routing_experiment_lifecycle_outcomes(
        store_obj,
        limit=limit,
        since=since,
        window_hours=window_hours if window_hours is not None else 168.0,
    )
    return {
        "schema": "tokenclaw.routing_experiment_report.v1",
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
            "routing_candidates": list(ROUTING_EXPERIMENT_POLICY.get("routing_candidates") or []),
            "routing_candidate_count": len(ROUTING_EXPERIMENT_POLICY.get("routing_candidates") or []),
            "categories": list(ROUTING_EXPERIMENT_POLICY.get("categories") or []),
            "workflow_phases": list(ROUTING_EXPERIMENT_POLICY.get("workflow_phases") or []),
            "min_samples_for_confidence": ROUTING_EXPERIMENT_MIN_SAMPLES,
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
            "readiness_scoreboard_counts": {
                item["status"]: item["count"]
                for item in readiness_scoreboard.get("readiness_counts", [])
                if isinstance(item, dict)
            },
            "routing_readiness_ready_count": readiness_scoreboard.get("summary", {}).get("ready_count", 0),
            "routing_readiness_insufficient_evidence_count": readiness_scoreboard.get("summary", {}).get("insufficient_evidence_count", 0),
            "routing_readiness_regressing_count": readiness_scoreboard.get("summary", {}).get("regressing_count", 0),
            "routing_lifecycle_outcome_count": lifecycle_outcomes.get("outcome_count", 0),
            "routing_lifecycle_applied_count": lifecycle_outcomes.get("summary", {}).get("applied_count", 0),
            "routing_lifecycle_holdout_count": lifecycle_outcomes.get("summary", {}).get("holdout_count", 0),
        },
        "decision_reasons": decision_reasons,
        "decision_surfaces": decision_surfaces,
        "eligibility_projection": eligibility_projection,
        "claude_shadow_yield": claude_shadow_yield,
        "post_fix_shadow_yield": post_fix_shadow_yield,
        "readiness_scoreboard": readiness_scoreboard,
        "lifecycle_outcomes": lifecycle_outcomes,
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
