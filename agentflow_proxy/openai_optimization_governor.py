from __future__ import annotations

import hashlib
import os
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint


SCHEMA = "agentflow.openai_optimization_governor.v1"
FAMILIES = ("routing", "old_context_summary", "cache_replay")
PRIORITY = ("routing", "old_context_summary", "cache_replay")


def _float_0_1(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _text_bucket(chars: Any) -> str:
    try:
        value = int(chars or 0)
    except (TypeError, ValueError):
        value = 0
    if value < 1500:
        return "lt-1_5k"
    if value < 8000:
        return "1_5k-8k"
    if value < 30000:
        return "8k-30k"
    return "gte-30k"


def _hash_value(parts: list[str]) -> tuple[str, float]:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return "sha256:" + digest, score


def _policy_source(*metas: dict[str, Any]) -> str:
    for meta in metas:
        source = meta.get("policy_source")
        if source in {"managed-recommended", "managed-enforced"}:
            return str(source)
    for meta in metas:
        source = meta.get("policy_source")
        if source in {"local-manual", "local-default"}:
            return str(source)
    return "local-default"


def _routing_decision(routing_meta: dict[str, Any]) -> dict[str, Any]:
    canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
    requested = str(routing_meta.get("requested_model") or canary.get("requested_model") or "")
    routed = str(routing_meta.get("routed_model") or canary.get("actual_forwarded_model") or requested)
    status = str(canary.get("status") or "")
    reason = str(canary.get("reason") or routing_meta.get("reason") or "")
    eligible = False
    selected = False
    suppressed: list[str] = []
    if canary:
        eligible = bool(canary.get("enabled")) and status not in {"disabled", "ineligible", "noop"}
        selected = status == "applied" and routed and requested and routed != requested
        if status in {"holdout", "not_selected"}:
            suppressed.append("missing-holdout" if status == "holdout" else "canary-not-selected")
        elif status in {"safety_stopped", "safety-stop-tripped"}:
            suppressed.append("stale-evidence")
        elif status == "ineligible":
            if reason in {"streaming-not-enabled", "unsupported-streaming-shape"}:
                suppressed.append("streaming-unsupported")
            else:
                suppressed.append(reason or "ineligible")
    elif routing_meta.get("enabled"):
        eligible = requested != "" and routed != "" and requested != routed
        selected = eligible
    return {
        "family": "routing",
        "eligible": eligible,
        "selected": selected,
        "status": status or ("applied" if selected else "not_selected"),
        "reason": reason,
        "policy_source": _policy_source(canary, routing_meta),
        "suppression_reasons": sorted(set(suppressed)),
    }


def _summary_decision(crunch_meta: dict[str, Any], summary_meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = summary_meta
    if not isinstance(meta, dict):
        existing = crunch_meta.get("old_context_summarization")
        meta = existing if isinstance(existing, dict) else {}
    status = str(meta.get("status") or "")
    reason_codes = [str(item) for item in meta.get("reason_codes") or []]
    enabled = bool(meta.get("enabled")) if "enabled" in meta else bool(meta)
    selected = bool(meta.get("applied")) or status == "applied"
    eligible = selected or status in {"holdout", "skipped", "not_evaluated"} or (enabled and status != "disabled")
    suppressed: list[str] = []
    if status == "holdout" or "holdout" in reason_codes:
        suppressed.append("missing-holdout")
    if any(code in reason_codes for code in ("summary_fetch_error", "summary_empty_or_malformed")):
        suppressed.append("summary-provider-unavailable")
    if "unsupported-streaming-shape" in reason_codes:
        suppressed.append("streaming-unsupported")
    return {
        "family": "old_context_summary",
        "eligible": eligible,
        "selected": selected,
        "status": status or ("applied" if selected else "not_evaluated"),
        "reason": ",".join(reason_codes),
        "policy_source": _policy_source(meta),
        "suppression_reasons": sorted(set(suppressed or reason_codes)),
    }


def _cache_decision(cache_meta: dict[str, Any]) -> dict[str, Any]:
    replay_canary = (
        cache_meta.get("cache_replay_canary")
        if isinstance(cache_meta.get("cache_replay_canary"), dict)
        else {}
    )
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    status = str(cache_meta.get("status") or "")
    reason = str(cache_meta.get("reason") or replay_canary.get("reason") or "")
    eligible = bool(pattern_rule or replay_canary or status == "hit")
    selected = status == "hit" and reason != "conflicts-with-selected-family"
    suppressed: list[str] = []
    replay_status = str(replay_canary.get("status") or "")
    if replay_status == "holdout":
        suppressed.append("missing-holdout")
    if reason in {"streaming", "unsupported-streaming-shape"}:
        suppressed.append("streaming-unsupported")
    if reason in {
        "file-dependency-missing",
        "dependency-cap-exceeded",
        "dependency-changed",
        "session-scope-missing",
        "cache-replay-invalidation-missing",
    }:
        suppressed.append("cache-replay-invalidation-missing")
    if reason == "conflicts-with-selected-family":
        suppressed.append("conflicts-with-selected-family")
    return {
        "family": "cache_replay",
        "eligible": eligible,
        "selected": selected,
        "status": status or replay_status or "not_evaluated",
        "reason": reason,
        "policy_source": _policy_source(replay_canary, pattern_rule, cache_meta),
        "suppression_reasons": sorted(set(suppressed)),
    }


def _governor_canary(
    *,
    endpoint: str,
    requested_model: str,
    category: str | None,
    text_chars: Any,
    has_tools: Any,
    stream: Any,
    session_id: str | None,
) -> dict[str, Any]:
    canary_fraction = _float_0_1(
        os.getenv("AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_CANARY_FRACTION"),
        1.0,
    )
    holdout_fraction = _float_0_1(
        os.getenv("AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_HOLDOUT_FRACTION"),
        0.0,
    )
    salt = os.getenv("AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_SALT", "agentflow-openai-optimization-governor-v1")
    session_hash = ""
    if session_id:
        session_hash, _ = _hash_value([salt, "session", session_id])
    cohort_key_hash, score = _hash_value([
        salt,
        endpoint,
        requested_model,
        str(category or "unknown"),
        _text_bucket(text_chars),
        "tools" if bool(has_tools) else "no-tools",
        "stream" if bool(stream) else "nonstream",
        session_hash,
    ])
    if score < holdout_fraction:
        cohort = "governor_holdout"
        selected = False
        holdout = True
    elif score < holdout_fraction + canary_fraction:
        cohort = "canary_applied"
        selected = True
        holdout = False
    else:
        cohort = "not_selected"
        selected = False
        holdout = False
    return {
        "cohort": cohort,
        "selected": selected,
        "holdout": holdout,
        "cohort_key_hash": cohort_key_hash,
        "cohort_score": round(score, 12),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "cohort_basis": "metadata-hash",
        "raw_session_id_included": False,
        "salt_included": False,
    }


def build_openai_optimization_governor(
    *,
    routing_meta: dict[str, Any] | None,
    crunch_meta: dict[str, Any] | None,
    cache_meta: dict[str, Any] | None,
    summary_meta: dict[str, Any] | None = None,
    path: str = "/v1/responses",
    requested_model: str | None = None,
    category: str | None = None,
    stream: bool = False,
    session_id: str | None = None,
    compatible_families: list[list[str]] | None = None,
) -> dict[str, Any]:
    routing = routing_meta if isinstance(routing_meta, dict) else {}
    crunch = crunch_meta if isinstance(crunch_meta, dict) else {}
    cache = cache_meta if isinstance(cache_meta, dict) else {}
    endpoint = openai_endpoint(path)
    requested = str(requested_model or routing.get("requested_model") or "")
    category_value = category if category is not None else routing.get("category")
    decisions = {
        "routing": _routing_decision(routing),
        "old_context_summary": _summary_decision(crunch, summary_meta),
        "cache_replay": _cache_decision(cache),
    }
    canary = _governor_canary(
        endpoint=endpoint,
        requested_model=requested,
        category=str(category_value or "unknown"),
        text_chars=routing.get("text_chars"),
        has_tools=routing.get("has_tools"),
        stream=stream or bool(routing.get("stream")),
        session_id=session_id,
    )
    selected = "none"
    if canary["selected"]:
        for family in PRIORITY:
            if decisions[family]["selected"]:
                selected = family
                break
    suppressed: list[dict[str, Any]] = []
    compatible_sets = [set(item) for item in compatible_families or []]
    for family, decision in decisions.items():
        reasons = list(decision.get("suppression_reasons") or [])
        if family != selected and selected != "none" and decision.get("eligible"):
            compatible = any({family, selected}.issubset(items) for items in compatible_sets)
            if not compatible:
                reasons.append("conflicts-with-selected-family")
        if not canary["selected"] and decision.get("eligible"):
            reasons.append("missing-holdout" if canary["holdout"] else "governor-canary-not-selected")
        if reasons:
            suppressed.append({
                "family": family,
                "reason_codes": sorted(set(str(item) for item in reasons if item)),
                "status": decision.get("status"),
            })

    eligible = [family for family, decision in decisions.items() if decision.get("eligible")]
    source = _policy_source(*(decisions[family] for family in FAMILIES))
    if selected != "none":
        source = str(decisions[selected].get("policy_source") or source)
    return {
        "schema": SCHEMA,
        "provider": "openai",
        "endpoint": endpoint,
        "eligible_action_families": eligible,
        "selected_action_family": selected,
        "selected_action_families": [] if selected == "none" else [selected],
        "suppressed_families": suppressed,
        "family_status": {
            family: {
                "eligible": bool(decisions[family].get("eligible")),
                "selected": family == selected,
                "status": decisions[family].get("status"),
                "reason": decisions[family].get("reason"),
                "policy_source": decisions[family].get("policy_source"),
            }
            for family in FAMILIES
        },
        "canary": canary,
        "source": source,
        "policy_source": source,
        "conservative_single_mutation": True,
        "privacy": {
            "raw_prompt_included": False,
            "raw_response_included": False,
            "raw_request_body_included": False,
            "request_id_included": False,
            "session_id_included": False,
            "file_paths_included": False,
            "cache_key_included": False,
            "provider_body_included": False,
        },
    }


def attach_openai_optimization_governor(
    *,
    routing_meta: dict[str, Any] | None,
    crunch_meta: dict[str, Any] | None,
    cache_meta: dict[str, Any] | None,
    summary_meta: dict[str, Any] | None = None,
    path: str = "/v1/responses",
    requested_model: str | None = None,
    category: str | None = None,
    stream: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    governor = build_openai_optimization_governor(
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
        summary_meta=summary_meta,
        path=path,
        requested_model=requested_model,
        category=category,
        stream=stream,
        session_id=session_id,
    )
    for meta in (routing_meta, crunch_meta, cache_meta):
        if isinstance(meta, dict):
            meta["openai_optimization_governor"] = governor
    return governor


def selected_openai_governor_family(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return "none"
    governor = meta.get("openai_optimization_governor")
    if not isinstance(governor, dict):
        return "none"
    return str(governor.get("selected_action_family") or "none")
