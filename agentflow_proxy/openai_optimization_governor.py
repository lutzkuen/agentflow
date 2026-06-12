from __future__ import annotations

import hashlib
import os
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint


SCHEMA = "agentflow.openai_optimization_governor.v1"
LIFECYCLE_SCHEMA = "agentflow.openai_optimization_lifecycle_feedback.v1"
LIFECYCLE_SOURCE_SURFACE = "openai_optimization_lifecycle"
FAMILIES = ("routing", "old_context_summary", "cache_replay")
PRIORITY = ("routing", "old_context_summary", "cache_replay")
RAW_IDENTIFIER_TERMS = {
    "apikey",
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "content",
    "file",
    "message",
    "path",
    "payload",
    "prompt",
    "request",
    "response",
    "secret",
    "session",
    "summary_text",
    "tenant",
    "tool",
    "transcript",
}


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


def _hash_identifier(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _public_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    lowered = text.lower()
    if (
        len(text) > 128
        or any(char.isspace() for char in text)
        or any(char in text for char in ("/", "\\", "{", "}", "[", "]", "\"", "'"))
        or any(term in lowered for term in RAW_IDENTIFIER_TERMS)
    ):
        return _hash_identifier(text)
    return text


def _reason_code(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    safe = all(char.isalnum() or char in {"-", ".", ":"} for char in text)
    if len(text) > 96 or not safe or any(term in text for term in RAW_IDENTIFIER_TERMS):
        return _hash_identifier(text)
    return text


def _reason_codes(*values: Any) -> list[str]:
    codes: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                code = _reason_code(item)
                if code:
                    codes.append(code)
        else:
            code = _reason_code(value)
            if code:
                codes.append(code)
    return sorted(set(codes))


def _status_code_bucket(value: Any) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if code < 200:
        return "lt_2xx"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _retry_bucket(value: Any) -> str:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 3:
        return "two_three"
    return "gte_4"


def _latency_bucket(value: Any) -> str:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if ms < 500:
        return "lt_500ms"
    if ms < 2_000:
        return "500ms_2s"
    if ms < 10_000:
        return "2s_10s"
    if ms < 30_000:
        return "10s_30s"
    return "gte_30s"


def _money(value: Any) -> float | None:
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


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


def _is_managed_policy_source(value: Any) -> bool:
    return value in {"managed-recommended", "managed-enforced"}


def _has_managed_evidence(meta: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, dict) and value:
            return True
    return False


def _routing_decision(routing_meta: dict[str, Any]) -> dict[str, Any]:
    canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
    requested = str(routing_meta.get("requested_model") or canary.get("requested_model") or "")
    routed = str(routing_meta.get("routed_model") or canary.get("actual_forwarded_model") or requested)
    status = str(canary.get("status") or "")
    reason = str(canary.get("reason") or routing_meta.get("reason") or "")
    policy_source = _policy_source(canary, routing_meta)
    managed_missing_evidence = _is_managed_policy_source(policy_source) and not canary
    eligible = False
    selected = False
    suppressed: list[str] = []
    if managed_missing_evidence:
        eligible = requested != "" and routed != "" and requested != routed
        selected = False
        suppressed.append("missing-canary-evidence")
    elif canary:
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
        "policy_source": policy_source,
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
    policy_source = _policy_source(meta)
    managed_missing_evidence = (
        _is_managed_policy_source(policy_source)
        and (bool(meta.get("applied")) or status == "applied")
        and not _has_managed_evidence(meta, "canary", "impact_evidence", "quality_gate")
    )
    selected = (bool(meta.get("applied")) or status == "applied") and not managed_missing_evidence
    eligible = selected or status in {"holdout", "skipped", "not_evaluated"} or (enabled and status != "disabled")
    suppressed: list[str] = []
    if managed_missing_evidence:
        suppressed.append("missing-canary-evidence")
    if status == "holdout" or "holdout" in reason_codes:
        suppressed.append("missing-holdout")
    if status in {"safety_stopped", "safety-stop-tripped"} or "stale-evidence" in reason_codes:
        suppressed.append("stale-evidence")
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
        "policy_source": policy_source,
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
    policy_source = _policy_source(replay_canary, pattern_rule, cache_meta)
    managed_missing_evidence = (
        _is_managed_policy_source(policy_source)
        and status == "hit"
        and not _has_managed_evidence(cache_meta, "cache_replay_canary", "pattern_rule")
    )
    eligible = bool(pattern_rule or replay_canary or status == "hit")
    selected = status == "hit" and reason != "conflicts-with-selected-family" and not managed_missing_evidence
    suppressed: list[str] = []
    replay_status = str(replay_canary.get("status") or "")
    if managed_missing_evidence:
        suppressed.append("missing-canary-evidence")
    if replay_status == "holdout":
        suppressed.append("missing-holdout")
    if replay_status in {"safety_stopped", "safety-stop-tripped"}:
        suppressed.append("stale-evidence")
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
        "policy_source": policy_source,
        "suppression_reasons": sorted(set(suppressed)),
    }


def _summary_meta(crunch_meta: dict[str, Any], summary_meta: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(summary_meta, dict):
        return summary_meta
    existing = crunch_meta.get("old_context_summarization")
    return existing if isinstance(existing, dict) else {}


def _cache_replay_meta(cache_meta: dict[str, Any]) -> dict[str, Any]:
    replay = cache_meta.get("cache_replay_canary")
    if isinstance(replay, dict):
        return replay
    rule = cache_meta.get("pattern_rule")
    if isinstance(rule, dict):
        return rule
    return {}


def _family_identifier_meta(
    family: str,
    *,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    summary_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if family == "routing":
        canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
        return {
            "policy_id": _public_identifier(canary.get("policy_id") or routing_meta.get("managed_policy_id")),
            "rule_id": _public_identifier(canary.get("rule_id") or canary.get("policy_id")),
            "candidate_id": _public_identifier(
                canary.get("candidate_id")
                or canary.get("target_candidate_id")
                or canary.get("promotion_action_id")
            ),
            "action_id": _public_identifier(canary.get("promotion_action_id")),
        }
    if family == "old_context_summary":
        meta = _summary_meta(crunch_meta, summary_meta)
        return {
            "policy_id": _public_identifier(meta.get("policy_id") or meta.get("rule_id")),
            "rule_id": _public_identifier(meta.get("rule_id")),
            "candidate_id": _public_identifier(meta.get("candidate_id") or meta.get("target_candidate_id")),
            "action_id": _public_identifier(meta.get("promotion_action_id")),
        }
    replay = _cache_replay_meta(cache_meta)
    return {
        "policy_id": _public_identifier(replay.get("policy_id") or replay.get("rule_id")),
        "rule_id": _public_identifier(replay.get("rule_id")),
        "candidate_id": _public_identifier(replay.get("candidate_id")),
        "action_id": _public_identifier(replay.get("promotion_action_id")),
    }


def _family_reason_codes(
    family: str,
    *,
    decision: dict[str, Any],
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    summary_meta: dict[str, Any] | None,
    suppressed: list[dict[str, Any]],
) -> list[str]:
    suppressed_codes: list[str] = []
    for item in suppressed:
        if item.get("family") == family:
            suppressed_codes.extend(str(code) for code in item.get("reason_codes") or [])
    if family == "routing":
        canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
        safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
        return _reason_codes(decision.get("reason"), canary.get("reason"), safety.get("reason_codes"), suppressed_codes)
    if family == "old_context_summary":
        meta = _summary_meta(crunch_meta, summary_meta)
        return _reason_codes(decision.get("reason"), meta.get("reason_codes"), suppressed_codes)
    replay = _cache_replay_meta(cache_meta)
    return _reason_codes(decision.get("reason"), replay.get("reason"), suppressed_codes)


def _family_cohort(
    family: str,
    *,
    selected: str,
    decision: dict[str, Any],
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    summary_meta: dict[str, Any] | None,
    reason_codes: list[str],
) -> str:
    status = str(decision.get("status") or "")
    if family == "routing":
        canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
        if canary.get("fallback_reason"):
            return "fallback"
        if canary.get("cohort") == "canary_holdout" or status == "holdout":
            return "holdout"
        safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
        if status in {"safety_stopped", "safety-stop-tripped"} or safety.get("tripped"):
            return "safety_stop"
    elif family == "old_context_summary":
        meta = _summary_meta(crunch_meta, summary_meta)
        canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
        if canary.get("cohort") == "canary_holdout" or status == "holdout":
            return "holdout"
        if status in {"safety_stopped", "safety-stop-tripped"}:
            return "safety_stop"
    else:
        replay = _cache_replay_meta(cache_meta)
        if replay.get("canary_cohort") == "canary_holdout" or status == "holdout":
            return "holdout"
        if status == "invalidated":
            return "invalidated"
        if status in {"safety_stopped", "safety-stop-tripped"}:
            return "safety_stop"
    if "cache-replay-invalidation-missing" in reason_codes or status == "invalidated":
        return "invalidated"
    if any(code.startswith("safety") or code == "stale-evidence" for code in reason_codes):
        return "safety_stop"
    if family == selected and selected != "none" and status in {"applied", "hit"}:
        return "applied"
    if decision.get("eligible"):
        return "suppressed"
    return "not_eligible"


def build_openai_optimization_lifecycle_event(
    *,
    routing_meta: dict[str, Any] | None,
    crunch_meta: dict[str, Any] | None,
    cache_meta: dict[str, Any] | None,
    summary_meta: dict[str, Any] | None = None,
    path: str = "/v1/responses",
    requested_model: str | None = None,
    routed_model: str | None = None,
    status_code: int | None = None,
    latency_ms: int | None = None,
    retry_count: int | None = None,
    cost_est_usd: float | None = None,
    cost_baseline_usd: float | None = None,
    category: str | None = None,
    stream: bool = False,
    call_id: str | None = None,
) -> dict[str, Any] | None:
    routing = routing_meta if isinstance(routing_meta, dict) else {}
    governor = routing.get("openai_optimization_governor")
    if not isinstance(governor, dict) or governor.get("schema") != SCHEMA:
        return None
    crunch = crunch_meta if isinstance(crunch_meta, dict) else {}
    cache = cache_meta if isinstance(cache_meta, dict) else {}
    family_status = governor.get("family_status") if isinstance(governor.get("family_status"), dict) else {}
    suppressed = governor.get("suppressed_families") if isinstance(governor.get("suppressed_families"), list) else []
    selected = str(governor.get("selected_action_family") or "none")
    observed_savings = None
    if cost_baseline_usd is not None and cost_est_usd is not None:
        observed_savings = round(float(cost_baseline_usd) - float(cost_est_usd), 8)

    events: list[dict[str, Any]] = []
    for family in FAMILIES:
        decision = family_status.get(family) if isinstance(family_status.get(family), dict) else {}
        identifiers = {
            key: value
            for key, value in _family_identifier_meta(
                family,
                routing_meta=routing,
                crunch_meta=crunch,
                cache_meta=cache,
                summary_meta=summary_meta,
            ).items()
            if value
        }
        reasons = _family_reason_codes(
            family,
            decision=decision,
            routing_meta=routing,
            crunch_meta=crunch,
            cache_meta=cache,
            summary_meta=summary_meta,
            suppressed=suppressed,
        )
        cohort = _family_cohort(
            family,
            selected=selected,
            decision=decision,
            routing_meta=routing,
            crunch_meta=crunch,
            cache_meta=cache,
            summary_meta=summary_meta,
            reason_codes=reasons,
        )
        if cohort == "not_eligible" and not identifiers:
            continue
        events.append({
            "action_family": family,
            "cohort": cohort,
            "selected": family == selected,
            "eligible": bool(decision.get("eligible")),
            "status": _reason_code(decision.get("status")) or "unknown",
            "reason_codes": reasons,
            **identifiers,
        })
    if not events:
        return None
    canary = governor.get("canary") if isinstance(governor.get("canary"), dict) else {}
    cost_est = _money(cost_est_usd)
    baseline_est = _money(cost_baseline_usd)
    return {
        "schema": LIFECYCLE_SCHEMA,
        "event_type": "openai_optimization_lifecycle",
        "provider": "openai",
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "endpoint": openai_endpoint(path),
        "category": category or routing.get("category") or "unknown",
        "stream": bool(stream or routing.get("stream")),
        "requested_model_family": _public_identifier(requested_model or routing.get("requested_model")),
        "routed_model_family": _public_identifier(routed_model or routing.get("routed_model")),
        "selected_action_family": selected,
        "family_events": events,
        "status_bucket": _status_code_bucket(status_code),
        "retry_bucket": _retry_bucket(retry_count),
        "latency_bucket": _latency_bucket(latency_ms),
        "cost_estimate_usd": cost_est,
        "baseline_estimate_usd": baseline_est,
        "observed_savings_estimate_usd": observed_savings,
        "governor": {
            "schema": SCHEMA,
            "cohort": canary.get("cohort"),
            "cohort_key_hash": canary.get("cohort_key_hash"),
            "selected": bool(canary.get("selected")),
            "holdout": bool(canary.get("holdout")),
            "conservative_single_mutation": True,
        },
        "local_call_hash": _hash_identifier(call_id) if call_id else None,
        "privacy": {
            "telemetry_profile": "metadata-only",
            "raw_body_storage": False,
            "metadata_only": True,
            "aggregate_only": False,
            "raw_payload_included": False,
            "raw_prompt_included": False,
            "raw_response_included": False,
            "provider_body_included": False,
            "provider_forwarding": False,
            "cache_key_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }


def openai_optimization_lifecycle_public_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.openai_optimization_lifecycle_feedback_queue_meta.v1",
        "enabled": bool(meta.get("enabled")),
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "endpoint": meta.get("endpoint"),
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "queue_id": meta.get("queue_id"),
        "attempts": meta.get("attempts"),
        "status_code": meta.get("status_code"),
        "latency_ms": meta.get("latency_ms"),
        "payload_included": False,
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
    enforcement = meta.get("optimization_coordinator_enforcement")
    if isinstance(enforcement, dict) and enforcement.get("enabled") and enforcement.get("status") == "applied":
        return str(enforcement.get("selected_family") or "none")
    governor = meta.get("openai_optimization_governor")
    if not isinstance(governor, dict):
        return "none"
    return str(governor.get("selected_action_family") or "none")
