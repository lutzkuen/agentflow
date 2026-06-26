from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.cache_smoke import build_cache_smoke_diagnostic
from tokenclaw.codex_turn_policy import (
    CODEX_APP_SOURCE_SURFACE,
    canonical_source_surface,
    is_codex_turn_source_surface,
)
from tokenclaw.limiter import model_tier
from tokenclaw.pricing import codex_app_pricing_basis, estimate_cost
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now

CODEX_APP_PRICING_BASIS = codex_app_pricing_basis()
CODEX_APP_MODEL = str(CODEX_APP_PRICING_BASIS["model"])
CODEX_APP_COST_BASIS = str(CODEX_APP_PRICING_BASIS["cost_basis"])
CODEX_APP_PROCESSING_MODE = str(CODEX_APP_PRICING_BASIS["processing_mode"])
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


def _breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": counts[key]}
        for key in sorted(counts, key=lambda key: (-counts[key], key))
    ]


def _source_surface(provider: str, path: str) -> str:
    provider_l = (provider or "").lower()
    path_l = (path or "").lower()
    if provider_l in {"codex-app", "codex_app"}:
        return CODEX_APP_SOURCE_SURFACE
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        if "chat/completions" in path_l:
            return "openai_chat"
        return "openai_responses"
    return "unknown"


def _endpoint_label(provider: str, path: str) -> str:
    provider_l = (provider or "").lower()
    path_l = (path or "").lower()
    if provider_l in {"codex-app", "codex_app"}:
        return "codex_app_turn"
    if "chat/completions" in path_l:
        return "chat_completions"
    if "responses" in path_l:
        return "responses"
    if "messages" in path_l:
        return "messages"
    return "unknown"


def _app_family_for_call(provider: str, requested_model: Any, path: str) -> str:
    provider_l = (provider or "").lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in (path or "").lower():
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def estimate_tokens_from_text_chars(chars: Any) -> int:
    return max(0, round(_as_int(chars) / TOKEN_CHARS))


def _codex_turn_estimates(input_text_chars: Any, result_chars: Any) -> dict[str, Any]:
    input_tokens = estimate_tokens_from_text_chars(input_text_chars)
    output_tokens = estimate_tokens_from_text_chars(result_chars)
    cost = estimate_cost(
        CODEX_APP_MODEL,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=CODEX_APP_PROCESSING_MODE,
    )
    cost_known = cost is not None
    cost_value = float(cost) if cost_known else None
    return {
        "model": CODEX_APP_MODEL,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": input_tokens + output_tokens,
        "cost_est_usd": cost_value,
        "baseline_cost_est_usd": cost_value,
        "hard_floor_usd": cost_value,
        "cost_basis": CODEX_APP_COST_BASIS,
        "pricing_basis": CODEX_APP_PRICING_BASIS,
        "cost_known": cost_known,
        "cost_estimated": cost_known,
    }


def _codex_estimates_with_cache(input_text_chars: Any, result_chars: Any, cache: dict[str, Any]) -> dict[str, Any]:
    estimates = _codex_turn_estimates(input_text_chars, result_chars)
    if cache.get("status") == "hit":
        baseline = float(estimates["baseline_cost_est_usd"] or estimates["cost_est_usd"] or 0.0)
        estimates["cost_est_usd"] = 0.0
        estimates["hard_floor_usd"] = 0.0
        estimates["baseline_cost_est_usd"] = baseline
        estimates["cache_savings_usd"] = baseline
        estimates["cost_known"] = True
        estimates["cost_estimated"] = True
    else:
        estimates["cache_savings_usd"] = 0.0
    return estimates

def _legacy_cache_decision(row: dict[str, Any]) -> dict[str, str]:
    status_code = _as_int(row.get("status_code"))
    source_surface = canonical_source_surface(
        row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
    )
    if _as_int(row.get("cache_hit")):
        return {
            "status": "hit",
            "reason": "legacy-cache-hit",
            "hit_type": "exact",
            "policy_source": "legacy-inferred",
            "source_surface": source_surface,
            "outcome_bucket": "hit",
        }
    if _as_int(row.get("stream")):
        return {
            "status": "skipped",
            "reason": "legacy-streaming",
            "hit_type": "",
            "policy_source": "legacy-inferred",
            "source_surface": source_surface,
            "outcome_bucket": "disabled",
        }
    if status_code >= 400:
        return {
            "status": "skipped",
            "reason": "legacy-upstream-error",
            "hit_type": "",
            "policy_source": "legacy-inferred",
            "source_surface": source_surface,
            "outcome_bucket": "unsafe-skip",
        }
    return {
        "status": "missing",
        "reason": "legacy-unknown",
        "hit_type": "",
        "policy_source": "legacy-inferred",
        "source_surface": source_surface,
        "outcome_bucket": "unknown",
    }


def _cache_outcome_bucket(status: str, reason: str, explicit: Any = None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if reason in {"codex-app-cache-disabled", "cache-disabled", "streaming-cache-disabled"}:
        return "disabled"
    if status == "hit":
        return "hit"
    if status == "holdout" or reason in {"codex-app-cache-canary-holdout", "canary_holdout"}:
        return "holdout"
    if status == "unsafe-skip" or reason in {
        "action-like-params",
        "non-text-input",
        "unknown-param-shape",
        "terminal-interaction-text",
        "file-affecting-text",
        "unsafe-cached-envelope",
    }:
        return "unsafe-skip"
    if reason in {"dependency-changed", "dependency-deleted", "codex-cache-ttl-expired"}:
        return "invalidated"
    if reason in {"stale-risk-blockers", "file-dependency-missing", "dependency-missing", "dependency-cap-exceeded", "file-watch-disabled"}:
        return "stale-risk"
    if status == "miss":
        return "miss"
    return status or "unknown"


def _cache_decision_for_breakdown(row: dict[str, Any]) -> dict[str, str]:
    cache = _json_obj(row.get("cache_json"))
    if cache:
        policy_source = str(cache.get("policy_source") or "unknown")
        source_surface = canonical_source_surface(
            cache.get("surface") or row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        )
        if not cache.get("status") and not cache.get("reason"):
            legacy_hit_type = str(cache.get("hit_type") or "")
            if legacy_hit_type == "skip-streaming":
                return {
                    "status": "skipped",
                    "reason": "legacy-streaming",
                    "hit_type": "",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                    "outcome_bucket": "disabled",
                }
            if legacy_hit_type == "miss":
                return {
                    "status": "miss",
                    "reason": "legacy-exact-miss",
                    "hit_type": "",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                    "outcome_bucket": "miss",
                }
            if legacy_hit_type == "hit":
                return {
                    "status": "hit",
                    "reason": "legacy-cache-hit",
                    "hit_type": "exact",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                    "outcome_bucket": "hit",
                }
            return {
                "status": "missing",
                "reason": "legacy-partial-cache-json",
                "hit_type": legacy_hit_type,
                "policy_source": policy_source,
                "source_surface": source_surface,
                "outcome_bucket": "unknown",
            }
        status = str(cache.get("status") or "missing")
        reason = str(cache.get("reason") or "unknown")
        return {
            "status": status,
            "reason": reason,
            "hit_type": str(cache.get("hit_type") or ""),
            "policy_source": policy_source,
            "source_surface": source_surface,
            "outcome_bucket": _cache_outcome_bucket(status, reason, cache.get("outcome_bucket")),
        }
    return _legacy_cache_decision(row)


def _cache_decision_breakdown(rows: list[dict[str, Any]], *, today_only: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        decision = _cache_decision_for_breakdown(row)
        key = (
            decision["source_surface"],
            decision["status"],
            decision["reason"],
            decision["hit_type"],
            decision["policy_source"],
            decision["outcome_bucket"],
        )
        bucket = grouped.setdefault(
            key,
            {
                "source_surface": key[0],
                "status": key[1],
                "reason": key[2],
                "hit_type": key[3],
                "policy_source": key[4],
                "outcome_bucket": key[5],
                "count": 0,
            },
        )
        bucket["count"] += 1

    breakdown = list(grouped.values())
    breakdown.sort(key=lambda r: r["count"], reverse=True)
    return breakdown

_CACHE_BLOCKER_SCAN_LIMIT = 1000

_CACHE_BLOCKER_NEXT_ACTIONS = {
    "skipped-streaming": {
        "family": "stage-replay-policy",
        "label": "stage streaming-safe cache replay policy or accept streaming traffic as non-cacheable",
    },
    "skipped-tools": {
        "family": "collect-dependency-evidence",
        "label": "collect dependency evidence before enabling tool-call cache replay",
    },
    "disabled": {
        "family": "reload-cache-policy",
        "label": "enable or reload local cache policy",
    },
    "staged-policy-not-loaded": {
        "family": "reload-cache-policy",
        "label": "reload staged cache policy before expecting hits",
    },
    "dependency-invalidation-blocked": {
        "family": "collect-dependency-evidence",
        "label": "collect stable dependency evidence or accept invalidated cache replay",
    },
    "holdout-only": {
        "family": "promote-or-rebalance-canary",
        "label": "review holdout evidence before promoting cache replay",
    },
    "true-miss": {
        "family": "accept-non-repeatable-traffic",
        "label": "accept non-repeatable traffic or look for repeated request-shape cohorts",
    },
    "cache-hit-observed": {
        "family": "none",
        "label": "cache hits are already present",
    },
    "safety-stopped": {
        "family": "review-safety-stop",
        "label": "review safety stop or unsafe cache envelope metadata",
    },
    "unknown-cache-decision": {
        "family": "instrument-cache-decision",
        "label": "record explicit cache decision metadata on this surface",
    },
}

_CACHE_ACTION_CANDIDATE_TITLES = {
    "stage-replay-policy": "Stage streaming cache replay pattern rule for {cohort}",
    "collect-dependency-evidence": "Collect cache invalidation evidence for {cohort}",
    "reload-cache-policy": "Enable or reload local cache policy for {cohort}",
    "promote-or-rebalance-canary": "Review cache replay holdout evidence for {cohort}",
    "review-safety-stop": "Review cache replay safety stop for {cohort}",
    "instrument-cache-decision": "Instrument explicit cache decisions for {cohort}",
    "accept-non-repeatable-traffic": "Confirm non-repeatable cache cohort for {cohort}",
}

_CACHE_ACTION_CANDIDATE_LOCAL_ACTIONS = {
    "stage-replay-policy": "draft a local streaming-safe cache replay rule and verify it with a dry-run before activation",
    "collect-dependency-evidence": "collect file-watch and invalidation evidence before enabling tool-call cache replay",
    "reload-cache-policy": "run a local cache policy reload or exact-cache smoke check so eligible requests can reach cache lookup",
    "promote-or-rebalance-canary": "review applied versus holdout cache replay evidence before promoting or rebalancing the canary",
    "review-safety-stop": "keep replay disabled for the cohort until the safety-stop reason is reviewed and narrowed",
    "instrument-cache-decision": "add explicit cache status and reason metadata for this surface before planning activation",
    "accept-non-repeatable-traffic": "treat the cohort as a no-op unless a narrower repeated request-shape bucket appears",
}

_CACHE_ACTION_CANDIDATE_ACCEPTANCE = {
    "stage-replay-policy": "A dry-run reports projected streaming replay hits for this cohort while holdout traffic still bypasses unchanged.",
    "collect-dependency-evidence": "The next metadata window reports stable invalidation evidence or a smaller dependency blocker count for this cohort.",
    "reload-cache-policy": "A cache policy reload or smoke diagnostic removes the disabled/reload blocker for new rows in this cohort.",
    "promote-or-rebalance-canary": "Applied and holdout cache replay counts produce a promote, rebalance, or keep-holdout decision for this cohort.",
    "review-safety-stop": "The safety-stop remains an explicit no-op or is replaced by a narrower safe bypass reduction check.",
    "instrument-cache-decision": "New rows for this provider/surface include non-unknown cache status and reason metadata.",
    "accept-non-repeatable-traffic": "The report marks this as research-only/no-op unless repeated-shape evidence appears in a bounded metadata window.",
}

_CACHE_ACTION_CANDIDATE_SAVINGS_PATH = {
    "stage-replay-policy": "Recover hits from high-volume streaming requests once replay safety is proven locally.",
    "collect-dependency-evidence": "Remove the stale-cache risk that blocks tool or file-dependent cache replay.",
    "reload-cache-policy": "Let already-configured exact-cache or replay rules participate in lookup instead of being skipped.",
    "promote-or-rebalance-canary": "Convert holdout-only replay evidence into captured cache savings when quality gates pass.",
    "review-safety-stop": "Prevent unsafe replay while identifying the smallest blocker that can be safely reduced later.",
    "instrument-cache-decision": "Remove the observability gap that prevents cache activation work from targeting a real blocker.",
    "accept-non-repeatable-traffic": "Avoid spending activation work on traffic that has no repeatable local cache opportunity.",
}

_CACHE_ACTION_CANDIDATE_CONCRETE = {
    "stage-replay-policy",
    "collect-dependency-evidence",
    "reload-cache-policy",
    "promote-or-rebalance-canary",
    "review-safety-stop",
    "instrument-cache-decision",
}


def _endpoint_for_cache_ladder(row: dict[str, Any]) -> str:
    endpoint = row.get("endpoint")
    if endpoint:
        return public_label(endpoint, "unknown")
    path = str(row.get("path") or "")
    path_l = path.lower()
    if "chat/completions" in path_l:
        return "chat"
    if "responses" in path_l:
        return "responses"
    if "messages" in path_l:
        return "messages"
    if path_l.startswith("codex-app://"):
        return "turn_start"
    return "unknown"


def _cache_ladder_bool_label(value: Any, *, true_label: str, false_label: str) -> str:
    if value is True:
        return true_label
    if value is False:
        return false_label
    return "unknown"


def _cache_ladder_has_tools(cache: dict[str, Any], routing: dict[str, Any]) -> bool | None:
    for key in ("has_tools", "has_tool_blocks", "tool_use_present"):
        if key in cache:
            return bool(cache.get(key))
    for key in ("has_tools", "has_tool_blocks", "tool_use_present"):
        if key in routing:
            return bool(routing.get(key))
    return None


def _cache_ladder_category(row: dict[str, Any], cache: dict[str, Any], routing: dict[str, Any]) -> str:
    return public_label(row.get("category") or cache.get("category") or routing.get("category"), "unknown")


def _cache_ladder_phase(cache: dict[str, Any], routing: dict[str, Any]) -> str:
    feature = routing.get("openai_feature_unit") if isinstance(routing.get("openai_feature_unit"), dict) else {}
    for key in ("workflow_phase", "phase"):
        if routing.get(key):
            return public_label(routing.get(key), "unknown")
        if cache.get(key):
            return public_label(cache.get(key), "unknown")
        if feature.get(key):
            return public_label(feature.get(key), "unknown")
    managed_features = routing.get("managed_pattern_features") if isinstance(routing.get("managed_pattern_features"), dict) else {}
    return public_label(managed_features.get("workflow_phase") or managed_features.get("category"), "unknown")


def _cache_action_candidate_cohort(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("blocker_code") or "cache-replay"),
        str(row.get("provider") or "unknown"),
        str(row.get("source_surface") or "unknown"),
        str(row.get("endpoint") or "unknown"),
        str(row.get("category") or "unknown"),
        str(row.get("workflow_phase") or "unknown"),
    ]
    return " / ".join(part for part in parts if part and part != "unknown") or "cache-replay"


def _cache_action_candidate_from_ladder_row(row: dict[str, Any], *, rank: int) -> dict[str, Any] | None:
    action_family = str(row.get("next_action_family") or "instrument-cache-decision")
    if action_family == "none":
        return None
    cohort = _cache_action_candidate_cohort(row)
    concrete = action_family in _CACHE_ACTION_CANDIDATE_CONCRETE
    title_template = _CACHE_ACTION_CANDIDATE_TITLES.get(
        action_family,
        _CACHE_ACTION_CANDIDATE_TITLES["instrument-cache-decision"],
    )
    local_action = _CACHE_ACTION_CANDIDATE_LOCAL_ACTIONS.get(
        action_family,
        _CACHE_ACTION_CANDIDATE_LOCAL_ACTIONS["instrument-cache-decision"],
    )
    acceptance = _CACHE_ACTION_CANDIDATE_ACCEPTANCE.get(
        action_family,
        _CACHE_ACTION_CANDIDATE_ACCEPTANCE["instrument-cache-decision"],
    )
    savings_path = _CACHE_ACTION_CANDIDATE_SAVINGS_PATH.get(
        action_family,
        _CACHE_ACTION_CANDIDATE_SAVINGS_PATH["instrument-cache-decision"],
    )
    evidence = {
        "count": _as_int(row.get("count")),
        "blocker_code": row.get("blocker_code"),
        "provider": row.get("provider"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "category": row.get("category"),
        "workflow_phase": row.get("workflow_phase"),
        "stream_mode": row.get("stream_mode"),
        "tool_presence": row.get("tool_presence"),
        "replayability_level": row.get("replayability_level"),
        "cache_status": row.get("cache_status"),
        "cache_reason": row.get("cache_reason"),
    }
    return {
        "rank": rank,
        "title": title_template.format(cohort=cohort),
        "labels": ["backlog", "status:ready", "priority:p1", "core-feature", "correctness", "cache", "privacy"],
        "cohort": cohort,
        "local_action": {
            "family": action_family,
            "label": row.get("next_action_label") or local_action,
            "implementation_hint": local_action,
            "concrete": concrete,
            "activation_mode": "activation-candidate" if concrete else "research-only",
        },
        "rationale": (
            "A zero-hit cache window is dominated by this explicit skip/blocker bucket. "
            "The next issue should target this local action instead of restating the generic hit-rate warning."
        ),
        "implementation_approach": [
            "Use only the aggregate cache decision bucket dimensions in this candidate.",
            local_action,
            "Record the follow-up result in machine-readable local cache metadata.",
            "Do not inspect prompts, provider bodies, file paths, request IDs, session IDs, cache keys, pattern hashes, or candidate identifiers.",
        ],
        "acceptance_metric": acceptance,
        "expected_savings_path_or_bottleneck_removed": savings_path,
        "sequencing_notes": (
            "Sequence before broad cache activation so replay work targets the highest-volume explicit zero-hit blocker first."
        ),
        "evidence": evidence,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
            "candidate_identifiers_included": False,
        },
    }


def _cache_ladder_replayability(cache: dict[str, Any], routing: dict[str, Any]) -> str:
    feature = routing.get("openai_feature_unit") if isinstance(routing.get("openai_feature_unit"), dict) else {}
    raw = cache.get("replayability_level") or feature.get("replayability_level")
    return public_label(raw, "unknown")


def _cache_ladder_policy_reload_required(cache: dict[str, Any]) -> bool:
    if cache.get("policy_reload_required") is True or cache.get("reload_required") is True:
        return True
    policy_state = cache.get("policy_state") if isinstance(cache.get("policy_state"), dict) else {}
    if policy_state.get("reload_required") is True:
        return True
    pattern_rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), dict) else {}
    if pattern_rules.get("reload_required") is True or pattern_rules.get("policy_reload_required") is True:
        return True
    return False


def _cache_ladder_dependency_blocked(cache: dict[str, Any], reason: str, outcome_bucket: str) -> bool:
    if outcome_bucket in {"invalidated", "stale-risk"}:
        return True
    if reason.startswith("dependency-") or reason.startswith("file-dependency-"):
        return True
    if reason in {"stale-risk-blockers", "file-watch-disabled"}:
        return True
    audit = cache.get("file_dependency_audit") if isinstance(cache.get("file_dependency_audit"), dict) else {}
    if audit.get("invalidation_reason"):
        return True
    if audit and audit.get("safe_invalidation_evidence") is False:
        return True
    replay_canary = cache.get("cache_replay_canary") if isinstance(cache.get("cache_replay_canary"), dict) else {}
    decision = replay_canary.get("decision") if isinstance(replay_canary.get("decision"), dict) else {}
    return str(decision.get("status") or "") in {"invalidated", "bypassed"} and "dependency" in str(decision.get("reason") or "")


def _cache_blocker_code(row: dict[str, Any], decision: dict[str, str], cache: dict[str, Any]) -> str:
    status = public_label(decision.get("status"), "unknown")
    reason = public_label(decision.get("reason"), "unknown")
    outcome_bucket = public_label(decision.get("outcome_bucket"), "unknown")
    if status == "hit":
        return "cache-hit-observed"
    if _cache_ladder_policy_reload_required(cache) or "reload" in reason:
        return "staged-policy-not-loaded"
    if status == "holdout" or outcome_bucket == "holdout" or reason in {"canary_holdout", "codex-app-cache-canary-holdout"}:
        return "holdout-only"
    if _cache_ladder_dependency_blocked(cache, reason, outcome_bucket):
        return "dependency-invalidation-blocked"
    if _as_int(row.get("stream")) or reason in {"streaming", "legacy-streaming", "streaming-cache-disabled", "streaming-tools-disabled"}:
        return "skipped-streaming"
    if reason in {"tools-disabled", "tool-cache-disabled"}:
        return "skipped-tools"
    if status == "disabled" or outcome_bucket == "disabled" or reason in {"cache-disabled", "codex-app-cache-disabled"}:
        return "disabled"
    if status == "miss" or outcome_bucket == "miss" or reason in {"exact-miss", "exact-pattern-miss", "semantic-miss", "exact-and-semantic-miss"}:
        return "true-miss"
    if status in {"unsafe-skip", "safety-stopped"} or outcome_bucket == "unsafe-skip":
        return "safety-stopped"
    return "unknown-cache-decision"


def _cache_zero_hit_blocker_ladder(rows: list[dict[str, Any]], *, scan_limit: int = _CACHE_BLOCKER_SCAN_LIMIT) -> dict[str, Any]:
    capped_limit = max(1, min(int(scan_limit or _CACHE_BLOCKER_SCAN_LIMIT), 5000))
    scanned = rows[:capped_limit]
    grouped: dict[tuple[str, str, str, str, str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    cache_hits = 0
    for row in scanned:
        cache = _json_obj(row.get("cache_json"))
        routing = _json_obj(row.get("routing_json"))
        decision = _cache_decision_for_breakdown(row)
        if decision.get("status") == "hit" or _as_int(row.get("cache_hit")):
            cache_hits += 1
        blocker = _cache_blocker_code(row, decision, cache)
        next_action = _CACHE_BLOCKER_NEXT_ACTIONS.get(blocker, _CACHE_BLOCKER_NEXT_ACTIONS["unknown-cache-decision"])
        source_surface = canonical_source_surface(
            cache.get("surface")
            or row.get("source_surface")
            or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        )
        has_tools = _cache_ladder_has_tools(cache, routing)
        category = _cache_ladder_category(row, cache, routing)
        workflow_phase = _cache_ladder_phase(cache, routing)
        key = (
            blocker,
            public_label(row.get("provider") or "anthropic", "unknown"),
            public_label(source_surface, "unknown"),
            _endpoint_for_cache_ladder(row),
            category,
            workflow_phase,
            _cache_ladder_bool_label(bool(_as_int(row.get("stream"))), true_label="stream", false_label="non-stream"),
            _cache_ladder_bool_label(has_tools, true_label="tools", false_label="no-tools"),
            _cache_ladder_replayability(cache, routing),
            public_label(decision.get("policy_source"), "unknown"),
            public_label(decision.get("status"), "unknown"),
            public_label(decision.get("reason"), "unknown"),
        )
        bucket = grouped.setdefault(
            key,
            {
                "blocker_code": key[0],
                "provider": key[1],
                "source_surface": key[2],
                "endpoint": key[3],
                "category": key[4],
                "workflow_phase": key[5],
                "stream_mode": key[6],
                "tool_presence": key[7],
                "replayability_level": key[8],
                "cache_policy_source": key[9],
                "cache_status": key[10],
                "cache_reason": key[11],
                "next_action_family": next_action["family"],
                "next_action_label": next_action["label"],
                "count": 0,
            },
        )
        bucket["count"] += 1

    ladder = list(grouped.values())
    ladder.sort(key=lambda row: (-row["count"], row["blocker_code"], row["provider"], row["source_surface"]))
    top = ladder[0] if ladder else None
    action_candidates = [
        candidate
        for candidate in (
            _cache_action_candidate_from_ladder_row(row, rank=index + 1)
            for index, row in enumerate(ladder)
            if row.get("blocker_code") != "cache-hit-observed"
        )
        if candidate is not None
    ][:10] if cache_hits == 0 else []
    return {
        "schema": "tokenclaw.cache_zero_hit_blocker_ladder.v1",
        "generated_at": utc_now(),
        "summary": {
            "scan_limit": capped_limit,
            "scanned_rows": len(scanned),
            "available_rows": len(rows),
            "bounded_recent_window": True,
            "cache_hits": cache_hits,
            "zero_hit_window": cache_hits == 0,
            "blocker_bucket_count": len(ladder),
            "action_candidate_count": len(action_candidates),
            "top_blocker_code": top.get("blocker_code") if top else None,
            "top_next_action_family": top.get("next_action_family") if top else None,
            "top_action_candidate_family": (
                ((action_candidates[0].get("local_action") or {}).get("family"))
                if action_candidates
                else None
            ),
        },
        "ladder": ladder[:50],
        "action_candidates": action_candidates,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "raw_request_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
            "candidate_identifiers_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }

_CACHE_REPLAY_INVALIDATION_REASONS = {
    "file-dependency-missing",
    "dependency-missing",
    "dependency-changed",
    "dependency-deleted",
    "dependency-created",
    "dependency-cap-exceeded",
    "file-watch-disabled",
    "file-dependency-evidence-absent",
    "safe-invalidation-required",
    "tool-cache-rule-requires-safe-invalidation",
    "tool-cache-rule-missing-safe-invalidation",
    "unsafe-tool-cache-pattern",
    "file-watch-required",
}
_CACHE_REPLAY_STALE_RISK_REASONS = {
    "current-state",
    "user-specific",
    "low-cacheability",
    "stale-risk-blockers",
}
_CACHE_REPLAY_SAFETY_STOP_REASON = "local-canary-safety-stop"


def _public_cache_canary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    public: dict[str, Any] = {}
    for key in ("enabled", "selected", "fraction", "threshold"):
        if value.get(key) is not None:
            public[key] = value.get(key)
    for key in ("cohort", "unit", "reason"):
        if value.get(key) is not None:
            public[key] = public_label(value.get(key), "unknown")
    public["pattern_hashes_included"] = False
    return public


def _public_cache_rollout(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: value.get(key)
        for key in (
            "enabled",
            "canary_enabled",
            "canary_fraction",
            "holdout_fraction",
            "rollout_fraction",
            "canary_unit",
            "unit",
        )
        if value.get(key) is not None
    }


def _cache_pattern_rules_from_meta(cache: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if isinstance(cache.get("pattern_rule"), dict):
        rules.append(cache["pattern_rule"])
    pattern_rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), dict) else {}
    for rule in pattern_rules.get("rules") or []:
        if isinstance(rule, dict):
            rules.append(rule)
    for skip in pattern_rules.get("skip_reasons") or []:
        if not isinstance(skip, dict):
            continue
        if skip.get("rule_id") or skip.get("candidate_id") or skip.get("canary") or skip.get("safety_stop"):
            rules.append(skip)

    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for rule in rules:
        key = (
            str(rule.get("rule_id") or ""),
            str(rule.get("candidate_id") or ""),
            str(rule.get("reason") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def _cache_rule_identity(
    *,
    cache: dict[str, Any],
    rule: dict[str, Any] | None,
    source_surface: str,
    category: str,
    stream: bool,
    has_tools: bool,
) -> dict[str, Any]:
    rule = rule if isinstance(rule, dict) else {}
    raw_rule_id = rule.get("rule_id") or cache.get("rule_id") or "unruled-cache-decision"
    raw_candidate_id = rule.get("candidate_id") if rule.get("candidate_id") is not None else cache.get("candidate_id")
    rule_id = public_id(raw_rule_id, prefix="rule-id", fallback="unruled-cache-decision") or "unruled-cache-decision"
    candidate_id = public_id(raw_candidate_id, prefix="candidate-id") if raw_candidate_id is not None else None
    policy_source = public_label(rule.get("policy_source") or cache.get("policy_source") or "unknown", "unknown")
    return {
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "policy_source": policy_source,
        "source_surface": canonical_source_surface(source_surface),
        "category": public_label(category or "unknown", "unknown"),
        "stream": bool(stream),
        "has_tools": bool(has_tools),
    }


def _cache_confidence_outcome(cache: dict[str, Any], decision: dict[str, Any], rule: dict[str, Any] | None) -> str:
    status = str(decision.get("status") or cache.get("status") or "missing")
    reason = str((rule or {}).get("reason") or decision.get("reason") or cache.get("reason") or "unknown")
    canary = (rule or {}).get("canary") if isinstance(rule, dict) else None
    if not isinstance(canary, dict):
        canary = cache.get("canary") if isinstance(cache.get("canary"), dict) else None
    cohort = str(cache.get("canary_cohort") or (canary or {}).get("cohort") or "")
    if reason == _CACHE_REPLAY_SAFETY_STOP_REASON or isinstance((rule or {}).get("safety_stop"), dict) or isinstance(cache.get("safety_stop"), dict):
        return "safety_stop"
    if reason == "canary_holdout" or cohort == "canary_holdout" or (isinstance(canary, dict) and canary.get("selected") is False):
        return "holdout"
    if status == "hit":
        return "hit"
    if status == "miss":
        return "miss"
    return "skip"


def _cache_confidence_reason_counts(
    *,
    cache: dict[str, Any],
    decision: dict[str, Any],
    rule: dict[str, Any] | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    invalidation: dict[str, int] = {}
    stale: dict[str, int] = {}
    safety: dict[str, int] = {}
    reasons = {
        str(decision.get("reason") or cache.get("reason") or ""),
        str((rule or {}).get("reason") or ""),
    }
    for blocker in (rule or {}).get("blockers") or cache.get("blockers") or []:
        if isinstance(blocker, (str, int, float, bool)):
            reasons.add(str(blocker))
    for blocker in (rule or {}).get("stale_risk_blockers") or cache.get("stale_risk_blockers") or []:
        if isinstance(blocker, (str, int, float, bool)):
            reasons.add(str(blocker))
    audit = cache.get("file_dependency_audit") if isinstance(cache.get("file_dependency_audit"), dict) else {}
    if audit.get("cap_exceeded"):
        reasons.add("dependency-cap-exceeded")
    if audit.get("invalidation_reason"):
        reasons.add(str(audit.get("invalidation_reason")))
    if _as_int(audit.get("changed_path_count")):
        reasons.add("dependency-changed")
    if _as_int(audit.get("deleted_path_count")):
        reasons.add("dependency-deleted")
    if _as_int(audit.get("missing_path_count")):
        reasons.add("dependency-missing")
    cacheability = cache.get("cacheability") if isinstance(cache.get("cacheability"), dict) else {}
    if cacheability.get("time_sensitive_hint"):
        reasons.add("current-state")
    if cacheability.get("user_specific_hint"):
        reasons.add("user-specific")
    if str(cacheability.get("cacheability_bucket") or "").lower() == "low":
        reasons.add("low-cacheability")
    if isinstance((rule or {}).get("safety_stop"), dict) or isinstance(cache.get("safety_stop"), dict):
        reasons.add(_CACHE_REPLAY_SAFETY_STOP_REASON)

    for reason in reasons:
        if not reason:
            continue
        if reason in _CACHE_REPLAY_INVALIDATION_REASONS:
            invalidation[reason] = invalidation.get(reason, 0) + 1
        if reason in _CACHE_REPLAY_STALE_RISK_REASONS:
            stale[reason] = stale.get(reason, 0) + 1
        if reason == _CACHE_REPLAY_SAFETY_STOP_REASON:
            safety[reason] = safety.get(reason, 0) + 1
    return invalidation, stale, safety


def _cache_replay_confidence_rows_from_store(store_obj: Any, *, limit: int) -> list[dict[str, Any]]:
    conn = store_obj.conn
    capped = max(1, min(int(limit or 1000), 10000))
    provider_rows = [
        dict(row)
        for row in conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, stream, cache_hit, status_code,
                   latency_ms, retry_count, cost_est_usd, cost_baseline_usd,
                   cache_json, routing_json, category,
                   null as response_error_code,
                   null as response_result_chars,
                   null as input_text_chars
            from calls
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]
    for row in provider_rows:
        row["source_surface"] = _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        row["granularity"] = "provider_request"

    codex_rows = [
        dict(row)
        for row in conn.execute(
            """
            select s.created_at,
                   s.routing_json,
                   s.cache_json,
                   s.input_text_chars,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars,
                   (
                       select r.error_code from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_error_code,
                   (
                       select r.latency_ms from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as latency_ms
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]
    for row in codex_rows:
        cache = _json_obj(row.get("cache_json"))
        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), row.get("response_result_chars"), cache)
        row.update(
            {
                "path": "codex-app://turn/start",
                "provider": "codex-app",
                "source_surface": CODEX_APP_SOURCE_SURFACE,
                "granularity": "agent_turn",
                "requested_model": CODEX_APP_MODEL,
                "routed_model": CODEX_APP_MODEL,
                "stream": 0,
                "cache_hit": 1 if cache.get("status") == "hit" else 0,
                "status_code": 500 if row.get("response_error_code") is not None else 200,
                "retry_count": 0,
                "cost_est_usd": estimates.get("cost_est_usd"),
                "cost_baseline_usd": estimates.get("baseline_cost_est_usd"),
                "category": _json_obj(row.get("routing_json")).get("category") or "codex_turn",
            }
        )

    return sorted(provider_rows + codex_rows, key=lambda row: str(row.get("created_at") or ""), reverse=True)[:capped]


async def stats_cache_replay_confidence(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    rows = _cache_replay_confidence_rows_from_store(store_obj, limit=limit)
    grouped: dict[tuple[str, str, str, str, str, bool, bool], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    invalidation_counts: dict[str, int] = {}
    stale_counts: dict[str, int] = {}
    safety_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for row in rows:
        cache = _json_obj(row.get("cache_json"))
        if not cache:
            continue
        routing = _json_obj(row.get("routing_json"))
        decision = _cache_decision_for_breakdown(row)
        source_surface = str(decision.get("source_surface") or row.get("source_surface") or "unknown")
        category = str(row.get("category") or routing.get("category") or cache.get("category") or "unknown")
        stream = bool(_as_int(row.get("stream")) or cache.get("stream"))
        has_tools = bool(routing.get("has_tools") or category.startswith("tool") or cache.get("has_tools"))
        rules = _cache_pattern_rules_from_meta(cache) or [None]
        for rule in rules:
            identity = _cache_rule_identity(
                cache=cache,
                rule=rule,
                source_surface=source_surface,
                category=category,
                stream=stream,
                has_tools=has_tools,
            )
            key = (
                identity["rule_id"],
                str(identity.get("candidate_id") or ""),
                identity["policy_source"],
                identity["source_surface"],
                identity["category"],
                identity["stream"],
                identity["has_tools"],
            )
            bucket = grouped.setdefault(
                key,
                {
                    **identity,
                    "granularities": set(),
                    "sample_count": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "holdout_count": 0,
                    "skip_count": 0,
                    "safety_stop_count": 0,
                    "invalidation_count": 0,
                    "stale_risk_blocked_count": 0,
                    "error_count": 0,
                    "retry_count": 0,
                    "replayed_count": 0,
                    "replayed_error_count": 0,
                    "replayed_retry_count": 0,
                    "replayed_latency_ms_total": 0,
                    "replayed_latency_sample_count": 0,
                    "replayed_estimated_saved_cost_usd": 0.0,
                    "holdout_error_count": 0,
                    "holdout_retry_count": 0,
                    "holdout_latency_ms_total": 0,
                    "holdout_latency_sample_count": 0,
                    "holdout_estimated_saved_cost_usd": 0.0,
                    "estimated_saved_cost_usd": 0.0,
                    "estimated_cost_usd": 0.0,
                    "baseline_cost_usd": 0.0,
                    "first_seen_at": row.get("created_at"),
                    "last_seen_at": row.get("created_at"),
                    "canary": None,
                    "rollout": None,
                    "safety_stop_active": False,
                    "safety_stop": None,
                    "_invalidation_reasons": {},
                    "_stale_reasons": {},
                    "_safety_reasons": {},
                },
            )
            outcome = _cache_confidence_outcome(cache, decision, rule)
            bucket["sample_count"] += 1
            bucket["granularities"].add(str(row.get("granularity") or "provider_request"))
            bucket[f"{outcome}_count"] = _as_int(bucket.get(f"{outcome}_count")) + 1
            if outcome == "hit":
                bucket["replayed_count"] += 1
            status_counts[outcome] = status_counts.get(outcome, 0) + 1
            source_counts[identity["policy_source"]] = source_counts.get(identity["policy_source"], 0) + 1

            status_code = _as_int(row.get("status_code")) if row.get("status_code") is not None else None
            retry_count = _as_int(row.get("retry_count"))
            errored = bool(status_code is not None and status_code >= 400)
            if errored:
                bucket["error_count"] += 1
            if retry_count:
                bucket["retry_count"] += retry_count
            if outcome == "hit":
                bucket["replayed_error_count"] += 1 if errored else 0
                bucket["replayed_retry_count"] += retry_count
            elif outcome == "holdout":
                bucket["holdout_error_count"] += 1 if errored else 0
                bucket["holdout_retry_count"] += retry_count

            saved = _as_float(cache.get("estimated_saved_cost_usd"))
            if not saved and outcome == "hit":
                saved = max(_as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd")), 0.0)
            latency_value = _as_int(row.get("latency_ms")) if row.get("latency_ms") is not None else None
            if outcome == "hit":
                bucket["replayed_estimated_saved_cost_usd"] += saved
                if latency_value is not None:
                    bucket["replayed_latency_ms_total"] += latency_value
                    bucket["replayed_latency_sample_count"] += 1
            elif outcome == "holdout":
                holdout_saved = max(_as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd")), 0.0)
                bucket["holdout_estimated_saved_cost_usd"] += holdout_saved
                if latency_value is not None:
                    bucket["holdout_latency_ms_total"] += latency_value
                    bucket["holdout_latency_sample_count"] += 1
            bucket["estimated_saved_cost_usd"] += saved
            bucket["estimated_cost_usd"] += _as_float(row.get("cost_est_usd"))
            bucket["baseline_cost_usd"] += _as_float(row.get("cost_baseline_usd"))
            if str(row.get("created_at") or "") < str(bucket.get("first_seen_at") or row.get("created_at") or ""):
                bucket["first_seen_at"] = row.get("created_at")
            if str(row.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
                bucket["last_seen_at"] = row.get("created_at")

            canary = _public_cache_canary((rule or {}).get("canary") if isinstance(rule, dict) else cache.get("canary"))
            if canary and bucket["canary"] is None:
                bucket["canary"] = canary
            rollout = _public_cache_rollout((rule or {}).get("rollout") if isinstance(rule, dict) else cache.get("rollout"))
            if rollout and bucket["rollout"] is None:
                bucket["rollout"] = rollout
            safety_stop = (rule or {}).get("safety_stop") if isinstance(rule, dict) else cache.get("safety_stop")
            if isinstance(safety_stop, dict):
                bucket["safety_stop"] = {
                    key: safety_stop.get(key)
                    for key in ("reason", "decision", "rule_id", "candidate_id", "sample_count", "error_rate", "retry_rate")
                    if safety_stop.get(key) is not None
                }
                bucket["safety_stop_active"] = True

            invalidation, stale, safety = _cache_confidence_reason_counts(cache=cache, decision=decision, rule=rule)
            for reason, count in invalidation.items():
                bucket["_invalidation_reasons"][reason] = bucket["_invalidation_reasons"].get(reason, 0) + count
                invalidation_counts[reason] = invalidation_counts.get(reason, 0) + count
            for reason, count in stale.items():
                bucket["_stale_reasons"][reason] = bucket["_stale_reasons"].get(reason, 0) + count
                stale_counts[reason] = stale_counts.get(reason, 0) + count
            for reason, count in safety.items():
                bucket["_safety_reasons"][reason] = bucket["_safety_reasons"].get(reason, 0) + count
                safety_counts[reason] = safety_counts.get(reason, 0) + count

    confidence_rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        samples = _as_int(bucket.get("sample_count"))
        replayed = _as_int(bucket.get("replayed_count"))
        holdout = _as_int(bucket.get("holdout_count"))
        bucket["granularities"] = sorted(bucket["granularities"])
        bucket["invalidation_count"] = sum(bucket["_invalidation_reasons"].values())
        bucket["stale_risk_blocked_count"] = sum(bucket["_stale_reasons"].values())
        bucket["safety_stop_count"] = sum(bucket["_safety_reasons"].values()) or bucket["safety_stop_count"]
        bucket["hit_rate"] = round(_as_int(bucket.get("hit_count")) / samples, 4) if samples else 0.0
        bucket["holdout_rate"] = round(holdout / samples, 4) if samples else 0.0
        bucket["error_rate"] = round(_as_int(bucket.get("error_count")) / samples, 4) if samples else 0.0
        bucket["retry_rate"] = round(_as_int(bucket.get("retry_count")) / samples, 4) if samples else 0.0
        bucket["replayed_error_rate"] = round(_as_int(bucket.get("replayed_error_count")) / replayed, 4) if replayed else 0.0
        bucket["holdout_error_rate"] = round(_as_int(bucket.get("holdout_error_count")) / holdout, 4) if holdout else 0.0
        bucket["replayed_retry_rate"] = round(_as_int(bucket.get("replayed_retry_count")) / replayed, 4) if replayed else 0.0
        bucket["holdout_retry_rate"] = round(_as_int(bucket.get("holdout_retry_count")) / holdout, 4) if holdout else 0.0
        replayed_latency_samples = _as_int(bucket.pop("replayed_latency_sample_count"))
        holdout_latency_samples = _as_int(bucket.pop("holdout_latency_sample_count"))
        replayed_latency_total = _as_int(bucket.pop("replayed_latency_ms_total"))
        holdout_latency_total = _as_int(bucket.pop("holdout_latency_ms_total"))
        bucket["replayed_avg_latency_ms"] = round(replayed_latency_total / replayed_latency_samples) if replayed_latency_samples else None
        bucket["holdout_avg_latency_ms"] = round(holdout_latency_total / holdout_latency_samples) if holdout_latency_samples else None
        bucket["replayed_estimated_saved_cost_usd"] = round(_as_float(bucket.get("replayed_estimated_saved_cost_usd")), 8)
        bucket["holdout_estimated_saved_cost_usd"] = round(_as_float(bucket.get("holdout_estimated_saved_cost_usd")), 8)
        bucket["replayed_savings_rate_usd"] = round(
            _as_float(bucket.get("replayed_estimated_saved_cost_usd")) / replayed,
            8,
        ) if replayed else 0.0
        bucket["holdout_savings_rate_usd"] = round(
            _as_float(bucket.get("holdout_estimated_saved_cost_usd")) / holdout,
            8,
        ) if holdout else 0.0
        bucket["estimated_saved_cost_usd"] = round(_as_float(bucket.get("estimated_saved_cost_usd")), 8)
        bucket["estimated_cost_usd"] = round(_as_float(bucket.get("estimated_cost_usd")), 8)
        bucket["baseline_cost_usd"] = round(_as_float(bucket.get("baseline_cost_usd")), 8)
        bucket["invalidation_reasons"] = _breakdown_from_counts(bucket.pop("_invalidation_reasons"))
        bucket["stale_risk_reasons"] = _breakdown_from_counts(bucket.pop("_stale_reasons"))
        bucket["safety_stop_reasons"] = _breakdown_from_counts(bucket.pop("_safety_reasons"))
        if bucket.get("canary") is None and bucket.get("rollout"):
            rollout = bucket["rollout"]
            fraction = rollout.get("canary_fraction") if rollout.get("canary_fraction") is not None else rollout.get("rollout_fraction")
            bucket["canary"] = {
                "enabled": bool(rollout.get("canary_enabled", True)),
                "fraction": fraction,
                "pattern_hashes_included": False,
            }
        confidence_rows.append(bucket)

    confidence_rows.sort(
        key=lambda row: (
            bool(row.get("safety_stop_active")),
            _as_int(row.get("sample_count")),
            _as_float(row.get("estimated_saved_cost_usd")),
            _as_int(row.get("hit_count")),
        ),
        reverse=True,
    )
    return {
        "schema": "tokenclaw.cache_replay_confidence.v1",
        "generated_at": utc_now(),
        "summary": {
            "rows_considered": len(rows),
            "rule_buckets": len(confidence_rows),
            "active_rule_count": sum(1 for row in confidence_rows if row.get("rule_id") != "unruled-cache-decision"),
            "hit_rows": status_counts.get("hit", 0),
            "miss_rows": status_counts.get("miss", 0),
            "holdout_rows": status_counts.get("holdout", 0),
            "skip_rows": status_counts.get("skip", 0),
            "safety_stop_rows": status_counts.get("safety_stop", 0),
            "invalidation_rows": sum(invalidation_counts.values()),
            "stale_risk_blocked_rows": sum(stale_counts.values()),
            "estimated_saved_cost_usd": round(sum(_as_float(row.get("estimated_saved_cost_usd")) for row in confidence_rows), 8),
            "safety_stop_active": any(bool(row.get("safety_stop_active")) for row in confidence_rows),
        },
        "policy_source_breakdown": _breakdown_from_counts(source_counts),
        "outcome_breakdown": _breakdown_from_counts(status_counts),
        "invalidation_breakdown": _breakdown_from_counts(invalidation_counts),
        "stale_risk_breakdown": _breakdown_from_counts(stale_counts),
        "safety_stop_breakdown": _breakdown_from_counts(safety_counts),
        "rules": confidence_rows[: max(1, min(int(limit or 50), 1000))],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "pattern_hashes_included": False,
            "provider_calls_made": 0,
            "dashboard_read_only": True,
            "basis": "stored cache decision metadata, canary cohorts, safety-stop summaries, status codes, retries, and cost estimates only",
        },
    }


def _cache_replay_rule_public_from_config(rule: dict[str, Any]) -> dict[str, Any]:
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    return {
        "rule_id": public_id(rule.get("id") or rule.get("rule_id") or "cache-pattern-rule", prefix="rule-id", fallback="cache-pattern-rule"),
        "candidate_id": public_id(rule.get("candidate_id"), prefix="candidate-id") if rule.get("candidate_id") is not None else None,
        "policy_source": public_label(rule.get("policy_source") or "managed-recommended", "unknown"),
        "enabled": bool(rule.get("enabled", True)),
        "allow_tool_calls": bool(action.get("allow_tool_calls")),
        "safe_invalidation": bool(action.get("safe_invalidation")),
        "streaming": bool(action.get("streaming")),
        "scope": public_label(action.get("scope") or "session", "unknown"),
        "replayability_levels": [
            public_label(item, "unknown")
            for item in conditions.get("replayability_levels") or []
            if isinstance(item, (str, int, float, bool))
        ],
        "pattern_hash_count": len(conditions.get("pattern_hashes") or []),
        "pattern_hashes_included": False,
        "rollout": _public_cache_rollout(rule.get("rollout")),
    }


def _cache_replay_readiness_state(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if bool(row.get("safety_stop_active")) or _as_int(row.get("safety_stop_count")):
        return "safety-stopped", [_CACHE_REPLAY_SAFETY_STOP_REASON]
    if _as_int(row.get("invalidation_count")):
        reasons.extend(str(item.get("value")) for item in row.get("invalidation_reasons") or [] if item.get("value"))
    if _as_int(row.get("stale_risk_blocked_count")):
        reasons.extend(str(item.get("value")) for item in row.get("stale_risk_reasons") or [] if item.get("value"))
    if reasons:
        return "blocked", sorted(set(reasons))

    replayed = _as_int(row.get("replayed_count"))
    holdout = _as_int(row.get("holdout_count"))
    hits = _as_int(row.get("hit_count"))
    misses = _as_int(row.get("miss_count"))
    skips = _as_int(row.get("skip_count"))
    if replayed and holdout:
        return "ready", ["replay-and-holdout-observed"]
    if hits:
        return "active-no-holdout", ["replay-observed-without-holdout"]
    if holdout:
        return "holdout-only", ["canary-holdout-observed"]
    if misses:
        return "miss-only", ["exact-miss-observed"]
    if skips:
        return "blocked", ["skipped-no-replay"]
    return "not-observed", ["no-recent-rule-metadata"]


async def stats_cache_replay_readiness(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    confidence = await stats_cache_replay_confidence(store_obj, limit=limit)
    replayability = await stats_cache_replayability(store_obj, limit=50)
    from tokenclaw.stats import stats_policies

    policies = await stats_policies()
    cache_policy = policies.get("cache") if isinstance(policies.get("cache"), dict) else {}
    cache_file = cache_policy.get("file") if isinstance(cache_policy.get("file"), dict) else {}

    try:
        from tokenclaw import cache as cache_module

        configured_rules = [
            _cache_replay_rule_public_from_config(dict(rule))
            for rule in getattr(cache_module, "CACHE_PATTERN_RULES", ())
            if isinstance(rule, dict)
        ]
    except Exception:
        configured_rules = []

    rows: list[dict[str, Any]] = []
    observed_rule_ids: set[str] = set()
    for rule in confidence.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        state, reason_codes = _cache_replay_readiness_state(rule)
        rule_id = str(rule.get("rule_id") or "unruled-cache-decision")
        if rule_id != "unruled-cache-decision":
            observed_rule_ids.add(rule_id)
        canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        dependency_status = "not-required"
        if bool(rule.get("has_tools")):
            dependency_status = "blocked" if _as_int(rule.get("invalidation_count")) else "ready"
        elif _as_int(rule.get("invalidation_count")):
            dependency_status = "invalidated"
        rows.append({
            "rule_id": rule_id,
            "candidate_id": rule.get("candidate_id"),
            "policy_source": rule.get("policy_source") or "unknown",
            "source_surface": rule.get("source_surface") or "unknown",
            "category": rule.get("category") or "unknown",
            "stream": bool(rule.get("stream")),
            "has_tools": bool(rule.get("has_tools")),
            "readiness": state,
            "reason_codes": reason_codes,
            "sample_count": _as_int(rule.get("sample_count")),
            "hit_count": _as_int(rule.get("hit_count")),
            "replayed_count": _as_int(rule.get("replayed_count")),
            "miss_count": _as_int(rule.get("miss_count")),
            "holdout_count": _as_int(rule.get("holdout_count")),
            "skip_count": _as_int(rule.get("skip_count")),
            "safety_stop_count": _as_int(rule.get("safety_stop_count")),
            "safety_stop_active": bool(rule.get("safety_stop_active")),
            "invalidation_count": _as_int(rule.get("invalidation_count")),
            "stale_risk_blocked_count": _as_int(rule.get("stale_risk_blocked_count")),
            "dependency_evidence_status": dependency_status,
            "canary_enabled": bool(canary.get("enabled") if canary else rollout.get("canary_enabled", False)),
            "canary_fraction": canary.get("fraction") if canary.get("fraction") is not None else rollout.get("canary_fraction"),
            "holdout_fraction_observed": rule.get("holdout_rate"),
            "replayed_error_rate": rule.get("replayed_error_rate"),
            "holdout_error_rate": rule.get("holdout_error_rate"),
            "replayed_retry_rate": rule.get("replayed_retry_rate"),
            "holdout_retry_rate": rule.get("holdout_retry_rate"),
            "replayed_avg_latency_ms": rule.get("replayed_avg_latency_ms"),
            "holdout_avg_latency_ms": rule.get("holdout_avg_latency_ms"),
            "replayed_savings_rate_usd": rule.get("replayed_savings_rate_usd"),
            "holdout_savings_rate_usd": rule.get("holdout_savings_rate_usd"),
            "estimated_saved_cost_usd": rule.get("estimated_saved_cost_usd"),
            "invalidation_reasons": rule.get("invalidation_reasons") or [],
            "stale_risk_reasons": rule.get("stale_risk_reasons") or [],
            "safety_stop_reasons": rule.get("safety_stop_reasons") or [],
            "last_seen_at": rule.get("last_seen_at"),
            "pattern_hashes_included": False,
        })

    for rule in configured_rules:
        rule_id = str(rule.get("rule_id") or "cache-pattern-rule")
        if rule_id in observed_rule_ids:
            continue
        state = "not-observed" if rule.get("enabled") else "disabled"
        reason = "no-recent-rule-metadata" if rule.get("enabled") else "disabled-rule"
        rows.append({
            **rule,
            "source_surface": "unknown",
            "category": "unknown",
            "stream": bool(rule.get("streaming")),
            "has_tools": bool(rule.get("allow_tool_calls")),
            "readiness": state,
            "reason_codes": [reason],
            "sample_count": 0,
            "hit_count": 0,
            "replayed_count": 0,
            "miss_count": 0,
            "holdout_count": 0,
            "skip_count": 0,
            "safety_stop_count": 0,
            "safety_stop_active": False,
            "invalidation_count": 0,
            "stale_risk_blocked_count": 0,
            "dependency_evidence_status": "not-observed",
            "canary_enabled": bool(((rule.get("rollout") or {}).get("canary_enabled")) if isinstance(rule.get("rollout"), dict) else False),
            "canary_fraction": ((rule.get("rollout") or {}).get("canary_fraction") if isinstance(rule.get("rollout"), dict) else None),
            "holdout_fraction_observed": 0.0,
            "estimated_saved_cost_usd": 0.0,
            "invalidation_reasons": [],
            "stale_risk_reasons": [],
            "safety_stop_reasons": [],
        })

    rows.sort(
        key=lambda row: (
            row.get("readiness") == "safety-stopped",
            row.get("readiness") == "blocked",
            _as_int(row.get("sample_count")),
            _as_float(row.get("estimated_saved_cost_usd")),
        ),
        reverse=True,
    )
    readiness_counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("readiness") or "unknown")
        readiness_counts[state] = readiness_counts.get(state, 0) + 1

    confidence_summary = confidence.get("summary") if isinstance(confidence.get("summary"), dict) else {}
    replay_summary = replayability.get("summary") if isinstance(replayability.get("summary"), dict) else {}
    session_memory_proposals = (
        replayability.get("session_memory_replay_proposals")
        if isinstance(replayability.get("session_memory_replay_proposals"), list)
        else []
    )
    active = sum(1 for row in rows if row.get("readiness") in {"ready", "active-no-holdout"})
    blocked = sum(1 for row in rows if row.get("readiness") in {"blocked", "safety-stopped", "disabled"})
    if confidence_summary.get("safety_stop_active"):
        status = "safety-stopped"
    elif active and not blocked:
        status = "ready"
    elif active:
        status = "partial"
    elif blocked:
        status = "blocked"
    elif configured_rules:
        status = "not-observed"
    else:
        status = "no-rules"

    return {
        "schema": "tokenclaw.cache_replay_readiness.v1",
        "generated_at": utc_now(),
        "status": status,
        "summary": {
            "configured_rule_count": len(configured_rules),
            "observed_rule_count": len(observed_rule_ids),
            "active_rule_count": active,
            "blocked_rule_count": blocked,
            "ready_rule_count": readiness_counts.get("ready", 0),
            "safety_stop_active": bool(confidence_summary.get("safety_stop_active")),
            "safety_stop_rows": _as_int(confidence_summary.get("safety_stop_rows")),
            "hit_rows": _as_int(confidence_summary.get("hit_rows")),
            "holdout_rows": _as_int(confidence_summary.get("holdout_rows")),
            "miss_rows": _as_int(confidence_summary.get("miss_rows")),
            "invalidation_rows": _as_int(confidence_summary.get("invalidation_rows")),
            "stale_risk_blocked_rows": _as_int(confidence_summary.get("stale_risk_blocked_rows")),
            "estimated_saved_cost_usd": _as_float(confidence_summary.get("estimated_saved_cost_usd")),
            "repeated_shape_groups": _as_int(replay_summary.get("repeated_shape_groups")),
            "repeated_shape_exists_but_cache_is_unsafe": bool(replay_summary.get("repeated_shape_exists_but_cache_is_unsafe")),
            "session_memory_replay_proposal_count": _as_int(replay_summary.get("session_memory_replay_proposal_count")),
            "session_memory_replay_eligible_count": _as_int(replay_summary.get("session_memory_replay_eligible_count")),
            "session_memory_replay_blocked_count": _as_int(replay_summary.get("session_memory_replay_blocked_count")),
            "policy_source": cache_policy.get("policy_source") or "unknown",
            "cache_enabled": bool(cache_policy.get("enabled")),
            "exact_cache_enabled": bool(((cache_policy.get("exact_cache") or {}) if isinstance(cache_policy.get("exact_cache"), dict) else {}).get("enabled")),
            "tool_cache_enabled": bool(((cache_policy.get("exact_cache") or {}) if isinstance(cache_policy.get("exact_cache"), dict) else {}).get("cache_tool_calls")),
            "file_watch_enabled": bool(((cache_policy.get("file_watch") or {}) if isinstance(cache_policy.get("file_watch"), dict) else {}).get("enabled")),
            "policy_reload_required": bool(cache_file.get("reload_required")),
        },
        "readiness_breakdown": _breakdown_from_counts(readiness_counts),
        "invalidation_breakdown": confidence.get("invalidation_breakdown") or [],
        "stale_risk_breakdown": confidence.get("stale_risk_breakdown") or [],
        "safety_stop_breakdown": confidence.get("safety_stop_breakdown") or [],
        "blocker_breakdown": replayability.get("blocker_breakdown") or [],
        "session_memory_replay_proposals": session_memory_proposals[: max(1, min(int(limit or 50), 1000))],
        "rules": rows[: max(1, min(int(limit or 50), 1000))],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "pattern_hashes_included": False,
            "provider_calls_made": 0,
            "dashboard_read_only": True,
            "basis": "local cache policy state, bounded replay canary metadata, invalidation reason codes, safety-stop summaries, and aggregate cache replayability blockers",
        },
    }


def _cache_replay_activation_provider_gate(cache: dict[str, Any], rule: dict[str, Any] | None) -> dict[str, Any]:
    candidates = [
        cache.get("provider_adoption_gate"),
        cache.get("provider_adoption_health"),
        cache.get("provider_adoption"),
        (rule or {}).get("provider_adoption_gate") if isinstance(rule, dict) else None,
        (rule or {}).get("provider_adoption_health") if isinstance(rule, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        status = public_label(candidate.get("status") or candidate.get("readiness") or "unknown", "unknown")
        reason_codes = [
            public_label(item, "unknown")
            for item in candidate.get("reason_codes") or candidate.get("blockers") or []
            if isinstance(item, (str, int, float, bool))
        ]
        blocking = bool(candidate.get("blocking")) or status in {"blocked", "failed", "regressed"}
        if blocking:
            return {"status": "blocked", "reason_codes": sorted(set(reason_codes or ["provider-adoption-regression"]))}
        if status in {"ready", "pass", "healthy"} or candidate.get("blocking") is False:
            return {"status": "ready", "reason_codes": sorted(set(reason_codes))}
        return {"status": status, "reason_codes": sorted(set(reason_codes))}
    return {"status": "not-observed", "reason_codes": []}


def _cache_replay_activation_projected_hits(*values: Any) -> int:
    for value in values:
        if isinstance(value, dict):
            for key in ("projected_hits", "projected_hit_count", "expected_hits", "expected_hit_count"):
                if value.get(key) is not None:
                    return _as_int(value.get(key))
    return 0


def _cache_replay_activation_projected_savings(*values: Any) -> float:
    for value in values:
        if isinstance(value, dict):
            for key in (
                "projected_saved_cost_usd",
                "projected_savings_usd",
                "projected_estimated_saved_cost_usd",
                "expected_saved_cost_usd",
            ):
                if value.get(key) is not None:
                    return _as_float(value.get(key))
    return 0.0


def _cache_replay_activation_state(bucket: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = set(str(item) for item in bucket.get("_reason_codes", set()) if item)
    provider = bucket.get("provider_adoption_gate") if isinstance(bucket.get("provider_adoption_gate"), dict) else {}
    if _as_int(bucket.get("safety_stop_count")):
        reasons.add(_CACHE_REPLAY_SAFETY_STOP_REASON)
        return "rollback recommended", sorted(reasons)
    replayed = _as_int(bucket.get("hit_count"))
    replayed_errors = _as_int(bucket.get("replayed_error_count"))
    if replayed and replayed_errors:
        reasons.add("replayed-errors-observed")
        return "rollback recommended", sorted(reasons)
    if (
        _as_int(bucket.get("invalidation_count"))
        or _as_int(bucket.get("bypass_count"))
        or str(provider.get("status") or "") == "blocked"
    ):
        if str(provider.get("status") or "") == "blocked":
            reasons.update(provider.get("reason_codes") or ["provider-adoption-regression"])
        return "hold", sorted(reasons or {"activation-blocked"})
    applied = _as_int(bucket.get("applied_count"))
    holdout = _as_int(bucket.get("holdout_count"))
    misses = _as_int(bucket.get("miss_count"))
    if replayed and holdout and not replayed_errors:
        reasons.add("replay-and-holdout-observed")
        return "widen candidate", sorted(reasons)
    if applied or replayed or misses:
        reasons.add("collect-hit-recovery-evidence" if not replayed else "canary-replay-observed")
        return "canary active", sorted(reasons)
    if holdout:
        reasons.add("holdout-only")
        return "hold", sorted(reasons)
    reasons.add("no-cache-replay-canary-evidence")
    return "needs evidence", sorted(reasons)


async def stats_cache_replay_activation_health(
    store_obj: Any,
    *,
    limit: int = 1000,
    scan_limit: int = 1000,
) -> dict[str, Any]:
    capped_scan = max(1, min(_as_int(scan_limit) or 1000, 10000))
    output_limit = max(1, min(_as_int(limit) or 50, 1000))
    rows = _cache_replay_confidence_rows_from_store(store_obj, limit=capped_scan)
    grouped: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    state_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}

    for row in rows:
        cache = _json_obj(row.get("cache_json"))
        if not cache:
            continue
        routing = _json_obj(row.get("routing_json"))
        decision = _cache_decision_for_breakdown(row)
        source_surface = canonical_source_surface(
            decision.get("source_surface") or row.get("source_surface") or "unknown"
        )
        category = public_label(row.get("category") or routing.get("category") or cache.get("category") or "unknown", "unknown")
        stream = bool(_as_int(row.get("stream")) or cache.get("stream"))
        has_tools = bool(routing.get("has_tools") or category.startswith("tool") or cache.get("has_tools"))
        workflow_phase = public_label(
            routing.get("workflow_phase")
            or cache.get("workflow_phase")
            or cache.get("phase")
            or "unknown",
            "unknown",
        )
        app_family = public_label(
            cache.get("app_family")
            or routing.get("app_family")
            or _app_family_for_call(row.get("provider"), row.get("requested_model"), row.get("path")),
            "unknown",
        )
        rules = _cache_pattern_rules_from_meta(cache)
        replay_canary = cache.get("cache_replay_canary") if isinstance(cache.get("cache_replay_canary"), dict) else {}
        if not rules and replay_canary:
            rules = [{
                "rule_id": replay_canary.get("rule_id"),
                "candidate_id": replay_canary.get("candidate_id"),
                "policy_source": replay_canary.get("policy_source"),
                "scope": replay_canary.get("scope"),
                "canary": replay_canary.get("canary"),
            }]
        if not rules:
            continue

        for rule in rules:
            identity = _cache_rule_identity(
                cache=cache,
                rule=rule,
                source_surface=source_surface,
                category=category,
                stream=stream,
                has_tools=has_tools,
            )
            replay_scope = public_label(
                (rule or {}).get("scope")
                or replay_canary.get("scope")
                or cache.get("replay_scope")
                or "unknown",
                "unknown",
            )
            provider_gate = _cache_replay_activation_provider_gate(cache, rule)
            key = (
                identity["rule_id"],
                str(identity.get("candidate_id") or ""),
                identity["policy_source"],
                identity["source_surface"],
                app_family,
                identity["category"],
                workflow_phase,
                replay_scope,
            )
            bucket = grouped.setdefault(
                key,
                {
                    **identity,
                    "app_family": app_family,
                    "workflow_phase": workflow_phase,
                    "replay_scope_class": replay_scope,
                    "granularities": set(),
                    "sample_count": 0,
                    "applied_count": 0,
                    "holdout_count": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "bypass_count": 0,
                    "invalidation_count": 0,
                    "safety_stop_count": 0,
                    "replayed_error_count": 0,
                    "estimated_saved_cost_usd": 0.0,
                    "projected_saved_cost_usd": 0.0,
                    "projected_hits": 0,
                    "canary_fraction": None,
                    "holdout_fraction": None,
                    "provider_adoption_gate": provider_gate,
                    "first_seen_at": row.get("created_at"),
                    "last_seen_at": row.get("created_at"),
                    "_reason_codes": set(),
                },
            )
            bucket["sample_count"] += 1
            bucket["granularities"].add(str(row.get("granularity") or "provider_request"))
            if str(provider_gate.get("status") or "") == "blocked":
                bucket["provider_adoption_gate"] = provider_gate
            elif str((bucket.get("provider_adoption_gate") or {}).get("status") or "") == "not-observed":
                bucket["provider_adoption_gate"] = provider_gate

            outcome = _cache_confidence_outcome(cache, decision, rule)
            if outcome == "hit":
                bucket["hit_count"] += 1
                bucket["applied_count"] += 1
            elif outcome == "miss":
                bucket["miss_count"] += 1
                bucket["applied_count"] += 1
            elif outcome == "holdout":
                bucket["holdout_count"] += 1
            elif outcome == "safety_stop":
                bucket["safety_stop_count"] += 1
            else:
                bucket["bypass_count"] += 1

            status_code = _as_int(row.get("status_code")) if row.get("status_code") is not None else None
            if outcome == "hit" and status_code is not None and status_code >= 400:
                bucket["replayed_error_count"] += 1

            invalidation, stale, safety = _cache_confidence_reason_counts(cache=cache, decision=decision, rule=rule)
            bucket["invalidation_count"] += sum(invalidation.values())
            bucket["safety_stop_count"] += sum(safety.values())
            for reason, count in {**invalidation, **stale, **safety}.items():
                if count:
                    bucket["_reason_codes"].add(str(reason))

            saved = _as_float(cache.get("estimated_saved_cost_usd"))
            if not saved and outcome == "hit":
                saved = max(_as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd")), 0.0)
            bucket["estimated_saved_cost_usd"] += saved
            bucket["projected_saved_cost_usd"] += _cache_replay_activation_projected_savings(cache, replay_canary, rule)
            bucket["projected_hits"] += _cache_replay_activation_projected_hits(cache, replay_canary, rule)

            canary = (rule or {}).get("canary") if isinstance(rule, dict) else None
            if not isinstance(canary, dict):
                canary = replay_canary.get("canary") if isinstance(replay_canary.get("canary"), dict) else cache.get("canary")
            rollout = (rule or {}).get("rollout") if isinstance(rule, dict) else cache.get("rollout")
            if isinstance(canary, dict):
                if bucket["canary_fraction"] is None and canary.get("fraction") is not None:
                    bucket["canary_fraction"] = canary.get("fraction")
                if bucket["holdout_fraction"] is None and canary.get("holdout_fraction") is not None:
                    bucket["holdout_fraction"] = canary.get("holdout_fraction")
            if isinstance(rollout, dict):
                if bucket["canary_fraction"] is None:
                    bucket["canary_fraction"] = rollout.get("canary_fraction") or rollout.get("rollout_fraction")
                if bucket["holdout_fraction"] is None:
                    bucket["holdout_fraction"] = rollout.get("holdout_fraction")
            if replay_canary.get("status") == "applied" or (isinstance(canary, dict) and canary.get("cohort") == "canary_applied"):
                if outcome not in {"hit", "miss"}:
                    bucket["applied_count"] += 1
            if replay_canary.get("reason"):
                bucket["_reason_codes"].add(public_label(replay_canary.get("reason"), "unknown"))
            if str(row.get("created_at") or "") < str(bucket.get("first_seen_at") or row.get("created_at") or ""):
                bucket["first_seen_at"] = row.get("created_at")
            if str(row.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
                bucket["last_seen_at"] = row.get("created_at")

    cohorts: list[dict[str, Any]] = []
    for bucket in grouped.values():
        state, reason_codes = _cache_replay_activation_state(bucket)
        bucket["state"] = state
        bucket["reason_codes"] = reason_codes
        bucket["granularities"] = sorted(bucket["granularities"])
        bucket["estimated_saved_cost_usd"] = round(_as_float(bucket.get("estimated_saved_cost_usd")), 8)
        bucket["projected_saved_cost_usd"] = round(_as_float(bucket.get("projected_saved_cost_usd")), 8)
        bucket["actual_hits"] = _as_int(bucket.get("hit_count"))
        bucket["actual_hit_recovery_rate"] = (
            round(_as_int(bucket.get("hit_count")) / max(_as_int(bucket.get("applied_count")), 1), 4)
            if _as_int(bucket.get("applied_count"))
            else 0.0
        )
        bucket["pattern_hashes_included"] = False
        bucket["cache_keys_included"] = False
        for key in list(bucket.keys()):
            if key.startswith("_"):
                bucket.pop(key, None)
        state_counts[state] = state_counts.get(state, 0) + 1
        provider_status = str((bucket.get("provider_adoption_gate") or {}).get("status") or "unknown")
        provider_counts[provider_status] = provider_counts.get(provider_status, 0) + 1
        for reason in reason_codes:
            blocker_counts[reason] = blocker_counts.get(reason, 0) + 1
        cohorts.append(bucket)

    cohorts.sort(
        key=lambda row: (
            row.get("state") == "rollback recommended",
            row.get("state") == "hold",
            row.get("state") == "widen candidate",
            _as_float(row.get("estimated_saved_cost_usd")),
            _as_int(row.get("sample_count")),
        ),
        reverse=True,
    )

    return {
        "schema": "tokenclaw.cache_replay_activation_health.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "status": "observed" if cohorts else "needs evidence",
        "summary": {
            "rows_considered": len(rows),
            "cohort_count": len(cohorts),
            "healthy_canary_count": state_counts.get("canary active", 0) + state_counts.get("widen candidate", 0),
            "blocked_or_hold_count": state_counts.get("hold", 0) + state_counts.get("rollback recommended", 0),
            "needs_evidence_count": state_counts.get("needs evidence", 0),
            "canary_active_count": state_counts.get("canary active", 0),
            "widen_candidate_count": state_counts.get("widen candidate", 0),
            "rollback_recommended_count": state_counts.get("rollback recommended", 0),
            "actual_hits": sum(_as_int(row.get("actual_hits")) for row in cohorts),
            "projected_hits": sum(_as_int(row.get("projected_hits")) for row in cohorts),
            "estimated_saved_cost_usd": round(sum(_as_float(row.get("estimated_saved_cost_usd")) for row in cohorts), 8),
            "projected_saved_cost_usd": round(sum(_as_float(row.get("projected_saved_cost_usd")) for row in cohorts), 8),
        },
        "state_breakdown": _breakdown_from_counts(state_counts),
        "provider_adoption_breakdown": _breakdown_from_counts(provider_counts),
        "blocker_breakdown": _breakdown_from_counts(blocker_counts),
        "cohorts": cohorts[:output_limit],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "provider_bodies_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "dashboard_read_only": True,
            "basis": "stored cache replay canary metadata, aggregate outcomes, dependency blocker codes, provider adoption gate summaries, and cost estimates only",
        },
    }


def _streaming_cache_rule_public_id(rule: dict[str, Any], cache: dict[str, Any], *, key: str, prefix: str) -> str | None:
    raw = rule.get(key) if isinstance(rule, dict) else None
    if raw is None and isinstance(cache, dict):
        raw = cache.get(key)
    return public_id(raw, prefix=prefix) if raw is not None else None


def _streaming_cache_cohort_label(rule: dict[str, Any], cache: dict[str, Any], replay_canary: dict[str, Any]) -> str:
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else None
    if not isinstance(canary, dict):
        canary = cache.get("canary") if isinstance(cache.get("canary"), dict) else None
    if not isinstance(canary, dict):
        canary = replay_canary.get("canary") if isinstance(replay_canary.get("canary"), dict) else {}
    return public_label(
        replay_canary.get("canary_cohort")
        or cache.get("canary_cohort")
        or canary.get("cohort")
        or "unassigned",
        "unassigned",
    )


def _streaming_cache_stage(
    *,
    row: dict[str, Any],
    cache: dict[str, Any],
    decision: dict[str, Any],
    rule: dict[str, Any],
    replay_canary: dict[str, Any],
) -> tuple[str, str]:
    outcome = _cache_confidence_outcome(cache, decision, rule)
    reason = public_label(
        replay_canary.get("reason")
        or rule.get("reason")
        or decision.get("reason")
        or cache.get("reason")
        or "unknown",
        "unknown",
    )
    status = public_label(replay_canary.get("status") or decision.get("status") or cache.get("status"), "unknown")
    if outcome == "hit" or _as_int(row.get("cache_hit")):
        return "hit", reason
    if outcome == "safety_stop" or status == "safety_stopped" or reason == _CACHE_REPLAY_SAFETY_STOP_REASON:
        return "safety_stop", _CACHE_REPLAY_SAFETY_STOP_REASON
    if outcome == "holdout" or status == "holdout" or reason == "canary_holdout":
        return "holdout", "canary_holdout"
    if outcome == "miss" or status == "applied":
        return "replay_attempt", reason
    if outcome == "skip" and status == "invalidated":
        return "invalidation", reason
    if _cache_ladder_dependency_blocked(cache, reason, public_label(decision.get("outcome_bucket"), "unknown")):
        return "invalidation", reason
    if status in {"bypassed", "blocked"}:
        return "replay_blocked", reason
    if rule.get("rule_id") or rule.get("candidate_id"):
        return "not_eligible", reason
    return "unknown", reason


def _streaming_cache_recovery_verdict(bucket: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = set(str(item) for item in bucket.get("_reason_codes", set()) if item)
    if _as_int(bucket.get("eligible_calls")) <= 0:
        reasons.add("no eligible streaming calls matched this active rule")
        return "no-eligible-traffic", sorted(reasons)
    if _as_int(bucket.get("successful_hits")) > 0:
        reasons.add("streaming cache hits recovered")
        return "hits-recovered", sorted(reasons)
    if _as_int(bucket.get("safety_stops")) > 0:
        reasons.add(_CACHE_REPLAY_SAFETY_STOP_REASON)
        return "safety-stopped", sorted(reasons)
    if _as_int(bucket.get("invalidations")) > 0:
        reasons.add("dependency invalidation blocked replay")
        return "invalidated", sorted(reasons)
    if _as_int(bucket.get("holdouts")) > 0 and _as_int(bucket.get("replay_attempts")) <= 0:
        reasons.add("canary holdout has no applied replay traffic")
        return "holdout-only", sorted(reasons)
    if _as_int(bucket.get("replay_attempts")) <= 0:
        reasons.add("eligible calls did not reach replay lookup")
        return "replay-blocked", sorted(reasons)
    reasons.add("replay lookup attempted but no cache hit recovered")
    return "store-missing", sorted(reasons)


def _streaming_cache_stored_response_count(cache: dict[str, Any]) -> int:
    for key in ("stream_cache_store", "cache_store", "store"):
        store_meta = cache.get(key) if isinstance(cache.get(key), dict) else None
        if not isinstance(store_meta, dict):
            continue
        if str(store_meta.get("status") or "").lower() in {"stored", "ok", "success"}:
            return max(1, _as_int(store_meta.get("entry_count")) or _as_int(store_meta.get("stored_count")) or 1)
    for key in ("stored_response", "response_stored", "stream_cache_stored"):
        if cache.get(key) is True:
            return 1
    return 0


async def stats_streaming_cache_hit_recovery(
    store_obj: Any,
    *,
    limit: int = 1000,
    scan_limit: int = 1000,
) -> dict[str, Any]:
    capped_scan = max(1, min(_as_int(scan_limit) or 1000, 10000))
    output_limit = max(1, min(_as_int(limit) or 50, 1000))
    rows = _cache_replay_confidence_rows_from_store(store_obj, limit=capped_scan)
    grouped: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    verdict_counts: dict[str, int] = {}
    bypass_counts: dict[str, int] = {}

    for row in rows:
        cache = _json_obj(row.get("cache_json"))
        if not cache:
            continue
        stream = bool(_as_int(row.get("stream")) or cache.get("stream"))
        if not stream:
            continue
        routing = _json_obj(row.get("routing_json"))
        decision = _cache_decision_for_breakdown(row)
        replay_canary = cache.get("cache_replay_canary") if isinstance(cache.get("cache_replay_canary"), dict) else {}
        rules = _cache_pattern_rules_from_meta(cache)
        if not rules and replay_canary:
            rules = [{
                "rule_id": replay_canary.get("rule_id"),
                "candidate_id": replay_canary.get("candidate_id"),
                "policy_source": replay_canary.get("policy_source"),
                "scope": replay_canary.get("scope"),
                "canary": replay_canary.get("canary"),
            }]
        if not rules:
            continue

        source_surface = canonical_source_surface(
            decision.get("source_surface")
            or cache.get("source_surface")
            or row.get("source_surface")
            or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        )
        category = public_label(row.get("category") or routing.get("category") or cache.get("category"), "unknown")
        workflow_phase = public_label(
            routing.get("workflow_phase")
            or cache.get("workflow_phase")
            or cache.get("phase")
            or "unknown",
            "unknown",
        )
        provider = public_label(row.get("provider") or "anthropic", "unknown")

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            stage, reason = _streaming_cache_stage(
                row=row,
                cache=cache,
                decision=decision,
                rule=rule,
                replay_canary=replay_canary,
            )
            policy_source = public_label(rule.get("policy_source") or cache.get("policy_source") or "unknown", "unknown")
            rule_id = _streaming_cache_rule_public_id(rule, cache, key="rule_id", prefix="rule-id") or "unruled-cache-decision"
            candidate_id = _streaming_cache_rule_public_id(rule, cache, key="candidate_id", prefix="candidate-id")
            policy_id = _streaming_cache_rule_public_id(rule, cache, key="policy_id", prefix="policy-id") or candidate_id or rule_id
            cohort = _streaming_cache_cohort_label(rule, cache, replay_canary)
            key = (
                provider,
                source_surface,
                category,
                workflow_phase,
                policy_source,
                policy_id,
                rule_id,
                cohort,
            )
            bucket = grouped.setdefault(
                key,
                {
                    "provider": provider,
                    "source_surface": source_surface,
                    "category": category,
                    "workflow_phase": workflow_phase,
                    "policy_source": policy_source,
                    "policy_id": policy_id,
                    "rule_id": rule_id,
                    "candidate_id": candidate_id,
                    "cohort_label": cohort,
                    "sample_count": 0,
                    "eligible_calls": 0,
                    "stored_responses": 0,
                    "replay_attempts": 0,
                    "successful_hits": 0,
                    "holdouts": 0,
                    "invalidations": 0,
                    "safety_stops": 0,
                    "replay_blocked": 0,
                    "bypass_reasons": {},
                    "first_seen_at": row.get("created_at"),
                    "last_seen_at": row.get("created_at"),
                    "_reason_codes": set(),
                },
            )
            bucket["sample_count"] += 1
            if stage != "not_eligible":
                bucket["eligible_calls"] += 1
            if stage == "hit":
                bucket["successful_hits"] += 1
                bucket["replay_attempts"] += 1
                bucket["stored_responses"] += 1
            elif stage == "replay_attempt":
                bucket["replay_attempts"] += 1
                bucket["stored_responses"] += _streaming_cache_stored_response_count(cache)
            elif stage == "holdout":
                bucket["holdouts"] += 1
            elif stage == "invalidation":
                bucket["invalidations"] += 1
            elif stage == "safety_stop":
                bucket["safety_stops"] += 1
            elif stage == "replay_blocked":
                bucket["replay_blocked"] += 1
            if stage in {"not_eligible", "replay_blocked", "invalidation", "safety_stop", "holdout"}:
                bucket["bypass_reasons"][reason] = bucket["bypass_reasons"].get(reason, 0) + 1
                bypass_counts[reason] = bypass_counts.get(reason, 0) + 1
            bucket["_reason_codes"].add(reason)
            if str(row.get("created_at") or "") < str(bucket.get("first_seen_at") or row.get("created_at") or ""):
                bucket["first_seen_at"] = row.get("created_at")
            if str(row.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
                bucket["last_seen_at"] = row.get("created_at")

    cohorts: list[dict[str, Any]] = []
    for bucket in grouped.values():
        verdict, reason_codes = _streaming_cache_recovery_verdict(bucket)
        bucket["recovery_verdict"] = verdict
        bucket["reason_codes"] = reason_codes
        bucket["hit_recovery_rate"] = (
            round(_as_int(bucket.get("successful_hits")) / max(_as_int(bucket.get("replay_attempts")), 1), 4)
            if _as_int(bucket.get("replay_attempts"))
            else 0.0
        )
        bucket["bypass_reasons"] = _breakdown_from_counts({
            str(reason): _as_int(count)
            for reason, count in (bucket.get("bypass_reasons") or {}).items()
        })
        bucket["cache_keys_included"] = False
        bucket["pattern_hashes_included"] = False
        for key in list(bucket.keys()):
            if key.startswith("_"):
                bucket.pop(key, None)
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        cohorts.append(bucket)

    verdict_rank = {
        "safety-stopped": 0,
        "hits-recovered": 1,
        "invalidated": 2,
        "store-missing": 3,
        "replay-blocked": 4,
        "holdout-only": 5,
        "no-eligible-traffic": 6,
    }
    cohorts.sort(
        key=lambda row: (
            verdict_rank.get(str(row.get("recovery_verdict")), 99),
            -_as_int(row.get("sample_count")),
            str(row.get("provider") or ""),
            str(row.get("source_surface") or ""),
        )
    )
    top = cohorts[0] if cohorts else None
    return {
        "schema": "tokenclaw.streaming_cache_hit_recovery.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "status": "observed" if cohorts else "needs evidence",
        "summary": {
            "rows_considered": len(rows),
            "streaming_rule_cohort_count": len(cohorts),
            "recovery_verdict": top.get("recovery_verdict") if top else "no-eligible-traffic",
            "eligible_calls": sum(_as_int(row.get("eligible_calls")) for row in cohorts),
            "stored_responses": sum(_as_int(row.get("stored_responses")) for row in cohorts),
            "replay_attempts": sum(_as_int(row.get("replay_attempts")) for row in cohorts),
            "successful_hits": sum(_as_int(row.get("successful_hits")) for row in cohorts),
            "holdouts": sum(_as_int(row.get("holdouts")) for row in cohorts),
            "invalidations": sum(_as_int(row.get("invalidations")) for row in cohorts),
            "safety_stops": sum(_as_int(row.get("safety_stops")) for row in cohorts),
            "replay_blocked": sum(_as_int(row.get("replay_blocked")) for row in cohorts),
        },
        "verdict_breakdown": _breakdown_from_counts(verdict_counts),
        "bypass_reason_breakdown": _breakdown_from_counts(bypass_counts),
        "cohorts": cohorts[:output_limit],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "provider_bodies_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "dashboard_read_only": True,
            "basis": "bounded local cache replay metadata for streaming rule cohorts only",
        },
    }


async def stats_cache_effectiveness(store_obj: Any, *, limit: int = 10, scan_limit: int = 5000) -> dict[str, Any]:
    return build_cache_smoke_diagnostic(store_obj, limit=limit, scan_limit=scan_limit)

def _size_bucket(value: Any) -> str:
    n = _as_int(value)
    if n <= 0:
        return "0"
    if n < 2_000:
        return "1_2k"
    if n < 8_000:
        return "2k_8k"
    if n < 32_000:
        return "8k_32k"
    if n < 128_000:
        return "32k_128k"
    return "128k_plus"


def _short_session_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:8] if text else None


def _cache_replayability_blockers(unit: dict[str, Any]) -> list[str]:
    cache = unit["cache"]
    reason = str(unit["cache_reason"] or "")
    category = str(unit.get("category") or "")
    cacheability = unit.get("cacheability") if isinstance(unit.get("cacheability"), dict) else {}
    blockers: set[str] = set()
    if unit.get("stream") or "streaming" in reason:
        blockers.add("streaming")
    if "tools-disabled" in reason or (unit.get("has_tools") and not bool(cache.get("tool_cache_enabled"))):
        blockers.add("tool-call-disabled")
    if (
        not bool(cache.get("semantic_enabled"))
        and (
            unit["cache_status"] == "skipped"
            or "semantic" in reason
            or "cache-disabled" in reason
        )
    ):
        blockers.add("semantic-cache-disabled")
    tool_like = bool(unit.get("has_tools")) or category.startswith("tool")
    if bool(cacheability.get("time_sensitive_hint")):
        blockers.add("current-state")
    if bool(cacheability.get("user_specific_hint")):
        blockers.add("user-specific")
    if str(unit.get("cacheability_bucket") or "") == "low":
        blockers.add("low-cacheability")
    audit = unit.get("file_dependency_audit") if isinstance(unit.get("file_dependency_audit"), dict) else {}
    if is_codex_turn_source_surface(str(unit.get("source_surface") or "")):
        blockers.add("turn-level-only")
    if unit["cache_status"] == "missing":
        blockers.add("missing-cache-metadata")
    if tool_like:
        audit_reason = audit.get("invalidation_reason")
        if audit.get("cap_exceeded"):
            blockers.add("dependency-cap-exceeded")
        elif audit_reason in {
            "file-dependency-missing",
            "dependency-missing",
            "dependency-changed",
            "dependency-deleted",
            "dependency-created",
            "file-watch-disabled",
        }:
            blockers.add(str(audit_reason))
        elif not bool(unit.get("safe_invalidation_evidence")):
            blockers.add("file-dependency-missing")
    return sorted(blockers)


_CACHE_REPLAY_DEPENDENCY_STALE_REASONS = {
    "dependency-cap-exceeded",
    "dependency-changed",
    "dependency-created",
    "dependency-deleted",
    "dependency-missing",
    "file-dependency-missing",
    "file-watch-disabled",
}


def _cache_replay_dependency_freshness(unit: dict[str, Any]) -> dict[str, Any]:
    """Classify stored dependency metadata without exposing paths or file stats."""
    has_tools = bool(unit.get("has_tools")) or str(unit.get("category") or "").startswith("tool")
    audit = unit.get("file_dependency_audit") if isinstance(unit.get("file_dependency_audit"), dict) else {}
    reason = str(audit.get("invalidation_reason") or "")
    safe = bool(unit.get("safe_invalidation_evidence") or audit.get("safe_invalidation_evidence"))
    evidence = bool(unit.get("file_dependency_evidence_available") or audit.get("file_dependency_evidence_available") or safe)

    if not has_tools:
        status = "not-required"
        freshness_reason = "not-required"
    elif audit.get("cap_exceeded"):
        status = "stale"
        freshness_reason = "dependency-cap-exceeded"
    elif reason in _CACHE_REPLAY_DEPENDENCY_STALE_REASONS:
        status = "stale" if reason in {"dependency-changed", "dependency-created", "dependency-deleted", "dependency-cap-exceeded"} else "unknown"
        freshness_reason = reason
    elif safe:
        status = "fresh"
        freshness_reason = "dependency-fresh"
    elif evidence:
        status = "unknown"
        freshness_reason = "dependency-freshness-unknown"
    else:
        status = "unknown"
        freshness_reason = "file-dependency-evidence-absent"

    return {
        "schema": "tokenclaw.cache_replay_dependency_freshness.v1",
        "status": status,
        "reason": freshness_reason,
        "tool_call_cohort": has_tools,
        "evidence_available": evidence,
        "safe_invalidation_evidence": safe,
        "snapshot_count_bucket": str(audit.get("snapshot_count_bucket") or "unknown"),
        "candidate_path_count_bucket": str(audit.get("candidate_path_count_bucket") or "unknown"),
        "raw_candidate_path_count_bucket": str(audit.get("raw_candidate_path_count_bucket") or "unknown"),
        "distinct_candidate_path_count_bucket": str(audit.get("distinct_candidate_path_count_bucket") or "unknown"),
        "cap_exceeded": bool(audit.get("cap_exceeded")),
        "cap_trimmed": bool(audit.get("cap_trimmed")),
        "dependency_capture_reason": public_label(audit.get("dependency_capture_reason") or "unknown", "unknown"),
        "paths_included": False,
        "root_path_included": False,
        "raw_stat_values_included": False,
    }


def _cache_replay_cost_bucket(cost: Any) -> str:
    value = _as_float(cost)
    if value <= 0:
        return "none"
    if value < 0.01:
        return "lt_1c"
    if value < 0.05:
        return "1c_5c"
    if value < 0.25:
        return "5c_25c"
    if value < 1.0:
        return "25c_1usd"
    return "gte_1usd"


def _cache_replay_next_action(blockers: list[str]) -> dict[str, str]:
    blocker_set = set(blockers)
    dependency_blockers = {
        "file-dependency-missing",
        "dependency-missing",
        "dependency-cap-exceeded",
        "dependency-changed",
        "dependency-deleted",
        "dependency-created",
        "file-watch-disabled",
    }
    stale_blockers = {"current-state", "user-specific", "low-cacheability"}
    if "tool-call-disabled" in blocker_set:
        return {
            "family": "tool_call_safety",
            "label": "Review tool-call cache safety and invalidation gates",
        }
    if dependency_blockers & blocker_set:
        return {
            "family": "dependency_evidence",
            "label": "Capture or validate file dependency evidence",
        }
    if "streaming" in blocker_set or "streaming-not-allowed" in blocker_set:
        return {
            "family": "streaming_replay",
            "label": "Canary session-scoped streaming replay",
        }
    if "semantic-cache-disabled" in blocker_set:
        return {
            "family": "canary_policy_loading",
            "label": "Load a reviewed cache replay policy or canary",
        }
    if "session-context-changed" in blocker_set:
        return {
            "family": "canary_policy_loading",
            "label": "Load a reviewed session-scoped replay policy",
        }
    if "missing-cache-metadata" in blocker_set:
        return {
            "family": "cache_metadata",
            "label": "Record cache decision metadata before replay",
        }
    if "turn-level-only" in blocker_set:
        return {
            "family": "turn_level_replay",
            "label": "Promote turn-level replay interfaces",
        }
    if stale_blockers & blocker_set:
        return {
            "family": "cacheability_evidence",
            "label": "Improve cacheability evidence before replay",
        }
    if "true-one-off-miss" in blocker_set:
        return {
            "family": "none",
            "label": "No repeated-shape burn-down action",
        }
    return {
        "family": "canary_policy_loading",
        "label": "Load a reviewed exact replay canary policy",
    }


def _cache_replay_blocker_burn_down(groups: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[tuple[str, ...], str, str, str, str], dict[str, Any]] = {}
    for group in groups:
        if not bool(group.get("repeated")):
            continue
        blockers = sorted(str(item) for item in group.get("replayability_blockers") or [])
        if not blockers:
            blockers = ["none"]
        next_action = _cache_replay_next_action(blockers)
        key = (
            tuple(blockers),
            str(group.get("source_surface") or "unknown"),
            str(group.get("workflow_phase") or "unknown"),
            str(group.get("category") or "unknown"),
            next_action["family"],
        )
        bucket = buckets.setdefault(
            key,
            {
                "blockers": blockers,
                "blocker_combination": " + ".join(blockers),
                "next_action_family": next_action["family"],
                "next_action_label": next_action["label"],
                "source_surface": str(group.get("source_surface") or "unknown"),
                "granularity": str(group.get("granularity") or "unknown"),
                "workflow_phase": str(group.get("workflow_phase") or "unknown"),
                "category": str(group.get("category") or "unknown"),
                "replay_candidate_classes": {},
                "shape_groups": 0,
                "calls": 0,
                "sessions": 0,
                "estimated_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "projected_repeated_call_cost_usd": 0.0,
                "projected_cost_bucket": "none",
                "example_shape_fingerprints": [],
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "file_paths_included": False,
                "cache_keys_included": False,
                "raw_session_ids_included": False,
            },
        )
        bucket["shape_groups"] += 1
        bucket["calls"] += _as_int(group.get("count"))
        bucket["sessions"] += _as_int(group.get("sessions"))
        bucket["estimated_cost_usd"] += _as_float(group.get("estimated_cost_usd"))
        bucket["baseline_cost_usd"] += _as_float(group.get("baseline_cost_usd"))
        bucket["projected_repeated_call_cost_usd"] += _as_float(group.get("projected_repeated_call_cost_usd"))
        replay_class = str(group.get("replay_candidate_class") or "unknown")
        bucket["replay_candidate_classes"][replay_class] = bucket["replay_candidate_classes"].get(replay_class, 0) + 1
        fingerprint = str(group.get("replay_fingerprint") or "")
        if fingerprint and fingerprint not in bucket["example_shape_fingerprints"] and len(bucket["example_shape_fingerprints"]) < 3:
            bucket["example_shape_fingerprints"].append(fingerprint)

    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        replay_classes = [
            {"value": key, "count": count}
            for key, count in sorted(
                bucket.pop("replay_candidate_classes").items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        projected_cost = float(bucket["projected_repeated_call_cost_usd"])
        bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]), 6)
        bucket["baseline_cost_usd"] = round(float(bucket["baseline_cost_usd"]), 6)
        bucket["projected_repeated_call_cost_usd"] = round(projected_cost, 6)
        bucket["projected_cost_bucket"] = _cache_replay_cost_bucket(projected_cost)
        bucket["replay_candidate_classes"] = replay_classes
        rows.append(bucket)

    rows.sort(
        key=lambda row: (
            _as_float(row.get("projected_repeated_call_cost_usd")),
            _as_float(row.get("estimated_cost_usd")),
            _as_int(row.get("calls")),
            str(row.get("blocker_combination") or ""),
        ),
        reverse=True,
    )
    return rows[: max(1, int(limit or 25))]


def _cacheability_meta_from_row(
    *,
    routing: dict[str, Any],
    crunch: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for source in (
        routing.get("managed_pattern_features") if isinstance(routing.get("managed_pattern_features"), dict) else {},
        cache.get("managed_pattern_features") if isinstance(cache.get("managed_pattern_features"), dict) else {},
        cache.get("cacheability") if isinstance(cache.get("cacheability"), dict) else {},
    ):
        if isinstance(source, dict):
            candidates.append(source)

    pattern_modules = crunch.get("pattern_modules") if isinstance(crunch.get("pattern_modules"), dict) else {}
    server_features = pattern_modules.get("server_features") if isinstance(pattern_modules.get("server_features"), dict) else {}
    for entry in server_features.get("features") or []:
        if not isinstance(entry, dict) or entry.get("family") != "cacheability":
            continue
        features = entry.get("features") if isinstance(entry.get("features"), dict) else {}
        if features:
            candidates.append(features)

    result = {
        "bucket": "unknown",
        "deterministic_answer_likelihood_bucket": "unknown",
        "static_information_hint": False,
        "time_sensitive_hint": False,
        "user_specific_hint": False,
        "exact_cache_candidate_hint": False,
        "preserved_by_default_reason": None,
        "metadata_available": False,
    }
    for candidate in candidates:
        bucket = candidate.get("cacheability_bucket")
        if isinstance(bucket, str) and bucket:
            result["bucket"] = bucket
            result["metadata_available"] = True
        deterministic = candidate.get("deterministic_answer_likelihood_bucket")
        if isinstance(deterministic, str) and deterministic:
            result["deterministic_answer_likelihood_bucket"] = deterministic
            result["metadata_available"] = True
        for key in (
            "static_information_hint",
            "time_sensitive_hint",
            "user_specific_hint",
            "exact_cache_candidate_hint",
        ):
            if candidate.get(key) is not None:
                result[key] = bool(candidate.get(key))
                result["metadata_available"] = True
        reason = candidate.get("cache_preserved_by_default_reason")
        if isinstance(reason, str) and reason:
            result["preserved_by_default_reason"] = reason
            result["metadata_available"] = True
    return result


_CACHE_PATTERN_SAFE_FEATURE_KEYS = (
    "pattern_hashes",
    "pattern_hash",
    "normalized_pattern_hash",
    "cache_pattern_hash",
    "source_surface",
    "app_family",
    "category",
    "workflow_phase",
    "text_bucket",
    "token_bucket",
    "requested_model",
    "candidate_target_model",
    "replayability_level",
    "has_tools",
    "stream",
    "request_fingerprint",
    "session_id_hash",
    "workflow_id_hash",
)


def _cache_pattern_features_from_row(
    *,
    source_surface: str,
    routing: dict[str, Any],
    cache: dict[str, Any],
    category: str,
    workflow_phase: str,
    requested_model: str,
    routed_model: str,
    stream: bool,
    has_tools: bool,
    text_chars: int,
    input_tokens: int,
    replayability_level: str,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for source in (
        routing.get("managed_pattern_features") if isinstance(routing.get("managed_pattern_features"), dict) else {},
        cache.get("managed_pattern_features") if isinstance(cache.get("managed_pattern_features"), dict) else {},
        cache.get("pattern_features") if isinstance(cache.get("pattern_features"), dict) else {},
    ):
        if not isinstance(source, dict):
            continue
        for key in _CACHE_PATTERN_SAFE_FEATURE_KEYS:
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                features[key] = value
            elif isinstance(value, list):
                features[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
    features.setdefault("source_surface", canonical_source_surface(source_surface))
    features.setdefault("category", category)
    features.setdefault("workflow_phase", workflow_phase)
    features.setdefault("requested_model", requested_model)
    features.setdefault("candidate_target_model", routed_model)
    features.setdefault("replayability_level", replayability_level)
    features.setdefault("has_tools", bool(has_tools))
    features.setdefault("stream", bool(stream))
    features.setdefault("text_bucket", f"{_size_bucket(text_chars)}_chars")
    features.setdefault("token_bucket", f"{_size_bucket(input_tokens)}_tokens")
    features["raw_pattern_strings_included"] = False
    return features


def _file_dependency_count(cache: dict[str, Any]) -> int:
    if cache.get("file_dependency_count") is not None:
        return _as_int(cache.get("file_dependency_count"))
    file_dependencies = cache.get("file_dependencies")
    if isinstance(file_dependencies, list):
        return len(file_dependencies)
    return 0


def _cache_file_dependency_audit_from_cache(cache: dict[str, Any]) -> dict[str, Any]:
    audit = cache.get("file_dependency_audit")
    if isinstance(audit, dict):
        safe = bool(audit.get("safe_invalidation_evidence"))
        return {
            "schema": str(audit.get("schema") or "tokenclaw.cache_file_dependency_audit.v1"),
            "file_watch_enabled": bool(audit.get("file_watch_enabled")),
            "snapshot_root_policy": str(audit.get("snapshot_root_policy") or "unknown"),
            "root_path_included": False,
            "snapshot_count": _as_int(audit.get("snapshot_count")),
            "snapshot_count_bucket": str(audit.get("snapshot_count_bucket") or _size_bucket(audit.get("snapshot_count"))),
            "candidate_path_count_bucket": str(audit.get("candidate_path_count_bucket") or "unknown"),
            "raw_candidate_path_count_bucket": str(audit.get("raw_candidate_path_count_bucket") or "unknown"),
            "distinct_candidate_path_count_bucket": str(
                audit.get("distinct_candidate_path_count_bucket") or audit.get("candidate_path_count_bucket") or "unknown"
            ),
            "max_paths": audit.get("max_paths"),
            "cap_exceeded": bool(audit.get("cap_exceeded")),
            "cap_trimmed": bool(audit.get("cap_trimmed")),
            "dependency_capture_reason": audit.get("dependency_capture_reason"),
            "present_path_count": _as_int(audit.get("present_path_count")),
            "missing_path_count": _as_int(audit.get("missing_path_count")),
            "changed_path_count": _as_int(audit.get("changed_path_count")),
            "deleted_path_count": _as_int(audit.get("deleted_path_count")),
            "created_path_count": _as_int(audit.get("created_path_count")),
            "invalidation_reason": audit.get("invalidation_reason"),
            "safe_invalidation_evidence": safe,
            "file_dependency_evidence_available": bool(audit.get("file_dependency_evidence_available") or safe),
            "paths_included": False,
        }
    count = _file_dependency_count(cache)
    evidence = bool(count > 0 or cache.get("file_dependency_evidence_available") or cache.get("safe_invalidation_evidence") or cache.get("safe_invalidation"))
    reason = cache.get("invalidation_reason")
    if reason is None and not evidence:
        reason = "file-dependency-missing"
    return {
        "schema": "tokenclaw.cache_file_dependency_audit.v1",
        "file_watch_enabled": bool(cache.get("file_watch_enabled")),
        "snapshot_root_policy": "unknown",
        "root_path_included": False,
        "snapshot_count": count,
        "snapshot_count_bucket": _size_bucket(count),
        "candidate_path_count_bucket": _size_bucket(count),
        "raw_candidate_path_count_bucket": _size_bucket(count),
        "distinct_candidate_path_count_bucket": _size_bucket(count),
        "max_paths": None,
        "cap_exceeded": False,
        "cap_trimmed": False,
        "dependency_capture_reason": "complete",
        "present_path_count": count,
        "missing_path_count": 0,
        "changed_path_count": 1 if reason in {"dependency-changed", "file-dependency-changed"} else 0,
        "deleted_path_count": 1 if reason == "dependency-deleted" else 0,
        "created_path_count": 0,
        "invalidation_reason": reason,
        "safe_invalidation_evidence": bool(cache.get("safe_invalidation_evidence") or cache.get("safe_invalidation")),
        "file_dependency_evidence_available": evidence,
        "paths_included": False,
    }


def _cache_replayability_unit(row: dict[str, Any], *, source_surface: str, granularity: str) -> dict[str, Any] | None:
    decision = _cache_decision_for_breakdown({**row, "source_surface": source_surface})
    raw_status = str(decision.get("status") or "missing")
    if raw_status == "hit":
        return None
    cache = _json_obj(row.get("cache_json"))
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    text_chars = _as_int(row.get("text_chars") if row.get("text_chars") is not None else routing.get("text_chars"))
    input_tokens = _as_int(row.get("input_tokens") if row.get("input_tokens") is not None else row.get("input_tokens_est"))
    if not text_chars and input_tokens:
        text_chars = input_tokens * TOKEN_CHARS
    category = public_label(row.get("category") or routing.get("category") or "unknown", "unknown")
    requested_model = public_label(row.get("requested_model") or "", "unknown")
    routed_model = public_label(row.get("routed_model") or requested_model, requested_model or "unknown")
    replayability_level = public_label(
        cache.get("replayability_level") or ("features_only" if granularity == "agent_turn" else "metadata_shape"),
        "metadata_shape",
    )
    pattern_diagnostics = routing.get("managed_pattern_features") if isinstance(routing.get("managed_pattern_features"), dict) else {}
    cacheability = _cacheability_meta_from_row(routing=routing, crunch=crunch, cache=cache)
    workflow_phase = public_label(
        cache.get("workflow_phase")
        or routing.get("workflow_phase")
        or pattern_diagnostics.get("workflow_phase")
        or category
        or "unknown",
        "unknown",
    )
    has_tools = bool(routing.get("has_tools") or category.startswith("tool"))
    file_dependency_audit = _cache_file_dependency_audit_from_cache(cache)
    file_dependency_count = _as_int(file_dependency_audit.get("snapshot_count"))
    file_dependency_evidence_available = bool(file_dependency_audit.get("file_dependency_evidence_available"))
    session_memory_hints = cache.get("session_memory_hints") if isinstance(cache.get("session_memory_hints"), dict) else {}
    session_memory_proposal = (
        session_memory_hints.get("dry_run_replay_proposal")
        if isinstance(session_memory_hints.get("dry_run_replay_proposal"), dict)
        else None
    )
    provider = public_label(row.get("provider") or ("codex-app" if is_codex_turn_source_surface(source_surface) else "unknown"), "unknown")
    endpoint = public_label(row.get("endpoint") or _endpoint_label(provider, str(row.get("path") or "")), "unknown")
    unit = {
        "source_surface": canonical_source_surface(source_surface),
        "granularity": granularity,
        "provider": provider,
        "endpoint": endpoint,
        "created_at": row.get("created_at"),
        "session_id": row.get("session_id"),
        "stream": bool(_as_int(row.get("stream"))),
        "cache_status": public_label(raw_status, "unknown"),
        "cache_reason": public_label(decision.get("reason") or "unknown", "unknown"),
        "hit_type": public_label(decision.get("hit_type") or "", ""),
        "policy_source": public_label(decision.get("policy_source") or "unknown", "unknown"),
        "category": category,
        "workflow_phase": workflow_phase,
        "requested_model": requested_model,
        "routed_model": routed_model,
        "requested_tier": model_tier(requested_model) if requested_model else "unknown",
        "target_tier": model_tier(routed_model) if routed_model else "unknown",
        "has_tools": has_tools,
        "eligible": bool(cache.get("eligible")),
        "replayability_level": replayability_level,
        "cacheability": cacheability,
        "cacheability_bucket": cacheability["bucket"],
        "file_dependency_evidence_available": file_dependency_evidence_available,
        "file_dependency_count": file_dependency_count,
        "safe_invalidation_evidence": bool(file_dependency_audit.get("safe_invalidation_evidence")),
        "file_dependency_audit": file_dependency_audit,
        "session_memory_replay_proposal": session_memory_proposal,
        "text_size_bucket": _size_bucket(text_chars),
        "input_items_bucket": _size_bucket(row.get("input_items")),
        "cost_est_usd": _as_float(row.get("cost_est_usd")),
        "baseline_cost_usd": _as_float(row.get("cost_baseline_usd")),
        "input_tokens": input_tokens,
        "text_chars": text_chars,
        "cache": cache,
    }
    unit["dependency_freshness"] = _cache_replay_dependency_freshness(unit)
    unit["pattern_features"] = _cache_pattern_features_from_row(
        source_surface=source_surface,
        routing=routing,
        cache=cache,
        category=category,
        workflow_phase=workflow_phase,
        requested_model=requested_model,
        routed_model=routed_model,
        stream=bool(_as_int(row.get("stream"))),
        has_tools=has_tools,
        text_chars=text_chars,
        input_tokens=input_tokens,
        replayability_level=replayability_level,
    )
    unit["blockers"] = _cache_replayability_blockers(unit)
    return unit


def _cache_replayability_fingerprint_basis(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_surface": unit["source_surface"],
        "granularity": unit["granularity"],
        "provider": unit.get("provider", "unknown"),
        "endpoint": unit.get("endpoint", "unknown"),
        "cache_status": unit["cache_status"],
        "cache_reason": unit["cache_reason"],
        "category": unit["category"],
        "workflow_phase": unit["workflow_phase"],
        "stream": unit["stream"],
        "has_tools": unit["has_tools"],
        "requested_tier": unit["requested_tier"],
        "target_tier": unit["target_tier"],
        "text_size_bucket": unit["text_size_bucket"],
        "input_items_bucket": unit["input_items_bucket"],
        "replayability_level": unit["replayability_level"],
        "cacheability_bucket": unit["cacheability_bucket"],
        "current_state": bool((unit.get("cacheability") or {}).get("time_sensitive_hint")),
        "user_specific": bool((unit.get("cacheability") or {}).get("user_specific_hint")),
        "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
        "file_dependency_audit_reason": (
            (unit.get("file_dependency_audit") or {}).get("invalidation_reason")
            if isinstance(unit.get("file_dependency_audit"), dict)
            else None
        ),
        "file_dependency_snapshot_count_bucket": (
            (unit.get("file_dependency_audit") or {}).get("snapshot_count_bucket")
            if isinstance(unit.get("file_dependency_audit"), dict)
            else "unknown"
        ),
        "eligible": unit["eligible"],
    }


def _cache_replayability_fingerprint(unit: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    basis = _cache_replayability_fingerprint_basis(unit)
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], basis


def _public_session_memory_replay_proposal(proposal: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(proposal, dict):
        return None
    fingerprint = str(proposal.get("proposal_fingerprint") or "")
    if not fingerprint.startswith("sha256:"):
        return None
    blockers = [str(item) for item in proposal.get("blockers") or [] if isinstance(item, (str, int, float, bool))]
    review_steps = [
        str(item)
        for item in proposal.get("review_steps") or []
        if isinstance(item, (str, int, float, bool))
    ][:8]
    families = proposal.get("blocker_families") if isinstance(proposal.get("blocker_families"), dict) else {}
    privacy = proposal.get("privacy") if isinstance(proposal.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.session_memory_cache_replay_proposal.v1",
        "status": public_label(proposal.get("status") or "unknown", "unknown"),
        "reason": public_label(proposal.get("reason") or "unknown", "unknown"),
        "proposal_id": f"session-memory-cache-replay:{fingerprint.removeprefix('sha256:')}",
        "proposal_fingerprint": fingerprint,
        "rule_id": public_id(
            proposal.get("rule_id") or "local-session-plateau-cache-hint",
            prefix="rule-id",
            fallback="local-session-plateau-cache-hint",
        ),
        "policy_source": public_label(proposal.get("policy_source") or "unknown", "unknown"),
        "policy_rule_path_included": False,
        "phase": public_label(proposal.get("phase") or "unknown", "unknown"),
        "category": public_label(proposal.get("category") or "unknown", "unknown"),
        "stream": bool(proposal.get("stream")),
        "has_tool_blocks": bool(proposal.get("has_tool_blocks")),
        "thinking_present": bool(proposal.get("thinking_present")),
        "text_size_bucket": str(proposal.get("text_size_bucket") or "unknown"),
        "projected_tokens_saved_est": _as_int(proposal.get("projected_tokens_saved_est")),
        "projected_savings_bucket": str(proposal.get("projected_savings_bucket") or "none"),
        "projected_cost_savings_bucket": str(proposal.get("projected_cost_savings_bucket") or "none"),
        "blockers": sorted({public_label(blocker, "unknown") for blocker in blockers}),
        "blocker_families": {
            key: bool(families.get(key))
            for key in (
                "streaming",
                "tool",
                "thinking",
                "safe_invalidation",
                "reviewed_pattern_rule",
                "session_memory",
                "quality",
            )
        },
        "review_steps": review_steps,
        "mutation_applied": False,
        "cache_mutation": False,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "provider_calls_made": 0,
        "managed_server_calls_made": 0,
        "dry_run": True,
        "privacy": {
            "metadata_only": bool(privacy.get("metadata_only", True)),
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "pattern_hashes_included": False,
        },
    }


def _session_memory_replay_proposal_rows(units: list[dict[str, Any]], *, limit: int = 25) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for unit in units:
        proposal = _public_session_memory_replay_proposal(
            unit.get("session_memory_replay_proposal")
            if isinstance(unit.get("session_memory_replay_proposal"), dict)
            else {}
        )
        if proposal is None:
            continue
        key = (
            str(proposal.get("proposal_fingerprint")),
            str(proposal.get("status")),
            str(proposal.get("rule_id")),
        )
        bucket = grouped.setdefault(
            key,
            {
                **proposal,
                "count": 0,
                "estimated_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "projected_repeated_call_cost_usd": 0.0,
                "first_seen_at": unit.get("created_at"),
                "last_seen_at": unit.get("created_at"),
                "_blockers": set(proposal.get("blockers") or []),
                "_review_steps": list(proposal.get("review_steps") or []),
            },
        )
        bucket["count"] += 1
        bucket["estimated_cost_usd"] += _as_float(unit.get("cost_est_usd"))
        bucket["baseline_cost_usd"] += _as_float(unit.get("baseline_cost_usd"))
        for blocker in proposal.get("blockers") or []:
            bucket["_blockers"].add(str(blocker))
        for step in proposal.get("review_steps") or []:
            if step not in bucket["_review_steps"] and len(bucket["_review_steps"]) < 8:
                bucket["_review_steps"].append(step)
        if str(unit.get("created_at") or "") < str(bucket.get("first_seen_at") or unit.get("created_at") or ""):
            bucket["first_seen_at"] = unit.get("created_at")
        if str(unit.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = unit.get("created_at")

    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        count = _as_int(bucket.get("count"))
        if count > 1:
            avg_cost = float(bucket["estimated_cost_usd"]) / count
            bucket["projected_repeated_call_cost_usd"] = max(0.0, float(bucket["estimated_cost_usd"]) - avg_cost)
        bucket["blockers"] = sorted(bucket.pop("_blockers"))
        bucket["review_steps"] = bucket.pop("_review_steps")
        bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]), 6)
        bucket["baseline_cost_usd"] = round(float(bucket["baseline_cost_usd"]), 6)
        bucket["projected_repeated_call_cost_usd"] = round(float(bucket["projected_repeated_call_cost_usd"]), 6)
        rows.append(bucket)
    rows.sort(
        key=lambda row: (
            row.get("status") == "session-plateau-dry-run-eligible",
            _as_float(row.get("projected_repeated_call_cost_usd")),
            _as_int(row.get("count")),
            _as_float(row.get("estimated_cost_usd")),
        ),
        reverse=True,
    )
    return rows[: max(1, int(limit or 25))]


_CACHE_REPLAY_EVIDENCE_BLOCKER_CODES = {
    "streaming": "streaming-response-cache-missing",
    "streaming-not-allowed": "streaming-response-cache-missing",
    "tool-call-disabled": "tool-cache-invalidator-missing",
    "safe-invalidation-required": "tool-cache-invalidator-missing",
    "file-dependency-evidence-absent": "tool-cache-invalidator-missing",
    "file-dependency-missing": "tool-cache-invalidator-missing",
    "dependency-missing": "tool-cache-invalidator-missing",
    "dependency-cap-exceeded": "tool-cache-invalidator-missing",
    "dependency-changed": "tool-cache-invalidator-missing",
    "dependency-created": "tool-cache-invalidator-missing",
    "dependency-deleted": "tool-cache-invalidator-missing",
    "file-watch-disabled": "tool-cache-invalidator-missing",
    "true-one-off-miss": "no-repeat-normalized-shape",
    "session-context-changed": "session-scoped-shape-changed",
    "semantic-cache-disabled": "reviewed-cache-replay-policy-missing",
    "missing-cache-metadata": "cache-decision-metadata-missing",
    "turn-level-only": "turn-level-cache-replay-interface-missing",
    "current-state": "time-sensitive-shape",
    "user-specific": "user-specific-shape",
    "low-cacheability": "low-cacheability-shape",
}


def _cache_replay_evidence_blocker_codes(group: dict[str, Any]) -> list[str]:
    blockers = [str(item) for item in group.get("replayability_blockers") or []]
    codes = {
        _CACHE_REPLAY_EVIDENCE_BLOCKER_CODES.get(blocker, blocker)
        for blocker in blockers
        if blocker
    }
    if not codes and _as_int(group.get("count")) <= 1:
        codes.add("no-repeat-normalized-shape")
    if bool(group.get("stream")):
        codes.add("streaming-response-cache-missing")
    if bool(group.get("has_tools")) and not bool(group.get("safe_invalidation_evidence")):
        codes.add("tool-cache-invalidator-missing")
    return sorted(codes)


def _cache_replay_evidence_status(group: dict[str, Any], blocker_codes: list[str]) -> str:
    if not blocker_codes and bool(group.get("repeated")):
        return "safe-replayable-cohort"
    if "no-repeat-normalized-shape" in blocker_codes:
        return "not-repeated"
    if "streaming-response-cache-missing" in blocker_codes:
        return "blocked-streaming-replay"
    if "tool-cache-invalidator-missing" in blocker_codes:
        return "blocked-tool-invalidation"
    return "blocked"


def _cache_replayability_evidence_from_report(
    *,
    groups: list[dict[str, Any]],
    summary: dict[str, Any],
    blocker_breakdown: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    safe_count = 0
    for group in groups:
        blocker_codes = _cache_replay_evidence_blocker_codes(group)
        status = _cache_replay_evidence_status(group, blocker_codes)
        if status == "safe-replayable-cohort":
            safe_count += 1
        for code in blocker_codes or ["none"]:
            blocker_counts[code] = blocker_counts.get(code, 0) + _as_int(group.get("count"))
        dependency = (
            group.get("file_dependency_audit")
            if isinstance(group.get("file_dependency_audit"), dict)
            else {}
        )
        rows.append({
            "rank": 0,
            "status": status,
            "provider": public_label(group.get("provider") or "unknown", "unknown"),
            "endpoint": public_label(group.get("endpoint") or "unknown", "unknown"),
            "source_surface": public_label(group.get("source_surface") or "unknown", "unknown"),
            "granularity": public_label(group.get("granularity") or "unknown", "unknown"),
            "shape_fingerprint": f"sha256:{group.get('shape_fingerprint')}",
            "request_shape": {
                "category": public_label(group.get("category") or "unknown", "unknown"),
                "workflow_phase": public_label(group.get("workflow_phase") or "unknown", "unknown"),
                "text_size_bucket": str(group.get("text_size_bucket") or "unknown"),
                "input_items_bucket": str(group.get("input_items_bucket") or "unknown"),
                "requested_tier": public_label(group.get("requested_tier") or "unknown", "unknown"),
                "target_tier": public_label(group.get("target_tier") or "unknown", "unknown"),
                "cacheability_bucket": public_label(group.get("cacheability_bucket") or "unknown", "unknown"),
            },
            "stream": bool(group.get("stream")),
            "has_tools": bool(group.get("has_tools")),
            "dependency_freshness_state": (
                "fresh"
                if bool(group.get("safe_invalidation_evidence"))
                else (
                    "not-required"
                    if not bool(group.get("has_tools"))
                    else public_label(dependency.get("invalidation_reason") or "unknown", "unknown")
                )
            ),
            "cache_decision_reason": public_label(group.get("cache_reason") or "unknown", "unknown"),
            "cache_decision_status": public_label(group.get("cache_status") or "unknown", "unknown"),
            "calls": _as_int(group.get("count")),
            "sessions": _as_int(group.get("sessions")),
            "repeated": bool(group.get("repeated")),
            "replayability_level": public_label(group.get("replayability_level") or "unknown", "unknown"),
            "blocker_codes": blocker_codes,
            "estimated_avoided_cost_usd": round(_as_float(group.get("projected_repeated_call_cost_usd")), 6),
            "estimated_cohort_cost_usd": round(_as_float(group.get("estimated_cost_usd")), 6),
            "aggregate_only": True,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
        })

    rows.sort(
        key=lambda row: (
            row["status"] == "safe-replayable-cohort",
            _as_float(row.get("estimated_avoided_cost_usd")),
            _as_int(row.get("calls")),
            _as_float(row.get("estimated_cohort_cost_usd")),
        ),
        reverse=True,
    )
    output_limit = max(1, int(limit or 25))
    for index, row in enumerate(rows[:output_limit], start=1):
        row["rank"] = index

    total_rows = _as_int(summary.get("candidate_rows"))
    repeated_groups = _as_int(summary.get("repeated_shape_groups"))
    if total_rows == 0:
        status = "no-cache-replayability-data"
        zero_hit_explanation = "no local non-hit cache metadata rows were available for replayability analysis"
    elif safe_count > 0:
        status = "safe-replayable-cohorts-present"
        zero_hit_explanation = "cache hits are zero but at least one repeated metadata shape appears safe enough for reviewed local replay canarying"
    elif repeated_groups > 0:
        status = "no-safe-replayable-cohorts"
        zero_hit_explanation = "cache hits are zero because repeated metadata shapes are blocked by streaming, tool invalidation, or cacheability evidence"
    else:
        status = "no-safe-replayable-cohorts"
        zero_hit_explanation = "cache hits are zero because no repeated normalized metadata shape was observed"

    return {
        "schema": "tokenclaw.cache_replayability_evidence.v1",
        "generated_at": utc_now(),
        "status": status,
        "zero_hit_explanation": zero_hit_explanation,
        "summary": {
            "total_rows_considered": total_rows,
            "shape_groups": _as_int(summary.get("shape_groups")),
            "repeated_shape_groups": repeated_groups,
            "safe_replayable_cohort_count": safe_count,
            "ranked_replayability_cohort_count": len(rows),
            "projected_repeated_call_cost_usd": round(_as_float(summary.get("projected_repeated_call_cost_usd")), 6),
            "top_blocker_code": next(iter(_breakdown_from_counts(blocker_counts)), {}).get("value") if blocker_counts else None,
            "legacy_blocker_breakdown_count": len(blocker_breakdown),
        },
        "blocker_code_breakdown": _breakdown_from_counts(blocker_counts),
        "ranked_replayability_cohorts": rows[:output_limit],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "provider_bodies_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "basis": "stored cache decision, route, dependency freshness, and aggregate cost metadata only",
        },
    }


def _cache_replayability_report_from_units(units: list[dict[str, Any]], *, limit: int = 25) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    one_off_rows = 0
    for unit in units:
        fingerprint, basis = _cache_replayability_fingerprint(unit)
        bucket = grouped.setdefault(
            fingerprint,
            {
                "shape_fingerprint": fingerprint,
                "fingerprint_basis": basis,
                "source_surface": unit["source_surface"],
                "granularity": unit["granularity"],
                "provider": unit.get("provider", "unknown"),
                "endpoint": unit.get("endpoint", "unknown"),
                "cache_status": unit["cache_status"],
                "cache_reason": unit["cache_reason"],
                "category": unit["category"],
                "workflow_phase": unit["workflow_phase"],
                "text_size_bucket": unit["text_size_bucket"],
                "input_items_bucket": unit["input_items_bucket"],
                "requested_tier": unit["requested_tier"],
                "target_tier": unit["target_tier"],
                "stream": unit["stream"],
                "has_tools": unit["has_tools"],
                "eligible": unit["eligible"],
                "replayability_level": unit["replayability_level"],
                "cacheability_bucket": unit["cacheability_bucket"],
                "cacheability": unit["cacheability"],
                "file_dependency_evidence_available": unit["file_dependency_evidence_available"],
                "file_dependency_count": 0,
                "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
                "file_dependency_audit": unit.get("file_dependency_audit"),
                "policy_source": unit["policy_source"],
                "count": 0,
                "sessions": set(),
                "example_sessions": [],
                "estimated_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "projected_repeated_call_cost_usd": 0.0,
                "input_tokens": 0,
                "text_chars": 0,
                "first_seen_at": unit.get("created_at"),
                "last_seen_at": unit.get("created_at"),
                "_blockers": set(unit.get("blockers") or []),
            },
        )
        bucket["count"] += 1
        session = str(unit.get("session_id") or "")
        if session:
            bucket["sessions"].add(session)
            short = _short_session_id(session)
            if short and short not in bucket["example_sessions"] and len(bucket["example_sessions"]) < 3:
                bucket["example_sessions"].append(short)
        bucket["estimated_cost_usd"] += _as_float(unit.get("cost_est_usd"))
        bucket["baseline_cost_usd"] += _as_float(unit.get("baseline_cost_usd"))
        bucket["input_tokens"] += _as_int(unit.get("input_tokens"))
        bucket["text_chars"] += _as_int(unit.get("text_chars"))
        bucket["file_dependency_count"] += _as_int(unit.get("file_dependency_count"))
        bucket["safe_invalidation_evidence"] = bool(
            bucket.get("safe_invalidation_evidence") or unit.get("safe_invalidation_evidence")
        )
        existing_audit = bucket.get("file_dependency_audit") if isinstance(bucket.get("file_dependency_audit"), dict) else {}
        unit_audit = unit.get("file_dependency_audit") if isinstance(unit.get("file_dependency_audit"), dict) else {}
        if unit_audit:
            bucket["file_dependency_audit"] = {
                **unit_audit,
                "snapshot_count": _as_int(existing_audit.get("snapshot_count")) + _as_int(unit_audit.get("snapshot_count")),
                "present_path_count": _as_int(existing_audit.get("present_path_count")) + _as_int(unit_audit.get("present_path_count")),
                "missing_path_count": _as_int(existing_audit.get("missing_path_count")) + _as_int(unit_audit.get("missing_path_count")),
                "changed_path_count": _as_int(existing_audit.get("changed_path_count")) + _as_int(unit_audit.get("changed_path_count")),
                "deleted_path_count": _as_int(existing_audit.get("deleted_path_count")) + _as_int(unit_audit.get("deleted_path_count")),
                "created_path_count": _as_int(existing_audit.get("created_path_count")) + _as_int(unit_audit.get("created_path_count")),
                "cap_exceeded": bool(existing_audit.get("cap_exceeded") or unit_audit.get("cap_exceeded")),
                "cap_trimmed": bool(existing_audit.get("cap_trimmed") or unit_audit.get("cap_trimmed")),
                "dependency_capture_reason": (
                    "dependency-cap-exceeded"
                    if bool(existing_audit.get("cap_exceeded") or unit_audit.get("cap_exceeded"))
                    else (
                        "dependency-cap-trimmed"
                        if bool(existing_audit.get("cap_trimmed") or unit_audit.get("cap_trimmed"))
                        else unit_audit.get("dependency_capture_reason")
                    )
                ),
                "safe_invalidation_evidence": bool(
                    existing_audit.get("safe_invalidation_evidence") or unit_audit.get("safe_invalidation_evidence")
                ),
                "file_dependency_evidence_available": bool(
                    existing_audit.get("file_dependency_evidence_available")
                    or unit_audit.get("file_dependency_evidence_available")
                ),
                "paths_included": False,
                "root_path_included": False,
            }
        for blocker in unit.get("blockers") or []:
            bucket["_blockers"].add(str(blocker))
        if str(unit.get("created_at") or "") < str(bucket.get("first_seen_at") or unit.get("created_at") or ""):
            bucket["first_seen_at"] = unit.get("created_at")
        if str(unit.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = unit.get("created_at")

    groups = []
    blocker_counts: dict[str, dict[str, Any]] = {}
    repeated_groups = 0
    repeated_rows = 0
    repeated_cost = 0.0
    projected_repeated_cost = 0.0
    unsafe_repeated_rows = 0
    unsafe_repeated_cost = 0.0
    for bucket in grouped.values():
        session_count = len(bucket["sessions"])
        blockers = set(bucket.pop("_blockers"))
        if session_count > 1:
            blockers.add("session-context-changed")
        if bucket["count"] == 1 and not blockers and bucket["cache_status"] == "miss":
            blockers.add("true-one-off-miss")
            one_off_rows += 1
        elif bucket["count"] == 1:
            one_off_rows += 1
        blocker_list = sorted(blockers)
        repeated = bucket["count"] > 1
        avg_cost = float(bucket["estimated_cost_usd"]) / bucket["count"] if bucket["count"] else 0.0
        projected = max(0.0, float(bucket["estimated_cost_usd"]) - avg_cost) if repeated else 0.0
        replay_candidate_class = "blocked-structural"
        if "true-one-off-miss" in blocker_list:
            replay_candidate_class = "one-off-miss"
        elif {"current-state", "user-specific", "low-cacheability"} & set(blocker_list):
            replay_candidate_class = "blocked-low-cacheability"
        elif (
            bool(bucket.get("stream"))
            and not bool(bucket.get("has_tools"))
            and bucket.get("cacheability_bucket") in {"high", "medium", "unknown"}
        ):
            replay_candidate_class = "streaming-non-tool-exact-candidate"
        elif bool(bucket.get("has_tools")) and {
            "file-dependency-missing",
            "dependency-missing",
            "dependency-cap-exceeded",
            "dependency-changed",
            "dependency-deleted",
            "dependency-created",
            "file-watch-disabled",
        } & set(blocker_list):
            replay_candidate_class = "blocked-tool-result-invalidation"
        elif not blocker_list and bucket.get("cacheability_bucket") in {"high", "medium", "unknown"}:
            replay_candidate_class = "replay-safe-exact-candidate"
        if repeated:
            repeated_groups += 1
            repeated_rows += bucket["count"]
            repeated_cost += float(bucket["estimated_cost_usd"])
            projected_repeated_cost += projected
            if blocker_list:
                unsafe_repeated_rows += bucket["count"]
                unsafe_repeated_cost += float(bucket["estimated_cost_usd"])
        for blocker in blocker_list or ["none"]:
            row = blocker_counts.setdefault(
                blocker,
                {"blocker": blocker, "groups": 0, "calls": 0, "estimated_cost_usd": 0.0, "projected_repeated_call_cost_usd": 0.0},
            )
            row["groups"] += 1
            row["calls"] += bucket["count"]
            row["estimated_cost_usd"] += float(bucket["estimated_cost_usd"])
            row["projected_repeated_call_cost_usd"] += projected
        finalized = {
            **bucket,
            "replay_fingerprint": f"sha256:{bucket['shape_fingerprint']}",
            "sessions": session_count,
            "repeated": repeated,
            "replay_candidate_class": replay_candidate_class,
            "replayability_blockers": blocker_list,
            "estimated_cost_usd": round(float(bucket["estimated_cost_usd"]), 6),
            "baseline_cost_usd": round(float(bucket["baseline_cost_usd"]), 6),
            "projected_repeated_call_cost_usd": round(projected, 6),
        }
        groups.append(finalized)

    groups.sort(key=lambda row: (row["count"], row["estimated_cost_usd"], str(row.get("last_seen_at") or "")), reverse=True)
    blocker_rows = [
        {
            **row,
            "estimated_cost_usd": round(float(row["estimated_cost_usd"]), 6),
            "projected_repeated_call_cost_usd": round(float(row["projected_repeated_call_cost_usd"]), 6),
        }
        for row in blocker_counts.values()
    ]
    blocker_rows.sort(key=lambda row: (row["calls"], row["projected_repeated_call_cost_usd"], row["estimated_cost_usd"]), reverse=True)
    blocker_burn_down = _cache_replay_blocker_burn_down(groups, limit=limit)
    session_memory_proposals = _session_memory_replay_proposal_rows(units, limit=limit)
    session_memory_status_counts: dict[str, int] = {}
    for row in session_memory_proposals:
        status = str(row.get("status") or "unknown")
        session_memory_status_counts[status] = session_memory_status_counts.get(status, 0) + _as_int(row.get("count"))
    summary = {
        "candidate_rows": len(units),
        "shape_groups": len(groups),
        "repeated_shape_groups": repeated_groups,
        "repeated_candidate_rows": repeated_rows,
        "one_off_candidate_rows": one_off_rows,
        "repeated_estimated_cost_usd": round(repeated_cost, 6),
        "projected_repeated_call_cost_usd": round(projected_repeated_cost, 6),
        "unsafe_repeated_rows": unsafe_repeated_rows,
        "unsafe_repeated_estimated_cost_usd": round(unsafe_repeated_cost, 6),
        "no_repeated_shape_exists": repeated_groups == 0,
        "repeated_shape_exists_but_cache_is_unsafe": unsafe_repeated_rows > 0,
        "blocker_burn_down_rows": len(blocker_burn_down),
        "top_blocker_burn_down_projected_cost_usd": (
            _as_float(blocker_burn_down[0].get("projected_repeated_call_cost_usd"))
            if blocker_burn_down
            else 0.0
        ),
        "top_blocker_burn_down_next_action_family": (
            blocker_burn_down[0].get("next_action_family")
            if blocker_burn_down
            else None
        ),
        "session_memory_replay_proposal_count": sum(_as_int(row.get("count")) for row in session_memory_proposals),
        "session_memory_replay_eligible_count": sum(
            _as_int(row.get("count"))
            for row in session_memory_proposals
            if row.get("status") == "session-plateau-dry-run-eligible"
        ),
        "session_memory_replay_blocked_count": sum(
            _as_int(row.get("count"))
            for row in session_memory_proposals
            if row.get("status") != "session-plateau-dry-run-eligible"
        ),
    }
    evidence = _cache_replayability_evidence_from_report(
        groups=groups,
        summary=summary,
        blocker_breakdown=blocker_rows,
        limit=limit,
    )
    return {
        "schema": "tokenclaw.cache_replayability.v1",
        "generated_at": utc_now(),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "basis": "metadata-derived shape fingerprints only; request/response bodies are not inspected",
        },
        "summary": summary,
        "cache_replayability_evidence": evidence,
        "blocker_breakdown": blocker_rows,
        "blocker_burn_down": blocker_burn_down,
        "session_memory_replay_proposal_breakdown": _breakdown_from_counts(session_memory_status_counts),
        "session_memory_replay_proposals": session_memory_proposals,
        "groups": groups[: max(1, int(limit or 25))],
    }


def _cache_replayability_units_from_store(store_obj: Any, *, row_limit: int | None = None) -> list[dict[str, Any]]:
    conn = store_obj.conn
    provider_rows = [
        dict(row)
        for row in conn.execute("""
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, stream, cache_hit, status_code,
                   cache_json, routing_json, crunch_json, category, session_id,
                   input_tokens_est, actual_input_tokens as input_tokens,
                   cost_est_usd, cost_baseline_usd,
                   null as input_items,
                   null as text_chars
            from calls
            order by created_at desc
        """).fetchall()
    ]
    if row_limit is not None:
        provider_rows = provider_rows[: max(0, int(row_limit))]
    units: list[dict[str, Any]] = []
    for row in provider_rows:
        surface = _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        unit = _cache_replayability_unit(row, source_surface=surface, granularity="provider_request")
        if unit is not None:
            units.append(unit)

    codex_rows = [
        dict(row)
        for row in conn.execute("""
            select s.created_at,
                   s.session_id,
                   s.routing_json,
                   s.crunch_json,
                   s.cache_json,
                   s.input_items,
                   s.input_text_chars as text_chars,
                   s.input_text_chars as input_tokens,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
        """).fetchall()
    ]
    if row_limit is not None:
        codex_rows = codex_rows[: max(0, int(row_limit))]
    for row in codex_rows:
        estimates = _codex_estimates_with_cache(row.get("text_chars"), row.get("response_result_chars"), _json_obj(row.get("cache_json")))
        prepared = {
            **row,
            "provider": "codex-app",
            "path": "codex_app_turn",
            "endpoint": "codex_app_turn",
            "requested_model": CODEX_APP_MODEL,
            "routed_model": CODEX_APP_MODEL,
            "stream": 0,
            "cache_hit": 1 if _json_obj(row.get("cache_json")).get("status") == "hit" else 0,
            "status_code": None,
            "category": "codex-app-turn",
            "input_tokens": estimates.get("input_tokens_est"),
            "cost_est_usd": estimates.get("cost_est_usd"),
            "cost_baseline_usd": estimates.get("baseline_cost_est_usd"),
        }
        unit = _cache_replayability_unit(prepared, source_surface=CODEX_APP_SOURCE_SURFACE, granularity="agent_turn")
        if unit is not None:
            units.append(unit)

    return units


async def stats_cache_replayability(store_obj: Any, limit: int = 25) -> dict[str, Any]:
    units = _cache_replayability_units_from_store(store_obj)
    return _cache_replayability_report_from_units(units, limit=limit)


_CACHE_REPLAY_STALE_BLOCKERS = {"current-state", "low-cacheability", "user-specific"}
_CACHE_REPLAY_ACTIVATION_SETUP_BLOCKERS = {
    "semantic-cache-disabled",
    "streaming",
    "tool-call-disabled",
}


def _cache_replay_dependency_state(unit: dict[str, Any]) -> str:
    freshness = unit.get("dependency_freshness") if isinstance(unit.get("dependency_freshness"), dict) else {}
    freshness_status = str(freshness.get("status") or "")
    if freshness_status == "fresh":
        return "stable"
    if freshness_status == "stale":
        return "invalidated"
    if freshness_status == "not-required":
        return "not-required"
    if not bool(unit.get("has_tools")):
        return "not-required"
    blockers = {str(item) for item in unit.get("blockers") or []}
    audit = unit.get("file_dependency_audit") if isinstance(unit.get("file_dependency_audit"), dict) else {}
    reason = str(audit.get("invalidation_reason") or "")
    if blockers & {"dependency-changed", "dependency-created", "dependency-deleted", "dependency-cap-exceeded"}:
        return "invalidated"
    if reason in {"dependency-changed", "dependency-created", "dependency-deleted"}:
        return "invalidated"
    if reason in {"file-dependency-missing", "dependency-missing", "file-watch-disabled"}:
        return "missing"
    if blockers & {"file-dependency-missing", "dependency-missing", "file-watch-disabled"}:
        return "missing"
    if bool(unit.get("safe_invalidation_evidence")):
        return "stable"
    if bool(unit.get("file_dependency_evidence_available")):
        return "evidence-without-safe-invalidation"
    return "missing"


def _cache_replay_provider_adoption_state(unit: dict[str, Any]) -> tuple[str, list[str]]:
    cache = unit.get("cache") if isinstance(unit.get("cache"), dict) else {}
    candidates = [
        cache.get("provider_adoption_gate"),
        cache.get("provider_adoption_health"),
        cache.get("provider_adoption"),
        (unit.get("pattern_features") or {}).get("provider_adoption_gate")
        if isinstance(unit.get("pattern_features"), dict)
        else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        reason_codes = [
            public_label(item, "unknown")
            for item in candidate.get("reason_codes") or candidate.get("blockers") or []
            if isinstance(item, (str, int, float, bool))
        ]
        status = public_label(candidate.get("status") or candidate.get("readiness") or "", "")
        if candidate.get("blocking") is True or status in {"blocked", "regressed", "failed"}:
            return "blocked", sorted(set(reason_codes or ["provider-adoption-regression"]))
        if status in {"ready", "pass", "healthy"} or candidate.get("blocking") is False:
            return "ready", sorted(set(reason_codes))
    return "not-observed", []


def _cache_replay_plateau_evidence(unit: dict[str, Any]) -> dict[str, Any]:
    proposal = _public_session_memory_replay_proposal(
        unit.get("session_memory_replay_proposal")
        if isinstance(unit.get("session_memory_replay_proposal"), dict)
        else {}
    )
    text_chars = _as_int(unit.get("text_chars"))
    evidence = {
        "session_memory_proposal": proposal is not None,
        "session_memory_status": proposal.get("status") if proposal else None,
        "large_context": text_chars >= 8000,
        "text_size_bucket": unit.get("text_size_bucket"),
    }
    evidence["present"] = bool(
        proposal is not None
        or (
            text_chars >= 8000
            and str(unit.get("category") or "").startswith("tool")
        )
    )
    return evidence


def _cache_replay_cohort_basis(unit: dict[str, Any]) -> dict[str, Any]:
    cache = unit.get("cache") if isinstance(unit.get("cache"), dict) else {}
    features = unit.get("pattern_features") if isinstance(unit.get("pattern_features"), dict) else {}
    dependency_state = _cache_replay_dependency_state(unit)
    provider_adoption_state, _provider_reasons = _cache_replay_provider_adoption_state(unit)
    replay_scope = public_label(cache.get("replay_scope") or features.get("replay_scope") or "metadata-shape", "metadata-shape")
    return {
        "source_surface": unit.get("source_surface"),
        "granularity": unit.get("granularity"),
        "app_family": public_label(features.get("app_family") or "unknown", "unknown"),
        "category": unit.get("category"),
        "workflow_phase": unit.get("workflow_phase"),
        "stream": bool(unit.get("stream")),
        "has_tools": bool(unit.get("has_tools")),
        "replay_scope": replay_scope,
        "replay_scope_id_available": bool(cache.get("replay_scope_id_available")),
        "replayability_level": unit.get("replayability_level"),
        "cacheability_bucket": unit.get("cacheability_bucket"),
        "requested_tier": unit.get("requested_tier"),
        "target_tier": unit.get("target_tier"),
        "text_size_bucket": unit.get("text_size_bucket"),
        "dependency_state": dependency_state,
        "dependency_snapshot_count_bucket": (
            (unit.get("file_dependency_audit") or {}).get("snapshot_count_bucket")
            if isinstance(unit.get("file_dependency_audit"), dict)
            else "unknown"
        ),
        "provider_adoption_state": provider_adoption_state,
    }


def _cache_replay_finalize_cohort(bucket: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(bucket.get("count"))
    blockers = set(bucket.pop("_blockers"))
    if count <= 1:
        blockers.add("insufficient-repeat-evidence")
    if not bool(bucket.get("plateau_evidence_present")):
        blockers.add("plateau-evidence-missing")
    dependency_state = str(bucket.get("dependency_state") or "unknown")
    if dependency_state == "invalidated":
        blockers.add("dependency-invalidated")
    elif dependency_state in {"missing", "evidence-without-safe-invalidation"}:
        blockers.add("dependency-evidence-missing")
    provider_reasons = bucket.pop("_provider_adoption_reason_codes")
    if str(bucket.get("provider_adoption_state")) == "blocked":
        blockers.update(provider_reasons or ["provider-adoption-regression"])

    hard_blockers = (
        (blockers & _CACHE_REPLAY_STALE_BLOCKERS)
        or "dependency-invalidated" in blockers
        or "missing-cache-metadata" in blockers
        or "turn-level-only" in blockers
        or "provider-adoption-regression" in blockers
    )
    evidence_blockers = {
        "dependency-evidence-missing",
        "insufficient-repeat-evidence",
        "plateau-evidence-missing",
        "provider-adoption-evidence-missing",
    }
    if hard_blockers:
        readiness = "blocked"
    elif blockers & evidence_blockers:
        readiness = "needs-more-evidence"
    else:
        readiness = "activation-ready"

    avg_cost = float(bucket["estimated_cost_usd"]) / count if count else 0.0
    projected_hits = max(0, count - 1)
    projected_savings = max(0.0, float(bucket["estimated_cost_usd"]) - avg_cost) if projected_hits else 0.0
    activation_blockers = sorted(blockers - _CACHE_REPLAY_ACTIVATION_SETUP_BLOCKERS)
    recommended_canary = None
    if readiness == "activation-ready":
        cohort_hash = str(bucket["cohort_id"]).split(":", 1)[-1]
        recommended_canary = {
            "rule_id": f"local-cache-replay-cohort:{cohort_hash}",
            "policy_source": "local-manual",
            "source_surface": bucket.get("source_surface"),
            "category": bucket.get("category"),
            "workflow_phase": bucket.get("workflow_phase"),
            "replay_scope": bucket.get("replay_scope"),
            "streaming": bool(bucket.get("stream")),
            "allow_tool_calls": bool(bucket.get("has_tools")),
            "safe_invalidation": bool(bucket.get("has_tools")),
            "replayability_levels": [bucket.get("replayability_level")],
            "canary_fraction": 0.05,
            "holdout_fraction": 0.95,
            "requires_human_review": True,
            "pattern_hashes_included": False,
        }

    public = {
        key: value
        for key, value in bucket.items()
        if key
        not in {
            "sessions",
            "estimated_cost_usd",
            "baseline_cost_usd",
            "input_tokens",
            "text_chars",
        }
    }
    public.update({
        "readiness": readiness,
        "blocker_reasons": activation_blockers,
        "setup_required": sorted(blockers & _CACHE_REPLAY_ACTIVATION_SETUP_BLOCKERS),
        "session_count": len(bucket["sessions"]),
        "projected_hits": projected_hits,
        "projected_saved_cost_usd": round(projected_savings, 6),
        "projected_cost_bucket": _cache_replay_cost_bucket(projected_savings),
        "estimated_cost_usd": round(float(bucket["estimated_cost_usd"]), 6),
        "baseline_cost_usd": round(float(bucket["baseline_cost_usd"]), 6),
        "avg_input_tokens": round(_as_int(bucket.get("input_tokens")) / count) if count else 0,
        "avg_text_chars": round(_as_int(bucket.get("text_chars")) / count) if count else 0,
        "provider_adoption_reason_codes": sorted(set(provider_reasons)),
        "recommended_canary": recommended_canary,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "pattern_hashes_included": False,
    })
    return public


def _cache_replay_cohort_ranking_from_units(units: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for unit in units:
        basis = _cache_replay_cohort_basis(unit)
        raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
        cohort_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        bucket = grouped.setdefault(
            cohort_hash,
            {
                "schema": "tokenclaw.cache_replay_plateau_cohort.v1",
                "cohort_id": f"cache-replay-cohort:{cohort_hash}",
                "cohort_basis": basis,
                "source_surface": basis["source_surface"],
                "granularity": basis["granularity"],
                "app_family": basis["app_family"],
                "category": basis["category"],
                "workflow_phase": basis["workflow_phase"],
                "stream": bool(basis["stream"]),
                "has_tools": bool(basis["has_tools"]),
                "replay_scope": basis["replay_scope"],
                "replay_scope_id_available": bool(basis["replay_scope_id_available"]),
                "replayability_level": basis["replayability_level"],
                "cacheability_bucket": basis["cacheability_bucket"],
                "dependency_state": basis["dependency_state"],
                "provider_adoption_state": basis["provider_adoption_state"],
                "count": 0,
                "sessions": set(),
                "estimated_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "input_tokens": 0,
                "text_chars": 0,
                "first_seen_at": unit.get("created_at"),
                "last_seen_at": unit.get("created_at"),
                "plateau_evidence_present": False,
                "session_memory_proposal_count": 0,
                "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
                "file_dependency_audit": unit.get("file_dependency_audit"),
                "_blockers": set(unit.get("blockers") or []),
                "_provider_adoption_reason_codes": set(),
            },
        )
        bucket["count"] += 1
        session = str(unit.get("session_id") or "")
        if session:
            bucket["sessions"].add(hashlib.sha256(session.encode("utf-8")).hexdigest()[:16])
        bucket["estimated_cost_usd"] += _as_float(unit.get("cost_est_usd"))
        bucket["baseline_cost_usd"] += _as_float(unit.get("baseline_cost_usd"))
        bucket["input_tokens"] += _as_int(unit.get("input_tokens"))
        bucket["text_chars"] += _as_int(unit.get("text_chars"))
        bucket["file_dependency_evidence_available"] = bool(
            bucket.get("file_dependency_evidence_available") or unit.get("file_dependency_evidence_available")
        )
        bucket["safe_invalidation_evidence"] = bool(bucket.get("safe_invalidation_evidence") or unit.get("safe_invalidation_evidence"))
        plateau = _cache_replay_plateau_evidence(unit)
        bucket["plateau_evidence_present"] = bool(bucket.get("plateau_evidence_present") or plateau.get("present"))
        if plateau.get("session_memory_proposal"):
            bucket["session_memory_proposal_count"] += 1
        for blocker in unit.get("blockers") or []:
            bucket["_blockers"].add(str(blocker))
        _state, provider_reason_codes = _cache_replay_provider_adoption_state(unit)
        for code in provider_reason_codes:
            bucket["_provider_adoption_reason_codes"].add(str(code))
        if str(unit.get("created_at") or "") < str(bucket.get("first_seen_at") or unit.get("created_at") or ""):
            bucket["first_seen_at"] = unit.get("created_at")
        if str(unit.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = unit.get("created_at")

    cohorts = [_cache_replay_finalize_cohort(bucket) for bucket in grouped.values()]
    readiness_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    dependency_counts: dict[str, int] = {}
    adoption_counts: dict[str, int] = {}
    for row in cohorts:
        readiness = str(row.get("readiness") or "unknown")
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        dependency = str(row.get("dependency_state") or "unknown")
        dependency_counts[dependency] = dependency_counts.get(dependency, 0) + 1
        adoption = str(row.get("provider_adoption_state") or "unknown")
        adoption_counts[adoption] = adoption_counts.get(adoption, 0) + 1
        for blocker in row.get("blocker_reasons") or []:
            blocker_counts[str(blocker)] = blocker_counts.get(str(blocker), 0) + 1

    cohorts.sort(
        key=lambda row: (
            row.get("readiness") == "activation-ready",
            row.get("readiness") == "needs-more-evidence",
            _as_float(row.get("projected_saved_cost_usd")),
            _as_int(row.get("projected_hits")),
            _as_int(row.get("count")),
            _as_float(row.get("estimated_cost_usd")),
        ),
        reverse=True,
    )
    output_limit = max(1, min(int(limit or 25), 1000))
    return {
        "schema": "tokenclaw.cache_replay_plateau_cohort_ranking.v1",
        "generated_at": utc_now(),
        "summary": {
            "candidate_rows": len(units),
            "cohort_count": len(cohorts),
            "activation_ready_count": readiness_counts.get("activation-ready", 0),
            "needs_more_evidence_count": readiness_counts.get("needs-more-evidence", 0),
            "blocked_count": readiness_counts.get("blocked", 0),
            "projected_ready_hits": sum(
                _as_int(row.get("projected_hits"))
                for row in cohorts
                if row.get("readiness") == "activation-ready"
            ),
            "projected_ready_saved_cost_usd": round(
                sum(
                    _as_float(row.get("projected_saved_cost_usd"))
                    for row in cohorts
                    if row.get("readiness") == "activation-ready"
                ),
                6,
            ),
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "readiness_breakdown": _breakdown_from_counts(readiness_counts),
        "dependency_breakdown": _breakdown_from_counts(dependency_counts),
        "provider_adoption_breakdown": _breakdown_from_counts(adoption_counts),
        "blocker_breakdown": _breakdown_from_counts(blocker_counts),
        "cohorts": cohorts[:output_limit],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "provider_bodies_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "basis": "stored cache decision, dependency, session-memory plateau, provider-adoption, and cost metadata only",
        },
    }


async def stats_cache_replay_cohort_ranking(
    store_obj: Any,
    *,
    limit: int = 25,
    row_limit: int | None = None,
) -> dict[str, Any]:
    scan_limit = max(1, min(_as_int(row_limit if row_limit is not None else 1000) or 1000, 10000))
    output_limit = max(1, min(_as_int(limit) or 25, 1000))
    units = _cache_replayability_units_from_store(store_obj, row_limit=scan_limit)
    return _cache_replay_cohort_ranking_from_units(units, limit=output_limit)


def _cache_table_count(conn: Any) -> int | None:
    try:
        row = conn.execute("select count(*) as c from cache").fetchone()
        return _as_int(row["c"] if row is not None else 0)
    except Exception:
        return None


def _dry_run_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float, bool))]
    return []


def _dry_run_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return False


def _dry_run_condition_matches(conditions: dict[str, Any], key: str, actual: Any) -> bool:
    if key not in conditions:
        return True
    expected = {item.lower() for item in _dry_run_values(conditions.get(key))}
    if not expected:
        return True
    return str(actual or "").lower() in expected


def _cache_dry_run_public_canary(canary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(canary, dict):
        return None
    public = dict(canary)
    pattern_hash_count = len(public.get("pattern_hashes") or []) if isinstance(public.get("pattern_hashes"), list) else 0
    public.pop("pattern_hashes", None)
    for key in ("cohort", "unit", "reason", "status", "rule_id", "candidate_id"):
        if key in public:
            public[key] = public_id(public[key], prefix=key.replace("_", "-")) if key.endswith("_id") else public_label(public[key], "unknown")
    public["pattern_hash_count"] = pattern_hash_count
    public["pattern_hashes_included"] = False
    return public


def _cache_replay_dry_run_decision(unit: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    from tokenclaw.cache import cache_pattern_hashes_from_features
    from tokenclaw.pattern_rollout import pattern_canary_decision, pattern_rollout_public_meta

    features = unit.get("pattern_features") if isinstance(unit.get("pattern_features"), dict) else {}
    feature_hashes = set(cache_pattern_hashes_from_features(features))
    last_reason = "no-matching-rule"
    saw_source_surface_mismatch = False
    saw_replayability_mismatch = False
    if not feature_hashes:
        last_reason = "pattern-features-missing"

    for rule in rules:
        rule_id = public_id(rule.get("id") or "cache-pattern-rule", prefix="rule-id", fallback="cache-pattern-rule")
        candidate_id = public_id(rule.get("candidate_id"), prefix="candidate-id") if rule.get("candidate_id") is not None else None
        policy_source = public_label(rule.get("policy_source") or "managed-recommended", "unknown")
        if not rule.get("enabled", True):
            last_reason = "disabled"
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        rule_hashes = {str(item) for item in conditions.get("pattern_hashes") or []}
        matched_hashes = sorted(feature_hashes.intersection(rule_hashes))
        if not matched_hashes:
            last_reason = "pattern-hash-mismatch"
            continue
        model_pattern = conditions.get("model_pattern")
        if isinstance(model_pattern, str) and model_pattern.strip():
            requested_model = str(features.get("requested_model") or unit.get("requested_model") or "").lower()
            routed_model = str(features.get("candidate_target_model") or unit.get("routed_model") or "").lower()
            pattern = model_pattern.strip().lower()
            if pattern not in requested_model and pattern not in routed_model:
                last_reason = "model-pattern-mismatch"
                continue
        if "has_tools" in conditions and _dry_run_bool(conditions.get("has_tools")) != bool(unit.get("has_tools")):
            last_reason = "has-tools-mismatch"
            continue
        if "stream" in conditions and _dry_run_bool(conditions.get("stream")) != bool(unit.get("stream")):
            last_reason = "stream-mismatch"
            continue
        if not _dry_run_condition_matches(conditions, "category", unit.get("category")):
            last_reason = "category-mismatch"
            continue
        excluded_categories = {item.lower() for item in _dry_run_values(conditions.get("category_not_in"))}
        if excluded_categories and str(unit.get("category") or "").lower() in excluded_categories:
            last_reason = "category-excluded"
            continue
        mismatched_key = None
        for key in ("workflow_phase", "source_surface", "app_family", "text_bucket", "token_bucket"):
            actual = features.get(key)
            if key == "source_surface":
                actual = unit.get("source_surface")
            if not _dry_run_condition_matches(conditions, key, actual):
                mismatched_key = key
                break
        if mismatched_key:
            last_reason = f"{mismatched_key}-mismatch"
            if mismatched_key == "source_surface":
                saw_source_surface_mismatch = True
            continue
        replayability_levels = {item.lower() for item in _dry_run_values(conditions.get("replayability_levels"))}
        if replayability_levels and str(unit.get("replayability_level") or "").lower() not in replayability_levels:
            last_reason = "replayability-gate-mismatch"
            saw_replayability_mismatch = True
            continue

        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        base = {
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "policy_source": policy_source,
            "matched_pattern_hash_count": len(matched_hashes),
            "matched_pattern_hashes_included": False,
            "rollout": pattern_rollout_public_meta(rule.get("rollout")),
        }
        dependency_freshness = (
            unit.get("dependency_freshness")
            if isinstance(unit.get("dependency_freshness"), dict)
            else _cache_replay_dependency_freshness(unit)
        )
        if action.get("type") not in {"exact_cache", "exact_cache_pattern"}:
            return {**base, "status": "blocked", "reason": "unsupported-action", "blockers": ["unsupported-action"]}
        if bool(unit.get("stream")) and not bool(action.get("streaming")):
            return {**base, "status": "blocked", "reason": "streaming-not-allowed", "blockers": ["streaming-not-allowed"]}
        if bool(unit.get("has_tools")) and not bool(action.get("allow_tool_calls")):
            return {
                **base,
                "status": "invalidation-required",
                "reason": "tool-cache-rule-requires-safe-invalidation",
                "blockers": ["tool-call-disabled", "safe-invalidation-required"],
                "requires_file_dependency_evidence": True,
                "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                "dependency_freshness": dependency_freshness,
            }
        if bool(unit.get("has_tools")) and not bool(action.get("safe_invalidation")):
            return {
                **base,
                "status": "invalidation-required",
                "reason": "tool-cache-rule-missing-safe-invalidation",
                "blockers": ["safe-invalidation-required"],
                "requires_file_dependency_evidence": True,
                "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                "dependency_freshness": dependency_freshness,
            }
        if bool(unit.get("has_tools")):
            audit_blocker = None
            freshness_status = str(dependency_freshness.get("status") or "unknown")
            if freshness_status == "stale":
                audit_blocker = str(dependency_freshness.get("reason") or "dependency-stale")
            elif freshness_status == "unknown":
                audit_blocker = str(dependency_freshness.get("reason") or "dependency-freshness-unknown")
            if audit_blocker:
                return {
                    **base,
                    "status": "invalidation-required",
                    "reason": audit_blocker,
                    "blockers": [audit_blocker],
                    "requires_file_dependency_evidence": True,
                    "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                    "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
                    "file_dependency_audit": unit.get("file_dependency_audit"),
                    "dependency_freshness": dependency_freshness,
                }
        if bool(unit.get("has_tools")) and not bool(unit.get("file_dependency_evidence_available")):
            return {
                **base,
                "status": "invalidation-required",
                "reason": "file-dependency-evidence-absent",
                "blockers": ["file-dependency-evidence-absent"],
                "requires_file_dependency_evidence": True,
                "file_dependency_evidence_available": False,
                "dependency_freshness": dependency_freshness,
            }
        stale_blockers = [
            blocker
            for blocker in unit.get("blockers") or []
            if blocker in {"current-state", "user-specific", "low-cacheability"}
        ]
        if stale_blockers:
            return {
                **base,
                "status": "blocked",
                "reason": "stale-risk-blockers",
                "blockers": stale_blockers,
                "stale_risk_blockers": stale_blockers,
            }
        canary = pattern_canary_decision(
            rollout=rule.get("rollout"),
            rule_id=rule_id,
            candidate_id=candidate_id,
            pattern_hashes=matched_hashes,
            features=features,
        )
        if canary.get("enabled") and not canary.get("selected", True):
            return {
                **base,
                "status": "holdout",
                "reason": "canary_holdout",
                "blockers": ["canary_holdout"],
                "canary": _cache_dry_run_public_canary(canary),
            }
        return {
            **base,
            "status": "projected-streaming-candidate" if bool(unit.get("stream")) else "projected-exact-candidate",
            "reason": "rule-match",
            "blockers": [],
            "canary": _cache_dry_run_public_canary(canary) if canary.get("enabled") else None,
            "requires_file_dependency_evidence": bool(unit.get("has_tools")),
            "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
            "dependency_freshness": dependency_freshness,
        }

    if saw_source_surface_mismatch:
        return {"status": "unsupported-source-surface", "reason": "source_surface-mismatch", "blockers": ["unsupported-source-surface"]}
    if saw_replayability_mismatch:
        return {"status": "blocked", "reason": "replayability-gate-mismatch", "blockers": ["replayability-gate-mismatch"]}
    return {"status": "unmatched", "reason": last_reason, "blockers": [last_reason] if last_reason else []}


def _cache_replay_dry_run_from_units(
    units: list[dict[str, Any]],
    *,
    rules: list[dict[str, Any]],
    limit: int,
    cache_rows_before: int | None,
    cache_rows_after: int | None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    freshness_counts: dict[str, int] = {}
    session_memory_proposals = _session_memory_replay_proposal_rows(units, limit=limit)
    for unit in units:
        decision = _cache_replay_dry_run_decision(unit, rules)
        status = str(decision.get("status") or "unknown")
        reason = str(decision.get("reason") or "unknown")
        fingerprint, basis = _cache_replayability_fingerprint(unit)
        dependency_freshness = (
            decision.get("dependency_freshness")
            if isinstance(decision.get("dependency_freshness"), dict)
            else (
                unit.get("dependency_freshness")
                if isinstance(unit.get("dependency_freshness"), dict)
                else _cache_replay_dependency_freshness(unit)
            )
        )
        freshness_status = str(dependency_freshness.get("status") or "unknown")
        key = (status, str(decision.get("rule_id") or ""), fingerprint)
        bucket = grouped.setdefault(
            key,
            {
                "status": status,
                "reason": reason,
                "rule_id": decision.get("rule_id"),
                "candidate_id": decision.get("candidate_id"),
                "policy_source": decision.get("policy_source"),
                "replay_fingerprint": f"sha256:{fingerprint}",
                "fingerprint_basis": basis,
                "source_surface": unit.get("source_surface"),
                "granularity": unit.get("granularity"),
                "category": unit.get("category"),
                "workflow_phase": unit.get("workflow_phase"),
                "stream": bool(unit.get("stream")),
                "has_tools": bool(unit.get("has_tools")),
                "replayability_level": unit.get("replayability_level"),
                "cacheability_bucket": unit.get("cacheability_bucket"),
                "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
                "file_dependency_audit": unit.get("file_dependency_audit"),
                "dependency_freshness": dependency_freshness,
                "dependency_freshness_status": freshness_status,
                "requires_file_dependency_evidence": bool(decision.get("requires_file_dependency_evidence")),
                "matched_pattern_hash_count": _as_int(decision.get("matched_pattern_hash_count")),
                "matched_pattern_hashes_included": False,
                "count": 0,
                "estimated_cost_usd": 0.0,
                "projected_hits": 0,
                "estimated_saved_cost_usd": 0.0,
                "first_seen_at": unit.get("created_at"),
                "last_seen_at": unit.get("created_at"),
                "_blockers": set(decision.get("blockers") or []),
                "_stale": set(decision.get("stale_risk_blockers") or []),
            },
        )
        bucket["count"] += 1
        bucket["estimated_cost_usd"] += _as_float(unit.get("cost_est_usd"))
        if str(unit.get("created_at") or "") < str(bucket.get("first_seen_at") or unit.get("created_at") or ""):
            bucket["first_seen_at"] = unit.get("created_at")
        if str(unit.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = unit.get("created_at")
        if decision.get("canary") and not bucket.get("canary"):
            bucket["canary"] = decision.get("canary")
        if decision.get("rollout") and not bucket.get("rollout"):
            bucket["rollout"] = decision.get("rollout")
        bucket_freshness = bucket.get("dependency_freshness") if isinstance(bucket.get("dependency_freshness"), dict) else {}
        if bucket_freshness.get("status") in {None, "", "not-required"} and freshness_status != "not-required":
            bucket["dependency_freshness"] = dependency_freshness
            bucket["dependency_freshness_status"] = freshness_status
        for blocker in decision.get("blockers") or []:
            bucket["_blockers"].add(str(blocker))
        for blocker in decision.get("stale_risk_blockers") or []:
            bucket["_stale"].add(str(blocker))
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        freshness_counts[freshness_status] = freshness_counts.get(freshness_status, 0) + 1
        for blocker in decision.get("blockers") or []:
            label = str(blocker)
            blocker_counts[label] = blocker_counts.get(label, 0) + 1

    rows: list[dict[str, Any]] = []
    projected_exact_hits = 0
    projected_streaming_hits = 0
    estimated_saved = 0.0
    for bucket in grouped.values():
        status = str(bucket.get("status") or "")
        candidate = status in {"projected-exact-candidate", "projected-streaming-candidate"}
        if candidate and _as_int(bucket.get("count")) > 1:
            projected_hits = _as_int(bucket.get("count")) - 1
            avg_cost = float(bucket["estimated_cost_usd"]) / _as_int(bucket.get("count"))
            saved = max(0.0, float(bucket["estimated_cost_usd"]) - avg_cost)
            bucket["projected_hits"] = projected_hits
            bucket["estimated_saved_cost_usd"] = saved
            estimated_saved += saved
            if status == "projected-streaming-candidate":
                projected_streaming_hits += projected_hits
            else:
                projected_exact_hits += projected_hits
        finalized = {
            key: value
            for key, value in bucket.items()
            if key not in {"_blockers", "_stale"}
        }
        finalized["blockers"] = sorted(bucket["_blockers"])
        finalized["stale_risk_blockers"] = sorted(bucket["_stale"])
        finalized["estimated_cost_usd"] = round(float(finalized["estimated_cost_usd"]), 6)
        finalized["estimated_saved_cost_usd"] = round(float(finalized["estimated_saved_cost_usd"]), 6)
        rows.append(finalized)

    rows.sort(
        key=lambda row: (
            _as_float(row.get("estimated_saved_cost_usd")),
            _as_int(row.get("projected_hits")),
            _as_int(row.get("count")),
            _as_float(row.get("estimated_cost_usd")),
        ),
        reverse=True,
    )
    candidate_rows = status_counts.get("projected-exact-candidate", 0) + status_counts.get("projected-streaming-candidate", 0)
    holdout_rows = status_counts.get("holdout", 0)
    invalidation_rows = status_counts.get("invalidation-required", 0)
    unsupported_rows = status_counts.get("unsupported-source-surface", 0)
    stale_rows = sum(_as_int(row.get("count")) for row in rows if row.get("stale_risk_blockers"))
    blocked_rows = sum(
        count
        for status, count in status_counts.items()
        if status in {"blocked", "invalidation-required", "unsupported-source-surface"}
    )
    return {
        "schema": "tokenclaw.cache_replay_dry_run.v1",
        "generated_at": utc_now(),
        "summary": {
            "rows_considered": len(units),
            "policy_rule_count": len(rules),
            "matched_rows": len(units) - status_counts.get("unmatched", 0),
            "candidate_rows": candidate_rows,
            "projected_exact_hits": projected_exact_hits,
            "projected_streaming_hits": projected_streaming_hits,
            "projected_total_hits": projected_exact_hits + projected_streaming_hits,
            "holdout_rows": holdout_rows,
            "blocked_rows": blocked_rows,
            "invalidation_required_rows": invalidation_rows,
            "unsupported_source_surface_rows": unsupported_rows,
            "stale_risk_blocked_rows": stale_rows,
            "estimated_saved_cost_usd": round(estimated_saved, 6),
            "cache_rows_before": cache_rows_before,
            "cache_rows_after": cache_rows_after,
            "cache_table_mutated": cache_rows_before != cache_rows_after,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "session_memory_replay_proposal_count": sum(_as_int(row.get("count")) for row in session_memory_proposals),
            "session_memory_replay_eligible_count": sum(
                _as_int(row.get("count"))
                for row in session_memory_proposals
                if row.get("status") == "session-plateau-dry-run-eligible"
            ),
        },
        "status_breakdown": _breakdown_from_counts(status_counts),
        "reason_breakdown": _breakdown_from_counts(reason_counts),
        "blocker_breakdown": _breakdown_from_counts(blocker_counts),
        "dependency_freshness_breakdown": _breakdown_from_counts(freshness_counts),
        "session_memory_replay_proposals": session_memory_proposals,
        "rows": rows[: max(1, int(limit or 50))],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "pattern_hashes_included": False,
            "file_paths_included": False,
            "raw_file_stat_values_included": False,
            "basis": "proposed cache pattern rules evaluated against metadata-derived replay fingerprints only",
        },
    }


async def stats_cache_replay_dry_run(
    store_obj: Any,
    proposed_policy: Any,
    *,
    limit: int = 1000,
    row_limit: int | None = None,
) -> dict[str, Any]:
    from tokenclaw.cache import cache_pattern_rules_from_policy_payload

    rules = cache_pattern_rules_from_policy_payload(proposed_policy)
    scan_limit = max(1, min(_as_int(row_limit if row_limit is not None else limit) or 1000, 10000))
    output_limit = max(1, min(_as_int(limit) or 50, 1000))
    cache_rows_before = _cache_table_count(store_obj.conn)
    units = _cache_replayability_units_from_store(store_obj, row_limit=scan_limit)
    cache_rows_after = _cache_table_count(store_obj.conn)
    return _cache_replay_dry_run_from_units(
        units,
        rules=rules,
        limit=output_limit,
        cache_rows_before=cache_rows_before,
        cache_rows_after=cache_rows_after,
    )
