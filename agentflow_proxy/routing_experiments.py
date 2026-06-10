from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Callable

import yaml

from agentflow_proxy.crunch import build_embedding, sha256_text
from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.store import cosine_similarity, stable_json


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
        "profile_id": "first-safe-openai-codex-ab-v1",
        "enabled": False,
        "kill_switch": False,
        "sample_rate": 0.0,
        "daily_budget_usd": 0.0,
        "min_text_chars": 0,
        "max_text_chars": 8000,
        "providers": ["openai"],
        "source_surfaces": ["openai_responses", "openai_chat", "codex_turn"],
        "model_pairs": [
            {"requested_model": "gpt-5-codex", "routed_model": "gpt-5-mini"},
            {"requested_model": "gpt-5.4", "routed_model": "gpt-5.4-mini"},
        ],
        "workflow_phases": [],
        "categories": ["chat", "short-completion"],
        "similarity_threshold": 0.86,
        "min_samples_for_confidence": 20,
        "store_response_bodies": False,
    }


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(Path.home() / ".agentflow" / filename)
    return candidates


def _apply_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    if data.get("profile_id") not in (None, ""):
        policy["profile_id"] = str(data["profile_id"])
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
    for key in ("providers", "source_surfaces", "workflow_phases", "categories"):
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
ROUTING_EXPERIMENT_SAMPLE_RATE = float(ROUTING_EXPERIMENT_POLICY["sample_rate"])
ROUTING_EXPERIMENT_DAILY_BUDGET_USD = float(ROUTING_EXPERIMENT_POLICY["daily_budget_usd"])
ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD = float(ROUTING_EXPERIMENT_POLICY["similarity_threshold"])
ROUTING_EXPERIMENT_MIN_SAMPLES = int(ROUTING_EXPERIMENT_POLICY["min_samples_for_confidence"])
ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES = bool(ROUTING_EXPERIMENT_POLICY["store_response_bodies"])
ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE = "routing_experiment_outcome"


def _today_shadow_spend_usd(store_obj: Any | None, *, provider: str | None = None, source_surface: str | None = None) -> float:
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
    category = str(routing_meta.get("category") or "")
    workflow_phase = str(routing_meta.get("workflow_phase") or "")
    text_chars = int(routing_meta.get("text_chars") or 0)
    categories = set(str(c) for c in ROUTING_EXPERIMENT_POLICY.get("categories") or [])
    budget_limit = ROUTING_EXPERIMENT_DAILY_BUDGET_USD
    budget_spent = _today_shadow_spend_usd(store_obj, provider=provider, source_surface=source_surface)
    budget_remaining = max(0.0, budget_limit - budget_spent)

    meta = {
        "schema": "agentflow.routing_experiment_decision.v1",
        "enabled": ROUTING_EXPERIMENT_ENABLED,
        "kill_switch": bool(ROUTING_EXPERIMENT_POLICY.get("kill_switch")),
        "status": "skipped",
        "sampled": False,
        "reason": "disabled",
        "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
        "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
        "sample_rate": ROUTING_EXPERIMENT_SAMPLE_RATE,
        "daily_budget_usd": round(budget_limit, 6),
        "profile_id": str(ROUTING_EXPERIMENT_POLICY.get("profile_id") or ""),
        "budget_spent_usd": round(budget_spent, 6),
        "budget_remaining_usd": round(budget_remaining, 6),
        "budget_exhausted": budget_limit <= 0 or budget_spent >= budget_limit,
        "similarity_threshold": ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD,
        "min_samples_for_confidence": ROUTING_EXPERIMENT_MIN_SAMPLES,
        "provider": provider,
        "source_surface": source_surface,
        "requested_model": requested,
        "routed_model": routed,
        "shadow_model": requested,
        "category": category,
        "workflow_phase": workflow_phase,
        "text_chars": text_chars,
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
    if stream:
        meta["reason"] = "streaming"
        return meta
    if not requested or requested == routed:
        meta["reason"] = "not-routed-down"
        return meta
    if not _model_pair_allowed(requested, routed):
        meta["reason"] = "model-pair-not-enabled"
        return meta
    if routing_meta.get("fallback_reason"):
        meta["reason"] = "fallback-used"
        return meta
    min_chars = int(ROUTING_EXPERIMENT_POLICY.get("min_text_chars") or 0)
    max_chars = int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0)
    if text_chars < min_chars:
        meta["reason"] = "request-too-small"
        return meta
    if max_chars > 0 and text_chars > max_chars:
        meta["reason"] = "request-too-large"
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
    if ROUTING_EXPERIMENT_SAMPLE_RATE <= 0:
        meta["reason"] = "sample-rate-zero"
        return meta
    if random_value() >= ROUTING_EXPERIMENT_SAMPLE_RATE:
        meta["reason"] = "not-sampled"
        return meta

    meta["status"] = "selected"
    meta["sampled"] = True
    meta["reason"] = "sampled-routed-down-call"
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
    requested_model = str(routing_meta.get("requested_model") or experiment_meta.get("requested_model") or "")
    routed_model = str(routing_meta.get("routed_model") or experiment_meta.get("routed_model") or "")
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


def build_routing_experiment_report(store_obj: Any, *, limit: int = 20) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 1000))
    conn = store_obj.conn
    rows = conn.execute(
        """
        select coalesce(provider, 'anthropic') as provider,
               coalesce(source_surface, 'anthropic_messages') as source_surface,
               requested_model,
               routed_model,
               coalesce(category, 'unknown') as category,
               coalesce(routing_reason, 'unknown') as routing_reason,
               count(*) as samples,
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
                              then primary_latency_ms - shadow_latency_ms else null end), 2) as avg_latency_delta_ms,
               max(created_at) as last_sample_at
        from routing_experiments
        group by coalesce(provider, 'anthropic'), coalesce(source_surface, 'anthropic_messages'),
                 requested_model, routed_model, coalesce(category, 'unknown'), coalesce(routing_reason, 'unknown')
        order by samples desc, last_sample_at desc
        limit ?
        """,
        (capped,),
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        compared_samples = int(item.get("compared_samples") or 0)
        avg_similarity = item.get("avg_similarity")
        pass_rate = item.get("pass_rate")
        confidence_score = 0.0
        if avg_similarity is not None and compared_samples > 0:
            confidence_score = float(avg_similarity) * min(1.0, compared_samples / ROUTING_EXPERIMENT_MIN_SAMPLES)
        item["compared_samples"] = compared_samples
        item["avg_similarity"] = round(float(avg_similarity), 6) if avg_similarity is not None else None
        item["pass_rate"] = round(float(pass_rate), 4) if pass_rate is not None else None
        item["confidence_score"] = round(confidence_score, 6)
        item["min_samples_for_confidence"] = ROUTING_EXPERIMENT_MIN_SAMPLES
        candidates.append(item)

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
    today_spend = _today_shadow_spend_usd(store_obj)
    budget_limit = ROUTING_EXPERIMENT_DAILY_BUDGET_USD
    feedback_status_counts: dict[str, int] = {}
    for row in conn.execute("select experiment_json from routing_experiments where experiment_json is not null").fetchall():
        try:
            experiment = yaml.safe_load(row["experiment_json"]) or {}
        except Exception:
            status = "invalid-json"
        else:
            feedback = experiment.get("managed_feedback") if isinstance(experiment, dict) else None
            status = str((feedback or {}).get("status") or "not-exported") if isinstance(feedback, dict) else "not-exported"
        feedback_status_counts[status] = feedback_status_counts.get(status, 0) + 1

    decision_reason_counts: dict[tuple[str, str, str, str], int] = {}
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
        try:
            routing = yaml.safe_load(row["routing_json"]) or {}
        except Exception:
            reason = "invalid-routing-json"
            status = "unknown"
        else:
            experiment = routing.get("routing_experiment") if isinstance(routing, dict) else None
            if not isinstance(experiment, dict):
                continue
            reason = str(experiment.get("reason") or "unknown")
            status = str(experiment.get("status") or "unknown")
        key = (str(row["provider"]), str(row["source_surface"]), status, reason)
        decision_reason_counts[key] = decision_reason_counts.get(key, 0) + 1
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

    compared_total = int((summary_row or {}).get("compared_samples") or 0)
    avg_similarity_total = (summary_row or {}).get("avg_similarity")
    pass_rate_total = (summary_row or {}).get("pass_rate")
    return {
        "schema": "agentflow.routing_experiment_report.v1",
        "generated_at": utc_now(),
        "policy": {
            "profile_id": str(ROUTING_EXPERIMENT_POLICY.get("profile_id") or ""),
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
            "model_pairs": list(ROUTING_EXPERIMENT_POLICY.get("model_pairs") or []),
            "categories": list(ROUTING_EXPERIMENT_POLICY.get("categories") or []),
            "workflow_phases": list(ROUTING_EXPERIMENT_POLICY.get("workflow_phases") or []),
            "min_text_chars": int(ROUTING_EXPERIMENT_POLICY.get("min_text_chars") or 0),
            "max_text_chars": int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0),
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
        },
        "decision_reasons": decision_reasons,
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
