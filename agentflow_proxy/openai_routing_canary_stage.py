from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from agentflow_proxy.policy_files import stage_policy_draft
from agentflow_proxy.pricing import pricing_basis
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_routing_canary_stage.v1"
OMISSION_SCHEMA = "agentflow.openai_routing_canary_stage_omission.v1"
STAGED_SCHEMA = "agentflow.openai_routing_canary_staged_draft.v1"
PRIVACY = {
    "local_only": True,
    "metadata_only": True,
    "raw_prompts_included": False,
    "raw_messages_included": False,
    "provider_bodies_included": False,
    "raw_provider_bodies_included": False,
    "request_ids_included": False,
    "raw_request_ids_included": False,
    "session_ids_included": False,
    "raw_session_ids_included": False,
    "cache_keys_included": False,
    "file_paths_included": False,
    "secrets_included": False,
    "provider_calls_made": False,
    "managed_server_calls_made": False,
}
PAYLOAD_PRIVACY = {
    "local_only": True,
    "metadata_only": True,
    "provider_calls_made": False,
    "managed_server_calls_made": False,
}

SUPPORTED_SURFACES = {"openai", "openai_responses", "openai_chat", "openai_chat_completions"}
SUPPORTED_ENDPOINTS = {"responses", "chat", "chat_completions"}
DEFAULT_EXCLUDED_CATEGORIES = ["tool-result", "tool-heavy", "tool-light", "code-gen", "long-context"]
PASS_THROUGH_SCHEMA = "agentflow.pass_through_routing_activation_candidates.v1"
RAW_KEYS = {
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "content",
    "file_path",
    "messages",
    "password",
    "prompt",
    "provider_body",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_id",
    "response",
    "secret",
    "session_id",
    "system_prompt",
    "thread_id",
    "tool_input",
    "tool_payload",
    "tool_result",
    "transcript",
}
ALLOWED_RAW_FLAG_KEYS = {
    "raw_prompts_included",
    "raw_messages_included",
    "raw_provider_bodies_included",
    "raw_request_ids_included",
    "raw_session_ids_included",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, *items: Any) -> str:
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")[:72] or "openai-routing-canary"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item or "").strip()})
    text = _string(value)
    return [text] if text else []


def _privacy_errors(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key).strip()
                lowered = key_text.lower()
                child_path = f"{item_path}.{key_text}"
                if lowered in ALLOWED_RAW_FLAG_KEYS:
                    continue
                if lowered in RAW_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "OpenAI canary staging accepts metadata only, not raw prompts, provider bodies, request identifiers, file paths, cache keys, or secrets",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def _candidate_id(candidate: dict[str, Any]) -> str:
    explicit = _string(candidate.get("candidate_id") or candidate.get("target_candidate_id"))
    if explicit:
        safe = "".join(char for char in explicit if char.isalnum() or char in {"-", "_", ":"}).strip("-_:")
        if safe:
            return safe[:120]
    return _stable_id(
        "openai-routing-candidate",
        candidate.get("source_surface"),
        candidate.get("endpoint"),
        candidate.get("requested_model"),
        candidate.get("target_model"),
        candidate.get("category"),
        candidate.get("text_bucket"),
        candidate.get("token_bucket"),
    )


def _bucket_range(bucket: Any, *, kind: str) -> tuple[int, int]:
    text_ranges = {
        "lt-1_5k": (0, 1500),
        "1_5k-6k": (1500, 6000),
        "6k-32k": (6000, 32000),
        "gte-32k": (32000, 0),
    }
    token_ranges = {
        "lt-1k": (0, 1000),
        "1k-4k": (1000, 4000),
        "4k-16k": (4000, 16000),
        "gte-16k": (16000, 0),
    }
    table = token_ranges if kind == "tokens" else text_ranges
    return table.get(_string(bucket), (0, 0))


def _pricing_known(model: str) -> bool:
    return bool(model and pricing_basis(model, provider="openai").get("cost_known"))


def _candidate_omission_reason(candidate: dict[str, Any], *, min_samples: int) -> str | None:
    source_surface = _string(candidate.get("source_surface")).lower()
    endpoint = _string(candidate.get("endpoint")).lower()
    requested = _string(candidate.get("requested_model"))
    target = _string(candidate.get("target_model"))
    blockers = {str(item).strip() for item in candidate.get("blockers") or [] if str(item or "").strip()}

    if source_surface not in SUPPORTED_SURFACES:
        return "unsupported-source-surface"
    if endpoint and endpoint not in SUPPORTED_ENDPOINTS:
        return "unsupported-endpoint"
    if _as_int(candidate.get("matched_count")) < min_samples:
        return "insufficient-samples"
    if _as_int(candidate.get("current_routed_count")) > 0:
        return "already-routed"
    if _as_int(candidate.get("blocked_count")) > 0:
        if "tools-disabled" in blockers or bool(candidate.get("has_tools")):
            return "tool-safety-blocker"
        if "missing-baseline-cost" in blockers:
            return "missing-baseline-cost"
        if "unsupported-target-model" in blockers:
            return "unsupported-target-model"
        if "unknown-model-family" in blockers:
            return "unknown-pricing"
        if "high-recent-error-rate" in blockers:
            return "high-error-rate"
        if "high-recent-retry-rate" in blockers:
            return "high-retry-rate"
        if "stream-only-evidence" in blockers:
            return "stream-only-evidence"
        if "category-safety-blocker" in blockers:
            return "category-safety-blocker"
        return sorted(blockers)[0] if blockers else "blocked-candidate"
    if bool(candidate.get("has_tools")) and not bool(candidate.get("allow_tools")):
        return "tool-safety-blocker"
    if bool(candidate.get("stream")):
        return "streaming-not-enabled"
    if _as_float(candidate.get("error_rate")) > 0.05:
        return "high-error-rate"
    if _as_float(candidate.get("retry_rate")) > 0.20:
        return "high-retry-rate"
    if not _pricing_known(requested):
        return "unknown-pricing"
    if not _pricing_known(target):
        return "unsupported-target-model"
    if "mini" not in target.lower() or "mini" in requested.lower() or target.lower() == requested.lower():
        return "not-large-to-mini"
    if _as_float(candidate.get("projected_savings_usd")) <= 0:
        return "missing-baseline-cost"
    return None


def _omission(candidate: dict[str, Any], reason: str, *, path: str | None = None) -> dict[str, Any]:
    return {
        "schema": OMISSION_SCHEMA,
        "status": "omitted",
        "reason": reason,
        "path": path,
        "target_candidate_id": _candidate_id(candidate),
        "provider": "openai",
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "requested_model": candidate.get("requested_model"),
        "target_model": candidate.get("target_model"),
        "blocker_codes": _string_list(candidate.get("blockers")),
        "privacy": PRIVACY,
    }


def _rollback_metadata() -> dict[str, Any]:
    return {
        "schema": "agentflow.openai_routing_canary_rollback.v1",
        "rollback_action_type": "disable_openai_canary",
        "rollback_canary_fraction": 0.0,
        "rollback_holdout_fraction": 0.0,
        "rollback_reason_codes": [
            "safety-stop-observed",
            "error-rate-regression",
            "retry-or-fallback-regression",
            "latency-regression",
            "operator-requested",
        ],
        "preserve_previous_rule_required": True,
    }


def _candidate_payload(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    canary_fraction: float,
    holdout_fraction: float,
) -> tuple[str, dict[str, Any]]:
    candidate_id = _candidate_id(candidate)
    text_min, text_max = _bucket_range(candidate.get("text_bucket"), kind="text")
    token_min, token_max = _bucket_range(candidate.get("token_bucket"), kind="tokens")
    matched = max(1, _as_int(candidate.get("matched_count"), 1))
    avg_input_tokens = math.ceil(_as_int(candidate.get("input_tokens")) / matched)
    if token_max <= 0 and avg_input_tokens > 0:
        token_max = max(avg_input_tokens * 2, token_min)
    if text_max <= 0:
        text_max = max(_as_int(candidate.get("input_tokens")) * 4 // matched, text_min)

    evidence_category = _string(candidate.get("category") or "chat")
    category = evidence_category if evidence_category in {"tool-result", "tool-heavy", "tool-light", "long-context", "short-completion", "code-gen", "chat"} else "chat"
    excluded = [item for item in DEFAULT_EXCLUDED_CATEGORIES if item != category]
    allow_tools = bool(candidate.get("allow_tools"))
    policy_id = f"local-openai-routing-canary-{_safe_id(candidate_id)}"
    rollback = _rollback_metadata()
    reason_codes = sorted({
        "eligible-openai-large-to-mini",
        _string(candidate.get("simulated_reason") or "openai-routing-opportunity"),
        f"endpoint:{_string(candidate.get('endpoint') or 'unknown')}",
    })
    bounded_canary_fraction = _bounded_fraction(canary_fraction, 0.05)
    bounded_holdout_fraction = _bounded_fraction(holdout_fraction, 0.10)
    projected_holdout_count = min(matched, int(math.ceil(matched * bounded_holdout_fraction)))
    projected_canary_count = min(
        max(0, matched - projected_holdout_count),
        int(math.ceil(matched * max(0.0, min(1.0, bounded_holdout_fraction + bounded_canary_fraction) - bounded_holdout_fraction))),
    )
    savings_per_1000 = _as_float(candidate.get("estimated_savings_per_1000_calls_usd"))
    if savings_per_1000 <= 0:
        savings_per_1000 = (_as_float(candidate.get("projected_savings_usd")) / matched) * 1000.0
    canary = {
        "enabled": False,
        "review_only": True,
        "policy_id": policy_id,
        "target_candidate_id": candidate_id,
        "policy_source": "local-manual",
        "model_pattern": _string(candidate.get("requested_model")),
        "target_model": _string(candidate.get("target_model")),
        "eligible_categories": [category],
        "excluded_categories": excluded,
        "allow_tools": allow_tools,
        "allow_stream": False,
        "min_text_chars": text_min,
        "max_text_chars": text_max,
        "min_input_tokens_est": token_min,
        "max_input_tokens_est": token_max,
        "canary_fraction": bounded_canary_fraction,
        "holdout_fraction": bounded_holdout_fraction,
        "salt": _stable_id("openai-routing-canary-salt", candidate_id, candidate.get("requested_model"), candidate.get("target_model")),
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 20,
            "min_holdout_samples": 10,
            "max_error_rate": 0.03,
            "max_retry_rate": 0.10,
            "max_fallback_rate": 0.10,
            "max_latency_regression_ratio": 1.50,
            "limit": 1000,
        },
        "fallback": {
            "enabled": True,
            "fallback_model": _string(candidate.get("requested_model")),
            "reason_codes": [
                "rate_limited",
                "upstream_error",
                "local-canary-safety-stop",
                "operator-rollback",
            ],
        },
        "promotion": {
            "schema": "agentflow.openai_routing_canary_stage_metadata.v1",
            "source": "pass_through_routing_report" if report.get("schema") == PASS_THROUGH_SCHEMA else "openai_routing_report",
            "source_report_schema": report.get("schema"),
            "source_report_generated_at": report.get("generated_at"),
            "candidate_id": candidate_id,
            "provider": "openai",
            "source_surface": candidate.get("source_surface"),
            "endpoint": candidate.get("endpoint"),
            "requested_model": candidate.get("requested_model"),
            "target_model": candidate.get("target_model"),
            "category": evidence_category,
            "applied_canary_category": category,
            "matched_count": _as_int(candidate.get("matched_count")),
            "source_actionability": candidate.get("actionability"),
            "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 6),
            "estimated_savings_per_1000_calls_usd": round(savings_per_1000, 6),
            "estimated_baseline_cost_usd": round(_as_float(candidate.get("estimated_baseline_cost_usd")), 6),
            "projected_cohort_counts": {
                "matched": matched,
                "canary_applied": projected_canary_count,
                "canary_holdout": projected_holdout_count,
                "bypassed_or_disabled": max(0, matched - projected_canary_count - projected_holdout_count),
            },
            "reason_codes": reason_codes,
            "rollback_metadata": rollback,
            "privacy": PAYLOAD_PRIVACY,
        },
    }
    return candidate_id, {"openai_canary": canary}


def _candidate_from_pass_through_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    provider = _string(bucket.get("provider")).lower()
    requested = _string(bucket.get("requested_model"))
    target = _string(bucket.get("candidate_target_model") or bucket.get("target_model"))
    category = _string(bucket.get("category") or "unknown")
    sample_count = _as_int(bucket.get("sample_count") or bucket.get("count"))
    savings_per_1000 = _as_float(bucket.get("estimated_savings_per_1000_calls_usd"))
    actionability = _string(bucket.get("actionability"))
    blockers: list[str] = []
    if actionability and actionability != "actionable":
        blockers.append(actionability)
    if provider != "openai":
        blockers.append("unsupported-source-surface")
    if category in {"tool-result", "tool-heavy", "code-gen", "long-context"}:
        blockers.append("category-safety-blocker")
    has_tools = category.startswith("tool-")
    allow_tools = category == "tool-light"
    return {
        "candidate_id": _stable_id(
            "openai-pass-through-route",
            bucket.get("rank"),
            bucket.get("source_surface"),
            bucket.get("endpoint"),
            requested,
            bucket.get("routed_model"),
            target,
            category,
        ),
        "source_surface": bucket.get("source_surface") or "openai_provider_request",
        "endpoint": bucket.get("endpoint") or "responses",
        "requested_model": requested,
        "target_model": target,
        "category": category,
        "matched_count": sample_count,
        "blocked_count": sample_count if blockers else 0,
        "current_routed_count": 0,
        "projected_savings_usd": round((savings_per_1000 * sample_count) / 1000.0, 6),
        "estimated_savings_per_1000_calls_usd": savings_per_1000,
        "estimated_baseline_cost_usd": 0.0,
        "text_bucket": bucket.get("text_bucket") or "unknown",
        "token_bucket": bucket.get("token_bucket") or "unknown",
        "input_tokens": _as_int(bucket.get("input_tokens")),
        "has_tools": has_tools,
        "allow_tools": allow_tools,
        "stream": False,
        "simulated_policy": bucket.get("required_local_executor") or "openai-routing-canary",
        "simulated_reason": bucket.get("candidate_reason") or "pass-through-routing-activation-candidate",
        "actionability": actionability,
        "blockers": sorted(set(blockers)),
    }


def _candidate_list_from_report(report: dict[str, Any]) -> list[Any] | None:
    candidates = report.get("candidates")
    if isinstance(candidates, list):
        return candidates
    if report.get("schema") == PASS_THROUGH_SCHEMA and isinstance(report.get("buckets"), list):
        return [
            _candidate_from_pass_through_bucket(bucket)
            for bucket in report.get("buckets") or []
            if isinstance(bucket, dict) and _string(bucket.get("provider")).lower() == "openai"
        ]
    return None


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "summary": {
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "staged_count": 0,
            "omitted_count": 0,
        },
        "staged_drafts": [],
        "omitted": [],
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


async def stage_openai_routing_canary_drafts(
    routing_report: dict[str, Any],
    *,
    draft_id: str | None = None,
    workspace: str | None = None,
    canary_fraction: float = 0.05,
    holdout_fraction: float = 0.10,
    min_samples: int = 5,
    top_candidates: int | None = 1,
) -> dict[str, Any]:
    if not isinstance(routing_report, dict):
        return _error_result("invalid_report", "OpenAI routing report must be a JSON object")
    raw_errors = _privacy_errors(routing_report)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "OpenAI routing report contains raw prompt, response, provider body, identifier, file path, cache key, or secret fields",
            errors=raw_errors,
        )
    candidates = _candidate_list_from_report(routing_report)
    if not isinstance(candidates, list):
        return _error_result("invalid_report", "OpenAI routing report must include a candidates list or pass-through buckets list")

    staged: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    eligible_count = 0
    stage_limit = None if top_candidates is None else max(0, int(top_candidates))
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            omitted.append({
                "schema": OMISSION_SCHEMA,
                "status": "omitted",
                "reason": "invalid-candidate",
                "path": f"$.candidates[{index}]",
                "target_candidate_id": None,
                "privacy": PRIVACY,
            })
            continue
        reason = _candidate_omission_reason(candidate, min_samples=min_samples)
        if reason is not None:
            omitted.append(_omission(candidate, reason, path=f"$.candidates[{index}]"))
            continue
        eligible_count += 1
        if stage_limit is not None and len(staged) >= stage_limit:
            omitted.append(_omission(candidate, "lower-ranked-candidate-not-staged", path=f"$.candidates[{index}]"))
            continue
        candidate_id, payload = _candidate_payload(
            routing_report,
            candidate,
            canary_fraction=canary_fraction,
            holdout_fraction=holdout_fraction,
        )
        candidate_draft_id = draft_id or _safe_id(candidate_id)
        if draft_id and len(candidates) > 1:
            candidate_draft_id = f"{draft_id}-{len(staged) + 1}"
        draft_result = await stage_policy_draft(
            payload,
            section="routing",
            draft_id=candidate_draft_id,
            workspace=workspace,
            metadata={
                "openai_routing_canary_stage": {
                    "schema": "agentflow.openai_routing_canary_stage_manifest_metadata.v1",
                    "candidate_id": candidate_id,
                    "source_report_schema": routing_report.get("schema"),
                    "privacy": PRIVACY,
                }
            },
        )
        canary = payload["openai_canary"]
        staged.append({
            "schema": STAGED_SCHEMA,
            "candidate_id": candidate_id,
            "section": "routing",
            "target_local_policy": "openai_canary",
            "draft_id": draft_result.get("draft_id"),
            "ok": bool(draft_result.get("ok")),
            "workspace": draft_result.get("workspace"),
            "bundle_path": draft_result.get("bundle_path"),
            "manifest_path": draft_result.get("manifest_path"),
            "changed_sections": (draft_result.get("draft") or {}).get("changed_sections", []),
            "change_count": (draft_result.get("draft") or {}).get("change_count", 0),
            "provider": "openai",
            "source_surface": candidate.get("source_surface"),
            "endpoint": candidate.get("endpoint"),
            "requested_model": candidate.get("requested_model"),
            "target_model": candidate.get("target_model"),
            "canary_fraction": canary["canary_fraction"],
            "holdout_fraction": canary["holdout_fraction"],
            "reason_codes": canary["promotion"]["reason_codes"],
            "projected_savings_usd": canary["promotion"]["projected_savings_usd"],
            "estimated_savings_per_1000_calls_usd": canary["promotion"]["estimated_savings_per_1000_calls_usd"],
            "projected_cohort_counts": canary["promotion"]["projected_cohort_counts"],
            "rollback_metadata": canary["promotion"]["rollback_metadata"],
            "draft": draft_result.get("draft"),
            "error": draft_result.get("error"),
            "privacy": PRIVACY,
        })

    omission_counts: dict[str, int] = {}
    for item in omitted:
        reason = str(item.get("reason") or "unknown")
        omission_counts[reason] = omission_counts.get(reason, 0) + 1
    ok = bool(staged) and all(bool(item.get("ok")) for item in staged)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "generated_at": utc_now(),
        "source_report_schema": routing_report.get("schema"),
        "source_report_generated_at": routing_report.get("generated_at"),
        "summary": {
            "candidate_count": len(candidates),
            "eligible_candidate_count": eligible_count,
            "staged_count": len(staged),
            "omitted_count": len(omitted),
            "projected_savings_usd": round(sum(_as_float(item.get("projected_savings_usd")) for item in staged), 6),
            "estimated_savings_per_1000_calls_usd": round(
                max((_as_float(item.get("estimated_savings_per_1000_calls_usd")) for item in staged), default=0.0),
                6,
            ),
            "omission_reason_counts": [{"value": key, "count": omission_counts[key]} for key in sorted(omission_counts)],
        },
        "staged_drafts": staged,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
        "error": None if ok else {"type": "no_staged_drafts", "message": "no eligible OpenAI large-to-mini routing candidates were staged"},
    }
