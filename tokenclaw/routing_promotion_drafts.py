from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from tokenclaw.policy_files import stage_policy_draft
from tokenclaw.store import utc_now


ROUTING_PROMOTION_DRAFT_STAGE_SCHEMA = "tokenclaw.routing_promotion_draft_stage.v1"
ROUTING_PROMOTION_DRAFT_PRIVACY = {
    "local_only": True,
    "metadata_only": True,
    "raw_prompts_included": False,
    "raw_responses_included": False,
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

_OPENAI_SURFACES = {"openai", "openai_responses", "openai_chat", "openai_chat_completions", "codex_turn"}
_FORBIDDEN_KEYS = {
    "api_key",
    "api_keys",
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "content",
    "contents",
    "file_content",
    "file_contents",
    "file_path",
    "file_paths",
    "message",
    "messages",
    "password",
    "passwords",
    "prompt",
    "prompts",
    "provider_body",
    "raw_context",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_id",
    "request_ids",
    "response",
    "secret",
    "secrets",
    "session_id",
    "session_ids",
    "system_prompt",
    "thread_id",
    "thread_ids",
    "tool_input",
    "tool_inputs",
    "tool_payload",
    "tool_payloads",
    "tool_result",
    "tool_results",
    "transcript",
    "transcripts",
}
_ALLOWED_RAW_FLAG_KEYS = {
    "raw_content_included",
    "raw_messages_included",
    "raw_params_included",
    "raw_payload_included",
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_responses_included_by_default",
    "raw_responses_included",
    "raw_request_ids_included",
    "raw_session_ids_included",
    "raw_tool_payloads_included",
    "raw_transcripts_included",
}
_SERVER_CONTENT_KEYS = {
    "replacement_prompt",
    "provider_body_patch",
    "provider_body_rewrite",
    "requires_provider_body_rewrite",
    "requires_server_content_processing",
    "server_content_processing",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, *items: Any) -> str:
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(*values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                text = _string(item)
                if text:
                    result.append(text)
        else:
            text = _string(value)
            if text:
                result.append(text)
    return sorted(set(result))


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
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


def _privacy_errors(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key).strip()
                lowered = key_text.lower()
                child_path = f"{item_path}.{key_text}"
                if lowered in _ALLOWED_RAW_FLAG_KEYS:
                    continue
                if lowered in _FORBIDDEN_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "routing promotion drafts accept metadata only, not raw prompt, response, provider body, identifier, file path, cache key, or secret fields",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def _requires_server_content(candidate: dict[str, Any]) -> bool:
    for key in _SERVER_CONTENT_KEYS:
        value = candidate.get(key)
        if value not in (None, "", False):
            return True
    omitted = candidate.get("omitted_actions")
    if isinstance(omitted, list):
        for item in omitted:
            if isinstance(item, dict) and _string(item.get("action")) in {"prompt_replacement", "provider_body_rewrite"}:
                return True
    return False


def _candidate_id(candidate: dict[str, Any]) -> str:
    explicit = _string(candidate.get("candidate_id") or candidate.get("target_candidate_id"))
    if explicit:
        safe = "".join(char for char in explicit if char.isalnum() or char in {"-", "_", ":"}).strip("-_:")
        if safe:
            return safe[:96]
    return _stable_id(
        "routing-promotion-candidate",
        candidate.get("provider"),
        candidate.get("source_surface"),
        candidate.get("requested_model"),
        candidate.get("routed_model"),
        candidate.get("category"),
        candidate.get("workflow_phase"),
        candidate.get("stream"),
        candidate.get("mode"),
    )


def _source_policy_allowed(report: dict[str, Any], candidate: dict[str, Any]) -> bool:
    report_policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    for source in (candidate.get("policy_source"), report_policy.get("policy_source")):
        if source in (None, ""):
            continue
        if str(source) not in {"local-default", "local-manual", "managed-recommended"}:
            return False
    return True


def _candidate_promotion(candidate: dict[str, Any]) -> dict[str, Any]:
    promotion = candidate.get("promotion")
    if isinstance(promotion, dict):
        return promotion
    return {
        "verdict": candidate.get("promotion_verdict") or candidate.get("verdict"),
        "promotion_ready": candidate.get("promotion_verdict") == "promote" or candidate.get("verdict") == "promote",
        "reason_codes": candidate.get("promotion_reason_codes") or candidate.get("reason_codes") or [],
        "thresholds": candidate.get("thresholds") or {},
        "coverage": {
            "samples": candidate.get("samples"),
            "compared_samples": candidate.get("compared_samples"),
            "compared_coverage": candidate.get("compared_coverage"),
        },
    }


def _omission(candidate: dict[str, Any], reason: str, *, path: str | None = None) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.routing_promotion_draft_omission.v1",
        "status": "omitted",
        "reason": reason,
        "path": path,
        "target_candidate_id": _candidate_id(candidate),
        "promotion_verdict": _string(candidate.get("promotion_verdict") or candidate.get("verdict") or (_candidate_promotion(candidate).get("verdict"))),
        "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
    }


def _candidate_omission_reason(report: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    promotion = _candidate_promotion(candidate)
    verdict = _string(promotion.get("verdict") or candidate.get("promotion_verdict") or candidate.get("verdict"))
    if verdict != "promote" or not bool(promotion.get("promotion_ready", verdict == "promote")):
        return "not-promoted"
    if not _source_policy_allowed(report, candidate):
        return "unsafe-policy-source"
    if _requires_server_content(candidate):
        return "requires-server-content-processing"
    for key in ("provider", "source_surface", "requested_model", "routed_model", "category"):
        if not _string(candidate.get(key)):
            return "missing-evidence"
    if not _string(candidate.get("workflow_phase")):
        return "missing-evidence"
    if candidate.get("stream") in (None, ""):
        return "missing-evidence"
    if candidate.get("samples") in (None, "") or candidate.get("compared_samples") in (None, ""):
        return "missing-evidence"
    reason_codes = {str(code) for code in promotion.get("reason_codes") or candidate.get("promotion_reason_codes") or []}
    if "stale-evidence" in reason_codes:
        return "stale-evidence"
    thresholds = promotion.get("thresholds") if isinstance(promotion.get("thresholds"), dict) else {}
    max_age = _as_float(thresholds.get("freshness_max_age_hours"), 168.0)
    last_age = candidate.get("last_sample_age_hours")
    if last_age is None:
        last_age = _age_hours(candidate.get("last_sample_at"))
    if last_age is None or _as_float(last_age, max_age + 1.0) > max_age:
        return "stale-evidence"
    privacy = candidate.get("privacy")
    if isinstance(privacy, dict):
        if privacy.get("provider_bodies_included") or privacy.get("raw_provider_bodies_included"):
            return "provider-bodies-included"
        if privacy.get("raw_prompts_included") or privacy.get("raw_responses_included"):
            return "raw-content-included"
    return None


def _target_local_policy(candidate: dict[str, Any]) -> tuple[str, str]:
    source_surface = _string(candidate.get("source_surface")).lower()
    provider = _string(candidate.get("provider")).lower()
    if provider == "openai" or source_surface in _OPENAI_SURFACES or "openai" in source_surface or source_surface == "codex_turn":
        return "routing", "openai_canary"
    return "routing", "phase_canary"


def _policy_id(candidate_id: str, target_key: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in candidate_id.lower()).strip("-")
    safe = safe[:54] or hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"routing-promotion-{target_key.replace('_', '-')}-{safe}"


def _evidence_summary(report: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    promotion = _candidate_promotion(candidate)
    return {
        "schema": "tokenclaw.routing_promotion_evidence_summary.v1",
        "source_report_schema": report.get("schema"),
        "source_report_generated_at": report.get("generated_at"),
        "provider": candidate.get("provider"),
        "source_surface": candidate.get("source_surface"),
        "stream": bool(candidate.get("stream")),
        "mode": candidate.get("mode"),
        "promotion_scope": promotion.get("promotion_scope"),
        "evidence_kind": promotion.get("evidence_kind"),
        "samples": _as_int(candidate.get("samples")),
        "compared_samples": _as_int(candidate.get("compared_samples")),
        "compared_coverage": candidate.get("compared_coverage"),
        "avg_similarity": candidate.get("avg_similarity"),
        "pass_rate": candidate.get("pass_rate"),
        "primary_error_rate": candidate.get("primary_error_rate"),
        "shadow_error_rate": candidate.get("shadow_error_rate"),
        "fallback_or_retry_count": _as_int(candidate.get("fallback_or_retry_count")),
        "cost_delta_usd": candidate.get("cost_delta_usd"),
        "avg_latency_delta_ms": candidate.get("avg_latency_delta_ms"),
        "last_sample_at": candidate.get("last_sample_at"),
        "last_sample_age_hours": candidate.get("last_sample_age_hours"),
        "effective_min_text_chars": candidate.get("effective_min_text_chars"),
        "effective_max_text_chars": candidate.get("effective_max_text_chars"),
        "promotion_reason_codes": list(promotion.get("reason_codes") or []),
        "thresholds": promotion.get("thresholds") if isinstance(promotion.get("thresholds"), dict) else {},
        "coverage": promotion.get("coverage") if isinstance(promotion.get("coverage"), dict) else {},
        "budget": promotion.get("budget") if isinstance(promotion.get("budget"), dict) else {},
    }


def _safety_stop(candidate: dict[str, Any]) -> dict[str, Any]:
    promotion = _candidate_promotion(candidate)
    thresholds = promotion.get("thresholds") if isinstance(promotion.get("thresholds"), dict) else {}
    coverage = promotion.get("coverage") if isinstance(promotion.get("coverage"), dict) else {}
    return {
        "enabled": True,
        "window_hours": 24,
        "min_samples": max(5, min(20, _as_int(coverage.get("compared_samples"), 10))),
        "min_holdout_samples": 5,
        "max_error_rate": _as_float(thresholds.get("max_shadow_error_rate"), 0.05),
        "max_retry_rate": 0.20,
        "max_fallback_rate": 0.20,
        "max_latency_regression_ratio": 1.50,
        "limit": 500,
    }


def _routing_payload(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float,
    holdout_fraction: float,
) -> tuple[str, dict[str, Any]]:
    candidate_id = _candidate_id(candidate)
    section, target_key = _target_local_policy(candidate)
    policy_id = _policy_id(candidate_id, target_key)
    category = _string(candidate.get("category"))
    workflow_phase = _string(candidate.get("workflow_phase"))
    requested = _string(candidate.get("requested_model"))
    routed = _string(candidate.get("routed_model"))
    report_policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    min_text_chars = _as_int(report_policy.get("min_text_chars"), 0)
    max_text_chars = _as_int(report_policy.get("max_text_chars"), 30000 if target_key == "phase_canary" else 8000)
    min_text_chars = _as_int(candidate.get("effective_min_text_chars"), min_text_chars)
    max_text_chars = _as_int(candidate.get("effective_max_text_chars"), max_text_chars)
    provider = _string(candidate.get("provider"))
    source_surface = _string(candidate.get("source_surface"))
    stream = bool(candidate.get("stream"))
    canary = {
        "enabled": True,
        "policy_id": policy_id,
        "target_candidate_id": candidate_id,
        "provider": provider,
        "source_surface": source_surface,
        "app_family": candidate.get("app_family") or "anthropic",
        "policy_source": "local-manual",
        "model_pattern": requested,
        "target_model": routed,
        "requested_model": requested,
        "routed_model": routed,
        "stream": stream,
        "eligible_categories": _string_list(category),
        "excluded_categories": ["code-gen"] if category != "code-gen" else [],
        "min_text_chars": min_text_chars,
        "max_text_chars": max_text_chars,
        "canary_fraction": _bounded_fraction(initial_canary_fraction, 0.10),
        "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
        "salt": _stable_id("routing-promotion-salt", candidate_id, requested, routed),
        "cohort_unit": "session",
        "safety_gates": {
            "provider": provider,
            "source_surface": source_surface,
            "stream": stream,
            "block_thinking_history": True,
            "block_top_level_thinking": True,
            "strip_model_incompatible_params": True,
            "fallback_to_requested_on_rate_limit": True,
            "content_free": True,
            "provider_calls_made_by_apply": False,
        },
        "safety_stop": _safety_stop(candidate),
        "promotion": {
            "schema": "tokenclaw.routing_promotion_local_draft_metadata.v1",
            "source": "routing_experiment_report",
            "candidate_id": candidate_id,
            "source_surface": candidate.get("source_surface"),
            "provider": candidate.get("provider"),
            "requested_model": requested,
            "routed_model": routed,
            "category": category,
            "workflow_phase": workflow_phase,
            "evidence_summary": _evidence_summary(report, candidate),
            "rollback_metadata": {
                "rollback_action_type": "disable_canary",
                "rollback_canary_fraction": 0.0,
                "rollback_reason_codes": [
                    "safety-stop-observed",
                    "error-rate-regression",
                    "retry-or-fallback-regression",
                    "operator-requested",
                ],
                "preserve_previous_rule_required": True,
            },
            "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
        },
    }
    if target_key == "phase_canary":
        canary["eligible_workflow_phases"] = _string_list(workflow_phase)
        canary["excluded_workflow_phases"] = ["planning", "thinking"] if workflow_phase != "unknown" else ["planning", "thinking", "unknown"]
        canary["min_workflow_phase_confidence"] = "medium"
        return section, {"phase_canary": canary}
    canary["allow_tools"] = False
    canary["allow_stream"] = False
    canary["max_input_tokens_est"] = 2000
    return section, {"openai_canary": canary}


def _routing_experiment_payload(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float,
) -> tuple[str, dict[str, Any]]:
    candidate_id = _candidate_id(candidate)
    promotion = _candidate_promotion(candidate)
    report_policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    thresholds = promotion.get("thresholds") if isinstance(promotion.get("thresholds"), dict) else {}
    policy = {
        "profile_id": _policy_id(candidate_id, "shadow-sampling"),
        "mode": "shadow_candidate_pass_through",
        "enabled": True,
        "kill_switch": False,
        "sample_rate": _bounded_fraction(initial_canary_fraction, 0.10),
        "daily_budget_usd": _as_float(report_policy.get("daily_budget_usd"), 10.0),
        "min_text_chars": _as_int(report_policy.get("min_text_chars"), 0),
        "max_text_chars": _as_int(report_policy.get("max_text_chars"), 8000),
        "providers": _string_list(candidate.get("provider")),
        "source_surfaces": _string_list(candidate.get("source_surface")),
        "model_pairs": [{
            "requested_model": _string(candidate.get("requested_model")),
            "routed_model": _string(candidate.get("routed_model")),
        }],
        "workflow_phases": _string_list(candidate.get("workflow_phase")),
        "categories": _string_list(candidate.get("category")),
        "similarity_threshold": _as_float(thresholds.get("min_similarity_pass_rate"), 0.90),
        "min_samples_for_confidence": _as_int(thresholds.get("min_samples"), 20),
        "store_response_bodies": False,
        "promotion": {
            "schema": "tokenclaw.routing_promotion_shadow_sampling_draft_metadata.v1",
            "candidate_id": candidate_id,
            "evidence_summary": _evidence_summary(report, candidate),
            "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
        },
    }
    return "routing_experiments", policy


def draft_payload_for_candidate(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float = 0.10,
    holdout_fraction: float = 0.10,
) -> tuple[str, dict[str, Any], str]:
    promotion = _candidate_promotion(candidate)
    scope = _string(promotion.get("promotion_scope") or candidate.get("promotion_scope"))
    if scope in {"continue_shadow_sampling", "more_shadow_sampling", "shadow_sampling"}:
        section, payload = _routing_experiment_payload(
            report,
            candidate,
            initial_canary_fraction=initial_canary_fraction,
        )
    else:
        section, payload = _routing_payload(
            report,
            candidate,
            initial_canary_fraction=initial_canary_fraction,
            holdout_fraction=holdout_fraction,
        )
    return section, payload, _candidate_id(candidate)


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": ROUTING_PROMOTION_DRAFT_STAGE_SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "summary": {
            "candidate_count": 0,
            "promoted_candidate_count": 0,
            "staged_count": 0,
            "omitted_count": 0,
        },
        "staged_drafts": [],
        "omitted": [],
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


async def stage_routing_promotion_drafts(
    promotion_report: dict[str, Any],
    *,
    draft_id: str | None = None,
    workspace: str | None = None,
    initial_canary_fraction: float = 0.10,
    holdout_fraction: float = 0.10,
) -> dict[str, Any]:
    if not isinstance(promotion_report, dict):
        return _error_result("invalid_report", "routing promotion report must be a JSON object")

    raw_errors = _privacy_errors(promotion_report)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "routing promotion report contains raw prompt, response, provider body, identifier, file path, cache key, or secret fields",
            errors=raw_errors,
        )

    candidates = promotion_report.get("candidates")
    if not isinstance(candidates, list):
        return _error_result("invalid_report", "routing promotion report must include a candidates list")

    staged: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    promoted_count = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            omitted.append({
                "schema": "tokenclaw.routing_promotion_draft_omission.v1",
                "status": "omitted",
                "reason": "invalid-candidate",
                "path": f"$.candidates[{index}]",
                "target_candidate_id": None,
                "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
            })
            continue
        promotion = _candidate_promotion(candidate)
        verdict = _string(promotion.get("verdict") or candidate.get("promotion_verdict") or candidate.get("verdict"))
        promotion_ready = bool(promotion.get("promotion_ready", verdict == "promote"))
        if verdict == "promote" and promotion_ready:
            promoted_count += 1
        omission_reason = _candidate_omission_reason(promotion_report, candidate)
        if omission_reason == "not-promoted":
            omitted.append(_omission(candidate, omission_reason, path=f"$.candidates[{index}]"))
            continue
        if omission_reason is not None:
            omitted.append(_omission(candidate, omission_reason, path=f"$.candidates[{index}]"))
            continue
        section, payload, candidate_id = draft_payload_for_candidate(
            promotion_report,
            candidate,
            initial_canary_fraction=initial_canary_fraction,
            holdout_fraction=holdout_fraction,
        )
        candidate_draft_id = draft_id or candidate_id.replace(":", "-")
        if len(candidates) > 1 and draft_id:
            candidate_draft_id = f"{draft_id}-{len(staged) + 1}"
        draft_result = await stage_policy_draft(
            payload,
            section=section,
            draft_id=candidate_draft_id,
            workspace=workspace,
        )
        staged.append({
            "schema": "tokenclaw.routing_promotion_staged_draft.v1",
            "candidate_id": candidate_id,
            "section": section,
            "target_local_policy": "openai_canary" if "openai_canary" in payload else ("phase_canary" if "phase_canary" in payload else "policy"),
            "draft_id": draft_result.get("draft_id"),
            "ok": bool(draft_result.get("ok")),
            "workspace": draft_result.get("workspace"),
            "bundle_path": draft_result.get("bundle_path"),
            "manifest_path": draft_result.get("manifest_path"),
            "changed_sections": (draft_result.get("draft") or {}).get("changed_sections", []),
            "change_count": (draft_result.get("draft") or {}).get("change_count", 0),
            "canary_fraction": _bounded_fraction(initial_canary_fraction, 0.10),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10) if section == "routing" else 0.0,
            "evidence_summary": _evidence_summary(promotion_report, candidate),
            "rollback_metadata": (payload.get("phase_canary") or payload.get("openai_canary") or {}).get("promotion", {}).get("rollback_metadata"),
            "draft": draft_result.get("draft"),
            "error": draft_result.get("error"),
            "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
        })

    section_counts: dict[str, int] = {}
    for item in staged:
        section = str(item.get("section") or "unknown")
        section_counts[section] = section_counts.get(section, 0) + 1
    omission_counts: dict[str, int] = {}
    for item in omitted:
        reason = str(item.get("reason") or "unknown")
        omission_counts[reason] = omission_counts.get(reason, 0) + 1

    ok = bool(staged) and all(bool(item.get("ok")) for item in staged)
    return {
        "schema": ROUTING_PROMOTION_DRAFT_STAGE_SCHEMA,
        "ok": ok,
        "generated_at": utc_now(),
        "source_report_schema": promotion_report.get("schema"),
        "source_report_generated_at": promotion_report.get("generated_at"),
        "summary": {
            "candidate_count": len(candidates),
            "promoted_candidate_count": promoted_count,
            "staged_count": len(staged),
            "omitted_count": len(omitted),
            "section_counts": [{"value": key, "count": section_counts[key]} for key in sorted(section_counts)],
            "omission_reason_counts": [{"value": key, "count": omission_counts[key]} for key in sorted(omission_counts)],
        },
        "staged_drafts": staged,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": ROUTING_PROMOTION_DRAFT_PRIVACY,
        "error": None if ok else {"type": "no_staged_drafts", "message": "no promoted routing candidates were staged"},
    }
