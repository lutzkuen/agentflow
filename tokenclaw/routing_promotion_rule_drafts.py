from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.routing_promotion_draft_dry_run.v1"
RULE_DRAFT_SCHEMA = "tokenclaw.routing_promotion_rule_draft.v1"
OMISSION_SCHEMA = "tokenclaw.routing_promotion_draft_omission.v1"

_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")
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
    "raw_payload_included",
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_responses_included",
    "raw_request_bodies_included",
    "raw_session_ids_included",
    "raw_source_reports_included",
    "raw_tool_payloads_included",
    "raw_transcripts_included",
}
_NON_BLOCKING_REASONS = {"target-savings-met", "canary-full-coverage", "promotion-ready", "widen-ready"}
_REGRESSION_REASONS = {
    "applied-error-rate-above-threshold",
    "error-rate-regression",
    "fallback-rate-regression",
    "latency-regression",
    "rate-limit-fallback-regression",
    "retry-rate-regression",
    "rollback-error-rate",
    "rollback-fallback-rate",
}


def _privacy() -> dict[str, Any]:
    return {
        "local_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "read_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, *items: Any) -> str:
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


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


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _reason(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    return text if _REASON_RE.match(text) else "unsanitized-reason-code"


def _reason_list(*values: Any) -> list[str]:
    reasons: set[str] = set()
    for value in values:
        if isinstance(value, list):
            for item in value:
                reason = _reason(item)
                if reason:
                    reasons.add(reason)
        else:
            reason = _reason(value)
            if reason:
                reasons.add(reason)
    return sorted(reasons)


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


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "invalid",
        "generated_at": utc_now(),
        "summary": {
            "candidate_count": 0,
            "routing_candidate_count": 0,
            "promotion_ready_count": 0,
            "draft_count": 0,
            "omitted_count": 0,
        },
        "drafts": [],
        "omitted": [],
        "wrote_active_policy_files": False,
        "wrote_local_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


def _candidate_id(candidate: dict[str, Any]) -> str:
    explicit = _string(candidate.get("candidate_id") or candidate.get("policy_id"))
    if explicit:
        safe = "".join(char for char in explicit if char.isalnum() or char in {"-", "_", ":"}).strip("-_:")
        if safe:
            return safe[:96]
    return _stable_id(
        "routing-promotion-candidate",
        candidate.get("source_evidence_schema"),
        candidate.get("source_rank"),
        candidate.get("provider"),
        candidate.get("source_surface"),
        candidate.get("category"),
        candidate.get("workflow_phase"),
        candidate.get("requested_model"),
        candidate.get("target_model"),
    )


def _safe_rule_id(candidate_id: str) -> str:
    suffix = "".join(char if char.isalnum() else "-" for char in candidate_id.lower()).strip("-")
    suffix = suffix[:54] or hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"promotion-anthropic-routing-{suffix}"


def _omission(candidate: dict[str, Any], reason: str, *, path: str | None = None) -> dict[str, Any]:
    return {
        "schema": OMISSION_SCHEMA,
        "status": "omitted",
        "reason": reason,
        "path": path,
        "target_candidate_id": _candidate_id(candidate),
        "action_family": candidate.get("action_family"),
        "readiness_state": candidate.get("readiness_state"),
        "blocker_codes": _reason_list(candidate.get("blocker_codes"), reason),
        "no_op_reason": reason,
        "privacy": _privacy(),
    }


def _candidate_omission_reason(candidate: dict[str, Any], *, max_evidence_age_hours: float) -> str | None:
    if candidate.get("action_family") != "routing":
        return "not-routing-candidate"
    if candidate.get("target_local_rule_file") not in (None, "", "routing_rules.yaml"):
        return "unsupported-target-rule-file"
    if _string(candidate.get("provider")) not in {"", "anthropic"}:
        return "unsupported-routing-provider"
    if _string(candidate.get("source_surface")) not in {"", "anthropic_messages"}:
        return "unsupported-routing-surface"
    if _string(candidate.get("endpoint")) not in {"", "messages"}:
        return "unsupported-routing-endpoint"
    if "sonnet" not in _string(candidate.get("requested_model")).lower():
        return "unsupported-requested-model"
    target_model = _string(candidate.get("target_model"))
    if not target_model:
        return "missing-routed-model-metadata"
    if "haiku" not in target_model.lower():
        return "unsupported-target-model"
    blockers = _reason_list(candidate.get("blocker_codes"), candidate.get("reason_codes"))
    for reason in blockers:
        if reason in _REGRESSION_REASONS:
            return reason
    if "thinking-routing-guard" in blockers or "thinking-history-blocked" in blockers or _string(candidate.get("workflow_phase")) == "thinking":
        return "thinking-routing-guard"
    if any("safety-stop" in reason for reason in blockers) or _as_int(candidate.get("safety_stop_count")) > 0:
        return "safety-stop-observed"
    if "stale-evidence" in blockers:
        return "stale-evidence"
    stale = candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {}
    if stale.get("stale"):
        return "stale-evidence"
    last_age = candidate.get("last_observed_age_hours")
    if last_age is None:
        last_age = _age_hours(candidate.get("last_observed_at") or candidate.get("latest_observed_at"))
    if last_age is None:
        return "missing-freshness-evidence"
    if _as_float(last_age, max_evidence_age_hours + 1.0) > max_evidence_age_hours:
        return "stale-evidence"
    if not bool(candidate.get("promotion_ready")):
        reason = _reason(candidate.get("no_op_reason"))
        return reason if reason and reason not in _NON_BLOCKING_REASONS else "not-promotion-ready"
    if _as_int(candidate.get("applied_count")) <= 0:
        return "missing-applied-coverage"
    if _as_int(candidate.get("holdout_count")) <= 0:
        return "missing-holdout-coverage"
    if _as_int(candidate.get("error_count")) > 0:
        return "error-observed"
    if _as_float(candidate.get("observed_savings_usd")) <= 0 and _as_float(candidate.get("projected_savings_usd")) <= 0:
        return "non-positive-routing-savings"
    privacy = candidate.get("privacy")
    if isinstance(privacy, dict):
        if privacy.get("provider_bodies_included") or privacy.get("raw_provider_bodies_included"):
            return "provider-bodies-included"
        if privacy.get("raw_prompts_included") or privacy.get("raw_messages_included") or privacy.get("raw_request_bodies_included"):
            return "raw-content-included"
        if privacy.get("tool_payloads_included") or privacy.get("request_ids_included") or privacy.get("session_ids_included"):
            return "unsafe-identifiers-included"
    return None


def _evidence_summary(report: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.routing_promotion_evidence_summary.v1",
        "source_report_schema": report.get("schema"),
        "source_report_generated_at": report.get("generated_at"),
        "source_evidence_schema": candidate.get("source_evidence_schema"),
        "source_evidence_status": candidate.get("source_evidence_status"),
        "source_rank": candidate.get("source_rank"),
        "provider": candidate.get("provider") or "anthropic",
        "source_surface": candidate.get("source_surface") or "anthropic_messages",
        "endpoint": candidate.get("endpoint") or "messages",
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "workflow_phase_confidence": candidate.get("workflow_phase_confidence"),
        "stream": bool(candidate.get("stream")),
        "requested_model": candidate.get("requested_model"),
        "target_model": candidate.get("target_model"),
        "sample_count": _as_int(candidate.get("sample_count")),
        "applied_count": _as_int(candidate.get("applied_count")),
        "holdout_count": _as_int(candidate.get("holdout_count")),
        "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
        "error_count": _as_int(candidate.get("error_count")),
        "retry_count": _as_int(candidate.get("retry_count")),
        "fallback_count": _as_int(candidate.get("fallback_count")),
        "observed_savings_usd": round(_as_float(candidate.get("observed_savings_usd")), 8),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "first_observed_at": candidate.get("first_observed_at") or candidate.get("oldest_observed_at"),
        "last_observed_at": candidate.get("last_observed_at") or candidate.get("latest_observed_at"),
        "blocker_codes": _reason_list(candidate.get("blocker_codes")),
        "reason_codes": _reason_list(candidate.get("reason_codes")),
        "stale_evidence": candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {},
        "privacy": _privacy(),
    }


def _model_pattern(model: Any) -> str:
    text = str(model or "").lower()
    for family in ("sonnet", "haiku", "opus"):
        if family in text:
            return family
    return str(model or "").strip() or "sonnet"


def _draft_rule(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    rule_id = _safe_rule_id(candidate_id)
    canary_fraction = _bounded_fraction(initial_canary_fraction, 0.10)
    holdout = _bounded_fraction(holdout_fraction, 0.10)
    evidence = _evidence_summary(report, candidate)
    category = _string(candidate.get("category")) or "tool-result"
    workflow_phase = _string(candidate.get("workflow_phase")) or "tool-execution"
    confidence = _string(candidate.get("workflow_phase_confidence")) or "medium"
    rule = {
        "id": rule_id,
        "enabled": True,
        "policy_source": "local-promoted",
        "candidate_id": candidate_id,
        "conditions": {
            "model_pattern": _model_pattern(candidate.get("requested_model")),
            "category": category,
            "workflow_phase": workflow_phase,
            "workflow_phase_confidence_gte": confidence,
            "stream": bool(candidate.get("stream")),
            "has_tools": True if category.startswith("tool-") else None,
        },
        "action": {
            "route_to": _string(candidate.get("target_model")),
            "reason": f"guarded Anthropic routing promotion {candidate_id} to local Haiku rule",
        },
        "metadata": {
            "schema": "tokenclaw.routing_promotion_local_draft_metadata.v1",
            "source": "local_promotion_candidates",
            "promoted_from_canary": True,
            "promotion_source_policy_id": candidate.get("policy_id"),
            "target_candidate_id": candidate_id,
            "target_local_rule_file": "routing_rules.yaml",
            "target_local_policy_section": "routing.rules",
            "drafted_at": utc_now(),
            "evidence": evidence,
            "safety_gates": {
                "block_top_level_thinking": True,
                "block_thinking_history": True,
                "strip_model_incompatible_params": True,
                "fallback_to_requested_on_rate_limit": True,
                "require_applied_and_holdout_lifecycle": True,
                "require_zero_safety_stops": True,
                "require_zero_errors": True,
                "content_free": True,
            },
            "privacy": _privacy(),
        },
        "rollout": {
            "schema": "tokenclaw.routing_promotion_rollout.v1",
            "recommendation_mode": "routing-promotion-rule-draft",
            "canary_enabled": True,
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout,
            "canary_salt": _stable_id("routing-promotion-salt", candidate_id, rule_id),
            "canary_unit": "session",
        },
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 5,
            "min_holdout_samples": 1,
            "max_error_rate": 0.05,
            "max_retry_rate": 0.10,
            "max_fallback_rate": 0.10,
            "max_latency_regression_ratio": 1.50,
            "limit": 500,
        },
        "promotion": {
            "schema": "tokenclaw.routing_promotion_local_draft_metadata.v1",
            "source": "local_promotion_candidates",
            "target_candidate_id": candidate_id,
            "target_local_rule_file": "routing_rules.yaml",
            "target_local_policy_section": "routing.rules",
            "evidence_summary": evidence,
            "dry_run_impact_estimate": {
                "schema": "tokenclaw.routing_promotion_dry_run_impact_estimate.v1",
                "sample_count": evidence["sample_count"],
                "applied_count": evidence["applied_count"],
                "holdout_count": evidence["holdout_count"],
                "activation_fraction": canary_fraction,
                "holdout_fraction": holdout,
                "projected_canary_sample_count": round(evidence["sample_count"] * canary_fraction, 4),
                "projected_holdout_sample_count": round(evidence["sample_count"] * holdout, 4),
                "observed_savings_usd": evidence["observed_savings_usd"],
                "projected_savings_usd": evidence["projected_savings_usd"],
            },
            "rollback_metadata": {
                "rollback_action_type": "disable_rule",
                "rollback_canary_fraction": 0.0,
                "rollback_reason_codes": [
                    "safety-stop-observed",
                    "error-rate-regression",
                    "retry-rate-regression",
                    "fallback-rate-regression",
                    "thinking-routing-guard",
                    "operator-requested",
                ],
                "preserve_previous_rule_required": True,
            },
            "privacy": _privacy(),
        },
    }
    rule["conditions"] = {key: value for key, value in rule["conditions"].items() if value not in (None, "")}
    return {
        "schema": RULE_DRAFT_SCHEMA,
        "status": "drafted",
        "target_local_rule_file": "routing_rules.yaml",
        "target_local_policy_section": "routing.rules",
        "target_candidate_id": candidate_id,
        "rule_id": rule_id,
        "activation_fraction": canary_fraction,
        "holdout_fraction": holdout,
        "proposed_rule": rule,
        "evidence_summary": evidence,
        "dry_run_impact_estimate": rule["promotion"]["dry_run_impact_estimate"],
        "rollback_metadata": rule["promotion"]["rollback_metadata"],
        "privacy": _privacy(),
    }


def dry_run_routing_promotion_drafts(
    promotion_report: dict[str, Any],
    *,
    initial_canary_fraction: float = 0.10,
    holdout_fraction: float = 0.10,
    max_evidence_age_hours: float = 72.0,
) -> dict[str, Any]:
    if not isinstance(promotion_report, dict):
        return _error_result("invalid_report", "local promotion report must be a JSON object")
    raw_errors = _privacy_errors(promotion_report)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "local promotion report contains raw prompt, response, provider body, identifier, file path, cache key, or secret fields",
            errors=raw_errors,
        )
    candidates = promotion_report.get("candidates")
    if not isinstance(candidates, list):
        return _error_result("invalid_report", "local promotion report must include a candidates list")

    drafts: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    promotion_ready_count = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            omitted.append({
                "schema": OMISSION_SCHEMA,
                "status": "omitted",
                "reason": "invalid-candidate",
                "path": f"$.candidates[{index}]",
                "target_candidate_id": None,
                "privacy": _privacy(),
            })
            continue
        if candidate.get("action_family") != "routing":
            continue
        if bool(candidate.get("promotion_ready")):
            promotion_ready_count += 1
        omission_reason = _candidate_omission_reason(candidate, max_evidence_age_hours=max_evidence_age_hours)
        if omission_reason is not None:
            omitted.append(_omission(candidate, omission_reason, path=f"$.candidates[{index}]"))
            continue
        drafts.append(_draft_rule(
            promotion_report,
            candidate,
            initial_canary_fraction=initial_canary_fraction,
            holdout_fraction=holdout_fraction,
        ))

    reason_counts: dict[str, int] = {}
    for item in omitted:
        reason = str(item.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    ok = bool(drafts)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "status": "drafted" if ok else "no-op",
        "generated_at": utc_now(),
        "source_report_schema": promotion_report.get("schema"),
        "source_report_generated_at": promotion_report.get("generated_at"),
        "summary": {
            "candidate_count": len(candidates),
            "routing_candidate_count": sum(1 for item in candidates if isinstance(item, dict) and item.get("action_family") == "routing"),
            "promotion_ready_count": promotion_ready_count,
            "draft_count": len(drafts),
            "omitted_count": len(omitted),
            "omission_reason_counts": [{"value": key, "count": reason_counts[key]} for key in sorted(reason_counts)],
            "activation_fraction": _bounded_fraction(initial_canary_fraction, 0.10),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
            "target_local_rule_file": "routing_rules.yaml",
            "target_local_policy_section": "routing.rules",
            "projected_savings_usd": round(sum(_as_float(item.get("evidence_summary", {}).get("projected_savings_usd")) for item in drafts), 8),
            "observed_savings_usd": round(sum(_as_float(item.get("evidence_summary", {}).get("observed_savings_usd")) for item in drafts), 8),
        },
        "drafts": drafts,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "wrote_local_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "error": None if ok else {"type": "no_drafts", "message": "no promotion-ready Anthropic routing candidates were drafted"},
    }
