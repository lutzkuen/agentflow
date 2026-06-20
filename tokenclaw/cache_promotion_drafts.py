from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from tokenclaw.pattern_rollout import PATTERN_ROLLOUT_SCHEMA
from tokenclaw.store import utc_now


SCHEMA = "agentflow.cache_promotion_draft_dry_run.v1"
RULE_DRAFT_SCHEMA = "agentflow.cache_promotion_rule_draft.v1"
OMISSION_SCHEMA = "agentflow.cache_promotion_draft_omission.v1"

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
    "request_fingerprint",
    "request_fingerprints",
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
_BLOCKING_CACHE_REPLAY_REASONS = {
    "invalidation-evidence-missing",
    "tools-present",
    "unsafe-tool-calls-without-invalidation",
    "streaming-replay-not-supported",
}
_REPLAY_READY_PROJECTION_ALLOWED_REASONS = {
    "missing-applied-evidence",
    "missing-holdout-evidence",
    "missing-measured-cache-canary-impact",
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
        "request_fingerprints_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "individual_candidate_ids_included": False,
        "pattern_hashes_included": False,
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


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return default


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
                        "message": "cache promotion drafts accept metadata only, not raw prompt, response, provider body, identifier, file path, cache key, fingerprint, or secret fields",
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
    return _stable_id(
        "cache-promotion-candidate",
        candidate.get("source_evidence_schema"),
        candidate.get("source_rank"),
        candidate.get("source_surface"),
        candidate.get("endpoint"),
        candidate.get("category"),
        candidate.get("workflow_phase"),
        candidate.get("stream"),
        candidate.get("has_tools"),
        candidate.get("text_bucket"),
        candidate.get("token_bucket"),
    )


def _safe_rule_id(candidate_id: str) -> str:
    suffix = "".join(char if char.isalnum() else "-" for char in candidate_id.lower()).strip("-")
    suffix = suffix[:58] or hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"promotion-openai-exact-cache-{suffix}"


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
    if candidate.get("action_family") != "cache":
        return "not-cache-candidate"
    if candidate.get("target_local_rule_file") not in (None, "", "cache_rules.yaml"):
        return "unsupported-target-rule-file"
    if _string(candidate.get("source_surface")) != "openai_responses":
        return "unsupported-cache-promotion-surface"
    if _string(candidate.get("endpoint")) not in {"", "responses"}:
        return "unsupported-cache-promotion-endpoint"
    blockers = _reason_list(candidate.get("blocker_codes"))
    for reason in _BLOCKING_CACHE_REPLAY_REASONS:
        if reason in blockers:
            return reason
    if _bool(candidate.get("has_tools")):
        return "tools-present"
    if _bool(candidate.get("stream")):
        return "streaming-replay-not-supported"
    replay_ready_projection = (
        _string(candidate.get("source_evidence_schema")) == "agentflow.request_shape_cache_replayability_dry_run.v1"
        and (_bool(candidate.get("replay_ready")) or _string(candidate.get("readiness")) == "replay-ready")
        and not any(reason not in _REPLAY_READY_PROJECTION_ALLOWED_REASONS for reason in blockers)
    )
    if any("safety-stop" in reason for reason in blockers) or _as_int(candidate.get("safety_stop_count")) > 0:
        return "safety-stop-observed"
    if "stale-evidence" in blockers:
        return "stale-evidence"
    last_age = candidate.get("last_observed_age_hours")
    if last_age is None:
        last_age = _age_hours(candidate.get("last_observed_at") or candidate.get("latest_observed_at"))
    if last_age is None and not replay_ready_projection:
        return "missing-freshness-evidence"
    if last_age is not None and _as_float(last_age, max_evidence_age_hours + 1.0) > max_evidence_age_hours:
        return "stale-evidence"
    if not bool(candidate.get("promotion_ready")) and not replay_ready_projection:
        return _string(candidate.get("no_op_reason")) or "not-promotion-ready"
    if _as_int(candidate.get("applied_count")) <= 0 and not replay_ready_projection:
        return "missing-applied-evidence"
    if _as_int(candidate.get("holdout_count")) <= 0 and not replay_ready_projection:
        return "missing-holdout-evidence"
    if _as_int(candidate.get("projected_hits")) <= 0:
        return "non-positive-projected-hits"
    if _as_float(candidate.get("projected_savings_usd")) <= 0 and _as_float(candidate.get("dry_run_projected_savings_usd")) <= 0:
        return "non-positive-projected-savings"
    privacy = candidate.get("privacy")
    if isinstance(privacy, dict):
        if privacy.get("provider_bodies_included") or privacy.get("raw_provider_bodies_included"):
            return "provider-bodies-included"
        if privacy.get("raw_prompts_included") or privacy.get("raw_messages_included") or privacy.get("raw_request_bodies_included"):
            return "raw-content-included"
        if privacy.get("cache_keys_included") or privacy.get("request_fingerprints_included"):
            return "cache-identifiers-included"
    return None


def _cohort_bucket(candidate: dict[str, Any]) -> str:
    return "/".join(
        str(value or "unknown").replace("/", "_")
        for value in (
            candidate.get("source_surface") or "openai_responses",
            candidate.get("endpoint") or "responses",
            candidate.get("category") or "chat",
            candidate.get("workflow_phase") or candidate.get("category") or "chat",
            candidate.get("text_bucket") or "unknown_text_bucket",
            candidate.get("token_bucket") or "unknown_token_bucket",
        )
    )


def _evidence_summary(report: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    measurement = candidate.get("canary_hit_measurement") if isinstance(candidate.get("canary_hit_measurement"), dict) else {}
    return {
        "schema": "agentflow.cache_promotion_evidence_summary.v1",
        "source_report_schema": report.get("schema"),
        "source_report_generated_at": report.get("generated_at"),
        "source_evidence_schema": candidate.get("source_evidence_schema"),
        "source_evidence_status": candidate.get("source_evidence_status"),
        "source_rank": candidate.get("source_rank"),
        "provider": candidate.get("provider") or "openai",
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "stream": bool(candidate.get("stream")),
        "has_tools": bool(candidate.get("has_tools")),
        "text_bucket": candidate.get("text_bucket"),
        "token_bucket": candidate.get("token_bucket"),
        "replay_source_schema": candidate.get("replay_source_schema"),
        "replay_ready": candidate.get("replay_ready"),
        "readiness": candidate.get("readiness"),
        "sample_count": _as_int(candidate.get("sample_count")),
        "applied_count": _as_int(candidate.get("applied_count")),
        "holdout_count": _as_int(candidate.get("holdout_count")),
        "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
        "projected_hits": _as_int(candidate.get("projected_hits")),
        "actual_hits": _as_int(candidate.get("actual_hits")),
        "miss_count": _as_int(candidate.get("miss_count")),
        "invalidated_count": _as_int(candidate.get("invalidated_count")),
        "observed_savings_usd": round(_as_float(candidate.get("observed_savings_usd")), 8),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "dry_run_projected_savings_usd": round(_as_float(candidate.get("dry_run_projected_savings_usd")), 8),
        "hit_realization_rate": measurement.get("hit_realization_rate"),
        "savings_realization_rate": measurement.get("savings_realization_rate"),
        "first_observed_at": candidate.get("first_observed_at"),
        "last_observed_at": candidate.get("last_observed_at") or candidate.get("latest_observed_at"),
        "blocker_codes": _reason_list(candidate.get("blocker_codes")),
        "privacy": _privacy(),
    }


def _draft_rule(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float,
    holdout_fraction: float,
    ttl_seconds: int,
) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    rule_id = _safe_rule_id(candidate_id)
    canary_fraction = _bounded_fraction(initial_canary_fraction, 0.10)
    holdout = _bounded_fraction(holdout_fraction, 0.10)
    ttl = max(60, _as_int(ttl_seconds, 86_400))
    evidence = _evidence_summary(report, candidate)
    conditions = {
        "pattern_hashes": ["sha256:*"],
        "source_surface": _string(candidate.get("source_surface")) or "openai_responses",
        "endpoint": _string(candidate.get("endpoint")) or "responses",
        "category": _string(candidate.get("category")) or "chat",
        "workflow_phase": _string(candidate.get("workflow_phase")) or _string(candidate.get("category")) or "chat",
        "text_bucket": candidate.get("text_bucket"),
        "token_bucket": candidate.get("token_bucket"),
        "has_tools": False,
        "stream": False,
        "replayability_levels": ["features_only", "local-exact-response"],
    }
    conditions = {key: value for key, value in conditions.items() if value not in (None, "", [])}
    rule = {
        "id": rule_id,
        "enabled": True,
        "policy_source": "local-manual",
        "candidate_id": candidate_id,
        "description": "Local OpenAI exact-cache draft for replay-ready non-tool Responses request shapes.",
        "conditions": conditions,
        "action": {
            "type": "exact_cache_pattern",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "streaming": False,
            "scope": "session",
            "min_call_count": 2,
            "ttl_seconds": ttl,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "invalidation_assumptions": {
                "schema": "agentflow.cache_promotion_invalidation_assumptions.v1",
                "tool_call_caching_enabled": False,
                "streaming_replay_enabled": False,
                "session_scoped_keys_required": True,
                "exact_request_fingerprint_match_required": True,
                "invalidate_on_request_shape_change": True,
                "file_dependency_invalidation_required": False,
                "ttl_seconds": ttl,
            },
        },
        "rollout": {
            "schema": PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "cache-promotion-exact-cache-rule-draft",
            "canary_enabled": True,
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout,
            "canary_salt": _stable_id("cache-promotion-salt", candidate_id, rule_id),
            "canary_unit": "request_fingerprint",
            "local_feedback_fields": [
                "cache_hit",
                "status_code",
                "retry_count",
                "latency_ms",
                "cost_est_usd",
                "cost_baseline_usd",
                "cache_replay_canary",
            ],
        },
        "graduation": {
            "schema": "agentflow.cache_promotion_local_draft_metadata.v1",
            "source": "local_promotion_candidates",
            "target_candidate_id": candidate_id,
            "target_local_rule_file": "cache_rules.yaml",
            "target_local_policy_section": "cache.pattern_rules",
            "source_reason": "replay-ready-exact-non-tool-shape",
            "cohort_bucket": _cohort_bucket(candidate),
            "aggregate_only": True,
            "projected_hits": evidence["projected_hits"],
            "projected_savings_usd": evidence["projected_savings_usd"],
            "observed_savings_usd": evidence["observed_savings_usd"],
            "sample_count": evidence["sample_count"],
            "applied_count": evidence["applied_count"],
            "holdout_count": evidence["holdout_count"],
            "graduated_at": utc_now(),
            "privacy": _privacy(),
        },
        "promotion": {
            "schema": "agentflow.cache_promotion_local_draft_metadata.v1",
            "source": "local_promotion_candidates",
            "target_candidate_id": candidate_id,
            "target_local_rule_file": "cache_rules.yaml",
            "target_local_policy_section": "cache.pattern_rules",
            "evidence_summary": evidence,
            "dry_run_impact_estimate": {
                "schema": "agentflow.cache_promotion_dry_run_impact_estimate.v1",
                "sample_count": evidence["sample_count"],
                "applied_count": evidence["applied_count"],
                "holdout_count": evidence["holdout_count"],
                "activation_fraction": canary_fraction,
                "holdout_fraction": holdout,
                "projected_canary_sample_count": round(evidence["sample_count"] * canary_fraction, 4),
                "projected_holdout_sample_count": round(evidence["sample_count"] * holdout, 4),
                "projected_hits": evidence["projected_hits"],
                "actual_hits": evidence["actual_hits"],
                "observed_savings_usd": evidence["observed_savings_usd"],
                "projected_savings_usd": evidence["projected_savings_usd"],
            },
            "rollback_metadata": {
                "rollback_action_type": "disable_rule",
                "rollback_canary_fraction": 0.0,
                "rollback_reason_codes": [
                    "safety-stop-observed",
                    "cache-hit-rate-regression",
                    "error-rate-regression",
                    "retry-rate-regression",
                    "stale-evidence",
                    "operator-requested",
                ],
                "preserve_previous_rule_required": True,
            },
            "privacy": _privacy(),
        },
    }
    return {
        "schema": RULE_DRAFT_SCHEMA,
        "status": "drafted",
        "target_local_rule_file": "cache_rules.yaml",
        "target_local_policy_section": "cache.pattern_rules",
        "target_candidate_id": candidate_id,
        "rule_id": rule_id,
        "activation_fraction": canary_fraction,
        "holdout_fraction": holdout,
        "ttl_seconds": ttl,
        "proposed_rule": rule,
        "evidence_summary": evidence,
        "dry_run_impact_estimate": rule["promotion"]["dry_run_impact_estimate"],
        "rollback_metadata": rule["promotion"]["rollback_metadata"],
        "privacy": _privacy(),
    }


def dry_run_cache_promotion_drafts(
    promotion_report: dict[str, Any],
    *,
    initial_canary_fraction: float = 0.10,
    holdout_fraction: float = 0.10,
    max_evidence_age_hours: float = 168.0,
    ttl_seconds: int = 86_400,
    max_drafts: int = 10,
) -> dict[str, Any]:
    if not isinstance(promotion_report, dict):
        return _error_result("invalid_report", "local promotion report must be a JSON object")
    raw_errors = _privacy_errors(promotion_report)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "local promotion report contains raw prompt, response, provider body, identifier, file path, cache key, fingerprint, or secret fields",
            errors=raw_errors,
        )
    candidates = promotion_report.get("candidates")
    if not isinstance(candidates, list):
        return _error_result("invalid_report", "local promotion report must include a candidates list")

    drafts: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    promotion_ready_count = 0
    draft_limit = max(1, min(_as_int(max_drafts, 10), 100))
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
        if candidate.get("action_family") != "cache":
            continue
        if bool(candidate.get("promotion_ready")):
            promotion_ready_count += 1
        omission_reason = _candidate_omission_reason(candidate, max_evidence_age_hours=max_evidence_age_hours)
        if omission_reason is not None:
            omitted.append(_omission(candidate, omission_reason, path=f"$.candidates[{index}]"))
            continue
        if len(drafts) >= draft_limit:
            omitted.append(_omission(candidate, "max-drafts-reached", path=f"$.candidates[{index}]"))
            continue
        drafts.append(_draft_rule(
            promotion_report,
            candidate,
            initial_canary_fraction=initial_canary_fraction,
            holdout_fraction=holdout_fraction,
            ttl_seconds=ttl_seconds,
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
            "cache_candidate_count": sum(1 for item in candidates if isinstance(item, dict) and item.get("action_family") == "cache"),
            "promotion_ready_count": promotion_ready_count,
            "draft_count": len(drafts),
            "omitted_count": len(omitted),
            "omission_reason_counts": [{"value": key, "count": reason_counts[key]} for key in sorted(reason_counts)],
            "activation_fraction": _bounded_fraction(initial_canary_fraction, 0.10),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
            "ttl_seconds": max(60, _as_int(ttl_seconds, 86_400)),
            "max_drafts": draft_limit,
            "target_local_rule_file": "cache_rules.yaml",
            "target_local_policy_section": "cache.pattern_rules",
            "projected_hits": sum(_as_int(item.get("evidence_summary", {}).get("projected_hits")) for item in drafts),
            "projected_savings_usd": round(sum(_as_float(item.get("evidence_summary", {}).get("projected_savings_usd")) for item in drafts), 8),
        },
        "drafts": drafts,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "wrote_local_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "error": None if ok else {"type": "no_drafts", "message": "no promotion-ready OpenAI exact-cache candidates were drafted"},
    }
