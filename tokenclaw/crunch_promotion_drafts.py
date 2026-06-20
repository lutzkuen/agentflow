from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "agentflow.crunch_promotion_draft_dry_run.v1"
RULE_DRAFT_SCHEMA = "agentflow.crunch_promotion_rule_draft.v1"
OMISSION_SCHEMA = "agentflow.crunch_promotion_draft_omission.v1"

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
        "raw_thinking_text_included": False,
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
                        "message": "crunch promotion drafts accept metadata only, not raw prompt, response, provider body, identifier, file path, cache key, or secret fields",
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
        "crunch-promotion-candidate",
        candidate.get("source_evidence_schema"),
        candidate.get("source_rank"),
        candidate.get("source_surface"),
        candidate.get("endpoint"),
        candidate.get("category"),
        candidate.get("workflow_phase"),
        candidate.get("stream"),
        candidate.get("requested_model_family") or candidate.get("requested_model"),
    )


def _safe_rule_id(candidate_id: str) -> str:
    suffix = "".join(char if char.isalnum() else "-" for char in candidate_id.lower()).strip("-")
    suffix = suffix[:54] or hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"promotion-repeated-context-crunch-{suffix}"


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
    if candidate.get("action_family") != "crunch":
        return "not-crunch-candidate"
    if candidate.get("target_local_rule_file") not in (None, "", "crunch_rules.yaml"):
        return "unsupported-target-rule-file"
    if _string(candidate.get("source_surface")) != "anthropic_messages":
        return "unsupported-crunch-promotion-surface"
    blockers = _reason_list(candidate.get("blocker_codes"))
    if any("safety-stop" in reason for reason in blockers) or _as_int(candidate.get("safety_stop_count")) > 0:
        return "safety-stop-observed"
    if "stale-evidence" in blockers:
        return "stale-evidence"
    last_age = candidate.get("last_observed_age_hours")
    if last_age is None:
        last_age = _age_hours(candidate.get("last_observed_at"))
    if last_age is None:
        return "missing-freshness-evidence"
    if _as_float(last_age, max_evidence_age_hours + 1.0) > max_evidence_age_hours:
        return "stale-evidence"
    if not bool(candidate.get("promotion_ready")):
        return _string(candidate.get("no_op_reason")) or "not-promotion-ready"
    if _as_int(candidate.get("applied_count")) <= 0:
        return "missing-applied-evidence"
    if _as_int(candidate.get("holdout_count")) <= 0:
        return "missing-holdout-evidence"
    privacy = candidate.get("privacy")
    if isinstance(privacy, dict):
        if privacy.get("provider_bodies_included") or privacy.get("raw_provider_bodies_included"):
            return "provider-bodies-included"
        if privacy.get("raw_prompts_included") or privacy.get("raw_messages_included") or privacy.get("raw_request_bodies_included"):
            return "raw-content-included"
    return None


def _min_text_chars(candidate: dict[str, Any]) -> int:
    explicit = _as_int(candidate.get("min_text_chars"))
    if explicit > 0:
        return explicit
    bucket = _string(candidate.get("text_bucket"))
    if bucket.startswith("gte_128k"):
        return 128000
    if bucket.startswith("32k_128k"):
        return 32000
    if bucket.startswith("8k_32k"):
        return 8000
    category = _string(candidate.get("category"))
    return 128000 if category == "tool-result" else 32000


def _evidence_summary(report: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.crunch_promotion_evidence_summary.v1",
        "source_report_schema": report.get("schema"),
        "source_report_generated_at": report.get("generated_at"),
        "source_evidence_schema": candidate.get("source_evidence_schema"),
        "source_evidence_status": candidate.get("source_evidence_status"),
        "source_rank": candidate.get("source_rank"),
        "provider": candidate.get("provider") or "anthropic",
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "stream": bool(candidate.get("stream")),
        "requested_model_family": candidate.get("requested_model_family"),
        "routed_model_family": candidate.get("routed_model_family"),
        "sample_count": _as_int(candidate.get("sample_count")),
        "applied_count": _as_int(candidate.get("applied_count")),
        "holdout_count": _as_int(candidate.get("holdout_count")),
        "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
        "observed_savings_usd": round(_as_float(candidate.get("observed_savings_usd")), 8),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "observed_saved_tokens": _as_int(candidate.get("observed_saved_tokens")),
        "projected_saved_tokens": _as_int(candidate.get("projected_saved_tokens")),
        "avg_crunch_ratio": candidate.get("avg_crunch_ratio"),
        "first_observed_at": candidate.get("first_observed_at"),
        "last_observed_at": candidate.get("last_observed_at"),
        "canary_impact_decision": candidate.get("canary_impact_decision"),
        "budget_governor_action": candidate.get("budget_governor_action"),
        "blocker_codes": _reason_list(candidate.get("blocker_codes")),
        "privacy": _privacy(),
    }


def _draft_rule(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    rule_id = _safe_rule_id(candidate_id)
    category = _string(candidate.get("category")) or "tool-result"
    workflow_phase = _string(candidate.get("workflow_phase")) or category
    requested_family = _string(candidate.get("requested_model_family") or candidate.get("requested_model")) or "sonnet"
    source_surface = _string(candidate.get("source_surface")) or "anthropic_messages"
    min_text_chars = _min_text_chars(candidate)
    canary_fraction = _bounded_fraction(initial_canary_fraction, 0.10)
    holdout = _bounded_fraction(holdout_fraction, 0.10)
    evidence = _evidence_summary(report, candidate)
    rule = {
        "id": rule_id,
        "enabled": True,
        "policy_source": "local-manual",
        "candidate_id": candidate_id,
        "conditions": {
            "source_surface": source_surface,
            "endpoint": _string(candidate.get("endpoint")) or "messages",
            "category": category,
            "phase": workflow_phase,
            "text_bucket": candidate.get("text_bucket"),
            "token_bucket": candidate.get("token_bucket"),
            "model_pattern": requested_family,
            "has_tools": True,
            "stream": bool(candidate.get("stream")),
            "min_text_chars": min_text_chars,
        },
        "action": {
            "type": "compact_thinking_history_block",
            "min_text_chars": min_text_chars,
            "min_block_chars": 2000,
            "similarity_threshold": 0.95,
            "preserve_tool_protocol": True,
            "preserve_assistant_text_fallback": True,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "block_top_level_thinking": True,
        "canary": {
            "enabled": True,
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout,
            "canary_salt": _stable_id("crunch-promotion-salt", candidate_id, rule_id),
            "canary_unit": "thinking_block_local_fingerprint",
            "activation_fraction": canary_fraction,
        },
        "safety_stop": {
            "enabled": True,
            "min_outcome_samples": 5,
            "window": 500,
            "max_error_rate": 0.1,
            "max_retry_rate": 0.25,
            "max_negative_savings_rate": 0.25,
            "max_missing_usage_rate": 0.1,
            "max_error_rate_delta": 0.05,
        },
        "promotion": {
            "schema": "agentflow.crunch_promotion_local_draft_metadata.v1",
            "source": "local_promotion_candidates",
            "target_candidate_id": candidate_id,
            "target_local_rule_file": "crunch_rules.yaml",
            "target_local_policy_section": "anthropic_thinking_history_compaction.rules",
            "evidence_summary": evidence,
            "dry_run_impact_estimate": {
                "schema": "agentflow.crunch_promotion_dry_run_impact_estimate.v1",
                "sample_count": evidence["sample_count"],
                "applied_count": evidence["applied_count"],
                "holdout_count": evidence["holdout_count"],
                "activation_fraction": canary_fraction,
                "holdout_fraction": holdout,
                "projected_canary_sample_count": round(evidence["sample_count"] * canary_fraction, 4),
                "projected_holdout_sample_count": round(evidence["sample_count"] * holdout, 4),
                "observed_savings_usd": evidence["observed_savings_usd"],
                "projected_savings_usd": evidence["projected_savings_usd"],
                "observed_saved_tokens": evidence["observed_saved_tokens"],
                "projected_saved_tokens": evidence["projected_saved_tokens"],
            },
            "rollback_metadata": {
                "rollback_action_type": "disable_rule",
                "rollback_canary_fraction": 0.0,
                "rollback_reason_codes": [
                    "safety-stop-observed",
                    "error-rate-regression",
                    "retry-rate-regression",
                    "negative-savings-regression",
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
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "anthropic_thinking_history_compaction.rules",
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


def dry_run_crunch_promotion_drafts(
    promotion_report: dict[str, Any],
    *,
    initial_canary_fraction: float = 0.10,
    holdout_fraction: float = 0.10,
    max_evidence_age_hours: float = 168.0,
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
        if candidate.get("action_family") != "crunch":
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
            "crunch_candidate_count": sum(1 for item in candidates if isinstance(item, dict) and item.get("action_family") == "crunch"),
            "promotion_ready_count": promotion_ready_count,
            "draft_count": len(drafts),
            "omitted_count": len(omitted),
            "omission_reason_counts": [{"value": key, "count": reason_counts[key]} for key in sorted(reason_counts)],
            "activation_fraction": _bounded_fraction(initial_canary_fraction, 0.10),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
            "target_local_rule_file": "crunch_rules.yaml",
            "target_local_policy_section": "anthropic_thinking_history_compaction.rules",
        },
        "drafts": drafts,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "wrote_local_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "error": None if ok else {"type": "no_drafts", "message": "no promotion-ready repeated-context crunch candidates were drafted"},
    }
