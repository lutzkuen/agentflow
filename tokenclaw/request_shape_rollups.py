from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from tokenclaw.paths import tokenclaw_config_path
from tokenclaw.pricing import pricing_basis
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import stable_json, utc_now


SCHEMA = "tokenclaw.request_shape_rollups.v1"
ROLLUP_ROW_SCHEMA = "tokenclaw.request_shape_rollup_row.v1"
ROLLUP_SNAPSHOT_SCHEMA = "tokenclaw.request_shape_rollup_snapshot.v1"
CONTEXT_PLATEAU_ROLLUP_SCHEMA = "tokenclaw.context_plateau_crunch_rollups.v1"
CONTEXT_PLATEAU_ROLLUP_ROW_SCHEMA = "tokenclaw.context_plateau_crunch_rollup_row.v1"
REPLAYABILITY_DRY_RUN_SCHEMA = "tokenclaw.request_shape_cache_replayability_dry_run.v1"
REPLAY_INVALIDATION_EVIDENCE_SCHEMA = "tokenclaw.request_shape_cache_invalidation_evidence.v1"
REPLAY_INVALIDATION_EVIDENCE_COHORT_SCHEMA = "tokenclaw.request_shape_cache_invalidation_evidence_cohort.v1"
REPLAY_DEPENDENCY_FINGERPRINT_COVERAGE_SCHEMA = "tokenclaw.request_shape_tool_cache_dependency_fingerprint_coverage.v1"
REPLAY_DEPENDENCY_FINGERPRINT_COVERAGE_ROW_SCHEMA = (
    "tokenclaw.request_shape_tool_cache_dependency_fingerprint_coverage_row.v1"
)
REPLAY_SKIPPED_OPENAI_BLOCKERS_SCHEMA = "tokenclaw.request_shape_skipped_openai_cache_replay_blockers.v1"
REPLAY_SKIPPED_OPENAI_BLOCKER_ROW_SCHEMA = "tokenclaw.request_shape_skipped_openai_cache_replay_blocker.v1"
REPLAY_TOOL_REPLAY_EVIDENCE_SCHEMA = "tokenclaw.request_shape_tool_cache_replay_evidence.v1"
REPLAY_TOOL_REPLAY_EVIDENCE_ROW_SCHEMA = "tokenclaw.request_shape_tool_cache_replay_evidence_row.v1"
REPLAY_TOOL_REVIEW_CANDIDATES_SCHEMA = "tokenclaw.request_shape_tool_cache_review_candidates.v1"
REPLAY_TOOL_REVIEW_CANDIDATE_ROW_SCHEMA = "tokenclaw.request_shape_tool_cache_review_candidate.v1"
REPLAY_TOOL_MANAGED_LOCAL_PREVIEWS_SCHEMA = "tokenclaw.request_shape_tool_cache_managed_local_replay_previews.v1"
REPLAY_TOOL_MANAGED_LOCAL_PREVIEW_ROW_SCHEMA = "tokenclaw.request_shape_tool_cache_managed_local_replay_preview.v1"
REPLAY_TOOL_MANAGED_PREVIEW_AGREEMENT_SCHEMA = "tokenclaw.request_shape_tool_cache_managed_preview_agreement.v1"
REPLAY_BLOCKER_CLASSIFICATION_SCHEMA = "tokenclaw.request_shape_cache_replay_blocker_classification.v1"
REPLAY_BLOCKER_CLASSIFICATION_ROW_SCHEMA = "tokenclaw.request_shape_cache_replay_blocker_classification_row.v1"
REPLAY_CACHE_CANARY_ACTION_SCHEMA = "tokenclaw.request_shape_cache_replay_canary_action.v1"
REPLAY_CACHE_CANARY_STAGE_SCHEMA = "tokenclaw.request_shape_cache_replay_canary_stage.v1"
REPLAY_CACHE_CANARY_APPLY_SCHEMA = "tokenclaw.request_shape_cache_replay_canary_apply.v1"
REPLAY_CACHE_CANARY_EVIDENCE_SCHEMA = "tokenclaw.request_shape_cache_replay_evidence.v1"
REPLAY_CACHE_POLICY_DECISION_SCHEMA = "tokenclaw.request_shape_cache_replay_policy_decision.v1"
CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA = "tokenclaw.request_shape_crunch_opportunity_dry_run.v1"
CACHE_REPLAY_SUCCESSOR_DRY_RUN_SCHEMA = "tokenclaw.request_shape_cache_replay_successor_dry_run.v1"
CACHE_REPLAY_SUCCESSOR_COHORT_SCHEMA = "tokenclaw.request_shape_cache_replay_successor_cohort.v1"
CRUNCH_CANARY_ACTION_SCHEMA = "tokenclaw.request_shape_crunch_canary_action.v1"
CRUNCH_CANARY_STAGE_SCHEMA = "tokenclaw.request_shape_repeated_context_crunch_canary_stage.v1"
CRUNCH_CANARY_APPLY_SCHEMA = "tokenclaw.request_shape_crunch_canary_apply.v1"
CRUNCH_CANARY_APPLY_BATCH_SCHEMA = "tokenclaw.request_shape_crunch_canary_apply_batch.v1"
CRUNCH_CANARY_LIFECYCLE_SCHEMA = "tokenclaw.request_shape_crunch_canary_lifecycle.v1"
CRUNCH_CANARY_IMPACT_SCHEMA = "tokenclaw.request_shape_crunch_canary_impact.v1"
CRUNCH_CAPTURED_SAVINGS_SCHEMA = "tokenclaw.request_shape_crunch_captured_savings.v1"
CRUNCH_CANARY_IMPACT_ROWS_SCHEMA = "tokenclaw.request_shape_crunch_canary_impact_rows.v1"
CRUNCH_CANARY_IMPACT_ROW_SCHEMA = "tokenclaw.request_shape_crunch_canary_impact_row.v1"
CRUNCH_POLICY_DECISION_SCHEMA = "tokenclaw.request_shape_crunch_policy_decision.v1"
CRUNCH_ACTIVATION_EVIDENCE_SCHEMA = "tokenclaw.request_shape_crunch_activation_evidence.v1"
CRUNCH_REMAINING_MEASUREMENT_SCHEMA = "tokenclaw.request_shape_crunch_remaining_measurement_cohorts.v1"
CRUNCH_POLICY_DECISION_APPLY_SCHEMA = "tokenclaw.request_shape_crunch_policy_decision_apply.v1"
CRUNCH_POLICY_DECISION_LEDGER_SCHEMA = "tokenclaw.request_shape_crunch_policy_decision_ledger.v1"
CRUNCH_POLICY_DECISION_LEDGER_ENTRY_SCHEMA = "tokenclaw.request_shape_crunch_policy_decision_ledger_entry.v1"
CRUNCH_POST_MAX_ROLLOUT_DECISION_SCHEMA = "tokenclaw.request_shape_crunch_post_max_rollout_decision.v1"
FOLLOW_UP_CANDIDATES_SCHEMA = "tokenclaw.request_shape_follow_up_candidates.v1"
FOLLOW_UP_BLOCKER_COHORT_SCHEMA = "tokenclaw.request_shape_blocker_cohort.v1"
LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA = "tokenclaw.request_shape_local_activation_candidate_queue.v1"
LOCAL_ACTIVATION_CANDIDATE_QUEUE_ENTRY_SCHEMA = "tokenclaw.request_shape_local_activation_candidate_queue_entry.v1"
ROLLUP_SOURCE_DECLARATION_SCHEMA = "tokenclaw.request_shape_rollup_source_declaration.v1"
SOURCE_TRAFFIC_ACQUISITION_SCHEMA = "tokenclaw.source_traffic_acquisition_action.v1"
DASHBOARD_ROUTING_CANDIDATE_ROLLUP_SOURCE_SCHEMA = "tokenclaw.dashboard_routing_candidate_rollups.v1"
ROUTING_DOWNGRADE_DRILL_SCHEMA = "tokenclaw.request_shape_routing_downgrade_drills.v1"
ROUTING_DOWNGRADE_DRILL_ROW_SCHEMA = "tokenclaw.request_shape_routing_downgrade_drill.v1"
PHASE_AWARE_ROUTING_DRY_RUN_SCHEMA = "tokenclaw.request_shape_phase_aware_routing_dry_run.v1"
PHASE_AWARE_ROUTING_DRY_RUN_ROW_SCHEMA = "tokenclaw.request_shape_phase_aware_routing_delta.v1"
PHASE_AWARE_ROUTING_RULE_SECTION_SCHEMA = "tokenclaw.request_shape_phase_aware_routing_rule_section.v1"
DEPENDENCY_EVIDENCE_CLASSES = (
    "missing-dependency-evidence",
    "stable-dependency-evidence",
    "stale-dependency-evidence",
    "unsafe-dependency-evidence",
    "unknown-dependency-evidence",
)
DEPENDENCY_EVIDENCE_DECISION_OPTIONS = (
    "missing-dependency-evidence",
    "stable-dependency-evidence",
    "stale-risk-blocker",
    "unsafe-dependency-evidence",
    "unknown-dependency-evidence",
    "not-required",
)
REPEATED_CONTEXT_TEXT_BUCKETS = {"8k_32k_chars", "32k_128k_chars", "gte_128k_chars"}
LARGE_CONTEXT_TOKEN_BUCKETS = {"8k_32k_tokens", "gte_32k_tokens"}
# Ordinal ranking of text-size buckets used as an aggregate-only "median text size"
# signal when promoting repeated-context drills into ranked crunch cohorts.
REPEATED_CONTEXT_TEXT_BUCKET_ORDINALS = {
    "unknown": 0,
    "lt_2k_chars": 1,
    "2k_8k_chars": 2,
    "8k_32k_chars": 3,
    "32k_128k_chars": 4,
    "gte_128k_chars": 5,
}
# A repeated-context shape needs at least this many sampled calls before it is
# treated as repeat evidence rather than a one-off request.
REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES = 2
REPLAY_SUPPORTED_ENDPOINTS = {"messages", "responses", "chat_completions", "chat"}
REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE = 0.05
DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_MAX_NEW_STAGE_ACTIONS = 10
DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS = 72.0
DEFAULT_CRUNCH_CANARY_WIDEN_FRACTION = 0.10
DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION = 0.50
DEFAULT_CRUNCH_CANARY_ROLLBACK_ERROR_RATE = 0.20
DEFAULT_CRUNCH_CANARY_ROLLBACK_RETRY_RATE_DELTA = 0.10
DEFAULT_CRUNCH_CANARY_ROLLBACK_FALLBACK_RATE_DELTA = 0.05
DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION = 0.10
DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION = 0.10
DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS = 3600
DEFAULT_CACHE_REPLAY_CANARY_MAX_EVIDENCE_AGE_HOURS = 72.0
DEFAULT_CACHE_REPLAY_MIN_STAGE_ROWS = 10
DEFAULT_CACHE_REPLAY_MIN_STAGE_PROJECTED_HITS = 5
DEFAULT_CACHE_REPLAY_MIN_STAGE_SAVINGS_USD = 0.01
DEFAULT_ROLLUP_SNAPSHOT_MAX_AGE_HOURS = 72.0
DEFAULT_ROUTING_DOWNGRADE_DRILL_CANARY_FRACTION = 0.10
DEFAULT_ROUTING_DOWNGRADE_DRILL_HOLDOUT_FRACTION = 0.10
DEFAULT_ROUTING_DOWNGRADE_DRILL_MIN_SAMPLES = 5
DEFAULT_ROUTING_DOWNGRADE_DRILL_MAX_ERROR_RATE = 0.05
DEFAULT_ROUTING_DOWNGRADE_DRILL_MAX_RETRY_RATE = 0.25
DEFAULT_PHASE_AWARE_ROUTING_CANARY_FRACTION = 0.10
DEFAULT_PHASE_AWARE_ROUTING_HOLDOUT_FRACTION = 0.10
DEFAULT_PHASE_AWARE_ROUTING_MIN_SAMPLES = 5
DEFAULT_PHASE_AWARE_ROUTING_MAX_ERROR_RATE = 0.05
DEFAULT_PHASE_AWARE_ROUTING_MAX_RETRY_RATE = 0.25
DEFAULT_PHASE_AWARE_ROUTING_MAX_FALLBACK_RATE = 0.25


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    label = public_label(key, "unknown")
    counter[label] = counter.get(label, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _public_label_list(values: Any) -> list[str]:
    if not isinstance(values, list | tuple | set):
        return []
    return sorted(
        {
            public_label(item, "unknown")
            for item in values
            if public_label(item, "unknown") != "unknown"
        }
    )


def _provider_family(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "").strip().lower()
    if provider:
        return public_label(provider, "unknown")
    path = str(row.get("path") or "")
    if "responses" in path or "chat/completions" in path:
        return "openai"
    if "messages" in path:
        return "anthropic"
    return "unknown"


def _endpoint(row: dict[str, Any]) -> str:
    endpoint = row.get("endpoint")
    if endpoint:
        return public_label(endpoint, "unknown")
    path = str(row.get("path") or "")
    if "chat/completions" in path:
        return "chat_completions"
    if "responses" in path:
        return "responses"
    if "messages" in path:
        return "messages"
    return "unknown"


def _source_surface(row: dict[str, Any], provider: str, endpoint: str) -> str:
    source = row.get("source_surface")
    if source:
        return public_label(source, "unknown")
    if provider == "openai":
        return f"openai_{endpoint}"
    if provider == "anthropic":
        return "anthropic_messages"
    return "unknown"


def _app_family(provider: str, source_surface: str, requested_model: Any) -> str:
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


def _model_family(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    if "claude" in text:
        for family in ("haiku", "sonnet", "opus"):
            if family in text:
                return f"claude-{family}"
        return "claude"
    if text.startswith("gpt-5"):
        return "gpt-5"
    if text.startswith("gpt-4"):
        return "gpt-4"
    return public_label(text, fallback)


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _token_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "unknown"
    if tokens < 500:
        return "lt_500_tokens"
    if tokens < 2_000:
        return "500_2k_tokens"
    if tokens < 8_000:
        return "2k_8k_tokens"
    if tokens < 32_000:
        return "8k_32k_tokens"
    return "gte_32k_tokens"


def _cost_bucket(cost: float) -> str:
    if cost <= 0:
        return "unknown"
    if cost < 0.001:
        return "lt_0_001_usd"
    if cost < 0.01:
        return "0_001_0_01_usd"
    if cost < 0.05:
        return "0_01_0_05_usd"
    if cost < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


def _savings_bucket(savings: float) -> str:
    if savings <= 0:
        return "none"
    if savings < 0.001:
        return "lt_0_001_usd"
    if savings < 0.01:
        return "0_001_0_01_usd"
    if savings < 0.05:
        return "0_01_0_05_usd"
    if savings < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


def _input_savings_usd(tokens: int, *, provider: str, model: str, fallback_cost: float = 0.0, fallback_tokens: int = 0) -> float:
    if tokens <= 0:
        return 0.0
    basis = pricing_basis(model, provider)
    price = _as_float(basis.get("input_usd_per_million"))
    if price > 0:
        return (tokens / 1_000_000.0) * price
    if fallback_cost > 0 and fallback_tokens > 0:
        return fallback_cost * (tokens / float(fallback_tokens))
    return 0.0


def _crunch_saved_tokens(crunch: dict[str, Any]) -> int:
    for key in ("tokens_saved_est", "saved_tokens", "tokens_saved", "crunch_tokens_saved"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    saved_chars = _crunch_saved_chars(crunch)
    return saved_chars // 4


def _crunch_saved_chars(crunch: dict[str, Any]) -> int:
    for key in ("saved_chars", "chars_saved", "crunch_chars_saved"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    before = _as_int(crunch.get("before_chars") or crunch.get("original_chars"))
    after = _as_int(crunch.get("after_chars") or crunch.get("result_chars"))
    if before > after > 0:
        return before - after
    return 0


def _status_bucket(status_code: Any) -> str:
    code = _as_int(status_code, -1)
    if code < 0:
        return "unknown"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _retry_bucket(retries: int) -> str:
    if retries <= 0:
        return "0"
    if retries == 1:
        return "1"
    if retries <= 3:
        return "2_3"
    return "4_plus"


def _cache_status(row: dict[str, Any], cache: dict[str, Any]) -> str:
    status = str(cache.get("status") or "").strip().lower()
    if status:
        return public_label(status, "unknown")
    return "hit" if _as_int(row.get("cache_hit")) else "missing"


def _sanitized_file_dependency_audit(cache: dict[str, Any]) -> dict[str, Any]:
    audit = cache.get("file_dependency_audit")
    if isinstance(audit, dict):
        safe = bool(audit.get("safe_invalidation_evidence"))
        return {
            "schema": str(audit.get("schema") or "tokenclaw.cache_file_dependency_audit.v1"),
            "file_watch_enabled": bool(audit.get("file_watch_enabled")),
            "snapshot_root_policy": public_label(audit.get("snapshot_root_policy"), "unknown"),
            "root_path_included": False,
            "snapshot_count": _as_int(audit.get("snapshot_count")),
            "snapshot_count_bucket": public_label(audit.get("snapshot_count_bucket"), "unknown"),
            "candidate_path_count_bucket": public_label(audit.get("candidate_path_count_bucket"), "unknown"),
            "raw_candidate_path_count_bucket": public_label(audit.get("raw_candidate_path_count_bucket"), "unknown"),
            "distinct_candidate_path_count_bucket": public_label(
                audit.get("distinct_candidate_path_count_bucket") or audit.get("candidate_path_count_bucket"),
                "unknown",
            ),
            "max_paths": _as_int(audit.get("max_paths")),
            "cap_exceeded": bool(audit.get("cap_exceeded")),
            "cap_trimmed": bool(audit.get("cap_trimmed")),
            "dependency_capture_reason": public_label(audit.get("dependency_capture_reason"), "unknown"),
            "present_path_count": _as_int(audit.get("present_path_count")),
            "missing_path_count": _as_int(audit.get("missing_path_count")),
            "changed_path_count": _as_int(audit.get("changed_path_count")),
            "deleted_path_count": _as_int(audit.get("deleted_path_count")),
            "created_path_count": _as_int(audit.get("created_path_count")),
            "invalidation_reason": public_label(audit.get("invalidation_reason"), "none"),
            "safe_invalidation_evidence": safe,
            "file_dependency_evidence_available": bool(audit.get("file_dependency_evidence_available") or safe),
            "paths_included": False,
            "path_hashes_included": False,
            "raw_stat_values_included": False,
        }
    evidence = bool(cache.get("file_dependency_evidence_available") or cache.get("safe_invalidation_evidence"))
    return {
        "schema": "tokenclaw.cache_file_dependency_audit.v1",
        "file_watch_enabled": bool(cache.get("file_watch_enabled")),
        "snapshot_root_policy": "unknown",
        "root_path_included": False,
        "snapshot_count": _as_int(cache.get("file_dependency_count")),
        "snapshot_count_bucket": public_label(cache.get("file_dependency_count_bucket"), "unknown"),
        "candidate_path_count_bucket": "unknown",
        "raw_candidate_path_count_bucket": "unknown",
        "distinct_candidate_path_count_bucket": "unknown",
        "max_paths": 0,
        "cap_exceeded": False,
        "cap_trimmed": False,
        "dependency_capture_reason": "complete" if evidence else "file-dependency-missing",
        "present_path_count": _as_int(cache.get("file_dependency_count")),
        "missing_path_count": 0,
        "changed_path_count": 0,
        "deleted_path_count": 0,
        "created_path_count": 0,
        "invalidation_reason": public_label(cache.get("invalidation_reason"), "none"),
        "safe_invalidation_evidence": bool(cache.get("safe_invalidation_evidence")),
        "file_dependency_evidence_available": evidence,
        "paths_included": False,
        "path_hashes_included": False,
        "raw_stat_values_included": False,
    }


def _file_dependency_status(audit: dict[str, Any]) -> str:
    reason = public_label(audit.get("invalidation_reason"), "none")
    if audit.get("cap_exceeded"):
        return "invalidated"
    if reason in {"dependency-changed", "dependency-deleted", "dependency-created", "dependency-cap-exceeded"}:
        return "invalidated"
    if audit.get("safe_invalidation_evidence"):
        return "stable"
    if reason in {"file-dependency-missing", "dependency-missing", "file-watch-disabled"}:
        return "missing"
    if not audit.get("file_dependency_evidence_available"):
        return "missing"
    if reason in {"none", "unknown", "dependency-evidence-unknown"}:
        return "unknown"
    if audit.get("file_dependency_evidence_available") and not audit.get("safe_invalidation_evidence"):
        return "unsafe"
    return "unknown"


def _file_dependency_fingerprint_available(cache: dict[str, Any]) -> bool:
    fingerprint = cache.get("file_dependency_fingerprint")
    if isinstance(fingerprint, dict):
        return bool(fingerprint.get("fingerprint_available") or fingerprint.get("fingerprint_sha256"))
    return bool(cache.get("file_dependency_fingerprint_available") or cache.get("file_dependency_fingerprint_sha256"))


def _local_dependency_fingerprint_metadata(available: bool, audit: dict[str, Any] | None = None) -> dict[str, Any]:
    audit = audit if isinstance(audit, dict) else {}
    return {
        "schema": "tokenclaw.local_dependency_fingerprint_metadata.v1",
        "fingerprint_available": bool(available),
        "fingerprint_value_included": False,
        "fingerprint_sha256_included": False,
        "path_hashes_included": False,
        "paths_included": False,
        "snapshot_count": _as_int(audit.get("snapshot_count")),
        "snapshot_count_bucket": public_label(audit.get("snapshot_count_bucket"), "unknown"),
        "candidate_path_count_bucket": public_label(audit.get("candidate_path_count_bucket"), "unknown"),
        "stable_dependency_snapshot": bool(available and audit.get("safe_invalidation_evidence")),
        "metadata_only": True,
        "aggregate_only": True,
    }


def _merge_file_dependency_audit(left: dict[str, Any] | None, right: dict[str, Any]) -> dict[str, Any]:
    if not left:
        return {
            **right,
            "paths_included": False,
            "path_hashes_included": False,
            "raw_stat_values_included": False,
            "root_path_included": False,
        }
    merged = {**left}
    for key in (
        "snapshot_count",
        "present_path_count",
        "missing_path_count",
        "changed_path_count",
        "deleted_path_count",
        "created_path_count",
    ):
        merged[key] = _as_int(merged.get(key)) + _as_int(right.get(key))
    merged["cap_exceeded"] = bool(merged.get("cap_exceeded") or right.get("cap_exceeded"))
    merged["cap_trimmed"] = bool(merged.get("cap_trimmed") or right.get("cap_trimmed"))
    merged["safe_invalidation_evidence"] = bool(
        merged.get("safe_invalidation_evidence") or right.get("safe_invalidation_evidence")
    )
    merged["file_dependency_evidence_available"] = bool(
        merged.get("file_dependency_evidence_available") or right.get("file_dependency_evidence_available")
    )
    if public_label(merged.get("invalidation_reason"), "none") == "none":
        merged["invalidation_reason"] = right.get("invalidation_reason")
    merged["paths_included"] = False
    merged["path_hashes_included"] = False
    merged["raw_stat_values_included"] = False
    merged["root_path_included"] = False
    return merged


def _routing_status(row: dict[str, Any], routing: dict[str, Any]) -> str:
    requested = str(row.get("requested_model") or routing.get("requested_model") or "")
    routed = str(row.get("routed_model") or routing.get("routed_model") or requested)
    if requested and routed and requested != routed:
        return "routed"
    if routing.get("enabled") is False:
        return "disabled"
    return "passthrough"


def _has_tools(row: dict[str, Any], routing: dict[str, Any], cache: dict[str, Any]) -> bool:
    if routing.get("has_tools") is not None:
        return bool(routing.get("has_tools"))
    tool_features = routing.get("tool_features") if isinstance(routing.get("tool_features"), dict) else {}
    if tool_features.get("has_tools") is not None:
        return bool(tool_features.get("has_tools"))
    if cache.get("has_tools") is not None:
        return bool(cache.get("has_tools"))
    category = str(row.get("category") or routing.get("category") or "").lower()
    reason = str(cache.get("reason") or "").lower()
    return category.startswith("tool") or "tool" in reason


def _workflow_phase(row: dict[str, Any], routing: dict[str, Any]) -> str:
    for key in ("workflow_phase", "phase", "category"):
        value = routing.get(key)
        if value:
            return public_label(value, "unknown")
    return public_label(row.get("category"), "unknown")


def _endpoint_label_from_row(row: dict[str, Any]) -> str:
    endpoint = str(row.get("endpoint") or row.get("path") or "").strip().lower()
    if endpoint.startswith("/v1/"):
        endpoint = endpoint.removeprefix("/v1/")
    return endpoint.strip("/").replace("/", "_") or "unknown"


def _is_anthropic_messages_row(row: dict[str, Any]) -> bool:
    provider = str(row.get("provider_family") or row.get("provider") or "").strip().lower()
    source_surface = str(row.get("source_surface") or "").strip().lower()
    return _endpoint_label_from_row(row) == "messages" and (
        provider == "anthropic" or source_surface == "anthropic_messages"
    )


def _streaming_exact_replay_supported(
    *,
    row: dict[str, Any],
    cache: dict[str, Any],
    routing: dict[str, Any],
    has_tools: bool,
) -> bool:
    if has_tools:
        return False
    if not _is_anthropic_messages_row(row):
        return False
    reason = str(cache.get("reason") or "").lower()
    routing_reason = str(routing.get("reason") or "").lower()
    if "thinking" in reason or _routing_thinking_blockers(routing, routing_reason):
        return False
    unsupported_reasons = {
        "streaming-cache-disabled",
        "streaming-pattern-rule-required",
        "streaming-thinking-disabled",
        "streaming-tools-disabled",
        "streaming-not-allowed",
    }
    return reason not in unsupported_reasons


def _routing_thinking_blockers(routing: dict[str, Any], routing_reason: str) -> list[str]:
    blockers: set[str] = set()
    phase_canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
    safety_stop = phase_canary.get("safety_stop") if isinstance(phase_canary.get("safety_stop"), dict) else {}
    for code in _public_label_list(safety_stop.get("reason_codes")):
        if code in {"top-level-thinking-blocked", "thinking-history-blocked", "thinking-routing-guard"}:
            blockers.add(code)

    gate = routing.get("thinking_gate") if isinstance(routing.get("thinking_gate"), dict) else {}
    if bool(gate.get("top_level_thinking")):
        blockers.add("top-level-thinking-blocked")
    if bool(gate.get("assistant_thinking_history")):
        blockers.add("thinking-history-blocked")

    reason = str(routing_reason or "").strip().lower().replace(" ", "-")
    if "thinking" not in reason:
        return sorted(blockers)
    evidence_only = (
        "thinking-tool-safety-evidence" in reason
        or "thinking/tool-safety-evidence" in reason
        or "thinking-safety-evidence" in reason
        or "thinking-lifecycle-evidence" in reason
    )
    if blockers or evidence_only:
        return sorted(blockers)
    if "top-level-thinking" in reason or "current-thinking" in reason:
        blockers.add("top-level-thinking-blocked")
    elif "thinking-request" in reason:
        blockers.add("thinking-routing-guard")
    elif "assistant-thinking-history" in reason or "thinking-history" in reason:
        blockers.add("thinking-history-blocked")
    elif "thinking-safety-gate" in reason or "thinking-blocked" in reason or "adaptive-thinking" in reason:
        blockers.add("thinking-routing-guard")
    return sorted(blockers)


def _blocker_codes(
    *,
    row: dict[str, Any],
    cache: dict[str, Any],
    routing: dict[str, Any],
    cache_status: str,
    routing_status: str,
    stream: bool,
    has_tools: bool,
    file_dependency_status: str = "missing",
) -> list[str]:
    blockers: set[str] = set()
    reason = str(cache.get("reason") or "").lower()
    routing_reason = str(routing.get("reason") or "").lower()
    status_bucket = _status_bucket(row.get("status_code"))
    if stream and not _streaming_exact_replay_supported(
        row=row,
        cache=cache,
        routing=routing,
        has_tools=has_tools,
    ):
        blockers.add("unsupported-streaming-shape")
    if has_tools and ("tools-disabled" in reason or "tool" in reason and "disabled" in reason):
        blockers.add("tool-call-cache-disabled")
    if has_tools:
        if file_dependency_status == "stable":
            blockers.add("safe-invalidation-evidence-present")
        elif file_dependency_status == "invalidated":
            blockers.add("stale-dependency-evidence")
        elif file_dependency_status == "unsafe":
            blockers.add("unsafe-dependency-evidence")
            blockers.add("unsafe-tool-calls-without-invalidation")
        elif file_dependency_status == "missing":
            blockers.add("invalidation-evidence-missing")
        else:
            blockers.add("dependency-evidence-unknown")
    if "semantic" in reason and "disabled" in reason:
        blockers.add("semantic-cache-disabled")
    if cache_status in {"miss", "missing"} or "exact-miss" in reason:
        blockers.add("exact-cache-miss")
    if cache_status == "skipped" and not blockers:
        blockers.add("cache-skipped")
    if cache_status == "hit":
        blockers.add("already-cache-hit")
    blockers.update(_routing_thinking_blockers(routing, routing_reason))
    if "rate" in routing_reason or status_bucket in {"4xx", "5xx"} and _as_int(row.get("retry_count")) > 0:
        blockers.add("rate-or-error-pressure")
    if routing_status == "passthrough" and not blockers:
        blockers.add("routing-rule-required")
    return sorted(public_label(code, "unknown") for code in blockers if code)


def _candidate_families(
    *,
    cache_status: str,
    routing_status: str,
    blockers: list[str],
    observed_savings: float,
    cost: float,
) -> list[str]:
    families: set[str] = set()
    if cache_status != "hit":
        families.add("cache_replay")
    if any(
        code.startswith("exact-cache")
        or code.startswith("cache-")
        or code in {"unsupported-streaming-shape", "tool-call-cache-disabled", "semantic-cache-disabled"}
        for code in blockers
    ):
        families.add("cache_blocker")
    if routing_status == "passthrough" and cost > 0:
        families.add("routing_candidate")
    if routing_status == "routed" or observed_savings > 0:
        families.add("routing_evidence")
    return sorted(families or {"observability"})


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = stable_json(basis)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    provider = str(basis.get("provider_family") or "unknown").replace("_", "-")
    endpoint = str(basis.get("endpoint") or "unknown").replace("_", "-")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"request-shape:{provider}:{endpoint}:{category}:{digest}"


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _codex_metadata_window_rows(store_obj: Any, *, limit: int) -> list[dict[str, Any]]:
    try:
        raw_rows = store_obj.conn.execute(
            """
            select id, created_at, method, direction, message_chars, params_chars,
                   input_items, input_text_chars, result_chars, error_code,
                   error_message, latency_ms, routing_json, crunch_json, cache_json,
                   event_window_json, metadata_json
            from codex_app_events
            where direction = 'client_to_server'
              and method = 'turn/start'
            order by created_at desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        source = dict(raw)
        routing = _json_obj(source.get("routing_json"))
        cache = _json_obj(source.get("cache_json"))
        crunch = _json_obj(source.get("crunch_json"))
        event_window = _json_obj(source.get("event_window_json"))
        metadata = _json_obj(source.get("metadata_json"))

        input_text_chars = (
            _as_int(source.get("input_text_chars"))
            or _as_int(event_window.get("input_text_chars"))
            or _as_int(source.get("message_chars"))
        )
        input_tokens = _as_int(event_window.get("input_tokens_est")) or max(0, input_text_chars // 4)
        output_tokens = _as_int(event_window.get("output_tokens_est")) or max(0, _as_int(source.get("result_chars")) // 4)
        error_code = _as_int(source.get("error_code"))
        status_code = 500 if error_code else 200
        method = _first_text(source.get("method"), event_window.get("method"), "turn/start").replace("/", "_")
        provider = public_label(_first_text(routing.get("provider"), metadata.get("provider"), "openai"), "openai")
        source_surface = public_label(
            _first_text(routing.get("source_surface"), event_window.get("source_surface"), metadata.get("source_surface"), "codex_turn"),
            "codex_turn",
        )
        endpoint = public_label(_first_text(routing.get("endpoint"), event_window.get("endpoint"), method), "turn_start")
        requested_model = _first_text(
            routing.get("requested_model"),
            routing.get("model"),
            event_window.get("requested_model"),
            event_window.get("model"),
            metadata.get("requested_model"),
        )
        routed_model = _first_text(routing.get("routed_model"), event_window.get("routed_model"), metadata.get("routed_model"), requested_model)
        category = public_label(_first_text(routing.get("category"), event_window.get("category"), metadata.get("category"), "codex-turn"), "codex-turn")
        workflow_phase = public_label(
            _first_text(routing.get("workflow_phase"), event_window.get("workflow_phase"), metadata.get("workflow_phase"), category),
            "unknown",
        )
        method_counts = event_window.get("method_counts") if isinstance(event_window.get("method_counts"), dict) else {}
        has_tools = bool(
            routing.get("has_tools")
            or event_window.get("has_tools")
            or _as_int(event_window.get("tool_use_count"))
            or _as_int(event_window.get("tool_result_count"))
            or any("tool" in str(key).lower() or "command" in str(key).lower() for key in method_counts)
        )

        routing_meta = {
            **routing,
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "requested_model": requested_model,
            "routed_model": routed_model or requested_model,
            "text_chars": input_text_chars,
            "has_tools": has_tools,
            "category": category,
            "workflow_phase": workflow_phase,
            "reason": routing.get("reason") or "codex-app-metadata-window-backfill",
            "metadata_window_backfill": True,
        }
        cache_meta = {
            "status": "skipped",
            "reason": "codex-app-cache-disabled",
            "policy_source": "local-default",
            **cache,
        }
        rows.append(
            {
                "id": f"codex-metadata-window:{hashlib.sha256(str(source.get('id') or source.get('created_at')).encode('utf-8')).hexdigest()[:16]}",
                "created_at": source.get("created_at"),
                "path": method,
                "provider": provider,
                "source_surface": source_surface,
                "endpoint": endpoint,
                "requested_model": requested_model,
                "routed_model": routed_model or requested_model,
                "requested_model_family": _model_family(requested_model) if requested_model else "unknown",
                "routed_model_family": _model_family(routed_model or requested_model) if (routed_model or requested_model) else "unknown",
                "stream": 0,
                "cache_hit": 1 if cache_meta.get("status") == "hit" else 0,
                "status_code": status_code,
                "latency_ms": source.get("latency_ms"),
                "input_tokens_est": input_tokens,
                "output_tokens_est": output_tokens,
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
                "cost_est_usd": _as_float(metadata.get("cost_est_usd") or event_window.get("cost_est_usd")),
                "cost_baseline_usd": _as_float(metadata.get("cost_baseline_usd") or event_window.get("cost_baseline_usd")),
                "retry_count": _as_int(metadata.get("retry_count") or event_window.get("retry_count")),
                "category": category,
                "crunch_json": stable_json(crunch),
                "routing_json": stable_json(routing_meta),
                "cache_json": stable_json(cache_meta),
                "_metadata_window_backfill": True,
            }
        )
    return rows


def _crunch_canary_cohort_id(row: dict[str, Any]) -> str:
    basis = {
        "provider_family": row.get("provider_family"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "category": row.get("category"),
        "workflow_phase": row.get("workflow_phase"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": row.get("text_bucket"),
        "token_bucket": row.get("token_bucket"),
        "cache_status": row.get("cache_status"),
        "routing_status": row.get("routing_status"),
    }
    digest = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:16]
    provider = str(row.get("provider_family") or "unknown").replace("_", "-")
    endpoint = str(row.get("endpoint") or "unknown").replace("_", "-")
    category = str(row.get("category") or "unknown").replace("_", "-")
    return f"request-shape-crunch:{provider}:{endpoint}:{category}:{digest}"


def _crunch_canary_policy_id(cohort_id: str) -> str:
    digest = hashlib.sha256(cohort_id.encode("utf-8")).hexdigest()[:12]
    return f"local-repeated-context-crunch-canary-{digest}"


def _crunch_rules_candidate_paths(rules_path: str | Path | None = None) -> list[Path]:
    if rules_path is not None:
        return [Path(rules_path)]
    import os

    candidates: list[Path] = []
    env_path = os.getenv("TOKENCLAW_CRUNCH_RULES")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / "crunch_rules.yaml")
    candidates.append(Path.home() / ".tokenclaw" / "crunch_rules.yaml")
    candidates.append(Path(__file__).parent / "crunch_rules.yaml")
    return candidates


def _request_shape_crunch_canary_rule_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    conditions: dict[str, Any] = {}
    string_keys = {
        "provider_family",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "text_bucket",
        "token_bucket",
        "cache_status",
        "routing_status",
    }
    bool_keys = {"stream", "has_tools"}
    for key in string_keys:
        if value.get(key) is not None:
            conditions[key] = public_label(value.get(key), "unknown")
    for key in bool_keys:
        if value.get(key) is not None:
            conditions[key] = bool(value.get(key))
    return conditions


def _load_request_shape_crunch_canary_rules(rules_path: str | Path | None = None) -> list[dict[str, Any]]:
    for path in _crunch_rules_candidate_paths(rules_path):
        if not path.exists():
            continue
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(loaded, dict):
            return []
        section = loaded.get("request_shape_repeated_context_canaries")
        if not isinstance(section, dict) or not bool(section.get("enabled", True)):
            return []
        raw_rules = section.get("rules")
        if not isinstance(raw_rules, list):
            return []
        rules: list[dict[str, Any]] = []
        for index, item in enumerate(raw_rules):
            if not isinstance(item, dict) or not bool(item.get("enabled", True)):
                continue
            rollout = item.get("rollout") if isinstance(item.get("rollout"), dict) else {}
            safety_gates = item.get("safety_gates") if isinstance(item.get("safety_gates"), dict) else {}
            policy_decision = item.get("policy_decision") if isinstance(item.get("policy_decision"), dict) else {}
            decision_value = public_label(policy_decision.get("decision") or policy_decision.get("graduation_decision"), "unknown")
            canary_fraction = _bounded_fraction(
                rollout.get("canary_fraction", rollout.get("fraction", 0.0)),
                0.0,
            )
            full_rollout_fraction = _bounded_fraction(rollout.get("full_rollout_fraction"), 0.0)
            full_rollout_active = (
                bool(rollout.get("full_rollout_enabled"))
                or full_rollout_fraction > 0.0
                or decision_value == "promote-full"
            )
            if full_rollout_active:
                full_rollout_fraction = full_rollout_fraction or 1.0
                canary_fraction = max(canary_fraction, full_rollout_fraction)
            max_rollout_fraction = _bounded_fraction(
                safety_gates.get("max_rollout_fraction", DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION),
                0.0,
            )
            active_at_max_rollout = (
                decision_value == "widen"
                and not full_rollout_active
                and canary_fraction > 0
                and max_rollout_fraction > 0
                and canary_fraction >= max_rollout_fraction
            )
            rules.append(
                {
                    "id": public_label(item.get("id") or item.get("policy_id") or f"local-repeated-context-crunch-canary-{index + 1}", "unknown"),
                    "policy_id": public_label(item.get("policy_id") or item.get("id"), "unknown"),
                    "cohort_id": public_label(item.get("cohort_id"), "unknown"),
                    "policy_source": public_label(item.get("policy_source"), "unknown"),
                    "conditions": _request_shape_crunch_canary_rule_conditions(item.get("conditions")),
                    "policy_decision": {
                        "decision": decision_value,
                        "graduation_decision": public_label(policy_decision.get("graduation_decision") or decision_value, "unknown"),
                        "decision_id": public_label(policy_decision.get("decision_id"), "unknown"),
                        "source_evidence_schema": public_label(policy_decision.get("source_evidence_schema"), "unknown"),
                    },
                    "full_rollout_active": full_rollout_active,
                    "active_at_max_rollout": active_at_max_rollout,
                    "full_rollout_fraction": round(full_rollout_fraction, 6),
                    "max_rollout_fraction": round(max_rollout_fraction, 6),
                    "rollout": {
                        "canary_enabled": bool(rollout.get("canary_enabled", True)),
                        "full_rollout_enabled": full_rollout_active,
                        "full_rollout_fraction": full_rollout_fraction,
                        "canary_fraction": canary_fraction,
                        "holdout_fraction": 0.0
                        if full_rollout_active
                        else _bounded_fraction(rollout.get("holdout_fraction", 0.0), 0.0),
                    },
                }
            )
        return rules
    return []


def _request_shape_crunch_cohort_matches_rule(cohort: dict[str, Any], rule: dict[str, Any]) -> bool:
    rule_cohort_id = public_label(rule.get("cohort_id"), "unknown")
    cohort_id = public_label(cohort.get("cohort_id"), "unknown")
    if rule_cohort_id != "unknown" and cohort_id != "unknown" and rule_cohort_id == cohort_id:
        return True
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    return _request_shape_crunch_cohort_matches_conditions(cohort, conditions)


def _request_shape_crunch_cohort_matches_conditions(cohort: dict[str, Any], conditions: dict[str, Any]) -> bool:
    if not conditions:
        return False
    for key, expected in conditions.items():
        if expected is None:
            continue
        actual = cohort.get(key)
        if key in {"stream", "has_tools"}:
            if bool(actual) != bool(expected):
                return False
        elif public_label(actual, "unknown") != public_label(expected, "unknown"):
            return False
    return True


def _request_shape_crunch_cohort_duplicate_suppression(
    cohort: dict[str, Any],
    rules: list[dict[str, Any]],
) -> dict[str, Any]:
    readiness = public_label(cohort.get("readiness"), "unknown")
    lifecycle_suppressed = readiness in {"canary-staged", "canary-applied", "canary-holdout", "canary-safety-stopped"}
    for rule in rules:
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        full_rollout_active = bool(rule.get("full_rollout_active") or rollout.get("full_rollout_enabled"))
        if not bool(rollout.get("canary_enabled", True)) and not full_rollout_active:
            continue
        if _request_shape_crunch_cohort_matches_rule(cohort, rule):
            active_at_max_rollout = bool(rule.get("active_at_max_rollout"))
            full_rollout_active = bool(rule.get("full_rollout_active") or rollout.get("full_rollout_enabled"))
            reason = (
                "repeated-context-crunch-full-rollout-active"
                if full_rollout_active
                else "repeated-context-crunch-active-at-max-rollout"
                if active_at_max_rollout
                else "matching-repeated-context-crunch-canary-already-staged-in-local-policy"
            )
            if full_rollout_active:
                reason = "repeated-context-crunch-full-rollout-active"
            policy_decision = rule.get("policy_decision") if isinstance(rule.get("policy_decision"), dict) else {}
            return {
                "schema": "tokenclaw.request_shape_crunch_stage_duplicate_suppression.v1",
                "suppressed": True,
                "suppresses_new_stage_action": True,
                "reason": reason,
                "full_rollout_active": full_rollout_active,
                "active_at_max_rollout": active_at_max_rollout,
                "matching_local_policy": "crunch_rules",
                "matching_policy_id": public_label(rule.get("policy_id") or rule.get("id"), "unknown"),
                "matching_cohort_id": public_label(rule.get("cohort_id"), "unknown"),
                "matching_rollout_fraction": round(_as_float(rollout.get("canary_fraction")), 6),
                "matching_full_rollout_fraction": round(_as_float(rollout.get("full_rollout_fraction")), 6),
                "matching_holdout_fraction": round(_as_float(rollout.get("holdout_fraction")), 6),
                "matching_max_rollout_fraction": round(_as_float(rule.get("max_rollout_fraction")), 6),
                "matching_policy_decision": {
                    "decision": public_label(policy_decision.get("decision"), "unknown"),
                    "graduation_decision": public_label(policy_decision.get("graduation_decision"), "unknown"),
                    "decision_id": public_label(policy_decision.get("decision_id"), "unknown"),
                    "source_evidence_schema": public_label(policy_decision.get("source_evidence_schema"), "unknown"),
                    "metadata_only": True,
                    "aggregate_only": True,
                }
                if policy_decision
                else None,
                "metadata_only": True,
                "aggregate_only": True,
                "privacy": _crunch_opportunity_privacy(),
            }
    reason = (
        f"matching-repeated-context-crunch-canary-{readiness}"
        if lifecycle_suppressed
        else None
    )
    return {
        "schema": "tokenclaw.request_shape_crunch_stage_duplicate_suppression.v1",
        "suppressed": lifecycle_suppressed,
        "suppresses_new_stage_action": lifecycle_suppressed,
        "reason": reason,
        "matching_local_policy": "crunch_rules" if lifecycle_suppressed else None,
        "matching_policy_id": public_label((cohort.get("crunch_canary_lifecycle") or {}).get("policy_id"), "unknown")
        if isinstance(cohort.get("crunch_canary_lifecycle"), dict)
        else "unknown",
        "matching_cohort_id": public_label((cohort.get("crunch_canary_lifecycle") or {}).get("cohort_id"), "unknown")
        if isinstance(cohort.get("crunch_canary_lifecycle"), dict)
        else public_label(cohort.get("cohort_id"), "unknown"),
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_canary_lifecycle_from_meta(crunch: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "request_shape_repeated_context_canary",
        "repeated_context_crunch_canary",
        "request_shape_crunch_canary",
        "crunch_canary",
    ):
        meta = crunch.get(key)
        if isinstance(meta, dict):
            status = public_label(meta.get("status") or meta.get("lifecycle") or meta.get("cohort"), "unknown")
            cohort = public_label(meta.get("cohort") or status, "unknown")
            if status in {"canary-applied", "canary_applied"}:
                status = "applied"
            elif status in {"canary-holdout", "canary_holdout"}:
                status = "holdout"
            if cohort == "canary-applied":
                cohort = "canary_applied"
            elif cohort == "canary-holdout":
                cohort = "canary_holdout"
            return {
                "status": status,
                "cohort": cohort,
                "policy_id": public_label(meta.get("policy_id"), "unknown"),
                "cohort_id": public_label(meta.get("cohort_id"), "unknown"),
                "policy_source": public_label(meta.get("policy_source"), "local-manual"),
                "source_evidence_schema": public_label(meta.get("source_evidence_schema"), "unknown"),
                "source_evidence_schemas": _public_label_list(meta.get("source_evidence_schemas")),
                "rule_group": public_label(meta.get("rule_group") or meta.get("candidate_rule") or "repeated-context-conservative", "repeated-context-conservative"),
                "staged_at": meta.get("staged_at") if isinstance(meta.get("staged_at"), str) else None,
                "projected_saved_chars": _as_int(meta.get("projected_saved_chars")),
                "projected_saved_tokens": _as_int(meta.get("projected_saved_tokens")),
                "projected_saved_usd": round(_as_float(meta.get("projected_saved_usd")), 8),
                "rollback_metadata_present": bool(meta.get("rollback_metadata_present")),
                "safety_stop": bool(meta.get("safety_stop")) or status == "safety-stopped",
            }
    return None


def _crunch_canary_cohort_name(lifecycle: dict[str, Any]) -> str:
    status = str(lifecycle.get("status") or "").strip().lower().replace("_", "-")
    cohort = str(lifecycle.get("cohort") or "").strip().lower().replace("-", "_")
    if status == "applied" or cohort == "canary_applied":
        return "canary_applied"
    if status == "holdout" or cohort == "canary_holdout":
        return "canary_holdout"
    if bool(lifecycle.get("safety_stop")) or status in {"safety-stopped", "safety-stop"} or cohort in {"safety_stopped", "safety_stop"}:
        return "safety_stopped"
    if status in {"fallback", "fallback-applied"} or cohort in {"fallback", "fallback_applied"}:
        return "fallback"
    if status in {"rollback", "rollback-required"} or cohort in {"rollback", "rollback_required"}:
        return "rollback"
    if status in {"skipped", "disabled", "ineligible"} or cohort in {"skipped", "bypassed_or_disabled"}:
        return "skipped"
    return "unknown"


def _crunch_before_chars(crunch: dict[str, Any], *, text_chars: int, input_tokens: int) -> int:
    for key in ("before_chars", "original_chars", "input_chars", "text_chars_before"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    saved = _crunch_saved_chars(crunch)
    for key in ("after_chars", "result_chars", "output_chars", "text_chars_after"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value + saved
    return max(text_chars, input_tokens * 4 if input_tokens > 0 else 0, saved)


def _crunch_after_chars(crunch: dict[str, Any], before_chars: int) -> int:
    for key in ("after_chars", "result_chars", "output_chars", "text_chars_after"):
        value = _as_int(crunch.get(key))
        if value > 0:
            return value
    saved = _crunch_saved_chars(crunch)
    return max(0, before_chars - saved)


def _crunch_savings_usd(
    crunch: dict[str, Any],
    *,
    tokens_saved: int,
    provider: str,
    model: str,
    fallback_cost: float,
    fallback_tokens: int,
) -> float:
    for key in ("savings_usd", "saved_usd", "crunch_savings_usd", "estimated_savings_usd"):
        value = _as_float(crunch.get(key))
        if value > 0:
            return value
    return _input_savings_usd(
        tokens_saved,
        provider=provider,
        model=model,
        fallback_cost=fallback_cost,
        fallback_tokens=fallback_tokens,
    )


def _empty_crunch_impact_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "before_chars": 0,
        "after_chars": 0,
        "saved_chars": 0,
        "saved_tokens": 0,
        "saved_usd": 0.0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "safety_stop_count": 0,
        "rollback_count": 0,
    }


def _empty_crunch_impact_candidate(policy_id: str, cohort_id: str, row: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "cohort_metadata": {
            "provider_family": row.get("provider_family"),
            "source_surface": row.get("source_surface"),
            "endpoint": row.get("endpoint"),
            "category": row.get("category"),
            "workflow_phase": row.get("workflow_phase"),
            "stream": bool(row.get("stream")),
            "has_tools": bool(row.get("has_tools")),
            "text_bucket": row.get("text_bucket"),
            "token_bucket": row.get("token_bucket"),
            "cache_status": row.get("cache_status"),
            "routing_status": row.get("routing_status"),
        },
        "policy_source": public_label(lifecycle.get("policy_source") or "local-manual", "local-manual"),
        "source_evidence_schema": public_label(lifecycle.get("source_evidence_schema"), "unknown"),
        "source_evidence_schemas": _public_label_list(lifecycle.get("source_evidence_schemas")),
        "rule_group": public_label(lifecycle.get("rule_group") or "repeated-context-conservative", "repeated-context-conservative"),
        "staged_at": lifecycle.get("staged_at") if isinstance(lifecycle.get("staged_at"), str) else None,
        "projected_saved_chars": _as_int(lifecycle.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(lifecycle.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(lifecycle.get("projected_saved_usd")), 8),
        "rollback_metadata_present": bool(lifecycle.get("rollback_metadata_present")),
        "cohorts": {
            "canary_applied": _empty_crunch_impact_cohort(),
            "canary_holdout": _empty_crunch_impact_cohort(),
            "safety_stopped": _empty_crunch_impact_cohort(),
            "fallback": _empty_crunch_impact_cohort(),
            "rollback": _empty_crunch_impact_cohort(),
            "skipped": _empty_crunch_impact_cohort(),
            "unknown": _empty_crunch_impact_cohort(),
        },
        "status_counts": {},
        "reason_counts": {},
        "first_observed_at": None,
        "latest_observed_at": None,
    }


def _add_crunch_impact_row(candidate: dict[str, Any], row: dict[str, Any], crunch: dict[str, Any], lifecycle: dict[str, Any]) -> None:
    cohort_name = _crunch_canary_cohort_name(lifecycle)
    cohort = candidate["cohorts"].setdefault(cohort_name, _empty_crunch_impact_cohort())
    input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    text_chars = _as_int(row.get("text_chars"))
    before_chars = _crunch_before_chars(crunch, text_chars=text_chars, input_tokens=input_tokens)
    after_chars = _crunch_after_chars(crunch, before_chars)
    saved_chars = _crunch_saved_chars(crunch)
    saved_tokens = _crunch_saved_tokens(crunch)
    model = str(row.get("routed_model") or row.get("requested_model") or "")
    saved_usd = _crunch_savings_usd(
        crunch,
        tokens_saved=saved_tokens,
        provider=str(row.get("provider_family") or "unknown"),
        model=model,
        fallback_cost=_as_float(row.get("cost_est_usd")),
        fallback_tokens=input_tokens,
    )
    cohort["count"] += 1
    cohort["before_chars"] += before_chars
    cohort["after_chars"] += after_chars
    cohort["saved_chars"] += saved_chars
    cohort["saved_tokens"] += saved_tokens
    cohort["saved_usd"] += saved_usd
    cost_est = _as_float(row.get("cost_est_usd"))
    baseline = _as_float(row.get("cost_baseline_usd") or row.get("baseline_cost_usd"))
    cohort["cost_est_usd"] += cost_est
    cohort["baseline_cost_usd"] += baseline if baseline > 0 else cost_est
    latency_ms = _as_int(row.get("latency_ms"), -1)
    if latency_ms >= 0:
        cohort["latency_ms_total"] += latency_ms
        cohort["latency_sample_count"] += 1
    if _status_bucket(row.get("status_code")) in {"4xx", "5xx"}:
        cohort["error_count"] += 1
    cohort["retry_count"] += _as_int(row.get("retry_count"))
    if cohort_name == "fallback":
        cohort["fallback_count"] += 1
    if cohort_name == "safety_stopped":
        cohort["safety_stop_count"] += 1
    if cohort_name == "rollback":
        cohort["rollback_count"] += 1
    _increment(candidate["status_counts"], cohort_name)
    _increment(candidate["reason_counts"], lifecycle.get("reason") or cohort_name)
    created_at = str(row.get("created_at") or "")
    if created_at:
        first = candidate.get("first_observed_at")
        latest = candidate.get("latest_observed_at")
        candidate["first_observed_at"] = created_at if first is None else min(str(first), created_at)
        candidate["latest_observed_at"] = created_at if latest is None else max(str(latest), created_at)


def _finalize_crunch_impact_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    errors = _as_int(raw.get("error_count"))
    retries = _as_int(raw.get("retry_count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    return {
        "count": count,
        "before_chars": _as_int(raw.get("before_chars")),
        "after_chars": _as_int(raw.get("after_chars")),
        "saved_chars": _as_int(raw.get("saved_chars")),
        "saved_tokens": _as_int(raw.get("saved_tokens")),
        "saved_usd": round(_as_float(raw.get("saved_usd")), 8),
        "cost_est_usd": round(_as_float(raw.get("cost_est_usd")), 8),
        "baseline_cost_usd": round(_as_float(raw.get("baseline_cost_usd")), 8),
        "cost_delta_usd": round(
            _as_float(raw.get("baseline_cost_usd")) - _as_float(raw.get("cost_est_usd")),
            8,
        ),
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "error_count": errors,
        "retry_count": retries,
        "fallback_count": _as_int(raw.get("fallback_count")),
        "safety_stop_count": _as_int(raw.get("safety_stop_count")),
        "rollback_count": _as_int(raw.get("rollback_count")),
        "error_rate": round(errors / count, 6) if count else 0.0,
        "retry_rate": round(retries / count, 6) if count else 0.0,
        "avg_saved_tokens": round(_as_int(raw.get("saved_tokens")) / count, 2) if count else 0.0,
        "avg_saved_usd": round(_as_float(raw.get("saved_usd")) / count, 8) if count else 0.0,
    }


def _crunch_impact_stale(latest_observed_at: str | None, *, max_age_hours: float) -> bool:
    latest = _parse_utc(latest_observed_at)
    if latest is None:
        return False
    age_hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600.0
    return age_hours > max_age_hours


def _crunch_impact_verdict(
    *,
    applied: dict[str, Any],
    holdout: dict[str, Any],
    safety: dict[str, Any],
    fallback: dict[str, Any],
    rollback: dict[str, Any],
    stale: bool,
) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    if _as_int(rollback.get("count")) or _as_int(rollback.get("rollback_count")):
        reasons.append("rollback-observed")
    if _as_int(safety.get("count")) or _as_int(safety.get("safety_stop_count")):
        reasons.append("canary-safety-stopped")
    if stale:
        reasons.append("stale-canary-impact-evidence")
    if _as_int(applied.get("count")) <= 0:
        reasons.append("missing-applied-coverage")
    if _as_int(holdout.get("count")) <= 0:
        reasons.append("missing-holdout-coverage")
    if _as_int(applied.get("count")) > 0 and _as_int(applied.get("saved_tokens")) <= 0 and _as_float(applied.get("saved_usd")) <= 0:
        reasons.append("no-applied-savings")
    if _as_float(applied.get("error_rate")) > _as_float(holdout.get("error_rate")):
        reasons.append("error-rate-regression")
    if _as_float(applied.get("retry_rate")) > _as_float(holdout.get("retry_rate")):
        reasons.append("retry-rate-regression")
    if _as_int(fallback.get("count")) or _as_int(fallback.get("fallback_count")):
        reasons.append("fallback-observed")
    if reasons:
        return "no-widen", sorted(set(reasons), key=reasons.index), reasons[0]
    return "widen-ready", ["applied-savings-with-holdout-no-regression"], "ready-to-widen-repeated-context-crunch-canary"


def _crunch_impact_recommendation(
    *,
    verdict: str,
    reasons: list[str],
    applied: dict[str, Any],
    holdout: dict[str, Any],
) -> tuple[str, str]:
    reason_set = set(reasons)
    if verdict == "widen-ready":
        return "promotion-ready", "widen-repeated-context-crunch-canary"
    if reason_set & {
        "rollback-observed",
        "canary-safety-stopped",
        "error-rate-regression",
        "retry-rate-regression",
        "fallback-observed",
    }:
        return "rollback", "rollback-repeated-context-crunch-canary"
    if reason_set & {"missing-applied-coverage", "missing-holdout-coverage", "stale-canary-impact-evidence"}:
        return "collect-more-evidence", "collect-repeated-context-crunch-canary-impact-evidence"
    if _as_int(applied.get("count")) > 0 and _as_int(holdout.get("count")) > 0:
        return "keep-blocked", "keep-repeated-context-crunch-canary-blocked"
    return "collect-more-evidence", "collect-repeated-context-crunch-canary-impact-evidence"


def _crunch_impact_next_action(
    *,
    impact_recommendation: str | None,
    applied_count: int,
    holdout_count: int,
    reason_codes: list[str],
) -> str:
    if impact_recommendation == "promotion-ready":
        return "widen"
    if impact_recommendation == "rollback":
        return "rollback"
    if applied_count <= 0 or "missing-applied-coverage" in reason_codes:
        return "stage-canary-first"
    return "keep-observing"


def _crunch_impact_graduation_decision(
    *,
    impact_recommendation: str | None,
    applied_count: int,
    holdout_count: int,
    reason_codes: list[str],
) -> str:
    if impact_recommendation == "promotion-ready":
        return "widen"
    if impact_recommendation == "rollback":
        return "rollback"
    if impact_recommendation == "collect-more-evidence":
        return "keep-staged"
    if impact_recommendation == "keep-blocked":
        return "keep-blocked"
    if applied_count <= 0 or holdout_count <= 0:
        return "keep-staged"
    if reason_codes:
        return "keep-blocked"
    return "keep-staged"


def _crunch_impact_durable_next_action(
    *,
    impact_recommendation: str | None,
    reason_codes: list[str],
    applied_count: int,
    holdout_count: int,
) -> str:
    if impact_recommendation == "promotion-ready":
        return "widen"
    if impact_recommendation == "rollback":
        return "rollback"
    if impact_recommendation == "keep-blocked":
        return "blocked"
    if impact_recommendation == "collect-more-evidence":
        return "measure-more" if applied_count > 0 or holdout_count > 0 else "keep-staged"
    if reason_codes:
        return "blocked"
    return "keep-staged"


def _crunch_impact_missing_measurements_for_candidate(
    *,
    applied_count: int,
    holdout_count: int,
    saved_tokens: int,
    saved_usd: float,
) -> list[str]:
    missing: list[str] = []
    if applied_count <= 0:
        missing.append("applied-crunch-canary-coverage")
    if holdout_count <= 0:
        missing.append("holdout-crunch-canary-coverage")
    if applied_count > 0 and saved_tokens <= 0 and saved_usd <= 0:
        missing.append("applied-crunch-savings-measurement")
    return missing


def _crunch_impact_coverage(
    *,
    applied_count: int,
    holdout_count: int,
    skipped_count: int = 0,
    fallback_count: int = 0,
    safety_stop_count: int = 0,
    rollback_count: int = 0,
    unknown_count: int = 0,
) -> dict[str, Any]:
    observed_count = (
        applied_count
        + holdout_count
        + skipped_count
        + fallback_count
        + safety_stop_count
        + rollback_count
        + unknown_count
    )
    return {
        "schema": "tokenclaw.request_shape_crunch_canary_coverage.v1",
        "observed_count": observed_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "fallback_count": fallback_count,
        "safety_stop_count": safety_stop_count,
        "rollback_count": rollback_count,
        "unknown_count": unknown_count,
        "has_applied_coverage": applied_count > 0,
        "has_holdout_coverage": holdout_count > 0,
        "applied_coverage_rate": round(applied_count / observed_count, 6) if observed_count else 0.0,
        "holdout_coverage_rate": round(holdout_count / observed_count, 6) if observed_count else 0.0,
        "applied_to_holdout_ratio": round(applied_count / holdout_count, 6) if holdout_count else None,
        "aggregate_only": True,
        "metadata_only": True,
    }


def _crunch_captured_savings(
    *,
    policy_id: Any,
    cohort_id: Any,
    rule_group: Any,
    applied: dict[str, Any],
    holdout: dict[str, Any],
    projected_saved_tokens: int,
    projected_saved_usd: float,
) -> dict[str, Any]:
    applied_count = _as_int(applied.get("count"))
    holdout_count = _as_int(holdout.get("count"))
    applied_cost_delta = _as_float(applied.get("cost_delta_usd"))
    holdout_cost_delta = _as_float(holdout.get("cost_delta_usd"))
    holdout_avg_delta = holdout_cost_delta / holdout_count if holdout_count else 0.0
    expected_holdout_delta = holdout_avg_delta * applied_count
    captured_usd = max(0.0, applied_cost_delta - expected_holdout_delta)

    applied_saved_tokens = _as_int(applied.get("saved_tokens"))
    holdout_saved_tokens = _as_int(holdout.get("saved_tokens"))
    holdout_avg_tokens = holdout_saved_tokens / holdout_count if holdout_count else 0.0
    captured_tokens = max(0, int(round(applied_saved_tokens - (holdout_avg_tokens * applied_count))))
    realization_ratio = (
        round(captured_usd / _as_float(projected_saved_usd), 6)
        if _as_float(projected_saved_usd) > 0
        else None
    )
    status = "captured" if applied_count > 0 and holdout_count > 0 and captured_usd > 0 else "no-captured-savings"
    if applied_count <= 0 or holdout_count <= 0:
        status = "missing-applied-or-holdout-coverage"
    return {
        "schema": CRUNCH_CAPTURED_SAVINGS_SCHEMA,
        "status": status,
        "policy_id": public_label(policy_id, "unknown"),
        "cohort_id": public_label(cohort_id, "unknown"),
        "rule_group": public_label(rule_group, "repeated-context-conservative"),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "applied_cost_est_usd": round(_as_float(applied.get("cost_est_usd")), 8),
        "applied_baseline_cost_usd": round(_as_float(applied.get("baseline_cost_usd")), 8),
        "holdout_cost_est_usd": round(_as_float(holdout.get("cost_est_usd")), 8),
        "holdout_baseline_cost_usd": round(_as_float(holdout.get("baseline_cost_usd")), 8),
        "applied_cost_delta_usd": round(applied_cost_delta, 8),
        "holdout_cost_delta_usd": round(holdout_cost_delta, 8),
        "holdout_avg_cost_delta_usd": round(holdout_avg_delta, 8),
        "expected_holdout_cost_delta_usd": round(expected_holdout_delta, 8),
        "captured_saved_tokens": captured_tokens,
        "captured_saved_usd": round(captured_usd, 8),
        "projected_saved_tokens": _as_int(projected_saved_tokens),
        "projected_saved_usd": round(_as_float(projected_saved_usd), 8),
        "projection_realization_ratio": realization_ratio,
        "measurement_basis": "canary-applied baseline-minus-actual cost delta minus holdout average cost delta scaled to applied coverage",
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_captured_savings_summary(captured_rows: list[dict[str, Any]]) -> dict[str, Any]:
    captured_count = sum(1 for row in captured_rows if row.get("status") == "captured")
    applied = sum(_as_int(row.get("applied_count")) for row in captured_rows)
    holdout = sum(_as_int(row.get("holdout_count")) for row in captured_rows)
    captured_usd = sum(_as_float(row.get("captured_saved_usd")) for row in captured_rows)
    projected_usd = sum(_as_float(row.get("projected_saved_usd")) for row in captured_rows)
    rule_groups: dict[str, int] = {}
    for row in captured_rows:
        _increment(rule_groups, row.get("rule_group") or "repeated-context-conservative")
    return {
        "schema": "tokenclaw.request_shape_crunch_captured_savings_summary.v1",
        "status": "captured" if captured_count else "no-captured-savings",
        "cohort_count": len(captured_rows),
        "captured_cohort_count": captured_count,
        "applied_count": applied,
        "holdout_count": holdout,
        "captured_saved_tokens": sum(_as_int(row.get("captured_saved_tokens")) for row in captured_rows),
        "captured_saved_usd": round(captured_usd, 8),
        "projected_saved_tokens": sum(_as_int(row.get("projected_saved_tokens")) for row in captured_rows),
        "projected_saved_usd": round(projected_usd, 8),
        "projection_realization_ratio": round(captured_usd / projected_usd, 6) if projected_usd > 0 else None,
        "rule_group_breakdown": _breakdown(rule_groups),
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_activation_lifecycle_feedback(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    family_state_counts: dict[str, int] = {}
    metadata: list[dict[str, Any]] = []
    for candidate in candidates:
        verdict = str(candidate.get("verdict") or "unknown")
        state = "healthy_canary" if verdict == "widen-ready" else "suppressed"
        _increment(state_counts, state)
        _increment(family_state_counts, f"crunch:{state}")
        cohorts = candidate.get("cohorts") if isinstance(candidate.get("cohorts"), dict) else {}
        for name, cohort in cohorts.items():
            count = _as_int(cohort.get("count")) if isinstance(cohort, dict) else 0
            if count:
                _increment(cohort_counts, name, count)
        metadata.append(
            {
                "policy_ref": public_label(candidate.get("policy_id"), "unknown"),
                "cohort_label": "canary",
                "action_family": "crunch",
                "event_count": _as_int(candidate.get("observed_count")),
                "applied_count": _as_int(candidate.get("applied_count")),
                "holdout_count": _as_int(candidate.get("holdout_count")),
                "fallback_count": _as_int(candidate.get("fallback_count")),
                "error_count": _as_int(candidate.get("applied_error_count")),
                "retry_count": _as_int(candidate.get("applied_retry_count")),
                "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
                "savings_estimate_usd": round(_as_float(candidate.get("saved_usd")), 8),
                "reason_codes": candidate.get("reason_codes") or [],
                "blocker_reason_breakdown": [
                    {"value": reason, "count": 1}
                    for reason in candidate.get("reason_codes") or []
                ],
            }
        )
    return {
        "schema": "tokenclaw.activation_staged_lifecycle_feedback_summary.v1",
        "queue_rows": 0,
        "family_event_count": sum(_as_int(item.get("observed_count")) for item in candidates),
        "state_breakdown": _breakdown(state_counts),
        "event_phase_breakdown": [{"value": "impact", "count": len(candidates)}] if candidates else [],
        "cohort_breakdown": _breakdown(cohort_counts),
        "family_state_breakdown": _breakdown(family_state_counts),
        "candidate_id_breakdown": [],
        "cohort_lifecycle_metadata": metadata[:50],
        "payload_json_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_stage_follow_up_action(action: dict[str, Any], *, rank: int) -> dict[str, Any]:
    conditions = action.get("conditions") if isinstance(action.get("conditions"), dict) else {}
    return {
        "schema": "tokenclaw.request_shape_crunch_canary_stage_follow_up.v1",
        "rank": rank,
        "action_type": public_label(action.get("action_type") or "stage-local-repeated-context-crunch-canary", "unknown"),
        "target_local_policy": "crunch_rules",
        "policy_id": public_label(action.get("policy_id"), "unknown"),
        "cohort_id": public_label(action.get("cohort_id"), "unknown"),
        "conditions": _request_shape_crunch_canary_rule_conditions(conditions),
        "rollout_fraction": round(_as_float(action.get("rollout_fraction")), 6),
        "holdout_fraction": round(_as_float(action.get("holdout_fraction")), 6),
        "projected_saved_chars": _as_int(action.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(action.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(action.get("projected_saved_usd")), 8),
        "source_evidence_schema": public_label(action.get("source_evidence_schema"), "unknown"),
        "source_evidence_schemas": _public_label_list(action.get("source_evidence_schemas")),
        "local_only_reason": public_label(
            action.get("local_only_reason") or "file-backed-local-policy-no-managed-dependency",
            "unknown",
        ),
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_activation_ready_measurements(
    *,
    opportunity_report: dict[str, Any] | None,
    impact_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    measured_by_cohort = {
        public_label(candidate.get("cohort_id"), "unknown"): candidate
        for candidate in impact_candidates
        if public_label(candidate.get("cohort_id"), "unknown") != "unknown"
    }
    cohorts = opportunity_report.get("cohorts") if isinstance(opportunity_report, dict) else []
    if not isinstance(cohorts, list):
        cohorts = []
    recommended_actions = opportunity_report.get("recommended_actions") if isinstance(opportunity_report, dict) else []
    if not isinstance(recommended_actions, list):
        recommended_actions = []

    rows: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    projected_tokens = 0
    projected_savings = 0.0
    observed_tokens = 0
    observed_savings = 0.0
    suppressed_cohort_ids: set[str] = set()

    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        cohort_id = public_label(cohort.get("cohort_id"), "unknown")
        measured = measured_by_cohort.get(cohort_id)
        readiness = public_label(cohort.get("readiness"), "unknown")
        lifecycle = cohort.get("crunch_canary_lifecycle") if isinstance(cohort.get("crunch_canary_lifecycle"), dict) else {}
        duplicate = cohort.get("duplicate_suppression") if isinstance(cohort.get("duplicate_suppression"), dict) else {}
        blockers = _public_label_list(cohort.get("blockers") or cohort.get("evidence_blocker_codes"))
        if measured and (
            _as_int(measured.get("applied_count"))
            or _as_int(measured.get("holdout_count"))
            or _as_int(measured.get("safety_stop_count"))
            or _as_int(measured.get("fallback_count"))
            or _as_int(measured.get("rollback_count"))
        ):
            state = "measured"
            next_action = public_label(measured.get("durable_next_action") or measured.get("next_action"), "unknown")
            observed_tokens += _as_int(measured.get("saved_tokens"))
            observed_savings += _as_float(measured.get("saved_usd"))
            reason_codes = _public_label_list(measured.get("reason_codes"))
        elif readiness in {"canary-staged", "canary-applied", "canary-holdout"} or bool(duplicate.get("suppresses_new_stage_action")):
            state = "keep-staged"
            next_action = "measure-repeated-context-crunch-canary-impact"
            reason_codes = [public_label(duplicate.get("reason") or "matching-repeated-context-crunch-canary-already-staged", "unknown")]
        elif readiness in {"measurement-ready", "activation-ready"}:
            state = "stageable"
            next_action = "stage-repeated-context-crunch-canary"
            reason_codes = []
        else:
            state = "blocked"
            next_action = "resolve-repeated-context-crunch-cohort-blocker"
            reason_codes = blockers or [public_label(cohort.get("reason"), "unknown")]
        projected_tokens += _as_int(cohort.get("projected_saved_tokens"))
        projected_savings += _as_float(cohort.get("projected_saved_usd"))
        applied_count = _as_int(measured.get("applied_count")) if measured else _as_int(lifecycle.get("applied_count"))
        holdout_count = _as_int(measured.get("holdout_count")) if measured else _as_int(lifecycle.get("holdout_count"))
        skipped_count = (
            _as_int((measured.get("cohorts") or {}).get("skipped", {}).get("count"))
            if measured
            else _as_int(lifecycle.get("skipped_count"))
        )
        fallback_count = _as_int(measured.get("fallback_count")) if measured else _as_int(lifecycle.get("fallback_count"))
        safety_stop_count = _as_int(measured.get("safety_stop_count")) if measured else _as_int(lifecycle.get("safety_stopped_count"))
        rollback_count = _as_int(measured.get("rollback_count")) if measured else _as_int(lifecycle.get("rollback_count"))
        retry_count = (
            _as_int(measured.get("applied_retry_count")) + _as_int(measured.get("holdout_retry_count"))
            if measured
            else _as_int(lifecycle.get("retry_count"))
        )
        error_count = (
            _as_int(measured.get("applied_error_count")) + _as_int(measured.get("holdout_error_count"))
            if measured
            else _as_int(lifecycle.get("error_count"))
        )
        observed_saved_tokens = _as_int(measured.get("saved_tokens")) if measured else 0
        observed_saved_usd = round(_as_float(measured.get("saved_usd")), 8) if measured else 0.0
        missing_measurements = _public_label_list(measured.get("missing_measurements")) if measured else []
        if applied_count <= 0:
            missing_measurements.append("applied-crunch-canary-coverage")
        if holdout_count <= 0:
            missing_measurements.append("holdout-crunch-canary-coverage")
        if applied_count > 0 and observed_saved_tokens <= 0 and observed_saved_usd <= 0:
            missing_measurements.append("crunch-canary-impact-observed-savings")
        missing_measurements = sorted(set(missing_measurements))
        duplicate_suppressed = bool(duplicate.get("suppressed") or duplicate.get("suppresses_new_stage_action"))
        if duplicate_suppressed and cohort_id != "unknown":
            suppressed_cohort_ids.add(cohort_id)
        _increment(state_counts, state)
        rows.append(
            {
                "schema": "tokenclaw.request_shape_crunch_activation_ready_cohort_measurement.v1",
                "rank": _as_int(cohort.get("rank")),
                "cohort_id": cohort_id,
                "policy_id": public_label(cohort.get("policy_id"), "unknown"),
                "state": state,
                "readiness": readiness,
                "next_action": next_action,
                "provider_family": public_label(cohort.get("provider_family"), "unknown"),
                "source_surface": public_label(cohort.get("source_surface"), "unknown"),
                "endpoint": public_label(cohort.get("endpoint"), "unknown"),
                "category": public_label(cohort.get("category"), "unknown"),
                "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
                "stream": bool(cohort.get("stream")),
                "has_tools": bool(cohort.get("has_tools")),
                "cache_status": public_label(cohort.get("cache_status"), "unknown"),
                "routing_status": public_label(cohort.get("routing_status"), "unknown"),
                "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
                "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
                "sample_count": _as_int(cohort.get("row_count")),
                "row_count": _as_int(cohort.get("row_count")),
                "projected_saved_tokens": _as_int(cohort.get("projected_saved_tokens")),
                "projected_saved_usd": round(_as_float(cohort.get("projected_saved_usd")), 8),
                "projected_saved_chars": _as_int(cohort.get("projected_saved_chars")),
                "current_conservative_saved_tokens": _as_int(cohort.get("current_conservative_tokens_saved")),
                "current_conservative_saved_chars": _as_int(cohort.get("current_conservative_chars_saved")),
                "current_conservative_saved_usd": round(_as_float(cohort.get("current_conservative_savings_usd")), 8),
                "observed_saved_tokens": observed_saved_tokens,
                "observed_saved_usd": observed_saved_usd,
                "applied_count": applied_count,
                "holdout_count": holdout_count,
                "skipped_count": skipped_count,
                "fallback_count": fallback_count,
                "error_count": error_count,
                "retry_count": retry_count,
                "rollback_count": rollback_count,
                "safety_stop_count": safety_stop_count,
                "reason_codes": [code for code in reason_codes if code and code != "unknown"],
                "missing_measurements": missing_measurements,
                "evidence_blocker_codes": blockers,
                "duplicate_suppression": {
                    "suppressed": duplicate_suppressed,
                    "reason": public_label(duplicate.get("reason"), "unknown"),
                    "active_at_max_rollout": bool(duplicate.get("active_at_max_rollout")),
                    "matching_local_policy": public_label(duplicate.get("matching_local_policy"), "unknown"),
                    "matching_policy_id": public_label(duplicate.get("matching_policy_id"), "unknown"),
                    "matching_cohort_id": public_label(duplicate.get("matching_cohort_id"), "unknown"),
                    "matching_max_rollout_fraction": round(_as_float(duplicate.get("matching_max_rollout_fraction")), 6),
                    "metadata_only": True,
                    "aggregate_only": True,
                },
                "privacy": _crunch_opportunity_privacy(),
            }
        )

    rows.sort(
        key=lambda item: (
            {"measured": 3, "keep-staged": 2, "stageable": 1, "blocked": 0}.get(str(item.get("state")), 0),
            _as_float(item.get("observed_saved_usd")) + _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    stage_follow_ups = [
        _crunch_impact_stage_follow_up_action(action, rank=rank)
        for rank, action in enumerate(
            [
                action
                for action in recommended_actions
                if isinstance(action, dict)
                and public_label(action.get("cohort_id"), "unknown") not in suppressed_cohort_ids
            ][:10],
            start=1,
        )
    ]
    return {
        "schema": "tokenclaw.request_shape_crunch_activation_ready_measurements.v1",
        "status": "classified" if rows else "no-activation-ready-cohorts",
        "cohort_count": len(rows),
        "measured_count": _as_int(state_counts.get("measured")),
        "keep_staged_count": _as_int(state_counts.get("keep-staged")),
        "stageable_count": _as_int(state_counts.get("stageable")),
        "blocked_count": _as_int(state_counts.get("blocked")),
        "projected_saved_tokens": projected_tokens,
        "projected_saved_usd": round(projected_savings, 8),
        "observed_saved_tokens": observed_tokens,
        "observed_saved_usd": round(observed_savings, 8),
        "state_breakdown": _breakdown(state_counts),
        "bounded_stage_recommendation_count": len(stage_follow_ups),
        "bounded_stage_recommendations": stage_follow_ups,
        "cohorts": rows[:50],
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_row_ref(value: dict[str, Any], *, prefix: str) -> str:
    return public_id(stable_json(value), prefix=prefix, fallback=f"{prefix}:unknown") or f"{prefix}:unknown"


def _crunch_impact_state_from_measurement(value: dict[str, Any]) -> str:
    state = public_label(value.get("state"), "unknown")
    if state == "measured":
        return "measured"
    if state in {"keep-staged", "stageable"}:
        return "measurement-required"
    if state == "blocked":
        return "blocked"
    return "measurement-required"


def _crunch_impact_state_from_candidate(value: dict[str, Any]) -> str:
    recommendation = public_label(value.get("impact_recommendation"), "unknown")
    reasons = {public_label(reason, "unknown") for reason in (value.get("reason_codes") or [])}
    if recommendation == "promotion-ready":
        return "measured"
    if recommendation in {"rollback", "keep-blocked"}:
        return "blocked"
    if reasons & {"canary-safety-stopped", "rollback-observed", "fallback-observed", "error-rate-regression"}:
        return "blocked"
    return "measurement-required"


def _crunch_impact_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    state_priority = {"measured": 4, "measurement-required": 3, "blocked": 2, "superseded": 1}
    next_action_priority = {
        "measure-full-rollout-repeated-context-crunch-outcomes": 5,
        "measure-repeated-context-crunch-canary-impact": 4,
        "stage-repeated-context-crunch-canary": 3,
        "widen": 2,
        "rollback": 2,
    }
    return (
        public_label(row.get("local_action_family"), "unknown") == "crunch",
        state_priority.get(public_label(row.get("measurement_state"), "unknown"), 0),
        next_action_priority.get(public_label(row.get("next_action"), "unknown"), 0),
        _as_float(row.get("observed_saved_usd")),
        _as_float(row.get("projected_saved_usd")),
        _as_int(row.get("sample_count")),
    )


def _crunch_impact_rows_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("cohort_metadata") if isinstance(candidate.get("cohort_metadata"), dict) else {}
    coverage = candidate.get("coverage") if isinstance(candidate.get("coverage"), dict) else {}
    row_basis = {
        "source": "impact_candidate",
        "policy": candidate.get("policy_id"),
        "cohort": candidate.get("cohort_id"),
        "metadata": metadata,
    }
    applied_count = _as_int(candidate.get("applied_count"))
    holdout_count = _as_int(candidate.get("holdout_count"))
    skipped_count = _as_int(coverage.get("skipped_count"))
    sample_count = _as_int(candidate.get("observed_count")) or applied_count + holdout_count + skipped_count
    return {
        "schema": CRUNCH_CANARY_IMPACT_ROW_SCHEMA,
        "source": "crunch_canary_impact",
        "source_schema": public_label(candidate.get("schema"), "unknown"),
        "cohort_ref": _crunch_impact_row_ref(row_basis, prefix="crunch-impact"),
        "policy_ref": _crunch_impact_row_ref({"policy": candidate.get("policy_id")}, prefix="policy"),
        "measurement_state": _crunch_impact_state_from_candidate(candidate),
        "local_action_family": "crunch",
        "readiness_state": public_label(candidate.get("verdict"), "unknown"),
        "next_action": public_label(candidate.get("durable_next_action") or candidate.get("next_action"), "unknown"),
        "blocker_codes": _public_label_list(candidate.get("reason_codes")),
        "sample_count": sample_count,
        "row_count": sample_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "fallback_count": _as_int(candidate.get("fallback_count")),
        "error_count": _as_int(candidate.get("applied_error_count")) + _as_int(candidate.get("holdout_error_count")),
        "retry_count": _as_int(candidate.get("applied_retry_count")) + _as_int(candidate.get("holdout_retry_count")),
        "rollback_count": _as_int(candidate.get("rollback_count")),
        "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
        "projected_saved_tokens": _as_int(candidate.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(candidate.get("projected_saved_usd")), 8),
        "observed_saved_tokens": _as_int(candidate.get("saved_tokens")),
        "observed_saved_usd": round(_as_float(candidate.get("saved_usd")), 8),
        "provider_family": public_label(metadata.get("provider_family"), "unknown"),
        "source_surface": public_label(metadata.get("source_surface"), "unknown"),
        "endpoint": public_label(metadata.get("endpoint"), "unknown"),
        "category": public_label(metadata.get("category"), "unknown"),
        "workflow_phase": public_label(metadata.get("workflow_phase"), "unknown"),
        "stream": bool(metadata.get("stream")),
        "has_tools": bool(metadata.get("has_tools")),
        "cache_status": public_label(metadata.get("cache_status"), "unknown"),
        "routing_status": public_label(metadata.get("routing_status"), "unknown"),
        "text_bucket": public_label(metadata.get("text_bucket"), "unknown"),
        "token_bucket": public_label(metadata.get("token_bucket"), "unknown"),
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_rows_from_measurement(row: dict[str, Any]) -> dict[str, Any]:
    row_basis = {
        "source": "activation_ready_measurement",
        "rank": row.get("rank"),
        "cohort": row.get("cohort_id"),
        "shape": _crunch_remaining_shape_key(row),
    }
    return {
        "schema": CRUNCH_CANARY_IMPACT_ROW_SCHEMA,
        "source": "activation_ready_measurement",
        "source_schema": public_label(row.get("schema"), "unknown"),
        "cohort_ref": _crunch_impact_row_ref(row_basis, prefix="crunch-impact"),
        "policy_ref": _crunch_impact_row_ref({"policy": row.get("policy_id")}, prefix="policy"),
        "measurement_state": _crunch_impact_state_from_measurement(row),
        "local_action_family": "crunch",
        "readiness_state": public_label(row.get("readiness"), "unknown"),
        "next_action": public_label(row.get("next_action"), "unknown"),
        "blocker_codes": _public_label_list(row.get("reason_codes") or row.get("evidence_blocker_codes")),
        "sample_count": _as_int(row.get("sample_count") or row.get("row_count")),
        "row_count": _as_int(row.get("row_count") or row.get("sample_count")),
        "applied_count": _as_int(row.get("applied_count")),
        "holdout_count": _as_int(row.get("holdout_count")),
        "skipped_count": _as_int(row.get("skipped_count")),
        "fallback_count": _as_int(row.get("fallback_count")),
        "error_count": _as_int(row.get("error_count")),
        "retry_count": _as_int(row.get("retry_count")),
        "rollback_count": _as_int(row.get("rollback_count")),
        "safety_stop_count": _as_int(row.get("safety_stop_count")),
        "projected_saved_tokens": _as_int(row.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(row.get("projected_saved_usd")), 8),
        "observed_saved_tokens": _as_int(row.get("observed_saved_tokens")),
        "observed_saved_usd": round(_as_float(row.get("observed_saved_usd")), 8),
        "provider_family": public_label(row.get("provider_family"), "unknown"),
        "source_surface": public_label(row.get("source_surface"), "unknown"),
        "endpoint": public_label(row.get("endpoint"), "unknown"),
        "category": public_label(row.get("category"), "unknown"),
        "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "cache_status": public_label(row.get("cache_status"), "unknown"),
        "routing_status": public_label(row.get("routing_status"), "unknown"),
        "text_bucket": public_label(row.get("text_bucket"), "unknown"),
        "token_bucket": public_label(row.get("token_bucket"), "unknown"),
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_rows_from_follow_up(
    candidate: dict[str, Any],
    *,
    superseded: bool,
) -> dict[str, Any]:
    state = "superseded" if superseded else "measurement-required"
    readiness_state = public_label(candidate.get("readiness_state"), "unknown")
    if readiness_state == "blocked":
        state = "blocked"
    row_basis = {
        "source": "follow_up_candidate",
        "rank": candidate.get("rank"),
        "shape": _crunch_remaining_shape_key(candidate),
        "next_action": candidate.get("next_action"),
    }
    projected_saved_tokens = _as_int(candidate.get("projected_saved_tokens") or candidate.get("projected_crunch_tokens_saved"))
    projected_saved_usd = _as_float(candidate.get("projected_savings_usd") or candidate.get("projected_crunch_savings_usd"))
    observed_saved_usd = _as_float(candidate.get("observed_savings_usd"))
    return {
        "schema": CRUNCH_CANARY_IMPACT_ROW_SCHEMA,
        "source": "follow_up_candidates",
        "source_schema": public_label(candidate.get("schema"), FOLLOW_UP_CANDIDATES_SCHEMA),
        "cohort_ref": _crunch_impact_row_ref(row_basis, prefix="crunch-impact"),
        "policy_ref": None,
        "measurement_state": state,
        "local_action_family": public_label(candidate.get("local_action_family"), "unknown"),
        "readiness_state": readiness_state,
        "next_action": public_label(candidate.get("next_action"), "unknown"),
        "blocker_codes": _public_label_list(candidate.get("blocker_codes")),
        "sample_count": _as_int(candidate.get("sample_count") or candidate.get("row_count")),
        "row_count": _as_int(candidate.get("row_count") or candidate.get("sample_count")),
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "fallback_count": _as_int(candidate.get("fallback_count")),
        "error_count": _as_int(candidate.get("error_count")),
        "retry_count": _as_int(candidate.get("retry_count")),
        "rollback_count": _as_int(candidate.get("rollback_count")),
        "safety_stop_count": _as_int(candidate.get("safety_stop_count")),
        "projected_saved_tokens": projected_saved_tokens,
        "projected_saved_usd": round(projected_saved_usd, 8),
        "observed_saved_tokens": _as_int(candidate.get("observed_saved_tokens")),
        "observed_saved_usd": round(observed_saved_usd, 8),
        "provider_family": public_label(candidate.get("provider_family"), "unknown"),
        "source_surface": public_label(candidate.get("source_surface"), "unknown"),
        "endpoint": public_label(candidate.get("endpoint"), "unknown"),
        "category": public_label(candidate.get("category"), "unknown"),
        "workflow_phase": public_label(candidate.get("workflow_phase"), "unknown"),
        "stream": bool(candidate.get("stream")),
        "has_tools": bool(candidate.get("has_tools")),
        "cache_status": public_label(candidate.get("cache_status"), "unknown"),
        "routing_status": public_label(candidate.get("routing_status"), "unknown"),
        "text_bucket": public_label(candidate.get("text_bucket"), "unknown"),
        "token_bucket": public_label(candidate.get("token_bucket"), "unknown"),
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_row_from_activation_evidence(activation_evidence: dict[str, Any]) -> dict[str, Any] | None:
    summary = activation_evidence.get("summary") if isinstance(activation_evidence.get("summary"), dict) else {}
    if not summary and not activation_evidence:
        return None
    applied_count = _as_int(summary.get("applied_count") or activation_evidence.get("applied_count"))
    holdout_count = _as_int(summary.get("holdout_count") or activation_evidence.get("holdout_count"))
    observed_saved_tokens = _as_int(summary.get("observed_saved_tokens") or activation_evidence.get("observed_saved_tokens"))
    observed_saved_usd = _as_float(summary.get("observed_saved_usd") or activation_evidence.get("observed_saved_usd"))
    if applied_count <= 0 and holdout_count <= 0 and observed_saved_tokens <= 0 and observed_saved_usd <= 0:
        return None
    source_status = public_label(activation_evidence.get("status") or summary.get("post_widening_status"), "unknown")
    row_basis = {
        "source": "activation_evidence",
        "decision": activation_evidence.get("decision_id") or summary.get("decision_id"),
        "status": source_status,
        "next_action": activation_evidence.get("next_action") or summary.get("next_action"),
    }
    return {
        "schema": CRUNCH_CANARY_IMPACT_ROW_SCHEMA,
        "source": "crunch_activation_evidence",
        "source_schema": public_label(activation_evidence.get("schema"), CRUNCH_ACTIVATION_EVIDENCE_SCHEMA),
        "cohort_ref": _crunch_impact_row_ref(row_basis, prefix="crunch-impact"),
        "policy_ref": _crunch_impact_row_ref({"decision": activation_evidence.get("decision_id")}, prefix="policy"),
        "measurement_state": "measured",
        "local_action_family": "crunch",
        "readiness_state": public_label(summary.get("post_widening_status") or source_status, "unknown"),
        "next_action": public_label(activation_evidence.get("next_action") or summary.get("next_action"), "unknown"),
        "blocker_codes": _public_label_list(
            activation_evidence.get("missing_measurements")
            or summary.get("post_widening_reason_codes")
            or summary.get("post_max_rollout_reason_codes")
        ),
        "sample_count": applied_count + holdout_count + _as_int(summary.get("skipped_count")),
        "row_count": applied_count + holdout_count + _as_int(summary.get("skipped_count")),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": _as_int(summary.get("skipped_count")),
        "fallback_count": _as_int(summary.get("fallback_count")),
        "error_count": 0,
        "retry_count": 0,
        "rollback_count": _as_int(summary.get("rollback_count")),
        "safety_stop_count": _as_int(summary.get("safety_stop_count")),
        "projected_saved_tokens": _as_int(summary.get("observed_saved_tokens") or summary.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(summary.get("observed_saved_usd") or summary.get("projected_saved_usd")), 8),
        "observed_saved_tokens": observed_saved_tokens,
        "observed_saved_usd": round(observed_saved_usd, 8),
        "provider_family": "unknown",
        "source_surface": "unknown",
        "endpoint": "unknown",
        "category": "unknown",
        "workflow_phase": "unknown",
        "stream": False,
        "has_tools": False,
        "cache_status": "unknown",
        "routing_status": "unknown",
        "text_bucket": "unknown",
        "token_bucket": "unknown",
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_canary_impact_rows_report(
    *,
    impact_candidates: list[dict[str, Any]],
    activation_ready_measurements: dict[str, Any] | None = None,
    follow_up_candidates: dict[str, Any] | None = None,
    activation_evidence: dict[str, Any] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in impact_candidates:
        if isinstance(candidate, dict):
            rows.append(_crunch_impact_rows_from_candidate(candidate))

    measurement_rows = (
        activation_ready_measurements.get("cohorts")
        if isinstance(activation_ready_measurements, dict) and isinstance(activation_ready_measurements.get("cohorts"), list)
        else []
    )
    for row in measurement_rows:
        if isinstance(row, dict):
            rows.append(_crunch_impact_rows_from_measurement(row))

    duplicate = activation_evidence.get("duplicate_suppression") if isinstance(activation_evidence, dict) and isinstance(activation_evidence.get("duplicate_suppression"), dict) else {}
    superseded_by_active_rule = bool(
        duplicate.get("suppresses_new_activation_issue")
        or duplicate.get("suppresses_generic_crunch_activation_issue")
    )
    follow_up_rows = (
        follow_up_candidates.get("candidates")
        if isinstance(follow_up_candidates, dict) and isinstance(follow_up_candidates.get("candidates"), list)
        else []
    )
    for candidate in follow_up_rows:
        if not isinstance(candidate, dict):
            continue
        if public_label(candidate.get("local_action_family"), "unknown") != "crunch":
            continue
        if public_label(candidate.get("next_action"), "unknown") not in {
            "measure-repeated-context-crunch-canary-impact",
            "measure-full-rollout-repeated-context-crunch-outcomes",
            "stage-repeated-context-crunch-canary",
        }:
            continue
        rows.append(_crunch_impact_rows_from_follow_up(candidate, superseded=superseded_by_active_rule))

    activation_row = _crunch_impact_row_from_activation_evidence(activation_evidence or {})
    if activation_row is not None:
        rows.append(activation_row)

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("source"),
            row.get("cohort_ref"),
            row.get("next_action"),
            row.get("measurement_state"),
        )
        existing = deduped.get(key)
        if existing is None or _crunch_impact_row_sort_key(row) > _crunch_impact_row_sort_key(existing):
            deduped[key] = row
    ranked_rows = sorted(deduped.values(), key=_crunch_impact_row_sort_key, reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    state_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    local_family_counts: dict[str, int] = {}
    for row in ranked_rows:
        _increment(state_counts, row.get("measurement_state"))
        _increment(next_action_counts, row.get("next_action"))
        _increment(readiness_counts, row.get("readiness_state"))
        _increment(local_family_counts, row.get("local_action_family"))

    capped_limit = max(1, min(_as_int(limit, 50), 100))
    return {
        "schema": CRUNCH_CANARY_IMPACT_ROWS_SCHEMA,
        "status": "ranked" if ranked_rows else "no-repeated-context-crunch-impact-rows",
        "read_only": True,
        "row_count": len(ranked_rows),
        "reported_count": min(len(ranked_rows), capped_limit),
        "state_breakdown": _breakdown(state_counts),
        "next_action_breakdown": _breakdown(next_action_counts),
        "readiness_state_breakdown": _breakdown(readiness_counts),
        "local_action_family_breakdown": _breakdown(local_family_counts),
        "summary": {
            "ranked_row_count": len(ranked_rows),
            "measured_count": _as_int(state_counts.get("measured")),
            "measurement_required_count": _as_int(state_counts.get("measurement-required")),
            "blocked_count": _as_int(state_counts.get("blocked")),
            "superseded_count": _as_int(state_counts.get("superseded")),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in ranked_rows),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in ranked_rows),
            "skipped_count": sum(_as_int(row.get("skipped_count")) for row in ranked_rows),
            "projected_saved_tokens": sum(_as_int(row.get("projected_saved_tokens")) for row in ranked_rows),
            "projected_saved_usd": round(sum(_as_float(row.get("projected_saved_usd")) for row in ranked_rows), 8),
            "observed_saved_tokens": sum(_as_int(row.get("observed_saved_tokens")) for row in ranked_rows),
            "observed_saved_usd": round(sum(_as_float(row.get("observed_saved_usd")) for row in ranked_rows), 8),
            "top_state": ranked_rows[0]["measurement_state"] if ranked_rows else None,
            "top_next_action": ranked_rows[0]["next_action"] if ranked_rows else None,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "acceptance": {
            "has_ranked_repeated_context_crunch_impact_rows": bool(ranked_rows),
            "has_rank": all(_as_int(row.get("rank")) > 0 for row in ranked_rows),
            "has_blocker_codes": all(isinstance(row.get("blocker_codes"), list) for row in ranked_rows),
            "has_local_action_family": all(public_label(row.get("local_action_family"), "unknown") != "unknown" for row in ranked_rows),
            "has_readiness_state": all(public_label(row.get("readiness_state"), "unknown") != "unknown" for row in ranked_rows),
            "has_next_action": all(public_label(row.get("next_action"), "unknown") != "unknown" for row in ranked_rows),
            "has_sample_count": all("sample_count" in row for row in ranked_rows),
            "has_canary_counts": all("applied_count" in row and "holdout_count" in row and "skipped_count" in row for row in ranked_rows),
            "has_projected_and_observed_savings": all("projected_saved_usd" in row and "observed_saved_usd" in row for row in ranked_rows),
            "emits_durable_measurement_state": all(
                public_label(row.get("measurement_state"), "unknown")
                in {"measured", "measurement-required", "blocked", "superseded"}
                for row in ranked_rows
            ),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "rows": ranked_rows[:capped_limit],
        "top_row": ranked_rows[0] if ranked_rows else None,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_impact_newly_staged_measurement(measurements: dict[str, Any]) -> dict[str, Any]:
    rows = measurements.get("cohorts") if isinstance(measurements.get("cohorts"), list) else []
    newly_staged: list[dict[str, Any]] = []
    active_max_rollout: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        duplicate = row.get("duplicate_suppression") if isinstance(row.get("duplicate_suppression"), dict) else {}
        if not bool(duplicate.get("suppressed")):
            continue
        if bool(duplicate.get("active_at_max_rollout")) or duplicate.get("reason") == "repeated-context-crunch-active-at-max-rollout":
            active_max_rollout.append(row)
            continue
        if row.get("state") in {"measured", "keep-staged"}:
            newly_staged.append(row)

    applied_count = sum(_as_int(row.get("applied_count")) for row in newly_staged)
    holdout_count = sum(_as_int(row.get("holdout_count")) for row in newly_staged)
    skipped_count = sum(_as_int(row.get("skipped_count")) for row in newly_staged)
    error_count = sum(_as_int(row.get("error_count")) for row in newly_staged)
    retry_count = sum(_as_int(row.get("retry_count")) for row in newly_staged)
    fallback_count = sum(_as_int(row.get("fallback_count")) for row in newly_staged)
    rollback_count = sum(_as_int(row.get("rollback_count")) for row in newly_staged)
    safety_stop_count = sum(_as_int(row.get("safety_stop_count")) for row in newly_staged)
    observed_saved_tokens = sum(_as_int(row.get("observed_saved_tokens")) for row in newly_staged)
    observed_saved_usd = sum(_as_float(row.get("observed_saved_usd")) for row in newly_staged)
    projected_saved_tokens = sum(_as_int(row.get("projected_saved_tokens")) for row in newly_staged)
    projected_saved_usd = sum(_as_float(row.get("projected_saved_usd")) for row in newly_staged)
    reason_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    for row in newly_staged:
        for reason in row.get("reason_codes") or []:
            _increment(reason_counts, reason)
        for missing in row.get("missing_measurements") or []:
            _increment(missing_counts, missing)

    if not newly_staged:
        status = "no-newly-staged-cohorts"
        next_action = "stage-repeated-context-crunch-canary"
    elif applied_count > 0 and holdout_count > 0:
        status = "measured"
        next_action = "review-repeated-context-crunch-canary-impact"
    else:
        status = "awaiting-live-coverage"
        next_action = "measure-repeated-context-crunch-canary-impact"

    return {
        "schema": "tokenclaw.request_shape_crunch_newly_staged_measurement.v1",
        "status": status,
        "next_action": next_action,
        "cohort_count": len(newly_staged),
        "measured_cohort_count": sum(
            1
            for row in newly_staged
            if _as_int(row.get("applied_count")) > 0 or _as_int(row.get("holdout_count")) > 0
        ),
        "active_max_rollout_suppressed_count": len(active_max_rollout),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "retry_count": retry_count,
        "fallback_count": fallback_count,
        "rollback_count": rollback_count,
        "safety_stop_count": safety_stop_count,
        "observed_saved_tokens": observed_saved_tokens,
        "observed_saved_usd": round(observed_saved_usd, 8),
        "projected_saved_tokens": projected_saved_tokens,
        "projected_saved_usd": round(projected_saved_usd, 8),
        "reason_breakdown": _breakdown(reason_counts),
        "missing_measurement_breakdown": _breakdown(missing_counts),
        "cohorts": newly_staged[:50],
        "active_max_rollout_suppression": [
            {
                "schema": "tokenclaw.request_shape_crunch_active_max_rollout_suppression.v1",
                "cohort_id": row.get("cohort_id"),
                "policy_id": row.get("policy_id"),
                "reason": (row.get("duplicate_suppression") or {}).get("reason")
                if isinstance(row.get("duplicate_suppression"), dict)
                else "repeated-context-crunch-active-at-max-rollout",
                "applied_count": _as_int(row.get("applied_count")),
                "holdout_count": _as_int(row.get("holdout_count")),
                "matching_local_policy": (row.get("duplicate_suppression") or {}).get("matching_local_policy")
                if isinstance(row.get("duplicate_suppression"), dict)
                else "crunch_rules",
                "metadata_only": True,
                "aggregate_only": True,
                "privacy": _crunch_opportunity_privacy(),
            }
            for row in active_max_rollout[:50]
        ],
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_canary_impact_report(
    rows: list[dict[str, Any]],
    *,
    max_evidence_age_hours: float = DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS,
    opportunity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    observed_rows = 0
    for row in rows:
        crunch = _json_obj(row.get("crunch_json"))
        lifecycle = _crunch_canary_lifecycle_from_meta(crunch)
        if lifecycle is None:
            continue
        policy_id = public_label(lifecycle.get("policy_id"), "unknown")
        cohort_id = public_label(lifecycle.get("cohort_id"), "unknown")
        key = (policy_id, cohort_id)
        candidate = candidates.setdefault(key, _empty_crunch_impact_candidate(policy_id, cohort_id, row, lifecycle))
        _add_crunch_impact_row(candidate, row, crunch, lifecycle)
        observed_rows += 1

    finalized: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for raw in candidates.values():
        cohorts = {
            key: _finalize_crunch_impact_cohort(value)
            for key, value in raw["cohorts"].items()
        }
        applied = cohorts["canary_applied"]
        holdout = cohorts["canary_holdout"]
        fallback = cohorts["fallback"]
        safety = cohorts["safety_stopped"]
        rollback = cohorts["rollback"]
        stale = _crunch_impact_stale(
            raw.get("latest_observed_at"),
            max_age_hours=max(0.0, _as_float(max_evidence_age_hours, DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS)),
        )
        verdict, reasons, top_blocker = _crunch_impact_verdict(
            applied=applied,
            holdout=holdout,
            safety=safety,
            fallback=fallback,
            rollback=rollback,
            stale=stale,
        )
        impact_recommendation, recommended_next_action = _crunch_impact_recommendation(
            verdict=verdict,
            reasons=reasons,
            applied=applied,
            holdout=holdout,
        )
        public_next_action = _crunch_impact_next_action(
            impact_recommendation=impact_recommendation,
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            reason_codes=reasons,
        )
        graduation_decision = _crunch_impact_graduation_decision(
            impact_recommendation=impact_recommendation,
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            reason_codes=reasons,
        )
        durable_next_action = _crunch_impact_durable_next_action(
            impact_recommendation=impact_recommendation,
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            reason_codes=reasons,
        )
        candidate_missing_measurements = _crunch_impact_missing_measurements_for_candidate(
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            saved_tokens=_as_int(applied.get("saved_tokens")),
            saved_usd=_as_float(applied.get("saved_usd")),
        )
        _increment(verdict_counts, verdict)
        if verdict != "widen-ready":
            for reason in reasons:
                _increment(blocker_counts, reason)
        latency_delta = None
        if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
            latency_delta = round(_as_float(applied.get("latency_avg_ms")) - _as_float(holdout.get("latency_avg_ms")), 2)
        observed_count = sum(_as_int(cohort.get("count")) for cohort in cohorts.values())
        coverage = _crunch_impact_coverage(
            applied_count=_as_int(applied.get("count")),
            holdout_count=_as_int(holdout.get("count")),
            skipped_count=_as_int(cohorts["skipped"].get("count")),
            fallback_count=_as_int(fallback.get("count")),
            safety_stop_count=_as_int(safety.get("count")),
            rollback_count=_as_int(rollback.get("count")),
            unknown_count=_as_int(cohorts["unknown"].get("count")),
        )
        captured_savings = _crunch_captured_savings(
            policy_id=raw["policy_id"],
            cohort_id=raw["cohort_id"],
            rule_group=raw.get("rule_group"),
            applied=applied,
            holdout=holdout,
            projected_saved_tokens=_as_int(raw.get("projected_saved_tokens")),
            projected_saved_usd=_as_float(raw.get("projected_saved_usd")),
        )
        finalized.append(
            {
                "schema": "tokenclaw.request_shape_crunch_canary_impact_candidate.v1",
                "policy_id": raw["policy_id"],
                "cohort_id": raw["cohort_id"],
                "rule_group": raw.get("rule_group"),
                "cohort_metadata": raw["cohort_metadata"],
                "policy_source": raw["policy_source"],
                "source_evidence_schema": raw.get("source_evidence_schema"),
                "source_evidence_schemas": raw.get("source_evidence_schemas") or [],
                "staged_at": raw.get("staged_at"),
                "freshly_staged": bool(raw.get("staged_at")),
                "projected_saved_chars": _as_int(raw.get("projected_saved_chars")),
                "projected_saved_tokens": _as_int(raw.get("projected_saved_tokens")),
                "projected_saved_usd": round(_as_float(raw.get("projected_saved_usd")), 8),
                "rollback_metadata_present": bool(raw.get("rollback_metadata_present")),
                "observed_count": observed_count,
                "applied_count": _as_int(applied.get("count")),
                "holdout_count": _as_int(holdout.get("count")),
                "fallback_count": _as_int(fallback.get("count")),
                "safety_stop_count": _as_int(safety.get("count")),
                "rollback_count": _as_int(rollback.get("count")),
                "saved_chars": _as_int(applied.get("saved_chars")),
                "saved_tokens": _as_int(applied.get("saved_tokens")),
                "saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                "estimated_saved_chars": _as_int(applied.get("saved_chars")),
                "estimated_saved_tokens": _as_int(applied.get("saved_tokens")),
                "estimated_saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                "applied_error_count": _as_int(applied.get("error_count")),
                "holdout_error_count": _as_int(holdout.get("error_count")),
                "applied_retry_count": _as_int(applied.get("retry_count")),
                "holdout_retry_count": _as_int(holdout.get("retry_count")),
                "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
                "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
                "latency_avg_delta_ms": latency_delta,
                "fallback_rate_delta": round(
                    (_as_int(fallback.get("count")) / max(1, _as_int(applied.get("count"))))
                    - 0.0,
                    6,
                ),
                "stale_evidence": {
                    "stale": stale,
                    "max_age_hours": round(max(0.0, _as_float(max_evidence_age_hours, DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS)), 6),
                },
                "first_observed_at": raw.get("first_observed_at"),
                "latest_observed_at": raw.get("latest_observed_at"),
                "verdict": verdict,
                "impact_recommendation": impact_recommendation,
                "promotion_recommendation": impact_recommendation,
                "graduation_decision": graduation_decision,
                "recommended_next_action": recommended_next_action,
                "next_action": public_next_action,
                "durable_next_action": durable_next_action,
                "top_blocker": top_blocker if verdict != "widen-ready" else None,
                "reason_codes": reasons,
                "missing_measurements": candidate_missing_measurements,
                "coverage": coverage,
                "applied_vs_holdout_coverage": coverage,
                "captured_savings": captured_savings,
                "captured_saved_tokens": _as_int(captured_savings.get("captured_saved_tokens")),
                "captured_saved_usd": round(_as_float(captured_savings.get("captured_saved_usd")), 8),
                "projection_realization_ratio": captured_savings.get("projection_realization_ratio"),
                "promotion_metadata": {
                    "schema": "tokenclaw.request_shape_crunch_canary_promotion_recommendation.v1",
                    "action_family": "crunch",
                    "local_action_family": "crunch",
                    "target_local_policy": "crunch_rules",
                    "impact_recommendation": impact_recommendation,
                    "graduation_decision": graduation_decision,
                    "recommended_next_action": recommended_next_action,
                    "next_action": public_next_action,
                    "durable_next_action": durable_next_action,
                    "reason_codes": reasons,
                    "missing_measurements": candidate_missing_measurements,
                    "applied_count": _as_int(applied.get("count")),
                    "holdout_count": _as_int(holdout.get("count")),
                    "safety_stop_count": _as_int(safety.get("count")),
                    "fallback_count": _as_int(fallback.get("count")),
                    "rollback_count": _as_int(rollback.get("count")),
                    "observed_saved_tokens": _as_int(applied.get("saved_tokens")),
                    "observed_saved_usd": round(_as_float(applied.get("saved_usd")), 8),
                    "captured_saved_tokens": _as_int(captured_savings.get("captured_saved_tokens")),
                    "captured_saved_usd": round(_as_float(captured_savings.get("captured_saved_usd")), 8),
                    "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
                    "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
                    "latency_avg_delta_ms": latency_delta,
                    "privacy": _crunch_opportunity_privacy(),
                },
                "status_breakdown": _breakdown(raw.get("status_counts", {})),
                "reason_breakdown": _breakdown(raw.get("reason_counts", {})),
                "cohorts": cohorts,
                "privacy": _crunch_opportunity_privacy(),
            }
        )

    finalized.sort(
        key=lambda item: (
            item.get("verdict") == "widen-ready",
            _as_float(item.get("saved_usd")),
            _as_int(item.get("saved_tokens")),
            _as_int(item.get("observed_count")),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(finalized, start=1):
        candidate["rank"] = rank
    blocker_breakdown = _breakdown(blocker_counts)
    status = "no-crunch-canary-impact-metadata"
    if finalized:
        status = "widen-ready" if any(item.get("verdict") == "widen-ready" for item in finalized) else "no-widen"
    recommendation_counts: dict[str, int] = {}
    for item in finalized:
        _increment(recommendation_counts, item.get("impact_recommendation") or "unknown")
    recommendation_breakdown = _breakdown(recommendation_counts)
    top_recommendation = recommendation_breakdown[0]["value"] if recommendation_breakdown else None
    top_next_action = None
    if finalized:
        top_next_action = str(finalized[0].get("next_action") or "")
    top_graduation_decision = str(finalized[0].get("graduation_decision") or "") if finalized else "keep-staged"
    total_applied = sum(_as_int(item.get("applied_count")) for item in finalized)
    total_holdout = sum(_as_int(item.get("holdout_count")) for item in finalized)
    total_skipped = sum(_as_int((item.get("cohorts") or {}).get("skipped", {}).get("count")) for item in finalized)
    total_fallback = sum(_as_int(item.get("fallback_count")) for item in finalized)
    total_safety = sum(_as_int(item.get("safety_stop_count")) for item in finalized)
    total_rollback = sum(_as_int(item.get("rollback_count")) for item in finalized)
    total_unknown = sum(_as_int((item.get("cohorts") or {}).get("unknown", {}).get("count")) for item in finalized)
    cohort_family_actions = [
        {
            "schema": "tokenclaw.request_shape_crunch_canary_cohort_family_action.v1",
            "rank": item.get("rank"),
            "policy_id": item.get("policy_id"),
            "cohort_id": item.get("cohort_id"),
            "cohort_family": "crunch",
            "policy_source": item.get("policy_source"),
            "source_evidence_schema": item.get("source_evidence_schema"),
            "staged_at": item.get("staged_at"),
            "freshly_staged": bool(item.get("freshly_staged")),
            "applied_count": _as_int(item.get("applied_count")),
            "holdout_count": _as_int(item.get("holdout_count")),
            "saved_tokens": _as_int(item.get("saved_tokens")),
            "saved_usd": round(_as_float(item.get("saved_usd")), 8),
            "projected_saved_tokens": _as_int(item.get("projected_saved_tokens")),
            "projected_saved_usd": round(_as_float(item.get("projected_saved_usd")), 8),
            "impact_recommendation": item.get("impact_recommendation"),
            "graduation_decision": item.get("graduation_decision"),
            "durable_next_action": item.get("durable_next_action"),
            "next_action": item.get("next_action"),
            "recommended_next_action": item.get("recommended_next_action"),
            "reason_codes": item.get("reason_codes") or [],
            "missing_measurements": item.get("missing_measurements") or [],
            "coverage": item.get("coverage"),
            "privacy": _crunch_opportunity_privacy(),
        }
        for item in finalized
    ]
    durable_action_counts: dict[str, int] = {}
    for action in cohort_family_actions:
        _increment(durable_action_counts, action.get("durable_next_action") or "unknown")
    coverage = _crunch_impact_coverage(
        applied_count=total_applied,
        holdout_count=total_holdout,
        skipped_count=total_skipped,
        fallback_count=total_fallback,
        safety_stop_count=total_safety,
        rollback_count=total_rollback,
        unknown_count=total_unknown,
    )
    if finalized and total_applied <= 0 and total_safety <= 0 and total_rollback <= 0:
        status = "no-applied-coverage"
    if not finalized:
        status = "no-applied-coverage"
        top_next_action = "stage-canary-first"
    missing_measurements = []
    if total_applied <= 0 or (finalized and total_holdout <= 0) or not finalized:
        missing_measurements.append("missing-applied-or-holdout-coverage")
    if total_applied <= 0:
        missing_measurements.append("applied-crunch-canary-coverage")
    if finalized and total_holdout <= 0:
        missing_measurements.append("holdout-crunch-canary-coverage")
    if not finalized:
        missing_measurements.append("crunch-canary-lifecycle-metadata")
    activation_ready_measurements = _crunch_impact_activation_ready_measurements(
        opportunity_report=opportunity_report,
        impact_candidates=finalized,
    )
    newly_staged_measurement = _crunch_impact_newly_staged_measurement(activation_ready_measurements)
    repeated_context_impact_rows = build_request_shape_crunch_canary_impact_rows_report(
        impact_candidates=finalized,
        activation_ready_measurements=activation_ready_measurements,
    )
    captured_savings_rows = [
        item["captured_savings"]
        for item in finalized
        if isinstance(item.get("captured_savings"), dict)
    ]
    captured_savings = _crunch_captured_savings_summary(captured_savings_rows)
    return {
        "schema": CRUNCH_CANARY_IMPACT_SCHEMA,
        "status": status,
        "ok": True,
        "read_only": True,
        "next_action": top_next_action or "stage-canary-first",
        "graduation_decision": top_graduation_decision,
        "recommended_next_action": str(finalized[0].get("recommended_next_action") or "") if finalized else "stage-repeated-context-crunch-canary",
        "missing_measurements": missing_measurements,
        "summary": {
            "candidate_count": len(finalized),
            "observed_canary_metadata_row_count": observed_rows,
            "applied_count": total_applied,
            "holdout_count": total_holdout,
            "saved_chars": sum(_as_int(item.get("saved_chars")) for item in finalized),
            "saved_tokens": sum(_as_int(item.get("saved_tokens")) for item in finalized),
            "saved_usd": round(sum(_as_float(item.get("saved_usd")) for item in finalized), 8),
            "captured_saved_tokens": captured_savings["captured_saved_tokens"],
            "captured_saved_usd": captured_savings["captured_saved_usd"],
            "captured_savings_status": captured_savings["status"],
            "projection_realization_ratio": captured_savings["projection_realization_ratio"],
            "estimated_saved_chars": sum(_as_int(item.get("estimated_saved_chars")) for item in finalized),
            "estimated_saved_tokens": sum(_as_int(item.get("estimated_saved_tokens")) for item in finalized),
            "estimated_saved_usd": round(sum(_as_float(item.get("estimated_saved_usd")) for item in finalized), 8),
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in finalized),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in finalized),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in finalized), 8),
            "error_rate_delta": round(max((_as_float(item.get("error_rate_delta")) for item in finalized), default=0.0), 6),
            "retry_rate_delta": round(max((_as_float(item.get("retry_rate_delta")) for item in finalized), default=0.0), 6),
            "latency_avg_delta_ms": max(
                (
                    _as_float(item.get("latency_avg_delta_ms"))
                    for item in finalized
                    if item.get("latency_avg_delta_ms") is not None
                ),
                default=None,
            ),
            "fallback_count": total_fallback,
            "safety_stop_count": total_safety,
            "rollback_count": total_rollback,
            "widen_ready_count": sum(1 for item in finalized if item.get("verdict") == "widen-ready"),
            "no_widen_count": sum(1 for item in finalized if item.get("verdict") != "widen-ready"),
            "promotion_ready_count": sum(1 for item in finalized if item.get("impact_recommendation") == "promotion-ready"),
            "rollback_recommended_count": sum(1 for item in finalized if item.get("impact_recommendation") == "rollback"),
            "keep_blocked_count": sum(1 for item in finalized if item.get("impact_recommendation") == "keep-blocked"),
            "collect_more_evidence_count": sum(
                1 for item in finalized if item.get("impact_recommendation") == "collect-more-evidence"
            ),
            "top_impact_recommendation": top_recommendation,
            "top_blocker_code": blocker_breakdown[0]["value"] if blocker_breakdown else None,
            "next_action": top_next_action or "stage-canary-first",
            "top_next_action": top_next_action or "stage-canary-first",
            "graduation_decision": top_graduation_decision,
            "top_graduation_decision": top_graduation_decision,
            "recommended_next_action": str(finalized[0].get("recommended_next_action") or "") if finalized else "stage-repeated-context-crunch-canary",
            "top_durable_next_action": str(finalized[0].get("durable_next_action") or "") if finalized else "keep-staged",
            "durable_next_action_breakdown": _breakdown(durable_action_counts),
            "cohort_family_action_count": len(cohort_family_actions),
            "freshly_staged_cohort_count": sum(1 for item in finalized if item.get("freshly_staged")),
            "coverage": coverage,
            "applied_vs_holdout_coverage": coverage,
            "cohort_counts": {
                "canary_applied": total_applied,
                "canary_holdout": total_holdout,
                "skipped": total_skipped,
                "fallback": total_fallback,
                "safety_stopped": total_safety,
                "rollback": total_rollback,
                "unknown": total_unknown,
            },
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
            "activation_ready_cohort_count": activation_ready_measurements["cohort_count"],
            "measured_cohort_count": activation_ready_measurements["measured_count"],
            "keep_staged_cohort_count": activation_ready_measurements["keep_staged_count"],
            "stageable_cohort_count": activation_ready_measurements["stageable_count"],
            "blocked_cohort_count": activation_ready_measurements["blocked_count"],
            "bounded_stage_recommendation_count": activation_ready_measurements["bounded_stage_recommendation_count"],
            "newly_staged_measurement_cohort_count": newly_staged_measurement["cohort_count"],
            "newly_staged_measured_cohort_count": newly_staged_measurement["measured_cohort_count"],
            "newly_staged_applied_count": newly_staged_measurement["applied_count"],
            "newly_staged_holdout_count": newly_staged_measurement["holdout_count"],
            "newly_staged_skipped_count": newly_staged_measurement["skipped_count"],
            "newly_staged_error_count": newly_staged_measurement["error_count"],
            "newly_staged_retry_count": newly_staged_measurement["retry_count"],
            "newly_staged_fallback_count": newly_staged_measurement["fallback_count"],
            "newly_staged_rollback_count": newly_staged_measurement["rollback_count"],
            "newly_staged_safety_stop_count": newly_staged_measurement["safety_stop_count"],
            "active_max_rollout_suppressed_cohort_count": newly_staged_measurement["active_max_rollout_suppressed_count"],
            "repeated_context_impact_row_count": repeated_context_impact_rows["summary"]["ranked_row_count"],
            "repeated_context_measured_count": repeated_context_impact_rows["summary"]["measured_count"],
            "repeated_context_measurement_required_count": repeated_context_impact_rows["summary"]["measurement_required_count"],
            "repeated_context_blocked_count": repeated_context_impact_rows["summary"]["blocked_count"],
            "repeated_context_superseded_count": repeated_context_impact_rows["summary"]["superseded_count"],
        },
        "captured_savings": captured_savings,
        "captured_savings_rows": captured_savings_rows,
        "verdict_breakdown": _breakdown(verdict_counts),
        "impact_recommendation_breakdown": recommendation_breakdown,
        "durable_next_action_breakdown": _breakdown(durable_action_counts),
        "blocker_reason_breakdown": blocker_breakdown,
        "cohort_family_actions": cohort_family_actions,
        "activation_ready_measurements": activation_ready_measurements,
        "newly_staged_measurement": newly_staged_measurement,
        "repeated_context_impact_rows": repeated_context_impact_rows,
        "candidates": finalized,
        "activation_lifecycle_feedback": _crunch_impact_activation_lifecycle_feedback(finalized),
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_policy_decision_id(candidate: dict[str, Any], decision: str) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "schema": CRUNCH_POLICY_DECISION_SCHEMA,
                "policy_id": candidate.get("policy_id"),
                "cohort_id": candidate.get("cohort_id"),
                "decision": decision,
                "reason_codes": candidate.get("reason_codes") or [],
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"request-shape-crunch-policy-decision:{digest}"


def _crunch_policy_decision_value(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "blocked"
    recommendation = str(candidate.get("impact_recommendation") or candidate.get("promotion_recommendation") or "")
    if recommendation == "promotion-ready":
        return "widen"
    if recommendation == "rollback":
        return "rollback"
    if recommendation == "collect-more-evidence":
        return "keep-staged"
    return "blocked"


def _crunch_policy_promotion_decision(decision: str) -> str:
    return {
        "widen": "promote",
        "rollback": "rollback",
        "keep-staged": "keep-staged",
        "blocked": "keep-blocked",
    }.get(decision, "keep-blocked")


def _crunch_policy_promotion_readiness(decision: str) -> str:
    return {
        "widen": "promotion-ready",
        "rollback": "rollback-required",
        "keep-staged": "needs-more-evidence",
        "blocked": "keep-blocked",
    }.get(decision, "keep-blocked")


def _crunch_policy_decision_reason(candidate: dict[str, Any] | None, decision: str) -> str:
    if not candidate:
        return "missing-applied-or-holdout-coverage"
    reasons = candidate.get("reason_codes") if isinstance(candidate.get("reason_codes"), list) else []
    if reasons:
        return public_label(reasons[0], "unknown")
    if decision == "widen":
        return "applied-savings-with-holdout-no-regression"
    if decision == "rollback":
        return "rollback-recommended"
    if decision == "keep-staged":
        return "collect-more-canary-impact-evidence"
    return public_label(candidate.get("top_blocker") or "blocked", "blocked")


def _crunch_policy_graduation_decision(decision: str, reason_codes: list[str]) -> str:
    if decision in {"widen", "rollback", "keep-staged", "blocked"}:
        return decision
    if any(reason in {"missing-applied-or-holdout-coverage", "missing-applied-coverage", "missing-holdout-coverage"} for reason in reason_codes):
        return "keep-staged"
    return "blocked"


def _crunch_policy_decision_rollback_metadata(candidate: dict[str, Any] | None, decision: str) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.request_shape_crunch_policy_decision_rollback_metadata.v1",
        "rollback_action_type": "disable_repeated_context_crunch_canary",
        "target_policy_id": candidate.get("policy_id") if candidate else None,
        "target_cohort_id": candidate.get("cohort_id") if candidate else None,
        "target_local_rule_file": "crunch_rules.yaml",
        "rollback_reason_codes": [
            "safety-stop-observed",
            "error-rate-regression",
            "retry-rate-regression",
            "fallback-observed",
            "rollback-observed",
            "operator-requested",
        ],
        "required_for_promotion": True,
        "present": bool(candidate),
        "selected_decision": decision,
        "policy_files_written": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_policy_decision_metrics(candidate: dict[str, Any] | None) -> dict[str, Any]:
    coverage = (
        candidate.get("coverage")
        if isinstance(candidate, dict) and isinstance(candidate.get("coverage"), dict)
        else _crunch_impact_coverage(applied_count=0, holdout_count=0)
    )
    cohorts = candidate.get("cohorts") if isinstance(candidate, dict) and isinstance(candidate.get("cohorts"), dict) else {}
    applied = cohorts.get("canary_applied") if isinstance(cohorts.get("canary_applied"), dict) else {}
    holdout = cohorts.get("canary_holdout") if isinstance(cohorts.get("canary_holdout"), dict) else {}
    return {
        "schema": "tokenclaw.request_shape_crunch_policy_decision_metrics.v1",
        "coverage": coverage,
        "applied_count": _as_int(candidate.get("applied_count")) if candidate else 0,
        "holdout_count": _as_int(candidate.get("holdout_count")) if candidate else 0,
        "observed_saved_chars": _as_int(candidate.get("saved_chars")) if candidate else 0,
        "observed_saved_tokens": _as_int(candidate.get("saved_tokens")) if candidate else 0,
        "observed_saved_usd": round(_as_float(candidate.get("saved_usd")) if candidate else 0.0, 8),
        "captured_saved_tokens": _as_int(candidate.get("captured_saved_tokens")) if candidate else 0,
        "captured_saved_usd": round(_as_float(candidate.get("captured_saved_usd")) if candidate else 0.0, 8),
        "projected_saved_tokens": _as_int(candidate.get("projected_saved_tokens")) if candidate else 0,
        "projected_saved_usd": round(_as_float(candidate.get("projected_saved_usd")) if candidate else 0.0, 8),
        "projection_realization_ratio": candidate.get("projection_realization_ratio") if candidate else None,
        "applied_error_count": _as_int(candidate.get("applied_error_count")) if candidate else 0,
        "holdout_error_count": _as_int(candidate.get("holdout_error_count")) if candidate else 0,
        "applied_retry_count": _as_int(candidate.get("applied_retry_count")) if candidate else 0,
        "holdout_retry_count": _as_int(candidate.get("holdout_retry_count")) if candidate else 0,
        "fallback_count": _as_int(candidate.get("fallback_count")) if candidate else 0,
        "safety_stop_count": _as_int(candidate.get("safety_stop_count")) if candidate else 0,
        "rollback_count": _as_int(candidate.get("rollback_count")) if candidate else 0,
        "error_rate_delta": round(_as_float(candidate.get("error_rate_delta")) if candidate else 0.0, 6),
        "retry_rate_delta": round(_as_float(candidate.get("retry_rate_delta")) if candidate else 0.0, 6),
        "fallback_rate_delta": round(_as_float(candidate.get("fallback_rate_delta")) if candidate else 0.0, 6),
        "latency_avg_delta_ms": candidate.get("latency_avg_delta_ms") if candidate else None,
        "applied_error_rate": _as_float(applied.get("error_rate")),
        "holdout_error_rate": _as_float(holdout.get("error_rate")),
        "applied_retry_rate": _as_float(applied.get("retry_rate")),
        "holdout_retry_rate": _as_float(holdout.get("retry_rate")),
        "aggregate_only": True,
        "metadata_only": True,
    }


def _crunch_policy_decision_patch(candidate: dict[str, Any] | None, decision: str) -> dict[str, Any] | None:
    if not candidate:
        return None
    patch_type = {
        "widen": "widen_repeated_context_crunch_canary",
        "rollback": "rollback_repeated_context_crunch_canary",
        "keep-staged": "keep_repeated_context_crunch_canary_staged",
        "blocked": "keep_repeated_context_crunch_canary_blocked",
    }[decision]
    return {
        "schema": "tokenclaw.request_shape_crunch_policy_decision_local_patch.v1",
        "status": "drafted" if decision == "widen" else "not-written",
        "patch_type": patch_type,
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "target_policy_id": candidate.get("policy_id"),
        "target_cohort_id": candidate.get("cohort_id"),
        "policy_source": "local-manual" if decision == "promote" else public_label(candidate.get("policy_source") or "local-manual", "local-manual"),
        "policy_files_written": False,
        "requires_operator_apply": True,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_policy_decision_duplicate_suppression(
    candidate: dict[str, Any] | None,
    decision: str,
    promotion_decision: str,
) -> dict[str, Any]:
    has_candidate = bool(candidate and (candidate.get("policy_id") or candidate.get("cohort_id")))
    suppresses_activation = bool(has_candidate)
    return {
        "schema": "tokenclaw.request_shape_crunch_policy_decision_duplicate_suppression.v1",
        "suppresses_new_activation_issue": suppresses_activation,
        "suppresses_generic_crunch_activation_issue": suppresses_activation,
        "suppresses_generic_crunch_promotion_issue": suppresses_activation,
        "reason": "durable-repeated-context-crunch-policy-decision" if suppresses_activation else "missing-canary-candidate",
        "decision": decision,
        "promotion_decision": promotion_decision,
        "matching_local_policy": "crunch_rules" if suppresses_activation else None,
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "policy_id_included": False,
        "cohort_id_included": False,
        "fingerprint": (
            "activation:"
            + hashlib.sha256(
                stable_json(
                    {
                        "schema": "tokenclaw.request_shape_crunch_policy_decision_duplicate_suppression.v1",
                        "policy_id": candidate.get("policy_id") if candidate else None,
                        "cohort_id": candidate.get("cohort_id") if candidate else None,
                        "decision": decision,
                        "promotion_decision": promotion_decision,
                    }
                ).encode("utf-8")
            ).hexdigest()[:16]
        ),
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_policy_decision_from_candidate(candidate: dict[str, Any] | None) -> dict[str, Any]:
    decision = _crunch_policy_decision_value(candidate)
    promotion_decision = _crunch_policy_promotion_decision(decision)
    reason = _crunch_policy_decision_reason(candidate, decision)
    rollback_metadata = _crunch_policy_decision_rollback_metadata(candidate, decision)
    metrics = _crunch_policy_decision_metrics(candidate)
    promotion_allowed = (
        decision == "widen"
        and metrics["applied_count"] > 0
        and metrics["holdout_count"] > 0
        and metrics["observed_saved_tokens"] > 0
        and rollback_metadata["present"]
    )
    if candidate is None:
        candidate = {}
    decision_id = _crunch_policy_decision_id(candidate, decision)
    duplicate_suppression = _crunch_policy_decision_duplicate_suppression(candidate, decision, promotion_decision)
    return {
        "schema": "tokenclaw.request_shape_crunch_policy_decision_entry.v1",
        "decision_id": decision_id,
        "decision": decision,
        "promotion_decision": promotion_decision,
        "promotion_readiness": _crunch_policy_promotion_readiness(decision),
        "graduation_decision": _crunch_policy_graduation_decision(decision, candidate.get("reason_codes") or [reason]),
        "status": "decided",
        "reason": reason,
        "reason_codes": candidate.get("reason_codes") or [reason],
        "action_family": "crunch",
        "local_action_family": "crunch",
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "policy_source": public_label(candidate.get("policy_source") or "local-manual", "local-manual"),
        "decision_options": ["widen", "rollback", "keep-staged", "blocked"],
        "promotion_decision_options": ["promote", "keep-staged", "rollback", "keep-blocked"],
        "policy_id": candidate.get("policy_id"),
        "cohort_id": candidate.get("cohort_id"),
        "promotion_allowed": promotion_allowed,
        "rollback_required": decision == "rollback",
        "keep_staged": decision == "keep-staged",
        "keep_blocked": decision == "blocked",
        "metrics": metrics,
        "coverage": metrics["coverage"],
        "observed_saved_tokens": metrics["observed_saved_tokens"],
        "observed_saved_usd": metrics["observed_saved_usd"],
        "captured_saved_tokens": metrics["captured_saved_tokens"],
        "captured_saved_usd": metrics["captured_saved_usd"],
        "projected_saved_tokens": metrics["projected_saved_tokens"],
        "projected_saved_usd": metrics["projected_saved_usd"],
        "projection_realization_ratio": metrics["projection_realization_ratio"],
        "error_rate_delta": metrics["error_rate_delta"],
        "retry_rate_delta": metrics["retry_rate_delta"],
        "fallback_rate_delta": metrics["fallback_rate_delta"],
        "safety_stop_state": "observed" if metrics["safety_stop_count"] > 0 else "none",
        "local_policy_patch": _crunch_policy_decision_patch(candidate, decision),
        "rollback_metadata": rollback_metadata,
        "duplicate_suppression": duplicate_suppression,
        "source_candidate_schema": candidate.get("schema"),
        "source_impact_recommendation": candidate.get("impact_recommendation"),
        "source_recommended_next_action": candidate.get("recommended_next_action"),
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_policy_decision_report(impact_report: dict[str, Any]) -> dict[str, Any]:
    candidates = impact_report.get("candidates") if isinstance(impact_report, dict) else []
    if not isinstance(candidates, list):
        candidates = []
    decision_entries = [
        _crunch_policy_decision_from_candidate(candidate)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    if not decision_entries:
        decision_entries = [_crunch_policy_decision_from_candidate(None)]

    decision_entries.sort(
        key=lambda item: (
            item["decision"] == "widen",
            item["decision"] == "rollback",
            item["decision"] == "keep-staged",
            _as_float(item.get("observed_saved_usd")),
            _as_int(item.get("observed_saved_tokens")),
        ),
        reverse=True,
    )
    top = decision_entries[0]
    decision_counts: dict[str, int] = {}
    for item in decision_entries:
        _increment(decision_counts, item.get("decision"))
    metrics = top["metrics"] if isinstance(top.get("metrics"), dict) else _crunch_policy_decision_metrics(None)
    report = {
        "schema": CRUNCH_POLICY_DECISION_SCHEMA,
        "ok": True,
        "status": "decided",
        "read_only": True,
        "generated_at": utc_now(),
        "decision": top["decision"],
        "promotion_decision": top["promotion_decision"],
        "promotion_readiness": top["promotion_readiness"],
        "graduation_decision": top["graduation_decision"],
        "decision_id": top["decision_id"],
        "top_decision": top,
        "decisions": decision_entries,
        "duplicate_suppression": top.get("duplicate_suppression"),
        "summary": {
            "decision": top["decision"],
            "promotion_decision": top["promotion_decision"],
            "promotion_readiness": top["promotion_readiness"],
            "graduation_decision": top["graduation_decision"],
            "decision_id": top["decision_id"],
            "decision_count": len(decision_entries),
            "decision_breakdown": _breakdown(decision_counts),
            "promotion_allowed": bool(top.get("promotion_allowed")),
            "rollback_required": bool(top.get("rollback_required")),
            "keep_staged": bool(top.get("keep_staged")),
            "keep_blocked": bool(top.get("keep_blocked")),
            "applied_count": metrics["applied_count"],
            "holdout_count": metrics["holdout_count"],
            "observed_saved_tokens": metrics["observed_saved_tokens"],
            "observed_saved_usd": metrics["observed_saved_usd"],
            "captured_saved_tokens": metrics["captured_saved_tokens"],
            "captured_saved_usd": metrics["captured_saved_usd"],
            "projected_saved_tokens": metrics["projected_saved_tokens"],
            "projected_saved_usd": metrics["projected_saved_usd"],
            "projection_realization_ratio": metrics["projection_realization_ratio"],
            "error_rate_delta": metrics["error_rate_delta"],
            "retry_rate_delta": metrics["retry_rate_delta"],
            "fallback_rate_delta": metrics["fallback_rate_delta"],
            "safety_stop_state": top["safety_stop_state"],
            "policy_source": top["policy_source"],
            "target_local_rule_file": "crunch_rules.yaml",
            "target_local_policy_section": "crunch.rules",
            "coverage": top["coverage"],
            "duplicate_activation_issue_suppressed": bool(
                (top.get("duplicate_suppression") or {}).get("suppresses_new_activation_issue")
            )
            if isinstance(top.get("duplicate_suppression"), dict)
            else False,
            "source_impact_status": impact_report.get("status") if isinstance(impact_report, dict) else None,
            "source_impact_recommendation": top.get("source_impact_recommendation"),
            "policy_files_written": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
        },
        "source_report": {
            "schema": impact_report.get("schema") if isinstance(impact_report, dict) else None,
            "status": impact_report.get("status") if isinstance(impact_report, dict) else None,
            "summary": impact_report.get("summary") if isinstance(impact_report.get("summary"), dict) else {},
        },
        "privacy": _crunch_opportunity_privacy(),
    }
    report["ledger_update"] = build_request_shape_crunch_policy_decision_ledger(report)
    return report


def _crunch_policy_decision_ledger_status(top: dict[str, Any]) -> tuple[str, bool]:
    decision = str(top.get("decision") or "")
    reason_codes = set(str(item) for item in (top.get("reason_codes") or []))
    if decision == "widen":
        return "positive", False
    if decision == "rollback":
        return "rollback-needed", True
    if decision == "keep-staged":
        return "needs-more-samples", False
    if reason_codes & {"missing-applied-or-holdout-coverage", "missing-applied-coverage", "missing-holdout-coverage"}:
        return "needs-more-samples", False
    if reason_codes & {"no-applied-savings", "stale-canary-impact-evidence"}:
        return "needs-review", False
    return "regression-flagged" if reason_codes else "needs-review", False


def build_request_shape_crunch_policy_decision_ledger(
    decision_report: dict[str, Any],
    *,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    recorded = recorded_at or str(decision_report.get("generated_at") or utc_now())
    top = decision_report.get("top_decision") if isinstance(decision_report.get("top_decision"), dict) else {}
    metrics = top.get("metrics") if isinstance(top.get("metrics"), dict) else {}
    status, rollback_needed = _crunch_policy_decision_ledger_status(top)
    policy_id = str(top.get("policy_id") or "unknown")
    decision_id = str(top.get("decision_id") or "")
    reason_codes = [public_label(reason, "unknown") for reason in (top.get("reason_codes") or [])]
    entry_id = "request-shape-crunch-decision:" + hashlib.sha256(
        stable_json(
            {
                "schema": CRUNCH_POLICY_DECISION_LEDGER_ENTRY_SCHEMA,
                "recorded_at": recorded,
                "decision_id": decision_id,
                "policy_id": policy_id,
                "decision": top.get("decision"),
                "reason_codes": reason_codes,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    entry = {
        "schema": CRUNCH_POLICY_DECISION_LEDGER_ENTRY_SCHEMA,
        "id": entry_id,
        "created_at": recorded,
        "impact_generated_at": decision_report.get("generated_at"),
        "policy_id": policy_id,
        "action_family": "crunch",
        "policy_section": "crunch.rules",
        "rule_source": public_label(top.get("policy_source") or "local-manual", "local-manual"),
        "rule_id": top.get("policy_id"),
        "candidate_id": top.get("cohort_id"),
        "action_id": decision_id,
        "source_evidence_schema": CRUNCH_POLICY_DECISION_SCHEMA,
        "status": status,
        "recommendation": top.get("graduation_decision") or decision_report.get("graduation_decision"),
        "rollback_needed": rollback_needed,
        "reason_codes": reason_codes,
        "observed_savings_usd": round(_as_float(metrics.get("observed_saved_usd")), 8),
        "projected_savings_usd": round(_as_float(metrics.get("observed_saved_usd")), 8),
        "projection_realization_ratio": 1.0 if _as_float(metrics.get("observed_saved_usd")) > 0 else None,
        "applied_count": _as_int(metrics.get("applied_count")),
        "holdout_count": _as_int(metrics.get("holdout_count")),
        "skipped_count": _as_int((metrics.get("coverage") or {}).get("skipped_count")) if isinstance(metrics.get("coverage"), dict) else 0,
        "bypassed_count": 0,
        "safety_stop_count": _as_int(metrics.get("safety_stop_count")),
        "error_rate_delta": round(_as_float(metrics.get("error_rate_delta")), 6),
        "retry_rate_delta": round(_as_float(metrics.get("retry_rate_delta")), 6),
        "latency_delta_ms": metrics.get("latency_avg_delta_ms"),
        "cohort_metrics": {
            "coverage": metrics.get("coverage") if isinstance(metrics.get("coverage"), dict) else {},
            "observed_saved_tokens": _as_int(metrics.get("observed_saved_tokens")),
            "observed_saved_usd": round(_as_float(metrics.get("observed_saved_usd")), 8),
            "safety_stop_state": top.get("safety_stop_state"),
            "graduation_decision": top.get("graduation_decision") or decision_report.get("graduation_decision"),
        },
        "privacy": _crunch_opportunity_privacy(),
    }
    entry = {key: value for key, value in entry.items() if value not in (None, "", [], {})}
    return {
        "schema": CRUNCH_POLICY_DECISION_LEDGER_SCHEMA,
        "ok": True,
        "status": "recordable",
        "append_only": True,
        "wrote_store": False,
        "entry_count": 1,
        "entries": [entry],
        "summary": {
            "entry_count": 1,
            "rows_written": 0,
            "status": status,
            "rollback_needed_count": 1 if rollback_needed else 0,
            "observed_savings_usd": entry.get("observed_savings_usd", 0.0),
            "applied_count": entry.get("applied_count", 0),
            "holdout_count": entry.get("holdout_count", 0),
        },
        "privacy": _crunch_opportunity_privacy(),
    }


def record_request_shape_crunch_policy_decision_ledger(
    decision_report: dict[str, Any],
    *,
    store_obj: Any,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    ledger = build_request_shape_crunch_policy_decision_ledger(decision_report, recorded_at=recorded_at)
    rows_written = 0
    for entry in ledger.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        store_obj.log_promotion_outcome_feedback(
            id=entry.get("id"),
            created_at=entry.get("created_at"),
            impact_generated_at=entry.get("impact_generated_at"),
            policy_id=entry.get("policy_id"),
            action_family=entry.get("action_family"),
            policy_section=entry.get("policy_section"),
            rule_source=entry.get("rule_source"),
            rule_id=entry.get("rule_id"),
            candidate_id=entry.get("candidate_id"),
            action_id=entry.get("action_id"),
            source_evidence_schema=entry.get("source_evidence_schema"),
            status=entry.get("status"),
            recommendation=entry.get("recommendation"),
            rollback_needed=1 if entry.get("rollback_needed") else 0,
            observed_savings_usd=entry.get("observed_savings_usd"),
            projected_savings_usd=entry.get("projected_savings_usd"),
            projection_realization_ratio=entry.get("projection_realization_ratio"),
            applied_count=entry.get("applied_count"),
            holdout_count=entry.get("holdout_count"),
            skipped_count=entry.get("skipped_count"),
            bypassed_count=entry.get("bypassed_count"),
            safety_stop_count=entry.get("safety_stop_count"),
            error_rate_delta=entry.get("error_rate_delta"),
            retry_rate_delta=entry.get("retry_rate_delta"),
            latency_delta_ms=entry.get("latency_delta_ms"),
            feedback_json=stable_json(entry),
        )
        rows_written += 1
    ledger["wrote_store"] = rows_written > 0
    ledger["status"] = "recorded" if rows_written else ledger.get("status")
    ledger["summary"]["rows_written"] = rows_written
    return ledger


def _new_group(basis: dict[str, Any], *, candidate_id: str, rollup_key: str) -> dict[str, Any]:
    return {
        "schema": ROLLUP_ROW_SCHEMA,
        "rollup_key": rollup_key,
        "candidate_id": candidate_id,
        **basis,
        "row_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "cache_hit_count": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "observed_savings_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "successful_input_tokens": 0,
        "input_token_cost_usd": 0.0,
        "current_crunch_tokens_saved": 0,
        "current_crunch_chars_saved": 0,
        "current_crunch_savings_usd": 0.0,
        "candidate_family_counts": {},
        "blocker_counts": {},
        "status_counts": {},
        "retry_bucket_counts": {},
        "cost_bucket_counts": {},
        "savings_bucket_counts": {},
        "cache_reason_counts": {},
        "file_dependency_status_counts": {},
        "file_dependency_fingerprint_availability_counts": {},
        "file_dependency_audit": None,
        "crunch_canary_lifecycle_counts": {},
        "crunch_canary_policy_counts": {},
    }


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    candidate_family_counts = group.pop("candidate_family_counts", {})
    blocker_counts = group.pop("blocker_counts", {})
    file_dependency_status_counts = group.pop("file_dependency_status_counts", {})
    file_dependency_fingerprint_counts = group.pop("file_dependency_fingerprint_availability_counts", {})
    file_dependency_audit = group.pop("file_dependency_audit", None)
    crunch_canary_lifecycle_counts = group.pop("crunch_canary_lifecycle_counts", {})
    crunch_canary_policy_counts = group.pop("crunch_canary_policy_counts", {})
    candidate_families = sorted(candidate_family_counts)
    blocker_codes = sorted(blocker_counts)
    candidate_classes = _candidate_work_classes(
        row_count=_as_int(group.get("row_count")),
        text_bucket=str(group.get("text_bucket") or "unknown"),
        token_bucket=str(group.get("token_bucket") or "unknown"),
        candidate_families=candidate_families,
        blocker_codes=blocker_codes,
        routing_status=str(group.get("routing_status") or "unknown"),
        observed_savings=_as_float(group.get("observed_savings_usd")),
    )
    metadata = {
        "schema": "tokenclaw.request_shape_rollup_metadata.v1",
        "status_breakdown": _breakdown(group.pop("status_counts", {})),
        "retry_bucket_breakdown": _breakdown(group.pop("retry_bucket_counts", {})),
        "cost_bucket_breakdown": _breakdown(group.pop("cost_bucket_counts", {})),
        "savings_bucket_breakdown": _breakdown(group.pop("savings_bucket_counts", {})),
        "cache_reason_breakdown": _breakdown(group.pop("cache_reason_counts", {})),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "file_dependency_status_breakdown": _breakdown(file_dependency_status_counts),
        "file_dependency_fingerprint_availability_breakdown": _breakdown(file_dependency_fingerprint_counts),
        "crunch_canary_lifecycle_breakdown": _breakdown(crunch_canary_lifecycle_counts),
        "crunch_canary_policy_breakdown": _breakdown(crunch_canary_policy_counts),
        "candidate_class_breakdown": [{"value": value, "count": _as_int(group.get("row_count"))} for value in candidate_classes],
        "raw_body_required": False,
        "aggregate_only": True,
    }
    group["candidate_families"] = candidate_families
    group["candidate_work_classes"] = candidate_classes
    group["blocker_codes"] = blocker_codes
    group["file_dependency_audit"] = file_dependency_audit
    group["file_dependency_status_breakdown"] = metadata["file_dependency_status_breakdown"]
    group["file_dependency_fingerprint_availability_breakdown"] = metadata[
        "file_dependency_fingerprint_availability_breakdown"
    ]
    group["cost_est_usd"] = round(_as_float(group.get("cost_est_usd")), 6)
    group["baseline_cost_usd"] = round(_as_float(group.get("baseline_cost_usd")), 6)
    group["observed_savings_usd"] = round(_as_float(group.get("observed_savings_usd")), 6)
    group["input_token_cost_usd"] = round(_as_float(group.get("input_token_cost_usd")), 6)
    group["current_crunch_savings_usd"] = round(_as_float(group.get("current_crunch_savings_usd")), 6)
    repeated_weight = 0.0
    row_count = _as_int(group.get("row_count"))
    if row_count > 1:
        repeated_weight = (row_count - 1) / float(row_count)
    if "repeated_context" in candidate_classes and "crunch" in candidate_classes:
        projected_tokens = int(
            _as_int(group.get("successful_input_tokens"))
            * REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE
            * repeated_weight
        )
        projected_savings = (
            _as_float(group.get("input_token_cost_usd"))
            * REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE
            * repeated_weight
        )
    else:
        projected_tokens = 0
        projected_savings = 0.0
    group["projected_crunch_tokens_saved"] = max(0, projected_tokens)
    group["projected_crunch_chars_saved"] = max(0, projected_tokens * 4)
    group["projected_crunch_savings_usd"] = round(max(0.0, projected_savings), 6)
    group["crunch_canary_lifecycle"] = {
        "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
        "cohort_id": _crunch_canary_cohort_id(group),
        "policy_id": _crunch_canary_policy_id(_crunch_canary_cohort_id(group)),
        "applied_count": _as_int(crunch_canary_lifecycle_counts.get("applied"))
        + _as_int(crunch_canary_lifecycle_counts.get("canary_applied")),
        "holdout_count": _as_int(crunch_canary_lifecycle_counts.get("holdout"))
        + _as_int(crunch_canary_lifecycle_counts.get("canary_holdout")),
        "skipped_count": _as_int(crunch_canary_lifecycle_counts.get("skipped")),
        "safety_stopped_count": _as_int(crunch_canary_lifecycle_counts.get("safety-stopped"))
        + _as_int(crunch_canary_lifecycle_counts.get("safety_stop")),
        "fallback_count": _as_int(crunch_canary_lifecycle_counts.get("fallback")),
        "rollback_count": _as_int(crunch_canary_lifecycle_counts.get("rollback")),
        "status_breakdown": _breakdown(crunch_canary_lifecycle_counts),
        "policy_breakdown": _breakdown(crunch_canary_policy_counts),
        "metadata_only": True,
        "aggregate_only": True,
    }
    group["metadata"] = metadata
    group["privacy"] = {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
    }
    return group


def _rollup_source_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _request_shape_rollup_source_declaration(
    *,
    source: str,
    status: str,
    rows_considered: int,
    rollups: list[dict[str, Any]],
    acquisition_reason: str | None = None,
) -> dict[str, Any]:
    source_surface_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    for row in rollups:
        if not isinstance(row, dict):
            continue
        row_count = _as_int(row.get("row_count") or row.get("sample_count") or row.get("count"))
        _increment(source_surface_counts, row.get("source_surface"), row_count)
        _increment(provider_counts, row.get("provider_family"), row_count)
        _increment(endpoint_counts, row.get("endpoint"), row_count)
    return {
        "schema": ROLLUP_SOURCE_DECLARATION_SCHEMA,
        "source": public_label(source, "unknown"),
        "status": public_label(status, "unknown"),
        "reason": public_label(acquisition_reason, None),
        "rows_considered": max(0, _as_int(rows_considered)),
        "rollup_count": len([row for row in rollups if isinstance(row, dict)]),
        "source_surface_breakdown": _breakdown(source_surface_counts),
        "provider_breakdown": _breakdown(provider_counts),
        "endpoint_breakdown": _breakdown(endpoint_counts),
        "from_local_metadata": source in {
            "recent-local-call-metadata",
            "recent-local-metadata-window-backfill",
            "dashboard-routing-candidates",
        },
        "read_only": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
        "privacy": _rollup_source_privacy(),
    }


def _replayability_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _crunch_opportunity_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _crunch_rule_candidate_paths(rules_path: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if rules_path:
        candidates.append(Path(rules_path))
    env_path = os.getenv("TOKENCLAW_CRUNCH_RULES")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / "crunch_rules.yaml")
    candidates.append(tokenclaw_config_path("crunch_rules.yaml"))
    candidates.append(Path(__file__).parent / "crunch_rules.yaml")
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.expanduser())
    return deduped


def _active_request_shape_crunch_rules(rules_path: str | Path | None = None) -> list[dict[str, Any]]:
    loaded: dict[str, Any] | None = None
    for path in _crunch_rule_candidate_paths(rules_path):
        if not path.exists():
            continue
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(value, dict):
            loaded = value
            break
    if loaded is None:
        return []

    section = loaded.get("request_shape_repeated_context_canaries")
    raw_rules = section.get("rules") if isinstance(section, dict) and isinstance(section.get("rules"), list) else []
    rules: list[dict[str, Any]] = []
    for item in raw_rules:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        decision = item.get("policy_decision") if isinstance(item.get("policy_decision"), dict) else {}
        if decision.get("schema") != "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1":
            continue
        rollout = item.get("rollout") if isinstance(item.get("rollout"), dict) else {}
        safety_gates = item.get("safety_gates") if isinstance(item.get("safety_gates"), dict) else {}
        rollback_metadata = item.get("rollback_metadata") if isinstance(item.get("rollback_metadata"), dict) else {}
        decision_value = public_label(decision.get("decision") or "unknown", "unknown")
        full_rollout_fraction = _as_float(rollout.get("full_rollout_fraction"))
        full_rollout_active = (
            bool(rollout.get("full_rollout_enabled"))
            or full_rollout_fraction > 0.0
            or decision_value == "promote-full"
        )
        canary_fraction = _as_float(decision.get("widened_canary_fraction") or rollout.get("canary_fraction"))
        if full_rollout_active:
            full_rollout_fraction = full_rollout_fraction or 1.0
            canary_fraction = max(canary_fraction, full_rollout_fraction)
        rules.append(
            {
                "rank": len(rules) + 1,
                "rule_ref": public_id(item.get("id") or f"request-shape-crunch-rule-{len(rules) + 1}", prefix="rule"),
                "policy_source": public_label(item.get("policy_source") or "local-manual", "local-manual"),
                "decision": decision_value,
                "graduation_decision": public_label(decision.get("graduation_decision") or decision_value, "unknown"),
                "decision_id": public_label(decision.get("decision_id") or "unknown", "unknown"),
                "source_evidence_schema": public_label(decision.get("source_evidence_schema") or "unknown", "unknown"),
                "applied_count": _as_int(decision.get("applied_count")),
                "holdout_count": _as_int(decision.get("holdout_count")),
                "skipped_count": _as_int(decision.get("skipped_count")),
                "blocked_count": _as_int(decision.get("blocked_count")),
                "observed_saved_tokens": _as_int(decision.get("observed_saved_tokens")),
                "observed_saved_usd": round(_as_float(decision.get("observed_saved_usd")), 8),
                "error_rate_delta": round(_as_float(decision.get("error_rate_delta")), 6),
                "retry_rate_delta": round(_as_float(decision.get("retry_rate_delta")), 6),
                "fallback_rate_delta": round(_as_float(decision.get("fallback_rate_delta")), 6),
                "safety_stop_state": public_label(decision.get("safety_stop_state") or "none", "none"),
                "previous_canary_fraction": round(_as_float(decision.get("previous_canary_fraction")), 6),
                "canary_fraction": round(canary_fraction, 6),
                "canary_enabled": bool(rollout.get("canary_enabled", True)),
                "full_rollout_active": full_rollout_active,
                "full_rollout_fraction": round(full_rollout_fraction, 6),
                "holdout_fraction": round(
                    0.0
                    if full_rollout_active
                    else _as_float(decision.get("holdout_fraction") or rollout.get("holdout_fraction")),
                    6,
                ),
                "max_rollout_fraction": round(
                    _as_float(safety_gates.get("max_rollout_fraction"), DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION),
                    6,
                ),
                "rollback_metadata_present": bool(rollback_metadata),
                "rollback_action_type": public_label(rollback_metadata.get("rollback_action_type") or "unknown", "unknown"),
                "metadata_only": True,
                "aggregate_only": True,
            }
        )
    return rules


def _crunch_activation_post_widening_decision(
    *,
    applied_count: int,
    holdout_count: int,
    observed_saved_tokens: int,
    observed_saved_usd: float,
    fallback_count: int,
    safety_stop_count: int,
    rollback_count: int,
    error_rate_delta: float,
    retry_rate_delta: float,
    fallback_rate_delta: float,
    canary_fraction: float,
    max_rollout_fraction: float,
) -> tuple[str, str, list[str]]:
    blockers: list[str] = []
    if safety_stop_count > 0:
        blockers.append("safety-stop-observed")
    if rollback_count > 0:
        blockers.append("rollback-observed")
    if fallback_count > 0 or fallback_rate_delta > 0:
        blockers.append("fallback-observed")
    if error_rate_delta > DEFAULT_CRUNCH_CANARY_ROLLBACK_ERROR_RATE:
        blockers.append("error-rate-regression")
    if retry_rate_delta > DEFAULT_CRUNCH_CANARY_ROLLBACK_RETRY_RATE_DELTA:
        blockers.append("retry-rate-regression")
    if blockers:
        return "post-widening-rollback-required", "rollback", blockers

    missing: list[str] = []
    if applied_count <= 0:
        missing.append("post-widening-applied-coverage")
    if holdout_count <= 0:
        missing.append("post-widening-holdout-coverage")
    if observed_saved_tokens <= 0 and observed_saved_usd <= 0:
        missing.append("post-widening-savings-observation")
    if missing:
        return "post-widening-measurement-incomplete", "keep-active", missing

    if max_rollout_fraction > 0 and canary_fraction >= max_rollout_fraction:
        return "post-widening-active-at-max-rollout", "keep-active", []
    if max_rollout_fraction > 0 and canary_fraction > 0 and canary_fraction < max_rollout_fraction:
        return "post-widening-widen-ready", "widen-further", []
    return "post-widening-active-observed", "monitor-post-widening-crunch-activation", []


def _crunch_post_max_rollout_local_policy_patch(
    *,
    patch_type: str,
    decision_id: str,
    active_rule_ref: str | None,
    canary_fraction: float,
    max_rollout_fraction: float,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "schema": "tokenclaw.request_shape_crunch_post_max_rollout_policy_patch.v1",
        "patch_type": patch_type,
        "target_local_policy": "crunch_rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "decision_id": public_label(decision_id, "unknown"),
        "active_rule_ref": public_label(active_rule_ref, "unknown") if active_rule_ref else None,
        "dry_run": True,
        "operator_review_required": True,
        "policy_files_written": False,
        "policy_file_contents_included": False,
        "current_rollout": {
            "schema": "tokenclaw.request_shape_crunch_rollout_snapshot.v1",
            "canary_fraction": round(canary_fraction, 6),
            "max_rollout_fraction": round(max_rollout_fraction, 6),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _crunch_opportunity_privacy(),
    }
    if patch_type == "promote_repeated_context_crunch_rule_full_rollout":
        patch["rollout_update"] = {
            "schema": "tokenclaw.request_shape_crunch_full_rollout_update.v1",
            "canary_enabled": False,
            "full_rollout_fraction": 1.0,
            "canary_fraction": 1.0,
            "holdout_fraction": 0.0,
            "metadata_only": True,
            "aggregate_only": True,
        }
    elif patch_type == "rollback_repeated_context_crunch_rule":
        patch["rollout_update"] = {
            "schema": "tokenclaw.request_shape_crunch_rollback_update.v1",
            "enabled": False,
            "canary_enabled": False,
            "metadata_only": True,
            "aggregate_only": True,
        }
    return {key: value for key, value in patch.items() if value not in (None, "", [], {})}


def _crunch_activation_post_max_rollout_decision(
    *,
    post_widening_status: str,
    post_widening_next_action: str,
    post_widening_reasons: list[str],
    decision_id: str,
    active_rule_ref: str | None,
    canary_fraction: float,
    max_rollout_fraction: float,
    rollback_metadata_present: bool,
    full_rollout_active: bool = False,
) -> dict[str, Any]:
    reason_codes = [public_label(reason, "unknown") for reason in post_widening_reasons if public_label(reason, "unknown") != "unknown"]
    decision = "not-applicable"
    status = "post-max-rollout-not-applicable"
    next_action = post_widening_next_action
    cap_reason: str | None = None
    local_policy_patch: dict[str, Any] | None = None

    if full_rollout_active:
        decision = "full-rollout-applied"
        status = "post-max-rollout-full-rollout-applied"
        next_action = "measure-full-rollout-repeated-context-crunch-outcomes"
        reason_codes = reason_codes or ["full-rollout-policy-active"]
    elif post_widening_next_action == "rollback":
        decision = "rollback"
        status = "post-max-rollout-rollback-required"
        next_action = "rollback"
        local_policy_patch = _crunch_post_max_rollout_local_policy_patch(
            patch_type="rollback_repeated_context_crunch_rule",
            decision_id=decision_id,
            active_rule_ref=active_rule_ref,
            canary_fraction=canary_fraction,
            max_rollout_fraction=max_rollout_fraction,
        )
    elif post_widening_status == "post-widening-active-at-max-rollout":
        if rollback_metadata_present:
            decision = "promote-full"
            status = "post-max-rollout-full-rollout-ready"
            next_action = "promote-full-repeated-context-crunch-rule"
            reason_codes = reason_codes or ["max-rollout-cap-only"]
            local_policy_patch = _crunch_post_max_rollout_local_policy_patch(
                patch_type="promote_repeated_context_crunch_rule_full_rollout",
                decision_id=decision_id,
                active_rule_ref=active_rule_ref,
                canary_fraction=canary_fraction,
                max_rollout_fraction=max_rollout_fraction,
            )
        else:
            decision = "keep-capped"
            status = "post-max-rollout-keep-capped"
            next_action = "keep-capped-add-rollback-proof"
            cap_reason = "rollback-metadata-missing"
            reason_codes = reason_codes or [cap_reason]
    elif post_widening_status == "post-widening-measurement-incomplete" and max_rollout_fraction > 0 and canary_fraction >= max_rollout_fraction:
        decision = "keep-capped"
        status = "post-max-rollout-keep-capped"
        next_action = "keep-capped-collect-missing-measurements"
        cap_reason = reason_codes[0] if reason_codes else "post-widening-measurement-incomplete"
        reason_codes = reason_codes or [cap_reason]

    rollback_metadata = {
        "schema": "tokenclaw.request_shape_crunch_post_max_rollout_rollback_metadata.v1",
        "present": bool(rollback_metadata_present),
        "required_for_promotion": True,
        "rollback_action_type": "disable_repeated_context_crunch_canary",
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "active_rule_ref": public_label(active_rule_ref, "unknown") if active_rule_ref else None,
        "policy_file_contents_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }
    result = {
        "schema": CRUNCH_POST_MAX_ROLLOUT_DECISION_SCHEMA,
        "status": status,
        "decision": decision,
        "decision_options": ["promote-full", "full-rollout-applied", "keep-capped", "rollback"],
        "next_action": next_action,
        "promotion_allowed": decision == "promote-full",
        "full_rollout_allowed": decision in {"promote-full", "full-rollout-applied"},
        "full_rollout_active": bool(full_rollout_active),
        "cap_reason": cap_reason,
        "reason_codes": reason_codes,
        "decision_id": public_label(decision_id, "unknown"),
        "active_rule_ref": public_label(active_rule_ref, "unknown") if active_rule_ref else None,
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "canary_fraction": round(canary_fraction, 6),
        "max_rollout_fraction": round(max_rollout_fraction, 6),
        "rollback_metadata": {key: value for key, value in rollback_metadata.items() if value not in (None, "", [], {})},
        "local_policy_patch": local_policy_patch,
        "privacy": _crunch_opportunity_privacy(),
    }
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def build_request_shape_crunch_activation_evidence_report(
    *,
    crunch_policy_decision: dict[str, Any],
    crunch_canary_impact: dict[str, Any],
    rules_path: str | Path | None = None,
) -> dict[str, Any]:
    decision_summary = crunch_policy_decision.get("summary") if isinstance(crunch_policy_decision.get("summary"), dict) else {}
    impact_summary = crunch_canary_impact.get("summary") if isinstance(crunch_canary_impact.get("summary"), dict) else {}
    decision_id = public_label(
        crunch_policy_decision.get("decision_id") or decision_summary.get("decision_id") or "unknown",
        "unknown",
    )
    active_rules = _active_request_shape_crunch_rules(rules_path)
    matching_rules = [rule for rule in active_rules if rule.get("decision_id") == decision_id]
    evidence_rules = matching_rules or active_rules
    top_rule = evidence_rules[0] if evidence_rules else {}

    applied_count = _as_int(decision_summary.get("applied_count") or impact_summary.get("applied_count") or top_rule.get("applied_count"))
    holdout_count = _as_int(decision_summary.get("holdout_count") or impact_summary.get("holdout_count") or top_rule.get("holdout_count"))
    coverage = decision_summary.get("coverage") if isinstance(decision_summary.get("coverage"), dict) else {}
    skipped_count = _as_int(coverage.get("skipped_count") or impact_summary.get("skipped_count") or top_rule.get("skipped_count"))
    observed_saved_tokens = _as_int(
        decision_summary.get("observed_saved_tokens")
        or impact_summary.get("saved_tokens")
        or top_rule.get("observed_saved_tokens")
    )
    observed_saved_usd = _as_float(
        decision_summary.get("observed_saved_usd")
        or impact_summary.get("saved_usd")
        or top_rule.get("observed_saved_usd")
    )
    safety_stop_count = _as_int(coverage.get("safety_stop_count") or impact_summary.get("safety_stop_count"))
    rollback_count = _as_int(coverage.get("rollback_count") or impact_summary.get("rollback_count"))
    fallback_count = _as_int(coverage.get("fallback_count") or impact_summary.get("fallback_count"))
    error_rate_delta = round(
        _as_float(decision_summary.get("error_rate_delta") if "error_rate_delta" in decision_summary else top_rule.get("error_rate_delta")),
        6,
    )
    retry_rate_delta = round(
        _as_float(decision_summary.get("retry_rate_delta") if "retry_rate_delta" in decision_summary else top_rule.get("retry_rate_delta")),
        6,
    )
    fallback_rate_delta = round(
        _as_float(decision_summary.get("fallback_rate_delta") if "fallback_rate_delta" in decision_summary else top_rule.get("fallback_rate_delta")),
        6,
    )
    canary_fraction = _as_float(top_rule.get("canary_fraction"))
    max_rollout_fraction = _as_float(top_rule.get("max_rollout_fraction"), DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION)
    full_rollout_active = bool(top_rule.get("full_rollout_active"))
    widened_rule_count = sum(1 for rule in active_rules if rule.get("decision") == "widen")
    full_rollout_rule_count = sum(1 for rule in active_rules if rule.get("full_rollout_active"))
    matching_widened_rule_count = sum(1 for rule in matching_rules if rule.get("decision") == "widen")
    matching_full_rollout_rule_count = sum(1 for rule in matching_rules if rule.get("full_rollout_active"))
    decision = public_label(crunch_policy_decision.get("decision") or decision_summary.get("decision") or "unknown", "unknown")
    has_active_decision_rule = bool(matching_rules)
    has_measured_decision = applied_count > 0 and holdout_count > 0
    post_widening_status, post_widening_next_action, post_widening_reasons = _crunch_activation_post_widening_decision(
        applied_count=applied_count,
        holdout_count=holdout_count,
        observed_saved_tokens=observed_saved_tokens,
        observed_saved_usd=observed_saved_usd,
        fallback_count=fallback_count,
        safety_stop_count=safety_stop_count,
        rollback_count=rollback_count,
        error_rate_delta=error_rate_delta,
        retry_rate_delta=retry_rate_delta,
        fallback_rate_delta=fallback_rate_delta,
        canary_fraction=canary_fraction,
        max_rollout_fraction=max_rollout_fraction,
    )
    if has_active_decision_rule and has_measured_decision:
        status = "active-rule-evidence-observed"
        next_action = post_widening_next_action
        missing_measurements = post_widening_reasons if post_widening_status == "post-widening-measurement-incomplete" else []
    elif has_measured_decision:
        status = "policy-decision-without-active-rule"
        next_action = "inspect-crunch-rule-file-activation"
        missing_measurements = ["matching-active-crunch-rule"]
    else:
        status = "missing-crunch-activation-evidence"
        next_action = "measure-request-shape-crunch-canary-impact"
        missing_measurements = ["applied-and-holdout-crunch-decision-coverage"]
    active_rule_ref = str(top_rule.get("rule_ref") or top_rule.get("rule_id") or "") or None
    post_max_rollout_decision = _crunch_activation_post_max_rollout_decision(
        post_widening_status=post_widening_status,
        post_widening_next_action=post_widening_next_action,
        post_widening_reasons=post_widening_reasons,
        decision_id=decision_id,
        active_rule_ref=active_rule_ref,
        canary_fraction=canary_fraction,
        max_rollout_fraction=max_rollout_fraction,
        rollback_metadata_present=bool(top_rule.get("rollback_metadata_present")),
        full_rollout_active=full_rollout_active,
    )
    post_max_rollout_next_action = str(post_max_rollout_decision.get("next_action") or post_widening_next_action)
    keep_active_outcome = (
        status == "active-rule-evidence-observed"
        and post_widening_status == "post-widening-active-at-max-rollout"
    )
    full_rollout_outcome = status == "active-rule-evidence-observed" and full_rollout_active
    if keep_active_outcome or full_rollout_outcome:
        next_action = post_max_rollout_next_action
    duplicate_reason = (
        "repeated-context-crunch-full-rollout-active"
        if full_rollout_outcome
        else "repeated-context-crunch-active-at-max-rollout"
        if keep_active_outcome
        else None
    )
    duplicate_suppression = {
        "schema": "tokenclaw.request_shape_crunch_keep_active_duplicate_suppression.v1",
        "suppresses_new_activation_issue": keep_active_outcome or full_rollout_outcome,
        "suppresses_generic_crunch_activation_issue": keep_active_outcome or full_rollout_outcome,
        "reason": duplicate_reason,
        "fingerprint": public_id(
            "|".join(
                part
                for part in (
                    "crunch",
                    decision_id,
                    active_rule_ref or "",
                    "crunch_rules.yaml",
                    "crunch.rules",
                )
                if part
            ),
            prefix="activation",
            fallback="activation:unknown",
        ),
        "matching_local_policy": "crunch_rules" if keep_active_outcome else None,
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "metadata_only": True,
        "aggregate_only": True,
    }
    activation_follow_up = {
        "schema": "tokenclaw.request_shape_crunch_activation_follow_up.v1",
        "status": "full-rollout-outcome-recorded"
        if full_rollout_outcome
        else "keep-active-outcome-recorded"
        if keep_active_outcome
        else status,
        "savings_status": "active-rule-evidence-observed" if keep_active_outcome or full_rollout_outcome else status,
        "report_key": "request_shape_crunch_activation_evidence",
        "evidence_schema": CRUNCH_ACTIVATION_EVIDENCE_SCHEMA,
        "activation_state": "full-rollout-active"
        if full_rollout_outcome
        else "measured-active"
        if keep_active_outcome
        else ("missing-measurement" if missing_measurements else "measured-savings"),
        "activation_mode": "active-local-policy",
        "next_action": next_action,
        "post_max_rollout_decision": post_max_rollout_decision,
        "target_local_policy": "crunch_rules",
        "policy_section": "crunch",
        "local_file_backed": True,
        "projected_saved_tokens": observed_saved_tokens,
        "projected_saved_usd": round(observed_saved_usd, 6),
        "canary_applied_rows": applied_count,
        "canary_holdout_rows": holdout_count,
        "canary_already_staged": has_active_decision_rule,
        "canary_already_applied": applied_count > 0,
        "no_op_reason": duplicate_reason,
        "duplicate_suppression": duplicate_suppression,
        "missing_measurements": missing_measurements,
        "privacy": _crunch_opportunity_privacy(),
    }

    return {
        "schema": CRUNCH_ACTIVATION_EVIDENCE_SCHEMA,
        "status": status,
        "ok": True,
        "read_only": True,
        "decision_id": decision_id,
        "decision": decision,
        "graduation_decision": public_label(
            crunch_policy_decision.get("graduation_decision") or decision_summary.get("graduation_decision") or decision,
            "unknown",
        ),
        "next_action": next_action,
        "source_reports": {
            "policy_decision_schema": public_label(crunch_policy_decision.get("schema") or "unknown", "unknown"),
            "canary_impact_schema": public_label(crunch_canary_impact.get("schema") or "unknown", "unknown"),
            "active_rule_source_evidence_schema": public_label(top_rule.get("source_evidence_schema") or "unknown", "unknown"),
        },
        "summary": {
            "active_rule_count": len(active_rules),
            "matching_active_rule_count": len(matching_rules),
            "widened_rule_count": widened_rule_count,
            "matching_widened_rule_count": matching_widened_rule_count,
            "full_rollout_rule_count": full_rollout_rule_count,
            "matching_full_rollout_rule_count": matching_full_rollout_rule_count,
            "decision": decision,
            "graduation_decision": public_label(decision_summary.get("graduation_decision") or decision, "unknown"),
            "decision_id": decision_id,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "blocked_count": safety_stop_count + rollback_count + fallback_count,
            "fallback_count": fallback_count,
            "safety_stop_count": safety_stop_count,
            "rollback_count": rollback_count,
            "error_rate_delta": error_rate_delta,
            "retry_rate_delta": retry_rate_delta,
            "fallback_rate_delta": fallback_rate_delta,
            "safety_stop_state": public_label(top_rule.get("safety_stop_state") or decision_summary.get("safety_stop_state") or "none", "none"),
            "observed_saved_tokens": observed_saved_tokens,
            "observed_saved_usd": round(observed_saved_usd, 8),
            "policy_source": public_label(top_rule.get("policy_source") or decision_summary.get("policy_source") or "local-manual", "local-manual"),
            "previous_canary_fraction": round(_as_float(top_rule.get("previous_canary_fraction")), 6),
            "canary_fraction": round(canary_fraction, 6),
            "full_rollout_active": full_rollout_active,
            "full_rollout_fraction": round(_as_float(top_rule.get("full_rollout_fraction")), 6),
            "holdout_fraction": round(_as_float(top_rule.get("holdout_fraction")), 6),
            "max_rollout_fraction": round(max_rollout_fraction, 6),
            "post_widening_status": post_widening_status,
            "post_widening_next_action": post_widening_next_action,
            "post_widening_reason_codes": post_widening_reasons,
            "post_max_rollout_status": post_max_rollout_decision.get("status"),
            "post_max_rollout_decision": post_max_rollout_decision.get("decision"),
            "post_max_rollout_next_action": post_max_rollout_decision.get("next_action"),
            "post_max_rollout_reason_codes": post_max_rollout_decision.get("reason_codes", []),
            "post_max_rollout_promotion_allowed": bool(post_max_rollout_decision.get("promotion_allowed")),
            "post_max_rollout_cap_reason": post_max_rollout_decision.get("cap_reason"),
            "target_local_rule_file": "crunch_rules.yaml",
            "target_local_policy_section": "crunch.rules",
            "policy_files_written": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "next_action": next_action,
        },
        "rules": evidence_rules[:5],
        "post_max_rollout_decision": post_max_rollout_decision,
        "activation_follow_up": activation_follow_up,
        "duplicate_suppression": duplicate_suppression,
        "missing_measurements": missing_measurements,
        "privacy": _crunch_opportunity_privacy(),
    }


def _crunch_remaining_shape_key(cohort: dict[str, Any]) -> tuple[Any, ...]:
    return (
        public_label(cohort.get("provider_family"), "unknown"),
        public_label(cohort.get("source_surface"), "unknown"),
        public_label(cohort.get("endpoint"), "unknown"),
        public_label(cohort.get("category"), "unknown"),
        public_label(cohort.get("workflow_phase"), "unknown"),
        bool(cohort.get("stream")),
        bool(cohort.get("has_tools")),
        public_label(cohort.get("cache_status"), "unknown"),
        public_label(cohort.get("routing_status"), "unknown"),
        public_label(cohort.get("text_bucket"), "unknown"),
        public_label(cohort.get("token_bucket"), "unknown"),
    )


def _crunch_remaining_active_rule_coverage(
    candidate: dict[str, Any],
    *,
    active_rules: list[dict[str, Any]],
    activation_evidence: dict[str, Any],
) -> dict[str, Any]:
    active_summary = activation_evidence.get("summary") if isinstance(activation_evidence.get("summary"), dict) else {}
    active_next_action = public_label(
        activation_evidence.get("next_action") or active_summary.get("next_action") or "unknown",
        "unknown",
    )
    active_status = public_label(activation_evidence.get("status") or "unknown", "unknown")
    for rule in active_rules:
        if _request_shape_crunch_cohort_matches_rule(candidate, rule):
            return {
                "schema": "tokenclaw.request_shape_crunch_remaining_active_rule_coverage.v1",
                "status": "covered-by-active-rule",
                "covered": True,
                "matching_local_policy": "crunch_rules",
                "matching_policy_id": public_label(rule.get("policy_id") or rule.get("id"), "unknown"),
                "matching_cohort_id": public_label(rule.get("cohort_id"), "unknown"),
                "active_rule_status": active_status,
                "active_rule_next_action": active_next_action,
                "target_local_rule_file": "crunch_rules.yaml",
                "target_local_policy_section": "crunch.rules",
                "metadata_only": True,
                "aggregate_only": True,
                "privacy": _crunch_opportunity_privacy(),
            }
    if active_rules:
        status = "not-covered-by-active-rule"
    elif active_status == "active-rule-evidence-observed":
        status = "active-rule-coverage-unmatched"
    else:
        status = "no-active-rule-coverage"
    return {
        "schema": "tokenclaw.request_shape_crunch_remaining_active_rule_coverage.v1",
        "status": status,
        "covered": False,
        "matching_local_policy": None,
        "matching_policy_id": None,
        "matching_cohort_id": None,
        "active_rule_status": active_status,
        "active_rule_next_action": active_next_action,
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }


def build_request_shape_crunch_remaining_measurement_report(
    *,
    follow_up_candidates: dict[str, Any],
    crunch_opportunity: dict[str, Any],
    activation_evidence: dict[str, Any],
    rules_path: str | Path | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    candidates = follow_up_candidates.get("candidates") if isinstance(follow_up_candidates.get("candidates"), list) else []
    opportunity_cohorts = crunch_opportunity.get("cohorts") if isinstance(crunch_opportunity.get("cohorts"), list) else []
    opportunity_by_shape = {
        _crunch_remaining_shape_key(cohort): cohort
        for cohort in opportunity_cohorts
        if isinstance(cohort, dict)
    }
    active_rules = _load_request_shape_crunch_canary_rules(rules_path)
    rows: list[dict[str, Any]] = []
    excluded_active_rule_count = 0
    projected_tokens = 0
    projected_savings = 0.0
    applied_count = 0
    holdout_count = 0
    fallback_count = 0
    retry_count = 0
    rollback_count = 0
    safety_stop_count = 0

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if public_label(candidate.get("local_action_family"), "unknown") != "crunch":
            continue
        if public_label(candidate.get("readiness_state"), "unknown") != "measurement-required":
            continue
        if public_label(candidate.get("next_action"), "unknown") != "measure-repeated-context-crunch-canary-impact":
            continue

        opportunity = opportunity_by_shape.get(_crunch_remaining_shape_key(candidate), {})
        lifecycle = opportunity.get("crunch_canary_lifecycle") if isinstance(opportunity.get("crunch_canary_lifecycle"), dict) else {}
        duplicate = opportunity.get("duplicate_suppression") if isinstance(opportunity.get("duplicate_suppression"), dict) else {}
        coverage = _crunch_remaining_active_rule_coverage(
            candidate,
            active_rules=active_rules,
            activation_evidence=activation_evidence,
        )
        if coverage["covered"]:
            excluded_active_rule_count += 1
            continue

        row_applied = _as_int(lifecycle.get("applied_count"))
        row_holdout = _as_int(lifecycle.get("holdout_count"))
        row_fallback = _as_int(lifecycle.get("fallback_count"))
        row_retry = _as_int(lifecycle.get("retry_count"))
        row_rollback = _as_int(lifecycle.get("rollback_count"))
        row_safety = _as_int(lifecycle.get("safety_stopped_count"))
        row_projected_tokens = _as_int(candidate.get("projected_saved_tokens") or candidate.get("projected_crunch_tokens_saved"))
        row_projected_savings = _as_float(candidate.get("projected_savings_usd") or candidate.get("projected_crunch_savings_usd"))
        projected_tokens += row_projected_tokens
        projected_savings += row_projected_savings
        applied_count += row_applied
        holdout_count += row_holdout
        fallback_count += row_fallback
        retry_count += row_retry
        rollback_count += row_rollback
        safety_stop_count += row_safety
        rows.append(
            {
                "schema": "tokenclaw.request_shape_crunch_remaining_measurement_cohort.v1",
                "rank": _as_int(candidate.get("rank")),
                "row_count": _as_int(candidate.get("row_count")),
                "sample_count": _as_int(candidate.get("sample_count") or candidate.get("row_count")),
                "readiness_state": public_label(candidate.get("readiness_state"), "unknown"),
                "next_action": public_label(candidate.get("next_action"), "unknown"),
                "actionability_reason": public_label(candidate.get("actionability_reason"), "unknown"),
                "blocker_codes": _public_label_list(candidate.get("blocker_codes")),
                "evidence_blocker_codes": _public_label_list(opportunity.get("evidence_blocker_codes")),
                "projected_saved_tokens": row_projected_tokens,
                "projected_saved_usd": round(row_projected_savings, 8),
                "projected_crunch_tokens_saved": _as_int(candidate.get("projected_crunch_tokens_saved")),
                "projected_crunch_savings_usd": round(_as_float(candidate.get("projected_crunch_savings_usd")), 8),
                "applied_count": row_applied,
                "holdout_count": row_holdout,
                "fallback_count": row_fallback,
                "retry_count": row_retry,
                "rollback_count": row_rollback,
                "safety_stop_count": row_safety,
                "active_rule_coverage_status": coverage["status"],
                "active_rule_coverage": coverage,
                "duplicate_suppression": {
                    "schema": public_label(duplicate.get("schema"), "tokenclaw.request_shape_crunch_stage_duplicate_suppression.v1"),
                    "suppressed": bool(duplicate.get("suppressed") or duplicate.get("suppresses_new_stage_action")),
                    "suppresses_new_stage_action": bool(duplicate.get("suppresses_new_stage_action")),
                    "reason": public_label(duplicate.get("reason"), "unknown"),
                    "matching_local_policy": public_label(duplicate.get("matching_local_policy"), "unknown"),
                    "metadata_only": True,
                    "aggregate_only": True,
                    "privacy": _crunch_opportunity_privacy(),
                },
                "provider_family": public_label(candidate.get("provider_family"), "unknown"),
                "source_surface": public_label(candidate.get("source_surface"), "unknown"),
                "endpoint": public_label(candidate.get("endpoint"), "unknown"),
                "category": public_label(candidate.get("category"), "unknown"),
                "workflow_phase": public_label(candidate.get("workflow_phase"), "unknown"),
                "stream": bool(candidate.get("stream")),
                "has_tools": bool(candidate.get("has_tools")),
                "text_bucket": public_label(candidate.get("text_bucket"), "unknown"),
                "token_bucket": public_label(candidate.get("token_bucket"), "unknown"),
                "cache_status": public_label(candidate.get("cache_status"), "unknown"),
                "routing_status": public_label(candidate.get("routing_status"), "unknown"),
                "privacy": _crunch_opportunity_privacy(),
            }
        )

    rows.sort(
        key=lambda item: (
            _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("projected_saved_tokens")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit, 10), 50))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    activation_summary = activation_evidence.get("summary") if isinstance(activation_evidence.get("summary"), dict) else {}
    duplicate_suppression = activation_evidence.get("duplicate_suppression") if isinstance(activation_evidence.get("duplicate_suppression"), dict) else {}
    missing_measurements = [] if rows else ["remaining-measurement-required-crunch-cohorts"]
    return {
        "schema": CRUNCH_REMAINING_MEASUREMENT_SCHEMA,
        "status": "ranked" if rows else "no-remaining-measurement-required-cohorts",
        "ok": True,
        "read_only": True,
        "source_reports": {
            "follow_up_candidates_schema": public_label(follow_up_candidates.get("schema"), "unknown"),
            "crunch_opportunity_schema": public_label(crunch_opportunity.get("schema"), "unknown"),
            "activation_evidence_schema": public_label(activation_evidence.get("schema"), "unknown"),
        },
        "summary": {
            "remaining_measurement_required_count": len(rows),
            "reported_count": min(len(rows), capped_limit),
            "excluded_active_rule_covered_count": excluded_active_rule_count,
            "active_rule_count": _as_int(activation_summary.get("active_rule_count")),
            "active_rule_status": public_label(activation_evidence.get("status"), "unknown"),
            "active_rule_next_action": public_label(activation_evidence.get("next_action"), "unknown"),
            "active_rule_duplicate_suppresses_new_activation_issue": bool(duplicate_suppression.get("suppresses_new_activation_issue")),
            "projected_saved_tokens": projected_tokens,
            "projected_saved_usd": round(projected_savings, 8),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "rollback_count": rollback_count,
            "safety_stop_count": safety_stop_count,
            "top_active_rule_coverage_status": rows[0]["active_rule_coverage_status"] if rows else None,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "cohorts": rows[:capped_limit],
        "top_cohort": rows[0] if rows else None,
        "missing_measurements": missing_measurements,
        "privacy": _crunch_opportunity_privacy(),
    }


def _shape_crunch_decision(row: dict[str, Any]) -> dict[str, Any]:
    row_count = _as_int(row.get("row_count") or row.get("count"))
    classes = {str(item) for item in row.get("candidate_work_classes") or []}
    text_bucket = str(row.get("text_bucket") or "unknown")
    token_bucket = str(row.get("token_bucket") or "unknown")
    projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    observed_tokens = _as_int(row.get("current_crunch_tokens_saved"))
    lifecycle = row.get("crunch_canary_lifecycle") if isinstance(row.get("crunch_canary_lifecycle"), dict) else {}
    applied_count = _as_int(lifecycle.get("applied_count"))
    holdout_count = _as_int(lifecycle.get("holdout_count"))
    safety_stopped_count = _as_int(lifecycle.get("safety_stopped_count"))
    blockers: set[str] = set()

    if safety_stopped_count > 0:
        return {
            "readiness": "canary-safety-stopped",
            "reason": "repeated-context-crunch-canary-safety-stopped",
            "blockers": ["canary-safety-stopped"],
        }
    if applied_count > 0 and holdout_count > 0:
        return {
            "readiness": "canary-staged",
            "reason": "repeated-context-crunch-canary-applied-and-holdout",
            "blockers": [],
        }
    if applied_count > 0:
        return {
            "readiness": "canary-applied",
            "reason": "repeated-context-crunch-canary-applied",
            "blockers": [],
        }
    if holdout_count > 0:
        return {
            "readiness": "canary-holdout",
            "reason": "repeated-context-crunch-canary-holdout",
            "blockers": [],
        }

    if row_count < 2:
        blockers.add("insufficient-repeat-evidence")
    if text_bucket not in REPEATED_CONTEXT_TEXT_BUCKETS and token_bucket not in LARGE_CONTEXT_TOKEN_BUCKETS:
        blockers.add("not-large-context")
    if "crunch" not in classes and "repeated_context" not in classes:
        blockers.add("not-crunch-work-class")
    activation_readiness = public_label(row.get("activation_candidate_readiness_state"), "")
    if activation_readiness and activation_readiness != "activation-ready":
        blockers.add(f"activation-candidate-{activation_readiness}")
    freshness_state = public_label(row.get("freshness_state") or row.get("snapshot_freshness_state"), "")
    if freshness_state in {"stale", "rollup-stale", "snapshot-stale"}:
        blockers.add("stale-rollup-evidence")
    if _as_int(row.get("successful_input_tokens") or row.get("input_tokens")) <= 0:
        blockers.add("missing-token-metadata")
    if projected_tokens <= 0 and observed_tokens <= 0:
        blockers.add("non-positive-projection")

    if not blockers:
        return {
            "readiness": "measurement-ready",
            "reason": "repeated-context-crunch-opportunity",
            "blockers": [],
        }

    reason_priority = (
        "stale-rollup-evidence",
        "activation-candidate-blocked",
        "activation-candidate-stale",
        "activation-candidate-missing-evidence",
        "missing-token-metadata",
        "insufficient-repeat-evidence",
        "not-large-context",
        "not-crunch-work-class",
        "non-positive-projection",
    )
    return {
        "readiness": "skipped",
        "reason": next((item for item in reason_priority if item in blockers), sorted(blockers)[0]),
        "blockers": sorted(blockers),
    }


def _shape_crunch_candidate_status(decision: dict[str, Any]) -> str:
    readiness = str(decision.get("readiness") or "unknown")
    reason = str(decision.get("reason") or "")
    blockers = {str(item) for item in decision.get("blockers") or []}
    if readiness == "measurement-ready":
        return "candidate"
    if readiness == "canary-safety-stopped" or "canary-safety-stopped" in blockers:
        return "safety-blocked"
    if readiness in {"canary-staged", "canary-applied", "canary-holdout"}:
        return "policy-write-required"
    if reason in {"insufficient-repeat-evidence", "not-large-context"} or blockers.intersection(
        {"insufficient-repeat-evidence", "not-large-context"}
    ):
        return "too-small"
    if reason in {"missing-token-metadata", "non-positive-projection"} or blockers.intersection(
        {"missing-token-metadata", "non-positive-projection"}
    ):
        return "missing-observed-savings"
    return "blocked"


def _shape_crunch_policy_write_status(candidate_status: str) -> str:
    if candidate_status == "candidate":
        return "policy-write-required"
    if candidate_status == "policy-write-required":
        return "policy-already-written"
    return "no-policy-write"


def _bounded_fraction(value: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _scaled_projection(total: int, selected: int, matched: int) -> int:
    if total <= 0 or selected <= 0 or matched <= 0:
        return 0
    return max(0, int(round(total * (selected / float(matched)))))


def _request_shape_crunch_canary_lifecycle_projection(
    cohort: dict[str, Any],
    *,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    matched = _as_int(cohort.get("row_count") or cohort.get("cohort_row_count"))
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)

    readiness = public_label(cohort.get("readiness"), "unknown")
    reason = public_label(cohort.get("reason"), "unknown")
    blockers = [
        public_label(item, "unknown")
        for item in cohort.get("blockers") or []
        if public_label(item, "unknown") != "unknown"
    ]
    evidence_blockers = _public_label_list(cohort.get("evidence_blocker_codes") or cohort.get("blocker_codes"))
    applied = holdout_count = skipped = safety_stopped = 0
    lifecycle_status = "skipped"
    explicit_reason = reason

    if readiness == "measurement-ready" and matched > 0:
        holdout_count = min(matched, int(math.ceil(matched * holdout))) if holdout > 0 else 0
        remaining = max(0, matched - holdout_count)
        applied = min(remaining, int(math.ceil(matched * rollout))) if rollout > 0 else 0
        skipped = max(0, matched - applied - holdout_count)
        lifecycle_status = "projected-applied-holdout" if applied > 0 and holdout_count > 0 else "projected-partial"
        explicit_reason = "projected-canary-applied-and-holdout" if applied > 0 and holdout_count > 0 else "projected-canary-partial"
    elif readiness == "canary-safety-stopped":
        safety_stopped = matched
        lifecycle_status = "safety-stopped"
        explicit_reason = reason or "repeated-context-crunch-canary-safety-stopped"
    else:
        skipped = matched
        lifecycle_status = "skipped"
        explicit_reason = reason or (blockers[0] if blockers else "not-stageable")

    projected_tokens = _as_int(cohort.get("projected_saved_tokens"))
    projected_chars = _as_int(cohort.get("projected_saved_chars"))
    projected_usd = _as_float(cohort.get("projected_saved_usd"))

    return {
        "schema": "tokenclaw.request_shape_crunch_canary_projected_lifecycle.v1",
        "status": lifecycle_status,
        "reason": explicit_reason,
        "readiness": readiness,
        "matched_count": matched,
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "projected_canary_applied_count": applied,
        "projected_canary_holdout_count": holdout_count,
        "projected_skipped_count": skipped,
        "projected_safety_stopped_count": safety_stopped,
        "projected_fallback_count": 0,
        "projected_rollback_count": 0,
        "projected_saved_tokens": projected_tokens,
        "projected_saved_chars": projected_chars,
        "projected_saved_usd": round(projected_usd, 6),
        "projected_applied_saved_tokens": _scaled_projection(projected_tokens, applied, matched),
        "projected_applied_saved_chars": _scaled_projection(projected_chars, applied, matched),
        "projected_applied_saved_usd": round(projected_usd * (applied / float(matched)), 6) if matched and applied else 0.0,
        "projected_holdout_saved_tokens": _scaled_projection(projected_tokens, holdout_count, matched),
        "projected_holdout_saved_chars": _scaled_projection(projected_chars, holdout_count, matched),
        "projected_holdout_saved_usd": round(projected_usd * (holdout_count / float(matched)), 6) if matched and holdout_count else 0.0,
        "blocker_reasons": blockers,
        "evidence_blocker_codes": evidence_blockers,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }


def _request_shape_crunch_canary_action(
    cohort: dict[str, Any],
    *,
    candidate_count: int,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    cohort_id = str(cohort.get("cohort_id") or _crunch_canary_cohort_id(cohort))
    policy_id = _crunch_canary_policy_id(cohort_id)
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    lifecycle_projection = _request_shape_crunch_canary_lifecycle_projection(
        cohort,
        rollout_fraction=rollout,
        holdout_fraction=holdout,
    )
    evidence_blockers = _public_label_list(cohort.get("evidence_blocker_codes") or cohort.get("blocker_codes"))
    cohort_selector = {
        "schema": "tokenclaw.request_shape_crunch_canary_cohort_selector.v1",
        "provider_family": cohort.get("provider_family"),
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "stream": bool(cohort.get("stream")),
        "has_tools": bool(cohort.get("has_tools")),
        "text_bucket": cohort.get("text_bucket"),
        "token_bucket": cohort.get("token_bucket"),
        "cache_status": cohort.get("cache_status"),
        "routing_status": cohort.get("routing_status"),
        "metadata_only": True,
        "aggregate_only": True,
    }
    rollback_metadata = {
        "schema": "tokenclaw.request_shape_crunch_canary_rollback_metadata.v1",
        "present": True,
        "required_for_promotion": True,
        "rollback_action_type": "disable_repeated_context_crunch_canary",
        "rollback_threshold": DEFAULT_CRUNCH_CANARY_ROLLBACK_ERROR_RATE,
        "rollback_error_rate": DEFAULT_CRUNCH_CANARY_ROLLBACK_ERROR_RATE,
        "rollback_retry_rate_delta": DEFAULT_CRUNCH_CANARY_ROLLBACK_RETRY_RATE_DELTA,
        "rollback_fallback_rate_delta": DEFAULT_CRUNCH_CANARY_ROLLBACK_FALLBACK_RATE_DELTA,
        "rollback_reason_codes": [
            "safety-stop-observed",
            "error-rate-regression",
            "retry-rate-regression",
            "fallback-observed",
            "rollback-observed",
            "operator-requested",
        ],
        "target_policy_id": policy_id,
        "target_cohort_id": cohort_id,
        "target_local_rule_file": "crunch_rules.yaml",
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }
    return {
        "schema": CRUNCH_CANARY_ACTION_SCHEMA,
        "action_type": "stage-local-repeated-context-crunch-canary",
        "target_local_policy": "crunch_rules",
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "policy_section": "crunch",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "source_readiness": "activation-ready",
        "source_readiness_aliases": ["activation-ready", "measurement-ready"],
        "source_evidence_schema": cohort.get("source_evidence_schema") or CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        "source_evidence_schemas": [
            FOLLOW_UP_CANDIDATES_SCHEMA,
            CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        ],
        "source_activation_fingerprint": cohort.get("source_activation_fingerprint"),
        "source_activation_candidate_rank": cohort.get("source_activation_candidate_rank"),
        "source_activation_candidate_next_action": cohort.get("source_activation_candidate_next_action"),
        "local_only_reason": "file-backed-local-policy-no-managed-dependency",
        "candidate_rule": cohort.get("candidate_rule"),
        "candidate_count": candidate_count,
        "cohort_row_count": _as_int(cohort.get("row_count")),
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "canary_fraction": round(rollout, 6),
        "policy_source": "local-manual",
        "cohort_selector": cohort_selector,
        "conditions": {
            "provider_family": cohort.get("provider_family"),
            "source_surface": cohort.get("source_surface"),
            "endpoint": cohort.get("endpoint"),
            "category": cohort.get("category"),
            "workflow_phase": cohort.get("workflow_phase"),
            "stream": bool(cohort.get("stream")),
            "has_tools": bool(cohort.get("has_tools")),
            "text_bucket": cohort.get("text_bucket"),
            "token_bucket": cohort.get("token_bucket"),
            "cache_status": cohort.get("cache_status"),
            "routing_status": cohort.get("routing_status"),
        },
        "projected_saved_chars": _as_int(cohort.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(cohort.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(cohort.get("projected_saved_usd")), 6),
        "evidence_blocker_codes": evidence_blockers,
        "duplicate_suppression": cohort.get("duplicate_suppression")
        if isinstance(cohort.get("duplicate_suppression"), dict)
        else _request_shape_crunch_cohort_duplicate_suppression(cohort, []),
        "projected_lifecycle": lifecycle_projection,
        "safety_gates": {
            "metadata_only": True,
            "aggregate_only": True,
            "local_file_backed": True,
            "local_only": True,
            "tool_call_cache_enabled": False,
            "tool_call_cache_enablement_allowed": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "tool_payloads_included": False,
            "holdout_required": holdout > 0,
            "max_rollout_fraction": DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
            "records_applied_holdout_skipped_safety_stopped_fallback_rollback": True,
        },
        "lifecycle_metadata": {
            "schema": "tokenclaw.request_shape_crunch_canary_stage_lifecycle_metadata.v1",
            "emits_applied": True,
            "emits_holdout": True,
            "emits_skipped": True,
            "emits_safety_stopped": True,
            "emits_fallback": True,
            "emits_rollback": True,
            "projected_canary_applied_count": lifecycle_projection["projected_canary_applied_count"],
            "projected_canary_holdout_count": lifecycle_projection["projected_canary_holdout_count"],
            "projected_skipped_count": lifecycle_projection["projected_skipped_count"],
            "projected_safety_stopped_count": lifecycle_projection["projected_safety_stopped_count"],
            "evidence_blocker_codes": evidence_blockers,
            "impact_report": CRUNCH_CANARY_IMPACT_SCHEMA,
            "lifecycle_schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "rollback_threshold": DEFAULT_CRUNCH_CANARY_ROLLBACK_ERROR_RATE,
        "rollback_metadata": rollback_metadata,
        "next_action": "apply-local-crunch-canary-after-review",
        "privacy": _crunch_opportunity_privacy(),
    }


def _request_shape_crunch_stage_rollup_selection_review(
    *,
    cohorts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    stage_action_limit: int,
) -> dict[str, Any]:
    actions_by_cohort = {
        public_label(action.get("cohort_id"), "unknown"): action
        for action in actions
        if isinstance(action, dict)
    }
    rows: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}

    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        cohort_id = public_label(cohort.get("cohort_id"), "unknown")
        readiness = public_label(cohort.get("readiness"), "unknown")
        activation_ready = readiness in {"activation-ready", "measurement-ready"}
        duplicate = cohort.get("duplicate_suppression") if isinstance(cohort.get("duplicate_suppression"), dict) else {}
        action = actions_by_cohort.get(cohort_id)
        if action:
            state = "drafted"
            reason = "activation-ready-stage-action-drafted"
            skipped = False
        else:
            state = "skipped"
            skipped = True
            if activation_ready and bool(duplicate.get("suppresses_new_stage_action")):
                reason = public_label(duplicate.get("reason"), "matching-repeated-context-crunch-canary-already-staged-in-local-policy")
            elif activation_ready:
                reason = "stage-action-limit-reached"
            else:
                blockers = _public_label_list(cohort.get("blockers") or cohort.get("evidence_blocker_codes"))
                reason = public_label(cohort.get("reason") or (blockers[0] if blockers else "not-activation-ready"), "not-activation-ready")
            _increment(skipped_reasons, reason, _as_int(cohort.get("row_count")) or 1)
        _increment(state_counts, state)
        row: dict[str, Any] = {
            "schema": "tokenclaw.request_shape_crunch_canary_stage_rollup_selection_row.v1",
            "rank": _as_int(cohort.get("rank")),
            "cohort_id": cohort_id,
            "policy_id": public_label(cohort.get("policy_id"), "unknown"),
            "state": state,
            "selected_for_stage": bool(action),
            "skipped": skipped,
            "skip_reason": None if action else reason,
            "selection_reason": reason,
            "readiness": readiness,
            "activation_readiness": "activation-ready" if activation_ready else "not-activation-ready",
            "next_action": "stage-repeated-context-crunch-canary" if activation_ready else "skip-repeated-context-crunch-canary",
            "source_evidence_schema": public_label(cohort.get("source_evidence_schema"), "unknown"),
            "provider_family": public_label(cohort.get("provider_family"), "unknown"),
            "source_surface": public_label(cohort.get("source_surface"), "unknown"),
            "endpoint": public_label(cohort.get("endpoint"), "unknown"),
            "category": public_label(cohort.get("category"), "unknown"),
            "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
            "stream": bool(cohort.get("stream")),
            "has_tools": bool(cohort.get("has_tools")),
            "cache_status": public_label(cohort.get("cache_status"), "unknown"),
            "routing_status": public_label(cohort.get("routing_status"), "unknown"),
            "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
            "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
            "row_count": _as_int(cohort.get("row_count")),
            "sample_count": _as_int(cohort.get("row_count")),
            "projected_saved_tokens": _as_int(cohort.get("projected_saved_tokens")),
            "projected_saved_usd": round(_as_float(cohort.get("projected_saved_usd")), 8),
            "target_local_policy_section": "crunch.rules",
            "target_local_rule_file": "crunch_rules.yaml",
            "duplicate_suppression": {
                "suppressed": bool(duplicate.get("suppressed") or duplicate.get("suppresses_new_stage_action")),
                "reason": public_label(duplicate.get("reason"), "unknown"),
                "matching_local_policy": public_label(duplicate.get("matching_local_policy"), "unknown"),
                "metadata_only": True,
                "aggregate_only": True,
            },
            "privacy": _crunch_opportunity_privacy(),
        }
        if action:
            row.update(
                {
                    "canary_fraction": round(_as_float(action.get("canary_fraction") or action.get("rollout_fraction")), 6),
                    "holdout_fraction": round(_as_float(action.get("holdout_fraction")), 6),
                    "rollback_metadata": action.get("rollback_metadata"),
                }
            )
        rows.append(row)

    activation_ready_count = sum(1 for row in rows if row["activation_readiness"] == "activation-ready")
    drafted_count = _as_int(state_counts.get("drafted"))
    skipped_count = _as_int(state_counts.get("skipped"))
    return {
        "schema": "tokenclaw.request_shape_crunch_canary_stage_rollup_selection.v1",
        "status": "drafted" if drafted_count else "no-activation-ready-drafts",
        "source_schema": CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        "cohort_count": len(rows),
        "activation_ready_cohort_count": activation_ready_count,
        "drafted_count": drafted_count,
        "skipped_count": skipped_count,
        "stage_action_limit": stage_action_limit,
        "skipped_reason_breakdown": _breakdown(skipped_reasons),
        "state_breakdown": _breakdown(state_counts),
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "policy_files_written": False,
        "provider_calls_made": 0,
        "managed_server_calls_made": 0,
        "rows": rows[:50],
        "privacy": _crunch_opportunity_privacy(),
    }


def request_shape_crunch_canary_lifecycle(action: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    conditions = action.get("conditions") if isinstance(action.get("conditions"), dict) else {}
    cohort_id = str(action.get("cohort_id") or _crunch_canary_cohort_id(conditions))
    policy_id = str(action.get("policy_id") or _crunch_canary_policy_id(cohort_id))
    mismatch = [
        key
        for key, expected in conditions.items()
        if expected is not None and features.get(key) is not None and features.get(key) != expected
    ]
    base = {
        "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "rule_group": public_label(action.get("rule_group") or action.get("candidate_rule") or "repeated-context-conservative", "repeated-context-conservative"),
        "source_evidence_schema": public_label(action.get("source_evidence_schema"), "unknown"),
        "source_evidence_schemas": _public_label_list(action.get("source_evidence_schemas")),
        "projected_saved_chars": _as_int(action.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(action.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(action.get("projected_saved_usd")), 8),
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "metadata_only": True,
    }
    if mismatch:
        return {
            **base,
            "status": "skipped",
            "cohort": "skipped",
            "reason": "cohort-mismatch",
            "mismatched_conditions": sorted(public_label(item, "unknown") for item in mismatch),
        }

    rollout = _bounded_fraction(action.get("rollout_fraction", action.get("canary_fraction")), DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(action.get("holdout_fraction"), DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    unit = str(features.get("request_fingerprint") or features.get("cohort_sample_id") or stable_json({
        key: features.get(key)
        for key in sorted(conditions)
        if features.get(key) is not None
    }))
    material = stable_json({"policy_id": policy_id, "cohort_id": cohort_id, "unit": unit})
    bucket = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if bucket < rollout:
        status = "applied"
        cohort = "canary_applied"
        reason = "selected-canary"
    elif bucket < rollout + holdout:
        status = "holdout"
        cohort = "canary_holdout"
        reason = "selected-holdout"
    else:
        status = "skipped"
        cohort = "skipped"
        reason = "outside-canary-and-holdout"
    return {
        **base,
        "status": status,
        "cohort": cohort,
        "reason": reason,
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "bucket": round(bucket, 8),
        "cohort_key_hash": "sha256:" + hashlib.sha256(stable_json({
            "policy_id": policy_id,
            "cohort_id": cohort_id,
            "unit": unit,
        }).encode("utf-8")).hexdigest(),
    }


def apply_request_shape_crunch_canary_action(
    action: dict[str, Any],
    *,
    rules_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if action.get("schema") != CRUNCH_CANARY_ACTION_SCHEMA:
        errors.append({"path": "$.schema", "message": "unsupported request-shape crunch canary action schema"})
    if action.get("action_type") != "stage-local-repeated-context-crunch-canary":
        errors.append({"path": "$.action_type", "message": "unsupported request-shape crunch canary action type"})
    if action.get("target_local_policy") != "crunch_rules":
        errors.append({"path": "$.target_local_policy", "message": "request-shape crunch canary must target crunch_rules"})
    privacy = action.get("privacy") if isinstance(action.get("privacy"), dict) else {}
    safety = action.get("safety_gates") if isinstance(action.get("safety_gates"), dict) else {}
    for key in ("raw_prompts_included", "provider_bodies_included", "request_ids_included", "session_ids_included"):
        if privacy.get(key) or safety.get(key):
            errors.append({"path": f"$.privacy.{key}", "message": "request-shape crunch canary action is not metadata-only"})
    if _as_float(action.get("projected_saved_tokens")) <= 0 and _as_float(action.get("projected_saved_chars")) <= 0:
        errors.append({"path": "$.projected_saved_tokens", "message": "request-shape crunch canary needs positive projected savings"})

    path = Path(rules_path)
    existing: dict[str, Any] = {}
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing = loaded if isinstance(loaded, dict) else {}
    if errors:
        return {
            "schema": CRUNCH_CANARY_APPLY_SCHEMA,
            "ok": False,
            "dry_run": bool(dry_run),
            "wrote_policy_files": False,
            "rules_path_included": False,
            "errors": errors,
            "privacy": _crunch_opportunity_privacy(),
        }

    policy_id = str(action.get("policy_id") or _crunch_canary_policy_id(str(action.get("cohort_id") or "")))
    canary_rule = {
        "id": policy_id,
        "enabled": True,
        "policy_source": "local-manual",
        "cohort_id": action.get("cohort_id"),
        "source_evidence_schema": action.get("source_evidence_schema"),
        "source_evidence_schemas": _public_label_list(action.get("source_evidence_schemas")),
        "local_only_reason": public_label(action.get("local_only_reason"), "file-backed-local-policy-no-managed-dependency"),
        "evidence_blocker_codes": _public_label_list(action.get("evidence_blocker_codes")),
        "conditions": action.get("conditions") if isinstance(action.get("conditions"), dict) else {},
        "rollout": {
            "schema": "tokenclaw.request_shape_crunch_canary_rollout.v1",
            "canary_enabled": True,
            "canary_fraction": _bounded_fraction(action.get("rollout_fraction"), DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION),
            "holdout_fraction": _bounded_fraction(action.get("holdout_fraction"), DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION),
            "canary_salt": policy_id,
            "canary_unit": "request_shape_cohort",
        },
        "projected_saved_chars": _as_int(action.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(action.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(action.get("projected_saved_usd")), 6),
        "safety_gates": action.get("safety_gates") if isinstance(action.get("safety_gates"), dict) else {},
        "lifecycle_metadata": action.get("lifecycle_metadata") if isinstance(action.get("lifecycle_metadata"), dict) else {},
        "rollback_metadata": action.get("rollback_metadata") if isinstance(action.get("rollback_metadata"), dict) else {},
        "privacy": _crunch_opportunity_privacy(),
        "staged_at": utc_now(),
    }
    updated = dict(existing)
    section = updated.get("request_shape_repeated_context_canaries")
    if not isinstance(section, dict):
        section = {}
    rules = section.get("rules") if isinstance(section.get("rules"), list) else []
    kept = [rule for rule in rules if not (isinstance(rule, dict) and rule.get("id") == policy_id)]
    section.update({
        "enabled": True,
        "schema": "tokenclaw.request_shape_repeated_context_canaries.v1",
        "rules": kept + [canary_rule],
    })
    updated["request_shape_repeated_context_canaries"] = section
    if not dry_run:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")

    return {
        "schema": CRUNCH_CANARY_APPLY_SCHEMA,
        "ok": True,
        "dry_run": bool(dry_run),
        "wrote_policy_files": not dry_run,
        "target_local_policy": "crunch_rules",
        "policy_id": policy_id,
        "cohort_id": action.get("cohort_id"),
        "canary_fraction": canary_rule["rollout"]["canary_fraction"],
        "holdout_fraction": canary_rule["rollout"]["holdout_fraction"],
        "rollback_metadata": canary_rule["rollback_metadata"],
        "rules_path_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def apply_request_shape_crunch_canary_actions(
    actions: list[dict[str, Any]],
    *,
    rules_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a bounded batch of staged repeated-context crunch canary actions.

    Each action is written via apply_request_shape_crunch_canary_action against the
    same local rules file, so cohorts already represented (by policy_id) are upserted
    in place rather than duplicated. Safe to call repeatedly: re-applying an
    already-applied action is idempotent.
    """
    results = [
        apply_request_shape_crunch_canary_action(action, rules_path=rules_path, dry_run=dry_run)
        for action in actions
        if isinstance(action, dict)
    ]
    applied_count = sum(1 for result in results if bool(result.get("ok")))
    return {
        "schema": CRUNCH_CANARY_APPLY_BATCH_SCHEMA,
        "ok": bool(results) and all(bool(result.get("ok")) for result in results),
        "dry_run": bool(dry_run),
        "wrote_policy_files": any(bool(result.get("wrote_policy_files")) for result in results),
        "target_local_policy": "crunch_rules",
        "applied_count": applied_count,
        "failed_count": len(results) - applied_count,
        "policy_ids": [result.get("policy_id") for result in results if result.get("policy_id")],
        "cohort_ids": [result.get("cohort_id") for result in results if result.get("cohort_id")],
        "results": results,
        "rules_path_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def _request_shape_crunch_policy_apply_error(
    *,
    dry_run: bool,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema": CRUNCH_POLICY_DECISION_APPLY_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "wrote_policy_files": False,
        "rules_path_included": False,
        "errors": errors,
        "privacy": _crunch_opportunity_privacy(),
    }


def _request_shape_crunch_policy_apply_decision(
    decision_report: dict[str, Any],
    *,
    decision_id: str | None = None,
) -> dict[str, Any] | None:
    decisions = decision_report.get("decisions") if isinstance(decision_report.get("decisions"), list) else []
    if decision_id:
        for item in decisions:
            if isinstance(item, dict) and item.get("decision_id") == decision_id:
                return item
    top = decision_report.get("top_decision") if isinstance(decision_report.get("top_decision"), dict) else None
    if top is not None:
        return top
    for item in decisions:
        if isinstance(item, dict):
            return item
    return None


def _crunch_policy_decision_application_fingerprint(decision: dict[str, Any]) -> str:
    metrics = decision.get("metrics") if isinstance(decision.get("metrics"), dict) else {}
    coverage = metrics.get("coverage") if isinstance(metrics.get("coverage"), dict) else {}
    payload = {
        "schema": "tokenclaw.request_shape_crunch_policy_decision_application_fingerprint.v1",
        "decision_id": decision.get("decision_id"),
        "decision": decision.get("decision"),
        "policy_id": decision.get("policy_id"),
        "cohort_id": decision.get("cohort_id"),
        "graduation_decision": decision.get("graduation_decision"),
        "applied_count": _as_int(metrics.get("applied_count")),
        "holdout_count": _as_int(metrics.get("holdout_count")),
        "observed_count": _as_int(coverage.get("observed_count")),
        "skipped_count": _as_int(coverage.get("skipped_count")),
        "observed_saved_tokens": _as_int(metrics.get("observed_saved_tokens")),
        "observed_saved_usd": round(_as_float(metrics.get("observed_saved_usd")), 8),
        "error_rate_delta": round(_as_float(metrics.get("error_rate_delta")), 6),
        "retry_rate_delta": round(_as_float(metrics.get("retry_rate_delta")), 6),
        "fallback_rate_delta": round(_as_float(metrics.get("fallback_rate_delta")), 6),
        "safety_stop_count": _as_int(metrics.get("safety_stop_count")),
        "rollback_count": _as_int(metrics.get("rollback_count")),
    }
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"request-shape-crunch-policy-apply:{digest}"


def _crunch_policy_decision_full_rollout_fingerprint(decision: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "schema": "tokenclaw.request_shape_crunch_full_rollout_application_fingerprint.v1",
                "decision_id": decision.get("decision_id"),
                "application_fingerprint": _crunch_policy_decision_application_fingerprint(decision),
                "policy_id": decision.get("policy_id"),
                "cohort_id": decision.get("cohort_id"),
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"request-shape-crunch-full-rollout:{digest}"


def _crunch_policy_decision_legacy_metadata_matches(existing: dict[str, Any], decision: dict[str, Any]) -> bool:
    metrics = decision.get("metrics") if isinstance(decision.get("metrics"), dict) else {}
    checks = {
        "applied_count": _as_int(metrics.get("applied_count")),
        "holdout_count": _as_int(metrics.get("holdout_count")),
        "observed_saved_tokens": _as_int(metrics.get("observed_saved_tokens")),
        "observed_saved_usd": round(_as_float(metrics.get("observed_saved_usd")), 8),
        "error_rate_delta": round(_as_float(metrics.get("error_rate_delta")), 6),
        "retry_rate_delta": round(_as_float(metrics.get("retry_rate_delta")), 6),
        "fallback_rate_delta": round(_as_float(metrics.get("fallback_rate_delta")), 6),
        "safety_stop_state": public_label(decision.get("safety_stop_state") or "none", "none"),
    }
    for key, value in checks.items():
        if isinstance(value, float):
            if round(_as_float(existing.get(key)), 8 if key == "observed_saved_usd" else 6) != value:
                return False
        elif existing.get(key) != value:
            return False
    return True


def _widened_fraction(current: Any, *, widen_fraction: float, max_canary_fraction: float, holdout_fraction: float) -> float:
    current_value = _bounded_fraction(current, DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    step = _bounded_fraction(widen_fraction, DEFAULT_CRUNCH_CANARY_WIDEN_FRACTION)
    max_fraction = _bounded_fraction(max_canary_fraction, DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION)
    max_fraction = min(max_fraction, max(0.0, 1.0 - holdout_fraction))
    return round(min(max_fraction, current_value + step), 6)


def apply_request_shape_crunch_policy_decision(
    decision_report: dict[str, Any],
    *,
    rules_path: str | Path,
    dry_run: bool = False,
    decision_id: str | None = None,
    widen_fraction: float = DEFAULT_CRUNCH_CANARY_WIDEN_FRACTION,
    max_canary_fraction: float = DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION,
    promote_full_rollout: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if decision_report.get("schema") != CRUNCH_POLICY_DECISION_SCHEMA:
        errors.append({"path": "$.schema", "message": "unsupported request-shape crunch policy decision schema"})
    decision = _request_shape_crunch_policy_apply_decision(decision_report, decision_id=decision_id)
    if decision is None:
        errors.append({"path": "$.top_decision", "message": "missing request-shape crunch policy decision"})
        return _request_shape_crunch_policy_apply_error(dry_run=dry_run, errors=errors)
    if decision_id and decision.get("decision_id") != decision_id:
        errors.append({"path": "$.decision_id", "message": "requested decision id was not found"})
    if decision.get("decision") != "widen":
        errors.append({"path": "$.decision", "message": "only widen decisions can update local crunch rollout"})
    if not decision.get("promotion_allowed"):
        errors.append({"path": "$.promotion_allowed", "message": "widen decision is not promotion allowed"})
    rollback_metadata = decision.get("rollback_metadata") if isinstance(decision.get("rollback_metadata"), dict) else {}
    if not rollback_metadata.get("present"):
        errors.append({"path": "$.rollback_metadata", "message": "widen decision requires rollback metadata"})
    privacy = decision.get("privacy") if isinstance(decision.get("privacy"), dict) else {}
    rollback_privacy = rollback_metadata.get("privacy") if isinstance(rollback_metadata.get("privacy"), dict) else {}
    for key in ("raw_prompts_included", "provider_bodies_included", "request_ids_included", "session_ids_included", "cache_keys_included", "tool_payloads_included"):
        if privacy.get(key) or rollback_privacy.get(key):
            errors.append({"path": f"$.privacy.{key}", "message": "request-shape crunch policy decision is not metadata-only"})

    policy_id = str(decision.get("policy_id") or "")
    cohort_id = str(decision.get("cohort_id") or "")
    decision_identifier = str(decision.get("decision_id") or decision_id or "")
    if not policy_id:
        errors.append({"path": "$.policy_id", "message": "widen decision is missing target policy id"})
    if not cohort_id:
        errors.append({"path": "$.cohort_id", "message": "widen decision is missing target cohort id"})
    if not decision_identifier:
        errors.append({"path": "$.decision_id", "message": "widen decision is missing decision id"})

    path = Path(rules_path)
    existing: dict[str, Any] = {}
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing = loaded if isinstance(loaded, dict) else {}
    section = existing.get("request_shape_repeated_context_canaries") if isinstance(existing, dict) else None
    rules = section.get("rules") if isinstance(section, dict) and isinstance(section.get("rules"), list) else []
    target_index: int | None = None
    target_rule: dict[str, Any] | None = None
    for index, item in enumerate(rules):
        if not isinstance(item, dict):
            continue
        if item.get("id") == policy_id or item.get("policy_id") == policy_id or item.get("cohort_id") == cohort_id:
            target_index = index
            target_rule = item
            break
    if target_rule is None:
        errors.append({"path": "$.request_shape_repeated_context_canaries.rules", "message": "matching staged crunch canary rule was not found"})
    if errors:
        return _request_shape_crunch_policy_apply_error(dry_run=dry_run, errors=errors)

    assert target_index is not None
    assert target_rule is not None
    rollout = target_rule.get("rollout") if isinstance(target_rule.get("rollout"), dict) else {}
    previous_canary = _bounded_fraction(rollout.get("canary_fraction", rollout.get("fraction")), DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(rollout.get("holdout_fraction"), DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION)
    safety_gates = target_rule.get("safety_gates") if isinstance(target_rule.get("safety_gates"), dict) else {}
    rule_max_rollout = _bounded_fraction(
        safety_gates.get("max_rollout_fraction", max_canary_fraction),
        _bounded_fraction(max_canary_fraction, DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION),
    )
    existing_policy_decision = target_rule.get("policy_decision") if isinstance(target_rule.get("policy_decision"), dict) else {}
    application_fingerprint = _crunch_policy_decision_application_fingerprint(decision)
    full_rollout_fingerprint = _crunch_policy_decision_full_rollout_fingerprint(decision)
    existing_application_fingerprint = str(existing_policy_decision.get("application_fingerprint") or "")
    existing_full_rollout_fingerprint = str(existing_policy_decision.get("full_rollout_fingerprint") or "")
    existing_full_rollout = (
        existing_policy_decision.get("decision") == "promote-full"
        and (
            existing_full_rollout_fingerprint == full_rollout_fingerprint
            or existing_application_fingerprint == application_fingerprint
        )
    )
    already_applied = (
        existing_policy_decision.get("decision_id") == decision_identifier
        and (
            existing_application_fingerprint == application_fingerprint
            or (
                not existing_application_fingerprint
                and _crunch_policy_decision_legacy_metadata_matches(existing_policy_decision, decision)
            )
        )
    )
    full_rollout_ready = (
        promote_full_rollout
        and previous_canary > 0.0
        and rule_max_rollout > 0.0
        and previous_canary >= rule_max_rollout
    )
    widened_canary = (
        1.0
        if full_rollout_ready
        else previous_canary
        if already_applied
        else _widened_fraction(
            previous_canary,
            widen_fraction=widen_fraction,
            max_canary_fraction=max_canary_fraction,
            holdout_fraction=holdout,
        )
    )
    updated_rule = dict(target_rule)
    updated_rollout = dict(rollout)
    if full_rollout_ready:
        updated_rollout.update({
            "schema": "tokenclaw.request_shape_crunch_canary_rollout.v1",
            "canary_enabled": False,
            "full_rollout_enabled": True,
            "full_rollout_fraction": 1.0,
            "canary_fraction": 1.0,
            "holdout_fraction": 0.0,
            "canary_salt": str(updated_rollout.get("canary_salt") or policy_id),
            "canary_unit": str(updated_rollout.get("canary_unit") or "request_shape_cohort"),
        })
    else:
        updated_rollout.update({
            "schema": "tokenclaw.request_shape_crunch_canary_rollout.v1",
            "canary_enabled": True,
            "full_rollout_enabled": False,
            "full_rollout_fraction": 0.0,
            "canary_fraction": widened_canary,
            "holdout_fraction": holdout,
            "canary_salt": str(updated_rollout.get("canary_salt") or policy_id),
            "canary_unit": str(updated_rollout.get("canary_unit") or "request_shape_cohort"),
        })
    updated_safety_gates = dict(safety_gates)
    updated_safety_gates.update({
        "metadata_only": True,
        "aggregate_only": True,
        "local_file_backed": True,
        "local_only": True,
        "holdout_required": False if full_rollout_ready else holdout > 0,
        "max_rollout_fraction": 1.0
        if full_rollout_ready
        else max(widened_canary, _as_float(updated_safety_gates.get("max_rollout_fraction"))),
        "previous_max_rollout_fraction": rule_max_rollout if full_rollout_ready else _as_float(updated_safety_gates.get("previous_max_rollout_fraction")),
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "tool_payloads_included": False,
    })
    metrics = decision.get("metrics") if isinstance(decision.get("metrics"), dict) else {}
    updated_rule.update({
        "id": policy_id,
        "enabled": True,
        "policy_source": public_label(decision.get("policy_source") or target_rule.get("policy_source") or "local-manual", "local-manual"),
        "cohort_id": cohort_id,
        "rollout": updated_rollout,
        "safety_gates": updated_safety_gates,
        "policy_decision": {
            "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
            "decision_id": decision_identifier,
            "application_fingerprint": application_fingerprint,
            "full_rollout_fingerprint": full_rollout_fingerprint if full_rollout_ready else None,
            "source_evidence_schema": CRUNCH_ACTIVATION_EVIDENCE_SCHEMA if full_rollout_ready else CRUNCH_POLICY_DECISION_SCHEMA,
            "source_policy_decision_schema": CRUNCH_POLICY_DECISION_SCHEMA if full_rollout_ready else None,
            "decision": "promote-full" if full_rollout_ready else "widen",
            "graduation_decision": "promote-full"
            if full_rollout_ready
            else public_label(decision.get("graduation_decision") or "widen", "widen"),
            "applied_count": _as_int(metrics.get("applied_count")),
            "holdout_count": _as_int(metrics.get("holdout_count")),
            "observed_saved_tokens": _as_int(metrics.get("observed_saved_tokens")),
            "observed_saved_usd": round(_as_float(metrics.get("observed_saved_usd")), 8),
            "error_rate_delta": round(_as_float(metrics.get("error_rate_delta")), 6),
            "retry_rate_delta": round(_as_float(metrics.get("retry_rate_delta")), 6),
            "fallback_rate_delta": round(_as_float(metrics.get("fallback_rate_delta")), 6),
            "safety_stop_state": public_label(decision.get("safety_stop_state") or "none", "none"),
            "previous_canary_fraction": previous_canary,
            "widened_canary_fraction": widened_canary,
            "full_rollout_fraction": 1.0 if full_rollout_ready else 0.0,
            "holdout_fraction": 0.0 if full_rollout_ready else holdout,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "rollback_metadata": {
            **rollback_metadata,
            "selected_decision": "promote-full" if full_rollout_ready else "widen",
            "policy_files_written": not dry_run,
            "target_policy_id": policy_id,
            "target_cohort_id": cohort_id,
            "target_local_rule_file": "crunch_rules.yaml",
            "privacy": _crunch_opportunity_privacy(),
        },
        "widened_at": utc_now(),
        "privacy": _crunch_opportunity_privacy(),
    })
    updated_rules = list(rules)
    updated_rules[target_index] = updated_rule
    updated_section = dict(section) if isinstance(section, dict) else {}
    updated_section.update({
        "enabled": True,
        "schema": "tokenclaw.request_shape_repeated_context_canaries.v1",
        "rules": updated_rules,
    })
    updated = dict(existing)
    updated["request_shape_repeated_context_canaries"] = updated_section
    updated_rule["policy_decision"] = {
        key: value
        for key, value in updated_rule["policy_decision"].items()
        if value not in (None, "", [], {})
    }
    should_write = not dry_run and not (existing_full_rollout if full_rollout_ready else already_applied)
    if should_write:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    status = (
        "already-full-rollout"
        if full_rollout_ready and existing_full_rollout
        else "full-rollout-applied"
        if full_rollout_ready and not dry_run
        else "full-rollout-drafted"
        if full_rollout_ready
        else "already-applied"
        if already_applied
        else "applied"
        if not dry_run
        else "drafted"
    )

    return {
        "schema": CRUNCH_POLICY_DECISION_APPLY_SCHEMA,
        "ok": True,
        "status": status,
        "dry_run": bool(dry_run),
        "wrote_policy_files": should_write,
        "already_applied": existing_full_rollout if full_rollout_ready else already_applied,
        "full_rollout_ready": full_rollout_ready,
        "full_rollout_applied": full_rollout_ready and not dry_run,
        "target_local_policy": "crunch_rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "decision_id": decision_identifier,
        "application_fingerprint": application_fingerprint,
        "full_rollout_fingerprint": full_rollout_fingerprint if full_rollout_ready else None,
        "previous_canary_fraction": previous_canary,
        "canary_fraction": widened_canary,
        "full_rollout_fraction": 1.0 if full_rollout_ready else 0.0,
        "holdout_fraction": 0.0 if full_rollout_ready else holdout,
        "previous_max_rollout_fraction": rule_max_rollout,
        "canary_enabled": not full_rollout_ready,
        "widen_fraction": round(_bounded_fraction(widen_fraction, DEFAULT_CRUNCH_CANARY_WIDEN_FRACTION), 6),
        "max_canary_fraction": round(_bounded_fraction(max_canary_fraction, DEFAULT_CRUNCH_CANARY_MAX_WIDENED_FRACTION), 6),
        "rollback_metadata": updated_rule["rollback_metadata"],
        "widened_cohort": {
            "schema": "tokenclaw.request_shape_crunch_policy_decision_widened_cohort.v1",
            "policy_id": policy_id,
            "cohort_id": cohort_id,
            "decision_id": decision_identifier,
            "application_fingerprint": application_fingerprint,
            "previous_canary_fraction": previous_canary,
            "canary_fraction": widened_canary,
            "full_rollout_fraction": 1.0 if full_rollout_ready else 0.0,
            "holdout_fraction": 0.0 if full_rollout_ready else holdout,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "rules_path_included": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def _request_shape_crunch_follow_up(
    *,
    status: str,
    report_key: str,
    evidence_schema: str,
    candidate_count: int,
    matched_count: int,
    rows_considered: int,
    recommended_action_count: int,
    canary_applied_rows: int,
    canary_holdout_rows: int,
    canary_safety_stopped_rows: int,
    projected_saved_chars: int,
    projected_saved_tokens: int,
    projected_saved_usd: float,
    top_blocker: str | None,
    missing_measurements: list[str],
) -> dict[str, Any]:
    if canary_safety_stopped_rows > 0:
        activation_state = "blocked"
        next_action = "review-repeated-context-crunch-canary-safety-stop"
        activation_mode = "review-required"
        blocker = "canary-safety-stopped"
        no_op_reason = "matching-repeated-context-crunch-canary-safety-stopped"
    elif recommended_action_count > 0:
        activation_state = "activation-ready"
        next_action = "stage-repeated-context-crunch-canary"
        activation_mode = "canary-candidate"
        blocker = None
        no_op_reason = None
    elif canary_applied_rows > 0 or canary_holdout_rows > 0:
        activation_state = "measurement-required"
        next_action = "measure-repeated-context-crunch-canary-impact"
        activation_mode = "staged-canary-measurement"
        blocker = "missing-crunch-canary-impact-measurement"
        no_op_reason = "matching-repeated-context-crunch-canary-already-staged"
    elif status == "no-repeated-context-crunch-cohorts":
        activation_state = "missing-evidence"
        next_action = "rank-repeated-context-crunch-dry-run"
        activation_mode = "evidence-required"
        blocker = "repeated-context-crunch-cohorts"
        no_op_reason = blocker
    elif missing_measurements:
        activation_state = "missing-measurement"
        next_action = "inspect-crunch-coverage-and-projection"
        activation_mode = "evidence-required"
        blocker = missing_measurements[0]
        no_op_reason = blocker
    else:
        activation_state = "ranked"
        next_action = "rank-crunch-opportunity-follow-up"
        activation_mode = "review-required"
        blocker = top_blocker
        no_op_reason = blocker

    follow_up_missing = list(dict.fromkeys(str(item) for item in missing_measurements if str(item or "").strip()))
    if activation_state == "measurement-required" and blocker not in follow_up_missing:
        follow_up_missing.append(blocker)
    if activation_state == "blocked" and blocker not in follow_up_missing:
        follow_up_missing.append(blocker)
    canary_already_staged = canary_applied_rows > 0 or canary_holdout_rows > 0
    savings_status = (
        "projected-savings-ranked"
        if projected_saved_chars > 0 or projected_saved_tokens > 0 or projected_saved_usd > 0
        else "no-positive-projection"
    )

    return {
        "schema": "tokenclaw.request_shape_crunch_activation_follow_up.v1",
        "status": status,
        "savings_status": savings_status,
        "report_key": report_key,
        "evidence_schema": evidence_schema,
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "rows_considered": rows_considered,
        "activation_state": activation_state,
        "activation_mode": activation_mode,
        "next_action": next_action,
        "target_local_policy": "crunch_rules",
        "policy_section": "crunch",
        "local_file_backed": True,
        "projected_saved_chars": projected_saved_chars,
        "projected_saved_tokens": projected_saved_tokens,
        "projected_saved_usd": round(projected_saved_usd, 6),
        "recommended_action_count": recommended_action_count,
        "canary_applied_rows": canary_applied_rows,
        "canary_holdout_rows": canary_holdout_rows,
        "canary_safety_stopped_rows": canary_safety_stopped_rows,
        "canary_already_staged": canary_already_staged,
        "canary_already_applied": canary_applied_rows > 0,
        "no_op_reason": no_op_reason,
        "duplicate_suppression": {
            "schema": "tokenclaw.request_shape_crunch_follow_up_duplicate_suppression.v1",
            "suppresses_new_stage_action": recommended_action_count == 0 and (canary_already_staged or canary_safety_stopped_rows > 0),
            "reason": no_op_reason,
            "matching_local_policy": "crunch_rules" if canary_already_staged or canary_safety_stopped_rows > 0 else None,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "top_blocker": blocker or top_blocker,
        "missing_measurements": follow_up_missing,
        "privacy": _crunch_opportunity_privacy(),
    }


def _repeated_context_drill_signal(row: dict[str, Any], *, row_count: int, candidate_status: str) -> dict[str, Any]:
    """Aggregate-only ranking signal that promotes a repeated-context rollup drill.

    Surfaces the levers the crunch dry-run ranks repeated-context cohorts on:
    sample count, a bucketed/median text-size signal, the plateau/repetition signal
    derived from how many of the sampled calls are repeats, projected saved tokens,
    and whether a canary safety blocker demotes the cohort. No raw bodies, prompts,
    or candidate identifiers are read.
    """

    sample_count = max(0, _as_int(row_count))
    text_bucket = str(row.get("text_bucket") or "unknown")
    token_bucket = str(row.get("token_bucket") or "unknown")
    row_input_tokens = _as_int(row.get("successful_input_tokens") or row.get("input_tokens"))
    median_input_tokens = int(round(row_input_tokens / sample_count)) if sample_count > 0 else 0
    # Plateau/repetition signal: fraction of the sampled calls that are repeats of the
    # shared shape. Approaches 1.0 as repeated context accumulates, 0.0 for a one-off.
    repetition_signal = round((sample_count - 1) / sample_count, 6) if sample_count > 0 else 0.0
    large_context = text_bucket in REPEATED_CONTEXT_TEXT_BUCKETS or token_bucket in LARGE_CONTEXT_TOKEN_BUCKETS
    repeat_evidence = sample_count >= REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES
    safety_blocked = candidate_status == "safety-blocked"
    projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    projected_usd = _as_float(row.get("projected_crunch_savings_usd"))
    observed_tokens = _as_int(row.get("current_crunch_tokens_saved"))

    if safety_blocked:
        drill_state = "safety-blocked"
    elif not repeat_evidence:
        drill_state = "no-repeat"
    elif not large_context:
        drill_state = "too-small"
    elif projected_tokens <= 0 and observed_tokens <= 0:
        drill_state = "missing-projection"
    else:
        drill_state = "ranked"

    return {
        "schema": "tokenclaw.request_shape_repeated_context_drill_signal.v1",
        "drill_state": drill_state,
        "sample_count": sample_count,
        "text_bucket": public_label(text_bucket, "unknown"),
        "token_bucket": public_label(token_bucket, "unknown"),
        "text_size_ordinal": REPEATED_CONTEXT_TEXT_BUCKET_ORDINALS.get(text_bucket, 0),
        "median_sample_input_tokens": median_input_tokens,
        "repetition_signal": repetition_signal,
        "large_context": large_context,
        "repeat_evidence": repeat_evidence,
        "safety_blocked": safety_blocked,
        "projected_saved_tokens": projected_tokens,
        "projected_saved_usd": round(projected_usd, 6),
        "metadata_only": True,
        "aggregate_only": True,
    }


def _repeated_context_drill_rank_key(signal: dict[str, Any]) -> tuple[Any, ...]:
    # Rank order from issue #797: sample count, median text size, plateau/repetition
    # signal, projected saved tokens; safety blockers demote a cohort below all others.
    return (
        not bool(signal.get("safety_blocked")),
        _as_int(signal.get("sample_count")),
        _as_int(signal.get("text_size_ordinal")),
        _as_int(signal.get("median_sample_input_tokens")),
        _as_float(signal.get("repetition_signal")),
        _as_int(signal.get("projected_saved_tokens")),
        round(_as_float(signal.get("projected_saved_usd")), 6),
    )


def _build_repeated_context_crunch_drill(
    cohorts: list[dict[str, Any]],
    *,
    crunch_row_count: int,
    matched_count: int,
    has_cohort_filter: bool,
) -> dict[str, Any]:
    """Deterministic drill block that either ranks repeated large-context cohorts or
    explains which threshold (source / repeat / size) is missing.

    This graduates zero-row crunch drills into ranked repeated-context cohorts when
    local rollups meet the sample and size thresholds, and otherwise emits an explicit
    ``no-source`` / ``no-repeat`` / ``too-small`` state. Policy writes stay disabled;
    only ranked evidence and a dry-run state are produced.
    """

    signals = [
        cohort.get("repeated_context_drill_signal")
        for cohort in cohorts
        if isinstance(cohort.get("repeated_context_drill_signal"), dict)
    ]
    repeat_evidence_count = sum(1 for signal in signals if signal.get("repeat_evidence"))
    large_context_count = sum(1 for signal in signals if signal.get("large_context"))
    repeated_large_context = [
        signal
        for signal in signals
        if signal.get("repeat_evidence") and signal.get("large_context")
    ]
    no_repeat_count = sum(1 for signal in signals if not signal.get("repeat_evidence"))
    too_small_count = sum(
        1 for signal in signals if signal.get("repeat_evidence") and not signal.get("large_context")
    )
    safety_blocked_count = sum(1 for signal in signals if signal.get("safety_blocked"))

    if not signals or matched_count <= 0:
        state = "no-source"
        missing_threshold = "matching-repeated-context-source-traffic" if has_cohort_filter else "repeated-context-source-traffic"
        next_action = "collect-source-traffic"
    elif not repeated_large_context:
        if repeat_evidence_count <= 0:
            state = "no-repeat"
            missing_threshold = f"repeated-context-min-samples-{REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES}"
            next_action = "collect-repeated-context-samples"
        else:
            state = "too-small"
            missing_threshold = "repeated-context-large-context-size"
            next_action = "collect-large-context-samples"
    else:
        state = "ranked"
        missing_threshold = None
        next_action = "rank-repeated-context-crunch-cohorts"

    ranked = sorted(repeated_large_context, key=_repeated_context_drill_rank_key, reverse=True)
    ranked_cohorts: list[dict[str, Any]] = []
    cohort_by_signal_id = {id(cohort.get("repeated_context_drill_signal")): cohort for cohort in cohorts}
    for rank, signal in enumerate(ranked, start=1):
        cohort = cohort_by_signal_id.get(id(signal), {})
        ranked_cohorts.append(
            {
                "schema": "tokenclaw.request_shape_repeated_context_drill_cohort.v1",
                "rank": rank,
                "cohort_id": cohort.get("cohort_id"),
                "policy_id": cohort.get("policy_id"),
                "readiness": cohort.get("readiness"),
                "candidate_status": cohort.get("candidate_status"),
                "sample_count": _as_int(signal.get("sample_count")),
                "text_bucket": signal.get("text_bucket"),
                "median_sample_input_tokens": _as_int(signal.get("median_sample_input_tokens")),
                "repetition_signal": _as_float(signal.get("repetition_signal")),
                "projected_saved_tokens": _as_int(signal.get("projected_saved_tokens")),
                "projected_saved_usd": round(_as_float(signal.get("projected_saved_usd")), 6),
                "safety_blocked": bool(signal.get("safety_blocked")),
            }
        )

    projected_tokens = sum(_as_int(item["projected_saved_tokens"]) for item in ranked_cohorts)
    projected_usd = round(sum(_as_float(item["projected_saved_usd"]) for item in ranked_cohorts), 6)

    return {
        "schema": "tokenclaw.request_shape_repeated_context_crunch_drill.v1",
        "state": state,
        "missing_threshold": missing_threshold,
        "next_action": next_action,
        "crunch_row_count": max(0, _as_int(crunch_row_count)),
        "matched_count": max(0, _as_int(matched_count)),
        "candidate_cohort_count": len(signals),
        "ranked_cohort_count": len(ranked_cohorts),
        "repeat_evidence_cohort_count": repeat_evidence_count,
        "large_context_cohort_count": large_context_count,
        "no_repeat_cohort_count": no_repeat_count,
        "too_small_cohort_count": too_small_count,
        "safety_blocked_cohort_count": safety_blocked_count,
        "min_sample_threshold": REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES,
        "large_context_text_buckets": sorted(REPEATED_CONTEXT_TEXT_BUCKETS),
        "large_context_token_buckets": sorted(LARGE_CONTEXT_TOKEN_BUCKETS),
        "projected_saved_tokens": projected_tokens,
        "projected_saved_usd": projected_usd,
        "ranked_cohorts": ranked_cohorts,
        "policy_files_written": False,
        "privacy": _crunch_opportunity_privacy(),
    }


def _activation_candidate_queue_crunch_rollups(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(source, dict):
        return [], {
            "source_schema": ROLLUP_ROW_SCHEMA,
            "source_queue_status": None,
            "source_queue_entry_count": 0,
            "source_queue_crunch_entry_count": 0,
        }

    entries = source.get("entries") if isinstance(source.get("entries"), list) else []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if public_label(entry.get("local_action_family"), "unknown") != "crunch":
            continue
        sample_count = _as_int(entry.get("sample_count") or entry.get("row_count"))
        projected_tokens = _as_int(entry.get("projected_saved_tokens"))
        projected_chars = _as_int(entry.get("projected_saved_chars")) or projected_tokens * 4
        estimated_input_tokens = (
            _as_int(entry.get("successful_input_tokens"))
            or _as_int(entry.get("input_tokens"))
            or (
                int(round(projected_tokens / REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE))
                if projected_tokens > 0 and REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE > 0
                else 0
            )
        )
        blocker_codes = [
            public_label(item, "unknown")
            for item in entry.get("blocker_codes") or []
            if public_label(item, "unknown") != "unknown"
        ]
        rows.append(
            {
                "schema": ROLLUP_ROW_SCHEMA,
                "source_schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_ENTRY_SCHEMA,
                "source_activation_fingerprint": public_label(entry.get("fingerprint"), "unknown"),
                "source_activation_candidate_rank": _as_int(entry.get("rank")),
                "provider_family": public_label(entry.get("provider_family"), "unknown"),
                "source_surface": public_label(entry.get("source_surface"), "unknown"),
                "endpoint": public_label(entry.get("endpoint"), "unknown"),
                "app_family": public_label(entry.get("app_family"), "unknown"),
                "requested_model_family": public_label(entry.get("requested_model_family"), "unknown"),
                "routed_model_family": public_label(entry.get("routed_model_family"), "unknown"),
                "category": public_label(entry.get("category"), "unknown"),
                "workflow_phase": public_label(entry.get("workflow_phase"), "unknown"),
                "stream": bool(entry.get("stream")),
                "has_tools": bool(entry.get("has_tools")),
                "cache_status": public_label(entry.get("cache_status"), "unknown"),
                "routing_status": public_label(entry.get("routing_status"), "unknown"),
                "text_bucket": public_label(entry.get("text_bucket"), "unknown"),
                "token_bucket": public_label(entry.get("token_bucket"), "unknown"),
                "candidate_work_classes": ["crunch", "repeated_context"],
                "candidate_families": ["crunch_candidate"],
                "blocker_codes": blocker_codes,
                "row_count": sample_count,
                "sample_count": sample_count,
                "successful_input_tokens": estimated_input_tokens,
                "input_tokens": estimated_input_tokens,
                "projected_crunch_tokens_saved": projected_tokens,
                "projected_crunch_chars_saved": projected_chars,
                "projected_crunch_savings_usd": round(_as_float(entry.get("projected_savings_usd")), 8),
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "crunch_canary_lifecycle": entry.get("crunch_canary_lifecycle")
                if isinstance(entry.get("crunch_canary_lifecycle"), dict)
                else {},
                "freshness_state": public_label(entry.get("freshness_state"), "unknown"),
                "activation_candidate_readiness_state": public_label(entry.get("readiness_state"), "unknown"),
                "activation_candidate_next_action": public_label(entry.get("recommended_next_action"), "unknown"),
                "privacy": _shape_follow_up_privacy(),
            }
        )
    return rows, {
        "source_schema": public_label(source.get("schema"), LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA),
        "source_queue_status": public_label(source.get("status"), "unknown"),
        "source_queue_entry_count": len(entries),
        "source_queue_crunch_entry_count": len(rows),
    }


def _preview_gate_verified(row: dict[str, Any]) -> bool:
    gate = row.get("managed_preview_gate") if isinstance(row.get("managed_preview_gate"), dict) else {}
    status = public_label(row.get("preview_verification_status") or gate.get("status"), "")
    decision = public_label(row.get("preview_verification_decision") or gate.get("decision"), "")
    return bool(
        row.get("preview_verified")
        or gate.get("verified")
        or status in {"preview-verified", "verified", "agreed"}
        or decision in {"ready", "review-ready"}
    )


def _local_executor_gate_passed(row: dict[str, Any]) -> bool:
    gate = row.get("managed_preview_gate") if isinstance(row.get("managed_preview_gate"), dict) else {}
    local_gate = gate.get("local_executor_gate") if isinstance(gate.get("local_executor_gate"), dict) else {}
    if local_gate.get("passed") is not None:
        return bool(local_gate.get("passed"))
    if row.get("local_executor_gate_passed") is not None:
        return bool(row.get("local_executor_gate_passed"))
    if row.get("policy_files_written") or row.get("provider_calls_made"):
        return False
    return bool(_preview_gate_verified(row))


def _local_activation_successor_crunch_rollups(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(source, dict):
        return [], {
            "source_schema": "tokenclaw.local_activation_next_action_queue.v1",
            "source_queue_status": None,
            "source_queue_entry_count": 0,
            "source_queue_crunch_entry_count": 0,
            "source_queue_preview_verified_crunch_entry_count": 0,
        }

    actions = source.get("successor_actions") if isinstance(source.get("successor_actions"), list) else []
    if not actions:
        actions = source.get("entries") if isinstance(source.get("entries"), list) else []
    rows: list[dict[str, Any]] = []
    verified_crunch_count = 0
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        if public_label(entry.get("local_action_family") or entry.get("lever"), "unknown") != "crunch":
            continue
        if not _preview_gate_verified(entry) or not _local_executor_gate_passed(entry):
            continue
        if public_label(entry.get("target_local_policy_section"), "unknown") not in {"crunch.rules", "unknown"}:
            continue
        if public_label(entry.get("target_local_rule_file"), "crunch_rules.yaml") != "crunch_rules.yaml":
            continue
        if _as_int(entry.get("safety_stop_count")) > 0 or _as_int(entry.get("rollback_count")) > 0:
            continue
        verified_crunch_count += 1
        sample_count = _as_int(entry.get("sample_count") or entry.get("row_count"))
        projected_tokens = _as_int(entry.get("projected_saved_tokens") or entry.get("observed_saved_tokens"))
        projected_chars = _as_int(entry.get("projected_saved_chars")) or projected_tokens * 4
        estimated_input_tokens = (
            _as_int(entry.get("successful_input_tokens"))
            or _as_int(entry.get("input_tokens"))
            or (
                int(round(projected_tokens / REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE))
                if projected_tokens > 0 and REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE > 0
                else 0
            )
        )
        blocker_codes = [
            public_label(item, "unknown")
            for item in entry.get("blocker_codes") or []
            if public_label(item, "unknown") != "unknown"
        ]
        rows.append(
            {
                "schema": ROLLUP_ROW_SCHEMA,
                "source_schema": "tokenclaw.local_activation_successor_action.v1",
                "source_activation_fingerprint": public_label(
                    entry.get("source_fingerprint") or entry.get("fingerprint"),
                    "unknown",
                ),
                "source_activation_candidate_rank": _as_int(
                    entry.get("source_queue_rank") or entry.get("rank") or entry.get("source_ledger_rank")
                ),
                "provider_family": public_label(entry.get("provider_family"), "unknown"),
                "source_surface": public_label(entry.get("source_surface"), "unknown"),
                "endpoint": public_label(entry.get("endpoint"), "unknown"),
                "app_family": public_label(entry.get("app_family"), "unknown"),
                "requested_model_family": public_label(entry.get("requested_model_family"), "unknown"),
                "routed_model_family": public_label(entry.get("routed_model_family") or entry.get("target_model_family"), "unknown"),
                "category": public_label(entry.get("category"), "unknown"),
                "workflow_phase": public_label(entry.get("workflow_phase"), "unknown"),
                "stream": bool(entry.get("stream")),
                "has_tools": bool(entry.get("has_tools")),
                "cache_status": public_label(entry.get("cache_status"), "unknown"),
                "routing_status": public_label(entry.get("routing_status"), "unknown"),
                "text_bucket": public_label(entry.get("text_bucket"), "unknown"),
                "token_bucket": public_label(entry.get("token_bucket"), "unknown"),
                "candidate_work_classes": ["crunch", "repeated_context"],
                "candidate_families": ["crunch_candidate"],
                "blocker_codes": blocker_codes,
                "row_count": sample_count,
                "sample_count": sample_count,
                "successful_input_tokens": estimated_input_tokens,
                "input_tokens": estimated_input_tokens,
                "projected_crunch_tokens_saved": projected_tokens,
                "projected_crunch_chars_saved": projected_chars,
                "projected_crunch_savings_usd": round(
                    _as_float(entry.get("projected_saved_usd") or entry.get("projected_savings_usd") or entry.get("observed_saved_usd")),
                    8,
                ),
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "crunch_canary_lifecycle": entry.get("crunch_canary_lifecycle")
                if isinstance(entry.get("crunch_canary_lifecycle"), dict)
                else {},
                "freshness_state": public_label(entry.get("freshness_state"), "fresh"),
                "activation_candidate_readiness_state": "activation-ready",
                "activation_candidate_next_action": public_label(
                    entry.get("recommended_next_action") or entry.get("next_action"),
                    "stage-repeated-context-crunch-canary",
                ),
                "preview_verified": True,
                "preview_verification_status": public_label(entry.get("preview_verification_status"), "preview-verified"),
                "privacy": _shape_follow_up_privacy(),
            }
        )
    return rows, {
        "source_schema": public_label(source.get("schema"), "tokenclaw.local_activation_next_action_queue.v1"),
        "source_queue_status": public_label(source.get("status"), "unknown"),
        "source_queue_entry_count": len(actions),
        "source_queue_crunch_entry_count": len(rows),
        "source_queue_preview_verified_crunch_entry_count": verified_crunch_count,
    }


def _crunch_dry_run_source_rows(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(source, dict) and source.get("schema") == LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA:
        return _activation_candidate_queue_crunch_rollups(source)
    if isinstance(source, dict) and source.get("schema") == "tokenclaw.local_activation_next_action_queue.v1":
        return _local_activation_successor_crunch_rollups(source)
    if isinstance(source, list):
        rows = [row for row in source if isinstance(row, dict)]
        source_schemas = {
            public_label(row.get("source_schema") or row.get("schema"), "unknown")
            for row in rows
            if public_label(row.get("source_schema") or row.get("schema"), "unknown") != "unknown"
        }
        source_schema = source_schemas.pop() if len(source_schemas) == 1 else ROLLUP_ROW_SCHEMA
        fresh_plateau_rows = [
            row
            for row in rows
            if (row.get("source_schema") or row.get("schema")) == CONTEXT_PLATEAU_ROLLUP_ROW_SCHEMA
            and public_label(row.get("source_rollup_freshness_state") or row.get("freshness_state"), "unknown") == "fresh"
        ]
        return rows, {
            "source_schema": source_schema,
            "source_queue_status": None,
            "source_queue_entry_count": 0,
            "source_queue_crunch_entry_count": 0,
            "fresh_plateau_rollup_count": len(fresh_plateau_rows),
        }
    return [], {
        "source_schema": ROLLUP_ROW_SCHEMA,
        "source_queue_status": None,
        "source_queue_entry_count": 0,
        "source_queue_crunch_entry_count": 0,
        "fresh_plateau_rollup_count": 0,
    }


def build_request_shape_crunch_opportunity_dry_run(
    rollups: list[dict[str, Any]] | dict[str, Any],
    *,
    limit: int = 25,
    rollout_fraction: float = DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
    holdout_fraction: float = DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION,
    existing_canary_rules: list[dict[str, Any]] | None = None,
    max_canary_actions: int = DEFAULT_CRUNCH_CANARY_MAX_NEW_STAGE_ACTIONS,
    cohort_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_rows, source_metadata = _crunch_dry_run_source_rows(rollups)
    crunch_rows = [
        row
        for row in source_rows
        if isinstance(row, dict)
        and (
            "crunch" in {str(item) for item in row.get("candidate_work_classes") or []}
            or "repeated_context" in {str(item) for item in row.get("candidate_work_classes") or []}
        )
    ]
    cohorts: list[dict[str, Any]] = []
    readiness_counts: dict[str, int] = {}
    candidate_status_counts: dict[str, int] = {}
    policy_write_status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    work_class_counts: dict[str, int] = {}
    projected_tokens = 0
    projected_chars = 0
    projected_savings = 0.0
    observed_tokens = 0
    observed_chars = 0
    observed_savings = 0.0
    canary_applied_rows = 0
    canary_holdout_rows = 0
    canary_safety_stopped_rows = 0
    existing_rules = existing_canary_rules if isinstance(existing_canary_rules, list) else []
    filter_conditions = _request_shape_crunch_canary_rule_conditions(cohort_filter)
    matched_count = 0

    for row in crunch_rows:
        if filter_conditions and not _request_shape_crunch_cohort_matches_conditions(row, filter_conditions):
            continue
        decision = _shape_crunch_decision(row)
        row_count = _as_int(row.get("row_count") or row.get("count"))
        source_evidence_schema = row.get("source_schema") or row.get("schema") or ROLLUP_ROW_SCHEMA
        source_rollup_freshness = public_label(
            row.get("source_rollup_freshness_state") or row.get("freshness_state"),
            "unknown",
        )
        classes = sorted({public_label(item, "unknown") for item in row.get("candidate_work_classes") or []})
        readiness = str(decision["readiness"])
        reason = str(decision["reason"])
        candidate_status = _shape_crunch_candidate_status(decision)
        policy_write_status = _shape_crunch_policy_write_status(candidate_status)
        row_projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
        row_projected_chars = _as_int(row.get("projected_crunch_chars_saved"))
        row_projected_savings = _as_float(row.get("projected_crunch_savings_usd"))
        row_observed_tokens = _as_int(row.get("current_crunch_tokens_saved"))
        row_observed_chars = _as_int(row.get("current_crunch_chars_saved"))
        row_observed_savings = _as_float(row.get("current_crunch_savings_usd"))
        lifecycle = row.get("crunch_canary_lifecycle") if isinstance(row.get("crunch_canary_lifecycle"), dict) else {}
        cohort_id = str(lifecycle.get("cohort_id") or _crunch_canary_cohort_id(row))
        policy_id = str(lifecycle.get("policy_id") or _crunch_canary_policy_id(cohort_id))
        applied_count = _as_int(lifecycle.get("applied_count"))
        holdout_count = _as_int(lifecycle.get("holdout_count"))
        safety_stopped_count = _as_int(lifecycle.get("safety_stopped_count"))

        _increment(readiness_counts, readiness)
        _increment(candidate_status_counts, candidate_status, row_count)
        _increment(policy_write_status_counts, policy_write_status, row_count)
        _increment(reason_counts, reason)
        for blocker in decision.get("blockers") or []:
            _increment(blocker_counts, blocker)
        for work_class in classes:
            _increment(work_class_counts, work_class, row_count)
        if readiness in {"measurement-ready", "canary-staged", "canary-applied", "canary-holdout"}:
            projected_tokens += row_projected_tokens
            projected_chars += row_projected_chars
            projected_savings += row_projected_savings
            observed_tokens += row_observed_tokens
            observed_chars += row_observed_chars
            observed_savings += row_observed_savings
        canary_applied_rows += applied_count
        canary_holdout_rows += holdout_count
        canary_safety_stopped_rows += safety_stopped_count
        matched_count += row_count

        cohort = {
                "schema": "tokenclaw.request_shape_crunch_opportunity_cohort.v1",
                "cohort_id": cohort_id,
                "policy_id": policy_id,
                "fingerprint": public_id(
                    stable_json(
                        {
                            "schema": "tokenclaw.request_shape_crunch_opportunity_cohort.v1",
                            "cohort_id": cohort_id,
                            "source_activation_fingerprint": row.get("source_activation_fingerprint"),
                            "provider_family": row.get("provider_family"),
                            "source_surface": row.get("source_surface"),
                            "endpoint": row.get("endpoint"),
                            "category": row.get("category"),
                            "workflow_phase": row.get("workflow_phase"),
                            "text_bucket": row.get("text_bucket"),
                            "token_bucket": row.get("token_bucket"),
                        }
                    ),
                    prefix="crunch-opportunity",
                ),
                "source_evidence_schema": source_evidence_schema,
                "source_rollup_freshness_state": source_rollup_freshness,
                "fresh_plateau_rollup": (
                    source_evidence_schema == CONTEXT_PLATEAU_ROLLUP_ROW_SCHEMA
                    and source_rollup_freshness == "fresh"
                ),
                "context_plateau_pair_count": _as_int(row.get("context_plateau_pair_count")),
                "context_plateau_session_count": _as_int(row.get("context_plateau_session_count")),
                "source_activation_fingerprint": row.get("source_activation_fingerprint"),
                "source_activation_candidate_rank": row.get("source_activation_candidate_rank"),
                "source_activation_candidate_next_action": row.get("activation_candidate_next_action"),
                "readiness": readiness,
                "candidate_status": candidate_status,
                "policy_write_status": policy_write_status,
                "policy_write_required": policy_write_status == "policy-write-required",
                "reason": reason,
                "blockers": decision.get("blockers") or [],
                "blocker_codes": decision.get("blockers") or [],
                "evidence_blocker_codes": _public_label_list(row.get("blocker_codes")),
                "provider_family": row.get("provider_family"),
                "source_surface": row.get("source_surface"),
                "endpoint": row.get("endpoint"),
                "category": row.get("category"),
                "workflow_phase": row.get("workflow_phase"),
                "stream": bool(row.get("stream")),
                "has_tools": bool(row.get("has_tools")),
                "cache_status": row.get("cache_status"),
                "routing_status": row.get("routing_status"),
                "text_bucket": row.get("text_bucket"),
                "token_bucket": row.get("token_bucket"),
                "row_count": row_count,
                "sample_count": row_count,
                "work_classes": classes,
                "current_conservative_tokens_saved": row_observed_tokens,
                "current_conservative_chars_saved": row_observed_chars,
                "current_conservative_savings_usd": round(row_observed_savings, 6),
                "projected_saved_tokens": row_projected_tokens,
                "projected_saved_chars": row_projected_chars,
                "projected_saved_usd": round(row_projected_savings, 6),
                "crunch_canary_lifecycle": lifecycle,
                "candidate_rule": "repeated-context-conservative-dry-run",
                "target_local_policy": "crunch_rules",
                "target_local_policy_section": "crunch.rules",
                "target_local_rule_file": "crunch_rules.yaml",
                "estimate_basis": (
                    "metadata-only projection using aggregate input tokens, repeated-shape row count, "
                    f"and {REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE:.0%} conservative input-token reduction"
                ),
                "aggregate_only": True,
                "privacy": _crunch_opportunity_privacy(),
            }
        cohort["duplicate_suppression"] = _request_shape_crunch_cohort_duplicate_suppression(cohort, existing_rules)
        cohort["repeated_context_drill_signal"] = _repeated_context_drill_signal(
            row,
            row_count=row_count,
            candidate_status=candidate_status,
        )
        cohorts.append(cohort)

    cohorts.sort(
        key=lambda item: (
            item.get("readiness") == "measurement-ready",
            _as_float(item.get("projected_saved_usd")) + _as_float(item.get("current_conservative_savings_usd")),
            _as_int(item.get("projected_saved_tokens")) + _as_int(item.get("current_conservative_tokens_saved")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(cohorts, start=1):
        row["rank"] = rank

    stageable_cohorts = [
        cohort
        for cohort in cohorts
        if cohort.get("readiness") == "measurement-ready"
        and not (
            isinstance(cohort.get("duplicate_suppression"), dict)
            and bool(cohort["duplicate_suppression"].get("suppresses_new_stage_action"))
        )
    ]
    stage_action_limit = max(0, _as_int(max_canary_actions) or DEFAULT_CRUNCH_CANARY_MAX_NEW_STAGE_ACTIONS)
    recommended_actions = [
        _request_shape_crunch_canary_action(
            cohort,
            candidate_count=len(cohorts),
            rollout_fraction=rollout_fraction,
            holdout_fraction=holdout_fraction,
        )
        for cohort in stageable_cohorts[:stage_action_limit]
    ]
    duplicate_suppressed_count = sum(
        1
        for cohort in cohorts
        if isinstance(cohort.get("duplicate_suppression"), dict)
        and bool(cohort["duplicate_suppression"].get("suppresses_new_stage_action"))
    )
    active_max_rollout_suppressed_count = sum(
        1
        for cohort in cohorts
        if isinstance(cohort.get("duplicate_suppression"), dict)
        and bool(cohort["duplicate_suppression"].get("suppresses_new_stage_action"))
        and bool(cohort["duplicate_suppression"].get("active_at_max_rollout"))
    )
    blocker_breakdown = _breakdown(blocker_counts)
    top_blocker = blocker_breakdown[0]["value"] if blocker_breakdown else None
    status = "ranked" if projected_tokens > 0 or observed_tokens > 0 or projected_savings > 0 or observed_savings > 0 else "no-positive-crunch-opportunity"
    if canary_safety_stopped_rows:
        status = "canary-safety-stopped"
    elif canary_applied_rows or canary_holdout_rows:
        status = "canary-staged"
    if not cohorts:
        status = "no-repeated-context-crunch-cohorts"
    missing = []
    if not cohorts:
        missing.append("matching-repeated-context-crunch-cohorts" if filter_conditions else "repeated-context-crunch-cohorts")
    if projected_tokens <= 0 and observed_tokens <= 0 and projected_savings <= 0 and observed_savings <= 0:
        missing.append("positive-observed-or-projected-savings")
    no_op_reason = None
    if status in {"no-positive-crunch-opportunity", "no-repeated-context-crunch-cohorts"}:
        no_op_reason = (
            "no-repeated-context-crunch-cohorts"
            if not cohorts
            else top_blocker
            or (
                "too-small"
                if candidate_status_counts.get("too-small")
                else "missing-observed-savings"
                if candidate_status_counts.get("missing-observed-savings")
                else "no-positive-crunch-opportunity"
            )
        )
    activation_follow_up = _request_shape_crunch_follow_up(
        status=status,
        report_key="request_shape_crunch_opportunity",
        evidence_schema=CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        candidate_count=len(cohorts),
        matched_count=matched_count,
        rows_considered=matched_count,
        recommended_action_count=len(recommended_actions),
        canary_applied_rows=canary_applied_rows,
        canary_holdout_rows=canary_holdout_rows,
        canary_safety_stopped_rows=canary_safety_stopped_rows,
        projected_saved_chars=projected_chars,
        projected_saved_tokens=projected_tokens,
        projected_saved_usd=projected_savings,
        top_blocker=top_blocker,
        missing_measurements=missing,
    )
    activation_follow_up["duplicate_suppression"].update(
        {
            "suppressed_existing_cohort_count": duplicate_suppressed_count,
            "active_max_rollout_suppressed_cohort_count": active_max_rollout_suppressed_count,
            "stageable_unsuppressed_cohort_count": len(stageable_cohorts),
            "newly_staged_cohort_count": len(recommended_actions),
            "suppresses_new_stage_action": bool(
                activation_follow_up["duplicate_suppression"].get("suppresses_new_stage_action")
            )
            or (not recommended_actions and duplicate_suppressed_count > 0),
            "reason": activation_follow_up["duplicate_suppression"].get("reason")
            or (
                "matching-repeated-context-crunch-canary-already-staged-in-local-policy"
                if duplicate_suppressed_count
                else None
            ),
            "matching_local_policy": activation_follow_up["duplicate_suppression"].get("matching_local_policy")
            or ("crunch_rules" if duplicate_suppressed_count else None),
        }
    )

    repeated_context_drill = _build_repeated_context_crunch_drill(
        cohorts,
        crunch_row_count=len(crunch_rows),
        matched_count=matched_count,
        has_cohort_filter=bool(filter_conditions),
    )
    fresh_plateau_candidate_count = sum(1 for cohort in cohorts if bool(cohort.get("fresh_plateau_rollup")))
    fresh_plateau_stageable_count = sum(1 for cohort in stageable_cohorts if bool(cohort.get("fresh_plateau_rollup")))
    has_recommended_action = len(recommended_actions) > 0
    has_holdout_plan = any(_as_float(action.get("holdout_fraction")) > 0 for action in recommended_actions)
    has_safety_stop_fields = any(
        isinstance(action.get("safety_gates"), dict)
        and bool(action["safety_gates"].get("records_applied_holdout_skipped_safety_stopped_fallback_rollback"))
        and isinstance(action.get("rollback_metadata"), dict)
        for action in recommended_actions
    )
    has_target_rule_section = any(
        action.get("target_local_policy_section") == "crunch.rules"
        and action.get("target_local_rule_file") == "crunch_rules.yaml"
        for action in recommended_actions
    )
    has_duplicate_suppression = all(
        isinstance(action.get("duplicate_suppression"), dict) for action in recommended_actions
    ) if recommended_actions else False
    acceptance = {
        "schema": "tokenclaw.request_shape_crunch_opportunity_acceptance.v1",
        "emits_ranked_candidate_from_fresh_plateau_rollups": fresh_plateau_stageable_count > 0 and has_recommended_action,
        "fresh_plateau_candidate_count": fresh_plateau_candidate_count,
        "fresh_plateau_stageable_count": fresh_plateau_stageable_count,
        "has_projected_saved_tokens": projected_tokens > 0,
        "has_projected_saved_usd": projected_savings > 0,
        "has_holdout_plan": has_holdout_plan,
        "has_safety_stop_fields": has_safety_stop_fields,
        "has_target_local_rule_section": has_target_rule_section,
        "has_duplicate_suppression": has_duplicate_suppression,
        "metadata_only": True,
        "aggregate_only": True,
        "policy_files_written": False,
    }

    return {
        "schema": CRUNCH_OPPORTUNITY_DRY_RUN_SCHEMA,
        "status": status,
        "source_schema": source_metadata["source_schema"],
        "source_queue_status": source_metadata["source_queue_status"],
        "source_queue_entry_count": source_metadata["source_queue_entry_count"],
        "source_queue_crunch_entry_count": source_metadata["source_queue_crunch_entry_count"],
        "summary": {
            "candidate_count": len(cohorts),
            "matched_count": matched_count,
            "rows_considered": matched_count,
            "source_schema": source_metadata["source_schema"],
            "source_queue_status": source_metadata["source_queue_status"],
            "source_queue_entry_count": source_metadata["source_queue_entry_count"],
            "source_queue_crunch_entry_count": source_metadata["source_queue_crunch_entry_count"],
            "fresh_plateau_rollup_count": _as_int(source_metadata.get("fresh_plateau_rollup_count")),
            "fresh_plateau_candidate_count": fresh_plateau_candidate_count,
            "fresh_plateau_stageable_count": fresh_plateau_stageable_count,
            "measurement_ready_cohort_count": readiness_counts.get("measurement-ready", 0),
            "canary_staged_cohort_count": readiness_counts.get("canary-staged", 0),
            "canary_applied_cohort_count": readiness_counts.get("canary-applied", 0),
            "canary_holdout_cohort_count": readiness_counts.get("canary-holdout", 0),
            "canary_safety_stopped_cohort_count": readiness_counts.get("canary-safety-stopped", 0),
            "candidate_status_counts": _breakdown(candidate_status_counts),
            "policy_write_status_counts": _breakdown(policy_write_status_counts),
            "policy_write_required_count": candidate_status_counts.get("candidate", 0),
            "canary_applied_rows": canary_applied_rows,
            "canary_holdout_rows": canary_holdout_rows,
            "canary_safety_stopped_rows": canary_safety_stopped_rows,
            "recommended_action_count": len(recommended_actions),
            "newly_staged_cohort_count": len(recommended_actions),
            "stage_action_limit": stage_action_limit,
            "duplicate_suppressed_cohort_count": duplicate_suppressed_count,
            "active_max_rollout_suppressed_cohort_count": active_max_rollout_suppressed_count,
            "stageable_unsuppressed_cohort_count": len(stageable_cohorts),
            "skipped_cohort_count": readiness_counts.get("skipped", 0),
            "current_conservative_tokens_saved": observed_tokens,
            "current_conservative_chars_saved": observed_chars,
            "current_conservative_savings_usd": round(observed_savings, 6),
            "projected_saved_tokens": projected_tokens,
            "projected_saved_chars": projected_chars,
            "projected_saved_usd": round(projected_savings, 6),
            "top_blocker_code": top_blocker,
            "no_op_reason": no_op_reason,
            "target_local_policy_section": "crunch.rules" if cohorts else None,
            "target_local_rule_file": "crunch_rules.yaml" if cohorts else None,
            "activation_state": activation_follow_up["activation_state"],
            "top_next_action": activation_follow_up["next_action"],
            "repeated_context_drill_state": repeated_context_drill["state"],
            "repeated_context_ranked_cohort_count": repeated_context_drill["ranked_cohort_count"],
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
        },
        "acceptance": acceptance,
        "activation_follow_up": activation_follow_up,
        "repeated_context_drill": repeated_context_drill,
        "recommended_actions": recommended_actions,
        "readiness_breakdown": _breakdown(readiness_counts),
        "candidate_status_breakdown": _breakdown(candidate_status_counts),
        "policy_write_status_breakdown": _breakdown(policy_write_status_counts),
        "reason_breakdown": _breakdown(reason_counts),
        "blocker_reason_breakdown": blocker_breakdown,
        "work_class_breakdown": _breakdown(work_class_counts),
        "cohorts": cohorts[: max(1, min(_as_int(limit) or 25, 1000))],
        "no_op_reason": no_op_reason,
        "missing_measurements": missing,
        "privacy": _crunch_opportunity_privacy(),
    }


def _cache_replay_successor_privacy() -> dict[str, Any]:
    privacy = _replayability_privacy()
    privacy.update(
        {
            "policy_files_written": False,
            "cache_entries_written": False,
            "cache_apply_actions_emitted": False,
        }
    )
    return privacy


# Activation-queue blocker codes that name a tool-cache dependency invalidation state.
_DEPENDENCY_BLOCKER_TO_FILE_STATUS = {
    "safe-invalidation-evidence-present": "stable",
    "stale-dependency-evidence": "invalidated",
    "unsafe-dependency-evidence": "unsafe",
    "unsafe-tool-calls-without-invalidation": "unsafe",
    "invalidation-evidence-missing": "missing",
    "dependency-evidence-unknown": "unknown",
}


def _cache_successor_file_dependency_status(blocker_set: set[str], *, has_tools: bool) -> str:
    """Map an activation cohort's blocker codes to a file-dependency status.

    Unsafe evidence dominates, then stale, then missing, then explicit unknown.
    Non-tool exact replay has no file dependency to invalidate, so it reports
    ``not-applicable`` and freshness alone drives staleness downstream.
    """
    ranked = ("unsafe", "invalidated", "missing", "stable", "unknown")
    observed: set[str] = set()
    for code in blocker_set:
        status = _DEPENDENCY_BLOCKER_TO_FILE_STATUS.get(code)
        if status:
            observed.add(status)
    for status in ranked:
        if status in observed:
            return status
    if has_tools:
        # A tool-cache cohort with no invalidation signal cannot be trusted.
        return "missing"
    return "not-applicable"


def _cache_successor_evidence_class(
    *,
    decision: str,
    has_tools: bool,
    evidence_stale: bool,
    evidence_unknown: bool,
) -> str:
    """Normalise to the five-value vocabulary the successor queue classifies on."""
    mapping = {
        "stable-dependency-evidence": "stable",
        "stale-risk-blocker": "stale",
        "unsafe-dependency-evidence": "unsafe",
        "missing-dependency-evidence": "missing",
        "unknown-dependency-evidence": "unknown",
    }
    if decision in mapping:
        return mapping[decision]
    # decision == "not-required": exact non-tool replay. Freshness is the only
    # staleness signal, so age-out becomes stale and a present cohort is stable.
    if evidence_stale:
        return "stale"
    if evidence_unknown:
        return "unknown"
    return "stable"


def _cache_successor_repeat_proof(entry: dict[str, Any]) -> dict[str, Any]:
    cache_status = public_label(entry.get("cache_status"), "unknown")
    live_repeat_hits = (
        _as_int(entry.get("observed_hit_count"))
        or _as_int(entry.get("cache_hit_count"))
        or _as_int(entry.get("observed_cache_hits"))
    )
    observed_savings = _as_float(entry.get("observed_savings_usd"))
    live_repeat = cache_status == "hit" or live_repeat_hits > 0
    observed_hit = live_repeat or observed_savings > 0.0
    if live_repeat:
        basis = "live-cache-repeat-observed"
    elif observed_hit:
        basis = "observed-replay-savings"
    else:
        basis = "no-live-repeat-or-observed-hit"
    return {
        "schema": "tokenclaw.request_shape_cache_replay_successor_repeat_proof.v1",
        "live_repeat": live_repeat,
        "observed_hit": observed_hit,
        "has_repeat_proof": bool(live_repeat or observed_hit),
        "live_repeat_hit_count": live_repeat_hits,
        "observed_savings_usd": round(observed_savings, 6),
        "basis": basis,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_successor_decision(
    *,
    evidence_class: str,
    has_tools: bool,
    stream: bool,
    evidence_stale: bool,
    repeat_proof: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one cache-family activation cohort to a successor action.

    Unsafe and missing evidence never produce a cache apply action or cache
    entry.  Stale evidence reobserves before any rollback.  Tool and streaming
    replay stay disabled unless invalidation evidence is stable *and* a live
    repeat or observed-hit proof exists.
    """
    has_proof = bool(repeat_proof.get("has_repeat_proof"))
    tool_or_stream = bool(has_tools or stream)
    base = {
        "tool_cache_replay_enabled": False,
        "streaming_replay_enabled": False,
        "emits_cache_apply_action": False,
        "cache_entries_written": False,
    }
    if evidence_class == "unsafe":
        return {
            **base,
            "successor_decision": "hold-unsafe-no-apply",
            "successor_next_action": "keep-cache-replay-blocked-unsafe-evidence",
            "successor_reason": "unsafe-tool-calls-without-invalidation",
            "blocker_codes": ["unsafe-dependency-evidence"],
        }
    if evidence_class == "missing":
        return {
            **base,
            "successor_decision": "collect-invalidation-evidence",
            "successor_next_action": "collect-tool-call-cache-invalidation-evidence",
            "successor_reason": "invalidation-evidence-missing",
            "blocker_codes": ["invalidation-evidence-missing"],
        }
    if evidence_class == "unknown":
        return {
            **base,
            "successor_decision": "classify-dependency-evidence",
            "successor_next_action": "collect-cache-replay-dependency-evidence",
            "successor_reason": "dependency-evidence-unknown",
            "blocker_codes": ["dependency-evidence-unknown"],
        }
    if evidence_class == "stale" or evidence_stale:
        # Refresh stale evidence before deciding to roll back or widen.
        return {
            **base,
            "successor_decision": "reobserve-before-rollback",
            "successor_next_action": "reobserve-cache-replay-evidence",
            "successor_reason": "stale-cache-replay-evidence-older-than-max-age",
            "blocker_codes": ["stale-cache-replay-evidence-older-than-max-age"],
        }
    # evidence_class == "stable" and fresh.
    if tool_or_stream:
        if has_proof:
            return {
                "tool_cache_replay_enabled": bool(has_tools),
                "streaming_replay_enabled": bool(stream),
                "emits_cache_apply_action": True,
                "cache_entries_written": False,
                "successor_decision": "apply-stable-with-repeat-proof",
                "successor_next_action": "stage-cache-replay-canary",
                "successor_reason": "stable-invalidation-evidence-with-repeat-proof",
                "blocker_codes": [],
            }
        return {
            **base,
            "successor_decision": "keep-tool-streaming-replay-disabled",
            "successor_next_action": "collect-cache-replay-repeat-proof",
            "successor_reason": "stable-evidence-without-live-repeat-or-observed-hit",
            "blocker_codes": ["missing-cache-replay-repeat-proof"],
        }
    if has_proof:
        return {
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
            "emits_cache_apply_action": True,
            "cache_entries_written": False,
            "successor_decision": "apply-stable-with-repeat-proof",
            "successor_next_action": "stage-cache-replay-canary",
            "successor_reason": "stable-exact-replay-with-repeat-proof",
            "blocker_codes": [],
        }
    return {
        **base,
        "successor_decision": "stage-canary-for-repeat-proof",
        "successor_next_action": "stage-cache-replay-canary",
        "successor_reason": "stable-exact-replay-pending-repeat-proof",
        "blocker_codes": ["missing-cache-replay-repeat-proof"],
    }


def _activation_candidate_queue_cache_entries(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(source, dict):
        return [], {
            "source_schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA,
            "source_queue_status": None,
            "source_queue_entry_count": 0,
            "source_queue_cache_entry_count": 0,
        }
    entries = source.get("entries") if isinstance(source.get("entries"), list) else []
    cache_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and public_label(entry.get("local_action_family"), "unknown") == "cache"
    ]
    return cache_entries, {
        "source_schema": public_label(source.get("schema"), LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA),
        "source_queue_status": public_label(source.get("status"), "unknown"),
        "source_queue_entry_count": len(entries),
        "source_queue_cache_entry_count": len(cache_entries),
    }


def _cache_successor_source_entries(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(source, dict) and source.get("schema") == LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA:
        return _activation_candidate_queue_cache_entries(source)
    if isinstance(source, dict) and isinstance(source.get("entries"), list):
        return _activation_candidate_queue_cache_entries(source)
    if isinstance(source, list):
        cache_entries = [
            entry
            for entry in source
            if isinstance(entry, dict)
            and public_label(entry.get("local_action_family"), "unknown") == "cache"
        ]
        return cache_entries, {
            "source_schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA,
            "source_queue_status": None,
            "source_queue_entry_count": len(source),
            "source_queue_cache_entry_count": len(cache_entries),
        }
    return [], {
        "source_schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA,
        "source_queue_status": None,
        "source_queue_entry_count": 0,
        "source_queue_cache_entry_count": 0,
    }


def build_request_shape_cache_replay_successor_dry_run(
    source: list[dict[str, Any]] | dict[str, Any],
    *,
    limit: int = 25,
    max_age_hours: float = DEFAULT_ROLLUP_SNAPSHOT_MAX_AGE_HOURS,
) -> dict[str, Any]:
    """Consume cache-family activation candidates and resolve stale-evidence successors.

    For every cache-family entry in the ranked local activation candidate queue
    this classifies the tool/non-tool dependency evidence as ``stable``,
    ``stale``, ``unsafe``, ``unknown`` or ``missing`` and emits exactly one
    metadata-only successor: stale evidence reobserves before rollback, unsafe
    and missing evidence emit no cache apply action and write no cache entry,
    and tool or streaming replay stays disabled unless stable evidence is paired
    with a live-repeat or observed-hit proof.
    """
    cache_entries, source_metadata = _cache_successor_source_entries(source)
    cohorts: list[dict[str, Any]] = []
    evidence_class_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    freshness_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    apply_action_cohorts = 0
    reobserve_cohorts = 0
    unsafe_cohorts = 0
    missing_cohorts = 0
    tool_or_streaming_blocked = 0

    for entry in cache_entries:
        rank = _as_int(entry.get("rank"))
        has_tools = bool(entry.get("has_tools"))
        stream = bool(entry.get("stream"))
        sample_count = _as_int(entry.get("sample_count") or entry.get("row_count"))
        blocker_set = {
            public_label(item, "unknown")
            for item in entry.get("blocker_codes") or []
            if public_label(item, "unknown") != "unknown"
        }
        freshness_state = public_label(entry.get("freshness_state"), "unknown")
        evidence_stale = freshness_state in {"stale", "snapshot-stale", "rollup-stale"}
        evidence_unknown_freshness = freshness_state == "unknown"
        file_dependency_status = _cache_successor_file_dependency_status(blocker_set, has_tools=has_tools)
        dependency_decision = _cache_dependency_evidence_decision(
            file_dependency_status=file_dependency_status if file_dependency_status != "not-applicable" else "stable",
            next_action=public_label(entry.get("recommended_next_action") or entry.get("next_action"), "inspect-local-evidence"),
            requires_invalidation=has_tools,
            has_tools=has_tools,
        )
        evidence_class = _cache_successor_evidence_class(
            decision=str(dependency_decision.get("decision")),
            has_tools=has_tools,
            evidence_stale=evidence_stale,
            evidence_unknown=evidence_unknown_freshness,
        )
        repeat_proof = _cache_successor_repeat_proof(entry)
        decision = _cache_successor_decision(
            evidence_class=evidence_class,
            has_tools=has_tools,
            stream=stream,
            evidence_stale=evidence_stale,
            repeat_proof=repeat_proof,
        )
        cohort_fingerprint = public_id(
            stable_json(
                {
                    "schema": CACHE_REPLAY_SUCCESSOR_COHORT_SCHEMA,
                    "source_activation_fingerprint": public_label(entry.get("fingerprint"), "unknown"),
                    "evidence_class": evidence_class,
                    "successor_next_action": decision["successor_next_action"],
                }
            ),
            prefix="cache-replay-successor",
        )
        cohort = {
            "schema": CACHE_REPLAY_SUCCESSOR_COHORT_SCHEMA,
            "fingerprint": cohort_fingerprint,
            "rank": rank,
            "source_evidence_schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_ENTRY_SCHEMA,
            "source_activation_fingerprint": public_label(entry.get("fingerprint"), "unknown"),
            "source_activation_candidate_rank": rank,
            "source_activation_candidate_next_action": public_label(
                entry.get("recommended_next_action") or entry.get("next_action"), "unknown"
            ),
            "provider_family": public_label(entry.get("provider_family"), "unknown"),
            "source_surface": public_label(entry.get("source_surface"), "unknown"),
            "endpoint": public_label(entry.get("endpoint"), "unknown"),
            "category": public_label(entry.get("category"), "unknown"),
            "workflow_phase": public_label(entry.get("workflow_phase"), "unknown"),
            "stream": stream,
            "has_tools": has_tools,
            "cache_status": public_label(entry.get("cache_status"), "unknown"),
            "text_bucket": public_label(entry.get("text_bucket"), "unknown"),
            "token_bucket": public_label(entry.get("token_bucket"), "unknown"),
            "sample_count": sample_count,
            "row_count": sample_count,
            "projected_hits": _as_int(entry.get("projected_hits")),
            "projected_savings_usd": round(_as_float(entry.get("projected_savings_usd")), 6),
            "observed_savings_usd": round(_as_float(entry.get("observed_savings_usd")), 6),
            "replay_kind": "tool-cache" if has_tools else "streaming-cache" if stream else "non-tool-cache",
            "freshness_state": freshness_state,
            "evidence_stale": evidence_stale,
            "file_dependency_status": file_dependency_status,
            "dependency_evidence_class": evidence_class,
            "dependency_evidence_decision": dependency_decision,
            "repeat_proof": repeat_proof,
            "successor_decision": decision["successor_decision"],
            "successor_next_action": decision["successor_next_action"],
            "successor_reason": decision["successor_reason"],
            "blocker_codes": decision["blocker_codes"],
            "tool_cache_replay_enabled": decision["tool_cache_replay_enabled"],
            "streaming_replay_enabled": decision["streaming_replay_enabled"],
            "emits_cache_apply_action": decision["emits_cache_apply_action"],
            "cache_entries_written": False,
            "policy_files_written": False,
            "max_age_hours": float(max_age_hours),
            "metadata_only": True,
            "aggregate_only": True,
            "privacy": _cache_replay_successor_privacy(),
        }
        cohorts.append(cohort)

        _increment(evidence_class_counts, evidence_class, sample_count)
        _increment(decision_counts, decision["successor_decision"], sample_count)
        _increment(freshness_counts, freshness_state, sample_count)
        _increment(next_action_counts, decision["successor_next_action"], sample_count)
        if decision["emits_cache_apply_action"]:
            apply_action_cohorts += 1
        if decision["successor_decision"] == "reobserve-before-rollback":
            reobserve_cohorts += 1
        if evidence_class == "unsafe":
            unsafe_cohorts += 1
        if evidence_class == "missing":
            missing_cohorts += 1
        if (has_tools or stream) and not decision["emits_cache_apply_action"]:
            tool_or_streaming_blocked += 1

    cohorts.sort(
        key=lambda item: (
            item.get("successor_decision") == "reobserve-before-rollback",
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("sample_count")),
        ),
        reverse=True,
    )
    for index, cohort in enumerate(cohorts, start=1):
        cohort["rank"] = index

    capped = cohorts[: max(1, min(_as_int(limit) or 25, 1000))]
    apply_actions = [
        {
            "schema": "tokenclaw.request_shape_cache_replay_successor_apply_action.v1",
            "source_activation_fingerprint": cohort["source_activation_fingerprint"],
            "successor_next_action": cohort["successor_next_action"],
            "replay_kind": cohort["replay_kind"],
            "tool_cache_replay_enabled": cohort["tool_cache_replay_enabled"],
            "streaming_replay_enabled": cohort["streaming_replay_enabled"],
            "cache_entries_written": False,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        }
        for cohort in capped
        if cohort.get("emits_cache_apply_action")
    ]
    if not cache_entries:
        status = "no-cache-replay-successor-candidates"
    elif reobserve_cohorts:
        status = "reobserve-stale-cache-replay-successors"
    elif apply_actions:
        status = "apply-ready-cache-replay-successors"
    else:
        status = "blocked-cache-replay-successors"
    missing_measurements = [] if cache_entries else ["cache-family-activation-candidates"]
    top = capped[0] if capped else None
    return {
        "schema": CACHE_REPLAY_SUCCESSOR_DRY_RUN_SCHEMA,
        "status": status,
        "source_schema": source_metadata["source_schema"],
        "source_queue_status": source_metadata["source_queue_status"],
        "source_queue_entry_count": source_metadata["source_queue_entry_count"],
        "source_queue_cache_entry_count": source_metadata["source_queue_cache_entry_count"],
        "summary": {
            "cache_successor_cohort_count": len(cohorts),
            "source_queue_status": source_metadata["source_queue_status"],
            "source_queue_entry_count": source_metadata["source_queue_entry_count"],
            "source_queue_cache_entry_count": source_metadata["source_queue_cache_entry_count"],
            "reobserve_cohort_count": reobserve_cohorts,
            "unsafe_cohort_count": unsafe_cohorts,
            "missing_evidence_cohort_count": missing_cohorts,
            "apply_action_cohort_count": apply_action_cohorts,
            "tool_or_streaming_disabled_cohort_count": tool_or_streaming_blocked,
            "top_successor_decision": top.get("successor_decision") if top else None,
            "top_successor_next_action": top.get("successor_next_action") if top else None,
            "top_dependency_evidence_class": top.get("dependency_evidence_class") if top else None,
            "evidence_class_breakdown": _breakdown(evidence_class_counts),
            "successor_decision_breakdown": _breakdown(decision_counts),
            "freshness_breakdown": _breakdown(freshness_counts),
            "next_action_breakdown": _breakdown(next_action_counts),
            "supported_evidence_classes": ["stable", "stale", "unsafe", "unknown", "missing"],
            "cache_apply_actions_emitted": len(apply_actions),
            "cache_entries_written": 0,
            "policy_files_written": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
        },
        "top_cohort": top,
        "cohorts": capped,
        "cache_apply_actions": apply_actions,
        "missing_measurements": missing_measurements,
        "privacy": _cache_replay_successor_privacy(),
    }


def build_request_shape_crunch_canary_stage_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    run_id: str | None = None,
    persist_rollups: bool = False,
    rollout_fraction: float = DEFAULT_CRUNCH_CANARY_ROLLOUT_FRACTION,
    holdout_fraction: float = DEFAULT_CRUNCH_CANARY_HOLDOUT_FRACTION,
    rules_path: str | Path | None = None,
    max_new_canaries: int = DEFAULT_CRUNCH_CANARY_MAX_NEW_STAGE_ACTIONS,
    cohort_filter: dict[str, Any] | None = None,
    source_rollup_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(source_rollup_report, dict):
        rollup_report = source_rollup_report
        if run_id and not rollup_report.get("run_id"):
            rollup_report = {**rollup_report, "run_id": run_id}
    else:
        rollup_report = build_request_shape_rollups_report(
            store_obj,
            limit=limit,
            persist=persist_rollups,
            run_id=run_id,
        )
    existing_canary_rules = _load_request_shape_crunch_canary_rules(rules_path)
    has_cohort_filter = bool(_request_shape_crunch_canary_rule_conditions(cohort_filter))
    follow_up = rollup_report.get("follow_up_candidates") if isinstance(rollup_report.get("follow_up_candidates"), dict) else {}
    activation_queue = (
        follow_up.get("activation_candidate_queue")
        if isinstance(follow_up.get("activation_candidate_queue"), dict)
        else None
    )
    dry_run_source: list[dict[str, Any]] | dict[str, Any]
    rollup_rows = [row for row in rollup_report.get("rollups") or [] if isinstance(row, dict)]
    if rollup_report.get("schema") == "tokenclaw.local_activation_next_action_queue.v1":
        dry_run_source = rollup_report
    elif isinstance(activation_queue, dict) and not rollup_rows:
        dry_run_source = activation_queue
    else:
        dry_run_source = rollup_rows
    dry_run = build_request_shape_crunch_opportunity_dry_run(
        dry_run_source,
        limit=25,
        rollout_fraction=rollout_fraction,
        holdout_fraction=holdout_fraction,
        existing_canary_rules=existing_canary_rules,
        max_canary_actions=max_new_canaries,
        cohort_filter=cohort_filter,
    )
    actions = [
        action
        for action in dry_run.get("recommended_actions") or []
        if isinstance(action, dict) and action.get("schema") == CRUNCH_CANARY_ACTION_SCHEMA
    ]
    cohorts = [cohort for cohort in dry_run.get("cohorts") or [] if isinstance(cohort, dict)]
    stage_action_limit = (dry_run.get("summary") or {}).get("stage_action_limit", max_new_canaries)
    rollup_selection = _request_shape_crunch_stage_rollup_selection_review(
        cohorts=cohorts,
        actions=actions,
        stage_action_limit=_as_int(stage_action_limit) or max_new_canaries,
    )
    top_action = actions[0] if actions else None
    top_cohort = cohorts[0] if cohorts else None
    top_stage_cohort = next(
        (
            cohort
            for cohort in cohorts
            if top_action is not None
            and cohort.get("cohort_id") == top_action.get("cohort_id")
        ),
        None,
    )
    already_staged_cohorts = [
        cohort
        for cohort in cohorts
        if isinstance(cohort.get("duplicate_suppression"), dict)
        and bool(cohort["duplicate_suppression"].get("suppresses_new_stage_action"))
    ]
    top_reported_canary = top_action or (already_staged_cohorts[0] if already_staged_cohorts else top_cohort)
    lifecycle_projections = [
        _request_shape_crunch_canary_lifecycle_projection(
            cohort,
            rollout_fraction=rollout_fraction,
            holdout_fraction=holdout_fraction,
        )
        for cohort in cohorts
    ]
    skipped_or_safety_reasons: dict[str, int] = {}
    for item in lifecycle_projections:
        skipped_or_safety = _as_int(item.get("projected_skipped_count")) + _as_int(item.get("projected_safety_stopped_count"))
        if skipped_or_safety:
            _increment(skipped_or_safety_reasons, item.get("reason") or "unknown", skipped_or_safety)
    stage_lifecycle_projection = {
        "schema": "tokenclaw.request_shape_crunch_canary_stage_lifecycle_projection.v1",
        "cohort_count": len(lifecycle_projections),
        "matched_count": sum(_as_int(item.get("matched_count")) for item in lifecycle_projections),
        "projected_canary_applied_count": sum(_as_int(item.get("projected_canary_applied_count")) for item in lifecycle_projections),
        "projected_canary_holdout_count": sum(_as_int(item.get("projected_canary_holdout_count")) for item in lifecycle_projections),
        "projected_skipped_count": sum(_as_int(item.get("projected_skipped_count")) for item in lifecycle_projections),
        "projected_safety_stopped_count": sum(_as_int(item.get("projected_safety_stopped_count")) for item in lifecycle_projections),
        "projected_fallback_count": sum(_as_int(item.get("projected_fallback_count")) for item in lifecycle_projections),
        "projected_rollback_count": sum(_as_int(item.get("projected_rollback_count")) for item in lifecycle_projections),
        "projected_applied_saved_tokens": sum(_as_int(item.get("projected_applied_saved_tokens")) for item in lifecycle_projections),
        "projected_applied_saved_chars": sum(_as_int(item.get("projected_applied_saved_chars")) for item in lifecycle_projections),
        "projected_applied_saved_usd": round(sum(_as_float(item.get("projected_applied_saved_usd")) for item in lifecycle_projections), 6),
        "skipped_or_safety_reasons": _breakdown(skipped_or_safety_reasons),
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }
    duplicate_suppression = {
        "schema": "tokenclaw.request_shape_crunch_stage_duplicate_suppression_summary.v1",
        "existing_local_rule_count": len(existing_canary_rules),
        "suppressed_existing_cohort_count": (dry_run.get("summary") or {}).get("duplicate_suppressed_cohort_count", 0),
        "active_max_rollout_suppressed_cohort_count": (dry_run.get("summary") or {}).get(
            "active_max_rollout_suppressed_cohort_count",
            0,
        ),
        "stageable_unsuppressed_cohort_count": (dry_run.get("summary") or {}).get("stageable_unsuppressed_cohort_count", 0),
        "newly_staged_cohort_count": len(actions),
        "stage_action_limit": (dry_run.get("summary") or {}).get("stage_action_limit", max_new_canaries),
        "suppresses_new_stage_action": not actions and _as_int((dry_run.get("summary") or {}).get("duplicate_suppressed_cohort_count")) > 0,
        "reason": ((dry_run.get("activation_follow_up") or {}).get("duplicate_suppression") or {}).get("reason"),
        "matching_local_policy": "crunch_rules" if existing_canary_rules else None,
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _crunch_opportunity_privacy(),
    }
    validation_rules_path = _crunch_rules_candidate_paths(rules_path)[0]
    validation = apply_request_shape_crunch_canary_actions(
        actions,
        rules_path=validation_rules_path,
        dry_run=True,
    )
    if actions:
        status = "staged"
        next_action = "apply-local-crunch-canary-after-review"
        reason = "staged-repeated-context-crunch-canary"
        ok = True
    elif (
        _as_int(duplicate_suppression.get("suppressed_existing_cohort_count")) > 0
        and _as_int(duplicate_suppression.get("stageable_unsuppressed_cohort_count")) == 0
        and any(
            isinstance(cohort.get("duplicate_suppression"), dict)
            and cohort["duplicate_suppression"].get("reason")
            == "matching-repeated-context-crunch-canary-already-staged-in-local-policy"
            for cohort in already_staged_cohorts
        )
    ):
        status = "already-staged"
        next_action = "measure-repeated-context-crunch-canary-impact"
        reason = duplicate_suppression.get("reason") or "matching-repeated-context-crunch-canary-already-staged-in-local-policy"
        ok = True
    elif has_cohort_filter and len(cohorts) == 1 and len(already_staged_cohorts) == 1:
        status = "already-staged"
        next_action = "measure-repeated-context-crunch-canary-impact"
        duplicate = already_staged_cohorts[0].get("duplicate_suppression")
        reason = (
            duplicate.get("reason")
            if isinstance(duplicate, dict) and duplicate.get("reason")
            else "matching-repeated-context-crunch-canary-already-staged-in-local-policy"
        )
        ok = True
    else:
        status = "no-stageable-cohort"
        next_action = (dry_run.get("activation_follow_up") or {}).get("next_action") or "rank-repeated-context-crunch-dry-run"
        reason = (dry_run.get("activation_follow_up") or {}).get("top_blocker") or dry_run.get("status") or "no-stageable-cohort"
        ok = False
    reported_canary_count = len(actions) + (len(already_staged_cohorts) if not actions else 0)
    return {
        "schema": CRUNCH_CANARY_STAGE_SCHEMA,
        "status": status,
        "ok": ok,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "run_id": rollup_report.get("run_id"),
        "reason": reason,
        "next_action": next_action,
        "staged_canary_count": len(actions),
        "already_staged_canary_count": len(already_staged_cohorts),
        "reported_canary_count": reported_canary_count,
        "stage_actions": actions,
        "activation_ready_rollup_selection": rollup_selection,
        "top_stage_action": top_action,
        "top_reported_canary": top_reported_canary,
        "top_cohort": top_cohort,
        "top_stage_cohort": top_stage_cohort,
        "duplicate_suppression": duplicate_suppression,
        "validation": validation,
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "stage_lifecycle_projection": stage_lifecycle_projection,
        "cohort_lifecycle_projections": lifecycle_projections[:25],
        "source_report": {
            "schema": rollup_report.get("schema"),
            "window": rollup_report.get("window"),
            "summary": {
                "rows_considered": (rollup_report.get("summary") or {}).get("rows_considered"),
                "rollup_count": (rollup_report.get("summary") or {}).get("rollup_count"),
                "top_next_action": (rollup_report.get("summary") or {}).get("top_next_action"),
                "body_rows_read": (rollup_report.get("summary") or {}).get("body_rows_read"),
            },
            "crunch_opportunity_summary": dry_run.get("summary"),
            "activation_follow_up": dry_run.get("activation_follow_up"),
        },
        "acceptance": {
            "stages_one_repeated_context_crunch_canary": len(actions) == 1,
            "has_validation_metadata": validation.get("schema") == CRUNCH_CANARY_APPLY_BATCH_SCHEMA
            and bool(actions)
            and bool(validation.get("ok"))
            and bool(validation.get("dry_run"))
            and not bool(validation.get("wrote_policy_files")),
            "reports_one_new_or_existing_repeated_context_crunch_canary": reported_canary_count == 1,
            "has_projected_tokens": bool(top_reported_canary and _as_int(top_reported_canary.get("projected_saved_tokens")) > 0),
            "has_projected_savings": bool(top_reported_canary and _as_float(top_reported_canary.get("projected_saved_usd")) > 0),
            "has_holdout_metadata": bool(
                (top_action and _as_float(top_action.get("holdout_fraction")) > 0)
                or (
                    isinstance((top_reported_canary or {}).get("duplicate_suppression"), dict)
                    and _as_float(top_reported_canary["duplicate_suppression"].get("matching_holdout_fraction")) > 0
                )
            ),
            "has_projected_lifecycle_split": bool(
                stage_lifecycle_projection["projected_canary_applied_count"] > 0
                and stage_lifecycle_projection["projected_canary_holdout_count"] > 0
            ),
            "has_safety_stop_metadata": bool(
                top_action
                and isinstance(top_action.get("lifecycle_metadata"), dict)
                and bool(top_action["lifecycle_metadata"].get("emits_safety_stopped"))
            ),
            "has_cohort_selector": bool(
                top_action
                and isinstance(top_action.get("cohort_selector"), dict)
                and top_action["cohort_selector"].get("schema") == "tokenclaw.request_shape_crunch_canary_cohort_selector.v1"
            ),
            "has_rollback_threshold": bool(top_action and _as_float(top_action.get("rollback_threshold")) > 0),
            "has_duplicate_suppression": isinstance(duplicate_suppression, dict)
            and duplicate_suppression.get("schema") == "tokenclaw.request_shape_crunch_stage_duplicate_suppression_summary.v1",
            "has_file_backed_target": True,
            "has_activation_ready_rollup_selection": rollup_selection.get("schema")
            == "tokenclaw.request_shape_crunch_canary_stage_rollup_selection.v1",
            "drafts_only_activation_ready_rollups": all(
                row.get("activation_readiness") == "activation-ready"
                for row in rollup_selection.get("rows") or []
                if row.get("selected_for_stage")
            ),
            "reports_skipped_rollup_reasons": rollup_selection.get("skipped_count") == 0
            or bool(rollup_selection.get("skipped_reason_breakdown")),
            "unsafe_or_stale_cohorts_remain_skipped": all(
                item.get("readiness") == "measurement-ready"
                or _as_int(item.get("projected_skipped_count")) > 0
                or _as_int(item.get("projected_safety_stopped_count")) > 0
                for item in lifecycle_projections
            ),
            "stages_all_unsuppressed_cohorts_within_bound": len(actions)
            == min(
                _as_int(duplicate_suppression.get("stageable_unsuppressed_cohort_count")),
                _as_int(duplicate_suppression.get("stage_action_limit")) or DEFAULT_CRUNCH_CANARY_MAX_NEW_STAGE_ACTIONS,
            ),
            "does_not_restage_suppressed_or_existing_widened_cohorts": all(
                not (
                    isinstance(action.get("duplicate_suppression"), dict)
                    and bool(action["duplicate_suppression"].get("suppresses_new_stage_action"))
                )
                for action in actions
            ),
        },
        "privacy": _crunch_opportunity_privacy(),
    }


def _cache_replay_canary_cohort_id(cohort: dict[str, Any]) -> str:
    basis = {
        "schema": "tokenclaw.request_shape_cache_replay_canary_cohort_id_basis.v1",
        "provider_family": cohort.get("provider_family"),
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "stream": bool(cohort.get("stream")),
        "has_tools": bool(cohort.get("has_tools")),
        "cache_status": cohort.get("cache_status"),
        "routing_status": cohort.get("routing_status"),
        "text_bucket": cohort.get("text_bucket"),
        "token_bucket": cohort.get("token_bucket"),
    }
    digest = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:16]
    endpoint = public_label(cohort.get("endpoint"), "unknown").replace("_", "-")
    category = public_label(cohort.get("category"), "unknown").replace("_", "-")
    return f"request-shape-cache-replay:{endpoint}:{category}:{digest}"


def _cache_replay_canary_policy_id(cohort_id: str) -> str:
    digest = hashlib.sha256(cohort_id.encode("utf-8")).hexdigest()[:16]
    return f"local-openai-cache-replay-canary:{digest}"


def _cache_replay_shape_key(shape: dict[str, Any]) -> tuple[Any, ...]:
    return (
        public_label(shape.get("provider_family") or "openai", "unknown"),
        public_label(shape.get("source_surface"), "unknown"),
        public_label(shape.get("endpoint"), "unknown"),
        public_label(shape.get("category"), "unknown"),
        public_label(shape.get("workflow_phase"), "unknown"),
        public_label(shape.get("text_bucket"), "unknown"),
        public_label(shape.get("token_bucket"), "unknown"),
        bool(shape.get("stream")),
        bool(shape.get("has_tools")),
    )


def _cache_replay_cohort_shape_key(cohort: dict[str, Any]) -> tuple[Any, ...]:
    return _cache_replay_shape_key(
        {
            "provider_family": cohort.get("provider_family"),
            "source_surface": cohort.get("source_surface"),
            "endpoint": cohort.get("endpoint"),
            "category": cohort.get("category"),
            "workflow_phase": cohort.get("workflow_phase"),
            "text_bucket": cohort.get("text_bucket"),
            "token_bucket": cohort.get("token_bucket"),
            "stream": bool(cohort.get("stream")),
            "has_tools": bool(cohort.get("has_tools")),
        }
    )


def _cache_replay_handled_rule_state(rule: dict[str, Any]) -> str:
    if not bool(rule.get("enabled", True)):
        return "blocked-local-policy"
    source_file = str(rule.get("source_policy_file") or "")
    target = rule.get("target_cache_policy") if isinstance(rule.get("target_cache_policy"), dict) else {}
    target_file = str(target.get("target_local_rule_file") or source_file)
    rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
    if target_file == "cache_rules.yaml" or source_file == "cache_rules.yaml" or rollout.get("canary_enabled") is False:
        return "active-local-policy"
    return "staged-canary"


def _cache_replay_handled_rule_public(rule: dict[str, Any]) -> dict[str, Any]:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
    return {
        "schema": "tokenclaw.request_shape_cache_replay_handled_local_policy.v1",
        "handled": True,
        "handled_state": _cache_replay_handled_rule_state(rule),
        "policy_source": public_label(rule.get("policy_source"), "unknown"),
        "source_policy_file": public_label(rule.get("source_policy_file"), "unknown"),
        "target_local_rule_file": public_label(rule.get("target_local_rule_file") or rule.get("source_policy_file"), "unknown"),
        "shape": _cache_replay_public_shape_from_rule(rule),
        "source_schema": graduation.get("source_schema"),
        "sample_count": _as_int(graduation.get("sample_count")),
        "source_rank": _as_int(graduation.get("rank") or graduation.get("cohort_rank")),
        "projected_hits": _as_int(graduation.get("projected_hits") or graduation.get("projected_hit_count")),
        "projected_savings_usd": round(
            _as_float(graduation.get("projected_savings_usd") or graduation.get("projected_saved_cost_usd")),
            6,
        ),
        "metadata_only": True,
        "aggregate_only": True,
        "rule_ids_included": False,
        "cohort_ids_included": False,
        "policy_paths_included": False,
    }


def _cache_replay_handled_match(
    cohort: dict[str, Any],
    handled_rules: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not handled_rules:
        return None
    cohort_key = _cache_replay_cohort_shape_key(cohort)
    for rule in handled_rules:
        if _cache_replay_shape_key(_cache_replay_public_shape_from_rule(rule)) == cohort_key:
            return _cache_replay_handled_rule_public(rule)
    return None


def _request_shape_cache_replay_canary_lifecycle_projection(
    cohort: dict[str, Any],
    *,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    matched = _as_int(cohort.get("row_count") or cohort.get("cohort_row_count"))
    projected_hits = _as_int(cohort.get("projected_hits"))
    projected_savings = _as_float(cohort.get("projected_savings_usd"))
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)

    readiness = public_label(cohort.get("readiness"), "unknown")
    reason = public_label(cohort.get("reason"), "unknown")
    blockers = [
        public_label(item, "unknown")
        for item in cohort.get("blockers") or []
        if public_label(item, "unknown") != "unknown"
    ]
    applied = holdout_count = skipped = 0
    lifecycle_status = "skipped"
    explicit_reason = reason

    if readiness == "replay-ready" and matched > 0:
        holdout_count = min(matched, int(math.ceil(matched * holdout))) if holdout > 0 else 0
        remaining = max(0, matched - holdout_count)
        applied = min(remaining, int(math.ceil(matched * rollout))) if rollout > 0 else 0
        skipped = max(0, matched - applied - holdout_count)
        lifecycle_status = "projected-applied-holdout" if applied > 0 and holdout_count > 0 else "projected-partial"
        explicit_reason = "projected-cache-replay-applied-and-holdout" if applied > 0 and holdout_count > 0 else "projected-cache-replay-partial"
    else:
        skipped = matched
        lifecycle_status = "skipped"
        explicit_reason = reason or (blockers[0] if blockers else "not-stageable")

    return {
        "schema": "tokenclaw.request_shape_cache_replay_canary_projected_lifecycle.v1",
        "status": lifecycle_status,
        "reason": explicit_reason,
        "readiness": readiness,
        "matched_count": matched,
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "canary_applied_eligible": applied > 0,
        "canary_holdout_eligible": holdout_count > 0,
        "projected_canary_applied_count": applied,
        "projected_canary_holdout_count": holdout_count,
        "projected_skipped_count": skipped,
        "projected_invalidated_count": 0,
        "projected_bypassed_count": 0,
        "projected_hits": projected_hits,
        "projected_savings_usd": round(projected_savings, 6),
        "projected_applied_hits": _scaled_projection(projected_hits, applied, matched),
        "projected_applied_savings_usd": round(projected_savings * (applied / float(matched)), 6) if matched and applied else 0.0,
        "projected_holdout_hits": _scaled_projection(projected_hits, holdout_count, matched),
        "projected_holdout_savings_usd": round(projected_savings * (holdout_count / float(matched)), 6) if matched and holdout_count else 0.0,
        "blocker_reasons": blockers,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _replayability_privacy(),
    }


def _request_shape_cache_replay_canary_action(
    cohort: dict[str, Any],
    *,
    candidate_count: int,
    rollout_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    cohort_id = str(cohort.get("cohort_id") or _cache_replay_canary_cohort_id(cohort))
    policy_id = _cache_replay_canary_policy_id(cohort_id)
    rollout = _bounded_fraction(rollout_fraction, DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(holdout_fraction, DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    lifecycle_projection = _request_shape_cache_replay_canary_lifecycle_projection(
        cohort,
        rollout_fraction=rollout,
        holdout_fraction=holdout,
    )
    shape = {
        "provider_family": cohort.get("provider_family"),
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "stream": bool(cohort.get("stream")),
        "has_tools": bool(cohort.get("has_tools")),
        "cache_status": cohort.get("cache_status"),
        "routing_status": cohort.get("routing_status"),
        "text_bucket": cohort.get("text_bucket"),
        "token_bucket": cohort.get("token_bucket"),
        "readiness": cohort.get("readiness"),
        "reason": cohort.get("reason"),
    }
    target_cache_policy = {
        "schema": "tokenclaw.request_shape_cache_replay_target_policy.v1",
        "policy_section": "cache.pattern_rules",
        "target_local_policy": "cache_canary_policy",
        "target_local_rule_file": "cache_canary_policy.yaml",
        "policy_source": "local-manual",
        "local_file_backed": True,
        "managed_dependency": "optional",
        "rules_path_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }
    return {
        "schema": REPLAY_CACHE_CANARY_ACTION_SCHEMA,
        "action_type": "stage-local-openai-cache-replay-canary",
        "target_local_policy": "cache_canary_policy",
        "target_cache_policy": target_cache_policy,
        "target_local_rule_file": "cache_canary_policy.yaml",
        "policy_section": "cache",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "rank": _as_int(cohort.get("rank")),
        "cohort_rank": _as_int(cohort.get("rank")),
        "candidate_count": candidate_count,
        "eligible_cohort_count": candidate_count,
        "cohort_row_count": _as_int(cohort.get("row_count")),
        "rollout_fraction": round(rollout, 6),
        "holdout_fraction": round(holdout, 6),
        "canary_fraction": round(rollout, 6),
        "policy_source": "local-manual",
        "shape": shape,
        "conditions": shape,
        "ttl_seconds": DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS,
        "invalidation": {
            "schema": "tokenclaw.request_shape_cache_replay_invalidation_assumptions.v1",
            "strategy": "session-scoped-exact-non-tool",
            "scope": "session",
            "safe_invalidation": False,
            "tool_call_cache_enabled": False,
            "streaming_replay_enabled": False,
            "ttl_seconds": DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS,
            "assumptions": [
                "non-streaming-openai-responses-only",
                "no-tool-calls",
                "exact-request-shape-replay",
                "session-scoped-keying",
                "ttl-limited-replay-window",
            ],
            "bypass_reasons": [
                "tools-present",
                "streaming-replay-not-supported",
                "invalidation-evidence-missing",
                "unsafe-tool-calls-without-invalidation",
                "cohort-mismatch",
            ],
            "metadata_only": True,
            "aggregate_only": True,
        },
        "projected_hits": _as_int(cohort.get("projected_hits")),
        "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
        "projected_lifecycle": lifecycle_projection,
        "canary_applied_eligible": lifecycle_projection["canary_applied_eligible"],
        "canary_holdout_eligible": lifecycle_projection["canary_holdout_eligible"],
        "safety_gates": {
            "metadata_only": True,
            "aggregate_only": True,
            "local_file_backed": True,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "cache_entries_written": False,
            "policy_files_written": False,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "openai_responses_only": cohort.get("source_surface") == "openai_responses" and cohort.get("endpoint") == "responses",
            "exact_non_tool_only": not bool(cohort.get("has_tools")) and not bool(cohort.get("stream")),
            "tool_call_cache_enabled": False,
            "streaming_replay_enabled": False,
            "ttl_seconds": DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS,
            "session_scoped": True,
            "holdout_required": holdout > 0,
            "records_applied_holdout_skipped_invalidation_blocked": True,
            "records_applied_holdout_skipped_bypassed_invalidated_hit_miss": True,
            "records_applied_holdout_hit_miss_bypass_invalidation_blocked_stale_risk": True,
        },
        "cache_decision_metadata": {
            "schema": "tokenclaw.request_shape_cache_replay_decision_metadata.v1",
            "cache_json_field": "cache_replay_canary",
            "emits_statuses": [
                "applied",
                "holdout",
                "skipped",
                "bypass",
                "bypassed",
                "invalidated",
                "invalidation_blocked",
                "stale_risk",
                "cache_hit",
                "cache_miss",
            ],
            "records_applied": True,
            "records_holdout": True,
            "records_skipped": True,
            "records_bypass": True,
            "records_bypassed": True,
            "records_invalidated": True,
            "records_invalidation_blocked": True,
            "records_stale_risk": True,
            "records_cache_hit": True,
            "records_cache_miss": True,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "lifecycle_metadata": {
            "schema": "tokenclaw.request_shape_cache_replay_canary_stage_lifecycle_metadata.v1",
            "emits_applied": True,
            "emits_holdout": True,
            "emits_skipped": True,
            "emits_bypass": True,
            "emits_bypassed": True,
            "emits_invalidated": True,
            "emits_invalidation_blocked": True,
            "emits_stale_risk": True,
            "emits_cache_hit": True,
            "emits_cache_miss": True,
            "canary_applied_eligible": lifecycle_projection["canary_applied_eligible"],
            "canary_holdout_eligible": lifecycle_projection["canary_holdout_eligible"],
            "projected_canary_applied_count": lifecycle_projection["projected_canary_applied_count"],
            "projected_canary_holdout_count": lifecycle_projection["projected_canary_holdout_count"],
            "projected_skipped_count": lifecycle_projection["projected_skipped_count"],
            "projected_invalidated_count": lifecycle_projection["projected_invalidated_count"],
            "projected_bypassed_count": lifecycle_projection["projected_bypassed_count"],
            "impact_report": "tokenclaw.openai_cache_replay_impact.v1",
            "lifecycle_feedback_schema": "tokenclaw.openai_cache_replay_lifecycle_feedback.v1",
            "metadata_only": True,
            "aggregate_only": True,
        },
        "next_action": "apply-local-cache-replay-canary-after-review",
        "privacy": _replayability_privacy(),
    }


def _cache_replay_stage_skipped_guard_summary(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    skipped = [
        cohort
        for cohort in cohorts
        if isinstance(cohort, dict) and cohort.get("readiness") == "skipped"
    ]
    blocker_counts: dict[str, int] = {}
    skipped_rows = 0
    tool_count = 0
    streaming_count = 0
    invalidation_missing_count = 0
    stale_risk_count = 0
    examples: list[dict[str, Any]] = []

    for cohort in skipped:
        row_count = _as_int(cohort.get("row_count"))
        skipped_rows += row_count
        blockers = [public_label(item, "unknown") for item in cohort.get("blockers") or [] if item]
        reason = public_label(cohort.get("reason"), "unknown")
        if not blockers and reason != "unknown":
            blockers = [reason]
        for blocker in blockers:
            _increment(blocker_counts, blocker)
        has_tools = bool(cohort.get("has_tools")) or any("tool" in blocker for blocker in blockers)
        is_streaming = bool(cohort.get("stream")) or "streaming-replay-not-supported" in blockers
        invalidation_missing = "invalidation-evidence-missing" in blockers
        stale_risk = any("stale" in blocker for blocker in blockers)
        if has_tools:
            tool_count += 1
        if is_streaming:
            streaming_count += 1
        if invalidation_missing:
            invalidation_missing_count += 1
        if stale_risk:
            stale_risk_count += 1
        if len(examples) < 5:
            examples.append(
                {
                    "rank": _as_int(cohort.get("rank")),
                    "reason": reason,
                    "blockers": blockers,
                    "provider_family": cohort.get("provider_family"),
                    "source_surface": cohort.get("source_surface"),
                    "endpoint": cohort.get("endpoint"),
                    "category": cohort.get("category"),
                    "stream": bool(cohort.get("stream")),
                    "has_tools": bool(cohort.get("has_tools")),
                    "row_count": row_count,
                    "projected_hits": _as_int(cohort.get("projected_hits")),
                }
            )

    return {
        "schema": "tokenclaw.request_shape_cache_replay_canary_skipped_guards.v1",
        "skipped_cohort_count": len(skipped),
        "skipped_rows": skipped_rows,
        "tool_cohort_count": tool_count,
        "streaming_cohort_count": streaming_count,
        "invalidation_missing_cohort_count": invalidation_missing_count,
        "stale_risk_cohort_count": stale_risk_count,
        "blocker_breakdown": _breakdown(blocker_counts),
        "examples": examples,
        "tool_streaming_and_invalidation_missing_remain_skipped": (
            all(cohort.get("readiness") == "skipped" for cohort in skipped)
            and tool_count > 0
            and streaming_count > 0
            and invalidation_missing_count > 0
        ),
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
    }


def build_request_shape_cache_replay_canary_stage_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    run_id: str | None = None,
    persist_rollups: bool = False,
    rollout_fraction: float = DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION,
    holdout_fraction: float = DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION,
    mark_handled_cache_replay_cohorts: bool = True,
) -> dict[str, Any]:
    rollup_report = build_request_shape_rollups_report(
        store_obj,
        limit=limit,
        persist=persist_rollups,
        run_id=run_id,
        mark_handled_cache_replay_cohorts=mark_handled_cache_replay_cohorts,
    )
    dry_run = (
        rollup_report.get("cache_replayability_dry_run")
        if isinstance(rollup_report.get("cache_replayability_dry_run"), dict)
        else {}
    )
    source_cohorts = (
        dry_run.get("remaining_replay_ready_cohorts")
        if mark_handled_cache_replay_cohorts and isinstance(dry_run.get("remaining_replay_ready_cohorts"), list)
        else dry_run.get("cohorts")
    )
    exact_safe_categories = {"chat", "short-completion"}
    cohorts = [
        cohort
        for cohort in source_cohorts or []
        if isinstance(cohort, dict)
        and cohort.get("readiness") == "replay-ready"
        and not bool(cohort.get("handled_by_local_policy"))
        and bool(cohort.get("remaining_replay_ready", True))
        and cohort.get("next_action") == "stage-cache-replay-canary"
        and cohort.get("provider_family") == "openai"
        and cohort.get("source_surface") == "openai_responses"
        and cohort.get("endpoint") == "responses"
        and str(cohort.get("category") or "") in exact_safe_categories
        and not bool(cohort.get("has_tools"))
        and not bool(cohort.get("stream"))
        and _as_int(cohort.get("projected_hits")) > 0
        and _as_float(cohort.get("projected_savings_usd")) > 0
    ]
    top_stageable_cohorts = cohorts
    actions = [
        _request_shape_cache_replay_canary_action(
            cohort,
            candidate_count=len(cohorts),
            rollout_fraction=rollout_fraction,
            holdout_fraction=holdout_fraction,
        )
        for cohort in top_stageable_cohorts
    ]
    skipped_guards = _cache_replay_stage_skipped_guard_summary(
        [cohort for cohort in dry_run.get("cohorts") or [] if isinstance(cohort, dict)]
    )
    top_action = actions[0] if actions else None
    top_cohort = cohorts[0] if cohorts else None
    if actions:
        status = "staged"
        next_action = "apply-local-cache-replay-canary-after-review"
        reason = "staged-openai-responses-cache-replay-canary"
    elif _as_int((dry_run.get("summary") or {}).get("gated_too_small_replay_ready_cohort_count")) > 0:
        status = "no-stageable-cohort"
        next_action = "no-op-too-small-without-live-repeat"
        reason = "too-small-without-live-repeat"
    else:
        status = "no-stageable-cohort"
        next_action = "rank-request-shape-cache-replayability"
        reason = (dry_run.get("summary") or {}).get("top_blocker_code") or dry_run.get("status") or "no-stageable-cohort"
    return {
        "schema": REPLAY_CACHE_CANARY_STAGE_SCHEMA,
        "status": status,
        "ok": bool(actions),
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "run_id": rollup_report.get("run_id"),
        "reason": reason,
        "next_action": next_action,
        "staged_canary_count": len(actions),
        "eligible_stageable_cohort_count": len(cohorts),
        "stage_actions": actions,
        "top_stage_action": top_action,
        "top_cohort": top_cohort,
        "skipped_cohort_guards": skipped_guards,
        "source_report": {
            "schema": rollup_report.get("schema"),
            "window": rollup_report.get("window"),
            "summary": {
                "rows_considered": (rollup_report.get("summary") or {}).get("rows_considered"),
                "rollup_count": (rollup_report.get("summary") or {}).get("rollup_count"),
                "top_next_action": (rollup_report.get("summary") or {}).get("top_next_action"),
                "body_rows_read": (rollup_report.get("summary") or {}).get("body_rows_read"),
            },
            "cache_replayability_summary": dry_run.get("summary"),
            "readiness_breakdown": dry_run.get("readiness_breakdown"),
            "blocker_breakdown": dry_run.get("blocker_breakdown"),
            "handled_policy_summary": dry_run.get("handled_policy_summary"),
        },
        "acceptance": {
            "stages_single_top_ranked_cohort": len(actions) == 1 and bool(top_action and _as_int(top_action.get("rank")) == _as_int((top_cohort or {}).get("rank"))),
            "stages_top_ranked_cohort": bool(
                top_action and _as_int(top_action.get("rank")) == _as_int((top_cohort or {}).get("rank"))
            ),
            "stages_all_remaining_exact_safe_replay_ready_cohorts": len(actions) == len(cohorts),
            "has_replay_ready_openai_responses_cohort": bool(actions),
            "stages_remaining_unhandled_replay_ready_cohort": bool(
                top_cohort
                and top_cohort.get("readiness") == "replay-ready"
                and bool(top_cohort.get("remaining_replay_ready", True))
                and not bool(top_cohort.get("handled_by_local_policy"))
            ),
            "excludes_already_handled_local_policy_cohorts": bool(
                not mark_handled_cache_replay_cohorts
                or all(not bool(action.get("handled_by_local_policy")) for action in actions)
            ),
            "has_rank": bool(top_action and _as_int(top_action.get("rank")) > 0),
            "has_shape_buckets": bool(
                top_action
                and isinstance(top_action.get("shape"), dict)
                and bool(top_action["shape"].get("text_bucket"))
                and bool(top_action["shape"].get("token_bucket"))
            ),
            "has_target_cache_policy_metadata": bool(
                top_action
                and isinstance(top_action.get("target_cache_policy"), dict)
                and top_action["target_cache_policy"].get("target_local_rule_file") == "cache_canary_policy.yaml"
                and top_action["target_cache_policy"].get("policy_section") == "cache.pattern_rules"
            ),
            "has_projected_hits": bool(top_action and _as_int(top_action.get("projected_hits")) > 0),
            "has_projected_savings": bool(top_action and _as_float(top_action.get("projected_savings_usd")) > 0),
            "writes_no_provider_bodies": bool(top_action and not top_action["safety_gates"]["provider_bodies_included"]),
            "writes_no_cache_entries": bool(top_action and not top_action["safety_gates"]["cache_entries_written"]),
            "has_holdout_metadata": bool(top_action and _as_float(top_action.get("holdout_fraction")) > 0),
            "has_lifecycle_metadata": bool(
                top_action
                and isinstance(top_action.get("lifecycle_metadata"), dict)
                and bool(top_action["lifecycle_metadata"].get("emits_applied"))
                and bool(top_action["lifecycle_metadata"].get("emits_holdout"))
                and bool(top_action["lifecycle_metadata"].get("emits_skipped"))
                and bool(top_action["lifecycle_metadata"].get("emits_bypass"))
                and bool(top_action["lifecycle_metadata"].get("emits_invalidated"))
                and bool(top_action["lifecycle_metadata"].get("emits_invalidation_blocked"))
                and bool(top_action["lifecycle_metadata"].get("emits_stale_risk"))
            ),
            "has_applied_and_holdout_eligibility": bool(
                top_action
                and bool(top_action.get("canary_applied_eligible"))
                and bool(top_action.get("canary_holdout_eligible"))
                and isinstance(top_action.get("projected_lifecycle"), dict)
                and _as_int(top_action["projected_lifecycle"].get("projected_canary_applied_count")) > 0
                and _as_int(top_action["projected_lifecycle"].get("projected_canary_holdout_count")) > 0
            ),
            "records_hit_miss_bypass_invalidation_and_stale_risk": bool(
                top_action
                and isinstance(top_action.get("cache_decision_metadata"), dict)
                and bool(top_action["cache_decision_metadata"].get("records_cache_hit"))
                and bool(top_action["cache_decision_metadata"].get("records_cache_miss"))
                and bool(top_action["cache_decision_metadata"].get("records_bypass"))
                and bool(top_action["cache_decision_metadata"].get("records_invalidated"))
                and bool(top_action["cache_decision_metadata"].get("records_invalidation_blocked"))
                and bool(top_action["cache_decision_metadata"].get("records_stale_risk"))
            ),
            "preserves_tool_and_streaming_guards": all(
                not bool(action.get("conditions", {}).get("has_tools")) and not bool(action.get("conditions", {}).get("stream"))
                for action in actions
            ),
            "stages_only_openai_responses_exact_safe_categories": all(
                action.get("conditions", {}).get("provider_family") == "openai"
                and action.get("conditions", {}).get("source_surface") == "openai_responses"
                and action.get("conditions", {}).get("endpoint") == "responses"
                and action.get("conditions", {}).get("category") in exact_safe_categories
                for action in actions
            ),
            "tool_streaming_and_invalidation_missing_cohorts_skipped": bool(
                skipped_guards.get("tool_streaming_and_invalidation_missing_remain_skipped")
            ),
        },
        "privacy": _replayability_privacy(),
    }


def apply_request_shape_cache_replay_canary_action(
    action: dict[str, Any],
    *,
    rules_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if action.get("schema") != REPLAY_CACHE_CANARY_ACTION_SCHEMA:
        errors.append({"path": "$.schema", "message": "unsupported request-shape cache replay canary action schema"})
    if action.get("action_type") != "stage-local-openai-cache-replay-canary":
        errors.append({"path": "$.action_type", "message": "unsupported request-shape cache replay canary action type"})
    if action.get("target_local_policy") != "cache_canary_policy":
        errors.append({"path": "$.target_local_policy", "message": "request-shape cache replay canary must target cache_canary_policy"})
    privacy = action.get("privacy") if isinstance(action.get("privacy"), dict) else {}
    safety = action.get("safety_gates") if isinstance(action.get("safety_gates"), dict) else {}
    for key in (
        "raw_prompts_included",
        "provider_bodies_included",
        "raw_responses_included",
        "cache_keys_included",
        "request_fingerprints_included",
        "request_ids_included",
        "session_ids_included",
    ):
        if privacy.get(key) or safety.get(key):
            errors.append({"path": f"$.privacy.{key}", "message": "request-shape cache replay canary action is not metadata-only"})
    conditions = action.get("conditions") if isinstance(action.get("conditions"), dict) else {}
    if conditions.get("provider_family") != "openai" or conditions.get("source_surface") != "openai_responses" or conditions.get("endpoint") != "responses":
        errors.append({"path": "$.conditions", "message": "request-shape cache replay canary must target OpenAI Responses"})
    if bool(conditions.get("has_tools")):
        errors.append({"path": "$.conditions.has_tools", "message": "tool-call cache replay requires separate invalidation evidence"})
    if bool(conditions.get("stream")):
        errors.append({"path": "$.conditions.stream", "message": "streaming cache replay is not supported by this canary"})
    if _as_int(action.get("projected_hits")) <= 0:
        errors.append({"path": "$.projected_hits", "message": "request-shape cache replay canary needs positive projected hits"})
    if _as_float(action.get("projected_savings_usd")) <= 0:
        errors.append({"path": "$.projected_savings_usd", "message": "request-shape cache replay canary needs positive projected savings"})

    path = Path(rules_path)
    existing: dict[str, Any] = {}
    if path.exists():
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        existing = loaded if isinstance(loaded, dict) else {}
    if errors:
        return {
            "schema": REPLAY_CACHE_CANARY_APPLY_SCHEMA,
            "ok": False,
            "dry_run": bool(dry_run),
            "wrote_policy_files": False,
            "target_local_policy": "cache_canary_policy",
            "rules_path_included": False,
            "errors": errors,
            "privacy": _replayability_privacy(),
        }

    policy_id = str(action.get("policy_id") or _cache_replay_canary_policy_id(str(action.get("cohort_id") or "")))
    cohort_id = str(action.get("cohort_id") or "")
    rollout = _bounded_fraction(action.get("rollout_fraction", action.get("canary_fraction")), DEFAULT_CACHE_REPLAY_CANARY_ROLLOUT_FRACTION)
    holdout = _bounded_fraction(action.get("holdout_fraction"), DEFAULT_CACHE_REPLAY_CANARY_HOLDOUT_FRACTION)
    if rollout + holdout > 1.0:
        holdout = max(0.0, 1.0 - rollout)
    ttl_seconds = max(60, _as_int(action.get("ttl_seconds"), DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS))
    rule_conditions = {
        "pattern_hashes": ["sha256:*"],
        "provider_family": conditions.get("provider_family"),
        "source_surface": conditions.get("source_surface"),
        "endpoint": conditions.get("endpoint"),
        "category": conditions.get("category"),
        "workflow_phase": conditions.get("workflow_phase"),
        "text_bucket": conditions.get("text_bucket"),
        "token_bucket": conditions.get("token_bucket"),
        "has_tools": False,
        "stream": False,
        "replayability_levels": ["features_only", "local-exact-response"],
    }
    rule_conditions = {key: value for key, value in rule_conditions.items() if value not in (None, "", [])}
    canary_rule = {
        "id": policy_id,
        "enabled": True,
        "policy_source": "local-manual",
        "candidate_id": cohort_id,
        "description": "Local OpenAI Responses exact-cache replay canary staged from aggregate request-shape evidence.",
        "target_cache_policy": action.get("target_cache_policy")
        if isinstance(action.get("target_cache_policy"), dict)
        else {
            "schema": "tokenclaw.request_shape_cache_replay_target_policy.v1",
            "policy_section": "cache.pattern_rules",
            "target_local_policy": "cache_canary_policy",
            "target_local_rule_file": "cache_canary_policy.yaml",
            "policy_source": "local-manual",
            "local_file_backed": True,
            "managed_dependency": "optional",
            "rules_path_included": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "conditions": rule_conditions,
        "action": {
            "type": "exact_cache_pattern",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "streaming": False,
            "scope": "session",
            "min_call_count": 2,
            "ttl_seconds": ttl_seconds,
        },
        "rollout": {
            "schema": "tokenclaw.pattern_policy_rollout.v1",
            "recommendation_mode": "openai-cache-replay-request-shape-canary",
            "canary_enabled": True,
            "canary_fraction": round(rollout, 6),
            "holdout_fraction": round(holdout, 6),
            "canary_salt": policy_id,
            "canary_unit": "request_fingerprint",
            "local_feedback_fields": [
                "cache_hit",
                "status_code",
                "retry_count",
                "latency_ms",
                "cost_est_usd",
                "cost_baseline_usd",
                "cache_replay_canary",
                "invalidation_reason",
            ],
        },
        "graduation": {
            "schema": "tokenclaw.request_shape_cache_replay_shape_activation.v1",
            "source_schema": REPLAYABILITY_DRY_RUN_SCHEMA,
            "source_reason": conditions.get("reason"),
            "cohort_id": cohort_id,
            "source_surface": conditions.get("source_surface"),
            "endpoint": conditions.get("endpoint"),
            "category": conditions.get("category"),
            "workflow_phase": conditions.get("workflow_phase"),
            "text_bucket": conditions.get("text_bucket"),
            "token_bucket": conditions.get("token_bucket"),
            "sample_count": _as_int(action.get("cohort_row_count")),
            "rank": _as_int(action.get("rank") or action.get("cohort_rank")),
            "cohort_rank": _as_int(action.get("cohort_rank") or action.get("rank")),
            "shape": action.get("shape") if isinstance(action.get("shape"), dict) else conditions,
            "projected_hits": _as_int(action.get("projected_hits")),
            "projected_savings_usd": round(_as_float(action.get("projected_savings_usd")), 6),
            "aggregate_only": True,
            "staged_at": utc_now(),
        },
        "invalidation": action.get("invalidation") if isinstance(action.get("invalidation"), dict) else {},
        "safety_gates": action.get("safety_gates") if isinstance(action.get("safety_gates"), dict) else {},
        "lifecycle_metadata": action.get("lifecycle_metadata") if isinstance(action.get("lifecycle_metadata"), dict) else {},
        "privacy": _replayability_privacy(),
        "staged_at": utc_now(),
    }

    updated = dict(existing)
    updated["schema"] = str(updated.get("schema") or "tokenclaw.openai_cache_replay_canary_policy.v1")
    updated["policy_source"] = "local-manual"
    updated["generated_at"] = utc_now()
    rules = updated.get("pattern_rules") if isinstance(updated.get("pattern_rules"), list) else []
    kept = [rule for rule in rules if not (isinstance(rule, dict) and rule.get("id") == policy_id)]
    updated["pattern_rules"] = kept + [canary_rule]
    if not dry_run:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")

    return {
        "schema": REPLAY_CACHE_CANARY_APPLY_SCHEMA,
        "ok": True,
        "dry_run": bool(dry_run),
        "wrote_policy_files": not dry_run,
        "target_local_policy": "cache_canary_policy",
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "canary_fraction": canary_rule["rollout"]["canary_fraction"],
        "holdout_fraction": canary_rule["rollout"]["holdout_fraction"],
        "ttl_seconds": ttl_seconds,
        "rules_path_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "cache_entries_written": False,
        "privacy": _replayability_privacy(),
    }


def _cache_replay_evidence_privacy() -> dict[str, Any]:
    privacy = dict(_replayability_privacy())
    privacy.update({
        "policy_ids_included": False,
        "rule_ids_included": False,
        "cohort_ids_included": False,
        "cache_canary_policy_path_included": False,
    })
    return privacy


def _load_cache_replay_canary_policy(path: str | Path) -> tuple[dict[str, Any], bool]:
    policy_path = Path(path).expanduser()
    if not policy_path.exists():
        return {}, False
    import yaml

    loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    return (loaded if isinstance(loaded, dict) else {}), True


def _request_shape_cache_replay_policy_rules(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rules = policy.get("pattern_rules") if isinstance(policy.get("pattern_rules"), list) else []
    staged: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        target_cache_policy = rule.get("target_cache_policy") if isinstance(rule.get("target_cache_policy"), dict) else {}
        if graduation.get("source_schema") != REPLAYABILITY_DRY_RUN_SCHEMA:
            continue
        staged.append({
            "index": index + 1,
            "rule_id": str(rule.get("id") or rule.get("rule_id") or ""),
            "candidate_id": str(rule.get("candidate_id") or ""),
            "policy_source": public_label(rule.get("policy_source"), "unknown"),
            "enabled": bool(rule.get("enabled", True)),
            "target_cache_policy": target_cache_policy,
            "target_local_rule_file": public_label(target_cache_policy.get("target_local_rule_file"), "unknown"),
            "conditions": conditions,
            "action": action,
            "rollout": rollout,
            "graduation": graduation,
            "staged_at": rule.get("staged_at") or graduation.get("staged_at"),
        })
    return staged


def _cache_replay_policy_file_candidates(filename: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if filename == "cache_rules.yaml":
        candidates.append((filename, Path(__file__).with_name(filename)))
    candidates.append((filename, Path.cwd() / "config" / filename))
    candidates.append((filename, tokenclaw_config_path(filename)))
    return candidates


def _request_shape_cache_replay_handled_policy_rules() -> list[dict[str, Any]]:
    handled: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for filename in ("cache_rules.yaml", "cache_canary_policy.yaml"):
        for public_filename, path in _cache_replay_policy_file_candidates(filename):
            if not path.exists():
                continue
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(loaded, dict):
                continue
            for rule in _request_shape_cache_replay_policy_rules(loaded):
                rule = dict(rule)
                rule["source_policy_file"] = public_filename
                if not rule.get("target_local_rule_file") or rule.get("target_local_rule_file") == "unknown":
                    rule["target_local_rule_file"] = public_filename
                key = (
                    _cache_replay_shape_key(_cache_replay_public_shape_from_rule(rule)),
                    _cache_replay_handled_rule_state(rule),
                    rule.get("source_policy_file"),
                )
                if key in seen:
                    continue
                seen.add(key)
                handled.append(rule)
    return handled


def _cache_replay_public_shape_from_rule(rule: dict[str, Any]) -> dict[str, Any]:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    return {
        "provider_family": public_label(conditions.get("provider_family") or "openai", "unknown"),
        "source_surface": public_label(graduation.get("source_surface") or conditions.get("source_surface"), "unknown"),
        "endpoint": public_label(graduation.get("endpoint") or conditions.get("endpoint"), "unknown"),
        "category": public_label(graduation.get("category") or conditions.get("category"), "unknown"),
        "workflow_phase": public_label(graduation.get("workflow_phase") or conditions.get("workflow_phase"), "unknown"),
        "text_bucket": public_label(graduation.get("text_bucket") or conditions.get("text_bucket"), "unknown"),
        "token_bucket": public_label(graduation.get("token_bucket") or conditions.get("token_bucket"), "unknown"),
        "stream": bool(conditions.get("stream")),
        "has_tools": bool(conditions.get("has_tools")),
    }


def _cache_replay_staged_policy_summary(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for rank, rule in enumerate(rules, start=1):
        graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        summaries.append({
            "rank": rank,
            "policy_source": rule.get("policy_source") or "unknown",
            "enabled": bool(rule.get("enabled", True)),
            "shape": _cache_replay_public_shape_from_rule(rule),
            "sample_count": _as_int(graduation.get("sample_count")),
            "projected_hits": _as_int(graduation.get("projected_hits") or graduation.get("projected_hit_count")),
            "projected_savings_usd": round(_as_float(
                graduation.get("projected_savings_usd") or graduation.get("projected_saved_cost_usd")
            ), 6),
            "canary_fraction": round(_as_float(rollout.get("canary_fraction")), 6),
            "holdout_fraction": round(_as_float(rollout.get("holdout_fraction")), 6),
            "ttl_seconds": _as_int(action.get("ttl_seconds"), DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS),
            "source_schema": graduation.get("source_schema"),
            "aggregate_only": True,
            "metadata_only": True,
        })
    return summaries


def _cache_replay_staged_rule_observation_keys(
    cache_meta: dict[str, Any],
    *,
    staged_rule_ids: set[str],
    staged_candidate_ids: set[str],
) -> tuple[set[str], set[str]]:
    observed_rule_ids: set[str] = set()
    observed_candidate_ids: set[str] = set()
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    replay_canary = (
        cache_meta.get("cache_replay_canary")
        if isinstance(cache_meta.get("cache_replay_canary"), dict)
        else {}
    )
    for meta in (pattern_rule, replay_canary):
        rule_id = str(meta.get("rule_id") or meta.get("id") or "")
        candidate_id = str(meta.get("candidate_id") or "")
        if rule_id and rule_id in staged_rule_ids:
            observed_rule_ids.add(rule_id)
        if candidate_id and candidate_id in staged_candidate_ids:
            observed_candidate_ids.add(candidate_id)
    return observed_rule_ids, observed_candidate_ids


def _cache_replay_row_matches_staged_rules(
    cache_meta: dict[str, Any],
    *,
    staged_rule_ids: set[str],
    staged_candidate_ids: set[str],
) -> bool:
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    replay_canary = (
        cache_meta.get("cache_replay_canary")
        if isinstance(cache_meta.get("cache_replay_canary"), dict)
        else {}
    )
    for meta in (pattern_rule, replay_canary):
        rule_id = str(meta.get("rule_id") or meta.get("id") or "")
        candidate_id = str(meta.get("candidate_id") or "")
        if rule_id and rule_id in staged_rule_ids:
            return True
        if candidate_id and candidate_id in staged_candidate_ids:
            return True
    graduation = pattern_rule.get("graduation") if isinstance(pattern_rule.get("graduation"), dict) else {}
    projection = replay_canary.get("projection") if isinstance(replay_canary.get("projection"), dict) else {}
    return graduation.get("source_schema") == REPLAYABILITY_DRY_RUN_SCHEMA or projection.get("source_schema") == REPLAYABILITY_DRY_RUN_SCHEMA


def _cache_replay_stale_zero_traffic_rules(
    staged_rules: list[dict[str, Any]],
    *,
    observed_rule_ids: set[str],
    observed_candidate_ids: set[str],
    now: datetime,
    max_age_hours: float,
) -> list[dict[str, Any]]:
    stale_rules: list[dict[str, Any]] = []
    for rule in staged_rules:
        rule_id = str(rule.get("rule_id") or "")
        candidate_id = str(rule.get("candidate_id") or "")
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        if not bool(rule.get("enabled", True)) or rollout.get("canary_enabled") is False:
            continue
        if (rule_id and rule_id in observed_rule_ids) or (candidate_id and candidate_id in observed_candidate_ids):
            continue
        staged_at = _parse_utc(rule.get("staged_at"))
        if staged_at is None:
            continue
        age_hours = round((now - staged_at).total_seconds() / 3600.0, 3)
        if age_hours <= max_age_hours:
            continue
        stale_rules.append({
            "rank": rule.get("index"),
            "rule_id": rule_id or None,
            "staged_at": staged_at.isoformat(),
            "age_hours": age_hours,
            "max_age_hours": float(max_age_hours),
            "reason": "stale-no-canary-traffic",
            "metadata_only": True,
            "aggregate_only": True,
        })
    return stale_rules


def _row_shape_matches_staged_rule(row: dict[str, Any], routing: dict[str, Any], rule: dict[str, Any]) -> bool:
    """True when a row's shape attributes match a staged rule's conditions.

    Intentionally skips token_bucket (not populated in the OpenAI feature unit)
    and replayability_levels (requires canary metadata to evaluate).  Used as a
    fallback when _cache_replay_row_matches_staged_rules finds no strict match,
    so rows processed before the canary rule was loaded are classified as bypass.
    """
    provider = str(row.get("provider") or "anthropic").lower()
    if provider != "openai":
        return False
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    feature: dict[str, Any] = {}
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        candidate = routing.get(key)
        if isinstance(candidate, dict):
            feature = candidate
            break
    row_surface = str(row.get("source_surface") or feature.get("source_surface") or "")
    row_endpoint = str(row.get("endpoint") or feature.get("endpoint") or "")
    row_category = str(row.get("category") or feature.get("category") or "")
    row_text_bucket = str(feature.get("text_bucket") or "")
    for cond_key, actual in (
        ("source_surface", row_surface),
        ("endpoint", row_endpoint),
        ("category", row_category),
        ("text_bucket", row_text_bucket),
    ):
        expected = conditions.get(cond_key)
        if expected is None:
            continue
        expected_set = {str(v).lower() for v in ([expected] if isinstance(expected, str) else list(expected))}
        if expected_set and str(actual).lower() not in expected_set:
            return False
    if "stream" in conditions:
        row_stream = bool(int(row.get("stream") or 0))
        if bool(conditions["stream"]) != row_stream:
            return False
    if "has_tools" in conditions:
        row_has_tools = bool(routing.get("has_tools") or feature.get("has_tools"))
        if bool(conditions["has_tools"]) != row_has_tools:
            return False
    return True


def _cache_replay_evidence_class(row: dict[str, Any], cache_meta: dict[str, Any]) -> tuple[str, str]:
    replay_canary = (
        cache_meta.get("cache_replay_canary")
        if isinstance(cache_meta.get("cache_replay_canary"), dict)
        else {}
    )
    cache_status = str(cache_meta.get("status") or "").strip().lower()
    canary_status = str(replay_canary.get("status") or cache_meta.get("canary_cohort") or "").strip().lower()
    reason = public_label(replay_canary.get("reason") or cache_meta.get("reason") or cache_status, "unknown")
    cache_hit = bool(row.get("cache_hit")) or cache_status == "hit"
    unsupported_reasons = {
        "stream-mismatch",
        "streaming-not-allowed",
        "streaming-replay-not-supported",
        "has-tools-mismatch",
        "unsafe-tool-cache-pattern",
        "file-watch-required",
        "unsupported-endpoint",
        "cohort-mismatch",
        "replayability-gate-mismatch",
    }
    if canary_status == "holdout":
        return "holdout", reason
    if canary_status == "applied":
        if cache_hit:
            return "exact_hit", reason
        if cache_status == "miss":
            return "miss", reason
        return "applied", reason
    if canary_status in {"invalidated", "invalidation_blocked"}:
        return "invalidation_skipped", reason
    if canary_status in {"bypassed", "bypass", "safety_stopped"}:
        return "bypassed", reason
    if cache_status == "hit":
        return "exact_hit", reason
    if cache_status == "miss":
        return "miss", reason
    if reason in unsupported_reasons or cache_status == "skipped":
        return "unsupported_shape", reason
    return "bypassed", reason


def _cache_replay_applied_miss_blockers(row: dict[str, Any], cache_meta: dict[str, Any]) -> list[str]:
    replay_canary = (
        cache_meta.get("cache_replay_canary")
        if isinstance(cache_meta.get("cache_replay_canary"), dict)
        else {}
    )
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    pattern_rules = cache_meta.get("pattern_rules") if isinstance(cache_meta.get("pattern_rules"), dict) else {}
    store_meta = (
        cache_meta.get("cache_replay_store")
        if isinstance(cache_meta.get("cache_replay_store"), dict)
        else {}
    )
    reasons = {
        public_label(value, "unknown")
        for value in (
            cache_meta.get("reason"),
            replay_canary.get("reason"),
            cache_meta.get("invalidation_reason"),
            store_meta.get("reason"),
        )
        if public_label(value, "unknown") != "unknown"
    }
    for item in pattern_rules.get("skip_reasons") or []:
        if isinstance(item, dict):
            reason = public_label(item.get("reason"), "unknown")
            if reason != "unknown":
                reasons.add(reason)
    cache_replay_blocker_reasons = {
        public_label(value, "unknown")
        for value in (cache_meta.get("cache_replay_blocker_reasons") or [])
        if public_label(value, "unknown") != "unknown"
    }
    audit = replay_canary.get("dependency_audit")
    if not isinstance(audit, dict):
        audit = cache_meta.get("file_dependency_audit") if isinstance(cache_meta.get("file_dependency_audit"), dict) else {}

    blockers: set[str] = set()
    requires_invalidation_evidence = _cache_replay_requires_invalidation_evidence(pattern_rule, replay_canary, cache_meta)
    invalidation_reasons = reasons | cache_replay_blocker_reasons
    if any("ttl" in reason or "expired" in reason for reason in reasons):
        blockers.add("ttl-expiry")
    if (
        cache_meta.get("invalidated")
        or any(
            "invalidation" in reason
            or reason in {"dependency-changed", "dependency-deleted", "dependency-created", "dependency-cap-exceeded"}
            for reason in invalidation_reasons
        )
        or (
            requires_invalidation_evidence
            and any(reason.startswith("dependency-") or reason == "file-dependency-missing" for reason in invalidation_reasons)
        )
        or (
            requires_invalidation_evidence
            and isinstance(audit, dict)
            and audit.get("safe_invalidation_evidence") is False
        )
    ):
        blockers.add("invalidation-risk")
    mismatch_reasons = {
        reason
        for reason in reasons
        if reason.endswith("-mismatch")
        or reason in {
            "cohort-mismatch",
            "category-excluded",
            "unsupported-endpoint",
            "replayability-gate-mismatch",
        }
    }
    if "pattern-hash-mismatch" in mismatch_reasons:
        blockers.add("fingerprint-drift")
    cohort_mismatch_reasons = {
        "source_surface-mismatch",
        "endpoint-mismatch",
        "category-mismatch",
        "workflow_phase-mismatch",
        "text_bucket-mismatch",
        "token_bucket-mismatch",
        "cohort-mismatch",
    }
    if any(reason in mismatch_reasons for reason in cohort_mismatch_reasons):
        blockers.add("cohort-mismatch")
    if any("normalization" in reason for reason in reasons):
        blockers.add("normalization-mismatch")
    if any("ttl-window" in reason or "ttl-not-elapsed" in reason for reason in reasons):
        blockers.add("ttl-window-not-elapsed")
    if any("bypass" in reason for reason in reasons):
        blockers.add("canary-bypass")
    if not pattern_rule:
        blockers.add("cohort-mismatch")
    store_status = public_label(store_meta.get("status"), "unknown")
    if store_status == "stored":
        store_reason = public_label(store_meta.get("reason"), "unknown")
        if store_reason in {"compatible-success-response", "chat-compatible", "responses-compatible"}:
            blockers.add("first-seen-cache-warmup")
        elif any("write-read" in reason or "hit-recovery" in reason for reason in reasons | {store_reason}):
            blockers.add("replay-write-read-disconnect")
        else:
            blockers.add("first-seen-cache-warmup")
    elif store_status == "skipped":
        blockers.add("cache-write-absence")
    elif _as_int(row.get("status_code"), 200) >= 400:
        blockers.add("upstream-error-before-cache-write")
    elif cache_meta.get("status") == "miss":
        blockers.add("cache-write-absence")
    if "cache-warmup-miss" in reasons and not blockers:
        blockers.add("first-seen-cache-warmup")
    if not blockers:
        blockers.add("uncategorized-applied-cache-miss")
    return sorted(blockers)


def _cache_replay_row_observed_savings(row: dict[str, Any], cache_meta: dict[str, Any], evidence_class: str) -> float:
    if evidence_class != "exact_hit":
        return 0.0
    saved = _as_float(cache_meta.get("estimated_saved_cost_usd"))
    if saved > 0:
        return saved
    baseline = _as_float(row.get("cost_baseline_usd"))
    actual = _as_float(row.get("cost_est_usd"))
    return max(0.0, baseline - actual)


def _cache_replay_warmup_window_analysis(
    *,
    warmup_miss_count: int,
    non_warmup_miss_count: int,
    applied_miss_count: int,
    observed_hits: int,
    projected_hits: int,
    projected_savings_usd: float,
    ttl_seconds: int,
    first_warmup_miss_at: datetime | None,
    latest_warmup_miss_at: datetime | None,
    now: datetime,
    top_applied_miss_blocker: str | None,
) -> dict[str, Any]:
    ttl = max(0, int(ttl_seconds or 0))
    ttl_hours = round(ttl / 3600.0, 6) if ttl else None
    first_age_hours = (
        round((now - first_warmup_miss_at).total_seconds() / 3600.0, 3)
        if first_warmup_miss_at
        else None
    )
    latest_age_hours = (
        round((now - latest_warmup_miss_at).total_seconds() / 3600.0, 3)
        if latest_warmup_miss_at
        else None
    )
    warmup_only = bool(warmup_miss_count > 0 and non_warmup_miss_count == 0 and observed_hits <= 0)
    repeat_window_elapsed = bool(
        warmup_only
        and ttl_hours is not None
        and first_age_hours is not None
        and first_age_hours >= ttl_hours
    )
    later_exact_repeat_expected = bool(warmup_only and projected_hits > 0)
    later_exact_repeat_absent = bool(later_exact_repeat_expected and observed_hits <= 0 and repeat_window_elapsed)
    if observed_hits > 0:
        status = "live-repeat-observed"
        classification = "hit-recovered"
        next_action = "review-cache-replay-promotion"
    elif warmup_only and repeat_window_elapsed:
        status = "repeat-window-elapsed-no-live-repeat"
        classification = "first-seen-warmup-no-later-repeat-yet"
        next_action = "keep-staged-until-live-repeat-or-blocker"
    elif warmup_only:
        status = "warmup-window-open"
        classification = "first-seen-warmup"
        next_action = "continue-cache-replay-warmup"
    elif non_warmup_miss_count > 0:
        status = "ineffective-replay-evidence"
        classification = "non-warmup-applied-miss"
        next_action = "keep-cache-replay-blocked"
    else:
        status = "no-applied-warmup-misses"
        classification = "no-warmup-evidence"
        next_action = "collect-cache-replay-canary-traffic"
    return {
        "schema": "tokenclaw.request_shape_cache_replay_warmup_analysis.v1",
        "status": status,
        "classification": classification,
        "next_action": next_action,
        "warmup_only_applied_misses": warmup_only,
        "warmup_miss_count": warmup_miss_count,
        "applied_miss_count": applied_miss_count,
        "non_warmup_miss_count": non_warmup_miss_count,
        "observed_hit_blocker": top_applied_miss_blocker if warmup_only and observed_hits <= 0 else None,
        "first_warmup_age_hours": first_age_hours,
        "latest_warmup_age_hours": latest_age_hours,
        "repeat_window": {
            "schema": "tokenclaw.request_shape_cache_replay_repeat_window.v1",
            "ttl_seconds": ttl,
            "ttl_hours": ttl_hours,
            "eligible": bool(warmup_miss_count > 0 and ttl > 0),
            "elapsed": repeat_window_elapsed,
            "projected_hits": projected_hits,
            "observed_hits": observed_hits,
            "projected_savings_usd": round(projected_savings_usd, 6),
            "later_exact_repeat_expected": later_exact_repeat_expected,
            "later_exact_repeat_absent": later_exact_repeat_absent,
            "reason": status,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
        "cache_entries_written": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_replay_reobserve_successor_resolution(
    *,
    decision: str,
    applied_count: int,
    holdout_count: int,
    observed_hits: int,
    warmup_analysis: dict[str, Any],
) -> str:
    """Map a reobserve-window decision to one acceptance successor outcome.

    A stale staged cache replay successor must resolve to one of
    ``rollback-required``, ``retire-staged-no-repeat``,
    ``fresh-applied-holdout-evidence`` or ``keep-staged-warmup`` instead of
    being left only as ``evidence-older-than-max-age``.  Decisions that simply
    keep collecting bounded fresh traffic stay ``reobserve-window-open``.
    """
    if decision == "rollback-required":
        return "rollback-required"
    if decision == "retire-staged-no-repeat":
        return "retire-staged-no-repeat"
    has_applied_holdout = applied_count > 0 and holdout_count > 0
    if has_applied_holdout and observed_hits > 0:
        return "fresh-applied-holdout-evidence"
    if (
        has_applied_holdout
        and observed_hits <= 0
        and bool(warmup_analysis.get("warmup_only_applied_misses"))
    ):
        return "keep-staged-warmup"
    if decision == "reobserve":
        return "reobserve-window-open"
    return "not-required"


def _cache_replay_bounded_reobserve_window(
    *,
    status: str,
    reason: str,
    stale: bool,
    stale_reason: str,
    stale_zero_traffic_rule_count: int,
    staged_canaries: list[dict[str, Any]],
    observed_rows: int,
    applied_count: int,
    holdout_count: int,
    observed_hits: int,
    projected_hits: int,
    projected_savings_usd: float,
    observed_savings_usd: float,
    warmup_analysis: dict[str, Any],
    max_age_hours: float,
    lifecycle_counts: dict[str, Any] | None = None,
    blocker_breakdown: list[dict[str, Any]] | None = None,
    age_hours: float | None = None,
) -> dict[str, Any]:
    top_canary = staged_canaries[0] if staged_canaries else {}
    lifecycle_counts = lifecycle_counts or {}
    ttl_seconds = _as_int(top_canary.get("ttl_seconds"), DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS)
    repeat_window = (
        warmup_analysis.get("repeat_window")
        if isinstance(warmup_analysis.get("repeat_window"), dict)
        else {}
    )
    traffic_floor = {
        "schema": "tokenclaw.request_shape_cache_replay_reobserve_traffic_floor.v1",
        "minimum_observed_rows": DEFAULT_CACHE_REPLAY_MIN_STAGE_ROWS,
        "minimum_applied_count": 1,
        "minimum_holdout_count": 1,
        "minimum_observed_hits_for_promotion": 1,
        "minimum_repeat_window_seconds": ttl_seconds,
        "projected_hits": projected_hits,
        "projected_savings_usd": round(projected_savings_usd, 6),
        "metadata_only": True,
        "aggregate_only": True,
    }
    traffic_floor_met = bool(
        observed_rows >= traffic_floor["minimum_observed_rows"]
        and applied_count >= traffic_floor["minimum_applied_count"]
        and holdout_count >= traffic_floor["minimum_holdout_count"]
    )
    repeat_absent_after_floor = bool(
        traffic_floor_met
        and repeat_window.get("elapsed")
        and repeat_window.get("later_exact_repeat_expected")
        and repeat_window.get("later_exact_repeat_absent")
    )
    if repeat_absent_after_floor:
        decision = "retire-staged-no-repeat"
        window_status = "reobserve-window-complete-no-repeat"
        next_action = "retire-cache-replay-canary-no-repeat"
        blocker_codes = ["repeat-window-elapsed-no-live-repeat"]
        opens_after = "already-open"
    elif stale:
        decision = "rollback-required"
        window_status = "rollback-required-before-reobserve"
        next_action = "apply-cache-replay-rollback-before-reobserve"
        blocker_codes = [stale_reason or reason or "stale-cache-replay-evidence"]
        opens_after = "rollback-applied"
    elif status in {"staged-no-traffic", "observed"}:
        decision = "reobserve"
        window_status = "reobserve-window-open"
        next_action = "collect-cache-replay-canary-traffic"
        blocker_codes = [] if observed_rows > 0 else ["missing-observed-cache-replay-traffic"]
        opens_after = "already-open"
    else:
        decision = "not-required"
        window_status = "not-required"
        next_action = reason or "cache-replay-reobserve-not-required"
        blocker_codes = []
        opens_after = "not-required"
    successor_resolution = _cache_replay_reobserve_successor_resolution(
        decision=decision,
        applied_count=applied_count,
        holdout_count=holdout_count,
        observed_hits=observed_hits,
        warmup_analysis=warmup_analysis,
    )
    freshness_status = "stale" if stale else ("fresh" if observed_rows > 0 else "no-fresh-evidence")
    safe_blocker_breakdown = [
        {"value": public_label(item.get("value"), "unknown"), "count": _as_int(item.get("count"))}
        for item in (blocker_breakdown or [])
        if isinstance(item, dict) and public_label(item.get("value"), "unknown") != "unknown"
    ]
    recorded_evidence = {
        "schema": "tokenclaw.request_shape_cache_replay_reobserve_recorded_evidence.v1",
        "freshness_status": freshness_status,
        "age_hours": age_hours,
        "max_age_hours": float(max_age_hours),
        "canary_fraction": round(_as_float(top_canary.get("canary_fraction")), 6),
        "holdout_fraction": round(_as_float(top_canary.get("holdout_fraction")), 6),
        "projected_hits": projected_hits,
        "projected_savings_usd": round(projected_savings_usd, 6),
        "observed_savings_usd": round(observed_savings_usd, 6),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "observed_hits": observed_hits,
        "exact_hit_count": observed_hits,
        "miss_count": _as_int(lifecycle_counts.get("miss_count")),
        "warmup_miss_count": _as_int(warmup_analysis.get("warmup_miss_count")),
        "non_warmup_miss_count": _as_int(warmup_analysis.get("non_warmup_miss_count")),
        "repeat_window_status": public_label(repeat_window.get("reason") or warmup_analysis.get("status"), "unknown"),
        "repeat_window_elapsed": bool(repeat_window.get("elapsed")),
        "later_exact_repeat_expected": bool(repeat_window.get("later_exact_repeat_expected")),
        "later_exact_repeat_absent": bool(repeat_window.get("later_exact_repeat_absent")),
        "retry_count": _as_int(lifecycle_counts.get("retry_count")),
        "error_count": _as_int(lifecycle_counts.get("error_count")),
        "fallback_count": _as_int(lifecycle_counts.get("fallback_count")),
        "invalidation_skipped_count": _as_int(lifecycle_counts.get("invalidation_skipped_count")),
        "unsupported_shape_count": _as_int(lifecycle_counts.get("unsupported_shape_count")),
        "blocker_breakdown": safe_blocker_breakdown,
        "retirement_required": decision == "retire-staged-no-repeat",
        "rollback_required": decision == "rollback-required",
        "metadata_only": True,
        "aggregate_only": True,
    }
    durable_decision = {
        "schema": "tokenclaw.request_shape_cache_replay_reobserve_durable_decision.v1",
        "decision": decision,
        "status": window_status,
        "freshness_status": freshness_status,
        "successor_resolution": successor_resolution,
        "next_action": next_action,
        "observed_coverage": {
            "observed_row_count": observed_rows,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "exact_hit_count": observed_hits,
            "miss_count": _as_int(lifecycle_counts.get("miss_count")),
            "warmup_miss_count": _as_int(warmup_analysis.get("warmup_miss_count")),
            "non_warmup_miss_count": _as_int(warmup_analysis.get("non_warmup_miss_count")),
            "invalidation_skipped_count": _as_int(lifecycle_counts.get("invalidation_skipped_count")),
            "fallback_count": _as_int(lifecycle_counts.get("fallback_count")),
            "error_count": _as_int(lifecycle_counts.get("error_count")),
            "retry_count": _as_int(lifecycle_counts.get("retry_count")),
            "observed_savings_usd": round(observed_savings_usd, 6),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "blocker_breakdown": safe_blocker_breakdown,
        "emits_cache_apply_action": False,
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "metadata_only": True,
        "aggregate_only": True,
    }
    return {
        "schema": "tokenclaw.request_shape_cache_replay_bounded_reobserve_window.v1",
        "status": window_status,
        "decision": decision,
        "freshness_status": freshness_status,
        "successor_resolution": successor_resolution,
        "durable_decision": durable_decision,
        "next_action": next_action,
        "reason": stale_reason if stale else reason,
        "blocker_codes": blocker_codes,
        "opens_after": opens_after,
        "stale_zero_traffic_rule_count": stale_zero_traffic_rule_count,
        "recorded_evidence": recorded_evidence,
        "traffic_floor": traffic_floor,
        "traffic_floor_met": traffic_floor_met,
        "repeat_absent_after_floor": repeat_absent_after_floor,
        "observed": {
            "observed_row_count": observed_rows,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "observed_hits": observed_hits,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "expiry": {
            "schema": "tokenclaw.request_shape_cache_replay_reobserve_window_expiry.v1",
            "reference": "rollback_applied_at" if stale else "latest_observed_at",
            "max_age_hours": float(max_age_hours),
            "ttl_seconds": ttl_seconds,
            "expires_at_included": False,
            "expired": bool(stale),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def build_request_shape_cache_replay_evidence_report(
    store_obj: Any,
    *,
    rules_path: str | Path,
    limit: int = 10000,
    max_age_hours: float = DEFAULT_CACHE_REPLAY_CANARY_MAX_EVIDENCE_AGE_HOURS,
) -> dict[str, Any]:
    policy_file_name = Path(rules_path).name
    policy, policy_exists = _load_cache_replay_canary_policy(rules_path)
    staged_rules = _request_shape_cache_replay_policy_rules(policy)
    staged_rule_ids = {str(rule.get("rule_id")) for rule in staged_rules if rule.get("rule_id")}
    staged_candidate_ids = {str(rule.get("candidate_id")) for rule in staged_rules if rule.get("candidate_id")}
    candidate_to_rule_ids: dict[str, set[str]] = {}
    for rule in staged_rules:
        candidate_id = str(rule.get("candidate_id") or "")
        rule_id = str(rule.get("rule_id") or "")
        if candidate_id and rule_id:
            candidate_to_rule_ids.setdefault(candidate_id, set()).add(rule_id)
    capped_limit = max(1, min(int(limit or 1), 10000))
    rows = store_obj.optimization_action_ledger_rows(limit=capped_limit)

    lifecycle_counts = {
        "canary_applied_count": 0,
        "canary_holdout_count": 0,
        "exact_hit_count": 0,
        "miss_count": 0,
        "bypass_count": 0,
        "invalidation_skipped_count": 0,
        "unsupported_shape_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "error_count": 0,
    }
    blocker_counts: dict[str, int] = {}
    applied_miss_blocker_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    canary_status_counts: dict[str, int] = {}
    observed_rows = 0
    observed_savings = 0.0
    latest_observed: datetime | None = None
    first_warmup_miss_at: datetime | None = None
    latest_warmup_miss_at: datetime | None = None
    warmup_miss_count = 0
    non_warmup_miss_count = 0
    observed_staged_rule_ids: set[str] = set()
    observed_staged_candidate_ids: set[str] = set()

    for row in rows:
        cache_meta = _json_obj(row.get("cache_json"))
        if not cache_meta or not staged_rules:
            continue
        if not _cache_replay_row_matches_staged_rules(
            cache_meta,
            staged_rule_ids=staged_rule_ids,
            staged_candidate_ids=staged_candidate_ids,
        ):
            continue

        routing = _json_obj(row.get("routing_json"))
        observed_rows += 1
        row_rule_ids, row_candidate_ids = _cache_replay_staged_rule_observation_keys(
            cache_meta,
            staged_rule_ids=staged_rule_ids,
            staged_candidate_ids=staged_candidate_ids,
        )
        observed_staged_rule_ids.update(row_rule_ids)
        observed_staged_candidate_ids.update(row_candidate_ids)
        for candidate_id in row_candidate_ids:
            observed_staged_rule_ids.update(candidate_to_rule_ids.get(candidate_id, set()))
        replay_canary = (
            cache_meta.get("cache_replay_canary")
            if isinstance(cache_meta.get("cache_replay_canary"), dict)
            else {}
        )
        cache_status = public_label(cache_meta.get("status"), "unknown")
        canary_status = public_label(replay_canary.get("status"), "unknown")
        _increment(cache_status_counts, cache_status)
        _increment(canary_status_counts, canary_status)
        evidence_class, reason = _cache_replay_evidence_class(row, cache_meta)
        if evidence_class == "exact_hit":
            lifecycle_counts["canary_applied_count"] += 1
            lifecycle_counts["exact_hit_count"] += 1
        elif evidence_class == "miss":
            lifecycle_counts["canary_applied_count"] += 1
            lifecycle_counts["miss_count"] += 1
            miss_blockers = _cache_replay_applied_miss_blockers(row, cache_meta)
            for blocker in miss_blockers:
                _increment(applied_miss_blocker_counts, blocker)
            observed_at = _parse_utc(row.get("created_at"))
            if miss_blockers and all(blocker in _CACHE_REPLAY_NON_BLOCKING_APPLIED_MISS_BLOCKERS for blocker in miss_blockers):
                warmup_miss_count += 1
                if observed_at and (first_warmup_miss_at is None or observed_at < first_warmup_miss_at):
                    first_warmup_miss_at = observed_at
                if observed_at and (latest_warmup_miss_at is None or observed_at > latest_warmup_miss_at):
                    latest_warmup_miss_at = observed_at
            else:
                non_warmup_miss_count += 1
        elif evidence_class == "applied":
            lifecycle_counts["canary_applied_count"] += 1
        elif evidence_class == "holdout":
            lifecycle_counts["canary_holdout_count"] += 1
        elif evidence_class == "invalidation_skipped":
            lifecycle_counts["invalidation_skipped_count"] += 1
            _increment(blocker_counts, reason)
        elif evidence_class == "unsupported_shape":
            lifecycle_counts["unsupported_shape_count"] += 1
            _increment(blocker_counts, reason)
        else:
            lifecycle_counts["bypass_count"] += 1
            _increment(blocker_counts, reason)
        retry_count = _as_int(row.get("retry_count"))
        if retry_count > 0:
            lifecycle_counts["retry_count"] += retry_count
        status_code = _as_int(row.get("status_code"), 200)
        if status_code >= 400:
            lifecycle_counts["error_count"] += 1
            _increment(blocker_counts, "upstream-error-observed")
        if any(
            bool(container.get("fallback_reason") or container.get("fallback_applied") or container.get("fallback_required"))
            for container in (cache_meta, routing)
            if isinstance(container, dict)
        ):
            lifecycle_counts["fallback_count"] += 1
            _increment(blocker_counts, "fallback-observed")
        observed_savings += _cache_replay_row_observed_savings(row, cache_meta, evidence_class)
        observed_at = _parse_utc(row.get("created_at"))
        if observed_at and (latest_observed is None or observed_at > latest_observed):
            latest_observed = observed_at

    # Shape-based bypass detection: when no rows matched by canary metadata,
    # scan for rows whose cohort shape matches the staged rule but whose
    # cache_json lacks canary metadata (e.g. the proxy hasn't reloaded yet,
    # or the token_bucket condition in the rule can't be evaluated at runtime).
    # These are counted as bypass outcomes so lifecycle counts are nonzero and
    # the next_action becomes a narrower, more actionable signal than the
    # generic collect-cache-replay-canary-traffic.
    shape_bypass_count = 0
    if observed_rows == 0 and staged_rules:
        for row in rows:
            cache_meta = _json_obj(row.get("cache_json"))
            if not cache_meta:
                continue
            routing = _json_obj(row.get("routing_json"))
            for rule in staged_rules:
                if _row_shape_matches_staged_rule(row, routing, rule):
                    shape_bypass_count += 1
                    lifecycle_counts["bypass_count"] += 1
                    _increment(blocker_counts, "canary-rule-not-active")
                    observed_at = _parse_utc(row.get("created_at"))
                    if observed_at and (latest_observed is None or observed_at > latest_observed):
                        latest_observed = observed_at
                    break
    observed_rows += shape_bypass_count

    now = _parse_utc(utc_now()) or datetime.now(timezone.utc)
    stale_zero_traffic_rules = _cache_replay_stale_zero_traffic_rules(
        staged_rules,
        observed_rule_ids=observed_staged_rule_ids,
        observed_candidate_ids=observed_staged_candidate_ids,
        now=now,
        max_age_hours=max_age_hours,
    )
    staged_times = [
        parsed
        for rule in staged_rules
        if (parsed := _parse_utc(rule.get("staged_at"))) is not None
    ]
    reference_time = latest_observed or (min(staged_times) if staged_times else None)
    age_hours = round((now - reference_time).total_seconds() / 3600.0, 3) if reference_time else None
    stale = bool(stale_zero_traffic_rules or (age_hours is not None and age_hours > max_age_hours))
    durable_outcome: dict[str, Any] | None = None
    if stale_zero_traffic_rules:
        status = "staged-stale-no-traffic"
        reason = "stale-no-canary-traffic"
        next_action = "rollback-cache-replay-rule"
        durable_outcome = {
            "schema": "tokenclaw.request_shape_cache_replay_durable_outcome.v1",
            "decision": "rollback",
            "reason": reason,
            "next_action": next_action,
            "source_canary_policy_file": policy_file_name,
            "target_local_rule_file": "cache_rules.yaml",
            "policy_files_written": False,
            "cache_entries_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "metadata_only": True,
            "aggregate_only": True,
        }
    elif not policy_exists:
        status = "no-canary-policy"
        reason = "cache-canary-policy-missing"
        next_action = "stage-cache-replay-canary"
    elif not staged_rules:
        status = "no-request-shape-cache-replay-canary"
        reason = "request-shape-cache-replay-canary-missing"
        next_action = "stage-cache-replay-canary"
    elif observed_rows == 0:
        if stale:
            status = "staged-stale-no-traffic"
            reason = "stale-cache-replay-evidence"
            next_action = "rollback-cache-replay-rule"
            durable_outcome = {
                "schema": "tokenclaw.request_shape_cache_replay_durable_outcome.v1",
                "decision": "rollback",
                "reason": reason,
                "next_action": next_action,
                "source_canary_policy_file": policy_file_name,
                "target_local_rule_file": "cache_rules.yaml",
                "policy_files_written": False,
                "cache_entries_written": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "metadata_only": True,
                "aggregate_only": True,
            }
        else:
            status = "staged-no-traffic"
            reason = "missing-observed-cache-replay-traffic"
            next_action = "collect-cache-replay-canary-traffic"
    elif shape_bypass_count > 0 and observed_rows == shape_bypass_count:
        # All observed rows match the cohort shape but have no canary metadata,
        # meaning the canary rule is staged but not yet active in the proxy.
        status = "staged-bypass"
        reason = "canary-rule-not-active-in-proxy"
        next_action = "activate-cache-replay-canary-in-proxy"
    else:
        status = "observed"
        reason = "cache-replay-canary-evidence-observed"
        next_action = "review-cache-replay-canary-promotion-readiness"
    projected_hits = sum(
        _as_int((rule.get("graduation") or {}).get("projected_hits") or (rule.get("graduation") or {}).get("projected_hit_count"))
        for rule in staged_rules
        if isinstance(rule.get("graduation"), dict)
    )
    projected_savings = sum(
        _as_float((rule.get("graduation") or {}).get("projected_savings_usd") or (rule.get("graduation") or {}).get("projected_saved_cost_usd"))
        for rule in staged_rules
        if isinstance(rule.get("graduation"), dict)
    )
    staged_canary_summaries = _cache_replay_staged_policy_summary(staged_rules)
    observed_hits = lifecycle_counts["exact_hit_count"]
    applied_miss_blocker_breakdown = _breakdown(applied_miss_blocker_counts)
    top_applied_miss_blocker = applied_miss_blocker_breakdown[0]["value"] if applied_miss_blocker_breakdown else None
    top_ttl_seconds = _as_int(
        staged_canary_summaries[0].get("ttl_seconds") if staged_canary_summaries else None,
        DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS,
    )
    warmup_analysis = _cache_replay_warmup_window_analysis(
        warmup_miss_count=warmup_miss_count,
        non_warmup_miss_count=non_warmup_miss_count,
        applied_miss_count=lifecycle_counts["miss_count"],
        observed_hits=observed_hits,
        projected_hits=projected_hits,
        projected_savings_usd=projected_savings,
        ttl_seconds=top_ttl_seconds,
        first_warmup_miss_at=first_warmup_miss_at,
        latest_warmup_miss_at=latest_warmup_miss_at,
        now=now,
        top_applied_miss_blocker=top_applied_miss_blocker,
    )
    stale_reason = (
        "stale-no-canary-traffic"
        if stale_zero_traffic_rules
        else ("evidence-older-than-max-age" if stale else "fresh-or-not-yet-observed")
    )
    reobserve_window = _cache_replay_bounded_reobserve_window(
        status=status,
        reason=reason,
        stale=stale,
        stale_reason=stale_reason,
        stale_zero_traffic_rule_count=len(stale_zero_traffic_rules),
        staged_canaries=staged_canary_summaries,
        observed_rows=observed_rows,
        applied_count=lifecycle_counts["canary_applied_count"],
        holdout_count=lifecycle_counts["canary_holdout_count"],
        observed_hits=observed_hits,
        projected_hits=projected_hits,
        projected_savings_usd=projected_savings,
        observed_savings_usd=observed_savings,
        warmup_analysis=warmup_analysis,
        max_age_hours=max_age_hours,
        lifecycle_counts=lifecycle_counts,
        blocker_breakdown=_breakdown(blocker_counts),
        age_hours=age_hours,
    )
    return {
        "schema": REPLAY_CACHE_CANARY_EVIDENCE_SCHEMA,
        "status": status,
        "ok": bool(staged_rules),
        "generated_at": utc_now(),
        "reason": reason,
        "next_action": (
            reobserve_window["next_action"]
            if reobserve_window["decision"] in {"rollback-required", "retire-staged-no-repeat"}
            else next_action
        ),
        "policy_next_action": next_action,
        "source": {
            "policy_file": policy_file_name,
            "policy_file_present": policy_exists,
            "policy_path_included": False,
            "rows_scanned": len(rows),
            "lookback_limit": capped_limit,
        },
        "staged_canary_count": len(staged_rules),
        "staged_canaries": staged_canary_summaries,
        "stale_zero_traffic_rule_count": len(stale_zero_traffic_rules),
        "stale_zero_traffic_rules": stale_zero_traffic_rules,
        "summary": {
            "observed_row_count": observed_rows,
            "applied_count": lifecycle_counts["canary_applied_count"],
            "holdout_count": lifecycle_counts["canary_holdout_count"],
            "exact_hit_count": lifecycle_counts["exact_hit_count"],
            "miss_count": lifecycle_counts["miss_count"],
            "bypass_count": lifecycle_counts["bypass_count"],
            "invalidation_skipped_count": lifecycle_counts["invalidation_skipped_count"],
            "unsupported_shape_count": lifecycle_counts["unsupported_shape_count"],
            "retry_count": lifecycle_counts["retry_count"],
            "fallback_count": lifecycle_counts["fallback_count"],
            "error_count": lifecycle_counts["error_count"],
            "projected_hits": projected_hits,
            "observed_hits": observed_hits,
            "projected_savings_usd": round(projected_savings, 6),
            "observed_savings_usd": round(observed_savings, 6),
            "hit_observation_rate": round(observed_hits / float(lifecycle_counts["canary_applied_count"]), 6)
            if lifecycle_counts["canary_applied_count"] else 0.0,
            "top_blocker": _breakdown(blocker_counts)[0]["value"] if blocker_counts else None,
            "top_applied_miss_blocker": top_applied_miss_blocker,
            "warmup_miss_count": warmup_analysis["warmup_miss_count"],
            "non_warmup_miss_count": warmup_analysis["non_warmup_miss_count"],
            "warmup_status": warmup_analysis["status"],
            "repeat_window_elapsed": warmup_analysis["repeat_window"]["elapsed"],
            "later_exact_repeat_expected": warmup_analysis["repeat_window"]["later_exact_repeat_expected"],
            "later_exact_repeat_absent": warmup_analysis["repeat_window"]["later_exact_repeat_absent"],
            "stale_zero_traffic_rule_count": len(stale_zero_traffic_rules),
            "reobserve_window_status": reobserve_window["status"],
            "reobserve_window_decision": reobserve_window["decision"],
            "reobserve_window_freshness_status": reobserve_window["freshness_status"],
            "reobserve_window_successor_resolution": reobserve_window["successor_resolution"],
            "reobserve_window_next_action": reobserve_window["next_action"],
        },
        "lifecycle_counts": lifecycle_counts,
        "cache_status_breakdown": _breakdown(cache_status_counts),
        "canary_status_breakdown": _breakdown(canary_status_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "miss_reason_breakdown": applied_miss_blocker_breakdown,
        "applied_miss_blocker_breakdown": applied_miss_blocker_breakdown,
        "warmup_analysis": warmup_analysis,
        "reobserve_window": reobserve_window,
        "durable_outcome": durable_outcome,
        "stale_evidence": {
            "max_age_hours": float(max_age_hours),
            "latest_observed_at": latest_observed.isoformat() if latest_observed else None,
            "reference_time": reference_time.isoformat() if reference_time else None,
            "age_hours": age_hours,
            "stale": stale,
            "reason": stale_reason,
            "zero_traffic_rule_count": len(stale_zero_traffic_rules),
        },
        "acceptance": {
            "has_staged_canary_metadata": bool(staged_rules),
            "reports_applied_and_holdout_counts": "canary_applied_count" in lifecycle_counts and "canary_holdout_count" in lifecycle_counts,
            "reports_projected_vs_observed_hits": projected_hits >= 0 and observed_hits >= 0,
            "reports_observed_savings_estimate": True,
            "reports_blocker_breakdown": isinstance(_breakdown(blocker_counts), list),
            "reports_applied_miss_blocker_breakdown": isinstance(applied_miss_blocker_breakdown, list),
            "reports_warmup_analysis": isinstance(warmup_analysis, dict),
            "reports_repeat_window_metadata": isinstance(warmup_analysis.get("repeat_window"), dict),
            "reports_bounded_reobserve_window": isinstance(reobserve_window, dict),
            "reports_reobserve_traffic_floor": isinstance(reobserve_window.get("traffic_floor"), dict),
            "reports_reobserve_recorded_evidence": isinstance(reobserve_window.get("recorded_evidence"), dict),
            "reports_reobserve_freshness_status": reobserve_window.get("freshness_status") in {"fresh", "stale", "no-fresh-evidence"},
            "reports_durable_reobserve_decision": isinstance(reobserve_window.get("durable_decision"), dict),
            "resolves_stale_successor_beyond_evidence_age": (
                (not stale)
                or reobserve_window["successor_resolution"]
                in {"rollback-required", "retire-staged-no-repeat", "fresh-applied-holdout-evidence", "keep-staged-warmup"}
            ),
            "reobserve_window_writes_no_cache_entries": reobserve_window["cache_entries_written"] == 0,
            "reobserve_window_emits_no_cache_apply_actions": reobserve_window["cache_apply_action_count"] == 0,
            "distinguishes_first_seen_warmup_from_ineffective_replay": True,
            "reports_stale_evidence_metadata": True,
            "reports_durable_rollback_or_retirement_reason": durable_outcome is not None,
            "aggregate_only": True,
        },
        "privacy": {
            **_cache_replay_evidence_privacy(),
            "rule_ids_included": bool(stale_zero_traffic_rules),
        },
    }


def _cache_replay_policy_decision_privacy() -> dict[str, Any]:
    privacy = dict(_cache_replay_evidence_privacy())
    privacy.update({
        "policy_patch_includes_raw_ids": False,
        "policy_files_written": False,
        "cache_entries_written": False,
    })
    return privacy


def _cache_replay_policy_decision_id(evidence: dict[str, Any], decision: str) -> str:
    top = _cache_replay_policy_decision_top_canary(evidence)
    basis = {
        "schema": REPLAY_CACHE_POLICY_DECISION_SCHEMA,
        "decision": decision,
        "shape": top.get("shape") if isinstance(top, dict) else {},
        "projected_hits": (evidence.get("summary") or {}).get("projected_hits")
        if isinstance(evidence.get("summary"), dict)
        else 0,
        "observed_hits": (evidence.get("summary") or {}).get("observed_hits")
        if isinstance(evidence.get("summary"), dict)
        else 0,
    }
    return f"cache-replay-policy-decision:{hashlib.sha256(stable_json(basis).encode('utf-8')).hexdigest()[:16]}"


def _cache_replay_policy_rule_id(evidence: dict[str, Any]) -> str:
    top = _cache_replay_policy_decision_top_canary(evidence)
    shape = top.get("shape") if isinstance(top, dict) else {}
    digest = hashlib.sha256(stable_json(shape).encode("utf-8")).hexdigest()[:16]
    return f"local-openai-cache-replay-promoted:{digest}"


def _cache_replay_policy_source_file(evidence: dict[str, Any] | None) -> str:
    source = evidence.get("source") if isinstance(evidence, dict) and isinstance(evidence.get("source"), dict) else {}
    return public_label(source.get("policy_file"), "cache_canary_policy.yaml")


def _cache_replay_policy_disable_patch_rules(evidence: dict[str, Any] | None, reason: str) -> list[dict[str, Any]]:
    stale_rules = (
        evidence.get("stale_zero_traffic_rules")
        if isinstance(evidence, dict) and isinstance(evidence.get("stale_zero_traffic_rules"), list)
        else []
    )
    patch_rules: list[dict[str, Any]] = []
    for rule in stale_rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("rule_id") or "")
        if not rule_id:
            continue
        patch_rules.append({
            "id": rule_id,
            "enabled": False,
            "disabled_reason": reason,
            "rollback_reason": "stale-no-canary-traffic",
            "evidence_age_hours": rule.get("age_hours"),
        })
    if patch_rules:
        return patch_rules
    return [
        {
            "id": _cache_replay_policy_rule_id(evidence or {}),
            "enabled": False,
            "disabled_reason": reason,
        }
    ]


def _cache_replay_policy_decision_top_canary(evidence: dict[str, Any]) -> dict[str, Any]:
    canaries = evidence.get("staged_canaries") if isinstance(evidence.get("staged_canaries"), list) else []
    for canary in canaries:
        if isinstance(canary, dict):
            return canary
    return {}


_CACHE_REPLAY_NON_BLOCKING_APPLIED_MISS_BLOCKERS = {
    "cache-warmup-miss",
    "first-seen-cache-warmup",
}
_CACHE_REPLAY_APPLIED_MISS_BLOCKER_PRIORITY = (
    "invalidation-risk",
    "fingerprint-drift",
    "cache-write-absence",
    "cohort-mismatch",
    "normalization-mismatch",
    "ttl-window-not-elapsed",
    "ttl-expiry",
    "canary-bypass",
    "replay-write-read-disconnect",
    "upstream-error-before-cache-write",
    "uncategorized-applied-cache-miss",
)


def _cache_replay_top_blocking_applied_miss_blocker(blockers: list[str]) -> str | None:
    blocking = [blocker for blocker in blockers if blocker not in _CACHE_REPLAY_NON_BLOCKING_APPLIED_MISS_BLOCKERS]
    if not blocking:
        return None
    for preferred in _CACHE_REPLAY_APPLIED_MISS_BLOCKER_PRIORITY:
        if preferred in blocking:
            return preferred
    return sorted(blocking)[0]


def _cache_replay_requires_invalidation_evidence(pattern_rule: dict[str, Any], replay_canary: dict[str, Any], cache_meta: dict[str, Any]) -> bool:
    action = pattern_rule.get("action") if isinstance(pattern_rule.get("action"), dict) else {}
    conditions = pattern_rule.get("conditions") if isinstance(pattern_rule.get("conditions"), dict) else {}
    if bool(action.get("allow_tool_calls") or action.get("safe_invalidation")):
        return True
    if bool(conditions.get("has_tools")):
        return True
    if bool(replay_canary.get("allow_tool_calls") or cache_meta.get("has_tools")):
        return True
    return False


def _cache_replay_policy_decision_metrics(evidence: dict[str, Any] | None) -> dict[str, Any]:
    summary = evidence.get("summary") if isinstance(evidence, dict) and isinstance(evidence.get("summary"), dict) else {}
    stale = evidence.get("stale_evidence") if isinstance(evidence, dict) and isinstance(evidence.get("stale_evidence"), dict) else {}
    warmup = evidence.get("warmup_analysis") if isinstance(evidence, dict) and isinstance(evidence.get("warmup_analysis"), dict) else {}
    repeat_window = warmup.get("repeat_window") if isinstance(warmup.get("repeat_window"), dict) else {}
    applied_miss_blockers = (
        evidence.get("applied_miss_blocker_breakdown")
        if isinstance(evidence, dict) and isinstance(evidence.get("applied_miss_blocker_breakdown"), list)
        else []
    )
    applied = _as_int(summary.get("applied_count"))
    holdout = _as_int(summary.get("holdout_count"))
    observed_hits = _as_int(summary.get("observed_hits"))
    observed_savings = _as_float(summary.get("observed_savings_usd"))
    projected_hits = _as_int(summary.get("projected_hits"))
    projected_savings = _as_float(summary.get("projected_savings_usd"))
    applied_miss_blocker_values = [
        public_label(item.get("value"), "unknown")
        for item in applied_miss_blockers
        if isinstance(item, dict) and _as_int(item.get("count")) > 0
    ]
    applied_miss_blocker_values = [value for value in applied_miss_blocker_values if value != "unknown"]
    return {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision_metrics.v1",
        "staged_canary_count": _as_int(evidence.get("staged_canary_count")) if isinstance(evidence, dict) else 0,
        "observed_row_count": _as_int(summary.get("observed_row_count")),
        "applied_count": applied,
        "holdout_count": holdout,
        "exact_hit_count": _as_int(summary.get("exact_hit_count")),
        "miss_count": _as_int(summary.get("miss_count")),
        "bypass_count": _as_int(summary.get("bypass_count")),
        "invalidation_skipped_count": _as_int(summary.get("invalidation_skipped_count")),
        "unsupported_shape_count": _as_int(summary.get("unsupported_shape_count")),
        "retry_count": _as_int(summary.get("retry_count")),
        "fallback_count": _as_int(summary.get("fallback_count")),
        "error_count": _as_int(summary.get("error_count")),
        "projected_hits": projected_hits,
        "observed_hits": observed_hits,
        "projected_savings_usd": round(projected_savings, 6),
        "observed_savings_usd": round(observed_savings, 6),
        "hit_observation_rate": round(_as_float(summary.get("hit_observation_rate")), 6),
        "savings_realization_ratio": round(observed_savings / projected_savings, 6) if projected_savings > 0 else 0.0,
        "top_applied_miss_blocker": applied_miss_blockers[0].get("value")
        if applied_miss_blockers and isinstance(applied_miss_blockers[0], dict)
        else summary.get("top_applied_miss_blocker"),
        "top_blocking_applied_miss_blocker": _cache_replay_top_blocking_applied_miss_blocker(applied_miss_blocker_values),
        "warmup_status": public_label(warmup.get("status"), "unknown"),
        "warmup_classification": public_label(warmup.get("classification"), "unknown"),
        "warmup_miss_count": _as_int(warmup.get("warmup_miss_count") or summary.get("warmup_miss_count")),
        "non_warmup_miss_count": _as_int(warmup.get("non_warmup_miss_count") or summary.get("non_warmup_miss_count")),
        "first_warmup_age_hours": warmup.get("first_warmup_age_hours"),
        "latest_warmup_age_hours": warmup.get("latest_warmup_age_hours"),
        "repeat_window_elapsed": bool(repeat_window.get("elapsed") or summary.get("repeat_window_elapsed")),
        "later_exact_repeat_expected": bool(
            repeat_window.get("later_exact_repeat_expected") or summary.get("later_exact_repeat_expected")
        ),
        "later_exact_repeat_absent": bool(
            repeat_window.get("later_exact_repeat_absent") or summary.get("later_exact_repeat_absent")
        ),
        "stale_evidence": bool(stale.get("stale")),
        "evidence_age_hours": stale.get("age_hours"),
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_replay_shapes_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left or not right:
        return False
    comparable_keys = (
        "provider_family",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "text_bucket",
        "token_bucket",
        "has_tools",
        "stream",
    )
    shared_keys = [key for key in comparable_keys if key in left and key in right]
    return bool(shared_keys) and all(left.get(key) == right.get(key) for key in shared_keys)


def _cache_replay_policy_hit_recovery_metrics(
    hit_recovery_report: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    top = _cache_replay_policy_decision_top_canary(evidence or {})
    top_shape = top.get("shape") if isinstance(top.get("shape"), dict) else {}
    if not isinstance(hit_recovery_report, dict):
        return {
            "schema": "tokenclaw.request_shape_cache_replay_policy_hit_recovery_metrics.v1",
            "source_schema": None,
            "status": "unavailable",
            "reason": "hit-recovery-smoke-not-provided",
            "hit_recovery_demonstrated": False,
            "synthetic_exact_hit_count": 0,
            "synthetic_observed_hits": 0,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "cache_entries_written": False,
            "target_matches_staged_canary_shape": False,
            "target_rule_id_included": False,
            "synthetic_only": True,
            "metadata_only": True,
            "aggregate_only": True,
        }
    summary = hit_recovery_report.get("summary") if isinstance(hit_recovery_report.get("summary"), dict) else {}
    privacy = hit_recovery_report.get("privacy") if isinstance(hit_recovery_report.get("privacy"), dict) else {}
    target_shape = hit_recovery_report.get("target_shape") if isinstance(hit_recovery_report.get("target_shape"), dict) else {}
    return {
        "schema": "tokenclaw.request_shape_cache_replay_policy_hit_recovery_metrics.v1",
        "source_schema": hit_recovery_report.get("schema"),
        "status": public_label(hit_recovery_report.get("status"), "unknown"),
        "reason": public_label(hit_recovery_report.get("reason"), "unknown"),
        "hit_recovery_demonstrated": bool(summary.get("hit_recovery_demonstrated")),
        "synthetic_exact_hit_count": _as_int(summary.get("exact_hit_count")),
        "synthetic_observed_hits": _as_int(summary.get("observed_hits")),
        "provider_calls_made": bool(summary.get("provider_calls_made")),
        "managed_server_calls_made": bool(privacy.get("managed_server_calls_made")),
        "cache_entries_written": bool(summary.get("cache_entries_written")),
        "target_shape": {
            key: target_shape.get(key)
            for key in (
                "provider_family",
                "source_surface",
                "endpoint",
                "category",
                "workflow_phase",
                "text_bucket",
                "token_bucket",
                "has_tools",
                "stream",
            )
            if key in target_shape
        },
        "target_matches_staged_canary_shape": _cache_replay_shapes_match(target_shape, top_shape),
        "target_rule_id_included": False,
        "synthetic_only": bool(privacy.get("synthetic_only", True)),
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_replay_applied_miss_blocker_values(evidence: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence, dict):
        return []
    breakdown = evidence.get("applied_miss_blocker_breakdown")
    if not isinstance(breakdown, list):
        return []
    values: list[str] = []
    for item in breakdown:
        if not isinstance(item, dict):
            continue
        value = public_label(item.get("value"), "unknown")
        if value != "unknown" and _as_int(item.get("count")) > 0:
            values.append(value)
    return list(dict.fromkeys(values))


def _cache_replay_warmup_only_applied_miss(evidence: dict[str, Any] | None) -> bool:
    metrics = _cache_replay_policy_decision_metrics(evidence)
    blockers = _cache_replay_applied_miss_blocker_values(evidence)
    if not blockers:
        return False
    return (
        metrics["applied_count"] > 0
        and metrics["holdout_count"] > 0
        and metrics["miss_count"] > 0
        and metrics["observed_hits"] <= 0
        and all(blocker in _CACHE_REPLAY_NON_BLOCKING_APPLIED_MISS_BLOCKERS for blocker in blockers)
    )


def _cache_replay_retire_no_repeat_warmup(
    evidence: dict[str, Any] | None,
    hit_recovery_metrics: dict[str, Any] | None,
) -> bool:
    metrics = _cache_replay_policy_decision_metrics(evidence)
    if not _cache_replay_warmup_only_applied_miss(evidence):
        return False
    if not isinstance(hit_recovery_metrics, dict):
        return False
    return (
        bool(hit_recovery_metrics.get("hit_recovery_demonstrated"))
        and bool(hit_recovery_metrics.get("target_matches_staged_canary_shape"))
        and metrics["repeat_window_elapsed"]
        and metrics["later_exact_repeat_expected"]
        and metrics["later_exact_repeat_absent"]
        and metrics["observed_hits"] <= 0
    )


def _cache_replay_policy_warmup_analysis(evidence: dict[str, Any] | None, metrics: dict[str, Any]) -> dict[str, Any]:
    if isinstance(evidence, dict) and isinstance(evidence.get("warmup_analysis"), dict):
        warmup = dict(evidence["warmup_analysis"])
        repeat_window = warmup.get("repeat_window") if isinstance(warmup.get("repeat_window"), dict) else {}
        warmup["repeat_window"] = {
            **repeat_window,
            "metadata_only": True,
            "aggregate_only": True,
        }
        warmup.update({
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_written": False,
            "cache_entries_written": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "metadata_only": True,
            "aggregate_only": True,
        })
        return warmup
    top = metrics.get("top_applied_miss_blocker")
    warmup_only = bool(_cache_replay_warmup_only_applied_miss(evidence))
    return {
        "schema": "tokenclaw.request_shape_cache_replay_warmup_analysis.v1",
        "status": "warmup-only-without-window-metadata" if warmup_only else "unavailable",
        "classification": "first-seen-warmup" if warmup_only else "unknown",
        "next_action": "continue-cache-replay-warmup" if warmup_only else "collect-cache-replay-canary-traffic",
        "warmup_only_applied_misses": warmup_only,
        "warmup_miss_count": metrics.get("miss_count", 0) if warmup_only else 0,
        "applied_miss_count": metrics.get("miss_count", 0),
        "non_warmup_miss_count": 0 if warmup_only else metrics.get("miss_count", 0),
        "observed_hit_blocker": top if warmup_only and metrics.get("observed_hits", 0) <= 0 else None,
        "first_warmup_age_hours": metrics.get("evidence_age_hours"),
        "latest_warmup_age_hours": metrics.get("evidence_age_hours"),
        "repeat_window": {
            "schema": "tokenclaw.request_shape_cache_replay_repeat_window.v1",
            "ttl_seconds": None,
            "ttl_hours": None,
            "eligible": False,
            "elapsed": False,
            "projected_hits": metrics.get("projected_hits", 0),
            "observed_hits": metrics.get("observed_hits", 0),
            "projected_savings_usd": metrics.get("projected_savings_usd", 0.0),
            "later_exact_repeat_expected": bool(warmup_only and metrics.get("projected_hits", 0) > 0),
            "later_exact_repeat_absent": False,
            "reason": "warmup-window-metadata-unavailable",
            "metadata_only": True,
            "aggregate_only": True,
        },
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
        "cache_entries_written": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_replay_policy_reason_codes(evidence: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence, dict):
        return ["missing-cache-replay-evidence"]
    metrics = _cache_replay_policy_decision_metrics(evidence)
    codes: list[str] = []
    if metrics["staged_canary_count"] <= 0:
        codes.append("missing-cache-replay-canary-policy")
    if metrics["applied_count"] <= 0:
        codes.append("missing-applied-coverage")
    if metrics["holdout_count"] <= 0:
        codes.append("missing-holdout-coverage")
    if metrics["observed_hits"] <= 0:
        codes.append("missing-observed-cache-hits")
    if metrics["observed_savings_usd"] <= 0:
        codes.append("missing-observed-cache-savings")
    if metrics["applied_count"] > 0 and metrics["holdout_count"] > 0 and metrics["miss_count"] > 0 and metrics["observed_hits"] <= 0:
        codes.append("applied-cache-replay-miss-observed")
        if _cache_replay_warmup_only_applied_miss(evidence):
            top_warmup_blocker = metrics.get("top_applied_miss_blocker") or "cache-warmup-miss"
            codes.append(public_label(top_warmup_blocker, "cache-warmup-miss"))
    if metrics["stale_evidence"]:
        codes.append("stale-cache-replay-evidence")
    if _as_int(evidence.get("stale_zero_traffic_rule_count")) > 0:
        codes.append("stale-no-canary-traffic")
    if metrics["invalidation_skipped_count"] > 0:
        codes.append("invalidation-or-stale-risk-observed")
    if metrics["unsupported_shape_count"] > 0:
        codes.append("unsupported-cache-replay-shape-observed")
    if metrics["fallback_count"] > 0:
        codes.append("cache-replay-fallback-observed")
    if metrics["error_count"] > 0:
        codes.append("cache-replay-error-observed")
    if metrics["retry_count"] > 0:
        codes.append("cache-replay-retry-observed")
    blocker_breakdown = evidence.get("blocker_breakdown") if isinstance(evidence.get("blocker_breakdown"), list) else []
    for item in blocker_breakdown[:3]:
        if isinstance(item, dict):
            code = public_label(item.get("value"), "unknown")
            if code != "unknown":
                codes.append(code)
    applied_miss_breakdown = (
        evidence.get("applied_miss_blocker_breakdown")
        if isinstance(evidence.get("applied_miss_blocker_breakdown"), list)
        else []
    )
    for item in applied_miss_breakdown[:3]:
        if isinstance(item, dict):
            code = public_label(item.get("value"), "unknown")
            if code != "unknown":
                codes.append(f"applied-miss:{code}")
    return list(dict.fromkeys(codes))


def _cache_replay_policy_decision_value(
    evidence: dict[str, Any] | None,
    hit_recovery_metrics: dict[str, Any] | None = None,
) -> str:
    metrics = _cache_replay_policy_decision_metrics(evidence)
    if metrics["stale_evidence"]:
        return "rollback"
    if metrics["invalidation_skipped_count"] > 0 or metrics["unsupported_shape_count"] > 0:
        return "keep-blocked"
    if metrics["fallback_count"] > 0 or metrics["error_count"] > 0:
        return "keep-blocked"
    if metrics["staged_canary_count"] <= 0:
        return "keep-blocked"
    if (
        metrics["applied_count"] > 0
        and metrics["holdout_count"] > 0
        and metrics["observed_hits"] > 0
        and metrics["observed_savings_usd"] > 0
    ):
        return "widen"
    if metrics["applied_count"] > 0 and metrics["holdout_count"] > 0 and metrics["miss_count"] > 0:
        if _cache_replay_retire_no_repeat_warmup(evidence, hit_recovery_metrics):
            return "retire-staged-no-repeat"
        if _cache_replay_warmup_only_applied_miss(evidence):
            return "keep-staged"
        return "keep-blocked"
    return "keep-staged"


def _cache_replay_policy_decision_reason(evidence: dict[str, Any] | None, decision: str) -> str:
    if decision == "widen":
        return "cache-replay-canary-hit-recovery-observed"
    if decision == "rollback":
        if isinstance(evidence, dict) and _as_int(evidence.get("stale_zero_traffic_rule_count")) > 0:
            return "stale-no-canary-traffic"
        return "stale-cache-replay-evidence"
    if decision == "retire-staged-no-repeat":
        return "repeat-window-elapsed-no-live-repeat"
    metrics = _cache_replay_policy_decision_metrics(evidence)
    codes = _cache_replay_policy_reason_codes(evidence)
    if decision == "keep-staged" and _cache_replay_warmup_only_applied_miss(evidence):
        return str(metrics.get("top_applied_miss_blocker") or "cache-warmup-miss")
    if decision == "keep-blocked" and metrics.get("top_blocking_applied_miss_blocker"):
        return str(metrics["top_blocking_applied_miss_blocker"])
    if decision == "keep-blocked" and "applied-cache-replay-miss-observed" in codes:
        return "applied-cache-replay-miss-observed"
    return codes[0] if codes else "cache-replay-promotion-blocked"


def _cache_replay_policy_promotion_blocker(
    evidence: dict[str, Any] | None,
    decision: str,
    promotion_readiness: str,
    reason: str,
) -> str | None:
    if promotion_readiness == "promotion-ready":
        return None
    if decision == "keep-staged" and _cache_replay_warmup_only_applied_miss(evidence):
        metrics = _cache_replay_policy_decision_metrics(evidence)
        return str(metrics.get("top_applied_miss_blocker") or reason or "first-seen-cache-warmup")
    return reason or None


def _cache_replay_policy_observed_hit_blocker(
    evidence: dict[str, Any] | None,
    promotion_blocker: str | None,
) -> str | None:
    metrics = _cache_replay_policy_decision_metrics(evidence)
    if metrics["observed_hits"] > 0:
        return None
    return promotion_blocker or "missing-observed-cache-hits"


def _cache_replay_policy_recommended_next_action(decision: str) -> str:
    return {
        "widen": "promote-cache-replay-rule",
        "rollback": "rollback-cache-replay-rule",
        "retire-staged-no-repeat": "retire-cache-replay-canary-no-repeat",
        "keep-staged": "keep-cache-replay-canary-staged",
        "keep-blocked": "keep-cache-replay-blocked",
    }[decision]


def _cache_replay_canary_promotion_decision(evidence: dict[str, Any] | None, decision: str) -> str:
    if decision == "widen":
        return "promote"
    if decision == "retire-staged-no-repeat":
        return "retire-staged-no-repeat"
    if decision == "keep-staged" and _cache_replay_warmup_only_applied_miss(evidence):
        return "keep-staged-warmup"
    return "keep-blocked"


def _cache_replay_policy_promotion_readiness(evidence: dict[str, Any] | None, decision: str) -> str:
    if decision == "widen":
        return "promotion-ready"
    if decision == "rollback":
        return "rollback-required"
    if decision == "retire-staged-no-repeat":
        return "retire-staged-no-repeat"
    if decision == "keep-staged" and _cache_replay_warmup_only_applied_miss(evidence):
        return "keep-staged-warmup"
    return decision


def _cache_replay_policy_rollback_metadata(evidence: dict[str, Any] | None, decision: str) -> dict[str, Any]:
    reason = _cache_replay_policy_decision_reason(evidence, decision)
    patch_rules = _cache_replay_policy_disable_patch_rules(evidence, reason)
    return {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision_rollback_metadata.v1",
        "required_for_promotion": True,
        "rollback_action_type": "disable_openai_exact_cache_replay_policy",
        "target_local_rule_file": "cache_rules.yaml",
        "source_canary_policy_file": _cache_replay_policy_source_file(evidence),
        "rule_id": patch_rules[0]["id"] if patch_rules else None,
        "rule_count": len(patch_rules),
        "reason": reason,
        "disable_patch": {
            "pattern_rules": patch_rules,
        },
        "metadata_only": True,
        "aggregate_only": True,
        "rules_path_included": False,
    }


def _cache_replay_policy_post_rollback_reobserve_window(
    evidence: dict[str, Any] | None,
    decision: str,
    promotion_readiness: str,
    metrics: dict[str, Any],
    warmup_analysis: dict[str, Any],
) -> dict[str, Any]:
    top = _cache_replay_policy_decision_top_canary(evidence or {})
    shape = top.get("shape") if isinstance(top.get("shape"), dict) else {}
    stale = evidence.get("stale_evidence") if isinstance(evidence, dict) and isinstance(evidence.get("stale_evidence"), dict) else {}
    max_age_hours = _as_float(stale.get("max_age_hours"), DEFAULT_CACHE_REPLAY_CANARY_MAX_EVIDENCE_AGE_HOURS)
    ttl_seconds = _as_int(top.get("ttl_seconds"), DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS)
    observed_rows = metrics["observed_row_count"]
    applied_count = metrics["applied_count"]
    holdout_count = metrics["holdout_count"]
    observed_hits = metrics["observed_hits"]
    repeat_window = warmup_analysis.get("repeat_window") if isinstance(warmup_analysis.get("repeat_window"), dict) else {}
    blocker_breakdown = (
        evidence.get("blocker_breakdown")
        if isinstance(evidence, dict) and isinstance(evidence.get("blocker_breakdown"), list)
        else []
    )
    traffic_floor = {
        "schema": "tokenclaw.request_shape_cache_replay_reobserve_traffic_floor.v1",
        "minimum_observed_rows": DEFAULT_CACHE_REPLAY_MIN_STAGE_ROWS,
        "minimum_applied_count": 1,
        "minimum_holdout_count": 1,
        "minimum_observed_hits_for_promotion": 1,
        "minimum_repeat_window_seconds": ttl_seconds,
        "projected_hits": metrics["projected_hits"],
        "projected_savings_usd": metrics["projected_savings_usd"],
        "sample_count": _as_int(top.get("sample_count")) or metrics["observed_row_count"],
        "metadata_only": True,
        "aggregate_only": True,
    }
    traffic_floor_met = bool(
        observed_rows >= traffic_floor["minimum_observed_rows"]
        and applied_count >= traffic_floor["minimum_applied_count"]
        and holdout_count >= traffic_floor["minimum_holdout_count"]
    )
    enough_for_promotion_review = bool(traffic_floor_met and observed_hits >= traffic_floor["minimum_observed_hits_for_promotion"])
    repeat_absent_after_floor = bool(
        traffic_floor_met
        and repeat_window.get("elapsed")
        and repeat_window.get("later_exact_repeat_expected")
        and repeat_window.get("later_exact_repeat_absent")
    )
    if decision == "rollback":
        state = "rollback-required"
        next_state = "reobserve-window-open"
        next_action = "apply-cache-replay-rollback-before-reobserve"
        status = "fresh-reobserve-window-after-rollback"
    elif promotion_readiness == "promotion-ready":
        state = "promotion-ready"
        next_state = "promotion-ready"
        next_action = "promote-cache-replay-rule"
        status = "reobserve-window-complete"
    elif decision == "retire-staged-no-repeat" or repeat_absent_after_floor:
        state = "retire-no-repeat"
        next_state = "retire-no-repeat"
        next_action = "retire-cache-replay-canary-no-repeat"
        status = "reobserve-window-complete"
    elif traffic_floor_met:
        state = "reobserve-window-open"
        next_state = "review-cache-replay-evidence"
        next_action = "review-cache-replay-canary-promotion-readiness"
        status = "reobserve-traffic-floor-met"
    else:
        state = "reobserve-window-open" if not metrics["stale_evidence"] else "rollback-required"
        next_state = "reobserve-window-open"
        next_action = "collect-cache-replay-canary-traffic"
        status = "reobserve-window-open"
    if decision == "rollback":
        successor_resolution = "rollback-required"
    elif decision == "retire-staged-no-repeat" or repeat_absent_after_floor:
        successor_resolution = "retire-staged-no-repeat"
    elif promotion_readiness == "promotion-ready" or (traffic_floor_met and observed_hits > 0):
        successor_resolution = "fresh-applied-holdout-evidence"
    elif promotion_readiness == "keep-staged-warmup" or (
        applied_count > 0
        and holdout_count > 0
        and observed_hits <= 0
        and bool(warmup_analysis.get("warmup_only_applied_misses"))
    ):
        successor_resolution = "keep-staged-warmup"
    else:
        successor_resolution = "reobserve-window-open"
    freshness_status = "stale" if metrics["stale_evidence"] else ("fresh" if observed_rows > 0 else "no-fresh-evidence")
    safe_blocker_breakdown = [
        {"value": public_label(item.get("value"), "unknown"), "count": _as_int(item.get("count"))}
        for item in blocker_breakdown
        if isinstance(item, dict) and public_label(item.get("value"), "unknown") != "unknown"
    ]
    recorded_evidence = {
        "schema": "tokenclaw.request_shape_cache_replay_reobserve_recorded_evidence.v1",
        "freshness_status": freshness_status,
        "age_hours": stale.get("age_hours") if stale.get("age_hours") is not None else metrics.get("evidence_age_hours"),
        "max_age_hours": max_age_hours,
        "canary_fraction": round(_as_float(top.get("canary_fraction")), 6),
        "holdout_fraction": round(_as_float(top.get("holdout_fraction")), 6),
        "projected_hits": metrics["projected_hits"],
        "projected_savings_usd": metrics["projected_savings_usd"],
        "observed_savings_usd": metrics["observed_savings_usd"],
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "observed_hits": observed_hits,
        "exact_hit_count": metrics["exact_hit_count"],
        "miss_count": metrics["miss_count"],
        "warmup_miss_count": metrics["warmup_miss_count"],
        "non_warmup_miss_count": metrics["non_warmup_miss_count"],
        "repeat_window_status": public_label(repeat_window.get("reason") or warmup_analysis.get("status"), "unknown"),
        "repeat_window_elapsed": bool(repeat_window.get("elapsed")),
        "later_exact_repeat_expected": bool(repeat_window.get("later_exact_repeat_expected")),
        "later_exact_repeat_absent": bool(repeat_window.get("later_exact_repeat_absent")),
        "retry_count": metrics["retry_count"],
        "error_count": metrics["error_count"],
        "fallback_count": metrics["fallback_count"],
        "invalidation_skipped_count": metrics["invalidation_skipped_count"],
        "unsupported_shape_count": metrics["unsupported_shape_count"],
        "blocker_breakdown": safe_blocker_breakdown,
        "retirement_required": successor_resolution == "retire-staged-no-repeat",
        "rollback_required": decision == "rollback",
        "metadata_only": True,
        "aggregate_only": True,
    }
    durable_decision = {
        "schema": "tokenclaw.request_shape_cache_replay_reobserve_durable_decision.v1",
        "decision": "reobserve-after-rollback" if decision == "rollback" else state,
        "status": status,
        "freshness_status": freshness_status,
        "successor_resolution": successor_resolution,
        "next_action": next_action,
        "observed_coverage": {
            "observed_row_count": observed_rows,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "exact_hit_count": metrics["exact_hit_count"],
            "miss_count": metrics["miss_count"],
            "warmup_miss_count": metrics["warmup_miss_count"],
            "non_warmup_miss_count": metrics["non_warmup_miss_count"],
            "invalidation_skipped_count": metrics["invalidation_skipped_count"],
            "fallback_count": metrics["fallback_count"],
            "error_count": metrics["error_count"],
            "retry_count": metrics["retry_count"],
            "observed_savings_usd": metrics["observed_savings_usd"],
            "metadata_only": True,
            "aggregate_only": True,
        },
        "blocker_breakdown": safe_blocker_breakdown,
        "emits_cache_apply_action": False,
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "metadata_only": True,
        "aggregate_only": True,
    }
    return {
        "schema": "tokenclaw.request_shape_cache_replay_post_rollback_reobserve_window.v1",
        "status": status,
        "decision": "reobserve-after-rollback" if decision == "rollback" else state,
        "state": state,
        "freshness_status": freshness_status,
        "successor_resolution": successor_resolution,
        "durable_decision": durable_decision,
        "recorded_evidence": recorded_evidence,
        "next_state": next_state,
        "next_action": next_action,
        "opens_after": "rollback-applied" if decision == "rollback" else "already-open",
        "rollback_required": decision == "rollback",
        "traffic_floor": traffic_floor,
        "traffic_floor_met": traffic_floor_met,
        "enough_for_promotion_review": enough_for_promotion_review,
        "repeat_absent_after_floor": repeat_absent_after_floor,
        "expiry": {
            "schema": "tokenclaw.request_shape_cache_replay_reobserve_window_expiry.v1",
            "reference": "rollback_applied_at" if decision == "rollback" else "latest_observed_at",
            "max_age_hours": max_age_hours,
            "ttl_seconds": ttl_seconds,
            "expires_at_included": False,
            "expired": False if decision == "rollback" else bool(metrics["stale_evidence"]),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "target_shape": {
            key: shape.get(key)
            for key in (
                "provider_family",
                "source_surface",
                "endpoint",
                "category",
                "workflow_phase",
                "text_bucket",
                "token_bucket",
                "has_tools",
                "stream",
            )
            if key in shape
        },
        "lifecycle_states": [
            "rollback-required",
            "reobserve-window-open",
            "retire-no-repeat",
            "promotion-ready",
        ],
        "observed": {
            "observed_row_count": observed_rows,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "observed_hits": observed_hits,
            "observed_savings_usd": metrics["observed_savings_usd"],
            "metadata_only": True,
            "aggregate_only": True,
        },
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_replay_policy_duplicate_suppression(
    decision: str,
    promotion_readiness: str,
    hit_recovery_metrics: dict[str, Any],
) -> dict[str, Any]:
    reason = promotion_readiness or decision
    if promotion_readiness == "keep-staged-warmup" and hit_recovery_metrics.get("hit_recovery_demonstrated"):
        reason = "synthetic-hit-recovery-proven-live-traffic-warmup-only"
    if decision == "retire-staged-no-repeat":
        reason = "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired"
    return {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision_duplicate_suppression.v1",
        "reason": reason,
        "metadata_only": True,
        "aggregate_only": True,
        "target_local_policy_section": "cache.pattern_rules",
        "target_local_rule_file": "cache_rules.yaml",
        "suppresses_generic_replay_ready_issue": True,
        "suppresses_new_cache_replay_stage_issue": decision in {"widen", "keep-staged", "retire-staged-no-repeat"},
        "suppresses_generic_cache_replay_activation_issue": decision in {
            "widen",
            "keep-staged",
            "keep-blocked",
            "rollback",
            "retire-staged-no-repeat",
        },
    }


def _cache_replay_policy_conditions_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    conditions = {
        "pattern_hashes": ["sha256:*"],
        "provider_family": shape.get("provider_family") or "openai",
        "source_surface": shape.get("source_surface"),
        "endpoint": shape.get("endpoint"),
        "category": shape.get("category"),
        "workflow_phase": shape.get("workflow_phase"),
        "text_bucket": shape.get("text_bucket"),
        "token_bucket": shape.get("token_bucket"),
        "has_tools": False,
        "stream": False,
        "replayability_levels": ["features_only", "local-exact-response"],
    }
    return {key: value for key, value in conditions.items() if value not in (None, "", [])}


def _cache_replay_policy_patch(evidence: dict[str, Any] | None, decision: str) -> dict[str, Any] | None:
    if not isinstance(evidence, dict):
        return None
    top = _cache_replay_policy_decision_top_canary(evidence)
    shape = top.get("shape") if isinstance(top.get("shape"), dict) else {}
    metrics = _cache_replay_policy_decision_metrics(evidence)
    rule_id = _cache_replay_policy_rule_id(evidence)
    reason = _cache_replay_policy_decision_reason(evidence, decision)
    if decision in {"rollback", "retire-staged-no-repeat"}:
        return {
            "schema": "tokenclaw.request_shape_cache_replay_policy_decision_local_patch.v1",
            "patch_type": (
                "retire_openai_exact_cache_replay_canary"
                if decision == "retire-staged-no-repeat"
                else "rollback_openai_exact_cache_replay_policy"
            ),
            "target_local_rule_file": "cache_rules.yaml",
            "source_canary_policy_file": _cache_replay_policy_source_file(evidence),
            "pattern_rules": _cache_replay_policy_disable_patch_rules(evidence, reason),
            "metadata_only": True,
            "aggregate_only": True,
            "rules_path_included": False,
        }
    if decision != "widen":
        return None
    ttl_seconds = _as_int(top.get("ttl_seconds"), DEFAULT_CACHE_REPLAY_CANARY_TTL_SECONDS)
    return {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision_local_patch.v1",
        "patch_type": "widen_openai_exact_cache_replay_canary",
        "target_local_rule_file": "cache_rules.yaml",
        "source_canary_policy_file": "cache_canary_policy.yaml",
        "policy_source": "local-manual",
        "pattern_rules": [
            {
                "id": rule_id,
                "enabled": True,
                "policy_source": "local-manual",
                "description": "Promoted OpenAI Responses exact-cache replay rule from local request-shape canary evidence.",
                "conditions": _cache_replay_policy_conditions_from_shape(shape),
                "action": {
                    "type": "exact_cache_pattern",
                    "allow_tool_calls": False,
                    "safe_invalidation": False,
                    "streaming": False,
                    "scope": "session",
                    "min_call_count": 2,
                    "ttl_seconds": ttl_seconds,
                },
                "rollout": {
                    "schema": "tokenclaw.pattern_policy_rollout.v1",
                    "recommendation_mode": "active",
                    "canary_enabled": False,
                    "canary_fraction": 1.0,
                    "holdout_fraction": 0.0,
                    "canary_salt": rule_id,
                    "canary_unit": "request_fingerprint",
                },
                "graduation": {
                    "schema": "tokenclaw.request_shape_cache_replay_policy_graduation.v1",
                    "source_schema": REPLAY_CACHE_CANARY_EVIDENCE_SCHEMA,
                    "source_verdict": "widen",
                    "source_surface": shape.get("source_surface"),
                    "endpoint": shape.get("endpoint"),
                    "category": shape.get("category"),
                    "workflow_phase": shape.get("workflow_phase"),
                    "text_bucket": shape.get("text_bucket"),
                    "token_bucket": shape.get("token_bucket"),
                    "sample_count": metrics["observed_row_count"],
                    "applied_count": metrics["applied_count"],
                    "holdout_count": metrics["holdout_count"],
                    "projected_hits": metrics["projected_hits"],
                    "observed_hits": metrics["observed_hits"],
                    "projected_savings_usd": metrics["projected_savings_usd"],
                    "observed_savings_usd": metrics["observed_savings_usd"],
                    "aggregate_only": True,
                    "graduated_at": utc_now(),
                },
            }
        ],
        "metadata_only": True,
        "aggregate_only": True,
        "rules_path_included": False,
        "policy_files_written": False,
        "cache_entries_written": False,
    }


def build_request_shape_cache_replay_policy_decision_report(
    evidence_report: dict[str, Any],
    *,
    hit_recovery_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = evidence_report if isinstance(evidence_report, dict) else {}
    metrics = _cache_replay_policy_decision_metrics(evidence)
    warmup_analysis = _cache_replay_policy_warmup_analysis(evidence, metrics)
    hit_recovery_metrics = _cache_replay_policy_hit_recovery_metrics(hit_recovery_report, evidence)
    decision = _cache_replay_policy_decision_value(evidence, hit_recovery_metrics)
    reason = _cache_replay_policy_decision_reason(evidence, decision)
    reason_codes = _cache_replay_policy_reason_codes(evidence)
    recommended_next_action = _cache_replay_policy_recommended_next_action(decision)
    promotion_decision = _cache_replay_canary_promotion_decision(evidence, decision)
    promotion_readiness = _cache_replay_policy_promotion_readiness(evidence, decision)
    promotion_blocker = _cache_replay_policy_promotion_blocker(
        evidence,
        decision,
        promotion_readiness,
        reason,
    )
    observed_hit_blocker = _cache_replay_policy_observed_hit_blocker(evidence, promotion_blocker)
    duplicate_suppression = _cache_replay_policy_duplicate_suppression(
        decision,
        promotion_readiness,
        hit_recovery_metrics,
    )
    rollback_metadata = _cache_replay_policy_rollback_metadata(evidence, decision)
    local_policy_patch = _cache_replay_policy_patch(evidence, decision)
    promotion_allowed = decision == "widen"
    rollback_required = decision == "rollback"
    retirement_required = decision == "retire-staged-no-repeat"
    post_rollback_observation = _cache_replay_policy_post_rollback_reobserve_window(
        evidence,
        decision,
        promotion_readiness,
        metrics,
        warmup_analysis,
    )
    if promotion_readiness == "promotion-ready":
        reason_codes = list(dict.fromkeys(["promotion-ready", *reason_codes]))
    if retirement_required:
        reason_codes = list(dict.fromkeys(["retire-staged-no-repeat", reason, *reason_codes]))
    entry = {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision_entry.v1",
        "decision_id": _cache_replay_policy_decision_id(evidence, decision),
        "decision": decision,
        "promotion_decision": promotion_decision,
        "promotion_readiness": promotion_readiness,
        "impact_recommendation": promotion_readiness,
        "promotion_recommendation": promotion_readiness,
        "reason": reason,
        "promotion_blocker": promotion_blocker,
        "observed_hit_blocker": observed_hit_blocker,
        "reason_codes": reason_codes,
        "recommended_next_action": recommended_next_action,
        "next_action": recommended_next_action,
        "target_local_policy": "cache_rules",
        "target_local_rule_file": "cache_rules.yaml",
        "target_local_policy_section": "cache.pattern_rules",
        "source_canary_policy_file": _cache_replay_policy_source_file(evidence),
        "policy_source": "local-manual" if metrics["staged_canary_count"] else "unknown",
        "decision_options": ["widen", "rollback", "retire-staged-no-repeat", "keep-staged", "keep-blocked"],
        "promotion_decision_options": ["promote", "keep-staged-warmup", "retire-staged-no-repeat", "keep-blocked"],
        "promotion_readiness_options": [
            "promotion-ready",
            "retire-staged-no-repeat",
            "keep-staged-warmup",
            "keep-staged",
            "keep-blocked",
            "rollback-required",
        ],
        "promotion_allowed": promotion_allowed,
        "promotion_ready": promotion_readiness == "promotion-ready",
        "rollback_required": rollback_required,
        "retirement_required": retirement_required,
        "keep_staged": decision == "keep-staged",
        "keep_blocked": decision == "keep-blocked",
        "coverage": {
            "schema": "tokenclaw.request_shape_cache_replay_policy_decision_coverage.v1",
            "has_applied_coverage": metrics["applied_count"] > 0,
            "has_holdout_coverage": metrics["holdout_count"] > 0,
            "has_observed_hits": metrics["observed_hits"] > 0,
            "has_observed_savings": metrics["observed_savings_usd"] > 0,
            "has_fresh_evidence": not metrics["stale_evidence"],
            "has_no_invalidation_skips": metrics["invalidation_skipped_count"] == 0,
            "has_supported_shapes": metrics["unsupported_shape_count"] == 0,
            "has_no_fallbacks": metrics["fallback_count"] == 0,
            "has_no_errors": metrics["error_count"] == 0,
            "applied_count": metrics["applied_count"],
            "holdout_count": metrics["holdout_count"],
            "miss_count": metrics["miss_count"],
            "observed_hits": metrics["observed_hits"],
            "exact_hit_count": metrics["exact_hit_count"],
            "metadata_only": True,
            "aggregate_only": True,
        },
        "metrics": metrics,
        "warmup_analysis": warmup_analysis,
        "hit_recovery_metrics": hit_recovery_metrics,
        "applied_miss_blocker_breakdown": (
            evidence.get("applied_miss_blocker_breakdown")
            if isinstance(evidence.get("applied_miss_blocker_breakdown"), list)
            else []
        ),
        "rollback_metadata": rollback_metadata,
        "post_rollback_observation": post_rollback_observation,
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "duplicate_suppression": duplicate_suppression,
        "local_policy_patch": local_policy_patch,
        "privacy": _cache_replay_policy_decision_privacy(),
    }
    return {
        "schema": REPLAY_CACHE_POLICY_DECISION_SCHEMA,
        "status": "decided",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "decision": decision,
        "promotion_decision": promotion_decision,
        "promotion_readiness": promotion_readiness,
        "impact_recommendation": promotion_readiness,
        "promotion_recommendation": promotion_readiness,
        "reason": reason,
        "promotion_blocker": promotion_blocker,
        "observed_hit_blocker": observed_hit_blocker,
        "reason_codes": reason_codes,
        "next_action": recommended_next_action,
        "top_decision": entry,
        "decisions": [entry],
        "warmup_analysis": warmup_analysis,
        "hit_recovery_metrics": hit_recovery_metrics,
        "post_rollback_observation": post_rollback_observation,
        "duplicate_suppression": duplicate_suppression,
        "summary": {
            "decision": decision,
            "promotion_decision": promotion_decision,
            "promotion_readiness": promotion_readiness,
            "impact_recommendation": promotion_readiness,
            "promotion_recommendation": promotion_readiness,
            "next_action": recommended_next_action,
            "promotion_allowed": promotion_allowed,
            "promotion_ready": promotion_readiness == "promotion-ready",
            "rollback_required": rollback_required,
            "retirement_required": retirement_required,
            "keep_staged_warmup": promotion_decision == "keep-staged-warmup",
            "retire_staged_no_repeat": retirement_required,
            "keep_staged": decision == "keep-staged",
            "keep_blocked": decision == "keep-blocked",
            "staged_canary_count": metrics["staged_canary_count"],
            "observed_row_count": metrics["observed_row_count"],
            "applied_count": metrics["applied_count"],
            "holdout_count": metrics["holdout_count"],
            "exact_hit_count": metrics["exact_hit_count"],
            "miss_count": metrics["miss_count"],
            "bypass_count": metrics["bypass_count"],
            "invalidation_skipped_count": metrics["invalidation_skipped_count"],
            "unsupported_shape_count": metrics["unsupported_shape_count"],
            "retry_count": metrics["retry_count"],
            "fallback_count": metrics["fallback_count"],
            "error_count": metrics["error_count"],
            "projected_hits": metrics["projected_hits"],
            "observed_hits": metrics["observed_hits"],
            "projected_savings_usd": metrics["projected_savings_usd"],
            "observed_savings_usd": metrics["observed_savings_usd"],
            "hit_observation_rate": metrics["hit_observation_rate"],
            "hit_recovery_demonstrated": bool(hit_recovery_metrics.get("hit_recovery_demonstrated")),
            "synthetic_hit_recovery_exact_hit_count": hit_recovery_metrics["synthetic_exact_hit_count"],
            "synthetic_hit_recovery_status": hit_recovery_metrics["status"],
            "target_matches_hit_recovery_shape": bool(hit_recovery_metrics.get("target_matches_staged_canary_shape")),
            "savings_realization_ratio": metrics["savings_realization_ratio"],
            "top_applied_miss_blocker": metrics["top_applied_miss_blocker"],
            "top_blocking_applied_miss_blocker": metrics["top_blocking_applied_miss_blocker"],
            "warmup_status": warmup_analysis["status"],
            "warmup_classification": warmup_analysis["classification"],
            "warmup_miss_count": warmup_analysis["warmup_miss_count"],
            "non_warmup_miss_count": warmup_analysis["non_warmup_miss_count"],
            "first_warmup_age_hours": warmup_analysis["first_warmup_age_hours"],
            "latest_warmup_age_hours": warmup_analysis["latest_warmup_age_hours"],
            "repeat_window_elapsed": warmup_analysis["repeat_window"]["elapsed"],
            "later_exact_repeat_expected": warmup_analysis["repeat_window"]["later_exact_repeat_expected"],
            "later_exact_repeat_absent": warmup_analysis["repeat_window"]["later_exact_repeat_absent"],
            "promotion_blocker": promotion_blocker,
            "observed_hit_blocker": observed_hit_blocker,
            "stale_evidence": metrics["stale_evidence"],
            "post_rollback_observation_status": post_rollback_observation["status"],
            "post_rollback_observation_state": post_rollback_observation["state"],
            "post_rollback_observation_freshness_status": post_rollback_observation["freshness_status"],
            "post_rollback_observation_successor_resolution": post_rollback_observation["successor_resolution"],
            "post_rollback_observation_next_state": post_rollback_observation["next_state"],
            "post_rollback_observation_next_action": post_rollback_observation["next_action"],
            "reobserve_traffic_floor_met": post_rollback_observation["traffic_floor_met"],
            "reobserve_window_max_age_hours": post_rollback_observation["expiry"]["max_age_hours"],
            "policy_source": entry["policy_source"],
            "target_local_rule_file": "cache_rules.yaml",
            "target_local_policy_section": "cache.pattern_rules",
            "source_canary_policy_file": _cache_replay_policy_source_file(evidence),
            "policy_files_written": False,
            "cache_apply_action_count": 0,
            "cache_entries_written": False,
        },
        "source_evidence": {
            "schema": evidence.get("schema"),
            "status": evidence.get("status"),
            "summary": evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {},
            "applied_miss_blocker_breakdown": (
                evidence.get("applied_miss_blocker_breakdown")
                if isinstance(evidence.get("applied_miss_blocker_breakdown"), list)
                else []
            ),
            "hit_recovery_metrics": hit_recovery_metrics,
            "warmup_analysis": warmup_analysis,
            "reobserve_window": evidence.get("reobserve_window") if isinstance(evidence.get("reobserve_window"), dict) else {},
            "stale_evidence": evidence.get("stale_evidence") if isinstance(evidence.get("stale_evidence"), dict) else {},
            "privacy": _cache_replay_evidence_privacy(),
        },
        "acceptance": {
            "single_durable_decision": True,
            "records_durable_decision": decision in {
                "widen",
                "rollback",
                "retire-staged-no-repeat",
                "keep-staged",
                "keep-blocked",
            },
            "emits_explicit_canary_promotion_decision": promotion_decision
            in {"promote", "keep-staged-warmup", "retire-staged-no-repeat", "keep-blocked"},
            "emits_explicit_promotion_readiness": promotion_readiness
            in {
                "promotion-ready",
                "retire-staged-no-repeat",
                "keep-staged-warmup",
                "keep-staged",
                "keep-blocked",
                "rollback-required",
            },
            "reports_hit_recovery": metrics["observed_hits"] >= 0 and metrics["projected_hits"] >= 0,
            "reports_synthetic_hit_recovery_smoke": hit_recovery_metrics["status"] != "unavailable",
            "reports_holdout_coverage": metrics["holdout_count"] >= 0,
            "reports_applied_miss_blocker_breakdown": isinstance(
                evidence.get("applied_miss_blocker_breakdown"), list
            ),
            "reports_observed_hit_blocker": bool(observed_hit_blocker) or metrics["observed_hits"] > 0,
            "reports_warmup_analysis": isinstance(warmup_analysis, dict),
            "reports_repeat_window_metadata": isinstance(warmup_analysis.get("repeat_window"), dict),
            "emits_post_rollback_reobserve_window": isinstance(post_rollback_observation, dict),
            "reports_reobserve_traffic_floor": isinstance(post_rollback_observation.get("traffic_floor"), dict),
            "reports_reobserve_expiry_metadata": isinstance(post_rollback_observation.get("expiry"), dict),
            "reports_reobserve_recorded_evidence": isinstance(post_rollback_observation.get("recorded_evidence"), dict),
            "reports_reobserve_freshness_status": post_rollback_observation.get("freshness_status") in {"fresh", "stale", "no-fresh-evidence"},
            "reports_durable_reobserve_decision": isinstance(post_rollback_observation.get("durable_decision"), dict),
            "resolves_stale_successor_beyond_evidence_age": (
                (not metrics["stale_evidence"])
                or post_rollback_observation["successor_resolution"]
                in {"rollback-required", "retire-staged-no-repeat", "fresh-applied-holdout-evidence", "keep-staged-warmup"}
            ),
            "reports_reobserve_next_state": bool(post_rollback_observation.get("next_state")),
            "reobserve_window_writes_no_cache_entries": post_rollback_observation["cache_entries_written"] == 0,
            "reobserve_window_emits_no_cache_apply_actions": post_rollback_observation["cache_apply_action_count"] == 0,
            "distinguishes_first_seen_warmup_from_ineffective_replay": True,
            "drafts_local_policy_patch_or_blocker": bool(local_policy_patch) or bool(reason_codes),
            "targets_file_backed_cache_policy": entry["target_local_rule_file"] == "cache_rules.yaml",
            "suppresses_generic_replay_ready_issue_recreation": duplicate_suppression[
                "suppresses_generic_replay_ready_issue"
            ],
            "keeps_tool_and_streaming_replay_blocked": bool(
                local_policy_patch is None
                or all(
                    isinstance(rule, dict)
                    and not bool((rule.get("conditions") or {}).get("has_tools"))
                    and not bool((rule.get("conditions") or {}).get("stream"))
                    and not bool((rule.get("action") or {}).get("allow_tool_calls"))
                    and not bool((rule.get("action") or {}).get("streaming"))
                    for rule in local_policy_patch.get("pattern_rules", [])
                )
            ),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _cache_replay_policy_decision_privacy(),
    }


def _shape_replayability_decision(row: dict[str, Any]) -> dict[str, Any]:
    row_count = _as_int(row.get("row_count") or row.get("count"))
    endpoint = _endpoint_label_from_row(row)
    cache_status = str(row.get("cache_status") or "unknown")
    stream = bool(row.get("stream"))
    has_tools = bool(row.get("has_tools"))
    dependency_status = public_label(row.get("file_dependency_status"), "missing")
    blockers: set[str] = set()

    if endpoint not in REPLAY_SUPPORTED_ENDPOINTS:
        blockers.add("unsupported-endpoint")
    if cache_status == "hit":
        blockers.add("already-cache-hit")
    if row_count < 2:
        blockers.add("insufficient-repeat-evidence")
    if stream and not (not has_tools and _is_anthropic_messages_row(row)):
        blockers.add("streaming-replay-not-supported")
    if has_tools:
        blockers.add("tools-present")
        if dependency_status == "stable":
            blockers.add("safe-invalidation-evidence-present")
        elif dependency_status == "invalidated":
            blockers.add("stale-dependency-evidence")
            blockers.add("unsafe-tool-calls-without-invalidation")
        elif dependency_status == "unsafe":
            blockers.add("unsafe-dependency-evidence")
            blockers.add("unsafe-tool-calls-without-invalidation")
        elif dependency_status == "missing":
            blockers.add("invalidation-evidence-missing")
            blockers.add("unsafe-tool-calls-without-invalidation")
        else:
            blockers.add("dependency-evidence-unknown")
            blockers.add("unsafe-tool-calls-without-invalidation")

    if not blockers:
        return {
            "readiness": "replay-ready",
            "reason": "replay-ready-exact-non-tool-shape",
            "blockers": [],
            "projected_hits": max(0, row_count - 1),
        }

    reason_priority = (
        "unsupported-endpoint",
        "already-cache-hit",
        "streaming-replay-not-supported",
        "stale-dependency-evidence",
        "unsafe-dependency-evidence",
        "invalidation-evidence-missing",
        "unsafe-tool-calls-without-invalidation",
        "dependency-evidence-unknown",
        "safe-invalidation-evidence-present",
        "tools-present",
        "insufficient-repeat-evidence",
    )
    reason = next((item for item in reason_priority if item in blockers), sorted(blockers)[0])
    return {
        "readiness": "skipped",
        "reason": reason,
        "blockers": sorted(blockers),
        "projected_hits": 0,
    }


def _cache_invalidation_next_actions(cohort: dict[str, Any]) -> tuple[str, list[str], str, bool]:
    blockers = {
        public_label(item, "unknown")
        for item in cohort.get("blockers") or []
        if public_label(item, "unknown") != "unknown"
    }
    reason = public_label(cohort.get("reason"), "unknown")
    if reason != "unknown":
        blockers.add(reason)
    has_tools = bool(cohort.get("has_tools"))
    stream = bool(cohort.get("stream"))
    readiness = public_label(cohort.get("readiness"), "unknown")
    secondary: list[str] = []

    if readiness == "replay-ready":
        return "exact-non-tool-only", secondary, "ready-exact-non-tool", False
    if "safe-invalidation-evidence-present" in blockers:
        if has_tools:
            secondary.append("keep-tool-cache-blocked")
        return "rank-safe-tool-cache-replay-readiness", secondary, "safe-invalidation-evidence-present", False
    if "stale-dependency-evidence" in blockers:
        if has_tools:
            secondary.append("keep-tool-cache-blocked")
        return "refresh-file-invalidation-evidence", secondary, "stale-dependency-evidence", True
    if "unsafe-dependency-evidence" in blockers:
        if has_tools:
            secondary.append("keep-tool-cache-blocked")
        return "collect-file-invalidation-evidence", secondary, "unsafe-dependency-evidence", True
    if "invalidation-evidence-missing" in blockers or "unsafe-tool-calls-without-invalidation" in blockers:
        if has_tools:
            secondary.append("keep-tool-cache-blocked")
        if stream:
            secondary.append("stage-streaming-replay-buffer-fixture")
        return "collect-file-invalidation-evidence", secondary, "missing-safe-invalidation-evidence", True
    if has_tools or "tools-present" in blockers or "tool-call-cache-disabled" in blockers:
        return "keep-tool-cache-blocked", secondary, "tool-cache-disabled-without-invalidation-proof", True
    if stream or "streaming-replay-not-supported" in blockers or "unsupported-streaming-shape" in blockers:
        return "stage-streaming-replay-buffer-fixture", secondary, "streaming-replay-buffer-fixture-needed", False
    if "insufficient-repeat-evidence" in blockers:
        return "collect-repeat-evidence", secondary, "insufficient-repeat-evidence", False
    return "keep-cache-replay-noop", secondary, public_label(reason, "unsupported-cache-replay-shape"), False


def _cache_dependency_evidence_decision(
    *,
    file_dependency_status: str,
    next_action: str,
    requires_invalidation: bool,
    has_tools: bool,
) -> dict[str, Any]:
    status = public_label(file_dependency_status, "missing")
    if not has_tools and not requires_invalidation:
        decision = "not-required"
        reason = "exact-non-tool-cache-replay"
    elif status == "stable":
        decision = "stable-dependency-evidence"
        reason = "safe-invalidation-evidence-present"
    elif status == "invalidated":
        decision = "stale-risk-blocker"
        reason = "stale-dependency-evidence"
    elif status == "unsafe":
        decision = "unsafe-dependency-evidence"
        reason = "unsafe-tool-calls-without-invalidation"
    elif status == "missing":
        decision = "missing-dependency-evidence"
        reason = "invalidation-evidence-missing"
    else:
        decision = "unknown-dependency-evidence"
        reason = "dependency-evidence-unknown"
    evidence_class = {
        "stable-dependency-evidence": "stable-dependency-evidence",
        "stale-risk-blocker": "stale-dependency-evidence",
        "unsafe-dependency-evidence": "unsafe-dependency-evidence",
        "missing-dependency-evidence": "missing-dependency-evidence",
        "not-required": "not-required",
    }.get(decision, "unknown-dependency-evidence")
    return {
        "schema": "tokenclaw.request_shape_cache_dependency_evidence_decision.v1",
        "status": status,
        "decision": decision,
        "evidence_class": evidence_class,
        "reason": reason,
        "next_action": next_action,
        "safe_invalidation_evidence": decision == "stable-dependency-evidence",
        "stale_risk_blocker": decision == "stale-risk-blocker",
        "unsafe_dependency_evidence": decision == "unsafe-dependency-evidence",
        "missing_dependency_evidence": decision == "missing-dependency-evidence",
        "tool_cache_replay_enabled": False,
        "streaming_replay_enabled": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _cache_dependency_evidence_state(decision: str) -> str:
    if decision == "stable-dependency-evidence":
        return "dependency-gated-review-ready"
    if decision == "stale-risk-blocker":
        return "blocked-stale-dependency-evidence"
    if decision == "unsafe-dependency-evidence":
        return "blocked-unsafe-dependency-evidence"
    if decision == "missing-dependency-evidence":
        return "blocked-missing-dependency-evidence"
    if decision == "not-required":
        return "exact-non-tool-only"
    return "blocked-unknown-dependency-evidence"


def _dependency_evidence_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported_classes = set(DEPENDENCY_EVIDENCE_CLASSES)
    known_classes = supported_classes | {"not-required"}
    observed_classes = {
        public_label(row.get("dependency_evidence_decision", {}).get("evidence_class"), "unknown-dependency-evidence")
        for row in rows
        if isinstance(row.get("dependency_evidence_decision"), dict)
    }
    all_rows_classified = bool(rows) and all(
        public_label(row.get("dependency_evidence_decision", {}).get("evidence_class"), "unknown-dependency-evidence")
        in known_classes
        for row in rows
        if isinstance(row.get("dependency_evidence_decision"), dict)
    )
    return {
        "schema": "tokenclaw.request_shape_cache_dependency_evidence_classification.v1",
        "supported_evidence_classes": list(DEPENDENCY_EVIDENCE_CLASSES),
        "decision_options": list(DEPENDENCY_EVIDENCE_DECISION_OPTIONS),
        "observed_evidence_classes": sorted(observed_classes),
        "missing_observed_evidence_classes": sorted(supported_classes - observed_classes),
        "observed_evidence_class_count": len(observed_classes),
        "all_rows_classified_into_supported_evidence_classes": all_rows_classified,
        "maps_stale_risk_to_stale_dependency_evidence": True,
        "supports_four_way_dependency_evidence_split": True,
        "classification_buckets": list(DEPENDENCY_EVIDENCE_CLASSES),
        "supports_five_way_dependency_evidence_split": True,
        "tool_cache_replay_enabled": False,
        "streaming_replay_enabled": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _dependency_evidence_burndown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    burndown: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision_meta = row.get("dependency_evidence_decision")
        if not isinstance(decision_meta, dict):
            continue
        decision = public_label(decision_meta.get("decision"), "unknown-dependency-evidence")
        evidence_class = public_label(decision_meta.get("evidence_class"), decision)
        status = public_label(row.get("dependency_evidence_status") or decision_meta.get("status"), "unknown")
        next_action = public_label(row.get("next_action") or decision_meta.get("next_action"), "keep-cache-replay-blocked")
        evidence_state = public_label(row.get("evidence_state"), _cache_dependency_evidence_state(decision))
        reason = public_label(
            row.get("evidence_reason") or row.get("reason") or decision_meta.get("reason"),
            "unknown",
        )
        count = _as_int(row.get("row_count") or row.get("sample_count"))
        if count <= 0:
            continue
        key = (decision, status, evidence_state, next_action)
        entry = burndown.setdefault(
            key,
            {
                "schema": "tokenclaw.request_shape_cache_dependency_evidence_burndown_row.v1",
                "dependency_evidence_decision": decision,
                "dependency_evidence_class": evidence_class,
                "dependency_evidence_status": status,
                "evidence_state": evidence_state,
                "next_action": next_action,
                "row_count": 0,
                "tool_cache_replay_enabled": False,
                "streaming_replay_enabled": False,
                "cache_entries_written": 0,
                "policy_files_written": False,
                "_reason_counts": {},
                "metadata_only": True,
                "aggregate_only": True,
            },
        )
        entry["row_count"] += count
        entry["tool_cache_replay_enabled"] = bool(
            entry["tool_cache_replay_enabled"] or row.get("tool_cache_replay_enabled")
        )
        entry["streaming_replay_enabled"] = bool(
            entry["streaming_replay_enabled"] or row.get("streaming_replay_enabled")
        )
        entry["cache_entries_written"] += _as_int(row.get("cache_entries_written"))
        entry["policy_files_written"] = bool(entry["policy_files_written"] or row.get("policy_files_written"))
        if reason != "unknown":
            _increment(entry["_reason_counts"], reason, count)

    priority = {
        "blocked-missing-dependency-evidence": 5,
        "blocked-stale-dependency-evidence": 5,
        "blocked-unsafe-dependency-evidence": 5,
        "dependency-gated-review-ready": 4,
        "blocked-unknown-dependency-evidence": 3,
        "exact-non-tool-only": 1,
    }
    rows_out = []
    for entry in burndown.values():
        reason_breakdown = _breakdown(entry.pop("_reason_counts", {}))
        entry["reason_breakdown"] = reason_breakdown
        entry["top_reason"] = reason_breakdown[0]["value"] if reason_breakdown else None
        entry["top_blocker_reason"] = entry["top_reason"]
        rows_out.append(entry)
    return sorted(
        rows_out,
        key=lambda item: (
            priority.get(str(item.get("evidence_state")), 0),
            _as_int(item.get("row_count")),
            str(item.get("dependency_evidence_decision")),
            str(item.get("next_action")),
        ),
        reverse=True,
    )


def _dependency_fingerprint_missing_reason(row: dict[str, Any]) -> str:
    audit = row.get("file_dependency_audit") if isinstance(row.get("file_dependency_audit"), dict) else {}
    decision_meta = row.get("dependency_evidence_decision")
    decision = (
        public_label(decision_meta.get("decision"), "unknown-dependency-evidence")
        if isinstance(decision_meta, dict)
        else "unknown-dependency-evidence"
    )
    status = public_label(row.get("file_dependency_status") or row.get("dependency_evidence_status"), "missing")
    invalidation_reason = public_label(audit.get("invalidation_reason"), "none")
    capture_reason = public_label(audit.get("dependency_capture_reason"), "unknown")
    snapshot_bucket = public_label(audit.get("snapshot_count_bucket"), "unknown")
    candidate_bucket = public_label(audit.get("candidate_path_count_bucket"), "unknown")
    raw_candidate_bucket = public_label(audit.get("raw_candidate_path_count_bucket"), "unknown")

    if decision == "stable-dependency-evidence" or status == "stable":
        return "safe-invalidation-evidence-present"
    if decision == "stale-risk-blocker" or status == "invalidated":
        return "stale-dependency-evidence"
    if decision == "unsafe-dependency-evidence" or status == "unsafe":
        return "unsafe-tool-calls-without-invalidation"
    if decision == "unknown-dependency-evidence" or status == "unknown":
        return "dependency-evidence-unknown"
    if capture_reason not in {"unknown", "complete"}:
        return capture_reason
    if invalidation_reason not in {"none", "unknown"}:
        return invalidation_reason
    if snapshot_bucket in {"0", "unknown"} and raw_candidate_bucket not in {"0", "unknown"}:
        return "candidate-paths-not-snapshotted"
    if snapshot_bucket in {"0", "unknown"} and candidate_bucket in {"0", "unknown"}:
        return "no-stable-file-dependency-snapshots"
    return "invalidation-evidence-missing"


def _dependency_fingerprint_coverage_next_action(
    *,
    stable_rows: int,
    stale_rows: int,
    unsafe_rows: int,
    missing_rows: int,
    unknown_rows: int,
) -> str:
    if stable_rows > 0:
        return "rank-safe-tool-cache-replay-readiness"
    if stale_rows > 0:
        return "refresh-file-invalidation-evidence"
    if unsafe_rows > 0 or missing_rows > 0 or unknown_rows > 0:
        return "collect-file-invalidation-evidence"
    return "keep-tool-cache-replay-blocked"


def _dependency_fingerprint_coverage_decision(
    *,
    stable_rows: int,
    stale_rows: int,
    unsafe_rows: int,
    missing_rows: int,
    unknown_rows: int,
) -> str:
    if stable_rows > 0:
        return "stable-coverage-observed"
    if stale_rows > 0 or unsafe_rows > 0:
        return "nonstable-coverage-blocked"
    if missing_rows > 0:
        return "missing-stable-coverage"
    if unknown_rows > 0:
        return "unknown-coverage"
    return "no-tool-cache-dependency-coverage"


def _build_dependency_fingerprint_coverage_report(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    coverage_rows: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    evidence_state_counts: dict[str, int] = {}
    missing_reason_counts: dict[str, int] = {}
    snapshot_bucket_counts: dict[str, int] = {}
    candidate_bucket_counts: dict[str, int] = {}
    raw_candidate_bucket_counts: dict[str, int] = {}
    fingerprint_available_rows = 0
    fingerprint_missing_rows = 0
    safe_invalidation_rows = 0
    stable_rows = 0
    stale_rows = 0
    missing_rows = 0
    unsafe_rows = 0
    unknown_rows = 0
    affected_rows = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_count = _as_int(row.get("row_count") or row.get("sample_count"))
        if row_count <= 0:
            continue
        decision_meta = row.get("dependency_evidence_decision")
        decision = (
            public_label(decision_meta.get("decision"), "unknown-dependency-evidence")
            if isinstance(decision_meta, dict)
            else "unknown-dependency-evidence"
        )
        evidence_class = (
            public_label(decision_meta.get("evidence_class"), decision)
            if isinstance(decision_meta, dict)
            else "unknown-dependency-evidence"
        )
        status = public_label(row.get("dependency_evidence_status") or row.get("file_dependency_status"), "missing")
        evidence_state = public_label(row.get("evidence_state"), _cache_dependency_evidence_state(decision))
        audit = row.get("file_dependency_audit") if isinstance(row.get("file_dependency_audit"), dict) else {}
        snapshot_bucket = public_label(audit.get("snapshot_count_bucket"), "unknown")
        candidate_bucket = public_label(audit.get("candidate_path_count_bucket"), "unknown")
        raw_candidate_bucket = public_label(audit.get("raw_candidate_path_count_bucket"), "unknown")
        missing_reason = _dependency_fingerprint_missing_reason(row)
        fingerprint_available = bool(row.get("file_dependency_fingerprint_available"))
        safe_invalidation = bool(row.get("safe_invalidation_evidence"))

        affected_rows += row_count
        _increment(decision_counts, decision, row_count)
        _increment(status_counts, status, row_count)
        _increment(evidence_state_counts, evidence_state, row_count)
        _increment(missing_reason_counts, missing_reason, row_count)
        _increment(snapshot_bucket_counts, snapshot_bucket, row_count)
        _increment(candidate_bucket_counts, candidate_bucket, row_count)
        _increment(raw_candidate_bucket_counts, raw_candidate_bucket, row_count)
        if fingerprint_available:
            fingerprint_available_rows += row_count
        else:
            fingerprint_missing_rows += row_count
        if safe_invalidation:
            safe_invalidation_rows += row_count
        if decision == "stable-dependency-evidence":
            stable_rows += row_count
        elif decision == "stale-risk-blocker":
            stale_rows += row_count
        elif decision == "missing-dependency-evidence":
            missing_rows += row_count
        elif decision == "unsafe-dependency-evidence":
            unsafe_rows += row_count
        else:
            unknown_rows += row_count

        coverage_rows.append(
            {
                "schema": REPLAY_DEPENDENCY_FINGERPRINT_COVERAGE_ROW_SCHEMA,
                "rank": 0,
                "provider_family": public_label(row.get("provider_family"), "unknown"),
                "source_surface": public_label(row.get("source_surface"), "unknown"),
                "endpoint": public_label(row.get("endpoint"), "unknown"),
                "category": public_label(row.get("category"), "unknown"),
                "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
                "stream": bool(row.get("stream")),
                "has_tools": bool(row.get("has_tools")),
                "row_count": row_count,
                "sample_count": row_count,
                "dependency_evidence_decision": decision,
                "dependency_evidence_class": evidence_class,
                "dependency_evidence_status": status,
                "evidence_state": evidence_state,
                "file_dependency_status": public_label(row.get("file_dependency_status"), status),
                "file_dependency_fingerprint_available": fingerprint_available,
                "safe_invalidation_evidence": safe_invalidation,
                "missing_or_blocked_reason": missing_reason,
                "snapshot_count_bucket": snapshot_bucket,
                "candidate_path_count_bucket": candidate_bucket,
                "raw_candidate_path_count_bucket": raw_candidate_bucket,
                "local_dependency_fingerprint": row.get("local_dependency_fingerprint")
                if isinstance(row.get("local_dependency_fingerprint"), dict)
                else _local_dependency_fingerprint_metadata(fingerprint_available, audit),
                "next_action": public_label(row.get("next_action"), "collect-file-invalidation-evidence"),
                "tool_cache_replay_enabled": False,
                "streaming_replay_enabled": False,
                "emits_cache_apply_action": False,
                "cache_entries_written": 0,
                "policy_files_written": False,
                "aggregate_only": True,
                "metadata_only": True,
            }
        )

    coverage_rows.sort(
        key=lambda item: (
            {
                "missing-dependency-evidence": 5,
                "stale-risk-blocker": 4,
                "unsafe-dependency-evidence": 4,
                "unknown-dependency-evidence": 3,
                "stable-dependency-evidence": 2,
            }.get(str(item.get("dependency_evidence_decision")), 0),
            _as_int(item.get("row_count")),
            str(item.get("source_surface")),
            str(item.get("endpoint")),
            str(item.get("category")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(coverage_rows[:capped_limit], start=1):
        row["rank"] = rank

    next_action = _dependency_fingerprint_coverage_next_action(
        stable_rows=stable_rows,
        stale_rows=stale_rows,
        unsafe_rows=unsafe_rows,
        missing_rows=missing_rows,
        unknown_rows=unknown_rows,
    )
    coverage_decision = _dependency_fingerprint_coverage_decision(
        stable_rows=stable_rows,
        stale_rows=stale_rows,
        unsafe_rows=unsafe_rows,
        missing_rows=missing_rows,
        unknown_rows=unknown_rows,
    )
    missing_reason_breakdown = _breakdown(missing_reason_counts)
    top_missing_reason = missing_reason_breakdown[0]["value"] if missing_reason_breakdown else None
    return {
        "schema": REPLAY_DEPENDENCY_FINGERPRINT_COVERAGE_SCHEMA,
        "status": "reported" if coverage_rows else "no-tool-cache-dependency-fingerprint-coverage",
        "read_only": True,
        "coverage_decision": coverage_decision,
        "next_action": next_action,
        "no_apply_guarantee": {
            "schema": "tokenclaw.request_shape_tool_cache_dependency_fingerprint_no_apply_guarantee.v1",
            "emits_cache_apply_action": False,
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "summary": {
            "cohort_count": len(coverage_rows),
            "affected_rows": affected_rows,
            "sample_count": affected_rows,
            "stable_dependency_evidence_rows": stable_rows,
            "stale_dependency_evidence_rows": stale_rows,
            "missing_dependency_evidence_rows": missing_rows,
            "unsafe_dependency_evidence_rows": unsafe_rows,
            "unknown_dependency_evidence_rows": unknown_rows,
            "file_dependency_fingerprint_available_rows": fingerprint_available_rows,
            "file_dependency_fingerprint_missing_rows": fingerprint_missing_rows,
            "safe_invalidation_evidence_rows": safe_invalidation_rows,
            "safe_invalidation_coverage_rate": round((safe_invalidation_rows / affected_rows), 6)
            if affected_rows
            else 0.0,
            "top_missing_or_blocked_reason": top_missing_reason,
            "top_next_action": next_action,
            "cache_apply_action_count": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
        },
        "dependency_evidence_decision_breakdown": _breakdown(decision_counts),
        "dependency_evidence_status_breakdown": _breakdown(status_counts),
        "evidence_state_breakdown": _breakdown(evidence_state_counts),
        "missing_or_blocked_reason_breakdown": missing_reason_breakdown,
        "snapshot_count_bucket_breakdown": _breakdown(snapshot_bucket_counts),
        "candidate_path_count_bucket_breakdown": _breakdown(candidate_bucket_counts),
        "raw_candidate_path_count_bucket_breakdown": _breakdown(raw_candidate_bucket_counts),
        "cohorts": coverage_rows[:capped_limit],
        "acceptance": {
            "reports_dependency_fingerprint_coverage_after_capture": bool(coverage_rows),
            "reports_stable_stale_missing_unsafe_unknown_rows": all(
                key in {
                    "stable_dependency_evidence_rows",
                    "stale_dependency_evidence_rows",
                    "missing_dependency_evidence_rows",
                    "unsafe_dependency_evidence_rows",
                    "unknown_dependency_evidence_rows",
                }
                for key in (
                    "stable_dependency_evidence_rows",
                    "stale_dependency_evidence_rows",
                    "missing_dependency_evidence_rows",
                    "unsafe_dependency_evidence_rows",
                    "unknown_dependency_evidence_rows",
                )
            ),
            "reports_narrow_no_safe_invalidation_reason": bool(coverage_rows)
            and (
                stable_rows > 0
                or bool(top_missing_reason and top_missing_reason != "invalidation-evidence-missing")
                or bool(missing_reason_breakdown)
            ),
            "stable_dependency_evidence_does_not_activate_replay": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in coverage_rows
                if row.get("dependency_evidence_decision") == "stable-dependency-evidence"
            ),
            "emits_no_cache_apply_actions": True,
            "tool_and_streaming_replay_remain_disabled": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in coverage_rows
            ),
            "no_cache_entries_written": True,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def build_request_shape_cache_invalidation_evidence_report(
    cohorts: list[dict[str, Any]],
    *,
    limit: int = 25,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    next_action_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    row_count_total = 0
    tool_blocked_rows = 0
    streaming_blocked_rows = 0
    invalidation_missing_rows = 0
    exact_non_tool_rows = 0
    stable_dependency_evidence_rows = 0
    stale_dependency_evidence_rows = 0
    missing_dependency_evidence_rows = 0
    unsafe_dependency_evidence_rows = 0
    unknown_dependency_evidence_rows = 0
    dependency_decision_counts: dict[str, int] = {}

    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        row_count = _as_int(cohort.get("row_count"))
        if row_count <= 0:
            continue
        readiness = public_label(cohort.get("readiness"), "unknown")
        reason = public_label(cohort.get("reason"), "unknown")
        blockers = sorted(
            {
                public_label(item, "unknown")
                for item in cohort.get("blockers") or []
                if public_label(item, "unknown") != "unknown"
            }
        )
        next_action, secondary_actions, evidence_status, requires_invalidation = _cache_invalidation_next_actions(cohort)
        has_tools = bool(cohort.get("has_tools"))
        stream = bool(cohort.get("stream"))
        exact_non_tool_only = readiness == "replay-ready" and not has_tools and not stream
        file_dependency_status = public_label(cohort.get("file_dependency_status"), "missing")
        dependency_evidence_decision = _cache_dependency_evidence_decision(
            file_dependency_status=file_dependency_status,
            next_action=next_action,
            requires_invalidation=requires_invalidation,
            has_tools=has_tools,
        )

        row_count_total += row_count
        _increment(readiness_counts, readiness, row_count)
        _increment(next_action_counts, next_action, row_count)
        for blocker in blockers or [reason]:
            _increment(blocker_counts, blocker, row_count)
        if has_tools:
            tool_blocked_rows += row_count
        if stream:
            streaming_blocked_rows += row_count
        if requires_invalidation:
            invalidation_missing_rows += row_count
        if exact_non_tool_only:
            exact_non_tool_rows += row_count
        decision = str(dependency_evidence_decision.get("decision") or "unknown-dependency-evidence")
        _increment(dependency_decision_counts, decision, row_count)
        if decision == "stable-dependency-evidence":
            stable_dependency_evidence_rows += row_count
        elif decision == "stale-risk-blocker":
            stale_dependency_evidence_rows += row_count
        elif decision == "unsafe-dependency-evidence":
            unsafe_dependency_evidence_rows += row_count
        elif decision == "missing-dependency-evidence":
            missing_dependency_evidence_rows += row_count
        elif decision == "unknown-dependency-evidence":
            unknown_dependency_evidence_rows += row_count

        rows.append(
            {
                "schema": REPLAY_INVALIDATION_EVIDENCE_COHORT_SCHEMA,
                "rank": 0,
                "provider_family": public_label(cohort.get("provider_family"), "unknown"),
                "source_surface": public_label(cohort.get("source_surface"), "unknown"),
                "endpoint": public_label(cohort.get("endpoint"), "unknown"),
                "category": public_label(cohort.get("category"), "unknown"),
                "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
                "stream": stream,
                "has_tools": has_tools,
                "cache_status": public_label(cohort.get("cache_status"), "unknown"),
                "routing_status": public_label(cohort.get("routing_status"), "unknown"),
                "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
                "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
                "row_count": row_count,
                "readiness": readiness,
                "evidence_status": evidence_status,
                "reason": reason,
                "blocker_codes": blockers,
                "next_action": next_action,
                "secondary_next_actions": sorted(set(secondary_actions)),
                "projected_hits": _as_int(cohort.get("projected_hits")),
                "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
                "requires_explicit_invalidation_safety_evidence": requires_invalidation,
                "dependency_evidence_status": dependency_evidence_decision["status"],
                "dependency_evidence_decision": dependency_evidence_decision,
                "file_dependency_status": file_dependency_status,
                "file_dependency_fingerprint_available": bool(cohort.get("file_dependency_fingerprint_available")),
                "local_dependency_fingerprint": _local_dependency_fingerprint_metadata(
                    bool(cohort.get("file_dependency_fingerprint_available")),
                    cohort.get("file_dependency_audit") if isinstance(cohort.get("file_dependency_audit"), dict) else None,
                ),
                "file_dependency_audit": cohort.get("file_dependency_audit")
                if isinstance(cohort.get("file_dependency_audit"), dict)
                else None,
                "safe_invalidation_evidence": bool(file_dependency_status == "stable"),
                "tool_cache_replay_enabled": False,
                "streaming_replay_enabled": False,
                "cache_entries_written": 0,
                "policy_files_written": False,
                "local_file_backed_policy_compatibility": {
                    "schema": "tokenclaw.request_shape_cache_invalidation_policy_compatibility.v1",
                    "compatible": True,
                    "policy_source": "local-file-backed",
                    "policy_section": "cache",
                    "rule_file": "cache_rules.yaml",
                    "managed_dependency": "optional",
                    "tool_call_cache_enabled": False,
                    "streaming_replay_enabled": False,
                    "requires_operator_apply": True,
                    "reason": "file-backed-local-cache-policy",
                },
                "aggregate_only": True,
                "metadata_only": True,
                "privacy": _replayability_privacy(),
            }
        )

    rows.sort(
        key=lambda item: (
            {
                "collect-file-invalidation-evidence": 5,
                "rank-safe-tool-cache-replay-readiness": 5,
                "refresh-file-invalidation-evidence": 5,
                "keep-tool-cache-blocked": 4,
                "stage-streaming-replay-buffer-fixture": 3,
                "exact-non-tool-only": 2,
                "collect-repeat-evidence": 1,
            }.get(str(item.get("next_action")), 0),
            _as_int(item.get("row_count")),
            str(item.get("provider_family")),
            str(item.get("endpoint")),
            str(item.get("category")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    next_action_breakdown = _breakdown(next_action_counts)
    dependency_evidence_burndown = _dependency_evidence_burndown(rows)
    dependency_evidence_classification = _dependency_evidence_classification(rows)
    return {
        "schema": REPLAY_INVALIDATION_EVIDENCE_SCHEMA,
        "status": "ranked" if rows else "no-cache-invalidation-evidence",
        "read_only": True,
        "next_action": next_action_breakdown[0]["value"] if next_action_breakdown else "keep-cache-replay-observing",
        "summary": {
            "cohort_count": len(rows),
            "row_count": row_count_total,
            "ranked_blocker_cohort_count": sum(1 for row in rows if row.get("readiness") != "replay-ready"),
            "tool_blocked_rows": tool_blocked_rows,
            "streaming_blocked_rows": streaming_blocked_rows,
            "invalidation_missing_rows": invalidation_missing_rows,
            "stable_dependency_evidence_rows": stable_dependency_evidence_rows,
            "stale_dependency_evidence_rows": stale_dependency_evidence_rows,
            "missing_dependency_evidence_rows": missing_dependency_evidence_rows,
            "unsafe_dependency_evidence_rows": unsafe_dependency_evidence_rows,
            "unknown_dependency_evidence_rows": unknown_dependency_evidence_rows,
            "dependency_evidence_decision_count": len(dependency_decision_counts),
            "exact_non_tool_rows": exact_non_tool_rows,
            "top_next_action": next_action_breakdown[0]["value"] if next_action_breakdown else None,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "next_action_breakdown": next_action_breakdown,
        "readiness_breakdown": _breakdown(readiness_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "dependency_evidence_decision_breakdown": _breakdown(dependency_decision_counts),
        "dependency_evidence_burndown": dependency_evidence_burndown,
        "dependency_evidence_classification": dependency_evidence_classification,
        "local_file_backed_policy_compatibility": {
            "schema": "tokenclaw.request_shape_cache_invalidation_policy_compatibility.v1",
            "compatible": True,
            "policy_source": "local-file-backed",
            "policy_section": "cache",
            "rule_file": "cache_rules.yaml",
            "managed_dependency": "optional",
            "tool_call_cache_enabled": False,
            "streaming_replay_enabled": False,
            "requires_operator_apply": True,
            "reason": "file-backed-local-cache-policy",
        },
        "cohorts": rows[:capped_limit],
        "acceptance": {
            "has_ranked_blocker_cohorts": any(row.get("readiness") != "replay-ready" for row in rows),
            "has_next_action": bool(next_action_breakdown),
            "has_local_file_backed_policy_compatibility": True,
            "tool_cohorts_require_invalidation_evidence": all(
                bool(row.get("requires_explicit_invalidation_safety_evidence"))
                for row in rows
                if bool(row.get("has_tools"))
            ),
            "tool_and_streaming_replay_remain_disabled": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in rows
                if bool(row.get("has_tools")) or bool(row.get("stream"))
            ),
            "reports_dependency_evidence_decisions": all(
                isinstance(row.get("dependency_evidence_decision"), dict)
                and bool(row.get("dependency_evidence_decision", {}).get("decision"))
                for row in rows
            ),
            "reports_dependency_evidence_burndown": bool(dependency_evidence_burndown)
            and all(_as_int(item.get("row_count")) > 0 for item in dependency_evidence_burndown),
            "distinguishes_missing_stable_and_stale_dependency_evidence": bool(rows)
            and set(DEPENDENCY_EVIDENCE_CLASSES).issubset(
                set(dependency_evidence_classification["supported_evidence_classes"])
            )
            and all(
                public_label(
                    row.get("dependency_evidence_decision", {}).get("evidence_class"),
                    "unknown-dependency-evidence",
                )
                in set(DEPENDENCY_EVIDENCE_CLASSES) | {"unknown-dependency-evidence", "not-required"}
                for row in rows
                if isinstance(row.get("dependency_evidence_decision"), dict)
            ),
            "distinguishes_missing_stable_stale_and_unsafe_dependency_evidence": bool(rows)
            and set(DEPENDENCY_EVIDENCE_CLASSES).issubset(
                set(dependency_evidence_classification["supported_evidence_classes"])
            ),
            "distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence": bool(rows)
            and set(DEPENDENCY_EVIDENCE_CLASSES).issubset(
                set(dependency_evidence_classification["classification_buckets"])
            )
            and bool(dependency_evidence_classification["all_rows_classified_into_supported_evidence_classes"]),
            "stable_dependency_evidence_does_not_activate_replay": all(
                row.get("next_action") == "rank-safe-tool-cache-replay-readiness"
                and not bool(row.get("tool_cache_replay_enabled"))
                for row in rows
                if row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
            ),
            "stale_or_missing_dependency_evidence_keeps_replay_blocked": all(
                bool(row.get("requires_explicit_invalidation_safety_evidence"))
                and not bool(row.get("tool_cache_replay_enabled"))
                for row in rows
                if row.get("dependency_evidence_decision", {}).get("decision")
                in {"stale-risk-blocker", "missing-dependency-evidence"}
            ),
            "unsafe_dependency_evidence_keeps_replay_blocked": all(
                bool(row.get("requires_explicit_invalidation_safety_evidence"))
                and not bool(row.get("tool_cache_replay_enabled"))
                for row in rows
                if row.get("dependency_evidence_decision", {}).get("decision") == "unsafe-dependency-evidence"
            ),
            "no_cache_entries_written": True,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def _skipped_openai_cache_replay_next_action(blocker: str) -> str:
    code = public_label(blocker, "unknown")
    if code == "safe-invalidation-evidence-present":
        return "rank-safe-tool-cache-replay-readiness"
    if code == "stale-dependency-evidence":
        return "refresh-invalidation-evidence"
    if code in {"invalidation-evidence-missing", "unsafe-dependency-evidence", "unsafe-tool-calls-without-invalidation"}:
        return "add-invalidation-evidence"
    if code in {"tools-present", "tool-call-cache-disabled"}:
        return "keep-tool-cache-disabled"
    if code in {"streaming-replay-not-supported", "unsupported-streaming-shape"}:
        return "wait-for-streaming-replay-support"
    if code == "unsupported-endpoint":
        return "unsupported-endpoint"
    if code == "insufficient-repeat-evidence":
        return "collect-more-repeat-evidence"
    if code == "already-cache-hit":
        return "already-cache-hit"
    return "keep-cache-replay-blocked"


def _primary_skipped_openai_cache_replay_next_action(blockers: list[str], reason: str) -> str:
    candidates = [public_label(item, "unknown") for item in blockers if public_label(item, "unknown") != "unknown"]
    if not candidates and public_label(reason, "unknown") != "unknown":
        candidates = [public_label(reason, "unknown")]
    action_priority = {
        "rank-safe-tool-cache-replay-readiness": -1,
        "refresh-invalidation-evidence": -1,
        "add-invalidation-evidence": 0,
        "keep-tool-cache-disabled": 1,
        "wait-for-streaming-replay-support": 2,
        "unsupported-endpoint": 3,
        "collect-more-repeat-evidence": 4,
        "already-cache-hit": 5,
        "keep-cache-replay-blocked": 6,
    }
    actions = sorted(
        {_skipped_openai_cache_replay_next_action(blocker) for blocker in candidates},
        key=lambda action: (action_priority.get(action, 99), action),
    )
    return actions[0] if actions else "keep-cache-replay-blocked"


def _top_skipped_openai_cache_replay_blocker(
    rows: list[dict[str, Any]],
    blocker_breakdown: list[dict[str, Any]],
) -> str | None:
    if rows:
        top_row = rows[0]
        top_action = str(top_row.get("next_action") or "")
        blockers = [
            public_label(item, "unknown")
            for item in top_row.get("blocker_codes") or []
            if public_label(item, "unknown") != "unknown"
        ]
        for blocker in blockers:
            if _skipped_openai_cache_replay_next_action(blocker) == top_action:
                return blocker
        if blockers:
            return blockers[0]
    return blocker_breakdown[0]["value"] if blocker_breakdown else None


def build_request_shape_skipped_openai_cache_replay_blockers_report(
    cohorts: list[dict[str, Any]],
    *,
    limit: int = 25,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    source_surface_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    sample_count_total = 0
    openai_replay_ready_count = 0
    openai_replay_ready_rows = 0
    openai_skipped_count = 0
    openai_skipped_rows = 0

    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        if public_label(cohort.get("provider_family"), "unknown") != "openai":
            continue
        sample_count = _as_int(cohort.get("row_count") or cohort.get("sample_count"))
        if sample_count <= 0:
            continue
        readiness = public_label(cohort.get("readiness"), "unknown")
        if readiness == "replay-ready":
            openai_replay_ready_count += 1
            openai_replay_ready_rows += sample_count
        elif readiness == "skipped":
            openai_skipped_count += 1
            openai_skipped_rows += sample_count
        if readiness != "skipped":
            continue
        reason = public_label(cohort.get("reason"), "unknown")
        blocker_codes = sorted(
            {
                public_label(item, "unknown")
                for item in cohort.get("blockers") or []
                if public_label(item, "unknown") != "unknown"
            }
        )
        if not blocker_codes and reason != "unknown":
            blocker_codes = [reason]
        next_action = _primary_skipped_openai_cache_replay_next_action(blocker_codes, reason)
        blocker_actions = [
            {
                "blocker_code": blocker,
                "next_action": _skipped_openai_cache_replay_next_action(blocker),
            }
            for blocker in blocker_codes
        ]
        sample_count_total += sample_count
        _increment(next_action_counts, next_action, sample_count)
        _increment(source_surface_counts, cohort.get("source_surface"), sample_count)
        _increment(endpoint_counts, cohort.get("endpoint"), sample_count)
        for blocker in blocker_codes:
            _increment(blocker_counts, blocker, sample_count)
        rows.append(
            {
                "schema": REPLAY_SKIPPED_OPENAI_BLOCKER_ROW_SCHEMA,
                "rank": 0,
                "provider_family": "openai",
                "source_surface": public_label(cohort.get("source_surface"), "unknown"),
                "endpoint": public_label(cohort.get("endpoint"), "unknown"),
                "category": public_label(cohort.get("category"), "unknown"),
                "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
                "stream": bool(cohort.get("stream")),
                "has_tools": bool(cohort.get("has_tools")),
                "cache_status": public_label(cohort.get("cache_status"), "unknown"),
                "routing_status": public_label(cohort.get("routing_status"), "unknown"),
                "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
                "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
                "sample_count": sample_count,
                "row_count": sample_count,
                "projected_hits": _as_int(cohort.get("projected_hits")),
                "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
                "reason": reason,
                "blocker_codes": blocker_codes,
                "blocker_actions": blocker_actions,
                "next_action": next_action,
                "file_dependency_status": public_label(cohort.get("file_dependency_status"), "missing"),
                "file_dependency_fingerprint_available": bool(cohort.get("file_dependency_fingerprint_available")),
                "file_dependency_audit": cohort.get("file_dependency_audit")
                if isinstance(cohort.get("file_dependency_audit"), dict)
                else None,
                "safe_invalidation_evidence": bool(cohort.get("file_dependency_status") == "stable"),
                "tool_cache_replay_enabled": False,
                "streaming_replay_enabled": False,
                "emits_cache_apply_action": False,
                "requires_operator_apply": False,
                "aggregate_only": True,
                "metadata_only": True,
                "privacy": _replayability_privacy(),
            }
        )

    action_priority = {
        "rank-safe-tool-cache-replay-readiness": 5,
        "refresh-invalidation-evidence": 5,
        "add-invalidation-evidence": 5,
        "keep-tool-cache-disabled": 4,
        "wait-for-streaming-replay-support": 3,
        "unsupported-endpoint": 2,
        "collect-more-repeat-evidence": 1,
    }
    rows.sort(
        key=lambda item: (
            action_priority.get(str(item.get("next_action")), 0),
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("projected_hits")),
            _as_int(item.get("sample_count")),
            str(item.get("source_surface")),
            str(item.get("endpoint")),
            str(item.get("category")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    blocker_breakdown = _breakdown(blocker_counts)
    next_action_breakdown = _breakdown(next_action_counts)
    top_blocker_code = _top_skipped_openai_cache_replay_blocker(rows, blocker_breakdown)
    top_blocker_count = blocker_counts.get(top_blocker_code or "", 0)
    return {
        "schema": REPLAY_SKIPPED_OPENAI_BLOCKERS_SCHEMA,
        "status": "ranked" if rows else "no-skipped-openai-cache-replay-cohorts",
        "read_only": True,
        "wrote_store": False,
        "wrote_local_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "next_action": next_action_breakdown[0]["value"] if next_action_breakdown else "keep-cache-replay-observing",
        "summary": {
            "skipped_openai_cohort_count": len(rows),
            "replay_ready_count": openai_replay_ready_count,
            "replay_ready_rows": openai_replay_ready_rows,
            "skipped_count": openai_skipped_count,
            "skipped_rows": openai_skipped_rows,
            "sample_count": sample_count_total,
            "affected_rows": sample_count_total,
            "projected_hits": sum(_as_int(row.get("projected_hits")) for row in rows),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in rows), 6),
            "top_blocker_code": top_blocker_code,
            "top_blocker_count": top_blocker_count,
            "blocker_code_count": len(blocker_counts),
            "top_next_action": next_action_breakdown[0]["value"] if next_action_breakdown else None,
            "cache_apply_action_count": 0,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "source_surface_breakdown": _breakdown(source_surface_counts),
        "endpoint_breakdown": _breakdown(endpoint_counts),
        "blocker_breakdown": blocker_breakdown,
        "next_action_breakdown": next_action_breakdown,
        "cohorts": rows[:capped_limit],
        "acceptance": {
            "has_ranked_skipped_openai_cohorts": bool(rows),
            "has_rank": all(_as_int(row.get("rank")) > 0 for row in rows[:capped_limit]),
            "has_blocker_codes": all(bool(row.get("blocker_codes")) for row in rows[:capped_limit]),
            "has_sample_count": all(_as_int(row.get("sample_count")) > 0 for row in rows[:capped_limit]),
            "has_deterministic_next_action": all(bool(row.get("next_action")) for row in rows[:capped_limit]),
            "has_acceptance_summary_fields": bool(
                openai_replay_ready_count >= 0
                and openai_skipped_count >= 0
                and sample_count_total >= 0
                and (not rows or blocker_breakdown)
                and (not rows or next_action_breakdown)
            ),
            "covers_required_blockers": all(
                code in blocker_counts
                for code in (
                    "invalidation-evidence-missing",
                    "tools-present",
                    "unsafe-tool-calls-without-invalidation",
                    "streaming-replay-not-supported",
                )
            ),
            "emits_no_cache_apply_actions": True,
            "tool_and_streaming_replay_remain_disabled": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in rows
            ),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def _tool_cache_replay_evidence_state(
    *,
    dependency_decision: str,
    next_action: str,
) -> tuple[str, str, str]:
    if dependency_decision == "stable-dependency-evidence":
        return (
            "dependency-gated-review-ready",
            "safe-invalidation-evidence-present",
            "rank-safe-tool-cache-replay-readiness",
        )
    if dependency_decision == "stale-risk-blocker":
        return (
            "blocked-stale-dependency-evidence",
            "stale-dependency-evidence",
            "refresh-file-invalidation-evidence",
        )
    if dependency_decision == "unsafe-dependency-evidence":
        return (
            "blocked-unsafe-dependency-evidence",
            "unsafe-tool-calls-without-invalidation",
            "collect-file-invalidation-evidence",
        )
    if dependency_decision == "missing-dependency-evidence":
        return (
            "blocked-missing-dependency-evidence",
            "invalidation-evidence-missing",
            "collect-file-invalidation-evidence",
        )
    return (
        "blocked-unknown-dependency-evidence",
        "dependency-evidence-unknown",
        next_action or "collect-file-invalidation-evidence",
    )


def _tool_cache_review_candidate_blocker(dependency_decision: str) -> tuple[str, str]:
    if dependency_decision == "stale-risk-blocker":
        return "stale-dependency-evidence", "refresh-file-invalidation-evidence"
    if dependency_decision == "unsafe-dependency-evidence":
        return "unsafe-tool-calls-without-invalidation", "collect-file-invalidation-evidence"
    if dependency_decision == "missing-dependency-evidence":
        return "invalidation-evidence-missing", "collect-file-invalidation-evidence"
    return "dependency-evidence-unknown", "collect-file-invalidation-evidence"


def _tool_cache_replay_proof(row: dict[str, Any]) -> dict[str, Any]:
    cache_hit_count = _as_int(row.get("cache_hit_count"))
    observed_hit_count = max(
        _as_int(row.get("observed_hits")),
        _as_int(row.get("exact_hit_count")),
        _as_int(row.get("live_repeat_cache_hit_count")),
    )
    cache_status = public_label(row.get("cache_status"), "unknown")
    live_repeat_confirmed = bool(row.get("live_repeat_confirmed")) or cache_hit_count > 0 or cache_status == "hit"
    observed_hit_proof = bool(row.get("observed_hit_proof")) or observed_hit_count > 0
    proof_available = live_repeat_confirmed or observed_hit_proof
    if live_repeat_confirmed:
        reason = "live-repeat-confirmed"
    elif observed_hit_proof:
        reason = "observed-hit-proof"
    else:
        reason = "missing-live-repeat-or-observed-hit-proof"
    return {
        "schema": "tokenclaw.request_shape_tool_cache_replay_proof.v1",
        "proof_available": proof_available,
        "reason": reason,
        "live_repeat_confirmed": live_repeat_confirmed,
        "observed_hit_proof": observed_hit_proof,
        "live_repeat_cache_hit_count": cache_hit_count,
        "observed_hit_count": observed_hit_count,
        "requires_live_repeat_or_observed_hit_proof": True,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _replayability_privacy(),
    }


def _tool_cache_review_readiness_gate(
    readiness_gate: dict[str, Any],
    replay_proof: dict[str, Any],
    *,
    dependency_decision: str,
) -> dict[str, Any]:
    gate = dict(readiness_gate)
    proof_required = dependency_decision == "stable-dependency-evidence"
    proof_available = bool(replay_proof.get("proof_available"))
    gate["tool_cache_replay_review_gate"] = {
        "schema": "tokenclaw.request_shape_tool_cache_replay_review_gate.v1",
        "requires_stable_dependency_evidence": True,
        "requires_live_repeat_or_observed_hit_proof": proof_required,
        "proof_gate_passed": (not proof_required) or proof_available,
        "live_repeat_confirmed": bool(replay_proof.get("live_repeat_confirmed")),
        "observed_hit_proof": bool(replay_proof.get("observed_hit_proof")),
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _replayability_privacy(),
    }
    if proof_required and not proof_available:
        gate["raw_stage_gate_status"] = gate.get("gate_status")
        gate["stage_allowed"] = False
        gate["gate_status"] = "missing-live-repeat-or-observed-hit-proof"
        gate["next_action"] = "wait-for-live-repeat-or-observed-hit-proof"
        gate["reason"] = "missing-live-repeat-or-observed-hit-proof"
    return gate


def _tool_cache_retired_no_repeat(row: dict[str, Any]) -> bool:
    values: list[str] = []
    for key in (
        "current_status",
        "state",
        "reason",
        "evidence_reason",
        "promotion_decision",
        "promotion_readiness",
        "policy_decision",
        "observed_hit_blocker",
        "no_op_reason",
    ):
        value = row.get(key)
        if value is not None:
            values.append(public_label(value, "unknown"))
    for key in ("blocker_codes", "reason_codes"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(public_label(item, "unknown") for item in value)
    return any(
        value
        in {
            "retired-no-repeat",
            "retire-staged-no-repeat",
            "repeat-window-elapsed-no-live-repeat",
            "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired",
        }
        for value in values
    )


def _tool_cache_review_candidate_from_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    row_count = _as_int(row.get("row_count") or row.get("sample_count"))
    projected_hits = _as_int(row.get("projected_hits"))
    projected_savings = round(_as_float(row.get("projected_savings_usd")), 6)
    readiness_gate = _cache_replay_readiness_gate(
        row_count=row_count,
        projected_hits=projected_hits,
        projected_savings_usd=projected_savings,
        cache_hit_count=_as_int(row.get("cache_hit_count")),
    )
    dependency_decision_meta = (
        row.get("dependency_evidence_decision")
        if isinstance(row.get("dependency_evidence_decision"), dict)
        else {}
    )
    dependency_decision = public_label(
        dependency_decision_meta.get("decision"),
        "unknown-dependency-evidence",
    )
    replay_proof = _tool_cache_replay_proof(row)
    readiness_gate = _tool_cache_review_readiness_gate(
        readiness_gate,
        replay_proof,
        dependency_decision=dependency_decision,
    )
    retired_no_repeat = _tool_cache_retired_no_repeat(row)
    streaming_replay_blocked = bool(row.get("stream"))
    if retired_no_repeat:
        candidate_status = "blocked"
        candidate_decision = "no-op-retired-no-repeat"
        blocker_reason = "retire-staged-no-repeat"
        next_action = "keep-tool-cache-replay-retired-no-repeat"
        stageable_after_review = False
    elif streaming_replay_blocked:
        candidate_status = "blocked"
        candidate_decision = "no-op-streaming-replay-not-supported"
        blocker_reason = "streaming-replay-not-supported"
        next_action = "stage-streaming-replay-buffer-fixture"
        stageable_after_review = False
        readiness_gate["raw_stage_gate_status"] = readiness_gate.get("gate_status")
        readiness_gate["stage_allowed"] = False
        readiness_gate["gate_status"] = "streaming-replay-not-supported"
        readiness_gate["next_action"] = next_action
        readiness_gate["reason"] = blocker_reason
    elif dependency_decision == "stable-dependency-evidence":
        proof_available = bool(replay_proof.get("proof_available"))
        candidate_status = "review-ready" if proof_available else "review-only-gated"
        candidate_decision = "review-only-candidate" if proof_available else "no-op-missing-live-repeat-proof"
        blocker_reason = None if proof_available else "missing-live-repeat-or-observed-hit-proof"
        next_action = (
            "review-tool-cache-replay-candidate"
            if proof_available
            else "wait-for-live-repeat-or-observed-hit-proof"
        )
        stageable_after_review = proof_available
    else:
        blocker_reason, next_action = _tool_cache_review_candidate_blocker(dependency_decision)
        candidate_status = "blocked"
        candidate_decision = "blocked"
        stageable_after_review = False

    return {
        "schema": REPLAY_TOOL_REVIEW_CANDIDATE_ROW_SCHEMA,
        "rank": 0,
        "provider_family": "openai",
        "source_surface": public_label(row.get("source_surface"), "unknown"),
        "endpoint": public_label(row.get("endpoint"), "unknown"),
        "category": public_label(row.get("category"), "unknown"),
        "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": True,
        "sample_count": row_count,
        "row_count": row_count,
        "review_only": candidate_status == "review-ready",
        "candidate_status": candidate_status,
        "candidate_decision": candidate_decision,
        "stageable_after_review": stageable_after_review,
        "stage_allowed": False,
        "next_action": next_action,
        "blocker_reason": blocker_reason,
        "no_op_reason": blocker_reason if candidate_decision.startswith("no-op") else None,
        "retired_no_repeat": retired_no_repeat,
        "dependency_evidence_status": public_label(
            row.get("dependency_evidence_status") or row.get("file_dependency_status"),
            "missing",
        ),
        "dependency_evidence_decision": dependency_decision_meta,
        "evidence_state": public_label(row.get("evidence_state"), _cache_dependency_evidence_state(dependency_decision)),
        "blocker_codes": row.get("blocker_codes") if isinstance(row.get("blocker_codes"), list) else [],
        "projected_hits": projected_hits,
        "projected_savings_usd": projected_savings,
        "readiness_gate": readiness_gate,
        "safe_invalidation_evidence": bool(row.get("safe_invalidation_evidence")),
        "file_dependency_status": public_label(row.get("file_dependency_status"), "missing"),
        "file_dependency_fingerprint_available": bool(row.get("file_dependency_fingerprint_available")),
        "local_dependency_fingerprint": row.get("local_dependency_fingerprint")
        if isinstance(row.get("local_dependency_fingerprint"), dict)
        else _local_dependency_fingerprint_metadata(False),
        "replay_proof": replay_proof,
        "requires_live_repeat_or_observed_hit_proof": dependency_decision == "stable-dependency-evidence",
        "allows_savings_floor_without_replay_proof": False,
        "requires_operator_review": True,
        "requires_explicit_later_promotion": True,
        "tool_cache_replay_enabled": False,
        "streaming_replay_enabled": False,
        "emits_cache_apply_action": False,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "aggregate_only": True,
        "metadata_only": True,
        "privacy": _replayability_privacy(),
    }


def _build_tool_cache_review_candidates(rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    dependency_decision_counts: dict[str, int] = {}
    review_ready_rows = 0
    review_only_gated_rows = 0
    blocked_rows = 0
    stageable_after_review_rows = 0
    projected_hits_total = 0
    projected_savings_total = 0.0

    for row in rows:
        if not isinstance(row, dict):
            continue
        dependency_decision_meta = row.get("dependency_evidence_decision")
        dependency_decision = (
            public_label(dependency_decision_meta.get("decision"), "unknown-dependency-evidence")
            if isinstance(dependency_decision_meta, dict)
            else "unknown-dependency-evidence"
        )
        if dependency_decision not in {
            "stable-dependency-evidence",
            "stale-risk-blocker",
            "unsafe-dependency-evidence",
            "missing-dependency-evidence",
            "unknown-dependency-evidence",
        }:
            continue
        candidate = _tool_cache_review_candidate_from_evidence_row(row)
        row_count = _as_int(candidate.get("row_count"))
        candidates.append(candidate)
        _increment(status_counts, candidate.get("candidate_status"), row_count)
        _increment(next_action_counts, candidate.get("next_action"), row_count)
        _increment(dependency_decision_counts, dependency_decision, row_count)
        for blocker in candidate.get("blocker_codes") or [candidate.get("blocker_reason")]:
            _increment(blocker_counts, blocker, row_count)
        if candidate["candidate_status"] == "review-ready":
            review_ready_rows += row_count
        elif candidate["candidate_status"] == "review-only-gated":
            review_only_gated_rows += row_count
        else:
            blocked_rows += row_count
        if candidate["stageable_after_review"]:
            stageable_after_review_rows += row_count
        projected_hits_total += _as_int(candidate.get("projected_hits"))
        projected_savings_total += _as_float(candidate.get("projected_savings_usd"))

    candidates.sort(
        key=lambda item: (
            {
                "review-ready": 5,
                "review-only-gated": 4,
                "blocked": 3,
            }.get(str(item.get("candidate_status")), 0),
            bool(item.get("stageable_after_review")),
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("projected_hits")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(candidates[:capped_limit], start=1):
        row["rank"] = rank

    return {
        "schema": REPLAY_TOOL_REVIEW_CANDIDATES_SCHEMA,
        "status": "ranked" if candidates else "no-tool-cache-review-candidates",
        "read_only": True,
        "next_action": _breakdown(next_action_counts)[0]["value"] if next_action_counts else "keep-tool-cache-replay-blocked",
        "summary": {
            "candidate_count": len(candidates),
            "review_only_candidate_count": sum(1 for row in candidates if row.get("review_only")),
            "review_ready_rows": review_ready_rows,
            "review_only_gated_rows": review_only_gated_rows,
            "blocked_rows": blocked_rows,
            "stageable_after_review_rows": stageable_after_review_rows,
            "projected_hits": projected_hits_total,
            "projected_savings_usd": round(projected_savings_total, 6),
            "cache_apply_action_count": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
        },
        "status_breakdown": _breakdown(status_counts),
        "next_action_breakdown": _breakdown(next_action_counts),
        "dependency_evidence_decision_breakdown": _breakdown(dependency_decision_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "candidates": candidates[:capped_limit],
        "acceptance": {
            "stable_dependency_evidence_emits_review_only_candidate": any(
                row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
                and row.get("review_only") is True
                and row.get("candidate_status") == "review-ready"
                and isinstance(row.get("replay_proof"), dict)
                and bool(row["replay_proof"].get("proof_available"))
                for row in candidates
            ),
            "review_only_candidates_have_stable_dependency_and_proof": all(
                row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
                and row.get("candidate_status") == "review-ready"
                and isinstance(row.get("replay_proof"), dict)
                and bool(row["replay_proof"].get("proof_available"))
                for row in candidates
                if row.get("review_only") is True
            ),
            "requires_live_repeat_or_observed_hit_proof": all(
                isinstance(row.get("replay_proof"), dict)
                and (
                    row.get("candidate_status") != "review-ready"
                    or bool(row.get("replay_proof", {}).get("proof_available"))
                )
                for row in candidates
                if row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
            ),
            "stable_without_live_repeat_or_observed_hit_is_noop": all(
                row.get("candidate_decision") == "no-op-missing-live-repeat-proof"
                for row in candidates
                if row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
                and isinstance(row.get("replay_proof"), dict)
                and not bool(row.get("replay_proof", {}).get("proof_available"))
            ),
            "retired_no_repeat_emits_noop": all(
                row.get("candidate_decision") == "no-op-retired-no-repeat"
                for row in candidates
                if row.get("retired_no_repeat") is True
            ),
            "streaming_candidates_do_not_become_review_ready": all(
                row.get("candidate_status") != "review-ready"
                and row.get("candidate_decision") == "no-op-streaming-replay-not-supported"
                and row.get("blocker_reason") == "streaming-replay-not-supported"
                for row in candidates
                if bool(row.get("stream"))
            ),
            "does_not_allow_savings_floor_without_replay_proof": all(
                not bool(row.get("allows_savings_floor_without_replay_proof"))
                for row in candidates
                if row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
            ),
            "blocked_dependency_evidence_has_distinct_reason_codes": all(
                bool(row.get("blocker_reason"))
                for row in candidates
                if row.get("candidate_status") == "blocked"
            ),
            "emits_no_cache_apply_actions": True,
            "tool_and_streaming_replay_remain_disabled": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in candidates
            ),
            "no_cache_entries_written": True,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def _managed_preview_outcome_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows = report.get("outcomes")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _normalized_tool_cache_preview_action(value: Any, *, decision: Any = None) -> str:
    text = public_label(value, "unknown")
    decision_text = public_label(decision, "unknown")
    review_actions = {
        "review-tool-cache-replay-candidate",
        "rank-safe-tool-cache-replay-readiness",
        "review-only-recommendation",
        "preview-tool-cache-replay-candidate",
        "draft-tool-cache-replay-review",
    }
    if text in review_actions or decision_text == "review-only-recommendation":
        return "review-tool-cache-replay-candidate"
    if text in {
        "wait-for-live-repeat-or-observed-hit-proof",
        "keep-tool-cache-replay-retired-no-repeat",
        "keep-cache-replay-noop",
        "noop",
        "no-op",
    } or decision_text in {"no-op", "noop"}:
        return "no-op"
    if text in {
        "collect-file-invalidation-evidence",
        "refresh-file-invalidation-evidence",
        "keep-tool-cache-blocked",
        "add-invalidation-evidence",
    } or decision_text in {"keep-blocked", "omitted"}:
        return "keep-blocked"
    return text


def _managed_tool_cache_preview_outcome(managed_preview_outcomes: dict[str, Any] | None) -> dict[str, Any]:
    rows = [
        row
        for row in _managed_preview_outcome_rows(managed_preview_outcomes)
        if public_label(row.get("local_action_family"), "unknown") == "cache"
        and public_label(row.get("evidence_schema"), "unknown") == REPLAY_TOOL_REPLAY_EVIDENCE_SCHEMA
    ]
    if not rows:
        return {}

    def _priority(row: dict[str, Any]) -> tuple[int, float]:
        classification = public_label(row.get("classification"), "unknown")
        if bool(row.get("failed_closed")):
            score = 0
        elif bool(row.get("stale")) or classification == "stale-preview":
            score = 1
        elif bool(row.get("missing_preview_decision")):
            score = 2
        elif bool(row.get("disagrees_with_local_evidence")) or classification == "managed-local-disagreement":
            score = 3
        elif classification == "review-only":
            score = 5
        else:
            score = 4
        age = _as_float(row.get("preview_age_hours"), 999999.0)
        return (score, -age)

    return sorted(rows, key=_priority, reverse=True)[0]


def _tool_cache_managed_preview_agreement(
    candidate: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = _managed_tool_cache_preview_outcome(managed_preview_outcomes)
    local_next_action = public_label(candidate.get("next_action"), "unknown")
    local_decision = public_label(candidate.get("candidate_decision"), "unknown")
    normalized_local = _normalized_tool_cache_preview_action(local_next_action, decision=local_decision)
    required = candidate.get("candidate_status") == "review-ready"

    if not selected:
        reason = "missing-managed-preview-outcome" if required else "managed-preview-not-required-for-local-noop"
        normalized_managed = "missing"
        agreed = False
    elif bool(selected.get("failed_closed")):
        reason = "managed-preview-failed-closed"
        normalized_managed = _normalized_tool_cache_preview_action(selected.get("next_action"), decision=selected.get("decision"))
        agreed = False
    elif bool(selected.get("stale")) or public_label(selected.get("classification"), "unknown") == "stale-preview":
        reason = "stale-managed-preview-outcome"
        normalized_managed = _normalized_tool_cache_preview_action(selected.get("next_action"), decision=selected.get("decision"))
        agreed = False
    elif bool(selected.get("missing_preview_decision")):
        reason = "missing-managed-preview-decision"
        normalized_managed = _normalized_tool_cache_preview_action(selected.get("next_action"), decision=selected.get("decision"))
        agreed = False
    elif bool(selected.get("disagrees_with_local_evidence")) or public_label(
        selected.get("classification"),
        "unknown",
    ) == "managed-local-disagreement":
        reason = "managed-local-disagreement"
        normalized_managed = _normalized_tool_cache_preview_action(selected.get("next_action"), decision=selected.get("decision"))
        agreed = False
    else:
        normalized_managed = _normalized_tool_cache_preview_action(selected.get("next_action"), decision=selected.get("decision"))
        agreed = normalized_local == normalized_managed and (not required or normalized_local == "review-tool-cache-replay-candidate")
        if agreed:
            reason = "local-managed-preview-agree" if required else "local-noop-managed-preview-compatible"
        elif not required and normalized_managed == "review-tool-cache-replay-candidate":
            reason = "local-proof-gate-blocks-managed-preview"
        else:
            reason = "managed-preview-action-disagreement"

    return {
        "schema": REPLAY_TOOL_MANAGED_PREVIEW_AGREEMENT_SCHEMA,
        "required": required,
        "agreed": agreed,
        "reason": reason,
        "local_next_action": local_next_action,
        "managed_next_action": selected.get("next_action") if selected else None,
        "normalized_local_action": normalized_local,
        "normalized_managed_action": normalized_managed,
        "managed_classification": selected.get("classification") if selected else None,
        "managed_decision": selected.get("decision") if selected else None,
        "managed_preview_age_hours": selected.get("preview_age_hours") if selected else None,
        "handoff_ref": selected.get("handoff_ref") if selected else None,
        "preview_ref": selected.get("preview_ref") if selected else None,
        "review_only": True,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _replayability_privacy(),
    }


def _tool_cache_replay_preview_from_candidate(
    candidate: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any]:
    agreement = _tool_cache_managed_preview_agreement(candidate, managed_preview_outcomes)
    candidate_status = public_label(candidate.get("candidate_status"), "blocked")
    if candidate_status == "review-ready":
        preview_status = "review-ready"
        preview_decision = "review-only-replay-preview"
    elif candidate_status == "review-only-gated":
        preview_status = "review-only-noop"
        preview_decision = "no-op-missing-live-repeat-proof"
    else:
        preview_status = "blocked"
        preview_decision = "keep-blocked"

    return {
        "schema": REPLAY_TOOL_MANAGED_LOCAL_PREVIEW_ROW_SCHEMA,
        "rank": 0,
        "provider_family": "openai",
        "source_surface": public_label(candidate.get("source_surface"), "unknown"),
        "endpoint": public_label(candidate.get("endpoint"), "unknown"),
        "category": public_label(candidate.get("category"), "unknown"),
        "workflow_phase": public_label(candidate.get("workflow_phase"), "unknown"),
        "stream": bool(candidate.get("stream")),
        "has_tools": True,
        "sample_count": _as_int(candidate.get("sample_count") or candidate.get("row_count")),
        "row_count": _as_int(candidate.get("row_count") or candidate.get("sample_count")),
        "preview_status": preview_status,
        "preview_decision": preview_decision,
        "candidate_status": candidate_status,
        "candidate_decision": public_label(candidate.get("candidate_decision"), "unknown"),
        "local_next_action": public_label(candidate.get("next_action"), "unknown"),
        "next_action": public_label(candidate.get("next_action"), "unknown"),
        "no_op_reason": public_label(candidate.get("no_op_reason"), "none")
        if candidate.get("no_op_reason")
        else None,
        "blocker_reason": public_label(candidate.get("blocker_reason"), "none")
        if candidate.get("blocker_reason")
        else None,
        "review_only": True,
        "stageable_after_review": bool(candidate.get("stageable_after_review")),
        "stage_allowed": False,
        "requires_operator_review": True,
        "requires_explicit_later_promotion": True,
        "managed_preview_agreement": agreement,
        "dependency_evidence_status": public_label(candidate.get("dependency_evidence_status"), "missing"),
        "dependency_evidence_decision": candidate.get("dependency_evidence_decision")
        if isinstance(candidate.get("dependency_evidence_decision"), dict)
        else {},
        "file_dependency_status": public_label(candidate.get("file_dependency_status"), "missing"),
        "safe_invalidation_evidence": bool(candidate.get("safe_invalidation_evidence")),
        "file_dependency_fingerprint_available": bool(candidate.get("file_dependency_fingerprint_available")),
        "local_dependency_fingerprint": candidate.get("local_dependency_fingerprint")
        if isinstance(candidate.get("local_dependency_fingerprint"), dict)
        else _local_dependency_fingerprint_metadata(False),
        "replay_proof": candidate.get("replay_proof") if isinstance(candidate.get("replay_proof"), dict) else {},
        "projected_hits": _as_int(candidate.get("projected_hits")),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 6),
        "tool_cache_replay_enabled": False,
        "streaming_replay_enabled": False,
        "emits_cache_apply_action": False,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "aggregate_only": True,
        "metadata_only": True,
        "privacy": _replayability_privacy(),
    }


def _build_tool_cache_managed_local_replay_previews(
    review_candidates: dict[str, Any],
    *,
    managed_preview_outcomes: dict[str, Any] | None = None,
    limit: int,
) -> dict[str, Any]:
    raw_candidates = review_candidates.get("candidates") if isinstance(review_candidates, dict) else []
    previews = [
        _tool_cache_replay_preview_from_candidate(candidate, managed_preview_outcomes)
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    ]
    previews.sort(
        key=lambda item: (
            {
                "review-ready": 5,
                "review-only-noop": 4,
                "blocked": 3,
            }.get(str(item.get("preview_status")), 0),
            bool(item.get("managed_preview_agreement", {}).get("agreed")),
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("projected_hits")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(previews[:capped_limit], start=1):
        row["rank"] = rank

    status_counts: dict[str, int] = {}
    agreement_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    dependency_counts: dict[str, int] = {}
    review_ready_rows = 0
    noop_rows = 0
    blocked_rows = 0
    managed_required_rows = 0
    managed_agreed_rows = 0
    managed_missing_rows = 0
    for row in previews:
        row_count = _as_int(row.get("row_count"))
        _increment(status_counts, row.get("preview_status"), row_count)
        _increment(action_counts, row.get("next_action"), row_count)
        decision_meta = row.get("dependency_evidence_decision") if isinstance(row.get("dependency_evidence_decision"), dict) else {}
        _increment(dependency_counts, decision_meta.get("decision") or "unknown-dependency-evidence", row_count)
        agreement = row.get("managed_preview_agreement") if isinstance(row.get("managed_preview_agreement"), dict) else {}
        _increment(agreement_counts, agreement.get("reason") or "unknown", row_count)
        if row.get("preview_status") == "review-ready":
            review_ready_rows += row_count
        elif row.get("preview_status") == "review-only-noop":
            noop_rows += row_count
        else:
            blocked_rows += row_count
        if agreement.get("required"):
            managed_required_rows += row_count
            if agreement.get("agreed"):
                managed_agreed_rows += row_count
            if agreement.get("reason") == "missing-managed-preview-outcome":
                managed_missing_rows += row_count

    return {
        "schema": REPLAY_TOOL_MANAGED_LOCAL_PREVIEWS_SCHEMA,
        "status": "ranked" if previews else "no-tool-cache-replay-previews",
        "read_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "next_action": _breakdown(action_counts)[0]["value"] if action_counts else "keep-tool-cache-replay-blocked",
        "summary": {
            "preview_count": len(previews),
            "review_ready_preview_rows": review_ready_rows,
            "review_only_noop_rows": noop_rows,
            "blocked_preview_rows": blocked_rows,
            "managed_preview_required_rows": managed_required_rows,
            "managed_preview_agreement_rows": managed_agreed_rows,
            "managed_preview_missing_rows": managed_missing_rows,
            "cache_apply_action_count": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
        },
        "status_breakdown": _breakdown(status_counts),
        "next_action_breakdown": _breakdown(action_counts),
        "managed_preview_agreement_breakdown": _breakdown(agreement_counts),
        "dependency_evidence_decision_breakdown": _breakdown(dependency_counts),
        "previews": previews[:capped_limit],
        "acceptance": {
            "emits_managed_local_replay_previews": bool(previews),
            "stable_with_proof_emits_review_ready_preview": any(
                row.get("preview_status") == "review-ready"
                and isinstance(row.get("dependency_evidence_decision"), dict)
                and row["dependency_evidence_decision"].get("decision") == "stable-dependency-evidence"
                and bool(row.get("replay_proof", {}).get("proof_available"))
                for row in previews
            ),
            "stable_without_live_repeat_or_observed_hit_is_noop": all(
                row.get("preview_decision") == "no-op-missing-live-repeat-proof"
                for row in previews
                if isinstance(row.get("dependency_evidence_decision"), dict)
                and row["dependency_evidence_decision"].get("decision") == "stable-dependency-evidence"
                and isinstance(row.get("replay_proof"), dict)
                and not bool(row.get("replay_proof", {}).get("proof_available"))
            ),
            "unsafe_or_missing_dependency_emits_no_apply_actions": all(
                not bool(row.get("emits_cache_apply_action"))
                and not bool(row.get("tool_cache_replay_enabled"))
                and _as_int(row.get("cache_entries_written")) == 0
                for row in previews
                if isinstance(row.get("dependency_evidence_decision"), dict)
                and row["dependency_evidence_decision"].get("decision")
                in {"unsafe-dependency-evidence", "missing-dependency-evidence", "unknown-dependency-evidence"}
            ),
            "review_ready_previews_require_stable_dependency_and_proof": all(
                isinstance(row.get("dependency_evidence_decision"), dict)
                and row["dependency_evidence_decision"].get("decision") == "stable-dependency-evidence"
                and isinstance(row.get("replay_proof"), dict)
                and bool(row["replay_proof"].get("proof_available"))
                and not bool(row.get("stream"))
                for row in previews
                if row.get("preview_status") == "review-ready"
            ),
            "streaming_shapes_do_not_become_review_ready_previews": all(
                row.get("preview_status") != "review-ready"
                and row.get("preview_decision") == "keep-blocked"
                and row.get("blocker_reason") == "streaming-replay-not-supported"
                for row in previews
                if bool(row.get("stream"))
            ),
            "managed_preview_agreement_is_review_only": all(
                isinstance(row.get("managed_preview_agreement"), dict)
                and row["managed_preview_agreement"].get("review_only") is True
                and not bool(row["managed_preview_agreement"].get("policy_files_written"))
                and not bool(row["managed_preview_agreement"].get("provider_calls_made"))
                for row in previews
            ),
            "emits_no_cache_apply_actions": True,
            "tool_and_streaming_replay_remain_disabled": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in previews
            ),
            "no_cache_entries_written": True,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def build_request_shape_tool_cache_replay_evidence_report(
    cohorts: list[dict[str, Any]],
    *,
    limit: int = 25,
    managed_preview_outcomes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    evidence_state_counts: dict[str, int] = {}
    dependency_decision_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    sample_count_total = 0
    stable_dependency_rows = 0
    stale_dependency_rows = 0
    missing_dependency_rows = 0
    unsafe_dependency_rows = 0
    unknown_dependency_rows = 0
    unsafe_tool_call_blocker_rows = 0

    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        if public_label(cohort.get("provider_family"), "unknown") != "openai":
            continue
        if not bool(cohort.get("has_tools")):
            continue
        sample_count = _as_int(cohort.get("row_count") or cohort.get("sample_count"))
        if sample_count <= 0:
            continue
        reason = public_label(cohort.get("reason"), "unknown")
        blocker_codes = sorted(
            {
                public_label(item, "unknown")
                for item in cohort.get("blockers") or []
                if public_label(item, "unknown") != "unknown"
            }
        )
        if not blocker_codes and reason != "unknown":
            blocker_codes = [reason]
        readiness = public_label(cohort.get("readiness"), "unknown")
        file_dependency_status = public_label(cohort.get("file_dependency_status"), "missing")
        next_action, secondary_actions, _evidence_status, requires_invalidation = _cache_invalidation_next_actions(cohort)
        dependency_evidence_decision = _cache_dependency_evidence_decision(
            file_dependency_status=file_dependency_status,
            next_action=next_action,
            requires_invalidation=requires_invalidation,
            has_tools=True,
        )
        dependency_decision = public_label(
            dependency_evidence_decision.get("decision"),
            "unknown-dependency-evidence",
        )
        evidence_state, evidence_reason, durable_next_action = _tool_cache_replay_evidence_state(
            dependency_decision=dependency_decision,
            next_action=next_action,
        )

        sample_count_total += sample_count
        _increment(evidence_state_counts, evidence_state, sample_count)
        _increment(dependency_decision_counts, dependency_decision, sample_count)
        _increment(next_action_counts, durable_next_action, sample_count)
        for blocker in blocker_codes:
            _increment(blocker_counts, blocker, sample_count)
        if dependency_decision == "stable-dependency-evidence":
            stable_dependency_rows += sample_count
        elif dependency_decision == "stale-risk-blocker":
            stale_dependency_rows += sample_count
        elif dependency_decision == "unsafe-dependency-evidence":
            unsafe_dependency_rows += sample_count
        elif dependency_decision == "missing-dependency-evidence":
            missing_dependency_rows += sample_count
        else:
            unknown_dependency_rows += sample_count
        if "unsafe-tool-calls-without-invalidation" in blocker_codes:
            unsafe_tool_call_blocker_rows += sample_count

        rows.append(
            {
                "schema": REPLAY_TOOL_REPLAY_EVIDENCE_ROW_SCHEMA,
                "rank": 0,
                "provider_family": "openai",
                "source_surface": public_label(cohort.get("source_surface"), "unknown"),
                "endpoint": public_label(cohort.get("endpoint"), "unknown"),
                "category": public_label(cohort.get("category"), "unknown"),
                "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
                "stream": bool(cohort.get("stream")),
                "has_tools": True,
                "cache_status": public_label(cohort.get("cache_status"), "unknown"),
                "routing_status": public_label(cohort.get("routing_status"), "unknown"),
                "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
                "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
                "sample_count": sample_count,
                "row_count": sample_count,
                "readiness": readiness,
                "reason": reason,
                "evidence_state": evidence_state,
                "evidence_reason": evidence_reason,
                "blocker_codes": blocker_codes,
                "dependency_evidence_status": dependency_evidence_decision["status"],
                "dependency_evidence_decision": dependency_evidence_decision,
                "next_action": durable_next_action,
                "secondary_next_actions": sorted(set(secondary_actions + ["keep-tool-cache-blocked"])),
                "projected_hits": _as_int(cohort.get("projected_hits")),
                "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
                "cache_hit_count": _as_int(cohort.get("cache_hit_count")),
                "observed_hits": _as_int(cohort.get("observed_hits")),
                "exact_hit_count": _as_int(cohort.get("exact_hit_count")),
                "live_repeat_confirmed": bool(cohort.get("live_repeat_confirmed")),
                "observed_hit_proof": bool(cohort.get("observed_hit_proof")),
                "current_status": public_label(cohort.get("current_status"), "unknown"),
                "state": public_label(cohort.get("state"), "unknown"),
                "promotion_decision": public_label(cohort.get("promotion_decision"), "unknown"),
                "promotion_readiness": public_label(cohort.get("promotion_readiness"), "unknown"),
                "policy_decision": public_label(cohort.get("policy_decision"), "unknown"),
                "observed_hit_blocker": public_label(cohort.get("observed_hit_blocker"), "unknown"),
                "reason_codes": [
                    public_label(item, "unknown")
                    for item in cohort.get("reason_codes") or []
                    if public_label(item, "unknown") != "unknown"
                ],
                "file_dependency_status": file_dependency_status,
                "file_dependency_fingerprint_available": bool(cohort.get("file_dependency_fingerprint_available")),
                "local_dependency_fingerprint": _local_dependency_fingerprint_metadata(
                    bool(cohort.get("file_dependency_fingerprint_available")),
                    cohort.get("file_dependency_audit") if isinstance(cohort.get("file_dependency_audit"), dict) else None,
                ),
                "safe_invalidation_evidence": bool(file_dependency_status == "stable"),
                "tools_present_replay_evidence": True,
                "generic_tools_present_blocker_reduced": True,
                "requires_explicit_invalidation_safety_evidence": dependency_decision != "stable-dependency-evidence",
                "tool_cache_replay_enabled": False,
                "streaming_replay_enabled": False,
                "emits_cache_apply_action": False,
                "cache_entries_written": 0,
                "policy_files_written": False,
                "aggregate_only": True,
                "metadata_only": True,
                "privacy": _replayability_privacy(),
            }
        )

    rows.sort(
        key=lambda item: (
            {
                "dependency-gated-review-ready": 5,
                "blocked-stale-dependency-evidence": 4,
                "blocked-unsafe-dependency-evidence": 4,
                "blocked-missing-dependency-evidence": 3,
                "blocked-unknown-dependency-evidence": 2,
            }.get(str(item.get("evidence_state")), 0),
            _as_int(item.get("sample_count")),
            str(item.get("source_surface")),
            str(item.get("endpoint")),
            str(item.get("category")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    evidence_state_breakdown = _breakdown(evidence_state_counts)
    next_action_breakdown = _breakdown(next_action_counts)
    dependency_evidence_burndown = _dependency_evidence_burndown(rows)
    dependency_fingerprint_coverage = _build_dependency_fingerprint_coverage_report(rows, limit=capped_limit)
    review_only_candidates = _build_tool_cache_review_candidates(rows, limit=capped_limit)
    managed_local_replay_previews = _build_tool_cache_managed_local_replay_previews(
        review_only_candidates,
        managed_preview_outcomes=managed_preview_outcomes,
        limit=capped_limit,
    )
    dependency_evidence_classification = _dependency_evidence_classification(rows)
    return {
        "schema": REPLAY_TOOL_REPLAY_EVIDENCE_SCHEMA,
        "status": "ranked" if rows else "no-tool-cache-replay-evidence",
        "read_only": True,
        "wrote_store": False,
        "wrote_local_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "next_action": next_action_breakdown[0]["value"] if next_action_breakdown else "keep-tool-cache-replay-blocked",
        "summary": {
            "tool_cache_replay_evidence_cohort_count": len(rows),
            "sample_count": sample_count_total,
            "affected_rows": sample_count_total,
            "tools_present_rows": sample_count_total,
            "tools_present_replay_evidence_rows": sample_count_total,
            "generic_tools_present_blocker_reduced_rows": sample_count_total,
            "unsafe_tool_call_blocker_rows": unsafe_tool_call_blocker_rows,
            "stable_dependency_evidence_rows": stable_dependency_rows,
            "stale_dependency_evidence_rows": stale_dependency_rows,
            "missing_dependency_evidence_rows": missing_dependency_rows,
            "unsafe_dependency_evidence_rows": unsafe_dependency_rows,
            "unknown_dependency_evidence_rows": unknown_dependency_rows,
            "file_dependency_fingerprint_available_rows": dependency_fingerprint_coverage["summary"][
                "file_dependency_fingerprint_available_rows"
            ],
            "file_dependency_fingerprint_missing_rows": dependency_fingerprint_coverage["summary"][
                "file_dependency_fingerprint_missing_rows"
            ],
            "safe_invalidation_evidence_rows": dependency_fingerprint_coverage["summary"][
                "safe_invalidation_evidence_rows"
            ],
            "review_only_candidate_count": review_only_candidates["summary"]["review_only_candidate_count"],
            "review_ready_rows": review_only_candidates["summary"]["review_ready_rows"],
            "review_only_gated_rows": review_only_candidates["summary"]["review_only_gated_rows"],
            "stageable_after_review_rows": review_only_candidates["summary"]["stageable_after_review_rows"],
            "managed_local_replay_preview_count": managed_local_replay_previews["summary"]["preview_count"],
            "review_ready_preview_rows": managed_local_replay_previews["summary"]["review_ready_preview_rows"],
            "managed_preview_agreement_rows": managed_local_replay_previews["summary"]["managed_preview_agreement_rows"],
            "managed_preview_missing_rows": managed_local_replay_previews["summary"]["managed_preview_missing_rows"],
            "dependency_fingerprint_coverage_decision": dependency_fingerprint_coverage["coverage_decision"],
            "dependency_fingerprint_top_missing_or_blocked_reason": dependency_fingerprint_coverage["summary"][
                "top_missing_or_blocked_reason"
            ],
            "dependency_evidence_decision_count": len(dependency_decision_counts),
            "top_evidence_state": evidence_state_breakdown[0]["value"] if evidence_state_breakdown else None,
            "top_next_action": next_action_breakdown[0]["value"] if next_action_breakdown else None,
            "top_blocker_code": _breakdown(blocker_counts)[0]["value"] if blocker_counts else None,
            "cache_apply_action_count": 0,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
        },
        "evidence_state_breakdown": evidence_state_breakdown,
        "dependency_evidence_decision_breakdown": _breakdown(dependency_decision_counts),
        "dependency_evidence_burndown": dependency_evidence_burndown,
        "dependency_evidence_classification": dependency_evidence_classification,
        "dependency_fingerprint_coverage": dependency_fingerprint_coverage,
        "review_only_candidates": review_only_candidates,
        "managed_local_replay_previews": managed_local_replay_previews,
        "next_action_breakdown": next_action_breakdown,
        "blocker_breakdown": _breakdown(blocker_counts),
        "cohorts": rows[:capped_limit],
        "acceptance": {
            "has_ranked_tool_cache_replay_evidence": bool(rows),
            "reports_tools_present_replay_evidence": all(bool(row.get("tools_present_replay_evidence")) for row in rows[:capped_limit]),
            "reduces_generic_tools_present_blocker": all(
                bool(row.get("generic_tools_present_blocker_reduced")) for row in rows[:capped_limit]
            ),
            "reports_dependency_evidence_decisions": all(
                isinstance(row.get("dependency_evidence_decision"), dict)
                and bool(row.get("dependency_evidence_decision", {}).get("decision"))
                for row in rows[:capped_limit]
            ),
            "reports_dependency_evidence_burndown": bool(dependency_evidence_burndown)
            and all(_as_int(item.get("row_count")) > 0 for item in dependency_evidence_burndown),
            "reports_dependency_fingerprint_coverage_after_capture": bool(
                dependency_fingerprint_coverage.get("acceptance", {}).get(
                    "reports_dependency_fingerprint_coverage_after_capture"
                )
            ),
            "reports_narrow_no_safe_invalidation_reason": bool(
                dependency_fingerprint_coverage.get("acceptance", {}).get(
                    "reports_narrow_no_safe_invalidation_reason"
                )
            ),
            "promotes_stable_dependency_evidence_to_review_only_candidates": bool(
                review_only_candidates.get("acceptance", {}).get(
                    "stable_dependency_evidence_emits_review_only_candidate"
                )
            ),
            "promotes_stable_dependency_evidence_to_managed_local_replay_previews": bool(
                managed_local_replay_previews.get("acceptance", {}).get("emits_managed_local_replay_previews")
            ),
            "stable_with_proof_emits_review_ready_replay_preview": bool(
                managed_local_replay_previews.get("acceptance", {}).get(
                    "stable_with_proof_emits_review_ready_preview"
                )
            ),
            "review_only_candidates_do_not_allow_savings_floor_without_replay_proof": bool(
                review_only_candidates.get("acceptance", {}).get(
                    "does_not_allow_savings_floor_without_replay_proof"
                )
            ),
            "review_only_candidates_require_live_repeat_or_observed_hit_proof": bool(
                review_only_candidates.get("acceptance", {}).get("requires_live_repeat_or_observed_hit_proof")
            ),
            "streaming_candidates_do_not_become_review_ready": bool(
                review_only_candidates.get("acceptance", {}).get(
                    "streaming_candidates_do_not_become_review_ready"
                )
            ),
            "stable_without_live_repeat_or_observed_hit_is_noop": bool(
                review_only_candidates.get("acceptance", {}).get("stable_without_live_repeat_or_observed_hit_is_noop")
            ),
            "stable_without_live_repeat_or_observed_hit_preview_is_noop": bool(
                managed_local_replay_previews.get("acceptance", {}).get(
                    "stable_without_live_repeat_or_observed_hit_is_noop"
                )
            ),
            "streaming_shapes_do_not_become_review_ready_previews": bool(
                managed_local_replay_previews.get("acceptance", {}).get(
                    "streaming_shapes_do_not_become_review_ready_previews"
                )
            ),
            "review_ready_previews_require_stable_dependency_and_proof": bool(
                managed_local_replay_previews.get("acceptance", {}).get(
                    "review_ready_previews_require_stable_dependency_and_proof"
                )
            ),
            "retired_no_repeat_emits_noop": bool(
                review_only_candidates.get("acceptance", {}).get("retired_no_repeat_emits_noop")
            ),
            "distinguishes_missing_stable_and_stale_dependency_evidence": bool(rows)
            and set(DEPENDENCY_EVIDENCE_CLASSES).issubset(
                set(dependency_evidence_classification["supported_evidence_classes"])
            )
            and all(
                public_label(
                    row.get("dependency_evidence_decision", {}).get("evidence_class"),
                    "unknown-dependency-evidence",
                )
                in set(DEPENDENCY_EVIDENCE_CLASSES) | {"unknown-dependency-evidence", "not-required"}
                for row in rows
                if isinstance(row.get("dependency_evidence_decision"), dict)
            ),
            "distinguishes_missing_stable_stale_and_unsafe_dependency_evidence": bool(rows)
            and set(DEPENDENCY_EVIDENCE_CLASSES).issubset(
                set(dependency_evidence_classification["supported_evidence_classes"])
            ),
            "distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence": bool(rows)
            and set(DEPENDENCY_EVIDENCE_CLASSES).issubset(
                set(dependency_evidence_classification["classification_buckets"])
            )
            and bool(dependency_evidence_classification["all_rows_classified_into_supported_evidence_classes"]),
            "stable_dependency_evidence_does_not_activate_replay": all(
                row.get("next_action") == "rank-safe-tool-cache-replay-readiness"
                and not bool(row.get("tool_cache_replay_enabled"))
                for row in rows
                if row.get("dependency_evidence_decision", {}).get("decision") == "stable-dependency-evidence"
            ),
            "unsafe_or_missing_dependency_keeps_tool_replay_blocked": all(
                not bool(row.get("tool_cache_replay_enabled"))
                for row in rows
                if row.get("dependency_evidence_decision", {}).get("decision")
                in {
                    "stale-risk-blocker",
                    "unsafe-dependency-evidence",
                    "missing-dependency-evidence",
                    "unknown-dependency-evidence",
                }
            ),
            "unsafe_or_missing_dependency_previews_emit_no_apply_actions": bool(
                managed_local_replay_previews.get("acceptance", {}).get(
                    "unsafe_or_missing_dependency_emits_no_apply_actions"
                )
            ),
            "managed_local_replay_previews_are_review_only": bool(
                managed_local_replay_previews.get("acceptance", {}).get(
                    "managed_preview_agreement_is_review_only"
                )
            ),
            "emits_no_cache_apply_actions": True,
            "tool_and_streaming_replay_remain_disabled": all(
                not bool(row.get("tool_cache_replay_enabled")) and not bool(row.get("streaming_replay_enabled"))
                for row in rows
            ),
            "no_cache_entries_written": True,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def build_request_shape_cache_replayability_dry_run(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 25,
    handled_policy_rules: list[dict[str, Any]] | None = None,
    managed_preview_outcomes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay_rows = [
        row
        for row in rollups
        if isinstance(row, dict)
        and (
            "replayability" in {str(item) for item in row.get("candidate_work_classes") or []}
            or "cache_replay" in {str(item) for item in row.get("candidate_families") or []}
            or "cache_blocker" in {str(item) for item in row.get("candidate_families") or []}
        )
    ]
    cohorts: list[dict[str, Any]] = []
    readiness_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    replay_ready_rows = 0
    skipped_rows = 0
    projected_hits = 0
    projected_savings = 0.0
    remaining_projected_hits = 0
    remaining_projected_savings = 0.0
    handled_projected_hits = 0
    handled_projected_savings = 0.0
    handled_rules = handled_policy_rules or []
    gated_too_small_count = 0
    gated_too_small_rows = 0
    gated_too_small_projected_hits = 0
    gated_too_small_projected_savings = 0.0
    live_repeat_confirmed_count = 0
    savings_floor_met_count = 0

    for row in replay_rows:
        decision = _shape_replayability_decision(row)
        row_count = _as_int(row.get("row_count") or row.get("count"))
        cost = _as_float(row.get("cost_est_usd"))
        hits = _as_int(decision.get("projected_hits"))
        saved = 0.0
        if hits and row_count:
            avg_cost = cost / row_count
            saved = max(0.0, cost - avg_cost)
        readiness = str(decision["readiness"])
        reason = str(decision["reason"])
        cache_hit_count = _as_int(row.get("cache_hit_count"))
        readiness_gate = (
            _cache_replay_readiness_gate(
                row_count=row_count,
                projected_hits=hits,
                projected_savings_usd=saved,
                cache_hit_count=cache_hit_count,
            )
            if readiness == "replay-ready"
            else None
        )
        _increment(readiness_counts, readiness)
        _increment(reason_counts, reason)
        for blocker in decision.get("blockers") or []:
            _increment(blocker_counts, blocker)
        if readiness == "replay-ready":
            replay_ready_rows += row_count
            projected_hits += hits
            projected_savings += saved
        else:
            skipped_rows += row_count
        cohort = {
            "schema": "tokenclaw.request_shape_cache_replayability_cohort.v1",
            "readiness": readiness,
            "reason": reason,
            "blockers": decision.get("blockers") or [],
            "provider_family": row.get("provider_family"),
            "source_surface": row.get("source_surface"),
            "endpoint": row.get("endpoint"),
            "category": row.get("category"),
            "workflow_phase": row.get("workflow_phase"),
            "stream": bool(row.get("stream")),
            "has_tools": bool(row.get("has_tools")),
            "cache_status": row.get("cache_status"),
            "routing_status": row.get("routing_status"),
            "file_dependency_status": row.get("file_dependency_status"),
            "file_dependency_fingerprint_available": bool(row.get("file_dependency_fingerprint_available")),
            "file_dependency_audit": row.get("file_dependency_audit")
            if isinstance(row.get("file_dependency_audit"), dict)
            else None,
            "text_bucket": row.get("text_bucket"),
            "token_bucket": row.get("token_bucket"),
            "row_count": row_count,
            "cache_hit_count": cache_hit_count,
            "projected_hits": hits,
            "projected_savings_usd": round(saved, 6),
            "live_repeat_confirmed": bool(
                isinstance(readiness_gate, dict) and readiness_gate.get("live_repeat_confirmed")
            ),
            "readiness_gate": readiness_gate,
            "aggregate_only": True,
            "privacy": _replayability_privacy(),
        }
        handled = _cache_replay_handled_match(cohort, handled_rules) if readiness == "replay-ready" else None
        if handled:
            cohort["handled_by_local_policy"] = True
            cohort["handled_local_policy"] = handled
            cohort["remaining_replay_ready"] = False
            cohort["next_action"] = "already-handled-by-local-cache-policy"
            handled_projected_hits += hits
            handled_projected_savings += saved
        else:
            cohort["handled_by_local_policy"] = False
            gate_allows_stage = bool(readiness_gate.get("stage_allowed")) if isinstance(readiness_gate, dict) else False
            cohort["remaining_replay_ready"] = readiness == "replay-ready" and gate_allows_stage
            if readiness == "replay-ready":
                cohort["next_action"] = (
                    "stage-cache-replay-canary" if gate_allows_stage else "no-op-too-small-without-live-repeat"
                )
                if gate_allows_stage:
                    remaining_projected_hits += hits
                    remaining_projected_savings += saved
                else:
                    gated_too_small_count += 1
                    gated_too_small_rows += row_count
                    gated_too_small_projected_hits += hits
                    gated_too_small_projected_savings += saved
            if isinstance(readiness_gate, dict):
                if readiness_gate.get("live_repeat_confirmed"):
                    live_repeat_confirmed_count += 1
                if readiness_gate.get("savings_floor_met") or readiness_gate.get("row_floor_met"):
                    savings_floor_met_count += 1
        cohorts.append(cohort)

    cohorts.sort(
        key=lambda item: (
            item.get("readiness") == "replay-ready" and not bool(item.get("handled_by_local_policy")),
            item.get("readiness") == "replay-ready",
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("projected_hits")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(cohorts, start=1):
        row["rank"] = rank
    remaining_replay_ready = [
        row
        for row in cohorts
        if row.get("readiness") == "replay-ready"
        and not bool(row.get("handled_by_local_policy"))
        and bool(row.get("remaining_replay_ready"))
    ]
    handled_replay_ready = [
        row
        for row in cohorts
        if row.get("readiness") == "replay-ready" and bool(row.get("handled_by_local_policy"))
    ]
    for rank, row in enumerate(remaining_replay_ready, start=1):
        row["remaining_rank"] = rank
    for rank, row in enumerate(handled_replay_ready, start=1):
        row["handled_rank"] = rank

    top_blocker = None
    blocker_breakdown = _breakdown(blocker_counts)
    if blocker_breakdown:
        top_blocker = blocker_breakdown[0]["value"]
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    invalidation_evidence = build_request_shape_cache_invalidation_evidence_report(
        cohorts,
        limit=capped_limit,
    )
    skipped_openai_blockers = build_request_shape_skipped_openai_cache_replay_blockers_report(
        cohorts,
        limit=capped_limit,
    )
    tool_replay_evidence = build_request_shape_tool_cache_replay_evidence_report(
        cohorts,
        limit=capped_limit,
        managed_preview_outcomes=managed_preview_outcomes,
    )
    return {
        "schema": REPLAYABILITY_DRY_RUN_SCHEMA,
        "status": "ranked" if cohorts else "no-replayability-cohorts",
        "summary": {
            "cohort_count": len(cohorts),
            "rows_considered": sum(_as_int(row.get("row_count") or row.get("count")) for row in replay_rows),
            "replay_ready_cohort_count": readiness_counts.get("replay-ready", 0),
            "replay_ready_rows": replay_ready_rows,
            "remaining_replay_ready_cohort_count": len(remaining_replay_ready),
            "remaining_replay_ready_rows": sum(_as_int(row.get("row_count")) for row in remaining_replay_ready),
            "handled_replay_ready_cohort_count": len(handled_replay_ready),
            "handled_replay_ready_rows": sum(_as_int(row.get("row_count")) for row in handled_replay_ready),
            "skipped_cohort_count": readiness_counts.get("skipped", 0),
            "skipped_rows": skipped_rows,
            "projected_hits": projected_hits,
            "projected_savings_usd": round(projected_savings, 6),
            "remaining_projected_hits": remaining_projected_hits,
            "remaining_projected_savings_usd": round(remaining_projected_savings, 6),
            "handled_projected_hits": handled_projected_hits,
            "handled_projected_savings_usd": round(handled_projected_savings, 6),
            "gated_too_small_replay_ready_cohort_count": gated_too_small_count,
            "gated_too_small_replay_ready_rows": gated_too_small_rows,
            "gated_too_small_projected_hits": gated_too_small_projected_hits,
            "gated_too_small_projected_savings_usd": round(gated_too_small_projected_savings, 6),
            "live_repeat_confirmed_replay_ready_cohort_count": live_repeat_confirmed_count,
            "savings_or_repeat_floor_met_replay_ready_cohort_count": savings_floor_met_count,
            "minimum_replay_stage_rows": DEFAULT_CACHE_REPLAY_MIN_STAGE_ROWS,
            "minimum_replay_stage_projected_hits": DEFAULT_CACHE_REPLAY_MIN_STAGE_PROJECTED_HITS,
            "minimum_replay_stage_savings_usd": DEFAULT_CACHE_REPLAY_MIN_STAGE_SAVINGS_USD,
            "top_blocker_code": top_blocker,
            "top_remaining_replay_ready_rank": _as_int(remaining_replay_ready[0].get("rank")) if remaining_replay_ready else 0,
            "top_remaining_replay_ready_projected_hits": _as_int(remaining_replay_ready[0].get("projected_hits")) if remaining_replay_ready else 0,
            "top_remaining_replay_ready_projected_savings_usd": (
                remaining_replay_ready[0].get("projected_savings_usd") if remaining_replay_ready else 0.0
            ),
            "dependency_fingerprint_coverage_decision": tool_replay_evidence["summary"].get(
                "dependency_fingerprint_coverage_decision"
            ),
            "dependency_fingerprint_top_missing_or_blocked_reason": tool_replay_evidence["summary"].get(
                "dependency_fingerprint_top_missing_or_blocked_reason"
            ),
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "readiness_breakdown": _breakdown(readiness_counts),
        "skipped_reason_breakdown": _breakdown(reason_counts),
        "blocker_breakdown": blocker_breakdown,
        "cache_invalidation_evidence": invalidation_evidence,
        "skipped_openai_blockers": skipped_openai_blockers,
        "tool_replay_evidence": tool_replay_evidence,
        "remaining_replay_ready_cohorts": remaining_replay_ready[:capped_limit],
        "handled_replay_ready_cohorts": handled_replay_ready[:capped_limit],
        "handled_policy_summary": {
            "schema": "tokenclaw.request_shape_cache_replay_handled_policy_summary.v1",
            "handled_rule_count": len(handled_rules),
            "matched_handled_cohort_count": len(handled_replay_ready),
            "policy_paths_included": False,
            "rule_ids_included": False,
            "cohort_ids_included": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "acceptance": {
            "emits_durable_invalidation_evidence": invalidation_evidence["schema"] == REPLAY_INVALIDATION_EVIDENCE_SCHEMA,
            "emits_skipped_openai_blocker_ranking": skipped_openai_blockers["schema"] == REPLAY_SKIPPED_OPENAI_BLOCKERS_SCHEMA,
            "emits_tool_replay_evidence": tool_replay_evidence["schema"] == REPLAY_TOOL_REPLAY_EVIDENCE_SCHEMA,
            "ranks_remaining_replay_ready_cohorts": all(
                _as_int(row.get("remaining_rank")) > 0 for row in remaining_replay_ready[:capped_limit]
            ),
            "marks_already_handled_replay_ready_cohorts": all(
                bool(row.get("handled_by_local_policy")) and isinstance(row.get("handled_local_policy"), dict)
                for row in handled_replay_ready[:capped_limit]
            ),
            "reports_remaining_projected_hits_and_savings": remaining_projected_hits >= 0
            and remaining_projected_savings >= 0.0,
            "gates_tiny_replay_ready_cohorts_without_live_repeat": all(
                row.get("next_action") == "no-op-too-small-without-live-repeat"
                and not bool(row.get("remaining_replay_ready"))
                for row in cohorts
                if isinstance(row.get("readiness_gate"), dict)
                and row["readiness_gate"].get("gate_status") == "replay-ready-but-too-small"
            ),
            "reports_live_repeat_and_savings_floor_fields": all(
                isinstance(row.get("readiness_gate"), dict)
                and "live_repeat_confirmed" in row["readiness_gate"]
                and "minimum_projected_savings_usd" in row["readiness_gate"]
                for row in cohorts
                if row.get("readiness") == "replay-ready"
            ),
            "has_ranked_blocker_cohorts": bool(
                invalidation_evidence.get("acceptance", {}).get("has_ranked_blocker_cohorts")
            ),
            "has_ranked_skipped_openai_cohorts": bool(
                skipped_openai_blockers.get("acceptance", {}).get("has_ranked_skipped_openai_cohorts")
            ),
            "has_ranked_tool_cache_replay_evidence": bool(
                tool_replay_evidence.get("acceptance", {}).get("has_ranked_tool_cache_replay_evidence")
            ),
            "reduces_generic_tools_present_blocker": bool(
                tool_replay_evidence.get("acceptance", {}).get("reduces_generic_tools_present_blocker")
            ),
            "reports_dependency_fingerprint_coverage_after_capture": bool(
                tool_replay_evidence.get("acceptance", {}).get(
                    "reports_dependency_fingerprint_coverage_after_capture"
                )
            ),
            "reports_narrow_no_safe_invalidation_reason": bool(
                tool_replay_evidence.get("acceptance", {}).get("reports_narrow_no_safe_invalidation_reason")
            ),
            "tool_and_streaming_replay_remain_disabled": bool(
                invalidation_evidence.get("acceptance", {}).get("tool_and_streaming_replay_remain_disabled")
            ),
            "has_local_file_backed_policy_compatibility": bool(
                invalidation_evidence.get("acceptance", {}).get("has_local_file_backed_policy_compatibility")
            ),
            "no_cache_entries_written": True,
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "cohorts": cohorts[:capped_limit],
        "privacy": _replayability_privacy(),
    }


def _cache_replay_blocker_classification(blocker: str) -> tuple[str, str, str, str]:
    code = public_label(blocker, "unknown")
    if code in {"invalidation-evidence-missing", "unsafe-dependency-evidence", "unsafe-tool-calls-without-invalidation"}:
        return (
            "collect-invalidation-evidence",
            "collect-cache-invalidation-evidence",
            "blocked",
            "cache",
        )
    if code in {"tools-present", "tool-call-cache-disabled"}:
        return (
            "keep-tool-cache-disabled",
            "keep-tool-cache-disabled",
            "blocked",
            "cache",
        )
    if code in {"streaming-replay-not-supported", "unsupported-streaming-shape"}:
        return (
            "streaming-replay-support-needed",
            "design-streaming-cache-replay-support",
            "blocked",
            "cache",
        )
    if code == "insufficient-repeat-evidence":
        return (
            "insufficient-repeat-evidence",
            "collect-more-repeat-evidence",
            "needs-evidence",
            "cache",
        )
    return (
        "unsupported-safety-shape",
        "keep-cache-replay-noop",
        "blocked",
        "cache",
    )


def build_request_shape_cache_replay_blocker_classification_report(
    cache_replayability_dry_run: dict[str, Any],
    *,
    limit: int = 25,
) -> dict[str, Any]:
    cohorts = [
        cohort
        for cohort in cache_replayability_dry_run.get("cohorts") or []
        if isinstance(cohort, dict) and cohort.get("readiness") == "skipped"
    ]
    class_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for cohort in cohorts:
        row_count = _as_int(cohort.get("row_count"))
        blockers = [
            public_label(item, "unknown")
            for item in cohort.get("blockers") or []
            if public_label(item, "unknown") != "unknown"
        ]
        reason = public_label(cohort.get("reason"), "unknown")
        if not blockers and reason != "unknown":
            blockers = [reason]
        classes: dict[str, dict[str, Any]] = {}
        for blocker in blockers:
            blocker_class, next_action, status, family = _cache_replay_blocker_classification(blocker)
            _increment(blocker_counts, blocker, row_count)
            _increment(class_counts, blocker_class, row_count)
            _increment(next_action_counts, next_action, row_count)
            _increment(status_counts, status, row_count)
            _increment(family_counts, family, row_count)
            classified = classes.setdefault(
                blocker_class,
                {
                    "class": blocker_class,
                    "next_action": next_action,
                    "status": status,
                    "local_action_family": family,
                    "blockers": [],
                },
            )
            classified["blockers"].append(blocker)

        if not classes:
            blocker_class, next_action, status, family = _cache_replay_blocker_classification(reason)
            _increment(class_counts, blocker_class, row_count)
            _increment(next_action_counts, next_action, row_count)
            _increment(status_counts, status, row_count)
            _increment(family_counts, family, row_count)
            classes[blocker_class] = {
                "class": blocker_class,
                "next_action": next_action,
                "status": status,
                "local_action_family": family,
                "blockers": [reason],
            }

        primary = sorted(
            classes.values(),
            key=lambda item: (
                {
                    "collect-invalidation-evidence": 0,
                    "keep-tool-cache-disabled": 1,
                    "streaming-replay-support-needed": 2,
                    "insufficient-repeat-evidence": 3,
                    "unsupported-safety-shape": 4,
                }.get(str(item.get("class")), 99),
                str(item.get("class")),
            ),
        )[0]
        classified_blockers = [
            {
                **item,
                "blockers": sorted(set(public_label(blocker, "unknown") for blocker in item.get("blockers") or [])),
                "emits_cache_apply_action": False,
                "requires_explicit_invalidation_safety_evidence": item.get("class") == "collect-invalidation-evidence",
            }
            for item in sorted(classes.values(), key=lambda item: str(item.get("class")))
        ]
        rows.append(
            {
                "schema": REPLAY_BLOCKER_CLASSIFICATION_ROW_SCHEMA,
                "rank": 0,
                "provider_family": public_label(cohort.get("provider_family"), "unknown"),
                "source_surface": public_label(cohort.get("source_surface"), "unknown"),
                "endpoint": public_label(cohort.get("endpoint"), "unknown"),
                "category": public_label(cohort.get("category"), "unknown"),
                "workflow_phase": public_label(cohort.get("workflow_phase"), "unknown"),
                "stream": bool(cohort.get("stream")),
                "has_tools": bool(cohort.get("has_tools")),
                "cache_status": public_label(cohort.get("cache_status"), "unknown"),
                "routing_status": public_label(cohort.get("routing_status"), "unknown"),
                "text_bucket": public_label(cohort.get("text_bucket"), "unknown"),
                "token_bucket": public_label(cohort.get("token_bucket"), "unknown"),
                "row_count": row_count,
                "projected_hits": _as_int(cohort.get("projected_hits")),
                "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
                "readiness": "blocked",
                "reason": reason,
                "blocker_codes": sorted(set(blockers)),
                "blocker_class": primary.get("class"),
                "next_action": primary.get("next_action"),
                "local_action_family": primary.get("local_action_family"),
                "classified_blockers": classified_blockers,
                "emits_cache_apply_action": False,
                "requires_explicit_invalidation_safety_evidence": any(
                    item.get("requires_explicit_invalidation_safety_evidence")
                    for item in classified_blockers
                ),
                "aggregate_only": True,
                "privacy": _replayability_privacy(),
            }
        )

    rows.sort(
        key=lambda item: (
            {
                "collect-invalidation-evidence": 5,
                "keep-tool-cache-disabled": 4,
                "streaming-replay-support-needed": 3,
                "insufficient-repeat-evidence": 2,
                "unsupported-safety-shape": 1,
            }.get(str(item.get("blocker_class")), 0),
            _as_int(item.get("row_count")),
            str(item.get("endpoint")),
            str(item.get("category")),
        ),
        reverse=True,
    )
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    action_rows = [row for row in rows if bool(row.get("emits_cache_apply_action"))]
    unsafe_apply_rows = [
        row
        for row in action_rows
        if not bool(row.get("requires_explicit_invalidation_safety_evidence"))
    ]
    class_breakdown = _breakdown(class_counts)
    top_class = class_breakdown[0]["value"] if class_breakdown else None
    next_action_breakdown = _breakdown(next_action_counts)
    top_next_action = next_action_breakdown[0]["value"] if next_action_breakdown else None
    return {
        "schema": REPLAY_BLOCKER_CLASSIFICATION_SCHEMA,
        "status": "classified" if rows else "no-skipped-cache-replay-cohorts",
        "summary": {
            "skipped_cohort_count": len(rows),
            "skipped_rows": sum(_as_int(row.get("row_count")) for row in rows),
            "classified_blocker_count": sum(
                len(row.get("classified_blockers") or [])
                for row in rows
            ),
            "collect_invalidation_evidence_rows": class_counts.get("collect-invalidation-evidence", 0),
            "keep_tool_cache_disabled_rows": class_counts.get("keep-tool-cache-disabled", 0),
            "streaming_replay_support_needed_rows": class_counts.get("streaming-replay-support-needed", 0),
            "insufficient_repeat_evidence_rows": class_counts.get("insufficient-repeat-evidence", 0),
            "unsupported_safety_shape_rows": class_counts.get("unsupported-safety-shape", 0),
            "top_blocker_class": top_class,
            "top_next_action": top_next_action,
            "cache_apply_action_count": len(action_rows),
            "unsafe_cache_apply_action_count": len(unsafe_apply_rows),
            "provider_calls_made": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "class_breakdown": class_breakdown,
        "next_action_breakdown": next_action_breakdown,
        "status_breakdown": _breakdown(status_counts),
        "local_action_family_breakdown": _breakdown(family_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "classifications": rows[:capped_limit],
        "acceptance": {
            "has_tool_blocker_class": class_counts.get("keep-tool-cache-disabled", 0) > 0,
            "has_invalidation_evidence_class": class_counts.get("collect-invalidation-evidence", 0) > 0,
            "has_streaming_support_class": class_counts.get("streaming-replay-support-needed", 0) > 0,
            "has_insufficient_repeat_class": class_counts.get("insufficient-repeat-evidence", 0) > 0,
            "has_unsupported_safety_shape_class": class_counts.get("unsupported-safety-shape", 0) > 0,
            "no_cache_apply_without_invalidation_safety_evidence": len(unsafe_apply_rows) == 0,
            "emits_no_cache_apply_actions": len(action_rows) == 0,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _replayability_privacy(),
    }


def _candidate_work_classes(
    *,
    row_count: int,
    text_bucket: str,
    token_bucket: str,
    candidate_families: list[str],
    blocker_codes: list[str],
    routing_status: str,
    observed_savings: float,
) -> list[str]:
    classes: set[str] = set()
    repeated_large_context = (
        row_count >= REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES and text_bucket in REPEATED_CONTEXT_TEXT_BUCKETS
    )
    token_heavy_context = token_bucket in LARGE_CONTEXT_TOKEN_BUCKETS
    if repeated_large_context or (row_count >= REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES and token_heavy_context):
        classes.add("repeated_context")
        classes.add("crunch")
    if any(family in {"cache_replay", "cache_blocker"} for family in candidate_families) or any(
        code
        in {
            "unsupported-streaming-shape",
            "tool-call-cache-disabled",
            "semantic-cache-disabled",
            "exact-cache-miss",
            "cache-skipped",
        }
        for code in blocker_codes
    ):
        classes.add("replayability")
    if "routing_candidate" in candidate_families or routing_status == "passthrough":
        classes.add("routing")
    if "routing_evidence" in candidate_families or observed_savings > 0:
        classes.add("routing_evidence")
    return sorted(classes or {"observability"})


def _shape_next_action(classes: list[str], blockers: list[str]) -> str:
    class_set = set(classes)
    blocker_set = set(blockers)
    if "repeated_context" in class_set and "replayability" in class_set:
        return "rank-repeated-context-replayability-cohort"
    if "repeated_context" in class_set and "crunch" in class_set:
        return "rank-repeated-context-crunch-dry-run"
    if "routing" in class_set:
        return "stage-routing-lifecycle-evidence"
    if blocker_set:
        return "classify-request-shape-blocker"
    return "keep-observability-only"


def _shape_local_action_family(next_action: str, classes: list[str]) -> str:
    if next_action in {"stage-repeated-context-crunch-canary", "measure-repeated-context-crunch-canary-impact"}:
        return "crunch"
    if next_action == "stage-cache-replay-canary":
        return "cache"
    if next_action in {"collect-thinking-routing-lifecycle-evidence", "stage-routing-lifecycle-evidence"}:
        return "routing"
    if next_action == "rank-repeated-context-replayability-cohort":
        return "cache"
    if next_action == "rank-repeated-context-crunch-dry-run":
        return "crunch"
    if next_action == "stage-routing-lifecycle-evidence":
        return "routing"
    if "replayability" in classes:
        return "cache"
    if "crunch" in classes:
        return "crunch"
    if "routing" in classes:
        return "routing"
    return "cohort-ranking"


def _shape_follow_up_privacy() -> dict[str, Any]:
    privacy = _replayability_privacy()
    privacy["policy_files_written"] = False
    return privacy


def _routing_downgrade_privacy() -> dict[str, Any]:
    privacy = _shape_follow_up_privacy()
    privacy.update(
        {
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_response_bodies_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_written": False,
        }
    )
    return privacy


def _routing_downgrade_target(
    *,
    provider: str,
    requested_model_family: str,
) -> tuple[str | None, str | None, str]:
    provider = public_label(provider, "unknown")
    requested = public_label(requested_model_family, "unknown")
    if provider == "anthropic":
        if "opus" in requested:
            return "claude-sonnet-4.5", "claude-sonnet", "opus-to-sonnet-downgrade-drill"
        if "sonnet" in requested:
            return "claude-haiku-4.5", "claude-haiku", "sonnet-to-haiku-downgrade-drill"
        return None, None, "no-cheaper-anthropic-routing-target"
    if provider == "openai":
        if requested in {"gpt-5", "gpt-5.3", "gpt-5.3-codex", "gpt-5.2", "gpt-5.2-codex", "gpt-5-codex"}:
            return "gpt-5-mini", "gpt-5-mini", "gpt5-to-mini-downgrade-drill"
        if requested in {"gpt-5.5", "gpt-5.4"}:
            return "gpt-5.4-mini", "gpt-5.4-mini", "gpt5-large-to-mini-downgrade-drill"
        if requested == "gpt-4.1":
            return "gpt-4.1-mini", "gpt-4.1-mini", "gpt41-to-mini-downgrade-drill"
        if requested == "gpt-4o":
            return "gpt-4o-mini", "gpt-4o-mini", "gpt4o-to-mini-downgrade-drill"
        return None, None, "no-cheaper-openai-routing-target"
    return None, None, "unknown-provider-routing-target"


def _routing_token_cost_usd(tokens: int, *, provider: str, model: str, field: str) -> float:
    if tokens <= 0:
        return 0.0
    basis = pricing_basis(model, provider)
    price = _as_float(basis.get(field))
    if price <= 0:
        return 0.0
    return (tokens / 1_000_000.0) * price


def _routing_downgrade_projection(
    row: dict[str, Any],
    *,
    provider: str,
    requested_model: str,
    target_model: str,
    sample_count: int,
) -> dict[str, Any]:
    input_tokens = _as_int(row.get("successful_input_tokens") or row.get("input_tokens"))
    output_tokens = _as_int(row.get("output_tokens"))
    requested_cost = _routing_token_cost_usd(
        input_tokens,
        provider=provider,
        model=requested_model,
        field="input_usd_per_million",
    ) + _routing_token_cost_usd(
        output_tokens,
        provider=provider,
        model=requested_model,
        field="output_usd_per_million",
    )
    target_cost = _routing_token_cost_usd(
        input_tokens,
        provider=provider,
        model=target_model,
        field="input_usd_per_million",
    ) + _routing_token_cost_usd(
        output_tokens,
        provider=provider,
        model=target_model,
        field="output_usd_per_million",
    )
    projected = max(0.0, requested_cost - target_cost)
    per_call = projected / float(sample_count) if sample_count > 0 else 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "requested_model_cost_usd": round(requested_cost, 8),
        "target_model_cost_usd": round(target_cost, 8),
        "projected_savings_usd": round(projected, 8),
        "projected_savings_per_1000_calls_usd": round(per_call * 1000.0, 8),
    }


def _routing_drill_fingerprint(material: dict[str, Any]) -> str:
    digest = hashlib.sha256(stable_json(material).encode("utf-8")).hexdigest()[:16]
    return f"routing-drill:{digest}"


def _routing_drill_stale(row: dict[str, Any]) -> bool:
    if bool(row.get("stale") or row.get("snapshot_stale")):
        return True
    freshness_state = public_label(row.get("freshness_state") or row.get("freshness_status"), "fresh")
    if freshness_state in {"stale", "snapshot-stale", "rollup-stale"}:
        return True
    freshness = row.get("snapshot_freshness") if isinstance(row.get("snapshot_freshness"), dict) else {}
    return bool(freshness.get("stale"))


def _routing_drill_blockers(
    row: dict[str, Any],
    *,
    target_model: str | None,
    target_reason: str,
    projection: dict[str, Any],
    min_samples: int,
    max_error_rate: float,
    max_retry_rate: float,
) -> list[str]:
    blockers: set[str] = set()
    sample_count = _as_int(row.get("row_count") or row.get("sample_count"))
    error_rate = _as_int(row.get("error_count")) / float(sample_count) if sample_count > 0 else 0.0
    retry_rate = _as_int(row.get("retry_count")) / float(sample_count) if sample_count > 0 else 0.0
    category = public_label(row.get("category"), "unknown")
    phase = public_label(row.get("workflow_phase"), "unknown")
    existing_blockers = {public_label(item, "unknown") for item in row.get("blocker_codes") or []}
    if not target_model:
        blockers.add(target_reason)
    if _routing_drill_stale(row):
        blockers.add("stale-request-shape-rollup")
    if sample_count < min_samples:
        blockers.add("too-small-routing-drill-sample")
    if _as_float(projection.get("projected_savings_per_1000_calls_usd")) <= 0:
        blockers.add("non-positive-routing-savings-projection")
    thinking_blockers = existing_blockers & {
        "top-level-thinking-blocked",
        "thinking-history-blocked",
        "thinking-routing-guard",
    }
    if thinking_blockers:
        blockers.update(thinking_blockers)
    elif category == "thinking" or phase == "thinking":
        blockers.add("thinking-routing-guard")
    if category in {"code-gen", "coding", "planning"} or phase in {"planning", "code-generation"}:
        blockers.add("high-downgrade-risk-shape")
    if error_rate > max_error_rate:
        blockers.add("elevated-error-rate")
    if retry_rate > max_retry_rate:
        blockers.add("elevated-retry-rate")
    if public_label(row.get("routing_status"), "unknown") != "routed":
        blockers.add("missing-quality-evidence")
    priority = {
        "stale-request-shape-rollup": 0,
        "top-level-thinking-blocked": 1,
        "thinking-history-blocked": 1,
        "thinking-routing-guard": 1,
        "high-downgrade-risk-shape": 2,
        "elevated-error-rate": 3,
        "elevated-retry-rate": 4,
        "too-small-routing-drill-sample": 5,
        "non-positive-routing-savings-projection": 6,
        "no-cheaper-anthropic-routing-target": 7,
        "no-cheaper-openai-routing-target": 7,
        "unknown-provider-routing-target": 7,
        "missing-quality-evidence": 20,
    }
    return sorted(blockers, key=lambda code: (priority.get(code, 10), code))


def build_request_shape_routing_downgrade_drill_report(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 25,
    canary_fraction: float = DEFAULT_ROUTING_DOWNGRADE_DRILL_CANARY_FRACTION,
    holdout_fraction: float = DEFAULT_ROUTING_DOWNGRADE_DRILL_HOLDOUT_FRACTION,
    min_samples: int = DEFAULT_ROUTING_DOWNGRADE_DRILL_MIN_SAMPLES,
    max_error_rate: float = DEFAULT_ROUTING_DOWNGRADE_DRILL_MAX_ERROR_RATE,
    max_retry_rate: float = DEFAULT_ROUTING_DOWNGRADE_DRILL_MAX_RETRY_RATE,
) -> dict[str, Any]:
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    total_samples = 0
    total_projected = 0.0

    for row in rollups:
        if not isinstance(row, dict):
            continue
        classes = {public_label(item, "unknown") for item in row.get("candidate_work_classes") or []}
        families = {public_label(item, "unknown") for item in row.get("candidate_families") or []}
        routing_status = public_label(row.get("routing_status"), "unknown")
        if "routing" not in classes and "routing_candidate" not in families and routing_status not in {"passthrough", "routed"}:
            continue

        provider = public_label(row.get("provider_family"), "unknown")
        requested_family = public_label(row.get("requested_model_family"), "unknown")
        target_model, target_family, target_reason = _routing_downgrade_target(
            provider=provider,
            requested_model_family=requested_family,
        )
        sample_count = _as_int(row.get("row_count") or row.get("sample_count"))
        projection = _routing_downgrade_projection(
            row,
            provider=provider,
            requested_model=requested_family,
            target_model=target_model or requested_family,
            sample_count=sample_count,
        )
        blockers = _routing_drill_blockers(
            row,
            target_model=target_model,
            target_reason=target_reason,
            projection=projection,
            min_samples=min_samples,
            max_error_rate=max_error_rate,
            max_retry_rate=max_retry_rate,
        )
        hard_blockers = {
            "stale-request-shape-rollup",
            "too-small-routing-drill-sample",
            "non-positive-routing-savings-projection",
            "top-level-thinking-blocked",
            "thinking-history-blocked",
            "thinking-routing-guard",
            "high-downgrade-risk-shape",
            "elevated-error-rate",
            "elevated-retry-rate",
            "no-cheaper-anthropic-routing-target",
            "no-cheaper-openai-routing-target",
            "unknown-provider-routing-target",
        }
        status = "blocked" if hard_blockers.intersection(blockers) else "review-ready"
        next_action = (
            "review-routing-downgrade-canary"
            if status == "review-ready"
            else "keep-routing-downgrade-drill-blocked"
        )
        fingerprint_material = {
            "schema": ROUTING_DOWNGRADE_DRILL_ROW_SCHEMA,
            "provider_family": provider,
            "source_surface": public_label(row.get("source_surface"), "unknown"),
            "endpoint": public_label(row.get("endpoint"), "unknown"),
            "requested_model_family": requested_family,
            "target_model_family": target_family,
            "category": public_label(row.get("category"), "unknown"),
            "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
            "stream": bool(row.get("stream")),
            "has_tools": bool(row.get("has_tools")),
            "text_bucket": public_label(row.get("text_bucket"), "unknown"),
            "token_bucket": public_label(row.get("token_bucket"), "unknown"),
        }
        canary_sample_count = int(math.ceil(sample_count * canary_fraction)) if status == "review-ready" else 0
        holdout_sample_count = int(math.ceil(sample_count * holdout_fraction)) if status == "review-ready" else 0
        drill = {
            "schema": ROUTING_DOWNGRADE_DRILL_ROW_SCHEMA,
            "fingerprint": _routing_drill_fingerprint(fingerprint_material),
            "source_evidence_schema": row.get("source_schema") or row.get("schema") or ROLLUP_ROW_SCHEMA,
            **fingerprint_material,
            "candidate_id_included": False,
            "requested_model_family": requested_family,
            "candidate_target_model": target_model,
            "target_model_family": target_family,
            "target_reason": target_reason,
            "routing_status": routing_status,
            "sample_count": sample_count,
            "row_count": sample_count,
            "error_count": _as_int(row.get("error_count")),
            "retry_count": _as_int(row.get("retry_count")),
            "error_rate": round(_as_int(row.get("error_count")) / float(sample_count), 6) if sample_count else 0.0,
            "retry_rate": round(_as_int(row.get("retry_count")) / float(sample_count), 6) if sample_count else 0.0,
            "input_tokens": projection["input_tokens"],
            "output_tokens": projection["output_tokens"],
            "projected_savings_usd": projection["projected_savings_usd"],
            "projected_savings_per_1000_calls_usd": projection["projected_savings_per_1000_calls_usd"],
            "recommended_canary_fraction": round(canary_fraction, 4) if status == "review-ready" else 0.0,
            "recommended_holdout_fraction": round(holdout_fraction, 4) if status == "review-ready" else 0.0,
            "recommended_canary_sample_count": canary_sample_count,
            "recommended_holdout_sample_count": holdout_sample_count,
            "top_blocker_code": blockers[0] if blockers else None,
            "blocker_codes": blockers,
            "status": status,
            "readiness_state": status,
            "recommended_next_action": next_action,
            "local_action_family": "routing",
            "review_only": True,
            "emits_routing_apply_action": False,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "aggregate_only": True,
            "privacy": _routing_downgrade_privacy(),
        }
        rows.append(drill)
        total_samples += sample_count
        if status == "review-ready":
            total_projected += _as_float(drill.get("projected_savings_usd"))
        _increment(status_counts, status, sample_count)
        _increment(target_counts, target_family or target_reason, sample_count)
        _increment(source_counts, drill.get("source_surface"), sample_count)
        for blocker in blockers:
            _increment(blocker_counts, blocker, sample_count)

    rows.sort(
        key=lambda item: (
            item.get("status") == "review-ready",
            _as_float(item.get("projected_savings_per_1000_calls_usd")),
            _as_int(item.get("sample_count")),
            str(item.get("fingerprint")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    ranked = rows[:capped_limit]
    top = ranked[0] if ranked else None
    blocker_breakdown = _breakdown(blocker_counts)
    return {
        "schema": ROUTING_DOWNGRADE_DRILL_SCHEMA,
        "status": "ranked" if ranked else "no-routing-downgrade-drill-candidates",
        "summary": {
            "rows_considered": len([row for row in rollups if isinstance(row, dict)]),
            "candidate_count": len(ranked),
            "review_ready_count": sum(1 for row in ranked if row.get("status") == "review-ready"),
            "blocked_count": sum(1 for row in ranked if row.get("status") == "blocked"),
            "sample_count": total_samples,
            "top_fingerprint": top.get("fingerprint") if top else None,
            "top_next_action": top.get("recommended_next_action") if top else None,
            "top_blocker_code": top.get("top_blocker_code") if top else None,
            "top_projected_savings_per_1000_calls_usd": (
                top.get("projected_savings_per_1000_calls_usd") if top else 0.0
            ),
            "total_projected_savings_usd": round(total_projected, 8),
            "minimum_sample_count": min_samples,
            "default_canary_fraction": round(canary_fraction, 4),
            "default_holdout_fraction": round(holdout_fraction, 4),
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "status_breakdown": _breakdown(status_counts),
        "target_breakdown": _breakdown(target_counts),
        "source_surface_breakdown": _breakdown(source_counts),
        "blocker_breakdown": blocker_breakdown,
        "top_candidate": top,
        "candidates": ranked,
        "acceptance": {
            "has_stable_fingerprints": all(str(row.get("fingerprint", "")).startswith("routing-drill:") for row in ranked),
            "has_sample_counts": all(_as_int(row.get("sample_count")) >= 0 for row in ranked),
            "has_projected_savings_per_1000_calls": all(
                "projected_savings_per_1000_calls_usd" in row for row in ranked
            ),
            "has_canary_and_holdout_sizing": all(
                "recommended_canary_fraction" in row and "recommended_holdout_fraction" in row for row in ranked
            ),
            "emits_no_routing_apply_actions": all(not bool(row.get("emits_routing_apply_action")) for row in ranked),
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _routing_downgrade_privacy(),
    }


def _phase_aware_routing_privacy() -> dict[str, Any]:
    return _routing_downgrade_privacy()


def _phase_aware_routing_policy_coverage(
    *,
    provider: str,
    requested_model_family: str,
    routed_model_family: str,
    candidate_target_model: str | None,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
) -> dict[str, Any]:
    """Read active-routing-policy coverage for a cohort shape (metadata-only).

    Degrades to ``coverage-unknown`` when the local routing policy module is not
    importable so the dry run never depends on routing experiments being wired.
    Intentionally drops the policy decision's ``rule_path`` (a filesystem path)
    to honor the metadata-only privacy contract.
    """
    coverage = {
        "schema": "tokenclaw.request_shape_phase_aware_routing_policy_coverage.v1",
        "coverage_state": "coverage-unknown",
        "policy_decision": "unknown",
        "policy_blocked": False,
        "has_active_preferred_pathway": False,
        "active_policy_target_model_family": None,
        "policy_source": "unknown",
    }
    try:
        from tokenclaw import routing_experiments

        decision = routing_experiments.routing_pathway_policy_decision(
            provider=provider,
            requested_model=requested_model_family,
            current_model=routed_model_family or requested_model_family,
            target_model=candidate_target_model,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
        )
    except Exception:
        return coverage
    if not isinstance(decision, dict):
        return coverage
    preferred = decision.get("preferred_pathway") if isinstance(decision.get("preferred_pathway"), dict) else None
    policy_decision = public_label(decision.get("decision"), "unknown")
    blocked = policy_decision == "blocked" or bool(decision.get("blocklist_match"))
    if blocked:
        coverage_state = "blocked-by-policy"
    elif preferred is not None:
        coverage_state = "covered-by-active-policy"
    else:
        coverage_state = "uncovered"
    active_target = preferred.get("routed_model") if preferred else decision.get("target_model")
    coverage.update(
        {
            "coverage_state": coverage_state,
            "policy_decision": policy_decision,
            "policy_blocked": blocked,
            "has_active_preferred_pathway": preferred is not None,
            "active_policy_target_model_family": (
                _model_family(str(active_target)) if active_target else None
            ),
            "policy_source": public_label(decision.get("policy_source"), "unknown"),
        }
    )
    return coverage


def _phase_aware_routing_blockers(
    row: dict[str, Any],
    *,
    target_model: str | None,
    target_reason: str,
    projection: dict[str, Any],
    coverage: dict[str, Any],
    min_samples: int,
    max_error_rate: float,
    max_retry_rate: float,
    max_fallback_rate: float,
) -> list[str]:
    blockers = set(
        _routing_drill_blockers(
            row,
            target_model=target_model,
            target_reason=target_reason,
            projection=projection,
            min_samples=min_samples,
            max_error_rate=max_error_rate,
            max_retry_rate=max_retry_rate,
        )
    )
    sample_count = _as_int(row.get("row_count") or row.get("sample_count"))
    fallback_rate = _as_int(row.get("fallback_count")) / float(sample_count) if sample_count > 0 else 0.0
    if fallback_rate > max_fallback_rate:
        blockers.add("elevated-fallback-rate")
    if coverage.get("policy_blocked"):
        blockers.add("active-routing-policy-blocklist")
    priority = {
        "stale-request-shape-rollup": 0,
        "active-routing-policy-blocklist": 1,
        "thinking-routing-guard": 2,
        "high-downgrade-risk-shape": 3,
        "elevated-error-rate": 4,
        "elevated-fallback-rate": 5,
        "elevated-retry-rate": 6,
        "too-small-routing-drill-sample": 7,
        "non-positive-routing-savings-projection": 8,
        "no-cheaper-anthropic-routing-target": 9,
        "no-cheaper-openai-routing-target": 9,
        "unknown-provider-routing-target": 9,
        "missing-quality-evidence": 20,
    }
    return sorted(blockers, key=lambda code: (priority.get(code, 12), code))


def _phase_aware_routing_rule_section(
    *,
    provider: str,
    requested_model_family: str,
    candidate_target_model: str,
    target_model_family: str | None,
    source_surface: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    has_tools: bool,
    text_bucket: str,
    token_bucket: str,
) -> dict[str, Any]:
    """A stageable, file-backed routing-rule template for a review-ready cohort.

    Shaped like a ``preferred_pathways`` entry in routing_experiments.yaml but
    explicitly review-only: it carries no concrete request binding and is gated
    behind the existing local promotion/stage command (no policy file writes).
    """
    return {
        "schema": PHASE_AWARE_ROUTING_RULE_SECTION_SCHEMA,
        "target_file": "routing_experiments.yaml",
        "target_section": "preferred_pathways",
        "stageable": True,
        "requires_promotion_command": True,
        "promotion_command": "tokenclaw-routing-experiments-stage",
        "policy_files_written": False,
        "rule_template": {
            "provider": provider,
            "requested_model_family": requested_model_family,
            "routed_model": candidate_target_model,
            "routed_model_family": target_model_family,
            "source_surface": source_surface,
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": text_bucket,
            "token_bucket": token_bucket,
            "mode": "shadow_candidate_pass_through",
        },
        "requires_concrete_model_binding": True,
        "metadata_only": True,
    }


def build_request_shape_phase_aware_routing_dry_run(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 25,
    canary_fraction: float = DEFAULT_PHASE_AWARE_ROUTING_CANARY_FRACTION,
    holdout_fraction: float = DEFAULT_PHASE_AWARE_ROUTING_HOLDOUT_FRACTION,
    min_samples: int = DEFAULT_PHASE_AWARE_ROUTING_MIN_SAMPLES,
    max_error_rate: float = DEFAULT_PHASE_AWARE_ROUTING_MAX_ERROR_RATE,
    max_retry_rate: float = DEFAULT_PHASE_AWARE_ROUTING_MAX_RETRY_RATE,
    max_fallback_rate: float = DEFAULT_PHASE_AWARE_ROUTING_MAX_FALLBACK_RATE,
) -> dict[str, Any]:
    """Attach phase-aware routing dry-run deltas to ranked request-shape cohorts.

    Consumes ranked request-shape rollups, compares each routing-eligible cohort
    against the active local routing policy, and emits review-only route deltas
    for cheaper target tiers (or explicit blocker/coverage rows). No policy file
    is written, no provider or managed-server call is made, and a stageable
    file-backed rule template is included only when a cohort is review-ready and
    not already covered by the active policy.
    """
    capped_limit = max(1, min(_as_int(limit) or 25, 1000))
    rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    coverage_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    total_samples = 0
    total_projected = 0.0
    stageable_count = 0

    for row in rollups:
        if not isinstance(row, dict):
            continue
        classes = {public_label(item, "unknown") for item in row.get("candidate_work_classes") or []}
        families = {public_label(item, "unknown") for item in row.get("candidate_families") or []}
        routing_status = public_label(row.get("routing_status"), "unknown")
        if "routing" not in classes and "routing_candidate" not in families and routing_status not in {"passthrough", "routed"}:
            continue

        provider = public_label(row.get("provider_family"), "unknown")
        requested_family = public_label(row.get("requested_model_family"), "unknown")
        routed_family = public_label(row.get("routed_model_family") or requested_family, requested_family)
        source_surface = public_label(row.get("source_surface"), "unknown")
        category = public_label(row.get("category"), "unknown")
        workflow_phase = public_label(row.get("workflow_phase") or category, "unknown")
        stream = bool(row.get("stream"))
        has_tools = bool(row.get("has_tools"))
        text_bucket = public_label(row.get("text_bucket"), "unknown")
        token_bucket = public_label(row.get("token_bucket"), "unknown")
        sample_count = _as_int(row.get("row_count") or row.get("sample_count"))

        target_model, target_family, target_reason = _routing_downgrade_target(
            provider=provider,
            requested_model_family=requested_family,
        )
        projection = _routing_downgrade_projection(
            row,
            provider=provider,
            requested_model=requested_family,
            target_model=target_model or requested_family,
            sample_count=sample_count,
        )
        coverage = _phase_aware_routing_policy_coverage(
            provider=provider,
            requested_model_family=requested_family,
            routed_model_family=routed_family,
            candidate_target_model=target_model,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
        )
        blockers = _phase_aware_routing_blockers(
            row,
            target_model=target_model,
            target_reason=target_reason,
            projection=projection,
            coverage=coverage,
            min_samples=min_samples,
            max_error_rate=max_error_rate,
            max_retry_rate=max_retry_rate,
            max_fallback_rate=max_fallback_rate,
        )
        hard_blockers = {
            "stale-request-shape-rollup",
            "active-routing-policy-blocklist",
            "too-small-routing-drill-sample",
            "non-positive-routing-savings-projection",
            "thinking-routing-guard",
            "high-downgrade-risk-shape",
            "elevated-error-rate",
            "elevated-fallback-rate",
            "elevated-retry-rate",
            "no-cheaper-anthropic-routing-target",
            "no-cheaper-openai-routing-target",
            "unknown-provider-routing-target",
        }
        coverage_state = public_label(coverage.get("coverage_state"), "coverage-unknown")
        if hard_blockers.intersection(blockers):
            status = "blocked"
            next_action = "keep-phase-aware-route-candidate-blocked"
        elif coverage_state == "covered-by-active-policy":
            status = "already-covered"
            next_action = "observe-active-routing-policy-coverage"
        else:
            status = "review-ready"
            next_action = "review-phase-aware-route-canary"

        is_stageable = status == "review-ready" and coverage_state in {"uncovered", "coverage-unknown"}
        rule_section = (
            _phase_aware_routing_rule_section(
                provider=provider,
                requested_model_family=requested_family,
                candidate_target_model=target_model,
                target_model_family=target_family,
                source_surface=source_surface,
                category=category,
                workflow_phase=workflow_phase,
                stream=stream,
                has_tools=has_tools,
                text_bucket=text_bucket,
                token_bucket=token_bucket,
            )
            if is_stageable and target_model
            else None
        )
        canary_sample_count = int(math.ceil(sample_count * canary_fraction)) if status == "review-ready" else 0
        holdout_sample_count = int(math.ceil(sample_count * holdout_fraction)) if status == "review-ready" else 0
        fingerprint_material = {
            "schema": PHASE_AWARE_ROUTING_DRY_RUN_ROW_SCHEMA,
            "provider_family": provider,
            "source_surface": source_surface,
            "endpoint": public_label(row.get("endpoint"), "unknown"),
            "requested_model_family": requested_family,
            "target_model_family": target_family,
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": text_bucket,
            "token_bucket": token_bucket,
        }
        delta = {
            "schema": PHASE_AWARE_ROUTING_DRY_RUN_ROW_SCHEMA,
            "fingerprint": _routing_drill_fingerprint(fingerprint_material),
            "source_evidence_schema": row.get("source_schema") or row.get("schema") or ROLLUP_ROW_SCHEMA,
            **fingerprint_material,
            "candidate_id_included": False,
            "requested_model_family": requested_family,
            "current_model_family": routed_family,
            "candidate_target_model": target_model,
            "candidate_target_tier": target_family,
            "target_reason": target_reason,
            "routing_status": routing_status,
            "sample_count": sample_count,
            "row_count": sample_count,
            "error_count": _as_int(row.get("error_count")),
            "retry_count": _as_int(row.get("retry_count")),
            "fallback_count": _as_int(row.get("fallback_count")),
            "error_rate": round(_as_int(row.get("error_count")) / float(sample_count), 6) if sample_count else 0.0,
            "retry_rate": round(_as_int(row.get("retry_count")) / float(sample_count), 6) if sample_count else 0.0,
            "fallback_rate": round(_as_int(row.get("fallback_count")) / float(sample_count), 6) if sample_count else 0.0,
            "input_tokens": projection["input_tokens"],
            "output_tokens": projection["output_tokens"],
            "projected_savings_usd": projection["projected_savings_usd"],
            "projected_savings_per_1000_calls_usd": projection["projected_savings_per_1000_calls_usd"],
            "recommended_canary_fraction": round(canary_fraction, 4) if status == "review-ready" else 0.0,
            "recommended_holdout_fraction": round(holdout_fraction, 4) if status == "review-ready" else 0.0,
            "recommended_canary_sample_count": canary_sample_count,
            "recommended_holdout_sample_count": holdout_sample_count,
            "active_policy_coverage": coverage,
            "coverage_state": coverage_state,
            "safety_reason_codes": blockers,
            "top_blocker_code": blockers[0] if blockers else None,
            "blocker_codes": blockers,
            "status": status,
            "readiness_state": status,
            "recommended_next_action": next_action,
            "stageable_routing_rule_section": rule_section,
            "local_action_family": "routing",
            "review_only": True,
            "emits_routing_apply_action": False,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "aggregate_only": True,
            "privacy": _phase_aware_routing_privacy(),
        }
        rows.append(delta)
        total_samples += sample_count
        if status == "review-ready":
            total_projected += _as_float(delta.get("projected_savings_usd"))
        if rule_section is not None:
            stageable_count += 1
        _increment(status_counts, status, sample_count)
        _increment(phase_counts, workflow_phase, sample_count)
        _increment(coverage_counts, coverage_state, sample_count)
        _increment(target_counts, target_family or target_reason, sample_count)
        for blocker in blockers:
            _increment(blocker_counts, blocker, sample_count)

    rows.sort(
        key=lambda item: (
            item.get("status") == "review-ready",
            item.get("stageable_routing_rule_section") is not None,
            _as_float(item.get("projected_savings_per_1000_calls_usd")),
            _as_int(item.get("sample_count")),
            str(item.get("fingerprint")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows[:capped_limit], start=1):
        row["rank"] = rank

    ranked = rows[:capped_limit]
    top = ranked[0] if ranked else None
    return {
        "schema": PHASE_AWARE_ROUTING_DRY_RUN_SCHEMA,
        "status": "ranked" if ranked else "no-phase-aware-routing-candidates",
        "summary": {
            "rows_considered": len([row for row in rollups if isinstance(row, dict)]),
            "candidate_count": len(ranked),
            "review_ready_count": sum(1 for row in ranked if row.get("status") == "review-ready"),
            "blocked_count": sum(1 for row in ranked if row.get("status") == "blocked"),
            "already_covered_count": sum(1 for row in ranked if row.get("status") == "already-covered"),
            "stageable_rule_section_count": sum(
                1 for row in ranked if row.get("stageable_routing_rule_section") is not None
            ),
            "sample_count": total_samples,
            "top_fingerprint": top.get("fingerprint") if top else None,
            "top_next_action": top.get("recommended_next_action") if top else None,
            "top_blocker_code": top.get("top_blocker_code") if top else None,
            "top_candidate_target_tier": top.get("candidate_target_tier") if top else None,
            "top_projected_savings_per_1000_calls_usd": (
                top.get("projected_savings_per_1000_calls_usd") if top else 0.0
            ),
            "total_projected_savings_usd": round(total_projected, 8),
            "minimum_sample_count": min_samples,
            "default_canary_fraction": round(canary_fraction, 4),
            "default_holdout_fraction": round(holdout_fraction, 4),
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "status_breakdown": _breakdown(status_counts),
        "workflow_phase_breakdown": _breakdown(phase_counts),
        "coverage_breakdown": _breakdown(coverage_counts),
        "target_tier_breakdown": _breakdown(target_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "top_candidate": top,
        "candidates": ranked,
        "acceptance": {
            "has_stable_fingerprints": all(
                str(row.get("fingerprint", "")).startswith("routing-drill:") for row in ranked
            ),
            "has_projected_savings_per_1000_calls": all(
                "projected_savings_per_1000_calls_usd" in row for row in ranked
            ),
            "has_candidate_target_tier": all("candidate_target_tier" in row for row in ranked),
            "has_safety_reason_codes": all("safety_reason_codes" in row for row in ranked),
            "has_active_policy_coverage": all("active_policy_coverage" in row for row in ranked),
            "has_next_action": all(row.get("recommended_next_action") for row in ranked),
            "emits_ranked_or_blocker_rows": bool(ranked),
            "emits_no_routing_apply_actions": all(not bool(row.get("emits_routing_apply_action")) for row in ranked),
            "stageable_writes_gated_behind_promotion": all(
                (row.get("stageable_routing_rule_section") or {}).get("requires_promotion_command", True)
                for row in ranked
                if row.get("stageable_routing_rule_section") is not None
            ),
            "policy_files_written": False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _phase_aware_routing_privacy(),
    }


def _source_traffic_acquisition_attempt(
    *,
    rows_considered: int,
    rollup_count: int,
    source: str = "recent-local-call-metadata",
) -> dict[str, Any]:
    status = "completed" if rows_considered > 0 and rollup_count > 0 else "no-source-traffic"
    return {
        "schema": SOURCE_TRAFFIC_ACQUISITION_SCHEMA,
        "status": status,
        "action_type": "source-traffic-acquisition",
        "source_schema": FOLLOW_UP_CANDIDATES_SCHEMA,
        "recommended_next_action": "emit-request-shape-rollups",
        "recommended_command": "tokenclaw-request-shape-rollups",
        "recommended_module": "tokenclaw.request_shape_rollups",
        "target_downstream_lever": "cohort-ranking",
        "blocker_code": None if status == "completed" else "no-source-traffic-for-request-shape-rollups",
        "attempted": True,
        "rows_considered": rows_considered,
        "rollup_count": rollup_count,
        "source": public_label(source, "recent-local-call-metadata"),
        "metadata_only": True,
        "aggregate_only": True,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _shape_follow_up_privacy(),
    }


def _estimated_cache_replay_savings(row: dict[str, Any], row_count: int) -> tuple[int, float]:
    projected_hits = max(0, row_count - 1)
    if projected_hits <= 0:
        return 0, 0.0
    cost = _as_float(row.get("cost_est_usd"))
    return projected_hits, cost * (projected_hits / float(row_count)) if row_count > 0 else 0.0


def _cache_replay_readiness_gate(
    *,
    row_count: int,
    projected_hits: int,
    projected_savings_usd: float,
    cache_hit_count: int = 0,
) -> dict[str, Any]:
    live_repeat_confirmed = cache_hit_count > 0
    row_floor_met = (
        row_count >= DEFAULT_CACHE_REPLAY_MIN_STAGE_ROWS
        and projected_hits >= DEFAULT_CACHE_REPLAY_MIN_STAGE_PROJECTED_HITS
    )
    savings_floor_met = projected_savings_usd >= DEFAULT_CACHE_REPLAY_MIN_STAGE_SAVINGS_USD
    stage_allowed = live_repeat_confirmed or row_floor_met or savings_floor_met
    if live_repeat_confirmed:
        gate_status = "live-repeat-confirmed"
    elif savings_floor_met:
        gate_status = "savings-floor-met"
    elif row_floor_met:
        gate_status = "repeat-floor-met"
    else:
        gate_status = "replay-ready-but-too-small"
    return {
        "schema": "tokenclaw.request_shape_cache_replay_readiness_gate.v1",
        "gate_status": gate_status,
        "stage_allowed": stage_allowed,
        "next_action": "stage-cache-replay-canary" if stage_allowed else "no-op-too-small-without-live-repeat",
        "reason": gate_status if stage_allowed else "too-small-without-live-repeat",
        "row_count": row_count,
        "projected_hits": projected_hits,
        "projected_savings_usd": round(projected_savings_usd, 6),
        "live_repeat_confirmed": live_repeat_confirmed,
        "live_repeat_cache_hit_count": cache_hit_count,
        "minimum_row_count": DEFAULT_CACHE_REPLAY_MIN_STAGE_ROWS,
        "minimum_projected_hits": DEFAULT_CACHE_REPLAY_MIN_STAGE_PROJECTED_HITS,
        "minimum_projected_savings_usd": DEFAULT_CACHE_REPLAY_MIN_STAGE_SAVINGS_USD,
        "row_floor_met": row_floor_met,
        "savings_floor_met": savings_floor_met,
        "metadata_only": True,
        "aggregate_only": True,
        "privacy": _replayability_privacy(),
    }


def _shape_activation_decision(row: dict[str, Any], classes: list[str], blockers: list[str]) -> dict[str, Any]:
    class_set = set(classes)
    blocker_set = set(blockers)
    row_count = _as_int(row.get("row_count") or row.get("count"))
    projected_hits = 0
    projected_savings = 0.0
    projected_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    projected_crunch_savings = _as_float(row.get("projected_crunch_savings_usd"))

    if "crunch" in class_set or "repeated_context" in class_set:
        crunch_decision = _shape_crunch_decision(row)
        crunch_readiness = str(crunch_decision.get("readiness") or "unknown")
        if crunch_readiness == "measurement-ready":
            return {
                "readiness_state": "activation-ready",
                "next_action": "stage-repeated-context-crunch-canary",
                "local_action_family": "crunch",
                "actionability_reason": str(crunch_decision.get("reason") or "repeated-context-crunch-opportunity"),
                "projected_hits": 0,
                "projected_saved_tokens": projected_tokens,
                "projected_savings_usd": projected_crunch_savings,
                "blocker_codes": blockers,
            }
        if crunch_readiness in {"canary-staged", "canary-applied", "canary-holdout"}:
            return {
                "readiness_state": "measurement-required",
                "next_action": "measure-repeated-context-crunch-canary-impact",
                "local_action_family": "crunch",
                "actionability_reason": str(crunch_decision.get("reason") or "missing-crunch-canary-impact-measurement"),
                "projected_hits": 0,
                "projected_saved_tokens": projected_tokens,
                "projected_savings_usd": projected_crunch_savings,
                "blocker_codes": list(crunch_decision.get("blockers") or blockers),
            }
        if crunch_readiness == "canary-safety-stopped":
            return {
                "readiness_state": "blocked",
                "next_action": "review-repeated-context-crunch-canary-safety-stop",
                "local_action_family": "crunch",
                "actionability_reason": str(crunch_decision.get("reason") or "canary-safety-stopped"),
                "projected_hits": 0,
                "projected_saved_tokens": projected_tokens,
                "projected_savings_usd": projected_crunch_savings,
                "blocker_codes": list(crunch_decision.get("blockers") or blockers),
            }

    cache_status = public_label(row.get("cache_status"), "unknown")
    stream = bool(row.get("stream"))
    has_tools = bool(row.get("has_tools"))
    if "replayability" in class_set and not stream and not has_tools and cache_status in {"miss", "missing"} and row_count >= 2:
        projected_hits, projected_savings = _estimated_cache_replay_savings(row, row_count)
        gate = _cache_replay_readiness_gate(
            row_count=row_count,
            projected_hits=projected_hits,
            projected_savings_usd=projected_savings,
            cache_hit_count=_as_int(row.get("cache_hit_count")),
        )
        if not bool(gate["stage_allowed"]):
            return {
                "readiness_state": "blocked",
                "next_action": gate["next_action"],
                "local_action_family": "cache",
                "actionability_reason": gate["reason"],
                "projected_hits": projected_hits,
                "projected_saved_tokens": 0,
                "projected_savings_usd": projected_savings,
                "blocker_codes": ["too-small-without-live-repeat"],
            }
        return {
            "readiness_state": "activation-ready",
            "next_action": "stage-cache-replay-canary",
            "local_action_family": "cache",
            "actionability_reason": gate["gate_status"],
            "projected_hits": projected_hits,
            "projected_saved_tokens": 0,
            "projected_savings_usd": projected_savings,
            "blocker_codes": [blocker for blocker in blockers if blocker != "exact-cache-miss"],
        }
    if "tool-call-cache-disabled" in blocker_set:
        return {
            "readiness_state": "blocked",
            "next_action": "collect-tool-call-cache-invalidation-evidence",
            "local_action_family": "cache",
            "actionability_reason": "tool-call-cache-needs-invalidation-evidence",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }
    if "unsupported-streaming-shape" in blocker_set and "replayability" in class_set:
        return {
            "readiness_state": "blocked",
            "next_action": "add-streaming-cache-replay-support",
            "local_action_family": "cache",
            "actionability_reason": "streaming-cache-replay-not-supported",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }
    if "thinking-routing-guard" in blocker_set:
        return {
            "readiness_state": "needs-lifecycle-evidence",
            "next_action": "collect-thinking-routing-lifecycle-evidence",
            "local_action_family": "routing",
            "actionability_reason": "thinking-routing-guard-needs-lifecycle-evidence",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }
    if "routing" in class_set:
        return {
            "readiness_state": "needs-lifecycle-evidence",
            "next_action": "stage-routing-lifecycle-evidence",
            "local_action_family": "routing",
            "actionability_reason": "routing-candidate-needs-lifecycle-evidence",
            "projected_hits": 0,
            "projected_saved_tokens": 0,
            "projected_savings_usd": 0.0,
            "blocker_codes": blockers,
        }

    next_action = _shape_next_action(classes, blockers)
    return {
        "readiness_state": "needs-classification" if blockers else "observability-only",
        "next_action": next_action,
        "local_action_family": _shape_local_action_family(next_action, classes),
        "actionability_reason": blockers[0] if blockers else "no-actionable-blocker",
        "projected_hits": 0,
        "projected_saved_tokens": projected_tokens,
        "projected_savings_usd": projected_crunch_savings,
        "blocker_codes": blockers,
    }


def _shape_follow_up_candidate(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    classes = sorted(public_label(item, "unknown") for item in row.get("candidate_work_classes") or [])
    families = sorted(public_label(item, "unknown") for item in row.get("candidate_families") or [])
    blockers = sorted(public_label(item, "unknown") for item in row.get("blocker_codes") or [])
    decision = _shape_activation_decision(row, classes, blockers)
    next_action = str(decision["next_action"])
    provider = public_label(row.get("provider_family"), "unknown")
    source_surface = public_label(row.get("source_surface"), "unknown")
    endpoint = public_label(row.get("endpoint"), "unknown")
    app_family = public_label(
        row.get("app_family")
        or _app_family(provider, source_surface, row.get("requested_model_family")),
        "unknown",
    )
    row_count = _as_int(row.get("row_count") or row.get("count"))
    cost = _as_float(row.get("cost_est_usd"))
    observed_savings = _as_float(row.get("observed_savings_usd"))
    error_count = _as_int(row.get("error_count"))
    retry_count = _as_int(row.get("retry_count"))
    projected_crunch_savings = _as_float(row.get("projected_crunch_savings_usd"))
    projected_crunch_tokens = _as_int(row.get("projected_crunch_tokens_saved"))
    projected_crunch_chars = _as_int(row.get("projected_crunch_chars_saved"))
    crunch_canary_lifecycle = _shape_crunch_lifecycle_summary(row)
    readiness = str(decision.get("readiness_state") or "unknown")
    readiness_weight = {
        "activation-ready": 500.0,
        "measurement-required": 350.0,
        "needs-lifecycle-evidence": 150.0,
        "blocked": 50.0,
    }.get(readiness, 0.0)
    replay_weight = 100.0 if "replayability" in classes else 0.0
    repeated_weight = 150.0 if "repeated_context" in classes else 0.0
    routing_weight = 75.0 if "routing" in classes else 0.0
    crunch_weight = 75.0 if "crunch" in classes else 0.0
    score = (
        row_count
        + cost * 1000.0
        + observed_savings * 2000.0
        + projected_crunch_savings * 2500.0
        + repeated_weight
        + replay_weight
        + routing_weight
        + crunch_weight
        + readiness_weight
        - error_count * 5.0
        - retry_count * 0.5
    )
    fingerprint_material = {
        "schema": FOLLOW_UP_BLOCKER_COHORT_SCHEMA,
        "provider_family": provider,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "app_family": app_family,
        "requested_model_family": public_label(row.get("requested_model_family"), "unknown"),
        "routed_model_family": public_label(row.get("routed_model_family"), "unknown"),
        "category": public_label(row.get("category"), "unknown"),
        "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": public_label(row.get("text_bucket"), "unknown"),
        "token_bucket": public_label(row.get("token_bucket"), "unknown"),
        "cache_status": public_label(row.get("cache_status"), "unknown"),
        "routing_status": public_label(row.get("routing_status"), "unknown"),
        "candidate_work_classes": classes,
        "candidate_families": families,
    }
    digest = hashlib.sha256(stable_json(fingerprint_material).encode("utf-8")).hexdigest()[:16]
    freshness_state = _shape_candidate_freshness_state(row)
    preview_requirement, managed_preview_required = _shape_candidate_preview_requirement(decision)
    return {
        "schema": FOLLOW_UP_BLOCKER_COHORT_SCHEMA,
        "fingerprint": f"request-shape-follow-up:{digest}",
        "duplicate_suppression_key": f"request-shape-follow-up:{digest}:{next_action}",
        "rank": rank,
        "provider_surface_bucket": "/".join(part for part in (provider, source_surface, endpoint) if part) or "mixed",
        "provider_family": provider,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "app_family": app_family,
        "requested_model_family": public_label(row.get("requested_model_family"), "unknown"),
        "routed_model_family": public_label(row.get("routed_model_family"), "unknown"),
        "category": public_label(row.get("category"), "unknown"),
        "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": public_label(row.get("text_bucket"), "unknown"),
        "token_bucket": public_label(row.get("token_bucket"), "unknown"),
        "cache_status": public_label(row.get("cache_status"), "unknown"),
        "routing_status": public_label(row.get("routing_status"), "unknown"),
        "row_count": row_count,
        "sample_count": row_count,
        "error_count": error_count,
        "retry_count": retry_count,
        "cost_est_usd": round(cost, 6),
        "observed_savings_usd": round(observed_savings, 6),
        "projected_hits": _as_int(decision.get("projected_hits")),
        "projected_crunch_tokens_saved": projected_crunch_tokens,
        "projected_crunch_chars_saved": projected_crunch_chars,
        "projected_crunch_savings_usd": round(projected_crunch_savings, 6),
        "projected_saved_tokens": _as_int(decision.get("projected_saved_tokens")),
        "projected_saved_chars": projected_crunch_chars,
        "projected_savings_usd": round(_as_float(decision.get("projected_savings_usd")), 6),
        "successful_input_tokens": _as_int(row.get("successful_input_tokens")),
        "input_tokens": _as_int(row.get("input_tokens")),
        "crunch_canary_lifecycle": crunch_canary_lifecycle,
        "candidate_work_classes": classes,
        "candidate_families": families,
        "blocker_codes": sorted(public_label(item, "unknown") for item in decision.get("blocker_codes") or []),
        "readiness_state": readiness,
        "freshness_state": freshness_state,
        "preview_requirement": preview_requirement,
        "managed_preview_required": managed_preview_required,
        "actionability_reason": public_label(decision.get("actionability_reason"), "unknown"),
        "next_action": next_action,
        "recommended_next_action": next_action,
        "local_action_family": public_label(decision.get("local_action_family"), "cohort-ranking"),
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "read_only": True,
        "aggregate_only": True,
        "privacy": _shape_follow_up_privacy(),
        "_score": score,
    }


def _shape_candidate_freshness_state(row: dict[str, Any]) -> str:
    if bool(row.get("stale") or row.get("snapshot_stale")):
        return "stale"
    freshness = row.get("snapshot_freshness") if isinstance(row.get("snapshot_freshness"), dict) else {}
    if freshness.get("stale") is True:
        return "stale"
    status = public_label(
        row.get("freshness_state")
        or row.get("freshness_status")
        or freshness.get("status"),
        "",
    )
    if status in {"stale", "snapshot-stale", "rollup-stale"}:
        return "stale"
    if status in {"fresh", "snapshot-fresh"}:
        return "fresh"
    if _as_int(row.get("row_count") or row.get("sample_count") or row.get("count")) > 0:
        return "fresh"
    return "unknown"


def _shape_candidate_preview_requirement(decision: dict[str, Any]) -> tuple[str, bool]:
    family = public_label(decision.get("local_action_family"), "cohort-ranking")
    readiness = public_label(decision.get("readiness_state"), "unknown")
    if readiness != "activation-ready":
        return "not-required-for-ranking", False
    if family in {"cache", "routing"}:
        return "managed-preview-required-before-policy-write", True
    return "managed-preview-optional", False


def _shape_crunch_lifecycle_summary(row: dict[str, Any]) -> dict[str, Any]:
    lifecycle = row.get("crunch_canary_lifecycle") if isinstance(row.get("crunch_canary_lifecycle"), dict) else {}
    if not lifecycle:
        return {}
    return {
        "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
        "applied_count": _as_int(lifecycle.get("applied_count")),
        "holdout_count": _as_int(lifecycle.get("holdout_count")),
        "skipped_count": _as_int(lifecycle.get("skipped_count")),
        "safety_stopped_count": _as_int(lifecycle.get("safety_stopped_count")),
        "fallback_count": _as_int(lifecycle.get("fallback_count")),
        "rollback_count": _as_int(lifecycle.get("rollback_count")),
        "status_breakdown": lifecycle.get("status_breakdown") if isinstance(lifecycle.get("status_breakdown"), list) else [],
        "metadata_only": True,
        "aggregate_only": True,
    }


def _shape_candidate_expected_savings_path(candidate: dict[str, Any]) -> str:
    family = public_label(candidate.get("local_action_family"), "cohort-ranking")
    action = public_label(candidate.get("recommended_next_action") or candidate.get("next_action"), "inspect-local-evidence")
    if family == "crunch":
        return "Convert repeated-context rollup evidence into measured local crunch canaries with holdout coverage."
    if family == "cache":
        return "Convert replayable request-shape cohorts into exact-cache canary or dependency evidence follow-up."
    if family == "routing":
        return "Convert request-shape routing cohorts into measured downgrade lifecycle evidence before policy changes."
    return f"Keep aggregate request-shape rollups moving through `{action}` until a concrete local savings lever is selected."


def _shape_activation_candidate_entry(candidate: dict[str, Any], *, rank: int) -> dict[str, Any]:
    family = public_label(candidate.get("local_action_family"), "cohort-ranking")
    next_action = public_label(candidate.get("recommended_next_action") or candidate.get("next_action"), "inspect-local-evidence")
    source_fingerprint = str(candidate.get("fingerprint") or "").strip()
    material = {
        "schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_ENTRY_SCHEMA,
        "source_fingerprint": source_fingerprint,
        "local_action_family": family,
        "recommended_next_action": next_action,
    }
    fingerprint = public_id(json.dumps(material, sort_keys=True), prefix="activation")
    sample_count = _as_int(candidate.get("sample_count") or candidate.get("row_count"))
    projected_savings = round(_as_float(candidate.get("projected_savings_usd")), 8)
    savings_per_1000 = round((projected_savings / sample_count) * 1000.0, 8) if sample_count > 0 and projected_savings > 0 else 0.0
    readiness = public_label(candidate.get("readiness_state"), "unknown")
    blocker_codes = [
        public_label(item, "unknown")
        for item in candidate.get("blocker_codes") or []
        if public_label(item, "unknown") != "unknown"
    ]
    state = "ranked-evidence" if readiness == "activation-ready" else readiness or "blocked"
    current_status = "projected" if readiness == "activation-ready" else "blocked"
    target_rule_file = {
        "cache": "cache_rules.yaml",
        "crunch": "crunch_rules.yaml",
        "routing": "routing_rules.yaml",
    }.get(family)
    target_section = {
        "cache": "cache.pattern_rules",
        "crunch": "crunch.rules",
        "routing": "routing.rules",
    }.get(family)
    return {
        "schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_ENTRY_SCHEMA,
        "rank": rank,
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "duplicate_suppression_key": public_label(candidate.get("duplicate_suppression_key"), fingerprint),
        "lever": "request-shape-rollups",
        "local_action_family": family,
        "state": state,
        "current_status": current_status,
        "issue_worthy_status": "review" if readiness == "activation-ready" else "blocked",
        "readiness_state": readiness,
        "freshness_state": public_label(candidate.get("freshness_state"), "unknown"),
        "recommended_next_action": next_action,
        "next_action": next_action,
        "blocking_reason": blocker_codes[0] if blocker_codes else public_label(candidate.get("actionability_reason"), None),
        "unblock_reason": blocker_codes[0] if blocker_codes else public_label(candidate.get("actionability_reason"), None),
        "blocker_codes": blocker_codes,
        "sample_count": sample_count,
        "row_count": _as_int(candidate.get("row_count")),
        "projected_hits": _as_int(candidate.get("projected_hits")),
        "projected_saved_tokens": _as_int(candidate.get("projected_saved_tokens")),
        "projected_saved_chars": _as_int(candidate.get("projected_saved_chars")),
        "projected_savings_usd": projected_savings,
        "projected_saved_usd": projected_savings,
        "savings_per_1000_calls_usd": savings_per_1000,
        "successful_input_tokens": _as_int(candidate.get("successful_input_tokens")),
        "input_tokens": _as_int(candidate.get("input_tokens")),
        "crunch_canary_lifecycle": candidate.get("crunch_canary_lifecycle")
        if isinstance(candidate.get("crunch_canary_lifecycle"), dict)
        else {},
        "observed_savings_usd": round(_as_float(candidate.get("observed_savings_usd")), 8),
        "expected_savings_path": _shape_candidate_expected_savings_path(candidate),
        "provider_family": public_label(candidate.get("provider_family"), "unknown"),
        "source_surface": public_label(candidate.get("source_surface"), "unknown"),
        "endpoint": public_label(candidate.get("endpoint"), "unknown"),
        "category": public_label(candidate.get("category"), "unknown"),
        "workflow_phase": public_label(candidate.get("workflow_phase"), "unknown"),
        "stream": bool(candidate.get("stream")),
        "has_tools": bool(candidate.get("has_tools")),
        "text_bucket": public_label(candidate.get("text_bucket"), "unknown"),
        "token_bucket": public_label(candidate.get("token_bucket"), "unknown"),
        "cache_status": public_label(candidate.get("cache_status"), "unknown"),
        "routing_status": public_label(candidate.get("routing_status"), "unknown"),
        "requested_model_family": public_label(candidate.get("requested_model_family"), "unknown"),
        "routed_model_family": public_label(candidate.get("routed_model_family"), "unknown"),
        "target_local_rule_file": target_rule_file,
        "target_local_policy_section": target_section,
        "source_evidence_schema": FOLLOW_UP_CANDIDATES_SCHEMA,
        "evidence_schema": FOLLOW_UP_CANDIDATES_SCHEMA,
        "preview_requirement": public_label(candidate.get("preview_requirement"), "not-required-for-ranking"),
        "managed_preview_required": bool(candidate.get("managed_preview_required")),
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "read_only": True,
        "privacy": _shape_follow_up_privacy(),
    }


def _shape_activation_candidate_queue(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        _shape_activation_candidate_entry(candidate, rank=index)
        for index, candidate in enumerate(candidates, start=1)
    ]
    family_counts: dict[str, int] = {}
    freshness_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    total_projected = 0.0
    for entry in entries:
        sample_count = _as_int(entry.get("sample_count"))
        total_projected += _as_float(entry.get("projected_savings_usd"))
        _increment(family_counts, entry.get("local_action_family"), sample_count)
        _increment(freshness_counts, entry.get("freshness_state"), sample_count)
        _increment(action_counts, entry.get("recommended_next_action"), sample_count)
    top = entries[0] if entries else None
    return {
        "schema": LOCAL_ACTIVATION_CANDIDATE_QUEUE_SCHEMA,
        "status": "ranked" if entries else "empty",
        "source_schema": FOLLOW_UP_CANDIDATES_SCHEMA,
        "read_only": True,
        "summary": {
            "queued_candidate_count": len(entries),
            "top_local_action_family": top.get("local_action_family") if top else None,
            "top_next_action": top.get("recommended_next_action") if top else None,
            "top_freshness_state": top.get("freshness_state") if top else None,
            "total_projected_savings_usd": round(total_projected, 8),
            "policy_files_written": False,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "local_action_family_breakdown": _breakdown(family_counts),
            "freshness_breakdown": _breakdown(freshness_counts),
            "next_action_breakdown": _breakdown(action_counts),
        },
        "top_candidate": top,
        "entries": entries,
        "privacy": _shape_follow_up_privacy(),
    }


def build_request_shape_follow_up_candidates(
    rollups: list[dict[str, Any]],
    *,
    limit: int = 10,
    source: str = "recent-local-call-metadata",
) -> dict[str, Any]:
    relevant_classes = {"repeated_context", "replayability", "routing", "crunch"}
    candidates = [
        _shape_follow_up_candidate(row, rank=index)
        for index, row in enumerate(rollups, start=1)
        if isinstance(row, dict)
        and relevant_classes.intersection({str(item) for item in row.get("candidate_work_classes") or []})
    ]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("_score")),
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("cost_est_usd")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )

    class_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    clean: list[dict[str, Any]] = []
    capped_limit = max(1, min(_as_int(limit, 10), 100))
    for rank, candidate in enumerate(candidates[:capped_limit], start=1):
        candidate["rank"] = rank
        row_count = _as_int(candidate.get("row_count"))
        for work_class in candidate.get("candidate_work_classes") or []:
            _increment(class_counts, work_class, row_count)
        for blocker in candidate.get("blocker_codes") or []:
            _increment(blocker_counts, blocker, row_count)
        _increment(action_counts, candidate.get("next_action"), row_count)
        _increment(family_counts, candidate.get("local_action_family"), row_count)
        _increment(readiness_counts, candidate.get("readiness_state"), row_count)
        item = dict(candidate)
        item.pop("_score", None)
        clean.append(item)
    activation_queue = _shape_activation_candidate_queue(clean)

    no_source_traffic_reason = None
    if not clean and not rollups:
        no_source_traffic_reason = "no-source-traffic-for-request-shape-rollups"
    status = (
        "candidates-ranked"
        if clean
        else "no-source-traffic"
        if no_source_traffic_reason
        else "no-request-shape-follow-up-candidates"
    )
    top = clean[0] if clean else None
    missing = [] if clean else [no_source_traffic_reason or "request_shape_follow_up_candidates"]
    rows_considered = sum(_as_int(row.get("row_count") or row.get("count")) for row in rollups if isinstance(row, dict))
    source_traffic_acquisition = _source_traffic_acquisition_attempt(
        rows_considered=rows_considered,
        rollup_count=len([row for row in rollups if isinstance(row, dict)]),
        source=source,
    )
    return {
        "schema": FOLLOW_UP_CANDIDATES_SCHEMA,
        "status": status,
        "action_type": "source-traffic-acquisition",
        "source_traffic_acquisition_status": source_traffic_acquisition["status"],
        "source_traffic_acquisition": source_traffic_acquisition,
        "summary": {
            "rows_considered": rows_considered,
            "rollup_count": len([row for row in rollups if isinstance(row, dict)]),
            "ranked_candidate_count": len(clean),
            "top_next_action": top.get("next_action") if top else "emit-request-shape-rollups" if no_source_traffic_reason else None,
            "top_local_action_family": top.get("local_action_family") if top else "cohort-ranking" if no_source_traffic_reason else None,
            "top_readiness_state": top.get("readiness_state") if top else "blocked" if no_source_traffic_reason else None,
            "no_source_traffic_reason": no_source_traffic_reason,
            "source_traffic_acquisition_status": source_traffic_acquisition["status"],
            "source_traffic_acquisition_attempted": True,
            "activation_ready_count": sum(1 for item in clean if item.get("readiness_state") == "activation-ready"),
            "activation_candidate_count": _as_int(activation_queue["summary"].get("queued_candidate_count")),
            "activation_candidate_top_next_action": activation_queue["summary"].get("top_next_action"),
            "activation_candidate_top_freshness_state": activation_queue["summary"].get("top_freshness_state"),
            "class_breakdown": _breakdown(class_counts),
            "blocker_breakdown": _breakdown(blocker_counts),
            "readiness_breakdown": _breakdown(readiness_counts),
            "next_action_breakdown": _breakdown(action_counts),
            "local_action_family_breakdown": _breakdown(family_counts),
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "top_candidate": top,
        "top_blocker_cohort": top,
        "candidates": clean,
        "blocker_cohorts": clean,
        "activation_candidate_queue": activation_queue,
        "local_activation_candidate_queue": activation_queue,
        "missing_measurements": missing,
        "privacy": _shape_follow_up_privacy(),
    }


def _dashboard_candidate_source_surface(candidate: dict[str, Any], provider: str) -> str:
    source_surface = public_label(candidate.get("source_surface"), "")
    if source_surface:
        return source_surface
    if provider == "anthropic":
        return "anthropic_messages"
    if provider == "openai":
        return "openai_responses"
    return "unknown"


def _dashboard_candidate_endpoint(provider: str, source_surface: str) -> str:
    if source_surface == "anthropic_messages":
        return "messages"
    if source_surface in {"openai_chat", "openai_chat_completions"}:
        return "chat_completions"
    if source_surface in {"openai_responses", "codex_turn"} or provider == "openai":
        return "responses"
    return "unknown"


def _dashboard_candidate_representative_text_chars(candidate: dict[str, Any]) -> int:
    minimum = _as_int(candidate.get("min_text_chars"))
    maximum = _as_int(candidate.get("max_text_chars"))
    if minimum > 0 and maximum > 0:
        return max(1, (minimum + maximum) // 2)
    if maximum > 0:
        return max(1, maximum - 1)
    if minimum > 0:
        return minimum
    return 0


def _dashboard_candidate_sample_count(candidate: dict[str, Any]) -> int:
    weighted = _as_float(candidate.get("sample_weight"))
    if weighted > 0:
        return max(1, int(math.ceil(weighted)))
    return 1


def _dashboard_routing_candidate_rollup_rows(*, limit: int) -> list[dict[str, Any]]:
    try:
        from tokenclaw import routing_experiments

        all_candidates = getattr(routing_experiments, "_all_routing_candidates")()
        is_dashboard_candidate = getattr(routing_experiments, "_is_dashboard_routing_candidate")
    except Exception:
        return []

    groups: dict[str, dict[str, Any]] = {}
    for candidate in all_candidates:
        if not isinstance(candidate, dict) or not is_dashboard_candidate(candidate):
            continue
        provider = public_label(candidate.get("provider"), "unknown")
        requested_model = public_label(candidate.get("requested_model"), "unknown")
        routed_model = public_label(candidate.get("routed_model"), requested_model)
        source_surface = _dashboard_candidate_source_surface(candidate, provider)
        endpoint = _dashboard_candidate_endpoint(provider, source_surface)
        app_family = public_label(
            candidate.get("app_family") or _app_family(provider, source_surface, requested_model),
            "unknown",
        )
        category = public_label(candidate.get("category"), "unknown")
        workflow_phase = public_label(candidate.get("workflow_phase") or category, "unknown")
        stream = bool(candidate.get("stream")) if candidate.get("stream") is not None else False
        has_tools = category.startswith("tool")
        text_chars = _dashboard_candidate_representative_text_chars(candidate)
        input_tokens = _as_int(candidate.get("max_input_tokens") or candidate.get("min_input_tokens"))
        if input_tokens <= 0 and text_chars > 0:
            input_tokens = max(1, text_chars // 4)
        sample_count = _dashboard_candidate_sample_count(candidate)
        basis = {
            "source_surface": source_surface,
            "endpoint": endpoint,
            "provider_family": provider,
            "app_family": app_family,
            "requested_model_family": _model_family(requested_model),
            "routed_model_family": _model_family(routed_model, _model_family(requested_model)),
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": _text_bucket(text_chars),
            "token_bucket": _token_bucket(input_tokens),
            "cache_status": "skipped" if stream or has_tools else "missing",
            "routing_status": "passthrough",
            "file_dependency_status": "missing",
            "file_dependency_fingerprint_available": False,
        }
        rollup_key = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
        group = groups.setdefault(
            rollup_key,
            _new_group(basis, candidate_id=_candidate_id(basis), rollup_key=rollup_key),
        )
        group["source_schema"] = DASHBOARD_ROUTING_CANDIDATE_ROLLUP_SOURCE_SCHEMA
        group["row_count"] += sample_count
        group["input_tokens"] += input_tokens * sample_count
        group["successful_input_tokens"] += input_tokens * sample_count
        _increment(group["status_counts"], "dashboard-candidate-metadata", sample_count)
        _increment(group["candidate_family_counts"], "routing_candidate", sample_count)
        _increment(group["blocker_counts"], "dashboard-routing-candidate-needs-lifecycle-evidence", sample_count)
        _increment(group["cache_reason_counts"], "dashboard-candidate-metadata", sample_count)
        _increment(group["file_dependency_status_counts"], "missing", sample_count)
        _increment(group["file_dependency_fingerprint_availability_counts"], "missing", sample_count)

    rollups = [_finalize_group(group) for group in groups.values()]
    for row in rollups:
        row["source_schema"] = DASHBOARD_ROUTING_CANDIDATE_ROLLUP_SOURCE_SCHEMA
        row["sample_count"] = _as_int(row.get("row_count"))
        row["metadata"]["source"] = "dashboard-routing-candidates"
        row["metadata"]["dashboard_routing_candidate_rollup"] = True
        row["privacy"]["individual_candidate_ids_included"] = False
    rollups.sort(
        key=lambda item: (
            _as_int(item.get("row_count")),
            item.get("provider_family") or "",
            item.get("source_surface") or "",
            item.get("category") or "",
        ),
        reverse=True,
    )
    return rollups[: max(1, min(_as_int(limit, 1000), 10_000))]


def _context_plateau_source_surface(row: dict[str, Any]) -> str:
    source = public_label(row.get("source_surface"), "")
    if source:
        return source
    surfaces = row.get("source_surfaces")
    if isinstance(surfaces, list):
        for item in surfaces:
            if isinstance(item, dict):
                source = public_label(item.get("source_surface"), "")
                if source:
                    return source
    app_family = public_label(row.get("app_family"), "")
    if app_family == "codex":
        return "codex_app"
    if app_family in {"claude", "claude-code", "claude_code"}:
        return "anthropic_messages"
    return "unknown"


def _context_plateau_provider_endpoint(source_surface: str, row: dict[str, Any]) -> tuple[str, str]:
    provider = public_label(row.get("provider_family") or row.get("provider"), "")
    endpoint = public_label(row.get("endpoint"), "")
    if provider and endpoint:
        return provider, endpoint
    if source_surface.startswith("openai"):
        return provider or "openai", endpoint or ("chat" if "chat" in source_surface else "responses")
    if source_surface.startswith("anthropic"):
        return provider or "anthropic", endpoint or "messages"
    if source_surface == "codex_app":
        return provider or "openai", endpoint or "turn_start"
    return provider or "unknown", endpoint or "unknown"


def _context_plateau_model(provider: str, source_surface: str) -> str:
    if provider == "anthropic" or source_surface.startswith("anthropic"):
        return "claude-sonnet"
    if provider == "openai" or source_surface.startswith("openai") or source_surface == "codex_app":
        return "gpt-5"
    return "unknown"


def _context_plateau_rollup_groups(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows = stats.get("context_plateaus")
    if not isinstance(rows, list):
        sessions = stats.get("sessions") if isinstance(stats.get("sessions"), dict) else {}
        rows = sessions.get("context_plateaus") if isinstance(sessions.get("context_plateaus"), list) else []

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        plateau_pairs = _as_int(row.get("plateau_pairs") or row.get("context_plateau_pairs"))
        if plateau_pairs <= 0:
            continue
        median_chars = _as_int(row.get("median_text_chars"))
        p90_chars = _as_int(row.get("p90_text_chars"))
        representative_chars = median_chars or p90_chars
        if representative_chars <= 0:
            continue

        source_surface = _context_plateau_source_surface(row)
        provider, endpoint = _context_plateau_provider_endpoint(source_surface, row)
        app_family = public_label(row.get("app_family"), "unknown")
        category = public_label(
            row.get("category") or ("chat" if source_surface.startswith("openai") else "tool-result"),
            "unknown",
        )
        workflow_phase = public_label(
            row.get("workflow_phase") or ("chat" if category == "chat" else "tool-execution"),
            "unknown",
        )
        stream = bool(row.get("stream")) if row.get("stream") is not None else source_surface.startswith("anthropic")
        has_tools = bool(row.get("has_tools")) if row.get("has_tools") is not None else category.startswith("tool")
        sample_count = max(_as_int(row.get("calls")), plateau_pairs + 1, REPEATED_CONTEXT_CRUNCH_MIN_SAMPLES)
        token_count = max(1, representative_chars // 4)
        total_input_tokens = token_count * sample_count
        repetition_signal = min(1.0, plateau_pairs / float(max(sample_count, 1)))
        cost = _as_float(row.get("cost_usd"))
        model = _context_plateau_model(provider, source_surface)
        input_token_cost = cost if cost > 0 else _input_savings_usd(total_input_tokens, provider=provider, model=model)
        projected_tokens = int(total_input_tokens * REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE * repetition_signal)
        projected_savings = input_token_cost * REPEATED_CONTEXT_CRUNCH_PROJECTION_RATE * repetition_signal
        current_saved_chars = _as_int(row.get("crunch_saved_chars"))
        current_saved_tokens = max(0, current_saved_chars // 4)
        current_savings = _input_savings_usd(
            current_saved_tokens,
            provider=provider,
            model=model,
            fallback_cost=cost,
            fallback_tokens=total_input_tokens,
        )
        basis = {
            "source_surface": source_surface,
            "endpoint": endpoint,
            "provider_family": provider,
            "requested_model_family": model,
            "routed_model_family": model,
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": _text_bucket(representative_chars),
            "token_bucket": _token_bucket(token_count),
            "cache_status": public_label(row.get("cache_status"), "skipped" if stream or has_tools else "miss"),
            "routing_status": public_label(row.get("routing_status"), "passthrough"),
        }
        rollup_key = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
        group = groups.setdefault(
            rollup_key,
            {
                "schema": CONTEXT_PLATEAU_ROLLUP_ROW_SCHEMA,
                "source_schema": "tokenclaw.context_plateau_session_aggregate.v1",
                "rollup_key": rollup_key,
                "candidate_id": _candidate_id(basis),
                **basis,
                "row_count": 0,
                "sample_count": 0,
                "context_plateau_pair_count": 0,
                "context_plateau_session_count": 0,
                "freshness_state": "fresh",
                "source_rollup_freshness_state": "fresh",
                "error_count": 0,
                "retry_count": 0,
                "cache_hit_count": 0,
                "cost_est_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "observed_savings_usd": 0.0,
                "input_tokens": 0,
                "successful_input_tokens": 0,
                "output_tokens": 0,
                "input_token_cost_usd": 0.0,
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "projected_crunch_tokens_saved": 0,
                "projected_crunch_chars_saved": 0,
                "projected_crunch_savings_usd": 0.0,
                "candidate_families": ["crunch_candidate"],
                "candidate_work_classes": ["repeated_context", "crunch"],
                "blocker_codes": [],
                "metadata": {
                    "schema": "tokenclaw.context_plateau_crunch_rollup_metadata.v1",
                    "source": "context_plateaus",
                    "app_family_breakdown": {},
                    "source_surface_breakdown": {},
                    "raw_body_required": False,
                    "aggregate_only": True,
                    "metadata_only": True,
                },
                "privacy": _shape_follow_up_privacy(),
            },
        )
        group["row_count"] += sample_count
        group["sample_count"] += sample_count
        group["context_plateau_pair_count"] += plateau_pairs
        group["context_plateau_session_count"] += 1
        group["cost_est_usd"] += cost
        group["baseline_cost_usd"] += cost
        group["input_tokens"] += total_input_tokens
        group["successful_input_tokens"] += total_input_tokens
        group["input_token_cost_usd"] += input_token_cost
        group["current_crunch_tokens_saved"] += current_saved_tokens
        group["current_crunch_chars_saved"] += current_saved_chars
        group["current_crunch_savings_usd"] += current_savings
        group["projected_crunch_tokens_saved"] += projected_tokens
        group["projected_crunch_chars_saved"] += projected_tokens * 4
        group["projected_crunch_savings_usd"] += projected_savings
        metadata = group["metadata"]
        app_counts = metadata["app_family_breakdown"]
        source_counts = metadata["source_surface_breakdown"]
        app_counts[app_family] = app_counts.get(app_family, 0) + sample_count
        source_counts[source_surface] = source_counts.get(source_surface, 0) + sample_count

    rollups = []
    for group in groups.values():
        group["cost_est_usd"] = round(_as_float(group.get("cost_est_usd")), 6)
        group["baseline_cost_usd"] = round(_as_float(group.get("baseline_cost_usd")), 6)
        group["input_token_cost_usd"] = round(_as_float(group.get("input_token_cost_usd")), 6)
        group["current_crunch_savings_usd"] = round(_as_float(group.get("current_crunch_savings_usd")), 6)
        group["projected_crunch_savings_usd"] = round(_as_float(group.get("projected_crunch_savings_usd")), 6)
        lifecycle = {
            "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
            "cohort_id": _crunch_canary_cohort_id(group),
            "policy_id": _crunch_canary_policy_id(_crunch_canary_cohort_id(group)),
            "applied_count": 0,
            "holdout_count": 0,
            "skipped_count": 0,
            "safety_stopped_count": 0,
            "fallback_count": 0,
            "rollback_count": 0,
            "metadata_only": True,
            "aggregate_only": True,
        }
        group["crunch_canary_lifecycle"] = lifecycle
        metadata = group["metadata"]
        metadata["app_family_breakdown"] = _breakdown(metadata["app_family_breakdown"])
        metadata["source_surface_breakdown"] = _breakdown(metadata["source_surface_breakdown"])
        metadata["candidate_class_breakdown"] = [
            {"value": "repeated_context", "count": _as_int(group.get("row_count"))},
            {"value": "crunch", "count": _as_int(group.get("row_count"))},
        ]
        rollups.append(_snapshot_safe_rollup_row(group, source_schema=CONTEXT_PLATEAU_ROLLUP_ROW_SCHEMA))
    rollups.sort(
        key=lambda item: (
            _as_float(item.get("projected_crunch_savings_usd")),
            _as_int(item.get("projected_crunch_tokens_saved")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    return rollups


def build_context_plateau_crunch_rollup_report(
    stats: dict[str, Any],
    *,
    limit: int = 25,
) -> dict[str, Any] | None:
    if not isinstance(stats, dict):
        return None
    rollups = _context_plateau_rollup_groups(stats)
    if not rollups:
        return None
    capped_limit = max(1, min(_as_int(limit, 25), 100))
    rollups = rollups[:capped_limit]
    follow_up = build_request_shape_follow_up_candidates(rollups, limit=min(capped_limit, 10))
    crunch = build_request_shape_crunch_opportunity_dry_run(rollups, limit=capped_limit)
    summary = {
        "rows_considered": sum(_as_int(row.get("row_count")) for row in rollups),
        "rollup_count": len(rollups),
        "collapsed_rows": 0,
        "metadata_window_backfilled": False,
        "body_rows_read": 0,
        "follow_up_candidate_count": _as_int(follow_up["summary"].get("ranked_candidate_count")),
        "top_next_action": follow_up["summary"].get("top_next_action"),
        "top_local_action_family": follow_up["summary"].get("top_local_action_family"),
        "top_readiness_state": follow_up["summary"].get("top_readiness_state"),
        "projected_crunch_tokens_saved": _as_int(crunch["summary"].get("projected_saved_tokens")),
        "projected_crunch_savings_usd": round(_as_float(crunch["summary"].get("projected_saved_usd")), 8),
        "total_projected_savings_usd": round(_as_float(crunch["summary"].get("projected_saved_usd")), 8),
        "source": "context-plateau-session-aggregates",
        "policy_files_written": False,
    }
    return {
        "schema": CONTEXT_PLATEAU_ROLLUP_SCHEMA,
        "generated_at": utc_now(),
        "source": "context-plateau-session-aggregates",
        "window": {
            "source": "context_plateaus",
        },
        "summary": summary,
        "follow_up_candidates": follow_up,
        "crunch_opportunity_dry_run": crunch,
        "rollups": rollups,
        "privacy": _shape_follow_up_privacy(),
    }


def _persistable_row(
    *,
    run_id: str,
    generated_at: str,
    window_start: str | None,
    window_end: str | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{run_id}:{row['rollup_key']}",
        "run_id": run_id,
        "generated_at": generated_at,
        "window_start": window_start,
        "window_end": window_end,
        "rollup_key": row["rollup_key"],
        "candidate_id": row["candidate_id"],
        "source_surface": row["source_surface"],
        "endpoint": row["endpoint"],
        "provider_family": row["provider_family"],
        "requested_model_family": row["requested_model_family"],
        "routed_model_family": row["routed_model_family"],
        "category": row["category"],
        "workflow_phase": row["workflow_phase"],
        "stream": 1 if row["stream"] else 0,
        "has_tools": 1 if row["has_tools"] else 0,
        "text_bucket": row["text_bucket"],
        "token_bucket": row["token_bucket"],
        "cache_status": row["cache_status"],
        "routing_status": row["routing_status"],
        "candidate_families_json": stable_json(row["candidate_families"]),
        "blocker_codes_json": stable_json(row["blocker_codes"]),
        "row_count": row["row_count"],
        "error_count": row["error_count"],
        "retry_count": row["retry_count"],
        "cache_hit_count": row["cache_hit_count"],
        "cost_est_usd": row["cost_est_usd"],
        "baseline_cost_usd": row["baseline_cost_usd"],
        "observed_savings_usd": row["observed_savings_usd"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "metadata_json": stable_json(row["metadata"]),
    }


def _snapshot_safe_rollup_row(row: dict[str, Any], *, source_schema: str | None = None) -> dict[str, Any]:
    lifecycle = row.get("crunch_canary_lifecycle") if isinstance(row.get("crunch_canary_lifecycle"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "schema": ROLLUP_ROW_SCHEMA,
        "source_schema": source_schema or row.get("source_schema") or row.get("schema") or ROLLUP_ROW_SCHEMA,
        "provider_family": public_label(row.get("provider_family"), "unknown"),
        "source_surface": public_label(row.get("source_surface"), "unknown"),
        "endpoint": public_label(row.get("endpoint"), "unknown"),
        "app_family": public_label(row.get("app_family"), "unknown"),
        "requested_model_family": public_label(row.get("requested_model_family"), "unknown"),
        "routed_model_family": public_label(row.get("routed_model_family"), "unknown"),
        "category": public_label(row.get("category"), "unknown"),
        "workflow_phase": public_label(row.get("workflow_phase"), "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": public_label(row.get("text_bucket"), "unknown"),
        "token_bucket": public_label(row.get("token_bucket"), "unknown"),
        "cache_status": public_label(row.get("cache_status"), "unknown"),
        "routing_status": public_label(row.get("routing_status"), "unknown"),
        "candidate_families": _public_label_list(row.get("candidate_families")),
        "candidate_work_classes": _public_label_list(row.get("candidate_work_classes")),
        "blocker_codes": _public_label_list(row.get("blocker_codes")),
        "freshness_state": public_label(row.get("freshness_state"), "unknown"),
        "source_rollup_freshness_state": public_label(
            row.get("source_rollup_freshness_state") or row.get("freshness_state"),
            "unknown",
        ),
        "context_plateau_pair_count": _as_int(row.get("context_plateau_pair_count")),
        "context_plateau_session_count": _as_int(row.get("context_plateau_session_count")),
        "row_count": _as_int(row.get("row_count")),
        "sample_count": _as_int(row.get("row_count") or row.get("sample_count")),
        "error_count": _as_int(row.get("error_count")),
        "retry_count": _as_int(row.get("retry_count")),
        "cache_hit_count": _as_int(row.get("cache_hit_count")),
        "input_tokens": _as_int(row.get("input_tokens")),
        "successful_input_tokens": _as_int(row.get("successful_input_tokens") or row.get("input_tokens")),
        "output_tokens": _as_int(row.get("output_tokens")),
        "cost_est_usd": round(_as_float(row.get("cost_est_usd")), 6),
        "baseline_cost_usd": round(_as_float(row.get("baseline_cost_usd")), 6),
        "observed_savings_usd": round(_as_float(row.get("observed_savings_usd")), 6),
        "input_token_cost_usd": round(_as_float(row.get("input_token_cost_usd")), 6),
        "current_crunch_tokens_saved": _as_int(row.get("current_crunch_tokens_saved")),
        "current_crunch_chars_saved": _as_int(row.get("current_crunch_chars_saved")),
        "current_crunch_savings_usd": round(_as_float(row.get("current_crunch_savings_usd")), 6),
        "projected_crunch_tokens_saved": _as_int(row.get("projected_crunch_tokens_saved")),
        "projected_crunch_chars_saved": _as_int(row.get("projected_crunch_chars_saved")),
        "projected_crunch_savings_usd": round(_as_float(row.get("projected_crunch_savings_usd")), 6),
        "crunch_canary_lifecycle": {
            "schema": CRUNCH_CANARY_LIFECYCLE_SCHEMA,
            "cohort_id": public_label(lifecycle.get("cohort_id"), "unknown"),
            "policy_id": public_label(lifecycle.get("policy_id"), "unknown"),
            "applied_count": _as_int(lifecycle.get("applied_count")),
            "holdout_count": _as_int(lifecycle.get("holdout_count")),
            "skipped_count": _as_int(lifecycle.get("skipped_count")),
            "safety_stopped_count": _as_int(lifecycle.get("safety_stopped_count")),
            "fallback_count": _as_int(lifecycle.get("fallback_count")),
            "rollback_count": _as_int(lifecycle.get("rollback_count")),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "metadata": {
            "schema": "tokenclaw.request_shape_rollup_metadata.v1",
            "candidate_class_breakdown": metadata.get("candidate_class_breakdown")
            if isinstance(metadata.get("candidate_class_breakdown"), list)
            else [],
            "blocker_breakdown": metadata.get("blocker_breakdown")
            if isinstance(metadata.get("blocker_breakdown"), list)
            else [],
            "aggregate_only": True,
            "raw_body_required": False,
        },
        "privacy": _shape_follow_up_privacy(),
    }


def _snapshot_safe_rollup_rows(report: dict[str, Any], *, limit: int = 25) -> list[dict[str, Any]]:
    source_schema = report.get("schema") or SCHEMA
    rows = [
        _snapshot_safe_rollup_row(row, source_schema=source_schema)
        for row in report.get("rollups") or []
        if isinstance(row, dict)
    ]
    rows.sort(
        key=lambda item: (
            _as_float(item.get("projected_crunch_savings_usd")),
            _as_int(item.get("projected_crunch_tokens_saved")),
            _as_float(item.get("cost_est_usd")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    return rows[: max(1, min(_as_int(limit) or 25, 100))]


def _snapshot_rollup_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = snapshot.get("rollups")
    if not isinstance(rows, list):
        rows = snapshot.get("local_action_cohorts")
    if not isinstance(rows, list):
        return []
    source_schema = snapshot.get("source_schema") or ROLLUP_SNAPSHOT_SCHEMA
    return [
        _snapshot_safe_rollup_row(row, source_schema=str(source_schema))
        for row in rows
        if isinstance(row, dict)
    ]


def _rollup_snapshot_from_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    follow_up = report.get("follow_up_candidates") if isinstance(report.get("follow_up_candidates"), dict) else {}
    follow_up_summary = follow_up.get("summary") if isinstance(follow_up.get("summary"), dict) else {}
    replay = report.get("cache_replayability_dry_run") if isinstance(report.get("cache_replayability_dry_run"), dict) else {}
    replay_summary = replay.get("summary") if isinstance(replay.get("summary"), dict) else {}
    routing_drills = report.get("routing_downgrade_drills") if isinstance(report.get("routing_downgrade_drills"), dict) else {}
    routing_drill_summary = routing_drills.get("summary") if isinstance(routing_drills.get("summary"), dict) else {}
    crunch = report.get("crunch_opportunity_dry_run") if isinstance(report.get("crunch_opportunity_dry_run"), dict) else {}
    crunch_summary = crunch.get("summary") if isinstance(crunch.get("summary"), dict) else {}
    generated_at = str(report.get("generated_at") or utc_now())
    run_id = str(report.get("run_id") or f"shape-rollups-{uuid4().hex[:12]}")
    return {
        "schema": ROLLUP_SNAPSHOT_SCHEMA,
        "snapshot_id": f"{run_id}:request-shape-rollup-snapshot",
        "source_schema": report.get("schema") or SCHEMA,
        "generated_at": generated_at,
        "run_id": run_id,
        "window": report.get("window") if isinstance(report.get("window"), dict) else {},
        "summary": {
            "rows_considered": _as_int(summary.get("rows_considered")),
            "rollup_count": _as_int(summary.get("rollup_count")),
            "rollup_source_count": _as_int(summary.get("rollup_source_count")),
            "top_rollup_source": summary.get("top_rollup_source"),
            "ranked_candidate_count": _as_int(follow_up_summary.get("ranked_candidate_count")),
            "top_next_action": follow_up_summary.get("top_next_action") or summary.get("top_next_action"),
            "top_local_action_family": follow_up_summary.get("top_local_action_family") or summary.get("top_local_action_family"),
            "top_readiness_state": follow_up_summary.get("top_readiness_state"),
            "class_breakdown": follow_up_summary.get("class_breakdown")
            if isinstance(follow_up_summary.get("class_breakdown"), list)
            else [],
            "blocker_breakdown": follow_up_summary.get("blocker_breakdown")
            if isinstance(follow_up_summary.get("blocker_breakdown"), list)
            else [],
            "readiness_breakdown": follow_up_summary.get("readiness_breakdown")
            if isinstance(follow_up_summary.get("readiness_breakdown"), list)
            else [],
            "next_action_breakdown": follow_up_summary.get("next_action_breakdown")
            if isinstance(follow_up_summary.get("next_action_breakdown"), list)
            else [],
            "local_action_family_breakdown": follow_up_summary.get("local_action_family_breakdown")
            if isinstance(follow_up_summary.get("local_action_family_breakdown"), list)
            else [],
            "candidate_family_breakdown": report.get("candidate_family_breakdown")
            if isinstance(report.get("candidate_family_breakdown"), list)
            else [],
            "blocker_code_breakdown": report.get("blocker_code_breakdown")
            if isinstance(report.get("blocker_code_breakdown"), list)
            else [],
            "cache_replayability_replay_ready_cohort_count": _as_int(replay_summary.get("replay_ready_cohort_count")),
            "cache_replayability_skipped_cohort_count": _as_int(replay_summary.get("skipped_cohort_count")),
            "cache_replayability_projected_hits": _as_int(replay_summary.get("projected_hits")),
            "cache_replayability_projected_savings_usd": round(_as_float(replay_summary.get("projected_savings_usd")), 8),
            "routing_downgrade_drill_candidate_count": _as_int(routing_drill_summary.get("candidate_count")),
            "routing_downgrade_drill_review_ready_count": _as_int(routing_drill_summary.get("review_ready_count")),
            "routing_downgrade_drill_top_projected_savings_per_1000_calls_usd": round(
                _as_float(routing_drill_summary.get("top_projected_savings_per_1000_calls_usd")),
                8,
            ),
            "routing_downgrade_drill_total_projected_savings_usd": round(
                _as_float(routing_drill_summary.get("total_projected_savings_usd")),
                8,
            ),
            "projected_crunch_tokens_saved": _as_int(crunch_summary.get("projected_saved_tokens")),
            "projected_crunch_savings_usd": round(_as_float(crunch_summary.get("projected_savings_usd")), 8),
            "total_projected_savings_usd": round(
                _as_float(replay_summary.get("projected_savings_usd"))
                + _as_float(routing_drill_summary.get("total_projected_savings_usd"))
                + _as_float(crunch_summary.get("projected_savings_usd")),
                8,
            ),
            "no_source_traffic_reason": None,
        },
        "rollup_source_declarations": report.get("rollup_source_declarations")
        if isinstance(report.get("rollup_source_declarations"), list)
        else [],
        "rollups": _snapshot_safe_rollup_rows(report),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "absolute_paths_included": False,
            "file_paths_included": False,
            "managed_server_calls_made": False,
            "provider_calls_made": False,
            "policy_files_written": False,
        },
    }


def _snapshot_age_metadata(
    snapshot: dict[str, Any],
    *,
    now: datetime | None = None,
    max_age_hours: float = DEFAULT_ROLLUP_SNAPSHOT_MAX_AGE_HOURS,
) -> dict[str, Any]:
    generated = _parse_utc(snapshot.get("generated_at"))
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_hours = None
    stale = True
    if generated is not None:
        age_hours = round(max(0.0, (now_dt - generated).total_seconds() / 3600.0), 3)
        stale = bool(max_age_hours > 0 and age_hours > max_age_hours)
    return {
        "schema": "tokenclaw.request_shape_rollup_snapshot_freshness.v1",
        "status": "snapshot-stale" if stale else "fresh",
        "stale": stale,
        "age_hours": age_hours,
        "max_age_hours": max_age_hours,
        "reason": "snapshot-stale" if stale else "snapshot-fresh",
    }


def latest_request_shape_rollup_snapshot_report(
    store_obj: Any,
    *,
    now: datetime | None = None,
    max_age_hours: float = DEFAULT_ROLLUP_SNAPSHOT_MAX_AGE_HOURS,
) -> dict[str, Any] | None:
    if not hasattr(store_obj, "latest_request_shape_rollup_snapshot"):
        return None
    snapshot = store_obj.latest_request_shape_rollup_snapshot()
    if not isinstance(snapshot, dict):
        return None
    freshness = _snapshot_age_metadata(snapshot, now=now, max_age_hours=max_age_hours)
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    if _as_int(summary.get("rollup_count")) <= 0 and _as_int(summary.get("ranked_candidate_count")) <= 0:
        return None
    rollups = _snapshot_rollup_rows(snapshot)
    follow_up_candidates = build_request_shape_follow_up_candidates(rollups, limit=10) if rollups else None
    routing_downgrade_drills = build_request_shape_routing_downgrade_drill_report(rollups, limit=25) if rollups else None
    crunch_opportunity_dry_run = build_request_shape_crunch_opportunity_dry_run(rollups, limit=25) if rollups else None
    follow_up_summary = {
        key: summary.get(key)
        for key in (
            "rows_considered",
            "rollup_count",
            "ranked_candidate_count",
            "top_next_action",
            "top_local_action_family",
            "top_readiness_state",
            "class_breakdown",
            "blocker_breakdown",
            "readiness_breakdown",
            "next_action_breakdown",
            "local_action_family_breakdown",
            "no_source_traffic_reason",
        )
    }
    if isinstance(follow_up_candidates, dict) and not freshness["stale"]:
        follow_up_summary = follow_up_candidates["summary"]
    return {
        "schema": SCHEMA,
        "generated_at": snapshot.get("generated_at"),
        "run_id": snapshot.get("run_id"),
        "source": "request-shape-rollup-snapshot",
        "snapshot_reused": not freshness["stale"],
        "snapshot_stale": freshness["stale"],
        "snapshot_freshness": freshness,
        "rollup_snapshot": snapshot,
        "window": snapshot.get("window") if isinstance(snapshot.get("window"), dict) else {},
        "summary": {
            "rows_considered": _as_int(summary.get("rows_considered")),
            "rollup_count": _as_int(summary.get("rollup_count")),
            "rollup_source_count": _as_int(summary.get("rollup_source_count")),
            "top_rollup_source": summary.get("top_rollup_source"),
            "collapsed_rows": 0,
            "metadata_window_backfilled": False,
            "body_rows_read": 0,
            "follow_up_candidate_count": _as_int(summary.get("ranked_candidate_count")),
            "top_next_action": summary.get("top_next_action"),
            "top_local_action_family": summary.get("top_local_action_family"),
            "top_readiness_state": summary.get("top_readiness_state"),
            "snapshot_status": freshness["status"],
            "snapshot_age_hours": freshness["age_hours"],
            "snapshot_max_age_hours": freshness["max_age_hours"],
            "total_projected_savings_usd": _as_float(summary.get("total_projected_savings_usd")),
            "snapshot_rehydrated_rollup_count": len(rollups),
            "snapshot_rehydrated_crunch_candidate_count": _as_int(
                (crunch_opportunity_dry_run or {}).get("summary", {}).get("candidate_count")
            ),
            "routing_downgrade_drill_candidate_count": _as_int(
                (routing_downgrade_drills or {}).get("summary", {}).get("candidate_count")
            ),
            "routing_downgrade_drill_review_ready_count": _as_int(
                (routing_downgrade_drills or {}).get("summary", {}).get("review_ready_count")
            ),
        },
        "candidate_family_breakdown": summary.get("candidate_family_breakdown")
        if isinstance(summary.get("candidate_family_breakdown"), list)
        else [],
        "blocker_code_breakdown": summary.get("blocker_code_breakdown")
        if isinstance(summary.get("blocker_code_breakdown"), list)
        else [],
        "rollup_source_declarations": snapshot.get("rollup_source_declarations")
        if isinstance(snapshot.get("rollup_source_declarations"), list)
        else [],
        "follow_up_candidates": follow_up_candidates
        if isinstance(follow_up_candidates, dict) and not freshness["stale"]
        else {
            "schema": FOLLOW_UP_CANDIDATES_SCHEMA,
            "status": "snapshot-stale" if freshness["stale"] else "snapshot-reused",
            "summary": follow_up_summary,
            "top_candidate": None,
            "top_blocker_cohort": None,
            "candidates": [],
            "blocker_cohorts": [],
            "missing_measurements": ["snapshot-stale"] if freshness["stale"] else [],
            "privacy": _shape_follow_up_privacy(),
        },
        "crunch_opportunity_dry_run": crunch_opportunity_dry_run
        if isinstance(crunch_opportunity_dry_run, dict) and not freshness["stale"]
        else None,
        "routing_downgrade_drills": routing_downgrade_drills
        if isinstance(routing_downgrade_drills, dict) and not freshness["stale"]
        else {
            "schema": ROUTING_DOWNGRADE_DRILL_SCHEMA,
            "status": "snapshot-stale" if freshness["stale"] else "no-routing-downgrade-drill-candidates",
            "summary": {
                "candidate_count": 0,
                "review_ready_count": 0,
                "blocked_count": 0,
                "top_blocker_code": "snapshot-stale" if freshness["stale"] else None,
                "provider_calls_made": 0,
                "managed_server_calls_made": 0,
                "policy_files_written": False,
            },
            "candidates": [],
            "privacy": _routing_downgrade_privacy(),
        },
        "rollups": rollups if not freshness["stale"] else [],
        "privacy": snapshot.get("privacy") if isinstance(snapshot.get("privacy"), dict) else _shape_follow_up_privacy(),
    }


def build_request_shape_rollups_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    persist: bool = True,
    run_id: str | None = None,
    max_crunch_canary_evidence_age_hours: float = DEFAULT_CRUNCH_CANARY_MAX_EVIDENCE_AGE_HOURS,
    mark_handled_cache_replay_cohorts: bool = True,
    managed_preview_outcomes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    generated_at = utc_now()
    run_id = run_id or f"shape-rollups-{uuid4().hex[:12]}"
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, crunch_json,
                   routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    metadata_window_backfilled = False
    dashboard_candidate_backfilled = False
    if not rows:
        rows = _codex_metadata_window_rows(store_obj, limit=capped_limit)
        metadata_window_backfilled = bool(rows)
    dashboard_candidate_rollups: list[dict[str, Any]] = []
    if not rows:
        dashboard_candidate_rollups = _dashboard_routing_candidate_rollup_rows(limit=capped_limit)
        dashboard_candidate_backfilled = bool(dashboard_candidate_rollups)

    groups: dict[str, dict[str, Any]] = {}
    provider_counts: dict[str, int] = {}
    candidate_family_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    body_rows_read = 0
    window_start: str | None = None
    window_end: str | None = None
    impact_rows: list[dict[str, Any]] = []

    for row in rows:
        created_at = str(row.get("created_at") or "")
        if created_at:
            window_start = created_at if window_start is None else min(window_start, created_at)
            window_end = created_at if window_end is None else max(window_end, created_at)
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        crunch = _json_obj(row.get("crunch_json"))
        provider = _provider_family(row)
        endpoint = _endpoint(row)
        source_surface = _source_surface(row, provider, endpoint)
        requested_family = public_label(row.get("requested_model_family"), "") or _model_family(row.get("requested_model"))
        routed_family = public_label(row.get("routed_model_family"), "") or _model_family(
            row.get("routed_model"),
            requested_family,
        )
        category = public_label(row.get("category") or routing.get("category"), "unknown")
        workflow_phase = _workflow_phase(row, routing)
        stream = bool(_as_int(row.get("stream")))
        has_tools = _has_tools(row, routing, cache)
        text_chars = _as_int(routing.get("text_chars"))
        input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        if text_chars <= 0 and input_tokens > 0:
            text_chars = input_tokens * 4
        projection_input_tokens = max(input_tokens, text_chars // 4 if text_chars > 0 else 0)
        output_tokens = _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))
        cost = _as_float(row.get("cost_est_usd"))
        baseline = _as_float(row.get("cost_baseline_usd"))
        observed_savings = max(0.0, baseline - cost)
        status_bucket = _status_bucket(row.get("status_code"))
        current_crunch_tokens = _crunch_saved_tokens(crunch)
        current_crunch_chars = _crunch_saved_chars(crunch)
        crunch_canary_lifecycle = _crunch_canary_lifecycle_from_meta(crunch)
        crunch_model = str(row.get("routed_model") or row.get("requested_model") or "")
        current_crunch_savings = _input_savings_usd(
            current_crunch_tokens,
            provider=provider,
            model=crunch_model,
            fallback_cost=cost,
            fallback_tokens=input_tokens,
        )
        input_token_cost = _input_savings_usd(
            projection_input_tokens,
            provider=provider,
            model=crunch_model,
            fallback_cost=cost,
            fallback_tokens=input_tokens or projection_input_tokens,
        )
        cache_status = _cache_status(row, cache)
        cache_reason = public_label(cache.get("reason"), "unknown")
        file_dependency_audit = _sanitized_file_dependency_audit(cache)
        file_dependency_status = _file_dependency_status(file_dependency_audit)
        file_dependency_fingerprint_available = _file_dependency_fingerprint_available(cache)
        routing_status = _routing_status(row, routing)
        blockers = _blocker_codes(
            row=row,
            cache=cache,
            routing=routing,
            cache_status=cache_status,
            routing_status=routing_status,
            stream=stream,
            has_tools=has_tools,
            file_dependency_status=file_dependency_status,
        )
        candidate_families = _candidate_families(
            cache_status=cache_status,
            routing_status=routing_status,
            blockers=blockers,
            observed_savings=observed_savings,
            cost=cost,
        )
        basis = {
            "source_surface": source_surface,
            "endpoint": endpoint,
            "provider_family": provider,
            "requested_model_family": requested_family,
            "routed_model_family": routed_family,
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": _text_bucket(text_chars),
            "token_bucket": _token_bucket(input_tokens),
            "cache_status": cache_status,
            "routing_status": routing_status,
            "file_dependency_status": file_dependency_status,
            "file_dependency_fingerprint_available": file_dependency_fingerprint_available,
        }
        impact_rows.append(
            {
                **row,
                **basis,
                "text_chars": text_chars,
                "input_tokens": input_tokens,
            }
        )
        rollup_key = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
        candidate_id = _candidate_id(basis)
        group = groups.setdefault(rollup_key, _new_group(basis, candidate_id=candidate_id, rollup_key=rollup_key))
        group["row_count"] += 1
        group["error_count"] += int(_status_bucket(row.get("status_code")) in {"4xx", "5xx"})
        group["retry_count"] += _as_int(row.get("retry_count"))
        group["cache_hit_count"] += int(_as_int(row.get("cache_hit")) > 0 or cache_status == "hit")
        group["cost_est_usd"] += cost
        group["baseline_cost_usd"] += baseline
        group["observed_savings_usd"] += observed_savings
        group["input_tokens"] += input_tokens
        group["output_tokens"] += output_tokens
        if status_bucket in {"2xx", "3xx"}:
            group["successful_input_tokens"] += projection_input_tokens
            group["input_token_cost_usd"] += input_token_cost
        group["current_crunch_tokens_saved"] += current_crunch_tokens
        group["current_crunch_chars_saved"] += current_crunch_chars
        group["current_crunch_savings_usd"] += current_crunch_savings
        group["file_dependency_audit"] = _merge_file_dependency_audit(
            group.get("file_dependency_audit"),
            file_dependency_audit,
        )
        _increment(provider_counts, provider)
        _increment(group["status_counts"], status_bucket)
        _increment(group["retry_bucket_counts"], _retry_bucket(_as_int(row.get("retry_count"))))
        _increment(group["cost_bucket_counts"], _cost_bucket(cost))
        _increment(group["savings_bucket_counts"], _savings_bucket(observed_savings))
        _increment(group["cache_reason_counts"], cache_reason)
        _increment(group["file_dependency_status_counts"], file_dependency_status)
        _increment(
            group["file_dependency_fingerprint_availability_counts"],
            "available" if file_dependency_fingerprint_available else "missing",
        )
        for family in candidate_families:
            _increment(candidate_family_counts, family)
            _increment(group["candidate_family_counts"], family)
        for blocker in blockers:
            _increment(blocker_counts, blocker)
            _increment(group["blocker_counts"], blocker)
        if crunch_canary_lifecycle:
            status = str(crunch_canary_lifecycle.get("status") or "unknown")
            _increment(group["crunch_canary_lifecycle_counts"], status)
            policy_id = str(crunch_canary_lifecycle.get("policy_id") or "unknown")
            _increment(group["crunch_canary_policy_counts"], policy_id)

    rollups = dashboard_candidate_rollups or [_finalize_group(group) for group in groups.values()]
    rollups.sort(
        key=lambda item: (
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("cost_est_usd")),
            _as_int(item.get("row_count")),
            item.get("candidate_id") or "",
        ),
        reverse=True,
    )
    if dashboard_candidate_backfilled:
        provider_counts = {}
        candidate_family_counts = {}
        blocker_counts = {}
        for row in rollups:
            row_count = _as_int(row.get("row_count") or row.get("sample_count"))
            _increment(provider_counts, row.get("provider_family"), row_count)
            for family in row.get("candidate_families") or []:
                _increment(candidate_family_counts, family, row_count)
            for blocker in row.get("blocker_codes") or []:
                _increment(blocker_counts, blocker, row_count)
    persistable = [
        _persistable_row(
            run_id=run_id,
            generated_at=generated_at,
            window_start=window_start,
            window_end=window_end,
            row=row,
        )
        for row in rollups
    ]
    persisted_count = 0
    if persist and hasattr(store_obj, "persist_request_shape_rollups"):
        persisted_count = store_obj.persist_request_shape_rollups(
            run_id=run_id,
            generated_at=generated_at,
            rows=persistable,
        )
    handled_cache_replay_rules = (
        _request_shape_cache_replay_handled_policy_rules()
        if mark_handled_cache_replay_cohorts
        else []
    )
    cache_replayability_dry_run = build_request_shape_cache_replayability_dry_run(
        rollups,
        limit=25,
        handled_policy_rules=handled_cache_replay_rules,
        managed_preview_outcomes=managed_preview_outcomes,
    )
    cache_replay_blocker_classification = build_request_shape_cache_replay_blocker_classification_report(
        cache_replayability_dry_run,
        limit=25,
    )
    if dashboard_candidate_backfilled:
        source = "dashboard-routing-candidates"
    elif metadata_window_backfilled:
        source = "recent-local-metadata-window-backfill"
    else:
        source = "recent-local-call-metadata"
    follow_up_candidates = build_request_shape_follow_up_candidates(rollups, limit=10, source=source)
    crunch_opportunity_dry_run = build_request_shape_crunch_opportunity_dry_run(
        follow_up_candidates["activation_candidate_queue"],
        limit=25,
    )
    crunch_canary_impact = build_request_shape_crunch_canary_impact_report(
        impact_rows,
        max_evidence_age_hours=max_crunch_canary_evidence_age_hours,
        opportunity_report=crunch_opportunity_dry_run,
    )
    crunch_policy_decision = build_request_shape_crunch_policy_decision_report(crunch_canary_impact)
    crunch_activation_evidence = build_request_shape_crunch_activation_evidence_report(
        crunch_policy_decision=crunch_policy_decision,
        crunch_canary_impact=crunch_canary_impact,
    )
    routing_downgrade_drills = build_request_shape_routing_downgrade_drill_report(rollups, limit=25)
    phase_aware_routing_dry_run = build_request_shape_phase_aware_routing_dry_run(rollups, limit=25)
    remaining_crunch_measurements = build_request_shape_crunch_remaining_measurement_report(
        follow_up_candidates=follow_up_candidates,
        crunch_opportunity=crunch_opportunity_dry_run,
        activation_evidence=crunch_activation_evidence,
        limit=10,
    )
    repeated_context_impact_rows = build_request_shape_crunch_canary_impact_rows_report(
        impact_candidates=crunch_canary_impact.get("candidates") if isinstance(crunch_canary_impact.get("candidates"), list) else [],
        activation_ready_measurements=crunch_canary_impact.get("activation_ready_measurements")
        if isinstance(crunch_canary_impact.get("activation_ready_measurements"), dict)
        else {},
        follow_up_candidates=follow_up_candidates,
        activation_evidence=crunch_activation_evidence,
    )
    crunch_canary_impact = dict(crunch_canary_impact)
    crunch_canary_impact["repeated_context_impact_rows"] = repeated_context_impact_rows
    if isinstance(crunch_canary_impact.get("summary"), dict):
        crunch_canary_impact["summary"] = dict(crunch_canary_impact["summary"])
        crunch_canary_impact["summary"]["repeated_context_impact_row_count"] = repeated_context_impact_rows["summary"]["ranked_row_count"]
        crunch_canary_impact["summary"]["repeated_context_measured_count"] = repeated_context_impact_rows["summary"]["measured_count"]
        crunch_canary_impact["summary"]["repeated_context_measurement_required_count"] = repeated_context_impact_rows["summary"]["measurement_required_count"]
        crunch_canary_impact["summary"]["repeated_context_blocked_count"] = repeated_context_impact_rows["summary"]["blocked_count"]
        crunch_canary_impact["summary"]["repeated_context_superseded_count"] = repeated_context_impact_rows["summary"]["superseded_count"]

    rows_considered = (
        sum(_as_int(row.get("row_count") or row.get("sample_count")) for row in rollups)
        if dashboard_candidate_backfilled
        else len(rows)
    )
    source_declarations = [
        _request_shape_rollup_source_declaration(
            source=source,
            status=follow_up_candidates["source_traffic_acquisition_status"],
            rows_considered=rows_considered,
            rollups=rollups,
            acquisition_reason=follow_up_candidates["summary"].get("no_source_traffic_reason"),
        )
    ]
    follow_up_candidates = dict(follow_up_candidates)
    follow_up_candidates["rollup_source_declarations"] = source_declarations
    if isinstance(follow_up_candidates.get("summary"), dict):
        follow_up_candidates["summary"] = dict(follow_up_candidates["summary"])
        follow_up_candidates["summary"]["rollup_source_count"] = len(source_declarations)
        follow_up_candidates["summary"]["top_rollup_source"] = source_declarations[0]["source"] if source_declarations else None
    report = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "run_id": run_id,
        "limit": capped_limit,
        "persisted": bool(persisted_count),
        "persisted_count": persisted_count,
        "window": {
            "start": window_start,
            "end": window_end,
            "source": source,
        },
        "summary": {
            "rows_considered": rows_considered,
            "rollup_count": len(rollups),
            "collapsed_rows": max(0, rows_considered - len(rollups)),
            "rollup_source_count": len(source_declarations),
            "top_rollup_source": source_declarations[0]["source"] if source_declarations else None,
            "metadata_window_backfilled": metadata_window_backfilled,
            "metadata_window_backfill_rows": len(rows) if metadata_window_backfilled else 0,
            "dashboard_candidate_backfilled": dashboard_candidate_backfilled,
            "dashboard_candidate_backfill_rows": rows_considered if dashboard_candidate_backfilled else 0,
            "source_traffic_acquisition_status": follow_up_candidates["source_traffic_acquisition_status"],
            "source_traffic_acquisition_attempted": True,
            "total_cost_est_usd": round(sum(_as_float(row.get("cost_est_usd")) for row in rollups), 6),
            "total_baseline_cost_usd": round(sum(_as_float(row.get("baseline_cost_usd")) for row in rollups), 6),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in rollups), 6),
            "body_rows_read": body_rows_read,
            "follow_up_candidate_count": follow_up_candidates["summary"]["ranked_candidate_count"],
            "top_next_action": follow_up_candidates["summary"]["top_next_action"],
            "top_local_action_family": follow_up_candidates["summary"]["top_local_action_family"],
            "routing_downgrade_drill_candidate_count": routing_downgrade_drills["summary"]["candidate_count"],
            "routing_downgrade_drill_review_ready_count": routing_downgrade_drills["summary"]["review_ready_count"],
            "routing_downgrade_drill_top_projected_savings_per_1000_calls_usd": routing_downgrade_drills["summary"][
                "top_projected_savings_per_1000_calls_usd"
            ],
            "phase_aware_routing_candidate_count": phase_aware_routing_dry_run["summary"]["candidate_count"],
            "phase_aware_routing_review_ready_count": phase_aware_routing_dry_run["summary"]["review_ready_count"],
            "phase_aware_routing_stageable_rule_section_count": phase_aware_routing_dry_run["summary"][
                "stageable_rule_section_count"
            ],
            "phase_aware_routing_top_projected_savings_per_1000_calls_usd": phase_aware_routing_dry_run["summary"][
                "top_projected_savings_per_1000_calls_usd"
            ],
            "remaining_crunch_measurement_count": remaining_crunch_measurements["summary"]["remaining_measurement_required_count"],
        },
        "provider_breakdown": _breakdown(provider_counts),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_code_breakdown": _breakdown(blocker_counts),
        "rollup_source_declarations": source_declarations,
        "source_traffic_acquisition": follow_up_candidates["source_traffic_acquisition"],
        "follow_up_candidates": follow_up_candidates,
        "routing_downgrade_drills": routing_downgrade_drills,
        "phase_aware_routing_dry_run": phase_aware_routing_dry_run,
        "cache_replayability_dry_run": cache_replayability_dry_run,
        "cache_replay_blocker_classification": cache_replay_blocker_classification,
        "crunch_opportunity_dry_run": crunch_opportunity_dry_run,
        "crunch_canary_impact": crunch_canary_impact,
        "crunch_policy_decision": crunch_policy_decision,
        "crunch_activation_evidence": crunch_activation_evidence,
        "remaining_crunch_measurements": remaining_crunch_measurements,
        "rollups": rollups,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "tenant_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
    snapshot = _rollup_snapshot_from_report(report)
    report["rollup_snapshot"] = snapshot
    if (
        persist
        and hasattr(store_obj, "persist_request_shape_rollup_snapshot")
        and (
            _as_int(snapshot.get("summary", {}).get("rollup_count")) > 0
            or _as_int(snapshot.get("summary", {}).get("ranked_candidate_count")) > 0
        )
    ):
        report["snapshot_persisted_count"] = store_obj.persist_request_shape_rollup_snapshot(snapshot)
    else:
        report["snapshot_persisted_count"] = 0
    report["snapshot_persisted"] = bool(report["snapshot_persisted_count"])
    return report
