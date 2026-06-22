"""Bounded post-rollback cache reobserve report.

After a preview-verified stale cache replay rollback is applied (see
``local_activation_executor.apply_local_activation_executor_bundle`` and issue
#844), the loop needs a bounded way to decide what happens to each affected
request-shape cohort next: retire it as no-repeat, restage a fresh cache replay
canary, or keep it blocked behind a narrow invalidation reason.

This module reads aggregate/metadata-only local call rows for the affected
request shapes and emits exactly one durable successor decision per cohort. It
never applies cache patches, never writes cache entries, never makes provider or
managed-server calls, and never surfaces raw prompts, provider bodies, request
IDs, cache keys, session IDs, or file paths.

The report also carries a ``next_research_plan`` block whose duplicate
suppression retires the predecessor stale cache rollback issue once a stale
no-traffic cohort has been retired, so research stops re-proposing the same
stale rollback work.
"""

from __future__ import annotations

import json
from typing import Any

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.cache_replay_post_rollback_reobserve_report.v1"
COHORT_DECISION_SCHEMA = "tokenclaw.cache_replay_post_rollback_successor_decision.v1"
OBSERVATION_SCHEMA = "tokenclaw.cache_replay_post_rollback_observation.v1"
NEXT_PLAN_SCHEMA = "tokenclaw.cache_replay_post_rollback_next_research_plan.v1"
SUPPRESSION_SCHEMA = (
    "tokenclaw.request_shape_cache_replay_stale_no_traffic_retirement_duplicate_suppression.v1"
)

TARGET_LOCAL_RULE_FILE = "cache_rules.yaml"
TARGET_LOCAL_POLICY_SECTION = "cache.pattern_rules"

DEFAULT_MAX_OBSERVATION_AGE_HOURS = 72.0
DEFAULT_RESTAGE_MIN_REPEATS = 2

# Successor decisions (exactly one is emitted per cohort).
DECISION_RETIRE = "retire-staged-no-repeat"
DECISION_RESTAGE = "restage-cache-replay-canary"
DECISION_KEEP_BLOCKED = "keep-blocked"

# Reobserve cohort states.
STATE_NO_FRESH_TRAFFIC = "no-fresh-traffic"
STATE_FRESH_REPEAT_NO_HIT_PROOF = "fresh-repeat-no-hit-proof"
STATE_HIT_RECOVERY_PROOF = "hit-recovery-proof"
STATE_INVALIDATION_BLOCKER = "invalidation-blocker"
STATE_RESTAGE_READY = "restage-ready"
STATE_REOBSERVE_WINDOW_OPEN = "reobserve-window-open"

_INVALIDATION_HINTS = (
    "invalidat",
    "dependency-changed",
    "dependency_changed",
    "stale-dependency",
    "fingerprint-mismatch",
)


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return default


def _blocker_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("invalidation_reason") or ""),
        str(row.get("blocking_reason") or ""),
        str(row.get("reason") or ""),
    ]
    codes = row.get("blocker_codes")
    if isinstance(codes, list):
        parts.extend(str(item) for item in codes)
    return " ".join(parts).lower()


def _has_invalidation_blocker(row: dict[str, Any]) -> bool:
    if _as_bool(_first(row, "invalidation_blocked", "dependency_invalidated", default=False)):
        return True
    text = _blocker_text(row)
    return any(hint in text for hint in _INVALIDATION_HINTS)


def _cohort_fingerprint(row: dict[str, Any]) -> str:
    basis = _first(
        row,
        "request_shape_fingerprint",
        "cohort_fingerprint",
        "shape_fingerprint",
        "cohort_label",
        "request_shape_label",
        "shape",
        "cohort",
        default="unknown",
    )
    return public_id(basis, prefix="cache-replay-cohort", fallback="cache-replay-cohort:unknown")


def _cohort_label(row: dict[str, Any]) -> str:
    basis = _first(
        row,
        "cohort_label",
        "request_shape_label",
        "shape",
        "cohort",
        default="cache-replay-cohort",
    )
    return public_label(basis, fallback="cache-replay-cohort")


def _observed_row_count(row: dict[str, Any], applied: int, holdout: int, miss: int, warmup: int, hits: int) -> int:
    explicit = _first(row, "observed_row_count", "row_count", "reobserve_row_count")
    if explicit is not None:
        return max(0, _as_int(explicit))
    return max(0, applied + holdout + miss + warmup + hits)


def _classify(row: dict[str, Any], *, max_age_hours: float, restage_min_repeats: int) -> dict[str, Any]:
    applied = _as_int(_first(row, "applied_count", default=0))
    holdout = _as_int(_first(row, "holdout_count", default=0))
    miss = _as_int(_first(row, "miss_count", default=0))
    warmup = _as_int(_first(row, "warmup_miss_count", default=0))
    hits = _as_int(_first(row, "exact_hit_count", "observed_hits", "actual_hits", default=0))
    savings = _as_float(_first(row, "observed_savings_usd", "observed_savings", default=0.0))
    repeats = _as_int(
        _first(row, "fresh_repeat_count", "repeat_count", "repeated_shape_count", default=0)
    )
    observed_rows = _observed_row_count(row, applied, holdout, miss, warmup, hits)
    if repeats <= 0 and observed_rows > 0:
        # Repeats default to observed rows beyond the first call in the cohort.
        repeats = max(0, observed_rows - 1)

    age_hours = _as_float(
        _first(row, "observation_age_hours", "evidence_age_hours", "reobserve_age_hours", default=0.0)
    )
    row_max_age = _first(row, "max_observation_age_hours", "max_evidence_age_hours")
    effective_max_age = _as_float(row_max_age, max_age_hours) if row_max_age is not None else max_age_hours
    window_elapsed = bool(age_hours >= effective_max_age) if effective_max_age > 0 else False

    observation = {
        "applied_count": applied,
        "holdout_count": holdout,
        "miss_count": miss,
        "warmup_miss_count": warmup,
        "exact_hit_count": hits,
        "observed_row_count": observed_rows,
        "fresh_repeat_count": repeats,
        "observed_savings_usd": round(savings, 6),
        "observation_age_hours": round(age_hours, 3),
        "max_observation_age_hours": round(effective_max_age, 3),
        "observation_window_elapsed": window_elapsed,
    }

    # Priority order keeps the decision deterministic and emits exactly one per cohort.
    if _has_invalidation_blocker(row):
        return {
            "decision": DECISION_KEEP_BLOCKED,
            "state": STATE_INVALIDATION_BLOCKER,
            "next_action": "keep-cache-replay-successor-blocked-on-invalidation",
            "reason": "post-rollback-reobserve-invalidation-blocker",
            "blocker_codes": ["cache-replay-invalidation-blocker"],
            "terminal": False,
            "observation": observation,
        }

    if hits >= 1 and savings > 0.0:
        return {
            "decision": DECISION_RESTAGE,
            "state": STATE_HIT_RECOVERY_PROOF,
            "next_action": "restage-cache-replay-canary-from-fresh-hit-recovery",
            "reason": "post-rollback-reobserve-fresh-hit-recovery-proof",
            "blocker_codes": [],
            "terminal": False,
            "observation": observation,
        }

    if observed_rows <= 0:
        if window_elapsed:
            return {
                "decision": DECISION_RETIRE,
                "state": STATE_NO_FRESH_TRAFFIC,
                "next_action": "retire-stale-cache-replay-successor-no-traffic",
                "reason": "post-rollback-observation-window-elapsed-no-traffic",
                "blocker_codes": ["post-rollback-observation-window-elapsed-no-traffic"],
                "terminal": True,
                "observation": observation,
            }
        return {
            "decision": DECISION_KEEP_BLOCKED,
            "state": STATE_REOBSERVE_WINDOW_OPEN,
            "next_action": "reobserve-cache-replay-after-rollback",
            "reason": "post-rollback-reobserve-window-open-no-traffic-yet",
            "blocker_codes": ["post-rollback-reobserve-window-open"],
            "terminal": False,
            "observation": observation,
        }

    if window_elapsed and repeats >= restage_min_repeats:
        return {
            "decision": DECISION_RESTAGE,
            "state": STATE_RESTAGE_READY,
            "next_action": "restage-cache-replay-canary-from-fresh-repeat-evidence",
            "reason": "post-rollback-reobserve-mature-fresh-repeat-evidence",
            "blocker_codes": [],
            "terminal": False,
            "observation": observation,
        }

    return {
        "decision": DECISION_KEEP_BLOCKED,
        "state": STATE_FRESH_REPEAT_NO_HIT_PROOF,
        "next_action": "reobserve-cache-replay-after-rollback",
        "reason": "post-rollback-reobserve-fresh-repeat-without-hit-proof",
        "blocker_codes": ["cache-replay-fresh-repeat-without-hit-proof"],
        "terminal": False,
        "observation": observation,
    }


def _decision_record(row: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    observation = dict(classification["observation"])
    observation["schema"] = OBSERVATION_SCHEMA
    observation["metadata_only"] = True
    observation["aggregate_only"] = True
    observation["emits_cache_apply_action"] = False
    observation["cache_apply_action_count"] = 0
    observation["cache_entries_written"] = 0
    observation["policy_files_written"] = False
    return {
        "schema": COHORT_DECISION_SCHEMA,
        "cohort_fingerprint": _cohort_fingerprint(row),
        "cohort_label": _cohort_label(row),
        "successor_decision": classification["decision"],
        "state": classification["state"],
        "next_action": classification["next_action"],
        "reason": classification["reason"],
        "blocker_codes": list(classification["blocker_codes"]),
        "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
        "target_local_policy_section": TARGET_LOCAL_POLICY_SECTION,
        "durable_action_ledger_entry": True,
        "terminal_successor_state": bool(classification["terminal"]),
        "stale_no_traffic_retirement": classification["decision"] == DECISION_RETIRE,
        "observation": observation,
        # Hard guarantees: this report never touches cache apply paths.
        "emits_cache_apply_action": False,
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": counter[key]}
        for key in sorted(counter, key=lambda item: (-counter[item], item))
    ]


def _next_research_plan(
    decisions: list[dict[str, Any]],
    *,
    retire_count: int,
) -> dict[str, Any]:
    retired_cohorts = [
        decision["cohort_fingerprint"]
        for decision in decisions
        if decision["successor_decision"] == DECISION_RETIRE
    ]
    suppresses_predecessor = retire_count >= 1
    suppression = {
        "schema": SUPPRESSION_SCHEMA,
        "reason": "rollback-stale-no-traffic-retired",
        "active": suppresses_predecessor,
        "metadata_only": True,
        "aggregate_only": True,
        "suppresses_generic_cache_replay_activation_issue": suppresses_predecessor,
        "suppresses_generic_replay_ready_issue": suppresses_predecessor,
        "suppresses_new_cache_replay_stage_issue": suppresses_predecessor,
        "suppresses_duplicate_successor_issue": suppresses_predecessor,
        "suppresses_closed_stage_replay_predecessor_titles": suppresses_predecessor,
        "suppressed_predecessor_next_actions": [
            "apply-cache-replay-rollback-before-reobserve",
            "rollback-cache-replay-rule",
            "reobserve-cache-replay-after-rollback",
            "stage-cache-replay-canary",
            "turn-cache-candidate-into-local-replay-evidence",
        ],
        "suppressed_predecessor_title_families": [
            "Apply preview-verified cache rollback patches through the local activation executor",
            "Run post-rollback cache reobserve windows and emit retire-or-restage decisions",
            "Stage cache replay canary from evidence-to-activation ledger",
            "Turn evidence-older-than-max-age cache candidate into local replay evidence",
            "Keep cache activation successor blocked on evidence-older-than-max-age",
        ],
        "retired_cohort_fingerprints": retired_cohorts,
        "retired_cohort_count": len(retired_cohorts),
        "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
        "target_local_policy_section": TARGET_LOCAL_POLICY_SECTION,
    }
    return {
        "schema": NEXT_PLAN_SCHEMA,
        "suppresses_predecessor_stale_rollback_issue": suppresses_predecessor,
        "duplicate_suppression": suppression,
    }


def build_cache_replay_post_rollback_reobserve_report(
    rows: Any,
    *,
    max_observation_age_hours: float = DEFAULT_MAX_OBSERVATION_AGE_HOURS,
    restage_min_repeats: int = DEFAULT_RESTAGE_MIN_REPEATS,
    now: str | None = None,
) -> dict[str, Any]:
    """Build the bounded post-rollback reobserve report from cohort metadata rows.

    ``rows`` is an iterable of aggregate, metadata-only cohort observation rows
    (one per affected request shape). Each row may carry counts such as
    ``applied_count``/``holdout_count``/``exact_hit_count``, ``fresh_repeat_count``,
    ``observed_savings_usd``, the reobserve window age, and an optional
    invalidation marker. Exactly one retire/restage/keep-blocked decision is
    emitted per cohort row, deduplicated by stable cohort fingerprint.
    """

    max_age = _as_float(max_observation_age_hours, DEFAULT_MAX_OBSERVATION_AGE_HOURS)
    if max_age <= 0:
        max_age = DEFAULT_MAX_OBSERVATION_AGE_HOURS
    min_repeats = max(1, _as_int(restage_min_repeats, DEFAULT_RESTAGE_MIN_REPEATS))

    decisions: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    state_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}

    for raw in rows or []:
        row = _json_obj(raw)
        if not row:
            continue
        classification = _classify(row, max_age_hours=max_age, restage_min_repeats=min_repeats)
        record = _decision_record(row, classification)
        fingerprint = record["cohort_fingerprint"]
        if fingerprint in seen_fingerprints:
            # One durable decision per cohort: keep the first deterministic record.
            continue
        seen_fingerprints.add(fingerprint)
        decisions.append(record)
        state_counts[record["state"]] = state_counts.get(record["state"], 0) + 1
        decision_counts[record["successor_decision"]] = (
            decision_counts.get(record["successor_decision"], 0) + 1
        )

    retire_count = decision_counts.get(DECISION_RETIRE, 0)
    restage_count = decision_counts.get(DECISION_RESTAGE, 0)
    keep_blocked_count = decision_counts.get(DECISION_KEEP_BLOCKED, 0)

    report = {
        "schema": SCHEMA,
        "generated_at": now or utc_now(),
        "max_observation_age_hours": round(max_age, 3),
        "restage_min_repeats": min_repeats,
        "summary": {
            "cohort_count": len(decisions),
            "retire_count": retire_count,
            "restage_count": restage_count,
            "keep_blocked_count": keep_blocked_count,
            "decisions_per_cohort": 1,
            "cache_apply_action_count": 0,
            "cache_entries_written": 0,
            "emits_cache_apply_action": False,
            "policy_files_written": False,
            "state_breakdown": _breakdown(state_counts),
            "decision_breakdown": _breakdown(decision_counts),
        },
        "cohorts": decisions,
        "next_research_plan": _next_research_plan(decisions, retire_count=retire_count),
        "privacy": {
            "schema": "tokenclaw.cache_replay_post_rollback_reobserve_privacy.v1",
            "telemetry_profile": "metadata-only",
            "local_only": True,
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "aggregate local call metadata for affected request shapes only",
        },
    }

    # Defensive: refuse to emit a report that would leak raw-like keys downstream.
    violations = managed_egress_violations(report)
    if violations:  # pragma: no cover - guarded by privacy-safe construction
        report["privacy"]["egress_violations"] = violations
        report["privacy"]["egress_safe"] = False
    else:
        report["privacy"]["egress_safe"] = True

    return report
