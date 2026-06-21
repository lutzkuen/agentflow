from __future__ import annotations

import json
import os
from typing import Any

from tokenclaw.pattern_rollout import normalize_pattern_rollout


PATTERN_CANARY_SAFETY_STOP_SCHEMA = "tokenclaw.pattern_canary_safety_stop.v1"
PATTERN_CANARY_SAFETY_STOP_ENV = "TOKENCLAW_PATTERN_CANARY_SAFETY_STOP"
PATTERN_CANARY_SAFETY_STOP_WINDOW_ENV = "TOKENCLAW_PATTERN_CANARY_SAFETY_STOP_WINDOW"
LOCAL_CANARY_SAFETY_STOP_REASON = "local-canary-safety-stop"

_LOGGED_STOP_KEYS: set[tuple[str, str, str, str]] = set()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, number)


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _rollback_threshold(rollout: dict[str, Any]) -> float | None:
    raw = rollout.get("rollback_threshold")
    if isinstance(raw, dict):
        for key in (
            "regression_rate",
            "max_regression_rate",
            "error_or_bypass_rate",
            "max_error_or_bypass_rate",
            "error_rate",
            "max_error_rate",
            "bypass_rate",
            "max_bypass_rate",
            "threshold",
        ):
            parsed = _as_float(raw.get(key), None)
            if parsed is not None:
                return min(1.0, parsed)
        return None
    parsed = _as_float(raw, None)
    return min(1.0, parsed) if parsed is not None else None


def _candidate_matches(current: Any, observed: Any) -> bool:
    if current is None:
        return observed in (None, "", "unknown")
    return str(current) == str(observed)


def _canary_applied(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    if not isinstance(canary, dict):
        return False
    return bool(canary.get("enabled")) and str(canary.get("cohort") or canary.get("status")) in {
        "canary_applied",
        "applied",
    }


def _is_local_safety_stop(meta: dict[str, Any]) -> bool:
    reason = str(meta.get("reason") or "")
    if reason == LOCAL_CANARY_SAFETY_STOP_REASON:
        return True
    safety_stop = meta.get("safety_stop")
    return isinstance(safety_stop, dict) and safety_stop.get("reason") == LOCAL_CANARY_SAFETY_STOP_REASON


def _summary_is_bypass(meta: dict[str, Any]) -> bool:
    reason = str(meta.get("reason") or "").lower()
    status = str(meta.get("status") or "").lower()
    outcome = str(meta.get("outcome") or "").lower()
    return outcome == "bypassed" or status in {"bypass", "bypassed"} or "bypass" in reason or "disabled" in reason


def _crunch_matches(
    *,
    crunch_meta: dict[str, Any],
    rule_id: str,
    candidate_id: Any,
    pattern_hash: str,
) -> list[dict[str, Any]]:
    pattern_rules = crunch_meta.get("pattern_rules")
    if not isinstance(pattern_rules, dict):
        return []
    matches: list[dict[str, Any]] = []
    for rule in pattern_rules.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("rule_id") or "") != rule_id:
            continue
        if not _candidate_matches(candidate_id, rule.get("candidate_id")):
            continue
        hashes = [str(item) for item in rule.get("matched_hashes") or []]
        if pattern_hash not in hashes:
            continue
        if not _canary_applied(rule) or _is_local_safety_stop(rule):
            continue
        matches.append(rule)
    return matches


def _cache_matches(
    *,
    cache_meta: dict[str, Any],
    rule_id: str,
    candidate_id: Any,
    pattern_hash: str,
) -> list[dict[str, Any]]:
    pattern_rule = cache_meta.get("pattern_rule")
    if not isinstance(pattern_rule, dict):
        return []
    if str(pattern_rule.get("rule_id") or "") != rule_id:
        return []
    if not _candidate_matches(candidate_id, pattern_rule.get("candidate_id")):
        return []
    hashes = [str(item) for item in pattern_rule.get("matched_hashes") or []]
    if pattern_hash not in hashes:
        return []
    if not _canary_applied(pattern_rule) or _is_local_safety_stop(cache_meta) or _is_local_safety_stop(pattern_rule):
        return []
    return [cache_meta]


def evaluate_pattern_canary_safety_stop(
    *,
    store_obj: Any | None,
    policy_section: str,
    rule_id: str,
    candidate_id: Any,
    pattern_hash: str,
    rollout: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not _env_bool(PATTERN_CANARY_SAFETY_STOP_ENV, True):
        return None
    normalized = normalize_pattern_rollout(rollout)
    if not normalized or not normalized.get("canary_enabled"):
        return None
    threshold = _rollback_threshold(normalized)
    if threshold is None:
        return None
    min_samples = max(1, _as_int(normalized.get("min_outcome_samples"), 5))
    window_limit = max(1, min(_env_int(PATTERN_CANARY_SAFETY_STOP_WINDOW_ENV, 500), 10_000))
    if store_obj is None or not hasattr(store_obj, "conn"):
        return None

    try:
        rows = store_obj.conn.execute(
            """
            select status_code, crunch_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (window_limit,),
        ).fetchall()
    except Exception:
        return None

    sample_count = 0
    error_count = 0
    bypassed_count = 0
    section = str(policy_section)
    for row in rows:
        row_dict = dict(row)
        status_code = row_dict.get("status_code")
        if section == "crunch":
            matches = _crunch_matches(
                crunch_meta=_json_obj(row_dict.get("crunch_json")),
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hash=pattern_hash,
            )
        elif section == "cache":
            matches = _cache_matches(
                cache_meta=_json_obj(row_dict.get("cache_json")),
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hash=pattern_hash,
            )
        else:
            matches = []
        for match in matches:
            sample_count += 1
            if status_code is not None and _as_int(status_code) >= 400:
                error_count += 1
            elif _summary_is_bypass(match):
                bypassed_count += 1

    regression_count = error_count + bypassed_count
    regression_rate = (regression_count / sample_count) if sample_count else 0.0
    stopped = sample_count >= min_samples and regression_count > 0 and regression_rate >= threshold
    if not stopped:
        return None
    return {
        "schema": PATTERN_CANARY_SAFETY_STOP_SCHEMA,
        "stopped": True,
        "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
        "policy_section": section,
        "rule_id": rule_id,
        "candidate_id": str(candidate_id) if candidate_id is not None else None,
        "pattern_hash": pattern_hash,
        "canary_cohort": "canary_applied",
        "sample_count": sample_count,
        "error_count": error_count,
        "bypassed_count": bypassed_count,
        "regression_count": regression_count,
        "error_rate": round(error_count / sample_count, 4) if sample_count else 0.0,
        "bypass_rate": round(bypassed_count / sample_count, 4) if sample_count else 0.0,
        "regression_rate": round(regression_rate, 4),
        "min_outcome_samples": min_samples,
        "rollback_threshold": threshold,
        "window_limit": window_limit,
        "raw_payload_included": False,
    }


def log_pattern_canary_safety_stop(stop: dict[str, Any] | None) -> None:
    if not isinstance(stop, dict) or not stop.get("stopped"):
        return
    key = (
        str(stop.get("policy_section") or ""),
        str(stop.get("rule_id") or ""),
        str(stop.get("candidate_id") or ""),
        str(stop.get("pattern_hash") or ""),
    )
    if key in _LOGGED_STOP_KEYS:
        return
    _LOGGED_STOP_KEYS.add(key)
    try:
        from tokenclaw.policy_events import log_policy_event

        log_policy_event(
            "pattern-canary-safety-stop",
            ok=True,
            details={
                name: stop.get(name)
                for name in (
                    "schema",
                    "reason",
                    "policy_section",
                    "rule_id",
                    "candidate_id",
                    "pattern_hash",
                    "canary_cohort",
                    "sample_count",
                    "error_count",
                    "bypassed_count",
                    "regression_rate",
                    "min_outcome_samples",
                    "rollback_threshold",
                    "window_limit",
                    "raw_payload_included",
                )
            },
        )
    except Exception:
        return
