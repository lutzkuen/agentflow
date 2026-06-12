from __future__ import annotations

import base64
import binascii
import os
import re
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.crunch import sha256_text
from agentflow_proxy.pattern_rollout import (
    normalize_pattern_rollout,
    pattern_canary_decision,
    pattern_rollout_public_meta,
)
from agentflow_proxy.pattern_safety import (
    LOCAL_CANARY_SAFETY_STOP_REASON,
    evaluate_pattern_canary_safety_stop,
    log_pattern_canary_safety_stop,
)
from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.store import stable_json


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


def _default_cache_policy() -> dict[str, Any]:
    return {
        "exact_cache": {
            "enabled": True,
            # Avoid caching tool-using agent turns by default. Exact cache can be dangerous when tools reflect filesystem state.
            "cache_tool_calls": False,
        },
        "semantic_cache": {
            "enabled": False,
            "threshold": 0.95,
        },
        "file_watch": {
            "enabled": True,
            "root": ".",
            "max_paths": 128,
            "capture_candidates": False,
        },
        "pattern_rules": [],
        "session_memory_hints": {
            "enabled": False,
            "rule_id": "local-session-plateau-cache-hint",
            "min_call_count": 4,
            "min_plateau_pairs": 3,
            "min_text_chars": 8000,
            "max_error_rate": 0.0,
            "allowed_phases": ["planning", "verification", "summary"],
            "block_tool_results": True,
            "block_thinking": True,
            "require_safe_invalidation": True,
            "require_reviewed_pattern_rule": True,
            "allow_tool_calls": False,
            "allow_streaming_replay": False,
        },
    }


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(Path.home() / ".agentflow" / filename)
    return candidates


def _first_existing_rule_path(filename: str, env_name: str) -> Path | None:
    for path in _manual_rule_candidates(filename, env_name):
        if path.exists():
            return path
    return None


def _apply_cache_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    exact = data.get("exact_cache") or {}
    if isinstance(exact, dict):
        policy["exact_cache"]["enabled"] = _as_bool(exact.get("enabled"), policy["exact_cache"]["enabled"])
        policy["exact_cache"]["cache_tool_calls"] = _as_bool(
            exact.get("cache_tool_calls"),
            policy["exact_cache"]["cache_tool_calls"],
        )
    semantic = data.get("semantic_cache") or {}
    if isinstance(semantic, dict):
        policy["semantic_cache"]["enabled"] = _as_bool(
            semantic.get("enabled"),
            policy["semantic_cache"]["enabled"],
        )
        if semantic.get("threshold") is not None:
            policy["semantic_cache"]["threshold"] = float(semantic["threshold"])
    file_watch = data.get("file_watch") or {}
    if isinstance(file_watch, dict):
        policy["file_watch"]["enabled"] = _as_bool(
            file_watch.get("enabled"),
            policy["file_watch"]["enabled"],
        )
        if file_watch.get("root") is not None:
            policy["file_watch"]["root"] = str(file_watch["root"])
        if file_watch.get("max_paths") is not None:
            policy["file_watch"]["max_paths"] = int(file_watch["max_paths"])
        if file_watch.get("capture_candidates") is not None:
            policy["file_watch"]["capture_candidates"] = _as_bool(
                file_watch.get("capture_candidates"),
                policy["file_watch"]["capture_candidates"],
            )
    policy["pattern_rules"] = _load_cache_pattern_rules(data.get("pattern_rules"))
    hints = data.get("session_memory_hints") or {}
    if isinstance(hints, dict):
        _apply_session_memory_hints_policy_yaml(policy, hints)
    return policy


def _apply_cache_canary_overlay(policy: dict[str, Any]) -> str | None:
    path = _first_existing_rule_path("cache_canary_policy.yaml", "AGENTFLOW_CACHE_CANARY_POLICY")
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return None
    overlay_rules = cache_pattern_rules_from_policy_payload(data)
    if not overlay_rules:
        return None
    policy["pattern_rules"] = [
        *(policy.get("pattern_rules") or []),
        *overlay_rules,
    ]
    policy["cache_canary_policy_path"] = str(path)
    policy["cache_canary_policy_source"] = str(data.get("policy_source") or "managed-recommended")
    return str(path)


def _apply_session_memory_hints_policy_yaml(policy: dict[str, Any], hints: dict[str, Any]) -> None:
    target = policy["session_memory_hints"]
    target["enabled"] = _as_bool(hints.get("enabled"), target["enabled"])
    for key in ("rule_id",):
        if hints.get(key) is not None:
            target[key] = str(hints[key])
    for key in (
        "block_tool_results",
        "block_thinking",
        "require_safe_invalidation",
        "require_reviewed_pattern_rule",
        "allow_tool_calls",
        "allow_streaming_replay",
    ):
        if hints.get(key) is not None:
            target[key] = _as_bool(hints.get(key), target[key])
    for key in ("min_call_count", "min_plateau_pairs", "min_text_chars"):
        if hints.get(key) is not None:
            target[key] = int(hints[key])
    if hints.get("max_error_rate") is not None:
        target["max_error_rate"] = float(hints["max_error_rate"])
    if hints.get("allowed_phases") is not None:
        raw = hints["allowed_phases"]
        if isinstance(raw, list):
            target["allowed_phases"] = [str(item) for item in raw]
        else:
            target["allowed_phases"] = [str(raw)]


def _normalize_pattern_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    digest = text.removeprefix("sha256:") if text.startswith("sha256:") else text
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return f"sha256:{digest}"


def _parse_pattern_hashes(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return sorted({normalized for item in value if (normalized := _normalize_pattern_hash(item))})


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []


def _load_cache_pattern_rules(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        conditions = item.get("conditions") if isinstance(item.get("conditions"), dict) else {}
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        pattern_hashes = _parse_pattern_hashes(
            conditions.get("pattern_hashes")
            or conditions.get("pattern_hash")
            or item.get("pattern_hashes")
            or item.get("pattern_hash")
        )
        if not pattern_hashes:
            continue
        rule_id = str(item.get("id") or item.get("rule_id") or f"cache-pattern-rule-{index + 1}")
        rules.append({
            "id": rule_id,
            "enabled": _as_bool(item.get("enabled"), True),
            "policy_source": str(item.get("policy_source") or "managed-recommended"),
            "candidate_id": item.get("candidate_id") or item.get("recommendation_id") or item.get("policy_id"),
            "conditions": {
                **conditions,
                "pattern_hashes": pattern_hashes,
                "replayability_levels": _string_list(
                    conditions.get("replayability_levels")
                    or conditions.get("replayability_level")
                ),
                "category_not_in": _string_list(conditions.get("category_not_in")),
            },
            "action": {
                **action,
                "type": str(action.get("type") or "exact_cache"),
                "allow_tool_calls": _as_bool(action.get("allow_tool_calls"), False),
                "safe_invalidation": _as_bool(
                    action.get("safe_invalidation")
                    if "safe_invalidation" in action
                    else action.get("safe_invalidation_evidence"),
                    False,
                ),
                "streaming": _as_bool(action.get("streaming"), False),
                "scope": str(action.get("scope") or "session"),
            },
            "rollout": normalize_pattern_rollout(item.get("rollout")),
        })
    return rules


def normalize_cache_pattern_rules(value: Any) -> list[dict[str, Any]]:
    """Normalize cache pattern rules for offline review and dry-run tools."""
    return _load_cache_pattern_rules(value)


def cache_pattern_rules_from_policy_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract normalized cache pattern rules from a policy bundle or cache section."""
    if isinstance(payload, list):
        return normalize_cache_pattern_rules(payload)
    if not isinstance(payload, dict):
        return []

    candidates: list[Any] = []
    if "pattern_rules" in payload:
        candidates.append(payload.get("pattern_rules"))
    cache_section = payload.get("cache")
    if isinstance(cache_section, dict):
        candidates.append(cache_section.get("pattern_rules"))
    policies = payload.get("policies")
    if isinstance(policies, dict):
        policy_cache = policies.get("cache")
        if isinstance(policy_cache, dict):
            candidates.append(policy_cache.get("pattern_rules"))

    for candidate in candidates:
        rules = normalize_cache_pattern_rules(candidate)
        if rules:
            return rules
    return []


def cache_pattern_hashes_from_features(pattern_features: dict[str, Any] | None) -> list[str]:
    """Return normalized pattern hashes from metadata-only pattern features."""
    return _feature_hashes(pattern_features)


def _load_cache_policy() -> tuple[dict[str, Any], str, str]:
    loaded_policy: dict[str, Any] | None = None
    loaded_source = "local-default"
    loaded_path: str | None = None
    for path in _manual_rule_candidates("cache_rules.yaml", "AGENTFLOW_CACHE_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            loaded_policy = _apply_cache_policy_yaml(_default_cache_policy(), data)
            loaded_source = "local-manual"
            loaded_path = str(path)
            break

    defaults_path = Path(__file__).parent / "cache_rules.yaml"
    policy = loaded_policy or _default_cache_policy()
    if loaded_policy is None:
        if defaults_path.exists():
            with open(defaults_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                policy = _apply_cache_policy_yaml(policy, data)
        loaded_path = str(defaults_path)
    if loaded_policy is None:
        policy["exact_cache"]["enabled"] = os.getenv("AGENTFLOW_CACHE", "1") != "0"
        policy["exact_cache"]["cache_tool_calls"] = os.getenv("AGENTFLOW_CACHE_TOOL_CALLS", "0") == "1"
        policy["semantic_cache"]["enabled"] = os.getenv("AGENTFLOW_SEMANTIC_CACHE", "0") == "1"
        policy["semantic_cache"]["threshold"] = float(
            os.getenv("AGENTFLOW_SEMANTIC_THRESHOLD", str(policy["semantic_cache"]["threshold"]))
        )
        policy["file_watch"]["enabled"] = _as_bool(
            os.getenv("AGENTFLOW_CACHE_FILE_WATCH"),
            policy["file_watch"]["enabled"],
        )
        policy["file_watch"]["root"] = os.getenv(
            "AGENTFLOW_CACHE_WATCH_ROOT",
            str(policy["file_watch"]["root"]),
        )
        policy["file_watch"]["max_paths"] = int(
            os.getenv("AGENTFLOW_CACHE_WATCH_MAX_PATHS", str(policy["file_watch"]["max_paths"]))
        )
        policy["file_watch"]["capture_candidates"] = _as_bool(
            os.getenv("AGENTFLOW_CACHE_CAPTURE_CANDIDATES"),
            policy["file_watch"]["capture_candidates"],
        )
    _apply_cache_canary_overlay(policy)
    return policy, loaded_source, str(loaded_path or defaults_path)


CACHE_POLICY, CACHE_POLICY_SOURCE, CACHE_RULES_PATH = _load_cache_policy()
CACHE_RULES_LOADED_AT = utc_now()
CACHE_RULES_LOADED_FILE = policy_file_snapshot(CACHE_RULES_PATH)
CACHE_ENABLED = bool(CACHE_POLICY["exact_cache"]["enabled"])
CACHE_TOOL_CALLS = bool(CACHE_POLICY["exact_cache"]["cache_tool_calls"])
SEMANTIC_CACHE_ENABLED = bool(CACHE_POLICY["semantic_cache"]["enabled"])
SEMANTIC_CACHE_THRESHOLD = float(CACHE_POLICY["semantic_cache"]["threshold"])
CACHE_FILE_WATCH_ENABLED = bool(CACHE_POLICY["file_watch"]["enabled"])
CACHE_FILE_WATCH_ROOT = str(CACHE_POLICY["file_watch"]["root"])
CACHE_FILE_WATCH_MAX_PATHS = int(CACHE_POLICY["file_watch"]["max_paths"])
CACHE_FILE_WATCH_CAPTURE_CANDIDATES = bool(CACHE_POLICY["file_watch"]["capture_candidates"])
CACHE_PATTERN_RULES = tuple(CACHE_POLICY.get("pattern_rules") or [])
CACHE_CANARY_RULES_PATH = CACHE_POLICY.get("cache_canary_policy_path")


def cache_decision_meta(
    status: str,
    reason: str,
    *,
    hit_type: str | None = None,
    enabled: bool | None = None,
    exact_enabled: bool | None = None,
    semantic_enabled: bool | None = None,
    tool_cache_enabled: bool | None = None,
) -> dict[str, Any]:
    exact = CACHE_ENABLED if exact_enabled is None else exact_enabled
    semantic = SEMANTIC_CACHE_ENABLED if semantic_enabled is None else semantic_enabled
    tool_cache = CACHE_TOOL_CALLS if tool_cache_enabled is None else tool_cache_enabled
    overall_enabled = (CACHE_ENABLED or SEMANTIC_CACHE_ENABLED) if enabled is None else enabled
    meta = {
        "enabled": bool(overall_enabled),
        "status": status,
        "reason": reason,
        "policy_source": CACHE_POLICY_SOURCE,
        "rule_path": CACHE_RULES_PATH,
        "exact_enabled": bool(exact),
        "semantic_enabled": bool(semantic),
        "tool_cache_enabled": bool(tool_cache),
        "semantic_threshold": SEMANTIC_CACHE_THRESHOLD,
        "file_watch_enabled": CACHE_FILE_WATCH_ENABLED,
    }
    if hit_type:
        meta["hit_type"] = hit_type
    return meta


def _dependency_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2_5"
    if count <= 20:
        return "6_20"
    if count <= 128:
        return "21_128"
    return "128_plus"


def _dependency_root_policy(root: Path) -> str:
    cwd = Path.cwd().resolve(strict=False)
    try:
        root.relative_to(cwd)
        return "cwd-relative"
    except ValueError:
        pass
    try:
        home = Path.home().resolve(strict=False)
    except RuntimeError:
        return "configured-local-root"
    try:
        root.relative_to(home)
        return "home-relative"
    except ValueError:
        return "configured-local-root"


def _feature_hashes(pattern_features: dict[str, Any] | None) -> list[str]:
    if not isinstance(pattern_features, dict):
        return []
    hashes: list[str] = []
    for key in ("pattern_hashes", "pattern_hash", "normalized_pattern_hash", "cache_pattern_hash"):
        value = pattern_features.get(key)
        if isinstance(value, list):
            hashes.extend(_parse_pattern_hashes(value))
        elif (normalized := _normalize_pattern_hash(value)) is not None:
            hashes.append(normalized)
    return sorted(set(hashes))


def _condition_matches(conditions: dict[str, Any], key: str, actual: Any) -> bool:
    expected = conditions.get(key)
    if expected is None:
        return True
    expected_values = _string_list(expected)
    if not expected_values:
        return True
    return str(actual or "").lower() in {value.lower() for value in expected_values}


def _cacheability_from_pattern_features(pattern_features: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pattern_features, dict):
        return {}
    cacheability = pattern_features.get("cacheability")
    if isinstance(cacheability, dict):
        return cacheability
    result: dict[str, Any] = {}
    for key in (
        "cacheability_bucket",
        "static_information_hint",
        "time_sensitive_hint",
        "user_specific_hint",
        "exact_cache_candidate_hint",
    ):
        if key in pattern_features:
            result[key] = pattern_features.get(key)
    return result


def _bool_condition_matches(conditions: dict[str, Any], key: str, actual: Any) -> bool:
    if key not in conditions:
        return True
    return _as_bool(conditions.get(key), False) == bool(actual)


def _streaming_static_replay_blocker(
    *,
    has_tool_blocks: bool,
    has_thinking_blocks: bool = False,
    pattern_features: dict[str, Any] | None,
    rule: dict[str, Any],
) -> str | None:
    if has_tool_blocks:
        return "streaming-tools-disabled"
    if has_thinking_blocks:
        return "streaming-thinking-disabled"
    if str(rule.get("policy_source") or "").lower() == "local-default":
        return "streaming-rule-source-not-reviewed"
    cacheability = _cacheability_from_pattern_features(pattern_features)
    if not cacheability:
        return "cacheability-features-missing"
    if str(cacheability.get("cacheability_bucket") or "").lower() != "high":
        return "low-cacheability"
    if not bool(cacheability.get("static_information_hint")):
        return "static-information-required"
    if bool(cacheability.get("time_sensitive_hint")):
        return "current-state"
    if bool(cacheability.get("user_specific_hint")):
        return "user-specific"
    if cacheability.get("exact_cache_candidate_hint") is False:
        return "exact-cache-candidate-required"
    return None


def _cache_pattern_rule_match(
    *,
    has_tool_blocks: bool,
    has_thinking_blocks: bool = False,
    stream: bool,
    pattern_features: dict[str, Any] | None,
    local_replayability_level: str,
    store_obj: Any | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    feature_hashes = set(_feature_hashes(pattern_features))
    skip_reasons: list[dict[str, Any]] = []
    if not CACHE_PATTERN_RULES:
        return None, skip_reasons
    if not feature_hashes:
        return None, [{"reason": "pattern-features-missing", "configured_count": len(CACHE_PATTERN_RULES)}]

    for rule in CACHE_PATTERN_RULES:
        rule_id = str(rule.get("id") or "cache-pattern-rule")
        if not rule.get("enabled", True):
            skip_reasons.append({"rule_id": rule_id, "reason": "disabled"})
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        rule_hashes = set(_parse_pattern_hashes(conditions.get("pattern_hashes")))
        matched_hashes = sorted(feature_hashes.intersection(rule_hashes))
        if not matched_hashes:
            skip_reasons.append({"rule_id": rule_id, "reason": "pattern-hash-mismatch"})
            continue
        if "has_tools" in conditions and _as_bool(conditions.get("has_tools"), False) != bool(has_tool_blocks):
            skip_reasons.append({"rule_id": rule_id, "reason": "has-tools-mismatch"})
            continue
        if "stream" in conditions and _as_bool(conditions.get("stream"), False) != bool(stream):
            skip_reasons.append({"rule_id": rule_id, "reason": "stream-mismatch"})
            continue
        if stream and has_tool_blocks:
            skip_reasons.append({"rule_id": rule_id, "reason": "streaming-tools-disabled"})
            continue
        model_pattern = conditions.get("model_pattern")
        if isinstance(model_pattern, str) and model_pattern.strip():
            requested_model = str((pattern_features or {}).get("requested_model") or "").lower()
            routed_model = str((pattern_features or {}).get("candidate_target_model") or "").lower()
            pattern = model_pattern.strip().lower()
            if pattern not in requested_model and pattern not in routed_model:
                skip_reasons.append({"rule_id": rule_id, "reason": "model-pattern-mismatch"})
                continue
        if not _condition_matches(conditions, "category", (pattern_features or {}).get("category")):
            skip_reasons.append({"rule_id": rule_id, "reason": "category-mismatch"})
            continue
        cacheability = _cacheability_from_pattern_features(pattern_features)
        if not _condition_matches(conditions, "cacheability_bucket", cacheability.get("cacheability_bucket")):
            skip_reasons.append({"rule_id": rule_id, "reason": "cacheability-bucket-mismatch"})
            continue
        bool_mismatch_reason = None
        for bool_key in (
            "static_information_hint",
            "time_sensitive_hint",
            "user_specific_hint",
            "exact_cache_candidate_hint",
        ):
            if not _bool_condition_matches(conditions, bool_key, cacheability.get(bool_key)):
                bool_mismatch_reason = f"{bool_key}-mismatch"
                break
        if bool_mismatch_reason:
            skip_reasons.append({"rule_id": rule_id, "reason": bool_mismatch_reason})
            continue
        excluded_categories = {item.lower() for item in _string_list(conditions.get("category_not_in"))}
        if excluded_categories and str((pattern_features or {}).get("category") or "").lower() in excluded_categories:
            skip_reasons.append({"rule_id": rule_id, "reason": "category-excluded"})
            continue
        for key in ("workflow_phase", "source_surface", "endpoint", "app_family", "text_bucket", "token_bucket"):
            if not _condition_matches(conditions, key, (pattern_features or {}).get(key)):
                skip_reasons.append({"rule_id": rule_id, "reason": f"{key}-mismatch"})
                break
        else:
            replayability_levels = {
                item.lower()
                for item in _string_list(conditions.get("replayability_levels"))
            }
            if replayability_levels and local_replayability_level.lower() not in replayability_levels:
                skip_reasons.append({"rule_id": rule_id, "reason": "replayability-gate-mismatch"})
                continue
            action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
            if action.get("type") not in {"exact_cache", "exact_cache_pattern"}:
                skip_reasons.append({"rule_id": rule_id, "reason": "unsupported-action"})
                continue
            if stream and not action.get("streaming"):
                skip_reasons.append({"rule_id": rule_id, "reason": "streaming-not-allowed"})
                continue
            if stream:
                blocker = _streaming_static_replay_blocker(
                    has_tool_blocks=has_tool_blocks,
                    has_thinking_blocks=has_thinking_blocks,
                    pattern_features=pattern_features,
                    rule=rule,
                )
                if blocker:
                    skip_reasons.append({"rule_id": rule_id, "reason": blocker})
                    continue
            if has_tool_blocks and not (action.get("allow_tool_calls") and action.get("safe_invalidation")):
                skip_reasons.append({"rule_id": rule_id, "reason": "unsafe-tool-cache-pattern"})
                continue
            if has_tool_blocks and not CACHE_FILE_WATCH_ENABLED:
                skip_reasons.append({"rule_id": rule_id, "reason": "file-watch-required"})
                continue
            canary = pattern_canary_decision(
                rollout=rule.get("rollout"),
                rule_id=rule_id,
                candidate_id=rule.get("candidate_id"),
                pattern_hashes=matched_hashes,
                features=pattern_features,
            )
            if canary.get("enabled") and not canary.get("selected", True):
                skip_reasons.append({
                    "rule_id": rule_id,
                    "candidate_id": rule.get("candidate_id"),
                    "policy_source": rule.get("policy_source") or "managed-recommended",
                    "reason": "canary_holdout",
                    "matched_hashes": matched_hashes,
                    "canary": canary,
                })
                continue
            safety_stop = None
            for pattern_hash in matched_hashes:
                safety_stop = evaluate_pattern_canary_safety_stop(
                    store_obj=store_obj,
                    policy_section="cache",
                    rule_id=rule_id,
                    candidate_id=rule.get("candidate_id"),
                    pattern_hash=pattern_hash,
                    rollout=rule.get("rollout"),
                )
                if safety_stop:
                    break
            if safety_stop:
                skip_reasons.append({
                    "rule_id": rule_id,
                    "candidate_id": rule.get("candidate_id"),
                    "policy_source": rule.get("policy_source") or "managed-recommended",
                    "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
                    "matched_hashes": matched_hashes,
                    "rollout": pattern_rollout_public_meta(rule.get("rollout")),
                    "canary": canary if canary.get("enabled") else None,
                    "safety_stop": safety_stop,
                })
                log_pattern_canary_safety_stop(safety_stop)
                continue
            return {
                "rule_id": rule_id,
                "candidate_id": rule.get("candidate_id"),
                "policy_source": rule.get("policy_source") or "managed-recommended",
                "matched_hashes": matched_hashes,
                "replayability_level": local_replayability_level,
                "allow_tool_calls": bool(action.get("allow_tool_calls")),
                "safe_invalidation": bool(action.get("safe_invalidation")),
                "scope": str(action.get("scope") or "session"),
                "rollout": pattern_rollout_public_meta(rule.get("rollout")),
                "canary": canary if canary.get("enabled") else None,
            }, skip_reasons
    return None, skip_reasons


def _attach_cache_pattern_meta(
    meta: dict[str, Any],
    *,
    pattern_rule: dict[str, Any] | None,
    skip_reasons: list[dict[str, Any]],
) -> dict[str, Any]:
    meta["pattern_rules"] = {
        "configured_count": len(CACHE_PATTERN_RULES),
        "matched_count": 1 if pattern_rule else 0,
    }
    if pattern_rule:
        meta["pattern_rule"] = {
            key: value
            for key, value in pattern_rule.items()
            if value is not None and key in {
                "rule_id",
                "candidate_id",
                "policy_source",
                "matched_hashes",
                "replayability_level",
                "allow_tool_calls",
                "safe_invalidation",
                "scope",
                "rollout",
                "canary",
            }
        }
        meta["pattern_rules"]["rules"] = [meta["pattern_rule"]]
    if skip_reasons:
        meta["pattern_rules"]["skip_reasons"] = skip_reasons[:20]
    return meta


def cache_hit_decision_meta(
    reason: str,
    *,
    hit_type: str,
    exact_enabled: bool,
    semantic_enabled: bool,
    lookup_meta: dict[str, Any] | None = None,
    estimated_saved_cost_usd: float | None = None,
) -> dict[str, Any]:
    meta = cache_decision_meta(
        "hit",
        reason,
        hit_type=hit_type,
        exact_enabled=exact_enabled,
        semantic_enabled=semantic_enabled,
    )
    if isinstance(lookup_meta, dict) and isinstance(lookup_meta.get("pattern_rule"), dict):
        meta["pattern_rule"] = lookup_meta["pattern_rule"]
        meta["pattern_rules"] = lookup_meta.get("pattern_rules", {
            "configured_count": len(CACHE_PATTERN_RULES),
            "matched_count": 1,
            "rules": [lookup_meta["pattern_rule"]],
        })
    if isinstance(lookup_meta, dict) and isinstance(lookup_meta.get("session_memory_hints"), dict):
        meta["session_memory_hints"] = lookup_meta["session_memory_hints"]
    if isinstance(lookup_meta, dict) and isinstance(lookup_meta.get("session_memory_replayability"), dict):
        meta["session_memory_replayability"] = lookup_meta["session_memory_replayability"]
    if isinstance(lookup_meta, dict) and isinstance(lookup_meta.get("cache_replay_canary"), dict):
        meta["cache_replay_canary"] = lookup_meta["cache_replay_canary"]
    if estimated_saved_cost_usd is not None:
        meta["estimated_saved_cost_usd"] = round(max(0.0, float(estimated_saved_cost_usd)), 9)
    return meta


def cache_replay_scope_for_meta(cache_meta: dict[str, Any], session_id: str | None) -> tuple[str | None, str | None, dict[str, Any] | None]:
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta, dict) else None
    if not isinstance(pattern_rule, dict):
        return None, None, None
    scope = str(pattern_rule.get("scope") or "session")
    if scope == "session":
        return scope, session_id, pattern_rule
    if scope in {"workflow", "grouping"}:
        grouping = cache_meta.get("grouping_scope") if isinstance(cache_meta.get("grouping_scope"), dict) else {}
        scope_id = grouping.get("scope_id") or grouping.get("workflow_id_hash")
        return scope, str(scope_id) if scope_id else None, pattern_rule
    return scope, None, pattern_rule


def cache_replay_canary_decision(
    *,
    cache_meta: dict[str, Any],
    dependency_audit: dict[str, Any] | None,
    session_id: str | None,
) -> tuple[bool, dict[str, Any]]:
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta, dict) else None
    if not isinstance(pattern_rule, dict):
        return True, {}
    canary = pattern_rule.get("canary") if isinstance(pattern_rule.get("canary"), dict) else None
    scope = str(pattern_rule.get("scope") or "session")
    decision: dict[str, Any] = {
        "schema": "agentflow.cache_replay_canary_decision.v1",
        "rule_id": pattern_rule.get("rule_id"),
        "candidate_id": pattern_rule.get("candidate_id"),
        "policy_source": pattern_rule.get("policy_source"),
        "scope": scope,
        "canary": canary,
        "canary_cohort": canary.get("cohort") if canary else None,
        "dependency_audit": dependency_audit,
    }
    if canary and (canary.get("selected") is False or canary.get("cohort") == "canary_holdout"):
        decision.update({"status": "holdout", "reason": "canary_holdout"})
        return False, decision
    if not canary or canary.get("status") != "applied" or canary.get("cohort") != "canary_applied":
        decision.update({"status": "bypassed", "reason": "canary-applied-required"})
        return False, decision
    if scope == "session" and not session_id:
        decision.update({"status": "bypassed", "reason": "session-scope-missing"})
        return False, decision
    requires_dependency_evidence = bool(pattern_rule.get("allow_tool_calls") or pattern_rule.get("safe_invalidation"))
    if not requires_dependency_evidence:
        decision.update({"status": "applied", "reason": "no-dependency-required"})
        return True, decision
    current_audit = cache_meta.get("file_dependency_audit") if isinstance(cache_meta.get("file_dependency_audit"), dict) else None
    if isinstance(current_audit, dict) and not current_audit.get("safe_invalidation_evidence"):
        decision.update({
            "status": "bypassed",
            "reason": current_audit.get("invalidation_reason") or "file-dependency-missing",
            "current_dependency_evidence": {
                "safe_invalidation_evidence": False,
                "reason": current_audit.get("invalidation_reason") or "file-dependency-missing",
                "snapshot_count_bucket": current_audit.get("snapshot_count_bucket"),
                "candidate_path_count_bucket": current_audit.get("candidate_path_count_bucket"),
                "raw_candidate_path_count_bucket": current_audit.get("raw_candidate_path_count_bucket"),
                "distinct_candidate_path_count_bucket": current_audit.get("distinct_candidate_path_count_bucket"),
                "cap_exceeded": bool(current_audit.get("cap_exceeded")),
                "cap_trimmed": bool(current_audit.get("cap_trimmed")),
                "dependency_capture_reason": current_audit.get("dependency_capture_reason"),
                "paths_included": False,
            },
        })
        return False, decision
    if not isinstance(dependency_audit, dict):
        decision.update({"status": "bypassed", "reason": "dependency-audit-missing"})
        return False, decision
    if not dependency_audit.get("safe_invalidation_evidence"):
        decision.update({
            "status": "invalidated" if dependency_audit.get("invalidation_reason") else "bypassed",
            "reason": dependency_audit.get("invalidation_reason") or "file-dependency-missing",
        })
        return False, decision
    decision.update({"status": "applied", "reason": "dependency-stable"})
    return True, decision


def _cache_replay_rule_from_meta(cache_meta: dict[str, Any]) -> dict[str, Any] | None:
    rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else None
    if isinstance(rule, dict):
        return rule
    pattern_rules = cache_meta.get("pattern_rules") if isinstance(cache_meta.get("pattern_rules"), dict) else {}
    for skip in reversed(pattern_rules.get("skip_reasons") or []):
        if isinstance(skip, dict) and (
            skip.get("rule_id")
            or skip.get("candidate_id")
            or skip.get("canary")
            or skip.get("safety_stop")
        ):
            return skip
    return None


def _cache_replay_public_canary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    public = {
        key: value.get(key)
        for key in ("enabled", "selected", "cohort", "fraction", "threshold", "unit", "reason", "status")
        if value.get(key) is not None
    }
    public["pattern_hashes_included"] = False
    return public


def _cache_replay_invalidation_reasons(
    *,
    cache_meta: dict[str, Any],
    replay_canary: dict[str, Any],
) -> list[str]:
    reasons: set[str] = set()
    for value in (
        cache_meta.get("invalidation_reason"),
        replay_canary.get("reason") if replay_canary.get("status") == "invalidated" else None,
    ):
        if value:
            reasons.add(str(value))
    audit = replay_canary.get("dependency_audit")
    if not isinstance(audit, dict):
        audit = cache_meta.get("file_dependency_audit") if isinstance(cache_meta.get("file_dependency_audit"), dict) else {}
    if audit.get("invalidation_reason"):
        reasons.add(str(audit.get("invalidation_reason")))
    if _as_bool(audit.get("cap_exceeded"), False):
        reasons.add("dependency-cap-exceeded")
    if int(audit.get("changed_path_count") or 0):
        reasons.add("dependency-changed")
    if int(audit.get("deleted_path_count") or 0):
        reasons.add("dependency-deleted")
    if int(audit.get("created_path_count") or 0):
        reasons.add("dependency-created")
    if int(audit.get("missing_path_count") or 0):
        reasons.add("dependency-missing")
    return sorted(reasons)


def _cache_replay_cohort(
    *,
    cache_meta: dict[str, Any],
    rule: dict[str, Any],
    replay_canary: dict[str, Any],
) -> tuple[str | None, str]:
    reason = str(rule.get("reason") or cache_meta.get("reason") or replay_canary.get("reason") or "")
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else cache_meta.get("canary")
    if not isinstance(canary, dict):
        canary = replay_canary.get("canary") if isinstance(replay_canary.get("canary"), dict) else {}
    safety_stop = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else cache_meta.get("safety_stop")
    if isinstance(safety_stop, dict) or reason == LOCAL_CANARY_SAFETY_STOP_REASON:
        return "safety_stopped", "safety-stop"
    if replay_canary.get("status") == "invalidated" or cache_meta.get("invalidated"):
        return "invalidated", "invalidation"
    if reason == "canary_holdout" or canary.get("cohort") == "canary_holdout" or canary.get("selected") is False:
        return "holdout", "holdout"
    if replay_canary.get("status") == "bypassed":
        return "bypassed", "replay-bypassed"
    if cache_meta.get("status") == "miss" and replay_canary.get("status") == "applied":
        return "applied", "cache-miss"
    if cache_meta.get("status") == "hit" and replay_canary.get("status") == "applied":
        return "replayed", "cache-hit"
    return None, "not-cache-replay-lifecycle"


def build_cache_replay_lifecycle_feedback(
    *,
    cache_meta: dict[str, Any],
    provider: str,
    source_surface: str,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    latency_ms: int | None,
    retry_count: int | None,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    category: str | None,
    stream: bool,
) -> dict[str, Any] | None:
    """Build metadata-only cache replay lifecycle feedback for managed/local stats."""
    if not isinstance(cache_meta, dict):
        return None
    rule = _cache_replay_rule_from_meta(cache_meta)
    if not isinstance(rule, dict):
        return None
    replay_canary = (
        cache_meta.get("cache_replay_canary")
        if isinstance(cache_meta.get("cache_replay_canary"), dict)
        else {}
    )
    cohort, event_reason = _cache_replay_cohort(cache_meta=cache_meta, rule=rule, replay_canary=replay_canary)
    if cohort is None:
        return None
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else cache_meta.get("canary")
    if not isinstance(canary, dict):
        canary = replay_canary.get("canary") if isinstance(replay_canary.get("canary"), dict) else {}
    safety_stop = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else cache_meta.get("safety_stop")
    saved_cost = cache_meta.get("estimated_saved_cost_usd")
    if saved_cost is None and cost_baseline_usd is not None and cost_est_usd is not None:
        saved_cost = max(0.0, float(cost_baseline_usd) - float(cost_est_usd))
    status_class = "unknown"
    if isinstance(status_code, int):
        if status_code < 400:
            status_class = "success"
        elif status_code < 500:
            status_class = "client_error"
        else:
            status_class = "server_error"
    event: dict[str, Any] = {
        "schema": "agentflow.cache_replay_lifecycle_feedback.v1",
        "provider": provider,
        "source_surface": source_surface,
        "policy_id": _cache_replay_public_id(
            rule.get("policy_id") or rule.get("candidate_id") or rule.get("rule_id"),
            "policy-id",
        ),
        "rule_id": _cache_replay_public_id(rule.get("rule_id"), "rule-id"),
        "candidate_id": _cache_replay_public_id(rule.get("candidate_id"), "candidate-id"),
        "policy_source": rule.get("policy_source") or cache_meta.get("policy_source") or "unknown",
        "cohort": cohort,
        "event_reason": event_reason,
        "cache_decision_status": cache_meta.get("status"),
        "cache_decision_reason": cache_meta.get("reason"),
        "cache_hit": cache_meta.get("status") == "hit",
        "hit_type": cache_meta.get("hit_type"),
        "status_class": status_class,
        "status_code": status_code,
        "retry_count": int(retry_count or 0),
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "cost_est_usd": round(float(cost_est_usd), 9) if cost_est_usd is not None else None,
        "cost_baseline_usd": round(float(cost_baseline_usd), 9) if cost_baseline_usd is not None else None,
        "estimated_saved_cost_usd": round(float(saved_cost), 9) if saved_cost is not None else None,
        "requested_model_family": _model_family(requested_model),
        "routed_model_family": _model_family(routed_model),
        "category": category,
        "stream": bool(stream),
        "canary": _cache_replay_public_canary(canary),
        "invalidation_reason_codes": _cache_replay_invalidation_reasons(
            cache_meta=cache_meta,
            replay_canary=replay_canary,
        ),
        "safety_stop": {
            key: safety_stop.get(key)
            for key in ("reason", "decision", "sample_count", "error_rate", "retry_rate")
            if isinstance(safety_stop, dict) and safety_stop.get(key) is not None
        } if isinstance(safety_stop, dict) else None,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "pattern_hashes_included": False,
        },
    }
    if replay_canary:
        event["dependency_evidence"] = {
            "status": replay_canary.get("status"),
            "reason": replay_canary.get("reason"),
            "safe_invalidation_evidence": bool(
                (replay_canary.get("dependency_audit") or {}).get("safe_invalidation_evidence")
            ) if isinstance(replay_canary.get("dependency_audit"), dict) else None,
        }
    return {
        key: value
        for key, value in event.items()
        if value not in (None, "", [], {})
    }


def cache_replay_lifecycle_feedback_public_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(meta.get("enabled")),
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "endpoint": meta.get("endpoint"),
        "queue_id": meta.get("queue_id"),
        "attempts": meta.get("attempts"),
        "status_code": meta.get("status_code"),
        "latency_ms": meta.get("latency_ms"),
        "payload_included": False,
    }


def _model_family(model: str | None) -> str | None:
    if not model:
        return None
    model_l = str(model).lower()
    for family in ("haiku", "sonnet", "opus", "codex", "gpt-5", "gpt-4", "gpt-3"):
        if family in model_l:
            return family
    return "other"


_CACHE_REPLAY_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_CACHE_REPLAY_RAW_VALUE_HINT_RE = re.compile(
    r"[/\\]|\s|cache[-_]?key|request[-_]?id|raw[-_]?prompt|provider[-_]?body|tool[-_]?payload",
    re.IGNORECASE,
)


def _cache_replay_public_id(value: Any, kind: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    is_sha256_value = text.lower().startswith("sha256:") and len(text) >= 71
    if (
        _CACHE_REPLAY_PUBLIC_ID_RE.match(text)
        and not is_sha256_value
        and not _CACHE_REPLAY_RAW_VALUE_HINT_RE.search(text)
    ):
        return text
    return f"redacted-{kind}-{sha256_text(kind + ':' + text)[:12]}"


_PATH_TRAILING_JUNK = ".,;:)\\]}>\"'"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_NUMERIC_PATH_FRAGMENT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)+$")


def _walk_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(_walk_strings(item))
    return strings


def _obvious_non_path_fragment(token: str) -> bool:
    normalized = token.replace("\\", "/")
    if _NUMERIC_PATH_FRAGMENT_RE.match(normalized):
        return True
    segments = [segment for segment in normalized.split("/") if segment and segment not in {".", "..", "~"}]
    if not segments:
        return True
    if (
        not token.startswith(("/", "./", "../", "~/"))
        and _WINDOWS_DRIVE_RE.match(token) is None
        and not any(char.isalpha() for char in normalized)
        and "." not in normalized
    ):
        return True
    return False


_KNOWN_EXTENSIONLESS_FILE_NAMES = {
    ".dockerignore",
    ".env",
    ".envrc",
    ".gitignore",
    "dockerfile",
    "license",
    "makefile",
    "notice",
    "procfile",
    "readme",
}

_KNOWN_FILE_EXTENSIONS = {
    "bash",
    "c",
    "cc",
    "cfg",
    "conf",
    "cpp",
    "cs",
    "css",
    "csv",
    "db",
    "env",
    "fish",
    "gif",
    "go",
    "h",
    "hpp",
    "html",
    "ini",
    "ipynb",
    "java",
    "jpeg",
    "jpg",
    "js",
    "json",
    "jsx",
    "kt",
    "kts",
    "lock",
    "log",
    "lua",
    "md",
    "pdf",
    "php",
    "pl",
    "pm",
    "png",
    "py",
    "r",
    "rb",
    "rs",
    "rst",
    "scss",
    "sh",
    "sql",
    "sqlite",
    "sqlite3",
    "svg",
    "swift",
    "toml",
    "ts",
    "tsx",
    "txt",
    "webp",
    "xml",
    "yaml",
    "yml",
    "zsh",
}


def _path_token_has_file_identity(token: str) -> bool:
    normalized = token.replace("\\", "/")
    if token.startswith(("/", "./", "../", "~/")) or _WINDOWS_DRIVE_RE.match(token):
        return True
    basename = normalized.rsplit("/", 1)[-1].strip()
    if not basename or basename in {".", "..", "~"}:
        return False
    lowered = basename.lower()
    if lowered in _KNOWN_EXTENSIONLESS_FILE_NAMES:
        return True
    if lowered.startswith(".") and lowered.count(".") == 1 and len(lowered) > 1:
        return True
    if "." not in basename or basename.endswith("."):
        return False
    extension = basename.rsplit(".", 1)[-1].lower()
    return extension in _KNOWN_FILE_EXTENSIONS


def _candidate_path_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.split(r"\s+", text):
        token = raw.strip("`'\"(<[{")
        token = token.rstrip(_PATH_TRAILING_JUNK)
        if not token or "://" in token or token.startswith("data:"):
            continue
        if "\x00" in token or "*" in token or "?" in token:
            continue
        if ":" in token and not _WINDOWS_DRIVE_RE.match(token):
            before, after = token.rsplit(":", 1)
            if after.isdigit():
                token = before.rstrip(_PATH_TRAILING_JUNK)
        if not token:
            continue
        path_like = (
            token.startswith(("/", "./", "../", "~/"))
            or _WINDOWS_DRIVE_RE.match(token) is not None
            or "/" in token
            or "\\" in token
        )
        if path_like and not _obvious_non_path_fragment(token):
            tokens.append(token)
    return tokens


def _expand_path_or_none(value: str | Path) -> Path | None:
    try:
        return Path(value).expanduser()
    except RuntimeError:
        return None


def _resolve_under_root(token: str, root: Path) -> Path | None:
    expanded = _expand_path_or_none(token)
    if expanded is None:
        return None
    if expanded.is_absolute():
        resolved = expanded.resolve(strict=False)
    else:
        resolved = (root / expanded).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _cache_file_dependency_scan(
    body: dict[str, Any],
    *,
    root: str | Path | None = None,
    max_paths: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded_root = _expand_path_or_none(root if root is not None else CACHE_FILE_WATCH_ROOT)
    watch_root = (expanded_root or Path.cwd()).resolve(strict=False)
    limit = max(0, CACHE_FILE_WATCH_MAX_PATHS if max_paths is None else int(max_paths))
    paths: dict[str, Path] = {}
    raw_candidate_count = 0
    seen: set[str] = set()
    for text in _walk_strings(body):
        for token in _candidate_path_tokens(text):
            raw_candidate_count += 1
            resolved = _resolve_under_root(token, watch_root)
            if resolved is None:
                continue
            try:
                token_is_file = resolved.is_file()
            except OSError:
                token_is_file = False
            if not token_is_file and not _path_token_has_file_identity(token):
                continue
            resolved_key = str(resolved)
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            if len(paths) < limit:
                paths[resolved_key] = resolved

    snapshots: list[dict[str, Any]] = []
    for path in sorted(paths):
        file_path = paths[path]
        try:
            stat = file_path.stat()
            exists = file_path.is_file()
        except OSError:
            stat = None
            exists = False
        snapshots.append({
            "path": path,
            "exists": bool(exists),
            "mtime_ns": int(stat.st_mtime_ns) if stat and exists else None,
            "size": int(stat.st_size) if stat and exists else None,
        })
    audit = cache_file_dependency_audit(
        snapshots=snapshots,
        enabled=CACHE_FILE_WATCH_ENABLED,
        candidate_count=len(seen),
        raw_candidate_count=raw_candidate_count,
        max_paths=limit,
        root=watch_root,
    )
    return snapshots, audit


def _cache_workspace_scan(
    *,
    root: str | Path | None = None,
    max_paths: int | None = None,
) -> list[dict[str, Any]]:
    """Scan watch root for a bounded set of files as workspace dependency evidence.

    Produces file snapshots from the local workspace even when the request body
    contains no explicit path references.  Only called when capture_candidates is
    enabled in the file_watch policy — the default is off (conservative).
    """
    expanded_root = _expand_path_or_none(root if root is not None else CACHE_FILE_WATCH_ROOT)
    watch_root = (expanded_root or Path.cwd()).resolve(strict=False)
    limit = max(0, CACHE_FILE_WATCH_MAX_PATHS if max_paths is None else int(max_paths))
    if limit <= 0:
        return []
    snapshots: list[dict[str, Any]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(str(watch_root)):
            dirnames.sort()
            for filename in sorted(filenames):
                if len(snapshots) >= limit:
                    return snapshots
                file_path = Path(dirpath) / filename
                try:
                    resolved = file_path.resolve(strict=False)
                    resolved.relative_to(watch_root)
                except ValueError:
                    continue
                try:
                    stat = file_path.stat()
                    exists = file_path.is_file()
                except OSError:
                    stat = None
                    exists = False
                snapshots.append({
                    "path": str(file_path),
                    "exists": bool(exists),
                    "mtime_ns": int(stat.st_mtime_ns) if stat and exists else None,
                    "size": int(stat.st_size) if stat and exists else None,
                })
    except OSError:
        pass
    return snapshots


def cache_file_dependency_snapshots(
    body: dict[str, Any],
    *,
    root: str | Path | None = None,
    max_paths: int | None = None,
) -> list[dict[str, Any]]:
    if not CACHE_FILE_WATCH_ENABLED:
        return []
    snapshots, _audit = _cache_file_dependency_scan(body, root=root, max_paths=max_paths)
    if not snapshots and CACHE_FILE_WATCH_CAPTURE_CANDIDATES:
        snapshots = _cache_workspace_scan(root=root, max_paths=max_paths)
    return snapshots


def cache_file_dependency_fingerprint(
    snapshots: list[dict[str, Any]] | None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build path-free dependency state metadata for cache replay decisions."""
    audit = audit if isinstance(audit, dict) else {}
    normalized: list[dict[str, Any]] = []
    for dep in snapshots or []:
        path = dep.get("path")
        if not path:
            continue
        normalized.append({
            "path_sha256": sha256_text(str(path)),
            "exists": bool(dep.get("exists")),
            "mtime_ns": dep.get("mtime_ns"),
            "size": dep.get("size"),
        })
    normalized.sort(key=lambda item: str(item["path_sha256"]))
    fingerprint = f"sha256:{sha256_text(stable_json(normalized))}" if normalized else None
    return {
        "schema": "agentflow.cache_file_dependency_fingerprint.v1",
        "fingerprint_sha256": fingerprint,
        "fingerprint_available": bool(fingerprint),
        "snapshot_count": int(audit.get("snapshot_count") or len(normalized)),
        "snapshot_count_bucket": str(audit.get("snapshot_count_bucket") or _dependency_count_bucket(len(normalized))),
        "candidate_path_count_bucket": str(audit.get("candidate_path_count_bucket") or "unknown"),
        "raw_candidate_path_count_bucket": str(audit.get("raw_candidate_path_count_bucket") or "unknown"),
        "distinct_candidate_path_count_bucket": str(
            audit.get("distinct_candidate_path_count_bucket") or audit.get("candidate_path_count_bucket") or "unknown"
        ),
        "safe_invalidation_evidence": bool(audit.get("safe_invalidation_evidence")),
        "file_dependency_evidence_available": bool(audit.get("file_dependency_evidence_available")),
        "invalidation_reason": audit.get("invalidation_reason"),
        "cap_trimmed": bool(audit.get("cap_trimmed")),
        "dependency_capture_reason": audit.get("dependency_capture_reason"),
        "paths_included": False,
        "path_hashes_included": False,
        "raw_stat_values_included": False,
    }


def attach_file_dependency_cache_meta(
    meta: dict[str, Any],
    *,
    snapshots: list[dict[str, Any]] | None,
    audit: dict[str, Any] | None,
    blocker_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Attach path-free dependency audit/fingerprint fields to cache metadata."""
    audit = audit if isinstance(audit, dict) else cache_file_dependency_audit(snapshots=snapshots or [])
    fingerprint = cache_file_dependency_fingerprint(snapshots or [], audit)
    meta["file_dependency_audit"] = audit
    meta["file_dependency_fingerprint"] = fingerprint
    meta["file_dependency_count"] = int(audit.get("snapshot_count") or 0)
    meta["file_dependency_count_bucket"] = str(audit.get("snapshot_count_bucket") or "unknown")
    meta["file_dependency_fingerprint_available"] = bool(fingerprint.get("fingerprint_available"))
    if fingerprint.get("fingerprint_sha256"):
        meta["file_dependency_fingerprint_sha256"] = fingerprint["fingerprint_sha256"]
    meta["file_dependency_evidence_available"] = bool(audit.get("file_dependency_evidence_available"))
    meta["safe_invalidation_evidence"] = bool(audit.get("safe_invalidation_evidence"))
    blockers = {
        str(reason)
        for reason in (blocker_reasons or [])
        if reason
    }
    if not audit.get("safe_invalidation_evidence"):
        blockers.add(str(audit.get("invalidation_reason") or "file-dependency-missing"))
    if blockers:
        meta["cache_replay_blocker_reasons"] = sorted(blockers)
    meta["paths_included"] = False
    return meta


def cache_file_dependency_audit(
    body: dict[str, Any] | None = None,
    *,
    snapshots: list[dict[str, Any]] | None = None,
    enabled: bool | None = None,
    candidate_count: int | None = None,
    raw_candidate_count: int | None = None,
    max_paths: int | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    watch_enabled = CACHE_FILE_WATCH_ENABLED if enabled is None else bool(enabled)
    expanded_root = _expand_path_or_none(root if root is not None else CACHE_FILE_WATCH_ROOT)
    watch_root = (expanded_root or Path.cwd()).resolve(strict=False)
    limit = max(0, CACHE_FILE_WATCH_MAX_PATHS if max_paths is None else int(max_paths))
    if snapshots is None:
        if not watch_enabled or body is None:
            snapshots = []
        else:
            body_snapshots, body_audit = _cache_file_dependency_scan(body, root=watch_root, max_paths=limit)
            if body_snapshots or not CACHE_FILE_WATCH_CAPTURE_CANDIDATES:
                return body_audit
            # Body scan found no paths; fall back to workspace snapshot when capture_candidates is on
            snapshots = _cache_workspace_scan(root=watch_root, max_paths=limit)
    snapshot_count = len(snapshots or [])
    candidate_total = snapshot_count if candidate_count is None else max(0, int(candidate_count))
    raw_candidate_total = candidate_total if raw_candidate_count is None else max(0, int(raw_candidate_count))
    cap_exceeded = candidate_total > limit
    cap_trimmed = bool(raw_candidate_total > limit and not cap_exceeded)
    missing_count = sum(1 for dep in snapshots or [] if not bool(dep.get("exists")))
    present_count = snapshot_count - missing_count
    safe = bool(watch_enabled and snapshot_count > 0 and not cap_exceeded and missing_count == 0)
    reason = "safe"
    if not watch_enabled:
        reason = "file-watch-disabled"
    elif cap_exceeded:
        reason = "dependency-cap-exceeded"
    elif snapshot_count <= 0:
        reason = "file-dependency-missing"
    elif missing_count:
        reason = "dependency-missing"
    capture_reason = "complete"
    if cap_exceeded:
        capture_reason = "dependency-cap-exceeded"
    elif cap_trimmed:
        capture_reason = "dependency-cap-trimmed"
    return {
        "schema": "agentflow.cache_file_dependency_audit.v1",
        "file_watch_enabled": bool(watch_enabled),
        "snapshot_root_policy": _dependency_root_policy(watch_root),
        "root_path_included": False,
        "snapshot_count": snapshot_count,
        "snapshot_count_bucket": _dependency_count_bucket(snapshot_count),
        "candidate_path_count_bucket": _dependency_count_bucket(candidate_total),
        "raw_candidate_path_count_bucket": _dependency_count_bucket(raw_candidate_total),
        "distinct_candidate_path_count_bucket": _dependency_count_bucket(candidate_total),
        "max_paths": limit,
        "cap_exceeded": bool(cap_exceeded),
        "cap_trimmed": bool(cap_trimmed),
        "dependency_capture_reason": capture_reason,
        "present_path_count": present_count,
        "missing_path_count": missing_count,
        "changed_path_count": 0,
        "deleted_path_count": 0,
        "created_path_count": 0,
        "invalidation_reason": None if reason == "safe" else reason,
        "safe_invalidation_evidence": safe,
        "file_dependency_evidence_available": safe,
        "paths_included": False,
    }


def cache_lookup_meta(
    has_tool_blocks: bool,
    *,
    pattern_features: dict[str, Any] | None = None,
    store_obj: Any | None = None,
    managed_profile: dict[str, Any] | None = None,
) -> tuple[bool, bool, dict[str, Any]]:
    managed_profile = managed_profile if isinstance(managed_profile, dict) else None
    base_exact_enabled = CACHE_ENABLED
    base_semantic_enabled = SEMANTIC_CACHE_ENABLED
    if managed_profile and managed_profile.get("exact_enabled") is not None:
        base_exact_enabled = bool(managed_profile.get("exact_enabled"))
    if managed_profile and managed_profile.get("semantic_enabled") is not None:
        base_semantic_enabled = bool(managed_profile.get("semantic_enabled"))
    exact_enabled = base_exact_enabled and (CACHE_TOOL_CALLS or not has_tool_blocks)
    semantic_enabled = base_semantic_enabled and not has_tool_blocks
    local_replayability_level = "local-exact-response" if CACHE_ENABLED else "features_only"
    pattern_rule, pattern_skip_reasons = _cache_pattern_rule_match(
        has_tool_blocks=has_tool_blocks,
        stream=False,
        pattern_features=pattern_features,
        local_replayability_level=local_replayability_level,
        store_obj=store_obj,
    )
    if pattern_rule and CACHE_ENABLED:
        exact_enabled = True
    if exact_enabled or semantic_enabled:
        if exact_enabled and semantic_enabled:
            reason = "exact-and-semantic-miss"
        elif exact_enabled:
            reason = "exact-pattern-miss" if pattern_rule else "exact-miss"
        else:
            reason = "semantic-miss"
        status = "miss"
    elif has_tool_blocks and (CACHE_ENABLED or SEMANTIC_CACHE_ENABLED):
        status = "skipped"
        reason = "tools-disabled"
    else:
        status = "skipped"
        reason = "cache-disabled"
    meta = _attach_cache_pattern_meta(cache_decision_meta(
        status,
        reason,
        enabled=base_exact_enabled or base_semantic_enabled,
        exact_enabled=exact_enabled,
        semantic_enabled=semantic_enabled,
    ), pattern_rule=pattern_rule, skip_reasons=pattern_skip_reasons)
    if managed_profile:
        meta["policy_source"] = str(managed_profile.get("policy_source") or "managed-recommended")
        meta["managed_profile"] = managed_profile
        if managed_profile.get("semantic_threshold") is not None:
            meta["semantic_threshold"] = float(managed_profile["semantic_threshold"])
    if pattern_skip_reasons:
        selected_skip = pattern_skip_reasons[-1]
        if selected_skip.get("reason") in {"canary_holdout", LOCAL_CANARY_SAFETY_STOP_REASON}:
            meta["pattern_rule"] = {
                key: selected_skip.get(key)
                for key in ("rule_id", "candidate_id", "policy_source", "matched_hashes", "canary", "safety_stop")
                if selected_skip.get(key) is not None
            }
            if isinstance(selected_skip.get("canary"), dict):
                meta["canary"] = selected_skip["canary"]
                meta["canary_cohort"] = selected_skip["canary"].get("cohort")
    return exact_enabled, semantic_enabled, meta


def streaming_cache_lookup_meta(
    has_tool_blocks: bool,
    *,
    has_thinking_blocks: bool = False,
    pattern_features: dict[str, Any] | None = None,
    store_obj: Any | None = None,
    managed_profile: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    managed_profile = managed_profile if isinstance(managed_profile, dict) else None
    base_exact_enabled = CACHE_ENABLED
    if managed_profile and managed_profile.get("exact_enabled") is not None:
        base_exact_enabled = bool(managed_profile.get("exact_enabled"))
    exact_enabled = False
    pattern_rule, pattern_skip_reasons = _cache_pattern_rule_match(
        has_tool_blocks=has_tool_blocks,
        has_thinking_blocks=has_thinking_blocks,
        stream=True,
        pattern_features=pattern_features,
        local_replayability_level="local-exact-response" if CACHE_ENABLED else "features_only",
        store_obj=store_obj,
    )
    if pattern_rule and CACHE_ENABLED and base_exact_enabled:
        exact_enabled = True
    if exact_enabled:
        status = "miss"
        reason = "streaming-exact-pattern-miss"
    elif has_tool_blocks and CACHE_ENABLED:
        status = "skipped"
        reason = "streaming-tools-disabled"
    elif has_thinking_blocks and CACHE_ENABLED:
        status = "skipped"
        reason = "streaming-thinking-disabled"
    elif pattern_skip_reasons:
        status = "skipped"
        reason = str(pattern_skip_reasons[-1].get("reason") or "streaming-pattern-rule-skipped")
    elif CACHE_ENABLED and base_exact_enabled:
        status = "skipped"
        reason = "streaming-pattern-rule-required"
    else:
        status = "skipped"
        reason = "streaming-cache-disabled"
    meta = _attach_cache_pattern_meta(cache_decision_meta(
        status,
        reason,
        enabled=base_exact_enabled,
        exact_enabled=exact_enabled,
        semantic_enabled=False,
    ), pattern_rule=pattern_rule, skip_reasons=pattern_skip_reasons)
    if managed_profile:
        meta["policy_source"] = str(managed_profile.get("policy_source") or "managed-recommended")
        meta["managed_profile"] = managed_profile
    if pattern_skip_reasons:
        selected_skip = pattern_skip_reasons[-1]
        if selected_skip.get("reason") in {"canary_holdout", LOCAL_CANARY_SAFETY_STOP_REASON}:
            meta["pattern_rule"] = {
                key: selected_skip.get(key)
                for key in ("rule_id", "candidate_id", "policy_source", "matched_hashes", "canary", "safety_stop")
                if selected_skip.get(key) is not None
            }
            if isinstance(selected_skip.get("canary"), dict):
                meta["canary"] = selected_skip["canary"]
                meta["canary_cohort"] = selected_skip["canary"].get("cohort")
    return exact_enabled, meta


def stream_cache_payload(
    frames: list[bytes],
    *,
    provider: str,
    usage: dict[str, Any] | None = None,
    output_text: str | None = None,
) -> dict[str, Any]:
    return {
        "agentflow_cache_type": "sse-stream",
        "version": 1,
        "provider": provider,
        "frames_b64": [base64.b64encode(frame).decode("ascii") for frame in frames],
        "sse": stream_cache_sse_metadata(frames, provider=provider),
        "usage": usage or {},
        "output_text": output_text or "",
    }


def is_stream_cache_payload(payload: Any, *, provider: str | None = None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("agentflow_cache_type") != "sse-stream":
        return False
    if provider is not None and payload.get("provider") != provider:
        return False
    return isinstance(payload.get("frames_b64"), list)


def stream_cache_frames(payload: dict[str, Any]) -> list[bytes]:
    frames: list[bytes] = []
    for item in payload.get("frames_b64") or []:
        if isinstance(item, str):
            frames.append(base64.b64decode(item.encode("ascii")))
    return frames


def _sse_event_name(frame_text: str) -> str | None:
    for line in frame_text.splitlines():
        if line.startswith("event:"):
            return line[6:].strip() or None
    return None


def stream_cache_sse_metadata(frames: list[bytes], *, provider: str | None = None) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    for frame in frames:
        try:
            text = frame.decode("utf-8")
        except UnicodeDecodeError:
            continue
        event_name = _sse_event_name(text)
        if event_name:
            event_counts[event_name] = event_counts.get(event_name, 0) + 1
    metadata = {
        "schema": "agentflow.stream_cache_sse.v1",
        "media_type": "text/event-stream",
        "frame_count": len(frames),
        "event_counts": event_counts,
    }
    if provider:
        metadata["provider"] = provider
    if provider == "anthropic":
        metadata["anthropic_message_start"] = event_counts.get("message_start", 0) > 0
        metadata["anthropic_message_stop"] = event_counts.get("message_stop", 0) > 0
        metadata["complete"] = bool(metadata["anthropic_message_start"] and metadata["anthropic_message_stop"])
    return metadata


def validate_stream_cache_payload(payload: Any, *, provider: str | None = None) -> tuple[list[bytes], dict[str, Any]]:
    validation = {
        "schema": "agentflow.stream_cache_validation.v1",
        "valid": False,
        "reason": "invalid-envelope",
        "provider": provider,
        "raw_payload_included": False,
    }
    if not is_stream_cache_payload(payload, provider=provider):
        return [], validation

    raw_frames = payload.get("frames_b64") or []
    if not raw_frames:
        validation["reason"] = "frames-missing"
        return [], validation

    frames: list[bytes] = []
    event_names: list[str] = []
    for index, item in enumerate(raw_frames):
        if not isinstance(item, str):
            validation.update({"reason": "frame-not-string", "frame_index": index})
            return [], validation
        try:
            frame = base64.b64decode(item.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError):
            validation.update({"reason": "invalid-base64", "frame_index": index})
            return [], validation
        if not frame:
            validation.update({"reason": "frame-empty", "frame_index": index})
            return [], validation
        try:
            text = frame.decode("utf-8")
        except UnicodeDecodeError:
            validation.update({"reason": "invalid-utf8", "frame_index": index})
            return [], validation
        if not frame.endswith(b"\n\n"):
            validation.update({"reason": "unterminated-sse-frame", "frame_index": index})
            return [], validation
        if not any(line.startswith("data:") for line in text.splitlines()):
            validation.update({"reason": "sse-data-missing", "frame_index": index})
            return [], validation
        event_name = _sse_event_name(text)
        if event_name:
            event_names.append(event_name)
        frames.append(frame)

    if provider == "anthropic":
        if not event_names or event_names[0] != "message_start":
            validation["reason"] = "anthropic-message-start-missing"
            return [], validation
        if "message_stop" not in event_names:
            validation["reason"] = "anthropic-message-stop-missing"
            return [], validation

    validation.update(stream_cache_sse_metadata(frames, provider=provider))
    validation.update({
        "valid": True,
        "reason": "ok",
        "event_names": event_names[:20],
    })
    return frames, validation


def _default_cache_provider() -> str:
    return os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()


def _default_cache_upstream(provider: str) -> str:
    if provider == "openai":
        return os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com").rstrip("/")
    return os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com").rstrip("/")


def cache_key_for(
    body: dict[str, Any],
    path: str,
    *,
    provider: str | None = None,
    upstream: str | None = None,
    namespace: str | None = None,
    replay_scope: str | None = None,
    replay_scope_id: str | None = None,
) -> str:
    # Do not include auth. Include endpoint and body after crunch/routing.
    # Namespacing prevents cache reuse across providers, upstreams, or user-selected projects.
    provider = (provider or _default_cache_provider()).lower()
    upstream = (upstream or _default_cache_upstream(provider)).rstrip("/")
    namespace = namespace if namespace is not None else os.getenv("AGENTFLOW_CACHE_NAMESPACE", "default")
    key_payload: dict[str, Any] = {
        "version": 2,
        "namespace": namespace,
        "provider": provider,
        "upstream": upstream,
        "path": path,
        "body": body,
    }
    if replay_scope:
        key_payload["replay_scope"] = replay_scope
        key_payload["replay_scope_id"] = replay_scope_id
    key_material = stable_json(key_payload)
    return sha256_text(key_material)


def response_output_text(resp: dict[str, Any]) -> str:
    parts = []
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)
