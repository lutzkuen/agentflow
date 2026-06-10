from __future__ import annotations

import base64
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
        },
        "pattern_rules": [],
    }


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(Path.home() / ".agentflow" / filename)
    return candidates


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
    policy["pattern_rules"] = _load_cache_pattern_rules(data.get("pattern_rules"))
    return policy


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
    for path in _manual_rule_candidates("cache_rules.yaml", "AGENTFLOW_CACHE_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return _apply_cache_policy_yaml(_default_cache_policy(), data), "local-manual", str(path)

    defaults_path = Path(__file__).parent / "cache_rules.yaml"
    policy = _default_cache_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _apply_cache_policy_yaml(policy, data)
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
    return policy, "local-default", str(defaults_path)


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
CACHE_PATTERN_RULES = tuple(CACHE_POLICY.get("pattern_rules") or [])


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
    pattern_features: dict[str, Any] | None,
    rule: dict[str, Any],
) -> str | None:
    if has_tool_blocks:
        return "streaming-tools-disabled"
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
        for key in ("workflow_phase", "source_surface", "app_family", "text_bucket", "token_bucket"):
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
    if estimated_saved_cost_usd is not None:
        meta["estimated_saved_cost_usd"] = round(max(0.0, float(estimated_saved_cost_usd)), 9)
    return meta


_PATH_TRAILING_JUNK = ".,;:)\\]}>\"'"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


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
        if path_like:
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
    candidate_count = 0
    seen: set[str] = set()
    for text in _walk_strings(body):
        for token in _candidate_path_tokens(text):
            resolved = _resolve_under_root(token, watch_root)
            if resolved is None:
                continue
            resolved_key = str(resolved)
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            candidate_count += 1
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
        candidate_count=candidate_count,
        max_paths=limit,
        root=watch_root,
    )
    return snapshots, audit


def cache_file_dependency_snapshots(
    body: dict[str, Any],
    *,
    root: str | Path | None = None,
    max_paths: int | None = None,
) -> list[dict[str, Any]]:
    if not CACHE_FILE_WATCH_ENABLED:
        return []
    snapshots, _audit = _cache_file_dependency_scan(body, root=root, max_paths=max_paths)
    return snapshots


def cache_file_dependency_audit(
    body: dict[str, Any] | None = None,
    *,
    snapshots: list[dict[str, Any]] | None = None,
    enabled: bool | None = None,
    candidate_count: int | None = None,
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
            snapshots, scanned = _cache_file_dependency_scan(body, root=watch_root, max_paths=limit)
            return scanned
    snapshot_count = len(snapshots or [])
    candidate_total = snapshot_count if candidate_count is None else max(0, int(candidate_count))
    cap_exceeded = candidate_total > limit
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
    return {
        "schema": "agentflow.cache_file_dependency_audit.v1",
        "file_watch_enabled": bool(watch_enabled),
        "snapshot_root_policy": _dependency_root_policy(watch_root),
        "root_path_included": False,
        "snapshot_count": snapshot_count,
        "snapshot_count_bucket": _dependency_count_bucket(snapshot_count),
        "candidate_path_count_bucket": _dependency_count_bucket(candidate_total),
        "max_paths": limit,
        "cap_exceeded": bool(cap_exceeded),
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
    return exact_enabled, semantic_enabled, meta


def streaming_cache_lookup_meta(
    has_tool_blocks: bool,
    *,
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
    elif pattern_skip_reasons:
        status = "skipped"
        reason = str(pattern_skip_reasons[-1].get("reason") or "streaming-pattern-rule-skipped")
    elif has_tool_blocks and CACHE_ENABLED:
        status = "skipped"
        reason = "streaming-tools-disabled"
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
) -> str:
    # Do not include auth. Include endpoint and body after crunch/routing.
    # Namespacing prevents cache reuse across providers, upstreams, or user-selected projects.
    provider = (provider or _default_cache_provider()).lower()
    upstream = (upstream or _default_cache_upstream(provider)).rstrip("/")
    namespace = namespace if namespace is not None else os.getenv("AGENTFLOW_CACHE_NAMESPACE", "default")
    key_material = stable_json({
        "version": 2,
        "namespace": namespace,
        "provider": provider,
        "upstream": upstream,
        "path": path,
        "body": body,
    })
    return sha256_text(key_material)


def response_output_text(resp: dict[str, Any]) -> str:
    parts = []
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)
