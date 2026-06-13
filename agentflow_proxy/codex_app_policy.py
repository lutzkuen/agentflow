from __future__ import annotations

import copy
import os
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.paths import agentflow_config_path
from agentflow_proxy.policy_files import policy_file_snapshot, policy_file_status, utc_now
from agentflow_proxy.public_metadata import public_id, public_label, public_path_state


DEFAULT_CODEX_APP_UPSTREAM = "ws://127.0.0.1:4014"
CODEX_TURN_SOURCE_SURFACE = "codex_turn"
LEGACY_CODEX_APP_SOURCE_SURFACE = "codex_app_turn"
CODEX_APP_SOURCE_SURFACE = CODEX_TURN_SOURCE_SURFACE
CODEX_APP_SOURCE_SURFACE_ALIASES = frozenset({
    CODEX_TURN_SOURCE_SURFACE,
    LEGACY_CODEX_APP_SOURCE_SURFACE,
})
CODEX_APP_POLICY_CONDITION_KEYS = (
    "app_family",
    "workflow_phase",
    "granularity",
    "model_field_state",
    "input_size_bucket",
    "cache_eligible",
    "cache_status",
    "replayability_level",
    "has_action_like_params",
    "stale_risk",
    "stale_risk_signal",
    "supported_action_family",
)
CODEX_APP_POLICY_ACTION_KEYS = (
    "recommended_model",
    "model_hint",
    "crunch_profile",
    "cache_eligible",
    "cache_eligibility_reason",
    "pass_through_reason",
    "reason",
)
CODEX_APP_POLICY_ACTION_FAMILIES = ("routing", "crunch", "cache")
CODEX_TERMINAL_TRANSCRIPT_COMPACTION_CONDITION_KEYS = (
    "source_surface",
    "app_family",
    "granularity",
    "workflow_phase",
    "text_bucket",
    "input_size_bucket",
    "terminal_fraction_bucket",
    "terminal_output_char_fraction_bucket",
    "terminal_event_count_bucket",
    "terminal_signal_source",
    "cache_status",
    "already_crunched_repeated_scaffold",
    "safety_preserve_diagnostics",
    "min_input_chars",
    "min_terminal_chars",
    "min_projected_saved_chars",
)
CODEX_TERMINAL_TRANSCRIPT_COMPACTION_ACTION_KEYS = (
    "type",
    "keep_recent_turns",
    "min_block_chars",
    "head_lines",
    "tail_lines",
    "max_evidence_lines",
    "min_saved_chars",
    "preserve_diagnostics",
    "preserve_tool_protocol",
    "preserve_recent_turns",
    "preserve_error_lines",
)

CODEX_ACTION_KEY_HINTS = {
    "approval",
    "approvalrequest",
    "approval_request",
    "apply_patch",
    "cmd",
    "command",
    "exec",
    "function_call",
    "patch",
    "shell",
    "tool_call",
    "tool_calls",
}
CODEX_ACTION_VALUE_HINTS = {
    "approval_request",
    "apply_patch",
    "command",
    "exec",
    "function_call",
    "shell",
    "tool_call",
    "tool_result",
    "tool_use",
}
CODEX_MODEL_FIELDS = ("model", "modelId", "model_id")
CODEX_MODEL_STATE_FIELDS = (
    "model",
    "modelId",
    "model_id",
    "activeModel",
    "active_model",
    "defaultModel",
    "default_model",
    "modelName",
    "model_name",
)
CODEX_MODEL_STATE_SKIP_KEYS = {
    "cmd",
    "command",
    "content",
    "input",
    "input_text",
    "inputtext",
    "instructions",
    "message",
    "messages",
    "patch",
    "prompt",
    "raw_request",
    "raw_response",
    "result",
    "response",
    "text",
    "tool_call",
    "tool_calls",
    "tool_result",
    "tool_results",
}
CODEX_SAFE_TURN_PARAM_KEYS = {
    "input",
    "instructions",
    "max_tokens",
    "maxTokens",
    "model",
    "modelId",
    "model_id",
    "temperature",
    "threadId",
    "thread_id",
    "top_p",
    "topP",
}
CODEX_TEXT_INPUT_TYPES = {"text", "input_text"}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return default


def _bounded_fraction(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return min(max(parsed, 0.0), 1.0)


def _bounded_int(value: Any, default: int, *, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _bounded_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return max(minimum, parsed)


def _env_bool(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() not in {"0", "false", "no", "off", ""}


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(agentflow_config_path(filename))
    return candidates


def _default_codex_app_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "summary_model_hint": {
            "enabled": False,
            "target_model": "gpt-5-codex",
            "canary": {
                "fraction": 1.0,
                "holdout_fraction": 0.0,
                "salt": "codex-app-summary-model-hint",
                "unit": "source_hash",
            },
        },
        "exact_cache": {
            "enabled": False,
            "namespace": os.getenv("AGENTFLOW_CACHE_NAMESPACE", "default"),
            "ttl_seconds": 24 * 60 * 60,
            "canary": {
                "fraction": 1.0,
                "holdout_fraction": 0.0,
                "salt": "codex-app-exact-cache",
                "unit": "source_hash",
            },
        },
        "crunch": {
            "profiles": ["codex-repeated-scaffolding"],
        },
        "terminal_transcript_compaction": {
            "enabled": False,
            "review_only": True,
            "policy_source": "local-default",
            "rule_id": "local-codex-terminal-transcript-compaction",
            "candidate_id": None,
            "action_id": None,
            "conditions": {
                "source_surface": CODEX_TURN_SOURCE_SURFACE,
                "app_family": "codex",
                "granularity": "agent_turn",
                "workflow_phase": "tool_execution",
                "text_bucket": ["32k_128k_chars", "gte_128k_chars"],
                "terminal_fraction_bucket": ["50_75pct", "gte_75pct"],
                "terminal_event_count_bucket": ["6_20", "21_100", "101_1000", "1000_plus"],
                "terminal_signal_source": [
                    "input-terminal-features",
                    "event-window-terminal-events",
                    "input-terminal-features+event-window",
                ],
                "cache_status": ["miss", "skipped", "eligible"],
                "already_crunched_repeated_scaffold": False,
                "safety_preserve_diagnostics": True,
                "min_input_chars": 8_000,
                "min_terminal_chars": 2_000,
                "min_projected_saved_chars": 500,
            },
            "action": {
                "type": "compact_terminal_transcript",
                "keep_recent_turns": 2,
                "min_block_chars": 2_000,
                "head_lines": 12,
                "tail_lines": 16,
                "max_evidence_lines": 80,
                "min_saved_chars": 500,
                "preserve_diagnostics": True,
                "preserve_tool_protocol": True,
                "preserve_recent_turns": True,
                "preserve_error_lines": True,
            },
            "canary": {
                "enabled": True,
                "fraction": 0.0,
                "holdout_fraction": 1.0,
                "salt": "",
                "unit": "source_hash",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 5,
                "window": 500,
                "max_error_rate": 0.1,
                "max_retry_rate": 0.25,
                "max_negative_savings_rate": 0.25,
                "max_error_rate_delta": 0.05,
            },
            "provenance": {
                "schema": "agentflow.codex_terminal_transcript_compaction_policy.v1",
                "issuer": "local-agentflow",
                "status": "local-default",
            },
            "rules": [],
        },
        "rules": [],
    }


def _sanitize_codex_terminal_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    aliases = {
        "phase": "workflow_phase",
        "source": "terminal_signal_source",
        "terminal_output_fraction_bucket": "terminal_fraction_bucket",
        "terminal_output_char_fraction_bucket": "terminal_fraction_bucket",
    }
    allowed = set(CODEX_TERMINAL_TRANSCRIPT_COMPACTION_CONDITION_KEYS)
    sanitized: dict[str, Any] = {}
    for key, raw in value.items():
        key_text = aliases.get(str(key), str(key))
        if key_text not in allowed or raw is None:
            continue
        if isinstance(raw, (str, int, float, bool)):
            sanitized[key_text] = raw
        elif isinstance(raw, list):
            clean = [item for item in raw if isinstance(item, (str, int, float, bool))]
            if clean:
                sanitized[key_text] = clean
    return sanitized


def _sanitize_codex_terminal_action(value: Any, base: dict[str, Any]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    action = dict(base)
    if source.get("type") is not None:
        action["type"] = public_label(source.get("type"), "compact_terminal_transcript")
    for key in ("keep_recent_turns", "min_block_chars", "head_lines", "tail_lines", "max_evidence_lines", "min_saved_chars"):
        if source.get(key) is not None:
            action[key] = _bounded_int(source.get(key), int(action.get(key) or 0))
    for key in ("preserve_diagnostics", "preserve_tool_protocol", "preserve_recent_turns", "preserve_error_lines"):
        if source.get(key) is not None:
            action[key] = _as_bool(source.get(key), bool(action.get(key, True)))
    return action


def _sanitize_codex_terminal_provenance(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    labels = {}
    for key in ("schema", "issuer", "server_id", "key_id", "algorithm", "status"):
        if value.get(key) is not None:
            labels[key] = public_label(value.get(key), "redacted")
    for key in ("decision_hash", "signature"):
        if value.get(key) is not None:
            labels[key] = public_id(value.get(key), prefix=key.replace("_", "-"), fallback="redacted")
    for key in ("verified", "generated_at", "expires_at"):
        if isinstance(value.get(key), (str, int, float, bool)):
            labels[key] = value[key]
    return labels or None


def _apply_codex_terminal_transcript_compaction_yaml(
    policy: dict[str, Any],
    data: dict[str, Any],
    *,
    default_policy_source: str,
) -> None:
    target = policy["terminal_transcript_compaction"]
    target["enabled"] = _as_bool(data.get("enabled"), target["enabled"])
    target["review_only"] = _as_bool(data.get("review_only"), target["review_only"])
    target["policy_source"] = str(data.get("policy_source") or target.get("policy_source") or default_policy_source)
    for key in ("rule_id", "candidate_id", "action_id"):
        if data.get(key) is not None:
            target[key] = str(data[key]).strip()
    if isinstance(data.get("conditions"), dict):
        target["conditions"] = _sanitize_codex_terminal_conditions(data["conditions"])
    action_data = data.get("action") if isinstance(data.get("action"), dict) else data
    target["action"] = _sanitize_codex_terminal_action(action_data, target["action"])
    canary = data.get("canary") or {}
    if isinstance(canary, dict):
        target_canary = target["canary"]
        target_canary["enabled"] = _as_bool(canary.get("enabled"), target_canary["enabled"])
        for source_key, target_key in (
            ("fraction", "fraction"),
            ("canary_fraction", "fraction"),
            ("rollout_fraction", "fraction"),
            ("holdout_fraction", "holdout_fraction"),
        ):
            if canary.get(source_key) is not None:
                target_canary[target_key] = _bounded_fraction(canary.get(source_key), target_canary[target_key])
        if canary.get("salt") is not None:
            target_canary["salt"] = str(canary["salt"])
        if canary.get("canary_salt") is not None:
            target_canary["salt"] = str(canary["canary_salt"])
        if canary.get("unit") is not None:
            target_canary["unit"] = public_label(canary.get("unit"), "source_hash")
        if canary.get("canary_unit") is not None:
            target_canary["unit"] = public_label(canary.get("canary_unit"), "source_hash")
    safety = data.get("safety_stop") or {}
    if isinstance(safety, dict):
        target_safety = target["safety_stop"]
        target_safety["enabled"] = _as_bool(safety.get("enabled"), target_safety["enabled"])
        for key in ("min_outcome_samples", "window"):
            if safety.get(key) is not None:
                target_safety[key] = _bounded_int(safety.get(key), int(target_safety.get(key) or 0))
        for key in ("max_error_rate", "max_retry_rate", "max_negative_savings_rate", "max_error_rate_delta"):
            if safety.get(key) is not None:
                target_safety[key] = _bounded_float(safety.get(key), float(target_safety.get(key) or 0.0))
    if isinstance(data.get("provenance"), dict):
        target["provenance"] = _sanitize_codex_terminal_provenance(data["provenance"])
    rules = data.get("rules")
    if isinstance(rules, list):
        parsed: list[dict[str, Any]] = []
        for index, item in enumerate(rules):
            if not isinstance(item, dict):
                continue
            rule = copy.deepcopy({key: value for key, value in target.items() if key not in {"rules"}})
            rule["enabled"] = _as_bool(item.get("enabled"), bool(rule.get("enabled")))
            rule["review_only"] = _as_bool(item.get("review_only"), bool(rule.get("review_only", True)))
            rule["policy_source"] = str(item.get("policy_source") or rule.get("policy_source") or default_policy_source)
            rule["rule_id"] = str(
                item.get("id")
                or item.get("rule_id")
                or item.get("policy_id")
                or item.get("candidate_id")
                or f"codex-terminal-transcript-compaction-rule-{index + 1}"
            )
            for key in ("candidate_id", "action_id"):
                if item.get(key) is not None:
                    rule[key] = str(item[key]).strip()
            if item.get("rollout_action_id") is not None and not rule.get("action_id"):
                rule["action_id"] = str(item["rollout_action_id"]).strip()
            if isinstance(item.get("conditions"), dict):
                rule["conditions"] = _sanitize_codex_terminal_conditions(item["conditions"])
            rule["action"] = _sanitize_codex_terminal_action(item.get("action") if isinstance(item.get("action"), dict) else item, rule["action"])
            if isinstance(item.get("canary"), dict):
                _apply_codex_terminal_transcript_compaction_yaml(
                    {"terminal_transcript_compaction": rule},
                    {"canary": item["canary"]},
                    default_policy_source=rule["policy_source"],
                )
            if isinstance(item.get("safety_stop"), dict):
                _apply_codex_terminal_transcript_compaction_yaml(
                    {"terminal_transcript_compaction": rule},
                    {"safety_stop": item["safety_stop"]},
                    default_policy_source=rule["policy_source"],
                )
            if isinstance(item.get("provenance"), dict):
                rule["provenance"] = _sanitize_codex_terminal_provenance(item["provenance"])
            parsed.append(rule)
        target["rules"] = parsed


def _apply_codex_app_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    if "enabled" in data:
        policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])

    hint = data.get("summary_model_hint") or {}
    if isinstance(hint, dict):
        policy["summary_model_hint"]["enabled"] = _as_bool(
            hint.get("enabled"),
            policy["summary_model_hint"]["enabled"],
        )
        target_model = hint.get("target_model", hint.get("model_hint"))
        if target_model is not None:
            policy["summary_model_hint"]["target_model"] = str(target_model).strip()
        canary = hint.get("canary") or {}
        if isinstance(canary, dict):
            policy_canary = policy["summary_model_hint"]["canary"]
            policy_canary["fraction"] = _bounded_fraction(canary.get("fraction"), policy_canary["fraction"])
            policy_canary["holdout_fraction"] = _bounded_fraction(
                canary.get("holdout_fraction"),
                policy_canary["holdout_fraction"],
            )
            if canary.get("salt") is not None:
                policy_canary["salt"] = str(canary["salt"]).strip() or policy_canary["salt"]
            if canary.get("unit") is not None:
                unit = str(canary["unit"]).strip().lower().replace("-", "_")
                if unit in {"source_hash", "thread_id", "model_and_size"}:
                    policy_canary["unit"] = unit

    exact_cache = data.get("exact_cache", data.get("cache") if isinstance(data.get("cache"), dict) else {})
    if isinstance(exact_cache, dict):
        policy["exact_cache"]["enabled"] = _as_bool(
            exact_cache.get("enabled"),
            policy["exact_cache"]["enabled"],
        )
        if exact_cache.get("namespace") is not None:
            policy["exact_cache"]["namespace"] = str(exact_cache["namespace"]).strip() or "default"
        if exact_cache.get("ttl_seconds") is not None:
            try:
                policy["exact_cache"]["ttl_seconds"] = max(0, int(exact_cache["ttl_seconds"]))
            except (TypeError, ValueError):
                pass
        canary = exact_cache.get("canary") or {}
        if isinstance(canary, dict):
            policy_canary = policy["exact_cache"]["canary"]
            policy_canary["fraction"] = _bounded_fraction(canary.get("fraction"), policy_canary["fraction"])
            policy_canary["holdout_fraction"] = _bounded_fraction(
                canary.get("holdout_fraction"),
                policy_canary["holdout_fraction"],
            )
            if canary.get("salt") is not None:
                policy_canary["salt"] = str(canary["salt"]).strip() or policy_canary["salt"]
            if canary.get("unit") is not None:
                unit = str(canary["unit"]).strip().lower().replace("-", "_")
                if unit in {"source_hash", "thread_id", "model_and_size"}:
                    policy_canary["unit"] = unit

    crunch = data.get("crunch") or {}
    if isinstance(crunch, dict):
        profiles = crunch.get("profiles")
        if isinstance(profiles, list):
            policy["crunch"]["profiles"] = [str(profile).strip() for profile in profiles if str(profile).strip()]
    terminal_compaction = data.get("terminal_transcript_compaction")
    if isinstance(terminal_compaction, dict):
        _apply_codex_terminal_transcript_compaction_yaml(
            policy,
            terminal_compaction,
            default_policy_source=str(data.get("policy_source") or "local-manual"),
        )
    rules = data.get("rules")
    if isinstance(rules, list):
        normalized_rules: list[dict[str, Any]] = []
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
            action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
            normalized: dict[str, Any] = {
                "id": str(rule.get("id") or rule.get("rule_id") or f"codex-app-rule-{index + 1}").strip(),
                "conditions": dict(conditions),
                "action": dict(action),
            }
            for key in ("candidate_id", "recommendation_id", "policy_id", "policy_source"):
                value = rule.get(key)
                if value is not None:
                    normalized[key] = str(value).strip()
            canary = rule.get("canary")
            if isinstance(canary, dict):
                normalized["canary"] = dict(canary)
            safety_stop = rule.get("safety_stop")
            if isinstance(safety_stop, dict):
                normalized["safety_stop"] = dict(safety_stop)
            managed = rule.get("managed_recommendation")
            if isinstance(managed, dict):
                normalized["managed_recommendation"] = {
                    key: value
                    for key, value in managed.items()
                    if key in {"candidate_id", "recommendation_id", "policy_id", "reason", "canary"}
                }
            normalized_rules.append(normalized)
        policy["rules"] = normalized_rules
    return policy


def _codex_terminal_canary_public(canary: Any) -> dict[str, Any]:
    source = canary if isinstance(canary, dict) else {}
    fraction = _bounded_fraction(source.get("fraction"), 0.0)
    holdout_fraction = _bounded_fraction(source.get("holdout_fraction"), max(0.0, 1.0 - fraction))
    return {
        "enabled": _as_bool(source.get("enabled"), True),
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "unit": public_label(source.get("unit") or "source_hash", "source_hash"),
        "salt_configured": bool(source.get("salt")),
    }


def _codex_terminal_safety_public(safety: Any) -> dict[str, Any]:
    source = safety if isinstance(safety, dict) else {}
    return {
        "enabled": _as_bool(source.get("enabled"), True),
        "min_outcome_samples": _bounded_int(source.get("min_outcome_samples"), 5),
        "window": _bounded_int(source.get("window"), 500),
        "max_error_rate": _bounded_float(source.get("max_error_rate"), 0.1),
        "max_retry_rate": _bounded_float(source.get("max_retry_rate"), 0.25),
        "max_negative_savings_rate": _bounded_float(source.get("max_negative_savings_rate"), 0.25),
        "max_error_rate_delta": _bounded_float(source.get("max_error_rate_delta"), 0.05),
    }


def _codex_terminal_action_public(action: Any) -> dict[str, Any]:
    source = action if isinstance(action, dict) else {}
    return {
        "type": public_label(source.get("type") or "compact_terminal_transcript", "compact_terminal_transcript"),
        "keep_recent_turns": _bounded_int(source.get("keep_recent_turns"), 2),
        "min_block_chars": _bounded_int(source.get("min_block_chars"), 2_000),
        "head_lines": _bounded_int(source.get("head_lines"), 12),
        "tail_lines": _bounded_int(source.get("tail_lines"), 16),
        "max_evidence_lines": _bounded_int(source.get("max_evidence_lines"), 80),
        "min_saved_chars": _bounded_int(source.get("min_saved_chars"), 500),
        "preserve_diagnostics": _as_bool(source.get("preserve_diagnostics"), True),
        "preserve_tool_protocol": _as_bool(source.get("preserve_tool_protocol"), True),
        "preserve_recent_turns": _as_bool(source.get("preserve_recent_turns"), True),
        "preserve_error_lines": _as_bool(source.get("preserve_error_lines"), True),
    }


def _codex_terminal_transcript_compaction_public_policy(
    policy: dict[str, Any] | None = None,
    *,
    include_rules: bool = True,
) -> dict[str, Any]:
    source = policy if isinstance(policy, dict) else CODEX_APP_POLICY["terminal_transcript_compaction"]
    canary = source.get("canary") if isinstance(source.get("canary"), dict) else {}
    public = {
        "schema": "agentflow.codex_terminal_transcript_compaction_policy.v1",
        "enabled": _as_bool(source.get("enabled"), False),
        "review_only": _as_bool(source.get("review_only"), True),
        "policy_source": public_label(source.get("policy_source") or CODEX_APP_POLICY_SOURCE, "unknown"),
        "rule_id": public_id(
            source.get("rule_id") or "local-codex-terminal-transcript-compaction",
            prefix="codex-terminal-transcript-rule",
            fallback="local-codex-terminal-transcript-compaction",
        ),
        "candidate_id": public_id(source.get("candidate_id"), prefix="codex-terminal-transcript-candidate")
        if source.get("candidate_id") is not None
        else None,
        "action_id": public_id(source.get("action_id"), prefix="codex-terminal-transcript-action")
        if source.get("action_id") is not None
        else None,
        "conditions": _sanitize_codex_terminal_conditions(source.get("conditions")),
        "action": _codex_terminal_action_public(source.get("action")),
        "canary": _codex_terminal_canary_public(canary),
        "safety_stop": _codex_terminal_safety_public(source.get("safety_stop")),
        "provenance": _sanitize_codex_terminal_provenance(source.get("provenance")),
        "rule_file": public_path_state(CODEX_APP_RULES_PATH),
        "raw_terminal_text_included": False,
        "raw_commands_included": False,
        "raw_paths_included": False,
        "raw_request_ids_included": False,
        "thread_ids_included": False,
        "secret_salt_included": False,
        "policy_file_contents_included": False,
        "runtime_mutation_enabled": False,
        "application": {
            "status": "review-only-not-applied",
            "reason": "Codex terminal-transcript compaction rules are loaded for dry-run/canary planning only.",
        },
    }
    if include_rules:
        raw_rules = source.get("rules") if isinstance(source.get("rules"), list) else []
        public["rules"] = [
            _codex_terminal_transcript_compaction_public_policy(rule, include_rules=False)
            for rule in raw_rules
            if isinstance(rule, dict)
        ]
        public["rule_count"] = len(public["rules"])
    return public


def codex_terminal_transcript_compaction_effective_policy() -> dict[str, Any]:
    return _codex_terminal_transcript_compaction_public_policy(
        CODEX_APP_POLICY.get("terminal_transcript_compaction")
        if isinstance(CODEX_APP_POLICY.get("terminal_transcript_compaction"), dict)
        else None,
        include_rules=True,
    )


def _load_codex_app_policy() -> tuple[dict[str, Any], str, str]:
    for path in _manual_rule_candidates("codex_app_rules.yaml", "AGENTFLOW_CODEX_APP_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return _apply_codex_app_policy_yaml(_default_codex_app_policy(), data), "local-manual", str(path)

    defaults_path = Path(__file__).parent / "codex_app_rules.yaml"
    policy = _default_codex_app_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _apply_codex_app_policy_yaml(policy, data)

    policy["enabled"] = _env_bool("AGENTFLOW_CODEX_APP_OPTIMIZE", policy["enabled"])
    policy["summary_model_hint"]["enabled"] = _env_bool(
        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT",
        policy["summary_model_hint"]["enabled"],
    )
    policy["summary_model_hint"]["target_model"] = os.getenv(
        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_TARGET",
        os.getenv(
            "AGENTFLOW_CODEX_APP_SUMMARY_TARGET_MODEL",
            str(policy["summary_model_hint"]["target_model"]),
        ),
    ).strip()
    policy["summary_model_hint"]["canary"]["fraction"] = _bounded_fraction(
        os.getenv("AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_CANARY_FRACTION"),
        policy["summary_model_hint"]["canary"]["fraction"],
    )
    policy["summary_model_hint"]["canary"]["holdout_fraction"] = _bounded_fraction(
        os.getenv("AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_HOLDOUT_FRACTION"),
        policy["summary_model_hint"]["canary"]["holdout_fraction"],
    )
    policy["summary_model_hint"]["canary"]["salt"] = os.getenv(
        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_CANARY_SALT",
        str(policy["summary_model_hint"]["canary"]["salt"]),
    ).strip() or "codex-app-summary-model-hint"
    unit = os.getenv(
        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_CANARY_UNIT",
        str(policy["summary_model_hint"]["canary"]["unit"]),
    ).strip().lower().replace("-", "_")
    if unit in {"source_hash", "thread_id", "model_and_size"}:
        policy["summary_model_hint"]["canary"]["unit"] = unit
    policy["exact_cache"]["enabled"] = _env_bool("AGENTFLOW_CODEX_APP_CACHE", policy["exact_cache"]["enabled"])
    policy["exact_cache"]["namespace"] = os.getenv(
        "AGENTFLOW_CODEX_APP_CACHE_NAMESPACE",
        os.getenv("AGENTFLOW_CACHE_NAMESPACE", str(policy["exact_cache"]["namespace"])),
    ).strip() or "default"
    try:
        policy["exact_cache"]["ttl_seconds"] = max(
            0,
            int(os.getenv("AGENTFLOW_CODEX_APP_CACHE_TTL_SECONDS", str(policy["exact_cache"]["ttl_seconds"]))),
        )
    except ValueError:
        pass
    policy["exact_cache"]["canary"]["fraction"] = _bounded_fraction(
        os.getenv("AGENTFLOW_CODEX_APP_CACHE_CANARY_FRACTION"),
        policy["exact_cache"]["canary"]["fraction"],
    )
    policy["exact_cache"]["canary"]["holdout_fraction"] = _bounded_fraction(
        os.getenv("AGENTFLOW_CODEX_APP_CACHE_HOLDOUT_FRACTION"),
        policy["exact_cache"]["canary"]["holdout_fraction"],
    )
    policy["exact_cache"]["canary"]["salt"] = os.getenv(
        "AGENTFLOW_CODEX_APP_CACHE_CANARY_SALT",
        str(policy["exact_cache"]["canary"]["salt"]),
    ).strip() or "codex-app-exact-cache"
    cache_unit = os.getenv(
        "AGENTFLOW_CODEX_APP_CACHE_CANARY_UNIT",
        str(policy["exact_cache"]["canary"]["unit"]),
    ).strip().lower().replace("-", "_")
    if cache_unit in {"source_hash", "thread_id", "model_and_size"}:
        policy["exact_cache"]["canary"]["unit"] = cache_unit
    return policy, "local-default", str(defaults_path)


def canonical_source_surface(value: Any) -> str:
    surface = str(value or "").strip()
    if surface in CODEX_APP_SOURCE_SURFACE_ALIASES:
        return CODEX_TURN_SOURCE_SURFACE
    return surface or "unknown"


def is_codex_turn_source_surface(value: Any) -> bool:
    return canonical_source_surface(value) == CODEX_TURN_SOURCE_SURFACE


def _normalized_model_field(value: Any) -> str:
    return str(value or "").replace("-", "_").lower()


def _normalized_model_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 100:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/")
    if any(char not in allowed for char in cleaned):
        return None
    return cleaned


def codex_model_state_signal(method: Any, params: Any) -> dict[str, Any] | None:
    if not isinstance(params, dict):
        return None
    model_fields = {_normalized_model_field(field) for field in CODEX_MODEL_STATE_FIELDS}
    stack: list[Any] = [params]
    explicit_absent: dict[str, Any] | None = None
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_s = str(key)
                normalized_key = _normalized_model_field(key_s)
                if normalized_key in model_fields:
                    normalized = _normalized_model_value(value)
                    if normalized:
                        return {
                            "state": "derived_present",
                            "field": key_s,
                            "normalized_model": normalized,
                            "source_method": str(method or "unknown"),
                            "confidence": "high",
                            "reason": "metadata-model-field",
                        }
                    if value is None or value == "":
                        explicit_absent = {
                            "state": "derived_absent",
                            "field": key_s,
                            "normalized_model": None,
                            "source_method": str(method or "unknown"),
                            "confidence": "high",
                            "reason": "metadata-model-field-empty",
                        }
                elif normalized_key not in CODEX_MODEL_STATE_SKIP_KEYS and isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (dict, list)))
    return explicit_absent


CODEX_APP_POLICY, CODEX_APP_POLICY_SOURCE, CODEX_APP_RULES_PATH = _load_codex_app_policy()
CODEX_APP_RULES_LOADED_AT = utc_now()
CODEX_APP_RULES_LOADED_FILE = policy_file_snapshot(CODEX_APP_RULES_PATH)


def codex_app_optimize_enabled() -> bool:
    if CODEX_APP_POLICY_SOURCE == "local-default":
        return _env_bool("AGENTFLOW_CODEX_APP_OPTIMIZE", bool(CODEX_APP_POLICY["enabled"]))
    return bool(CODEX_APP_POLICY["enabled"])


def codex_app_cache_enabled() -> bool:
    if CODEX_APP_POLICY_SOURCE == "local-default":
        return _env_bool("AGENTFLOW_CODEX_APP_CACHE", bool(CODEX_APP_POLICY["exact_cache"]["enabled"]))
    return bool(CODEX_APP_POLICY["exact_cache"]["enabled"])


def codex_app_summary_model_hint_enabled() -> bool:
    if CODEX_APP_POLICY_SOURCE == "local-default":
        return _env_bool(
            "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT",
            bool(CODEX_APP_POLICY["summary_model_hint"]["enabled"]),
        )
    return bool(CODEX_APP_POLICY["summary_model_hint"]["enabled"])


def codex_app_summary_model_hint_target() -> str:
    if CODEX_APP_POLICY_SOURCE == "local-default":
        return os.getenv(
            "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_TARGET",
            os.getenv(
                "AGENTFLOW_CODEX_APP_SUMMARY_TARGET_MODEL",
                str(CODEX_APP_POLICY["summary_model_hint"]["target_model"]),
            ),
        ).strip()
    return str(CODEX_APP_POLICY["summary_model_hint"]["target_model"]).strip()


def codex_app_summary_model_hint_canary() -> dict[str, Any]:
    canary = CODEX_APP_POLICY["summary_model_hint"].get("canary")
    if not isinstance(canary, dict):
        canary = {}
    if CODEX_APP_POLICY_SOURCE == "local-default":
        fraction = _bounded_fraction(
            os.getenv("AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_CANARY_FRACTION"),
            _bounded_fraction(canary.get("fraction"), 1.0),
        )
        holdout_fraction = _bounded_fraction(
            os.getenv("AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_HOLDOUT_FRACTION"),
            _bounded_fraction(canary.get("holdout_fraction"), 0.0),
        )
        salt = os.getenv(
            "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_CANARY_SALT",
            str(canary.get("salt") or "codex-app-summary-model-hint"),
        ).strip() or "codex-app-summary-model-hint"
        unit = os.getenv(
            "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_CANARY_UNIT",
            str(canary.get("unit") or "source_hash"),
        ).strip().lower().replace("-", "_")
    else:
        fraction = _bounded_fraction(canary.get("fraction"), 1.0)
        holdout_fraction = _bounded_fraction(canary.get("holdout_fraction"), 0.0)
        salt = str(canary.get("salt") or "codex-app-summary-model-hint").strip()
        unit = str(canary.get("unit") or "source_hash").strip().lower().replace("-", "_")
    if unit not in {"source_hash", "thread_id", "model_and_size"}:
        unit = "source_hash"
    return {
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "salt": salt,
        "unit": unit,
    }


def codex_app_cache_namespace() -> str:
    if CODEX_APP_POLICY_SOURCE == "local-default":
        return os.getenv(
            "AGENTFLOW_CODEX_APP_CACHE_NAMESPACE",
            os.getenv("AGENTFLOW_CACHE_NAMESPACE", str(CODEX_APP_POLICY["exact_cache"]["namespace"])),
        ).strip() or "default"
    return str(CODEX_APP_POLICY["exact_cache"]["namespace"]).strip() or "default"


def codex_app_cache_ttl_seconds() -> int:
    if CODEX_APP_POLICY_SOURCE == "local-default":
        try:
            return max(0, int(os.getenv(
                "AGENTFLOW_CODEX_APP_CACHE_TTL_SECONDS",
                str(CODEX_APP_POLICY["exact_cache"].get("ttl_seconds") or 0),
            )))
        except ValueError:
            return 0
    try:
        return max(0, int(CODEX_APP_POLICY["exact_cache"].get("ttl_seconds") or 0))
    except (TypeError, ValueError):
        return 0


def codex_app_cache_canary() -> dict[str, Any]:
    canary = CODEX_APP_POLICY["exact_cache"].get("canary")
    if not isinstance(canary, dict):
        canary = {}
    if CODEX_APP_POLICY_SOURCE == "local-default":
        fraction = _bounded_fraction(
            os.getenv("AGENTFLOW_CODEX_APP_CACHE_CANARY_FRACTION"),
            _bounded_fraction(canary.get("fraction"), 1.0),
        )
        holdout_fraction = _bounded_fraction(
            os.getenv("AGENTFLOW_CODEX_APP_CACHE_HOLDOUT_FRACTION"),
            _bounded_fraction(canary.get("holdout_fraction"), 0.0),
        )
        salt = os.getenv(
            "AGENTFLOW_CODEX_APP_CACHE_CANARY_SALT",
            str(canary.get("salt") or "codex-app-exact-cache"),
        ).strip() or "codex-app-exact-cache"
        unit = os.getenv(
            "AGENTFLOW_CODEX_APP_CACHE_CANARY_UNIT",
            str(canary.get("unit") or "source_hash"),
        ).strip().lower().replace("-", "_")
    else:
        fraction = _bounded_fraction(canary.get("fraction"), 1.0)
        holdout_fraction = _bounded_fraction(canary.get("holdout_fraction"), 0.0)
        salt = str(canary.get("salt") or "codex-app-exact-cache").strip()
        unit = str(canary.get("unit") or "source_hash").strip().lower().replace("-", "_")
    if unit not in {"source_hash", "thread_id", "model_and_size"}:
        unit = "source_hash"
    return {
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "salt": salt,
        "unit": unit,
    }


def codex_app_upstream() -> str:
    return os.getenv("AGENTFLOW_CODEX_APP_UPSTREAM", DEFAULT_CODEX_APP_UPSTREAM)


def codex_app_surface_policy_state(provider_policy_state: dict[str, Any]) -> dict[str, Any]:
    inherited_sections = {}
    reload_required_sections: list[str] = []
    for section in ("routing", "crunch", "cache"):
        policy = provider_policy_state.get(section)
        if not isinstance(policy, dict):
            continue
        file_status = policy.get("file") if isinstance(policy.get("file"), dict) else {}
        reload_required = bool(file_status.get("reload_required"))
        if reload_required:
            reload_required_sections.append(section)
        inherited_sections[section] = {
            "policy_source": policy.get("policy_source"),
            "rule_path": policy.get("rule_path"),
            "reload_required": reload_required,
            "file": file_status,
        }

    optimize_enabled = codex_app_optimize_enabled()
    cache_enabled = codex_app_cache_enabled()
    cache_ttl_seconds = codex_app_cache_ttl_seconds()
    cache_canary = codex_app_cache_canary()
    summary_model_hint_enabled = codex_app_summary_model_hint_enabled()
    summary_model_hint_target = codex_app_summary_model_hint_target()
    summary_model_hint_canary = codex_app_summary_model_hint_canary()
    terminal_compaction = codex_terminal_transcript_compaction_effective_policy()
    upstream = codex_app_upstream()
    namespace = codex_app_cache_namespace()
    file_state = policy_file_status(
        CODEX_APP_RULES_PATH,
        loaded_at=CODEX_APP_RULES_LOADED_AT,
        loaded_snapshot=CODEX_APP_RULES_LOADED_FILE,
    )
    reload_required = bool(file_state.get("reload_required"))
    policy_source = CODEX_APP_POLICY_SOURCE
    return {
        "surface": CODEX_APP_SOURCE_SURFACE,
        "name": "Codex app-server",
        "enabled": optimize_enabled,
        "policy_source": policy_source,
        "rule_path": CODEX_APP_RULES_PATH,
        "file": file_state,
        "runtime_flags": {
            "optimization_enabled": optimize_enabled,
            "cache_enabled": cache_enabled,
            "summary_model_hint_enabled": summary_model_hint_enabled,
            "terminal_transcript_compaction_enabled": bool(terminal_compaction.get("enabled")),
        },
        "optimization": {
            "enabled": optimize_enabled,
            "disabled_reason": None if optimize_enabled else "AGENTFLOW_CODEX_APP_OPTIMIZE=0",
            "scope": "metadata-only local JSON-RPC turn optimization",
        },
        "routing": {
            **inherited_sections.get("routing", {}),
            "summary_model_hint": {
                "enabled": summary_model_hint_enabled,
                "target_model": summary_model_hint_target,
                "canary": summary_model_hint_canary,
                "scope": "safe summary-phase turn/start frames with text-only input and known model field",
                "disabled_reason": None if summary_model_hint_enabled else "codex app summary model hint is disabled by local policy",
                "policy_source": policy_source,
            },
        },
        "crunch": inherited_sections.get("crunch", {}),
        "terminal_transcript_compaction": terminal_compaction,
        "cache": {
            **inherited_sections.get("cache", {}),
            "enabled": cache_enabled,
            "exact_cache": {
                "enabled": cache_enabled,
                "namespace": namespace,
                "ttl_seconds": cache_ttl_seconds,
                "canary": cache_canary,
                "provider": "codex-app",
                "upstream": upstream,
                "request_basis": "jsonrpc turn/start frame with request id removed",
                "cache_url": "codex-app://turn/start",
                "replayability_level": "local-exact-response",
            },
            "disabled_reason": None if cache_enabled else "AGENTFLOW_CODEX_APP_CACHE is not 1",
            "policy_source": policy_source,
        },
        "safe_turn_params": {
            "allowed_keys": sorted(CODEX_SAFE_TURN_PARAM_KEYS),
            "allowed_key_count": len(CODEX_SAFE_TURN_PARAM_KEYS),
            "model_fields": list(CODEX_MODEL_FIELDS),
            "text_input_types": sorted(CODEX_TEXT_INPUT_TYPES),
            "unknown_key_behavior": "skip-cache-and-summary-model-hint-and-keep-features-only",
        },
        "action_like_skip_behavior": {
            "enabled": True,
            "reason": "action-like-params",
            "applies_before": ["routing", "crunch", "cache"],
            "key_hints": sorted(CODEX_ACTION_KEY_HINTS),
            "value_hints": sorted(CODEX_ACTION_VALUE_HINTS),
        },
        "file_backed_policy_sections": inherited_sections,
        "reload_required": bool(reload_required_sections) or reload_required,
        "reload_required_sections": reload_required_sections + (["codex_app"] if reload_required else []),
        "managed_optimizer_required": False,
        "note": "Codex app-server local optimization is controlled by reviewable local policy; managed optimizer use remains opt-in.",
    }


def codex_app_bundle_policy_state() -> dict[str, Any]:
    optimize_enabled = codex_app_optimize_enabled()
    cache_enabled = codex_app_cache_enabled()
    cache_ttl_seconds = codex_app_cache_ttl_seconds()
    cache_canary = codex_app_cache_canary()
    hint_enabled = codex_app_summary_model_hint_enabled()
    hint_target = codex_app_summary_model_hint_target()
    hint_canary = codex_app_summary_model_hint_canary()
    terminal_compaction = codex_terminal_transcript_compaction_effective_policy()
    namespace = codex_app_cache_namespace()
    rules: list[dict[str, Any]] = []
    if hint_enabled:
        rules.append({
            "conditions": {
                "app_family": "codex",
                "workflow_phase": "summary",
                "model_field_state": "present",
                "cache_eligible": True,
                "has_action_like_params": False,
            },
            "action": {
                "model_hint": hint_target,
                "reason": "safe local summary-turn model hint",
            },
            "policy_source": CODEX_APP_POLICY_SOURCE,
        })
    if cache_enabled:
        rules.append({
            "conditions": {
                "app_family": "codex",
                "workflow_phase": "summary",
                "cache_eligible": True,
                "has_action_like_params": False,
            },
            "action": {
                "cache_eligible": True,
                "cache_eligibility_reason": "safe local exact summary-turn replay",
            },
            "policy_source": CODEX_APP_POLICY_SOURCE,
        })
    file_state = policy_file_status(
        CODEX_APP_RULES_PATH,
        loaded_at=CODEX_APP_RULES_LOADED_AT,
        loaded_snapshot=CODEX_APP_RULES_LOADED_FILE,
    )
    return {
        "enabled": optimize_enabled,
        "policy_source": CODEX_APP_POLICY_SOURCE,
        "rule_path": CODEX_APP_RULES_PATH,
        "file": file_state,
        "surface": CODEX_APP_SOURCE_SURFACE,
        "review_only": False,
        "runtime_flags": {
            "optimization_enabled": optimize_enabled,
            "cache_enabled": cache_enabled,
            "summary_model_hint_enabled": hint_enabled,
            "terminal_transcript_compaction_enabled": bool(terminal_compaction.get("enabled")),
        },
        "summary_model_hint": {
            "enabled": hint_enabled,
            "target_model": hint_target,
            "canary": hint_canary,
        },
        "exact_cache": {
            "enabled": cache_enabled,
            "namespace": namespace,
            "ttl_seconds": cache_ttl_seconds,
            "canary": cache_canary,
        },
        "crunch": {
            "profiles": list(CODEX_APP_POLICY.get("crunch", {}).get("profiles") or []),
        },
        "terminal_transcript_compaction": terminal_compaction,
        "rules": rules,
        "supported_conditions": list(CODEX_APP_POLICY_CONDITION_KEYS),
        "supported_actions": list(CODEX_APP_POLICY_ACTION_KEYS),
        "application": {
            "status": "applied-locally",
            "reason": "Safe Codex app summary hint and exact summary cache actions are applied by the local Codex app proxy when enabled.",
        },
        "managed_optimizer_required": False,
    }
