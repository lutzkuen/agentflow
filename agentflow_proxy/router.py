from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.paths import agentflow_config_path, default_db_path, safe_expanduser
from agentflow_proxy.pricing import estimate_cost
from agentflow_proxy.session_phase_memory import build_session_phase_memory_for_session
from agentflow_proxy.store import stable_json

HAIKU_DEFAULT = os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
SONNET_DEFAULT = os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4-6")
OPUS_DEFAULT = os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4-5")
OPENAI_LARGE_DEFAULT = os.getenv("AGENTFLOW_OPENAI_LARGE_MODEL", "gpt-5-codex")
OPENAI_SMALL_DEFAULT = os.getenv("AGENTFLOW_OPENAI_SMALL_MODEL", "gpt-5-mini")
OPENAI_TINY_DEFAULT = os.getenv("AGENTFLOW_OPENAI_TINY_MODEL", "gpt-5-nano")

ROUTING_ENABLED = os.getenv("AGENTFLOW_ROUTING", "1") != "0"
OPENAI_ROUTING_ENABLED = os.getenv("AGENTFLOW_OPENAI_ROUTING", "0") == "1"
ROUTING_RULES_PATH = os.getenv("AGENTFLOW_ROUTING_RULES", str(agentflow_config_path("routing_rules.yaml")))
STRIP_THINKING_HISTORY = os.getenv("AGENTFLOW_STRIP_THINKING_HISTORY", "0") == "1"


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return default


def extract_text(obj: Any) -> str:
    parts: list[str] = []
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            parts.append(extract_text(x))
    elif isinstance(obj, dict):
        # Only text-ish values for token estimate. Avoid binary/source blobs where possible.
        for k, v in obj.items():
            if k in {"text", "content", "input", "system", "name", "type"}:
                parts.append(extract_text(v))
            elif isinstance(v, (list, dict)):
                parts.append(extract_text(v))
    return "\n".join(p for p in parts if p)


def has_tools(body: dict[str, Any]) -> bool:
    if body.get("tools"):
        return True
    text = stable_json(body.get("messages", []))
    return "tool_use" in text or "tool_result" in text


def categorize_request(body: dict[str, Any]) -> str:
    tools = has_tools(body)
    text_chars = len(extract_text(body))
    msg_count = len(body.get("messages") or [])
    has_code = "```" in extract_text(body)

    messages = body.get("messages") or []
    if messages:
        last = messages[-1]
        if last.get("role") == "user":
            content = last.get("content", [])
            if isinstance(content, list) and content and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                return "tool-result"

    if tools and text_chars > 16000:
        return "tool-heavy"
    if tools:
        return "tool-light"
    if text_chars > 32000:
        return "long-context"
    if text_chars < 1500 and msg_count <= 2:
        return "short-completion"
    if has_code:
        return "code-gen"
    return "chat"


def _message_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _last_user_message(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def _last_user_text(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    return extract_text(message.get("content"))


def _assistant_tool_use_seen(messages: list[Any]) -> bool:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if any(block.get("type") == "tool_use" for block in _message_content_blocks(message)):
            return True
    return False


def _has_assistant_thinking_history(body: dict[str, Any]) -> bool:
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            if any(isinstance(b, dict) and b.get("type") == "thinking" for b in msg["content"]):
                return True
    return False


def classify_workflow_phase(body: dict[str, Any], category: str | None = None) -> dict[str, str]:
    """Classify Anthropic workflow phase using metadata-only request shape signals."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return {
            "workflow_phase": "unknown",
            "workflow_phase_reason": "missing-messages",
            "workflow_phase_confidence": "low",
        }

    category = category or categorize_request(body)
    text_chars = len(extract_text(body))
    msg_count = len([message for message in messages if isinstance(message, dict)])
    last_user = _last_user_message(messages)
    last_user_blocks = _message_content_blocks(last_user or {})
    last_user_text = _last_user_text(last_user).lower()
    last_user_block_types = {
        str(block.get("type") or "")
        for block in last_user_blocks
        if isinstance(block.get("type"), str)
    }

    if _has_top_level_thinking(body):
        return {
            "workflow_phase": "thinking",
            "workflow_phase_reason": "thinking-current-request",
            "workflow_phase_confidence": "high",
        }

    if "tool_result" in last_user_block_types:
        return {
            "workflow_phase": "tool-execution",
            "workflow_phase_reason": "last-user-tool-result",
            "workflow_phase_confidence": "high",
        }

    if _has_assistant_thinking_history(body):
        return {
            "workflow_phase": "thinking",
            "workflow_phase_reason": "thinking-history",
            "workflow_phase_confidence": "high",
        }

    summary_terms = ("summary", "summarize", "summarise", "recap", "final answer", "report back")
    if any(term in last_user_text for term in summary_terms):
        return {
            "workflow_phase": "summary",
            "workflow_phase_reason": "summary-intent-text",
            "workflow_phase_confidence": "medium",
        }
    if category == "short-completion" and msg_count > 2 and text_chars < 2000:
        return {
            "workflow_phase": "summary",
            "workflow_phase_reason": "short-late-conversation",
            "workflow_phase_confidence": "medium",
        }

    verification_terms = ("verify", "verification", "test", "tests", "failing", "failure", "error", "regression")
    if any(term in last_user_text for term in verification_terms):
        return {
            "workflow_phase": "verification",
            "workflow_phase_reason": "verification-intent-text",
            "workflow_phase_confidence": "medium",
        }
    if category == "code-gen":
        return {
            "workflow_phase": "verification",
            "workflow_phase_reason": "code-context",
            "workflow_phase_confidence": "low",
        }

    if has_tools(body) and msg_count <= 2 and not _assistant_tool_use_seen(messages):
        return {
            "workflow_phase": "planning",
            "workflow_phase_reason": "early-tool-capable-user-turn",
            "workflow_phase_confidence": "medium",
        }

    if category == "chat":
        return {
            "workflow_phase": "chat",
            "workflow_phase_reason": "non-tool-chat-category",
            "workflow_phase_confidence": "medium",
        }

    return {
        "workflow_phase": "unknown",
        "workflow_phase_reason": f"category-{category or 'unknown'}",
        "workflow_phase_confidence": "low",
    }


def uses_thinking(body: dict[str, Any]) -> bool:
    return _has_top_level_thinking(body) or _has_assistant_thinking_history(body)


def _has_top_level_thinking(body: dict[str, Any]) -> bool:
    if body.get("effort"):
        return True
    if body.get("interleaved_thinking"):
        return True
    thinking = body.get("thinking")
    if thinking:
        if isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "disabled":
            pass
        else:
            return True
    return False


def _thinking_gate_meta(body: dict[str, Any], category: str) -> dict[str, Any]:
    top_level = _has_top_level_thinking(body)
    historical = _has_assistant_thinking_history(body)
    if top_level or historical:
        return {
            "status": "blocked",
            "reason": "current-thinking-request" if top_level else "assistant-thinking-history",
            "top_level_thinking": top_level,
            "assistant_thinking_history": historical,
            "metadata_only": True,
        }
    return {
        "status": "clear",
        "reason": "no-thinking-signals",
        "top_level_thinking": False,
        "assistant_thinking_history": False,
        "metadata_only": True,
    }


def strip_thinking_history_blocks(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove type=='thinking' blocks from assistant messages; returns (modified_body, n_stripped)."""
    import copy
    body = copy.deepcopy(body)
    n_stripped = 0
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            filtered = [b for b in msg["content"] if not (isinstance(b, dict) and b.get("type") == "thinking")]
            n_stripped += len(msg["content"]) - len(filtered)
            msg["content"] = filtered
    return body, n_stripped


def _default_phase_canary_policy() -> dict[str, Any]:
    return {
        "enabled": False,
        "policy_id": "local-phase-sonnet-haiku-canary-v1",
        "provider": "anthropic",
        "source_surface": "anthropic_messages",
        "app_family": "anthropic",
        "model_pattern": "sonnet",
        "target_model": "haiku",
        "requested_model": "sonnet",
        "routed_model": "haiku",
        "eligible_workflow_phases": ["tool-execution", "summary"],
        "excluded_workflow_phases": ["planning", "thinking", "unknown"],
        "eligible_categories": [],
        "excluded_categories": ["code-gen"],
        "min_workflow_phase_confidence": "medium",
        "min_text_chars": 0,
        "max_text_chars": 30000,
        "canary_fraction": 0.0,
        "holdout_fraction": 0.0,
        "salt": "agentflow-phase-routing-canary-v1",
        "cohort_unit": "request_features",
        "safety_gates": {
            "block_thinking_history": True,
            "block_top_level_thinking": True,
            "strip_model_incompatible_params": True,
            "fallback_to_requested_on_rate_limit": True,
        },
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 10,
            "min_holdout_samples": 5,
            "max_error_rate": 0.05,
            "max_retry_rate": 0.20,
            "max_fallback_rate": 0.20,
            "max_latency_regression_ratio": 1.50,
            "limit": 500,
        },
    }


def _default_openai_canary_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "policy_id": "local-openai-routing-canary-v1",
        "model_pattern": "gpt-5.4",
        "target_model": "gpt-5.4-mini",
        "eligible_categories": ["chat", "short-completion", "tool-light"],
        "excluded_categories": ["tool-result", "tool-heavy", "code-gen", "long-context"],
        "allow_tools": True,
        "allow_stream": False,
        "min_text_chars": 0,
        "max_text_chars": 8000,
        "min_input_tokens_est": 0,
        "max_input_tokens_est": 2000,
        "canary_fraction": 0.05,
        "holdout_fraction": 0.05,
        "salt": "agentflow-openai-routing-canary-v1",
        "cohort_unit": "request_features_sequence",
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 20,
            "min_holdout_samples": 10,
            "max_error_rate": 0.03,
            "max_retry_rate": 0.10,
            "max_fallback_rate": 0.10,
            "max_latency_regression_ratio": 1.50,
            "limit": 1000,
        },
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def _float_0_1(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _int_min(value: Any, default: int, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _apply_phase_canary_yaml(policy: dict[str, Any], data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return policy
    policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
    for key in (
        "policy_id",
        "promotion_action_id",
        "target_candidate_id",
        "provider",
        "source_surface",
        "app_family",
        "policy_source",
        "model_pattern",
        "target_model",
        "requested_model",
        "routed_model",
        "min_workflow_phase_confidence",
        "salt",
        "cohort_unit",
    ):
        if data.get(key) not in (None, ""):
            policy[key] = str(data[key])
    for key in (
        "eligible_workflow_phases",
        "excluded_workflow_phases",
        "eligible_categories",
        "excluded_categories",
    ):
        if key in data:
            policy[key] = _string_list(data.get(key))
    for key in ("min_text_chars", "max_text_chars"):
        if key in data:
            policy[key] = _int_min(data.get(key), int(policy[key]), 0)
    if "stream" in data:
        policy["stream"] = _as_bool(data.get("stream"), False)
    for key in ("canary_fraction", "rollout_fraction", "holdout_fraction"):
        if key in data:
            target_key = "canary_fraction" if key == "rollout_fraction" else key
            policy[target_key] = _float_0_1(data.get(key), float(policy[target_key]))
    if isinstance(data.get("safety_gates"), dict):
        merged_gates = dict(policy.get("safety_gates") or {})
        for key, value in data["safety_gates"].items():
            if isinstance(value, bool):
                merged_gates[str(key)] = value
            elif value in (None, ""):
                continue
            elif isinstance(value, (str, int, float)):
                merged_gates[str(key)] = value
        policy["safety_gates"] = merged_gates

    safety = data.get("safety_stop")
    if isinstance(safety, dict):
        merged = dict(policy["safety_stop"])
        if "enabled" in safety:
            merged["enabled"] = _as_bool(safety.get("enabled"), bool(merged["enabled"]))
        for key in ("window_hours", "min_samples", "min_holdout_samples", "limit"):
            if key in safety:
                merged[key] = _int_min(safety.get(key), int(merged[key]), 0)
        for key in ("max_error_rate", "max_retry_rate", "max_fallback_rate", "max_latency_regression_ratio"):
            if key in safety:
                default = float(merged[key])
                try:
                    merged[key] = max(0.0, float(safety[key]))
                except (TypeError, ValueError):
                    merged[key] = default
        policy["safety_stop"] = merged
    return policy


def _apply_openai_canary_yaml(policy: dict[str, Any], data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return policy
    policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
    for key in ("policy_id", "promotion_action_id", "target_candidate_id", "policy_source", "model_pattern", "target_model", "salt", "cohort_unit"):
        if data.get(key) not in (None, ""):
            policy[key] = str(data[key])
    for key in ("eligible_categories", "excluded_categories"):
        if key in data:
            policy[key] = _string_list(data.get(key))
    for key in ("allow_tools", "allow_stream"):
        if key in data:
            policy[key] = _as_bool(data.get(key), bool(policy[key]))
    for key in ("min_text_chars", "max_text_chars", "min_input_tokens_est", "max_input_tokens_est"):
        if key in data:
            policy[key] = _int_min(data.get(key), int(policy[key]), 0)
    for key in ("canary_fraction", "rollout_fraction", "holdout_fraction"):
        if key in data:
            target_key = "canary_fraction" if key == "rollout_fraction" else key
            policy[target_key] = _float_0_1(data.get(key), float(policy[target_key]))

    safety = data.get("safety_stop")
    if isinstance(safety, dict):
        merged = dict(policy["safety_stop"])
        if "enabled" in safety:
            merged["enabled"] = _as_bool(safety.get("enabled"), bool(merged["enabled"]))
        for key in ("window_hours", "min_samples", "min_holdout_samples", "limit"):
            if key in safety:
                merged[key] = _int_min(safety.get(key), int(merged[key]), 0)
        for key in ("max_error_rate", "max_retry_rate", "max_fallback_rate", "max_latency_regression_ratio"):
            if key in safety:
                default = float(merged[key])
                try:
                    merged[key] = max(0.0, float(safety[key]))
                except (TypeError, ValueError):
                    merged[key] = default
        policy["safety_stop"] = merged
    return policy


def _load_routing_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return {"rules": data}
    if isinstance(data, dict):
        return data
    return {"rules": []}


def _rules_list(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    rules = data.get("rules")
    if not isinstance(rules, list):
        return []
    return [rule for rule in rules if isinstance(rule, dict)]


def _load_routing_rules() -> tuple[list[dict], dict[str, Any], dict[str, Any], str, str]:
    defaults_path = Path(__file__).parent / "routing_rules.yaml"
    defaults = _load_routing_yaml(defaults_path) or {"rules": []}
    env_path = os.getenv("AGENTFLOW_ROUTING_RULES")
    if env_path:
        p = safe_expanduser(env_path)
        data = _load_routing_yaml(p)
        if data is not None:
            canary = _apply_phase_canary_yaml(_default_phase_canary_policy(), data.get("phase_canary"))
            openai_canary = _apply_openai_canary_yaml(_default_openai_canary_policy(), data.get("openai_canary"))
            return _rules_list(data), canary, openai_canary, "local-manual", str(p)

    local_path = agentflow_config_path("routing_rules.yaml")
    local = _load_routing_yaml(local_path)
    if local is not None:
        canary = _apply_phase_canary_yaml(_default_phase_canary_policy(), local.get("phase_canary"))
        openai_canary = _apply_openai_canary_yaml(_default_openai_canary_policy(), local.get("openai_canary"))
        return _rules_list(local) + _rules_list(defaults), canary, openai_canary, "local-manual", str(local_path)

    canary = _apply_phase_canary_yaml(_default_phase_canary_policy(), defaults.get("phase_canary"))
    openai_canary = _apply_openai_canary_yaml(_default_openai_canary_policy(), defaults.get("openai_canary"))
    return _rules_list(defaults), canary, openai_canary, "local-default", str(defaults_path)


ROUTING_RULES, ROUTING_PHASE_CANARY, ROUTING_OPENAI_CANARY, ROUTING_RULES_SOURCE, ROUTING_RULES_PATH = _load_routing_rules()
ROUTING_RULES_LOADED_AT = utc_now()
ROUTING_RULES_LOADED_FILE = policy_file_snapshot(ROUTING_RULES_PATH)

_TIER_MAP = {"haiku": HAIKU_DEFAULT, "sonnet": SONNET_DEFAULT, "opus": OPUS_DEFAULT}
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_MODEL_FAMILY_RANK = {"unknown": 0, "other": 0, "haiku": 1, "gpt": 1, "sonnet": 2, "opus": 3}


def _text_bucket(text_chars: int) -> str:
    if text_chars < 2000:
        return "lt-2k"
    if text_chars < 8000:
        return "2k-8k"
    if text_chars < 30000:
        return "8k-30k"
    return "gte-30k"


def _phase_canary_db_path() -> str | None:
    database_url = os.getenv("AGENTFLOW_DATABASE_URL")
    if database_url and not database_url.startswith("sqlite:///"):
        return None
    if database_url and database_url.startswith("sqlite:///"):
        return database_url.removeprefix("sqlite:///")
    return str(default_db_path())


def _cohort_score(payload: dict[str, Any], salt: str) -> tuple[str, float]:
    digest = hashlib.sha256(f"{salt}:{stable_json(payload)}".encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return f"sha256:{digest}", score


def _hash_local_identifier(kind: str, value: str | None) -> str | None:
    if not value:
        return None
    digest = hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _input_tokens_est(text_chars: int) -> int:
    return max(1, text_chars // 4)


def _model_family(model: Any) -> str:
    value = str(model or "").lower()
    if "haiku" in value:
        return "haiku"
    if "sonnet" in value:
        return "sonnet"
    if "opus" in value:
        return "opus"
    if value.startswith("gpt-"):
        return "gpt"
    if not value:
        return "unknown"
    return "other"


def _count_value(rows: Any, value: str) -> int:
    if not isinstance(rows, list):
        return 0
    return sum(int(row.get("count") or 0) for row in rows if isinstance(row, dict) and row.get("value") == value)


def _dominant_count(memory: dict[str, Any], key: str, value: str) -> int:
    return _count_value(memory.get(key), value)


def _memory_policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    allow_blockers = sorted({"context_plateau_active", *_string_list(policy.get("allow_blockers"))})
    return {
        "enabled": True,
        "required": True,
        "window_size": _int_min(policy.get("window_size"), DEFAULT_SESSION_MEMORY_WINDOW, 1),
        "min_call_count": _int_min(policy.get("min_call_count"), 3, 1),
        "min_dominant_phase_count": _int_min(policy.get("min_dominant_phase_count"), 0, 0),
        "max_error_count": _int_min(policy.get("max_error_count"), 0, 0),
        "max_retry_count": _int_min(policy.get("max_retry_count"), 0, 0),
        "max_fallback_count": _int_min(policy.get("max_fallback_count"), 0, 0),
        "allow_thinking": _as_bool(policy.get("allow_thinking"), False),
        "allow_blockers": allow_blockers,
        "model_family_floor": str(policy.get("model_family_floor") or "").strip().lower() or None,
    }


DEFAULT_SESSION_MEMORY_WINDOW = 20


def _session_memory_db_path() -> str | None:
    return _phase_canary_db_path()


def _load_session_memory(session_id: str | None, *, limit: int, window_size: int) -> tuple[dict[str, Any] | None, str | None]:
    if not session_id:
        return None, "missing-session-id"
    db_path = _session_memory_db_path()
    if not db_path:
        return None, "non-sqlite-db"
    path = Path(db_path).expanduser()
    if not path.exists():
        return None, "db-missing"
    try:
        from agentflow_proxy.store import Store

        store = Store(str(path))
        try:
            memory = build_session_phase_memory_for_session(store, session_id, limit=limit, window_size=window_size)
        finally:
            store.conn.close()
    except (OSError, sqlite3.Error) as exc:
        return None, f"db-error:{exc.__class__.__name__}"
    if not memory:
        return None, "memory-missing"
    return memory, None


def _matches_session_memory_condition(
    policy: Any,
    *,
    session_id: str | None,
    requested_model: str,
    current_phase: str,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(policy, dict):
        return True, {}
    if not _as_bool(policy.get("enabled"), True):
        return True, {"enabled": False, "status": "ignored", "reason": "condition-disabled"}

    window_size = _int_min(policy.get("window_size"), DEFAULT_SESSION_MEMORY_WINDOW, 1)
    limit = _int_min(policy.get("limit"), window_size, window_size)
    meta = {
        "enabled": True,
        "status": "evaluating",
        "reason": "evaluating",
        "policy": _memory_policy_summary(policy),
    }
    memory, load_error = _load_session_memory(session_id, limit=limit, window_size=window_size)
    if load_error:
        meta.update({"status": "blocked", "reason": load_error})
        return False, meta
    assert memory is not None

    window = memory.get("window") if isinstance(memory.get("window"), dict) else {}
    blockers: list[str] = []
    call_count = int(window.get("call_count") or 0)
    dominant_phase = str(memory.get("dominant_phase") or "unknown")
    allowed_blockers = {"context_plateau_active", *_string_list(policy.get("allow_blockers"))}
    memory_blockers = [str(reason) for reason in memory.get("blocker_reasons") or []]
    effective_memory_blockers = [reason for reason in memory_blockers if reason not in allowed_blockers]
    if effective_memory_blockers:
        blockers.extend(effective_memory_blockers)

    min_call_count = _int_min(policy.get("min_call_count"), 3, 1)
    if call_count < min_call_count:
        blockers.append("stable_window_too_small")

    dominant_phase_expected = str(policy.get("dominant_phase") or "").strip()
    dominant_phase_in = _string_list(policy.get("dominant_phase_in"))
    if dominant_phase_expected and dominant_phase != dominant_phase_expected:
        blockers.append("dominant_phase_mismatch")
    if dominant_phase_in and dominant_phase not in set(dominant_phase_in):
        blockers.append("dominant_phase_mismatch")

    min_dominant_count = _int_min(policy.get("min_dominant_phase_count"), 0, 0)
    if min_dominant_count and _dominant_count(memory, "phase_counts", dominant_phase) < min_dominant_count:
        blockers.append("dominant_phase_window_too_short")

    stable_phase = policy.get("stable_phase")
    if stable_phase is not None and _as_bool(stable_phase):
        if call_count <= 0 or _dominant_count(memory, "phase_counts", dominant_phase) < call_count:
            blockers.append("phase_not_stable")

    blocked_phases = set(_string_list(policy.get("blocked_phases")))
    if current_phase in blocked_phases or dominant_phase in blocked_phases:
        blockers.append("blocked_phase_present")

    if not _as_bool(policy.get("allow_thinking"), False) and (
        dominant_phase == "thinking" or _count_value(memory.get("phase_counts"), "thinking") > 0
    ):
        blockers.append("thinking_present")

    max_error_count = _int_min(policy.get("max_error_count"), 0, 0)
    max_retry_count = _int_min(policy.get("max_retry_count"), 0, 0)
    max_fallback_count = _int_min(policy.get("max_fallback_count"), 0, 0)
    if int(memory.get("error_count") or 0) > max_error_count:
        blockers.append("recent_errors")
    if int(memory.get("retry_count") or 0) > max_retry_count:
        blockers.append("recent_retries")
    if int(memory.get("fallback_count") or 0) > max_fallback_count:
        blockers.append("recent_routing_fallback")

    floor = str(policy.get("model_family_floor") or "").strip().lower()
    if floor:
        floor_rank = _MODEL_FAMILY_RANK.get(floor, 0)
        requested_rank = _MODEL_FAMILY_RANK.get(_model_family(requested_model), 0)
        dominant_requested = str((memory.get("requested_model_family_counts") or [{}])[0].get("value") or "unknown")
        dominant_rank = _MODEL_FAMILY_RANK.get(dominant_requested, 0)
        if requested_rank < floor_rank or dominant_rank < floor_rank:
            blockers.append("model_family_floor_not_met")

    compact_memory = {
        "session_key": memory.get("session_key"),
        "session_key_kind": memory.get("session_key_kind"),
        "raw_session_id_included": False,
        "window": window,
        "dominant_phase": dominant_phase,
        "phase_counts": memory.get("phase_counts"),
        "category_counts": memory.get("category_counts"),
        "requested_model_family_counts": memory.get("requested_model_family_counts"),
        "routed_model_family_counts": memory.get("routed_model_family_counts"),
        "retry_count": memory.get("retry_count"),
        "fallback_count": memory.get("fallback_count"),
        "error_count": memory.get("error_count"),
        "blocker_reasons": memory_blockers,
        "privacy": {
            "metadata_only": True,
            "raw_session_ids_included": False,
            "request_json_read": False,
            "response_json_read": False,
            "error_text_included": False,
        },
    }
    meta["memory"] = compact_memory
    if blockers:
        meta.update({"status": "blocked", "reason": blockers[0], "reason_codes": sorted(set(blockers))})
        return False, meta
    meta.update({"status": "used", "reason": "matched", "reason_codes": []})
    return True, meta


def _safe_openai_target_model(target_model: Any) -> tuple[str | None, str | None]:
    if not isinstance(target_model, str) or not target_model.strip():
        return None, "missing-target-model"
    target = target_model.strip()
    lowered = target.lower()
    if lowered.startswith("claude-") or any(part in lowered for part in ("haiku", "sonnet", "opus")):
        return None, "provider-mismatch"
    if lowered.startswith(("gpt-", "o", "text-", "computer-use-", "codex")) or "gpt" in lowered:
        return target, None
    return None, "unsupported-openai-target-model"


def _phase_canary_base_meta(
    *,
    requested: str,
    target_model: str,
    text_chars: int,
    tools: bool,
    stream: bool,
    category: str,
    phase_meta: dict[str, str],
    reason: str,
    status: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(ROUTING_PHASE_CANARY.get("enabled")),
        "policy_id": str(ROUTING_PHASE_CANARY.get("policy_id") or "local-phase-sonnet-haiku-canary-v1"),
        "rule_id": str(ROUTING_PHASE_CANARY.get("policy_id") or "local-phase-sonnet-haiku-canary-v1"),
        "promotion_action_id": ROUTING_PHASE_CANARY.get("promotion_action_id"),
        "target_candidate_id": ROUTING_PHASE_CANARY.get("target_candidate_id"),
        "candidate_id": ROUTING_PHASE_CANARY.get("target_candidate_id") or ROUTING_PHASE_CANARY.get("promotion_action_id"),
        "status": status,
        "cohort": "none",
        "reason": reason,
        "requested_model": requested,
        "target_model": target_model,
        "actual_forwarded_model": requested,
        "provider": str(ROUTING_PHASE_CANARY.get("provider") or "anthropic"),
        "source_surface": str(ROUTING_PHASE_CANARY.get("source_surface") or "anthropic_messages"),
        "app_family": str(ROUTING_PHASE_CANARY.get("app_family") or "anthropic"),
        "workflow_phase": phase_meta.get("workflow_phase") or "unknown",
        "workflow_phase_confidence": phase_meta.get("workflow_phase_confidence") or "low",
        "category": category,
        "text_chars": text_chars,
        "text_bucket": _text_bucket(text_chars),
        "has_tools": tools,
        "stream": stream,
        "canary_fraction": float(ROUTING_PHASE_CANARY.get("canary_fraction") or 0.0),
        "holdout_fraction": float(ROUTING_PHASE_CANARY.get("holdout_fraction") or 0.0),
        "policy_source": ROUTING_PHASE_CANARY.get("policy_source") or ROUTING_RULES_SOURCE,
        "safety_gates": ROUTING_PHASE_CANARY.get("safety_gates") if isinstance(ROUTING_PHASE_CANARY.get("safety_gates"), dict) else {},
    }


def _phase_canary_safety_status(policy_id: str, safety: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "enabled": _as_bool(safety.get("enabled"), True),
        "status": "not-evaluated",
        "tripped": False,
        "reason_codes": [],
    }
    if not meta["enabled"]:
        meta["status"] = "disabled"
        return meta

    db_path = _phase_canary_db_path()
    if not db_path:
        meta["status"] = "skipped"
        meta["reason_codes"] = ["non-sqlite-db"]
        return meta
    path = Path(db_path).expanduser()
    if not path.exists():
        meta["status"] = "skipped"
        meta["reason_codes"] = ["db-missing"]
        return meta

    window_hours = _int_min(safety.get("window_hours"), 24, 1)
    limit = _int_min(safety.get("limit"), 500, 1)
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)).isoformat()
    rows: list[sqlite3.Row] = []
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                select created_at, status_code, retry_count, latency_ms, routing_json
                from calls
                where created_at >= ? and routing_json is not null
                order by created_at desc
                limit ?
                """,
                (cutoff, limit),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        meta["status"] = "skipped"
        meta["reason_codes"] = ["db-error"]
        meta["error_type"] = exc.__class__.__name__
        return meta

    applied: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in rows:
        try:
            routing = json.loads(row["routing_json"] or "{}")
        except (TypeError, ValueError):
            continue
        canary = routing.get("phase_canary")
        if not isinstance(canary, dict) or canary.get("policy_id") != policy_id:
            continue
        record = {
            "status_code": row["status_code"],
            "retry_count": row["retry_count"],
            "latency_ms": row["latency_ms"],
            "fallback": bool(routing.get("fallback_reason") or canary.get("fallback_reason")),
        }
        if canary.get("status") == "applied":
            applied.append(record)
        elif canary.get("status") == "holdout":
            holdout.append(record)

    min_samples = _int_min(safety.get("min_samples"), 10, 1)
    applied_count = len(applied)
    meta["status"] = "insufficient-samples" if applied_count < min_samples else "evaluated"
    meta["sample_count"] = applied_count
    meta["holdout_sample_count"] = len(holdout)
    meta["min_samples"] = min_samples
    meta["window_hours"] = window_hours
    if applied_count < min_samples:
        return meta

    error_count = sum(1 for item in applied if int(item.get("status_code") or 0) >= 400)
    retry_count = sum(1 for item in applied if int(item.get("retry_count") or 0) > 0)
    fallback_count = sum(1 for item in applied if item.get("fallback"))
    error_rate = error_count / applied_count
    retry_rate = retry_count / applied_count
    fallback_rate = fallback_count / applied_count
    meta.update({
        "error_rate": round(error_rate, 6),
        "retry_rate": round(retry_rate, 6),
        "fallback_rate": round(fallback_rate, 6),
    })

    reason_codes: list[str] = []
    if error_rate > float(safety.get("max_error_rate") or 0.0):
        reason_codes.append("error-rate")
    if retry_rate > float(safety.get("max_retry_rate") or 0.0):
        reason_codes.append("retry-rate")
    if fallback_rate > float(safety.get("max_fallback_rate") or 0.0):
        reason_codes.append("fallback-rate")

    applied_latencies = [int(item["latency_ms"]) for item in applied if item.get("latency_ms") is not None]
    holdout_latencies = [int(item["latency_ms"]) for item in holdout if item.get("latency_ms") is not None]
    min_holdout_samples = _int_min(safety.get("min_holdout_samples"), 5, 1)
    if len(applied_latencies) >= min_samples and len(holdout_latencies) >= min_holdout_samples:
        applied_avg = sum(applied_latencies) / len(applied_latencies)
        holdout_avg = sum(holdout_latencies) / len(holdout_latencies)
        ratio = applied_avg / holdout_avg if holdout_avg > 0 else 0.0
        meta["latency_regression_ratio"] = round(ratio, 6)
        if ratio > float(safety.get("max_latency_regression_ratio") or 0.0):
            reason_codes.append("latency-regression")

    if reason_codes:
        meta["tripped"] = True
        meta["status"] = "tripped"
        meta["reason_codes"] = reason_codes
    return meta


def phase_canary_decision(
    *,
    requested: str,
    text_chars: int,
    tools: bool,
    category: str,
    stream: bool,
    phase_meta: dict[str, str],
    thinking_gate: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    target_key = str(ROUTING_PHASE_CANARY.get("target_model") or "haiku")
    target_model = _TIER_MAP.get(target_key, target_key)
    meta = _phase_canary_base_meta(
        requested=requested,
        target_model=target_model,
        text_chars=text_chars,
        tools=tools,
        stream=stream,
        category=category,
        phase_meta=phase_meta,
        reason="disabled",
        status="disabled",
    )
    if not ROUTING_PHASE_CANARY.get("enabled"):
        return None, meta
    if meta["provider"] != "anthropic":
        meta.update({"status": "ineligible", "reason": "provider-not-supported"})
        return requested, meta
    if meta["source_surface"] != "anthropic_messages":
        meta.update({"status": "ineligible", "reason": "source-surface-not-supported"})
        return requested, meta
    if ROUTING_PHASE_CANARY.get("stream") is not None and bool(ROUTING_PHASE_CANARY.get("stream")) != bool(stream):
        meta.update({"status": "ineligible", "reason": "stream-scope-not-enabled"})
        return requested, meta

    requested_l = requested.lower()
    model_pattern = str(ROUTING_PHASE_CANARY.get("model_pattern") or "sonnet").lower()
    if model_pattern and model_pattern not in requested_l:
        meta.update({"status": "ineligible", "reason": "requested-model-not-enabled"})
        return requested, meta

    safety_gates = ROUTING_PHASE_CANARY.get("safety_gates") if isinstance(ROUTING_PHASE_CANARY.get("safety_gates"), dict) else {}
    gate = thinking_gate if isinstance(thinking_gate, dict) else {}
    top_level_thinking = bool(gate.get("top_level_thinking"))
    thinking_history = bool(gate.get("assistant_thinking_history"))
    reason_codes: list[str] = []
    if top_level_thinking and _as_bool(safety_gates.get("block_top_level_thinking"), True):
        reason_codes.append("top-level-thinking-blocked")
    if thinking_history and _as_bool(safety_gates.get("block_thinking_history"), True):
        reason_codes.append("thinking-history-blocked")
    if reason_codes:
        meta.update({
            "status": "safety_stopped",
            "cohort": "safety_stopped",
            "reason": "thinking-safety-gate",
            "actual_forwarded_model": requested,
            "safety_stop": {
                "enabled": True,
                "status": "tripped",
                "tripped": True,
                "reason_codes": sorted(reason_codes),
                "metadata_only": True,
            },
        })
        return requested, meta

    phase = str(phase_meta.get("workflow_phase") or "unknown")
    eligible_phases = set(_string_list(ROUTING_PHASE_CANARY.get("eligible_workflow_phases")))
    excluded_phases = set(_string_list(ROUTING_PHASE_CANARY.get("excluded_workflow_phases")))
    if phase in excluded_phases or (eligible_phases and phase not in eligible_phases):
        meta.update({"status": "ineligible", "reason": "workflow-phase-not-enabled"})
        return requested, meta

    confidence = str(phase_meta.get("workflow_phase_confidence") or "low")
    min_confidence = str(ROUTING_PHASE_CANARY.get("min_workflow_phase_confidence") or "medium")
    if _CONFIDENCE_ORDER.get(confidence, 0) < _CONFIDENCE_ORDER.get(min_confidence, 1):
        meta.update({"status": "ineligible", "reason": "workflow-phase-confidence-too-low"})
        return requested, meta

    eligible_categories = set(_string_list(ROUTING_PHASE_CANARY.get("eligible_categories")))
    excluded_categories = set(_string_list(ROUTING_PHASE_CANARY.get("excluded_categories")))
    if category in excluded_categories or (eligible_categories and category not in eligible_categories):
        meta.update({"status": "ineligible", "reason": "category-not-enabled"})
        return requested, meta

    min_chars = _int_min(ROUTING_PHASE_CANARY.get("min_text_chars"), 0, 0)
    max_chars = _int_min(ROUTING_PHASE_CANARY.get("max_text_chars"), 30000, 0)
    if text_chars < min_chars:
        meta.update({"status": "ineligible", "reason": "request-too-small"})
        return requested, meta
    if max_chars > 0 and text_chars > max_chars:
        meta.update({"status": "ineligible", "reason": "request-too-large"})
        return requested, meta

    safety = _phase_canary_safety_status(meta["policy_id"], ROUTING_PHASE_CANARY.get("safety_stop") or {})
    meta["safety_stop"] = safety
    if safety.get("tripped"):
        meta.update({"status": "safety_stopped", "reason": "safety-stop-tripped", "cohort": "bypassed_or_disabled"})
        return requested, meta

    request_cohort_payload = {
        "source_surface": meta["source_surface"],
        "app_family": meta["app_family"],
        "requested_model": requested,
        "target_model": target_model,
        "workflow_phase": phase,
        "category": category,
        "text_bucket": meta["text_bucket"],
        "has_tools": tools,
        "stream": stream,
        "policy_id": meta["policy_id"],
    }
    cohort_unit = str(ROUTING_PHASE_CANARY.get("cohort_unit") or "request_features")
    if cohort_unit in {"session", "session_id", "session_hash"}:
        cohort_payload = {
            "source_surface": meta["source_surface"],
            "app_family": meta["app_family"],
            "requested_model": requested,
            "target_model": target_model,
            "policy_id": meta["policy_id"],
            "cohort_unit": "session",
            "stream": stream,
            "session_id_hash": _hash_local_identifier("session_id", session_id),
        }
    else:
        cohort_payload = request_cohort_payload
        cohort_payload["cohort_unit"] = "request_features"
    cohort_hash, score = _cohort_score(cohort_payload, str(ROUTING_PHASE_CANARY.get("salt") or ""))
    holdout_fraction = float(ROUTING_PHASE_CANARY.get("holdout_fraction") or 0.0)
    canary_fraction = float(ROUTING_PHASE_CANARY.get("canary_fraction") or 0.0)
    meta.update({
        "cohort_hash": cohort_hash,
        "cohort_key_hash": cohort_hash,
        "cohort_score": round(score, 12),
        "cohort_features": cohort_payload,
    })
    if score < holdout_fraction:
        meta.update({
            "status": "holdout",
            "cohort": "canary_holdout",
            "reason": "selected-holdout",
            "actual_forwarded_model": requested,
        })
        return requested, meta
    if score < holdout_fraction + canary_fraction:
        meta.update({
            "status": "applied",
            "cohort": "canary_applied",
            "reason": "selected-canary",
            "actual_forwarded_model": target_model,
        })
        return target_model, meta
    meta.update({
        "status": "not_selected",
        "cohort": "skipped",
        "reason": "outside-canary-fraction",
        "actual_forwarded_model": requested,
    })
    return requested, meta


def _openai_canary_base_meta(
    *,
    requested: str,
    target_model: str,
    text_chars: int,
    input_tokens_est: int,
    tools: bool,
    stream: bool,
    category: str,
    reason: str,
    status: str,
) -> dict[str, Any]:
    current_input_cost = estimate_cost(requested, input_tokens_est, 0, provider="openai")
    target_input_cost = estimate_cost(target_model, input_tokens_est, 0, provider="openai")
    projected_savings = None
    if current_input_cost is not None and target_input_cost is not None:
        projected_savings = current_input_cost - target_input_cost
    policy_id = str(ROUTING_OPENAI_CANARY.get("policy_id") or "local-openai-routing-canary-v1")
    candidate_id = ROUTING_OPENAI_CANARY.get("target_candidate_id") or ROUTING_OPENAI_CANARY.get("promotion_action_id")
    return {
        "enabled": bool(ROUTING_OPENAI_CANARY.get("enabled")),
        "policy_id": policy_id,
        "rule_id": policy_id,
        "promotion_action_id": ROUTING_OPENAI_CANARY.get("promotion_action_id"),
        "target_candidate_id": ROUTING_OPENAI_CANARY.get("target_candidate_id"),
        "candidate_id": candidate_id,
        "status": status,
        "cohort": "none",
        "reason": reason,
        "original_model": requested,
        "requested_model": requested,
        "target_model": target_model,
        "actual_forwarded_model": requested,
        "source_surface": "openai_provider_request",
        "app_family": "generic_openai",
        "category": category,
        "text_chars": text_chars,
        "text_bucket": _text_bucket(text_chars),
        "input_tokens_est": input_tokens_est,
        "token_bucket": _text_bucket(input_tokens_est * 4),
        "current_input_cost_est_usd": current_input_cost,
        "target_input_cost_est_usd": target_input_cost,
        "projected_input_savings_usd": projected_savings,
        "has_tools": tools,
        "stream": stream,
        "canary_fraction": float(ROUTING_OPENAI_CANARY.get("canary_fraction") or 0.0),
        "holdout_fraction": float(ROUTING_OPENAI_CANARY.get("holdout_fraction") or 0.0),
        "policy_source": ROUTING_OPENAI_CANARY.get("policy_source") or ROUTING_RULES_SOURCE,
    }


def _openai_canary_safety_status(policy_id: str, safety: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "enabled": _as_bool(safety.get("enabled"), True),
        "status": "not-evaluated",
        "tripped": False,
        "reason_codes": [],
    }
    if not meta["enabled"]:
        meta["status"] = "disabled"
        return meta

    db_path = _phase_canary_db_path()
    if not db_path:
        meta["status"] = "skipped"
        meta["reason_codes"] = ["non-sqlite-db"]
        return meta
    path = Path(db_path).expanduser()
    if not path.exists():
        meta["status"] = "skipped"
        meta["reason_codes"] = ["db-missing"]
        return meta

    window_hours = _int_min(safety.get("window_hours"), 24, 1)
    limit = _int_min(safety.get("limit"), 1000, 1)
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)).isoformat()
    rows: list[sqlite3.Row] = []
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                select created_at, status_code, retry_count, latency_ms, routing_json
                from calls
                where provider = 'openai' and created_at >= ? and routing_json is not null
                order by created_at desc
                limit ?
                """,
                (cutoff, limit),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        meta["status"] = "skipped"
        meta["reason_codes"] = ["db-error"]
        meta["error_type"] = exc.__class__.__name__
        return meta

    applied: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in rows:
        try:
            routing = json.loads(row["routing_json"] or "{}")
        except (TypeError, ValueError):
            continue
        canary = routing.get("openai_canary")
        if not isinstance(canary, dict) or canary.get("policy_id") != policy_id:
            continue
        record = {
            "status_code": row["status_code"],
            "retry_count": row["retry_count"],
            "latency_ms": row["latency_ms"],
            "fallback": bool(routing.get("fallback_reason") or canary.get("fallback_reason")),
        }
        if canary.get("status") == "applied":
            applied.append(record)
        elif canary.get("status") == "holdout":
            holdout.append(record)

    min_samples = _int_min(safety.get("min_samples"), 20, 1)
    applied_count = len(applied)
    meta["status"] = "insufficient-samples" if applied_count < min_samples else "evaluated"
    meta["sample_count"] = applied_count
    meta["holdout_sample_count"] = len(holdout)
    meta["min_samples"] = min_samples
    meta["window_hours"] = window_hours
    if applied_count < min_samples:
        return meta

    error_count = sum(1 for item in applied if int(item.get("status_code") or 0) >= 400)
    retry_count = sum(1 for item in applied if int(item.get("retry_count") or 0) > 0)
    fallback_count = sum(1 for item in applied if item.get("fallback"))
    error_rate = error_count / applied_count
    retry_rate = retry_count / applied_count
    fallback_rate = fallback_count / applied_count
    meta.update({
        "error_rate": round(error_rate, 6),
        "retry_rate": round(retry_rate, 6),
        "fallback_rate": round(fallback_rate, 6),
    })

    reason_codes: list[str] = []
    if error_rate > float(safety.get("max_error_rate") or 0.0):
        reason_codes.append("error-rate")
    if retry_rate > float(safety.get("max_retry_rate") or 0.0):
        reason_codes.append("retry-rate")
    if fallback_rate > float(safety.get("max_fallback_rate") or 0.0):
        reason_codes.append("fallback-rate")

    applied_latencies = [int(item["latency_ms"]) for item in applied if item.get("latency_ms") is not None]
    holdout_latencies = [int(item["latency_ms"]) for item in holdout if item.get("latency_ms") is not None]
    min_holdout_samples = _int_min(safety.get("min_holdout_samples"), 10, 1)
    if len(applied_latencies) >= min_samples and len(holdout_latencies) >= min_holdout_samples:
        applied_avg = sum(applied_latencies) / len(applied_latencies)
        holdout_avg = sum(holdout_latencies) / len(holdout_latencies)
        ratio = applied_avg / holdout_avg if holdout_avg > 0 else 0.0
        meta["latency_regression_ratio"] = round(ratio, 6)
        if ratio > float(safety.get("max_latency_regression_ratio") or 0.0):
            reason_codes.append("latency-regression")

    if reason_codes:
        meta["tripped"] = True
        meta["status"] = "tripped"
        meta["reason_codes"] = reason_codes
    return meta


def _openai_canary_sequence_index(policy_id: str, cohort_payload: dict[str, Any], *, limit: int = 1000) -> int | None:
    db_path = _phase_canary_db_path()
    if not db_path:
        return None
    path = Path(db_path).expanduser()
    if not path.exists():
        return 0

    count = 0
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                select routing_json
                from calls
                where provider = 'openai' and routing_json is not null
                order by created_at desc
                limit ?
                """,
                (max(1, limit),),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None

    for row in rows:
        try:
            routing = json.loads(row["routing_json"] or "{}")
        except (TypeError, ValueError):
            continue
        canary = routing.get("openai_canary")
        if not isinstance(canary, dict):
            continue
        if canary.get("policy_id") != policy_id:
            continue
        if canary.get("cohort_features") != cohort_payload:
            continue
        count += 1
    return count


def _sequenced_canary_score(sequence_index: int, *, holdout_fraction: float, canary_fraction: float) -> float | None:
    if holdout_fraction >= 1.0:
        return min(0.999999, holdout_fraction / 2.0)
    if holdout_fraction <= 0.0 and canary_fraction >= 1.0:
        return min(0.999999, canary_fraction / 2.0)

    if holdout_fraction > 0.0:
        holdout_period = max(1, round(1.0 / holdout_fraction))
        if sequence_index % holdout_period == 0:
            return max(0.0, min(0.999999, holdout_fraction / 2.0))

    if canary_fraction > 0.0:
        canary_period = max(1, round(1.0 / canary_fraction))
        canary_offset = 1 if canary_period > 1 else 0
        if sequence_index % canary_period == canary_offset:
            return max(0.0, min(0.999999, holdout_fraction + (canary_fraction / 2.0)))

    total_fraction = holdout_fraction + canary_fraction
    if total_fraction >= 1.0:
        return min(0.999999, total_fraction - 0.000001)
    return None


def openai_canary_decision(
    *,
    requested: str,
    text_chars: int,
    input_tokens_est: int,
    tools: bool,
    stream: bool,
    category: str,
) -> tuple[str | None, dict[str, Any]]:
    target_model, target_error = _safe_openai_target_model(ROUTING_OPENAI_CANARY.get("target_model"))
    target = target_model or str(ROUTING_OPENAI_CANARY.get("target_model") or "")
    meta = _openai_canary_base_meta(
        requested=requested,
        target_model=target,
        text_chars=text_chars,
        input_tokens_est=input_tokens_est,
        tools=tools,
        stream=stream,
        category=category,
        reason="disabled",
        status="disabled",
    )
    if not ROUTING_OPENAI_CANARY.get("enabled"):
        return None, meta
    if target_error:
        meta.update({"status": "ineligible", "reason": target_error})
        return requested, meta

    requested_l = requested.lower()
    model_pattern = str(ROUTING_OPENAI_CANARY.get("model_pattern") or "").lower()
    if model_pattern and model_pattern not in requested_l:
        meta.update({"status": "ineligible", "reason": "requested-model-not-enabled"})
        return requested, meta
    if target_model == requested:
        meta.update({"status": "noop", "reason": "target-model-already-selected"})
        return requested, meta

    if tools and not _as_bool(ROUTING_OPENAI_CANARY.get("allow_tools"), False):
        meta.update({"status": "ineligible", "reason": "tool-request-not-enabled"})
        return requested, meta
    if stream and not _as_bool(ROUTING_OPENAI_CANARY.get("allow_stream"), False):
        meta.update({"status": "ineligible", "reason": "streaming-not-enabled"})
        return requested, meta

    eligible_categories = set(_string_list(ROUTING_OPENAI_CANARY.get("eligible_categories")))
    excluded_categories = set(_string_list(ROUTING_OPENAI_CANARY.get("excluded_categories")))
    if category in excluded_categories or (eligible_categories and category not in eligible_categories):
        meta.update({"status": "ineligible", "reason": "category-not-enabled"})
        return requested, meta

    min_chars = _int_min(ROUTING_OPENAI_CANARY.get("min_text_chars"), 0, 0)
    max_chars = _int_min(ROUTING_OPENAI_CANARY.get("max_text_chars"), 8000, 0)
    if text_chars < min_chars:
        meta.update({"status": "ineligible", "reason": "request-too-small"})
        return requested, meta
    if max_chars > 0 and text_chars > max_chars:
        meta.update({"status": "ineligible", "reason": "request-too-large"})
        return requested, meta

    min_tokens = _int_min(ROUTING_OPENAI_CANARY.get("min_input_tokens_est"), 0, 0)
    max_tokens = _int_min(ROUTING_OPENAI_CANARY.get("max_input_tokens_est"), 2000, 0)
    if input_tokens_est < min_tokens:
        meta.update({"status": "ineligible", "reason": "token-estimate-too-small"})
        return requested, meta
    if max_tokens > 0 and input_tokens_est > max_tokens:
        meta.update({"status": "ineligible", "reason": "token-estimate-too-large"})
        return requested, meta

    safety = _openai_canary_safety_status(meta["policy_id"], ROUTING_OPENAI_CANARY.get("safety_stop") or {})
    meta["safety_stop"] = safety
    if safety.get("tripped"):
        meta.update({"status": "safety_stopped", "reason": "safety-stop-tripped", "cohort": "bypassed_or_disabled"})
        return requested, meta

    cohort_payload = {
        "source_surface": meta["source_surface"],
        "app_family": meta["app_family"],
        "requested_model": requested,
        "target_model": target_model,
        "category": category,
        "text_bucket": meta["text_bucket"],
        "token_bucket": meta["token_bucket"],
        "has_tools": tools,
        "stream": stream,
        "policy_id": meta["policy_id"],
    }
    cohort_hash, score = _cohort_score(cohort_payload, str(ROUTING_OPENAI_CANARY.get("salt") or ""))
    holdout_fraction = float(ROUTING_OPENAI_CANARY.get("holdout_fraction") or 0.0)
    canary_fraction = float(ROUTING_OPENAI_CANARY.get("canary_fraction") or 0.0)
    cohort_unit = str(ROUTING_OPENAI_CANARY.get("cohort_unit") or "request_features")
    sequence_index: int | None = None
    sequenced_score = None
    if cohort_unit in {"request_features_sequence", "feature_sequence"}:
        sequence_index = _openai_canary_sequence_index(meta["policy_id"], cohort_payload)
        if sequence_index is not None:
            sequenced_score = _sequenced_canary_score(
                sequence_index,
                holdout_fraction=holdout_fraction,
                canary_fraction=canary_fraction,
            )
            if sequenced_score is not None:
                score = sequenced_score
    meta.update({
        "cohort_hash": cohort_hash,
        "cohort_key_hash": cohort_hash,
        "cohort_score": round(score, 12),
        "cohort_features": cohort_payload,
        "cohort_unit": cohort_unit,
    })
    if sequence_index is not None:
        meta["cohort_sequence_index"] = sequence_index
        meta["cohort_score_basis"] = "local-metadata-sequence" if sequenced_score is not None else "local-metadata-sequence-not-selected"
    if score < holdout_fraction:
        meta.update({
            "status": "holdout",
            "cohort": "canary_holdout",
            "reason": "selected-holdout",
            "actual_forwarded_model": requested,
        })
        return requested, meta
    if score < holdout_fraction + canary_fraction:
        meta.update({
            "status": "applied",
            "cohort": "canary_applied",
            "reason": "selected-canary",
            "actual_forwarded_model": target_model,
        })
        return target_model, meta
    meta.update({
        "status": "not_selected",
        "cohort": "skipped",
        "reason": "outside-canary-fraction",
        "actual_forwarded_model": requested,
    })
    return requested, meta


def _is_promoted_permanent_rule(rule: dict[str, Any]) -> bool:
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    if metadata.get("promoted_from_canary"):
        return True
    if metadata.get("source") == "claude-canary-promote":
        return True
    return str(rule.get("policy_source") or metadata.get("policy_source") or "") == "local-promoted"


def _rule_matches(
    rule: dict[str, Any],
    *,
    requested_l: str,
    text_chars: int,
    tools: bool,
    max_tokens: Any,
    category: str,
    phase_meta: dict[str, str],
    stream: bool,
    session_id: str | None,
    requested_model: str,
) -> tuple[bool, dict[str, Any] | None]:
    cond = rule.get("conditions") or {}
    if "model_pattern" in cond and str(cond["model_pattern"]).lower() not in requested_l:
        return False, None
    if "text_chars_lt" in cond and not (text_chars < int(cond["text_chars_lt"])):
        return False, None
    if "text_chars_gt" in cond and not (text_chars > int(cond["text_chars_gt"])):
        return False, None
    if "text_chars_lte" in cond and not (text_chars <= int(cond["text_chars_lte"])):
        return False, None
    if "text_chars_gte" in cond and not (text_chars >= int(cond["text_chars_gte"])):
        return False, None
    if "has_tools" in cond and bool(cond["has_tools"]) != tools:
        return False, None
    if "stream" in cond and bool(cond["stream"]) != stream:
        return False, None
    if "env_flag" in cond and not _env_flag_enabled(str(cond["env_flag"])):
        return False, None
    # A missing max_tokens value is unknown, not safely bounded.
    if "max_tokens_lte" in cond:
        if max_tokens is None or not (int(max_tokens) <= int(cond["max_tokens_lte"])):
            return False, None
    if "category" in cond and cond["category"] != category:
        return False, None
    if "category_not_in" in cond:
        excluded = cond["category_not_in"]
        if isinstance(excluded, str):
            excluded = [excluded]
        if category in set(excluded):
            return False, None
    if "workflow_phase" in cond and cond["workflow_phase"] != phase_meta.get("workflow_phase"):
        return False, None
    if "workflow_phase_confidence_gte" in cond:
        confidence = str(phase_meta.get("workflow_phase_confidence") or "low")
        required = str(cond.get("workflow_phase_confidence_gte") or "medium")
        if _CONFIDENCE_ORDER.get(confidence, 0) < _CONFIDENCE_ORDER.get(required, 1):
            return False, None
    session_memory_meta: dict[str, Any] | None = None
    if "session_memory" in cond:
        matched_memory, session_memory_meta = _matches_session_memory_condition(
            cond.get("session_memory"),
            session_id=session_id,
            requested_model=requested_model,
            current_phase=str(phase_meta.get("workflow_phase") or "unknown"),
        )
        if not matched_memory:
            return False, session_memory_meta
    return True, session_memory_meta


def _route_from_rule(
    rule: dict[str, Any],
    *,
    requested: str,
    text_chars: int,
    tools: bool,
    category: str,
    phase_meta: dict[str, str],
    session_memory_meta: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    action = rule.get("action") or {}
    route_key = str(action.get("route_to", ""))
    routed = _TIER_MAP.get(route_key, route_key) if route_key else requested
    reason = str(action.get("reason", "matched routing rule"))
    meta = {
        "enabled": True,
        "requested_model": requested,
        "routed_model": routed,
        "reason": reason,
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        **phase_meta,
        "policy_source": str(rule.get("policy_source") or ROUTING_RULES_SOURCE),
    }
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    if metadata.get("promoted_from_canary"):
        meta["canary_cohort"] = None
        meta["routing_rule"] = {
            "status": "applied",
            "source": metadata.get("source") or "claude-canary-promote",
            "promoted_from_canary": True,
            "rule_id": rule.get("id") or metadata.get("rule_id"),
            "target_candidate_id": metadata.get("target_candidate_id"),
            "promotion_source_policy_id": metadata.get("promotion_source_policy_id"),
        }
    if session_memory_meta:
        meta["session_phase_memory"] = session_memory_meta
    return routed, meta


def route_model(body: dict[str, Any], *, session_id: str | None = None) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or SONNET_DEFAULT)
    text_chars = len(extract_text(body))
    tools = has_tools(body)
    max_tokens = body.get("max_tokens")  # None when caller didn't set it
    category = categorize_request(body)
    phase_meta = classify_workflow_phase(body, category)
    stream = _as_bool(body.get("stream"), False)
    thinking_gate = _thinking_gate_meta(body, category)
    if not ROUTING_ENABLED:
        return requested, {
            "enabled": False,
            "requested_model": requested,
            "routed_model": requested,
            "reason": "routing disabled",
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            **phase_meta,
            "policy_source": "local-default",
            "thinking_gate": thinking_gate,
        }

    requested_l = requested.lower()

    last_session_memory_meta: dict[str, Any] | None = None
    matched_promoted_rules: set[int] = set()
    for index, rule in enumerate(ROUTING_RULES):
        if not _is_promoted_permanent_rule(rule):
            continue
        matched, session_memory_meta = _rule_matches(
            rule,
            requested_l=requested_l,
            text_chars=text_chars,
            tools=tools,
            max_tokens=max_tokens,
            category=category,
            phase_meta=phase_meta,
            stream=stream,
            session_id=session_id,
            requested_model=requested,
        )
        if session_memory_meta:
            last_session_memory_meta = session_memory_meta
        if matched:
            if thinking_gate.get("status") == "blocked":
                continue
            matched_promoted_rules.add(index)
            routed, meta = _route_from_rule(
                rule,
                requested=requested,
                text_chars=text_chars,
                tools=tools,
                category=category,
                phase_meta=phase_meta,
                session_memory_meta=session_memory_meta,
            )
            meta["thinking_gate"] = thinking_gate
            return routed, meta

    if ROUTING_PHASE_CANARY.get("enabled"):
        canary_routed, canary_meta = phase_canary_decision(
            requested=requested,
            text_chars=text_chars,
            tools=tools,
            category=category,
            stream=stream,
            phase_meta=phase_meta,
            thinking_gate=thinking_gate,
            session_id=session_id,
        )
        if canary_meta.get("reason") != "requested-model-not-enabled":
            routed = canary_routed or requested
            reason = "phase canary selected Sonnet-to-Haiku route"
            if canary_meta.get("status") == "holdout":
                reason = "phase canary holdout; keep requested model"
            elif canary_meta.get("status") == "not_selected":
                reason = "phase canary not selected; keep requested model"
            elif canary_meta.get("status") == "safety_stopped":
                reason = "phase canary safety stop; keep requested model"
            elif canary_meta.get("status") == "ineligible":
                reason = f"phase canary ineligible: {canary_meta.get('reason')}"
            meta = {
                "enabled": True,
                "requested_model": requested,
                "routed_model": routed,
                "reason": reason,
                "text_chars": text_chars,
                "has_tools": tools,
                "category": category,
                **phase_meta,
                "policy_source": ROUTING_RULES_SOURCE,
                "phase_canary": canary_meta,
            }
            meta["thinking_gate"] = thinking_gate
            return routed, meta

    if uses_thinking(body):
        return requested, {
            "enabled": True,
            "requested_model": requested,
            "routed_model": requested,
            "reason": "keep requested model for thinking request",
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            **phase_meta,
            "policy_source": ROUTING_RULES_SOURCE,
            "thinking_gate": thinking_gate,
        }

    for index, rule in enumerate(ROUTING_RULES):
        if index in matched_promoted_rules:
            continue
        matched, session_memory_meta = _rule_matches(
            rule,
            requested_l=requested_l,
            text_chars=text_chars,
            tools=tools,
            max_tokens=max_tokens,
            category=category,
            phase_meta=phase_meta,
            stream=stream,
            session_id=session_id,
            requested_model=requested,
        )
        if session_memory_meta:
            last_session_memory_meta = session_memory_meta
        if matched:
            routed, meta = _route_from_rule(
                rule,
                requested=requested,
                text_chars=text_chars,
                tools=tools,
                category=category,
                phase_meta=phase_meta,
                session_memory_meta=session_memory_meta,
            )
            meta["thinking_gate"] = thinking_gate
            return routed, meta

    meta = {
        "enabled": True,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "keep requested model",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        **phase_meta,
        "policy_source": ROUTING_RULES_SOURCE,
        "thinking_gate": thinking_gate,
    }
    if last_session_memory_meta:
        meta["session_phase_memory"] = last_session_memory_meta
        if last_session_memory_meta.get("status") == "blocked":
            meta["reason"] = f"session phase memory blocked: {last_session_memory_meta.get('reason')}"
    return requested, meta


def route_openai_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or OPENAI_LARGE_DEFAULT)
    text_chars = len(extract_text(body))
    tools = bool(body.get("tools"))
    stream = bool(body.get("stream"))
    category = categorize_request(body)
    input_tokens_est = _input_tokens_est(text_chars)
    openai_canary_enabled = bool(ROUTING_OPENAI_CANARY.get("enabled"))
    meta = {
        "enabled": bool(OPENAI_ROUTING_ENABLED or openai_canary_enabled),
        "requested_model": requested,
        "routed_model": requested,
        "reason": "openai routing disabled",
        "text_chars": text_chars,
        "input_tokens_est": input_tokens_est,
        "has_tools": tools,
        "stream": stream,
        "category": category,
        "policy_source": "local-default",
        "provider": "openai",
    }
    if not ROUTING_ENABLED:
        meta["enabled"] = False
        meta["reason"] = "routing disabled"
        return requested, meta

    if openai_canary_enabled:
        canary_routed, canary_meta = openai_canary_decision(
            requested=requested,
            text_chars=text_chars,
            input_tokens_est=input_tokens_est,
            tools=tools,
            stream=stream,
            category=category,
        )
        routed = canary_routed or requested
        reason = "OpenAI canary selected local route"
        if canary_meta.get("status") == "holdout":
            reason = "OpenAI canary holdout; keep requested model"
        elif canary_meta.get("status") == "not_selected":
            reason = "OpenAI canary not selected; keep requested model"
        elif canary_meta.get("status") == "safety_stopped":
            reason = "OpenAI canary safety stop; keep requested model"
        elif canary_meta.get("status") in {"ineligible", "noop"}:
            reason = f"OpenAI canary ineligible: {canary_meta.get('reason')}"
        meta.update({
            "enabled": True,
            "routed_model": routed,
            "reason": reason,
            "policy_source": canary_meta.get("policy_source") or ROUTING_RULES_SOURCE,
            "openai_canary": canary_meta,
        })
        if not (OPENAI_ROUTING_ENABLED and canary_meta.get("status") in {"ineligible", "noop"}):
            return routed, meta

    if not OPENAI_ROUTING_ENABLED:
        return requested, meta

    requested_l = requested.lower()
    if tools:
        meta["reason"] = "keep requested OpenAI model for tool request"
        meta["enabled"] = True
        meta["policy_source"] = "local-default"
        return requested, meta

    routed = requested
    reason = "keep requested OpenAI model"
    if requested_l == OPENAI_LARGE_DEFAULT.lower() and text_chars < int(os.getenv("AGENTFLOW_OPENAI_SMALL_TEXT_CHARS_LT", "6000")):
        routed = OPENAI_SMALL_DEFAULT
        reason = "small non-tool OpenAI request"
    elif requested_l == OPENAI_SMALL_DEFAULT.lower() and text_chars < int(os.getenv("AGENTFLOW_OPENAI_TINY_TEXT_CHARS_LT", "1500")):
        routed = OPENAI_TINY_DEFAULT
        reason = "tiny non-tool OpenAI request"

    meta.update({
        "enabled": True,
        "routed_model": routed,
        "reason": reason,
    })
    return routed, meta
