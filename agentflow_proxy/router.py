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
from agentflow_proxy.store import stable_json

HAIKU_DEFAULT = os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
SONNET_DEFAULT = os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4-6")
OPUS_DEFAULT = os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4-5")
OPENAI_LARGE_DEFAULT = os.getenv("AGENTFLOW_OPENAI_LARGE_MODEL", "gpt-5-codex")
OPENAI_SMALL_DEFAULT = os.getenv("AGENTFLOW_OPENAI_SMALL_MODEL", "gpt-5-mini")
OPENAI_TINY_DEFAULT = os.getenv("AGENTFLOW_OPENAI_TINY_MODEL", "gpt-5-nano")

ROUTING_ENABLED = os.getenv("AGENTFLOW_ROUTING", "1") != "0"
OPENAI_ROUTING_ENABLED = os.getenv("AGENTFLOW_OPENAI_ROUTING", "0") == "1"
ROUTING_RULES_PATH = os.getenv("AGENTFLOW_ROUTING_RULES", str(Path.home() / ".agentflow" / "routing_rules.yaml"))
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

    if uses_thinking(body):
        return {
            "workflow_phase": "thinking",
            "workflow_phase_reason": "thinking-flag-or-history",
            "workflow_phase_confidence": "high",
        }

    if "tool_result" in last_user_block_types:
        return {
            "workflow_phase": "tool-execution",
            "workflow_phase_reason": "last-user-tool-result",
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
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            if any(isinstance(b, dict) and b.get("type") == "thinking" for b in msg["content"]):
                return True
    return False


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
        "model_pattern": "sonnet",
        "target_model": "haiku",
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
        "policy_source",
        "model_pattern",
        "target_model",
        "min_workflow_phase_confidence",
        "salt",
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


def _load_routing_rules() -> tuple[list[dict], dict[str, Any], str, str]:
    p = Path(ROUTING_RULES_PATH)
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data, _default_phase_canary_policy(), "local-manual", str(p)
        if isinstance(data, dict) and "rules" in data:
            canary = _apply_phase_canary_yaml(_default_phase_canary_policy(), data.get("phase_canary"))
            return list(data["rules"]), canary, "local-manual", str(p)
    defaults = Path(__file__).parent / "routing_rules.yaml"
    with open(defaults) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data, _default_phase_canary_policy(), "local-default", str(defaults)
    canary = _apply_phase_canary_yaml(_default_phase_canary_policy(), data.get("phase_canary"))
    return list(data.get("rules", [])), canary, "local-default", str(defaults)


ROUTING_RULES, ROUTING_PHASE_CANARY, ROUTING_RULES_SOURCE, ROUTING_RULES_PATH = _load_routing_rules()
ROUTING_RULES_LOADED_AT = utc_now()
ROUTING_RULES_LOADED_FILE = policy_file_snapshot(ROUTING_RULES_PATH)

_TIER_MAP = {"haiku": HAIKU_DEFAULT, "sonnet": SONNET_DEFAULT, "opus": OPUS_DEFAULT}
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


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
    return os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))


def _cohort_score(payload: dict[str, Any], salt: str) -> tuple[str, float]:
    digest = hashlib.sha256(f"{salt}:{stable_json(payload)}".encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return f"sha256:{digest}", score


def _phase_canary_base_meta(
    *,
    requested: str,
    target_model: str,
    text_chars: int,
    tools: bool,
    category: str,
    phase_meta: dict[str, str],
    reason: str,
    status: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(ROUTING_PHASE_CANARY.get("enabled")),
        "policy_id": str(ROUTING_PHASE_CANARY.get("policy_id") or "local-phase-sonnet-haiku-canary-v1"),
        "promotion_action_id": ROUTING_PHASE_CANARY.get("promotion_action_id"),
        "target_candidate_id": ROUTING_PHASE_CANARY.get("target_candidate_id"),
        "status": status,
        "cohort": "none",
        "reason": reason,
        "requested_model": requested,
        "target_model": target_model,
        "source_surface": "anthropic_messages",
        "app_family": "anthropic",
        "workflow_phase": phase_meta.get("workflow_phase") or "unknown",
        "workflow_phase_confidence": phase_meta.get("workflow_phase_confidence") or "low",
        "category": category,
        "text_chars": text_chars,
        "text_bucket": _text_bucket(text_chars),
        "has_tools": tools,
        "canary_fraction": float(ROUTING_PHASE_CANARY.get("canary_fraction") or 0.0),
        "holdout_fraction": float(ROUTING_PHASE_CANARY.get("holdout_fraction") or 0.0),
        "policy_source": ROUTING_PHASE_CANARY.get("policy_source") or ROUTING_RULES_SOURCE,
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
    phase_meta: dict[str, str],
) -> tuple[str | None, dict[str, Any]]:
    target_key = str(ROUTING_PHASE_CANARY.get("target_model") or "haiku")
    target_model = _TIER_MAP.get(target_key, target_key)
    meta = _phase_canary_base_meta(
        requested=requested,
        target_model=target_model,
        text_chars=text_chars,
        tools=tools,
        category=category,
        phase_meta=phase_meta,
        reason="disabled",
        status="disabled",
    )
    if not ROUTING_PHASE_CANARY.get("enabled"):
        return None, meta

    requested_l = requested.lower()
    model_pattern = str(ROUTING_PHASE_CANARY.get("model_pattern") or "sonnet").lower()
    if model_pattern and model_pattern not in requested_l:
        meta.update({"status": "ineligible", "reason": "requested-model-not-enabled"})
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

    cohort_payload = {
        "source_surface": meta["source_surface"],
        "app_family": meta["app_family"],
        "requested_model": requested,
        "target_model": target_model,
        "workflow_phase": phase,
        "category": category,
        "text_bucket": meta["text_bucket"],
        "has_tools": tools,
        "policy_id": meta["policy_id"],
    }
    cohort_hash, score = _cohort_score(cohort_payload, str(ROUTING_PHASE_CANARY.get("salt") or ""))
    holdout_fraction = float(ROUTING_PHASE_CANARY.get("holdout_fraction") or 0.0)
    canary_fraction = float(ROUTING_PHASE_CANARY.get("canary_fraction") or 0.0)
    meta.update({
        "cohort_hash": cohort_hash,
        "cohort_score": round(score, 12),
        "cohort_features": cohort_payload,
    })
    if score < holdout_fraction:
        meta.update({"status": "holdout", "cohort": "canary_holdout", "reason": "selected-holdout"})
        return requested, meta
    if score < holdout_fraction + canary_fraction:
        meta.update({"status": "applied", "cohort": "canary_applied", "reason": "selected-canary"})
        return target_model, meta
    meta.update({"status": "not_selected", "cohort": "skipped", "reason": "outside-canary-fraction"})
    return requested, meta


def route_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or SONNET_DEFAULT)
    text_chars = len(extract_text(body))
    tools = has_tools(body)
    max_tokens = body.get("max_tokens")  # None when caller didn't set it
    category = categorize_request(body)
    phase_meta = classify_workflow_phase(body, category)
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
        }

    requested_l = requested.lower()

    if ROUTING_PHASE_CANARY.get("enabled"):
        canary_routed, canary_meta = phase_canary_decision(
            requested=requested,
            text_chars=text_chars,
            tools=tools,
            category=category,
            phase_meta=phase_meta,
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
            return routed, {
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
        }

    for rule in ROUTING_RULES:
        cond = rule.get("conditions") or {}
        if "model_pattern" in cond and cond["model_pattern"].lower() not in requested_l:
            continue
        if "text_chars_lt" in cond and not (text_chars < int(cond["text_chars_lt"])):
            continue
        if "text_chars_gt" in cond and not (text_chars > int(cond["text_chars_gt"])):
            continue
        if "text_chars_lte" in cond and not (text_chars <= int(cond["text_chars_lte"])):
            continue
        if "text_chars_gte" in cond and not (text_chars >= int(cond["text_chars_gte"])):
            continue
        if "has_tools" in cond and bool(cond["has_tools"]) != tools:
            continue
        if "env_flag" in cond and not _env_flag_enabled(str(cond["env_flag"])):
            continue
        # A missing max_tokens value is unknown, not safely bounded.
        if "max_tokens_lte" in cond:
            if max_tokens is None or not (int(max_tokens) <= int(cond["max_tokens_lte"])):
                continue
        if "category" in cond and cond["category"] != category:
            continue
        if "category_not_in" in cond:
            excluded = cond["category_not_in"]
            if isinstance(excluded, str):
                excluded = [excluded]
            if category in set(excluded):
                continue
        if "workflow_phase" in cond and cond["workflow_phase"] != phase_meta.get("workflow_phase"):
            continue

        action = rule.get("action") or {}
        route_key = str(action.get("route_to", ""))
        routed = _TIER_MAP.get(route_key, route_key) if route_key else requested
        reason = str(action.get("reason", "matched routing rule"))
        return routed, {
            "enabled": True,
            "requested_model": requested,
            "routed_model": routed,
            "reason": reason,
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            **phase_meta,
            "policy_source": ROUTING_RULES_SOURCE,
        }

    return requested, {
        "enabled": True,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "keep requested model",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        **phase_meta,
        "policy_source": ROUTING_RULES_SOURCE,
    }


def route_openai_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or OPENAI_LARGE_DEFAULT)
    text_chars = len(extract_text(body))
    tools = bool(body.get("tools"))
    category = categorize_request(body)
    meta = {
        "enabled": OPENAI_ROUTING_ENABLED,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "openai routing disabled",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        "policy_source": "local-default",
        "provider": "openai",
    }
    if not ROUTING_ENABLED or not OPENAI_ROUTING_ENABLED:
        return requested, meta

    requested_l = requested.lower()
    if tools:
        meta["reason"] = "keep requested OpenAI model for tool request"
        meta["enabled"] = True
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
