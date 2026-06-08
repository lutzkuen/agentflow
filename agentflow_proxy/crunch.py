from __future__ import annotations

import copy
import hashlib
import os
import re
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.store import stable_json

TOKEN_CHARS = 4  # rough estimator only


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


def _default_crunch_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "threshold_chars": 24000,
        "prompt_cache": {
            "enabled": True,
            "min_chars": 4096,
        },
        "old_context_summarization": {
            "enabled": False,
            "model": "claude-haiku-4-5-20251001",
            "placement": "system",
            "min_request_chars": 32000,
            "min_summarized_chars": 12000,
            "max_turns": 6,
            "keep_recent_turns": 4,
            "max_summary_chars": 4000,
            "max_source_chars": 80000,
        },
        "thinking_deduplication": {
            "enabled": True,
            "min_chars": 2000,
            "similarity_threshold": 0.95,
            "skip_latest_assistant": True,
        },
        "pattern_rules": [],
        "codex_repeated_scaffolding": {
            "enabled": True,
            "min_request_chars": 12000,
            "min_section_chars": 700,
            "keep_recent_input_blocks": 1,
            "older_block_min_chars": 24000,
            "older_block_head_chars": 6000,
            "older_block_tail_chars": 4000,
            "max_replacements": 32,
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


def _load_crunch_policy() -> tuple[dict[str, Any], str, str]:
    for path in _manual_rule_candidates("crunch_rules.yaml", "AGENTFLOW_CRUNCH_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _default_crunch_policy()
            policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
            if data.get("threshold_chars") is not None:
                policy["threshold_chars"] = int(data["threshold_chars"])
            prompt_cache = data.get("prompt_cache") or {}
            if isinstance(prompt_cache, dict):
                policy["prompt_cache"]["enabled"] = _as_bool(
                    prompt_cache.get("enabled"),
                    policy["prompt_cache"]["enabled"],
                )
                if prompt_cache.get("min_chars") is not None:
                    policy["prompt_cache"]["min_chars"] = int(prompt_cache["min_chars"])
            summary = data.get("old_context_summarization") or {}
            if isinstance(summary, dict):
                _apply_summary_policy_yaml(policy, summary)
            thinking_dedup = data.get("thinking_deduplication") or {}
            if isinstance(thinking_dedup, dict):
                _apply_thinking_dedup_policy_yaml(policy, thinking_dedup)
            pattern_rules = data.get("pattern_rules")
            if pattern_rules is not None:
                policy["pattern_rules"] = _parse_pattern_rules_yaml(pattern_rules, default_policy_source="local-manual")
            codex_scaffolding = data.get("codex_repeated_scaffolding") or {}
            if isinstance(codex_scaffolding, dict):
                _apply_codex_scaffolding_policy_yaml(policy, codex_scaffolding)
            return policy, "local-manual", str(path)

    defaults_path = Path(__file__).parent / "crunch_rules.yaml"
    policy = _default_crunch_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
            if data.get("threshold_chars") is not None:
                policy["threshold_chars"] = int(data["threshold_chars"])
            prompt_cache = data.get("prompt_cache") or {}
            if isinstance(prompt_cache, dict):
                policy["prompt_cache"]["enabled"] = _as_bool(
                    prompt_cache.get("enabled"),
                    policy["prompt_cache"]["enabled"],
                )
                if prompt_cache.get("min_chars") is not None:
                    policy["prompt_cache"]["min_chars"] = int(prompt_cache["min_chars"])
            summary = data.get("old_context_summarization") or {}
            if isinstance(summary, dict):
                _apply_summary_policy_yaml(policy, summary)
            thinking_dedup = data.get("thinking_deduplication") or {}
            if isinstance(thinking_dedup, dict):
                _apply_thinking_dedup_policy_yaml(policy, thinking_dedup)
            pattern_rules = data.get("pattern_rules")
            if pattern_rules is not None:
                policy["pattern_rules"] = _parse_pattern_rules_yaml(pattern_rules, default_policy_source="local-default")
            codex_scaffolding = data.get("codex_repeated_scaffolding") or {}
            if isinstance(codex_scaffolding, dict):
                _apply_codex_scaffolding_policy_yaml(policy, codex_scaffolding)
    policy["enabled"] = os.getenv("AGENTFLOW_CRUNCH", "1") != "0"
    policy["threshold_chars"] = int(os.getenv("AGENTFLOW_CRUNCH_THRESHOLD_CHARS", str(policy["threshold_chars"])))
    policy["prompt_cache"]["enabled"] = os.getenv("AGENTFLOW_PROMPT_CACHE", "1") != "0"
    policy["prompt_cache"]["min_chars"] = int(
        os.getenv("AGENTFLOW_PROMPT_CACHE_MIN_CHARS", str(policy["prompt_cache"]["min_chars"]))
    )
    summary = policy["old_context_summarization"]
    summary["enabled"] = os.getenv("AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT", "0") == "1"
    summary["model"] = os.getenv("AGENTFLOW_HAIKU_SUMMARY_MODEL", str(summary["model"]))
    summary["min_request_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MIN_REQUEST_CHARS", str(summary["min_request_chars"])))
    summary["min_summarized_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MIN_SUMMARIZED_CHARS", str(summary["min_summarized_chars"])))
    summary["max_turns"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MAX_TURNS", str(summary["max_turns"])))
    summary["keep_recent_turns"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_KEEP_RECENT_TURNS", str(summary["keep_recent_turns"])))
    summary["max_summary_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MAX_SUMMARY_CHARS", str(summary["max_summary_chars"])))
    summary["max_source_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MAX_SOURCE_CHARS", str(summary["max_source_chars"])))
    return policy, "local-default", str(defaults_path)


def _apply_summary_policy_yaml(policy: dict[str, Any], summary: dict[str, Any]) -> None:
    target = policy["old_context_summarization"]
    target["enabled"] = _as_bool(summary.get("enabled"), target["enabled"])
    if summary.get("model") is not None:
        target["model"] = str(summary["model"])
    if summary.get("placement") is not None:
        placement = str(summary["placement"]).strip().lower()
        if placement != "system":
            raise ValueError("old_context_summarization.placement currently supports only 'system'")
        target["placement"] = placement
    for key in (
        "min_request_chars",
        "min_summarized_chars",
        "max_turns",
        "keep_recent_turns",
        "max_summary_chars",
        "max_source_chars",
    ):
        if summary.get(key) is not None:
            target[key] = int(summary[key])


def _apply_thinking_dedup_policy_yaml(policy: dict[str, Any], thinking_dedup: dict[str, Any]) -> None:
    target = policy["thinking_deduplication"]
    target["enabled"] = _as_bool(thinking_dedup.get("enabled"), target["enabled"])
    if thinking_dedup.get("min_chars") is not None:
        target["min_chars"] = int(thinking_dedup["min_chars"])
    if thinking_dedup.get("similarity_threshold") is not None:
        target["similarity_threshold"] = float(thinking_dedup["similarity_threshold"])
    target["skip_latest_assistant"] = _as_bool(
        thinking_dedup.get("skip_latest_assistant"),
        target["skip_latest_assistant"],
    )


def _parse_pattern_hashes(value: Any) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    hashes: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            continue
        item = raw.strip()
        if not item:
            continue
        if not item.startswith("sha256:"):
            item = f"sha256:{item}"
        if item not in hashes:
            hashes.append(item)
    return hashes


def _parse_pattern_rules_yaml(value: Any, *, default_policy_source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        conditions = item.get("conditions") or {}
        if not isinstance(conditions, dict):
            conditions = {}
        action = item.get("action") or {}
        if not isinstance(action, dict):
            action = {}
        pattern_hashes = _parse_pattern_hashes(
            conditions.get("pattern_hashes")
            or conditions.get("pattern_hash")
            or item.get("pattern_hashes")
            or item.get("pattern_hash")
        )
        if not pattern_hashes:
            continue
        normalized_conditions: dict[str, Any] = {
            "pattern_hashes": pattern_hashes,
            "min_repeated_count": int(conditions.get("min_repeated_count", item.get("min_repeated_count", 2))),
            "keep_recent_matches": int(conditions.get("keep_recent_matches", item.get("keep_recent_matches", 1))),
        }
        for key in ("model_pattern", "category", "workflow_phase"):
            if conditions.get(key) is not None:
                normalized_conditions[key] = str(conditions[key])
        if conditions.get("category_not_in") is not None:
            raw_categories = conditions["category_not_in"]
            if isinstance(raw_categories, list):
                normalized_conditions["category_not_in"] = [str(category) for category in raw_categories]
            else:
                normalized_conditions["category_not_in"] = [str(raw_categories)]
        for key in ("min_text_chars", "max_text_chars", "max_applications"):
            if conditions.get(key) is not None:
                normalized_conditions[key] = int(conditions[key])
        action_type = str(action.get("type") or action.get("kind") or "shorten").strip().lower()
        if action_type not in {"shorten", "omit"}:
            action_type = "shorten"
        normalized_action: dict[str, Any] = {
            "type": action_type,
            "head_chars": int(action.get("head_chars", 1200)),
            "tail_chars": int(action.get("tail_chars", 800)),
            "max_replacement_chars": int(action.get("max_replacement_chars", item.get("max_replacement_chars", 2400))),
        }
        if action.get("marker") is not None:
            normalized_action["marker"] = str(action["marker"])
        rules.append({
            "id": str(item.get("id") or item.get("rule_id") or item.get("candidate_id") or f"pattern-rule-{index + 1}"),
            "candidate_id": item.get("candidate_id"),
            "enabled": _as_bool(item.get("enabled"), True),
            "policy_source": str(item.get("policy_source") or default_policy_source),
            "description": str(item.get("description") or ""),
            "conditions": normalized_conditions,
            "action": normalized_action,
        })
    return rules


def _apply_codex_scaffolding_policy_yaml(policy: dict[str, Any], codex_scaffolding: dict[str, Any]) -> None:
    target = policy["codex_repeated_scaffolding"]
    target["enabled"] = _as_bool(codex_scaffolding.get("enabled"), target["enabled"])
    for key in (
        "min_request_chars",
        "min_section_chars",
        "keep_recent_input_blocks",
        "older_block_min_chars",
        "older_block_head_chars",
        "older_block_tail_chars",
        "max_replacements",
    ):
        if codex_scaffolding.get(key) is not None:
            target[key] = int(codex_scaffolding[key])


CRUNCH_POLICY, CRUNCH_POLICY_SOURCE, CRUNCH_RULES_PATH = _load_crunch_policy()
CRUNCH_RULES_LOADED_AT = utc_now()
CRUNCH_RULES_LOADED_FILE = policy_file_snapshot(CRUNCH_RULES_PATH)
CRUNCH_ENABLED = bool(CRUNCH_POLICY["enabled"])
CRUNCH_THRESHOLD_CHARS = int(CRUNCH_POLICY["threshold_chars"])
PROMPT_CACHE_ENABLED = bool(CRUNCH_POLICY["prompt_cache"]["enabled"])
PROMPT_CACHE_MIN_CHARS = int(CRUNCH_POLICY["prompt_cache"]["min_chars"])
OLD_CONTEXT_SUMMARY_POLICY = CRUNCH_POLICY["old_context_summarization"]
OLD_CONTEXT_SUMMARY_ENABLED = bool(OLD_CONTEXT_SUMMARY_POLICY["enabled"])
OLD_CONTEXT_SUMMARY_MODEL = str(OLD_CONTEXT_SUMMARY_POLICY["model"])
OLD_CONTEXT_SUMMARY_PLACEMENT = str(OLD_CONTEXT_SUMMARY_POLICY["placement"])
OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["min_request_chars"])
OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["min_summarized_chars"])
OLD_CONTEXT_SUMMARY_MAX_TURNS = int(OLD_CONTEXT_SUMMARY_POLICY["max_turns"])
OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS = int(OLD_CONTEXT_SUMMARY_POLICY["keep_recent_turns"])
OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["max_summary_chars"])
OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["max_source_chars"])
THINKING_DEDUP_POLICY = CRUNCH_POLICY["thinking_deduplication"]
THINKING_DEDUP_ENABLED = bool(THINKING_DEDUP_POLICY["enabled"])
THINKING_DEDUP_MIN_CHARS = int(THINKING_DEDUP_POLICY["min_chars"])
THINKING_DEDUP_SIMILARITY_THRESHOLD = float(THINKING_DEDUP_POLICY["similarity_threshold"])
THINKING_DEDUP_SKIP_LATEST_ASSISTANT = bool(THINKING_DEDUP_POLICY["skip_latest_assistant"])
PATTERN_RULES = list(CRUNCH_POLICY["pattern_rules"])
CODEX_REPEATED_SCAFFOLDING_POLICY = CRUNCH_POLICY["codex_repeated_scaffolding"]
CODEX_REPEATED_SCAFFOLDING_ENABLED = bool(CODEX_REPEATED_SCAFFOLDING_POLICY["enabled"])
CODEX_REPEATED_SCAFFOLDING_MIN_REQUEST_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["min_request_chars"])
CODEX_REPEATED_SCAFFOLDING_MIN_SECTION_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["min_section_chars"])
CODEX_REPEATED_SCAFFOLDING_KEEP_RECENT_INPUT_BLOCKS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["keep_recent_input_blocks"])
CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_MIN_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["older_block_min_chars"])
CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_HEAD_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["older_block_head_chars"])
CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_TAIL_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["older_block_tail_chars"])
CODEX_REPEATED_SCAFFOLDING_MAX_REPLACEMENTS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["max_replacements"])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens_from_text(text: str) -> int:
    return max(1, int(len(text) / TOKEN_CHARS))


def normalize_text(s: str) -> str:
    # Conservative crunching: whitespace cleanup only; do not paraphrase.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip() if len(s) > 200 else s


def build_embedding(text: str) -> list[float]:
    buckets = [0.0] * 256
    for tok in re.findall(r"[a-z]+", text.lower()):
        buckets[hashlib.sha256(tok.encode()).digest()[0]] += 1.0
    norm = sum(x * x for x in buckets) ** 0.5
    if norm > 0:
        buckets = [x / norm for x in buckets]
    return buckets


def _shingles(text: str) -> frozenset:
    words = text.split()
    return frozenset(tuple(words[i:i + 4]) for i in range(len(words) - 3))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _extract_text_for_category(obj: Any) -> str:
    parts: list[str] = []
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            text = _extract_text_for_category(item)
            if text:
                parts.append(text)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"text", "content", "input", "system", "name", "type"}:
                text = _extract_text_for_category(value)
                if text:
                    parts.append(text)
            elif isinstance(value, (list, dict)):
                text = _extract_text_for_category(value)
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _body_has_tools(body: dict[str, Any]) -> bool:
    if body.get("tools"):
        return True
    serialized = stable_json(body.get("messages", []))
    return "tool_use" in serialized or "tool_result" in serialized


def _crunch_request_category(body: dict[str, Any]) -> str:
    tools = _body_has_tools(body)
    text = _extract_text_for_category(body)
    text_chars = len(text)
    messages = body.get("messages") or []
    if messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "user":
            content = last.get("content", [])
            if isinstance(content, list) and content and any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content
            ):
                return "tool-result"
    if tools and text_chars > 16000:
        return "tool-heavy"
    if tools:
        return "tool-light"
    if text_chars > 32000:
        return "long-context"
    if text_chars < 1500 and len(messages) <= 2:
        return "short-completion"
    if "```" in text:
        return "code-gen"
    return "chat"


def _pattern_hash_for_text(text: str) -> str:
    return f"sha256:{sha256_text(normalize_text(text))}"


def _pattern_rule_base_meta() -> dict[str, Any]:
    return {
        "enabled": bool(PATTERN_RULES),
        "configured_count": len(PATTERN_RULES),
        "applied_count": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "rules": [],
        "skip_reasons": [],
    }


def _pattern_rule_skip(skip_counts: dict[tuple[str, str], int], rule_id: str, reason: str, count: int = 1) -> None:
    skip_counts[(rule_id, reason)] = skip_counts.get((rule_id, reason), 0) + count


def _safe_pattern_text_entries(body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safe: list[dict[str, Any]] = []
    unsafe: list[dict[str, Any]] = []

    def add(container: Any, key: Any, text: str, location: str, *, unsafe_reason: str | None = None) -> None:
        entry = {
            "container": container,
            "key": key,
            "text": text,
            "location": location,
            "hash": _pattern_hash_for_text(text),
            "normalized_chars": len(normalize_text(text)),
        }
        if unsafe_reason:
            entry["unsafe_reason"] = unsafe_reason
            unsafe.append(entry)
        else:
            safe.append(entry)

    if isinstance(body.get("system"), str):
        add(body, "system", body["system"], "system")
    elif isinstance(body.get("system"), list):
        for index, block in enumerate(body["system"]):
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                add(block, "text", block["text"], f"system[{index}].text")

    if isinstance(body.get("instructions"), str):
        add(body, "instructions", body["instructions"], "instructions")

    input_value = body.get("input")
    if isinstance(input_value, str):
        add(body, "input", input_value, "input")
    else:
        for index, entry in enumerate(_codex_text_input_entries(input_value)):
            add(entry.get("container"), entry.get("key"), str(entry.get("text") or ""), f"input[{index}]")

    messages = body.get("messages") or []
    if isinstance(messages, list):
        for msg_index, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            role = str(msg.get("role") or "unknown")
            if isinstance(content, str):
                add(msg, "content", content, f"messages[{msg_index}].content", unsafe_reason=None)
                continue
            if not isinstance(content, list):
                continue
            unsafe_reason = None
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result", "thinking"}:
                    unsafe_reason = "unsafe-tool-or-action-payload"
                    break
            for block_index, block in enumerate(content):
                if isinstance(block, str):
                    add(content, block_index, block, f"messages[{msg_index}].content[{block_index}]", unsafe_reason=unsafe_reason)
                elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    add(
                        block,
                        "text",
                        block["text"],
                        f"messages[{msg_index}].content[{block_index}].text",
                        unsafe_reason=unsafe_reason,
                    )
                elif isinstance(block, dict) and role == "assistant" and block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                    add(
                        block,
                        "thinking",
                        block["thinking"],
                        f"messages[{msg_index}].content[{block_index}].thinking",
                        unsafe_reason="unsafe-thinking-payload",
                    )
    return safe, unsafe


def _set_pattern_text_entry(entry: dict[str, Any], text: str) -> None:
    container = entry.get("container")
    key = entry.get("key")
    if isinstance(container, dict) and isinstance(key, str):
        container[key] = text
    elif isinstance(container, list) and isinstance(key, int):
        container[key] = text


def _pattern_rule_matches_request(rule: dict[str, Any], body: dict[str, Any], category: str) -> bool:
    conditions = rule.get("conditions") or {}
    model_pattern = conditions.get("model_pattern")
    if model_pattern and str(model_pattern).lower() not in str(body.get("model") or "").lower():
        return False
    if conditions.get("category") and str(conditions["category"]) != category:
        return False
    if conditions.get("workflow_phase") and str(conditions["workflow_phase"]) != category:
        return False
    category_not_in = conditions.get("category_not_in") or []
    if category in {str(item) for item in category_not_in}:
        return False
    return True


def _build_pattern_replacement(entry: dict[str, Any], rule: dict[str, Any]) -> str | None:
    action = rule.get("action") or {}
    original = str(entry.get("text") or "")
    pattern_hash = str(entry["hash"])
    short_hash = pattern_hash.split("sha256:", 1)[-1][:12]
    rule_id = str(rule.get("id") or "pattern-rule")
    marker = str(action.get("marker") or "")
    if not marker:
        marker = (
            f"[AgentFlow: reviewed crunch pattern applied; rule_id={rule_id}; "
            f"pattern_hash={short_hash}; original_chars={len(original)}]"
        )
    max_replacement_chars = max(1, int(action.get("max_replacement_chars", 2400)))
    if str(action.get("type") or "shorten") == "omit":
        return marker[:max_replacement_chars]

    head = max(0, int(action.get("head_chars", 1200)))
    tail = max(0, int(action.get("tail_chars", 800)))
    replacement = original[:head] + "\n\n" + marker + "\n\n" + (original[-tail:] if tail else "")
    if len(replacement) > max_replacement_chars:
        remaining = max_replacement_chars - len(marker) - 4
        if remaining <= 0:
            replacement = marker[:max_replacement_chars]
        else:
            head = min(head, max(0, remaining // 2))
            tail = min(tail, max(0, remaining - head))
            replacement = original[:head] + "\n\n" + marker + "\n\n" + (original[-tail:] if tail else "")
            replacement = replacement[:max_replacement_chars]
    return replacement if len(replacement) < len(original) else None


def _apply_pattern_rules(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    meta = _pattern_rule_base_meta()
    before_chars = len(stable_json(body))
    if not PATTERN_RULES:
        meta["before_chars"] = before_chars
        meta["after_chars"] = before_chars
        return 0, meta

    entries, unsafe_entries = _safe_pattern_text_entries(body)
    category = _crunch_request_category(body)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry["hash"])] = counts.get(str(entry["hash"]), 0) + 1
    unsafe_counts: dict[str, int] = {}
    for entry in unsafe_entries:
        unsafe_counts[str(entry["hash"])] = unsafe_counts.get(str(entry["hash"]), 0) + 1

    total_saved = 0
    skip_counts: dict[tuple[str, str], int] = {}
    applied_by_hash: dict[str, int] = {}

    for rule in PATTERN_RULES:
        rule_id = str(rule.get("id") or "pattern-rule")
        rule_meta = {
            "rule_id": rule_id,
            "candidate_id": rule.get("candidate_id"),
            "enabled": bool(rule.get("enabled", True)),
            "policy_source": rule.get("policy_source") or CRUNCH_POLICY_SOURCE,
            "action": (rule.get("action") or {}).get("type", "shorten"),
            "matched_hashes": [],
            "applied_count": 0,
            "saved_chars": 0,
            "skip_reasons": [],
        }
        if not rule_meta["enabled"]:
            _pattern_rule_skip(skip_counts, rule_id, "disabled")
            rule_meta["skip_reasons"].append({"reason": "disabled", "count": 1})
            meta["rules"].append(rule_meta)
            continue
        if not _pattern_rule_matches_request(rule, body, category):
            _pattern_rule_skip(skip_counts, rule_id, "request-gate-not-matched")
            rule_meta["skip_reasons"].append({"reason": "request-gate-not-matched", "count": 1})
            meta["rules"].append(rule_meta)
            continue

        conditions = rule.get("conditions") or {}
        rule_hashes = [str(item) for item in conditions.get("pattern_hashes") or []]
        min_repeated_count = max(1, int(conditions.get("min_repeated_count", 2)))
        keep_recent_matches = max(0, int(conditions.get("keep_recent_matches", 1)))
        max_applications = max(0, int(conditions.get("max_applications", len(entries))))
        for pattern_hash in rule_hashes:
            safe_count = counts.get(pattern_hash, 0)
            unsafe_count = unsafe_counts.get(pattern_hash, 0)
            if unsafe_count:
                _pattern_rule_skip(skip_counts, rule_id, "unsafe-tool-or-action-payload", unsafe_count)
                rule_meta["skip_reasons"].append({"reason": "unsafe-tool-or-action-payload", "pattern_hash": pattern_hash, "count": unsafe_count})
            if safe_count < min_repeated_count:
                _pattern_rule_skip(skip_counts, rule_id, "min-repeated-count-not-met")
                rule_meta["skip_reasons"].append({"reason": "min-repeated-count-not-met", "pattern_hash": pattern_hash, "count": safe_count})
                continue
            occurrences = [entry for entry in entries if entry["hash"] == pattern_hash]
            protected_start = max(0, len(occurrences) - keep_recent_matches)
            for occurrence_index, entry in enumerate(occurrences):
                if entry.get("pattern_rule_applied"):
                    _pattern_rule_skip(skip_counts, rule_id, "already-applied-by-earlier-rule")
                    continue
                if rule_meta["applied_count"] >= max_applications:
                    _pattern_rule_skip(skip_counts, rule_id, "max-applications-reached")
                    break
                if occurrence_index >= protected_start:
                    _pattern_rule_skip(skip_counts, rule_id, "kept-recent-match")
                    continue
                if conditions.get("min_text_chars") is not None and int(entry["normalized_chars"]) < int(conditions["min_text_chars"]):
                    _pattern_rule_skip(skip_counts, rule_id, "text-too-small")
                    continue
                if conditions.get("max_text_chars") is not None and int(entry["normalized_chars"]) > int(conditions["max_text_chars"]):
                    _pattern_rule_skip(skip_counts, rule_id, "text-too-large")
                    continue
                replacement = _build_pattern_replacement(entry, rule)
                if replacement is None:
                    _pattern_rule_skip(skip_counts, rule_id, "replacement-not-smaller")
                    continue
                before_len = len(str(entry.get("text") or ""))
                _set_pattern_text_entry(entry, replacement)
                entry["pattern_rule_applied"] = True
                saved = before_len - len(replacement)
                total_saved += saved
                rule_meta["applied_count"] += 1
                rule_meta["saved_chars"] += saved
                applied_by_hash[pattern_hash] = applied_by_hash.get(pattern_hash, 0) + 1
                if pattern_hash not in rule_meta["matched_hashes"]:
                    rule_meta["matched_hashes"].append(pattern_hash)
        meta["rules"].append(rule_meta)

    meta["applied_count"] = sum(int(rule.get("applied_count") or 0) for rule in meta["rules"])
    meta["changed"] = meta["applied_count"] > 0
    after_chars = len(stable_json(body))
    meta["before_chars"] = before_chars
    meta["after_chars"] = after_chars
    meta["saved_chars"] = before_chars - after_chars
    meta["text_saved_chars"] = total_saved
    meta["tokens_saved_est"] = (before_chars - after_chars) // TOKEN_CHARS
    meta["category"] = category
    meta["applied_pattern_hashes"] = sorted(applied_by_hash)
    meta["skip_reasons"] = [
        {"rule_id": rule_id, "reason": reason, "count": count}
        for (rule_id, reason), count in sorted(skip_counts.items())
    ]
    return total_saved, meta


def _codex_scaffolding_meta(status: str, reason: str) -> dict[str, Any]:
    return {
        "enabled": CODEX_REPEATED_SCAFFOLDING_ENABLED,
        "status": status,
        "reason": reason,
        "changed": False,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "policy": _copy_codex_scaffolding_policy(),
        "patterns": [],
    }


def _copy_codex_scaffolding_policy() -> dict[str, Any]:
    return {
        "enabled": CODEX_REPEATED_SCAFFOLDING_ENABLED,
        "min_request_chars": CODEX_REPEATED_SCAFFOLDING_MIN_REQUEST_CHARS,
        "min_section_chars": CODEX_REPEATED_SCAFFOLDING_MIN_SECTION_CHARS,
        "keep_recent_input_blocks": CODEX_REPEATED_SCAFFOLDING_KEEP_RECENT_INPUT_BLOCKS,
        "older_block_min_chars": CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_MIN_CHARS,
        "older_block_head_chars": CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_HEAD_CHARS,
        "older_block_tail_chars": CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_TAIL_CHARS,
        "max_replacements": CODEX_REPEATED_SCAFFOLDING_MAX_REPLACEMENTS,
    }


def _codex_text_input_entries(input_value: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(input_value, str):
        entries.append({"container": None, "key": None, "text": input_value})
        return entries
    if isinstance(input_value, dict):
        block_type = str(input_value.get("type") or "text").strip().lower()
        for key in ("text", "input_text", "value"):
            if block_type in {"text", "input_text"} and isinstance(input_value.get(key), str):
                entries.append({"container": input_value, "key": key, "text": input_value[key]})
                break
        return entries
    if isinstance(input_value, list):
        for item_idx, item in enumerate(input_value):
            if isinstance(item, str):
                entries.append({"container": input_value, "key": item_idx, "text": item})
            elif isinstance(item, dict):
                block_type = str(item.get("type") or "text").strip().lower()
                for key in ("text", "input_text", "value"):
                    if block_type in {"text", "input_text"} and isinstance(item.get(key), str):
                        entries.append({"container": item, "key": key, "text": item[key]})
                        break
    return entries


def _set_codex_text_entry(entry: dict[str, Any], text: str) -> None:
    container = entry.get("container")
    key = entry.get("key")
    if container is None:
        entry["text"] = text
    elif isinstance(container, list) and isinstance(key, int):
        container[key] = text
    elif isinstance(container, dict) and isinstance(key, str):
        container[key] = text


def _crunch_codex_repeated_sections(
    text: str,
    *,
    block_number: int,
    is_recent_block: bool,
    seen_sections: dict[str, dict[str, int]],
    counters: dict[str, Any],
) -> str:
    parts = re.split(r"(\n\s*\n+)", text)
    section_part_indexes = [
        idx
        for idx in range(0, len(parts), 2)
        if len(normalize_text(parts[idx])) >= CODEX_REPEATED_SCAFFOLDING_MIN_SECTION_CHARS
    ]
    protected_parts = set(section_part_indexes[-1:]) if is_recent_block else set()
    section_number = 0

    for idx in range(0, len(parts), 2):
        section = parts[idx]
        normalized = normalize_text(section)
        if len(normalized) < CODEX_REPEATED_SCAFFOLDING_MIN_SECTION_CHARS:
            continue
        section_number += 1
        h = sha256_text(normalized)
        if idx in protected_parts:
            seen_sections.setdefault(h, {"block": block_number, "section": section_number})
            continue
        previous = seen_sections.get(h)
        if previous and counters["section_replacements"] < CODEX_REPEATED_SCAFFOLDING_MAX_REPLACEMENTS:
            counters["section_replacements"] += 1
            counters["section_saved_chars"] += max(0, len(section) - 140)
            if len(counters["section_hashes"]) < 8:
                counters["section_hashes"].append(h[:12])
            parts[idx] = (
                "[AgentFlow: repeated Codex input section omitted; "
                f"same_as=block:{previous['block']}/section:{previous['section']}; "
                f"hash={h[:12]}; original_chars={len(section)}]"
            )
        else:
            seen_sections.setdefault(h, {"block": block_number, "section": section_number})
    return "".join(parts)


def crunch_codex_turn_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Codex app-server specific deterministic crunching for text-only turn/start input.

    This operates only on input text blocks. It avoids tool/action payloads by being called
    after the Codex proxy's action-shape guard and preserves the newest input tail.
    """
    if not CODEX_REPEATED_SCAFFOLDING_ENABLED:
        return params, _codex_scaffolding_meta("skipped", "disabled")

    before = len(stable_json(params))
    if before < CODEX_REPEATED_SCAFFOLDING_MIN_REQUEST_CHARS:
        meta = _codex_scaffolding_meta("skipped", "request-too-small")
        meta["before_chars"] = before
        return params, meta

    input_value = params.get("input")
    entries = _codex_text_input_entries(input_value)
    if not entries:
        meta = _codex_scaffolding_meta("skipped", "no-text-input")
        meta["before_chars"] = before
        return params, meta

    new_params = copy.deepcopy(params)
    new_entries = _codex_text_input_entries(new_params.get("input"))
    recent_start = max(0, len(new_entries) - max(0, CODEX_REPEATED_SCAFFOLDING_KEEP_RECENT_INPUT_BLOCKS))
    seen_sections: dict[str, dict[str, int]] = {}
    counters: dict[str, Any] = {
        "section_replacements": 0,
        "section_saved_chars": 0,
        "section_hashes": [],
        "older_blocks_shortened": 0,
        "older_block_saved_chars": 0,
        "older_block_hashes": [],
    }

    for idx, entry in enumerate(new_entries):
        text = str(entry["text"])
        is_recent = idx >= recent_start
        text = _crunch_codex_repeated_sections(
            text,
            block_number=idx + 1,
            is_recent_block=is_recent,
            seen_sections=seen_sections,
            counters=counters,
        )
        if (
            not is_recent
            and len(text) >= CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_MIN_CHARS
            and counters["older_blocks_shortened"] < CODEX_REPEATED_SCAFFOLDING_MAX_REPLACEMENTS
        ):
            h = sha256_text(text)
            head = CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_HEAD_CHARS
            tail = CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_TAIL_CHARS
            notice = (
                "\n\n[AgentFlow: older Codex input block shortened; "
                f"hash={h[:12]}; original_chars={len(text)}]\n\n"
            )
            shortened = text[:head] + notice + text[-tail:]
            if len(shortened) < len(text):
                counters["older_blocks_shortened"] += 1
                counters["older_block_saved_chars"] += len(text) - len(shortened)
                if len(counters["older_block_hashes"]) < 8:
                    counters["older_block_hashes"].append(h[:12])
                text = shortened
        _set_codex_text_entry(entry, text)

    if isinstance(new_params.get("input"), str) and new_entries:
        new_params["input"] = new_entries[0]["text"]

    after = len(stable_json(new_params))
    meta = _codex_scaffolding_meta("applied" if after != before else "skipped", "codex-repeated-scaffolding-crunched" if after != before else "no-repeated-scaffolding")
    patterns: list[dict[str, Any]] = []
    if counters["section_replacements"]:
        patterns.append({
            "type": "repeated_input_section",
            "count": counters["section_replacements"],
            "saved_chars_est": counters["section_saved_chars"],
            "hashes": counters["section_hashes"],
        })
    if counters["older_blocks_shortened"]:
        patterns.append({
            "type": "older_input_head_tail",
            "count": counters["older_blocks_shortened"],
            "saved_chars_est": counters["older_block_saved_chars"],
            "hashes": counters["older_block_hashes"],
        })
    meta.update({
        "changed": after != before,
        "before_chars": before,
        "after_chars": after,
        "saved_chars": before - after,
        "tokens_saved_est": (before - after) // TOKEN_CHARS,
        "input_text_blocks": len(new_entries),
        "patterns": patterns,
        "pattern_types": [pattern["type"] for pattern in patterns],
        "repeated_sections_replaced": counters["section_replacements"],
        "older_input_blocks_shortened": counters["older_blocks_shortened"],
    })
    return new_params, meta


def _latest_assistant_message_index(messages: list[Any]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return idx
    return None


def _dedupe_thinking_blocks(messages: list[Any]) -> int:
    if not THINKING_DEDUP_ENABLED:
        return 0

    latest_assistant_idx = _latest_assistant_message_index(messages)
    seen_newer: list[tuple[frozenset, int]] = []
    removed = 0

    for msg_idx in range(len(messages) - 1, -1, -1):
        msg = messages[msg_idx]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        new_content: list[Any] = []
        changed = False
        for block_idx in range(len(content) - 1, -1, -1):
            block = content[block_idx]
            remove_block = False
            if isinstance(block, dict) and block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                thinking = block["thinking"]
                if len(thinking) >= THINKING_DEDUP_MIN_CHARS:
                    shingles = _shingles(thinking)
                    must_preserve = (
                        THINKING_DEDUP_SKIP_LATEST_ASSISTANT
                        and latest_assistant_idx is not None
                        and msg_idx == latest_assistant_idx
                    )
                    if not must_preserve and len(content) > 1:
                        for prev_shingles, _prev_idx in seen_newer:
                            if _jaccard(shingles, prev_shingles) > THINKING_DEDUP_SIMILARITY_THRESHOLD:
                                remove_block = True
                                removed += 1
                                changed = True
                                break
                    if not remove_block:
                        seen_newer.append((shingles, block_idx))
            if not remove_block:
                new_content.append(block)

        if changed:
            new_content.reverse()
            msg["content"] = new_content

    return removed


def _summary_base_meta(status: str, reason: str) -> dict[str, Any]:
    return {
        "enabled": OLD_CONTEXT_SUMMARY_ENABLED,
        "status": status,
        "reason": reason,
        "changed": False,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "model": OLD_CONTEXT_SUMMARY_MODEL,
        "placement": OLD_CONTEXT_SUMMARY_PLACEMENT,
        "min_request_chars": OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS,
        "min_summarized_chars": OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS,
        "max_turns": OLD_CONTEXT_SUMMARY_MAX_TURNS,
        "keep_recent_turns": OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS,
        "max_summary_chars": OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS,
        "max_source_chars": OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS,
    }


def _non_tool_message_text(msg: dict[str, Any]) -> str | None:
    if msg.get("role") not in {"user", "assistant"}:
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        block_type = block.get("type")
        if block_type in {"tool_use", "tool_result", "thinking"}:
            return None
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type is not None:
            return None
    text = "\n".join(parts)
    return text if text else None


def old_context_summary_plan(
    body: dict[str, Any],
    *,
    exact_cache_enabled: bool | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not OLD_CONTEXT_SUMMARY_ENABLED:
        return None, _summary_base_meta("skipped", "disabled")

    before_chars = len(stable_json(body))
    if before_chars < OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS:
        meta = _summary_base_meta("skipped", "request-too-small")
        meta["before_chars"] = before_chars
        return None, meta

    messages = body.get("messages") or []
    if not isinstance(messages, list) or len(messages) <= OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS:
        meta = _summary_base_meta("skipped", "not-enough-old-turns")
        meta["before_chars"] = before_chars
        return None, meta

    old_limit = max(0, len(messages) - OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS)
    candidates: list[dict[str, Any]] = []
    source_parts: list[str] = []
    total_chars = 0
    source_truncated = False
    for idx, msg in enumerate(messages[:old_limit]):
        if len(candidates) >= OLD_CONTEXT_SUMMARY_MAX_TURNS:
            break
        if not isinstance(msg, dict):
            continue
        text = _non_tool_message_text(msg)
        if text is None:
            continue
        normalized = normalize_text(text)
        if not normalized:
            continue
        remaining = OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS - total_chars
        if remaining <= 0:
            source_truncated = True
            break
        included = normalized[:remaining]
        if len(included) < len(normalized):
            source_truncated = True
        candidates.append({"index": idx, "role": msg.get("role"), "chars": len(normalized)})
        source_parts.append(f"<turn index=\"{idx}\" role=\"{msg.get('role')}\">\n{included}\n</turn>")
        total_chars += len(included)

    if total_chars < OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS or not candidates:
        meta = _summary_base_meta("skipped", "eligible-context-too-small")
        meta["before_chars"] = before_chars
        meta["eligible_turns"] = len(candidates)
        meta["eligible_chars"] = total_chars
        return None, meta

    source_text = "\n\n".join(source_parts)
    source_hash = sha256_text(source_text)
    summary_request = {
        "model": OLD_CONTEXT_SUMMARY_MODEL,
        "max_tokens": max(256, OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS // TOKEN_CHARS),
        "temperature": 0,
        "system": (
            "Summarize old conversation context for a coding agent. Preserve durable facts, "
            "decisions, constraints, file paths, command outcomes, and unresolved tasks. "
            "Do not invent details. Keep the result compact and factual."
        ),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize these old non-tool turns. The current request will keep recent "
                    "turns and all tool-use/tool-result protocol messages unchanged.\n\n"
                    f"{source_text}"
                ),
            }
        ],
    }
    plan = {
        "source_hash": source_hash,
        "cache_key": "agentflow-old-context-summary\n" + source_hash,
        "summary_request": summary_request,
        "placement": OLD_CONTEXT_SUMMARY_PLACEMENT,
        "candidate_indexes": [c["index"] for c in candidates],
        "candidate_roles": [c["role"] for c in candidates],
        "eligible_turns": len(candidates),
        "eligible_chars": total_chars,
        "source_truncated": source_truncated,
        "before_chars": before_chars,
    }
    meta = _summary_base_meta("planned", "eligible")
    meta.update({
        "before_chars": before_chars,
        "eligible_turns": len(candidates),
        "eligible_chars": total_chars,
        "source_hash": source_hash[:12],
        "source_truncated": source_truncated,
        "summary_cache_enabled": True,
        "exact_cache_enabled": exact_cache_enabled,
    })
    return plan, meta


def _summary_text_from_result(result: Any) -> str | None:
    if isinstance(result, str):
        text = result.strip()
        return text or None
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("summary"), str):
        text = result["summary"].strip()
        return text or None
    parts: list[str] = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    text = "\n".join(parts).strip()
    return text or None


def apply_old_context_summary(body: dict[str, Any], plan: dict[str, Any], summary: str) -> dict[str, Any]:
    new_body = copy.deepcopy(body)
    messages = new_body.get("messages") or []
    indexes = set(plan["candidate_indexes"])
    notice = (
        f"[AgentFlow: old non-tool context summarized by {OLD_CONTEXT_SUMMARY_MODEL}; "
        f"source_turns={plan['eligible_turns']}; source_chars={plan['eligible_chars']}; "
        f"source_hash={plan['source_hash'][:12]}]\n\n"
        f"{summary.strip()[:OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS]}"
    )
    system = new_body.get("system")
    summary_block = {"type": "text", "text": notice}
    if system is None:
        new_body["system"] = [summary_block]
    elif isinstance(system, str):
        new_body["system"] = [
            {"type": "text", "text": system},
            summary_block,
        ]
    elif isinstance(system, list):
        new_body["system"] = copy.deepcopy(system) + [summary_block]
    else:
        new_body["system"] = [summary_block]
    new_body["messages"] = [msg for idx, msg in enumerate(messages) if idx not in indexes]
    return new_body


async def maybe_summarize_old_context(
    body: dict[str, Any],
    *,
    exact_cache_enabled: bool,
    get_cached_summary: Any,
    set_cached_summary: Any,
    fetch_summary: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, meta = old_context_summary_plan(body, exact_cache_enabled=exact_cache_enabled)
    if plan is None:
        return body, meta

    cached = get_cached_summary(plan["cache_key"])
    summary = _summary_text_from_result(cached)
    summary_cache_hit = summary is not None
    fetch_result: Any = None
    if summary is None:
        fetch_result = await fetch_summary(plan["summary_request"])
        summary = _summary_text_from_result(fetch_result)
        if summary is None:
            meta.update({"status": "skipped", "reason": "summary-empty"})
            if isinstance(fetch_result, dict):
                for key in (
                    "summary_status_code",
                    "summary_error",
                    "summary_input_tokens",
                    "summary_output_tokens",
                    "summary_cost_est_usd",
                ):
                    if key in fetch_result:
                        meta[key] = fetch_result[key]
            return body, meta
        set_cached_summary(plan["cache_key"], {
            "summary": summary,
            "usage": fetch_result.get("usage") if isinstance(fetch_result, dict) else None,
        })

    summarized = apply_old_context_summary(body, plan, summary)
    after_chars = len(stable_json(summarized))
    meta.update({
        "status": "applied",
        "reason": "summary-cache-hit" if summary_cache_hit else "summary-created",
        "changed": after_chars != plan["before_chars"],
        "after_chars": after_chars,
        "saved_chars": plan["before_chars"] - after_chars,
        "tokens_saved_est": (plan["before_chars"] - after_chars) // TOKEN_CHARS,
        "summary_cache_hit": summary_cache_hit,
        "summary_chars": len(summary),
    })
    if isinstance(fetch_result, dict):
        for key in ("summary_input_tokens", "summary_output_tokens", "summary_cost_est_usd", "summary_status_code"):
            if key in fetch_result:
                meta[key] = fetch_result[key]
    return summarized, meta


def crunch_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservative request cruncher.

    It deliberately does NOT summarize with another model. For agent use this is safer.
    Current tactics:
    - normalize whitespace in large text blocks
    - deduplicate exact repeated text blocks within the same request
    - if extremely large, compress older non-tool text blocks to bounded heads/tails
    """
    if not CRUNCH_ENABLED:
        return body, {
            "enabled": False,
            "changed": False,
            "policy_source": CRUNCH_POLICY_SOURCE,
            "rule_path": CRUNCH_RULES_PATH,
        }

    new_body = copy.deepcopy(body)
    before = len(stable_json(new_body))
    _pattern_saved_chars, pattern_rules_meta = _apply_pattern_rules(new_body)
    seen: dict[str, int] = {}
    seen_shingles: list[tuple[frozenset, int]] = []
    replacements = 0
    near_replacements = 0
    thinking_near_replacements = 0
    shortened = 0

    def process_content(content: Any, allow_shorten: bool) -> Any:
        nonlocal replacements, near_replacements, shortened
        if isinstance(content, str):
            txt = normalize_text(content)
            h = sha256_text(txt)
            if len(txt) > 1000 and h in seen:
                replacements += 1
                return f"[AgentFlow: exact duplicate text block omitted; same as earlier block #{seen[h]} hash={h[:12]}]"
            seen[h] = len(seen) + 1
            if len(txt) > 2000:
                shingles = _shingles(txt)
                matched_idx = None
                for prev_shingles, prev_idx in seen_shingles:
                    if _jaccard(shingles, prev_shingles) > 0.85:
                        matched_idx = prev_idx
                        break
                if matched_idx is not None:
                    near_replacements += 1
                    return f"[AgentFlow: near-duplicate text block omitted; similar to earlier block #{matched_idx} jaccard>0.85; original_chars={len(txt)}]"
                seen_shingles.append((shingles, len(seen)))
            if allow_shorten and len(txt) > 8000:
                shortened += 1
                return txt[:3500] + f"\n\n[AgentFlow: middle of long older text block omitted; hash={h[:12]}; original_chars={len(txt)}]\n\n" + txt[-2500:]
            return txt
        if isinstance(content, list):
            out = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    item = copy.deepcopy(item)
                    item["text"] = process_content(item["text"], allow_shorten)
                elif isinstance(item, dict) and item.get("type") in {"tool_use", "tool_result"}:
                    # Tool blocks are protocol/state sensitive; don't alter.
                    pass
                elif isinstance(item, dict):
                    item = process_content(item, allow_shorten)
                elif isinstance(item, str):
                    item = process_content(item, allow_shorten)
                out.append(item)
            return out
        if isinstance(content, dict):
            out = copy.deepcopy(content)
            for k, v in list(out.items()):
                if k in {"text", "content"}:
                    out[k] = process_content(v, allow_shorten)
            return out
        return content

    # System can be string or list of blocks.
    if "system" in new_body:
        new_body["system"] = process_content(new_body["system"], allow_shorten=False)
    if "instructions" in new_body:
        new_body["instructions"] = process_content(new_body["instructions"], allow_shorten=False)
    if "input" in new_body:
        new_body["input"] = process_content(new_body["input"], allow_shorten=False)

    messages = new_body.get("messages") or []
    huge = before > CRUNCH_THRESHOLD_CHARS
    for idx, msg in enumerate(messages):
        # only shorten older text, not the latest user/assistant context
        allow_shorten = huge and idx < max(0, len(messages) - 4)
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = process_content(msg["content"], allow_shorten=allow_shorten)
    thinking_near_replacements = _dedupe_thinking_blocks(messages)

    after = len(stable_json(new_body))
    return new_body, {
        "enabled": True,
        "changed": after != before,
        "before_chars": before,
        "after_chars": after,
        "saved_chars": before - after,
        "tokens_before_est": before // TOKEN_CHARS,
        "tokens_after_est": after // TOKEN_CHARS,
        "tokens_saved_est": (before - after) // TOKEN_CHARS,
        "crunch_ratio": round((before - after) / before, 4) if before > 0 else 0,
        "duplicate_blocks_replaced": replacements,
        "near_duplicate_blocks_replaced": near_replacements,
        "thinking_near_duplicate_blocks_removed": thinking_near_replacements,
        "long_blocks_shortened": shortened,
        "pattern_rules_applied": pattern_rules_meta["applied_count"],
        "pattern_rule_saved_chars": pattern_rules_meta["saved_chars"],
        "pattern_rules": pattern_rules_meta,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "threshold_chars": CRUNCH_THRESHOLD_CHARS,
        "thinking_deduplication": {
            "enabled": THINKING_DEDUP_ENABLED,
            "min_chars": THINKING_DEDUP_MIN_CHARS,
            "similarity_threshold": THINKING_DEDUP_SIMILARITY_THRESHOLD,
            "skip_latest_assistant": THINKING_DEDUP_SKIP_LATEST_ASSISTANT,
        },
    }


def inject_prompt_cache(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not PROMPT_CACHE_ENABLED:
        return body, False
    system = body.get("system")
    if system is None:
        return body, False
    if isinstance(system, str):
        if len(system) < PROMPT_CACHE_MIN_CHARS:
            return body, False
        new_body = copy.deepcopy(body)
        new_body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return new_body, True
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("cache_control"):
                return body, False
        total_text = sum(len(b.get("text", "")) for b in system if isinstance(b, dict) and b.get("type") == "text")
        if total_text < PROMPT_CACHE_MIN_CHARS:
            return body, False
        new_body = copy.deepcopy(body)
        for i in range(len(new_body["system"]) - 1, -1, -1):
            block = new_body["system"][i]
            if isinstance(block, dict) and block.get("type") == "text":
                block["cache_control"] = {"type": "ephemeral"}
                break
        return new_body, True
    return body, False


def has_cache_control_blocks(body: dict[str, Any]) -> bool:
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("cache_control"):
                return True
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control"):
                    return True
    return False
