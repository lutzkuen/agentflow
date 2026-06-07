from __future__ import annotations

import copy
import hashlib
import os
import re
import yaml
from pathlib import Path
from typing import Any

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
            "min_request_chars": 32000,
            "min_summarized_chars": 12000,
            "max_turns": 6,
            "keep_recent_turns": 4,
            "max_summary_chars": 4000,
            "max_source_chars": 80000,
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


CRUNCH_POLICY, CRUNCH_POLICY_SOURCE, CRUNCH_RULES_PATH = _load_crunch_policy()
CRUNCH_ENABLED = bool(CRUNCH_POLICY["enabled"])
CRUNCH_THRESHOLD_CHARS = int(CRUNCH_POLICY["threshold_chars"])
PROMPT_CACHE_ENABLED = bool(CRUNCH_POLICY["prompt_cache"]["enabled"])
PROMPT_CACHE_MIN_CHARS = int(CRUNCH_POLICY["prompt_cache"]["min_chars"])
OLD_CONTEXT_SUMMARY_POLICY = CRUNCH_POLICY["old_context_summarization"]
OLD_CONTEXT_SUMMARY_ENABLED = bool(OLD_CONTEXT_SUMMARY_POLICY["enabled"])
OLD_CONTEXT_SUMMARY_MODEL = str(OLD_CONTEXT_SUMMARY_POLICY["model"])
OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["min_request_chars"])
OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["min_summarized_chars"])
OLD_CONTEXT_SUMMARY_MAX_TURNS = int(OLD_CONTEXT_SUMMARY_POLICY["max_turns"])
OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS = int(OLD_CONTEXT_SUMMARY_POLICY["keep_recent_turns"])
OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["max_summary_chars"])
OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["max_source_chars"])


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


def _summary_base_meta(status: str, reason: str) -> dict[str, Any]:
    return {
        "enabled": OLD_CONTEXT_SUMMARY_ENABLED,
        "status": status,
        "reason": reason,
        "changed": False,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "model": OLD_CONTEXT_SUMMARY_MODEL,
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


def old_context_summary_plan(body: dict[str, Any], *, exact_cache_enabled: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not OLD_CONTEXT_SUMMARY_ENABLED:
        return None, _summary_base_meta("skipped", "disabled")
    if not exact_cache_enabled:
        return None, _summary_base_meta("skipped", "exact-cache-required")

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
    first_idx = min(indexes)
    notice = (
        f"[AgentFlow: old non-tool context summarized by {OLD_CONTEXT_SUMMARY_MODEL}; "
        f"source_turns={plan['eligible_turns']}; source_chars={plan['eligible_chars']}; "
        f"source_hash={plan['source_hash'][:12]}]\n\n"
        f"{summary.strip()[:OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS]}"
    )
    replacement = {"role": "user", "content": [{"type": "text", "text": notice}]}
    new_messages = []
    for idx, msg in enumerate(messages):
        if idx == first_idx:
            new_messages.append(replacement)
        if idx in indexes:
            continue
        new_messages.append(msg)
    new_body["messages"] = new_messages
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
    seen: dict[str, int] = {}
    seen_shingles: list[tuple[frozenset, int]] = []
    replacements = 0
    near_replacements = 0
    shortened = 0

    def _shingles(text: str) -> frozenset:
        words = text.split()
        return frozenset(tuple(words[i:i + 4]) for i in range(len(words) - 3))

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
                    intersection = len(shingles & prev_shingles)
                    union = len(shingles | prev_shingles)
                    if union > 0 and intersection / union > 0.85:
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
        "long_blocks_shortened": shortened,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "threshold_chars": CRUNCH_THRESHOLD_CHARS,
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
