from __future__ import annotations

import copy
import hashlib
import os
import re
from typing import Any

from agentflow_proxy.store import stable_json

CRUNCH_ENABLED = os.getenv("AGENTFLOW_CRUNCH", "1") != "0"
CRUNCH_THRESHOLD_CHARS = int(os.getenv("AGENTFLOW_CRUNCH_THRESHOLD_CHARS", "24000"))
PROMPT_CACHE_ENABLED = os.getenv("AGENTFLOW_PROMPT_CACHE", "1") != "0"
PROMPT_CACHE_MIN_CHARS = int(os.getenv("AGENTFLOW_PROMPT_CACHE_MIN_CHARS", "4096"))

TOKEN_CHARS = 4  # rough estimator only


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


def crunch_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservative request cruncher.

    It deliberately does NOT summarize with another model. For agent use this is safer.
    Current tactics:
    - normalize whitespace in large text blocks
    - deduplicate exact repeated text blocks within the same request
    - if extremely large, compress older non-tool text blocks to bounded heads/tails
    """
    if not CRUNCH_ENABLED:
        return body, {"enabled": False, "changed": False, "policy_source": "local-default"}

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
        "policy_source": "local-default",
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
