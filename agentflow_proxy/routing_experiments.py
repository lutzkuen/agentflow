from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable

import yaml

from agentflow_proxy.crunch import build_embedding, sha256_text
from agentflow_proxy.cache import response_output_text
from agentflow_proxy.store import cosine_similarity, stable_json


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


def _default_experiment_policy() -> dict[str, Any]:
    return {
        "enabled": False,
        "sample_rate": 0.0,
        "min_text_chars": 0,
        "max_text_chars": 30000,
        "categories": [],
        "similarity_threshold": 0.86,
        "min_samples_for_confidence": 20,
        "store_response_bodies": False,
    }


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(Path.home() / ".agentflow" / filename)
    return candidates


def _apply_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
    if data.get("sample_rate") is not None:
        policy["sample_rate"] = max(0.0, min(1.0, float(data["sample_rate"])))
    if data.get("min_text_chars") is not None:
        policy["min_text_chars"] = int(data["min_text_chars"])
    if data.get("max_text_chars") is not None:
        policy["max_text_chars"] = int(data["max_text_chars"])
    categories = data.get("categories")
    if isinstance(categories, list):
        policy["categories"] = [str(c) for c in categories if c is not None]
    if data.get("similarity_threshold") is not None:
        policy["similarity_threshold"] = max(0.0, min(1.0, float(data["similarity_threshold"])))
    if data.get("min_samples_for_confidence") is not None:
        policy["min_samples_for_confidence"] = max(1, int(data["min_samples_for_confidence"]))
    policy["store_response_bodies"] = _as_bool(
        data.get("store_response_bodies"),
        policy["store_response_bodies"],
    )
    return policy


def _load_experiment_policy() -> tuple[dict[str, Any], str, str]:
    for path in _manual_rule_candidates("routing_experiments.yaml", "AGENTFLOW_ROUTING_EXPERIMENTS"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return _apply_policy_yaml(_default_experiment_policy(), data), "local-manual", str(path)

    defaults_path = Path(__file__).parent / "routing_experiments.yaml"
    policy = _default_experiment_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _apply_policy_yaml(policy, data)

    policy["enabled"] = os.getenv("AGENTFLOW_ROUTING_EXPERIMENTS_ENABLED", "1" if policy["enabled"] else "0") != "0"
    policy["sample_rate"] = max(
        0.0,
        min(1.0, float(os.getenv("AGENTFLOW_ROUTING_EXPERIMENT_SAMPLE_RATE", str(policy["sample_rate"])))),
    )
    policy["similarity_threshold"] = max(
        0.0,
        min(1.0, float(os.getenv("AGENTFLOW_ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD", str(policy["similarity_threshold"])))),
    )
    return policy, "local-default", str(defaults_path)


ROUTING_EXPERIMENT_POLICY, ROUTING_EXPERIMENT_POLICY_SOURCE, ROUTING_EXPERIMENT_RULES_PATH = _load_experiment_policy()
ROUTING_EXPERIMENT_ENABLED = bool(ROUTING_EXPERIMENT_POLICY["enabled"])
ROUTING_EXPERIMENT_SAMPLE_RATE = float(ROUTING_EXPERIMENT_POLICY["sample_rate"])
ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD = float(ROUTING_EXPERIMENT_POLICY["similarity_threshold"])
ROUTING_EXPERIMENT_MIN_SAMPLES = int(ROUTING_EXPERIMENT_POLICY["min_samples_for_confidence"])
ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES = bool(ROUTING_EXPERIMENT_POLICY["store_response_bodies"])


def routing_experiment_decision(
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    *,
    stream: bool,
    random_value: Callable[[], float] = random.random,
) -> dict[str, Any]:
    requested = str(routing_meta.get("requested_model") or body.get("model") or "")
    routed = str(routing_meta.get("routed_model") or body.get("model") or "")
    category = str(routing_meta.get("category") or "")
    text_chars = int(routing_meta.get("text_chars") or 0)
    categories = set(str(c) for c in ROUTING_EXPERIMENT_POLICY.get("categories") or [])

    meta = {
        "enabled": ROUTING_EXPERIMENT_ENABLED,
        "status": "skipped",
        "sampled": False,
        "reason": "disabled",
        "policy_source": ROUTING_EXPERIMENT_POLICY_SOURCE,
        "rule_path": ROUTING_EXPERIMENT_RULES_PATH,
        "sample_rate": ROUTING_EXPERIMENT_SAMPLE_RATE,
        "similarity_threshold": ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD,
        "min_samples_for_confidence": ROUTING_EXPERIMENT_MIN_SAMPLES,
        "requested_model": requested,
        "routed_model": routed,
        "shadow_model": requested,
        "category": category,
        "text_chars": text_chars,
    }
    if not ROUTING_EXPERIMENT_ENABLED:
        return meta
    if stream:
        meta["reason"] = "streaming"
        return meta
    if not requested or requested == routed:
        meta["reason"] = "not-routed-down"
        return meta
    if routing_meta.get("fallback_reason"):
        meta["reason"] = "fallback-used"
        return meta
    min_chars = int(ROUTING_EXPERIMENT_POLICY.get("min_text_chars") or 0)
    max_chars = int(ROUTING_EXPERIMENT_POLICY.get("max_text_chars") or 0)
    if text_chars < min_chars:
        meta["reason"] = "request-too-small"
        return meta
    if max_chars > 0 and text_chars > max_chars:
        meta["reason"] = "request-too-large"
        return meta
    if categories and category not in categories:
        meta["reason"] = "category-not-enabled"
        return meta
    if ROUTING_EXPERIMENT_SAMPLE_RATE <= 0:
        meta["reason"] = "sample-rate-zero"
        return meta
    if random_value() >= ROUTING_EXPERIMENT_SAMPLE_RATE:
        meta["reason"] = "not-sampled"
        return meta

    meta["status"] = "selected"
    meta["sampled"] = True
    meta["reason"] = "sampled-routed-down-call"
    return meta


def compare_response_outputs(primary_response: dict[str, Any] | None, shadow_response: dict[str, Any] | None) -> dict[str, Any]:
    primary_text = response_output_text(primary_response or {})
    shadow_text = response_output_text(shadow_response or {})
    if primary_text or shadow_text:
        similarity = cosine_similarity(build_embedding(primary_text), build_embedding(shadow_text))
    else:
        similarity = 1.0 if stable_json(primary_response or {}) == stable_json(shadow_response or {}) else 0.0
    return {
        "primary_output_chars": len(primary_text),
        "shadow_output_chars": len(shadow_text),
        "primary_output_sha256": sha256_text(primary_text),
        "shadow_output_sha256": sha256_text(shadow_text),
        "output_similarity": round(float(similarity), 6),
        "passed_threshold": float(similarity) >= ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD,
    }

