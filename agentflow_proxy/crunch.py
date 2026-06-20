from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.paths import agentflow_config_path
from agentflow_proxy.pricing import pricing_basis
from agentflow_proxy.pattern_rollout import (
    PATTERN_ROLLOUT_SCHEMA,
    normalize_pattern_rollout,
    pattern_canary_decision,
    pattern_rollout_public_meta,
)
from agentflow_proxy.pattern_safety import (
    LOCAL_CANARY_SAFETY_STOP_REASON,
    evaluate_pattern_canary_safety_stop,
    log_pattern_canary_safety_stop,
)
from agentflow_proxy.pattern_modules import evaluate_pattern_modules, registered_pattern_modules
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.request_shape_rollups import request_shape_crunch_canary_lifecycle
from agentflow_proxy.store import stable_json
from agentflow_proxy.terminal_compaction_dry_run import (
    DEFAULT_HEAD_LINES as TERMINAL_COMPACTION_DEFAULT_HEAD_LINES,
    DEFAULT_KEEP_RECENT_TURNS as TERMINAL_COMPACTION_DEFAULT_KEEP_RECENT_TURNS,
    DEFAULT_MAX_EVIDENCE_LINES as TERMINAL_COMPACTION_DEFAULT_MAX_EVIDENCE_LINES,
    DEFAULT_MIN_BLOCK_CHARS as TERMINAL_COMPACTION_DEFAULT_MIN_BLOCK_CHARS,
    DEFAULT_MIN_SAVED_CHARS as TERMINAL_COMPACTION_DEFAULT_MIN_SAVED_CHARS,
    DEFAULT_TAIL_LINES as TERMINAL_COMPACTION_DEFAULT_TAIL_LINES,
    plan_terminal_output_compaction,
)

TOKEN_CHARS = 4  # rough estimator only
ENHANCED_CRUNCH_PROVIDER_MODES = {
    "disabled",
    "local_provider_account",
    "customer_sidecar",
    "customer_controlled_endpoint",
}

CRUNCH_RULE_TAXONOMY_SCHEMA = "agentflow.crunch_rule_taxonomy.v1"
CRUNCH_RULE_DECISION_SCHEMA = "agentflow.crunch_rule_decision.v1"
CRUNCH_RULE_GROUP_BREAKDOWN_SCHEMA = "agentflow.crunch_rule_group_breakdown.v1"
CRUNCH_CANARY_STATES = {"off", "shadow", "canary", "active", "held", "rollback"}

CRUNCH_RULE_CATALOG: dict[str, dict[str, Any]] = {
    "whitespace_normalization": {
        "rule_group": "lossless_normalization",
        "lossiness_class": "lossless",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions", "codex_turn"],
        "requires_canary": False,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 1.0,
        "canary_state": "active",
    },
    "exact_duplicate_block_omission": {
        "rule_group": "structural_dedup",
        "lossiness_class": "structure_preserving",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions", "codex_turn"],
        "requires_canary": False,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 1.0,
        "canary_state": "active",
    },
    "near_duplicate_block_omission": {
        "rule_group": "structural_dedup",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions", "codex_turn"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 1.0,
        "canary_state": "active",
    },
    "old_text_collapse": {
        "rule_group": "structural_dedup",
        "lossiness_class": "structure_preserving",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions"],
        "requires_canary": False,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 1.0,
        "canary_state": "active",
    },
    "thinking_deduplication": {
        "rule_group": "structural_dedup",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 1.0,
        "canary_state": "active",
    },
    "terminal_log_boilerplate": {
        "rule_group": "lossless_normalization",
        "lossiness_class": "structure_preserving",
        "surfaces": ["anthropic_messages", "openai_responses", "codex_turn"],
        "requires_canary": False,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 1.0,
        "canary_state": "active",
    },
    "pattern_rules": {
        "rule_group": "structural_dedup",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions", "codex_turn"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
    "old_context_summarization": {
        "rule_group": "old_context_summarization",
        "lossiness_class": "model_generated_summary",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
    "anthropic_thinking_history_compaction": {
        "rule_group": "semantic_compaction",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
    "terminal_output_compaction": {
        "rule_group": "semantic_compaction",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
    "instruction_section_deduplication": {
        "rule_group": "semantic_compaction",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions", "codex_turn"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
    "repeated_provider_scaffolding": {
        "rule_group": "semantic_compaction",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
    "request_shape_repeated_context_canary": {
        "rule_group": "semantic_compaction",
        "lossiness_class": "semantic_loss_risk",
        "surfaces": ["anthropic_messages", "openai_responses", "openai_chat_completions"],
        "requires_canary": True,
        "requires_server_decision": False,
        "rollback_supported": True,
        "default_rollout_fraction": 0.0,
        "canary_state": "off",
    },
}


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


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_crunch_policy_yaml(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    nested = data.get("crunch")
    if not isinstance(nested, dict):
        return data
    merged = dict(data)
    for key, value in nested.items():
        merged[key] = value
    return merged


def _normalize_allowed_rules(value: Any) -> list[str] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    rules = [str(item).strip() for item in values if str(item).strip()]
    return sorted(set(rules))


def _canary_fraction(raw_canary: Any) -> float | None:
    if not isinstance(raw_canary, dict):
        return None
    for key in ("fraction", "canary_fraction", "rollout_fraction"):
        if raw_canary.get(key) is not None:
            try:
                return max(0.0, min(1.0, float(raw_canary[key])))
            except (TypeError, ValueError):
                return None
    return None


def _holdout_fraction(raw_canary: Any) -> float | None:
    if not isinstance(raw_canary, dict):
        return None
    if raw_canary.get("holdout_fraction") is None:
        return None
    try:
        return max(0.0, min(1.0, float(raw_canary["holdout_fraction"])))
    except (TypeError, ValueError):
        return None


def _derive_canary_state(
    *,
    enabled: bool | None,
    canary: Any = None,
    configured_state: Any = None,
    default_state: str = "off",
) -> str:
    raw_state = str(configured_state or "").strip().lower().replace("_", "-")
    normalized = raw_state.replace("-", "_")
    if normalized in CRUNCH_CANARY_STATES:
        return normalized
    if enabled is False:
        return "off"
    if not isinstance(canary, dict):
        return "active" if enabled else default_state
    if not _as_bool(canary.get("enabled"), False):
        return "active" if enabled else "off"
    fraction = _canary_fraction(canary)
    holdout = _holdout_fraction(canary)
    if fraction is None:
        fraction = 0.0
    if fraction >= 1.0:
        return "active"
    if fraction > 0.0:
        return "canary"
    if holdout and holdout > 0.0:
        return "held"
    return "shadow"


def _crunch_rule_metadata(
    rule_id: str,
    *,
    policy_source: str | None = None,
    rule_path: str | None = None,
    enabled: bool | None = None,
    canary: Any = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = dict(CRUNCH_RULE_CATALOG.get(rule_id) or {})
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                catalog[key] = value
    if not catalog:
        catalog = {
            "rule_group": "custom",
            "lossiness_class": "semantic_loss_risk",
            "surfaces": [],
            "requires_canary": True,
            "requires_server_decision": False,
            "rollback_supported": True,
            "default_rollout_fraction": 0.0,
            "canary_state": "off",
        }
    state = _derive_canary_state(
        enabled=enabled,
        canary=canary,
        configured_state=None if canary is not None else catalog.get("canary_state"),
        default_state=str(catalog.get("canary_state") or "off"),
    )
    meta = {
        "schema": CRUNCH_RULE_DECISION_SCHEMA,
        "rule_id": str(rule_id),
        "rule_group": str(catalog.get("rule_group") or "custom"),
        "lossiness_class": str(catalog.get("lossiness_class") or "semantic_loss_risk"),
        "surfaces": [str(item) for item in (catalog.get("surfaces") or [])],
        "requires_canary": bool(catalog.get("requires_canary")),
        "requires_server_decision": bool(catalog.get("requires_server_decision")),
        "rollback_supported": bool(catalog.get("rollback_supported")),
        "default_rollout_fraction": float(catalog.get("default_rollout_fraction") or 0.0),
        "canary_state": state,
    }
    if policy_source is not None:
        meta["policy_source"] = str(policy_source)
    if rule_path is not None:
        meta["rule_path"] = str(rule_path)
    return meta


def _allowed_crunch_rule_ids(policy: dict[str, Any]) -> set[str] | None:
    normalized = _normalize_allowed_rules(policy.get("allowed_rules"))
    return None if normalized is None else set(normalized)


def _crunch_rule_allowed(rule_id: str) -> bool:
    return ALLOWED_CRUNCH_RULES is None or rule_id in ALLOWED_CRUNCH_RULES


def _crunch_rule_decision(
    rule_id: str,
    *,
    status: str,
    reason: str,
    policy_source: str,
    rule_path: str,
    count: int = 1,
    saved_chars: int = 0,
    tokens_saved_est: int | None = None,
    enabled: bool | None = None,
    canary: Any = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = _crunch_rule_metadata(
        rule_id,
        policy_source=policy_source,
        rule_path=rule_path,
        enabled=enabled,
        canary=canary,
        overrides=overrides,
    )
    meta.update({
        "status": str(status),
        "reason": str(reason),
        "count": max(0, int(count)),
        "saved_chars": int(saved_chars or 0),
        "tokens_saved_est": int(tokens_saved_est if tokens_saved_est is not None else int(saved_chars or 0) // TOKEN_CHARS),
    })
    return meta


def _crunch_rule_group_breakdown(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in decisions:
        key = (
            str(item.get("rule_group") or "unknown"),
            str(item.get("lossiness_class") or "unknown"),
            str(item.get("status") or "unknown"),
        )
        bucket = grouped.setdefault(
            key,
            {
                "schema": CRUNCH_RULE_GROUP_BREAKDOWN_SCHEMA,
                "rule_group": key[0],
                "lossiness_class": key[1],
                "status": key[2],
                "count": 0,
                "decision_count": 0,
                "saved_chars": 0,
                "tokens_saved_est": 0,
                "rule_ids": [],
            },
        )
        bucket["count"] += int(item.get("count") or 0)
        bucket["decision_count"] += 1
        bucket["saved_chars"] += int(item.get("saved_chars") or 0)
        bucket["tokens_saved_est"] += int(item.get("tokens_saved_est") or 0)
        rule_id = str(item.get("rule_id") or "")
        if rule_id and rule_id not in bucket["rule_ids"]:
            bucket["rule_ids"].append(rule_id)
    return sorted(grouped.values(), key=lambda row: (row["status"] != "applied", -row["tokens_saved_est"], row["rule_group"]))


def _crunch_rule_taxonomy_meta(*, policy_source: str, rule_path: str) -> dict[str, Any]:
    allowed = sorted(ALLOWED_CRUNCH_RULES) if ALLOWED_CRUNCH_RULES is not None else None
    try:
        sections = {
            "old_context_summarization": OLD_CONTEXT_SUMMARY_POLICY,
            "thinking_deduplication": THINKING_DEDUP_POLICY,
            "terminal_log_boilerplate": TERMINAL_LOG_POLICY,
            "anthropic_thinking_history_compaction": ANTHROPIC_THINKING_COMPACTION_POLICY,
            "terminal_output_compaction": TERMINAL_OUTPUT_COMPACTION_POLICY,
            "instruction_section_deduplication": INSTRUCTION_SECTION_DEDUP_POLICY,
            "repeated_provider_scaffolding": REPEATED_PROVIDER_SCAFFOLDING_POLICY,
            "request_shape_repeated_context_canary": REQUEST_SHAPE_REPEATED_CONTEXT_CANARIES_POLICY,
        }
    except NameError:
        sections = {}
    return {
        "schema": CRUNCH_RULE_TAXONOMY_SCHEMA,
        "policy_source": policy_source,
        "rule_path": rule_path,
        "allowed_rules": allowed if allowed is not None else ["*"],
        "allowed_rule_count": len(allowed) if allowed is not None else len(CRUNCH_RULE_CATALOG),
        "rules": [
            _crunch_rule_metadata(
                rule_id,
                policy_source=policy_source,
                rule_path=rule_path,
                enabled=(
                    _crunch_rule_allowed(rule_id)
                    and _as_bool(sections.get(rule_id, {}).get("enabled"), True)
                    if isinstance(sections.get(rule_id), dict)
                    else _crunch_rule_allowed(rule_id)
                ),
                canary=sections.get(rule_id, {}).get("canary") if isinstance(sections.get(rule_id), dict) else None,
            )
            for rule_id in sorted(CRUNCH_RULE_CATALOG)
        ],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }


def _default_crunch_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "allowed_rules": None,
        "threshold_chars": 24000,
        "prompt_cache": {
            "enabled": True,
            "min_chars": 4096,
        },
        "enhanced_crunch_provider": {
            "mode": "disabled",
            "profile": "default",
            "model": None,
            "model_family": None,
            "endpoint_url": None,
            "max_summary_cost_usd": None,
        },
        "old_context_summarization": {
            "enabled": False,
            "rule_id": "local-old-context-summarization",
            "candidate_id": None,
            "model": "claude-haiku-4-5-20251001",
            "placement": "system",
            "min_request_chars": 32000,
            "min_summarized_chars": 12000,
            "max_turns": 6,
            "keep_recent_turns": 4,
            "max_summary_chars": 4000,
            "max_source_chars": 80000,
            "max_summary_cost_usd": 0.02,
            "excluded_categories": ["tool-heavy", "tool-result"],
            "block_tool_protocol": True,
            "block_thinking": True,
            "canary": {
                "enabled": False,
                "fraction": 1.0,
                "holdout_fraction": None,
                "salt": "",
                "unit": "source_hash",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 5,
                "window": 500,
                "max_error_rate": 0.1,
                "max_retry_rate": 0.25,
                "max_negative_net_savings_rate": 0.5,
                "max_summary_failure_rate": 0.1,
                "max_error_rate_delta": 0.05,
            },
        },
        "thinking_deduplication": {
            "enabled": True,
            "min_chars": 2000,
            "similarity_threshold": 0.95,
            "skip_latest_assistant": True,
        },
        "anthropic_thinking_history_compaction": {
            "enabled": False,
            "policy_source": "local-default",
            "rule_id": "local-anthropic-thinking-history-compaction-canary",
            "candidate_id": None,
            "action_id": None,
            "conditions": {},
            "min_text_chars": 8_000,
            "min_block_chars": 2_000,
            "similarity_threshold": 0.95,
            "replacement_notice": "[AgentFlow: older duplicate thinking history compacted]",
            "block_top_level_thinking": True,
            "canary": {
                "enabled": True,
                "fraction": 0.0,
                "holdout_fraction": 1.0,
                "salt": "",
                "unit": "thinking_block_local_fingerprint",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 5,
                "window": 500,
                "max_error_rate": 0.1,
                "max_retry_rate": 0.25,
                "max_negative_savings_rate": 0.25,
                "max_missing_usage_rate": 0.1,
                "max_error_rate_delta": 0.05,
            },
            "rules": [],
        },
        "instruction_section_deduplication": {
            "enabled": False,
            "policy_source": "local-default",
            "rules": [],
            "source_surfaces": [
                "anthropic_messages",
                "openai_responses",
                "openai_chat_completions",
                "codex_turn",
            ],
            "categories": [],
            "workflow_phases": [],
            "min_section_chars": 700,
            "min_repeated_count": 2,
            "keep_recent_sections": 1,
            "replacement_notice": "[repeated instruction section omitted by AgentFlow]",
            "max_replacements": 0,
            "block_tool_protocol": True,
            "block_tool_payloads": True,
            "block_responses": True,
            "block_thinking": True,
            "canary": {
                "enabled": True,
                "fraction": 0.0,
                "holdout_fraction": 1.0,
                "salt": "",
                "unit": "instruction_section_fingerprint",
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
        },
        "terminal_log_boilerplate": {
            "enabled": True,
            "min_lines": 8,
            "min_repeated_lines": 4,
            "max_annotations": 12,
        },
        "terminal_output_compaction": {
            "enabled": False,
            "rule_id": "local-terminal-output-compaction-canary",
            "candidate_id": None,
            "action_id": None,
            "conditions": {},
            "provenance": None,
            "keep_recent_turns": TERMINAL_COMPACTION_DEFAULT_KEEP_RECENT_TURNS,
            "min_block_chars": TERMINAL_COMPACTION_DEFAULT_MIN_BLOCK_CHARS,
            "head_lines": TERMINAL_COMPACTION_DEFAULT_HEAD_LINES,
            "tail_lines": TERMINAL_COMPACTION_DEFAULT_TAIL_LINES,
            "max_evidence_lines": TERMINAL_COMPACTION_DEFAULT_MAX_EVIDENCE_LINES,
            "min_saved_chars": TERMINAL_COMPACTION_DEFAULT_MIN_SAVED_CHARS,
            "block_thinking": True,
            "canary": {
                "enabled": True,
                "fraction": 0.0,
                "holdout_fraction": 1.0,
                "salt": "",
                "unit": "request_fingerprint",
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
            "rules": [],
        },
        "pattern_modules": {
            "diffs": {
                "enabled": True,
                "local_crunch_enabled": True,
            },
            "generated_artifacts": {
                "enabled": True,
                "local_crunch_enabled": True,
            },
            "terminal_logs": {
                "enabled": True,
                "local_crunch_enabled": False,
            },
            "prompt_role": {
                "enabled": True,
                "local_crunch_enabled": False,
            },
            "tool_results": {
                "enabled": True,
                "local_crunch_enabled": True,
            },
        },
        "pattern_rules": [],
        "request_shape_repeated_context_canaries": {
            "enabled": False,
            "schema": "agentflow.request_shape_repeated_context_canaries.v1",
            "rules": [],
        },
        "repeated_provider_scaffolding": {
            "enabled": False,
            "rules": [],
            "min_request_chars": 12000,
            "min_section_chars": 700,
            "keep_recent_messages": 2,
            "keep_recent_matches": 1,
            "max_replacements": 16,
            "block_tool_protocol": True,
            "block_thinking": True,
        },
        "session_memory_hints": {
            "enabled": False,
            "rule_id": "local-session-plateau-crunch-hint",
            "crunch_profile": "plateau-repeated-context-review",
            "old_context_summary_canary": False,
            "min_call_count": 4,
            "min_plateau_pairs": 3,
            "min_text_chars": 8000,
            "max_error_rate": 0.0,
            "allowed_phases": ["planning", "verification", "summary"],
            "block_tool_results": True,
            "block_thinking": True,
            "projected_savings_ratio": 0.10,
        },
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
    candidates.append(agentflow_config_path(filename))
    return candidates


def _first_existing_rule_path(filename: str, env_name: str) -> Path | None:
    for path in _manual_rule_candidates(filename, env_name):
        if path.exists():
            return path
    return None


def _apply_scaffold_canary_overlay(policy: dict[str, Any], *, base_source: str) -> str | None:
    path = _first_existing_rule_path("scaffold_canary_policy.yaml", "AGENTFLOW_SCAFFOLD_CANARY_POLICY")
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return None
    provider_scaffolding = data.get("repeated_provider_scaffolding") or {}
    if not isinstance(provider_scaffolding, dict):
        return None
    _apply_provider_scaffolding_policy_yaml(
        policy,
        provider_scaffolding,
        default_policy_source=str(data.get("policy_source") or "managed-recommended"),
    )
    target = policy["repeated_provider_scaffolding"]
    target["policy_source"] = str(data.get("policy_source") or target.get("policy_source") or "managed-recommended")
    target["overlay_rule_path"] = str(path)
    target["base_policy_source"] = base_source
    return str(path)


def _load_crunch_policy() -> tuple[dict[str, Any], str, str]:
    for path in _manual_rule_candidates("crunch_rules.yaml", "AGENTFLOW_CRUNCH_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = _normalized_crunch_policy_yaml(yaml.safe_load(f) or {})
        if isinstance(data, dict):
            policy = _default_crunch_policy()
            policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
            if "allowed_rules" in data:
                policy["allowed_rules"] = _normalize_allowed_rules(data.get("allowed_rules"))
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
            provider = data.get("enhanced_crunch_provider") or data.get("enhanced_provider") or {}
            if isinstance(provider, dict):
                _apply_enhanced_provider_yaml(policy, provider)
            thinking_dedup = data.get("thinking_deduplication") or {}
            if isinstance(thinking_dedup, dict):
                _apply_thinking_dedup_policy_yaml(policy, thinking_dedup)
            thinking_compaction = data.get("anthropic_thinking_history_compaction") or data.get("thinking_history_compaction") or {}
            if isinstance(thinking_compaction, dict):
                _apply_anthropic_thinking_compaction_policy_yaml(
                    policy,
                    thinking_compaction,
                    default_policy_source="local-manual",
                )
            instruction_dedup = data.get("instruction_section_deduplication") or {}
            if isinstance(instruction_dedup, dict):
                _apply_instruction_dedup_policy_yaml(
                    policy,
                    instruction_dedup,
                    default_policy_source="local-manual",
                )
            terminal_log = data.get("terminal_log_boilerplate") or {}
            if isinstance(terminal_log, dict):
                _apply_terminal_log_policy_yaml(policy, terminal_log)
            terminal_compaction = data.get("terminal_output_compaction") or {}
            if isinstance(terminal_compaction, dict):
                _apply_terminal_output_compaction_policy_yaml(
                    policy,
                    terminal_compaction,
                    default_policy_source="local-manual",
                )
            pattern_modules = data.get("pattern_modules") or {}
            if isinstance(pattern_modules, dict):
                _apply_pattern_modules_policy_yaml(policy, pattern_modules)
            pattern_rules = data.get("pattern_rules")
            if pattern_rules is not None:
                policy["pattern_rules"] = _parse_pattern_rules_yaml(pattern_rules, default_policy_source="local-manual")
            request_shape_canaries = data.get("request_shape_repeated_context_canaries") or {}
            if isinstance(request_shape_canaries, dict):
                _apply_request_shape_repeated_context_canaries_yaml(
                    policy,
                    request_shape_canaries,
                    default_policy_source="local-manual",
                )
            provider_scaffolding = data.get("repeated_provider_scaffolding") or {}
            if isinstance(provider_scaffolding, dict):
                _apply_provider_scaffolding_policy_yaml(policy, provider_scaffolding, default_policy_source="local-manual")
            session_memory_hints = data.get("session_memory_hints") or {}
            if isinstance(session_memory_hints, dict):
                _apply_session_memory_hints_policy_yaml(policy, session_memory_hints)
            codex_scaffolding = data.get("codex_repeated_scaffolding") or {}
            if isinstance(codex_scaffolding, dict):
                _apply_codex_scaffolding_policy_yaml(policy, codex_scaffolding)
            _promote_legacy_summary_provider(policy)
            _apply_scaffold_canary_overlay(policy, base_source="local-manual")
            return policy, "local-manual", str(path)

    defaults_path = Path(__file__).parent / "crunch_rules.yaml"
    policy = _default_crunch_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = _normalized_crunch_policy_yaml(yaml.safe_load(f) or {})
        if isinstance(data, dict):
            policy["enabled"] = _as_bool(data.get("enabled"), policy["enabled"])
            if "allowed_rules" in data:
                policy["allowed_rules"] = _normalize_allowed_rules(data.get("allowed_rules"))
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
            provider = data.get("enhanced_crunch_provider") or data.get("enhanced_provider") or {}
            if isinstance(provider, dict):
                _apply_enhanced_provider_yaml(policy, provider)
            thinking_dedup = data.get("thinking_deduplication") or {}
            if isinstance(thinking_dedup, dict):
                _apply_thinking_dedup_policy_yaml(policy, thinking_dedup)
            thinking_compaction = data.get("anthropic_thinking_history_compaction") or data.get("thinking_history_compaction") or {}
            if isinstance(thinking_compaction, dict):
                _apply_anthropic_thinking_compaction_policy_yaml(
                    policy,
                    thinking_compaction,
                    default_policy_source="local-default",
                )
            instruction_dedup = data.get("instruction_section_deduplication") or {}
            if isinstance(instruction_dedup, dict):
                _apply_instruction_dedup_policy_yaml(
                    policy,
                    instruction_dedup,
                    default_policy_source="local-default",
                )
            terminal_log = data.get("terminal_log_boilerplate") or {}
            if isinstance(terminal_log, dict):
                _apply_terminal_log_policy_yaml(policy, terminal_log)
            terminal_compaction = data.get("terminal_output_compaction") or {}
            if isinstance(terminal_compaction, dict):
                _apply_terminal_output_compaction_policy_yaml(
                    policy,
                    terminal_compaction,
                    default_policy_source="local-default",
                )
            pattern_modules = data.get("pattern_modules") or {}
            if isinstance(pattern_modules, dict):
                _apply_pattern_modules_policy_yaml(policy, pattern_modules)
            pattern_rules = data.get("pattern_rules")
            if pattern_rules is not None:
                policy["pattern_rules"] = _parse_pattern_rules_yaml(pattern_rules, default_policy_source="local-default")
            request_shape_canaries = data.get("request_shape_repeated_context_canaries") or {}
            if isinstance(request_shape_canaries, dict):
                _apply_request_shape_repeated_context_canaries_yaml(
                    policy,
                    request_shape_canaries,
                    default_policy_source="local-default",
                )
            provider_scaffolding = data.get("repeated_provider_scaffolding") or {}
            if isinstance(provider_scaffolding, dict):
                _apply_provider_scaffolding_policy_yaml(policy, provider_scaffolding, default_policy_source="local-default")
            session_memory_hints = data.get("session_memory_hints") or {}
            if isinstance(session_memory_hints, dict):
                _apply_session_memory_hints_policy_yaml(policy, session_memory_hints)
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
    if os.getenv("AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT") is not None:
        summary["enabled"] = os.getenv("AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT", "0") == "1"
    summary["model"] = os.getenv("AGENTFLOW_HAIKU_SUMMARY_MODEL", str(summary["model"]))
    summary["min_request_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MIN_REQUEST_CHARS", str(summary["min_request_chars"])))
    summary["min_summarized_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MIN_SUMMARIZED_CHARS", str(summary["min_summarized_chars"])))
    summary["max_turns"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MAX_TURNS", str(summary["max_turns"])))
    summary["keep_recent_turns"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_KEEP_RECENT_TURNS", str(summary["keep_recent_turns"])))
    summary["max_summary_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MAX_SUMMARY_CHARS", str(summary["max_summary_chars"])))
    summary["max_source_chars"] = int(os.getenv("AGENTFLOW_HAIKU_SUMMARY_MAX_SOURCE_CHARS", str(summary["max_source_chars"])))
    provider = policy["enhanced_crunch_provider"]
    provider["mode"] = _normalize_enhanced_provider_mode(
        os.getenv("AGENTFLOW_ENHANCED_CRUNCH_MODE", str(provider["mode"]))
    )
    provider["model"] = os.getenv("AGENTFLOW_ENHANCED_CRUNCH_MODEL", provider.get("model") or "") or provider.get("model")
    provider["model_family"] = (
        os.getenv("AGENTFLOW_ENHANCED_CRUNCH_MODEL_FAMILY", provider.get("model_family") or "")
        or provider.get("model_family")
    )
    provider["endpoint_url"] = (
        os.getenv("AGENTFLOW_ENHANCED_CRUNCH_ENDPOINT_URL", provider.get("endpoint_url") or "")
        or provider.get("endpoint_url")
    )
    if os.getenv("AGENTFLOW_ENHANCED_CRUNCH_MAX_SUMMARY_COST_USD") is not None:
        provider["max_summary_cost_usd"] = float(os.getenv("AGENTFLOW_ENHANCED_CRUNCH_MAX_SUMMARY_COST_USD", "0"))
    _promote_legacy_summary_provider(policy)
    _apply_scaffold_canary_overlay(policy, base_source="local-default")
    return policy, "local-default", str(defaults_path)


def _normalize_enhanced_provider_mode(value: Any) -> str:
    mode = str(value or "disabled").strip().lower().replace("-", "_")
    return mode if mode in ENHANCED_CRUNCH_PROVIDER_MODES else "disabled"


def _apply_enhanced_provider_yaml(policy: dict[str, Any], provider: dict[str, Any]) -> None:
    target = policy["enhanced_crunch_provider"]
    target["mode"] = _normalize_enhanced_provider_mode(provider.get("mode", target["mode"]))
    if provider.get("profile") is not None:
        target["profile"] = str(provider["profile"])
    if provider.get("model") is not None:
        target["model"] = str(provider["model"])
    if provider.get("model_family") is not None:
        target["model_family"] = str(provider["model_family"])
    if provider.get("endpoint_url") is not None:
        target["endpoint_url"] = str(provider["endpoint_url"])
    if provider.get("url") is not None:
        target["endpoint_url"] = str(provider["url"])
    if provider.get("max_summary_cost_usd") is not None:
        target["max_summary_cost_usd"] = float(provider["max_summary_cost_usd"])


def _promote_legacy_summary_provider(policy: dict[str, Any]) -> None:
    """Treat an explicit local summary policy as local provider configuration.

    Older configs only had old_context_summarization.enabled. Preserve that behavior
    while still requiring a configured provider for managed hints that enable
    summarization from the default disabled state.
    """
    provider = policy["enhanced_crunch_provider"]
    if _as_bool(policy["old_context_summarization"].get("enabled"), False) and provider.get("mode") == "disabled":
        provider["mode"] = "local_provider_account"
        provider["profile"] = provider.get("profile") or "old_context_summarization"


def _apply_summary_policy_yaml(policy: dict[str, Any], summary: dict[str, Any]) -> None:
    target = policy["old_context_summarization"]
    target["enabled"] = _as_bool(summary.get("enabled"), target["enabled"])
    if summary.get("rule_id") is not None:
        target["rule_id"] = str(summary["rule_id"])
    if summary.get("candidate_id") is not None:
        target["candidate_id"] = str(summary["candidate_id"])
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
    if summary.get("max_summary_cost_usd") is not None:
        target["max_summary_cost_usd"] = float(summary["max_summary_cost_usd"])
    if summary.get("excluded_categories") is not None:
        raw_categories = summary.get("excluded_categories")
        if isinstance(raw_categories, list):
            target["excluded_categories"] = [str(item) for item in raw_categories]
        elif raw_categories in ("", None):
            target["excluded_categories"] = []
        else:
            target["excluded_categories"] = [str(raw_categories)]
    target["block_tool_protocol"] = _as_bool(summary.get("block_tool_protocol"), target["block_tool_protocol"])
    target["block_thinking"] = _as_bool(summary.get("block_thinking"), target["block_thinking"])
    canary = summary.get("canary") or {}
    if isinstance(canary, dict):
        target_canary = target["canary"]
        target_canary["enabled"] = _as_bool(canary.get("enabled"), target_canary["enabled"])
        for source_key, target_key in (
            ("fraction", "fraction"),
            ("rollout_fraction", "fraction"),
            ("canary_fraction", "fraction"),
            ("holdout_fraction", "holdout_fraction"),
        ):
            if canary.get(source_key) is not None:
                target_canary[target_key] = max(0.0, min(1.0, float(canary[source_key])))
        if canary.get("salt") is not None:
            target_canary["salt"] = str(canary["salt"])
        if canary.get("canary_salt") is not None:
            target_canary["salt"] = str(canary["canary_salt"])
        if canary.get("unit") is not None:
            target_canary["unit"] = str(canary["unit"])
        if canary.get("canary_unit") is not None:
            target_canary["unit"] = str(canary["canary_unit"])
    safety = summary.get("safety_stop") or {}
    if isinstance(safety, dict):
        target_safety = target["safety_stop"]
        target_safety["enabled"] = _as_bool(safety.get("enabled"), target_safety["enabled"])
        for key in ("min_outcome_samples", "window"):
            if safety.get(key) is not None:
                target_safety[key] = int(safety[key])
        for key in (
            "max_error_rate",
            "max_retry_rate",
            "max_negative_net_savings_rate",
            "max_summary_failure_rate",
            "max_error_rate_delta",
        ):
            if safety.get(key) is not None:
                target_safety[key] = max(0.0, float(safety[key]))


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


def _apply_anthropic_thinking_compaction_policy_yaml(
    policy: dict[str, Any],
    thinking_compaction: dict[str, Any],
    *,
    default_policy_source: str,
) -> None:
    target = policy["anthropic_thinking_history_compaction"]
    action = thinking_compaction.get("action") if isinstance(thinking_compaction.get("action"), dict) else thinking_compaction
    target["enabled"] = _as_bool(thinking_compaction.get("enabled"), target["enabled"])
    target["policy_source"] = str(thinking_compaction.get("policy_source") or target.get("policy_source") or default_policy_source)
    for key in ("rule_id", "candidate_id", "action_id", "replacement_notice"):
        if thinking_compaction.get(key) is not None:
            target[key] = str(thinking_compaction[key])
    if isinstance(thinking_compaction.get("conditions"), dict):
        target["conditions"] = _sanitize_anthropic_thinking_compaction_conditions(thinking_compaction["conditions"])
    for key in ("min_text_chars", "min_block_chars"):
        if action.get(key) is not None:
            target[key] = int(action[key])
    if action.get("similarity_threshold") is not None:
        target["similarity_threshold"] = max(0.0, min(1.0, float(action["similarity_threshold"])))
    target["block_top_level_thinking"] = _as_bool(
        thinking_compaction.get("block_top_level_thinking"),
        target["block_top_level_thinking"],
    )
    _apply_fraction_canary_yaml(target["canary"], thinking_compaction.get("canary") or thinking_compaction.get("rollout"))
    safety = thinking_compaction.get("safety_stop") or {}
    if isinstance(safety, dict):
        target_safety = target["safety_stop"]
        target_safety["enabled"] = _as_bool(safety.get("enabled"), target_safety["enabled"])
        for key in ("min_outcome_samples", "window"):
            if safety.get(key) is not None:
                target_safety[key] = int(safety[key])
        for key in (
            "max_error_rate",
            "max_retry_rate",
            "max_negative_savings_rate",
            "max_missing_usage_rate",
            "max_error_rate_delta",
        ):
            if safety.get(key) is not None:
                target_safety[key] = max(0.0, float(safety[key]))
    rules = thinking_compaction.get("rules")
    if rules is None and (thinking_compaction.get("conditions") or thinking_compaction.get("candidate_id")):
        rules = [thinking_compaction]
    parsed = _parse_anthropic_thinking_compaction_rules_yaml(
        rules,
        base_policy=target,
        default_policy_source=target["policy_source"],
    )
    if parsed:
        target["rules"] = parsed


def _list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _apply_fraction_canary_yaml(target_canary: dict[str, Any], raw_canary: Any) -> None:
    if not isinstance(raw_canary, dict):
        return
    validation_errors = _instruction_canary_validation_errors(raw_canary)
    target_canary["enabled"] = _as_bool(raw_canary.get("enabled"), target_canary["enabled"])
    for source_key, target_key in (
        ("fraction", "fraction"),
        ("canary_fraction", "fraction"),
        ("rollout_fraction", "fraction"),
        ("holdout_fraction", "holdout_fraction"),
    ):
        if raw_canary.get(source_key) is not None:
            try:
                target_canary[target_key] = max(0.0, min(1.0, float(raw_canary[source_key])))
            except (TypeError, ValueError):
                pass
    if raw_canary.get("salt") is not None:
        target_canary["salt"] = str(raw_canary["salt"])
    if raw_canary.get("canary_salt") is not None:
        target_canary["salt"] = str(raw_canary["canary_salt"])
    if raw_canary.get("unit") is not None:
        target_canary["unit"] = str(raw_canary["unit"])
    if raw_canary.get("canary_unit") is not None:
        target_canary["unit"] = str(raw_canary["canary_unit"])
    if validation_errors:
        target_canary["validation_errors"] = validation_errors


def _instruction_canary_validation_errors(raw_canary: Any) -> list[str]:
    if raw_canary is None:
        return []
    if not isinstance(raw_canary, dict):
        return ["invalid-canary-configuration"]
    errors: list[str] = []

    def parse_unit_interval(key: str) -> float | None:
        if raw_canary.get(key) is None:
            return None
        try:
            value = float(raw_canary.get(key))
        except (TypeError, ValueError):
            errors.append(f"invalid-canary-{key.replace('_', '-')}")
            return None
        if value < 0.0 or value > 1.0:
            errors.append(f"invalid-canary-{key.replace('_', '-')}")
        return value

    fraction = parse_unit_interval("fraction")
    if fraction is None:
        fraction = parse_unit_interval("canary_fraction")
    if fraction is None:
        fraction = parse_unit_interval("rollout_fraction")
    holdout_fraction = parse_unit_interval("holdout_fraction")
    if fraction is not None and holdout_fraction is not None and 0.0 <= fraction <= 1.0 and 0.0 <= holdout_fraction <= 1.0:
        if fraction + holdout_fraction > 1.0:
            errors.append("invalid-canary-fraction-sum")
    configured_errors = raw_canary.get("validation_errors")
    if isinstance(configured_errors, list):
        errors.extend(public_label(item, "invalid-canary-configuration") for item in configured_errors if str(item))
    return sorted(set(errors))


def _apply_instruction_dedup_safety_yaml(target_safety: dict[str, Any], raw_safety: Any) -> None:
    if not isinstance(raw_safety, dict):
        return
    target_safety["enabled"] = _as_bool(raw_safety.get("enabled"), target_safety["enabled"])
    for key in ("min_outcome_samples", "window"):
        if raw_safety.get(key) is not None:
            target_safety[key] = int(raw_safety[key])
    for key in ("max_error_rate", "max_retry_rate", "max_negative_savings_rate", "max_error_rate_delta"):
        if raw_safety.get(key) is not None:
            target_safety[key] = max(0.0, float(raw_safety[key]))


def _parse_instruction_dedup_rule_yaml(
    item: dict[str, Any],
    *,
    index: int,
    base_policy: dict[str, Any],
    default_policy_source: str,
) -> dict[str, Any] | None:
    conditions = item.get("conditions") if isinstance(item.get("conditions"), dict) else {}
    action = item.get("action") if isinstance(item.get("action"), dict) else {}
    hashes = _parse_pattern_hashes(
        item.get("instruction_section_fingerprints")
        or item.get("instruction_section_fingerprint")
        or item.get("instruction_fingerprint_hashes")
        or item.get("instruction_fingerprint_hash")
        or conditions.get("instruction_section_fingerprints")
        or conditions.get("instruction_section_fingerprint")
        or conditions.get("instruction_fingerprint_hashes")
        or conditions.get("instruction_fingerprint_hash")
    )
    if not hashes:
        return None
    rule = {
        "id": str(item.get("id") or item.get("rule_id") or item.get("candidate_id") or f"instruction-dedup-rule-{index + 1}"),
        "candidate_id": item.get("candidate_id"),
        "enabled": _as_bool(item.get("enabled"), True),
        "policy_source": str(item.get("policy_source") or base_policy.get("policy_source") or default_policy_source),
        "instruction_section_fingerprints": hashes,
        "source_surfaces": _list_of_strings(
            item.get("source_surfaces")
            or item.get("source_surface")
            or conditions.get("source_surfaces")
            or conditions.get("source_surface")
        ),
        "categories": _list_of_strings(item.get("categories") or item.get("category") or conditions.get("categories") or conditions.get("category")),
        "workflow_phases": _list_of_strings(
            item.get("workflow_phases")
            or item.get("workflow_phase")
            or item.get("phases")
            or item.get("phase")
            or conditions.get("workflow_phases")
            or conditions.get("workflow_phase")
            or conditions.get("phases")
            or conditions.get("phase")
        ),
        "min_section_chars": int(conditions.get("min_section_chars", item.get("min_section_chars", base_policy["min_section_chars"]))),
        "min_repeated_count": int(conditions.get("min_repeated_count", item.get("min_repeated_count", base_policy["min_repeated_count"]))),
        "keep_recent_sections": int(item.get("keep_recent_sections", item.get("keep_recent_matches", base_policy["keep_recent_sections"]))),
        "replacement_notice": str(item.get("replacement_notice") or action.get("replacement_notice") or base_policy["replacement_notice"]),
        "max_replacements": int(item.get("max_replacements", action.get("max_replacements", base_policy["max_replacements"]))),
        "block_tool_protocol": _as_bool(item.get("block_tool_protocol"), bool(base_policy["block_tool_protocol"])),
        "block_tool_payloads": _as_bool(item.get("block_tool_payloads"), bool(base_policy["block_tool_payloads"])),
        "block_responses": _as_bool(item.get("block_responses"), bool(base_policy["block_responses"])),
        "block_thinking": _as_bool(item.get("block_thinking"), bool(base_policy["block_thinking"])),
        "action": {
            "type": str(action.get("type") or "omit_instruction_section"),
        },
        "canary": copy.deepcopy(base_policy["canary"]),
        "safety_stop": copy.deepcopy(base_policy["safety_stop"]),
    }
    _apply_fraction_canary_yaml(rule["canary"], item.get("canary") or item.get("rollout"))
    _apply_instruction_dedup_safety_yaml(rule["safety_stop"], item.get("safety_stop"))
    if item.get("description") is not None:
        rule["description"] = str(item["description"])
    return rule


def _apply_instruction_dedup_policy_yaml(
    policy: dict[str, Any],
    instruction_dedup: dict[str, Any],
    *,
    default_policy_source: str,
) -> None:
    target = policy["instruction_section_deduplication"]
    target["enabled"] = _as_bool(instruction_dedup.get("enabled"), target["enabled"])
    target["policy_source"] = str(instruction_dedup.get("policy_source") or default_policy_source)
    for key in ("source_surfaces", "categories", "workflow_phases"):
        if instruction_dedup.get(key) is not None:
            target[key] = _list_of_strings(instruction_dedup[key])
    for key in ("min_section_chars", "min_repeated_count", "keep_recent_sections", "max_replacements"):
        if instruction_dedup.get(key) is not None:
            target[key] = int(instruction_dedup[key])
    if instruction_dedup.get("replacement_notice") is not None:
        target["replacement_notice"] = str(instruction_dedup["replacement_notice"])
    for key in ("block_tool_protocol", "block_tool_payloads", "block_responses", "block_thinking"):
        if instruction_dedup.get(key) is not None:
            target[key] = _as_bool(instruction_dedup[key], bool(target[key]))
    _apply_fraction_canary_yaml(target["canary"], instruction_dedup.get("canary") or instruction_dedup.get("rollout"))
    _apply_instruction_dedup_safety_yaml(target["safety_stop"], instruction_dedup.get("safety_stop"))
    rules = instruction_dedup.get("rules")
    if rules is None and (
        instruction_dedup.get("instruction_section_fingerprint")
        or instruction_dedup.get("instruction_section_fingerprints")
        or instruction_dedup.get("instruction_fingerprint_hash")
        or instruction_dedup.get("instruction_fingerprint_hashes")
    ):
        rules = [instruction_dedup]
    if isinstance(rules, list):
        parsed = [
            rule
            for index, item in enumerate(rules)
            if isinstance(item, dict)
            for rule in [_parse_instruction_dedup_rule_yaml(
                item,
                index=index,
                base_policy=target,
                default_policy_source=target["policy_source"],
            )]
            if rule is not None
        ]
        if parsed:
            target["rules"] = parsed


def _apply_terminal_log_policy_yaml(policy: dict[str, Any], terminal_log: dict[str, Any]) -> None:
    target = policy["terminal_log_boilerplate"]
    target["enabled"] = _as_bool(terminal_log.get("enabled"), target["enabled"])
    for key in ("min_lines", "min_repeated_lines", "max_annotations"):
        if terminal_log.get(key) is not None:
            target[key] = int(terminal_log[key])


def _apply_terminal_output_compaction_policy_yaml(
    policy: dict[str, Any],
    terminal_compaction: dict[str, Any],
    *,
    default_policy_source: str,
) -> None:
    target = policy["terminal_output_compaction"]
    action = terminal_compaction.get("action") if isinstance(terminal_compaction.get("action"), dict) else terminal_compaction
    target["enabled"] = _as_bool(terminal_compaction.get("enabled"), target["enabled"])
    target["policy_source"] = str(terminal_compaction.get("policy_source") or target.get("policy_source") or default_policy_source)
    for key in ("rule_id", "candidate_id", "action_id"):
        if terminal_compaction.get(key) is not None:
            target[key] = str(terminal_compaction[key])
    if isinstance(terminal_compaction.get("conditions"), dict):
        target["conditions"] = _sanitize_terminal_output_compaction_conditions(terminal_compaction["conditions"])
    if isinstance(terminal_compaction.get("provenance"), dict):
        target["provenance"] = _sanitize_terminal_output_compaction_provenance(terminal_compaction["provenance"])
    for key in (
        "keep_recent_turns",
        "min_block_chars",
        "head_lines",
        "tail_lines",
        "max_evidence_lines",
        "min_saved_chars",
    ):
        if action.get(key) is not None:
            target[key] = int(action[key])
    target["block_thinking"] = _as_bool(terminal_compaction.get("block_thinking"), target["block_thinking"])
    canary = terminal_compaction.get("canary") or {}
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
                target_canary[target_key] = max(0.0, min(1.0, float(canary[source_key])))
        if canary.get("salt") is not None:
            target_canary["salt"] = str(canary["salt"])
        if canary.get("canary_salt") is not None:
            target_canary["salt"] = str(canary["canary_salt"])
        if canary.get("unit") is not None:
            target_canary["unit"] = str(canary["unit"])
        if canary.get("canary_unit") is not None:
            target_canary["unit"] = str(canary["canary_unit"])
    safety = terminal_compaction.get("safety_stop") or {}
    if isinstance(safety, dict):
        target_safety = target["safety_stop"]
        target_safety["enabled"] = _as_bool(safety.get("enabled"), target_safety["enabled"])
        for key in ("min_outcome_samples", "window"):
            if safety.get(key) is not None:
                target_safety[key] = int(safety[key])
        for key in ("max_error_rate", "max_retry_rate", "max_negative_savings_rate", "max_error_rate_delta"):
            if safety.get(key) is not None:
                target_safety[key] = max(0.0, float(safety[key]))
    rules = terminal_compaction.get("rules")
    if rules is None and (
        terminal_compaction.get("conditions")
        or terminal_compaction.get("action_id")
        or terminal_compaction.get("provenance")
    ):
        rules = [terminal_compaction]
    parsed = _parse_terminal_output_compaction_rules_yaml(
        rules,
        base_policy=target,
        default_policy_source=target["policy_source"],
    )
    if parsed:
        target["rules"] = parsed


def _sanitize_terminal_output_compaction_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    aliases = {
        "phase": "workflow_phase",
        "expected_saved_tokens_bucket": "expected_saved_token_bucket",
        "expected_saved_tokens_buckets": "expected_saved_token_buckets",
    }
    allowed = {
        "source_surface",
        "app_family",
        "category",
        "workflow_phase",
        "labels",
        "requested_model",
        "model_pattern",
        "text_bucket",
        "token_bucket",
        "expected_saved_token_bucket",
        "expected_saved_token_buckets",
        "terminal_output_char_fraction_bucket",
        "terminal_output_char_fraction_buckets",
        "has_tools",
        "stream",
        "uses_thinking",
        "min_text_chars",
        "max_text_chars",
        "min_saved_tokens",
        "category_not_in",
    }
    sanitized: dict[str, Any] = {}
    for key, raw in value.items():
        key_text = aliases.get(str(key), str(key))
        if key_text not in allowed or raw is None:
            continue
        if isinstance(raw, (str, int, float, bool)):
            sanitized[key_text] = raw
        elif isinstance(raw, list):
            sanitized[key_text] = [item for item in raw if isinstance(item, (str, int, float, bool))]
    return sanitized


def _sanitize_terminal_output_compaction_provenance(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = (
        "schema",
        "issuer",
        "server_id",
        "key_id",
        "decision_hash",
        "signature",
        "algorithm",
        "verified",
        "status",
        "generated_at",
        "expires_at",
    )
    sanitized = {
        key: value.get(key)
        for key in allowed
        if isinstance(value.get(key), (str, int, float, bool))
    }
    return sanitized or None


def _overlay_terminal_output_compaction_rule(
    base: dict[str, Any],
    item: dict[str, Any],
    *,
    index: int,
    default_policy_source: str,
) -> dict[str, Any]:
    rule = copy.deepcopy(base)
    action = item.get("action") if isinstance(item.get("action"), dict) else item
    rule["rules"] = []
    rule["enabled"] = _as_bool(item.get("enabled"), rule.get("enabled", True))
    rule["policy_source"] = str(item.get("policy_source") or rule.get("policy_source") or default_policy_source)
    rule["rule_id"] = str(
        item.get("id")
        or item.get("rule_id")
        or item.get("policy_id")
        or item.get("candidate_id")
        or f"terminal-output-compaction-rule-{index + 1}"
    )
    for key in ("candidate_id", "action_id"):
        if item.get(key) is not None:
            rule[key] = str(item[key])
    if item.get("rollout_action_id") is not None and not rule.get("action_id"):
        rule["action_id"] = str(item["rollout_action_id"])
    if isinstance(item.get("conditions"), dict):
        rule["conditions"] = _sanitize_terminal_output_compaction_conditions(item["conditions"])
    if isinstance(item.get("provenance"), dict):
        rule["provenance"] = _sanitize_terminal_output_compaction_provenance(item["provenance"])
    for key in (
        "keep_recent_turns",
        "min_block_chars",
        "head_lines",
        "tail_lines",
        "max_evidence_lines",
        "min_saved_chars",
    ):
        if action.get(key) is not None:
            rule[key] = int(action[key])
    if item.get("block_thinking") is not None:
        rule["block_thinking"] = _as_bool(item.get("block_thinking"), bool(rule.get("block_thinking", True)))
    canary = item.get("canary") or item.get("rollout") or {}
    if isinstance(canary, dict):
        target_canary = rule["canary"]
        target_canary["enabled"] = _as_bool(
            canary.get("enabled", canary.get("canary_enabled")),
            target_canary["enabled"],
        )
        for source_key, target_key in (
            ("fraction", "fraction"),
            ("canary_fraction", "fraction"),
            ("rollout_fraction", "fraction"),
            ("holdout_fraction", "holdout_fraction"),
        ):
            if canary.get(source_key) is not None:
                target_canary[target_key] = max(0.0, min(1.0, float(canary[source_key])))
        if canary.get("salt") is not None:
            target_canary["salt"] = str(canary["salt"])
        if canary.get("canary_salt") is not None:
            target_canary["salt"] = str(canary["canary_salt"])
        if canary.get("unit") is not None:
            target_canary["unit"] = str(canary["unit"])
        if canary.get("canary_unit") is not None:
            target_canary["unit"] = str(canary["canary_unit"])
    safety = item.get("safety_stop") or {}
    if isinstance(safety, dict):
        target_safety = rule["safety_stop"]
        target_safety["enabled"] = _as_bool(safety.get("enabled"), target_safety["enabled"])
        for key in ("min_outcome_samples", "window"):
            if safety.get(key) is not None:
                target_safety[key] = int(safety[key])
        for key in ("max_error_rate", "max_retry_rate", "max_negative_savings_rate", "max_error_rate_delta"):
            if safety.get(key) is not None:
                target_safety[key] = max(0.0, float(safety[key]))
    return rule


def _parse_terminal_output_compaction_rules_yaml(
    value: Any,
    *,
    base_policy: dict[str, Any],
    default_policy_source: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        rules.append(
            _overlay_terminal_output_compaction_rule(
                base_policy,
                item,
                index=index,
                default_policy_source=default_policy_source,
            )
        )
    return rules


def _sanitize_anthropic_thinking_compaction_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    aliases = {"phase": "workflow_phase"}
    allowed = {
        "source_surface",
        "category",
        "workflow_phase",
        "requested_model",
        "model_pattern",
        "text_bucket",
        "token_bucket",
        "has_tools",
        "stream",
        "min_text_chars",
        "max_text_chars",
        "category_not_in",
    }
    sanitized: dict[str, Any] = {}
    for key, raw in value.items():
        key_text = aliases.get(str(key), str(key))
        if key_text not in allowed or raw is None:
            continue
        if isinstance(raw, (str, int, float, bool)):
            sanitized[key_text] = raw
        elif isinstance(raw, list):
            sanitized[key_text] = [item for item in raw if isinstance(item, (str, int, float, bool))]
    return sanitized


def _overlay_anthropic_thinking_compaction_rule(
    base: dict[str, Any],
    item: dict[str, Any],
    *,
    index: int,
    default_policy_source: str,
) -> dict[str, Any]:
    rule = copy.deepcopy(base)
    action = item.get("action") if isinstance(item.get("action"), dict) else item
    rule["rules"] = []
    rule["enabled"] = _as_bool(item.get("enabled"), rule.get("enabled", True))
    rule["policy_source"] = str(item.get("policy_source") or rule.get("policy_source") or default_policy_source)
    rule["rule_id"] = str(
        item.get("id")
        or item.get("rule_id")
        or item.get("policy_id")
        or item.get("candidate_id")
        or f"anthropic-thinking-history-compaction-rule-{index + 1}"
    )
    for key in ("candidate_id", "action_id", "replacement_notice"):
        if item.get(key) is not None:
            rule[key] = str(item[key])
    if isinstance(item.get("conditions"), dict):
        rule["conditions"] = _sanitize_anthropic_thinking_compaction_conditions(item["conditions"])
    for key in ("min_text_chars", "min_block_chars"):
        if action.get(key) is not None:
            rule[key] = int(action[key])
    if action.get("similarity_threshold") is not None:
        rule["similarity_threshold"] = max(0.0, min(1.0, float(action["similarity_threshold"])))
    if item.get("block_top_level_thinking") is not None:
        rule["block_top_level_thinking"] = _as_bool(item.get("block_top_level_thinking"), True)
    _apply_fraction_canary_yaml(rule["canary"], item.get("canary") or item.get("rollout"))
    safety = item.get("safety_stop") or {}
    if isinstance(safety, dict):
        target_safety = rule["safety_stop"]
        target_safety["enabled"] = _as_bool(safety.get("enabled"), target_safety["enabled"])
        for key in ("min_outcome_samples", "window"):
            if safety.get(key) is not None:
                target_safety[key] = int(safety[key])
        for key in (
            "max_error_rate",
            "max_retry_rate",
            "max_negative_savings_rate",
            "max_missing_usage_rate",
            "max_error_rate_delta",
        ):
            if safety.get(key) is not None:
                target_safety[key] = max(0.0, float(safety[key]))
    return rule


def _parse_anthropic_thinking_compaction_rules_yaml(
    value: Any,
    *,
    base_policy: dict[str, Any],
    default_policy_source: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        rules.append(
            _overlay_anthropic_thinking_compaction_rule(
                base_policy,
                item,
                index=index,
                default_policy_source=default_policy_source,
            )
        )
    return rules


def _apply_pattern_modules_policy_yaml(policy: dict[str, Any], pattern_modules: dict[str, Any]) -> None:
    target = policy["pattern_modules"]
    for family, raw_config in pattern_modules.items():
        if str(family).strip() not in target:
            target[str(family).strip()] = {
                "enabled": True,
                "local_crunch_enabled": False,
            }
        module_config = target[str(family).strip()]
        if isinstance(raw_config, bool):
            module_config["enabled"] = raw_config
            continue
        if not isinstance(raw_config, dict):
            continue
        module_config["enabled"] = _as_bool(raw_config.get("enabled"), module_config["enabled"])
        module_config["local_crunch_enabled"] = _as_bool(
            raw_config.get("local_crunch_enabled"),
            module_config.get("local_crunch_enabled", False),
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
            "rollout": normalize_pattern_rollout(item.get("rollout")),
        })
    return rules


def _sanitize_request_shape_canary_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    conditions: dict[str, Any] = {}
    string_keys = {
        "provider_family",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "text_bucket",
        "token_bucket",
        "cache_status",
        "routing_status",
    }
    bool_keys = {"stream", "has_tools"}
    for key in string_keys:
        if value.get(key) is not None:
            conditions[key] = public_label(value.get(key), "unknown")
    for key in bool_keys:
        if value.get(key) is not None:
            conditions[key] = _as_bool(value.get(key), False)
    return conditions


def _parse_request_shape_repeated_context_canary_rules_yaml(
    value: Any,
    *,
    default_policy_source: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        rollout = item.get("rollout") if isinstance(item.get("rollout"), dict) else {}
        policy_id = str(item.get("id") or item.get("policy_id") or f"local-repeated-context-crunch-canary-{index + 1}")
        try:
            canary_fraction = max(0.0, min(1.0, float(rollout.get("canary_fraction", rollout.get("fraction", 0.0)))))
        except (TypeError, ValueError):
            canary_fraction = 0.0
        try:
            full_rollout_fraction = max(0.0, min(1.0, float(rollout.get("full_rollout_fraction", 0.0))))
        except (TypeError, ValueError):
            full_rollout_fraction = 0.0
        full_rollout_enabled = _as_bool(rollout.get("full_rollout_enabled"), False) or full_rollout_fraction > 0.0
        try:
            holdout_fraction = max(0.0, min(1.0, float(rollout.get("holdout_fraction", 0.0))))
        except (TypeError, ValueError):
            holdout_fraction = 0.0
        if full_rollout_enabled:
            full_rollout_fraction = full_rollout_fraction or 1.0
            canary_fraction = max(canary_fraction, full_rollout_fraction)
            holdout_fraction = 0.0
        if canary_fraction + holdout_fraction > 1.0:
            holdout_fraction = max(0.0, 1.0 - canary_fraction)
        rules.append({
            "id": policy_id,
            "policy_id": policy_id,
            "enabled": _as_bool(item.get("enabled"), True),
            "policy_source": str(item.get("policy_source") or default_policy_source),
            "cohort_id": str(item.get("cohort_id") or policy_id),
            "source_evidence_schema": public_label(item.get("source_evidence_schema"), "unknown"),
            "source_evidence_schemas": _list_of_strings(item.get("source_evidence_schemas")),
            "local_only_reason": public_label(item.get("local_only_reason"), "file-backed-local-policy-no-managed-dependency"),
            "evidence_blocker_codes": [public_label(code, "unknown") for code in _list_of_strings(item.get("evidence_blocker_codes"))],
            "conditions": _sanitize_request_shape_canary_conditions(item.get("conditions")),
            "rollout": {
                "schema": "agentflow.request_shape_crunch_canary_rollout.v1",
                "canary_enabled": _as_bool(rollout.get("canary_enabled"), True),
                "full_rollout_enabled": full_rollout_enabled,
                "full_rollout_fraction": full_rollout_fraction,
                "canary_fraction": canary_fraction,
                "holdout_fraction": holdout_fraction,
                "canary_salt": str(rollout.get("canary_salt") or policy_id),
                "canary_unit": str(rollout.get("canary_unit") or "request_shape_cohort"),
            },
            "projected_saved_chars": _safe_int(item.get("projected_saved_chars")),
            "projected_saved_tokens": _safe_int(item.get("projected_saved_tokens")),
            "projected_saved_usd": float(item.get("projected_saved_usd") or 0.0),
            "safety_gates": item.get("safety_gates") if isinstance(item.get("safety_gates"), dict) else {},
            "lifecycle_metadata": item.get("lifecycle_metadata") if isinstance(item.get("lifecycle_metadata"), dict) else {},
            "rollback_metadata": item.get("rollback_metadata") if isinstance(item.get("rollback_metadata"), dict) else {},
            "staged_at": str(item.get("staged_at") or "") or None,
        })
    return rules


def _apply_request_shape_repeated_context_canaries_yaml(
    policy: dict[str, Any],
    raw_canaries: dict[str, Any],
    *,
    default_policy_source: str,
) -> None:
    target = policy["request_shape_repeated_context_canaries"]
    target["enabled"] = _as_bool(raw_canaries.get("enabled"), target["enabled"])
    if raw_canaries.get("schema") is not None:
        target["schema"] = public_label(raw_canaries.get("schema"), target["schema"])
    parsed = _parse_request_shape_repeated_context_canary_rules_yaml(
        raw_canaries.get("rules"),
        default_policy_source=default_policy_source,
    )
    if parsed:
        target["rules"] = parsed


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


def _parse_provider_scaffolding_rules_yaml(value: Any, *, default_policy_source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        pattern_hashes = _parse_pattern_hashes(
            item.get("pattern_hashes")
            or item.get("pattern_hash")
            or (item.get("conditions") or {}).get("pattern_hashes")
            or (item.get("conditions") or {}).get("pattern_hash")
        )
        action = item.get("action") or {}
        if not isinstance(action, dict):
            action = {}
        conditions = item.get("conditions") or {}
        if not isinstance(conditions, dict):
            conditions = {}
        match_any_repeated = _as_bool(item.get("match_any_repeated", conditions.get("match_any_repeated")), False)
        if not pattern_hashes and not match_any_repeated:
            continue
        normalized: dict[str, Any] = {
            "id": str(item.get("id") or item.get("rule_id") or item.get("candidate_id") or f"repeated-provider-scaffold-{index + 1}"),
            "candidate_id": item.get("candidate_id"),
            "enabled": _as_bool(item.get("enabled"), True),
            "policy_source": str(item.get("policy_source") or default_policy_source),
            "pattern_hashes": pattern_hashes,
            "match_any_repeated": match_any_repeated,
            "min_repeated_count": int(conditions.get("min_repeated_count", item.get("min_repeated_count", 2))),
            "action": {
                "type": str(action.get("type") or item.get("action_type") or "omit").strip().lower(),
                "max_replacement_chars": int(action.get("max_replacement_chars", item.get("max_replacement_chars", 360))),
            },
            "rollout": normalize_pattern_rollout(item.get("rollout")),
        }
        if normalized["action"]["type"] not in {"omit"}:
            normalized["action"]["type"] = "omit"
        for key in ("min_section_chars", "min_request_chars"):
            value_for_key = conditions.get(key, item.get(key))
            if value_for_key is not None:
                normalized[key] = int(value_for_key)
        for key in ("keep_recent_matches", "max_applications"):
            value_for_key = conditions.get(key, item.get(key))
            if value_for_key is not None:
                normalized[key] = int(value_for_key)
        for key in ("block_tool_protocol", "block_thinking"):
            value_for_key = conditions.get(key, item.get(key))
            if value_for_key is not None:
                normalized[key] = _as_bool(value_for_key, True)
        safe_conditions: dict[str, Any] = {}
        for key in ("source_surface", "app_family", "phase", "category", "requested_model", "has_tools", "uses_thinking"):
            if conditions.get(key) is not None:
                safe_conditions[key] = conditions[key]
        if safe_conditions:
            normalized["conditions"] = safe_conditions
        if item.get("description") is not None:
            normalized["description"] = str(item["description"])
        rules.append(normalized)
    return rules


def _apply_provider_scaffolding_policy_yaml(
    policy: dict[str, Any],
    provider_scaffolding: dict[str, Any],
    *,
    default_policy_source: str,
) -> None:
    target = policy["repeated_provider_scaffolding"]
    target["enabled"] = _as_bool(provider_scaffolding.get("enabled"), target["enabled"])
    for key in (
        "min_request_chars",
        "min_section_chars",
        "keep_recent_messages",
        "keep_recent_matches",
        "max_replacements",
    ):
        if provider_scaffolding.get(key) is not None:
            target[key] = int(provider_scaffolding[key])
    for key in ("block_tool_protocol", "block_thinking"):
        if provider_scaffolding.get(key) is not None:
            target[key] = _as_bool(provider_scaffolding[key], target[key])
    rules = provider_scaffolding.get("rules")
    if rules is None and (
        provider_scaffolding.get("pattern_hash")
        or provider_scaffolding.get("pattern_hashes")
        or provider_scaffolding.get("candidate_id")
    ):
        rules = [provider_scaffolding]
    parsed = _parse_provider_scaffolding_rules_yaml(rules, default_policy_source=default_policy_source)
    if parsed:
        target["rules"] = parsed


def _apply_session_memory_hints_policy_yaml(policy: dict[str, Any], session_memory_hints: dict[str, Any]) -> None:
    target = policy["session_memory_hints"]
    target["enabled"] = _as_bool(session_memory_hints.get("enabled"), target["enabled"])
    for key in ("rule_id", "crunch_profile"):
        if session_memory_hints.get(key) is not None:
            target[key] = str(session_memory_hints[key])
    for key in ("old_context_summary_canary", "block_tool_results", "block_thinking"):
        if session_memory_hints.get(key) is not None:
            target[key] = _as_bool(session_memory_hints.get(key), target[key])
    for key in ("min_call_count", "min_plateau_pairs", "min_text_chars"):
        if session_memory_hints.get(key) is not None:
            target[key] = int(session_memory_hints[key])
    for key in ("max_error_rate", "projected_savings_ratio"):
        if session_memory_hints.get(key) is not None:
            target[key] = float(session_memory_hints[key])
    if session_memory_hints.get("allowed_phases") is not None:
        raw = session_memory_hints["allowed_phases"]
        if isinstance(raw, list):
            target["allowed_phases"] = [str(item) for item in raw]
        else:
            target["allowed_phases"] = [str(raw)]


CRUNCH_POLICY, CRUNCH_POLICY_SOURCE, CRUNCH_RULES_PATH = _load_crunch_policy()
CRUNCH_RULES_LOADED_AT = utc_now()
CRUNCH_RULES_LOADED_FILE = policy_file_snapshot(CRUNCH_RULES_PATH)
ALLOWED_CRUNCH_RULES = _allowed_crunch_rule_ids(CRUNCH_POLICY)
CRUNCH_ENABLED = bool(CRUNCH_POLICY["enabled"])
CRUNCH_THRESHOLD_CHARS = int(CRUNCH_POLICY["threshold_chars"])
PROMPT_CACHE_ENABLED = bool(CRUNCH_POLICY["prompt_cache"]["enabled"])
PROMPT_CACHE_MIN_CHARS = int(CRUNCH_POLICY["prompt_cache"]["min_chars"])
ENHANCED_CRUNCH_PROVIDER_POLICY = CRUNCH_POLICY["enhanced_crunch_provider"]
OLD_CONTEXT_SUMMARY_POLICY = CRUNCH_POLICY["old_context_summarization"]
OLD_CONTEXT_SUMMARY_ENABLED = bool(OLD_CONTEXT_SUMMARY_POLICY["enabled"])
OLD_CONTEXT_SUMMARY_MODEL = str(OLD_CONTEXT_SUMMARY_POLICY["model"])
OLD_CONTEXT_SUMMARY_RULE_ID = str(OLD_CONTEXT_SUMMARY_POLICY["rule_id"])
OLD_CONTEXT_SUMMARY_CANDIDATE_ID = OLD_CONTEXT_SUMMARY_POLICY.get("candidate_id")
OLD_CONTEXT_SUMMARY_PLACEMENT = str(OLD_CONTEXT_SUMMARY_POLICY["placement"])
OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["min_request_chars"])
OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["min_summarized_chars"])
OLD_CONTEXT_SUMMARY_MAX_TURNS = int(OLD_CONTEXT_SUMMARY_POLICY["max_turns"])
OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS = int(OLD_CONTEXT_SUMMARY_POLICY["keep_recent_turns"])
OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["max_summary_chars"])
OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS = int(OLD_CONTEXT_SUMMARY_POLICY["max_source_chars"])
OLD_CONTEXT_SUMMARY_MAX_COST_USD = float(OLD_CONTEXT_SUMMARY_POLICY["max_summary_cost_usd"])
OLD_CONTEXT_SUMMARY_EXCLUDED_CATEGORIES = {str(item) for item in OLD_CONTEXT_SUMMARY_POLICY["excluded_categories"]}
OLD_CONTEXT_SUMMARY_BLOCK_TOOL_PROTOCOL = bool(OLD_CONTEXT_SUMMARY_POLICY["block_tool_protocol"])
OLD_CONTEXT_SUMMARY_BLOCK_THINKING = bool(OLD_CONTEXT_SUMMARY_POLICY["block_thinking"])
THINKING_DEDUP_POLICY = CRUNCH_POLICY["thinking_deduplication"]
THINKING_DEDUP_ENABLED = bool(THINKING_DEDUP_POLICY["enabled"])
THINKING_DEDUP_MIN_CHARS = int(THINKING_DEDUP_POLICY["min_chars"])
THINKING_DEDUP_SIMILARITY_THRESHOLD = float(THINKING_DEDUP_POLICY["similarity_threshold"])
THINKING_DEDUP_SKIP_LATEST_ASSISTANT = bool(THINKING_DEDUP_POLICY["skip_latest_assistant"])
ANTHROPIC_THINKING_COMPACTION_POLICY = CRUNCH_POLICY["anthropic_thinking_history_compaction"]
INSTRUCTION_SECTION_DEDUP_POLICY = CRUNCH_POLICY["instruction_section_deduplication"]
TERMINAL_LOG_POLICY = CRUNCH_POLICY["terminal_log_boilerplate"]
TERMINAL_LOG_ENABLED = bool(TERMINAL_LOG_POLICY["enabled"])
TERMINAL_LOG_MIN_LINES = int(TERMINAL_LOG_POLICY["min_lines"])
TERMINAL_LOG_MIN_REPEATED_LINES = int(TERMINAL_LOG_POLICY["min_repeated_lines"])
TERMINAL_LOG_MAX_ANNOTATIONS = int(TERMINAL_LOG_POLICY["max_annotations"])
TERMINAL_OUTPUT_COMPACTION_POLICY = CRUNCH_POLICY["terminal_output_compaction"]
PATTERN_MODULES_POLICY = copy.deepcopy(CRUNCH_POLICY["pattern_modules"])
PATTERN_RULES = list(CRUNCH_POLICY["pattern_rules"])
REQUEST_SHAPE_REPEATED_CONTEXT_CANARIES_POLICY = CRUNCH_POLICY["request_shape_repeated_context_canaries"]
REPEATED_PROVIDER_SCAFFOLDING_POLICY = CRUNCH_POLICY["repeated_provider_scaffolding"]
CODEX_REPEATED_SCAFFOLDING_POLICY = CRUNCH_POLICY["codex_repeated_scaffolding"]
CODEX_REPEATED_SCAFFOLDING_ENABLED = bool(CODEX_REPEATED_SCAFFOLDING_POLICY["enabled"])
CODEX_REPEATED_SCAFFOLDING_MIN_REQUEST_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["min_request_chars"])
CODEX_REPEATED_SCAFFOLDING_MIN_SECTION_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["min_section_chars"])
CODEX_REPEATED_SCAFFOLDING_KEEP_RECENT_INPUT_BLOCKS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["keep_recent_input_blocks"])
CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_MIN_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["older_block_min_chars"])
CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_HEAD_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["older_block_head_chars"])
CODEX_REPEATED_SCAFFOLDING_OLDER_BLOCK_TAIL_CHARS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["older_block_tail_chars"])
CODEX_REPEATED_SCAFFOLDING_MAX_REPLACEMENTS = int(CODEX_REPEATED_SCAFFOLDING_POLICY["max_replacements"])


def _copy_summary_policy() -> dict[str, Any]:
    return copy.deepcopy(OLD_CONTEXT_SUMMARY_POLICY)


def _summary_profile_from_managed_profile(managed_profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(managed_profile, dict):
        return None
    summary = managed_profile.get("old_context_summarization") or managed_profile.get("enhanced_crunch")
    return summary if isinstance(summary, dict) else None


def _managed_summary_requested(managed_profile: dict[str, Any] | None) -> bool:
    summary = _summary_profile_from_managed_profile(managed_profile)
    return bool(summary and _as_bool(summary.get("enabled"), False))


def _effective_summary_policy(managed_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = _copy_summary_policy()
    policy["policy_source"] = CRUNCH_POLICY_SOURCE
    summary = _summary_profile_from_managed_profile(managed_profile)
    if not isinstance(summary, dict):
        return policy

    if summary.get("enabled") is not None:
        policy["enabled"] = _as_bool(summary.get("enabled"), bool(policy.get("enabled")))
    policy["policy_source"] = str((managed_profile or {}).get("policy_source") or "managed-recommended")
    for source_key, target_key in (
        ("rule_id", "rule_id"),
        ("policy_id", "rule_id"),
        ("candidate_id", "candidate_id"),
        ("model", "model"),
        ("model_hint", "model"),
        ("placement", "placement"),
        ("max_summary_cost_usd", "max_summary_cost_usd"),
    ):
        if summary.get(source_key) is not None:
            policy[target_key] = summary[source_key]
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    for key in (
        "min_request_chars",
        "min_summarized_chars",
        "max_turns",
        "keep_recent_turns",
        "max_summary_chars",
        "max_source_chars",
    ):
        value = summary.get(key, thresholds.get(key))
        if value is not None:
            policy[key] = int(value)
    if summary.get("excluded_categories") is not None:
        raw_categories = summary.get("excluded_categories")
        policy["excluded_categories"] = raw_categories if isinstance(raw_categories, list) else [str(raw_categories)]
    for key in ("block_tool_protocol", "block_thinking"):
        if summary.get(key) is not None:
            policy[key] = _as_bool(summary.get(key), bool(policy.get(key)))
    for key in ("canary", "safety_stop"):
        if isinstance(summary.get(key), dict):
            policy[key] = copy.deepcopy(summary[key])
    return policy


def _provider_endpoint_configured(provider: dict[str, Any]) -> bool:
    return bool(str(provider.get("endpoint_url") or "").strip())


def _enhanced_provider_configured(provider: dict[str, Any] | None = None) -> bool:
    provider = provider or ENHANCED_CRUNCH_PROVIDER_POLICY
    mode = _normalize_enhanced_provider_mode(provider.get("mode"))
    if mode == "disabled":
        return False
    if mode == "local_provider_account":
        return True
    return _provider_endpoint_configured(provider)


def enhanced_crunch_provider_public_meta(managed_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    provider = copy.deepcopy(ENHANCED_CRUNCH_PROVIDER_POLICY)
    summary = _summary_profile_from_managed_profile(managed_profile)
    recommended = bool(summary)
    mode = _normalize_enhanced_provider_mode(provider.get("mode"))
    configured = _enhanced_provider_configured(provider)
    model = provider.get("model") or (summary or {}).get("model_hint") or (summary or {}).get("model")
    model_family = provider.get("model_family") or (summary or {}).get("model_family")
    state = "configured" if configured else ("fallback-not-configured" if recommended else "disabled")
    return {
        "schema": "agentflow.enhanced_crunch_provider.v1",
        "recommended": recommended,
        "configured": configured,
        "state": state,
        "mode": mode,
        "profile": str(provider.get("profile") or (managed_profile or {}).get("profile") or "default"),
        "model": str(model) if model is not None else None,
        "model_family": str(model_family) if model_family is not None else None,
        "endpoint_configured": _provider_endpoint_configured(provider),
        "endpoint_url_included": False,
        "raw_source_included": False,
        "raw_summary_included": False,
        "provider_response_included": False,
        "cache_key_included": False,
        "policy_source": str((managed_profile or {}).get("policy_source") or CRUNCH_POLICY_SOURCE),
    }


def local_enhanced_crunch_configured() -> bool:
    return _enhanced_provider_configured()


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


def _request_shape_text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _request_shape_routing_status(routing_meta: dict[str, Any] | None) -> str | None:
    if not isinstance(routing_meta, dict):
        return None
    for key in ("routing_status", "status"):
        value = routing_meta.get(key)
        if isinstance(value, str) and value.strip():
            if value == "applied":
                return "routed"
            if value == "skipped":
                return "passthrough"
            return public_label(value, "unknown")
    requested = routing_meta.get("requested_model")
    routed = routing_meta.get("routed_model")
    if requested is not None and routed is not None:
        return "passthrough" if str(requested) == str(routed) else "routed"
    return None


def _request_shape_repeated_context_features(
    body: dict[str, Any],
    *,
    routing_meta: dict[str, Any] | None,
    provider: str | None,
    source_surface: str | None,
    endpoint: str | None,
    category: str,
) -> dict[str, Any]:
    routing = routing_meta if isinstance(routing_meta, dict) else {}
    text_chars = _safe_int(routing.get("text_chars"), len(_extract_text_for_category(body)))
    has_tools = routing.get("has_tools")
    if not isinstance(has_tools, bool):
        has_tools = _body_has_tools(body)
    stream = _as_bool(body.get("stream"), False)
    requested = routing.get("requested_model") or body.get("model")
    routed = routing.get("routed_model") or body.get("model")
    features = {
        "provider_family": public_label(provider or routing.get("provider"), "unknown"),
        "source_surface": public_label(source_surface or routing.get("source_surface"), "unknown"),
        "endpoint": public_label(endpoint or routing.get("endpoint"), "unknown"),
        "category": public_label(routing.get("category") or category, "unknown"),
        "workflow_phase": public_label(routing.get("workflow_phase") or routing.get("phase"), "unknown"),
        "stream": stream,
        "has_tools": bool(has_tools),
        "text_bucket": _request_shape_text_bucket(text_chars),
        "token_bucket": public_label(routing.get("token_bucket"), "unknown"),
        "cache_status": public_label(routing.get("cache_status") or ("skipped" if stream or has_tools else None), "unknown"),
        "routing_status": _request_shape_routing_status({**routing, "requested_model": requested, "routed_model": routed}),
        "request_fingerprint": "sha256:" + hashlib.sha256(stable_json(body).encode("utf-8")).hexdigest(),
    }
    return {key: value for key, value in features.items() if value not in (None, "unknown")}


def _request_shape_repeated_context_canary_base_meta(enabled: bool, reason: str) -> dict[str, Any]:
    rules = REQUEST_SHAPE_REPEATED_CONTEXT_CANARIES_POLICY.get("rules")
    return {
        "schema": "agentflow.request_shape_repeated_context_crunch_canary_runtime.v1",
        "enabled": bool(enabled),
        "status": "skipped",
        "reason": reason,
        "configured_count": len(rules) if isinstance(rules, list) else 0,
        "evaluated_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "safety_stopped_count": 0,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "rules": [],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
        },
    }


def _evaluate_request_shape_repeated_context_canaries(
    body: dict[str, Any],
    *,
    routing_meta: dict[str, Any] | None,
    provider: str | None,
    source_surface: str | None,
    endpoint: str | None,
    category: str,
) -> dict[str, Any]:
    policy = REQUEST_SHAPE_REPEATED_CONTEXT_CANARIES_POLICY
    rules = policy.get("rules") if isinstance(policy.get("rules"), list) else []
    enabled = _as_bool(policy.get("enabled"), False)
    if not enabled:
        return _request_shape_repeated_context_canary_base_meta(False, "disabled")
    if not rules:
        return _request_shape_repeated_context_canary_base_meta(True, "no-rules")

    features = _request_shape_repeated_context_features(
        body,
        routing_meta=routing_meta,
        provider=provider,
        source_surface=source_surface,
        endpoint=endpoint,
        category=category,
    )
    meta = _request_shape_repeated_context_canary_base_meta(True, "no-matching-rule")
    selected: dict[str, Any] | None = None
    for rule in rules:
        if not isinstance(rule, dict) or not _as_bool(rule.get("enabled"), True):
            continue
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        full_rollout_enabled = _as_bool(rollout.get("full_rollout_enabled"), False)
        if not _as_bool(rollout.get("canary_enabled"), True) and not full_rollout_enabled:
            continue
        rollout_fraction = rollout.get("full_rollout_fraction") if full_rollout_enabled else rollout.get("canary_fraction", 0.0)
        holdout_fraction = 0.0 if full_rollout_enabled else rollout.get("holdout_fraction", 0.0)
        action = {
            "policy_id": rule.get("policy_id") or rule.get("id"),
            "cohort_id": rule.get("cohort_id"),
            "conditions": rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {},
            "rollout_fraction": rollout_fraction,
            "holdout_fraction": holdout_fraction,
        }
        lifecycle = request_shape_crunch_canary_lifecycle(action, features)
        status = str(lifecycle.get("status") or "unknown")
        rule_meta = {
            "id": public_label(rule.get("id"), "unknown"),
            "policy_id": public_label(rule.get("policy_id"), "unknown"),
            "cohort_id": public_label(rule.get("cohort_id"), "unknown"),
            "status": public_label(status, "unknown"),
            "cohort": public_label(lifecycle.get("cohort"), "unknown"),
            "reason": public_label(lifecycle.get("reason"), "unknown"),
            "policy_source": public_label(rule.get("policy_source"), "unknown"),
            "rollout_mode": "full-rollout" if full_rollout_enabled else "canary",
            "rollout_fraction": lifecycle.get("rollout_fraction"),
            "holdout_fraction": lifecycle.get("holdout_fraction"),
            "staged_at": rule.get("staged_at"),
            "source_evidence_schema": public_label(rule.get("source_evidence_schema"), "unknown"),
            "metadata_only": True,
        }
        if lifecycle.get("mismatched_conditions"):
            rule_meta["mismatched_conditions"] = lifecycle["mismatched_conditions"]
        meta["rules"].append(rule_meta)
        meta["evaluated_count"] += 1
        if status == "applied":
            meta["applied_count"] += 1
        elif status == "holdout":
            meta["holdout_count"] += 1
        elif status in {"safety-stopped", "safety_stop"}:
            meta["safety_stopped_count"] += 1
        else:
            meta["skipped_count"] += 1
        if not lifecycle.get("mismatched_conditions") and selected is None:
            selected = {
                **lifecycle,
                "policy_source": rule_meta["policy_source"],
                "source_evidence_schema": public_label(rule.get("source_evidence_schema"), "unknown"),
                "source_evidence_schemas": [
                    public_label(schema, "unknown")
                    for schema in rule.get("source_evidence_schemas") or []
                    if public_label(schema, "unknown") != "unknown"
                ],
                "staged_at": rule.get("staged_at"),
                "projected_saved_chars": _safe_int(rule.get("projected_saved_chars")),
                "projected_saved_tokens": _safe_int(rule.get("projected_saved_tokens")),
                "projected_saved_usd": float(rule.get("projected_saved_usd") or 0.0),
                "rollback_metadata_present": bool(rule.get("rollback_metadata")),
                "rollout_mode": "full-rollout" if full_rollout_enabled else "canary",
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
            }

    if selected is not None:
        meta["status"] = public_label(selected.get("status"), "unknown")
        meta["reason"] = public_label(selected.get("reason"), "unknown")
        meta["selected_lifecycle"] = selected
    return meta


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


def _text_bucket(chars: int) -> str:
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _token_bucket(tokens: int) -> str:
    if tokens < 1_000:
        return "lt_1k_tokens"
    if tokens < 4_000:
        return "1k_4k_tokens"
    if tokens < 16_000:
        return "4k_16k_tokens"
    if tokens < 64_000:
        return "16k_64k_tokens"
    return "gte_64k_tokens"


_ANTHROPIC_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def _thinking_size_bucket(chars: int) -> str:
    if chars <= 0:
        return "0_chars"
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _thinking_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2_5"
    if count <= 20:
        return "6_20"
    return "gt_20"


def _assistant_age_bucket(age: int) -> str:
    if age <= 0:
        return "latest_assistant"
    if age == 1:
        return "previous_assistant"
    if age <= 5:
        return "assistant_age_2_5"
    return "assistant_age_gt_5"


def _thinking_top_level_active(body: dict[str, Any] | None) -> bool:
    if not isinstance(body, dict):
        return False
    thinking = body.get("thinking")
    if not thinking:
        return False
    if isinstance(thinking, dict) and str(thinking.get("type") or "").strip().lower() == "disabled":
        return False
    return True


def _thinking_public_breakdown(counter: dict[str, int], *, label: str) -> list[dict[str, Any]]:
    return [
        {label: key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _thinking_block_text(block: dict[str, Any]) -> str:
    for key in ("thinking", "text", "data"):
        value = block.get(key)
        if isinstance(value, str):
            return value
    return ""


def _thinking_assistant_text_available(content: list[Any]) -> bool:
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"].strip():
            return True
        if isinstance(block, str) and block.strip():
            return True
    return False


def _thinking_tool_use_ids(content: list[Any]) -> set[str]:
    ids: set[str] = set()
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_id = str(block.get("id") or "")
        if tool_id:
            ids.add(tool_id)
    return ids


def _thinking_tool_result_ids(content: list[Any]) -> set[str]:
    ids: set[str] = set()
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tool_id = str(block.get("tool_use_id") or "")
        if tool_id:
            ids.add(tool_id)
    return ids


def build_anthropic_thinking_history_metadata(
    body: dict[str, Any] | None,
    *,
    provider: str | None = None,
    source_surface: str | None = None,
    endpoint: str | None = None,
    category: str | None = None,
    policy_source: str | None = None,
    rule_path: str | None = None,
) -> dict[str, Any]:
    """Build bounded local diagnostics for Anthropic thinking-history compaction planning."""
    meta: dict[str, Any] = {
        "schema": "agentflow.anthropic_thinking_history_metadata.v1",
        "status": "blocked",
        "reason": "not-evaluated",
        "provider": provider,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "category": category,
        "policy_source": str(policy_source or CRUNCH_POLICY_SOURCE),
        "rule_path": str(rule_path or CRUNCH_RULES_PATH),
        "policy": {
            "enabled": THINKING_DEDUP_ENABLED,
            "min_chars": THINKING_DEDUP_MIN_CHARS,
            "similarity_threshold": THINKING_DEDUP_SIMILARITY_THRESHOLD,
            "skip_latest_assistant": THINKING_DEDUP_SKIP_LATEST_ASSISTANT,
        },
        "privacy_mode": "metadata-only",
        "body_available": isinstance(body, dict),
        "raw_thinking_text_included": False,
        "raw_prompt_text_included": False,
        "raw_messages_included": False,
        "raw_tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "local_thinking_block_fingerprints_included": False,
        "top_level_thinking_active": False,
        "thinking_block_count": 0,
        "redacted_thinking_block_count": 0,
        "thinking_history_chars": 0,
        "thinking_history_size_bucket": "0_chars",
        "thinking_block_count_bucket": "0",
        "local_thinking_block_fingerprints": [],
        "unique_local_thinking_block_fingerprint_count": 0,
        "exact_duplicate_thinking_block_count": 0,
        "near_duplicate_thinking_block_count": 0,
        "thinking_signal_kind": "none",
        "history_block_absence_reason": None,
        "route_crunch_mismatch_explained": False,
        "diagnosis": {
            "schema": "agentflow.anthropic_thinking_history_diagnosis.v1",
            "status": "not-evaluated",
            "reason": "not-evaluated",
            "thinking_signal_kind": "none",
            "route_crunch_mismatch_explained": False,
            "recommended_strategy": "inspect-history-blocks",
            "old_context_summarization_fallback": {
                "recommended": False,
                "reason": "not-needed",
                "policy_enabled": bool(OLD_CONTEXT_SUMMARY_ENABLED),
                "rule_id": str(OLD_CONTEXT_SUMMARY_RULE_ID),
            },
        },
        "assistant_message_count": 0,
        "assistant_message_with_thinking_count": 0,
        "missing_assistant_text_fallback_count": 0,
        "thinking_tool_use_message_count": 0,
        "adjacent_tool_use_dependency_count": 0,
        "unsupported_content_block_shape_count": 0,
        "block_size_bucket_breakdown": [],
        "assistant_age_bucket_breakdown": [],
        "blockers": [],
    }

    blockers: set[str] = set()
    provider_l = str(provider or "").lower()
    source_l = str(source_surface or "").lower()
    endpoint_l = str(endpoint or "").lower()
    if provider_l and provider_l != "anthropic":
        blockers.add("non-anthropic-provider")
    if source_l and source_l != "anthropic_messages":
        blockers.add("non-anthropic-source-surface")
    if endpoint_l and endpoint_l != "messages":
        blockers.add("non-anthropic-messages-endpoint")
    if not isinstance(body, dict):
        blockers.add("raw-body-unavailable")
        meta["reason"] = "raw-body-unavailable"
        meta["blockers"] = sorted(blockers)
        return meta

    meta["privacy_mode"] = "local-body-derived-metadata"
    top_level_active = _thinking_top_level_active(body)
    meta["top_level_thinking_active"] = top_level_active
    if top_level_active:
        blockers.add("active-top-level-thinking-request")

    messages = body.get("messages")
    if not isinstance(messages, list):
        blockers.add("unsupported-content-block-shape")
        meta["unsupported_content_block_shape_count"] = 1
        meta["reason"] = "unsupported-content-block-shape"
        meta["blockers"] = sorted(blockers)
        return meta

    assistant_indexes = [idx for idx, msg in enumerate(messages) if isinstance(msg, dict) and msg.get("role") == "assistant"]
    assistant_age_by_index = {
        msg_idx: len(assistant_indexes) - 1 - assistant_order
        for assistant_order, msg_idx in enumerate(assistant_indexes)
    }
    meta["assistant_message_count"] = len(assistant_indexes)

    size_buckets: dict[str, int] = {}
    age_buckets: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    seen_shingles: list[frozenset] = []
    local_hashes: list[str] = []
    previous_thinking_tool_use_ids: set[str] = set()

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            previous_thinking_tool_use_ids = set()
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and previous_thinking_tool_use_ids:
            if isinstance(content, list):
                if previous_thinking_tool_use_ids & _thinking_tool_result_ids(content):
                    meta["adjacent_tool_use_dependency_count"] += 1
                    blockers.add("adjacent-tool-use-dependency")
            else:
                blockers.add("unsupported-content-block-shape")
                meta["unsupported_content_block_shape_count"] += 1
            previous_thinking_tool_use_ids = set()
            continue
        previous_thinking_tool_use_ids = set()
        if role != "assistant":
            continue
        if not isinstance(content, list):
            blockers.add("unsupported-content-block-shape")
            meta["unsupported_content_block_shape_count"] += 1
            continue

        thinking_blocks = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") in _ANTHROPIC_THINKING_BLOCK_TYPES
        ]
        if not thinking_blocks:
            continue
        meta["assistant_message_with_thinking_count"] += 1
        if not _thinking_assistant_text_available(content):
            meta["missing_assistant_text_fallback_count"] += 1
            blockers.add("missing-assistant-text-fallback")
        tool_use_ids = _thinking_tool_use_ids(content)
        if tool_use_ids:
            meta["thinking_tool_use_message_count"] += 1
            previous_thinking_tool_use_ids = tool_use_ids

        age_bucket = _assistant_age_bucket(assistant_age_by_index.get(msg_idx, 0))
        for block in thinking_blocks:
            block_type = str(block.get("type") or "")
            if block_type == "redacted_thinking":
                meta["redacted_thinking_block_count"] += 1
                blockers.add("redacted-thinking-block")
            text = _thinking_block_text(block)
            if block_type == "thinking" and not text:
                blockers.add("unsupported-content-block-shape")
                meta["unsupported_content_block_shape_count"] += 1
            chars = len(text)
            meta["thinking_block_count"] += 1
            meta["thinking_history_chars"] += chars
            _bump_counter(size_buckets, _thinking_size_bucket(chars))
            _bump_counter(age_buckets, age_bucket)
            if not text:
                continue
            block_hash = "sha256:" + sha256_text(normalize_text(text))
            local_hashes.append(block_hash)
            seen_hashes[block_hash] = seen_hashes.get(block_hash, 0) + 1
            if seen_hashes[block_hash] > 1:
                meta["exact_duplicate_thinking_block_count"] += 1
                continue
            if chars >= THINKING_DEDUP_MIN_CHARS:
                shingles = _shingles(text)
                if any(_jaccard(shingles, older) >= THINKING_DEDUP_SIMILARITY_THRESHOLD for older in seen_shingles):
                    meta["near_duplicate_thinking_block_count"] += 1
                seen_shingles.append(shingles)

    meta["thinking_history_size_bucket"] = _thinking_size_bucket(int(meta["thinking_history_chars"]))
    meta["thinking_block_count_bucket"] = _thinking_count_bucket(int(meta["thinking_block_count"]))
    meta["local_thinking_block_fingerprints"] = sorted(set(local_hashes))[:16]
    meta["unique_local_thinking_block_fingerprint_count"] = len(set(local_hashes))
    meta["local_thinking_block_fingerprints_included"] = bool(meta["local_thinking_block_fingerprints"])
    meta["block_size_bucket_breakdown"] = _thinking_public_breakdown(size_buckets, label="bucket")
    meta["assistant_age_bucket_breakdown"] = _thinking_public_breakdown(age_buckets, label="bucket")

    thinking_block_count = int(meta["thinking_block_count"])
    if thinking_block_count > 0:
        meta["thinking_signal_kind"] = "top-level-and-history-blocks" if top_level_active else "history-blocks"
        meta["diagnosis"].update({
            "status": "history-blocks-present",
            "reason": "assistant-thinking-history-blocks-detected",
            "thinking_signal_kind": meta["thinking_signal_kind"],
            "recommended_strategy": "anthropic_thinking_history_compaction",
        })
    elif top_level_active:
        meta["thinking_signal_kind"] = "top-level-only"
        meta["history_block_absence_reason"] = "top-level-thinking-parameter-without-message-history-blocks"
        meta["route_crunch_mismatch_explained"] = True
        meta["diagnosis"].update({
            "status": "diagnosed",
            "reason": "top-level-thinking-parameter-without-message-history-blocks",
            "thinking_signal_kind": "top-level-only",
            "route_crunch_mismatch_explained": True,
            "recommended_strategy": "old_context_summarization",
            "old_context_summarization_fallback": {
                "recommended": True,
                "reason": "no-assistant-thinking-history-blocks-available-to-compact",
                "policy_enabled": bool(OLD_CONTEXT_SUMMARY_ENABLED),
                "rule_id": str(OLD_CONTEXT_SUMMARY_RULE_ID),
            },
        })
    else:
        meta["diagnosis"].update({
            "status": "no-thinking-signal",
            "reason": "no-top-level-or-history-thinking-signal",
            "thinking_signal_kind": "none",
            "recommended_strategy": "none",
        })

    if thinking_block_count <= 0:
        blockers.add("no-thinking-history-blocks")
    if blockers:
        meta["status"] = "blocked"
        if top_level_active and thinking_block_count <= 0:
            meta["reason"] = "top-level-thinking-without-history-blocks"
        else:
            meta["reason"] = sorted(blockers)[0]
    else:
        meta["status"] = "ready"
        meta["reason"] = "ready-for-thinking-compaction-planning"
        blockers.add("ready-for-thinking-compaction-planning")
    meta["blockers"] = sorted(blockers)
    return meta


def _pattern_canary_features(
    *,
    body: dict[str, Any],
    category: str,
    pattern_hash: str,
    before_chars: int,
) -> dict[str, Any]:
    return {
        "source_surface": "provider_request",
        "app_family": "unknown",
        "category": category,
        "workflow_phase": category,
        "text_bucket": _text_bucket(before_chars),
        "token_bucket": _token_bucket(max(1, before_chars // TOKEN_CHARS)),
        "requested_model": body.get("model"),
        "candidate_target_model": body.get("model"),
        "pattern_hashes": [pattern_hash],
    }


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


def _apply_pattern_rules(body: dict[str, Any], *, store_obj: Any | None = None) -> tuple[int, dict[str, Any]]:
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
            "rollout": pattern_rollout_public_meta(rule.get("rollout")),
            "matched_hashes": [],
            "applied_count": 0,
            "holdout_count": 0,
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
            if pattern_hash not in rule_meta["matched_hashes"]:
                rule_meta["matched_hashes"].append(pattern_hash)
            canary = pattern_canary_decision(
                rollout=rule.get("rollout"),
                rule_id=rule_id,
                candidate_id=rule.get("candidate_id"),
                pattern_hashes=[pattern_hash],
                features=_pattern_canary_features(
                    body=body,
                    category=category,
                    pattern_hash=pattern_hash,
                    before_chars=before_chars,
                ),
            )
            if canary.get("enabled"):
                rule_meta["canary"] = canary
            if canary.get("enabled") and not canary.get("selected", True):
                holdout_count = len(occurrences)
                rule_meta["holdout_count"] += holdout_count
                _pattern_rule_skip(skip_counts, rule_id, "canary_holdout", holdout_count)
                rule_meta["skip_reasons"].append({
                    "reason": "canary_holdout",
                    "pattern_hash": pattern_hash,
                    "count": holdout_count,
                    "canary": canary,
                })
                continue
            safety_stop = evaluate_pattern_canary_safety_stop(
                store_obj=store_obj,
                policy_section="crunch",
                rule_id=rule_id,
                candidate_id=rule.get("candidate_id"),
                pattern_hash=pattern_hash,
                rollout=rule.get("rollout"),
            )
            if safety_stop:
                holdout_count = len(occurrences)
                rule_meta["status"] = "bypass"
                rule_meta["reason"] = LOCAL_CANARY_SAFETY_STOP_REASON
                rule_meta["safety_stop"] = safety_stop
                rule_meta["holdout_count"] += holdout_count
                _pattern_rule_skip(skip_counts, rule_id, LOCAL_CANARY_SAFETY_STOP_REASON, holdout_count)
                rule_meta["skip_reasons"].append({
                    "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
                    "pattern_hash": pattern_hash,
                    "count": holdout_count,
                    "safety_stop": safety_stop,
                })
                log_pattern_canary_safety_stop(safety_stop)
                continue
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
        meta["rules"].append(rule_meta)

    meta["applied_count"] = sum(int(rule.get("applied_count") or 0) for rule in meta["rules"])
    meta["holdout_count"] = sum(int(rule.get("holdout_count") or 0) for rule in meta["rules"])
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


def _copy_repeated_provider_scaffolding_policy() -> dict[str, Any]:
    return copy.deepcopy(REPEATED_PROVIDER_SCAFFOLDING_POLICY)


def _provider_scaffolding_profile_from_managed_profile(managed_profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(managed_profile, dict):
        return None
    profile = managed_profile.get("repeated_provider_scaffolding")
    return profile if isinstance(profile, dict) else None


def _effective_repeated_provider_scaffolding_policy(
    managed_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = _copy_repeated_provider_scaffolding_policy()
    policy["policy_source"] = str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE)
    profile = _provider_scaffolding_profile_from_managed_profile(managed_profile)
    if not isinstance(profile, dict):
        return policy

    policy["policy_source"] = str((managed_profile or {}).get("policy_source") or profile.get("policy_source") or "managed-recommended")
    if profile.get("enabled") is not None:
        policy["enabled"] = _as_bool(profile.get("enabled"), bool(policy.get("enabled")))
    for key in (
        "min_request_chars",
        "min_section_chars",
        "keep_recent_messages",
        "keep_recent_matches",
        "max_replacements",
    ):
        if profile.get(key) is not None:
            policy[key] = int(profile[key])
    for key in ("block_tool_protocol", "block_thinking"):
        if profile.get(key) is not None:
            policy[key] = _as_bool(profile.get(key), bool(policy.get(key)))
    rules = profile.get("rules")
    if rules is None and (profile.get("pattern_hash") or profile.get("pattern_hashes") or profile.get("candidate_id")):
        rules = [profile]
    parsed = _parse_provider_scaffolding_rules_yaml(rules, default_policy_source=policy["policy_source"])
    if parsed:
        policy["rules"] = parsed
    return policy


def _provider_scaffolding_meta(policy: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "agentflow.repeated_provider_scaffolding.v1",
        "enabled": bool(policy.get("enabled")),
        "status": status,
        "reason": reason,
        "changed": False,
        "policy_source": str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE),
        "rule_path": str(policy.get("overlay_rule_path") or CRUNCH_RULES_PATH),
        "base_rule_path": CRUNCH_RULES_PATH,
        "configured_rule_count": len(policy.get("rules") or []),
        "rules": [],
        "skip_reasons": [],
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "raw_text_included": False,
        "raw_hashes_included": False,
        "system_instructions_preserved": True,
        "developer_instructions_preserved": True,
        "tool_protocol_preserved": True,
        "thinking_preserved": True,
        "newest_task_tail_preserved": True,
    }


def _message_role(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("role") or item.get("type") or "").strip().lower()
    return ""


def _content_has_tool_protocol(content: Any) -> bool:
    if isinstance(content, dict):
        block_type = str(content.get("type") or "").strip().lower()
        if block_type in {
            "tool_use",
            "tool_result",
            "function_call",
            "function_call_output",
            "web_search_call",
            "computer_call",
            "reasoning",
        }:
            return True
        if "tool_calls" in content or "tool_call_id" in content:
            return True
        return any(_content_has_tool_protocol(value) for value in content.values())
    if isinstance(content, list):
        return any(_content_has_tool_protocol(item) for item in content)
    return False


def _content_has_thinking(content: Any) -> bool:
    if isinstance(content, dict):
        block_type = str(content.get("type") or "").strip().lower()
        if block_type in {"thinking", "reasoning"} or "thinking" in content:
            return True
        return any(_content_has_thinking(value) for value in content.values())
    if isinstance(content, list):
        return any(_content_has_thinking(item) for item in content)
    return False


def _body_uses_thinking(body: dict[str, Any]) -> bool:
    if isinstance(body.get("thinking"), dict) and body.get("thinking"):
        return True
    messages = body.get("messages")
    if isinstance(messages, list):
        return any(isinstance(msg, dict) and _content_has_thinking(msg.get("content")) for msg in messages)
    input_value = body.get("input")
    if isinstance(input_value, list):
        return any(isinstance(item, dict) and _content_has_thinking(item.get("content")) for item in input_value)
    return False


INSTRUCTION_DEDUP_MIN_SECTION_CHARS = 80
_INSTRUCTION_LONG_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_INSTRUCTION_NUMBER_RE = re.compile(r"\b\d+\b")
_INSTRUCTION_PATH_RE = re.compile(r"(?:^|\s)(?:/|\.{1,2}/|[A-Za-z]:\\)[^\s]{2,}")
_INSTRUCTION_SPACE_RE = re.compile(r"\s+")


def _normalize_instruction_text(value: str) -> str:
    text = value.strip().lower()
    text = _INSTRUCTION_PATH_RE.sub(" <path> ", text)
    text = _INSTRUCTION_LONG_HEX_RE.sub("<hex>", text)
    text = _INSTRUCTION_NUMBER_RE.sub("<n>", text)
    return _INSTRUCTION_SPACE_RE.sub(" ", text).strip()


def _instruction_hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _instruction_source_allowed(value: str, allowed: list[Any]) -> bool:
    if not allowed:
        return True
    return value in {str(item) for item in allowed}


def _instruction_rule_matches(basis: dict[str, Any], section_fingerprint: str, rule: dict[str, Any]) -> bool:
    if not _as_bool(rule.get("enabled"), True):
        return False
    fingerprints = {str(item) for item in rule.get("instruction_section_fingerprints") or []}
    if fingerprints and section_fingerprint not in fingerprints:
        return False
    if not _instruction_source_allowed(str(basis.get("source_surface") or ""), rule.get("source_surfaces") or []):
        return False
    if not _instruction_source_allowed(str(basis.get("category") or ""), rule.get("categories") or []):
        return False
    if not _instruction_source_allowed(str(basis.get("workflow_phase") or ""), rule.get("workflow_phases") or []):
        return False
    return True


def _instruction_base_rule(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(policy.get("rule_id") or "instruction-section-dedup-policy"),
        "candidate_id": policy.get("candidate_id"),
        "enabled": _as_bool(policy.get("enabled"), False),
        "policy_source": str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE),
        "instruction_section_fingerprints": [],
        "source_surfaces": [str(item) for item in policy.get("source_surfaces") or []],
        "categories": [str(item) for item in policy.get("categories") or []],
        "workflow_phases": [str(item) for item in policy.get("workflow_phases") or []],
        "min_section_chars": int(policy.get("min_section_chars") or 700),
        "min_repeated_count": int(policy.get("min_repeated_count") or 2),
        "keep_recent_sections": int(policy.get("keep_recent_sections") or 1),
        "replacement_notice": str(policy.get("replacement_notice") or "[repeated instruction section omitted by AgentFlow]"),
        "max_replacements": int(policy.get("max_replacements") or 0),
        "canary": copy.deepcopy(policy.get("canary") if isinstance(policy.get("canary"), dict) else {}),
        "safety_stop": copy.deepcopy(policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {}),
    }


def _select_instruction_dedup_rule(policy: dict[str, Any], basis: dict[str, Any], section_fingerprint: str) -> dict[str, Any] | None:
    for rule in policy.get("rules") or []:
        if isinstance(rule, dict) and _instruction_rule_matches(basis, section_fingerprint, rule):
            return copy.deepcopy(rule)
    base = _instruction_base_rule(policy)
    return base if _instruction_rule_matches(basis, section_fingerprint, base) else None


def _instruction_dedup_cohort(rule: dict[str, Any], section_fingerprint: str, basis: dict[str, Any]) -> dict[str, Any]:
    raw_canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    enabled = _as_bool(raw_canary.get("enabled"), True)
    validation_errors = _instruction_canary_validation_errors(raw_canary)
    try:
        fraction = max(0.0, min(1.0, float(raw_canary.get("fraction", 0.0) or 0.0)))
    except (TypeError, ValueError):
        fraction = 0.0
    holdout_raw = raw_canary.get("holdout_fraction")
    try:
        holdout_fraction = None if holdout_raw is None else max(0.0, min(1.0, float(holdout_raw)))
    except (TypeError, ValueError):
        holdout_fraction = None
    holdout = holdout_fraction if holdout_fraction is not None else 0.0
    if fraction + holdout > 1.0:
        fraction = max(0.0, 1.0 - holdout)
    cohort_basis = {
        "salt": str(raw_canary.get("salt") or ""),
        "rule_id": rule.get("id"),
        "unit": raw_canary.get("unit") or "instruction_section_fingerprint",
        "fingerprint": section_fingerprint,
        "surface": basis.get("source_surface"),
        "category": basis.get("category"),
        "phase": basis.get("workflow_phase"),
    }
    digest = hashlib.sha256(json.dumps(cohort_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if validation_errors:
        cohort = "invalid"
        selected = False
        is_holdout = False
    elif not enabled:
        cohort = "disabled"
        selected = False
        is_holdout = False
    elif score < holdout:
        cohort = "holdout"
        selected = False
        is_holdout = True
    elif score < holdout + fraction:
        cohort = "canary"
        selected = True
        is_holdout = False
    else:
        cohort = "not_selected"
        selected = False
        is_holdout = False
    return {
        "enabled": enabled,
        "cohort": cohort,
        "selected": selected,
        "holdout": is_holdout,
        "canary_fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "holdout_configured": holdout_fraction is not None,
        "cohort_key_hash": digest[:16],
        "cohort_score": round(score, 12),
        "cohort_basis": "public-metadata-plus-hidden-instruction-fingerprint",
        "salt_included": False,
        "fingerprint_included": False,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
    }


def _instruction_dedup_coordinator_compatibility(row_meta: dict[str, Any]) -> dict[str, Any]:
    decision = row_meta.get("optimization_coordinator")
    if not isinstance(decision, dict):
        return {"status": "unknown", "compatible": True, "reason_codes": [], "selected_family": None}
    selected_raw = str(decision.get("selected_family") or decision.get("selected_action_family") or "none")
    selected = public_label(selected_raw, "unknown")
    reasons = [public_label(item, "sanitized-reason") for item in decision.get("reason_codes") or [] if str(item)]
    suppressed = decision.get("suppressed_families") if isinstance(decision.get("suppressed_families"), list) else []
    for item in suppressed:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "")
        if family in {"pattern_crunch", "prompt_role", "instruction_section_deduplication"} or "prompt_role" in family:
            reasons.extend(public_label(reason, "sanitized-reason") for reason in item.get("reason_codes") or [] if str(reason))
    if selected_raw not in {"none", "pattern_crunch", "prompt_role", "instruction_section_deduplication"}:
        reasons.append("conflicts-with-coordinator-selection")
        return {"status": "conflict", "compatible": False, "reason_codes": sorted(set(reasons)), "selected_family": selected}
    if "coordinator-holdout" in reasons or "coordinator-canary-not-selected" in reasons:
        return {"status": "blocked", "compatible": False, "reason_codes": sorted(set(reasons)), "selected_family": selected}
    return {"status": "compatible", "compatible": True, "reason_codes": sorted(set(reasons)), "selected_family": selected}


def _instruction_dedup_provider_and_surface(
    body: dict[str, Any],
    *,
    provider: str | None = None,
    source_surface: str | None = None,
    endpoint: str | None = None,
) -> tuple[str, str, str]:
    provider_text = str(provider or "").strip().lower()
    surface_text = str(source_surface or "").strip()
    endpoint_text = str(endpoint or "").strip()
    model = str(body.get("model") or "").lower()
    if surface_text == "codex_turn" or "codex" in model:
        return "openai", "codex_turn", endpoint_text or "responses"
    if provider_text in {"openai", "anthropic"}:
        inferred_surface = surface_text
        if not inferred_surface:
            inferred_surface = "anthropic_messages" if provider_text == "anthropic" else "openai_responses"
        inferred_endpoint = endpoint_text or ("messages" if provider_text == "anthropic" else "responses")
        return provider_text, inferred_surface, inferred_endpoint
    if "system" in body and "instructions" not in body:
        return "anthropic", surface_text or "anthropic_messages", endpoint_text or "messages"
    return "openai", surface_text or "openai_responses", endpoint_text or "responses"


def _instruction_dedup_text_from_content(content: Any) -> list[tuple[Any, Any, str]]:
    entries: list[tuple[Any, Any, str]] = []
    if isinstance(content, str):
        return []
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, str):
                entries.append((content, index, block))
            elif isinstance(block, dict):
                block_type = str(block.get("type") or "").strip().lower()
                if block_type in {"text", "input_text"} and isinstance(block.get("text"), str):
                    entries.append((block, "text", block["text"]))
        return entries
    return entries


def _instruction_dedup_entries(body: dict[str, Any], *, provider: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def add(container: Any, key: Any, text: str, source_field: str, order: int) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        normalized = _normalize_instruction_text(text)
        if len(normalized) < INSTRUCTION_DEDUP_MIN_SECTION_CHARS:
            return
        entries.append({
            "container": container,
            "key": key,
            "text": text,
            "chars": len(text),
            "normalized": normalized,
            "fingerprint": _instruction_hash_text(normalized),
            "source_field": source_field,
            "order": order,
        })

    order = 0
    provider_l = provider.lower()
    if provider_l == "anthropic":
        system = body.get("system")
        if isinstance(system, str):
            add(body, "system", system, "anthropic.system", order)
            order += 1
        elif isinstance(system, list):
            for index, block in enumerate(system):
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    add(block, "text", block["text"], f"anthropic.system[{index}].text", order)
                    order += 1
        messages = body.get("messages")
        if isinstance(messages, list):
            for msg_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").strip().lower()
                if role not in {"system", "developer"}:
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    add(message, "content", content, "anthropic.messages.system_or_developer", order)
                    order += 1
                for container, key, text in _instruction_dedup_text_from_content(content):
                    add(container, key, text, f"anthropic.messages[{msg_index}].system_or_developer", order)
                    order += 1
        return entries

    instructions = body.get("instructions")
    if isinstance(instructions, str):
        add(body, "instructions", instructions, "openai.instructions", order)
        order += 1
    elif isinstance(instructions, list):
        for index, block in enumerate(instructions):
            if isinstance(block, dict) and block.get("type") in {"text", "input_text"} and isinstance(block.get("text"), str):
                add(block, "text", block["text"], f"openai.instructions[{index}].text", order)
                order += 1

    for field_name in ("messages", "input"):
        value = body.get(field_name)
        if not isinstance(value, list):
            continue
        for msg_index, message in enumerate(value):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role not in {"system", "developer"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                add(message, "content", content, f"openai.{field_name}.system_or_developer", order)
                order += 1
            for container, key, text in _instruction_dedup_text_from_content(content):
                add(container, key, text, f"openai.{field_name}[{msg_index}].system_or_developer", order)
                order += 1
    return entries


def _set_instruction_dedup_entry(entry: dict[str, Any], text: str) -> None:
    container = entry.get("container")
    key = entry.get("key")
    if isinstance(container, dict) and isinstance(key, str):
        container[key] = text
    elif isinstance(container, list) and isinstance(key, int):
        container[key] = text


def _instruction_dedup_meta(policy: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "agentflow.instruction_section_deduplication.v1",
        "enabled": bool(policy.get("enabled")),
        "status": status,
        "reason": reason,
        "changed": False,
        "applied": False,
        "policy_source": public_label(policy.get("policy_source") or CRUNCH_POLICY_SOURCE, "unknown"),
        "rule_path_included": False,
        "configured_rule_count": len(policy.get("rules") or []),
        "selected_rule_id": None,
        "candidate_id": None,
        "source_surface": None,
        "provider": None,
        "endpoint": None,
        "category": None,
        "workflow_phase": None,
        "applied_count": 0,
        "holdout_count": 0,
        "eligible_section_count": 0,
        "matched_section_count": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "projected_saved_usd": 0.0,
        "reason_codes": [] if reason in {"disabled", "no-instruction-sections"} else [reason],
        "rules": [],
        "canary": None,
        "coordinator_compatibility": None,
        "privacy": {
            "metadata_only_output": True,
            "raw_instruction_text_included": False,
            "instruction_section_fingerprint_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "provider_body_included": False,
        },
        "tool_protocol_preserved": True,
        "thinking_preserved": True,
        "provider_responses_preserved": True,
        "policy_file_changed": False,
        "managed_server_call_made": False,
    }


def _instruction_dedup_public_canary(canary: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(canary)
    cohort = str(public.get("cohort") or "")
    if cohort == "canary":
        public["cohort"] = "canary_applied"
        public["status"] = "applied"
    elif cohort == "holdout":
        public["cohort"] = "canary_holdout"
        public["status"] = "holdout"
    elif cohort == "not_selected":
        public["status"] = "skipped"
    elif cohort == "disabled":
        public["status"] = "disabled"
    elif cohort == "invalid":
        public["status"] = "invalid"
    public["salt_included"] = False
    public["fingerprint_included"] = False
    return public


def _instruction_dedup_rule_meta(rule: dict[str, Any], canary: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": public_label(rule.get("id") or "instruction-section-dedup-policy", "instruction-section-dedup-policy"),
        "candidate_id": public_id(rule.get("candidate_id"), prefix="instruction-dedup-candidate") if rule.get("candidate_id") not in (None, "") else None,
        "enabled": bool(rule.get("enabled", True)),
        "policy_source": public_label(rule.get("policy_source") or CRUNCH_POLICY_SOURCE, "unknown"),
        "min_section_chars": int(rule.get("min_section_chars") or INSTRUCTION_DEDUP_MIN_SECTION_CHARS),
        "min_repeated_count": int(rule.get("min_repeated_count") or 2),
        "keep_recent_sections": int(rule.get("keep_recent_sections") or 1),
        "max_replacements": int(rule.get("max_replacements") or 0),
        "matched_section_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "projected_saved_usd": 0.0,
        "canary": canary,
        "reason_codes": [],
        "instruction_section": {
            "fingerprint_present": True,
            "fingerprint_included": False,
            "raw_text_included": False,
            "source_fields": [],
        },
    }


def _instruction_dedup_savings_usd(model: Any, provider: str, saved_tokens: int) -> float:
    basis = pricing_basis(str(model or ""), provider=provider if provider in {"anthropic", "openai"} else "openai")
    input_per_mtok = float(basis.get("input_per_mtok") or 0.0)
    return round(max(0, saved_tokens) * input_per_mtok / 1_000_000, 8)


def _instruction_dedup_safety_stop(
    store_obj: Any | None,
    *,
    rule: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else policy.get("safety_stop")
    if not isinstance(safety, dict) or not _as_bool(safety.get("enabled"), True):
        return None
    if store_obj is None or not hasattr(store_obj, "conn"):
        return None
    min_samples = max(0, int(safety.get("min_outcome_samples") or 5))
    window = max(min_samples, int(safety.get("window") or 500))
    rule_id = str(rule.get("id") or "instruction-section-dedup-policy")
    candidate_id = rule.get("candidate_id")
    try:
        rows = store_obj.conn.execute(
            """
            select status_code, retry_count, cost_est_usd, cost_baseline_usd, crunch_json
            from calls
            where crunch_json is not null
            order by created_at desc
            limit ?
            """,
            (window,),
        ).fetchall()
    except Exception:
        return None
    samples = 0
    errors = 0
    retries = 0
    negative_savings = 0
    for row in rows:
        row_dict = dict(row)
        try:
            meta = json.loads(row_dict.get("crunch_json") or "{}")
        except (TypeError, ValueError):
            continue
        dedup = meta.get("instruction_section_deduplication")
        if not isinstance(dedup, dict):
            continue
        if str(dedup.get("selected_rule_id") or "") != rule_id:
            continue
        if candidate_id and dedup.get("candidate_id") != candidate_id:
            continue
        if dedup.get("status") != "applied" and not dedup.get("applied"):
            continue
        samples += 1
        if int(row_dict.get("status_code") or 0) >= 400:
            errors += 1
        if int(row_dict.get("retry_count") or 0) > 0:
            retries += 1
        cost = row_dict.get("cost_est_usd")
        baseline = row_dict.get("cost_baseline_usd")
        if cost is not None and baseline is not None and float(cost) > float(baseline):
            negative_savings += 1
    if samples < min_samples:
        return None
    error_rate = errors / samples if samples else 0.0
    retry_rate = retries / samples if samples else 0.0
    negative_rate = negative_savings / samples if samples else 0.0
    reasons: list[str] = []
    if error_rate > float(safety.get("max_error_rate") or 0.1):
        reasons.append("instruction-dedup-error-rate-safety-stop")
    if retry_rate > float(safety.get("max_retry_rate") or 0.25):
        reasons.append("instruction-dedup-retry-rate-safety-stop")
    if negative_rate > float(safety.get("max_negative_savings_rate") or 0.25):
        reasons.append("instruction-dedup-negative-savings-safety-stop")
    if not reasons:
        return None
    return {
        "schema": "agentflow.instruction_section_dedup_safety_stop.v1",
        "status": "safety_stopped",
        "reason_codes": reasons,
        "sample_count": samples,
        "error_count": errors,
        "retry_count": retries,
        "negative_savings_count": negative_savings,
        "error_rate": round(error_rate, 6),
        "retry_rate": round(retry_rate, 6),
        "negative_savings_rate": round(negative_rate, 6),
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "metadata_only": True,
        "provider_body_included": False,
    }


def _apply_instruction_section_deduplication(
    body: dict[str, Any],
    *,
    store_obj: Any | None = None,
    routing_meta: dict[str, Any] | None = None,
    provider: str | None = None,
    source_surface: str | None = None,
    endpoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = copy.deepcopy(INSTRUCTION_SECTION_DEDUP_POLICY)
    policy["policy_source"] = str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE)
    meta = _instruction_dedup_meta(policy, "skipped", "disabled")
    before_chars = len(stable_json(body))
    meta["before_chars"] = before_chars
    meta["after_chars"] = before_chars
    if not _as_bool(policy.get("enabled"), False):
        return body, meta

    provider_name, surface, endpoint_name = _instruction_dedup_provider_and_surface(
        body,
        provider=provider,
        source_surface=source_surface,
        endpoint=endpoint,
    )
    category = str((routing_meta or {}).get("category") or _crunch_request_category(body))
    phase = str((routing_meta or {}).get("workflow_phase") or category)
    meta.update({
        "enabled": True,
        "status": "skipped",
        "reason": "no-instruction-sections",
        "reason_codes": [],
        "source_surface": surface,
        "provider": provider_name,
        "endpoint": endpoint_name,
        "category": category,
        "workflow_phase": phase,
    })

    if not _instruction_source_allowed(surface, policy.get("source_surfaces") or []):
        meta["reason"] = "unsafe-source-surface"
        meta["reason_codes"] = ["unsafe-source-surface"]
        return body, meta
    if _content_has_tool_protocol(body) and (
        _as_bool(policy.get("block_tool_protocol"), True) or _as_bool(policy.get("block_tool_payloads"), True)
    ):
        meta["reason"] = "tool-protocol-risk"
        meta["reason_codes"] = ["tool-protocol-risk"]
        return body, meta
    if _body_uses_thinking(body) and _as_bool(policy.get("block_thinking"), True):
        meta["reason"] = "thinking-content-risk"
        meta["reason_codes"] = ["thinking-content-risk"]
        return body, meta

    compatibility = _instruction_dedup_coordinator_compatibility(routing_meta or {})
    meta["coordinator_compatibility"] = compatibility
    if not compatibility.get("compatible"):
        meta["status"] = "suppressed"
        meta["reason"] = "coordinator-conflict"
        meta["reason_codes"] = ["coordinator-conflict", *[str(item) for item in compatibility.get("reason_codes") or []]]
        return body, meta

    entries = _instruction_dedup_entries(body, provider=provider_name)
    meta["eligible_section_count"] = len(entries)
    if not entries:
        return body, meta
    by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_fingerprint.setdefault(str(entry.get("fingerprint") or ""), []).append(entry)

    basis = {
        "source_surface": surface,
        "category": category,
        "workflow_phase": phase,
    }
    applied_count = 0
    holdout_count = 0
    saved_chars = 0
    reason_codes: set[str] = set()
    max_total = max(0, int(policy.get("max_replacements") or 0))
    model = body.get("model")

    for fingerprint, occurrences in sorted(by_fingerprint.items(), key=lambda item: min(int(e["order"]) for e in item[1])):
        rule = _select_instruction_dedup_rule(policy, basis, fingerprint)
        if rule is None:
            reason_codes.add("no-matching-instruction-dedup-rule")
            continue
        canary = _instruction_dedup_public_canary(
            _instruction_dedup_cohort(rule, fingerprint, basis)
        )
        rule_meta = _instruction_dedup_rule_meta(rule, canary)
        source_fields = sorted({str(entry.get("source_field") or "unknown") for entry in occurrences})
        rule_meta["instruction_section"]["source_fields"] = source_fields
        meta["rules"].append(rule_meta)
        repeated_count = len(occurrences)
        rule_meta["matched_section_count"] = repeated_count
        meta["matched_section_count"] += repeated_count
        min_repeated = max(1, int(rule.get("min_repeated_count") or policy.get("min_repeated_count") or 2))
        min_chars = max(0, int(rule.get("min_section_chars") or policy.get("min_section_chars") or INSTRUCTION_DEDUP_MIN_SECTION_CHARS))
        keep_recent = max(0, int(rule.get("keep_recent_sections") or policy.get("keep_recent_sections") or 1))
        max_rule = max(0, int(rule.get("max_replacements") or policy.get("max_replacements") or 0))
        if repeated_count < min_repeated:
            reason_codes.add("insufficient-repeated-instruction-sections")
            rule_meta["reason_codes"].append("insufficient-repeated-instruction-sections")
            continue
        if not canary.get("holdout_configured"):
            reason_codes.add("missing-holdout-configuration")
            rule_meta["reason_codes"].append("missing-holdout-configuration")
            continue
        if not canary.get("valid", True):
            reason_codes.add("invalid-canary-configuration")
            rule_meta["reason_codes"].append("invalid-canary-configuration")
            continue
        if canary.get("status") == "holdout":
            count = max(0, repeated_count - keep_recent)
            holdout_count += count
            rule_meta["holdout_count"] += count
            reason_codes.add("instruction-dedup-holdout")
            rule_meta["reason_codes"].append("instruction-dedup-holdout")
            continue
        if not canary.get("selected"):
            reason = "instruction-dedup-canary-not-selected"
            if canary.get("status") == "disabled":
                reason = "instruction-dedup-canary-disabled"
            reason_codes.add(reason)
            rule_meta["reason_codes"].append(reason)
            continue
        safety_stop = _instruction_dedup_safety_stop(store_obj, rule=rule, policy=policy)
        if safety_stop:
            meta["safety_stop"] = safety_stop
            meta["status"] = "safety_stopped"
            meta["reason"] = "local-canary-safety-stop"
            meta["reason_codes"] = sorted({"local-canary-safety-stop", *safety_stop.get("reason_codes", [])})
            rule_meta["reason_codes"].append("local-canary-safety-stop")
            rule_meta["safety_stop"] = safety_stop
            return body, meta
        protected_start = max(0, repeated_count - keep_recent)
        replacements_for_rule = 0
        for occurrence_index, entry in enumerate(sorted(occurrences, key=lambda item: int(item["order"]))):
            if occurrence_index >= protected_start:
                reason_codes.add("kept-recent-instruction-section")
                continue
            if len(str(entry.get("normalized") or "")) < min_chars:
                reason_codes.add("instruction-section-below-min-chars")
                rule_meta["reason_codes"].append("instruction-section-below-min-chars")
                continue
            if max_total and applied_count >= max_total:
                reason_codes.add("max-replacements-reached")
                break
            if max_rule and replacements_for_rule >= max_rule:
                reason_codes.add("rule-max-replacements-reached")
                break
            notice = str(rule.get("replacement_notice") or policy.get("replacement_notice") or "[repeated instruction section omitted by AgentFlow]")
            if len(notice) >= len(str(entry.get("text") or "")):
                reason_codes.add("replacement-not-smaller")
                rule_meta["reason_codes"].append("replacement-not-smaller")
                continue
            before_len = len(str(entry.get("text") or ""))
            _set_instruction_dedup_entry(entry, notice)
            saved = before_len - len(notice)
            applied_count += 1
            replacements_for_rule += 1
            saved_chars += saved
            rule_meta["applied_count"] += 1
            rule_meta["saved_chars"] += saved
            rule_meta["tokens_saved_est"] = rule_meta["saved_chars"] // TOKEN_CHARS
            rule_meta["projected_saved_usd"] = _instruction_dedup_savings_usd(model, provider_name, rule_meta["tokens_saved_est"])
            if meta["selected_rule_id"] is None:
                meta["selected_rule_id"] = rule_meta["rule_id"]
                meta["candidate_id"] = rule_meta["candidate_id"]
                meta["policy_source"] = rule_meta["policy_source"]
                meta["canary"] = canary

    after_chars = len(stable_json(body))
    meta["applied_count"] = applied_count
    meta["holdout_count"] = holdout_count
    meta["saved_chars"] = max(0, saved_chars)
    meta["tokens_saved_est"] = meta["saved_chars"] // TOKEN_CHARS
    meta["projected_saved_usd"] = _instruction_dedup_savings_usd(model, provider_name, int(meta["tokens_saved_est"]))
    meta["after_chars"] = after_chars
    meta["changed"] = applied_count > 0
    meta["applied"] = applied_count > 0
    if applied_count > 0:
        meta["status"] = "applied"
        meta["reason"] = "instruction-section-dedup-applied"
    elif holdout_count > 0:
        meta["status"] = "holdout"
        meta["reason"] = "instruction-dedup-holdout"
    elif reason_codes:
        meta["reason"] = sorted(reason_codes)[0]
    meta["reason_codes"] = sorted(reason_codes)
    return body, meta


def _provider_text_entries(body: dict[str, Any], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    skips = {
        "system_or_developer_message": 0,
        "tool_protocol_payload": 0,
        "thinking_payload": 0,
        "non_message_input": 0,
    }
    block_tool = _as_bool(policy.get("block_tool_protocol"), True)
    block_thinking = _as_bool(policy.get("block_thinking"), True)
    keep_recent_messages = max(0, int(policy.get("keep_recent_messages", 2)))

    def add_text(container: Any, key: Any, text: str, location: str, message_index: int, protected: bool) -> None:
        parts = re.split(r"(\n\s*\n+)", text)
        section_indexes = [
            idx
            for idx in range(0, len(parts), 2)
            if len(normalize_text(parts[idx])) >= int(policy.get("min_section_chars", 700))
        ]
        for section_number, part_index in enumerate(section_indexes, start=1):
            section = parts[part_index]
            normalized = normalize_text(section)
            entries.append({
                "container": container,
                "key": key,
                "parts": parts,
                "part_index": part_index,
                "section_number": section_number,
                "text": section,
                "hash": _pattern_hash_for_text(section),
                "normalized_chars": len(normalized),
                "location": location,
                "message_index": message_index,
                "protected": protected,
            })

    def collect_from_content(content: Any, location: str, message_index: int, protected: bool) -> None:
        if isinstance(content, str):
            add_text(None, None, content, location, message_index, protected)
            return
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                block_location = f"{location}[{block_index}]"
                if isinstance(block, str):
                    add_text(content, block_index, block, block_location, message_index, protected)
                elif isinstance(block, dict):
                    for key in ("text", "input_text", "output_text", "content"):
                        if isinstance(block.get(key), str):
                            add_text(block, key, block[key], f"{block_location}.{key}", message_index, protected)
                            break

    def collect_messages(messages: Any, prefix: str) -> None:
        if not isinstance(messages, list):
            return
        protect_start = max(0, len(messages) - keep_recent_messages)
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = _message_role(msg)
            if role in {"system", "developer"}:
                skips["system_or_developer_message"] += 1
                continue
            content = msg.get("content")
            has_tool_protocol = bool(msg.get("tool_calls") or msg.get("tool_call_id") or _content_has_tool_protocol(content))
            has_thinking = _content_has_thinking(content)
            if block_tool and has_tool_protocol:
                skips["tool_protocol_payload"] += 1
            if block_thinking and has_thinking:
                skips["thinking_payload"] += 1
            if (block_tool and has_tool_protocol) or (block_thinking and has_thinking):
                continue
            protected = idx >= protect_start
            if isinstance(content, str):
                add_text(msg, "content", content, f"{prefix}[{idx}].content", idx, protected)
            else:
                collect_from_content(content, f"{prefix}[{idx}].content", idx, protected)

    collect_messages(body.get("messages"), "messages")
    input_value = body.get("input")
    if isinstance(input_value, list):
        collect_messages(input_value, "input")
    elif input_value is not None:
        skips["non_message_input"] += 1
    return entries, skips


def _set_provider_section_entry(entry: dict[str, Any], replacement: str) -> None:
    parts = list(entry["parts"])
    parts[int(entry["part_index"])] = replacement
    new_text = "".join(parts)
    container = entry.get("container")
    key = entry.get("key")
    if isinstance(container, dict) and isinstance(key, str):
        container[key] = new_text
    elif isinstance(container, list) and isinstance(key, int):
        container[key] = new_text


def _provider_scaffold_notice(
    *,
    rule_id: str,
    candidate_id: Any,
    section_hash: str,
    original_chars: int,
    max_chars: int,
) -> str:
    short_hash = section_hash.split("sha256:", 1)[-1][:12]
    candidate = f"; candidate_id={candidate_id}" if candidate_id else ""
    notice = (
        "[AgentFlow: repeated provider scaffolding omitted; "
        f"rule_id={rule_id}{candidate}; scaffold_hash={short_hash}; original_chars={original_chars}]"
    )
    return notice[:max(1, max_chars)]


def _provider_scaffolding_canary_features(
    *,
    body: dict[str, Any],
    category: str,
    pattern_hash: str,
    before_chars: int,
) -> dict[str, Any]:
    return {
        "source_surface": "provider_request",
        "app_family": "unknown",
        "category": category,
        "workflow_phase": category,
        "text_bucket": _text_bucket(before_chars),
        "token_bucket": _token_bucket(max(1, before_chars // TOKEN_CHARS)),
        "requested_model": body.get("model"),
        "candidate_target_model": body.get("model"),
        "pattern_hashes": [pattern_hash],
    }


def _rule_conditions_match_provider_request(rule: dict[str, Any], *, body: dict[str, Any], category: str) -> tuple[bool, str | None]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    if not conditions:
        return True, None
    expected_model = conditions.get("requested_model")
    if expected_model is not None and str(expected_model) != str(body.get("model") or ""):
        return False, "requested-model-mismatch"
    expected_has_tools = conditions.get("has_tools")
    if expected_has_tools is not None and _as_bool(expected_has_tools, False) != _body_has_tools(body):
        return False, "has-tools-mismatch"
    expected_uses_thinking = conditions.get("uses_thinking")
    if expected_uses_thinking is not None and _as_bool(expected_uses_thinking, False) != _body_uses_thinking(body):
        return False, "uses-thinking-mismatch"
    return True, None


def _apply_repeated_provider_scaffolding(
    body: dict[str, Any],
    *,
    managed_profile: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    policy = _effective_repeated_provider_scaffolding_policy(managed_profile)
    before_chars = len(stable_json(body))
    meta = _provider_scaffolding_meta(policy, "skipped", "disabled")
    meta["before_chars"] = before_chars
    if not _as_bool(policy.get("enabled"), False):
        return 0, meta
    if not policy.get("rules"):
        meta["reason"] = "no-reviewed-rules"
        return 0, meta
    if before_chars < int(policy.get("min_request_chars", 12000)):
        meta["reason"] = "request-too-small"
        return 0, meta

    entries, safety_skips = _provider_text_entries(body, policy)
    meta["safety_skips"] = {key: value for key, value in safety_skips.items() if value}
    if not entries:
        meta["reason"] = "no-safe-provider-sections"
        return 0, meta

    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry["hash"])] = counts.get(str(entry["hash"]), 0) + 1
    total_saved = 0
    applied_count = 0
    holdout_count = 0
    skip_counts: dict[tuple[str, str], int] = {}
    category = _crunch_request_category(body)

    for rule in policy.get("rules") or []:
        rule_id = str(rule.get("id") or "repeated-provider-scaffold")
        rule_policy_source = str(rule.get("policy_source") or policy.get("policy_source") or CRUNCH_POLICY_SOURCE)
        rule_meta = {
            "rule_id": rule_id,
            "candidate_id": rule.get("candidate_id"),
            "enabled": _as_bool(rule.get("enabled"), True),
            "policy_source": rule_policy_source,
            "matched_pattern_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "saved_chars": 0,
            "rollout": pattern_rollout_public_meta(rule.get("rollout")),
            "skip_reasons": [],
        }
        if not rule_meta["enabled"]:
            skip_counts[(rule_id, "disabled")] = skip_counts.get((rule_id, "disabled"), 0) + 1
            rule_meta["skip_reasons"].append({"reason": "disabled", "count": 1})
            meta["rules"].append(rule_meta)
            continue
        conditions_match, mismatch_reason = _rule_conditions_match_provider_request(rule, body=body, category=category)
        if not conditions_match:
            reason = mismatch_reason or "conditions-mismatch"
            skip_counts[(rule_id, reason)] = skip_counts.get((rule_id, reason), 0) + 1
            rule_meta["skip_reasons"].append({"reason": reason, "count": 1})
            meta["rules"].append(rule_meta)
            continue
        min_repeated = max(1, int(rule.get("min_repeated_count", 2)))
        keep_recent_matches = max(0, int(rule.get("keep_recent_matches", policy.get("keep_recent_matches", 1))))
        max_applications = max(0, min(
            int(rule.get("max_applications", policy.get("max_replacements", 16))),
            int(policy.get("max_replacements", 16)),
        ))
        pattern_hashes = [str(item) for item in rule.get("pattern_hashes") or []]
        if not pattern_hashes and _as_bool(rule.get("match_any_repeated"), False):
            pattern_hashes = [
                item[0]
                for item in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                if item[1] >= min_repeated
            ]
            rule_meta["match_any_repeated"] = True
        for pattern_hash in pattern_hashes:
            occurrences = [entry for entry in entries if str(entry["hash"]) == pattern_hash]
            if len(occurrences) < min_repeated:
                skip_counts[(rule_id, "min-repeated-count-not-met")] = skip_counts.get((rule_id, "min-repeated-count-not-met"), 0) + 1
                rule_meta["skip_reasons"].append({"reason": "min-repeated-count-not-met", "count": len(occurrences)})
                continue
            rule_meta["matched_pattern_count"] += 1
            canary = pattern_canary_decision(
                rollout=rule.get("rollout"),
                rule_id=rule_id,
                candidate_id=rule.get("candidate_id"),
                pattern_hashes=[pattern_hash],
                features=_provider_scaffolding_canary_features(
                    body=body,
                    category=category,
                    pattern_hash=pattern_hash,
                    before_chars=before_chars,
                ),
            )
            canary.pop("pattern_hashes", None)
            if canary.get("enabled"):
                rule_meta["canary"] = canary
            if canary.get("enabled") and not canary.get("selected", True):
                count = len(occurrences)
                rule_meta["holdout_count"] += count
                holdout_count += count
                skip_counts[(rule_id, "canary_holdout")] = skip_counts.get((rule_id, "canary_holdout"), 0) + count
                rule_meta["skip_reasons"].append({"reason": "canary_holdout", "count": count, "canary": canary})
                continue
            mutable = [entry for entry in occurrences if not entry.get("protected")]
            protected_start = max(0, len(mutable) - keep_recent_matches)
            for occurrence_index, entry in enumerate(mutable):
                if rule_meta["applied_count"] >= max_applications:
                    skip_counts[(rule_id, "max-applications-reached")] = skip_counts.get((rule_id, "max-applications-reached"), 0) + 1
                    break
                if occurrence_index >= protected_start:
                    skip_counts[(rule_id, "kept-recent-match")] = skip_counts.get((rule_id, "kept-recent-match"), 0) + 1
                    continue
                if rule.get("min_section_chars") is not None and int(entry["normalized_chars"]) < int(rule["min_section_chars"]):
                    skip_counts[(rule_id, "section-too-small")] = skip_counts.get((rule_id, "section-too-small"), 0) + 1
                    continue
                replacement = _provider_scaffold_notice(
                    rule_id=rule_id,
                    candidate_id=rule.get("candidate_id"),
                    section_hash=pattern_hash,
                    original_chars=len(str(entry.get("text") or "")),
                    max_chars=int((rule.get("action") or {}).get("max_replacement_chars", 360)),
                )
                if len(replacement) >= len(str(entry.get("text") or "")):
                    skip_counts[(rule_id, "replacement-not-smaller")] = skip_counts.get((rule_id, "replacement-not-smaller"), 0) + 1
                    continue
                before_len = len(str(entry["text"]))
                _set_provider_section_entry(entry, replacement)
                saved = before_len - len(replacement)
                total_saved += saved
                applied_count += 1
                rule_meta["applied_count"] += 1
                rule_meta["saved_chars"] += saved
        meta["rules"].append(rule_meta)

    after_chars = len(stable_json(body))
    meta.update({
        "status": "applied" if applied_count else "skipped",
        "reason": "repeated-provider-scaffolding-crunched" if applied_count else "no-repeated-provider-scaffolding",
        "changed": applied_count > 0,
        "after_chars": after_chars,
        "saved_chars": before_chars - after_chars,
        "text_saved_chars": total_saved,
        "tokens_saved_est": (before_chars - after_chars) // TOKEN_CHARS,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "category": category,
        "skip_reasons": [
            {"rule_id": rule_id, "reason": reason, "count": count}
            for (rule_id, reason), count in sorted(skip_counts.items())
        ],
    })
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
    """Codex turn specific deterministic crunching for text-only turn/start input.

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


LOG_LEVEL_RE = re.compile(r"^(?:\[(?P<bracketed>TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL)\]|(?P<bare>TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL))\b[:\]\s-]*")
TIMESTAMP_RE = re.compile(
    r"^(?:\[(?:\d{4}-\d{2}-\d{2}[T ][0-9:.]+(?:Z|[+-]\d{2}:?\d{2})?|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]|\d{4}-\d{2}-\d{2}[T ][0-9:.]+(?:Z|[+-]\d{2}:?\d{2})?|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
)
PID_THREAD_RE = re.compile(
    r"^(?:\[(?:pid|process|thread|tid)?[=:\s-]*[A-Za-z0-9_.:-]{1,48}\]|\((?:pid|process|thread|tid)[=:\s-]*[A-Za-z0-9_.:-]{1,48}\)|(?:pid|process|thread|tid)[=:\s-]+[A-Za-z0-9_.:-]{1,48})\s+",
    re.IGNORECASE,
)
MODULE_RE = re.compile(r"^(?:[A-Za-z_][\w.:-]{1,80}|[A-Za-z_][\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|c|cc|cpp|h|hpp|sh|bash|zsh))(?::\d+(?::\d+)?)?\s*(?:-|:)\s+")
FILE_LINE_RE = re.compile(r"^((?:[A-Za-z]:)?[/\\]?[A-Za-z0-9_./\\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|c|cc|cpp|h|hpp|sh|bash|zsh|log):\d+(?::\d+)?(?::| - )\s+)(.+)$")
SHELL_PROMPT_RE = re.compile(r"^((?:\([^)]+\)\s*)?(?:[A-Za-z0-9_.-]+@[^:\s]+:)?[^#$%>\n]{0,120}[#$%>]\s+)(\S.*)$")
CANONICAL_SHEBANG_RE = re.compile(r"^#!\s*/(?:usr/bin/env\s+)?(?:bash|sh|zsh|python(?:3(?:\.\d+)?)?|node|ruby|perl)\s*$")
PURE_TEST_MARKER_RE = re.compile(r"^\s*(?:[=._-]{6,}|(?:\.|F|E|s|x){8,})\s*$")
DIAGNOSTIC_LINE_RE = re.compile(
    r"(?:\b(?:ERROR|ERR|FATAL|CRITICAL|FAILED|FAILURE|AssertionError|Traceback|Exception|exit code|exit status|non-zero|panic:)\b|^\s*File \"[^\"]+\", line \d+|^\s*at .+:\d+|^\s*E\s+|^\s*>\s+)"
)


def _terminal_log_meta(status: str, reason: str) -> dict[str, Any]:
    return {
        "enabled": TERMINAL_LOG_ENABLED,
        "status": status,
        "reason": reason,
        "changed": False,
        "policy_source": CRUNCH_POLICY_SOURCE,
        "rule_path": CRUNCH_RULES_PATH,
        "min_lines": TERMINAL_LOG_MIN_LINES,
        "min_repeated_lines": TERMINAL_LOG_MIN_REPEATED_LINES,
        "max_annotations": TERMINAL_LOG_MAX_ANNOTATIONS,
        "pattern_types": [],
        "matched_line_counts": {},
        "simplified_line_count": 0,
        "annotations_inserted": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "safety_skips": {},
        "error_bearing_lines_preserved": False,
        "raw_log_text_included": False,
    }


def _bump_counter(counts: dict[str, int], key: str, amount: int = 1) -> None:
    counts[key] = counts.get(key, 0) + amount


def _is_diagnostic_line(line: str) -> bool:
    return bool(DIAGNOSTIC_LINE_RE.search(line))


def _strip_log_prefix(line: str) -> tuple[tuple[str, ...], str] | None:
    rest = line
    pattern_types: list[str] = []
    while rest:
        original = rest
        match = TIMESTAMP_RE.match(rest)
        if match:
            pattern_types.append("timestamp_prefix")
            rest = rest[match.end():]
        match = LOG_LEVEL_RE.match(rest)
        if match:
            pattern_types.append("log_level_prefix")
            rest = rest[match.end():]
        match = PID_THREAD_RE.match(rest)
        if match:
            pattern_types.append("pid_thread_prefix")
            rest = rest[match.end():]
        match = MODULE_RE.match(rest)
        if match:
            pattern_types.append("module_prefix")
            rest = rest[match.end():]
        if rest == original:
            break
    rest = rest.lstrip(" -:\t")
    unique_types = tuple(dict.fromkeys(pattern_types))
    if not rest or len(unique_types) == 0:
        return None
    if "timestamp_prefix" not in unique_types and "log_level_prefix" not in unique_types:
        return None
    return unique_types, rest


def _terminal_line_candidate(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None
    if CANONICAL_SHEBANG_RE.match(line.strip()):
        return {
            "key": ("shebang_line", line.strip()),
            "pattern_type": "shebang_line",
            "payload": line.strip(),
            "preserve_once": True,
        }
    if PURE_TEST_MARKER_RE.match(line):
        return {
            "key": ("test_progress_marker", "marker"),
            "pattern_type": "test_progress_marker",
            "payload": "",
            "omit_line": True,
        }
    match = SHELL_PROMPT_RE.match(line)
    if match:
        return {
            "key": ("shell_prompt_prefix", match.group(1).strip()),
            "pattern_type": "shell_prompt_prefix",
            "payload": match.group(2).strip(),
            "payload_label": "command",
        }
    match = FILE_LINE_RE.match(line)
    if match:
        return {
            "key": ("file_line_prefix", match.group(1)),
            "pattern_type": "file_line_prefix",
            "payload": match.group(2).strip(),
            "shared_prefix": match.group(1).strip(),
        }
    stripped = _strip_log_prefix(line)
    if stripped:
        pattern_types, payload = stripped
        return {
            "key": ("log_prefix", pattern_types),
            "pattern_type": "+".join(pattern_types),
            "payload": payload,
        }
    return None


def _terminal_annotation(candidate: dict[str, Any], count: int) -> str:
    pattern_type = str(candidate["pattern_type"])
    if pattern_type == "shebang_line":
        return f"[AgentFlow: {count} repeated shebang lines collapsed; diagnostics preserved]"
    if pattern_type == "test_progress_marker":
        return f"[AgentFlow: {count} test progress/divider lines omitted; diagnostics preserved]"
    if pattern_type == "shell_prompt_prefix":
        return f"[AgentFlow: {count} shell prompt prefixes omitted; commands preserved]"
    if pattern_type == "file_line_prefix":
        prefix = str(candidate.get("shared_prefix") or "")
        return f"[AgentFlow: {count} repeated file/line prefixes omitted; shared_prefix={prefix}; diagnostics preserved]"
    rendered = pattern_type.replace("+", ", ")
    return f"[AgentFlow: {count} log lines shared {rendered}; prefixes omitted; diagnostics preserved]"


def _terminal_payload_line(candidate: dict[str, Any], *, first_occurrence: bool) -> str | None:
    if candidate.get("omit_line"):
        return None
    payload = str(candidate.get("payload") or "")
    if candidate.get("preserve_once"):
        return payload if first_occurrence else None
    label = candidate.get("payload_label")
    if label:
        return f"{label}: {payload}"
    return payload


def _simplify_terminal_log_boilerplate_text(text: str) -> tuple[str, dict[str, Any]]:
    meta = _terminal_log_meta("skipped", "no-terminal-log-boilerplate")
    if not TERMINAL_LOG_ENABLED:
        meta["reason"] = "disabled"
        return text, meta

    lines = text.splitlines()
    if len(lines) < TERMINAL_LOG_MIN_LINES:
        meta["reason"] = "not-enough-lines"
        meta["line_count"] = len(lines)
        return text, meta

    candidates: dict[int, dict[str, Any]] = {}
    counts: dict[tuple[Any, ...], int] = {}
    first_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    safety_skips: dict[str, int] = {}
    for index, line in enumerate(lines):
        if _is_diagnostic_line(line):
            _bump_counter(safety_skips, "diagnostic-line")
            continue
        candidate = _terminal_line_candidate(line)
        if not candidate:
            continue
        key = candidate["key"]
        candidates[index] = candidate
        counts[key] = counts.get(key, 0) + 1
        first_by_key.setdefault(key, candidate)

    eligible_keys = {
        key
        for key, count in counts.items()
        if count >= TERMINAL_LOG_MIN_REPEATED_LINES
    }
    if not eligible_keys:
        meta["line_count"] = len(lines)
        meta["safety_skips"] = safety_skips
        return text, meta

    annotation_count = 0
    emitted_keys: set[tuple[Any, ...]] = set()
    output: list[str] = []
    matched_line_counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        candidate = candidates.get(index)
        if not candidate or candidate["key"] not in eligible_keys:
            output.append(line)
            continue
        key = candidate["key"]
        pattern_type = str(candidate["pattern_type"])
        first_occurrence = key not in emitted_keys
        if first_occurrence:
            if annotation_count >= TERMINAL_LOG_MAX_ANNOTATIONS:
                _bump_counter(safety_skips, "annotation-limit")
                output.append(line)
                continue
            output.append(_terminal_annotation(first_by_key[key], counts[key]))
            annotation_count += 1
            emitted_keys.add(key)
        payload_line = _terminal_payload_line(candidate, first_occurrence=first_occurrence)
        if payload_line:
            output.append(payload_line)
        _bump_counter(matched_line_counts, pattern_type)

    trailing_newline = "\n" if text.endswith("\n") else ""
    simplified = "\n".join(output) + trailing_newline
    saved_chars = len(text) - len(simplified)
    if saved_chars <= 0:
        meta["line_count"] = len(lines)
        meta["safety_skips"] = safety_skips
        meta["reason"] = "replacement-not-smaller"
        return text, meta

    meta.update({
        "status": "applied",
        "reason": "terminal-log-boilerplate-simplified",
        "changed": True,
        "line_count": len(lines),
        "simplified_line_count": sum(matched_line_counts.values()),
        "annotations_inserted": annotation_count,
        "pattern_types": sorted(matched_line_counts),
        "matched_line_counts": matched_line_counts,
        "saved_chars": saved_chars,
        "tokens_saved_est": saved_chars // TOKEN_CHARS,
        "safety_skips": safety_skips,
        "error_bearing_lines_preserved": bool(safety_skips.get("diagnostic-line")),
    })
    return simplified, meta


def _terminal_log_aggregate_meta(metas: list[dict[str, Any]]) -> dict[str, Any]:
    base = _terminal_log_meta("skipped", "no-terminal-log-boilerplate")
    if not metas:
        return base
    matched_line_counts: dict[str, int] = {}
    safety_skips: dict[str, int] = {}
    saved_chars = 0
    annotations = 0
    simplified_lines = 0
    applied = 0
    reasons: dict[str, int] = {}
    for item in metas:
        _bump_counter(reasons, str(item.get("reason") or "unknown"))
        if item.get("changed"):
            applied += 1
        saved_chars += int(item.get("saved_chars") or 0)
        annotations += int(item.get("annotations_inserted") or 0)
        simplified_lines += int(item.get("simplified_line_count") or 0)
        for key, value in (item.get("matched_line_counts") or {}).items():
            _bump_counter(matched_line_counts, str(key), int(value or 0))
        for key, value in (item.get("safety_skips") or {}).items():
            _bump_counter(safety_skips, str(key), int(value or 0))
    base.update({
        "status": "applied" if applied else "skipped",
        "reason": "terminal-log-boilerplate-simplified" if applied else "no-terminal-log-boilerplate",
        "changed": bool(applied),
        "text_blocks_examined": len(metas),
        "text_blocks_changed": applied,
        "simplified_line_count": simplified_lines,
        "annotations_inserted": annotations,
        "pattern_types": sorted(matched_line_counts),
        "matched_line_counts": matched_line_counts,
        "saved_chars": saved_chars,
        "tokens_saved_est": saved_chars // TOKEN_CHARS,
        "safety_skips": safety_skips,
        "error_bearing_lines_preserved": bool(safety_skips.get("diagnostic-line")),
        "skip_reasons": reasons,
    })
    if not applied and set(reasons) == {"disabled"}:
        base["reason"] = "disabled"
    return base


def _summary_base_meta(status: str, reason: str, *, policy: dict[str, Any] | None = None, managed_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or _effective_summary_policy(managed_profile)
    excluded = policy.get("excluded_categories") or []
    return {
        "enabled": bool(policy.get("enabled")),
        "status": status,
        "reason": reason,
        "changed": False,
        "policy_source": str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE),
        "rule_path": CRUNCH_RULES_PATH,
        "rule_id": str(policy.get("rule_id") or OLD_CONTEXT_SUMMARY_RULE_ID),
        "candidate_id": str(policy.get("candidate_id")) if policy.get("candidate_id") is not None else None,
        "model": str(policy.get("model") or OLD_CONTEXT_SUMMARY_MODEL),
        "placement": str(policy.get("placement") or OLD_CONTEXT_SUMMARY_PLACEMENT),
        "min_request_chars": int(policy.get("min_request_chars") or OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS),
        "min_summarized_chars": int(policy.get("min_summarized_chars") or OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS),
        "max_turns": int(policy.get("max_turns") or OLD_CONTEXT_SUMMARY_MAX_TURNS),
        "keep_recent_turns": int(policy.get("keep_recent_turns") or OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS),
        "max_summary_chars": int(policy.get("max_summary_chars") or OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS),
        "max_source_chars": int(policy.get("max_source_chars") or OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS),
        "max_summary_cost_usd": float(policy.get("max_summary_cost_usd") or OLD_CONTEXT_SUMMARY_MAX_COST_USD),
        "excluded_categories": sorted({str(item) for item in excluded}),
        "block_tool_protocol": _as_bool(policy.get("block_tool_protocol"), OLD_CONTEXT_SUMMARY_BLOCK_TOOL_PROTOCOL),
        "block_thinking": _as_bool(policy.get("block_thinking"), OLD_CONTEXT_SUMMARY_BLOCK_THINKING),
        "canary": _summary_canary_public_meta(policy),
        "safety_stop": _summary_safety_public_meta(policy),
        "enhanced_crunch_provider": enhanced_crunch_provider_public_meta(managed_profile),
    }


def _summary_canary_public_meta(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or OLD_CONTEXT_SUMMARY_POLICY
    canary = policy.get("canary") or {}
    fraction = float(canary.get("fraction", 1.0))
    holdout_fraction = canary.get("holdout_fraction")
    if holdout_fraction is None:
        holdout_fraction = max(0.0, 1.0 - fraction)
    return {
        "enabled": bool(canary.get("enabled")),
        "fraction": fraction,
        "holdout_fraction": float(holdout_fraction),
        "salt": str(canary.get("salt") or ""),
        "unit": str(canary.get("unit") or "source_hash"),
    }


def _summary_safety_public_meta(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or OLD_CONTEXT_SUMMARY_POLICY
    safety = policy.get("safety_stop") or {}
    return {
        "enabled": bool(safety.get("enabled", True)),
        "min_outcome_samples": int(safety.get("min_outcome_samples", 5)),
        "window": int(safety.get("window", 500)),
        "max_error_rate": float(safety.get("max_error_rate", 0.1)),
        "max_retry_rate": float(safety.get("max_retry_rate", 0.25)),
        "max_negative_net_savings_rate": float(safety.get("max_negative_net_savings_rate", 0.5)),
        "max_summary_failure_rate": float(safety.get("max_summary_failure_rate", 0.1)),
        "max_error_rate_delta": float(safety.get("max_error_rate_delta", 0.05)),
    }


def _message_has_tool_protocol(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"} for block in content)


def _message_has_thinking(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "thinking" for block in content)


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
    managed_profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    policy = _effective_summary_policy(managed_profile)
    enabled = bool(policy.get("enabled"))
    model = str(policy.get("model") or OLD_CONTEXT_SUMMARY_MODEL)
    rule_id = str(policy.get("rule_id") or OLD_CONTEXT_SUMMARY_RULE_ID)
    candidate_id = policy.get("candidate_id")
    placement = str(policy.get("placement") or OLD_CONTEXT_SUMMARY_PLACEMENT)
    min_request_chars = int(policy.get("min_request_chars") or OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS)
    min_summarized_chars = int(policy.get("min_summarized_chars") or OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS)
    max_turns = int(policy.get("max_turns") or OLD_CONTEXT_SUMMARY_MAX_TURNS)
    keep_recent_turns = int(policy.get("keep_recent_turns") or OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS)
    max_summary_chars = int(policy.get("max_summary_chars") or OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS)
    max_source_chars = int(policy.get("max_source_chars") or OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS)
    excluded_categories = {str(item) for item in (policy.get("excluded_categories") or [])}
    block_tool_protocol = _as_bool(policy.get("block_tool_protocol"), OLD_CONTEXT_SUMMARY_BLOCK_TOOL_PROTOCOL)
    block_thinking = _as_bool(policy.get("block_thinking"), OLD_CONTEXT_SUMMARY_BLOCK_THINKING)
    if not enabled:
        return None, _summary_base_meta("skipped", "disabled", policy=policy, managed_profile=managed_profile)

    before_chars = len(stable_json(body))
    category = _crunch_request_category(body)
    if category in excluded_categories:
        meta = _summary_base_meta("skipped", "excluded-category", policy=policy, managed_profile=managed_profile)
        meta["before_chars"] = before_chars
        meta["category"] = category
        return None, meta

    if before_chars < min_request_chars:
        meta = _summary_base_meta("skipped", "request-too-small", policy=policy, managed_profile=managed_profile)
        meta["before_chars"] = before_chars
        meta["category"] = category
        return None, meta

    messages = body.get("messages") or []
    if not isinstance(messages, list) or len(messages) <= keep_recent_turns:
        meta = _summary_base_meta("skipped", "not-enough-old-turns", policy=policy, managed_profile=managed_profile)
        meta["before_chars"] = before_chars
        meta["category"] = category
        return None, meta

    old_limit = max(0, len(messages) - keep_recent_turns)
    if block_tool_protocol and any(
        isinstance(msg, dict) and _message_has_tool_protocol(msg) for msg in messages[:old_limit]
    ):
        meta = _summary_base_meta("skipped", "tool-protocol-context-blocked", policy=policy, managed_profile=managed_profile)
        meta["before_chars"] = before_chars
        meta["category"] = category
        return None, meta
    if block_thinking and any(
        isinstance(msg, dict) and _message_has_thinking(msg) for msg in messages[:old_limit]
    ):
        meta = _summary_base_meta("skipped", "thinking-context-blocked", policy=policy, managed_profile=managed_profile)
        meta["before_chars"] = before_chars
        meta["category"] = category
        return None, meta

    candidates: list[dict[str, Any]] = []
    source_parts: list[str] = []
    total_chars = 0
    source_truncated = False
    for idx, msg in enumerate(messages[:old_limit]):
        if len(candidates) >= max_turns:
            break
        if not isinstance(msg, dict):
            continue
        text = _non_tool_message_text(msg)
        if text is None:
            continue
        normalized = normalize_text(text)
        if not normalized:
            continue
        remaining = max_source_chars - total_chars
        if remaining <= 0:
            source_truncated = True
            break
        included = normalized[:remaining]
        if len(included) < len(normalized):
            source_truncated = True
        candidates.append({"index": idx, "role": msg.get("role"), "chars": len(normalized)})
        source_parts.append(f"<turn index=\"{idx}\" role=\"{msg.get('role')}\">\n{included}\n</turn>")
        total_chars += len(included)

    if total_chars < min_summarized_chars or not candidates:
        meta = _summary_base_meta("skipped", "eligible-context-too-small", policy=policy, managed_profile=managed_profile)
        meta["before_chars"] = before_chars
        meta["category"] = category
        meta["eligible_turns"] = len(candidates)
        meta["eligible_chars"] = total_chars
        return None, meta

    source_text = "\n\n".join(source_parts)
    source_hash = sha256_text(source_text)
    summary_request = {
        "model": model,
        "max_tokens": max(256, max_summary_chars // TOKEN_CHARS),
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
        "placement": placement,
        "candidate_indexes": [c["index"] for c in candidates],
        "candidate_roles": [c["role"] for c in candidates],
        "eligible_turns": len(candidates),
        "eligible_chars": total_chars,
        "source_truncated": source_truncated,
        "before_chars": before_chars,
        "category": category,
        "keep_recent_turns": keep_recent_turns,
        "summary_model": model,
        "max_summary_chars": max_summary_chars,
        "rule_id": rule_id,
        "candidate_id": str(candidate_id) if candidate_id is not None else None,
        "policy_source": str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE),
    }
    meta = _summary_base_meta("planned", "eligible", policy=policy, managed_profile=managed_profile)
    meta.update({
        "before_chars": before_chars,
        "category": category,
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
    summary_model = str(plan.get("summary_model") or OLD_CONTEXT_SUMMARY_MODEL)
    max_summary_chars = int(plan.get("max_summary_chars") or OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS)
    notice = (
        f"[AgentFlow: old non-tool context summarized by {summary_model}; "
        f"source_turns={plan['eligible_turns']}; source_chars={plan['eligible_chars']}; "
        f"source_hash={plan['source_hash'][:12]}]\n\n"
        f"{summary.strip()[:max_summary_chars]}"
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _summary_tool_protocol_fingerprints(body: dict[str, Any]) -> list[str]:
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return []
    fingerprints: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}:
                fingerprints.append(_canonical_json({
                    "role": msg.get("role"),
                    "block": block,
                }))
    return fingerprints


def _summary_recent_turn_fingerprints(body: dict[str, Any], keep_recent_turns: int) -> list[str]:
    if keep_recent_turns <= 0:
        return []
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return []
    return [_canonical_json(msg) for msg in messages[-keep_recent_turns:]]


def _old_context_summary_preservation_check(
    original: dict[str, Any],
    summarized: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    keep_recent_turns = int(plan.get("keep_recent_turns") or 0)
    original_tool_protocol = _summary_tool_protocol_fingerprints(original)
    summarized_tool_protocol = _summary_tool_protocol_fingerprints(summarized)
    original_recent = _summary_recent_turn_fingerprints(original, keep_recent_turns)
    summarized_recent = _summary_recent_turn_fingerprints(summarized, keep_recent_turns)
    tool_protocol_preserved = original_tool_protocol == summarized_tool_protocol
    recent_turns_preserved = original_recent == summarized_recent
    return {
        "schema": "agentflow.old_context_summary_preservation_check.v1",
        "ok": tool_protocol_preserved and recent_turns_preserved,
        "tool_protocol_blocks_preserved": tool_protocol_preserved,
        "recent_turns_preserved": recent_turns_preserved,
        "tool_protocol_block_count": len(original_tool_protocol),
        "recent_turn_count": len(original_recent),
        "raw_payload_included": False,
    }


def _summary_canary_decision(body: dict[str, Any], plan: dict[str, Any], *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or _effective_summary_policy(None)
    public = _summary_canary_public_meta(policy)
    rule_id = str(plan.get("rule_id") or policy.get("rule_id") or OLD_CONTEXT_SUMMARY_RULE_ID)
    candidate_id = plan.get("candidate_id")
    if candidate_id is None:
        candidate_id = policy.get("candidate_id")
    base: dict[str, Any] = {
        "schema": "agentflow.old_context_summary_canary_decision.v1",
        "enabled": public["enabled"],
        "selected": True,
        "status": "full",
        "cohort": "full",
        "rule_id": rule_id,
        "candidate_id": str(candidate_id) if candidate_id is not None else None,
        "source_hash": str(plan.get("source_hash") or "")[:12],
        "raw_context_included": False,
        "raw_summary_included": False,
    }
    if not public["enabled"]:
        return base

    fraction = max(0.0, min(1.0, float(public["fraction"])))
    unit = str(public["unit"] or "source_hash")
    request_hash = sha256_text(stable_json(body))
    if unit == "request_fingerprint":
        unit_value = request_hash
    elif unit == "category":
        unit_value = str(plan.get("category") or _crunch_request_category(body))
    else:
        unit = "source_hash"
        unit_value = str(plan.get("source_hash") or "")
    basis = {
        "unit": unit,
        "unit_hash": sha256_text(unit_value)[:16],
        "rule_id": rule_id,
        "candidate_id": str(candidate_id) if candidate_id is not None else None,
        "source_hash": str(plan.get("source_hash") or "")[:12],
        "category": str(plan.get("category") or ""),
        "requested_model": str(body.get("model") or ""),
        "eligible_turns": int(plan.get("eligible_turns") or 0),
        "eligible_chars_bucket": _text_bucket(int(plan.get("eligible_chars") or 0)),
        "request_chars_bucket": _text_bucket(int(plan.get("before_chars") or 0)),
    }
    material = {"salt": str(public["salt"] or ""), "basis": basis}
    digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    holdout_fraction = max(0.0, min(1.0, float(public.get("holdout_fraction", max(0.0, 1.0 - fraction)))))
    selected = bucket < fraction
    holdout = not selected and bucket < min(1.0, fraction + holdout_fraction)
    base.update({
        "selected": selected,
        "status": "applied" if selected else ("holdout" if holdout else "skipped"),
        "cohort": "canary_applied" if selected else ("canary_holdout" if holdout else "outside_canary_and_holdout"),
        "reason": None if selected else ("canary_holdout" if holdout else "outside-canary-and-holdout"),
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "salt": str(public["salt"] or ""),
        "unit": unit,
        "bucket": round(bucket, 8),
        "threshold": fraction,
        "holdout_threshold": min(1.0, fraction + holdout_fraction),
        "cohort_key_hash": "sha256:" + hashlib.sha256(_canonical_json(basis).encode("utf-8")).hexdigest(),
    })
    return base


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _summary_meta_from_crunch_json(raw: Any) -> dict[str, Any]:
    crunch_meta = _json_obj(raw)
    summary_meta = crunch_meta.get("old_context_summarization")
    return summary_meta if isinstance(summary_meta, dict) else {}


def _is_summary_canary_selected(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    return isinstance(canary, dict) and bool(canary.get("enabled")) and str(canary.get("cohort")) == "canary_applied"


def _is_summary_canary_holdout(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    return isinstance(canary, dict) and bool(canary.get("enabled")) and str(canary.get("cohort")) == "canary_holdout"


def _summary_failure(meta: dict[str, Any]) -> bool:
    status = str(meta.get("status") or "")
    reason = str(meta.get("reason") or "")
    status_code = meta.get("summary_status_code")
    try:
        if status_code is not None and int(status_code) >= 400:
            return True
    except (TypeError, ValueError):
        pass
    return reason in {"summary-empty", "summary-cost-too-high"} or status == "error"


def _rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def evaluate_old_context_summary_safety_stop(store_obj: Any | None) -> dict[str, Any] | None:
    safety = _summary_safety_public_meta()
    if not safety["enabled"]:
        return None
    if store_obj is None or not hasattr(store_obj, "conn"):
        return None
    window = max(1, min(int(safety["window"]), 10_000))
    try:
        rows = store_obj.conn.execute(
            """
            select status_code, retry_count, crunch_json
            from calls
            order by created_at desc
            limit ?
            """,
            (window,),
        ).fetchall()
    except Exception:
        return None

    applied = {"samples": 0, "errors": 0, "retries": 0, "negative_net_savings": 0, "summary_failures": 0}
    holdout = {"samples": 0, "errors": 0, "retries": 0}
    baseline = {"samples": 0, "errors": 0, "retries": 0}
    for row in rows:
        row_dict = dict(row)
        meta = _summary_meta_from_crunch_json(row_dict.get("crunch_json"))
        status_code = row_dict.get("status_code")
        retry_count = row_dict.get("retry_count")
        errored = False
        retried = False
        try:
            errored = status_code is not None and int(status_code) >= 400
        except (TypeError, ValueError):
            errored = False
        try:
            retried = retry_count is not None and int(retry_count) > 0
        except (TypeError, ValueError):
            retried = False
        if not meta or str(meta.get("rule_id") or "") != OLD_CONTEXT_SUMMARY_RULE_ID:
            baseline["samples"] += 1
            baseline["errors"] += int(errored)
            baseline["retries"] += int(retried)
            continue
        is_applied = _is_summary_canary_selected(meta)
        is_holdout = _is_summary_canary_holdout(meta)
        if not is_applied and not is_holdout:
            baseline["samples"] += 1
            baseline["errors"] += int(errored)
            baseline["retries"] += int(retried)
            continue
        if is_applied:
            applied["samples"] += 1
            applied["errors"] += int(errored)
            applied["retries"] += int(retried)
            applied["summary_failures"] += int(_summary_failure(meta))
            try:
                applied["negative_net_savings"] += int(float(meta.get("estimated_net_savings_usd") or 0.0) < 0.0)
            except (TypeError, ValueError):
                pass
        elif is_holdout:
            holdout["samples"] += 1
            holdout["errors"] += int(errored)
            holdout["retries"] += int(retried)

    samples = applied["samples"]
    min_samples = max(1, int(safety["min_outcome_samples"]))
    if samples < min_samples:
        return None

    applied_error_rate = _rate(applied["errors"], samples)
    applied_retry_rate = _rate(applied["retries"], samples)
    negative_net_rate = _rate(applied["negative_net_savings"], samples)
    summary_failure_rate = _rate(applied["summary_failures"], samples)
    holdout_error_rate = _rate(holdout["errors"], holdout["samples"])
    baseline_error_rate = _rate(baseline["errors"], baseline["samples"])
    triggers: list[dict[str, Any]] = []
    if applied_error_rate >= float(safety["max_error_rate"]):
        triggers.append({"metric": "error_rate", "value": round(applied_error_rate, 4), "threshold": safety["max_error_rate"]})
    if holdout["samples"] >= min_samples and (applied_error_rate - holdout_error_rate) >= float(safety["max_error_rate_delta"]):
        triggers.append({
            "metric": "error_rate_delta_vs_holdout",
            "value": round(applied_error_rate - holdout_error_rate, 4),
            "threshold": safety["max_error_rate_delta"],
        })
    if baseline["samples"] >= min_samples and (applied_error_rate - baseline_error_rate) >= float(safety["max_error_rate_delta"]):
        triggers.append({
            "metric": "error_rate_delta_vs_recent_baseline",
            "value": round(applied_error_rate - baseline_error_rate, 4),
            "threshold": safety["max_error_rate_delta"],
        })
    if applied_retry_rate >= float(safety["max_retry_rate"]):
        triggers.append({"metric": "retry_rate", "value": round(applied_retry_rate, 4), "threshold": safety["max_retry_rate"]})
    if negative_net_rate >= float(safety["max_negative_net_savings_rate"]):
        triggers.append({
            "metric": "negative_net_savings_rate",
            "value": round(negative_net_rate, 4),
            "threshold": safety["max_negative_net_savings_rate"],
        })
    if summary_failure_rate >= float(safety["max_summary_failure_rate"]):
        triggers.append({
            "metric": "summary_failure_rate",
            "value": round(summary_failure_rate, 4),
            "threshold": safety["max_summary_failure_rate"],
        })
    if not triggers:
        return None
    return {
        "schema": "agentflow.old_context_summary_safety_stop.v1",
        "stopped": True,
        "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
        "rule_id": OLD_CONTEXT_SUMMARY_RULE_ID,
        "candidate_id": str(OLD_CONTEXT_SUMMARY_CANDIDATE_ID) if OLD_CONTEXT_SUMMARY_CANDIDATE_ID is not None else None,
        "sample_count": samples,
        "holdout_sample_count": holdout["samples"],
        "baseline_sample_count": baseline["samples"],
        "error_count": applied["errors"],
        "retry_count": applied["retries"],
        "negative_net_savings_count": applied["negative_net_savings"],
        "summary_failure_count": applied["summary_failures"],
        "error_rate": round(applied_error_rate, 4),
        "holdout_error_rate": round(holdout_error_rate, 4),
        "baseline_error_rate": round(baseline_error_rate, 4),
        "retry_rate": round(applied_retry_rate, 4),
        "negative_net_savings_rate": round(negative_net_rate, 4),
        "summary_failure_rate": round(summary_failure_rate, 4),
        "min_outcome_samples": min_samples,
        "window": window,
        "triggers": triggers,
        "raw_payload_included": False,
    }


def _summary_input_savings_usd(model: str, tokens_saved: int) -> float:
    basis = pricing_basis(model or "claude-sonnet-4-6", provider="anthropic")
    input_price = float(basis.get("input_usd_per_million") or 0.0)
    return (max(0, int(tokens_saved)) / 1_000_000.0) * input_price


def _copy_terminal_output_compaction_policy() -> dict[str, Any]:
    return copy.deepcopy(TERMINAL_OUTPUT_COMPACTION_POLICY)


def _copy_anthropic_thinking_compaction_policy() -> dict[str, Any]:
    return copy.deepcopy(ANTHROPIC_THINKING_COMPACTION_POLICY)


def _anthropic_thinking_compaction_public_policy(
    policy: dict[str, Any] | None = None,
    *,
    include_rules: bool = True,
    include_salt: bool = True,
) -> dict[str, Any]:
    policy = policy or ANTHROPIC_THINKING_COMPACTION_POLICY
    canary = policy.get("canary") or {}
    safety = policy.get("safety_stop") or {}
    fraction = max(0.0, min(1.0, float(canary.get("fraction", 0.0))))
    holdout_fraction = max(0.0, min(1.0, float(canary.get("holdout_fraction", max(0.0, 1.0 - fraction)))))
    public = {
        "schema": "agentflow.anthropic_thinking_history_compaction_policy.v1",
        "enabled": _as_bool(policy.get("enabled"), False),
        "policy_source": str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE),
        "rule_path": CRUNCH_RULES_PATH,
        "rule_id": str(policy.get("rule_id") or "local-anthropic-thinking-history-compaction-canary"),
        "candidate_id": str(policy.get("candidate_id")) if policy.get("candidate_id") is not None else None,
        "action_id": str(policy.get("action_id")) if policy.get("action_id") is not None else None,
        "conditions": _sanitize_anthropic_thinking_compaction_conditions(policy.get("conditions")),
        "min_text_chars": int(policy.get("min_text_chars") or 8_000),
        "min_block_chars": int(policy.get("min_block_chars") or 2_000),
        "similarity_threshold": max(0.0, min(1.0, float(policy.get("similarity_threshold", 0.95)))),
        "replacement_notice": str(policy.get("replacement_notice") or "[AgentFlow: older duplicate thinking history compacted]"),
        "block_top_level_thinking": _as_bool(policy.get("block_top_level_thinking"), True),
        "canary": {
            "enabled": _as_bool(canary.get("enabled"), True),
            "fraction": fraction,
            "holdout_fraction": holdout_fraction,
            "salt": str(canary.get("salt") or "") if include_salt else "",
            "salt_included": bool(include_salt and canary.get("salt")),
            "salt_configured": bool(canary.get("salt")),
            "unit": str(canary.get("unit") or "thinking_block_local_fingerprint"),
            "validation_errors": sorted({
                public_label(item, "invalid-canary-configuration")
                for item in canary.get("validation_errors", [])
                if str(item)
            }) if isinstance(canary.get("validation_errors"), list) else [],
        },
        "safety_stop": {
            "enabled": _as_bool(safety.get("enabled"), True),
            "min_outcome_samples": int(safety.get("min_outcome_samples", 5)),
            "window": int(safety.get("window", 500)),
            "max_error_rate": float(safety.get("max_error_rate", 0.1)),
            "max_retry_rate": float(safety.get("max_retry_rate", 0.25)),
            "max_negative_savings_rate": float(safety.get("max_negative_savings_rate", 0.25)),
            "max_missing_usage_rate": float(safety.get("max_missing_usage_rate", 0.1)),
            "max_error_rate_delta": float(safety.get("max_error_rate_delta", 0.05)),
        },
    }
    public["action"] = {
        "type": "compact_thinking_history_block",
        "preserve_tool_protocol": True,
        "preserve_assistant_text_fallback": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    if include_rules:
        rules = policy.get("rules")
        public["rules"] = [
            _anthropic_thinking_compaction_public_policy(rule, include_rules=False, include_salt=include_salt)
            for rule in rules
            if isinstance(rule, dict)
        ] if isinstance(rules, list) else []
        public["rule_count"] = len(public["rules"])
    return public


def anthropic_thinking_compaction_effective_policy() -> dict[str, Any]:
    return _anthropic_thinking_compaction_public_policy(ANTHROPIC_THINKING_COMPACTION_POLICY, include_salt=False)


def _anthropic_thinking_compaction_base_meta(status: str, reason: str, *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    public = _anthropic_thinking_compaction_public_policy(policy, include_salt=False)
    return {
        "schema": "agentflow.anthropic_thinking_history_compaction_decision.v1",
        "enabled": public["enabled"],
        "status": status,
        "reason": reason,
        "changed": False,
        "applied": False,
        "policy_source": public["policy_source"],
        "rule_path": public["rule_path"],
        "rule_id": public["rule_id"],
        "candidate_id": public["candidate_id"],
        "action_id": public["action_id"],
        "conditions": public["conditions"],
        "canary": public["canary"],
        "safety_stop": public["safety_stop"],
        "raw_thinking_text_included": False,
        "thinking_block_fingerprints_included": False,
        "raw_request_body_included": False,
        "raw_tool_ids_included": False,
        "raw_session_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _anthropic_thinking_candidate_policies(base_policy: dict[str, Any]) -> list[dict[str, Any]]:
    rules = base_policy.get("rules")
    if isinstance(rules, list) and rules:
        return [copy.deepcopy(rule) for rule in rules if isinstance(rule, dict)]
    return [copy.deepcopy(base_policy)]


def _anthropic_thinking_compaction_features(
    body: dict[str, Any],
    *,
    category: str,
    before_chars: int,
    planned_saved_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "source_surface": "anthropic_messages",
        "category": category,
        "workflow_phase": category,
        "text_chars": before_chars,
        "text_bucket": _text_bucket(before_chars),
        "token_bucket": _token_bucket(max(1, before_chars // TOKEN_CHARS)),
        "requested_model": str(body.get("model") or ""),
        "has_tools": _body_has_tools(body),
        "stream": bool(body.get("stream")),
        "planned_saved_tokens": planned_saved_tokens,
        "expected_saved_token_bucket": _token_bucket(max(1, planned_saved_tokens)),
    }


def _anthropic_thinking_compaction_rule_matches(policy: dict[str, Any], features: dict[str, Any]) -> tuple[bool, list[str]]:
    conditions = _sanitize_anthropic_thinking_compaction_conditions(policy.get("conditions"))
    blockers: list[str] = []
    if not conditions:
        return True, blockers
    for key in ("source_surface", "category", "workflow_phase", "text_bucket", "token_bucket"):
        if key in conditions and str(features.get(key)) not in _condition_values(conditions[key]):
            blockers.append(f"{key}-not-matched")
    if "category_not_in" in conditions and str(features.get("category")) in _condition_values(conditions["category_not_in"]):
        blockers.append("category-excluded")
    if "requested_model" in conditions and str(features.get("requested_model")) not in _condition_values(conditions["requested_model"]):
        blockers.append("requested-model-not-matched")
    if "model_pattern" in conditions:
        patterns = _condition_values(conditions["model_pattern"])
        requested = str(features.get("requested_model") or "").lower()
        if not any(pattern.lower() in requested for pattern in patterns):
            blockers.append("model-pattern-not-matched")
    for key in ("has_tools", "stream"):
        if key in conditions and bool(features.get(key)) != _as_bool(conditions[key], False):
            blockers.append(f"{key}-not-matched")
    if "min_text_chars" in conditions and _safe_int(features.get("text_chars")) < _safe_int(conditions["min_text_chars"]):
        blockers.append("min-text-chars-not-met")
    if "max_text_chars" in conditions and _safe_int(features.get("text_chars")) > _safe_int(conditions["max_text_chars"]):
        blockers.append("max-text-chars-exceeded")
    return not blockers, blockers


def _anthropic_thinking_tool_ids(body: dict[str, Any]) -> tuple[set[str], set[str]]:
    tool_uses: set[str] = set()
    tool_results: set[str] = set()
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        tool_uses.update(_thinking_tool_use_ids(content))
        tool_results.update(_thinking_tool_result_ids(content))
    return tool_uses, tool_results


def _anthropic_thinking_compaction_targets(body: dict[str, Any], *, policy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return [], ["unsupported-content-block-shape"]
    assistant_indexes = [idx for idx, msg in enumerate(messages) if isinstance(msg, dict) and msg.get("role") == "assistant"]
    assistant_age_by_index = {
        msg_idx: len(assistant_indexes) - 1 - assistant_order
        for assistant_order, msg_idx in enumerate(assistant_indexes)
    }
    entries: list[dict[str, Any]] = []
    blockers: list[str] = []
    min_block_chars = int(policy.get("min_block_chars") or 2_000)
    similarity_threshold = float(policy.get("similarity_threshold") or 0.95)
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            blockers.append("unsupported-content-block-shape")
            continue
        assistant_text = _thinking_assistant_text_available(content)
        tool_use_ids = _thinking_tool_use_ids(content)
        next_message = messages[msg_idx + 1] if msg_idx + 1 < len(messages) else None
        next_results = set()
        if isinstance(next_message, dict) and isinstance(next_message.get("content"), list):
            next_results = _thinking_tool_result_ids(next_message["content"])
        unresolved_tool_use = bool(tool_use_ids and not tool_use_ids <= next_results)
        for block_idx, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") not in _ANTHROPIC_THINKING_BLOCK_TYPES:
                continue
            text = _thinking_block_text(block)
            block_blockers: set[str] = set()
            if block.get("type") == "redacted_thinking":
                block_blockers.add("redacted-thinking-block")
            if block.get("type") == "thinking" and not text:
                block_blockers.add("unsupported-content-block-shape")
            if not assistant_text:
                block_blockers.add("missing-assistant-text-fallback")
            if msg_idx >= len(messages) - 1:
                block_blockers.add("latest-message-not-old-history")
            if unresolved_tool_use:
                block_blockers.add("unresolved-tool-use-dependency")
            if len(text) < min_block_chars:
                block_blockers.add("thinking-block-below-min-chars")
            normalized = normalize_text(text)
            fingerprint = "sha256:" + sha256_text(normalized) if normalized else ""
            entries.append({
                "message_index": msg_idx,
                "block_index": block_idx,
                "chars": len(text),
                "fingerprint": fingerprint,
                "shingles": _shingles(normalized) if len(text) >= min_block_chars else frozenset(),
                "assistant_age": assistant_age_by_index.get(msg_idx, 0),
                "blockers": block_blockers,
            })
    for index, entry in enumerate(entries):
        duplicate_kind = "none"
        for newer in entries[index + 1:]:
            if not newer.get("fingerprint"):
                continue
            if entry.get("fingerprint") and entry["fingerprint"] == newer.get("fingerprint"):
                duplicate_kind = "exact"
                break
            if entry.get("shingles") and newer.get("shingles") and _jaccard(entry["shingles"], newer["shingles"]) >= similarity_threshold:
                duplicate_kind = "near"
                break
        entry["duplicate_kind"] = duplicate_kind
        if duplicate_kind == "none":
            entry["blockers"].add("no-newer-duplicate-thinking-block")
    targets = [
        entry for entry in entries
        if not entry["blockers"] and entry.get("duplicate_kind") in {"exact", "near"} and entry.get("chars", 0) > 0
    ]
    return targets, sorted(set(blockers))


def _anthropic_thinking_compaction_candidate_id(policy: dict[str, Any], targets: list[dict[str, Any]]) -> str:
    if policy.get("candidate_id"):
        return str(policy["candidate_id"])
    basis = {
        "rule_id": policy.get("rule_id"),
        "targets": [
            {
                "fingerprint": target.get("fingerprint"),
                "message_index": target.get("message_index"),
                "block_index": target.get("block_index"),
                "duplicate_kind": target.get("duplicate_kind"),
            }
            for target in targets[:24]
        ],
    }
    return "anthropic-thinking-compaction:" + sha256_text(stable_json(basis))[:20]


def _anthropic_thinking_compaction_canary_decision(
    body: dict[str, Any],
    *,
    policy: dict[str, Any],
    candidate_id: str,
    category: str,
    before_chars: int,
    planned_saved_tokens: int,
) -> dict[str, Any]:
    public = _anthropic_thinking_compaction_public_policy(policy)
    canary = public["canary"]
    rollout = {
        "schema": PATTERN_ROLLOUT_SCHEMA,
        "recommendation_mode": "canary-only",
        "canary_enabled": canary["enabled"],
        "canary_fraction": canary["fraction"],
        "canary_salt": canary["salt"],
        "canary_unit": canary["unit"],
    }
    features = _anthropic_thinking_compaction_features(
        body,
        category=category,
        before_chars=before_chars,
        planned_saved_tokens=planned_saved_tokens,
    )
    features["request_fingerprint"] = "sha256:" + sha256_text(stable_json(body))
    features["thinking_block_fingerprint"] = candidate_id
    decision = pattern_canary_decision(
        rollout=rollout,
        rule_id=public["rule_id"],
        candidate_id=candidate_id,
        pattern_hashes=[],
        features=features,
    )
    decision["schema"] = "agentflow.anthropic_thinking_history_compaction_canary_decision.v1"
    decision["holdout_fraction"] = canary["holdout_fraction"]
    decision.pop("salt", None)
    decision["salt_included"] = False
    decision["salt_configured"] = bool(canary.get("salt_configured"))
    decision["raw_request_body_included"] = False
    decision["raw_thinking_text_included"] = False
    decision["thinking_block_fingerprints_included"] = False
    return decision


def _anthropic_thinking_compaction_meta_from_crunch_json(raw: Any) -> dict[str, Any]:
    crunch_meta = _json_obj(raw)
    meta = crunch_meta.get("anthropic_thinking_history_compaction")
    return meta if isinstance(meta, dict) else {}


def _anthropic_thinking_compaction_is_applied(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    return bool(meta.get("applied")) and isinstance(canary, dict) and str(canary.get("cohort")) == "canary_applied"


def _anthropic_thinking_compaction_is_holdout(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    return isinstance(canary, dict) and str(canary.get("cohort")) == "canary_holdout"


def evaluate_anthropic_thinking_compaction_safety_stop(
    store_obj: Any | None,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    public = _anthropic_thinking_compaction_public_policy(policy)
    safety = public["safety_stop"]
    if not safety["enabled"]:
        return None
    if store_obj is None or not hasattr(store_obj, "conn"):
        return None
    window = max(1, min(int(safety["window"]), 10_000))
    try:
        rows = store_obj.conn.execute(
            """
            select status_code, retry_count, actual_input_tokens, actual_output_tokens, crunch_json
            from calls
            order by created_at desc
            limit ?
            """,
            (window,),
        ).fetchall()
    except Exception:
        return None

    applied = {"samples": 0, "errors": 0, "retries": 0, "negative_savings": 0, "missing_usage": 0}
    holdout = {"samples": 0, "errors": 0, "retries": 0}
    for row in rows:
        row_dict = dict(row)
        meta = _anthropic_thinking_compaction_meta_from_crunch_json(row_dict.get("crunch_json"))
        if not meta or str(meta.get("rule_id") or "") != public["rule_id"]:
            continue
        errored = _safe_int(row_dict.get("status_code")) >= 400
        retried = _safe_int(row_dict.get("retry_count")) > 0
        missing_usage = _safe_int(row_dict.get("actual_input_tokens")) <= 0 or _safe_int(row_dict.get("actual_output_tokens")) <= 0
        if _anthropic_thinking_compaction_is_applied(meta):
            applied["samples"] += 1
            applied["errors"] += int(errored)
            applied["retries"] += int(retried)
            applied["missing_usage"] += int(missing_usage)
            applied["negative_savings"] += int(_safe_int(meta.get("tokens_saved_est")) <= 0)
        elif _anthropic_thinking_compaction_is_holdout(meta):
            holdout["samples"] += 1
            holdout["errors"] += int(errored)
            holdout["retries"] += int(retried)

    samples = applied["samples"]
    min_samples = max(1, int(safety["min_outcome_samples"]))
    if samples < min_samples:
        return None
    error_rate = _rate(applied["errors"], samples)
    retry_rate = _rate(applied["retries"], samples)
    negative_savings_rate = _rate(applied["negative_savings"], samples)
    missing_usage_rate = _rate(applied["missing_usage"], samples)
    holdout_error_rate = _rate(holdout["errors"], holdout["samples"])
    triggers: list[dict[str, Any]] = []
    if error_rate >= float(safety["max_error_rate"]):
        triggers.append({"metric": "error_rate", "value": round(error_rate, 4), "threshold": safety["max_error_rate"]})
    if retry_rate >= float(safety["max_retry_rate"]):
        triggers.append({"metric": "retry_rate", "value": round(retry_rate, 4), "threshold": safety["max_retry_rate"]})
    if negative_savings_rate >= float(safety["max_negative_savings_rate"]):
        triggers.append({"metric": "negative_savings_rate", "value": round(negative_savings_rate, 4), "threshold": safety["max_negative_savings_rate"]})
    if missing_usage_rate >= float(safety["max_missing_usage_rate"]):
        triggers.append({"metric": "missing_usage_rate", "value": round(missing_usage_rate, 4), "threshold": safety["max_missing_usage_rate"]})
    if holdout["samples"] >= min_samples and (error_rate - holdout_error_rate) >= float(safety["max_error_rate_delta"]):
        triggers.append({
            "metric": "error_rate_delta_vs_holdout",
            "value": round(error_rate - holdout_error_rate, 4),
            "threshold": safety["max_error_rate_delta"],
        })
    if not triggers:
        return None
    return {
        "schema": "agentflow.anthropic_thinking_history_compaction_safety_stop.v1",
        "stopped": True,
        "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
        "rule_id": public["rule_id"],
        "candidate_id": public["candidate_id"],
        "sample_count": samples,
        "holdout_sample_count": holdout["samples"],
        "error_count": applied["errors"],
        "retry_count": applied["retries"],
        "negative_savings_count": applied["negative_savings"],
        "missing_usage_count": applied["missing_usage"],
        "error_rate": round(error_rate, 4),
        "retry_rate": round(retry_rate, 4),
        "negative_savings_rate": round(negative_savings_rate, 4),
        "missing_usage_rate": round(missing_usage_rate, 4),
        "holdout_error_rate": round(holdout_error_rate, 4),
        "min_outcome_samples": min_samples,
        "window": window,
        "triggers": triggers,
        "raw_payload_included": False,
    }


def _apply_anthropic_thinking_history_compaction_canary(
    body: dict[str, Any],
    *,
    store_obj: Any | None = None,
    policy_source: str,
    category: str,
    provider: str | None,
    source_surface: str | None,
    endpoint: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_policy = _copy_anthropic_thinking_compaction_policy()
    base_policy["policy_source"] = str(base_policy.get("policy_source") or policy_source)
    meta = _anthropic_thinking_compaction_base_meta("skipped", "disabled", policy=base_policy)
    before_chars = len(stable_json(body))
    meta["before_chars"] = before_chars
    meta["after_chars"] = before_chars
    meta["tokens_saved_est"] = 0
    if not _as_bool(base_policy.get("enabled"), False):
        return body, meta
    if str(provider or "anthropic").lower() != "anthropic":
        meta["reason"] = "non-anthropic-provider"
        return body, meta
    if source_surface and str(source_surface) != "anthropic_messages":
        meta["reason"] = "non-anthropic-source-surface"
        return body, meta
    if endpoint and str(endpoint) != "messages":
        meta["reason"] = "non-anthropic-messages-endpoint"
        return body, meta
    if category != "tool-result":
        meta["reason"] = "non-tool-result-category"
        meta["category"] = category
        return body, meta
    if before_chars < int(base_policy.get("min_text_chars") or 8_000):
        meta["reason"] = "below-min-text-size"
        return body, meta
    if _as_bool(base_policy.get("block_top_level_thinking"), True) and _thinking_top_level_active(body):
        meta["reason"] = "active-top-level-thinking-request"
        return body, meta

    initial_features = _anthropic_thinking_compaction_features(body, category=category, before_chars=before_chars)
    evaluated_rules: list[dict[str, Any]] = []
    selected_policy: dict[str, Any] | None = None
    selected_targets: list[dict[str, Any]] = []
    target_blockers: list[str] = []
    for raw_policy in _anthropic_thinking_candidate_policies(base_policy):
        policy = copy.deepcopy(raw_policy)
        policy["policy_source"] = str(policy.get("policy_source") or policy_source)
        public = _anthropic_thinking_compaction_public_policy(policy)
        rule_eval = {
            "rule_id": public["rule_id"],
            "candidate_id": public["candidate_id"],
            "action_id": public["action_id"],
            "enabled": public["enabled"],
            "policy_source": public["policy_source"],
            "conditions": public["conditions"],
            "status": "skipped",
            "reasons": [],
        }
        if not public["enabled"]:
            rule_eval["reasons"].append("disabled")
            evaluated_rules.append(rule_eval)
            continue
        matched, blockers = _anthropic_thinking_compaction_rule_matches(policy, initial_features)
        if not matched:
            rule_eval["reasons"].extend(blockers)
            evaluated_rules.append(rule_eval)
            continue
        targets, blockers = _anthropic_thinking_compaction_targets(body, policy=policy)
        if not targets:
            rule_eval["reasons"].extend(blockers or ["no-eligible-thinking-history-blocks"])
            evaluated_rules.append(rule_eval)
            target_blockers.extend(blockers)
            continue
        rule_eval.update({"status": "matched", "target_count": len(targets)})
        evaluated_rules.append(rule_eval)
        selected_policy = policy
        selected_targets = targets
        break
    meta["evaluated_rules"] = evaluated_rules
    meta["configured_rule_count"] = len(_anthropic_thinking_candidate_policies(base_policy))
    if selected_policy is None:
        first_reason = None
        for rule_eval in evaluated_rules:
            reasons = rule_eval.get("reasons") if isinstance(rule_eval, dict) else None
            if isinstance(reasons, list) and reasons:
                first_reason = str(reasons[0])
                break
        meta.update({
            "reason": first_reason or (target_blockers[0] if target_blockers else "no-eligible-thinking-history-blocks"),
            "category": category,
            "blockers": sorted(set(target_blockers)),
        })
        return body, meta

    policy = selected_policy
    public = _anthropic_thinking_compaction_public_policy(policy)
    validation_errors = public["canary"].get("validation_errors") or []
    if validation_errors:
        meta = _anthropic_thinking_compaction_base_meta("bypass", "policy-validation-error", policy=policy)
        meta.update({
            "category": category,
            "before_chars": before_chars,
            "after_chars": before_chars,
            "tokens_saved_est": 0,
            "validation_errors": validation_errors,
            "reason_codes": ["policy-validation-error", *validation_errors],
            "evaluated_rules": evaluated_rules,
            "configured_rule_count": len(_anthropic_thinking_candidate_policies(base_policy)),
            "lifecycle_feedback": {
                "schema": "agentflow.anthropic_thinking_history_compaction_lifecycle_feedback.v1",
                "status": "policy_validation_error",
                "cohort": "blocked",
                "candidate_id": None,
                "metadata_only": True,
                "raw_payload_included": False,
            },
        })
        return body, meta
    candidate_id = _anthropic_thinking_compaction_candidate_id(policy, selected_targets)
    planned_body = copy.deepcopy(body)
    messages = planned_body.get("messages")
    if not isinstance(messages, list):
        meta.update({"status": "bypass", "reason": "unsupported-content-block-shape"})
        return body, meta
    by_message: dict[int, set[int]] = {}
    for target in selected_targets:
        by_message.setdefault(int(target["message_index"]), set()).add(int(target["block_index"]))
    for msg_idx, block_indexes in by_message.items():
        message = messages[msg_idx]
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            meta.update({"status": "bypass", "reason": "unsupported-content-block-shape"})
            return body, meta
        message["content"] = [
            block for block_idx, block in enumerate(content)
            if block_idx not in block_indexes
        ]
        if not message["content"]:
            meta.update({"status": "bypass", "reason": "thinking-history-only-assistant-message"})
            return body, meta
    before_tool_uses, before_tool_results = _anthropic_thinking_tool_ids(body)
    after_tool_uses, after_tool_results = _anthropic_thinking_tool_ids(planned_body)
    if before_tool_uses != after_tool_uses or before_tool_results != after_tool_results:
        meta.update({"status": "bypass", "reason": "tool-protocol-reconstruction-mismatch"})
        return body, meta
    if body.get("stream") != planned_body.get("stream") or body.get("model") != planned_body.get("model"):
        meta.update({"status": "bypass", "reason": "provider-protocol-shape-mismatch"})
        return body, meta
    planned_after = len(stable_json(planned_body))
    planned_saved = before_chars - planned_after
    if planned_saved <= 0:
        meta.update({"status": "bypass", "reason": "compaction-savings-anomaly"})
        return body, meta
    planned_saved_tokens = planned_saved // TOKEN_CHARS
    target_summaries = [
        {
            "kind": "assistant_thinking_history",
            "duplicate_kind": str(target.get("duplicate_kind") or "unknown"),
            "assistant_age_bucket": _assistant_age_bucket(_safe_int(target.get("assistant_age"))),
            "size_bucket": _thinking_size_bucket(_safe_int(target.get("chars"))),
            "before_chars": _safe_int(target.get("chars")),
            "fingerprint_present": bool(target.get("fingerprint")),
            "fingerprint_included": False,
            "raw_text_included": False,
        }
        for target in selected_targets[:20]
    ]
    meta = _anthropic_thinking_compaction_base_meta("planned", "eligible", policy=policy)
    meta.update({
        "candidate_id": candidate_id,
        "category": category,
        "target_count": len(selected_targets),
        "before_chars": before_chars,
        "planned_after_chars": planned_after,
        "planned_saved_chars": planned_saved,
        "planned_saved_tokens": planned_saved_tokens,
        "after_chars": before_chars,
        "tokens_saved_est": 0,
        "target_summaries": target_summaries,
        "evaluated_rules": evaluated_rules,
        "configured_rule_count": len(_anthropic_thinking_candidate_policies(base_policy)),
        "lifecycle_feedback": {
            "schema": "agentflow.anthropic_thinking_history_compaction_lifecycle_feedback.v1",
            "status": "planned",
            "cohort": "pending",
            "candidate_id": public_id(candidate_id, prefix="candidate"),
            "metadata_only": True,
            "raw_payload_included": False,
        },
    })

    canary = _anthropic_thinking_compaction_canary_decision(
        body,
        policy=policy,
        candidate_id=candidate_id,
        category=category,
        before_chars=before_chars,
        planned_saved_tokens=planned_saved_tokens,
    )
    meta["canary"] = canary
    if canary.get("enabled") and not canary.get("selected", True):
        cohort = str(canary.get("cohort") or "canary_holdout")
        meta.update({
            "status": "holdout",
            "reason": str(canary.get("reason") or "canary_holdout"),
            "holdout": True,
            "after_chars": before_chars,
            "tokens_saved_est": 0,
        })
        meta["lifecycle_feedback"].update({"status": "holdout", "cohort": cohort})
        return body, meta

    safety_stop = evaluate_anthropic_thinking_compaction_safety_stop(store_obj, policy=policy)
    if safety_stop:
        meta.update({
            "status": "bypass",
            "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
            "safety_stop_state": "stopped",
            "safety_stop": safety_stop,
            "after_chars": before_chars,
            "tokens_saved_est": 0,
        })
        meta["lifecycle_feedback"].update({"status": "safety_stop", "cohort": "safety_stop"})
        log_pattern_canary_safety_stop(safety_stop)
        return body, meta

    meta.update({
        "status": "applied",
        "reason": "thinking-history-compaction-applied",
        "changed": True,
        "applied": True,
        "after_chars": planned_after,
        "saved_chars": planned_saved,
        "tokens_saved_est": planned_saved_tokens,
        "compaction_cost_usd": 0.0,
    })
    meta["lifecycle_feedback"].update({"status": "applied", "cohort": str(canary.get("cohort") or "canary_applied")})
    return planned_body, meta


def _terminal_output_compaction_action_public(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "compact_terminal_output",
        "keep_recent_turns": int(policy.get("keep_recent_turns") or TERMINAL_COMPACTION_DEFAULT_KEEP_RECENT_TURNS),
        "min_block_chars": int(policy.get("min_block_chars") or TERMINAL_COMPACTION_DEFAULT_MIN_BLOCK_CHARS),
        "head_lines": int(policy.get("head_lines") or TERMINAL_COMPACTION_DEFAULT_HEAD_LINES),
        "tail_lines": int(policy.get("tail_lines") or TERMINAL_COMPACTION_DEFAULT_TAIL_LINES),
        "max_evidence_lines": int(policy.get("max_evidence_lines") or TERMINAL_COMPACTION_DEFAULT_MAX_EVIDENCE_LINES),
        "min_saved_chars": int(policy.get("min_saved_chars") or TERMINAL_COMPACTION_DEFAULT_MIN_SAVED_CHARS),
        "preserve_diagnostics": True,
        "preserve_tool_protocol": True,
        "preserve_recent_turns": True,
    }


def _terminal_output_compaction_public_policy(
    policy: dict[str, Any] | None = None,
    *,
    include_rules: bool = True,
    include_salt: bool = True,
) -> dict[str, Any]:
    policy = policy or TERMINAL_OUTPUT_COMPACTION_POLICY
    canary = policy.get("canary") or {}
    safety = policy.get("safety_stop") or {}
    fraction = max(0.0, min(1.0, float(canary.get("fraction", 0.0))))
    holdout_fraction = max(0.0, min(1.0, float(canary.get("holdout_fraction", max(0.0, 1.0 - fraction)))))
    public = {
        "schema": "agentflow.terminal_output_compaction_policy.v1",
        "enabled": _as_bool(policy.get("enabled"), False),
        "policy_source": str(policy.get("policy_source") or CRUNCH_POLICY_SOURCE),
        "rule_path": CRUNCH_RULES_PATH,
        "rule_id": str(policy.get("rule_id") or "local-terminal-output-compaction-canary"),
        "candidate_id": str(policy.get("candidate_id")) if policy.get("candidate_id") is not None else None,
        "action_id": str(policy.get("action_id")) if policy.get("action_id") is not None else None,
        "conditions": _sanitize_terminal_output_compaction_conditions(policy.get("conditions")),
        "provenance": _sanitize_terminal_output_compaction_provenance(policy.get("provenance")),
        "keep_recent_turns": int(policy.get("keep_recent_turns") or TERMINAL_COMPACTION_DEFAULT_KEEP_RECENT_TURNS),
        "min_block_chars": int(policy.get("min_block_chars") or TERMINAL_COMPACTION_DEFAULT_MIN_BLOCK_CHARS),
        "head_lines": int(policy.get("head_lines") or TERMINAL_COMPACTION_DEFAULT_HEAD_LINES),
        "tail_lines": int(policy.get("tail_lines") or TERMINAL_COMPACTION_DEFAULT_TAIL_LINES),
        "max_evidence_lines": int(policy.get("max_evidence_lines") or TERMINAL_COMPACTION_DEFAULT_MAX_EVIDENCE_LINES),
        "min_saved_chars": int(policy.get("min_saved_chars") or TERMINAL_COMPACTION_DEFAULT_MIN_SAVED_CHARS),
        "block_thinking": _as_bool(policy.get("block_thinking"), True),
        "canary": {
            "enabled": _as_bool(canary.get("enabled"), True),
            "fraction": fraction,
            "holdout_fraction": holdout_fraction,
            "salt": str(canary.get("salt") or "") if include_salt else "",
            "salt_configured": bool(canary.get("salt")),
            "unit": str(canary.get("unit") or "request_fingerprint"),
        },
        "safety_stop": {
            "enabled": _as_bool(safety.get("enabled"), True),
            "min_outcome_samples": int(safety.get("min_outcome_samples", 5)),
            "window": int(safety.get("window", 500)),
            "max_error_rate": float(safety.get("max_error_rate", 0.1)),
            "max_retry_rate": float(safety.get("max_retry_rate", 0.25)),
            "max_negative_savings_rate": float(safety.get("max_negative_savings_rate", 0.25)),
            "max_error_rate_delta": float(safety.get("max_error_rate_delta", 0.05)),
        },
    }
    public["action"] = _terminal_output_compaction_action_public(public)
    if include_rules:
        rules = policy.get("rules")
        public["rules"] = [
            _terminal_output_compaction_public_policy(rule, include_rules=False, include_salt=include_salt)
            for rule in rules
            if isinstance(rule, dict)
        ] if isinstance(rules, list) else []
        public["rule_count"] = len(public["rules"])
    return public


def terminal_output_compaction_effective_policy() -> dict[str, Any]:
    """Return sanitized file-backed terminal compaction policy metadata."""
    public = _terminal_output_compaction_public_policy(TERMINAL_OUTPUT_COMPACTION_POLICY, include_salt=False)
    public["raw_terminal_text_included"] = False
    public["raw_request_body_included"] = False
    public["raw_tool_ids_included"] = False
    public["policy_file_contents_included"] = False
    return public


def _terminal_output_compaction_base_meta(status: str, reason: str, *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    public = _terminal_output_compaction_public_policy(policy)
    return {
        "schema": "agentflow.terminal_output_compaction_decision.v1",
        "enabled": public["enabled"],
        "status": status,
        "reason": reason,
        "changed": False,
        "applied": False,
        "policy_source": public["policy_source"],
        "rule_path": public["rule_path"],
        "rule_id": public["rule_id"],
        "candidate_id": public["candidate_id"],
        "action_id": public["action_id"],
        "conditions": public["conditions"],
        "provenance": public["provenance"],
        "canary": public["canary"],
        "safety_stop": public["safety_stop"],
        "raw_terminal_text_included": False,
        "raw_request_body_included": False,
        "raw_tool_ids_included": False,
        "raw_session_ids_included": False,
    }


def _terminal_output_compaction_canary_decision(
    body: dict[str, Any],
    plan: dict[str, Any],
    *,
    policy: dict[str, Any],
    category: str,
) -> dict[str, Any]:
    public = _terminal_output_compaction_public_policy(policy)
    canary = public["canary"]
    rollout = {
        "schema": PATTERN_ROLLOUT_SCHEMA,
        "recommendation_mode": "canary-only",
        "canary_enabled": canary["enabled"],
        "canary_fraction": canary["fraction"],
        "canary_salt": canary["salt"],
        "canary_unit": canary["unit"],
    }
    features = {
        "source_surface": "anthropic_messages",
        "app_family": "claude_code",
        "category": category,
        "workflow_phase": category,
        "text_bucket": _text_bucket(int(plan.get("before_chars") or 0)),
        "token_bucket": _token_bucket(max(1, int(plan.get("before_chars") or 0) // TOKEN_CHARS)),
        "requested_model": str(body.get("model") or ""),
        "candidate_target_model": str(body.get("model") or ""),
        "has_tools": True,
        "stream": bool(body.get("stream")),
        "request_fingerprint": "sha256:" + sha256_text(stable_json(body)),
    }
    decision = pattern_canary_decision(
        rollout=rollout,
        rule_id=public["rule_id"],
        candidate_id=public["candidate_id"],
        pattern_hashes=[],
        features=features,
    )
    decision["schema"] = "agentflow.terminal_output_compaction_canary_decision.v1"
    decision["holdout_fraction"] = canary["holdout_fraction"]
    decision["raw_request_body_included"] = False
    decision["raw_terminal_text_included"] = False
    return decision


def _terminal_output_compaction_meta_from_crunch_json(raw: Any) -> dict[str, Any]:
    crunch_meta = _json_obj(raw)
    meta = crunch_meta.get("terminal_output_compaction")
    return meta if isinstance(meta, dict) else {}


def _terminal_output_compaction_is_applied(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    return bool(meta.get("applied")) and isinstance(canary, dict) and str(canary.get("cohort")) == "canary_applied"


def _terminal_output_compaction_is_holdout(meta: dict[str, Any]) -> bool:
    canary = meta.get("canary")
    return isinstance(canary, dict) and str(canary.get("cohort")) == "canary_holdout"


def evaluate_terminal_output_compaction_safety_stop(
    store_obj: Any | None,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    public = _terminal_output_compaction_public_policy(policy)
    safety = public["safety_stop"]
    if not safety["enabled"]:
        return None
    if store_obj is None or not hasattr(store_obj, "conn"):
        return None
    window = max(1, min(int(safety["window"]), 10_000))
    try:
        rows = store_obj.conn.execute(
            """
            select status_code, retry_count, crunch_json
            from calls
            order by created_at desc
            limit ?
            """,
            (window,),
        ).fetchall()
    except Exception:
        return None

    applied = {"samples": 0, "errors": 0, "retries": 0, "negative_savings": 0}
    holdout = {"samples": 0, "errors": 0, "retries": 0}
    for row in rows:
        row_dict = dict(row)
        meta = _terminal_output_compaction_meta_from_crunch_json(row_dict.get("crunch_json"))
        if not meta or str(meta.get("rule_id") or "") != public["rule_id"]:
            continue
        errored = _safe_int(row_dict.get("status_code")) >= 400
        retried = _safe_int(row_dict.get("retry_count")) > 0
        if _terminal_output_compaction_is_applied(meta):
            applied["samples"] += 1
            applied["errors"] += int(errored)
            applied["retries"] += int(retried)
            applied["negative_savings"] += int(_safe_int(meta.get("tokens_saved_est")) <= 0)
        elif _terminal_output_compaction_is_holdout(meta):
            holdout["samples"] += 1
            holdout["errors"] += int(errored)
            holdout["retries"] += int(retried)

    samples = applied["samples"]
    min_samples = max(1, int(safety["min_outcome_samples"]))
    if samples < min_samples:
        return None
    error_rate = _rate(applied["errors"], samples)
    retry_rate = _rate(applied["retries"], samples)
    negative_savings_rate = _rate(applied["negative_savings"], samples)
    holdout_error_rate = _rate(holdout["errors"], holdout["samples"])
    triggers: list[dict[str, Any]] = []
    if error_rate >= float(safety["max_error_rate"]):
        triggers.append({"metric": "error_rate", "value": round(error_rate, 4), "threshold": safety["max_error_rate"]})
    if retry_rate >= float(safety["max_retry_rate"]):
        triggers.append({"metric": "retry_rate", "value": round(retry_rate, 4), "threshold": safety["max_retry_rate"]})
    if negative_savings_rate >= float(safety["max_negative_savings_rate"]):
        triggers.append({
            "metric": "negative_savings_rate",
            "value": round(negative_savings_rate, 4),
            "threshold": safety["max_negative_savings_rate"],
        })
    if holdout["samples"] >= min_samples and (error_rate - holdout_error_rate) >= float(safety["max_error_rate_delta"]):
        triggers.append({
            "metric": "error_rate_delta_vs_holdout",
            "value": round(error_rate - holdout_error_rate, 4),
            "threshold": safety["max_error_rate_delta"],
        })
    if not triggers:
        return None
    return {
        "schema": "agentflow.terminal_output_compaction_safety_stop.v1",
        "stopped": True,
        "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
        "rule_id": public["rule_id"],
        "candidate_id": public["candidate_id"],
        "sample_count": samples,
        "holdout_sample_count": holdout["samples"],
        "error_count": applied["errors"],
        "retry_count": applied["retries"],
        "negative_savings_count": applied["negative_savings"],
        "error_rate": round(error_rate, 4),
        "retry_rate": round(retry_rate, 4),
        "negative_savings_rate": round(negative_savings_rate, 4),
        "holdout_error_rate": round(holdout_error_rate, 4),
        "min_outcome_samples": min_samples,
        "window": window,
        "triggers": triggers,
        "raw_payload_included": False,
    }


def _terminal_output_compaction_candidate_policies(base_policy: dict[str, Any]) -> list[dict[str, Any]]:
    rules = base_policy.get("rules")
    if isinstance(rules, list) and rules:
        return [copy.deepcopy(rule) for rule in rules if isinstance(rule, dict)]
    return [copy.deepcopy(base_policy)]


def _terminal_output_compaction_features(
    body: dict[str, Any],
    *,
    category: str,
    before_chars: int,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "source_surface": "anthropic_messages",
        "app_family": "claude_code",
        "category": category,
        "workflow_phase": category,
        "text_chars": before_chars,
        "text_bucket": _text_bucket(before_chars),
        "token_bucket": _token_bucket(max(1, before_chars // TOKEN_CHARS)),
        "requested_model": str(body.get("model") or ""),
        "candidate_target_model": str(body.get("model") or ""),
        "has_tools": _body_has_tools(body),
        "stream": bool(body.get("stream")),
        "uses_thinking": _body_uses_thinking(body),
    }
    if isinstance(plan, dict):
        targets = [target for target in (plan.get("targets") or []) if isinstance(target, dict)]
        planned_saved_tokens = _safe_int(plan.get("estimated_saved_tokens"))
        buckets = sorted({
            str(target.get("terminal_output_char_fraction_bucket"))
            for target in targets
            if target.get("terminal_output_char_fraction_bucket") is not None
        })
        features.update({
            "target_count": len(targets),
            "planned_saved_tokens": planned_saved_tokens,
            "expected_saved_token_bucket": _token_bucket(max(1, planned_saved_tokens)),
            "terminal_output_char_fraction_buckets": buckets,
        })
        features["expected_saved_token_buckets"] = [features["expected_saved_token_bucket"]]
        if buckets:
            features["terminal_output_char_fraction_bucket"] = buckets[-1]
    return features


def _condition_values(value: Any) -> set[str]:
    raw_items = value if isinstance(value, list) else [value]
    return {str(item) for item in raw_items if item is not None}


def _terminal_output_compaction_rule_matches(
    policy: dict[str, Any],
    features: dict[str, Any],
    *,
    require_plan_conditions: bool,
) -> tuple[bool, list[str]]:
    conditions = _sanitize_terminal_output_compaction_conditions(policy.get("conditions"))
    blockers: list[str] = []
    if not conditions:
        return True, blockers

    for key in ("source_surface", "app_family", "category", "workflow_phase", "text_bucket", "token_bucket"):
        if key in conditions and str(features.get(key)) not in _condition_values(conditions[key]):
            blockers.append(f"{key}-not-matched")
    if "labels" in conditions:
        expected_labels = _condition_values(conditions["labels"])
        observed_labels = _condition_values(features.get("labels"))
        if not expected_labels.issubset(observed_labels):
            blockers.append("labels-not-matched")
    if "category_not_in" in conditions and str(features.get("category")) in _condition_values(conditions["category_not_in"]):
        blockers.append("category-excluded")
    if "requested_model" in conditions and str(features.get("requested_model")) not in _condition_values(conditions["requested_model"]):
        blockers.append("requested-model-not-matched")
    if "model_pattern" in conditions:
        patterns = _condition_values(conditions["model_pattern"])
        requested = str(features.get("requested_model") or "").lower()
        if not any(pattern.lower() in requested for pattern in patterns):
            blockers.append("model-pattern-not-matched")
    for key in ("has_tools", "stream", "uses_thinking"):
        if key in conditions and bool(features.get(key)) != _as_bool(conditions[key], False):
            blockers.append(f"{key}-not-matched")
    if "min_text_chars" in conditions and _safe_int(features.get("text_chars")) < _safe_int(conditions["min_text_chars"]):
        blockers.append("min-text-chars-not-met")
    if "max_text_chars" in conditions and _safe_int(features.get("text_chars")) > _safe_int(conditions["max_text_chars"]):
        blockers.append("max-text-chars-exceeded")
    if "terminal_output_char_fraction_bucket" in conditions or "terminal_output_char_fraction_buckets" in conditions:
        if not require_plan_conditions:
            return not blockers, blockers
        expected = _condition_values(
            conditions.get("terminal_output_char_fraction_bucket")
            if conditions.get("terminal_output_char_fraction_bucket") is not None
            else conditions.get("terminal_output_char_fraction_buckets")
        )
        observed = _condition_values(features.get("terminal_output_char_fraction_buckets"))
        if not expected.intersection(observed):
            blockers.append("terminal-output-fraction-bucket-not-matched")
    if "min_saved_tokens" in conditions:
        if not require_plan_conditions:
            return not blockers, blockers
        if _safe_int(features.get("planned_saved_tokens")) < _safe_int(conditions["min_saved_tokens"]):
            blockers.append("min-saved-tokens-not-met")
    if "expected_saved_token_bucket" in conditions or "expected_saved_token_buckets" in conditions:
        if not require_plan_conditions:
            return not blockers, blockers
        expected = _condition_values(
            conditions.get("expected_saved_token_bucket")
            if conditions.get("expected_saved_token_bucket") is not None
            else conditions.get("expected_saved_token_buckets")
        )
        observed = _condition_values(features.get("expected_saved_token_buckets"))
        if not expected.intersection(observed):
            blockers.append("expected-saved-token-bucket-not-matched")
    return not blockers, blockers


def _terminal_output_compaction_rule_sort_key(policy: dict[str, Any]) -> tuple[float, float, str]:
    public = _terminal_output_compaction_public_policy(policy)
    canary = public["canary"]
    return (
        float(canary.get("fraction") or 0.0),
        -float(canary.get("holdout_fraction") or 0.0),
        str(public.get("rule_id") or ""),
    )


def _select_terminal_output_compaction_policy(
    body: dict[str, Any],
    *,
    base_policy: dict[str, Any],
    category: str,
    before_chars: int,
    policy_source: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    candidates = _terminal_output_compaction_candidate_policies(base_policy)
    initial_features = _terminal_output_compaction_features(body, category=category, before_chars=before_chars)
    evaluated: list[dict[str, Any]] = []
    applicable: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for raw_policy in candidates:
        policy = copy.deepcopy(raw_policy)
        policy["policy_source"] = str(policy.get("policy_source") or policy_source)
        public = _terminal_output_compaction_public_policy(policy)
        rule_eval = {
            "rule_id": public["rule_id"],
            "candidate_id": public["candidate_id"],
            "action_id": public["action_id"],
            "enabled": public["enabled"],
            "policy_source": public["policy_source"],
            "conditions": public["conditions"],
            "status": "skipped",
            "reasons": [],
        }
        if not public["enabled"]:
            rule_eval["reasons"].append("disabled")
            evaluated.append(rule_eval)
            continue
        matched, blockers = _terminal_output_compaction_rule_matches(
            policy,
            initial_features,
            require_plan_conditions=False,
        )
        if not matched:
            rule_eval["reasons"].extend(blockers)
            evaluated.append(rule_eval)
            continue
        if public["block_thinking"] and bool(initial_features.get("uses_thinking")):
            rule_eval["reasons"].append("active-thinking-blocked")
            evaluated.append(rule_eval)
            continue
        plan, plan_meta = plan_terminal_output_compaction(
            body,
            keep_recent_turns=public["keep_recent_turns"],
            min_block_chars=public["min_block_chars"],
            head_lines=public["head_lines"],
            tail_lines=public["tail_lines"],
            max_evidence_lines=public["max_evidence_lines"],
            min_saved_chars=public["min_saved_chars"],
            policy_source=public["policy_source"],
        )
        if plan is None:
            rule_eval["reasons"].append(str(plan_meta.get("reason") or "not-eligible"))
            if plan_meta.get("blocker_counts"):
                rule_eval["blocker_counts"] = plan_meta.get("blocker_counts")
            evaluated.append(rule_eval)
            continue
        final_features = _terminal_output_compaction_features(
            body,
            category=category,
            before_chars=before_chars,
            plan=plan,
        )
        matched, blockers = _terminal_output_compaction_rule_matches(
            policy,
            final_features,
            require_plan_conditions=True,
        )
        if not matched:
            rule_eval["reasons"].extend(blockers)
            evaluated.append(rule_eval)
            continue
        rule_eval.update({
            "status": "matched",
            "planned_saved_tokens": _safe_int(plan.get("estimated_saved_tokens")),
            "target_count": _safe_int(plan.get("target_count")),
        })
        evaluated.append(rule_eval)
        applicable.append((policy, plan, plan_meta))

    if not applicable:
        return None, None, initial_features, evaluated
    applicable.sort(key=lambda item: _terminal_output_compaction_rule_sort_key(item[0]))
    selected_policy, selected_plan, selected_meta = applicable[0]
    return selected_policy, selected_plan, selected_meta, evaluated


def _apply_terminal_output_compaction_canary(
    body: dict[str, Any],
    *,
    store_obj: Any | None = None,
    policy_source: str,
    category: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_policy = _copy_terminal_output_compaction_policy()
    base_policy["policy_source"] = str(base_policy.get("policy_source") or policy_source)
    meta = _terminal_output_compaction_base_meta("skipped", "disabled", policy=base_policy)
    before_chars = len(stable_json(body))
    meta["before_chars"] = before_chars
    meta["after_chars"] = before_chars
    meta["tokens_saved_est"] = 0
    if not _as_bool(base_policy.get("enabled"), False):
        return body, meta
    if category != "tool-result":
        meta["reason"] = "non-tool-result-category"
        meta["category"] = category
        return body, meta

    policy, plan, plan_meta, evaluated_rules = _select_terminal_output_compaction_policy(
        body,
        base_policy=base_policy,
        category=category,
        before_chars=before_chars,
        policy_source=policy_source,
    )
    meta["evaluated_rules"] = evaluated_rules
    meta["configured_rule_count"] = len(_terminal_output_compaction_candidate_policies(base_policy))
    if plan is None:
        if policy is not None:
            meta = _terminal_output_compaction_base_meta("skipped", "not-eligible", policy=policy)
            meta["before_chars"] = before_chars
            meta["after_chars"] = before_chars
            meta["tokens_saved_est"] = 0
            meta["evaluated_rules"] = evaluated_rules
            meta["configured_rule_count"] = len(_terminal_output_compaction_candidate_policies(base_policy))
        first_reason = None
        for rule_eval in evaluated_rules:
            reasons = rule_eval.get("reasons") if isinstance(rule_eval, dict) else None
            if isinstance(reasons, list) and reasons:
                first_reason = str(reasons[0])
                break
        meta.update({
            "reason": (
                "no-conditional-rule-matched"
                if (base_policy.get("rules") or []) and first_reason not in {"active-thinking-blocked"}
                else (first_reason or str((plan_meta or {}).get("reason") or "not-eligible"))
            ),
            "category": category,
            "blocker_counts": (plan_meta or {}).get("blocker_counts", []) if isinstance(plan_meta, dict) else [],
        })
        return body, meta
    meta = _terminal_output_compaction_base_meta("skipped", "planned", policy=policy)
    meta["before_chars"] = before_chars
    meta["after_chars"] = before_chars
    meta["tokens_saved_est"] = 0
    meta["evaluated_rules"] = evaluated_rules
    meta["configured_rule_count"] = len(_terminal_output_compaction_candidate_policies(base_policy))

    target_summaries = [
        {
            "target_id": target.get("target_id"),
            "kind": target.get("kind"),
            "terminal_output_char_fraction_bucket": target.get("terminal_output_char_fraction_bucket"),
            "before_chars": target.get("before_chars"),
            "after_chars": target.get("after_chars"),
            "saved_chars": target.get("saved_chars"),
            "estimated_saved_tokens": target.get("estimated_saved_tokens"),
            "line_count": target.get("line_count"),
            "preserved_line_count": target.get("preserved_line_count"),
            "omitted_line_count": target.get("omitted_line_count"),
            "source_evidence_counts": target.get("source_evidence_counts"),
            "preserved_evidence_counts": target.get("preserved_evidence_counts"),
            "preservation_flags": target.get("preservation_flags"),
        }
        for target in plan.get("targets") or []
        if isinstance(target, dict)
    ]
    meta.update({
        "status": "planned",
        "reason": "eligible",
        "category": category,
        "target_count": int(plan.get("target_count") or 0),
        "before_chars": int(plan.get("before_chars") or before_chars),
        "planned_after_chars": int(plan.get("after_chars") or before_chars),
        "planned_saved_chars": int(plan.get("saved_chars") or 0),
        "planned_saved_tokens": int(plan.get("estimated_saved_tokens") or 0),
        "preservation_flags": plan.get("preservation_flags") or {},
        "target_summaries": target_summaries,
    })
    preservation_flags = plan.get("preservation_flags") if isinstance(plan.get("preservation_flags"), dict) else {}
    if not preservation_flags or not all(bool(value) for value in preservation_flags.values()):
        meta.update({"status": "bypass", "reason": "preservation-check-failed"})
        return body, meta
    planned_body = plan_meta.get("planned_body")
    if not isinstance(planned_body, dict):
        meta.update({"status": "bypass", "reason": "malformed-planned-body"})
        return body, meta
    if body.get("stream") != planned_body.get("stream"):
        meta.update({"status": "bypass", "reason": "streaming-protocol-mismatch"})
        return body, meta
    planned_after = len(stable_json(planned_body))
    planned_saved = before_chars - planned_after
    if planned_saved <= 0 or int(plan.get("estimated_saved_tokens") or 0) <= 0:
        meta.update({"status": "bypass", "reason": "compaction-savings-anomaly"})
        return body, meta

    canary = _terminal_output_compaction_canary_decision(body, plan, policy=policy, category=category)
    meta["canary"] = canary
    if canary.get("enabled") and not canary.get("selected", True):
        meta.update({
            "status": "holdout",
            "reason": str(canary.get("reason") or "canary_holdout"),
            "holdout": True,
            "after_chars": before_chars,
            "tokens_saved_est": 0,
        })
        return body, meta

    safety_stop = evaluate_terminal_output_compaction_safety_stop(store_obj, policy=policy)
    if safety_stop:
        meta.update({
            "status": "bypass",
            "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
            "safety_stop_state": "stopped",
            "safety_stop": safety_stop,
            "after_chars": before_chars,
            "tokens_saved_est": 0,
        })
        try:
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "terminal-output-compaction-safety-stop",
                ok=True,
                details={
                    key: safety_stop.get(key)
                    for key in (
                        "schema",
                        "reason",
                        "rule_id",
                        "candidate_id",
                        "sample_count",
                        "holdout_sample_count",
                        "error_count",
                        "retry_count",
                        "negative_savings_count",
                        "error_rate",
                        "retry_rate",
                        "negative_savings_rate",
                        "min_outcome_samples",
                        "window",
                        "raw_payload_included",
                    )
                },
            )
        except Exception:
            pass
        return body, meta

    meta.update({
        "status": "applied",
        "reason": "terminal-output-compaction-applied",
        "changed": True,
        "applied": True,
        "after_chars": planned_after,
        "saved_chars": planned_saved,
        "tokens_saved_est": planned_saved // TOKEN_CHARS,
        "compaction_cost_usd": 0.0,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    })
    return planned_body, meta


async def maybe_summarize_old_context(
    body: dict[str, Any],
    *,
    exact_cache_enabled: bool,
    get_cached_summary: Any,
    set_cached_summary: Any,
    fetch_summary: Any,
    store_obj: Any | None = None,
    managed_profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _effective_summary_policy(managed_profile)
    if _managed_summary_requested(managed_profile) and not _enhanced_provider_configured():
        meta = _summary_base_meta(
            "skipped",
            "fallback-not-configured",
            policy=policy,
            managed_profile=managed_profile,
        )
        meta.update({
            "before_chars": len(stable_json(body)),
            "category": _crunch_request_category(body),
            "enhanced_crunch_state": "fallback-not-configured",
            "configured": False,
            "applied": False,
        })
        return body, meta

    plan, meta = old_context_summary_plan(
        body,
        exact_cache_enabled=exact_cache_enabled,
        managed_profile=managed_profile,
    )
    if plan is None:
        return body, meta

    meta["configured"] = True
    meta["enhanced_crunch_state"] = "configured"
    canary = _summary_canary_decision(body, plan, policy=policy)
    meta["canary"] = canary
    if canary.get("enabled") and not canary.get("selected"):
        meta.update({
            "status": "skipped",
            "reason": str(canary.get("reason") or "canary_holdout"),
            "changed": False,
            "enhanced_crunch_state": "holdout" if canary.get("cohort") == "canary_holdout" else "outside-canary",
            "applied": False,
        })
        return body, meta

    safety_stop = evaluate_old_context_summary_safety_stop(store_obj)
    if safety_stop:
        meta.update({
            "status": "bypass",
            "reason": LOCAL_CANARY_SAFETY_STOP_REASON,
            "changed": False,
            "safety_stop_state": "stopped",
            "safety_stop": safety_stop,
            "enhanced_crunch_state": "bypassed",
            "applied": False,
        })
        log_pattern_canary_safety_stop(safety_stop)
        return body, meta

    cached = get_cached_summary(plan["cache_key"])
    summary = _summary_text_from_result(cached)
    summary_cache_hit = summary is not None
    fetch_result: Any = None
    if summary is None:
        try:
            fetch_result = await fetch_summary(plan["summary_request"])
        except Exception as exc:
            meta.update({
                "status": "error",
                "reason": "summary-fetch-error",
                "changed": False,
                "applied": False,
                "enhanced_crunch_state": "bypassed",
                "summary_error_type": type(exc).__name__,
            })
            return body, meta
        summary = _summary_text_from_result(fetch_result)
        if summary is None:
            reason = "summary-empty"
            if isinstance(fetch_result, dict):
                try:
                    if fetch_result.get("summary_status_code") is not None and int(fetch_result["summary_status_code"]) >= 400:
                        reason = "summary-error"
                except (TypeError, ValueError):
                    pass
            meta.update({
                "status": "skipped",
                "reason": reason,
                "changed": False,
                "applied": False,
                "enhanced_crunch_state": "bypassed",
            })
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
        summary_cost = 0.0
        if isinstance(fetch_result, dict):
            try:
                summary_cost = float(fetch_result.get("summary_cost_est_usd") or 0.0)
            except (TypeError, ValueError):
                summary_cost = 0.0
        max_cost = float(policy.get("max_summary_cost_usd") or OLD_CONTEXT_SUMMARY_MAX_COST_USD)
        if summary_cost > max_cost:
            meta.update({
                "status": "skipped",
                "reason": "summary-cost-too-high",
                "summary_cost_est_usd": summary_cost,
                "enhanced_crunch_state": "bypassed",
                "applied": False,
            })
            return body, meta
        set_cached_summary(plan["cache_key"], {
            "summary": summary,
            "usage": fetch_result.get("usage") if isinstance(fetch_result, dict) else None,
        })

    try:
        summarized = apply_old_context_summary(body, plan, summary)
    except Exception as exc:
        meta.update({
            "status": "error",
            "reason": "summary-apply-error",
            "changed": False,
            "applied": False,
            "enhanced_crunch_state": "bypassed",
            "summary_error_type": type(exc).__name__,
        })
        return body, meta
    preservation_check = _old_context_summary_preservation_check(body, summarized, plan)
    if not preservation_check["ok"]:
        meta.update({
            "status": "bypass",
            "reason": "tool-protocol-reconstruction-mismatch",
            "changed": False,
            "applied": False,
            "enhanced_crunch_state": "bypassed",
            "preservation_check": preservation_check,
        })
        return body, meta
    after_chars = len(stable_json(summarized))
    tokens_saved_est = (plan["before_chars"] - after_chars) // TOKEN_CHARS
    summary_cost_est = 0.0
    if isinstance(fetch_result, dict):
        try:
            summary_cost_est = float(fetch_result.get("summary_cost_est_usd") or 0.0)
        except (TypeError, ValueError):
            summary_cost_est = 0.0
    estimated_gross_savings = _summary_input_savings_usd(str(body.get("model") or ""), tokens_saved_est)
    meta.update({
        "status": "applied",
        "reason": "summary-cache-hit" if summary_cache_hit else "summary-created",
        "changed": after_chars != plan["before_chars"],
        "applied": True,
        "enhanced_crunch_state": "applied",
        "after_chars": after_chars,
        "saved_chars": plan["before_chars"] - after_chars,
        "tokens_saved_est": tokens_saved_est,
        "estimated_gross_savings_usd": round(estimated_gross_savings, 8),
        "estimated_net_savings_usd": round(estimated_gross_savings - summary_cost_est, 8),
        "summary_cache_hit": summary_cache_hit,
        "summary_chars": len(summary),
        "preservation_check": preservation_check,
        "tool_protocol_blocks_preserved": preservation_check["tool_protocol_blocks_preserved"],
        "recent_turns_preserved": preservation_check["recent_turns_preserved"],
    })
    if isinstance(fetch_result, dict):
        for key in ("summary_input_tokens", "summary_output_tokens", "summary_cost_est_usd", "summary_status_code"):
            if key in fetch_result:
                meta[key] = fetch_result[key]
    return summarized, meta


def crunch_body(
    body: dict[str, Any],
    *,
    store_obj: Any | None = None,
    managed_profile: dict[str, Any] | None = None,
    routing_meta: dict[str, Any] | None = None,
    provider: str | None = None,
    source_surface: str | None = None,
    endpoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservative request cruncher.

    It deliberately does NOT summarize with another model. For agent use this is safer.
    Current tactics:
    - normalize whitespace in large text blocks
    - deduplicate exact repeated text blocks within the same request
    - if extremely large, compress older non-tool text blocks to bounded heads/tails
    """
    managed_profile = managed_profile if isinstance(managed_profile, dict) else None
    policy_source = str((managed_profile or {}).get("policy_source") or CRUNCH_POLICY_SOURCE)
    threshold_chars = int((managed_profile or {}).get("threshold_chars") or CRUNCH_THRESHOLD_CHARS)

    if not CRUNCH_ENABLED:
        category = _crunch_request_category(body) if isinstance(body, dict) else None
        request_shape_canary_meta = _request_shape_repeated_context_canary_base_meta(False, "global-crunch-disabled")
        skipped_rules = [
            _crunch_rule_decision(
                rule_id,
                status="skipped",
                reason="global-crunch-disabled",
                policy_source=policy_source,
                rule_path=CRUNCH_RULES_PATH,
                enabled=False,
            )
            for rule_id in sorted(CRUNCH_RULE_CATALOG)
        ]
        meta = {
            "enabled": False,
            "changed": False,
            "policy_source": policy_source,
            "rule_path": CRUNCH_RULES_PATH,
            "rule_taxonomy": _crunch_rule_taxonomy_meta(policy_source=policy_source, rule_path=CRUNCH_RULES_PATH),
            "applied_rules": [],
            "skipped_rules": skipped_rules,
            "rule_group_breakdown": _crunch_rule_group_breakdown(skipped_rules),
            "managed_profile": managed_profile,
            "request_shape_repeated_context_canaries": request_shape_canary_meta,
            "anthropic_thinking_history_compaction": _anthropic_thinking_compaction_base_meta(
                "skipped",
                "global-crunch-disabled",
            ),
            "anthropic_thinking_history": build_anthropic_thinking_history_metadata(
                body if isinstance(body, dict) else None,
                provider=provider,
                source_surface=source_surface,
                endpoint=endpoint,
                category=category,
                policy_source=policy_source,
                rule_path=CRUNCH_RULES_PATH,
            ),
        }
        return body, meta

    new_body = copy.deepcopy(body)
    before = len(stable_json(new_body))
    category = _crunch_request_category(new_body)
    request_shape_canary_meta = _evaluate_request_shape_repeated_context_canaries(
        new_body,
        routing_meta=routing_meta,
        provider=provider,
        source_surface=source_surface,
        endpoint=endpoint,
        category=category,
    )
    thinking_history_meta = build_anthropic_thinking_history_metadata(
        body if isinstance(body, dict) else None,
        provider=provider,
        source_surface=source_surface,
        endpoint=endpoint,
        category=category,
        policy_source=policy_source,
        rule_path=CRUNCH_RULES_PATH,
    )
    new_body, thinking_compaction_meta = _apply_anthropic_thinking_history_compaction_canary(
        new_body,
        store_obj=store_obj,
        policy_source=policy_source,
        category=category,
        provider=provider,
        source_surface=source_surface,
        endpoint=endpoint,
    )
    new_body, terminal_output_compaction_meta = _apply_terminal_output_compaction_canary(
        new_body,
        store_obj=store_obj,
        policy_source=policy_source,
        category=category,
    )
    new_body, pattern_modules_meta = evaluate_pattern_modules(
        new_body,
        module_settings=PATTERN_MODULES_POLICY,
        apply_local_crunch=True,
        policy_source=policy_source,
        rule_path=CRUNCH_RULES_PATH,
        category=category,
    )
    new_body, instruction_dedup_meta = _apply_instruction_section_deduplication(
        new_body,
        store_obj=store_obj,
        routing_meta=routing_meta,
        provider=provider,
        source_surface=source_surface,
        endpoint=endpoint,
    )
    _provider_scaffolding_saved_chars, provider_scaffolding_meta = _apply_repeated_provider_scaffolding(
        new_body,
        managed_profile=managed_profile,
    )
    _pattern_saved_chars, pattern_rules_meta = _apply_pattern_rules(new_body, store_obj=store_obj)
    seen: dict[str, int] = {}
    seen_shingles: list[tuple[frozenset, int]] = []
    replacements = 0
    near_replacements = 0
    thinking_near_replacements = 0
    shortened = 0
    whitespace_normalized = 0
    skipped_rule_counts: dict[tuple[str, str], int] = {}
    terminal_log_metas: list[dict[str, Any]] = []

    def skip_rule(rule_id: str, reason: str, count: int = 1) -> None:
        skipped_rule_counts[(rule_id, reason)] = skipped_rule_counts.get((rule_id, reason), 0) + max(0, int(count))

    def process_content(content: Any, allow_shorten: bool) -> Any:
        nonlocal replacements, near_replacements, shortened, whitespace_normalized
        if isinstance(content, str):
            if _crunch_rule_allowed("terminal_log_boilerplate"):
                terminal_text, terminal_meta = _simplify_terminal_log_boilerplate_text(content)
            else:
                terminal_text = content
                terminal_meta = _terminal_log_meta("skipped", "rule-not-allowed")
                skip_rule("terminal_log_boilerplate", "rule-not-allowed")
            terminal_log_metas.append(terminal_meta)
            if _crunch_rule_allowed("whitespace_normalization"):
                txt = normalize_text(terminal_text)
                if txt != terminal_text:
                    whitespace_normalized += 1
            else:
                txt = terminal_text
                skip_rule("whitespace_normalization", "rule-not-allowed")
            h = sha256_text(txt)
            if len(txt) > 1000 and h in seen:
                if _crunch_rule_allowed("exact_duplicate_block_omission"):
                    replacements += 1
                    return f"[AgentFlow: exact duplicate text block omitted; same as earlier block #{seen[h]} hash={h[:12]}]"
                skip_rule("exact_duplicate_block_omission", "rule-not-allowed")
            seen[h] = len(seen) + 1
            if len(txt) > 2000:
                shingles = _shingles(txt)
                matched_idx = None
                for prev_shingles, prev_idx in seen_shingles:
                    if _jaccard(shingles, prev_shingles) > 0.85:
                        matched_idx = prev_idx
                        break
                if matched_idx is not None:
                    if _crunch_rule_allowed("near_duplicate_block_omission"):
                        near_replacements += 1
                        return f"[AgentFlow: near-duplicate text block omitted; similar to earlier block #{matched_idx} jaccard>0.85; original_chars={len(txt)}]"
                    skip_rule("near_duplicate_block_omission", "rule-not-allowed")
                seen_shingles.append((shingles, len(seen)))
            if allow_shorten and len(txt) > 8000:
                if _crunch_rule_allowed("old_text_collapse"):
                    shortened += 1
                    return txt[:3500] + f"\n\n[AgentFlow: middle of long older text block omitted; hash={h[:12]}; original_chars={len(txt)}]\n\n" + txt[-2500:]
                skip_rule("old_text_collapse", "rule-not-allowed")
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
    huge = before > threshold_chars
    for idx, msg in enumerate(messages):
        # only shorten older text, not the latest user/assistant context
        allow_shorten = huge and idx < max(0, len(messages) - 4)
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = process_content(msg["content"], allow_shorten=allow_shorten)
    if _crunch_rule_allowed("thinking_deduplication"):
        thinking_near_replacements = _dedupe_thinking_blocks(messages)
    else:
        skip_rule("thinking_deduplication", "rule-not-allowed")
    terminal_log_meta = _terminal_log_aggregate_meta(terminal_log_metas)

    after = len(stable_json(new_body))
    applied_rules: list[dict[str, Any]] = []
    skipped_rules: list[dict[str, Any]] = []

    def add_applied(rule_id: str, reason: str, count: int, saved_chars: int = 0, *, canary: Any = None, enabled: bool | None = True) -> None:
        if count <= 0:
            return
        applied_rules.append(_crunch_rule_decision(
            rule_id,
            status="applied",
            reason=reason,
            policy_source=policy_source,
            rule_path=CRUNCH_RULES_PATH,
            count=count,
            saved_chars=saved_chars,
            enabled=enabled,
            canary=canary,
        ))

    add_applied(
        "whitespace_normalization",
        "whitespace-normalized",
        whitespace_normalized,
    )
    add_applied(
        "exact_duplicate_block_omission",
        "exact-duplicate-text-block-omitted",
        replacements,
    )
    add_applied(
        "near_duplicate_block_omission",
        "near-duplicate-text-block-omitted",
        near_replacements,
    )
    add_applied(
        "old_text_collapse",
        "old-text-head-tail-collapse",
        shortened,
    )
    add_applied(
        "thinking_deduplication",
        "near-duplicate-thinking-block-omitted",
        thinking_near_replacements,
        enabled=THINKING_DEDUP_ENABLED,
    )
    add_applied(
        "terminal_log_boilerplate",
        "terminal-log-boilerplate-simplified",
        int(terminal_log_meta.get("text_blocks_changed") or 0),
        int(terminal_log_meta.get("saved_chars") or 0),
        enabled=TERMINAL_LOG_ENABLED,
    )
    if pattern_rules_meta.get("applied_count"):
        add_applied(
            "pattern_rules",
            "reviewed-pattern-rule-applied",
            int(pattern_rules_meta.get("applied_count") or 0),
            int(pattern_rules_meta.get("saved_chars") or 0),
            canary={"enabled": True, "canary_fraction": 1.0},
            enabled=bool(PATTERN_RULES),
        )
    if provider_scaffolding_meta.get("applied_count"):
        add_applied(
            "repeated_provider_scaffolding",
            "repeated-provider-scaffolding-collapsed",
            int(provider_scaffolding_meta.get("applied_count") or 0),
            int(provider_scaffolding_meta.get("saved_chars") or 0),
            enabled=bool(REPEATED_PROVIDER_SCAFFOLDING_POLICY.get("enabled")),
            canary=REPEATED_PROVIDER_SCAFFOLDING_POLICY.get("canary"),
        )
    for (rule_id, reason), count in sorted(skipped_rule_counts.items()):
        skipped_rules.append(_crunch_rule_decision(
            rule_id,
            status="skipped",
            reason=reason,
            policy_source=policy_source,
            rule_path=CRUNCH_RULES_PATH,
            count=count,
            enabled=False,
        ))
    for rule_id, section in (
        ("old_context_summarization", OLD_CONTEXT_SUMMARY_POLICY),
        ("anthropic_thinking_history_compaction", ANTHROPIC_THINKING_COMPACTION_POLICY),
        ("terminal_output_compaction", TERMINAL_OUTPUT_COMPACTION_POLICY),
        ("instruction_section_deduplication", INSTRUCTION_SECTION_DEDUP_POLICY),
    ):
        if not _crunch_rule_allowed(rule_id):
            skipped_rules.append(_crunch_rule_decision(
                rule_id,
                status="skipped",
                reason="rule-not-allowed",
                policy_source=policy_source,
                rule_path=CRUNCH_RULES_PATH,
                enabled=False,
                canary=section.get("canary") if isinstance(section, dict) else None,
            ))
    rule_group_breakdown = _crunch_rule_group_breakdown(applied_rules + skipped_rules)
    meta = {
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
        "terminal_log_boilerplate_simplified": terminal_log_meta["simplified_line_count"],
        "terminal_log_boilerplate_saved_chars": terminal_log_meta["saved_chars"],
        "terminal_log_boilerplate": terminal_log_meta,
        "anthropic_thinking_history_compaction": thinking_compaction_meta,
        "terminal_output_compaction": terminal_output_compaction_meta,
        "pattern_modules": pattern_modules_meta,
        "instruction_section_deduplication": instruction_dedup_meta,
        "registered_pattern_modules": registered_pattern_modules(),
        "pattern_rules_applied": pattern_rules_meta["applied_count"],
        "pattern_rule_saved_chars": pattern_rules_meta["saved_chars"],
        "pattern_rules": pattern_rules_meta,
        "repeated_provider_scaffolding": provider_scaffolding_meta,
        "request_shape_repeated_context_canaries": request_shape_canary_meta,
        "anthropic_thinking_history": thinking_history_meta,
        "policy_source": policy_source,
        "rule_path": CRUNCH_RULES_PATH,
        "rule_id": "local-crunch-policy",
        "rule_group": "multi_rule",
        "rule_taxonomy": _crunch_rule_taxonomy_meta(policy_source=policy_source, rule_path=CRUNCH_RULES_PATH),
        "applied_rules": applied_rules,
        "skipped_rules": skipped_rules,
        "rule_group_breakdown": rule_group_breakdown,
        "threshold_chars": threshold_chars,
        "managed_profile": managed_profile,
        "thinking_deduplication": {
            "enabled": THINKING_DEDUP_ENABLED,
            "min_chars": THINKING_DEDUP_MIN_CHARS,
            "similarity_threshold": THINKING_DEDUP_SIMILARITY_THRESHOLD,
            "skip_latest_assistant": THINKING_DEDUP_SKIP_LATEST_ASSISTANT,
        },
    }
    selected_lifecycle = request_shape_canary_meta.get("selected_lifecycle")
    if isinstance(selected_lifecycle, dict):
        meta["request_shape_repeated_context_canary"] = selected_lifecycle
    return new_body, meta


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
