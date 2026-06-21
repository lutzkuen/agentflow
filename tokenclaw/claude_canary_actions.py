from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tokenclaw.claude_canary_impact import SCHEMA as CLAUDE_IMPACT_SCHEMA
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.claude_canary_rollout_actions.v1"
ACTION_SCHEMA = "tokenclaw.claude_canary_rollout_action.v1"
APPLY_SCHEMA = "tokenclaw.claude_canary_rollout_actions_apply.v1"

_ROUTING_FILE = "routing_rules.yaml"
_ACTION_TYPES = {"widen", "hold", "rollback", "more-samples"}
_RAW_LIKE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cache_key",
    "content",
    "credential",
    "file_path",
    "message",
    "password",
    "prompt",
    "provider_body",
    "raw_context",
    "raw_payload",
    "raw_request",
    "raw_response",
    "request_id",
    "secret",
    "session_id",
    "system_prompt",
    "tool_payload",
    "tool_result",
    "transcript",
)
_ALLOWED_RAW_LIKE_KEYS = {
    "api_keys_included",
    "cache_keys_included",
    "content_free",
    "filesystem_paths_included",
    "provider_calls_made",
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_responses_included",
    "raw_session_ids_included",
    "raw_transcripts_included",
    "request_ids_included",
    "raw_request_ids_included",
    "session_id_hash",
    "tool_payloads_included",
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{_stable_hash(parts)[:24]}"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float = 0.0) -> float:
    return round(min(1.0, max(0.0, _as_float(value, default))), 6)


def _string(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    text = str(value).strip()
    return [text] if text else []


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_transcripts_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "api_keys_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "local_only": True,
    }


def _counter_rows(values: list[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _candidate_id(candidate: dict[str, Any]) -> str:
    explicit = _string(candidate.get("candidate_id") or candidate.get("target_candidate_id"))
    return explicit[:160] if explicit else _stable_id("claude-canary-candidate", candidate)


def _current_canary_fraction(candidate: dict[str, Any]) -> float:
    explicit = candidate.get("canary_fraction")
    if explicit is not None:
        return _bounded_fraction(explicit)
    counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    applied = _as_int(counts.get("canary_applied"))
    holdout = _as_int(counts.get("canary_holdout"))
    total = applied + holdout
    return _bounded_fraction(applied / total) if total else 0.0


def _current_holdout_fraction(candidate: dict[str, Any], default: float) -> float:
    explicit = candidate.get("holdout_fraction")
    if explicit is not None:
        return _bounded_fraction(explicit)
    counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    applied = _as_int(counts.get("canary_applied"))
    holdout = _as_int(counts.get("canary_holdout"))
    total = applied + holdout
    return _bounded_fraction(holdout / total) if total else _bounded_fraction(default)


def _action_type(candidate: dict[str, Any]) -> str:
    verdict = _string(candidate.get("verdict"))
    return "more-samples" if verdict == "needs_more_samples" else verdict


def _next_fractions(
    candidate: dict[str, Any],
    *,
    widen_step: float,
    max_canary_fraction: float,
    preserved_holdout_fraction: float,
) -> tuple[float, float]:
    action_type = _action_type(candidate)
    if action_type == "rollback":
        return 0.0, 0.0
    current = _current_canary_fraction(candidate)
    holdout = max(_current_holdout_fraction(candidate, preserved_holdout_fraction), _bounded_fraction(preserved_holdout_fraction))
    max_fraction = min(_bounded_fraction(max_canary_fraction, 1.0), max(0.0, 1.0 - holdout))
    if action_type == "widen":
        return _bounded_fraction(min(max_fraction, current + _bounded_fraction(widen_step))), _bounded_fraction(holdout)
    return _bounded_fraction(min(max_fraction, current)), _bounded_fraction(holdout)


def _rollback_reason_codes(candidate: dict[str, Any]) -> list[str]:
    reasons = _string_list(candidate.get("reason_codes"))
    mapped: set[str] = set()
    for reason in reasons:
        if "safety-stop" in reason:
            mapped.add("rollback-safety-stop")
        elif "error" in reason:
            mapped.add("rollback-error")
        elif "retry" in reason:
            mapped.add("rollback-retry")
        elif "fallback" in reason:
            mapped.add("rollback-fallback")
        elif "latency" in reason:
            mapped.add("rollback-latency")
        elif "saving" in reason or "cost" in reason:
            mapped.add("rollback-cost")
        elif "stale" in reason:
            mapped.add("rollback-stale-evidence")
        elif "holdout" in reason or "insufficient" in reason:
            mapped.add("rollback-insufficient-holdout")
    return sorted(mapped or {"rollback-operator-review"})


def _safe_evidence_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    cohorts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    deltas = candidate.get("applied_vs_holdout_deltas") if isinstance(candidate.get("applied_vs_holdout_deltas"), dict) else {}
    stale = candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {}
    return {
        "sample_count": _as_int(candidate.get("sample_count")),
        "cohort_counts": {str(key): _as_int(value) for key, value in cohorts.items()},
        "observed_savings_usd": round(_as_float(candidate.get("observed_savings_usd")), 8),
        "requested_model_fallback_cost_usd": round(_as_float(candidate.get("requested_model_fallback_cost_usd")), 8),
        "applied_vs_holdout_deltas": {
            key: value
            for key, value in deltas.items()
            if key in {
                "applied_minus_holdout_error_rate",
                "applied_minus_holdout_retry_rate",
                "applied_minus_holdout_fallback_rate",
                "applied_minus_holdout_rate_limit_fallback_rate",
                "applied_minus_holdout_latency_avg_ms",
                "applied_minus_holdout_cost_est_usd",
            }
            and isinstance(value, (int, float, type(None)))
        },
        "stale_evidence": {
            "stale": bool(stale.get("stale")),
            "age_hours": stale.get("age_hours") if isinstance(stale.get("age_hours"), (int, float, type(None))) else None,
            "max_age_hours": stale.get("max_age_hours") if isinstance(stale.get("max_age_hours"), (int, float, type(None))) else None,
        },
        "reason_codes": _string_list(candidate.get("reason_codes")),
        "warning_codes": _string_list(candidate.get("warning_codes")),
    }


def _local_policy_update(
    candidate: dict[str, Any],
    *,
    action_type: str,
    canary_fraction: float,
    holdout_fraction: float,
    preserved_holdout_fraction: float,
) -> dict[str, Any]:
    policy_id = _string(candidate.get("rule_id") or candidate.get("policy_id") or candidate.get("target_candidate_id"), "local-claude-routing-canary")
    category = _string(candidate.get("category"))
    workflow_phase = _string(candidate.get("workflow_phase"))
    eligible_categories = [category] if category else []
    eligible_phases = [workflow_phase] if workflow_phase else ["tool-execution", "summary"]
    return {
        "target_local_policy": "phase_canary",
        "policy_source": "local-manual",
        "managed_enforced": False,
        "required_local_review": True,
        "enabled": action_type not in {"rollback", "more-samples"},
        "policy_id": policy_id,
        "promotion_action_id": None,
        "target_candidate_id": _candidate_id(candidate),
        "provider": "anthropic",
        "source_surface": _string(candidate.get("source_surface"), "anthropic_messages"),
        "app_family": _string(candidate.get("app_family"), "anthropic"),
        "model_pattern": _string(candidate.get("original_model") or candidate.get("requested_model"), "sonnet"),
        "requested_model": _string(candidate.get("original_model") or candidate.get("requested_model"), "sonnet"),
        "target_model": _string(candidate.get("candidate_target_model") or candidate.get("target_model"), "haiku"),
        "routed_model": _string(candidate.get("candidate_target_model") or candidate.get("target_model"), "haiku"),
        "stream": bool(candidate.get("stream")),
        "eligible_workflow_phases": eligible_phases,
        "excluded_workflow_phases": ["planning", "thinking", "unknown"],
        "eligible_categories": eligible_categories,
        "excluded_categories": ["code-gen"],
        "min_workflow_phase_confidence": _string(candidate.get("workflow_phase_confidence"), "medium"),
        "min_text_chars": 0,
        "max_text_chars": 30000,
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "preserved_holdout_fraction": _bounded_fraction(preserved_holdout_fraction),
        "salt": _string(candidate.get("promotion_action_id") or candidate.get("target_candidate_id") or policy_id, policy_id),
        "cohort_unit": "request_features",
        "safety_gates": {
            "block_thinking_history": True,
            "block_top_level_thinking": True,
            "strip_model_incompatible_params": True,
            "fallback_to_requested_on_rate_limit": True,
            "content_free": True,
            "provider_calls_made_by_apply": False,
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


def _action(
    candidate: dict[str, Any],
    *,
    widen_step: float,
    max_canary_fraction: float,
    preserved_holdout_fraction: float,
) -> dict[str, Any]:
    action_type = _action_type(candidate)
    canary_fraction, holdout_fraction = _next_fractions(
        candidate,
        widen_step=widen_step,
        max_canary_fraction=max_canary_fraction,
        preserved_holdout_fraction=preserved_holdout_fraction,
    )
    candidate_id = _candidate_id(candidate)
    target_rule_id = _string(candidate.get("rule_id") or candidate.get("policy_id") or candidate_id, "local-claude-routing-canary")
    action_id = _stable_id("claude-canary-action", candidate_id, action_type, canary_fraction, holdout_fraction)
    local_update = _local_policy_update(
        candidate,
        action_type=action_type,
        canary_fraction=canary_fraction,
        holdout_fraction=holdout_fraction,
        preserved_holdout_fraction=preserved_holdout_fraction,
    )
    local_update["promotion_action_id"] = action_id
    return {
        "schema": ACTION_SCHEMA,
        "action_id": action_id,
        "status": "reviewable",
        "action_type": action_type,
        "verdict": _string(candidate.get("verdict")),
        "target_candidate_id": candidate_id,
        "target_rule_id": target_rule_id,
        "action_family": "routing",
        "optimization_family": "claude_phase_routing",
        "provider": "anthropic",
        "source_surface": _string(candidate.get("source_surface"), "anthropic_messages"),
        "app_family": _string(candidate.get("app_family"), "anthropic"),
        "policy_section": "routing",
        "target_local_policy_section": "routing.phase_canary",
        "current_canary_fraction": _current_canary_fraction(candidate),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "max_canary_fraction": _bounded_fraction(max_canary_fraction, 1.0),
        "preserved_holdout_fraction": _bounded_fraction(preserved_holdout_fraction),
        "local_policy_update": local_update,
        "evidence_summary": _safe_evidence_summary(candidate),
        "rollback_metadata": {
            "rollback_action_type": "rollback",
            "rollback_canary_fraction": 0.0,
            "rollback_reason_codes": _rollback_reason_codes(candidate),
            "preserve_previous_rule_required": True,
        },
        "local_review": {
            "required": True,
            "review_command": "tokenclaw-claude-canary-actions",
            "apply_preview_command": "tokenclaw-claude-canary-actions-apply --dry-run",
            "apply_write_command": "tokenclaw-claude-canary-actions-apply --write",
        },
        "privacy": _privacy_summary(),
    }


def build_claude_canary_actions(
    impact_report: dict[str, Any],
    *,
    widen_step: float = 0.25,
    max_canary_fraction: float = 1.0,
    preserved_holdout_fraction: float = 0.10,
) -> dict[str, Any]:
    candidates = impact_report.get("candidates") if isinstance(impact_report, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    actions: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            omitted.append({
                "status": "omitted",
                "reason": "invalid-candidate",
                "path": f"$.candidates[{index}]",
                "privacy": _privacy_summary(),
            })
            continue
        action_type = _action_type(candidate)
        if action_type not in _ACTION_TYPES:
            omitted.append({
                "status": "omitted",
                "reason": "unsupported-verdict",
                "target_candidate_id": _candidate_id(candidate),
                "verdict": candidate.get("verdict"),
                "privacy": _privacy_summary(),
            })
            continue
        actions.append(
            _action(
                candidate,
                widen_step=widen_step,
                max_canary_fraction=max_canary_fraction,
                preserved_holdout_fraction=preserved_holdout_fraction,
            )
        )

    actions.sort(key=lambda row: (str(row.get("action_type")), str(row.get("target_candidate_id"))))
    action_types = [str(action.get("action_type") or "unknown") for action in actions]
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "read_only": True,
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "source_report_schema": impact_report.get("schema") if isinstance(impact_report, dict) else None,
        "summary": {
            "candidate_count": len(candidates),
            "action_count": len(actions),
            "omitted_count": len(omitted),
            "action_type_counts": _counter_rows(action_types),
        },
        "actions": actions,
        "omitted": omitted,
        "privacy": _privacy_summary(),
    }


def _truthy(value: Any) -> bool:
    return value not in (None, False, 0, "", [], {})


def _scan_raw_like(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_LIKE_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    errors.append({"path": child_path, "message": "Claude canary action payloads must remain metadata-only"})
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        default = Path(__file__).with_name(_ROUTING_FILE)
        text = default.read_text(encoding="utf-8") if default.exists() else "rules: []\n"
    data = yaml.safe_load(text) or {}
    if isinstance(data, list):
        data = {"rules": data}
    if not isinstance(data, dict):
        data = {"rules": []}
    if not isinstance(data.get("rules"), list):
        data["rules"] = []
    return data, text


def _backup_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_policy_file(path: Path, text: str, *, backup_id: str | None = None) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: str | None = None
    if path.exists():
        suffix = backup_id or _backup_suffix()
        backup = path.with_name(f"{path.name}.bak-{suffix}")
        backup.write_bytes(path.read_bytes())
        backup_path = str(backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return backup_path


def _phase_canary_from_action(action: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    existing = dict(existing or {})
    action_type = str(action.get("action_type") or "")
    if action_type == "more-samples":
        return existing
    disabled = action_type == "rollback"
    canary = dict(existing)
    for key in (
        "policy_id",
        "promotion_action_id",
        "target_candidate_id",
        "policy_source",
        "provider",
        "source_surface",
        "app_family",
        "model_pattern",
        "target_model",
        "requested_model",
        "routed_model",
        "stream",
        "eligible_workflow_phases",
        "excluded_workflow_phases",
        "eligible_categories",
        "excluded_categories",
        "min_workflow_phase_confidence",
        "min_text_chars",
        "max_text_chars",
        "salt",
        "cohort_unit",
    ):
        if key in update:
            canary[key] = update[key]
    canary["enabled"] = False if disabled else bool(update.get("enabled", True))
    canary["canary_fraction"] = 0.0 if disabled else _bounded_fraction(action.get("canary_fraction"))
    canary["holdout_fraction"] = 0.0 if disabled else _bounded_fraction(action.get("holdout_fraction"))
    canary["managed_enforced"] = False
    if isinstance(update.get("safety_gates"), dict):
        canary["safety_gates"] = update["safety_gates"]
    if isinstance(update.get("safety_stop"), dict):
        canary["safety_stop"] = update["safety_stop"]
    canary["rollout_action"] = {
        "schema": ACTION_SCHEMA,
        "action_type": action_type,
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id"),
        "action_id": action.get("action_id"),
        "canary_fraction": canary["canary_fraction"],
        "holdout_fraction": canary["holdout_fraction"],
        "managed_enforced": False,
    }
    return {key: value for key, value in canary.items() if value is not None}


def apply_claude_canary_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = True,
    backup_id: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser()
    errors: list[dict[str, str]] = []
    actions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        errors.append({"path": "$.schema", "message": f"expected {SCHEMA}"})
        raw_actions = []
    else:
        raw_actions = bundle.get("actions") if isinstance(bundle.get("actions"), list) else []
        _scan_raw_like(bundle, "$", errors)

    routing_path = config_path / _ROUTING_FILE
    data, old_text = _load_yaml(routing_path)
    if not isinstance(data.get("phase_canary"), dict):
        data["phase_canary"] = {}

    if not errors:
        for index, action in enumerate(raw_actions):
            if not isinstance(action, dict):
                errors.append({"path": f"$.actions[{index}]", "message": "expected action object"})
                continue
            if action.get("schema") != ACTION_SCHEMA:
                errors.append({"path": f"$.actions[{index}].schema", "message": f"expected {ACTION_SCHEMA}"})
                continue
            if str(action.get("policy_section") or "") != "routing":
                actions.append({"path": f"$.actions[{index}]", "status": "skipped", "reason": "unsupported-policy-section"})
                continue
            action_type = str(action.get("action_type") or "")
            if action_type not in _ACTION_TYPES:
                actions.append({"path": f"$.actions[{index}]", "status": "rejected", "reason": "unsupported-action-type"})
                errors.append({"path": f"$.actions[{index}].action_type", "message": "unsupported Claude canary action type"})
                continue
            existing = data.get("phase_canary") if isinstance(data.get("phase_canary"), dict) else {}
            updated = _phase_canary_from_action(action, existing)
            data["phase_canary"] = updated
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": "routing",
                "target_local_policy": "phase_canary",
                "action_type": action_type,
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "action_id": action.get("action_id"),
                "rule_id": updated.get("policy_id"),
                "target_model": updated.get("target_model"),
                "canary_fraction": updated.get("canary_fraction"),
                "holdout_fraction": updated.get("holdout_fraction"),
            })

    if not errors:
        new_text = yaml.safe_dump(data, sort_keys=False)
        changed = old_text != new_text
        backup_path = None
        if changed and not dry_run:
            backup_path = _write_policy_file(routing_path, new_text, backup_id=backup_id)
        files.append({
            "section": "routing",
            "path": str(routing_path),
            "changed": bool(changed),
            "backup_path": backup_path,
            "bytes_after": len(new_text.encode("utf-8")),
        })

    return {
        "schema": APPLY_SCHEMA,
        "ok": not errors,
        "generated_at": utc_now(),
        "dry_run": bool(dry_run),
        "read_only": bool(dry_run),
        "wrote_policy_files": bool(files and not dry_run and any(file.get("changed") for file in files)),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "config_dir": str(config_path),
        "actions": actions,
        "files": files,
        "summary": {
            "action_count": len(raw_actions),
            "planned_action_count": sum(1 for action in actions if action.get("status") == "planned"),
            "error_count": len(errors),
        },
        "errors": errors,
        "privacy": _privacy_summary(),
    }


def is_claude_canary_impact_report(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("schema") == CLAUDE_IMPACT_SCHEMA
