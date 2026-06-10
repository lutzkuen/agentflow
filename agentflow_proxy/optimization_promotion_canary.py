from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.optimization_promotion_actions import ACTION_SCHEMA, SCHEMA as PROMOTION_ACTIONS_SCHEMA
from agentflow_proxy.store import utc_now


PROMOTION_CANARY_APPLY_SCHEMA = "agentflow.optimization_promotion_canary_apply.v1"
PROMOTION_CANARY_DECISION_SCHEMA = "agentflow.optimization_promotion_canary_decision.v1"
PROMOTION_CANARY_SAFETY_SCHEMA = "agentflow.optimization_promotion_canary_safety_stop.v1"

_POLICY_SECTION_FILES = {"routing": "routing_rules.yaml"}
_COHORT_SAFE_KEYS = (
    "request_fingerprint",
    "session_id_hash",
    "workflow_id_hash",
    "traffic_fingerprint",
    "optimization_unit_id",
    "source_surface",
    "app_family",
    "category",
    "workflow_phase",
    "text_bucket",
    "token_bucket",
    "requested_model",
    "candidate_target_model",
    "has_tools",
    "stream",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_cohort_features(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in _COHORT_SAFE_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return safe


def promotion_canary_decision(
    action: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    safety_stop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign a promotion rollout action to applied, holdout, or skipped using metadata only."""
    canary_fraction = _as_float(action.get("canary_fraction"))
    holdout_fraction = _as_float(action.get("holdout_fraction"))
    safe_features = _safe_cohort_features(metadata)
    action_id = str(action.get("action_id") or "")
    candidate_id = str(action.get("target_candidate_id") or "")
    target_rule_id = str(action.get("target_rule_id") or "")
    base = {
        "schema": PROMOTION_CANARY_DECISION_SCHEMA,
        "enabled": canary_fraction > 0 or holdout_fraction > 0,
        "selected": False,
        "status": "disabled",
        "cohort": "disabled",
        "reason": "disabled",
        "action_id": action_id,
        "target_candidate_id": candidate_id,
        "target_rule_id": target_rule_id,
        "policy_section": action.get("policy_section"),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "raw_content_included": False,
        "safe_feature_keys": sorted(safe_features),
    }
    if safety_stop and safety_stop.get("tripped"):
        base.update({
            "enabled": True,
            "status": "safety_stopped",
            "cohort": "bypassed_or_disabled",
            "reason": "local-canary-safety-stop",
            "safety_stop": safety_stop,
        })
        return base
    if not base["enabled"]:
        return base

    basis = {
        "action_id": action_id,
        "target_candidate_id": candidate_id,
        "target_rule_id": target_rule_id,
        "policy_section": action.get("policy_section"),
        "features": safe_features,
    }
    digest = _stable_hash({"salt": action_id or candidate_id or target_rule_id, "basis": basis})
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    base.update({
        "status": "skipped",
        "cohort": "skipped",
        "reason": "outside-canary-and-holdout",
        "bucket": round(bucket, 12),
        "cohort_key_hash": f"sha256:{_stable_hash(basis)}",
    })
    if bucket < holdout_fraction:
        base.update({
            "status": "holdout",
            "cohort": "canary_holdout",
            "reason": "selected-holdout",
        })
    elif bucket < holdout_fraction + canary_fraction:
        base.update({
            "selected": True,
            "status": "applied",
            "cohort": "canary_applied",
            "reason": "selected-canary",
        })
    return base


def evaluate_promotion_canary_safety_stop(
    action: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_source = thresholds if isinstance(thresholds, dict) else {}
    min_samples = _as_int(threshold_source.get("min_samples"), 1)
    max_error_rate = float(threshold_source.get("max_error_rate", 0.05))
    max_5xx_rate = float(threshold_source.get("max_5xx_rate", max_error_rate))
    max_unsupported_model_errors = _as_int(threshold_source.get("max_unsupported_model_errors"), 0)
    max_cache_stale_risk_blockers = _as_int(threshold_source.get("max_cache_stale_risk_blockers"), 0)

    applied = [
        row for row in records
        if str(row.get("cohort") or row.get("status") or "") in {"canary_applied", "applied"}
    ]
    sample_count = len(applied)
    status_codes = [_as_int(row.get("status_code")) for row in applied]
    error_count = sum(1 for code in status_codes if code >= 400)
    five_xx_count = sum(1 for code in status_codes if code >= 500)
    unsupported_model_errors = sum(
        1
        for row in applied
        if "unsupported" in str(row.get("error_bucket") or row.get("error") or "").lower()
        and "model" in str(row.get("error_bucket") or row.get("error") or "").lower()
    )
    stale_risk_blockers = sum(
        1
        for row in applied
        if "stale-risk" in str(row.get("reason") or row.get("blocker") or row.get("error_bucket") or "").lower()
    )
    error_rate = error_count / sample_count if sample_count else 0.0
    five_xx_rate = five_xx_count / sample_count if sample_count else 0.0
    reason_codes: list[str] = []
    if sample_count >= min_samples and error_rate > max_error_rate:
        reason_codes.append("error-rate")
    if sample_count >= min_samples and five_xx_rate > max_5xx_rate:
        reason_codes.append("provider-5xx-rate")
    if unsupported_model_errors > max_unsupported_model_errors:
        reason_codes.append("unsupported-model-errors")
    if stale_risk_blockers > max_cache_stale_risk_blockers:
        reason_codes.append("cache-stale-risk-blockers")
    return {
        "schema": PROMOTION_CANARY_SAFETY_SCHEMA,
        "enabled": True,
        "status": "tripped" if reason_codes else ("insufficient-samples" if sample_count < min_samples else "ok"),
        "tripped": bool(reason_codes),
        "action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "sample_count": sample_count,
        "min_samples": min_samples,
        "error_count": error_count,
        "error_rate": round(error_rate, 6),
        "five_xx_count": five_xx_count,
        "five_xx_rate": round(five_xx_rate, 6),
        "unsupported_model_errors": unsupported_model_errors,
        "cache_stale_risk_blockers": stale_risk_blockers,
        "reason_codes": reason_codes,
    }


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {"rules": []}, None
    text = path.read_text(encoding="utf-8")
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


def _write_policy_file(path: Path, text: str) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: str | None = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{_backup_suffix()}")
        backup.write_bytes(path.read_bytes())
        backup_path = str(backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return backup_path


def _routing_phase_canary(action: dict[str, Any]) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    target_model = local_update.get("candidate_target_model") or action.get("candidate_target_model") or "haiku"
    disabled = str(action.get("action_type") or "") == "rollback" or _as_float(action.get("canary_fraction")) <= 0
    return {
        "enabled": not disabled,
        "policy_id": str(action.get("target_rule_id") or action.get("target_candidate_id") or "promotion-routing-canary"),
        "promotion_action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "policy_source": "managed-recommended",
        "model_pattern": "sonnet",
        "target_model": target_model,
        "eligible_workflow_phases": ["tool-execution", "summary"],
        "excluded_workflow_phases": ["planning", "thinking", "unknown"],
        "eligible_categories": [],
        "excluded_categories": ["code-gen"],
        "min_workflow_phase_confidence": "medium",
        "min_text_chars": 0,
        "max_text_chars": 30000,
        "canary_fraction": 0.0 if disabled else _as_float(action.get("canary_fraction")),
        "holdout_fraction": 0.0 if disabled else _as_float(action.get("holdout_fraction")),
        "salt": str(action.get("action_id") or action.get("target_candidate_id") or "promotion-routing-canary"),
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


def apply_optimization_promotion_canaries(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = True,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser()
    requested_sections = set(sections or _POLICY_SECTION_FILES)
    errors: list[dict[str, str]] = []
    actions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    if not isinstance(bundle, dict) or bundle.get("schema") != PROMOTION_ACTIONS_SCHEMA:
        errors.append({"path": "$.schema", "message": f"expected {PROMOTION_ACTIONS_SCHEMA}"})
    raw_actions = bundle.get("actions") if isinstance(bundle, dict) else None
    if not isinstance(raw_actions, list):
        raw_actions = []
        errors.append({"path": "$.actions", "message": "expected list"})
    invalid_sections = sorted(requested_sections - set(_POLICY_SECTION_FILES))
    for section in invalid_sections:
        errors.append({"path": f"$.sections.{section}", "message": "unsupported promotion canary policy section"})

    if not errors:
        routing_path = config_path / _POLICY_SECTION_FILES["routing"]
        routing_data, old_text = _load_yaml(routing_path)
        for index, action in enumerate(raw_actions):
            if not isinstance(action, dict):
                errors.append({"path": f"$.actions[{index}]", "message": "expected action object"})
                continue
            if action.get("schema") != ACTION_SCHEMA:
                errors.append({"path": f"$.actions[{index}].schema", "message": f"expected {ACTION_SCHEMA}"})
                continue
            section = str(action.get("policy_section") or "")
            if section not in requested_sections:
                actions.append({"path": f"$.actions[{index}]", "status": "skipped", "reason": "not-requested", "policy_section": section})
                continue
            if section != "routing":
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "skipped",
                    "reason": "promotion-canary-apply-supports-routing-only",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            routing_data["phase_canary"] = _routing_phase_canary(action)
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": "routing",
                "action_type": action.get("action_type"),
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "action_id": action.get("action_id"),
                "canary_fraction": routing_data["phase_canary"]["canary_fraction"],
                "holdout_fraction": routing_data["phase_canary"]["holdout_fraction"],
            })
        text = yaml.safe_dump(routing_data, sort_keys=False)
        changed = old_text != text
        backup_path = None
        if changed and not dry_run:
            backup_path = _write_policy_file(routing_path, text)
        if any(action.get("status") == "planned" and action.get("policy_section") == "routing" for action in actions):
            files.append({
                "section": "routing",
                "path": str(routing_path),
                "changed": bool(changed),
                "backup_path": backup_path,
                "bytes_after": len(text.encode("utf-8")),
            })

    return {
        "schema": PROMOTION_CANARY_APPLY_SCHEMA,
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
            "skipped_action_count": sum(1 for action in actions if action.get("status") == "skipped"),
            "error_count": len(errors),
        },
        "errors": errors,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
        },
    }
