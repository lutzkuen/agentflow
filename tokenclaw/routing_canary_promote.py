from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tokenclaw.claude_canary_impact import SCHEMA as CLAUDE_IMPACT_SCHEMA
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.routing_canary_promotion_plan.v1"
APPLY_SCHEMA = "tokenclaw.routing_canary_promotion_apply.v1"
_ROUTING_FILE = "routing_rules.yaml"
_CANARY_FILE = "routing_canary_policy.yaml"
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
    "cache_keys_included",
    "content_free",
    "filesystem_paths_included",
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


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{_stable_hash(parts)[:24]}"


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


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
                    errors.append({"path": child_path, "message": "routing canary promotion payloads must remain metadata-only"})
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:500]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


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


def _cohort_count(candidate: dict[str, Any], name: str) -> int:
    counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    return _as_int(counts.get(name))


def _cohort_metric(candidate: dict[str, Any], cohort: str, field: str) -> float:
    metrics = candidate.get("cohort_metrics") if isinstance(candidate.get("cohort_metrics"), dict) else {}
    row = metrics.get(cohort) if isinstance(metrics.get(cohort), dict) else {}
    return _as_float(row.get(field))


def _delta(candidate: dict[str, Any], field: str) -> float | None:
    deltas = candidate.get("applied_vs_holdout_deltas") if isinstance(candidate.get("applied_vs_holdout_deltas"), dict) else {}
    value = deltas.get(field)
    if value is None:
        return None
    return _as_float(value)


def _candidate_id(candidate: dict[str, Any]) -> str:
    return _string(
        candidate.get("target_candidate_id")
        or candidate.get("candidate_id")
        or candidate.get("promotion_action_id")
        or candidate.get("rule_id")
        or candidate.get("policy_id"),
        "unknown-claude-canary",
    )[:160]


def _model_pattern(model: Any) -> str:
    text = str(model or "").lower()
    for family in ("sonnet", "haiku", "opus"):
        if family in text:
            return family
    return str(model or "").strip() or "sonnet"


def _route_target(candidate: dict[str, Any]) -> str:
    return _string(candidate.get("candidate_target_model") or candidate.get("target_model"), "haiku")


def _safe_evidence_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": _as_int(candidate.get("sample_count")),
        "cohort_counts": {
            "canary_applied": _cohort_count(candidate, "canary_applied"),
            "canary_holdout": _cohort_count(candidate, "canary_holdout"),
            "safety_stopped": _cohort_count(candidate, "safety_stopped"),
        },
        "observed_savings_usd": round(_as_float(candidate.get("observed_savings_usd")), 8),
        "applied_error_rate": round(_cohort_metric(candidate, "canary_applied", "error_rate"), 6),
        "applied_vs_holdout_deltas": {
            "applied_minus_holdout_error_rate": _delta(candidate, "applied_minus_holdout_error_rate"),
            "applied_minus_holdout_retry_rate": _delta(candidate, "applied_minus_holdout_retry_rate"),
            "applied_minus_holdout_fallback_rate": _delta(candidate, "applied_minus_holdout_fallback_rate"),
            "applied_minus_holdout_latency_avg_ms": _delta(candidate, "applied_minus_holdout_latency_avg_ms"),
        },
        "reason_codes": [str(item) for item in candidate.get("reason_codes") or []],
        "latest_observed_at": candidate.get("latest_observed_at"),
        "oldest_observed_at": candidate.get("oldest_observed_at"),
    }


def _permanent_rule(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    target_model = _route_target(candidate)
    rule_id = _string(candidate.get("rule_id") or candidate.get("policy_id"), candidate_id)
    conditions: dict[str, Any] = {
        "model_pattern": _model_pattern(candidate.get("original_model") or candidate.get("requested_model")),
    }
    category = _string(candidate.get("category"))
    if category and category != "unknown":
        conditions["category"] = category
    workflow_phase = _string(candidate.get("workflow_phase"))
    if workflow_phase and workflow_phase != "unknown":
        conditions["workflow_phase"] = workflow_phase
    confidence = _string(candidate.get("workflow_phase_confidence"))
    if confidence:
        conditions["workflow_phase_confidence_gte"] = confidence
    if candidate.get("stream") is not None:
        conditions["stream"] = bool(candidate.get("stream"))
    if category.startswith("tool-"):
        conditions["has_tools"] = True
    elif category in {"chat", "short-completion", "long-context", "code-gen"}:
        conditions["has_tools"] = False

    return {
        "id": f"promoted-{rule_id}",
        "policy_source": "local-promoted",
        "conditions": conditions,
        "action": {
            "route_to": target_model,
            "reason": f"promoted Claude routing canary {candidate_id} to permanent local rule",
        },
        "metadata": {
            "source": "claude-canary-promote",
            "promoted_from_canary": True,
            "promotion_source_policy_id": rule_id,
            "target_candidate_id": candidate_id,
            "promoted_at": utc_now(),
            "evidence": _safe_evidence_summary(candidate),
        },
    }


def _same_rule(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_meta = left.get("metadata") if isinstance(left.get("metadata"), dict) else {}
    right_meta = right.get("metadata") if isinstance(right.get("metadata"), dict) else {}
    if left_meta.get("target_candidate_id") and left_meta.get("target_candidate_id") == right_meta.get("target_candidate_id"):
        return True
    return left.get("conditions") == right.get("conditions") and left.get("action") == right.get("action")


def _gate_candidate(
    candidate: dict[str, Any],
    *,
    min_applied_samples: int,
    min_holdout_samples: int,
    max_error_rate: float,
    max_error_rate_delta: float,
    max_latency_regression_ms: int,
) -> list[str]:
    reasons: list[str] = []
    if str(candidate.get("verdict") or "") != "promote":
        reasons.append("not-promote-verdict")
    if _cohort_count(candidate, "canary_applied") < min_applied_samples:
        reasons.append("insufficient-applied-samples")
    if _cohort_count(candidate, "canary_holdout") < min_holdout_samples:
        reasons.append("insufficient-holdout-samples")
    if _cohort_metric(candidate, "canary_applied", "error_rate") > max_error_rate:
        reasons.append("applied-error-rate-above-threshold")
    error_delta = _delta(candidate, "applied_minus_holdout_error_rate")
    if error_delta is not None and error_delta > max_error_rate_delta:
        reasons.append("error-rate-regression")
    latency_delta = _delta(candidate, "applied_minus_holdout_latency_avg_ms")
    if latency_delta is not None and latency_delta > max_latency_regression_ms:
        reasons.append("latency-regression")
    stale = candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {}
    if stale.get("stale"):
        reasons.append("stale-evidence")
    if _as_float(candidate.get("observed_savings_usd")) <= 0:
        reasons.append("non-positive-observed-savings")
    return reasons


def build_routing_canary_promotion_plan(
    impact_report: dict[str, Any],
    *,
    config_dir: str | Path,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_error_rate: float = 0.05,
    max_error_rate_delta: float = 0.05,
    max_latency_regression_ms: int = 2000,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if not isinstance(impact_report, dict) or impact_report.get("schema") != CLAUDE_IMPACT_SCHEMA:
        errors.append({"path": "$.schema", "message": f"expected {CLAUDE_IMPACT_SCHEMA}"})
    else:
        _scan_raw_like(impact_report, "$", errors)

    config_path = Path(config_dir).expanduser()
    routing_path = config_path / _ROUTING_FILE
    routing_data, _old_text = _load_yaml(routing_path)
    existing_rules = routing_data.get("rules") if isinstance(routing_data.get("rules"), list) else []
    actions: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []

    candidates = impact_report.get("candidates") if isinstance(impact_report, dict) and isinstance(impact_report.get("candidates"), list) else []
    if not errors:
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                omitted.append({"path": f"$.candidates[{index}]", "status": "omitted", "reason": "invalid-candidate", "privacy": _privacy_summary()})
                continue
            candidate_id = _candidate_id(candidate)
            gate_reasons = _gate_candidate(
                candidate,
                min_applied_samples=min_applied_samples,
                min_holdout_samples=min_holdout_samples,
                max_error_rate=max_error_rate,
                max_error_rate_delta=max_error_rate_delta,
                max_latency_regression_ms=max_latency_regression_ms,
            )
            if gate_reasons:
                omitted.append({
                    "path": f"$.candidates[{index}]",
                    "status": "omitted",
                    "reason": gate_reasons[0],
                    "reason_codes": gate_reasons,
                    "target_candidate_id": candidate_id,
                    "verdict": candidate.get("verdict"),
                    "privacy": _privacy_summary(),
                })
                continue
            rule = _permanent_rule(candidate)
            if any(isinstance(existing, dict) and _same_rule(existing, rule) for existing in existing_rules):
                omitted.append({
                    "path": f"$.candidates[{index}]",
                    "status": "omitted",
                    "reason": "already-permanent",
                    "target_candidate_id": candidate_id,
                    "rule_id": rule.get("id"),
                    "privacy": _privacy_summary(),
                })
                continue
            actions.append({
                "action_id": _stable_id("routing-canary-promote", candidate_id, rule.get("id")),
                "status": "planned",
                "action_type": "promote",
                "target_candidate_id": candidate_id,
                "source_policy_id": candidate.get("rule_id") or candidate.get("policy_id"),
                "rule_id": rule.get("id"),
                "target_model": _route_target(candidate),
                "permanent_rule": rule,
                "evidence_summary": _safe_evidence_summary(candidate),
                "privacy": _privacy_summary(),
            })

    return {
        "schema": SCHEMA,
        "ok": not errors,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "config_dir": str(config_path),
        "routing_rules_path": str(routing_path),
        "routing_canary_policy_path": str(config_path / _CANARY_FILE),
        "summary": {
            "candidate_count": len(candidates),
            "promotion_action_count": len(actions),
            "omitted_count": len(omitted),
            "error_count": len(errors),
        },
        "actions": actions,
        "omitted": omitted,
        "errors": errors,
        "privacy": _privacy_summary(),
    }


def _canary_matches(candidate_ids: set[str], source_policy_ids: set[str], value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    identifiers = {
        _string(value.get("target_candidate_id")),
        _string(value.get("candidate_id")),
        _string(value.get("promotion_action_id")),
        _string(value.get("policy_id")),
        _string(value.get("rule_id")),
    }
    identifiers.discard("")
    return bool((identifiers & candidate_ids) or (identifiers & source_policy_ids))


def _remove_canary_entries(data: dict[str, Any], actions: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    candidate_ids = {_string(action.get("target_candidate_id")) for action in actions if _string(action.get("target_candidate_id"))}
    source_policy_ids = {_string(action.get("source_policy_id")) for action in actions if _string(action.get("source_policy_id"))}
    removed = 0
    updated = dict(data)

    phase = updated.get("phase_canary")
    if _canary_matches(candidate_ids, source_policy_ids, phase):
        updated["phase_canary"] = {"enabled": False, "canary_fraction": 0.0, "holdout_fraction": 0.0}
        removed += 1

    for key in ("canaries", "routing_canaries", "phase_canaries"):
        rows = updated.get(key)
        if not isinstance(rows, list):
            continue
        kept = [row for row in rows if not _canary_matches(candidate_ids, source_policy_ids, row)]
        removed += len(rows) - len(kept)
        updated[key] = kept

    if _canary_matches(candidate_ids, source_policy_ids, updated):
        updated["enabled"] = False
        updated["canary_fraction"] = 0.0
        updated["holdout_fraction"] = 0.0
        removed += 1
    return updated, removed


def apply_routing_canary_promotion_plan(
    plan: dict[str, Any],
    *,
    config_dir: str | Path,
    dry_run: bool = True,
    backup_id: str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        errors.append({"path": "$.schema", "message": f"expected {SCHEMA}"})
        actions: list[dict[str, Any]] = []
    else:
        _scan_raw_like(plan, "$", errors)
        actions = [action for action in plan.get("actions") or [] if isinstance(action, dict)]

    config_path = Path(config_dir).expanduser()
    routing_path = config_path / _ROUTING_FILE
    canary_path = config_path / _CANARY_FILE
    files: list[dict[str, Any]] = []

    routing_data, routing_old = _load_yaml(routing_path)
    canary_data, canary_old = _load_yaml(canary_path)
    if not isinstance(routing_data.get("rules"), list):
        routing_data["rules"] = []

    applied_actions: list[dict[str, Any]] = []
    if not errors:
        for action in actions:
            rule = action.get("permanent_rule") if isinstance(action.get("permanent_rule"), dict) else None
            if not rule:
                errors.append({"path": "$.actions[].permanent_rule", "message": "missing permanent rule"})
                continue
            if any(isinstance(existing, dict) and _same_rule(existing, rule) for existing in routing_data["rules"]):
                status = "unchanged"
            else:
                routing_data["rules"].insert(0, rule)
                status = "planned"
            applied_actions.append({
                "action_id": action.get("action_id"),
                "status": status,
                "action_type": "promote",
                "target_candidate_id": action.get("target_candidate_id"),
                "rule_id": rule.get("id"),
                "target_model": action.get("target_model"),
            })

        routing_data, routing_removed = _remove_canary_entries(routing_data, actions)
        canary_data, canary_removed = _remove_canary_entries(canary_data, actions)
        routing_new = yaml.safe_dump(routing_data, sort_keys=False)
        canary_new = yaml.safe_dump(canary_data, sort_keys=False)

        routing_changed = routing_old != routing_new
        canary_changed = canary_old is not None and canary_old != canary_new
        routing_backup = None
        canary_backup = None
        if not dry_run:
            if routing_changed:
                routing_backup = _write_policy_file(routing_path, routing_new, backup_id=backup_id)
            if canary_changed:
                canary_backup = _write_policy_file(canary_path, canary_new, backup_id=backup_id)
        files.append({
            "section": "routing",
            "path": str(routing_path),
            "changed": bool(routing_changed),
            "backup_path": routing_backup,
            "bytes_after": len(routing_new.encode("utf-8")),
            "removed_canary_entries": routing_removed,
        })
        files.append({
            "section": "routing_canary_policy",
            "path": str(canary_path),
            "changed": bool(canary_changed),
            "backup_path": canary_backup,
            "bytes_after": len(canary_new.encode("utf-8")),
            "removed_canary_entries": canary_removed,
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
        "actions": applied_actions,
        "files": files,
        "summary": {
            "action_count": len(actions),
            "planned_action_count": sum(1 for action in applied_actions if action.get("status") == "planned"),
            "unchanged_action_count": sum(1 for action in applied_actions if action.get("status") == "unchanged"),
            "removed_canary_entry_count": sum(_as_int(file.get("removed_canary_entries")) for file in files),
            "error_count": len(errors),
        },
        "errors": errors,
        "privacy": _privacy_summary(),
    }
