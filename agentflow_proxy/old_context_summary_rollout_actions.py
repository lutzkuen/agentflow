from __future__ import annotations

import difflib
import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.old_context_summary_impact import _call_rows, _cohort, _row_summary
from agentflow_proxy.policy_bundle import (
    MANAGED_POLICY_VERIFICATION_SECRET_ENV,
    POLICY_BUNDLE_PROVENANCE_SCHEMA,
    _hmac_signature,
    _normalize_signature,
    _secret_for_key_id,
)
from agentflow_proxy.store import utc_now


OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_SCHEMA = "agentflow.old_context_summary_rollout_action.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTIONS_SCHEMA = "agentflow.old_context_summary_rollout_actions.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_REVIEW_SCHEMA = "agentflow.old_context_summary_rollout_actions_review.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_APPLY_SCHEMA = "agentflow.old_context_summary_rollout_actions_apply.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_DRY_RUN_SCHEMA = "agentflow.old_context_summary_rollout_actions_dry_run.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_IMPACT_SCHEMA = "agentflow.old_context_summary_rollout_actions_impact.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_VALIDATION_SCHEMA = "agentflow.old_context_summary_rollout_actions_validation.v1"
OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_PROVENANCE_VERIFICATION_SCHEMA = "agentflow.old_context_summary_rollout_actions_provenance_verification.v1"

ACTION_TYPES = {"widen", "hold", "rollback", "retire", "disable", "more-samples", "request-more-samples"}
DISABLE_ACTION_TYPES = {"rollback", "retire", "disable"}
NO_WIDEN_ACTION_TYPES = {"hold", "more-samples", "request-more-samples"}
SAFE_POLICY_SOURCES = {"managed-recommended"}
CRUNCH_RULES_FILE = "crunch_rules.yaml"

RAW_LIKE_KEY_PARTS = (
    "account_id",
    "api_key",
    "apikey",
    "authorization",
    "cache_key",
    "content",
    "file_content",
    "generated_summary",
    "local_session",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_context",
    "raw_request",
    "raw_response",
    "request_id",
    "secret",
    "summary_text",
    "tenant_id",
    "tool_payload",
    "transcript",
)
SAFE_RAW_FLAG_KEYS = {
    "raw_body_storage",
    "raw_context_included",
    "raw_payloads_returned",
    "raw_prompts_included",
    "raw_request_bodies_included",
    "raw_responses_included",
    "raw_summaries_included",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _bundle_payload_for_hash(bundle: dict[str, Any]) -> dict[str, Any]:
    payload = _json_roundtrip(bundle)
    if isinstance(payload, dict):
        payload.pop("provenance", None)
    return payload if isinstance(payload, dict) else {}


def canonical_summary_rollout_action_bundle_hash(bundle: Any) -> str | None:
    if not isinstance(bundle, dict):
        return None
    payload = _bundle_payload_for_hash(bundle)
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def summary_rollout_action_id(action: dict[str, Any]) -> str:
    basis = {
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "action_type": action.get("action_type"),
    }
    digest = hashlib.sha256(_canonical_json(basis).encode("utf-8")).hexdigest()
    return f"old-context-summary-rollout-action:{digest[:24]}"


def _is_iso_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _add_error(errors: list[dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _add_warning(warnings: list[dict[str, str]], path: str, message: str) -> None:
    warnings.append({"path": path, "message": message})


def _truthy(value: Any) -> bool:
    return value not in (None, False, 0, "", [], {})


def _scan_raw_like(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in SAFE_RAW_FLAG_KEYS and any(part in lowered for part in RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    _add_error(errors, child_path, "raw or prompt-like old-context summary rollout action payloads are not accepted")
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:200]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


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


def verify_summary_rollout_action_provenance(bundle: Any) -> dict[str, Any]:
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    managed_bundle = isinstance(bundle, dict) and bundle.get("schema") == OLD_CONTEXT_SUMMARY_ROLLOUT_ACTIONS_SCHEMA
    secret, configured = _secret_for_key_id(provenance.get("key_id") if isinstance(provenance, dict) else None)
    result: dict[str, Any] = {
        "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_PROVENANCE_VERIFICATION_SCHEMA,
        "status": "missing",
        "ok": True,
        "managed_bundle": managed_bundle,
        "verification_configured": configured,
        "signature_required": bool(managed_bundle and configured),
        "algorithm": None,
        "issuer": None,
        "server_id": None,
        "key_id": None,
        "generated_at": None,
        "bundle_hash": None,
        "computed_bundle_hash": canonical_summary_rollout_action_bundle_hash(bundle),
        "signature_present": False,
        "errors": [],
        "warnings": [],
    }
    if not configured:
        result["status"] = "not-configured"
        if managed_bundle:
            result["warnings"].append({
                "path": "$.provenance",
                "message": f"managed old-context summary rollout action provenance was not verified because {MANAGED_POLICY_VERIFICATION_SECRET_ENV} is not configured",
            })
        return result
    if not isinstance(provenance, dict):
        result["ok"] = False
        result["errors"].append({
            "path": "$.provenance",
            "message": "managed old-context summary rollout action bundle is missing provenance required by configured verification",
        })
        return result
    for key in ("algorithm", "issuer", "server_id", "key_id", "generated_at", "bundle_hash"):
        result[key] = provenance.get(key)
    result["signature_present"] = bool(provenance.get("signature"))
    errors: list[dict[str, str]] = []
    if provenance.get("schema") != POLICY_BUNDLE_PROVENANCE_SCHEMA:
        _add_error(errors, "$.provenance.schema", f"expected {POLICY_BUNDLE_PROVENANCE_SCHEMA}")
    if provenance.get("algorithm") != "hmac-sha256":
        _add_error(errors, "$.provenance.algorithm", "expected hmac-sha256")
    for key in ("issuer", "server_id", "key_id"):
        if not isinstance(provenance.get(key), str) or not str(provenance.get(key)).strip():
            _add_error(errors, f"$.provenance.{key}", "expected non-empty string")
    if not _is_iso_datetime(provenance.get("generated_at")):
        _add_error(errors, "$.provenance.generated_at", "expected ISO-8601 timestamp string")
    if provenance.get("bundle_hash") != canonical_summary_rollout_action_bundle_hash(bundle):
        _add_error(errors, "$.provenance.bundle_hash", "bundle hash does not match canonical payload")
    signature = _normalize_signature(provenance.get("signature"))
    if not signature:
        _add_error(errors, "$.provenance.signature", "expected HMAC signature")
    elif secret is None:
        _add_error(errors, "$.provenance.key_id", "no configured verification secret for key_id")
    elif not hmac.compare_digest(signature, _hmac_signature(provenance, secret)):
        _add_error(errors, "$.provenance.signature", "HMAC signature does not match provenance metadata")
    result["errors"] = errors
    result["ok"] = not errors
    result["status"] = "invalid" if errors else "verified"
    return result


def attach_summary_rollout_action_provenance(
    bundle: dict[str, Any],
    *,
    secret: str,
    issuer: str,
    server_id: str,
    key_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    signed = _json_roundtrip(bundle)
    signed.pop("provenance", None)
    provenance = {
        "schema": POLICY_BUNDLE_PROVENANCE_SCHEMA,
        "algorithm": "hmac-sha256",
        "generated_at": generated_at or utc_now(),
        "issuer": issuer,
        "server_id": server_id,
        "key_id": key_id,
        "bundle_hash": canonical_summary_rollout_action_bundle_hash(signed),
    }
    provenance["signature"] = _hmac_signature(provenance, secret)
    signed["provenance"] = provenance
    return signed


def _quality_gate(action: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    for value in (
        action.get("quality_gate"),
        action.get("local_quality_gate"),
        bundle.get("quality_gate"),
        bundle.get("local_quality_gate"),
    ):
        if isinstance(value, dict):
            return value
    return {}


def _validate_action(action: Any, path: str, errors: list[dict[str, str]], bundle: dict[str, Any]) -> None:
    if not isinstance(action, dict):
        _add_error(errors, path, "expected old-context summary rollout action object")
        return
    if action.get("schema") != OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_SCHEMA:
        _add_error(errors, f"{path}.schema", f"expected {OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_SCHEMA}")
    action_type = str(action.get("action_type") or "").strip()
    if action_type not in ACTION_TYPES:
        _add_error(errors, f"{path}.action_type", "expected widen, hold, rollback, retire, disable, or more-samples")
    if action.get("policy_section") not in (None, "crunch"):
        _add_error(errors, f"{path}.policy_section", "expected crunch when present")
    if not isinstance(action.get("target_candidate_id"), str) or not action.get("target_candidate_id").strip():
        _add_error(errors, f"{path}.target_candidate_id", "expected non-empty string")
    if action.get("target_rule_id") is not None and (not isinstance(action.get("target_rule_id"), str) or not action.get("target_rule_id").strip()):
        _add_error(errors, f"{path}.target_rule_id", "expected non-empty string when present")
    for key in ("current_fraction", "recommended_fraction", "confidence"):
        if key not in action:
            continue
        try:
            number = float(action.get(key))
        except (TypeError, ValueError):
            _add_error(errors, f"{path}.{key}", "expected numeric value")
            continue
        if number < 0 or number > 1:
            _add_error(errors, f"{path}.{key}", "expected number between 0 and 1")
    if action.get("required_local_review") is not True:
        _add_error(errors, f"{path}.required_local_review", "expected true")
    if action.get("managed_enforced") is not False:
        _add_error(errors, f"{path}.managed_enforced", "expected false")
    gate = _quality_gate(action, bundle)
    if action_type == "widen" and gate.get("verdict") != "promote":
        _add_error(errors, f"{path}.quality_gate.verdict", "widen requires local old-context summary quality_gate verdict promote")


def validate_summary_rollout_action_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_summary_rollout_action_provenance(bundle)
    if not isinstance(bundle, dict):
        _add_error(errors, "$", "old-context summary rollout action bundle must be a JSON object")
        return {
            "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
        }
    if bundle.get("schema") != OLD_CONTEXT_SUMMARY_ROLLOUT_ACTIONS_SCHEMA:
        _add_error(errors, "$.schema", f"expected {OLD_CONTEXT_SUMMARY_ROLLOUT_ACTIONS_SCHEMA}")
    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")
    actions = bundle.get("actions")
    if not isinstance(actions, list):
        _add_error(errors, "$.actions", "expected list")
    else:
        for index, action in enumerate(actions):
            _validate_action(action, f"$.actions[{index}]", errors, bundle)
    _scan_raw_like(bundle, "$", errors)
    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            _add_warning(warnings, str(warning.get("path") or "$.provenance"), str(warning.get("message") or "provenance was not verified"))
    return {
        "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle.get("schema"),
        "action_count": len(actions) if isinstance(actions, list) else 0,
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _load_policy_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    return parsed if isinstance(parsed, dict) else {}, text


def _summary_matches(action: dict[str, Any], summary: dict[str, Any]) -> bool:
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    rule_id = str(summary.get("rule_id") or "").strip()
    candidate_id = str(summary.get("candidate_id") or "").strip()
    return (not target_rule_id or target_rule_id == rule_id) and (not target_candidate_id or target_candidate_id == candidate_id)


def _plan_summary_edit(summary: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    raw_action_type = str(action.get("action_type") or "")
    action_type = "more-samples" if raw_action_type == "request-more-samples" else raw_action_type
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    current_fraction = _as_float(canary.get("fraction"), _as_float(action.get("current_fraction"), 1.0))
    recommended_fraction = _as_float(action.get("recommended_fraction"), current_fraction)
    disable = action_type in DISABLE_ACTION_TYPES
    if action_type in NO_WIDEN_ACTION_TYPES:
        recommended_fraction = current_fraction
    proposed_canary = dict(canary)
    proposed_canary["enabled"] = False if disable else bool(proposed_canary.get("enabled", True))
    proposed_canary["fraction"] = 0.0 if disable else recommended_fraction
    proposed_summary = dict(summary)
    proposed_summary["enabled"] = False if disable else bool(summary.get("enabled", True))
    proposed_summary["canary"] = proposed_canary
    changed = summary.get("enabled", True) != proposed_summary["enabled"] or (summary.get("canary") or {}) != proposed_canary
    return {
        "action_type": action_type,
        "disable": disable,
        "current_fraction": current_fraction,
        "recommended_fraction": recommended_fraction,
        "proposed_summary": proposed_summary,
        "changed": changed,
    }


def plan_summary_rollout_actions(bundle: Any, *, config_dir: str | Path) -> dict[str, Any]:
    validation = validate_summary_rollout_action_bundle(bundle)
    config_path = Path(config_dir).expanduser()
    path = config_path / CRUNCH_RULES_FILE
    data, old_text = _load_policy_yaml(path)
    summary = data.get("old_context_summarization") if isinstance(data.get("old_context_summarization"), dict) else None
    actions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if validation["ok"]:
        for index, action in enumerate(bundle.get("actions") or []):
            if summary is None or not _summary_matches(action, summary):
                _add_error(errors, f"$.actions[{index}]", "old-context summary rollout action targets an unknown local summary rule")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unknown-rule",
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            policy_source = str(summary.get("policy_source") or "")
            if policy_source not in SAFE_POLICY_SOURCES:
                _add_error(errors, f"$.actions[{index}]", "old-context summary rollout action targets a rule with an unsafe policy source")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unsafe-policy-source",
                    "policy_source": policy_source,
                })
                continue
            edit = _plan_summary_edit(summary, action)
            gate = _quality_gate(action, bundle)
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": "crunch",
                "action_type": edit["action_type"],
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "rule_id": summary.get("rule_id"),
                "candidate_id": summary.get("candidate_id"),
                "confidence": action.get("confidence"),
                "blockers": action.get("blockers") if isinstance(action.get("blockers"), list) else [],
                "rationale": action.get("rationale"),
                "quality_gate": {
                    "verdict": gate.get("verdict"),
                    "reason_codes": gate.get("reason_codes") if isinstance(gate.get("reason_codes"), list) else [],
                    "warning_codes": gate.get("warning_codes") if isinstance(gate.get("warning_codes"), list) else [],
                },
                "current_rule": {
                    "enabled": bool(summary.get("enabled", True)),
                    "policy_source": policy_source,
                    "canary": summary.get("canary") if isinstance(summary.get("canary"), dict) else {},
                },
                "proposed_edit": {
                    "changed": edit["changed"],
                    "disable": edit["disable"],
                    "current_fraction": edit["current_fraction"],
                    "recommended_fraction": edit["recommended_fraction"],
                    "enabled": edit["proposed_summary"]["enabled"],
                    "canary": edit["proposed_summary"]["canary"],
                },
            })
    ok = bool(validation["ok"] and not errors)
    return {
        "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_REVIEW_SCHEMA,
        "ok": ok,
        "config_dir": str(config_path),
        "file": {"section": "crunch", "path": str(path), "exists": old_text is not None},
        "validation": validation,
        "provenance": validation.get("provenance"),
        "action_count": validation.get("action_count", 0),
        "planned_action_count": sum(1 for action in actions if action.get("status") == "planned"),
        "rejected_action_count": sum(1 for action in actions if action.get("status") == "rejected") + len(errors),
        "changed_action_count": sum(1 for action in actions if action.get("status") == "planned" and (action.get("proposed_edit") or {}).get("changed")),
        "actions": actions,
        "errors": [*validation.get("errors", []), *errors],
        "warnings": validation.get("warnings", []),
    }


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _policy_file_diff(path: Path, before: str | None, after: str) -> str:
    return "".join(difflib.unified_diff(
        (before or "").splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{path}.before",
        tofile=f"{path}.after",
    ))


def apply_summary_rollout_actions(bundle: Any, *, config_dir: str | Path, dry_run: bool = False) -> dict[str, Any]:
    review = plan_summary_rollout_actions(bundle, config_dir=config_dir)
    config_path = Path(config_dir).expanduser()
    path = config_path / CRUNCH_RULES_FILE
    result: dict[str, Any] = {
        "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_APPLY_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "applied_sections": [],
        "files": [],
        "actions": review.get("actions", []),
        "error": None,
    }
    if not review.get("ok"):
        result["error"] = {"type": "validation_failed", "message": "old-context summary rollout actions are invalid or target unknown local rules"}
        return result
    data, old_text = _load_policy_yaml(path)
    summary = data.get("old_context_summarization") if isinstance(data.get("old_context_summarization"), dict) else {}
    if not isinstance(summary, dict):
        result["error"] = {"type": "plan_mismatch", "message": "local crunch policy changed after rollout action review"}
        return result
    for planned in result["actions"]:
        if planned.get("status") != "planned":
            continue
        edit = planned.get("proposed_edit") or {}
        summary["enabled"] = bool(edit.get("enabled"))
        summary["canary"] = edit.get("canary")
        summary["rollout_action"] = {
            "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_SCHEMA,
            "action_type": planned.get("action_type"),
            "target_candidate_id": planned.get("target_candidate_id"),
            "target_rule_id": planned.get("target_rule_id"),
            "confidence": planned.get("confidence"),
            "quality_gate_verdict": (planned.get("quality_gate") or {}).get("verdict"),
            "quality_gate_reason_codes": (planned.get("quality_gate") or {}).get("reason_codes"),
            "blockers": planned.get("blockers", []),
            "rationale": planned.get("rationale"),
            "reviewed_at": utc_now(),
        }
    data["old_context_summarization"] = summary
    text = yaml.safe_dump(data, sort_keys=False)
    changed = old_text != text
    backup_path = None
    if changed and not dry_run:
        backup_path = _write_policy_file(path, text)
    file_result = {
        "section": "crunch",
        "path": str(path),
        "changed": bool(changed),
        "backup_path": backup_path,
        "sha256_before": _sha256_text(old_text),
        "sha256_after": _sha256_text(text),
        "bytes_after": len(text.encode("utf-8")),
    }
    if dry_run and changed:
        file_result["diff"] = _policy_file_diff(path, old_text, text)
    result["files"].append(file_result)
    result["applied_sections"].append("crunch")
    result["ok"] = True
    return result


def _summary_rows(store_obj: Any, *, limit: int, since: str | None = None) -> list[dict[str, Any]]:
    rows = _call_rows(store_obj, limit=max(1, min(int(limit or 500), 5000)), since=since)
    return [summary for row in rows if (summary := _row_summary(row)) is not None]


def _matches_action_summary(summary: dict[str, Any], action: dict[str, Any]) -> bool:
    meta = summary.get("summary") if isinstance(summary.get("summary"), dict) else {}
    return _summary_matches(action, meta)


def _actual_counts(matched: list[dict[str, Any]]) -> dict[str, Any]:
    applied = holdout = bypass = failures = 0
    tokens_saved = 0
    net_savings = 0.0
    reason_counts: dict[str, int] = {}
    for item in matched:
        meta = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        cohort = _cohort(meta)
        if cohort == "canary_applied":
            applied += 1
        elif cohort == "canary_holdout":
            holdout += 1
        elif cohort == "bypassed_or_disabled":
            bypass += 1
        failures += int(_as_int(meta.get("summary_status_code"), 0) >= 400 or bool(meta.get("summary_error")))
        tokens_saved += _as_int(meta.get("tokens_saved_est"))
        net_savings += float(meta.get("estimated_net_savings_usd") or 0.0)
        reason = str(meta.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "matched_metadata_row_count": len(matched),
        "canary_applied_count": applied,
        "canary_holdout_count": holdout,
        "bypassed_or_disabled_count": bypass,
        "summary_failure_count": failures,
        "tokens_saved_est": tokens_saved,
        "net_savings_usd": round(net_savings, 8),
        "reason_buckets": [{"value": key, "count": reason_counts[key]} for key in sorted(reason_counts)],
    }


def _projection(matched_count: int, actual: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    current_fraction = _as_float(edit.get("current_fraction"), 0.0)
    projected_fraction = _as_float(edit.get("recommended_fraction"), current_fraction)
    disable = bool(edit.get("disable"))
    if disable:
        return {
            "current_fraction": current_fraction,
            "projected_fraction": 0.0,
            "current_canary_applied_count": actual["canary_applied_count"],
            "current_canary_holdout_count": actual["canary_holdout_count"],
            "current_bypassed_or_disabled_count": actual["bypassed_or_disabled_count"],
            "projected_canary_applied_count": 0,
            "projected_canary_holdout_count": 0,
            "projected_local_bypass_or_disable_count": matched_count,
            "projected_additional_applied_count": 0,
        }
    projected_applied = int(round(matched_count * projected_fraction))
    projected_holdout = max(0, matched_count - projected_applied)
    return {
        "current_fraction": current_fraction,
        "projected_fraction": projected_fraction,
        "current_canary_applied_count": actual["canary_applied_count"],
        "current_canary_holdout_count": actual["canary_holdout_count"],
        "current_bypassed_or_disabled_count": actual["bypassed_or_disabled_count"],
        "projected_canary_applied_count": projected_applied,
        "projected_canary_holdout_count": projected_holdout,
        "projected_local_bypass_or_disable_count": 0,
        "projected_additional_applied_count": max(0, projected_applied - actual["canary_applied_count"]),
    }


def dry_run_summary_rollout_actions(
    bundle: Any,
    *,
    store_obj: Any,
    config_dir: str | Path,
    limit: int = 500,
) -> dict[str, Any]:
    review = plan_summary_rollout_actions(bundle, config_dir=config_dir)
    result: dict[str, Any] = {
        "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_DRY_RUN_SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "bundle_hash": canonical_summary_rollout_action_bundle_hash(bundle),
        "dry_run": True,
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "config_dir": str(Path(config_dir).expanduser()),
        "lookback_limit": max(1, min(int(limit or 500), 5000)),
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "actions": [],
        "summary": {},
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
        },
        "errors": review.get("errors", []),
        "warnings": review.get("warnings", []),
    }
    if not review.get("validation", {}).get("ok"):
        result["error"] = {"type": "validation_failed", "message": "old-context summary rollout action bundle failed validation"}
        return result
    if not review.get("ok"):
        result["error"] = {"type": "review_failed", "message": "old-context summary rollout actions failed local review"}
        return result
    summaries = _summary_rows(store_obj, limit=result["lookback_limit"])
    actions: list[dict[str, Any]] = []
    for planned in review.get("actions", []):
        if planned.get("status") != "planned":
            continue
        matched = [item for item in summaries if _matches_action_summary(item, planned)]
        actual = _actual_counts(matched)
        projection = _projection(len(matched), actual, planned.get("proposed_edit") or {})
        actions.append({
            "path": planned.get("path"),
            "action_id": summary_rollout_action_id(planned),
            "status": planned.get("status"),
            "policy_section": "crunch",
            "action_type": planned.get("action_type"),
            "target_candidate_id": planned.get("target_candidate_id"),
            "target_rule_id": planned.get("target_rule_id"),
            "rule_id": planned.get("rule_id"),
            "quality_gate": planned.get("quality_gate"),
            "affected_metadata_row_count": len(matched),
            **projection,
            "historical_tokens_saved_est": actual["tokens_saved_est"],
            "historical_net_savings_usd": actual["net_savings_usd"],
            "summary_failure_count": actual["summary_failure_count"],
            "summary_reason_buckets": actual["reason_buckets"],
            "proposed_edit": planned.get("proposed_edit"),
        })
    result.update({
        "ok": True,
        "actions": actions,
        "summary": {
            "sampled_provider_calls": len(_call_rows(store_obj, limit=result["lookback_limit"], since=None)),
            "old_context_summary_metadata_row_count": len(summaries),
            "affected_metadata_row_count": sum(_as_int(action.get("affected_metadata_row_count")) for action in actions),
            "projected_additional_applied_count": sum(_as_int(action.get("projected_additional_applied_count")) for action in actions),
            "projected_local_bypass_or_disable_count": sum(_as_int(action.get("projected_local_bypass_or_disable_count")) for action in actions),
            "historical_tokens_saved_est": sum(_as_int(action.get("historical_tokens_saved_est")) for action in actions),
            "historical_net_savings_usd": round(sum(float(action.get("historical_net_savings_usd") or 0.0) for action in actions), 8),
        },
    })
    return result


def measure_summary_rollout_action_impact(
    dry_run_report: Any,
    *,
    store_obj: Any,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 5000))
    result: dict[str, Any] = {
        "schema": OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_IMPACT_SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "post_apply_since": since,
        "actions": [],
        "summary": {},
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
        },
    }
    if not isinstance(dry_run_report, dict) or dry_run_report.get("schema") != OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_DRY_RUN_SCHEMA:
        result["error"] = {"type": "invalid_dry_run_report", "message": f"expected {OLD_CONTEXT_SUMMARY_ROLLOUT_ACTION_DRY_RUN_SCHEMA}"}
        return result
    if not dry_run_report.get("ok"):
        result["error"] = {"type": "dry_run_not_ok", "message": "old-context summary rollout action dry-run report was not successful"}
        return result
    since_value = since or dry_run_report.get("generated_at")
    result["post_apply_since"] = since_value
    summaries = _summary_rows(store_obj, limit=lookback_limit, since=since_value)
    actions: list[dict[str, Any]] = []
    for dry_action in dry_run_report.get("actions") or []:
        if not isinstance(dry_action, dict):
            continue
        matched = [item for item in summaries if _matches_action_summary(item, dry_action)]
        actual = _actual_counts(matched)
        projection = {
            "affected_metadata_row_count": _as_int(dry_action.get("affected_metadata_row_count")),
            "projected_canary_applied_count": _as_int(dry_action.get("projected_canary_applied_count")),
            "projected_canary_holdout_count": _as_int(dry_action.get("projected_canary_holdout_count")),
            "projected_local_bypass_or_disable_count": _as_int(dry_action.get("projected_local_bypass_or_disable_count")),
        }
        actions.append({
            "path": dry_action.get("path"),
            "status": "matched" if matched else "no-post-apply-matches",
            "action_id": dry_action.get("action_id") or summary_rollout_action_id(dry_action),
            "policy_section": "crunch",
            "action_type": dry_action.get("action_type"),
            "target_candidate_id": dry_action.get("target_candidate_id"),
            "target_rule_id": dry_action.get("target_rule_id"),
            "quality_gate": dry_action.get("quality_gate"),
            "projection": projection,
            "actual": {
                "matched_metadata_row_count": actual["matched_metadata_row_count"],
                "actual_canary_applied_count": actual["canary_applied_count"],
                "actual_canary_holdout_count": actual["canary_holdout_count"],
                "actual_bypassed_or_disabled_count": actual["bypassed_or_disabled_count"],
                "actual_tokens_saved_est": actual["tokens_saved_est"],
                "actual_net_savings_usd": actual["net_savings_usd"],
                "summary_failure_count": actual["summary_failure_count"],
            },
            "delta": {
                "matched_vs_projected_affected_delta": actual["matched_metadata_row_count"] - projection["affected_metadata_row_count"],
                "applied_vs_projected_delta": actual["canary_applied_count"] - projection["projected_canary_applied_count"],
                "holdout_vs_projected_delta": actual["canary_holdout_count"] - projection["projected_canary_holdout_count"],
                "bypass_or_disabled_vs_projected_delta": actual["bypassed_or_disabled_count"] - projection["projected_local_bypass_or_disable_count"],
            },
        })
    total_actual = sum(_as_int((action.get("actual") or {}).get("matched_metadata_row_count")) for action in actions)
    result.update({
        "ok": True,
        "status": "matched" if total_actual else "no-post-apply-matches",
        "actions": actions,
        "summary": {
            "sampled_provider_calls": len(_call_rows(store_obj, limit=lookback_limit, since=since_value)),
            "old_context_summary_metadata_row_count": len(summaries),
            "action_count": len(actions),
            "projected_affected_metadata_row_count": sum(_as_int((action.get("projection") or {}).get("affected_metadata_row_count")) for action in actions),
            "actual_matched_metadata_row_count": total_actual,
            "actual_canary_applied_count": sum(_as_int((action.get("actual") or {}).get("actual_canary_applied_count")) for action in actions),
            "actual_canary_holdout_count": sum(_as_int((action.get("actual") or {}).get("actual_canary_holdout_count")) for action in actions),
            "actual_bypassed_or_disabled_count": sum(_as_int((action.get("actual") or {}).get("actual_bypassed_or_disabled_count")) for action in actions),
            "actual_tokens_saved_est": sum(_as_int((action.get("actual") or {}).get("actual_tokens_saved_est")) for action in actions),
            "actual_net_savings_usd": round(sum(float((action.get("actual") or {}).get("actual_net_savings_usd") or 0.0) for action in actions), 8),
            "actions_without_post_apply_matches": sum(1 for action in actions if action.get("status") == "no-post-apply-matches"),
        },
    })
    return result
