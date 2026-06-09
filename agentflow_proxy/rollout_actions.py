from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.pattern_rollout import PATTERN_ROLLOUT_SCHEMA, normalize_pattern_rollout
from agentflow_proxy.policy_bundle import (
    MANAGED_POLICY_VERIFICATION_SECRET_ENV,
    POLICY_BUNDLE_PROVENANCE_SCHEMA,
    _hmac_signature,
    _normalize_signature,
    _secret_for_key_id,
)
from agentflow_proxy.store import utc_now


PATTERN_ROLLOUT_ACTION_SCHEMA = "agentflow.pattern_rollout_action.v1"
PATTERN_ROLLOUT_ACTIONS_SCHEMA = "agentflow.pattern_rollout_actions.v1"
PATTERN_ROLLOUT_ACTION_REVIEW_SCHEMA = "agentflow.pattern_rollout_actions_review.v1"
PATTERN_ROLLOUT_ACTION_APPLY_SCHEMA = "agentflow.pattern_rollout_actions_apply.v1"
PATTERN_ROLLOUT_ACTION_VALIDATION_SCHEMA = "agentflow.pattern_rollout_actions_validation.v1"
PATTERN_ROLLOUT_ACTION_PROVENANCE_VERIFICATION_SCHEMA = "agentflow.pattern_rollout_actions_provenance_verification.v1"

ROLLOUT_ACTION_TYPES = {
    "widen",
    "hold",
    "rollback",
    "retire",
    "disable",
    "more-samples",
    "request-more-samples",
}
_POLICY_SECTION_FILES = {
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
}
_SAFE_POLICY_SOURCES = {"managed-recommended"}
_RAW_LIKE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "command",
    "content",
    "credential",
    "file_content",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_pattern",
    "raw_request",
    "raw_response",
    "secret",
    "system",
    "tool_payload",
    "transcript",
)
_ALLOWED_RAW_KEYS = {
    "raw_body_storage",
    "raw_payloads_returned",
    "raw_prompts_included",
    "raw_params_included",
    "raw_responses_included",
    "raw_tool_payloads_included",
    "raw_provider_bodies_included",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _action_bundle_payload_for_hash(bundle: dict[str, Any]) -> dict[str, Any]:
    payload = _json_roundtrip(bundle)
    if isinstance(payload, dict):
        payload.pop("provenance", None)
    return payload if isinstance(payload, dict) else {}


def canonical_rollout_action_bundle_hash(bundle: Any) -> str | None:
    if not isinstance(bundle, dict):
        return None
    payload = _action_bundle_payload_for_hash(bundle)
    return f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"


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


def _normalize_pattern_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    digest = text.removeprefix("sha256:") if text.startswith("sha256:") else text
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return f"sha256:{digest}"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _truthy(value: Any) -> bool:
    if value in (None, False, 0, "", [], {}):
        return False
    return True


def _scan_raw_like(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    _add_error(errors, child_path, "raw or prompt-like rollout action payloads are not accepted")
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:200]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


def verify_rollout_action_provenance(bundle: Any) -> dict[str, Any]:
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    managed_bundle = isinstance(bundle, dict) and bundle.get("schema") == PATTERN_ROLLOUT_ACTIONS_SCHEMA
    secret, configured = _secret_for_key_id(provenance.get("key_id") if isinstance(provenance, dict) else None)
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_PROVENANCE_VERIFICATION_SCHEMA,
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
        "computed_bundle_hash": canonical_rollout_action_bundle_hash(bundle),
        "signature_present": False,
        "errors": [],
        "warnings": [],
    }

    if not configured:
        result["status"] = "not-configured"
        if managed_bundle:
            result["warnings"].append({
                "path": "$.provenance",
                "message": f"managed rollout action provenance was not verified because {MANAGED_POLICY_VERIFICATION_SECRET_ENV} is not configured",
            })
        return result

    if not isinstance(provenance, dict):
        result["ok"] = False
        result["status"] = "missing"
        result["errors"].append({
            "path": "$.provenance",
            "message": "managed rollout action bundle is missing provenance required by configured verification",
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
    if provenance.get("bundle_hash") != canonical_rollout_action_bundle_hash(bundle):
        _add_error(errors, "$.provenance.bundle_hash", "bundle hash does not match canonical payload")
    signature = _normalize_signature(provenance.get("signature"))
    if not signature:
        _add_error(errors, "$.provenance.signature", "expected HMAC signature")
    elif secret is None:
        _add_error(errors, "$.provenance.key_id", "no configured verification secret for key_id")
    elif not hmac.compare_digest(signature, _hmac_signature(provenance, secret)):
        _add_error(errors, "$.provenance.signature", "HMAC signature does not match provenance metadata")

    result["errors"] = errors
    if errors:
        result["status"] = "invalid"
        result["ok"] = False
    else:
        result["status"] = "verified"
        result["ok"] = True
    return result


def attach_rollout_action_provenance(
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
        "bundle_hash": canonical_rollout_action_bundle_hash(signed),
    }
    provenance["signature"] = _hmac_signature(provenance, secret)
    signed["provenance"] = provenance
    return signed


def validate_rollout_action_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_rollout_action_provenance(bundle)

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "rollout action bundle must be a JSON object")
        return {
            "schema": PATTERN_ROLLOUT_ACTION_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
        }
    if bundle.get("schema") != PATTERN_ROLLOUT_ACTIONS_SCHEMA:
        _add_error(errors, "$.schema", f"expected {PATTERN_ROLLOUT_ACTIONS_SCHEMA}")
    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")
    actions = bundle.get("actions")
    if not isinstance(actions, list):
        _add_error(errors, "$.actions", "expected list")
    else:
        for index, action in enumerate(actions):
            _validate_rollout_action(action, f"$.actions[{index}]", errors)
    _scan_raw_like(bundle, "$", errors)
    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            _add_warning(warnings, str(warning.get("path") or "$.provenance"), str(warning.get("message") or "provenance was not verified"))
    return {
        "schema": PATTERN_ROLLOUT_ACTION_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle.get("schema"),
        "action_count": len(actions) if isinstance(actions, list) else 0,
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _validate_rollout_action(action: Any, path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(action, dict):
        _add_error(errors, path, "expected rollout action object")
        return
    if action.get("schema") != PATTERN_ROLLOUT_ACTION_SCHEMA:
        _add_error(errors, f"{path}.schema", f"expected {PATTERN_ROLLOUT_ACTION_SCHEMA}")
    action_type = str(action.get("action_type") or "").strip()
    if action_type not in ROLLOUT_ACTION_TYPES:
        _add_error(errors, f"{path}.action_type", "expected widen, hold, rollback, retire, disable, or more-samples")
    if action.get("policy_section") not in _POLICY_SECTION_FILES:
        _add_error(errors, f"{path}.policy_section", "expected crunch or cache")
    if not isinstance(action.get("target_candidate_id"), str) or not action.get("target_candidate_id").strip():
        _add_error(errors, f"{path}.target_candidate_id", "expected non-empty string")
    if action.get("target_rule_id") is not None and (not isinstance(action.get("target_rule_id"), str) or not action.get("target_rule_id").strip()):
        _add_error(errors, f"{path}.target_rule_id", "expected non-empty string when present")
    if _normalize_pattern_hash(action.get("pattern_hash")) is None:
        _add_error(errors, f"{path}.pattern_hash", "expected sha256 pattern hash")
    for key in ("current_fraction", "recommended_fraction", "confidence"):
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


def _load_policy_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    return parsed if isinstance(parsed, dict) else {}, text


def _rule_hashes(rule: dict[str, Any]) -> set[str]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    values = conditions.get("pattern_hashes", conditions.get("pattern_hash"))
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return set()
    return {normalized for value in values if (normalized := _normalize_pattern_hash(value))}


def _find_rule(rules: list[Any], action: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    pattern_hash = _normalize_pattern_hash(action.get("pattern_hash"))
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        candidate_id = str(rule.get("candidate_id") or rule.get("recommendation_id") or rule.get("policy_id") or "").strip()
        rule_id_match = not target_rule_id or rule_id == target_rule_id
        candidate_id_match = not target_candidate_id or candidate_id == target_candidate_id
        hash_match = pattern_hash in _rule_hashes(rule)
        if rule_id_match and candidate_id_match and hash_match:
            return index, rule
    return None, None


def _rule_rollout(rule: dict[str, Any]) -> dict[str, Any]:
    rollout = normalize_pattern_rollout(rule.get("rollout"))
    if rollout is None:
        rollout = {
            "schema": PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "canary",
            "canary_enabled": True,
            "canary_fraction": 1.0,
            "canary_salt": "",
            "canary_unit": "request_fingerprint",
        }
    return rollout


def _plan_rule_edit(rule: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "")
    if action_type == "request-more-samples":
        action_type = "more-samples"
    current_rollout = _rule_rollout(rule)
    current_fraction = _as_float(current_rollout.get("canary_fraction"), 1.0)
    recommended_fraction = _as_float(action.get("recommended_fraction"), current_fraction)
    disable = action_type in {"rollback", "retire", "disable"}
    proposed_enabled = False if disable else bool(rule.get("enabled", True))
    proposed_rollout = dict(current_rollout)
    proposed_rollout["schema"] = str(proposed_rollout.get("schema") or PATTERN_ROLLOUT_SCHEMA)
    proposed_rollout["canary_fraction"] = 0.0 if disable else recommended_fraction
    proposed_rollout["canary_enabled"] = False if disable else bool(proposed_rollout.get("canary_enabled", True))
    proposed_rollout["recommendation_mode"] = "disabled-by-rollout-action" if disable else str(
        proposed_rollout.get("recommendation_mode") or "canary"
    )
    changed = bool(rule.get("enabled", True)) != proposed_enabled or normalize_pattern_rollout(rule.get("rollout")) != normalize_pattern_rollout(proposed_rollout)
    return {
        "action_type": action_type,
        "disable": disable,
        "current_fraction": current_fraction,
        "recommended_fraction": recommended_fraction,
        "proposed_enabled": proposed_enabled,
        "proposed_rollout": proposed_rollout,
        "changed": changed,
    }


def plan_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    validation = validate_rollout_action_bundle(bundle)
    requested_sections = set(sections or _POLICY_SECTION_FILES)
    config_path = Path(config_dir).expanduser()
    actions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    files: dict[str, dict[str, Any]] = {}

    if validation["ok"] and (invalid := sorted(requested_sections - set(_POLICY_SECTION_FILES))):
        for section in invalid:
            _add_error(errors, f"$.sections.{section}", "unknown rollout action policy section")

    if validation["ok"] and not errors:
        for index, action in enumerate(bundle.get("actions") or []):
            section = str(action.get("policy_section"))
            if section not in requested_sections:
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "skipped",
                    "reason": "not-requested",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            file_plan = files.get(section)
            if file_plan is None:
                path = config_path / _POLICY_SECTION_FILES[section]
                data, old_text = _load_policy_yaml(path)
                rules = data.get("pattern_rules")
                if not isinstance(rules, list):
                    rules = []
                    data["pattern_rules"] = rules
                file_plan = {"section": section, "path": path, "data": data, "old_text": old_text, "rules": rules}
                files[section] = file_plan

            rule_index, rule = _find_rule(file_plan["rules"], action)
            if rule is None or rule_index is None:
                _add_error(errors, f"$.actions[{index}]", "rollout action targets an unknown local pattern rule")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unknown-rule",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                    "pattern_hash": action.get("pattern_hash"),
                })
                continue
            policy_source = str(rule.get("policy_source") or "")
            if policy_source not in _SAFE_POLICY_SOURCES:
                _add_error(errors, f"$.actions[{index}]", "rollout action targets a rule with an unsafe policy source")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unsafe-policy-source",
                    "policy_section": section,
                    "rule_id": rule.get("id") or rule.get("rule_id"),
                    "policy_source": policy_source,
                })
                continue
            edit = _plan_rule_edit(rule, action)
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": section,
                "action_type": edit["action_type"],
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "rule_id": rule.get("id") or rule.get("rule_id"),
                "rule_index": rule_index,
                "pattern_hash": _normalize_pattern_hash(action.get("pattern_hash")),
                "confidence": action.get("confidence"),
                "blockers": action.get("blockers") if isinstance(action.get("blockers"), list) else [],
                "rationale": action.get("rationale"),
                "current_rule": {
                    "enabled": bool(rule.get("enabled", True)),
                    "policy_source": policy_source,
                    "rollout": normalize_pattern_rollout(rule.get("rollout")),
                },
                "proposed_edit": {
                    "changed": edit["changed"],
                    "disable": edit["disable"],
                    "current_fraction": edit["current_fraction"],
                    "recommended_fraction": edit["recommended_fraction"],
                    "enabled": edit["proposed_enabled"],
                    "rollout": edit["proposed_rollout"],
                },
            })

    ok = bool(validation["ok"] and not errors)
    return {
        "schema": PATTERN_ROLLOUT_ACTION_REVIEW_SCHEMA,
        "ok": ok,
        "config_dir": str(config_path),
        "validation": validation,
        "provenance": validation.get("provenance"),
        "action_count": validation.get("action_count", 0),
        "planned_action_count": sum(1 for action in actions if action.get("status") == "planned"),
        "rejected_action_count": sum(1 for action in actions if action.get("status") == "rejected") + len(errors),
        "changed_action_count": sum(
            1 for action in actions
            if action.get("status") == "planned" and (action.get("proposed_edit") or {}).get("changed")
        ),
        "actions": actions,
        "errors": [*validation.get("errors", []), *errors],
        "warnings": validation.get("warnings", []),
    }


def _sha256_text(value: str) -> str:
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


def apply_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    review = plan_rollout_actions(bundle, config_dir=config_dir, sections=sections)
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_APPLY_SCHEMA,
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
        result["error"] = {"type": "validation_failed", "message": "rollout actions are invalid or target unknown local rules"}
        return result

    file_sections = sorted({
        action["policy_section"]
        for action in review.get("actions", [])
        if action.get("status") == "planned"
    })
    for section in file_sections:
        path = config_path / _POLICY_SECTION_FILES[section]
        data, old_text = _load_policy_yaml(path)
        rules = data.get("pattern_rules")
        if not isinstance(rules, list):
            rules = []
            data["pattern_rules"] = rules
        section_actions = [
            action for action in review.get("actions", [])
            if action.get("status") == "planned" and action.get("policy_section") == section
        ]
        for planned in section_actions:
            rule_index = int(planned["rule_index"])
            if rule_index < 0 or rule_index >= len(rules) or not isinstance(rules[rule_index], dict):
                result["error"] = {"type": "plan_mismatch", "message": "local policy file changed after rollout action review"}
                return result
            edit = planned.get("proposed_edit") or {}
            rules[rule_index]["enabled"] = bool(edit.get("enabled"))
            rules[rule_index]["rollout"] = edit.get("rollout")
            rules[rule_index]["rollout_action"] = {
                "schema": PATTERN_ROLLOUT_ACTION_SCHEMA,
                "action_type": planned.get("action_type"),
                "target_candidate_id": planned.get("target_candidate_id"),
                "pattern_hash": planned.get("pattern_hash"),
                "confidence": planned.get("confidence"),
                "blockers": planned.get("blockers", []),
                "rationale": planned.get("rationale"),
                "reviewed_at": utc_now(),
            }
        text = yaml.safe_dump(data, sort_keys=False)
        changed = old_text != text
        backup_path = None
        if changed and not dry_run:
            backup_path = _write_policy_file(path, text)
        result["files"].append({
            "section": section,
            "path": str(path),
            "changed": bool(changed),
            "backup_path": backup_path,
            "sha256_before": _sha256_text(old_text) if old_text is not None else None,
            "sha256_after": _sha256_text(text),
            "bytes_after": len(text.encode("utf-8")),
        })
        result["applied_sections"].append(section)
    result["ok"] = True
    return result
