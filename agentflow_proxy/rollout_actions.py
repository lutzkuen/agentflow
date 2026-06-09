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
PATTERN_ROLLOUT_ACTION_DRY_RUN_SCHEMA = "agentflow.pattern_rollout_actions_dry_run.v1"
PATTERN_ROLLOUT_ACTION_IMPACT_SCHEMA = "agentflow.pattern_rollout_actions_impact.v1"
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
_PATTERN_ACTION_VALIDATION_DETAIL_SCHEMA = "agentflow.pattern_rollout_action_family_validation.v1"
_PATTERN_FAMILY_ALIASES = {
    "tool-result": "tool_results",
    "tool_results": "tool_results",
    "tool_result": "tool_results",
    "tool-results": "tool_results",
    "diff": "diffs",
    "diffs": "diffs",
    "generated-artifact": "generated_artifacts",
    "generated-artifacts": "generated_artifacts",
    "generated_artifact": "generated_artifacts",
    "generated_artifacts": "generated_artifacts",
    "tabular": "tabular_data",
    "tabular-data": "tabular_data",
    "tabular_data": "tabular_data",
    "terminal-log": "terminal_logs",
    "terminal-logs": "terminal_logs",
    "terminal_log": "terminal_logs",
    "terminal_logs": "terminal_logs",
    "cacheability": "cacheability",
    "cacheable": "cacheability",
}
_PATTERN_FAMILY_SECTIONS = {
    "tool_results": {"crunch", "cache"},
    "diffs": {"crunch", "cache"},
    "generated_artifacts": {"crunch", "cache"},
    "tabular_data": {"crunch", "cache"},
    "terminal_logs": {"crunch"},
    "cacheability": {"cache"},
}
_PATTERN_HOLDOUT_UNITS = {
    "request_fingerprint",
    "source_hash",
    "pattern_hash",
    "session_id_hash",
    "traffic_fingerprint",
    "optimization_unit_id",
}
_LOSSY_PATTERN_PROFILES = {
    "aggressive",
    "drop",
    "drop-middle",
    "lossy",
    "omit",
    "semantic",
    "truncate",
    "truncation",
}
_UNSAFE_CACHE_REPLAYABILITY_LEVELS = {
    "current-state",
    "current_state",
    "live",
    "local-state",
    "local_state",
    "user-specific",
    "user_specific",
    "workspace-state",
    "workspace_state",
}
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


def rollout_action_id(action: dict[str, Any]) -> str:
    basis = {
        "policy_section": action.get("policy_section"),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "pattern_hash": action.get("pattern_hash"),
        "action_type": action.get("action_type"),
    }
    digest = hashlib.sha256(_canonical_json(basis).encode("utf-8")).hexdigest()
    return f"rollout-action:{digest[:24]}"


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


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "-")


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_pattern_family(value: Any) -> str | None:
    text = _norm_text(value).replace("_", "-")
    if not text:
        return None
    if text in _PATTERN_FAMILY_ALIASES:
        return _PATTERN_FAMILY_ALIASES[text]
    for alias, family in _PATTERN_FAMILY_ALIASES.items():
        if alias in text:
            return family
    return None


def _pattern_family_from(*sources: Any) -> str | None:
    keys = (
        "module_family",
        "pattern_family",
        "candidate_family",
        "family",
        "action_family",
        "score_family",
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            family = _normalize_pattern_family(source.get(key))
            if family:
                return family
    return None


def _rollout_action_rule_family(action: dict[str, Any], rule: dict[str, Any]) -> tuple[str | None, str | None]:
    action_block = action.get("action") if isinstance(action.get("action"), dict) else {}
    rule_action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    managed = rule.get("managed_recommendation") if isinstance(rule.get("managed_recommendation"), dict) else {}
    action_family = _pattern_family_from(action, action_block)
    rule_family = _pattern_family_from(rule, conditions, rule_action, managed)
    return action_family or rule_family, rule_family


def _profile_from_action(action: dict[str, Any], rule: dict[str, Any]) -> str | None:
    action_block = action.get("action") if isinstance(action.get("action"), dict) else {}
    rule_action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    managed = rule.get("managed_recommendation") if isinstance(rule.get("managed_recommendation"), dict) else {}
    return _first_string(
        action.get("policy_profile"),
        action.get("profile"),
        action.get("crunch_profile"),
        action.get("cache_profile"),
        action.get("action_profile"),
        action_block.get("policy_profile"),
        action_block.get("profile"),
        rule.get("policy_profile"),
        rule.get("profile"),
        rule_action.get("policy_profile"),
        rule_action.get("profile"),
        managed.get("policy_profile"),
        managed.get("profile"),
    )


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    values = value if isinstance(value, list) else [value]
    return {_norm_text(item) for item in values if str(item or "").strip()}


def _rule_replayability_levels(rule: dict[str, Any], action: dict[str, Any]) -> set[str]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    action_block = action.get("action") if isinstance(action.get("action"), dict) else {}
    values: set[str] = set()
    for source in (action, action_block, conditions, rule):
        if not isinstance(source, dict):
            continue
        values.update(_as_string_set(source.get("replayability_level")))
        values.update(_as_string_set(source.get("replayability_levels")))
    return values


def _bool_field(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _validate_rollout_thresholds(rollout: Any, *, path: str) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if rollout is None:
        return errors
    if not isinstance(rollout, dict):
        _add_error(errors, path, "expected rollout object")
        return errors
    unit = str(rollout.get("canary_unit") or "request_fingerprint")
    if unit not in _PATTERN_HOLDOUT_UNITS:
        _add_error(errors, f"{path}.canary_unit", "unsupported canary holdout unit")
    for key in ("canary_fraction", "widening_threshold", "rollback_threshold"):
        if key not in rollout:
            continue
        try:
            number = float(rollout.get(key))
        except (TypeError, ValueError):
            _add_error(errors, f"{path}.{key}", "expected numeric value")
            continue
        if number < 0 or number > 1:
            _add_error(errors, f"{path}.{key}", "expected number between 0 and 1")
    if "min_outcome_samples" in rollout:
        try:
            samples = int(rollout.get("min_outcome_samples"))
        except (TypeError, ValueError):
            _add_error(errors, f"{path}.min_outcome_samples", "expected integer value")
        else:
            if samples < 0:
                _add_error(errors, f"{path}.min_outcome_samples", "expected integer >= 0")
    fields = rollout.get("local_feedback_fields")
    if fields is not None:
        if not isinstance(fields, list):
            _add_error(errors, f"{path}.local_feedback_fields", "expected list")
        else:
            for index, item in enumerate(fields):
                if not isinstance(item, str) or not item.strip():
                    _add_error(errors, f"{path}.local_feedback_fields[{index}]", "expected non-empty string")
    return errors


def _has_marker(action: dict[str, Any]) -> bool:
    return isinstance(action.get("marker"), str) and bool(action["marker"].strip())


def _exactness_marker_ok(action: dict[str, Any], *, require_generated_marker: bool = False) -> bool:
    marker = str(action.get("marker") or "")
    if not marker.strip():
        return False
    exactness = _bool_field(action.get("exactness_preserving_marker"))
    if exactness is True:
        return True
    if require_generated_marker:
        return "generated" in marker.lower() or "artifact" in marker.lower()
    return True


def _validate_family_specific_action(
    *,
    action: dict[str, Any],
    rule: dict[str, Any],
    section: str,
    path: str,
) -> dict[str, Any]:
    rule_action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    action_family, rule_family = _rollout_action_rule_family(action, rule)
    profile = _profile_from_action(action, rule)
    profile_key = _norm_text(profile)
    action_type = _norm_text(rule_action.get("type") or action.get("local_action") or action.get("action_family"))
    if not action_type:
        action_type = "exact_cache" if section == "cache" else "shorten"
    replayability_levels = _rule_replayability_levels(rule, action)
    errors: list[dict[str, str]] = []

    if action_family and rule_family and action_family != rule_family:
        _add_error(errors, f"{path}.module_family", "rollout action family does not match target local rule family")
    family = action_family or rule_family
    if family and section not in _PATTERN_FAMILY_SECTIONS.get(family, set()):
        _add_error(errors, f"{path}.policy_section", f"{family} actions are not supported for {section} rules")

    for error in _validate_rollout_thresholds(rule.get("rollout"), path=f"{path}.target_rule.rollout"):
        errors.append(error)

    if family and profile_key in _LOSSY_PATTERN_PROFILES:
        _add_error(errors, f"{path}.policy_profile", "lossy, semantic, or aggressive pattern profiles are not safe for local YAML rollout")

    if section == "crunch":
        if action_type not in {"shorten", "head-tail", "head_tail"}:
            _add_error(errors, f"{path}.target_rule.action.type", "family-specific crunch rules must use a bounded shorten action")
        if family in {"diffs", "generated_artifacts", "tabular_data", "terminal_logs"} and not _has_marker(rule_action):
            _add_error(errors, f"{path}.target_rule.action.marker", "family-specific crunch rules require an explicit local replacement marker")
        if family == "diffs":
            if action_type == "omit":
                _add_error(errors, f"{path}.target_rule.action.type", "diff crunching may not omit matched content")
            if _bool_field(rule_action.get("preserve_diff_headers")) is False or _bool_field(rule_action.get("preserve_hunk_boundaries")) is False:
                _add_error(errors, f"{path}.target_rule.action", "diff crunching must preserve diff headers and hunk boundaries")
        elif family == "generated_artifacts":
            if not _exactness_marker_ok(rule_action, require_generated_marker=True):
                _add_error(errors, f"{path}.target_rule.action.marker", "generated artifact crunching requires an exactness-preserving generated-artifact marker")
        elif family == "tool_results":
            if _bool_field(rule_action.get("preserve_tool_protocol")) is False:
                _add_error(errors, f"{path}.target_rule.action.preserve_tool_protocol", "tool-result crunching must preserve tool protocol boundaries")
        elif family == "cacheability":
            _add_error(errors, f"{path}.policy_section", "cacheability pattern actions can only write cache rules")
    elif section == "cache":
        if action_type not in {"exact_cache", "exact_cache_pattern"}:
            _add_error(errors, f"{path}.target_rule.action.type", "family-specific cache rules must use exact cache actions")
        if profile_key == "semantic":
            _add_error(errors, f"{path}.policy_profile", "semantic cache profiles are not accepted for managed pattern YAML rollout")
        if replayability_levels & _UNSAFE_CACHE_REPLAYABILITY_LEVELS:
            _add_error(errors, f"{path}.target_rule.conditions.replayability_levels", "current-state and user-specific patterns cannot be cache-enabled by managed rollout")
        if family == "tool_results":
            allow_tools = _bool_field(rule_action.get("allow_tool_calls")) is True
            safe_invalidation = _bool_field(rule_action.get("safe_invalidation")) is True or _bool_field(rule_action.get("safe_invalidation_evidence")) is True
            if allow_tools and not safe_invalidation:
                _add_error(errors, f"{path}.target_rule.action.safe_invalidation", "tool-result cache rules require safe invalidation evidence")
        if family == "terminal_logs":
            _add_error(errors, f"{path}.policy_section", "terminal-log pattern actions are not supported for cache rules")

    detail = {
        "schema": _PATTERN_ACTION_VALIDATION_DETAIL_SCHEMA,
        "status": "rejected" if errors else "accepted",
        "family": family,
        "target_rule_family": rule_family,
        "policy_section": section,
        "policy_profile": profile,
        "action_type": action_type,
        "replayability_levels": sorted(replayability_levels),
        "rollout": normalize_pattern_rollout(rule.get("rollout")),
        "errors": errors,
    }
    return detail


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


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_bucket(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _status_bucket(status_code: Any) -> str:
    status = _as_int(status_code)
    if status <= 0:
        return "unknown"
    if status < 200:
        return "lt_2xx"
    if status < 300:
        return "2xx"
    if status < 400:
        return "3xx"
    if status < 500:
        return "4xx"
    return "5xx"


def _error_bucket(error: Any, status_code: Any = None) -> str:
    status = _as_int(status_code)
    if status and status < 400 and not error:
        return "none"
    if not error:
        return f"http_{_status_bucket(status)}" if status >= 400 else "none"
    try:
        parsed = json.loads(str(error))
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        error_obj = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
        error_type = error_obj.get("type") if isinstance(error_obj, dict) else None
        if isinstance(error_type, str) and error_type:
            return error_type[:80]
    return f"http_{_status_bucket(status)}" if status >= 400 else "error"


def _latency_bucket(latency_ms: Any) -> str:
    latency = _as_int(latency_ms, -1)
    if latency < 0:
        return "unknown"
    if latency < 500:
        return "lt_500ms"
    if latency < 2000:
        return "500ms_2s"
    if latency < 10000:
        return "2s_10s"
    return "gte_10s"


def _saving_bucket(value: Any) -> str:
    amount = _as_float(value, 0.0)
    if amount <= 0:
        return "zero"
    if amount < 0.001:
        return "lt_0_001"
    if amount < 0.01:
        return "0_001_0_01"
    if amount < 0.05:
        return "0_01_0_05"
    return "gte_0_05"


def _cohort_for_summary(summary: dict[str, Any]) -> str:
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    status = str(summary.get("status") or "")
    outcome = str(summary.get("outcome") or "")
    reason = str(summary.get("reason") or "")
    cohort = str(summary.get("cohort") or canary.get("cohort") or "")
    if outcome == "holdout" or status == "holdout" or cohort == "canary_holdout" or canary.get("status") == "holdout":
        return "canary_holdout"
    if outcome == "bypassed" or status in {"bypass", "bypassed"} or "bypass" in reason or "disabled" in reason:
        return "bypassed"
    if _as_int(summary.get("applied_count")) > 0 and (canary.get("enabled") or cohort == "canary_applied"):
        return "canary_applied"
    if _as_int(summary.get("applied_count")) > 0 or outcome == "applied" or status == "applied":
        return "applied"
    return "received"


def _parse_utc_datetime(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _created_at_matches_since(raw: Any, since: datetime | None) -> bool:
    if since is None:
        return True
    created = _parse_utc_datetime(raw)
    return created is not None and created >= since


def _codex_estimated_cost(input_text_chars: Any, result_chars: Any) -> float:
    # Rollout action dry-runs are metadata-only; Codex app rows do not have provider-reported
    # billing fields yet, so expose count/risk impact without inventing model-specific spend.
    _ = input_text_chars, result_chars
    return 0.0


def _traffic_pattern_summaries(store_obj: Any, *, limit: int, since: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from agentflow_proxy.recommendations import pattern_decision_summaries

    capped_limit = max(1, min(int(limit or 500), 5000))
    since_dt = _parse_utc_datetime(since)
    conn = store_obj.conn
    summaries: list[dict[str, Any]] = []
    unknowns = {
        "provider_rows_considered": 0,
        "codex_turn_rows_considered": 0,
        "rows_without_pattern_decisions": 0,
        "summaries_missing_candidate_id": 0,
        "summaries_missing_rule_id": 0,
        "summaries_missing_pattern_hash": 0,
        "summaries_missing_canary_cohort": 0,
    }

    provider_rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   cost_est_usd, cost_baseline_usd, crunch_json, routing_json,
                   cache_json, category, error
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    provider_rows = [row for row in provider_rows if _created_at_matches_since(row.get("created_at"), since_dt)]
    unknowns["provider_rows_considered"] = len(provider_rows)
    for row in provider_rows:
        routing = _json_obj(row.get("routing_json"))
        cache_meta = _json_obj(row.get("cache_json"))
        crunch_meta = _json_obj(row.get("crunch_json"))
        rows = pattern_decision_summaries(
            provider=str(row.get("provider") or "anthropic"),
            path=str(row.get("path") or ""),
            requested_model=row.get("requested_model"),
            routed_model=row.get("routed_model"),
            status_code=_as_int(row.get("status_code")) if row.get("status_code") is not None else None,
            cost_est_usd=_as_float(row.get("cost_est_usd")) if row.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(row.get("cost_baseline_usd")) if row.get("cost_baseline_usd") is not None else None,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing,
            category=row.get("category") or routing.get("category"),
        )
        if not rows:
            unknowns["rows_without_pattern_decisions"] += 1
        for summary in rows:
            if not isinstance(summary, dict):
                continue
            item = dict(summary)
            item.update({
                "traffic_row_id": row.get("id"),
                "created_at": row.get("created_at"),
                "status_code": row.get("status_code"),
                "latency_ms": row.get("latency_ms"),
                "cost_est_usd": row.get("cost_est_usd"),
                "traffic_kind": "provider_call",
                "error_bucket": _error_bucket(row.get("error"), row.get("status_code")),
                "latency_bucket": _latency_bucket(row.get("latency_ms")),
                "cache_decision_status": cache_meta.get("status") or "missing",
                "crunch_decision_status": "changed" if crunch_meta.get("changed") else "unchanged" if "changed" in crunch_meta else "missing",
            })
            summaries.append(item)

    codex_rows = [
        dict(row)
        for row in conn.execute(
            """
            select s.id as start_event_id,
                   s.created_at,
                   s.request_id,
                   s.thread_id,
                   s.session_id,
                   s.input_text_chars,
                   s.routing_json,
                   s.crunch_json,
                   s.cache_json,
                   (
                       select r.id from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_event_id,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars,
                   (
                       select r.error_code from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_error_code,
                   (
                       select r.latency_ms from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_latency_ms
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    codex_rows = [row for row in codex_rows if _created_at_matches_since(row.get("created_at"), since_dt)]
    unknowns["codex_turn_rows_considered"] = len(codex_rows)
    for row in codex_rows:
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        crunch_meta = _json_obj(row.get("crunch_json"))
        status_code = 500 if row.get("response_error_code") is not None else (200 if row.get("response_event_id") else None)
        rows = pattern_decision_summaries(
            provider="openai",
            path="codex-app://turn/start",
            requested_model=routing.get("requested_model") or routing.get("requested_model_value"),
            routed_model=routing.get("routed_model") or routing.get("target_model") or routing.get("requested_model"),
            status_code=status_code,
            cost_est_usd=_codex_estimated_cost(row.get("input_text_chars"), row.get("response_result_chars")),
            cost_baseline_usd=_codex_estimated_cost(row.get("input_text_chars"), row.get("response_result_chars")),
            cache_meta=cache,
            crunch_meta=crunch_meta,
            routing_meta=routing,
            category=routing.get("category") or "codex_turn",
        )
        if not rows:
            unknowns["rows_without_pattern_decisions"] += 1
        for summary in rows:
            if not isinstance(summary, dict):
                continue
            item = dict(summary)
            item.update({
                "traffic_row_id": row.get("start_event_id"),
                "created_at": row.get("created_at"),
                "source_surface": "codex_turn",
                "app_family": "codex",
                "status_code": status_code,
                "latency_ms": row.get("response_latency_ms"),
                "cost_est_usd": 0.0,
                "traffic_kind": "codex_turn",
                "error_bucket": "codex_error" if row.get("response_error_code") is not None else "none",
                "latency_bucket": _latency_bucket(row.get("response_latency_ms")),
                "cache_decision_status": cache.get("status") or "missing",
                "crunch_decision_status": "changed" if crunch_meta.get("changed") else "unchanged" if "changed" in crunch_meta else "missing",
            })
            summaries.append(item)

    for summary in summaries:
        if not summary.get("candidate_id"):
            unknowns["summaries_missing_candidate_id"] += 1
        if not summary.get("rule_id"):
            unknowns["summaries_missing_rule_id"] += 1
        if not str(summary.get("pattern_hash") or "").startswith("sha256:"):
            unknowns["summaries_missing_pattern_hash"] += 1
        canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
        if not summary.get("cohort") and not canary.get("cohort") and not canary.get("status"):
            unknowns["summaries_missing_canary_cohort"] += 1

    return summaries, unknowns


def _summary_matches_action(summary: dict[str, Any], action: dict[str, Any]) -> bool:
    if str(summary.get("decision_type") or "") != str(action.get("policy_section") or ""):
        return False
    pattern_hash = _normalize_pattern_hash(action.get("pattern_hash"))
    summary_hashes = {
        normalized
        for value in [summary.get("pattern_hash"), *(summary.get("pattern_hashes") or [])]
        if (normalized := _normalize_pattern_hash(value))
    }
    if pattern_hash and pattern_hash not in summary_hashes:
        return False
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    if target_candidate_id and str(summary.get("candidate_id") or "").strip() != target_candidate_id:
        return False
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    if target_rule_id and str(summary.get("rule_id") or "").strip() != target_rule_id:
        return False
    return True


def _projected_counts(*, matched_count: int, applied_count: int, holdout_count: int, edit: dict[str, Any]) -> dict[str, Any]:
    proposed_fraction = _as_float(edit.get("recommended_fraction"), _as_float(edit.get("current_fraction"), 1.0))
    disable = bool(edit.get("disable"))
    if disable:
        projected_applied = 0
        projected_holdout = 0
        projected_disabled = matched_count
    else:
        projected_applied = max(applied_count, round(matched_count * proposed_fraction))
        projected_applied = min(projected_applied, matched_count)
        projected_holdout = max(0, matched_count - projected_applied)
        projected_disabled = 0
    return {
        "current_fraction": _as_float(edit.get("current_fraction"), 0.0),
        "projected_fraction": 0.0 if disable else proposed_fraction,
        "current_canary_applied_count": applied_count,
        "current_canary_holdout_count": holdout_count,
        "projected_canary_applied_count": projected_applied,
        "projected_canary_holdout_count": projected_holdout,
        "projected_local_bypass_or_disable_count": projected_disabled,
        "projected_additional_applied_count": max(0, projected_applied - applied_count),
    }


def dry_run_rollout_actions(
    bundle: Any,
    *,
    store_obj: Any,
    config_dir: str | Path,
    sections: list[str] | tuple[str, ...] | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    review = plan_rollout_actions(bundle, config_dir=config_dir, sections=sections)
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_DRY_RUN_SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "bundle_hash": canonical_rollout_action_bundle_hash(bundle),
        "dry_run": True,
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "config_dir": str(config_path),
        "lookback_limit": max(1, min(int(limit or 500), 5000)),
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "actions": [],
        "summary": {},
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "policy_files_written": False,
            "store_written": False,
            "basis": "stored pattern decision metadata, hashes, canary cohorts, status codes, latency, and size-derived savings only",
        },
        "errors": review.get("errors", []),
        "warnings": review.get("warnings", []),
    }
    if not review.get("validation", {}).get("ok"):
        result["error"] = {"type": "validation_failed", "message": "rollout action bundle failed validation"}
        return result

    summaries, unknowns = _traffic_pattern_summaries(store_obj, limit=result["lookback_limit"])
    action_results: list[dict[str, Any]] = []
    for planned in review.get("actions", []):
        action_index = int(str(planned.get("path") or "$.actions[0]").split("[")[-1].split("]")[0]) if "[" in str(planned.get("path") or "") else 0
        raw_action = (bundle.get("actions") or [])[action_index] if isinstance(bundle, dict) and isinstance(bundle.get("actions"), list) and action_index < len(bundle["actions"]) else {}
        matched = [summary for summary in summaries if _summary_matches_action(summary, raw_action)]
        status_counts: dict[str, int] = {}
        savings_counts: dict[str, int] = {}
        traffic_counts: dict[str, int] = {}
        bypass_reasons: dict[str, int] = {}
        applied_count = 0
        holdout_count = 0
        bypass_count = 0
        saved_chars = 0
        tokens_saved = 0
        cost_savings = 0.0
        for summary in matched:
            cohort = _cohort_for_summary(summary)
            if cohort == "canary_applied":
                applied_count += 1
            elif cohort == "canary_holdout":
                holdout_count += 1
            elif cohort == "bypassed":
                bypass_count += 1
                reason = str(summary.get("reason") or "unknown")
                bypass_reasons[reason] = bypass_reasons.get(reason, 0) + 1
            status_counts[_status_bucket(summary.get("status_code"))] = status_counts.get(_status_bucket(summary.get("status_code")), 0) + 1
            savings_counts[_saving_bucket(summary.get("estimated_cost_savings_usd"))] = savings_counts.get(_saving_bucket(summary.get("estimated_cost_savings_usd")), 0) + 1
            traffic_kind = str(summary.get("traffic_kind") or "unknown")
            traffic_counts[traffic_kind] = traffic_counts.get(traffic_kind, 0) + 1
            saved_chars += _as_int(summary.get("saved_chars"))
            tokens_saved += _as_int(summary.get("tokens_saved_est"))
            cost_savings += _as_float(summary.get("estimated_cost_savings_usd"), 0.0)
        edit = planned.get("proposed_edit") if isinstance(planned.get("proposed_edit"), dict) else {}
        projected = _projected_counts(
            matched_count=len(matched),
            applied_count=applied_count,
            holdout_count=holdout_count,
            edit=edit,
        )
        unknown_action = {
            "matched_summaries_missing_candidate_id": sum(1 for item in matched if not item.get("candidate_id")),
            "matched_summaries_missing_rule_id": sum(1 for item in matched if not item.get("rule_id")),
            "matched_summaries_missing_pattern_hash": sum(1 for item in matched if not str(item.get("pattern_hash") or "").startswith("sha256:")),
            "matched_summaries_missing_canary_cohort": sum(
                1
                for item in matched
                if not item.get("cohort")
                and not ((item.get("canary") if isinstance(item.get("canary"), dict) else {}) or {}).get("cohort")
                and not ((item.get("canary") if isinstance(item.get("canary"), dict) else {}) or {}).get("status")
            ),
        }
        action_results.append({
            "path": planned.get("path"),
            "action_id": rollout_action_id(raw_action) if isinstance(raw_action, dict) else rollout_action_id(planned),
            "status": planned.get("status"),
            "reason": planned.get("reason"),
            "policy_section": raw_action.get("policy_section") or planned.get("policy_section"),
            "action_type": raw_action.get("action_type") or planned.get("action_type"),
            "target_candidate_id": raw_action.get("target_candidate_id") or planned.get("target_candidate_id"),
            "target_rule_id": raw_action.get("target_rule_id") or planned.get("target_rule_id"),
            "rule_id": planned.get("rule_id"),
            "pattern_hash": _normalize_pattern_hash(raw_action.get("pattern_hash")) or planned.get("pattern_hash"),
            "affected_metadata_row_count": len(matched),
            "affected_provider_call_count": traffic_counts.get("provider_call", 0),
            "affected_codex_turn_count": traffic_counts.get("codex_turn", 0),
            "current_bypassed_or_disabled_count": bypass_count,
            **projected,
            "historical_tokens_saved_est": tokens_saved,
            "historical_saved_chars": saved_chars,
            "historical_estimated_cost_savings_usd": round(cost_savings, 8),
            "savings_buckets": _count_bucket(savings_counts),
            "status_risk_buckets": _count_bucket(status_counts),
            "local_bypass_reasons": _count_bucket(bypass_reasons),
            "unknowns": unknown_action,
            "proposed_edit": planned.get("proposed_edit"),
        })

    total_affected = sum(_as_int(action.get("affected_metadata_row_count")) for action in action_results)
    result.update({
        "ok": bool(review.get("ok")),
        "actions": action_results,
        "summary": {
            "sampled_provider_calls": unknowns["provider_rows_considered"],
            "sampled_codex_turns": unknowns["codex_turn_rows_considered"],
            "sampled_metadata_rows": unknowns["provider_rows_considered"] + unknowns["codex_turn_rows_considered"],
            "pattern_decision_summary_count": len(summaries),
            "affected_metadata_row_count": total_affected,
            "affected_provider_call_count": sum(_as_int(action.get("affected_provider_call_count")) for action in action_results),
            "affected_codex_turn_count": sum(_as_int(action.get("affected_codex_turn_count")) for action in action_results),
            "projected_additional_applied_count": sum(_as_int(action.get("projected_additional_applied_count")) for action in action_results),
            "projected_local_bypass_or_disable_count": sum(_as_int(action.get("projected_local_bypass_or_disable_count")) for action in action_results),
            "historical_tokens_saved_est": sum(_as_int(action.get("historical_tokens_saved_est")) for action in action_results),
            "historical_estimated_cost_savings_usd": round(sum(_as_float(action.get("historical_estimated_cost_savings_usd")) for action in action_results), 8),
            "unknowns": unknowns,
        },
    })
    if not review.get("ok"):
        result["error"] = {"type": "review_failed", "message": "rollout actions are invalid or target unknown local rules"}
    return result


def _dry_run_projection(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "affected_metadata_row_count": _as_int(action.get("affected_metadata_row_count")),
        "affected_provider_call_count": _as_int(action.get("affected_provider_call_count")),
        "affected_codex_turn_count": _as_int(action.get("affected_codex_turn_count")),
        "projected_canary_applied_count": _as_int(action.get("projected_canary_applied_count")),
        "projected_canary_holdout_count": _as_int(action.get("projected_canary_holdout_count")),
        "projected_local_bypass_or_disable_count": _as_int(action.get("projected_local_bypass_or_disable_count")),
        "projected_additional_applied_count": _as_int(action.get("projected_additional_applied_count")),
        "historical_tokens_saved_est": _as_int(action.get("historical_tokens_saved_est")),
        "historical_saved_chars": _as_int(action.get("historical_saved_chars")),
        "historical_estimated_cost_savings_usd": round(_as_float(action.get("historical_estimated_cost_savings_usd")), 8),
    }


def _increment_count(counts: dict[str, int], value: Any) -> None:
    key = str(value or "unknown")
    counts[key] = counts.get(key, 0) + 1


def _rollout_actual_impact(matched: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    latency_counts: dict[str, int] = {}
    savings_counts: dict[str, int] = {}
    traffic_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    crunch_status_counts: dict[str, int] = {}
    decision_status_counts: dict[str, int] = {}
    bypass_reasons: dict[str, int] = {}
    applied_count = 0
    holdout_count = 0
    bypass_count = 0
    saved_chars = 0
    tokens_saved = 0
    cost_savings = 0.0
    for summary in matched:
        cohort = _cohort_for_summary(summary)
        if cohort == "canary_applied":
            applied_count += 1
        elif cohort == "canary_holdout":
            holdout_count += 1
        elif cohort == "bypassed":
            bypass_count += 1
            _increment_count(bypass_reasons, summary.get("reason") or "unknown")
        _increment_count(status_counts, _status_bucket(summary.get("status_code")))
        _increment_count(error_counts, summary.get("error_bucket") or "none")
        _increment_count(latency_counts, summary.get("latency_bucket") or _latency_bucket(summary.get("latency_ms")))
        _increment_count(savings_counts, _saving_bucket(summary.get("estimated_cost_savings_usd")))
        _increment_count(traffic_counts, summary.get("traffic_kind") or "unknown")
        _increment_count(cache_status_counts, summary.get("cache_decision_status") or "missing")
        _increment_count(crunch_status_counts, summary.get("crunch_decision_status") or "missing")
        _increment_count(decision_status_counts, summary.get("status") or "unknown")
        saved_chars += _as_int(summary.get("saved_chars"))
        tokens_saved += _as_int(summary.get("tokens_saved_est"))
        cost_savings += _as_float(summary.get("estimated_cost_savings_usd"), 0.0)
    return {
        "matched_metadata_row_count": len(matched),
        "matched_provider_call_count": traffic_counts.get("provider_call", 0),
        "matched_codex_turn_count": traffic_counts.get("codex_turn", 0),
        "actual_canary_applied_count": applied_count,
        "actual_canary_holdout_count": holdout_count,
        "actual_bypassed_or_disabled_count": bypass_count,
        "actual_tokens_saved_est": tokens_saved,
        "actual_saved_chars": saved_chars,
        "actual_estimated_cost_savings_usd": round(cost_savings, 8),
        "status_risk_buckets": _count_bucket(status_counts),
        "error_buckets": _count_bucket(error_counts),
        "latency_buckets": _count_bucket(latency_counts),
        "savings_buckets": _count_bucket(savings_counts),
        "cache_decision_status_buckets": _count_bucket(cache_status_counts),
        "crunch_decision_status_buckets": _count_bucket(crunch_status_counts),
        "rollout_decision_status_buckets": _count_bucket(decision_status_counts),
        "safety_stop_outcomes": _count_bucket(bypass_reasons),
    }


def _impact_delta(projection: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "matched_vs_projected_affected_delta": _as_int(actual.get("matched_metadata_row_count")) - _as_int(projection.get("affected_metadata_row_count")),
        "applied_vs_projected_delta": _as_int(actual.get("actual_canary_applied_count")) - _as_int(projection.get("projected_canary_applied_count")),
        "holdout_vs_projected_delta": _as_int(actual.get("actual_canary_holdout_count")) - _as_int(projection.get("projected_canary_holdout_count")),
        "bypass_or_disable_vs_projected_delta": _as_int(actual.get("actual_bypassed_or_disabled_count")) - _as_int(projection.get("projected_local_bypass_or_disable_count")),
        "tokens_saved_vs_historical_projection_delta": _as_int(actual.get("actual_tokens_saved_est")) - _as_int(projection.get("historical_tokens_saved_est")),
        "saved_chars_vs_historical_projection_delta": _as_int(actual.get("actual_saved_chars")) - _as_int(projection.get("historical_saved_chars")),
        "estimated_cost_savings_vs_historical_projection_delta_usd": round(
            _as_float(actual.get("actual_estimated_cost_savings_usd")) - _as_float(projection.get("historical_estimated_cost_savings_usd")),
            8,
        ),
    }


def measure_rollout_action_impact(
    dry_run_report: Any,
    *,
    store_obj: Any,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 5000))
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_IMPACT_SCHEMA,
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
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_params_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
            "basis": "dry-run aggregate projections plus post-apply pattern decision metadata, status/error buckets, latency buckets, and size-derived savings only",
        },
        "warnings": [],
    }
    if not isinstance(dry_run_report, dict) or dry_run_report.get("schema") != PATTERN_ROLLOUT_ACTION_DRY_RUN_SCHEMA:
        result["error"] = {"type": "invalid_dry_run_report", "message": f"expected {PATTERN_ROLLOUT_ACTION_DRY_RUN_SCHEMA}"}
        return result
    if not dry_run_report.get("ok"):
        result["error"] = {"type": "dry_run_not_ok", "message": "dry-run report was not successful"}
        return result
    since_value = since or dry_run_report.get("generated_at")
    if since_value and _parse_utc_datetime(since_value) is None:
        result["error"] = {"type": "invalid_since", "message": "post-apply since timestamp must be ISO-8601"}
        return result
    result["post_apply_since"] = since_value
    result["dry_run"] = {
        "generated_at": dry_run_report.get("generated_at"),
        "bundle_hash": dry_run_report.get("bundle_hash") or ((dry_run_report.get("provenance") or {}).get("computed_bundle_hash") if isinstance(dry_run_report.get("provenance"), dict) else None),
        "summary": {
            "affected_metadata_row_count": _as_int((dry_run_report.get("summary") or {}).get("affected_metadata_row_count")) if isinstance(dry_run_report.get("summary"), dict) else 0,
            "projected_additional_applied_count": _as_int((dry_run_report.get("summary") or {}).get("projected_additional_applied_count")) if isinstance(dry_run_report.get("summary"), dict) else 0,
            "projected_local_bypass_or_disable_count": _as_int((dry_run_report.get("summary") or {}).get("projected_local_bypass_or_disable_count")) if isinstance(dry_run_report.get("summary"), dict) else 0,
            "historical_tokens_saved_est": _as_int((dry_run_report.get("summary") or {}).get("historical_tokens_saved_est")) if isinstance(dry_run_report.get("summary"), dict) else 0,
            "historical_estimated_cost_savings_usd": round(_as_float((dry_run_report.get("summary") or {}).get("historical_estimated_cost_savings_usd")) if isinstance(dry_run_report.get("summary"), dict) else 0.0, 8),
        },
    }

    summaries, unknowns = _traffic_pattern_summaries(store_obj, limit=lookback_limit, since=since_value)
    action_results: list[dict[str, Any]] = []
    for dry_action in dry_run_report.get("actions") or []:
        if not isinstance(dry_action, dict):
            continue
        matched = [summary for summary in summaries if _summary_matches_action(summary, dry_action)]
        projection = _dry_run_projection(dry_action)
        actual = _rollout_actual_impact(matched)
        unknown_action = {
            "matched_summaries_missing_candidate_id": sum(1 for item in matched if not item.get("candidate_id")),
            "matched_summaries_missing_rule_id": sum(1 for item in matched if not item.get("rule_id")),
            "matched_summaries_missing_pattern_hash": sum(1 for item in matched if not str(item.get("pattern_hash") or "").startswith("sha256:")),
            "matched_summaries_missing_canary_cohort": sum(
                1
                for item in matched
                if not item.get("cohort")
                and not ((item.get("canary") if isinstance(item.get("canary"), dict) else {}) or {}).get("cohort")
                and not ((item.get("canary") if isinstance(item.get("canary"), dict) else {}) or {}).get("status")
            ),
        }
        action_results.append({
            "path": dry_action.get("path"),
            "status": "matched" if matched else "no-post-apply-matches",
            "action_id": dry_action.get("action_id") or rollout_action_id(dry_action),
            "policy_section": dry_action.get("policy_section"),
            "action_type": dry_action.get("action_type"),
            "target_candidate_id": dry_action.get("target_candidate_id"),
            "target_rule_id": dry_action.get("target_rule_id"),
            "pattern_hash": _normalize_pattern_hash(dry_action.get("pattern_hash")),
            "projection": projection,
            "actual": actual,
            "delta": _impact_delta(projection, actual),
            "unknowns": unknown_action,
        })

    total_actual = sum(_as_int(action.get("actual", {}).get("matched_metadata_row_count")) for action in action_results)
    total_projected = sum(_as_int(action.get("projection", {}).get("affected_metadata_row_count")) for action in action_results)
    result.update({
        "ok": True,
        "status": "matched" if total_actual else "no-post-apply-matches",
        "actions": action_results,
        "summary": {
            "sampled_provider_calls": unknowns["provider_rows_considered"],
            "sampled_codex_turns": unknowns["codex_turn_rows_considered"],
            "pattern_decision_summary_count": len(summaries),
            "action_count": len(action_results),
            "projected_affected_metadata_row_count": total_projected,
            "actual_matched_metadata_row_count": total_actual,
            "actual_matched_provider_call_count": sum(_as_int(action.get("actual", {}).get("matched_provider_call_count")) for action in action_results),
            "actual_matched_codex_turn_count": sum(_as_int(action.get("actual", {}).get("matched_codex_turn_count")) for action in action_results),
            "actual_canary_applied_count": sum(_as_int(action.get("actual", {}).get("actual_canary_applied_count")) for action in action_results),
            "actual_canary_holdout_count": sum(_as_int(action.get("actual", {}).get("actual_canary_holdout_count")) for action in action_results),
            "actual_bypassed_or_disabled_count": sum(_as_int(action.get("actual", {}).get("actual_bypassed_or_disabled_count")) for action in action_results),
            "actual_tokens_saved_est": sum(_as_int(action.get("actual", {}).get("actual_tokens_saved_est")) for action in action_results),
            "actual_saved_chars": sum(_as_int(action.get("actual", {}).get("actual_saved_chars")) for action in action_results),
            "actual_estimated_cost_savings_usd": round(sum(_as_float(action.get("actual", {}).get("actual_estimated_cost_savings_usd")) for action in action_results), 8),
            "actions_without_post_apply_matches": sum(1 for action in action_results if action.get("status") == "no-post-apply-matches"),
            "unknowns": unknowns,
        },
    })
    return result


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
            family_validation = _validate_family_specific_action(
                action=action,
                rule=rule,
                section=section,
                path=f"$.actions[{index}]",
            )
            if family_validation.get("errors"):
                for error in family_validation.get("errors", []):
                    if isinstance(error, dict):
                        _add_error(errors, str(error.get("path") or f"$.actions[{index}]"), str(error.get("message") or "family-specific rollout action is invalid"))
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "family-specific-validation-failed",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                    "rule_id": rule.get("id") or rule.get("rule_id"),
                    "pattern_hash": action.get("pattern_hash"),
                    "family_validation": family_validation,
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
                "family_validation": family_validation,
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
                "current_fraction": edit.get("current_fraction"),
                "recommended_fraction": edit.get("recommended_fraction"),
                "family_validation": {
                    key: value
                    for key, value in (planned.get("family_validation") or {}).items()
                    if key in {"schema", "status", "family", "policy_section", "policy_profile", "action_type", "replayability_levels"}
                    and value not in (None, "", [], {})
                },
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
