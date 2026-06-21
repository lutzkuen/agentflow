from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tokenclaw import __version__
from tokenclaw.policy_bundle import (
    MANAGED_POLICY_VERIFICATION_SECRET_ENV,
    POLICY_BUNDLE_PROVENANCE_SCHEMA,
    _hmac_signature,
    _normalize_signature,
    _secret_for_key_id,
)
from tokenclaw.store import utc_now


OPENAI_CACHE_REPLAY_ROLLOUT_ACTIONS_SCHEMA = "tokenclaw.openai_cache_replay_rollout_actions.v1"
OPTIMIZATION_ROLLOUT_ACTION_SCHEMA = "tokenclaw.optimization_rollout_action.v1"
OPENAI_CACHE_REPLAY_REVIEW_ACTION_SCHEMA = "tokenclaw.openai_cache_replay_rollout_review_action.v1"
OPENAI_CACHE_REPLAY_CANARY_POLICY_SCHEMA = "tokenclaw.openai_cache_replay_canary_policy.v1"
OPENAI_CACHE_REPLAY_ROLLOUT_REVIEW_SCHEMA = "tokenclaw.openai_cache_replay_rollout_actions_review.v1"
OPENAI_CACHE_REPLAY_ROLLOUT_APPLY_SCHEMA = "tokenclaw.openai_cache_replay_rollout_actions_apply.v1"
OPENAI_CACHE_REPLAY_ROLLOUT_DRY_RUN_SCHEMA = "tokenclaw.openai_cache_replay_rollout_actions_dry_run.v1"
OPENAI_CACHE_REPLAY_ROLLOUT_VALIDATION_SCHEMA = "tokenclaw.openai_cache_replay_rollout_actions_validation.v1"
OPENAI_CACHE_REPLAY_ROLLOUT_PROVENANCE_SCHEMA = "tokenclaw.openai_cache_replay_rollout_actions_provenance_verification.v1"

ACTION_TYPES = {"widen", "hold", "rollback", "retire", "disable"}
DISABLE_ACTION_TYPES = {"rollback", "retire", "disable"}
RAW_LIKE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cache_key",
    "command",
    "content",
    "credential",
    "file_content",
    "file_path",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_payload",
    "raw_request",
    "raw_response",
    "request_id",
    "secret",
    "session_id",
    "system",
    "tool_payload",
    "transcript",
)
ALLOWED_RAW_LIKE_KEYS = {
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_payloads_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "raw_body_storage",
    "request_ids_returned",
    "tenant_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
    "session_id_hash",
    "workflow_id_hash",
    "traffic_fingerprint",
    "request_fingerprint",
    "provider_endpoint",
}
UNSAFE_PRIVACY_KEYS = {
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_payloads_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "raw_body_storage",
    "request_ids_returned",
    "tenant_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _payload_for_hash(bundle: dict[str, Any]) -> dict[str, Any]:
    payload = _json_roundtrip(bundle)
    if isinstance(payload, dict):
        payload.pop("provenance", None)
    return payload if isinstance(payload, dict) else {}


def canonical_openai_cache_replay_rollout_bundle_hash(bundle: Any) -> str | None:
    if not isinstance(bundle, dict):
        return None
    payload = _payload_for_hash(bundle)
    return f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"


def attach_openai_cache_replay_rollout_provenance(
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
        "bundle_hash": canonical_openai_cache_replay_rollout_bundle_hash(signed),
    }
    provenance["signature"] = _hmac_signature(provenance, secret)
    signed["provenance"] = provenance
    return signed


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_iso_datetime(value: Any) -> bool:
    return _parse_datetime(value) is not None


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
            if lowered not in ALLOWED_RAW_LIKE_KEYS and any(part in lowered for part in RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    _add_error(errors, child_path, "raw or local-identifier cache replay rollout payloads are not accepted")
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


def _privacy_flags_safe(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else f"$.{key}"
            lowered = str(key).lower()
            if lowered in UNSAFE_PRIVACY_KEYS and bool(item):
                _add_error(errors, child_path, "privacy summary reports raw payloads or local identifiers")
            _privacy_flags_safe(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _privacy_flags_safe(item, f"{path}[{index}]", errors)


def verify_openai_cache_replay_rollout_provenance(bundle: Any) -> dict[str, Any]:
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    managed_bundle = isinstance(bundle, dict) and bundle.get("schema") == OPENAI_CACHE_REPLAY_ROLLOUT_ACTIONS_SCHEMA
    secret, configured = _secret_for_key_id(provenance.get("key_id") if isinstance(provenance, dict) else None)
    result: dict[str, Any] = {
        "schema": OPENAI_CACHE_REPLAY_ROLLOUT_PROVENANCE_SCHEMA,
        "status": "missing",
        "ok": True,
        "managed_bundle": managed_bundle,
        "verification_configured": configured,
        "signature_required": bool(managed_bundle),
        "algorithm": None,
        "issuer": None,
        "server_id": None,
        "key_id": None,
        "generated_at": None,
        "bundle_hash": None,
        "computed_bundle_hash": canonical_openai_cache_replay_rollout_bundle_hash(bundle),
        "signature_present": False,
        "errors": [],
        "warnings": [],
    }
    if not managed_bundle:
        result["status"] = "not-managed-cache-replay"
        return result
    if not isinstance(provenance, dict):
        result["ok"] = False
        result["errors"].append({
            "path": "$.provenance",
            "message": "managed OpenAI cache replay rollout bundle is missing provenance required for local review",
        })
        return result
    for key in ("algorithm", "issuer", "server_id", "key_id", "generated_at", "bundle_hash"):
        result[key] = provenance.get(key)
    result["signature_present"] = bool(provenance.get("signature"))
    if not configured:
        result["status"] = "not-configured"
        result["warnings"].append({
            "path": "$.provenance",
            "message": f"managed OpenAI cache replay rollout provenance was not verified because {MANAGED_POLICY_VERIFICATION_SECRET_ENV} is not configured",
        })
        return result

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
    if provenance.get("bundle_hash") != canonical_openai_cache_replay_rollout_bundle_hash(bundle):
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


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.replace("-", ".").split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def _minimum_version_compatible(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    current = _version_tuple(__version__)
    required = _version_tuple(value)
    if not current or not required:
        return True
    return current >= required


def _nested_dict(source: Any, *path: str) -> dict[str, Any]:
    value = source
    for key in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _safe_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _validate_compatibility(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(value, dict):
        return
    if value.get("compatible") is False:
        _add_error(errors, path, "managed bundle reports local executor incompatibility")
    supported = set(_safe_string_list(value.get("supported_local_action_families")))
    if supported and "cache" not in supported:
        _add_error(errors, f"{path}.supported_local_action_families", "local executor compatibility does not include cache")
    if not _minimum_version_compatible(value.get("minimum_local_client_version")):
        _add_error(errors, f"{path}.minimum_local_client_version", "local AgentFlow version is below the managed rollout minimum")


def _validate_action(action: Any, path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(action, dict):
        _add_error(errors, path, "expected rollout action object")
        return
    if action.get("schema") != OPTIMIZATION_ROLLOUT_ACTION_SCHEMA:
        _add_error(errors, f"{path}.schema", f"expected {OPTIMIZATION_ROLLOUT_ACTION_SCHEMA}")
    if str(action.get("action_type") or "") not in ACTION_TYPES:
        _add_error(errors, f"{path}.action_type", "expected widen, hold, rollback, retire, or disable")
    if action.get("policy_section") != "cache":
        _add_error(errors, f"{path}.policy_section", "expected cache")
    if action.get("action_family") != "cache":
        _add_error(errors, f"{path}.action_family", "expected cache")
    if action.get("candidate_family") != "cache-policy-rule":
        _add_error(errors, f"{path}.candidate_family", "expected cache-policy-rule")
    if not isinstance(action.get("target_candidate_id"), str) or not action.get("target_candidate_id").strip():
        _add_error(errors, f"{path}.target_candidate_id", "expected non-empty string")
    if action.get("required_local_review") is not True:
        _add_error(errors, f"{path}.required_local_review", "expected true")
    if action.get("managed_enforced") is not False:
        _add_error(errors, f"{path}.managed_enforced", "expected false")
    _validate_compatibility(action.get("local_executor_compatibility"), f"{path}.local_executor_compatibility", errors)
    privacy = action.get("privacy_summary") if isinstance(action.get("privacy_summary"), dict) else {}
    if privacy.get("metadata_only") is not True or privacy.get("feature_only") is not True:
        _add_error(errors, f"{path}.privacy_summary", "expected feature-only metadata privacy summary")
    if privacy.get("provider_forwarding") is not False:
        _add_error(errors, f"{path}.privacy_summary.provider_forwarding", "expected false")
    _privacy_flags_safe(privacy, f"{path}.privacy_summary", errors)

    review_action = action.get("action") if isinstance(action.get("action"), dict) else {}
    if review_action.get("schema") != OPENAI_CACHE_REPLAY_REVIEW_ACTION_SCHEMA:
        _add_error(errors, f"{path}.action.schema", f"expected {OPENAI_CACHE_REPLAY_REVIEW_ACTION_SCHEMA}")
    if review_action.get("requires_local_review") is not True:
        _add_error(errors, f"{path}.action.requires_local_review", "expected true")
    if review_action.get("managed_enforced") is not False:
        _add_error(errors, f"{path}.action.managed_enforced", "expected false")
    if review_action.get("provider_forwarding") is not False:
        _add_error(errors, f"{path}.action.provider_forwarding", "expected false")
    proposed = review_action.get("proposed_edit") if isinstance(review_action.get("proposed_edit"), dict) else {}
    if proposed.get("policy_section") != "cache":
        _add_error(errors, f"{path}.action.proposed_edit.policy_section", "expected cache")
    if proposed.get("policy_source") != "managed-recommended":
        _add_error(errors, f"{path}.action.proposed_edit.policy_source", "expected managed-recommended")
    proposed_action = proposed.get("action") if isinstance(proposed.get("action"), dict) else {}
    if proposed_action.get("type") != "exact_cache_replay":
        _add_error(errors, f"{path}.action.proposed_edit.action.type", "expected exact_cache_replay")
    if proposed_action.get("review_only") is not True:
        _add_error(errors, f"{path}.action.proposed_edit.action.review_only", "expected true")
    if proposed_action.get("enabled") is False and str(action.get("action_type") or "") not in DISABLE_ACTION_TYPES:
        _add_error(errors, f"{path}.action.proposed_edit.action.enabled", "enabled=false is only accepted for rollback/retire/disable")
    requirements = proposed_action.get("dependency_requirements") if isinstance(proposed_action.get("dependency_requirements"), dict) else {}
    if requirements.get("safe_invalidation_evidence") is not True:
        _add_error(errors, f"{path}.action.proposed_edit.action.dependency_requirements.safe_invalidation_evidence", "expected true")


def validate_openai_cache_replay_rollout_bundle(bundle: Any, *, now: datetime | None = None) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_openai_cache_replay_rollout_provenance(bundle)
    now_dt = now or datetime.now(timezone.utc)
    if not isinstance(bundle, dict):
        _add_error(errors, "$", "rollout action bundle must be a JSON object")
        return {
            "schema": OPENAI_CACHE_REPLAY_ROLLOUT_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
        }
    if bundle.get("schema") != OPENAI_CACHE_REPLAY_ROLLOUT_ACTIONS_SCHEMA:
        _add_error(errors, "$.schema", f"expected {OPENAI_CACHE_REPLAY_ROLLOUT_ACTIONS_SCHEMA}")
    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")
    expires = _parse_datetime(bundle.get("expires_at"))
    if expires is not None and expires <= now_dt:
        _add_error(errors, "$.expires_at", "OpenAI cache replay rollout bundle is expired")
    _validate_compatibility(bundle.get("local_executor_compatibility"), "$.local_executor_compatibility", errors)
    _privacy_flags_safe(bundle.get("privacy_summary"), "$.privacy_summary", errors)
    _privacy_flags_safe(bundle.get("summary"), "$.summary", errors)
    actions = bundle.get("actions")
    if not isinstance(actions, list):
        _add_error(errors, "$.actions", "expected list")
        action_count = 0
    else:
        action_count = len(actions)
        for index, action in enumerate(actions):
            _validate_action(action, f"$.actions[{index}]", errors)
    _scan_raw_like(bundle, "$", errors)
    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            _add_warning(warnings, str(warning.get("path") or "$.provenance"), str(warning.get("message") or "provenance was not verified"))
    return {
        "schema": OPENAI_CACHE_REPLAY_ROLLOUT_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle.get("schema"),
        "action_count": action_count,
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _load_policy_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {"schema": OPENAI_CACHE_REPLAY_CANARY_POLICY_SCHEMA, "policy_source": "managed-recommended", "pattern_rules": []}, None
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    if not isinstance(parsed.get("pattern_rules"), list):
        parsed["pattern_rules"] = []
    return parsed, text


def _target_rule_id(action: dict[str, Any]) -> str | None:
    proposed = _nested_dict(action, "action", "proposed_edit")
    return _first_string(action.get("target_rule_id"), proposed.get("rule_id"))


def _find_rule(rules: list[Any], action: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    target_rule_id = _target_rule_id(action)
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        candidate_id = str(rule.get("candidate_id") or rule.get("recommendation_id") or rule.get("policy_id") or "").strip()
        rule_match = bool(target_rule_id and rule_id == target_rule_id)
        candidate_match = bool(target_candidate_id and candidate_id == target_candidate_id)
        if rule_match or candidate_match:
            return index, rule
    return None, None


def _rule_rollout(rule: dict[str, Any]) -> dict[str, Any]:
    rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
    return {
        "schema": str(rollout.get("schema") or "tokenclaw.pattern_policy_rollout.v1"),
        "recommendation_mode": str(rollout.get("recommendation_mode") or "canary"),
        "canary_enabled": bool(rollout.get("canary_enabled", True)),
        "canary_fraction": _as_float(rollout.get("canary_fraction"), 1.0),
        "holdout_fraction": _as_float(rollout.get("holdout_fraction"), 0.0),
        "canary_salt": str(rollout.get("canary_salt") or ""),
        "canary_unit": str(rollout.get("canary_unit") or "request_fingerprint"),
        **{
            key: value
            for key, value in rollout.items()
            if key not in {"schema", "recommendation_mode", "canary_enabled", "canary_fraction", "holdout_fraction", "canary_salt", "canary_unit"}
        },
    }


def _plan_rule_edit(rule: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "")
    proposed_action = _nested_dict(action, "action", "proposed_edit", "action")
    current_rollout = _rule_rollout(rule)
    current_fraction = _as_float(current_rollout.get("canary_fraction"), 1.0)
    current_holdout = _as_float(current_rollout.get("holdout_fraction"), 0.0)
    disable = action_type in DISABLE_ACTION_TYPES
    if action_type == "hold":
        recommended_fraction = current_fraction
        holdout_fraction = current_holdout
    else:
        recommended_fraction = _as_float(proposed_action.get("canary_fraction"), current_fraction)
        holdout_fraction = _as_float(proposed_action.get("holdout_fraction"), current_holdout)
    proposed_rollout = dict(current_rollout)
    proposed_rollout["canary_fraction"] = 0.0 if disable else recommended_fraction
    proposed_rollout["holdout_fraction"] = 0.0 if disable else holdout_fraction
    proposed_rollout["canary_enabled"] = False if disable else bool(proposed_rollout.get("canary_enabled", True))
    proposed_rollout["recommendation_mode"] = "disabled-by-openai-cache-replay-rollout-action" if disable else str(
        proposed_rollout.get("recommendation_mode") or "canary"
    )
    proposed_enabled = False if disable else bool(rule.get("enabled", True))
    changed = (
        bool(rule.get("enabled", True)) != proposed_enabled
        or _rule_rollout(rule) != _rule_rollout({"rollout": proposed_rollout})
    )
    return {
        "action_type": action_type,
        "disable": disable,
        "current_fraction": current_fraction,
        "recommended_fraction": recommended_fraction,
        "current_holdout_fraction": current_holdout,
        "recommended_holdout_fraction": holdout_fraction,
        "proposed_enabled": proposed_enabled,
        "proposed_rollout": proposed_rollout,
        "changed": changed,
    }


def review_openai_cache_replay_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    validation = validate_openai_cache_replay_rollout_bundle(bundle, now=now)
    config_path = Path(config_dir).expanduser()
    policy_path = config_path / "cache_canary_policy.yaml"
    actions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    data, _old_text = _load_policy_yaml(policy_path)
    rules = data.get("pattern_rules") if isinstance(data.get("pattern_rules"), list) else []
    if validation.get("ok"):
        for index, action in enumerate(bundle.get("actions") or []):
            if not isinstance(action, dict):
                continue
            rule_index, rule = _find_rule(rules, action)
            if rule is None or rule_index is None:
                _add_error(errors, f"$.actions[{index}]", "OpenAI cache replay action targets an unknown local cache canary rule")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unknown-rule",
                    "policy_section": "cache",
                    "action_type": action.get("action_type"),
                    "action_id": action.get("action_id"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": _target_rule_id(action),
                })
                continue
            if str(rule.get("policy_source") or "") != "managed-recommended":
                _add_error(errors, f"$.actions[{index}]", "OpenAI cache replay action targets a non-managed local cache rule")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unsafe-policy-source",
                    "policy_section": "cache",
                    "action_type": action.get("action_type"),
                    "action_id": action.get("action_id"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": _target_rule_id(action),
                    "rule_id": rule.get("id") or rule.get("rule_id"),
                    "policy_source": rule.get("policy_source"),
                })
                continue
            action_block = rule.get("action") if isinstance(rule.get("action"), dict) else {}
            if action_block.get("type") not in {"exact_cache", "exact_cache_pattern", "exact_cache_replay"}:
                _add_error(errors, f"$.actions[{index}]", "OpenAI cache replay action targets a non-exact local cache rule")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unsupported-local-cache-action",
                    "policy_section": "cache",
                    "action_type": action.get("action_type"),
                    "action_id": action.get("action_id"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": _target_rule_id(action),
                    "rule_id": rule.get("id") or rule.get("rule_id"),
                })
                continue
            if action_block.get("allow_tool_calls") and not (
                action_block.get("safe_invalidation") or action_block.get("safe_invalidation_evidence")
            ):
                _add_error(errors, f"$.actions[{index}]", "tool-call cache replay rules require safe invalidation evidence")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "missing-safe-invalidation-evidence",
                    "policy_section": "cache",
                    "action_type": action.get("action_type"),
                    "action_id": action.get("action_id"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": _target_rule_id(action),
                    "rule_id": rule.get("id") or rule.get("rule_id"),
                })
                continue
            edit = _plan_rule_edit(rule, action)
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": "cache",
                "action_type": edit["action_type"],
                "action_id": action.get("action_id"),
                "action_family": action.get("action_family"),
                "source_surface": action.get("source_surface"),
                "app_family": action.get("app_family"),
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": _target_rule_id(action),
                "rule_id": rule.get("id") or rule.get("rule_id"),
                "rule_index": rule_index,
                "confidence": action.get("confidence"),
                "current_rule": {
                    "enabled": bool(rule.get("enabled", True)),
                    "policy_source": rule.get("policy_source"),
                    "rollout": _rule_rollout(rule),
                },
                "proposed_edit": {
                    "changed": edit["changed"],
                    "disable": edit["disable"],
                    "current_fraction": edit["current_fraction"],
                    "recommended_fraction": edit["recommended_fraction"],
                    "current_holdout_fraction": edit["current_holdout_fraction"],
                    "recommended_holdout_fraction": edit["recommended_holdout_fraction"],
                    "enabled": edit["proposed_enabled"],
                    "rollout": edit["proposed_rollout"],
                },
            })
    ok = bool(validation.get("ok") and not errors)
    return {
        "schema": OPENAI_CACHE_REPLAY_ROLLOUT_REVIEW_SCHEMA,
        "ok": ok,
        "config_dir": str(config_path),
        "policy_file": "cache_canary_policy.yaml",
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
        "read_only": True,
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": {
            "metadata_only": True,
            "feature_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "filesystem_paths_included": False,
            "policy_file_contents_included": False,
        },
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _backup_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_policy_file(path: Path, text: str, *, backup_id: str | None = None) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: str | None = None
    if path.exists():
        suffix = backup_id if backup_id else _backup_suffix()
        backup = path.with_name(f"{path.name}.bak-{suffix}")
        backup.write_bytes(path.read_bytes())
        backup_path = str(backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return backup_path


def apply_openai_cache_replay_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    backup_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    review = review_openai_cache_replay_rollout_actions(bundle, config_dir=config_dir, now=now)
    config_path = Path(config_dir).expanduser()
    policy_path = config_path / "cache_canary_policy.yaml"
    result: dict[str, Any] = {
        "schema": OPENAI_CACHE_REPLAY_ROLLOUT_APPLY_SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "dry_run": bool(dry_run),
        "read_only": bool(dry_run),
        "config_dir": str(config_path),
        "policy_file": "cache_canary_policy.yaml",
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "applied_sections": [],
        "files": [],
        "actions": review.get("actions", []),
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "error": None,
        "privacy": review.get("privacy"),
    }
    if not review.get("ok"):
        result["error"] = {"type": "validation_failed", "message": "OpenAI cache replay rollout actions are invalid or target unknown local rules"}
        return result

    data, old_text = _load_policy_yaml(policy_path)
    data["schema"] = str(data.get("schema") or OPENAI_CACHE_REPLAY_CANARY_POLICY_SCHEMA)
    data["policy_source"] = str(data.get("policy_source") or "managed-recommended")
    rules = data.get("pattern_rules") if isinstance(data.get("pattern_rules"), list) else []
    data["pattern_rules"] = rules
    for planned in review.get("actions", []):
        if planned.get("status") != "planned":
            continue
        rule_index = int(planned["rule_index"])
        if rule_index < 0 or rule_index >= len(rules) or not isinstance(rules[rule_index], dict):
            result["error"] = {"type": "plan_mismatch", "message": "local cache canary policy changed after rollout action review"}
            return result
        edit = planned.get("proposed_edit") if isinstance(planned.get("proposed_edit"), dict) else {}
        rules[rule_index]["enabled"] = bool(edit.get("enabled"))
        rules[rule_index]["rollout"] = edit.get("rollout")
        rules[rule_index]["rollout_action"] = {
            "schema": OPENAI_CACHE_REPLAY_REVIEW_ACTION_SCHEMA,
            "action_type": planned.get("action_type"),
            "action_id": planned.get("action_id"),
            "target_candidate_id": planned.get("target_candidate_id"),
            "target_rule_id": planned.get("target_rule_id"),
            "confidence": planned.get("confidence"),
            "current_fraction": edit.get("current_fraction"),
            "recommended_fraction": edit.get("recommended_fraction"),
            "current_holdout_fraction": edit.get("current_holdout_fraction"),
            "recommended_holdout_fraction": edit.get("recommended_holdout_fraction"),
            "managed_enforced": False,
            "reviewed_at": utc_now(),
        }
    text = yaml.safe_dump(data, sort_keys=False)
    changed = old_text != text
    backup_path = None
    if changed and not dry_run:
        backup_path = _write_policy_file(policy_path, text, backup_id=backup_id)
    result["files"].append({
        "section": "cache",
        "path": str(policy_path),
        "policy_file": "cache_canary_policy.yaml",
        "changed": bool(changed),
        "backup_path": backup_path,
        "sha256_before": _sha256_text(old_text) if old_text is not None else None,
        "sha256_after": _sha256_text(text),
        "bytes_after": len(text.encode("utf-8")),
    })
    result["applied_sections"] = ["cache"] if review.get("planned_action_count") else []
    result["wrote_policy_files"] = bool(changed and not dry_run)
    result["ok"] = True
    result["summary"] = {
        "action_count": review.get("action_count", 0),
        "planned_action_count": review.get("planned_action_count", 0),
        "changed_action_count": review.get("changed_action_count", 0),
        "wrote_cache_canary_policy": bool(changed and not dry_run),
    }
    return result


def dry_run_openai_cache_replay_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    result = apply_openai_cache_replay_rollout_actions(
        bundle,
        config_dir=config_dir,
        dry_run=True,
        now=now,
    )
    result["schema"] = OPENAI_CACHE_REPLAY_ROLLOUT_DRY_RUN_SCHEMA
    result["read_only"] = True
    result["wrote_policy_files"] = False
    if result.get("files"):
        for item in result["files"]:
            if isinstance(item, dict):
                item["backup_path"] = None
    return result
