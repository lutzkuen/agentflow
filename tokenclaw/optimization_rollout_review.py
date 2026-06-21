from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw import __version__
from tokenclaw.policy_bundle import (
    MANAGED_POLICY_VERIFICATION_SECRET_ENV,
    POLICY_BUNDLE_PROVENANCE_SCHEMA,
    _hmac_signature,
    _normalize_signature,
    _secret_for_key_id,
)
from tokenclaw.store import utc_now


OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA = "tokenclaw.optimization_rollout_actions.v1"
OPTIMIZATION_ROLLOUT_ACTION_SCHEMA = "tokenclaw.optimization_rollout_action.v1"
OPTIMIZATION_ROLLOUT_OMITTED_ACTION_SCHEMA = "tokenclaw.optimization_rollout_omitted_action.v1"
OPTIMIZATION_ROLLOUT_ACTION_REVIEW_SCHEMA = "tokenclaw.optimization_rollout_actions_review.v1"
OPTIMIZATION_ROLLOUT_ACTION_VALIDATION_SCHEMA = "tokenclaw.optimization_rollout_actions_validation.v1"
OPTIMIZATION_ROLLOUT_ACTION_PROVENANCE_SCHEMA = "tokenclaw.optimization_rollout_actions_provenance_verification.v1"

SUPPORTED_LOCAL_ACTION_FAMILIES = {"routing", "crunch", "cache", "old_context_summarization"}
SUPPORTED_POLICY_SECTIONS = {"routing", "crunch", "cache", "old_context_summarization"}
ACTIONABLE_ACTION_TYPES = {"widen", "hold", "rollback", "retire", "disable"}
PASSING_EVAL_VERDICTS = {"widen"}
PASSING_ROLLBACK_VERDICTS = {"rollback", "hold", "widen"}
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
ALLOWED_RAW_KEYS = {
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


def canonical_optimization_rollout_bundle_hash(bundle: Any) -> str | None:
    if not isinstance(bundle, dict):
        return None
    payload = _payload_for_hash(bundle)
    return f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"


def attach_optimization_rollout_provenance(
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
        "bundle_hash": canonical_optimization_rollout_bundle_hash(signed),
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
    return parsed


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
            if lowered not in ALLOWED_RAW_KEYS and any(part in lowered for part in RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    _add_error(errors, child_path, "raw or local-identifier rollout payloads are not accepted")
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


def verify_optimization_rollout_provenance(bundle: Any) -> dict[str, Any]:
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    managed_bundle = isinstance(bundle, dict) and bundle.get("schema") == OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA
    secret, configured = _secret_for_key_id(provenance.get("key_id") if isinstance(provenance, dict) else None)
    result: dict[str, Any] = {
        "schema": OPTIMIZATION_ROLLOUT_ACTION_PROVENANCE_SCHEMA,
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
        "computed_bundle_hash": canonical_optimization_rollout_bundle_hash(bundle),
        "signature_present": False,
        "errors": [],
        "warnings": [],
    }
    if not configured:
        result["status"] = "not-configured"
        if managed_bundle:
            if not isinstance(provenance, dict):
                result["ok"] = False
                result["status"] = "missing"
                result["errors"].append({
                    "path": "$.provenance",
                    "message": "managed optimization rollout bundle is missing provenance required for local review",
                })
                return result
            result["warnings"].append({
                "path": "$.provenance",
                "message": f"managed optimization rollout provenance was not verified because {MANAGED_POLICY_VERIFICATION_SECRET_ENV} is not configured",
            })
        return result
    if not isinstance(provenance, dict):
        result["ok"] = False
        result["status"] = "missing"
        result["errors"].append({
            "path": "$.provenance",
            "message": "managed optimization rollout bundle is missing provenance required by configured verification",
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
    if provenance.get("bundle_hash") != canonical_optimization_rollout_bundle_hash(bundle):
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


def _minimum_version_compatible(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    current = _version_tuple(__version__)
    required = _version_tuple(value)
    if not current or not required:
        return True
    return current >= required


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.replace("-", ".").split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def _local_eval_verdict(action: dict[str, Any]) -> dict[str, Any] | None:
    evidence = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
    verdict = evidence.get("local_eval_verdict")
    return verdict if isinstance(verdict, dict) else None


def _validate_compatibility(
    value: Any,
    path: str,
    errors: list[dict[str, str]],
    *,
    family: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _add_error(errors, path, "local executor compatibility contract is required")
        return {}
    if not isinstance(value.get("minimum_local_client_version"), str) or not value.get("minimum_local_client_version").strip():
        _add_error(errors, f"{path}.minimum_local_client_version", "expected non-empty string")
    elif not _minimum_version_compatible(value.get("minimum_local_client_version")):
        _add_error(errors, f"{path}.minimum_local_client_version", "minimum local client version is newer than this package")
    if value.get("compatible") is False:
        _add_error(errors, f"{path}.compatible", "managed bundle reports local executor incompatibility")
    elif value.get("compatible") is not True:
        _add_error(errors, f"{path}.compatible", "expected true")
    supported = value.get("supported_local_action_families")
    if not isinstance(supported, list) or not supported:
        _add_error(errors, f"{path}.supported_local_action_families", "expected non-empty list")
    elif family and family not in {str(item) for item in supported}:
        _add_error(errors, f"{path}.supported_local_action_families", "action family is not supported by compatibility contract")
    return value


def _validate_action(action: Any, path: str, errors: list[dict[str, str]], *, now: datetime) -> None:
    if not isinstance(action, dict):
        _add_error(errors, path, "expected optimization rollout action object")
        return
    if action.get("schema") != OPTIMIZATION_ROLLOUT_ACTION_SCHEMA:
        _add_error(errors, f"{path}.schema", f"expected {OPTIMIZATION_ROLLOUT_ACTION_SCHEMA}")
    if str(action.get("action_type") or "") not in ACTIONABLE_ACTION_TYPES:
        _add_error(errors, f"{path}.action_type", "expected widen, hold, rollback, retire, or disable")
    if not isinstance(action.get("action_id"), str) or not action.get("action_id").strip():
        _add_error(errors, f"{path}.action_id", "expected non-empty string")
    if not isinstance(action.get("target_candidate_id"), str) or not action.get("target_candidate_id").strip():
        _add_error(errors, f"{path}.target_candidate_id", "expected non-empty string")
    family = str(action.get("action_family") or "")
    if family not in SUPPORTED_LOCAL_ACTION_FAMILIES:
        _add_error(errors, f"{path}.action_family", "unsupported local action family")
    if action.get("policy_section") not in SUPPORTED_POLICY_SECTIONS:
        _add_error(errors, f"{path}.policy_section", "unsupported local policy section")
    if action.get("required_local_review") is not True:
        _add_error(errors, f"{path}.required_local_review", "expected true")
    if action.get("managed_enforced") is not False:
        _add_error(errors, f"{path}.managed_enforced", "expected false")
    expires_at = _parse_datetime(action.get("expires_at"))
    if expires_at is None:
        _add_error(errors, f"{path}.expires_at", "expected ISO-8601 timestamp string")
    elif expires_at <= now:
        _add_error(errors, f"{path}.expires_at", "optimization rollout action is expired")
    _validate_compatibility(action.get("local_executor_compatibility"), f"{path}.local_executor_compatibility", errors, family=family)
    verdict = _local_eval_verdict(action)
    action_type = str(action.get("action_type") or "")
    if not verdict:
        _add_error(errors, f"{path}.evidence_summary.local_eval_verdict", "local eval verdict evidence is required")
    elif action_type in {"rollback", "retire", "disable"}:
        if str(verdict.get("verdict") or "") not in PASSING_ROLLBACK_VERDICTS:
            _add_error(errors, f"{path}.evidence_summary.local_eval_verdict.verdict", "rollback action requires rollback, hold, or widen local eval verdict")
    elif str(verdict.get("verdict") or "") not in PASSING_EVAL_VERDICTS:
        _add_error(errors, f"{path}.evidence_summary.local_eval_verdict.verdict", "local eval verdict must be widen")
    privacy = action.get("privacy_summary") if isinstance(action.get("privacy_summary"), dict) else {}
    for key in ("feature_only",):
        if privacy.get(key) is not True:
            _add_error(errors, f"{path}.privacy_summary.{key}", "expected true")
    for key in ("provider_forwarding", "managed_enforced"):
        if privacy.get(key) is not False:
            _add_error(errors, f"{path}.privacy_summary.{key}", "expected false")


def _validate_omitted_action(item: Any, path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(item, dict):
        _add_error(errors, path, "expected omitted optimization rollout action object")
        return
    if item.get("schema") != OPTIMIZATION_ROLLOUT_OMITTED_ACTION_SCHEMA:
        _add_error(errors, f"{path}.schema", f"expected {OPTIMIZATION_ROLLOUT_OMITTED_ACTION_SCHEMA}")
    if not isinstance(item.get("reason"), str) or not item.get("reason").strip():
        _add_error(errors, f"{path}.reason", "omitted action reason is required")
    if not isinstance(item.get("target_candidate_id"), str) or not item.get("target_candidate_id").strip():
        _add_error(errors, f"{path}.target_candidate_id", "expected non-empty string")
    family = str(item.get("action_family") or "")
    if not family:
        _add_error(errors, f"{path}.action_family", "expected non-empty string")
    privacy = item.get("privacy_summary") if isinstance(item.get("privacy_summary"), dict) else {}
    if privacy.get("metadata_only") is not True:
        _add_error(errors, f"{path}.privacy_summary.metadata_only", "expected true")
    if privacy.get("feature_only") is not True:
        _add_error(errors, f"{path}.privacy_summary.feature_only", "expected true")


def _max_evidence_age_seconds(bundle: dict[str, Any]) -> int:
    thresholds = bundle.get("thresholds") if isinstance(bundle.get("thresholds"), dict) else {}
    value = thresholds.get("max_evidence_age_seconds")
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _validate_action_evidence_freshness(
    action: dict[str, Any],
    path: str,
    errors: list[dict[str, str]],
    *,
    now: datetime,
    max_age_seconds: int,
) -> None:
    if max_age_seconds <= 0:
        return
    verdict = _local_eval_verdict(action)
    if not verdict:
        return
    latest = _parse_datetime(verdict.get("latest_eval_at"))
    if latest is None:
        _add_error(errors, f"{path}.evidence_summary.local_eval_verdict.latest_eval_at", "expected fresh local eval timestamp")
    elif (now - latest).total_seconds() > max_age_seconds:
        _add_error(errors, f"{path}.evidence_summary.local_eval_verdict.latest_eval_at", "local eval evidence is stale")


def validate_optimization_rollout_bundle(bundle: Any, *, now: datetime | None = None) -> dict[str, Any]:
    effective_now = now or datetime.now(timezone.utc)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_optimization_rollout_provenance(bundle)

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "optimization rollout bundle must be a JSON object")
        return {
            "schema": OPTIMIZATION_ROLLOUT_ACTION_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "action_count": 0,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
        }
    if bundle.get("schema") != OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA:
        _add_error(errors, "$.schema", f"expected {OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA}")
    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")
    expires_at = _parse_datetime(bundle.get("expires_at"))
    if expires_at is None:
        _add_error(errors, "$.expires_at", "expected ISO-8601 timestamp string")
    elif expires_at <= effective_now:
        _add_error(errors, "$.expires_at", "optimization rollout bundle is expired")
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    if summary.get("managed_enforced") is not False:
        _add_error(errors, "$.summary.managed_enforced", "expected false")
    if summary.get("provider_forwarding") is not False:
        _add_error(errors, "$.summary.provider_forwarding", "expected false")
    privacy = bundle.get("privacy_summary") if isinstance(bundle.get("privacy_summary"), dict) else {}
    if privacy.get("feature_only") is not True:
        _add_error(errors, "$.privacy_summary.feature_only", "expected true")
    for key in ("provider_forwarding", "managed_enforced"):
        if privacy.get(key) is not False:
            _add_error(errors, f"$.privacy_summary.{key}", "expected false")
    _validate_compatibility(bundle.get("local_executor_compatibility"), "$.local_executor_compatibility", errors)
    actions = bundle.get("actions")
    if not isinstance(actions, list):
        _add_error(errors, "$.actions", "expected list")
        actions = []
    max_age_seconds = _max_evidence_age_seconds(bundle)
    for index, action in enumerate(actions):
        path = f"$.actions[{index}]"
        _validate_action(action, path, errors, now=effective_now)
        if isinstance(action, dict):
            _validate_action_evidence_freshness(
                action,
                path,
                errors,
                now=effective_now,
                max_age_seconds=max_age_seconds,
            )
    omitted_actions = bundle.get("omitted_actions")
    if omitted_actions is None:
        omitted_actions = []
    if not isinstance(omitted_actions, list):
        _add_error(errors, "$.omitted_actions", "expected list")
        omitted_actions = []
    for index, item in enumerate(omitted_actions):
        _validate_omitted_action(item, f"$.omitted_actions[{index}]", errors)
    _scan_raw_like(bundle, "$", errors)
    _privacy_flags_safe(bundle, "$", errors)
    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            _add_warning(warnings, str(warning.get("path") or "$.provenance"), str(warning.get("message") or "provenance was not verified"))
    return {
        "schema": OPTIMIZATION_ROLLOUT_ACTION_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle.get("schema"),
        "action_count": len(actions),
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _action_review(action: dict[str, Any]) -> dict[str, Any]:
    nested = action.get("action") if isinstance(action.get("action"), dict) else {}
    proposed_edit = nested.get("proposed_edit") if isinstance(nested.get("proposed_edit"), dict) else {}
    return {
        "status": "accepted",
        "action_id": action.get("action_id"),
        "action_type": action.get("action_type"),
        "target_candidate_id": action.get("target_candidate_id"),
        "action_family": action.get("action_family"),
        "candidate_family": action.get("candidate_family"),
        "policy_section": action.get("policy_section"),
        "source_surface": action.get("source_surface"),
        "provider_endpoint": action.get("provider_endpoint"),
        "confidence": action.get("confidence"),
        "required_local_review": True,
        "managed_enforced": False,
        "locally_executed": False,
        "provider_forwarding": False,
        "target_rule_id": nested.get("target_rule_id") or proposed_edit.get("rule_id"),
        "local_apply_hint": {
            "review_only": True,
            "reuse_existing_apply_paths": True,
            "nested_action_schema": nested.get("schema"),
        },
        "evidence_summary": action.get("evidence_summary"),
    }


def review_optimization_rollout_actions(bundle: Any, *, now: datetime | None = None) -> dict[str, Any]:
    validation = validate_optimization_rollout_bundle(bundle, now=now)
    actions = bundle.get("actions") if isinstance(bundle, dict) and isinstance(bundle.get("actions"), list) else []
    accepted = [_action_review(action) for action in actions if isinstance(action, dict)] if validation["ok"] else []
    policy_sections = [str(row.get("policy_section") or "unknown") for row in accepted]
    action_families = [str(row.get("action_family") or "unknown") for row in accepted]
    return {
        "schema": OPTIMIZATION_ROLLOUT_ACTION_REVIEW_SCHEMA,
        "ok": bool(validation["ok"]),
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "locally_executed": False,
        "validation": validation,
        "provenance": validation.get("provenance"),
        "summary": {
            "action_count": len(actions),
            "accepted_action_count": len(accepted),
            "error_count": len(validation.get("errors", [])),
            "warning_count": len(validation.get("warnings", [])),
            "policy_section_counts": _counts(policy_sections),
            "action_family_counts": _counts(action_families),
        },
        "actions": accepted,
        "omitted_actions": bundle.get("omitted_actions", []) if isinstance(bundle, dict) and isinstance(bundle.get("omitted_actions"), list) else [],
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
        "privacy": {
            "metadata_only": True,
            "feature_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
        },
    }
