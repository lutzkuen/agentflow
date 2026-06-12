from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.optimization_rollout_review import verify_optimization_rollout_provenance
from agentflow_proxy.pattern_rollout import PATTERN_ROLLOUT_SCHEMA
from agentflow_proxy.store import stable_json, utc_now


SCAFFOLD_ROLLOUT_ACTIONS_REVIEW_SCHEMA = "agentflow.scaffold_rollout_actions_fetch_review.v1"
SCAFFOLD_ROLLOUT_ACTIONS_APPLY_SCHEMA = "agentflow.scaffold_rollout_actions_apply.v1"
SCAFFOLD_CANARY_POLICY_SCHEMA = "agentflow.scaffold_canary_policy.v1"
SCAFFOLD_CANARY_POLICY_FILE = "scaffold_canary_policy.yaml"

_REPEATED_SCAFFOLD_ACTION_TYPE = "review-local-repeated-scaffold-crunch-rule"
_REPEATED_SCAFFOLD_CANDIDATE_FAMILY = "repeated-scaffold-crunch-policy-rule"
_OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA = "agentflow.optimization_rollout_actions.v1"
_OPTIMIZATION_ROLLOUT_ACTION_SCHEMA = "agentflow.optimization_rollout_action.v1"
_RAW_LIKE_KEY_PARTS = (
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
_ALLOWED_RAW_KEYS = {
    "raw_body_storage",
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_payloads_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "request_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_responses_included",
}
_UNSAFE_PRIVACY_KEYS = {
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_payloads_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "raw_body_storage",
    "request_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "source_surface",
        "app_family",
        "phase",
        "category",
        "requested_model",
        "text_bucket",
        "token_bucket",
        "has_tools",
        "uses_thinking",
        "public_labels",
        "dominant_scaffold_bucket",
        "dominant_duplicate_bucket",
    }
    safe: dict[str, Any] = {}
    for key in allowed:
        item = value.get(key)
        if item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            safe[key] = item
        elif isinstance(item, list):
            safe[key] = [part for part in item if isinstance(part, (str, int, float, bool))]
    return safe


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


def _truthy(value: Any) -> bool:
    return value not in (None, False, 0, "", [], {})


def _add_error(errors: list[dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _scan_raw_like(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    _add_error(errors, child_path, "raw or local-identifier scaffold rollout payloads are not accepted")
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


def _privacy_flags_safe(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else f"$.{key}"
            if str(key).lower() in _UNSAFE_PRIVACY_KEYS and bool(item):
                _add_error(errors, child_path, "privacy summary reports raw payloads or local identifiers")
            _privacy_flags_safe(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _privacy_flags_safe(item, f"{path}[{index}]", errors)


def validate_scaffold_rollout_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_optimization_rollout_provenance(bundle)

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "scaffold rollout bundle must be a JSON object")
        return {
            "schema": "agentflow.scaffold_rollout_actions_validation.v1",
            "ok": False,
            "bundle_schema": None,
            "action_count": 0,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
        }
    if bundle.get("schema") != _OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA:
        _add_error(errors, "$.schema", f"expected {_OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA}")
    if _parse_datetime(bundle.get("generated_at")) is None:
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")
    expires_at = _parse_datetime(bundle.get("expires_at"))
    if expires_at is None:
        _add_error(errors, "$.expires_at", "expected ISO-8601 timestamp string")
    elif expires_at <= datetime.now(timezone.utc):
        _add_error(errors, "$.expires_at", "scaffold rollout bundle is expired")
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    for key in ("managed_enforced", "provider_forwarding", "server_content_processing"):
        if summary.get(key) is not False:
            _add_error(errors, f"$.summary.{key}", "expected false")
    privacy = bundle.get("privacy_summary") if isinstance(bundle.get("privacy_summary"), dict) else {}
    if privacy.get("feature_only") is not True:
        _add_error(errors, "$.privacy_summary.feature_only", "expected true")
    for key in ("provider_forwarding", "managed_enforced"):
        if privacy.get(key) is not False:
            _add_error(errors, f"$.privacy_summary.{key}", "expected false")
    compatibility = bundle.get("local_executor_compatibility") if isinstance(bundle.get("local_executor_compatibility"), dict) else {}
    if compatibility.get("compatible") is not True:
        _add_error(errors, "$.local_executor_compatibility.compatible", "expected true")
    supported = compatibility.get("supported_local_action_families")
    if not isinstance(supported, list) or "crunch" not in {str(item) for item in supported}:
        _add_error(errors, "$.local_executor_compatibility.supported_local_action_families", "expected crunch support")

    actions = bundle.get("actions")
    if not isinstance(actions, list):
        _add_error(errors, "$.actions", "expected list")
        actions = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            _add_error(errors, f"$.actions[{index}]", "expected scaffold rollout action object")
            continue
        if action.get("schema") != _OPTIMIZATION_ROLLOUT_ACTION_SCHEMA:
            _add_error(errors, f"$.actions[{index}].schema", f"expected {_OPTIMIZATION_ROLLOUT_ACTION_SCHEMA}")
        if action.get("action_type") == _REPEATED_SCAFFOLD_ACTION_TYPE or action.get("candidate_family") == _REPEATED_SCAFFOLD_CANDIDATE_FAMILY:
            _rule, action_errors = _repeated_scaffold_rule_from_action(action)
            for error in action_errors:
                _add_error(
                    errors,
                    f"$.actions[{index}]{str(error.get('path') or '$')[1:]}",
                    str(error.get("message") or "invalid repeated-scaffold action"),
                )

    _scan_raw_like(bundle, "$", errors)
    _privacy_flags_safe(bundle, "$", errors)
    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            warnings.append({"path": str(warning.get("path") or "$.provenance"), "message": str(warning.get("message") or "provenance was not verified")})

    return {
        "schema": "agentflow.scaffold_rollout_actions_validation.v1",
        "ok": not errors,
        "bundle_schema": bundle.get("schema"),
        "action_count": len(actions),
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _repeated_scaffold_rule_from_action(action: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    nested = action.get("action") if isinstance(action.get("action"), dict) else {}
    proposed = nested.get("proposed_edit") if isinstance(nested.get("proposed_edit"), dict) else {}
    proposed_action = proposed.get("action") if isinstance(proposed.get("action"), dict) else {}
    thresholds = nested.get("thresholds") if isinstance(nested.get("thresholds"), dict) else {}

    if action.get("action_type") != _REPEATED_SCAFFOLD_ACTION_TYPE:
        errors.append({"path": "$.action_type", "message": "not a repeated-scaffold rollout action"})
    if action.get("candidate_family") != _REPEATED_SCAFFOLD_CANDIDATE_FAMILY:
        errors.append({"path": "$.candidate_family", "message": "not a repeated-scaffold candidate"})
    if action.get("policy_section") != "crunch" or action.get("action_family") != "crunch":
        errors.append({"path": "$.policy_section", "message": "repeated-scaffold actions must target crunch policy"})
    if action.get("policy_source") != "managed-recommended" or proposed.get("policy_source") != "managed-recommended":
        errors.append({"path": "$.policy_source", "message": "only managed-recommended scaffold actions may be applied"})
    if action.get("managed_enforced") is not False or nested.get("managed_enforced") is not False:
        errors.append({"path": "$.managed_enforced", "message": "managed-enforced scaffold actions are not accepted locally"})
    if not _as_bool(action.get("required_local_review"), False) or not _as_bool(nested.get("review_only"), False):
        errors.append({"path": "$.required_local_review", "message": "scaffold rollout action must require local review"})
    compatibility = action.get("local_executor_compatibility") if isinstance(action.get("local_executor_compatibility"), dict) else {}
    if compatibility.get("compatible") is not True or not compatibility.get("requires_repeated_scaffold_support"):
        errors.append({"path": "$.local_executor_compatibility", "message": "local executor compatibility does not allow scaffold apply"})
    privacy = action.get("privacy_summary") if isinstance(action.get("privacy_summary"), dict) else {}
    for key in (
        "raw_payloads_returned",
        "raw_prompts_returned",
        "raw_responses_returned",
        "provider_bodies_returned",
        "request_ids_returned",
        "tenant_ids_returned",
        "cache_keys_returned",
        "file_paths_returned",
        "provider_forwarding",
        "server_content_processing",
        "managed_enforced",
    ):
        if bool(privacy.get(key)):
            errors.append({"path": f"$.privacy_summary.{key}", "message": "scaffold rollout action is not metadata-only"})

    if errors:
        return None, errors

    canary_fraction = _as_float(proposed_action.get("canary_fraction", nested.get("canary_fraction")), 0.25)
    holdout_fraction = _as_float(proposed_action.get("holdout_fraction", nested.get("holdout_fraction")), 1.0 - canary_fraction)
    rule_id = str(proposed.get("id") or action.get("target_rule_id") or action.get("target_candidate_id") or "managed-repeated-scaffold")
    candidate_id = str(proposed.get("candidate_id") or action.get("target_candidate_id") or rule_id)
    min_repeated = max(
        2,
        _as_int(
            proposed_action.get("min_duplicate_blocks", thresholds.get("min_duplicate_blocks")),
            _as_int(proposed_action.get("min_repeated_blocks", thresholds.get("min_repeated_blocks")), 2),
        ),
    )
    max_applications = max(1, _as_int(proposed_action.get("max_replacements_per_request"), 8))
    min_request_chars = max(1, _as_int(thresholds.get("min_text_chars"), 8000))
    replacement_notice = str(proposed_action.get("replacement_notice") or "[repeated provider-message scaffold removed locally]")
    max_replacement_chars = max(80, min(720, len(replacement_notice) + 240))

    rule = {
        "id": rule_id,
        "enabled": _as_bool(proposed.get("enabled"), True),
        "policy_source": "managed-recommended",
        "candidate_id": candidate_id,
        "description": str(proposed.get("description") or "Managed repeated provider-message scaffold crunch canary."),
        "match_any_repeated": True,
        "conditions": _safe_conditions(proposed.get("conditions")),
        "min_request_chars": min_request_chars,
        "min_repeated_count": min_repeated,
        "max_applications": max_applications,
        "block_tool_protocol": True,
        "block_thinking": True,
        "action": {
            "type": "omit",
            "max_replacement_chars": max_replacement_chars,
        },
        "rollout": {
            "schema": PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "managed-repeated-scaffold-canary",
            "canary_enabled": True,
            "canary_fraction": canary_fraction,
            "canary_salt": str(action.get("action_id") or candidate_id),
            "canary_unit": "request_fingerprint",
            "local_feedback_fields": [
                "applied_count",
                "holdout_count",
                "saved_chars",
                "tokens_saved_est",
                "status_code",
                "retry_count",
                "latency_ms",
            ],
        },
        "rollout_action": {
            "schema": "agentflow.repeated_scaffold_rollout_action.v1",
            "action_id": action.get("action_id"),
            "action_type": action.get("action_type"),
            "target_candidate_id": action.get("target_candidate_id"),
            "confidence": action.get("confidence"),
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "reviewed_at": utc_now(),
        },
    }
    return rule, []


def review_scaffold_rollout_actions(bundle: Any) -> dict[str, Any]:
    validation = validate_scaffold_rollout_bundle(bundle)
    actions = bundle.get("actions") if isinstance(bundle, dict) and isinstance(bundle.get("actions"), list) else []
    accepted_actions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if validation.get("ok"):
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            if action.get("action_type") != _REPEATED_SCAFFOLD_ACTION_TYPE and action.get("candidate_family") != _REPEATED_SCAFFOLD_CANDIDATE_FAMILY:
                continue
            rule, rule_errors = _repeated_scaffold_rule_from_action(action)
            if rule_errors:
                for error in rule_errors:
                    errors.append({
                        "path": f"$.actions[{index}]{error.get('path', '$')[1:]}",
                        "message": str(error.get("message") or "invalid repeated-scaffold action"),
                    })
                continue
            if rule is not None:
                accepted_actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "accepted",
                    "action_id": action.get("action_id"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": rule.get("id"),
                    "policy_section": "crunch",
                    "policy_source": "managed-recommended",
                    "canary_fraction": (rule.get("rollout") or {}).get("canary_fraction"),
                    "holdout_fraction": (rule.get("rollout_action") or {}).get("holdout_fraction"),
                    "proposed_rule": rule,
                })

    validation_errors = validation.get("errors", []) if isinstance(validation.get("errors"), list) else []
    ok = bool(validation.get("ok") and not errors)
    return {
        "schema": SCAFFOLD_ROLLOUT_ACTIONS_REVIEW_SCHEMA,
        "ok": ok,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "locally_executed": False,
        "validation": validation,
        "provenance": validation.get("provenance"),
        "action_count": len(actions),
        "accepted_action_count": len(accepted_actions),
        "errors": [*validation_errors, *errors],
        "warnings": validation.get("warnings", []),
        "actions": accepted_actions,
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


def apply_scaffold_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    review = review_scaffold_rollout_actions(bundle)
    config_path = Path(config_dir).expanduser()
    path = config_path / SCAFFOLD_CANARY_POLICY_FILE
    actions = [item for item in review.get("actions", []) if isinstance(item, dict)]
    rules = [(item.get("proposed_rule") or {}) for item in actions if isinstance(item.get("proposed_rule"), dict)]
    policy = {
        "schema": SCAFFOLD_CANARY_POLICY_SCHEMA,
        "generated_at": utc_now(),
        "policy_source": "managed-recommended",
        "repeated_provider_scaffolding": {
            "enabled": bool(rules),
            "rules": rules,
        },
    }
    text = yaml.safe_dump(policy, sort_keys=False)
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    changed = old_text != text
    backup_path = None

    result: dict[str, Any] = {
        "schema": SCAFFOLD_ROLLOUT_ACTIONS_APPLY_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "action_count": review.get("action_count", 0),
        "accepted_action_count": review.get("accepted_action_count", 0),
        "actions": actions,
        "files": [],
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "locally_executed": False,
        "error": None,
    }
    if not review.get("ok"):
        result["error"] = {"type": "validation_failed", "message": "scaffold rollout actions are invalid"}
        return result
    if not rules:
        result["ok"] = True
        result["files"].append({
            "section": "crunch",
            "path": str(path),
            "changed": False,
            "backup_path": None,
            "sha256_before": _sha256_text(old_text) if old_text is not None else None,
            "sha256_after": None,
            "bytes_after": 0,
            "reason": "no-accepted-scaffold-actions",
        })
        return result

    if changed and not dry_run:
        backup_path = _write_policy_file(path, text)
    result["files"].append({
        "section": "crunch",
        "path": str(path),
        "changed": bool(changed),
        "backup_path": backup_path,
        "sha256_before": _sha256_text(old_text) if old_text is not None else None,
        "sha256_after": _sha256_text(text),
        "bytes_after": len(text.encode("utf-8")),
    })
    result["ok"] = True
    result["wrote_policy_files"] = bool(changed and not dry_run)
    return result


def redacted_result_json(result: dict[str, Any]) -> str:
    return stable_json(result)
