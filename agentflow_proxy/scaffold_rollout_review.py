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
SCAFFOLD_LOCAL_CRUNCH_RULES_FILE = "crunch_rules.yaml"

_REPEATED_SCAFFOLD_ACTION_TYPE = "review-local-repeated-scaffold-crunch-rule"
_REPEATED_SCAFFOLD_CANDIDATE_FAMILY = "repeated-scaffold-crunch-policy-rule"
_OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA = "agentflow.optimization_rollout_actions.v1"
_OPTIMIZATION_ROLLOUT_ACTION_SCHEMA = "agentflow.optimization_rollout_action.v1"
_REPEATED_SCAFFOLD_DECISIONS = {"widen", "promote", "hold", "rollback", "suppress"}
_REPEATED_SCAFFOLD_ROLLBACK_DECISIONS = {"rollback", "suppress"}
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
    "provider_bodies_returned",
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


def _nested_dict(source: Any, key: str) -> dict[str, Any]:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else {}


def _nested_value(source: Any, *path: str) -> Any:
    value = source
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _scaffold_next_action(action: dict[str, Any]) -> str:
    raw_values = [
        _nested_value(action, "action", "next_action"),
        action.get("next_action"),
        _nested_value(action, "evidence_summary", "rollout_decision", "next_action"),
        _nested_value(action, "evidence_summary", "rollout_gate", "next_action"),
        _nested_value(action, "noop_action", "status"),
        action.get("action_type"),
        _nested_value(action, "action", "status"),
    ]
    for raw in raw_values:
        value = str(raw or "").strip().lower().replace("_", "-")
        if value in {"review-local-repeated-scaffold-crunch-rule", "canary", "apply"}:
            return "widen"
        if value in {"widen", "promote", "hold", "rollback", "suppress"}:
            return value
        if value in {"disable", "retire", "rejected", "safety-stop"}:
            return "rollback"
        if value in {"omitted", "unsupported", "blocked"}:
            return "suppress"
    return "widen"


def _is_repeated_scaffold_action(action: dict[str, Any]) -> bool:
    candidate_family = str(action.get("candidate_family") or "")
    target = str(action.get("target_candidate_id") or action.get("candidate_id") or "")
    return (
        action.get("action_type") == _REPEATED_SCAFFOLD_ACTION_TYPE
        or candidate_family == _REPEATED_SCAFFOLD_CANDIDATE_FAMILY
        or target.startswith("repeated-scaffold-candidate:")
    )


def _target_identifiers(action: dict[str, Any], rule: dict[str, Any] | None = None) -> dict[str, str | None]:
    nested = _nested_dict(action, "action")
    proposed = _nested_dict(nested, "proposed_edit")
    rule = rule or {}
    rule_id = str(
        rule.get("id")
        or proposed.get("id")
        or action.get("target_rule_id")
        or nested.get("target_rule_id")
        or ""
    ).strip() or None
    candidate_id = str(
        rule.get("candidate_id")
        or proposed.get("candidate_id")
        or action.get("target_candidate_id")
        or action.get("candidate_id")
        or ""
    ).strip() or None
    action_id = str(action.get("action_id") or "").strip() or None
    return {
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "action_id": action_id,
    }


def _target_matches_rule(rule: Any, target: dict[str, str | None]) -> bool:
    if not isinstance(rule, dict):
        return False
    rule_ids = {
        str(rule.get("id") or "").strip(),
        str(rule.get("rule_id") or "").strip(),
    }
    candidate_ids = {
        str(rule.get("candidate_id") or "").strip(),
        str(_nested_value(rule, "rollout_action", "target_candidate_id") or "").strip(),
    }
    action_ids = {
        str(_nested_value(rule, "rollout_action", "action_id") or "").strip(),
    }
    return bool(
        (target.get("rule_id") and target["rule_id"] in rule_ids)
        or (target.get("candidate_id") and target["candidate_id"] in candidate_ids)
        or (target.get("action_id") and target["action_id"] in action_ids)
    )


def _safe_reason_codes(action: dict[str, Any]) -> list[str]:
    raw = (
        action.get("reason_codes")
        or _nested_value(action, "evidence_summary", "rollout_decision", "reason_codes")
        or _nested_value(action, "evidence_summary", "rollout_gate", "reason_codes")
        or []
    )
    if not isinstance(raw, list):
        raw = [raw]
    return [str(item) for item in raw if isinstance(item, (str, int, float, bool))]


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
    omitted_actions = bundle.get("omitted_actions")
    if omitted_actions is None:
        omitted_actions = []
    if not isinstance(omitted_actions, list):
        _add_error(errors, "$.omitted_actions", "expected list")
        omitted_actions = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            _add_error(errors, f"$.actions[{index}]", "expected scaffold rollout action object")
            continue
        if action.get("schema") != _OPTIMIZATION_ROLLOUT_ACTION_SCHEMA:
            _add_error(errors, f"$.actions[{index}].schema", f"expected {_OPTIMIZATION_ROLLOUT_ACTION_SCHEMA}")
        if _is_repeated_scaffold_action(action):
            _rule, action_errors = _repeated_scaffold_rule_from_action(action)
            for error in action_errors:
                _add_error(
                    errors,
                    f"$.actions[{index}]{str(error.get('path') or '$')[1:]}",
                    str(error.get("message") or "invalid repeated-scaffold action"),
                )
    for index, action in enumerate(omitted_actions):
        if not isinstance(action, dict):
            _add_error(errors, f"$.omitted_actions[{index}]", "expected scaffold rollout omitted action object")
            continue
        if _is_repeated_scaffold_action(action):
            _action, action_errors = _repeated_scaffold_lifecycle_action(action)
            for error in action_errors:
                _add_error(
                    errors,
                    f"$.omitted_actions[{index}]{str(error.get('path') or '$')[1:]}",
                    str(error.get("message") or "invalid repeated-scaffold lifecycle action"),
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
        "action_count": len(actions) + len(omitted_actions),
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _repeated_scaffold_lifecycle_action(action: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    decision = _scaffold_next_action(action)
    if decision not in _REPEATED_SCAFFOLD_DECISIONS:
        errors.append({"path": "$.next_action", "message": "unsupported repeated-scaffold rollout decision"})
    if action.get("candidate_family") not in (None, _REPEATED_SCAFFOLD_CANDIDATE_FAMILY):
        errors.append({"path": "$.candidate_family", "message": "not a repeated-scaffold candidate"})
    if action.get("policy_section") not in (None, "crunch") or action.get("action_family") not in (None, "crunch"):
        errors.append({"path": "$.policy_section", "message": "repeated-scaffold actions must target crunch policy"})
    target = _target_identifiers(action)
    if not target.get("rule_id") and not target.get("candidate_id") and not target.get("action_id"):
        errors.append({"path": "$.target_candidate_id", "message": "repeated-scaffold lifecycle action needs a target id"})
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
    return {
        "decision": decision,
        "target": target,
        "reason_codes": _safe_reason_codes(action),
        "policy_source": "managed-recommended",
        "rollout_action": {
            "schema": "agentflow.repeated_scaffold_rollout_action.v1",
            "action_id": action.get("action_id"),
            "action_type": action.get("action_type") or decision,
            "next_action": decision,
            "target_candidate_id": action.get("target_candidate_id"),
            "confidence": action.get("confidence"),
            "reason_codes": _safe_reason_codes(action),
            "reviewed_at": utc_now(),
        },
    }, []


def _repeated_scaffold_rule_from_action(action: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    nested = action.get("action") if isinstance(action.get("action"), dict) else {}
    proposed = nested.get("proposed_edit") if isinstance(nested.get("proposed_edit"), dict) else {}
    proposed_action = proposed.get("action") if isinstance(proposed.get("action"), dict) else {}
    thresholds = nested.get("thresholds") if isinstance(nested.get("thresholds"), dict) else {}
    decision = _scaffold_next_action(action)

    if action.get("action_type") not in {_REPEATED_SCAFFOLD_ACTION_TYPE, *_REPEATED_SCAFFOLD_DECISIONS}:
        errors.append({"path": "$.action_type", "message": "not a supported repeated-scaffold rollout action"})
    if decision not in {"widen", "promote"}:
        return _repeated_scaffold_lifecycle_action(action)
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
            "next_action": decision,
            "target_candidate_id": action.get("target_candidate_id"),
            "confidence": action.get("confidence"),
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "reason_codes": _safe_reason_codes(action),
            "reviewed_at": utc_now(),
        },
    }
    return rule, []


def review_scaffold_rollout_actions(bundle: Any) -> dict[str, Any]:
    validation = validate_scaffold_rollout_bundle(bundle)
    actions = bundle.get("actions") if isinstance(bundle, dict) and isinstance(bundle.get("actions"), list) else []
    omitted_actions = bundle.get("omitted_actions") if isinstance(bundle, dict) and isinstance(bundle.get("omitted_actions"), list) else []
    accepted_actions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if validation.get("ok"):
        review_inputs = [
            ("actions", index, action)
            for index, action in enumerate(actions)
            if isinstance(action, dict)
        ] + [
            ("omitted_actions", index, action)
            for index, action in enumerate(omitted_actions)
            if isinstance(action, dict)
        ]
        for section, index, action in review_inputs:
            if not isinstance(action, dict):
                continue
            if not _is_repeated_scaffold_action(action):
                continue
            rule, rule_errors = _repeated_scaffold_rule_from_action(action)
            if rule_errors:
                for error in rule_errors:
                    errors.append({
                        "path": f"$.{section}[{index}]{error.get('path', '$')[1:]}",
                        "message": str(error.get("message") or "invalid repeated-scaffold action"),
                    })
                continue
            decision = str((rule or {}).get("decision") or _nested_value(rule, "rollout_action", "next_action") or _scaffold_next_action(action))
            target = _target_identifiers(action, rule)
            if rule is not None and "id" in rule:
                accepted_actions.append({
                    "path": f"$.{section}[{index}]",
                    "status": "accepted",
                    "action_id": action.get("action_id"),
                    "decision": decision,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": rule.get("id"),
                    "policy_section": "crunch",
                    "policy_source": "managed-recommended",
                    "target_local_policy": "crunch_rules" if decision == "promote" else "scaffold_canary",
                    "canary_fraction": (rule.get("rollout") or {}).get("canary_fraction"),
                    "holdout_fraction": (rule.get("rollout_action") or {}).get("holdout_fraction"),
                    "reason_codes": (rule.get("rollout_action") or {}).get("reason_codes", []),
                    "proposed_rule": rule,
                })
            elif rule is not None:
                accepted_actions.append({
                    "path": f"$.{section}[{index}]",
                    "status": "accepted",
                    "action_id": action.get("action_id"),
                    "decision": decision,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": target.get("rule_id"),
                    "target": target,
                    "policy_section": "crunch",
                    "policy_source": "managed-recommended",
                    "target_local_policy": "none" if decision == "hold" else "repeated_provider_scaffolding",
                    "reason_codes": rule.get("reason_codes", []),
                    "rollout_action": rule.get("rollout_action"),
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
        "omitted_action_count": len(omitted_actions),
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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _provider_scaffolding_section(data: dict[str, Any]) -> dict[str, Any]:
    section = data.get("repeated_provider_scaffolding")
    if not isinstance(section, dict):
        section = {}
        data["repeated_provider_scaffolding"] = section
    rules = section.get("rules")
    if not isinstance(rules, list):
        section["rules"] = []
    return section


def _promoted_rule(rule: dict[str, Any]) -> dict[str, Any]:
    promoted = {key: value for key, value in rule.items() if key != "decision"}
    rollout = promoted.get("rollout") if isinstance(promoted.get("rollout"), dict) else {}
    promoted["rollout"] = {
        **rollout,
        "schema": rollout.get("schema") or PATTERN_ROLLOUT_SCHEMA,
        "recommendation_mode": "managed-repeated-scaffold-promoted",
        "canary_enabled": False,
        "canary_fraction": 1.0,
    }
    rollout_action = promoted.get("rollout_action") if isinstance(promoted.get("rollout_action"), dict) else {}
    promoted["rollout_action"] = {
        **rollout_action,
        "next_action": "promote",
        "promoted_at": utc_now(),
    }
    promoted["enabled"] = True
    promoted["policy_source"] = "managed-recommended"
    return promoted


def _upsert_rule(rules: list[Any], rule: dict[str, Any]) -> bool:
    target = _target_identifiers({}, rule)
    for index, existing in enumerate(rules):
        if _target_matches_rule(existing, target):
            changed = existing != rule
            rules[index] = rule
            return changed
    rules.append(rule)
    return True


def _disable_matching_rules(
    rules: list[Any],
    *,
    target: dict[str, str | None],
    decision: str,
    reason_codes: list[str],
) -> int:
    changed = 0
    for rule in rules:
        if not _target_matches_rule(rule, target):
            continue
        if not isinstance(rule, dict):
            continue
        before = stable_json(rule)
        rule["enabled"] = False
        rollout = rule.get("rollout") if isinstance(rule.get("rollout"), dict) else {}
        rule["rollout"] = {
            **rollout,
            "schema": rollout.get("schema") or PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "managed-repeated-scaffold-rollback",
            "canary_enabled": False,
            "canary_fraction": 0.0,
        }
        rollout_action = rule.get("rollout_action") if isinstance(rule.get("rollout_action"), dict) else {}
        rule["rollout_action"] = {
            **rollout_action,
            "next_action": decision,
            "rollback_reason_codes": reason_codes,
            "rolled_back_at": utc_now(),
        }
        if stable_json(rule) != before:
            changed += 1
    return changed


def _dump_policy(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def _write_if_changed(path: Path, text: str, *, dry_run: bool) -> tuple[bool, str | None, str | None]:
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    changed = old_text != text
    backup_path = None
    if changed and not dry_run:
        backup_path = _write_policy_file(path, text)
    return changed, backup_path, old_text


def _file_result(
    *,
    section: str,
    path: Path,
    changed: bool,
    backup_path: str | None,
    old_text: str | None,
    new_text: str | None,
    reason: str,
    matched_rule_count: int | None = None,
) -> dict[str, Any]:
    result = {
        "section": section,
        "path": str(path),
        "changed": bool(changed),
        "backup_path": backup_path,
        "sha256_before": _sha256_text(old_text) if old_text is not None else None,
        "sha256_after": _sha256_text(new_text) if new_text is not None else None,
        "bytes_after": len(new_text.encode("utf-8")) if new_text is not None else 0,
        "reason": reason,
    }
    if matched_rule_count is not None:
        result["matched_rule_count"] = matched_rule_count
    return result


def apply_scaffold_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    review = review_scaffold_rollout_actions(bundle)
    config_path = Path(config_dir).expanduser()
    actions = [item for item in review.get("actions", []) if isinstance(item, dict)]

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

    promote_rules: list[dict[str, Any]] = []
    canary_rules: list[dict[str, Any]] = []
    rollback_actions: list[dict[str, Any]] = []
    hold_count = 0
    for action in actions:
        decision = str(action.get("decision") or "")
        rule = action.get("proposed_rule") if isinstance(action.get("proposed_rule"), dict) else None
        if decision == "hold":
            hold_count += 1
            continue
        if decision in _REPEATED_SCAFFOLD_ROLLBACK_DECISIONS:
            rollback_actions.append(action)
            continue
        if rule is None:
            continue
        if decision == "promote":
            promote_rules.append(_promoted_rule(rule))
        else:
            canary_rules.append(rule)

    if not promote_rules and not canary_rules and not rollback_actions:
        result["ok"] = True
        result["files"].append(
            _file_result(
                section="crunch",
                path=config_path / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE,
                changed=False,
                backup_path=None,
                old_text=None,
                new_text=None,
                reason="hold-no-local-policy-change" if hold_count else "no-accepted-scaffold-actions",
            )
        )
        return result

    if canary_rules:
        path = config_path / SCAFFOLD_CANARY_POLICY_FILE
        policy = {
            "schema": SCAFFOLD_CANARY_POLICY_SCHEMA,
            "generated_at": utc_now(),
            "policy_source": "managed-recommended",
            "repeated_provider_scaffolding": {
                "enabled": True,
                "rules": canary_rules,
            },
        }
        text = _dump_policy(policy)
        changed, backup_path, old_text = _write_if_changed(path, text, dry_run=dry_run)
        result["files"].append(
            _file_result(
                section="scaffold_canary",
                path=path,
                changed=changed,
                backup_path=backup_path,
                old_text=old_text,
                new_text=text,
                reason="widen-scaffold-canary",
            )
        )

    if promote_rules:
        path = config_path / SCAFFOLD_LOCAL_CRUNCH_RULES_FILE
        data = _load_yaml_mapping(path)
        section = _provider_scaffolding_section(data)
        section["enabled"] = True
        rules = section["rules"]
        for rule in promote_rules:
            _upsert_rule(rules, rule)
        text = _dump_policy(data)
        changed, backup_path, old_text = _write_if_changed(path, text, dry_run=dry_run)
        result["files"].append(
            _file_result(
                section="crunch",
                path=path,
                changed=changed,
                backup_path=backup_path,
                old_text=old_text,
                new_text=text,
                reason="promote-repeated-scaffold-rule",
            )
        )

    if rollback_actions:
        for filename, section_name in (
            (SCAFFOLD_CANARY_POLICY_FILE, "scaffold_canary"),
            (SCAFFOLD_LOCAL_CRUNCH_RULES_FILE, "crunch"),
        ):
            path = config_path / filename
            if not path.exists():
                result["files"].append(
                    _file_result(
                        section=section_name,
                        path=path,
                        changed=False,
                        backup_path=None,
                        old_text=None,
                        new_text=None,
                        reason="rollback-target-file-missing",
                        matched_rule_count=0,
                    )
                )
                continue
            data = _load_yaml_mapping(path)
            section = _provider_scaffolding_section(data)
            rules = section["rules"]
            matched = 0
            for action in rollback_actions:
                target = action.get("target") if isinstance(action.get("target"), dict) else {
                    "rule_id": action.get("target_rule_id"),
                    "candidate_id": action.get("target_candidate_id"),
                    "action_id": action.get("action_id"),
                }
                matched += _disable_matching_rules(
                    rules,
                    target=target,
                    decision=str(action.get("decision") or "rollback"),
                    reason_codes=[str(item) for item in action.get("reason_codes", [])],
                )
            section["enabled"] = any(isinstance(rule, dict) and _as_bool(rule.get("enabled"), True) for rule in rules)
            text = _dump_policy(data)
            changed, backup_path, old_text = _write_if_changed(path, text, dry_run=dry_run)
            result["files"].append(
                _file_result(
                    section=section_name,
                    path=path,
                    changed=changed,
                    backup_path=backup_path,
                    old_text=old_text,
                    new_text=text,
                    reason="rollback-repeated-scaffold-rule" if matched else "rollback-target-not-found",
                    matched_rule_count=matched,
                )
            )

    result["ok"] = True
    result["wrote_policy_files"] = any(bool(file.get("changed")) for file in result["files"]) and not dry_run
    return result


def redacted_result_json(result: dict[str, Any]) -> str:
    return stable_json(result)
