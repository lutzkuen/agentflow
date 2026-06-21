from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenclaw import __version__
from tokenclaw.policy_files import stage_policy_draft
from tokenclaw.policy_workbench import validate_staged_policy_draft
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.managed_activation_bundle_import.v1"
STAGED_ENTRY_SCHEMA = "tokenclaw.managed_activation_policy_draft_entry.v1"
SUPPORTED_INPUT_SCHEMA = "tokenclaw.local_activation_outcome_policy_bundle_drafts.v1"
SUPPORTED_FAMILIES = {"cache", "crunch"}
RULE_FILES = {
    "cache": "cache_rules.yaml",
    "crunch": "crunch_rules.yaml",
}
RULE_FILE_ALIASES = {
    "cache-rules.yaml": "cache_rules.yaml",
    "cache_rules.yaml": "cache_rules.yaml",
    "crunch-rules.yaml": "crunch_rules.yaml",
    "crunch_rules.yaml": "crunch_rules.yaml",
}
SENSITIVE_KEYS = {
    "api_key",
    "api_keys",
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "content",
    "contents",
    "file_content",
    "file_contents",
    "file_path",
    "file_paths",
    "message",
    "messages",
    "password",
    "passwords",
    "prompt",
    "prompts",
    "provider_body",
    "raw_context",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_fingerprint",
    "request_fingerprints",
    "request_id",
    "request_ids",
    "response",
    "secret",
    "secrets",
    "session_id",
    "session_ids",
    "tenant_id",
    "tenant_ids",
    "system_prompt",
    "thread_id",
    "thread_ids",
    "tool_input",
    "tool_inputs",
    "tool_payload",
    "tool_payloads",
    "tool_result",
    "tool_results",
    "transcript",
    "transcripts",
}
SENSITIVE_INCLUDED_FLAGS = {
    "absolute_paths_included",
    "cache_keys_included",
    "file_paths_included",
    "provider_bodies_included",
    "raw_messages_included",
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_request_bodies_included",
    "raw_response_bodies_included",
    "raw_responses_included",
    "raw_session_ids_included",
    "raw_source_reports_included",
    "raw_tool_payloads_included",
    "raw_transcripts_included",
    "request_fingerprints_included",
    "request_ids_included",
    "session_ids_included",
    "tenant_ids_included",
    "tool_payloads_included",
}


def _privacy() -> dict[str, Any]:
    return {
        "local_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "feature_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "file_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return default


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.replace("-", ".").split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            break
    return tuple(parts or [0])


def _minimum_version_supported(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    return _version_tuple(__version__) >= _version_tuple(value)


def _target_rule_file(value: Any, family: str) -> str | None:
    normalized = RULE_FILE_ALIASES.get(str(value or "").strip())
    expected = RULE_FILES.get(family)
    if normalized and normalized == expected:
        return normalized
    return None


def _content_errors(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key).strip()
                lowered = key_text.lower()
                child_path = f"{item_path}.{key_text}"
                if lowered in SENSITIVE_INCLUDED_FLAGS:
                    if _as_bool(child, False):
                        errors.append({
                            "path": child_path,
                            "message": "managed activation bundles must be metadata-only and may not include raw content, identifiers, cache keys, tool payloads, tenant IDs, or local paths",
                        })
                    continue
                if lowered in SENSITIVE_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "managed activation bundles may not contain raw provider content, prompts, messages, identifiers, cache keys, tool payloads, secrets, or local paths",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "rejected",
        "generated_at": utc_now(),
        "dry_run": True,
        "summary": {
            "action_count": 0,
            "staged_count": 0,
            "skipped_count": 0,
            "omitted_count": 0,
            "rejected_count": 1,
        },
        "staged": [],
        "skipped": [],
        "omitted": [],
        "rejected": [{"reason_codes": [error_type], "message": message}],
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


def _action_skip(action: dict[str, Any], reason: str, *, message: str | None = None) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.managed_activation_bundle_import_skip.v1",
        "status": "skipped",
        "reason_code": reason,
        "reason_codes": [reason],
        "message": message or reason,
        "action_id": action.get("action_id"),
        "draft_id": action.get("draft_id"),
        "recommendation_id": action.get("recommendation_id"),
        "local_action_family": action.get("local_action_family"),
        "policy_section": action.get("policy_section"),
        "target_local_rule_file": action.get("target_local_rule_file"),
        "policy_source": action.get("policy_source"),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _draft_for_action(bundle: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    drafts = bundle.get("drafts")
    if not isinstance(drafts, list):
        return {}
    draft_id = action.get("draft_id")
    for item in drafts:
        if isinstance(item, dict) and item.get("draft_id") == draft_id:
            return item
    return {}


def _compatibility(action: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    for source in (action, draft):
        value = source.get("local_executor_compatibility")
        if isinstance(value, dict):
            return value
    return {}


def _expiration(bundle: dict[str, Any], draft: dict[str, Any]) -> str | None:
    for source in (draft, bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}):
        value = source.get("expires_at") if isinstance(source, dict) else None
        if isinstance(value, str) and value.strip():
            return value
    return None


def _skip_reason(bundle: dict[str, Any], action: dict[str, Any], draft: dict[str, Any]) -> tuple[str | None, str | None]:
    family = str(action.get("local_action_family") or "").strip()
    if family not in SUPPORTED_FAMILIES:
        return "unsupported-local-action-family", "only cache and crunch activation bundles can be staged locally by this importer"
    if action.get("policy_section") not in (None, family):
        return "policy-section-mismatch", "local action family and policy section do not match"
    if _target_rule_file(action.get("target_local_rule_file") or draft.get("target_local_rule_file"), family) is None:
        return "unsupported-target-rule-file", "target local rule file is not cache_rules.yaml or crunch_rules.yaml"
    if action.get("policy_source") not in (None, "managed-recommended"):
        return "unsupported-policy-source", "only managed-recommended review drafts are accepted"
    if action.get("managed_enforced") is True or draft.get("managed_enforced") is True:
        return "managed-enforced-not-local-draft", "managed-enforced bundles cannot be imported as local review drafts"
    if action.get("provider_forwarding") is True or action.get("server_content_processing") is True:
        return "non-local-action-boundary", "bundle action would require provider forwarding or server content processing"
    if action.get("feature_only") is False or draft.get("feature_only") is False:
        return "not-feature-only", "bundle action is not marked feature-only"
    compatibility = _compatibility(action, draft)
    if compatibility.get("compatible") is False:
        return "local-executor-incompatible", "bundle action is not compatible with the local executor"
    if not _minimum_version_supported(compatibility.get("minimum_local_client_version")):
        return "minimum-local-client-version-not-met", "bundle action requires a newer local AgentFlow version"
    expires_at = _expiration(bundle, draft)
    parsed_expiration = _parse_time(expires_at)
    if expires_at and parsed_expiration is None:
        return "invalid-expiration", "bundle action expiration could not be parsed"
    now = _parse_time(utc_now())
    if parsed_expiration is not None and now is not None and parsed_expiration < now:
        return "expired-bundle-action", "bundle action has expired"
    return None, None


def _staged_entry(bundle: dict[str, Any], action: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    family = str(action.get("local_action_family") or draft.get("local_action_family") or "").strip()
    target_rule_file = _target_rule_file(action.get("target_local_rule_file") or draft.get("target_local_rule_file"), family)
    local_policy_draft = action.get("local_policy_draft")
    if not isinstance(local_policy_draft, dict):
        local_policy_draft = draft.get("recommended_local_rule") if isinstance(draft.get("recommended_local_rule"), dict) else {}
    entry = {
        "schema": STAGED_ENTRY_SCHEMA,
        "status": "review-required",
        "bundle_id": bundle.get("bundle_id"),
        "action_id": action.get("action_id"),
        "draft_id": action.get("draft_id") or draft.get("draft_id"),
        "recommendation_id": action.get("recommendation_id") or draft.get("recommendation_id"),
        "activation_order": action.get("activation_order") or draft.get("activation_order"),
        "apply_after": action.get("apply_after") if isinstance(action.get("apply_after"), list) else draft.get("apply_after", []),
        "activation_mode": action.get("activation_mode") or draft.get("activation_mode"),
        "policy_source": "managed-recommended",
        "required_local_review": True,
        "managed_enforced": False,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "local_action_family": family,
        "policy_section": family,
        "local_executor_family": action.get("local_executor_family") or draft.get("local_executor_family"),
        "target_local_rule_file": target_rule_file,
        "confidence": action.get("confidence") or draft.get("confidence"),
        "local_policy_draft": local_policy_draft,
        "expected_savings": action.get("expected_savings") if isinstance(action.get("expected_savings"), dict) else draft.get("expected_savings", {}),
        "required_coverage": action.get("required_coverage") if isinstance(action.get("required_coverage"), dict) else draft.get("required_coverage", {}),
        "rollback_criteria": action.get("rollback_criteria") if isinstance(action.get("rollback_criteria"), dict) else draft.get("rollback_criteria", {}),
        "keep_staged_criteria": action.get("keep_staged_criteria") if isinstance(action.get("keep_staged_criteria"), dict) else draft.get("keep_staged_criteria", {}),
        "risk_summary": action.get("risk_summary") if isinstance(action.get("risk_summary"), dict) else draft.get("risk_summary", {}),
        "candidate_bucket": action.get("candidate_bucket") if isinstance(action.get("candidate_bucket"), dict) else draft.get("candidate_bucket", {}),
        "provenance": action.get("provenance") if isinstance(action.get("provenance"), dict) else draft.get("provenance", {}),
        "source_bundle_provenance": bundle.get("provenance") if isinstance(bundle.get("provenance"), dict) else {},
        "privacy": _privacy(),
    }
    return {key: value for key, value in entry.items() if value not in (None, "", [])}


async def stage_managed_activation_bundle(
    bundle: Any,
    *,
    workspace: str | Path | None = None,
    config_dir: str | Path | None = None,
    db_path: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return _error_result("invalid-bundle", "managed activation bundle must be a JSON object")
    if bundle.get("schema") != SUPPORTED_INPUT_SCHEMA:
        return _error_result("unsupported-bundle-schema", f"expected {SUPPORTED_INPUT_SCHEMA}")
    if bundle.get("status") not in (None, "review-only"):
        return _error_result("unsupported-bundle-status", "only review-only activation bundles can be staged")

    content_errors = _content_errors(bundle)
    if content_errors:
        return _error_result(
            "content-bearing-bundle-rejected",
            "managed activation bundle contains raw content, identifiers, cache keys, tool payloads, secrets, tenant IDs, or local paths",
            errors=content_errors,
        )

    actions = bundle.get("local_actions")
    if not isinstance(actions, list):
        return _error_result("invalid-local-actions", "managed activation bundle must include a local_actions list")

    entries_by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in SUPPORTED_FAMILIES}
    skipped: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            skipped.append({
                "schema": "tokenclaw.managed_activation_bundle_import_skip.v1",
                "status": "skipped",
                "reason_code": "invalid-action",
                "reason_codes": ["invalid-action"],
                "message": "local action must be an object",
            })
            continue
        draft = _draft_for_action(bundle, action)
        reason, message = _skip_reason(bundle, action, draft)
        if reason is not None:
            skipped.append(_action_skip(action, reason, message=message))
            continue
        entry = _staged_entry(bundle, action, draft)
        entries_by_family[str(entry["local_action_family"])].append(entry)

    staged: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for family in ("cache", "crunch"):
        entries = entries_by_family[family]
        if not entries:
            continue
        payload = {
            "managed_activation_drafts": sorted(entries, key=lambda item: int(item.get("activation_order") or 0)),
        }
        draft_id = _stable_id(f"managed-activation-{family}", {
            "bundle_id": bundle.get("bundle_id"),
            "action_ids": [entry.get("action_id") for entry in payload["managed_activation_drafts"]],
        })
        stage = await stage_policy_draft(
            payload,
            section=family,
            draft_id=draft_id,
            workspace=workspace,
            metadata={
                "schema": "tokenclaw.managed_activation_bundle_import_metadata.v1",
                "source": "managed-activation-bundle-import",
                "bundle_id": bundle.get("bundle_id"),
                "policy_source": "managed-recommended",
                "target_local_rule_file": RULE_FILES[family],
                "staged_action_count": len(entries),
            },
        )
        validation = None
        if stage.get("ok") and validate:
            validation = await validate_staged_policy_draft(
                str(stage.get("draft_id")),
                workspace=workspace,
                config_dir=config_dir,
                db_path=db_path,
            )
        row = {
            "schema": "tokenclaw.managed_activation_bundle_import_staged.v1",
            "status": "staged" if stage.get("ok") else "rejected",
            "local_action_family": family,
            "policy_section": family,
            "target_local_rule_file": RULE_FILES[family],
            "policy_source": "managed-recommended",
            "draft_id": stage.get("draft_id"),
            "workspace": stage.get("workspace"),
            "staged_action_count": len(entries),
            "action_ids": [entry.get("action_id") for entry in entries],
            "stage": stage,
            "validation": validation,
            "validation_passed": bool((validation or {}).get("validation", {}).get("ok", stage.get("ok"))),
            "wrote_active_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        }
        if stage.get("ok"):
            staged.append(row)
        else:
            rejected.append(row)

    omitted = bundle.get("omitted_actions") if isinstance(bundle.get("omitted_actions"), list) else []
    ok = bool(staged) and not rejected
    status = "staged" if ok else "partial" if staged else "rejected"
    return {
        "schema": SCHEMA,
        "ok": ok,
        "status": status,
        "generated_at": utc_now(),
        "dry_run": True,
        "bundle_id": bundle.get("bundle_id"),
        "source_schema": bundle.get("schema"),
        "summary": {
            "action_count": len(actions),
            "staged_count": len(staged),
            "skipped_count": len(skipped),
            "omitted_count": len(omitted),
            "rejected_count": len(rejected),
            "staged_action_count": sum(int(row.get("staged_action_count") or 0) for row in staged),
            "target_local_rule_files": sorted({row["target_local_rule_file"] for row in staged}),
            "policy_source": "managed-recommended" if staged else "local-default",
            "required_local_review": True,
            "managed_enforced": False,
            "provider_forwarding": False,
            "server_content_processing": False,
        },
        "staged": staged,
        "skipped": skipped,
        "omitted": omitted,
        "rejected": rejected,
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "error": None if staged else {"type": "no-stageable-actions", "message": "bundle did not contain any stageable cache or crunch actions"},
    }


def stage_managed_activation_bundle_sync(
    bundle: Any,
    *,
    workspace: str | Path | None = None,
    config_dir: str | Path | None = None,
    db_path: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    return asyncio.run(stage_managed_activation_bundle(
        bundle,
        workspace=workspace,
        config_dir=config_dir,
        db_path=db_path,
        validate=validate,
    ))
