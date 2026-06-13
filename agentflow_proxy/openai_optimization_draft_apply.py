from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentflow_proxy.openai_optimization_draft_dry_run import dry_run_openai_optimization_draft
from agentflow_proxy.openai_optimization_drafts import _scan_safety
from agentflow_proxy.openai_optimization_governor import LIFECYCLE_SOURCE_SURFACE
from agentflow_proxy.paths import default_config_dir, safe_expanduser
from agentflow_proxy.optimization_promotion_actions import ACTION_SCHEMA, SCHEMA as PROMOTION_ACTIONS_SCHEMA
from agentflow_proxy.optimization_promotion_canary import apply_optimization_promotion_canaries
from agentflow_proxy.policy_bundle import validate_policy_bundle
from agentflow_proxy.policy_events import log_policy_event
from agentflow_proxy.policy_workbench import load_staged_policy_draft
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_optimization_draft_apply.v1"
LIFECYCLE_SCHEMA = "agentflow.openai_optimization_draft_apply_lifecycle_feedback.v1"
SUPPORTED_SECTIONS = ("routing", "crunch", "cache")
PRIVACY = {
    "metadata_only": True,
    "local_only": True,
    "raw_prompts_included": False,
    "raw_messages_included": False,
    "raw_request_bodies_included": False,
    "raw_responses_included": False,
    "provider_bodies_included": False,
    "tool_payloads_included": False,
    "file_paths_included": False,
    "request_ids_included": False,
    "session_ids_included": False,
    "cache_keys_included": False,
    "provider_calls_made": False,
    "managed_server_calls_made": False,
}


def _apply_id(draft_id: str | None) -> str:
    safe = "".join(char for char in str(draft_id or "openai-draft") if char.isalnum() or char in {"-", "_"}).strip("-_")
    safe = (safe or "openai-draft")[:48]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{safe}"


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _as_action(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"old_context_summary", "old_context_summarization", "summary", "summarization"}:
        return "old_context_summarization"
    if text in {"cache", "cache_replay"}:
        return "cache"
    return text


def _action_index(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    metadata = (manifest or {}).get("metadata") if isinstance((manifest or {}).get("metadata"), dict) else {}
    review = metadata.get("openai_optimization_review") if isinstance(metadata.get("openai_optimization_review"), dict) else {}
    indexed: dict[str, dict[str, Any]] = {}
    for item in review.get("selected_actions") if isinstance(review.get("selected_actions"), list) else []:
        if not isinstance(item, dict):
            continue
        for key in ("target_candidate_id", "candidate_id", "policy_id", "action_id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                indexed[value.strip()] = item
    return indexed


def _selected_families(manifest: dict[str, Any] | None) -> set[str]:
    metadata = (manifest or {}).get("metadata") if isinstance((manifest or {}).get("metadata"), dict) else {}
    review = metadata.get("openai_optimization_review") if isinstance(metadata.get("openai_optimization_review"), dict) else {}
    families: set[str] = set()
    for item in review.get("selected_actions") if isinstance(review.get("selected_actions"), list) else []:
        if isinstance(item, dict):
            families.add(_as_action(item.get("action_family")))
    return families


def _indexed_action(indexed: dict[str, dict[str, Any]], *ids: Any) -> dict[str, Any]:
    for value in ids:
        if isinstance(value, str) and value.strip() and value.strip() in indexed:
            return indexed[value.strip()]
    return {}


def _action_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _routing_action(bundle: dict[str, Any], indexed: dict[str, dict[str, Any]], *, canary_fraction: float, holdout_fraction: float) -> dict[str, Any] | None:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    routing = policies.get("routing") if isinstance(policies.get("routing"), dict) else {}
    openai = routing.get("openai") if isinstance(routing.get("openai"), dict) else {}
    canary = openai.get("canary") if isinstance(openai.get("canary"), dict) else routing.get("openai_canary")
    if not isinstance(canary, dict):
        return None
    meta = canary.get("managed_recommendation") if isinstance(canary.get("managed_recommendation"), dict) else {}
    selected = _indexed_action(indexed, canary.get("target_candidate_id"), canary.get("candidate_id"), canary.get("policy_id"), meta.get("target_candidate_id"))
    evidence = selected.get("evidence_freshness") if isinstance(selected.get("evidence_freshness"), dict) else meta.get("evidence_freshness")
    update = {key: value for key, value in canary.items() if key not in {"enabled", "canary_fraction", "holdout_fraction"}}
    update.setdefault("source_surface", "openai_responses")
    update.setdefault("provider_endpoint", "responses")
    update["canary"] = {
        "enabled": True,
        "fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "salt": canary.get("salt") or canary.get("target_candidate_id") or canary.get("policy_id"),
        "unit": canary.get("unit") or "request_fingerprint",
    }
    return {
        "schema": ACTION_SCHEMA,
        "action_id": selected.get("action_id") or _action_id("openai-optimization-draft-routing", update),
        "action_type": "widen",
        "action_family": "routing",
        "candidate_family": selected.get("candidate_family") or "provider-routing-rule",
        "policy_section": "routing",
        "source_surface": "openai_responses",
        "provider_endpoint": "responses",
        "provider_family": "openai",
        "target_candidate_id": canary.get("target_candidate_id") or selected.get("target_candidate_id") or canary.get("policy_id"),
        "target_rule_id": canary.get("policy_id") or selected.get("target_rule_id"),
        "candidate_target_model": canary.get("target_model"),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "policy_source": "managed-recommended",
        "managed_enforced": False,
        "local_policy_update": update,
        "evidence_freshness": evidence if isinstance(evidence, dict) else {},
    }


def _summary_action(bundle: dict[str, Any], indexed: dict[str, dict[str, Any]], *, canary_fraction: float, holdout_fraction: float) -> dict[str, Any] | None:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    crunch = policies.get("crunch") if isinstance(policies.get("crunch"), dict) else {}
    summary = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else None
    if not isinstance(summary, dict):
        return None
    meta = summary.get("managed_recommendation") if isinstance(summary.get("managed_recommendation"), dict) else {}
    selected = _indexed_action(indexed, summary.get("candidate_id"), summary.get("rule_id"), summary.get("promotion_action_id"))
    evidence = selected.get("evidence_freshness") if isinstance(selected.get("evidence_freshness"), dict) else meta.get("evidence_freshness")
    update = dict(summary)
    update["canary"] = {
        **(summary.get("canary") if isinstance(summary.get("canary"), dict) else {}),
        "enabled": True,
        "fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "salt": ((summary.get("canary") or {}).get("salt") if isinstance(summary.get("canary"), dict) else None) or summary.get("candidate_id") or summary.get("rule_id"),
        "unit": ((summary.get("canary") or {}).get("unit") if isinstance(summary.get("canary"), dict) else None) or "source_hash",
    }
    return {
        "schema": ACTION_SCHEMA,
        "action_id": selected.get("action_id") or _action_id("openai-optimization-draft-summary", update),
        "action_type": "widen",
        "action_family": "old_context_summarization",
        "candidate_family": selected.get("candidate_family") or "old-context-summary-policy-rule",
        "policy_section": "crunch",
        "target_candidate_id": summary.get("candidate_id") or selected.get("target_candidate_id"),
        "target_rule_id": summary.get("rule_id") or selected.get("target_rule_id"),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "policy_source": "managed-recommended",
        "managed_enforced": False,
        "local_policy_update": {"old_context_summarization": update, "canary": update["canary"]},
        "evidence_freshness": evidence if isinstance(evidence, dict) else {},
    }


def _cache_actions(bundle: dict[str, Any], indexed: dict[str, dict[str, Any]], *, canary_fraction: float, holdout_fraction: float) -> list[dict[str, Any]]:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    cache = policies.get("cache") if isinstance(policies.get("cache"), dict) else {}
    rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), list) else []
    actions: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        meta = rule.get("managed_recommendation") if isinstance(rule.get("managed_recommendation"), dict) else {}
        selected = _indexed_action(indexed, rule.get("candidate_id"), rule.get("id"), rule.get("promotion_action_id"))
        evidence = selected.get("evidence_freshness") if isinstance(selected.get("evidence_freshness"), dict) else meta.get("evidence_freshness")
        canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
        local_update = {
            "conditions": rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {},
            "action": rule.get("action") if isinstance(rule.get("action"), dict) else {},
            "canary": {
                "enabled": True,
                "fraction": canary_fraction,
                "holdout_fraction": holdout_fraction,
                "salt": canary.get("salt") or rule.get("candidate_id") or rule.get("id"),
                "unit": canary.get("unit") or "request_fingerprint",
            },
        }
        actions.append({
            "schema": ACTION_SCHEMA,
            "action_id": selected.get("action_id") or _action_id("openai-optimization-draft-cache", rule),
            "action_type": "widen",
            "action_family": "cache",
            "candidate_family": selected.get("candidate_family") or "cache-replay-policy-rule",
            "policy_section": "cache",
            "target_candidate_id": rule.get("candidate_id") or selected.get("target_candidate_id"),
            "target_rule_id": rule.get("id") or selected.get("target_rule_id"),
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "policy_source": "managed-recommended",
            "managed_enforced": False,
            "local_policy_update": local_update,
            "evidence_freshness": evidence if isinstance(evidence, dict) else {},
        })
    return actions


def _promotion_bundle(loaded: dict[str, Any], *, canary_fraction: float, holdout_fraction: float) -> dict[str, Any]:
    bundle = loaded["bundle"]
    manifest = loaded.get("manifest") if isinstance(loaded.get("manifest"), dict) else None
    indexed = _action_index(manifest)
    selected_families = _selected_families(manifest)
    actions = []
    if "routing" in selected_families:
        action = _routing_action(bundle, indexed, canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
        if action is not None:
            actions.append(action)
    if "old_context_summarization" in selected_families:
        action = _summary_action(bundle, indexed, canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
        if action is not None:
            actions.append(action)
    if "cache" in selected_families:
        actions.extend(_cache_actions(bundle, indexed, canary_fraction=canary_fraction, holdout_fraction=holdout_fraction))
    return {
        "schema": PROMOTION_ACTIONS_SCHEMA,
        "generated_at": utc_now(),
        "source_schema": bundle.get("schema"),
        "source": "openai_optimization_staged_draft",
        "actions": actions,
        "omitted": [],
        "privacy": {
            "metadata_only": True,
            "feature_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
        },
    }


def _block_result(
    *,
    draft: str,
    draft_id: str,
    apply_id: str,
    config_dir: str | Path,
    dry_run: bool,
    validation: dict[str, Any] | None,
    dry_run_projection: dict[str, Any] | None,
    error: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "blocked",
        "dry_run": bool(dry_run),
        "draft_id": draft_id or draft,
        "apply_id": apply_id,
        "backup_id": apply_id,
        "config_dir": str(Path(config_dir).expanduser()),
        "requested_sections": list(SUPPORTED_SECTIONS),
        "applied_sections": [],
        "changed_sections": [],
        "files": [],
        "backups": [],
        "actions": [],
        "summary": {"planned_action_count": 0, "changed_file_count": 0},
        "validation": validation,
        "dry_run_projection": dry_run_projection,
        "rollback_command": None,
        "rollback": None,
        "feedback": None,
        "privacy": {**PRIVACY, "active_policy_files_written": False},
        "error": error,
    }


def _validation_errors(
    loaded: dict[str, Any],
    *,
    require_verified_provenance: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    bundle = loaded.get("bundle")
    validation = validate_policy_bundle(bundle)
    errors: list[dict[str, str]] = []
    if not validation.get("ok"):
        errors.extend(error for error in validation.get("errors", []) if isinstance(error, dict))
    raw_errors: list[dict[str, str]] = []
    _scan_safety(bundle, raw_errors)
    errors.extend(raw_errors)
    provenance = validation.get("provenance") if isinstance(validation.get("provenance"), dict) else {}
    if require_verified_provenance and provenance.get("status") != "verified":
        errors.append({
            "path": "$.provenance",
            "message": "verified staged OpenAI optimization draft provenance is required before apply",
        })
    metadata = (loaded.get("manifest") or {}).get("metadata") if isinstance((loaded.get("manifest") or {}).get("metadata"), dict) else {}
    if "openai_optimization_review" not in metadata:
        errors.append({
            "path": "$.draft.metadata.openai_optimization_review",
            "message": "staged draft does not include OpenAI optimization review metadata",
        })
    else:
        review = metadata.get("openai_optimization_review") if isinstance(metadata.get("openai_optimization_review"), dict) else {}
        selected = review.get("selected_actions") if isinstance(review.get("selected_actions"), list) else []
        if not selected:
            errors.append({
                "path": "$.draft.metadata.openai_optimization_review.selected_actions",
                "message": "expected at least one selected OpenAI optimization action",
            })
        for index, action in enumerate(selected):
            family = _as_action(action.get("action_family")) if isinstance(action, dict) else ""
            if family not in {"routing", "old_context_summarization", "cache"}:
                errors.append({
                    "path": f"$.draft.metadata.openai_optimization_review.selected_actions[{index}].action_family",
                    "message": f"unsupported OpenAI optimization action family: {family or 'unknown'}",
                })
    return validation, errors


def _scan_apply_blockers(value: Any, errors: list[dict[str, str]], path: str = "$") -> None:
    if isinstance(value, dict):
        if isinstance(value.get("evidence_freshness"), dict) and value["evidence_freshness"].get("stale") is True:
            errors.append({
                "path": f"{path}.evidence_freshness",
                "message": "staged OpenAI optimization action has stale local evidence",
            })
        if isinstance(value.get("safety_stop"), dict) and value["safety_stop"].get("tripped") is True:
            errors.append({
                "path": f"{path}.safety_stop",
                "message": "staged OpenAI optimization action is suppressed by a local safety stop",
            })
        for key, item in value.items():
            _scan_apply_blockers(item, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_apply_blockers(item, errors, f"{path}[{index}]")


def _event_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "draft_id": result.get("draft_id"),
        "apply_id": result.get("apply_id"),
        "backup_id": result.get("backup_id"),
        "config_dir": result.get("config_dir"),
        "status": result.get("status"),
        "dry_run": result.get("dry_run"),
        "requested_sections": result.get("requested_sections", []),
        "applied_sections": result.get("applied_sections", []),
        "changed_sections": result.get("changed_sections", []),
        "changed_files": [file.get("path") for file in result.get("files", []) if isinstance(file, dict) and file.get("changed")],
        "backup_paths": [backup.get("path") for backup in result.get("backups", []) if isinstance(backup, dict) and backup.get("path")],
        "planned_action_count": (result.get("summary") or {}).get("planned_action_count") if isinstance(result.get("summary"), dict) else None,
        "rollback_command": result.get("rollback_command"),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "active_policy_files_written": bool((result.get("privacy") or {}).get("active_policy_files_written")) if isinstance(result.get("privacy"), dict) else False,
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "exit_code": 0 if result.get("ok") else 1,
    }


def _feedback_event(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": LIFECYCLE_SCHEMA,
        "event_type": "openai_optimization_draft_apply",
        "occurred_at": result.get("generated_at") or utc_now(),
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "lifecycle_phase": "dry_run" if result.get("dry_run") else ("apply" if result.get("ok") else "rejected"),
        "draft_id": result.get("draft_id"),
        "apply_id": result.get("apply_id"),
        "status": result.get("status"),
        "summary": result.get("summary"),
        "actions": [
            {
                "action_family": action.get("action_family"),
                "policy_section": action.get("policy_section"),
                "status": action.get("status"),
                "reason": action.get("reason"),
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "canary_fraction": action.get("canary_fraction"),
                "holdout_fraction": action.get("holdout_fraction"),
            }
            for action in result.get("actions", [])
            if isinstance(action, dict)
        ],
        "changed_sections": result.get("changed_sections", []),
        "privacy": result.get("privacy") or PRIVACY,
    }


async def _queue_feedback(store_obj: Any, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy.recommendations import queue_policy_event_feedback

    meta = await queue_policy_event_feedback(
        store_obj,
        _feedback_event(result),
        source_surface=LIFECYCLE_SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=False,
    )
    return {
        "schema": "agentflow.openai_optimization_draft_apply_feedback_queue.v1",
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "endpoint": meta.get("endpoint"),
        "queue_id": meta.get("queue_id"),
        "attempts": meta.get("attempts"),
        "payload_included": False,
    }


async def apply_openai_optimization_draft(
    draft: str,
    *,
    workspace: str | Path | None = None,
    config_dir: str | Path | None = None,
    store_obj: Any | None = None,
    dry_run: bool = True,
    canary_fraction: float = 0.10,
    holdout_fraction: float = 0.10,
    impact_limit: int = 1000,
    sections: list[str] | tuple[str, ...] | None = None,
    require_verified_provenance: bool = False,
    queue_feedback: bool = False,
    apply_id: str | None = None,
) -> dict[str, Any]:
    config_path = safe_expanduser(config_dir) if config_dir is not None else default_config_dir()
    loaded = load_staged_policy_draft(draft, workspace=workspace)
    manifest = loaded.get("manifest") if isinstance(loaded.get("manifest"), dict) else {}
    draft_id = str(manifest.get("draft_id") or draft)
    transaction_id = apply_id or _apply_id(draft_id)
    canary_fraction = _as_float(canary_fraction, 0.10)
    holdout_fraction = _as_float(holdout_fraction, 0.10)

    if not loaded.get("ok"):
        result = _block_result(
            draft=draft,
            draft_id=draft_id,
            apply_id=transaction_id,
            config_dir=config_path,
            dry_run=dry_run,
            validation=None,
            dry_run_projection=None,
            error=loaded.get("error") if isinstance(loaded.get("error"), dict) else {"type": "draft_not_found", "message": "staged draft could not be loaded"},
        )
        log_policy_event("draft-apply", ok=False, details={"source": "openai-optimization-draft-apply", **_event_details(result)})
        return result

    validation, validation_errors = _validation_errors(loaded, require_verified_provenance=require_verified_provenance)
    _scan_apply_blockers(loaded.get("bundle"), validation_errors)
    projection: dict[str, Any] | None = None
    if store_obj is not None:
        projection = await dry_run_openai_optimization_draft(
            draft,
            workspace=workspace,
            store_obj=store_obj,
            limit=impact_limit,
            canary_fraction=canary_fraction,
            holdout_fraction=holdout_fraction,
            queue_feedback=False,
        )
        projection_summary = projection.get("summary") if isinstance(projection.get("summary"), dict) else {}
        if projection.get("ok") and (
            (projection_summary.get("conflict_total") or 0) > 0
            or (projection_summary.get("suppressed_total") or 0) > 0
        ):
            validation_errors.append({
                "path": "$.openai_optimization_governor",
                "message": "staged OpenAI optimization actions conflict under the local governor projection",
            })
    if validation_errors:
        result = _block_result(
            draft=draft,
            draft_id=draft_id,
            apply_id=transaction_id,
            config_dir=config_path,
            dry_run=dry_run,
            validation=validation,
            dry_run_projection=projection,
            error={
                "type": "validation_failed",
                "message": "staged OpenAI optimization draft is not safe to apply",
                "errors": validation_errors,
            },
        )
        if queue_feedback and store_obj is not None:
            result["feedback"] = await _queue_feedback(store_obj, result)
        log_policy_event("draft-apply", ok=False, details={"source": "openai-optimization-draft-apply", **_event_details(result)})
        return result

    promotion = _promotion_bundle(loaded, canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
    requested_sections = [section for section in SUPPORTED_SECTIONS if section in set(sections or SUPPORTED_SECTIONS)]
    promotion_result = apply_optimization_promotion_canaries(
        promotion,
        config_dir=config_path,
        dry_run=dry_run,
        sections=requested_sections,
        backup_id=transaction_id,
    )
    files = promotion_result.get("files") if isinstance(promotion_result.get("files"), list) else []
    changed_files = [file for file in files if isinstance(file, dict) and file.get("changed")]
    changed_sections = [str(file.get("section")) for file in changed_files if file.get("section")]
    backups = [
        {
            "section": file.get("section"),
            "backup_id": transaction_id,
            "path": file.get("backup_path"),
        }
        for file in changed_files
        if file.get("backup_path")
    ]
    rollback_command = None
    if changed_sections and not dry_run:
        rollback_command = " ".join(
            ["agentflow-policy-rollback", "--config-dir", str(config_path), "--apply-id", transaction_id]
            + [part for section in changed_sections for part in ("--section", section)]
        )
    result = {
        "schema": SCHEMA,
        "ok": bool(promotion_result.get("ok")),
        "status": "dry-run" if dry_run and promotion_result.get("ok") else ("applied" if promotion_result.get("ok") else "blocked"),
        "generated_at": utc_now(),
        "dry_run": bool(dry_run),
        "draft_id": draft_id,
        "apply_id": transaction_id,
        "backup_id": transaction_id,
        "config_dir": str(config_path),
        "requested_sections": requested_sections,
        "applied_sections": changed_sections if not dry_run else [],
        "changed_sections": changed_sections,
        "files": files,
        "backups": backups,
        "actions": promotion_result.get("actions", []),
        "summary": {
            **(promotion_result.get("summary") if isinstance(promotion_result.get("summary"), dict) else {}),
            "changed_file_count": len(changed_files),
            "active_policy_files_written": bool(not dry_run and changed_files),
        },
        "validation": validation,
        "dry_run_projection": projection,
        "rollback_command": rollback_command,
        "rollback": {
            "backup_id": transaction_id,
            "command": rollback_command,
            "backup_paths": [backup["path"] for backup in backups if backup.get("path")],
        } if rollback_command else None,
        "feedback": None,
        "privacy": {
            **PRIVACY,
            "active_policy_files_written": bool(not dry_run and changed_files),
            "loopback_admin_calls_made": False,
        },
        "error": None if promotion_result.get("ok") else {
            "type": "apply_failed",
            "message": "staged OpenAI optimization draft canary apply failed",
            "errors": promotion_result.get("errors", []),
        },
    }
    if queue_feedback and store_obj is not None:
        result["feedback"] = await _queue_feedback(store_obj, result)
    log_policy_event("draft-apply", ok=bool(result.get("ok")) and not dry_run, details={"source": "openai-optimization-draft-apply", **_event_details(result)})
    return result
