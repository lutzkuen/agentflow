from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from tokenclaw.managed_activation_bundles import _content_errors
from tokenclaw.paths import default_config_dir, safe_expanduser
from tokenclaw.policy_events import log_policy_event
from tokenclaw.policy_workbench import load_staged_policy_draft
from tokenclaw.store import utc_now


SCHEMA = "agentflow.managed_activation_bundle_apply.v1"
APPLY_METADATA_SCHEMA = "agentflow.managed_activation_rule_apply_metadata.v1"
SUPPORTED_FAMILIES = ("cache", "crunch")
RULE_FILES = {
    "cache": "cache_rules.yaml",
    "crunch": "crunch_rules.yaml",
}
TARGET_PATHS = {
    "cache": ("pattern_rules", "cache.pattern_rules"),
    "crunch": ("pattern_rules", "anthropic_thinking_history_compaction.rules", "crunch.rules"),
}


def _privacy(*, wrote: bool, dry_run: bool) -> dict[str, Any]:
    return {
        "local_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "feature_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "dry_run": bool(dry_run),
        "policy_files_written": bool(wrote),
        "wrote_local_policy_files": bool(wrote),
    }


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_apply_id(draft_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in draft_id).strip("-_")
    safe = (safe or "managed-activation")[:48]
    stamp = utc_now().replace(":", "").replace("+", "Z").replace(".", "")
    return f"{stamp}-{safe}"


def _backup_path(path: Path, apply_id: str) -> Path:
    return path.with_name(f"{path.name}.bak-{apply_id}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_text(path: Path) -> tuple[str | None, bool]:
    try:
        return path.read_text(encoding="utf-8"), True
    except FileNotFoundError:
        return None, False


def _load_yaml_file(path: Path) -> dict[str, Any]:
    raw, exists = _read_text(path)
    if not exists or raw is None or not raw.strip():
        return {}
    parsed = yaml.safe_load(raw)
    return parsed if isinstance(parsed, dict) else {}


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _safe_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned or fallback)[:120]


def _selectors_for_entry(entry: dict[str, Any]) -> set[str]:
    values = {
        entry.get("action_id"),
        entry.get("draft_id"),
        entry.get("recommendation_id"),
    }
    draft = entry.get("local_policy_draft") if isinstance(entry.get("local_policy_draft"), dict) else {}
    values.update({
        draft.get("id"),
        draft.get("rule_id"),
        draft.get("candidate_id"),
        draft.get("action_id"),
        draft.get("draft_id"),
        draft.get("recommendation_id"),
    })
    return {str(value).strip() for value in values if isinstance(value, str) and value.strip()}


def _rule_matches(rule: Any, selectors: set[str]) -> bool:
    if not isinstance(rule, dict):
        return False
    for key in ("id", "rule_id", "candidate_id", "action_id", "draft_id", "recommendation_id"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip() in selectors:
            return True
    meta = rule.get("managed_activation_apply") if isinstance(rule.get("managed_activation_apply"), dict) else {}
    for key in ("action_id", "draft_id", "recommendation_id"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip() in selectors:
            return True
    return False


def _rule_list(payload: dict[str, Any], family: str, target_path: str) -> tuple[list[Any] | None, str]:
    if family == "cache":
        rules = payload.setdefault("pattern_rules", [])
        return (rules if isinstance(rules, list) else None), "$.pattern_rules"
    if target_path == "anthropic_thinking_history_compaction.rules" or target_path == "crunch.rules":
        compaction = payload.setdefault("anthropic_thinking_history_compaction", {})
        if not isinstance(compaction, dict):
            return None, "$.anthropic_thinking_history_compaction"
        rules = compaction.setdefault("rules", [])
        return (rules if isinstance(rules, list) else None), "$.anthropic_thinking_history_compaction.rules"
    rules = payload.setdefault("pattern_rules", [])
    return (rules if isinstance(rules, list) else None), "$.pattern_rules"


def _target_path(entry: dict[str, Any], local_draft: dict[str, Any]) -> str:
    for source in (
        local_draft,
        entry.get("candidate_bucket") if isinstance(entry.get("candidate_bucket"), dict) else {},
    ):
        value = source.get("target_local_policy_section") or source.get("policy_section_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    family = str(entry.get("local_action_family") or "")
    if family == "cache":
        return "pattern_rules"
    return "anthropic_thinking_history_compaction.rules"


def _rules_from_local_draft(entry: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], dict[str, Any] | None]:
    local_draft = entry.get("local_policy_draft")
    if not isinstance(local_draft, dict):
        return None, [], {"type": "missing-local-policy-draft", "message": "managed activation entry has no local_policy_draft object"}

    family = str(entry.get("local_action_family") or "").strip()
    target_path = _target_path(entry, local_draft)
    allowed = TARGET_PATHS.get(family, ())
    if target_path not in allowed:
        return None, [], {
            "type": "unsupported-target-policy-section",
            "message": "managed activation draft targets an unsupported local policy section",
            "target_policy_section": target_path,
        }

    patch = local_draft.get("local_policy_patch")
    if not isinstance(patch, dict):
        patch = local_draft.get("policy_patch")
    if not isinstance(patch, dict):
        patch = local_draft

    candidates: list[Any] = []
    for key in ("rule", "pattern_rule", "cache_rule", "crunch_rule", "anthropic_thinking_history_compaction_rule"):
        if isinstance(local_draft.get(key), dict):
            candidates.append(local_draft[key])
    if family == "cache":
        if isinstance(patch.get("pattern_rules"), list):
            candidates.extend(patch["pattern_rules"])
    else:
        if isinstance(patch.get("pattern_rules"), list) and target_path == "pattern_rules":
            candidates.extend(patch["pattern_rules"])
        compaction = patch.get("anthropic_thinking_history_compaction")
        if isinstance(compaction, dict) and isinstance(compaction.get("rules"), list):
            candidates.extend(compaction["rules"])
    if not candidates and isinstance(patch.get("conditions"), dict) and isinstance(patch.get("action"), dict):
        candidates.append({key: patch[key] for key in ("id", "rule_id", "candidate_id", "conditions", "action", "canary", "rollout", "safety_stop", "enabled") if key in patch})

    rules = [copy.deepcopy(rule) for rule in candidates if isinstance(rule, dict)]
    if not rules:
        return None, [], {
            "type": "missing-executable-rule",
            "message": "managed activation draft must include an explicit feature-only local rule patch before it can be applied",
        }
    return target_path, rules, None


def _entry_blocker(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return {"type": "invalid-entry", "message": "managed activation draft entry must be an object"}
    content_errors = _content_errors(entry, path="$")
    if content_errors:
        return {
            "type": "content-bearing-draft-rejected",
            "message": "managed activation drafts may not contain raw prompts, messages, provider bodies, identifiers, cache keys, tool payloads, tenant IDs, or local paths",
            "errors": content_errors,
        }
    family = str(entry.get("local_action_family") or "").strip()
    if family not in SUPPORTED_FAMILIES:
        return {"type": "unsupported-local-action-family", "message": "only cache and crunch managed activation drafts can be applied"}
    if entry.get("policy_section") not in (None, family):
        return {"type": "policy-section-mismatch", "message": "policy_section must match local_action_family"}
    if entry.get("target_local_rule_file") != RULE_FILES[family]:
        return {
            "type": "unsupported-target-rule-file",
            "message": "managed activation draft targets an unsupported local rule file",
            "target_local_rule_file": entry.get("target_local_rule_file"),
        }
    if entry.get("status") not in (None, "review-required"):
        return {"type": "draft-not-review-required", "message": "managed activation draft is not in review-required status"}
    if entry.get("policy_source") not in (None, "managed-recommended"):
        return {"type": "unsupported-policy-source", "message": "only managed-recommended drafts can be applied"}
    if entry.get("managed_enforced") is True:
        return {"type": "managed-enforced-rejected", "message": "managed-enforced drafts cannot be applied by the local review command"}
    if entry.get("provider_forwarding") is True or entry.get("server_content_processing") is True:
        return {"type": "non-local-action-boundary", "message": "draft would require provider forwarding or server content processing"}
    if entry.get("feature_only") is False or entry.get("locally_executed") is False:
        return {"type": "not-feature-only", "message": "draft must be feature-only and locally executed"}
    return None


def _annotate_rule(rule: dict[str, Any], entry: dict[str, Any], *, apply_id: str, previous_rule: dict[str, Any] | None) -> dict[str, Any]:
    annotated = copy.deepcopy(rule)
    fallback = _safe_id(entry.get("draft_id") or entry.get("action_id"), "managed-activation-rule")
    annotated.setdefault("id", fallback)
    annotated.setdefault("candidate_id", _safe_id(entry.get("recommendation_id") or entry.get("draft_id"), fallback))
    annotated["enabled"] = bool(annotated.get("enabled", True))
    annotated["policy_source"] = "managed-recommended"
    annotated["action_id"] = entry.get("action_id")
    annotated["draft_id"] = entry.get("draft_id")
    annotated["recommendation_id"] = entry.get("recommendation_id")
    annotated["managed_activation_apply"] = {
        "schema": APPLY_METADATA_SCHEMA,
        "applied_at": utc_now(),
        "apply_id": apply_id,
        "bundle_id": entry.get("bundle_id"),
        "action_id": entry.get("action_id"),
        "draft_id": entry.get("draft_id"),
        "recommendation_id": entry.get("recommendation_id"),
        "activation_mode": entry.get("activation_mode"),
        "policy_source": "managed-recommended",
        "rollback_ready": True,
        "previous_rule": _json_clone(previous_rule) if previous_rule is not None else None,
        "expected_savings": _json_clone(entry.get("expected_savings") if isinstance(entry.get("expected_savings"), dict) else {}),
        "required_coverage": _json_clone(entry.get("required_coverage") if isinstance(entry.get("required_coverage"), dict) else {}),
        "rollback_criteria": _json_clone(entry.get("rollback_criteria") if isinstance(entry.get("rollback_criteria"), dict) else {}),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    return annotated


def _mutate_payload(payload: dict[str, Any], entry: dict[str, Any], *, apply_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    target_path, rules, error = _rules_from_local_draft(entry)
    if error is not None or target_path is None:
        return None, [], error
    family = str(entry.get("local_action_family") or "").strip()
    mutated = copy.deepcopy(payload)
    active_rules, rules_path = _rule_list(mutated, family, target_path)
    if active_rules is None:
        return None, [], {"type": "invalid-rule-file", "message": f"{rules_path} must be a list"}

    selectors = _selectors_for_entry(entry)
    applied_rules: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule.get("conditions"), dict):
            return None, [], {"type": "invalid-rule", "message": "managed activation rule requires conditions object"}
        if not isinstance(rule.get("action"), dict):
            return None, [], {"type": "invalid-rule", "message": "managed activation rule requires action object"}
        index = next((idx for idx, existing in enumerate(active_rules) if _rule_matches(existing, selectors | _selectors_for_entry({"local_policy_draft": rule}))), None)
        previous = copy.deepcopy(active_rules[index]) if index is not None and isinstance(active_rules[index], dict) else None
        annotated = _annotate_rule(rule, entry, apply_id=apply_id, previous_rule=previous)
        if index is None:
            active_rules.append(annotated)
        else:
            active_rules[index] = annotated
        applied_rules.append({
            "rule_id": annotated.get("id"),
            "candidate_id": annotated.get("candidate_id"),
            "target_policy_section": target_path,
            "replaced_existing": previous is not None,
        })
    return mutated, applied_rules, None


def _selected(entry: dict[str, Any], selectors: set[str]) -> bool:
    if not selectors:
        return True
    values = _selectors_for_entry(entry)
    return bool(values & selectors)


def _entries_from_bundle(bundle: dict[str, Any], selectors: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for family in SUPPORTED_FAMILIES:
        policy = policies.get(family) if isinstance(policies.get(family), dict) else {}
        entries = policy.get("managed_activation_drafts")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                skipped.append({"status": "skipped", "reason": "invalid-entry"})
                continue
            if _selected(entry, selectors):
                selected.append(entry)
            else:
                skipped.append({
                    "status": "skipped",
                    "reason": "not-selected",
                    "action_id": entry.get("action_id"),
                    "draft_id": entry.get("draft_id"),
                    "recommendation_id": entry.get("recommendation_id"),
                    "local_action_family": entry.get("local_action_family"),
                })
    return selected, skipped


def _file_result(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": plan["section"],
        "path": str(plan["path"]),
        "changed": bool(plan["changed"]),
        "backup_path": str(plan["backup_path"]) if plan.get("backup_path") is not None else None,
        "sha256_before": plan.get("sha256_before"),
        "sha256_after": plan.get("sha256_after"),
        "bytes_after": plan.get("bytes_after"),
        "diff": plan.get("diff"),
    }


def _compact_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in result.get("applied") or []:
        if not isinstance(item, dict):
            continue
        entries.append({
            "status": item.get("status"),
            "local_action_family": item.get("local_action_family"),
            "action_id": item.get("action_id"),
            "draft_id": item.get("draft_id"),
            "recommendation_id": item.get("recommendation_id"),
            "target_local_rule_file": item.get("target_local_rule_file"),
        })
    for item in result.get("skipped") or []:
        if not isinstance(item, dict):
            continue
        entries.append({
            "status": "skipped",
            "local_action_family": item.get("local_action_family"),
            "action_id": item.get("action_id"),
            "draft_id": item.get("draft_id"),
            "recommendation_id": item.get("recommendation_id"),
            "reason": item.get("reason"),
        })
    for item in result.get("rejected") or []:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry") if isinstance(item.get("entry"), dict) else {}
        error = item.get("error") if isinstance(item.get("error"), dict) else {}
        entries.append({
            "status": "failed",
            "local_action_family": entry.get("local_action_family"),
            "action_id": entry.get("action_id"),
            "draft_id": entry.get("draft_id"),
            "recommendation_id": entry.get("recommendation_id"),
            "target_local_rule_file": entry.get("target_local_rule_file"),
            "reason": error.get("type"),
        })
    return entries


def _event_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "managed-activation-bundle-apply",
        "draft_id": result.get("draft_id"),
        "apply_id": result.get("apply_id"),
        "backup_id": result.get("backup_id"),
        "config_dir": result.get("config_dir"),
        "status": result.get("status"),
        "requested_sections": result.get("requested_sections", []),
        "applied_sections": result.get("applied_sections", []),
        "changed_sections": result.get("changed_sections", []),
        "changed_files": [item.get("path") for item in result.get("files", []) if isinstance(item, dict) and item.get("changed")],
        "backup_paths": [item.get("path") for item in result.get("backups", []) if isinstance(item, dict) and item.get("path")],
        "entries": _compact_entries(result),
        "reloaded_modules": False,
        "verification_ok": True,
        "validation_status": "pass" if result.get("ok") else "fail",
        "validation_can_apply": bool(result.get("ok")),
        "restored": False,
        "restore_ok": None,
        "rollback_command": result.get("rollback_command"),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "exit_code": 0 if result.get("ok") else 1,
    }


def apply_staged_managed_activation_bundle(
    draft: str,
    *,
    workspace: str | Path | None = None,
    config_dir: str | Path | None = None,
    selectors: list[str] | tuple[str, ...] | None = None,
    dry_run: bool = False,
    apply_id: str | None = None,
) -> dict[str, Any]:
    loaded = load_staged_policy_draft(draft, workspace=workspace)
    config_path = safe_expanduser(config_dir) if config_dir is not None else default_config_dir()
    transaction_id = apply_id or _stable_apply_id(str(draft))
    selector_set = {str(value).strip() for value in (selectors or []) if str(value).strip()}

    if not loaded.get("ok"):
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "blocked",
            "draft_id": draft,
            "apply_id": transaction_id,
            "backup_id": transaction_id,
            "dry_run": bool(dry_run),
            "config_dir": str(config_path),
            "requested_sections": list(SUPPORTED_FAMILIES),
            "applied_sections": [],
            "changed_sections": [],
            "files": [],
            "backups": [],
            "applied": [],
            "skipped": [],
            "rejected": [],
            "reloaded_modules": False,
            "rollback_command": None,
            "privacy": _privacy(wrote=False, dry_run=dry_run),
            "error": loaded.get("error"),
        }

    bundle = loaded["bundle"]
    entries, skipped = _entries_from_bundle(bundle, selector_set)
    if not entries:
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "blocked",
            "draft_id": str((loaded.get("manifest") or {}).get("draft_id") or draft),
            "apply_id": transaction_id,
            "backup_id": transaction_id,
            "dry_run": bool(dry_run),
            "config_dir": str(config_path),
            "requested_sections": list(SUPPORTED_FAMILIES),
            "applied_sections": [],
            "changed_sections": [],
            "files": [],
            "backups": [],
            "applied": [],
            "skipped": skipped,
            "rejected": [],
            "reloaded_modules": False,
            "rollback_command": None,
            "privacy": _privacy(wrote=False, dry_run=dry_run),
            "error": {"type": "no-selected-managed-activation-drafts", "message": "no staged managed activation draft matched the requested selection"},
        }

    payloads: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for entry in entries:
        blocker = _entry_blocker(entry)
        if blocker is not None:
            rejected.append({"entry": {key: entry.get(key) for key in ("action_id", "draft_id", "recommendation_id", "local_action_family", "target_local_rule_file")}, "error": blocker})
            continue
        family = str(entry.get("local_action_family"))
        payload = payloads.get(family)
        if payload is None:
            payload = _load_yaml_file(config_path / RULE_FILES[family])
        mutated, applied_rules, error = _mutate_payload(payload, entry, apply_id=transaction_id)
        if error is not None or mutated is None:
            rejected.append({"entry": {key: entry.get(key) for key in ("action_id", "draft_id", "recommendation_id", "local_action_family", "target_local_rule_file")}, "error": error})
            continue
        payloads[family] = mutated
        applied.append({
            "status": "applied" if not dry_run else "planned",
            "local_action_family": family,
            "action_id": entry.get("action_id"),
            "draft_id": entry.get("draft_id"),
            "recommendation_id": entry.get("recommendation_id"),
            "target_local_rule_file": RULE_FILES[family],
            "rules": applied_rules,
        })

    if rejected:
        result = {
            "schema": SCHEMA,
            "ok": False,
            "status": "blocked",
            "draft_id": str((loaded.get("manifest") or {}).get("draft_id") or draft),
            "apply_id": transaction_id,
            "backup_id": transaction_id,
            "dry_run": bool(dry_run),
            "config_dir": str(config_path),
            "requested_sections": sorted({str(entry.get("local_action_family")) for entry in entries if isinstance(entry.get("local_action_family"), str)}),
            "applied_sections": [],
            "changed_sections": [],
            "files": [],
            "backups": [],
            "applied": applied,
            "skipped": skipped,
            "rejected": rejected,
            "reloaded_modules": False,
            "rollback_command": None,
            "privacy": _privacy(wrote=False, dry_run=dry_run),
            "error": {"type": "managed-activation-apply-blocked", "message": "one or more selected managed activation drafts cannot be safely applied"},
        }
        log_policy_event("draft-apply", ok=False, details=_event_details(result))
        return result

    plans: list[dict[str, Any]] = []
    backups: list[dict[str, Any]] = []
    for family, payload in payloads.items():
        path = config_path / RULE_FILES[family]
        old_text, existed_before = _read_text(path)
        new_text = yaml.safe_dump(payload, sort_keys=False)
        changed = old_text != new_text
        backup = _backup_path(path, transaction_id) if changed and existed_before else None
        diff = "".join(difflib.unified_diff(
            (old_text or "").splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"{path.name}.before",
            tofile=f"{path.name}.after",
        ))
        plans.append({
            "section": family,
            "path": path,
            "old_text": old_text,
            "new_text": new_text,
            "existed_before": existed_before,
            "changed": changed,
            "backup_path": backup,
            "sha256_before": _sha256_text(old_text),
            "sha256_after": _sha256_text(new_text),
            "bytes_after": len(new_text.encode("utf-8")),
            "diff": diff,
        })

    write_plans = [plan for plan in plans if plan["changed"]]
    if not dry_run:
        try:
            for plan in write_plans:
                if plan.get("backup_path") is not None:
                    backup = plan["backup_path"]
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_text(plan.get("old_text") or "", encoding="utf-8")
                    backups.append({
                        "section": plan["section"],
                        "backup_id": transaction_id,
                        "path": str(backup),
                        "sha256": plan.get("sha256_before"),
                    })
            for plan in write_plans:
                _atomic_write_text(plan["path"], plan["new_text"])
        except OSError as exc:
            result = {
                "schema": SCHEMA,
                "ok": False,
                "status": "failed",
                "draft_id": str((loaded.get("manifest") or {}).get("draft_id") or draft),
                "apply_id": transaction_id,
                "backup_id": transaction_id,
                "dry_run": bool(dry_run),
                "config_dir": str(config_path),
                "requested_sections": [plan["section"] for plan in plans],
                "applied_sections": [],
                "changed_sections": [],
                "files": [_file_result(plan) for plan in plans],
                "backups": backups,
                "applied": applied,
                "skipped": skipped,
                "rejected": rejected,
                "reloaded_modules": False,
                "rollback_command": None,
                "privacy": _privacy(wrote=False, dry_run=dry_run),
                "error": {"type": "write-failed", "message": str(exc)},
            }
            log_policy_event("draft-apply", ok=False, details=_event_details(result))
            return result

    changed_sections = [plan["section"] for plan in write_plans]
    rollback_command = None
    if changed_sections and not dry_run:
        rollback_command = " ".join(
            ["agentflow-policy-rollback", "--config-dir", str(config_path), "--apply-id", transaction_id]
            + [part for section in changed_sections for part in ("--section", section)]
        )
    result = {
        "schema": SCHEMA,
        "ok": True,
        "status": "dry-run" if dry_run else "applied",
        "draft_id": str((loaded.get("manifest") or {}).get("draft_id") or draft),
        "apply_id": transaction_id,
        "backup_id": transaction_id,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "requested_sections": [plan["section"] for plan in plans],
        "applied_sections": [plan["section"] for plan in plans],
        "changed_sections": changed_sections,
        "files": [_file_result(plan) for plan in plans],
        "backups": backups,
        "applied": applied,
        "skipped": skipped,
        "rejected": rejected,
        "reloaded_modules": False,
        "verification": {"ok": True, "checks": [], "failures": []},
        "rollback_command": rollback_command,
        "rollback": {
            "backup_id": transaction_id,
            "command": rollback_command,
            "backup_paths": [backup["path"] for backup in backups],
        },
        "privacy": _privacy(wrote=bool(write_plans and not dry_run), dry_run=dry_run),
        "error": None,
    }
    log_policy_event("draft-apply", ok=bool(result.get("ok")) and not dry_run, details=_event_details(result))
    return result
