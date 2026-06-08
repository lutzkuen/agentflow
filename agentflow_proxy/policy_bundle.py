from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy import __version__
from agentflow_proxy.store import utc_now

POLICY_BUNDLE_APPLY_SCHEMA = "agentflow.policy_bundle_apply.v1"
POLICY_BUNDLE_SCHEMA = "agentflow.policy_bundle.v1"
POLICY_BUNDLE_DIFF_SCHEMA = "agentflow.policy_bundle_diff.v1"
POLICY_BUNDLE_REVIEW_SCHEMA = "agentflow.policy_bundle_review.v1"
POLICY_BUNDLE_ROLLBACK_SCHEMA = "agentflow.policy_bundle_rollback.v1"
POLICY_BUNDLE_VALIDATION_SCHEMA = "agentflow.policy_bundle_validation.v1"
POLICY_STATE_SCHEMA = "agentflow.policy_state.v1"
POLICY_SOURCES = {
    "local-default",
    "local-manual",
    "managed-recommended",
    "managed-enforced",
}
REQUIRED_POLICY_SECTIONS = (
    "routing",
    "crunch",
    "cache",
    "routing_experiments",
)
_POLICY_SECTION_FILES = {
    "routing": "routing_rules.yaml",
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
    "routing_experiments": "routing_experiments.yaml",
}


async def build_policy_bundle() -> dict[str, Any]:
    from agentflow_proxy import stats

    policy_state = await stats.stats_policies()
    return {
        "schema": POLICY_BUNDLE_SCHEMA,
        "generated_at": utc_now(),
        "generator": {
            "name": "agentflow-proxy",
            "version": __version__,
            "mode": "local-offline",
        },
        "managed_optimizer": {
            "enabled": False,
            "note": "Export only. No managed optimizer communication is performed by this command.",
        },
        "policies": policy_state,
    }


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


def validate_policy_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "bundle must be a JSON object")
        return {
            "schema": POLICY_BUNDLE_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
        }

    bundle_schema = bundle.get("schema")
    if bundle_schema != POLICY_BUNDLE_SCHEMA:
        _add_error(errors, "$.schema", f"expected {POLICY_BUNDLE_SCHEMA}")

    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")

    generator = bundle.get("generator")
    if not isinstance(generator, dict):
        _add_error(errors, "$.generator", "expected object")
    else:
        if generator.get("name") != "agentflow-proxy":
            _add_error(errors, "$.generator.name", "expected agentflow-proxy")
        if not isinstance(generator.get("version"), str) or not generator.get("version"):
            _add_error(errors, "$.generator.version", "expected non-empty string")
        if generator.get("mode") != "local-offline":
            _add_error(errors, "$.generator.mode", "expected local-offline")

    managed_optimizer = bundle.get("managed_optimizer")
    if not isinstance(managed_optimizer, dict):
        _add_error(errors, "$.managed_optimizer", "expected object")
    elif managed_optimizer.get("enabled") is not False:
        _add_error(errors, "$.managed_optimizer.enabled", "expected false for local offline bundles")

    policies = bundle.get("policies")
    if not isinstance(policies, dict):
        _add_error(errors, "$.policies", "expected object")
    else:
        if policies.get("schema") != POLICY_STATE_SCHEMA:
            _add_error(errors, "$.policies.schema", f"expected {POLICY_STATE_SCHEMA}")
        for section in REQUIRED_POLICY_SECTIONS:
            value = policies.get(section)
            if not isinstance(value, dict):
                _add_error(errors, f"$.policies.{section}", "expected policy section object")
                continue
            source = value.get("policy_source")
            if source is not None and source not in POLICY_SOURCES:
                _add_error(errors, f"$.policies.{section}.policy_source", "unknown policy source")

    return {
        "schema": POLICY_BUNDLE_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle_schema,
        "errors": errors,
        "warnings": warnings,
    }


_MISSING = object()


def _diff_values(path: str, before: Any, after: Any, changes: list[dict[str, Any]]) -> None:
    if before == after:
        return

    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}"
            _diff_values(child_path, before.get(key, _MISSING), after.get(key, _MISSING), changes)
        return

    if isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child_path = f"{path}[{index}]"
            before_value = before[index] if index < len(before) else _MISSING
            after_value = after[index] if index < len(after) else _MISSING
            _diff_values(child_path, before_value, after_value, changes)
        return

    if before is _MISSING:
        changes.append({"path": path, "change": "added", "old": None, "new": after})
    elif after is _MISSING:
        changes.append({"path": path, "change": "removed", "old": before, "new": None})
    else:
        changes.append({"path": path, "change": "changed", "old": before, "new": after})


def compare_policy_bundles(before: Any, after: Any) -> dict[str, Any]:
    before_validation = validate_policy_bundle(before)
    after_validation = validate_policy_bundle(after)
    ok = bool(before_validation["ok"] and after_validation["ok"])
    changes: list[dict[str, Any]] = []

    if ok:
        before_policies = before["policies"]
        after_policies = after["policies"]
        for section in REQUIRED_POLICY_SECTIONS:
            _diff_values(
                f"$.policies.{section}",
                before_policies.get(section, {}),
                after_policies.get(section, {}),
                changes,
            )

    changed_sections = sorted(
        {
            change["path"].removeprefix("$.policies.").split(".", 1)[0].split("[", 1)[0]
            for change in changes
        }
    )

    return {
        "schema": POLICY_BUNDLE_DIFF_SCHEMA,
        "ok": ok,
        "changed": bool(changes),
        "changed_sections": changed_sections,
        "change_count": len(changes),
        "changes": changes,
        "before_validation": before_validation,
        "after_validation": after_validation,
    }


def _add_warning(warnings: list[dict[str, str]], code: str, path: str, message: str) -> None:
    warnings.append({
        "code": code,
        "path": path,
        "severity": "warning",
        "message": message,
    })


def _section_policy(bundle: Any, section: str) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return {}
    policies = bundle.get("policies")
    if not isinstance(policies, dict):
        return {}
    value = policies.get(section)
    return value if isinstance(value, dict) else {}


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return False


def policy_bundle_safety_warnings(bundle: Any) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    if not isinstance(bundle, dict):
        return warnings

    managed_optimizer = bundle.get("managed_optimizer")
    if isinstance(managed_optimizer, dict) and managed_optimizer.get("enabled") is not False:
        _add_warning(
            warnings,
            "managed-optimizer-enabled",
            "$.managed_optimizer.enabled",
            "managed optimizer communication must remain opt-in and is not part of local offline review",
        )

    for section in REQUIRED_POLICY_SECTIONS:
        policy = _section_policy(bundle, section)
        if policy.get("policy_source") == "managed-enforced":
            _add_warning(
                warnings,
                "managed-enforced-policy-source",
                f"$.policies.{section}.policy_source",
                "managed-enforced policy source should not be accepted by the local module without an explicit future import/apply flow",
            )

    cache = _section_policy(bundle, "cache")
    exact_cache = cache.get("exact_cache") if isinstance(cache.get("exact_cache"), dict) else {}
    if _enabled(exact_cache.get("cache_tool_calls")):
        _add_warning(
            warnings,
            "tool-call-cache-enabled",
            "$.policies.cache.exact_cache.cache_tool_calls",
            "tool-call caching can return stale filesystem-dependent results unless invalidation is proven safe",
        )
    semantic_cache = cache.get("semantic_cache") if isinstance(cache.get("semantic_cache"), dict) else {}
    if _enabled(semantic_cache.get("enabled")):
        _add_warning(
            warnings,
            "semantic-cache-enabled",
            "$.policies.cache.semantic_cache.enabled",
            "semantic cache can produce false-positive response reuse and should stay opt-in with quality checks",
        )

    crunch = _section_policy(bundle, "crunch")
    old_context = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else {}
    if _enabled(old_context.get("enabled")):
        _add_warning(
            warnings,
            "old-context-summarization-enabled",
            "$.policies.crunch.old_context_summarization.enabled",
            "model-assisted summarization changes request context and should be reviewed against quality risk before enabling",
        )

    return warnings


def review_policy_bundle(current: Any, proposed: Any) -> dict[str, Any]:
    diff = compare_policy_bundles(current, proposed)
    warnings = policy_bundle_safety_warnings(proposed)
    return {
        "schema": POLICY_BUNDLE_REVIEW_SCHEMA,
        "ok": bool(diff["ok"]),
        "changed": bool(diff.get("changed", False)) if diff["ok"] else False,
        "changed_sections": diff.get("changed_sections", []) if diff["ok"] else [],
        "change_count": int(diff.get("change_count", 0)) if diff["ok"] else 0,
        "safety_warning_count": len(warnings),
        "safety_warnings": warnings,
        "current_validation": diff.get("before_validation"),
        "proposed_validation": diff.get("after_validation"),
        "diff": {
            "schema": diff.get("schema"),
            "ok": diff.get("ok"),
            "changed": diff.get("changed"),
            "changed_sections": diff.get("changed_sections", []),
            "change_count": diff.get("change_count", 0),
            "changes": diff.get("changes", []),
        },
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy_apply_yaml(section: str, policy: dict[str, Any]) -> dict[str, Any]:
    if section == "routing":
        return {"rules": policy.get("rules") if isinstance(policy.get("rules"), list) else []}
    if section == "crunch":
        payload: dict[str, Any] = {}
        if "enabled" in policy:
            payload["enabled"] = policy.get("enabled")
        if "threshold_chars" in policy:
            payload["threshold_chars"] = policy.get("threshold_chars")
        for key in ("prompt_cache", "old_context_summarization", "thinking_deduplication"):
            if isinstance(policy.get(key), dict):
                payload[key] = policy[key]
        return payload
    if section == "cache":
        return {key: policy[key] for key in ("exact_cache", "semantic_cache", "file_watch") if isinstance(policy.get(key), dict)}
    if section == "routing_experiments":
        experiment_policy = policy.get("policy")
        return experiment_policy if isinstance(experiment_policy, dict) else {}
    return {}


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


def _latest_policy_backup(path: Path) -> Path | None:
    backups = sorted(path.parent.glob(f"{path.name}.bak-*"))
    return backups[-1] if backups else None


def apply_policy_bundle(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    allow_risky: bool = False,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    validation = validate_policy_bundle(bundle)
    warnings = policy_bundle_safety_warnings(bundle)
    requested_sections = list(sections or REQUIRED_POLICY_SECTIONS)
    invalid_sections = sorted(set(requested_sections) - set(REQUIRED_POLICY_SECTIONS))
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": POLICY_BUNDLE_APPLY_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "applied_sections": [],
        "skipped_sections": [],
        "files": [],
        "validation": validation,
        "safety_warning_count": len(warnings),
        "safety_warnings": warnings,
        "error": None,
    }

    if invalid_sections:
        result["error"] = {
            "type": "invalid_sections",
            "message": "unknown policy section requested",
            "sections": invalid_sections,
        }
        return result
    if not validation["ok"]:
        result["error"] = {"type": "validation_failed", "message": "policy bundle is invalid"}
        return result
    if warnings and not allow_risky:
        result["error"] = {
            "type": "risky_policy",
            "message": "policy bundle has safety warnings; rerun with --allow-risky to apply explicitly",
        }
        return result

    policies = bundle["policies"]
    for section in REQUIRED_POLICY_SECTIONS:
        if section not in requested_sections:
            result["skipped_sections"].append({"section": section, "reason": "not-requested"})
            continue

        policy = policies.get(section) if isinstance(policies.get(section), dict) else {}
        yaml_payload = _policy_apply_yaml(section, policy)
        text = yaml.safe_dump(yaml_payload, sort_keys=False)
        path = config_path / _POLICY_SECTION_FILES[section]
        old_text: str | None = None
        if path.exists():
            try:
                old_text = path.read_text(encoding="utf-8")
            except OSError:
                old_text = None
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


def rollback_policy_files(
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    requested_sections = list(sections or REQUIRED_POLICY_SECTIONS)
    invalid_sections = sorted(set(requested_sections) - set(REQUIRED_POLICY_SECTIONS))
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": POLICY_BUNDLE_ROLLBACK_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "restored_sections": [],
        "skipped_sections": [],
        "files": [],
        "error": None,
    }

    if invalid_sections:
        result["error"] = {
            "type": "invalid_sections",
            "message": "unknown policy section requested",
            "sections": invalid_sections,
        }
        return result

    plans: list[dict[str, Any]] = []
    missing_sections: list[str] = []
    unreadable_backups: list[dict[str, str]] = []
    for section in REQUIRED_POLICY_SECTIONS:
        if section not in requested_sections:
            result["skipped_sections"].append({"section": section, "reason": "not-requested"})
            continue

        path = config_path / _POLICY_SECTION_FILES[section]
        backup = _latest_policy_backup(path)
        old_text: str | None = None
        if path.exists():
            try:
                old_text = path.read_text(encoding="utf-8")
            except OSError:
                old_text = None
        if backup is None:
            missing_sections.append(section)
            result["files"].append({
                "section": section,
                "path": str(path),
                "restored_from": None,
                "changed": False,
                "backup_path": None,
                "sha256_before": _sha256_text(old_text) if old_text is not None else None,
                "sha256_after": None,
                "bytes_after": None,
            })
            continue

        try:
            backup_text = backup.read_text(encoding="utf-8")
        except OSError as exc:
            unreadable_backups.append({"section": section, "path": str(backup), "message": str(exc)})
            continue

        plans.append({
            "section": section,
            "path": path,
            "backup": backup,
            "old_text": old_text,
            "backup_text": backup_text,
        })

    if missing_sections:
        result["error"] = {
            "type": "missing_backups",
            "message": "one or more requested policy sections have no backup file",
            "sections": missing_sections,
        }
        return result
    if unreadable_backups:
        result["error"] = {
            "type": "unreadable_backups",
            "message": "one or more requested policy backups could not be read",
            "backups": unreadable_backups,
        }
        return result

    for plan in plans:
        path = plan["path"]
        old_text = plan["old_text"]
        backup_text = plan["backup_text"]
        changed = old_text != backup_text
        backup_path = None
        if changed and not dry_run:
            backup_path = _write_policy_file(path, backup_text)

        result["files"].append({
            "section": plan["section"],
            "path": str(path),
            "restored_from": str(plan["backup"]),
            "changed": bool(changed),
            "backup_path": backup_path,
            "sha256_before": _sha256_text(old_text) if old_text is not None else None,
            "sha256_after": _sha256_text(backup_text),
            "bytes_after": len(backup_text.encode("utf-8")),
        })
        result["restored_sections"].append(plan["section"])

    result["ok"] = True
    return result
