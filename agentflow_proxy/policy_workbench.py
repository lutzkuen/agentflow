from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from agentflow_proxy.paths import default_config_dir, safe_expanduser
from agentflow_proxy.policy_bundle import APPLY_POLICY_SECTIONS
from agentflow_proxy.policy_files import _draft_workspace_root, utc_now


POLICY_DRAFT_VALIDATE_SCHEMA = "agentflow.policy_draft_validate.v1"
POLICY_DRAFT_APPLY_SCHEMA = "agentflow.policy_draft_apply.v1"
POLICY_DRAFT_ROLLBACK_SCHEMA = "agentflow.policy_draft_rollback.v1"
POLICY_DRAFT_VALIDATE_PRIVACY = {
    "local_only": True,
    "metadata_only": True,
    "raw_prompts_included": False,
    "raw_responses_included": False,
    "provider_bodies_included": False,
    "raw_provider_bodies_included": False,
    "raw_tool_payloads_included": False,
    "raw_session_ids_included": False,
    "raw_request_ids_included": False,
    "cache_keys_included": False,
    "provider_calls_made": False,
    "managed_server_calls_made": False,
}
POLICY_DRAFT_APPLY_PRIVACY = POLICY_DRAFT_VALIDATE_PRIVACY | {
    "loopback_admin_calls_made": False,
}

ReloadPolicyState = Callable[[], Awaitable[dict[str, Any]]]


def _safe_read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, {"type": "read_failed", "path": str(path), "message": str(exc)}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        return None, {"type": "parse_failed", "path": str(path), "message": str(exc)}
    if not isinstance(parsed, dict):
        return None, {"type": "parse_failed", "path": str(path), "message": "expected JSON object"}
    return parsed, None


def _safe_read_policy(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, {"type": "read_failed", "path": str(path), "message": str(exc)}
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, {"type": "parse_failed", "path": str(path), "message": str(exc)}
    if not isinstance(parsed, dict):
        return None, {"type": "parse_failed", "path": str(path), "message": "expected policy bundle object"}
    return parsed, None


def _draft_paths(path_or_id: str, *, workspace: str | Path | None = None) -> tuple[Path | None, Path | None, Path | None, dict[str, Any] | None]:
    raw = Path(path_or_id).expanduser()
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw)
    else:
        candidates.append(_draft_workspace_root(workspace) / path_or_id)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate / "draft.json", candidate / "policy_bundle.json", candidate, None
        if candidate.exists() and candidate.name == "draft.json":
            return candidate, candidate.parent / "policy_bundle.json", candidate.parent, None
        if candidate.exists():
            manifest = candidate.parent / "draft.json"
            return manifest if manifest.exists() else None, candidate, candidate.parent, None
    return None, None, None, {
        "type": "draft_not_found",
        "path": path_or_id,
        "message": "staged draft directory, manifest, or policy bundle was not found",
    }


def load_staged_policy_draft(path_or_id: str, *, workspace: str | Path | None = None) -> dict[str, Any]:
    manifest_path, bundle_path, draft_dir, path_error = _draft_paths(path_or_id, workspace=workspace)
    if path_error is not None:
        return {"ok": False, "manifest": None, "bundle": None, "error": path_error}

    manifest: dict[str, Any] | None = None
    manifest_error: dict[str, Any] | None = None
    if manifest_path is not None and manifest_path.exists():
        manifest, manifest_error = _safe_read_json(manifest_path)
        if manifest_error is not None:
            return {"ok": False, "manifest": None, "bundle": None, "error": manifest_error}
        manifest_bundle = manifest.get("bundle_path") if isinstance(manifest, dict) else None
        if isinstance(manifest_bundle, str) and manifest_bundle.strip():
            bundle_path = Path(manifest_bundle).expanduser()

    if bundle_path is None:
        return {
            "ok": False,
            "manifest": manifest,
            "bundle": None,
            "error": {"type": "bundle_not_found", "message": "staged draft policy bundle path could not be resolved"},
        }
    bundle, bundle_error = _safe_read_policy(bundle_path)
    if bundle_error is not None:
        return {"ok": False, "manifest": manifest, "bundle": None, "error": bundle_error}

    return {
        "ok": True,
        "manifest": manifest,
        "bundle": bundle,
        "manifest_path": str(manifest_path) if manifest_path is not None and manifest_path.exists() else None,
        "bundle_path": str(bundle_path),
        "workspace": str(draft_dir) if draft_dir is not None else None,
        "error": None,
    }


def _path_section(path: Any) -> str | None:
    text = str(path or "")
    if not text.startswith("$.policies."):
        return None
    section = text.removeprefix("$.policies.").split(".", 1)[0].split("[", 1)[0]
    return section if section in APPLY_POLICY_SECTIONS else None


def _section_items(items: Any, section: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and _path_section(item.get("path")) == section]


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_number(value: Any) -> int:
    return int(round(_number(value)))


def _sum_nested(value: Any, keys: set[str]) -> float:
    total = 0.0
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys:
                total += _number(child)
            elif isinstance(child, (dict, list)):
                total += _sum_nested(child, keys)
    elif isinstance(value, list):
        for child in value:
            total += _sum_nested(child, keys)
    return total


def _blocker_codes(*items: list[dict[str, Any]]) -> list[str]:
    codes: set[str] = set()
    for group in items:
        for item in group:
            for key in ("code", "reason", "type"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    codes.add(value.strip())
                    break
    return sorted(codes)


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _apply_id(draft_id: str | None) -> str:
    safe = "".join(char for char in str(draft_id or "draft") if char.isalnum() or char in {"-", "_"}).strip("-_")
    safe = (safe or "draft")[:48]
    stamp = utc_now().replace(":", "").replace("+", "Z").replace(".", "")
    return f"{stamp}-{safe}"


def _atomic_write_policy_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_policy_text(path: Path) -> tuple[str | None, bool]:
    try:
        return path.read_text(encoding="utf-8"), True
    except FileNotFoundError:
        return None, False


def _backup_path(path: Path, apply_id: str) -> Path:
    return path.with_name(f"{path.name}.bak-{apply_id}")


def _rollback_backup_path(path: Path, apply_id: str) -> Path:
    rollback_id = _apply_id(f"rollback-{apply_id}")
    return path.with_name(f"{path.name}.bak-{rollback_id}")


def _file_result(plan: dict[str, Any], *, restored: bool = False) -> dict[str, Any]:
    result = {
        "section": plan["section"],
        "path": str(plan["path"]),
        "changed": bool(plan["changed"]),
        "backup_path": str(plan["backup_path"]) if plan.get("backup_path") is not None else None,
        "sha256_before": plan.get("sha256_before"),
        "sha256_after": plan.get("sha256_after"),
        "bytes_after": plan.get("bytes_after"),
        "existed_before": bool(plan.get("existed_before")),
    }
    if restored:
        result["restored"] = True
        result["sha256_restored"] = plan.get("sha256_before")
    return result


def _restore_transaction_plans(plans: list[dict[str, Any]]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for plan in plans:
        if not plan.get("changed"):
            continue
        path = plan["path"]
        try:
            if plan.get("existed_before"):
                _atomic_write_policy_text(path, plan.get("old_text") or "")
            elif path.exists():
                path.unlink()
            files.append(_file_result(plan, restored=True))
        except OSError as exc:
            errors.append({"section": plan["section"], "path": str(path), "message": str(exc)})
    return {
        "ok": not errors,
        "files": files,
        "errors": errors,
    }


def _matching_backup_section(path: Path, apply_id: str) -> str | None:
    suffix = f".bak-{apply_id}"
    if not path.name.endswith(suffix):
        return None
    base_name = path.name[: -len(suffix)]
    from agentflow_proxy.policy_files import POLICY_DRAFT_SECTION_FILES

    for section, filename in POLICY_DRAFT_SECTION_FILES.items():
        if base_name == filename:
            return section
    return None


def _apply_event_for_id(apply_id: str) -> dict[str, Any] | None:
    from agentflow_proxy.policy_events import policy_events_log_path

    path = policy_events_log_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("action") != "draft-apply":
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if details.get("apply_id") == apply_id:
            return event
    return None


def _event_sections(event: dict[str, Any] | None) -> list[str]:
    if not isinstance(event, dict):
        return []
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    sections = details.get("changed_sections") if isinstance(details.get("changed_sections"), list) else []
    return [section for section in APPLY_POLICY_SECTIONS if section in set(sections)]


def _event_backup_paths(event: dict[str, Any] | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not isinstance(event, dict):
        return paths
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    backup_paths = details.get("backup_paths") if isinstance(details.get("backup_paths"), list) else []
    for raw in backup_paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw).expanduser()
        section = _matching_backup_section(path, str(details.get("apply_id") or ""))
        if section is not None:
            paths[section] = path
    return paths


def _workbench_rollback_error_result(
    *,
    apply_id: str,
    config_dir: str | Path,
    dry_run: bool,
    sections: list[str] | tuple[str, ...] | None,
    force: bool,
    error: dict[str, Any],
    event: dict[str, Any] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": POLICY_DRAFT_ROLLBACK_SCHEMA,
        "ok": False,
        "status": "blocked",
        "apply_id": apply_id,
        "backup_id": apply_id,
        "dry_run": bool(dry_run),
        "force": bool(force),
        "config_dir": str(Path(config_dir).expanduser()),
        "manifest_source": "policy-event" if event is not None else "backup-suffix",
        "apply_event_found": event is not None,
        "requested_sections": list(sections or APPLY_POLICY_SECTIONS),
        "restored_sections": [],
        "skipped_sections": [],
        "files": files or [],
        "current_backups": [],
        "reloaded_modules": False,
        "reload": None,
        "verification": None,
        "privacy": POLICY_DRAFT_APPLY_PRIVACY,
        "error": error,
    }


def _section_file_state(policies: dict[str, Any], section: str) -> dict[str, Any]:
    raw = policies.get(section) if isinstance(policies.get(section), dict) else {}
    return raw.get("file") if isinstance(raw.get("file"), dict) else {}


def _verify_reloaded_policy_state(
    policies: dict[str, Any],
    plans: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for plan in plans:
        if not plan.get("requested"):
            continue
        section = plan["section"]
        file_state = _section_file_state(policies, section)
        loaded = file_state.get("loaded") if isinstance(file_state.get("loaded"), dict) else {}
        current = file_state.get("current") if isinstance(file_state.get("current"), dict) else {}
        expected_sha = plan.get("sha256_after")
        loaded_sha = loaded.get("sha256")
        current_sha = current.get("sha256")
        reload_required = bool(file_state.get("reload_required"))
        ok = loaded_sha == expected_sha and current_sha == expected_sha and not reload_required
        check = {
            "section": section,
            "path": str(plan["path"]),
            "ok": ok,
            "expected_sha256": expected_sha,
            "loaded_sha256": loaded_sha,
            "current_sha256": current_sha,
            "reload_required": reload_required,
        }
        checks.append(check)
        if not ok:
            failures.append(check)
    return {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
    }


def _section_changed(manifest: dict[str, Any] | None, review: dict[str, Any], section: str) -> bool:
    changed = review.get("changed_sections") if isinstance(review.get("changed_sections"), list) else []
    if section in changed:
        return True
    if not isinstance(manifest, dict):
        return False
    return section in set(manifest.get("changed_sections") if isinstance(manifest.get("changed_sections"), list) else [])


def _file_projection(dry_run: dict[str, Any], section: str) -> dict[str, Any]:
    files = dry_run.get("files") if isinstance(dry_run.get("files"), list) else []
    for item in files:
        if isinstance(item, dict) and item.get("section") == section:
            return {
                "path": item.get("path"),
                "changed": bool(item.get("changed")),
                "bytes_after": item.get("bytes_after"),
                "sha256_before": item.get("sha256_before"),
                "sha256_after": item.get("sha256_after"),
                "diff_available": bool(item.get("diff")),
            }
    return {"path": None, "changed": False, "diff_available": False}


def _section_impact(section: str, review: dict[str, Any], codex_dry_run: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    impact = review.get("impact_summary") if isinstance(review.get("impact_summary"), dict) else {}
    sections = impact.get("sections") if isinstance(impact.get("sections"), dict) else {}
    raw = sections.get(section) if isinstance(sections.get(section), dict) else {}
    warnings = raw.get("warnings") if isinstance(raw.get("warnings"), list) else []

    if section == "codex_app" and isinstance(codex_dry_run, dict):
        summary = codex_dry_run.get("summary") if isinstance(codex_dry_run.get("summary"), dict) else {}
        return {
            "status": codex_dry_run.get("schema"),
            "projected_match_count": _int_number(summary.get("evaluated_rows")),
            "projected_applied_count": _int_number(summary.get("projected_applied_count")),
            "projected_holdout_count": _int_number(summary.get("projected_holdout_count")),
            "projected_skip_count": _int_number(summary.get("projected_skip_count")),
            "projected_savings_usd": _number(summary.get("projected_savings_usd")),
            "blocker_breakdown": codex_dry_run.get("blocker_breakdown", []),
            "raw_bodies_read": False,
            "metadata_only": True,
        }, warnings

    projected = {
        "status": raw.get("status"),
        "projected_match_count": _int_number(_sum_nested(raw, {"would_match_count", "eligible_call_count", "matched_count"})),
        "projected_applied_count": _int_number(_sum_nested(raw, {"would_apply_count", "projected_applied_count"})),
        "projected_holdout_count": _int_number(_sum_nested(raw, {"would_holdout_count", "projected_holdout_count"})),
        "projected_skip_count": _int_number(_sum_nested(raw, {"would_bypass_count", "projected_skip_count", "excluded_streaming_count", "excluded_tool_count", "excluded_thinking_count"})),
        "projected_savings_usd": round(_sum_nested(raw, {"estimated_savings_usd", "projected_savings_usd", "estimated_candidate_savings_usd"}), 8),
        "safety_blocker_count": _int_number(_sum_nested(raw, {"safety_blocker_count"})),
        "metadata_only": bool(raw.get("metadata_only", True)),
        "raw_bodies_read": bool(raw.get("raw_bodies_read", False)),
    }
    if section == "routing":
        projected["projected_applied_count"] = projected["projected_applied_count"] or projected["projected_match_count"]
    if section == "routing_experiments":
        projected["projected_applied_count"] = _int_number(raw.get("estimated_sample_count"))
        projected["projected_match_count"] = _int_number(raw.get("eligible_call_count"))
        projected["projected_holdout_count"] = max(0, projected["projected_match_count"] - projected["projected_applied_count"])
    if section == "crunch":
        projected["projected_applied_count"] = projected["projected_applied_count"] or projected["projected_match_count"]
    return projected, warnings


def _run_codex_dry_run(bundle: dict[str, Any], *, db_path: str | None, recent_limit: int) -> dict[str, Any] | None:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    codex = policies.get("codex_app") if isinstance(policies.get("codex_app"), dict) else {}
    if not codex.get("rules"):
        return None
    from agentflow_proxy.codex_app_dry_run import dry_run_codex_app_policy
    from agentflow_proxy.store import Store

    store = None
    if db_path:
        path = Path(db_path).expanduser()
        if path.exists():
            store = Store(str(path))
    try:
        return dry_run_codex_app_policy(
            bundle,
            store=store,
            recent_limit=max(0, recent_limit),
            include_synthetic=True,
        )
    finally:
        if store is not None:
            store.conn.close()


async def validate_staged_policy_draft(
    path_or_id: str,
    *,
    workspace: str | Path | None = None,
    config_dir: str | Path | None = None,
    db_path: str | None = None,
    impact_limit: int = 1000,
    codex_recent_limit: int = 200,
) -> dict[str, Any]:
    from agentflow_proxy.policy_bundle import apply_policy_bundle, build_policy_bundle, review_policy_bundle, validate_policy_bundle

    loaded = load_staged_policy_draft(path_or_id, workspace=workspace)
    if not loaded.get("ok"):
        return {
            "schema": POLICY_DRAFT_VALIDATE_SCHEMA,
            "ok": False,
            "status": "fail",
            "can_apply": False,
            "apply_blocked": True,
            "generated_at": utc_now(),
            "draft_id": path_or_id,
            "draft": None,
            "validation": None,
            "review": None,
            "dry_run_apply": None,
            "sections": [],
            "privacy": POLICY_DRAFT_VALIDATE_PRIVACY,
            "error": loaded.get("error"),
        }

    bundle = loaded["bundle"]
    manifest = loaded.get("manifest") if isinstance(loaded.get("manifest"), dict) else None
    current = await build_policy_bundle()
    validation = validate_policy_bundle(bundle)
    review = review_policy_bundle(current, bundle, impact_db_path=db_path, impact_limit=max(0, impact_limit))
    dry_run = apply_policy_bundle(
        bundle,
        config_dir=config_dir or str(default_config_dir()),
        dry_run=True,
        allow_risky=True,
    )
    codex_dry_run = _run_codex_dry_run(bundle, db_path=db_path, recent_limit=codex_recent_limit) if validation.get("ok") else None

    validation_errors = validation.get("errors") if isinstance(validation.get("errors"), list) else []
    safety_warnings = review.get("safety_warnings") if isinstance(review.get("safety_warnings"), list) else []
    dry_error = dry_run.get("error") if isinstance(dry_run.get("error"), dict) else None
    sections: list[dict[str, Any]] = []
    section_verdicts: list[str] = []

    for section in APPLY_POLICY_SECTIONS:
        section_validation_errors = _section_items(validation_errors, section)
        section_safety_warnings = _section_items(safety_warnings, section)
        projected, impact_warnings = _section_impact(section, review, codex_dry_run)
        warning_items = [*section_safety_warnings, *impact_warnings]
        dry_run_blocked = bool(dry_error and section in (review.get("changed_sections") or APPLY_POLICY_SECTIONS))
        if section_validation_errors or dry_run_blocked:
            verdict = "fail"
        elif warning_items:
            verdict = "warn"
        else:
            verdict = "pass"
        section_verdicts.append(verdict)
        sections.append({
            "section": section,
            "changed": _section_changed(manifest, review, section),
            "verdict": verdict,
            "blocker_reason_codes": _blocker_codes(section_validation_errors, section_safety_warnings, impact_warnings)
            + ([dry_error["type"]] if dry_run_blocked and isinstance(dry_error.get("type"), str) else []),
            "validation_error_count": len(section_validation_errors),
            "safety_warning_count": len(section_safety_warnings),
            "impact_warning_count": len(impact_warnings),
            "reload_required_after_apply": _section_changed(manifest, review, section),
            "dry_run": {
                "ok": bool(dry_run.get("ok")),
                "wrote_policy_files": False,
                "file": _file_projection(dry_run, section),
            },
            "projected_impact": projected,
            "privacy": POLICY_DRAFT_VALIDATE_PRIVACY,
        })

    top_status = "fail" if not validation.get("ok") or not dry_run.get("ok") else ("warn" if "warn" in section_verdicts else "pass")
    can_apply = bool(validation.get("ok") and dry_run.get("ok") and not safety_warnings)
    apply_blockers = _blocker_codes(validation_errors, safety_warnings)
    if dry_error and isinstance(dry_error.get("type"), str):
        apply_blockers.append(dry_error["type"])

    return {
        "schema": POLICY_DRAFT_VALIDATE_SCHEMA,
        "ok": can_apply,
        "status": top_status,
        "can_apply": can_apply,
        "apply_blocked": not can_apply,
        "generated_at": utc_now(),
        "draft_id": (manifest or {}).get("draft_id") or path_or_id,
        "draft": {
            "workspace": loaded.get("workspace"),
            "manifest_path": loaded.get("manifest_path"),
            "bundle_path": loaded.get("bundle_path"),
            "changed": bool((manifest or {}).get("changed", review.get("changed"))),
            "changed_sections": review.get("changed_sections", []),
        },
        "validation": validation,
        "review": {
            "schema": review.get("schema"),
            "ok": review.get("ok"),
            "changed": review.get("changed"),
            "changed_sections": review.get("changed_sections", []),
            "change_count": review.get("change_count", 0),
            "safety_warning_count": review.get("safety_warning_count", 0),
            "recommendation_health": review.get("recommendation_health"),
            "impact_summary": review.get("impact_summary"),
            "section_reviews": review.get("section_reviews", {}),
        },
        "dry_run_apply": {
            "schema": dry_run.get("schema"),
            "ok": dry_run.get("ok"),
            "dry_run": True,
            "wrote_policy_files": False,
            "applied_sections": dry_run.get("applied_sections", []),
            "skipped_sections": dry_run.get("skipped_sections", []),
            "safety_warning_count": dry_run.get("safety_warning_count", 0),
            "error": dry_run.get("error"),
            "old_context_summarization": dry_run.get("old_context_summarization"),
            "codex_app": dry_run.get("codex_app"),
        },
        "codex_app_dry_run": codex_dry_run,
        "apply_prerequisites": {
            "validation_ok": bool(validation.get("ok")),
            "dry_run_ok": bool(dry_run.get("ok")),
            "no_safety_warnings": not safety_warnings,
            "reload_required_after_apply": [
                section["section"] for section in sections if section["reload_required_after_apply"]
            ],
            "blocker_reason_codes": sorted(set(apply_blockers)),
        },
        "sections": sections,
        "privacy": POLICY_DRAFT_VALIDATE_PRIVACY,
        "error": None if can_apply else {
            "type": "apply_blocked",
            "message": "staged policy draft did not pass the local pre-apply gate",
            "blocker_reason_codes": sorted(set(apply_blockers)),
        },
    }


async def _default_reload_policy_state() -> dict[str, Any]:
    from agentflow_proxy.admin import reload_policy_modules

    return await reload_policy_modules()


def _requested_sections(sections: list[str] | tuple[str, ...] | None) -> tuple[list[str], list[str]]:
    requested = list(sections or APPLY_POLICY_SECTIONS)
    invalid = sorted(set(requested) - set(APPLY_POLICY_SECTIONS))
    ordered = [section for section in APPLY_POLICY_SECTIONS if section in set(requested)]
    return ordered, invalid


def _draft_apply_error_result(
    *,
    status: str,
    draft_id: str,
    apply_id: str,
    config_dir: str | Path,
    validation: dict[str, Any] | None,
    error: dict[str, Any],
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "schema": POLICY_DRAFT_APPLY_SCHEMA,
        "ok": False,
        "status": status,
        "draft_id": draft_id,
        "apply_id": apply_id,
        "backup_id": apply_id,
        "config_dir": str(Path(config_dir).expanduser()),
        "requested_sections": list(sections or APPLY_POLICY_SECTIONS),
        "applied_sections": [],
        "changed_sections": [],
        "files": [],
        "backups": [],
        "reloaded_modules": False,
        "reload": None,
        "verification": None,
        "validation": validation,
        "restored": False,
        "restore": None,
        "rollback_command": None,
        "privacy": POLICY_DRAFT_APPLY_PRIVACY,
        "error": error,
    }


async def apply_validated_policy_draft(
    path_or_id: str,
    *,
    workspace: str | Path | None = None,
    config_dir: str | Path | None = None,
    db_path: str | None = None,
    impact_limit: int = 1000,
    codex_recent_limit: int = 200,
    sections: list[str] | tuple[str, ...] | None = None,
    reload_policy_state: ReloadPolicyState | None = None,
    apply_id: str | None = None,
    event_source: str = "workbench",
    loopback_admin_calls_made: bool = False,
) -> dict[str, Any]:
    from agentflow_proxy.policy_bundle import _policy_apply_yaml
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.policy_files import POLICY_DRAFT_SECTION_FILES

    config_path = safe_expanduser(config_dir) if config_dir is not None else default_config_dir()
    validation = await validate_staged_policy_draft(
        path_or_id,
        workspace=workspace,
        config_dir=config_path,
        db_path=db_path,
        impact_limit=impact_limit,
        codex_recent_limit=codex_recent_limit,
    )
    draft_id = str((validation.get("draft") or {}).get("draft_id") or validation.get("draft_id") or path_or_id)
    transaction_id = apply_id or _apply_id(draft_id)
    privacy = POLICY_DRAFT_APPLY_PRIVACY | {"loopback_admin_calls_made": bool(loopback_admin_calls_made)}

    requested, invalid_sections = _requested_sections(sections)
    if invalid_sections:
        result = _draft_apply_error_result(
            status="failed",
            draft_id=draft_id,
            apply_id=transaction_id,
            config_dir=config_path,
            validation=validation,
            sections=sections,
            error={
                "type": "invalid_sections",
                "message": "unknown or review-only policy section requested",
                "sections": invalid_sections,
            },
        )
        result["privacy"] = privacy
        log_policy_event("draft-apply", ok=False, details={"source": event_source, **_policy_apply_event_details(result)})
        return result

    if not validation.get("can_apply"):
        result = _draft_apply_error_result(
            status="blocked",
            draft_id=draft_id,
            apply_id=transaction_id,
            config_dir=config_path,
            validation=validation,
            sections=sections,
            error=validation.get("error") if isinstance(validation.get("error"), dict) else {
                "type": "apply_blocked",
                "message": "staged policy draft did not pass validation",
            },
        )
        result["privacy"] = privacy
        log_policy_event("draft-apply", ok=False, details={"source": event_source, **_policy_apply_event_details(result)})
        return result

    loaded = load_staged_policy_draft(path_or_id, workspace=workspace)
    if not loaded.get("ok"):
        result = _draft_apply_error_result(
            status="failed",
            draft_id=draft_id,
            apply_id=transaction_id,
            config_dir=config_path,
            validation=validation,
            sections=sections,
            error=loaded.get("error") if isinstance(loaded.get("error"), dict) else {
                "type": "draft_not_found",
                "message": "staged draft could not be loaded for apply",
            },
        )
        result["privacy"] = privacy
        log_policy_event("draft-apply", ok=False, details={"source": event_source, **_policy_apply_event_details(result)})
        return result

    bundle = loaded["bundle"]
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    plans: list[dict[str, Any]] = []
    for section in APPLY_POLICY_SECTIONS:
        requested_section = section in requested
        policy = policies.get(section) if isinstance(policies.get(section), dict) else {}
        text = yaml.safe_dump(_policy_apply_yaml(section, policy), sort_keys=False)
        path = config_path / POLICY_DRAFT_SECTION_FILES[section]
        old_text, existed_before = _read_policy_text(path)
        changed = old_text != text
        backup = _backup_path(path, transaction_id) if changed and existed_before else None
        plans.append({
            "section": section,
            "requested": requested_section,
            "path": path,
            "new_text": text,
            "old_text": old_text,
            "existed_before": existed_before,
            "changed": bool(changed and requested_section),
            "backup_path": backup,
            "sha256_before": _sha256_text(old_text),
            "sha256_after": _sha256_text(text),
            "bytes_after": len(text.encode("utf-8")),
        })

    write_plans = [plan for plan in plans if plan["requested"] and plan["changed"]]
    backups: list[dict[str, Any]] = []
    try:
        for plan in write_plans:
            if plan.get("backup_path") is None:
                continue
            backup_path = plan["backup_path"]
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_text(plan.get("old_text") or "", encoding="utf-8")
            backups.append({
                "section": plan["section"],
                "backup_id": transaction_id,
                "path": str(backup_path),
                "sha256": plan.get("sha256_before"),
            })
        for plan in write_plans:
            _atomic_write_policy_text(plan["path"], plan["new_text"])
    except OSError as exc:
        restore = _restore_transaction_plans(write_plans)
        result = {
            "schema": POLICY_DRAFT_APPLY_SCHEMA,
            "ok": False,
            "status": "failed",
            "draft_id": draft_id,
            "apply_id": transaction_id,
            "backup_id": transaction_id,
            "config_dir": str(config_path),
            "requested_sections": requested,
            "applied_sections": [],
            "changed_sections": [],
            "files": [_file_result(plan) for plan in plans if plan["requested"]],
            "backups": backups,
            "reloaded_modules": False,
            "reload": None,
            "verification": None,
            "validation": validation,
            "restored": True,
            "restore": restore,
            "rollback_command": None,
            "privacy": privacy,
            "error": {"type": "write_failed", "message": str(exc)},
        }
        log_policy_event("draft-apply", ok=False, details={"source": event_source, **_policy_apply_event_details(result)})
        return result

    reload_payload: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    restore: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    reloader = reload_policy_state or _default_reload_policy_state
    if write_plans:
        try:
            reload_payload = await reloader()
            if not isinstance(reload_payload, dict) or not reload_payload.get("ok"):
                error = {
                    "type": "reload_failed",
                    "message": "policy reload did not return ok=true",
                    "reload": reload_payload,
                }
            else:
                policies_after = reload_payload.get("policies") if isinstance(reload_payload.get("policies"), dict) else {}
                verification = _verify_reloaded_policy_state(policies_after, [plan for plan in plans if plan["requested"]])
                if not verification.get("ok"):
                    error = {
                        "type": "verification_failed",
                        "message": "reloaded policy snapshots did not match applied files",
                        "failures": verification.get("failures", []),
                    }
        except Exception as exc:  # pragma: no cover - exercised through tests by type, not branch internals
            error = {"type": "reload_failed", "message": str(exc)}

    if error is not None:
        restore = _restore_transaction_plans(write_plans)
        restore_reload: dict[str, Any] | None = None
        try:
            restore_reload = await reloader()
        except Exception as exc:  # pragma: no cover
            restore_reload = {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}}
        restore["reload"] = restore_reload
        result = {
            "schema": POLICY_DRAFT_APPLY_SCHEMA,
            "ok": False,
            "status": "failed",
            "draft_id": draft_id,
            "apply_id": transaction_id,
            "backup_id": transaction_id,
            "config_dir": str(config_path),
            "requested_sections": requested,
            "applied_sections": [],
            "changed_sections": [],
            "files": [_file_result(plan) for plan in plans if plan["requested"]],
            "backups": backups,
            "reloaded_modules": bool(reload_payload and reload_payload.get("ok")),
            "reload": reload_payload,
            "verification": verification,
            "validation": validation,
            "restored": True,
            "restore": restore,
            "rollback_command": None,
            "privacy": privacy,
            "error": error,
        }
        log_policy_event("draft-apply", ok=False, details={"source": event_source, **_policy_apply_event_details(result)})
        return result

    changed_sections = [plan["section"] for plan in write_plans]
    rollback_command = None
    if changed_sections:
        rollback_command = " ".join(
            ["agentflow-policy-rollback", "--config-dir", str(config_path), "--apply-id", transaction_id]
            + [part for section in changed_sections for part in ("--section", section)]
        )
    result = {
        "schema": POLICY_DRAFT_APPLY_SCHEMA,
        "ok": True,
        "status": "applied",
        "draft_id": draft_id,
        "apply_id": transaction_id,
        "backup_id": transaction_id,
        "config_dir": str(config_path),
        "requested_sections": requested,
        "applied_sections": requested,
        "changed_sections": changed_sections,
        "files": [_file_result(plan) for plan in plans if plan["requested"]],
        "backups": backups,
        "reloaded_modules": bool(write_plans),
        "reload": reload_payload,
        "verification": verification or {"ok": True, "checks": [], "failures": []},
        "validation": validation,
        "restored": False,
        "restore": None,
        "rollback_command": rollback_command,
        "rollback": {
            "backup_id": transaction_id,
            "command": rollback_command,
            "backup_paths": [backup["path"] for backup in backups],
        },
        "privacy": privacy,
        "error": None,
    }
    log_policy_event("draft-apply", ok=True, details={"source": event_source, **_policy_apply_event_details(result)})
    return result


async def rollback_policy_apply(
    apply_id: str,
    *,
    config_dir: str | Path | None = None,
    sections: list[str] | tuple[str, ...] | None = None,
    dry_run: bool = False,
    force: bool = False,
    reload_policy_state: ReloadPolicyState | None = None,
    event_source: str = "workbench",
    loopback_admin_calls_made: bool = False,
) -> dict[str, Any]:
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.policy_files import POLICY_DRAFT_SECTION_FILES

    clean_apply_id = str(apply_id or "").strip()
    config_path = safe_expanduser(config_dir) if config_dir is not None else default_config_dir()
    privacy = POLICY_DRAFT_APPLY_PRIVACY | {"loopback_admin_calls_made": bool(loopback_admin_calls_made)}
    requested, invalid_sections = _requested_sections(sections)
    if not clean_apply_id:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            error={"type": "missing_apply_id", "message": "rollback requires an apply ID"},
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result
    if invalid_sections:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            error={
                "type": "invalid_sections",
                "message": "unknown or review-only policy section requested",
                "sections": invalid_sections,
            },
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    event = _apply_event_for_id(clean_apply_id)
    event_details = event.get("details") if isinstance(event, dict) and isinstance(event.get("details"), dict) else {}
    event_sections = _event_sections(event)
    event_backups = _event_backup_paths(event)
    discovered_backups: dict[str, Path] = {}
    for section, filename in POLICY_DRAFT_SECTION_FILES.items():
        backup = _backup_path(config_path / filename, clean_apply_id)
        if backup.exists():
            discovered_backups[section] = backup

    if event is not None and not event.get("ok") and not force:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            event=event,
            error={
                "type": "apply_not_successful",
                "message": "the matching apply event did not complete successfully",
                "apply_status": event_details.get("status"),
            },
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    if event is not None:
        expected_sections = [section for section in (requested if sections else event_sections) if section in APPLY_POLICY_SECTIONS]
    elif sections:
        expected_sections = requested
    elif force:
        expected_sections = [section for section in APPLY_POLICY_SECTIONS if section in discovered_backups]
    else:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            files=[
                {
                    "section": section,
                    "path": str(config_path / POLICY_DRAFT_SECTION_FILES[section]),
                    "restored_from": str(path),
                    "changed": False,
                }
                for section, path in discovered_backups.items()
            ],
            error={
                "type": "apply_event_not_found",
                "message": "no policy apply event was found for this apply ID; pass --section for an exact recovery set or --force for CLI-only recovery",
            },
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    if not expected_sections:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            event=event,
            error={
                "type": "empty_backup_set",
                "message": "no policy sections with backups were associated with this apply ID",
            },
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    if event is not None and not force:
        unexpected = sorted(set(discovered_backups) - set(event_sections))
        if unexpected:
            result = _workbench_rollback_error_result(
                apply_id=clean_apply_id,
                config_dir=config_path,
                dry_run=dry_run,
                sections=sections,
                force=force,
                event=event,
                error={
                    "type": "ambiguous_backup_set",
                    "message": "backup files exist for sections not recorded on the apply event",
                    "sections": unexpected,
                },
            )
            result["privacy"] = privacy
            log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
            return result

    plans: list[dict[str, Any]] = []
    missing_sections: list[str] = []
    unreadable_backups: list[dict[str, str]] = []
    for section in APPLY_POLICY_SECTIONS:
        if section not in expected_sections:
            continue
        path = config_path / POLICY_DRAFT_SECTION_FILES[section]
        backup = event_backups.get(section) or discovered_backups.get(section) or _backup_path(path, clean_apply_id)
        if not backup.exists():
            missing_sections.append(section)
            plans.append({
                "section": section,
                "path": path,
                "backup": backup,
                "missing": True,
            })
            continue
        current_text, current_exists = _read_policy_text(path)
        try:
            backup_text = backup.read_text(encoding="utf-8")
        except OSError as exc:
            unreadable_backups.append({"section": section, "path": str(backup), "message": str(exc)})
            continue
        plans.append({
            "section": section,
            "path": path,
            "backup": backup,
            "old_text": current_text,
            "existed_before": current_exists,
            "backup_text": backup_text,
            "changed": current_text != backup_text,
            "sha256_before": _sha256_text(current_text),
            "sha256_after": _sha256_text(backup_text),
            "bytes_after": len(backup_text.encode("utf-8")),
        })

    partial_sections = sorted(set(missing_sections))
    if partial_sections and not force:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            event=event,
            files=[
                {
                    "section": plan["section"],
                    "path": str(plan["path"]),
                    "restored_from": str(plan["backup"]) if plan.get("backup") else None,
                    "changed": False,
                    "backup_path": None,
                    "missing": bool(plan.get("missing")),
                }
                for plan in plans
            ],
            error={
                "type": "partial_backup_set",
                "message": "one or more requested apply backups are missing",
                "sections": partial_sections,
            },
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result
    if unreadable_backups:
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            event=event,
            error={
                "type": "unreadable_backups",
                "message": "one or more apply backups could not be read",
                "backups": unreadable_backups,
            },
        )
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    write_plans = [plan for plan in plans if not plan.get("missing")]
    current_backups: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    if dry_run:
        for plan in write_plans:
            files.append({
                "section": plan["section"],
                "path": str(plan["path"]),
                "restored_from": str(plan["backup"]),
                "changed": bool(plan["changed"]),
                "backup_path": None,
                "sha256_before": plan.get("sha256_before"),
                "sha256_after": plan.get("sha256_after"),
                "bytes_after": plan.get("bytes_after"),
            })
        result = {
            "schema": POLICY_DRAFT_ROLLBACK_SCHEMA,
            "ok": True,
            "status": "dry-run",
            "apply_id": clean_apply_id,
            "backup_id": clean_apply_id,
            "dry_run": True,
            "force": bool(force),
            "config_dir": str(config_path),
            "manifest_source": "policy-event" if event is not None else "backup-suffix",
            "apply_event_found": event is not None,
            "requested_sections": expected_sections,
            "restored_sections": [plan["section"] for plan in write_plans],
            "skipped_sections": [
                {"section": section, "reason": "not-requested"}
                for section in APPLY_POLICY_SECTIONS
                if section not in set(expected_sections)
            ],
            "files": files,
            "current_backups": [],
            "reloaded_modules": False,
            "reload": None,
            "verification": None,
            "privacy": privacy,
            "error": None,
        }
        log_policy_event("rollback", ok=True, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    written_plans: list[dict[str, Any]] = []
    try:
        for plan in write_plans:
            if plan.get("changed") and plan.get("existed_before"):
                current_backup = _rollback_backup_path(plan["path"], clean_apply_id)
                current_backup.write_text(plan.get("old_text") or "", encoding="utf-8")
                plan["current_backup_path"] = current_backup
                current_backups.append({
                    "section": plan["section"],
                    "path": str(current_backup),
                    "sha256": plan.get("sha256_before"),
                })
            if plan.get("changed"):
                _atomic_write_policy_text(plan["path"], plan.get("backup_text") or "")
                written_plans.append(plan)
    except OSError as exc:
        restore = _restore_transaction_plans(written_plans)
        files = [
            {
                "section": plan["section"],
                "path": str(plan["path"]),
                "restored_from": str(plan["backup"]),
                "changed": bool(plan.get("changed")),
                "backup_path": str(plan.get("current_backup_path")) if plan.get("current_backup_path") is not None else None,
                "sha256_before": plan.get("sha256_before"),
                "sha256_after": plan.get("sha256_after"),
                "bytes_after": plan.get("bytes_after"),
            }
            for plan in write_plans
        ]
        result = _workbench_rollback_error_result(
            apply_id=clean_apply_id,
            config_dir=config_path,
            dry_run=dry_run,
            sections=sections,
            force=force,
            event=event,
            files=files,
            error={"type": "write_failed", "message": str(exc), "restore": restore},
        )
        result["current_backups"] = current_backups
        result["privacy"] = privacy
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    reload_payload: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    reloader = reload_policy_state or _default_reload_policy_state
    try:
        reload_payload = await reloader()
        if not isinstance(reload_payload, dict) or not reload_payload.get("ok"):
            error = {
                "type": "reload_failed",
                "message": "policy reload did not return ok=true",
                "reload": reload_payload,
            }
        else:
            policies_after = reload_payload.get("policies") if isinstance(reload_payload.get("policies"), dict) else {}
            verification_plans = [
                {
                    "requested": True,
                    "section": plan["section"],
                    "path": plan["path"],
                    "sha256_after": plan.get("sha256_after"),
                }
                for plan in write_plans
            ]
            verification = _verify_reloaded_policy_state(policies_after, verification_plans)
            if not verification.get("ok"):
                error = {
                    "type": "verification_failed",
                    "message": "reloaded policy snapshots did not match rolled-back files",
                    "failures": verification.get("failures", []),
                }
    except Exception as exc:  # pragma: no cover - exercised through tests by type, not branch internals
        error = {"type": "reload_failed", "message": str(exc)}

    files = [
        {
            "section": plan["section"],
            "path": str(plan["path"]),
            "restored_from": str(plan["backup"]),
            "changed": bool(plan.get("changed")),
            "backup_path": str(plan.get("current_backup_path")) if plan.get("current_backup_path") is not None else None,
            "sha256_before": plan.get("sha256_before"),
            "sha256_after": plan.get("sha256_after"),
            "bytes_after": plan.get("bytes_after"),
        }
        for plan in write_plans
    ]
    if error is not None:
        restore = _restore_transaction_plans(written_plans)
        restore_reload: dict[str, Any] | None = None
        try:
            restore_reload = await reloader()
        except Exception as exc:  # pragma: no cover
            restore_reload = {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}}
        restore["reload"] = restore_reload
        result = {
            "schema": POLICY_DRAFT_ROLLBACK_SCHEMA,
            "ok": False,
            "status": "failed",
            "apply_id": clean_apply_id,
            "backup_id": clean_apply_id,
            "dry_run": False,
            "force": bool(force),
            "config_dir": str(config_path),
            "manifest_source": "policy-event" if event is not None else "backup-suffix",
            "apply_event_found": event is not None,
            "requested_sections": expected_sections,
            "restored_sections": [],
            "skipped_sections": [
                {"section": section, "reason": "not-requested"}
                for section in APPLY_POLICY_SECTIONS
                if section not in set(expected_sections)
            ],
            "files": files,
            "current_backups": current_backups,
            "reloaded_modules": bool(reload_payload and reload_payload.get("ok")),
            "reload": reload_payload,
            "verification": verification,
            "restored": True,
            "restore": restore,
            "privacy": privacy,
            "error": error,
        }
        log_policy_event("rollback", ok=False, details={"source": event_source, **_policy_rollback_event_details(result)})
        return result

    result = {
        "schema": POLICY_DRAFT_ROLLBACK_SCHEMA,
        "ok": True,
        "status": "rolled-back",
        "apply_id": clean_apply_id,
        "backup_id": clean_apply_id,
        "dry_run": False,
        "force": bool(force),
        "config_dir": str(config_path),
        "manifest_source": "policy-event" if event is not None else "backup-suffix",
        "apply_event_found": event is not None,
        "requested_sections": expected_sections,
        "restored_sections": [plan["section"] for plan in write_plans],
        "skipped_sections": [
            {"section": section, "reason": "not-requested"}
            for section in APPLY_POLICY_SECTIONS
            if section not in set(expected_sections)
        ],
        "files": files,
        "current_backups": current_backups,
        "reloaded_modules": True,
        "reload": reload_payload,
        "verification": verification or {"ok": True, "checks": [], "failures": []},
        "privacy": privacy,
        "error": None,
    }
    log_policy_event("rollback", ok=True, details={"source": event_source, **_policy_rollback_event_details(result)})
    return result


def _policy_apply_event_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "draft_id": result.get("draft_id"),
        "apply_id": result.get("apply_id"),
        "backup_id": result.get("backup_id"),
        "config_dir": result.get("config_dir"),
        "status": result.get("status"),
        "requested_sections": result.get("requested_sections", []),
        "applied_sections": result.get("applied_sections", []),
        "changed_sections": result.get("changed_sections", []),
        "changed_files": [
            file.get("path")
            for file in result.get("files", [])
            if isinstance(file, dict) and file.get("changed")
        ],
        "backup_paths": [
            backup.get("path")
            for backup in result.get("backups", [])
            if isinstance(backup, dict) and backup.get("path")
        ],
        "reloaded_modules": ((result.get("reload") or {}).get("reloaded_modules", []) if isinstance(result.get("reload"), dict) else []),
        "verification_ok": ((result.get("verification") or {}).get("ok") if isinstance(result.get("verification"), dict) else None),
        "validation_status": ((result.get("validation") or {}).get("status") if isinstance(result.get("validation"), dict) else None),
        "validation_can_apply": ((result.get("validation") or {}).get("can_apply") if isinstance(result.get("validation"), dict) else None),
        "restored": result.get("restored"),
        "restore_ok": ((result.get("restore") or {}).get("ok") if isinstance(result.get("restore"), dict) else None),
        "rollback_command": result.get("rollback_command"),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "exit_code": 0 if result.get("ok") else 1,
    }


def _policy_rollback_event_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "apply_id": result.get("apply_id"),
        "backup_id": result.get("backup_id"),
        "config_dir": result.get("config_dir"),
        "status": result.get("status"),
        "dry_run": result.get("dry_run"),
        "force": result.get("force"),
        "manifest_source": result.get("manifest_source"),
        "apply_event_found": result.get("apply_event_found"),
        "requested_sections": result.get("requested_sections", []),
        "restored_sections": result.get("restored_sections", []),
        "changed_files": [
            file.get("path")
            for file in result.get("files", [])
            if isinstance(file, dict) and file.get("changed")
        ],
        "restored_from": [
            file.get("restored_from")
            for file in result.get("files", [])
            if isinstance(file, dict) and file.get("restored_from")
        ],
        "current_backup_paths": [
            backup.get("path")
            for backup in result.get("current_backups", [])
            if isinstance(backup, dict) and backup.get("path")
        ],
        "reloaded_modules": ((result.get("reload") or {}).get("reloaded_modules", []) if isinstance(result.get("reload"), dict) else []),
        "verification_ok": ((result.get("verification") or {}).get("ok") if isinstance(result.get("verification"), dict) else None),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "exit_code": 0 if result.get("ok") else 1,
    }
