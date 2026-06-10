from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.policy_bundle import APPLY_POLICY_SECTIONS
from agentflow_proxy.policy_files import _draft_workspace_root, utc_now


POLICY_DRAFT_VALIDATE_SCHEMA = "agentflow.policy_draft_validate.v1"
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
        config_dir=config_dir or str(Path.home() / ".agentflow"),
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
