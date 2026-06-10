from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(mtime: float | None) -> str | None:
    if mtime is None:
        return None
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat()


def policy_file_snapshot(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    snapshot: dict[str, Any] = {
        "path": str(file_path),
        "exists": False,
        "is_file": False,
        "size": None,
        "mtime_ns": None,
        "mtime": None,
        "sha256": None,
    }
    try:
        stat = file_path.stat()
    except OSError as exc:
        snapshot["error"] = str(exc)
        return snapshot

    snapshot["exists"] = True
    snapshot["is_file"] = file_path.is_file()
    snapshot["size"] = int(stat.st_size)
    snapshot["mtime_ns"] = int(stat.st_mtime_ns)
    snapshot["mtime"] = _mtime_iso(stat.st_mtime)
    if snapshot["is_file"]:
        try:
            snapshot["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError as exc:
            snapshot["error"] = str(exc)
    return snapshot


def policy_file_status(
    path: str | Path,
    *,
    loaded_at: str,
    loaded_snapshot: dict[str, Any],
) -> dict[str, Any]:
    current = policy_file_snapshot(path)
    loaded_key = (
        bool(loaded_snapshot.get("exists")),
        bool(loaded_snapshot.get("is_file")),
        loaded_snapshot.get("size"),
        loaded_snapshot.get("mtime_ns"),
        loaded_snapshot.get("sha256"),
    )
    current_key = (
        bool(current.get("exists")),
        bool(current.get("is_file")),
        current.get("size"),
        current.get("mtime_ns"),
        current.get("sha256"),
    )
    return {
        "path": str(path),
        "loaded_at": loaded_at,
        "loaded": loaded_snapshot,
        "current": current,
        "reload_required": loaded_key != current_key,
    }


POLICY_DRAFT_SCHEMA = "agentflow.policy_draft.v1"
POLICY_DRAFT_STAGE_SCHEMA = "agentflow.policy_draft_stage.v1"
POLICY_DRAFT_SECTIONS = (
    "routing",
    "crunch",
    "cache",
    "routing_experiments",
    "codex_app",
)
POLICY_DRAFT_SECTION_FILES = {
    "routing": "routing_rules.yaml",
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
    "routing_experiments": "routing_experiments.yaml",
    "codex_app": "codex_app_rules.yaml",
}
_RAW_POLICY_PAYLOAD_KEYS = {
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
    "request_id",
    "request_ids",
    "response",
    "secret",
    "secrets",
    "session_id",
    "session_ids",
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
_RAW_POLICY_PAYLOAD_ALLOWED_KEYS = {
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "raw_tool_payloads_included",
    "raw_session_ids_included",
    "raw_request_ids_included",
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_draft_id(value: str | None, payload: Any) -> str:
    if value:
        cleaned = "".join(char for char in value if char.isalnum() or char in {"-", "_"}).strip("-_")
        if cleaned:
            return cleaned[:80]
    return f"draft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{_sha256_json(payload)[:12]}"


def _draft_workspace_root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return Path.home() / ".agentflow" / "policy_drafts"


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _raw_payload_errors(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{item_path}.{key}"
                lowered = str(key).strip().lower()
                if lowered in _RAW_POLICY_PAYLOAD_ALLOWED_KEYS:
                    walk(child, child_path)
                    continue
                if lowered in _RAW_POLICY_PAYLOAD_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "raw prompt, response, provider body, or replay payload fields are not accepted in policy drafts",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def parse_policy_payload(raw: str, *, path: str = "$") -> tuple[Any | None, dict[str, Any] | None]:
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, {
            "schema": "agentflow.policy_payload_parse_error.v1",
            "ok": False,
            "errors": [{"path": path, "message": f"invalid YAML/JSON: {exc}"}],
        }
    if parsed is None:
        parsed = {}
    return parsed, None


def _patch_section_policy(current_section: dict[str, Any], section: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    patched = _json_clone(current_section)
    if section == "routing":
        for key in ("enabled", "rules", "phase_canary"):
            if key in payload:
                patched[key] = payload[key]
        if isinstance(payload.get("openai_canary"), dict):
            patched.setdefault("openai", {})
            if isinstance(patched["openai"], dict):
                patched["openai"]["canary"] = payload["openai_canary"]
        if isinstance(payload.get("openai"), dict):
            patched["openai"] = payload["openai"]
        return patched
    if section == "routing_experiments":
        if "policy" in payload and isinstance(payload.get("policy"), dict):
            patched["policy"] = payload["policy"]
        else:
            patched["policy"] = payload
        for key in ("enabled", "policy_source"):
            if key in payload:
                patched[key] = payload[key]
        return patched
    for key, value in payload.items():
        patched[key] = value
    return patched


def build_policy_draft_bundle(current_bundle: dict[str, Any], payload: Any, *, section: str | None = None) -> dict[str, Any]:
    if section is None:
        return _json_clone(payload)
    if section not in POLICY_DRAFT_SECTIONS:
        raise ValueError(f"unknown policy draft section: {section}")
    draft = _json_clone(current_bundle)
    draft["generated_at"] = utc_now()
    draft["policies"][section] = _patch_section_policy(
        draft["policies"].get(section) if isinstance(draft["policies"].get(section), dict) else {},
        section,
        payload,
    )
    return draft


def _rule_ids(policy: Any) -> list[str]:
    if not isinstance(policy, dict):
        return []
    ids: list[str] = []
    rules = policy.get("rules")
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            value = rule.get("id") or rule.get("rule_id")
            if isinstance(value, str) and value.strip():
                ids.append(value)
            elif rule.get("conditions") or rule.get("action"):
                ids.append(f"index:{index}")
    summary = policy.get("old_context_summarization")
    if isinstance(summary, dict):
        value = summary.get("rule_id") or summary.get("id")
        if isinstance(value, str) and value.strip():
            ids.append(value)
    return ids


def _candidate_ids(value: Any) -> list[str]:
    ids: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "candidate_id" and isinstance(child, str) and child.strip():
                    ids.append(child)
                elif key == "candidate_ids" and isinstance(child, list):
                    ids.extend(str(candidate) for candidate in child if isinstance(candidate, str) and candidate.strip())
                else:
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return sorted(set(ids))


def _section_diff_summary(
    section: str,
    *,
    before_policy: dict[str, Any],
    after_policy: dict[str, Any],
    changes: list[dict[str, Any]],
    draft_file: Path,
) -> dict[str, Any]:
    active_file = before_policy.get("file") if isinstance(before_policy.get("file"), dict) else {}
    changed = bool(changes)
    target_file = before_policy.get("rule_path") or active_file.get("path")
    return {
        "section": section,
        "changed": changed,
        "change_count": len(changes),
        "policy_source_before": before_policy.get("policy_source"),
        "policy_source_after": after_policy.get("policy_source"),
        "target_file": target_file,
        "draft_file": str(draft_file),
        "active_file": active_file,
        "reload_required_after_apply": changed,
        "rule_ids": _rule_ids(after_policy),
        "candidate_ids": _candidate_ids(after_policy),
        "changes": changes,
    }


def _draft_error_result(
    *,
    error_type: str,
    message: str,
    draft_id: str | None,
    workspace: Path,
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": POLICY_DRAFT_STAGE_SCHEMA,
        "ok": False,
        "draft": None,
        "draft_id": draft_id,
        "workspace": str(workspace),
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "diff": None,
        "sections": [],
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


async def stage_policy_draft(
    payload: Any,
    *,
    section: str | None = None,
    draft_id: str | None = None,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    from agentflow_proxy.policy_bundle import build_policy_bundle, compare_policy_bundles, validate_policy_bundle
    from agentflow_proxy.policy_bundle import _policy_apply_yaml  # local file renderer used by apply dry-runs

    root = _draft_workspace_root(workspace)
    candidate_id = _safe_draft_id(draft_id, payload)
    if section is not None and section not in POLICY_DRAFT_SECTIONS:
        return _draft_error_result(
            error_type="invalid_section",
            message="unknown policy draft section",
            draft_id=candidate_id,
            workspace=root,
            errors=[{"path": "$.section", "message": f"expected one of {', '.join(POLICY_DRAFT_SECTIONS)}"}],
        )

    raw_errors = _raw_payload_errors(payload)
    if raw_errors:
        return _draft_error_result(
            error_type="raw_payload_rejected",
            message="policy draft contains raw prompt, response, provider body, or replay payload fields",
            draft_id=candidate_id,
            workspace=root,
            errors=raw_errors,
        )

    current = await build_policy_bundle()
    try:
        proposed = build_policy_draft_bundle(current, payload, section=section)
    except ValueError as exc:
        return _draft_error_result(
            error_type="invalid_section",
            message=str(exc),
            draft_id=candidate_id,
            workspace=root,
        )

    validation = validate_policy_bundle(proposed)
    if not validation["ok"]:
        return _draft_error_result(
            error_type="validation_failed",
            message="policy draft is invalid",
            draft_id=candidate_id,
            workspace=root,
            errors=validation.get("errors", []),
        ) | {"validation": validation}

    diff = compare_policy_bundles(current, proposed)
    draft_dir = root / candidate_id
    sections_dir = draft_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    draft_bundle_path = draft_dir / "policy_bundle.json"
    draft_manifest_path = draft_dir / "draft.json"
    draft_bundle_path.write_text(json.dumps(proposed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    changes_by_section: dict[str, list[dict[str, Any]]] = {section_name: [] for section_name in POLICY_DRAFT_SECTIONS}
    for change in diff.get("changes", []):
        if not isinstance(change, dict):
            continue
        path = str(change.get("path") or "")
        section_name = path.removeprefix("$.policies.").split(".", 1)[0].split("[", 1)[0]
        if section_name in changes_by_section:
            changes_by_section[section_name].append(change)

    sections: list[dict[str, Any]] = []
    for section_name in POLICY_DRAFT_SECTIONS:
        policy = proposed["policies"].get(section_name) if isinstance(proposed["policies"].get(section_name), dict) else {}
        yaml_payload = _policy_apply_yaml(section_name, policy)
        section_file = sections_dir / POLICY_DRAFT_SECTION_FILES[section_name]
        section_file.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
        sections.append(_section_diff_summary(
            section_name,
            before_policy=current["policies"].get(section_name) if isinstance(current["policies"].get(section_name), dict) else {},
            after_policy=policy,
            changes=changes_by_section[section_name],
            draft_file=section_file,
        ))

    manifest = {
        "schema": POLICY_DRAFT_SCHEMA,
        "draft_id": candidate_id,
        "created_at": utc_now(),
        "workspace": str(draft_dir),
        "bundle_path": str(draft_bundle_path),
        "requested_section": section,
        "changed": bool(diff.get("changed")),
        "changed_sections": diff.get("changed_sections", []),
        "change_count": diff.get("change_count", 0),
        "sections": sections,
        "privacy": {
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
    draft_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": POLICY_DRAFT_STAGE_SCHEMA,
        "ok": True,
        "draft": manifest,
        "draft_id": candidate_id,
        "workspace": str(draft_dir),
        "bundle_path": str(draft_bundle_path),
        "manifest_path": str(draft_manifest_path),
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "validation": validation,
        "diff": diff,
        "sections": sections,
        "error": None,
    }
