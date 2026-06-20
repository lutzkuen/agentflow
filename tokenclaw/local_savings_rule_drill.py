from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import yaml

SCHEMA = "agentflow.local_savings_rule_drill.v1"
FIXTURE_RULE_ID = "fixture-openai-routing-rollback-drill"
FIXTURE_REQUESTED_MODEL = "gpt-5.4"
FIXTURE_TARGET_MODEL = "gpt-5.4-mini"


def _privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "local_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _clean_routing_yaml() -> dict[str, Any]:
    return {
        "rules": [],
        "phase_canary": {"enabled": False},
        "openai_canary": {
            "enabled": False,
            "policy_id": "fixture-openai-canary-disabled",
            "model_pattern": FIXTURE_REQUESTED_MODEL,
            "target_model": FIXTURE_TARGET_MODEL,
            "eligible_categories": ["chat"],
            "excluded_categories": [],
            "allow_tools": False,
            "allow_stream": False,
            "canary_fraction": 0.0,
            "holdout_fraction": 0.0,
            "safety_stop": {"enabled": False},
        },
    }


def _routing_drill_patch() -> dict[str, Any]:
    clean = _clean_routing_yaml()
    clean["rules"] = [
        {
            "id": FIXTURE_RULE_ID,
            "policy_source": "local-manual",
            "conditions": {
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "model_pattern": FIXTURE_REQUESTED_MODEL,
                "category": "short-completion",
                "has_tools": False,
                "stream": False,
                "text_chars_lt": 4096,
            },
            "action": {
                "route_to": FIXTURE_TARGET_MODEL,
                "reason": "fixture local savings routing drill",
            },
        }
    ]
    return clean


def _fixture_request() -> dict[str, Any]:
    return {
        "model": FIXTURE_REQUESTED_MODEL,
        "stream": False,
        "input": "AgentFlow local savings rollback drill fixture.",
        "_agentflow_source_surface": "openai_responses",
        "_agentflow_endpoint": "responses",
    }


def _file_sha(path: Path) -> str | None:
    from tokenclaw.policy_files import policy_file_snapshot

    snapshot = policy_file_snapshot(path)
    return str(snapshot.get("sha256")) if snapshot.get("sha256") else None


def _decision_state(meta: dict[str, Any], requested: str, routed: str) -> str:
    if routed != requested:
        return "applied"
    if meta.get("enabled") is False:
        return "pass-through"
    status = ((meta.get("openai_canary") or {}).get("status") if isinstance(meta.get("openai_canary"), dict) else None)
    if status in {"holdout", "not_selected", "ineligible", "noop"}:
        return "blocked"
    return "pass-through"


def _observe_decision() -> dict[str, Any]:
    from tokenclaw import router

    body = _fixture_request()
    requested = str(body["model"])
    routed, meta = router.route_openai_model(body)
    return {
        "state": _decision_state(meta, requested, str(routed)),
        "requested_model": requested,
        "routed_model": str(routed),
        "rule_id": FIXTURE_RULE_ID if str(routed) == FIXTURE_TARGET_MODEL else None,
        "reason": meta.get("reason"),
        "policy_source": meta.get("policy_source"),
        "provider": "openai",
        "source_surface": "openai_responses",
        "endpoint": "responses",
        "category": meta.get("category"),
        "has_tools": bool(meta.get("has_tools")),
        "stream": bool(meta.get("stream")),
    }


async def _run_drill() -> dict[str, Any]:
    from tokenclaw.admin import reload_policy_modules
    from tokenclaw.policy_files import stage_policy_draft
    from tokenclaw.policy_workbench import apply_validated_policy_draft, rollback_policy_apply

    try:
        with tempfile.TemporaryDirectory(prefix="agentflow-rule-drill-") as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            workspace = root / "drafts"
            event_log = root / "policy-events.jsonl"
            config_dir.mkdir(parents=True, exist_ok=True)
            routing_path = config_dir / "routing_rules.yaml"
            routing_path.write_text(yaml.safe_dump(_clean_routing_yaml(), sort_keys=False), encoding="utf-8")

            env = {
                "AGENTFLOW_CONFIG_DIR": str(config_dir),
                "AGENTFLOW_POLICY_CONFIG_DIR": str(config_dir),
                "AGENTFLOW_POLICY_EVENTS_LOG": str(event_log),
                "AGENTFLOW_ROUTING_RULES": None,
                "AGENTFLOW_ROUTING": "1",
                "AGENTFLOW_OPENAI_ROUTING": "0",
            }
            with _temporary_env(env):
                await reload_policy_modules()
                before_sha = _file_sha(routing_path)
                before = _observe_decision()

                stage = await stage_policy_draft(
                    _routing_drill_patch(),
                    section="routing",
                    draft_id="local-savings-rule-rollback-drill",
                    workspace=workspace,
                    metadata={
                        "source": "local-savings-rule-drill",
                        "issue": 668,
                        "privacy": _privacy(),
                    },
                )
                if not stage.get("ok"):
                    return _blocked("stage_failed", before=before, stage=stage)

                apply_result = await apply_validated_policy_draft(
                    str(stage["workspace"]),
                    config_dir=config_dir,
                    sections=["routing"],
                    reload_policy_state=reload_policy_modules,
                    event_source="local-savings-rule-drill",
                    loopback_admin_calls_made=False,
                )
                after_apply_sha = _file_sha(routing_path)
                after_apply = _observe_decision()

                rollback_result = await rollback_policy_apply(
                    str(apply_result.get("apply_id") or ""),
                    config_dir=config_dir,
                    sections=["routing"],
                    reload_policy_state=reload_policy_modules,
                    event_source="local-savings-rule-drill",
                    loopback_admin_calls_made=False,
                )
                after_rollback_sha = _file_sha(routing_path)
                after_rollback = _observe_decision()
    finally:
        await reload_policy_modules()

    applied = bool(apply_result.get("ok") and after_apply["state"] == "applied")
    rollback_available = bool((apply_result.get("rollback") or {}).get("backup_id"))
    rollback_success = bool(
        rollback_result.get("ok")
        and after_rollback["state"] != "applied"
        and before_sha is not None
        and after_rollback_sha == before_sha
    )
    ok = bool(applied and rollback_available and rollback_success)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "status": "passed" if ok else "failed",
        "rule_family": "routing",
        "rule_id": FIXTURE_RULE_ID,
        "provider": "openai",
        "surface": "openai_responses",
        "managed_server_required": False,
        "provider_calls_made": False,
        "mocked_request": True,
        "applied": applied,
        "rollback_available": rollback_available,
        "rollback_success": rollback_success,
        "before_decision_state": before["state"],
        "after_apply_decision_state": after_apply["state"],
        "after_rollback_decision_state": after_rollback["state"],
        "decisions": {
            "before": before,
            "after_apply": after_apply,
            "after_rollback": after_rollback,
        },
        "policy_snapshot": {
            "section": "routing",
            "before_sha256": before_sha,
            "after_apply_sha256": after_apply_sha,
            "after_rollback_sha256": after_rollback_sha,
            "restored_previous_snapshot": bool(before_sha is not None and after_rollback_sha == before_sha),
        },
        "lifecycle": {
            "stage_status": stage.get("status") or ("staged" if stage.get("ok") else "failed"),
            "apply_status": apply_result.get("status"),
            "rollback_status": rollback_result.get("status"),
            "apply_id_present": bool(apply_result.get("apply_id")),
            "changed_sections": apply_result.get("changed_sections", []),
            "restored_sections": rollback_result.get("restored_sections", []),
            "verification": {
                "apply_ok": bool((apply_result.get("verification") or {}).get("ok", apply_result.get("ok"))),
                "rollback_ok": bool((rollback_result.get("verification") or {}).get("ok", rollback_result.get("ok"))),
            },
        },
        "privacy": _privacy(),
    }


def _blocked(reason: str, *, before: dict[str, Any] | None = None, stage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "blocked",
        "rule_family": "routing",
        "rule_id": FIXTURE_RULE_ID,
        "provider": "openai",
        "surface": "openai_responses",
        "managed_server_required": False,
        "provider_calls_made": False,
        "mocked_request": True,
        "applied": False,
        "rollback_available": False,
        "rollback_success": False,
        "before_decision_state": (before or {}).get("state"),
        "after_apply_decision_state": None,
        "after_rollback_decision_state": None,
        "decisions": {"before": before or {}, "after_apply": {}, "after_rollback": {}},
        "policy_snapshot": {
            "section": "routing",
            "before_sha256": None,
            "after_apply_sha256": None,
            "after_rollback_sha256": None,
            "restored_previous_snapshot": False,
        },
        "lifecycle": {
            "stage_status": (stage or {}).get("status"),
            "apply_status": None,
            "rollback_status": None,
            "apply_id_present": False,
            "changed_sections": [],
            "restored_sections": [],
            "verification": {"apply_ok": False, "rollback_ok": False},
        },
        "privacy": _privacy(),
        "error": {
            "type": reason,
            "message": "local savings rule drill could not complete",
        },
    }


def build_local_savings_rule_drill_summary() -> dict[str, Any]:
    """Run a no-provider apply/observe/rollback drill against temporary local policy files."""

    return asyncio.run(_run_drill())
