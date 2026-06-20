from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from tokenclaw.paths import safe_expanduser
from tokenclaw.policy_files import POLICY_DRAFT_SECTION_FILES, stage_policy_draft
from tokenclaw.policy_workbench import apply_validated_policy_draft
from tokenclaw.post_promotion_policy_drafts import DRAFT_SCHEMA
from tokenclaw.store import utc_now


SCHEMA = "agentflow.post_promotion_policy_draft_apply.v1"
OUTCOME_SCHEMA = "agentflow.post_promotion_policy_draft_apply_outcome.v1"

_SUPPORTED_SECTION_PATHS = {
    "routing": "routing.rules",
    "cache": "cache.pattern_rules",
    "crunch": "anthropic_thinking_history_compaction.rules",
}


def _privacy(*, wrote: bool, dry_run: bool) -> dict[str, Any]:
    return {
        "local_only": True,
        "metadata_only": True,
        "feature_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_enforced": False,
        "dry_run": bool(dry_run),
        "policy_files_written": bool(wrote),
        "wrote_local_policy_files": bool(wrote),
    }


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _safe_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned or fallback)[:80]


def _section_for_draft(draft: dict[str, Any]) -> str | None:
    family = str(draft.get("action_family") or "").strip()
    if family in _SUPPORTED_SECTION_PATHS:
        return family
    section = str(draft.get("target_local_policy_section") or "").strip()
    for candidate, supported in _SUPPORTED_SECTION_PATHS.items():
        if section == supported:
            return candidate
    return None


def _load_section_payload(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    parsed = yaml.safe_load(raw) if raw.strip() else {}
    return parsed if isinstance(parsed, dict) else {}


def _rule_list(payload: dict[str, Any], section: str) -> tuple[list[Any] | None, str]:
    if section == "routing":
        rules = payload.setdefault("rules", [])
        return (rules if isinstance(rules, list) else None), "$.rules"
    if section == "cache":
        rules = payload.setdefault("pattern_rules", [])
        return (rules if isinstance(rules, list) else None), "$.pattern_rules"
    if section == "crunch":
        compaction = payload.setdefault("anthropic_thinking_history_compaction", {})
        if not isinstance(compaction, dict):
            return None, "$.anthropic_thinking_history_compaction"
        rules = compaction.setdefault("rules", [])
        return (rules if isinstance(rules, list) else None), "$.anthropic_thinking_history_compaction.rules"
    return None, "$"


def _selector_values(draft: dict[str, Any]) -> set[str]:
    patch = draft.get("proposed_policy_patch") if isinstance(draft.get("proposed_policy_patch"), dict) else {}
    selector = patch.get("target_rule_selector") if isinstance(patch.get("target_rule_selector"), dict) else {}
    values = {
        draft.get("target_candidate_id"),
        draft.get("rule_id"),
        draft.get("draft_id"),
        selector.get("delta_id"),
        selector.get("candidate_id"),
        selector.get("rule_id"),
    }
    return {str(value).strip() for value in values if isinstance(value, str) and value.strip()}


def _rule_matches(rule: Any, selectors: set[str]) -> bool:
    if not isinstance(rule, dict):
        return False
    for key in ("id", "rule_id", "candidate_id", "action_id", "promotion_action_id", "target_candidate_id"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip() in selectors:
            return True
    return False


def _find_rule(rules: list[Any], selectors: set[str]) -> tuple[int | None, dict[str, Any] | None]:
    for index, rule in enumerate(rules):
        if _rule_matches(rule, selectors):
            return index, rule
    return None, None


def _canary_container(rule: dict[str, Any]) -> tuple[dict[str, Any], str]:
    for key in ("canary", "rollout"):
        value = rule.get(key)
        if isinstance(value, dict):
            return value, key
    rule["canary"] = {"enabled": True}
    return rule["canary"], "canary"


def _set_canary_fraction(container: dict[str, Any], key_name: str, fraction: float) -> None:
    if key_name == "rollout":
        container["canary_enabled"] = fraction > 0.0
    else:
        container["enabled"] = True
    container["canary_fraction"] = _bounded_fraction(fraction)


def _set_holdout_fraction(container: dict[str, Any], required: float) -> None:
    current = _as_float(container.get("holdout_fraction"), 0.0)
    container["holdout_fraction"] = _bounded_fraction(max(current, required))


def _attach_apply_metadata(rule: dict[str, Any], draft: dict[str, Any], *, apply_id: str, previous_rule: dict[str, Any]) -> None:
    rule["post_promotion_policy_apply"] = {
        "schema": "agentflow.post_promotion_policy_apply_rule_metadata.v1",
        "applied_at": utc_now(),
        "apply_id": apply_id,
        "draft_id": draft.get("draft_id"),
        "draft_action": draft.get("draft_action"),
        "target_candidate_id": draft.get("target_candidate_id"),
        "rollback_ready": True,
        "rollback_metadata": _json_clone(draft.get("rollback_metadata") if isinstance(draft.get("rollback_metadata"), dict) else {}),
        "previous_rule": previous_rule,
        "dry_run_impact_gate": _json_clone(draft.get("dry_run_impact_gate") if isinstance(draft.get("dry_run_impact_gate"), dict) else {}),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _blocker_for_draft(draft: Any) -> str | None:
    if not isinstance(draft, dict):
        return "invalid-draft"
    if draft.get("schema") != DRAFT_SCHEMA:
        return "unsupported-draft-schema"
    if draft.get("status") != "drafted":
        return "draft-not-ready"
    action = draft.get("draft_action")
    if action not in {"widen-local-policy", "rollback-local-policy"}:
        return "unsupported-draft-action"
    section = _section_for_draft(draft)
    if section is None:
        return "unsupported-policy-section"
    target_section = str(draft.get("target_local_policy_section") or "").strip()
    if target_section and target_section != _SUPPORTED_SECTION_PATHS[section]:
        return "unsupported-policy-section"
    gate = draft.get("dry_run_impact_gate")
    if not isinstance(gate, dict):
        return "missing-impact-gate"
    if gate.get("status") != "passed":
        return str(gate.get("reason") or "impact-gate-blocked")
    if gate.get("stale_evidence"):
        return "stale-evidence"
    if gate.get("safety_stop_active"):
        return "safety-stop-active"
    if action == "widen-local-policy" and not gate.get("holdout_coverage_present"):
        return "missing-holdout-coverage"
    rollback = draft.get("rollback_metadata")
    if not isinstance(rollback, dict):
        return "missing-rollback-metadata"
    if action == "rollback-local-policy" and not gate.get("preserved_previous_rule"):
        return "missing-preserved-previous-rule"
    return None


def _mutate_payload(payload: dict[str, Any], draft: dict[str, Any], *, apply_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    section = _section_for_draft(draft)
    if section is None:
        return None, {"type": "unsupported_policy_section", "message": "draft points at an unsupported local policy section"}
    rules, rules_path = _rule_list(payload, section)
    if rules is None:
        return None, {"type": "invalid_rule_file", "message": f"{rules_path} must be a list"}
    selectors = _selector_values(draft)
    _index, rule = _find_rule(rules, selectors)
    if rule is None:
        return None, {
            "type": "target_rule_not_found",
            "message": "no existing compatible local rule matched the post-promotion draft selector",
            "selector_values": sorted(selectors),
        }

    mutated = copy.deepcopy(payload)
    mutated_rules, _ = _rule_list(mutated, section)
    if mutated_rules is None:
        return None, {"type": "invalid_rule_file", "message": f"{rules_path} must be a list"}
    index, mutated_rule = _find_rule(mutated_rules, selectors)
    if index is None or mutated_rule is None:
        return None, {"type": "target_rule_not_found", "message": "target rule disappeared during mutation"}

    previous_rule = _json_clone(mutated_rule)
    gate = draft.get("dry_run_impact_gate") if isinstance(draft.get("dry_run_impact_gate"), dict) else {}
    action = str(draft.get("draft_action") or "")
    canary, canary_key = _canary_container(mutated_rule)
    if action == "widen-local-policy":
        projected = _bounded_fraction(gate.get("projected_canary_fraction"), _as_float(canary.get("canary_fraction"), 0.0))
        _set_canary_fraction(canary, canary_key, projected)
        _set_holdout_fraction(canary, _bounded_fraction(gate.get("required_holdout_fraction"), 0.0))
        mutated_rule["enabled"] = True
        mutated_rule["policy_source"] = "local-manual"
    elif action == "rollback-local-policy":
        _set_canary_fraction(canary, canary_key, 0.0)
        mutated_rule["enabled"] = False
        mutated_rule["policy_source"] = "local-manual"
    else:
        return None, {"type": "unsupported_draft_action", "message": "unsupported post-promotion draft action"}

    _attach_apply_metadata(mutated_rule, draft, apply_id=apply_id, previous_rule=previous_rule)
    return mutated, None


def _outcome(
    *,
    draft: Any,
    status: str,
    ok: bool,
    dry_run: bool,
    apply_id: str | None = None,
    section: str | None = None,
    stage: dict[str, Any] | None = None,
    apply: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": OUTCOME_SCHEMA,
        "ok": bool(ok),
        "status": status,
        "draft_id": draft.get("draft_id") if isinstance(draft, dict) else None,
        "draft_action": draft.get("draft_action") if isinstance(draft, dict) else None,
        "target_candidate_id": draft.get("target_candidate_id") if isinstance(draft, dict) else None,
        "action_family": draft.get("action_family") if isinstance(draft, dict) else None,
        "section": section,
        "apply_id": apply_id,
        "stage": stage,
        "apply": apply,
        "error": error,
        "privacy": _privacy(wrote=bool(apply and apply.get("ok")), dry_run=dry_run),
    }


async def apply_post_promotion_policy_drafts(
    policy_draft_report: Any,
    *,
    config_dir: str | Path | None = None,
    workspace: str | Path | None = None,
    dry_run: bool = False,
    max_apply: int = 20,
    reload_policy_state: Callable[[], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    config_path = safe_expanduser(config_dir) if config_dir is not None else safe_expanduser("~/.agentflow")
    workspace_path = safe_expanduser(workspace) if workspace is not None else safe_expanduser("~/.agentflow/policy_drafts")
    if not isinstance(policy_draft_report, dict):
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "invalid",
            "generated_at": utc_now(),
            "summary": {"draft_count": 0, "applied_count": 0, "blocked_count": 0, "dry_run_count": 0},
            "outcomes": [],
            "privacy": _privacy(wrote=False, dry_run=dry_run),
            "error": {"type": "invalid_report", "message": "post-promotion policy draft report must be a JSON object"},
        }
    drafts = policy_draft_report.get("drafts")
    if not isinstance(drafts, list):
        return {
            "schema": SCHEMA,
            "ok": False,
            "status": "invalid",
            "generated_at": utc_now(),
            "summary": {"draft_count": 0, "applied_count": 0, "blocked_count": 0, "dry_run_count": 0},
            "outcomes": [],
            "privacy": _privacy(wrote=False, dry_run=dry_run),
            "error": {"type": "invalid_report", "message": "post-promotion policy draft report must include a drafts list"},
        }

    outcomes: list[dict[str, Any]] = []
    for index, draft in enumerate(drafts[: max(0, min(int(max_apply or 0), 100))]):
        blocker = _blocker_for_draft(draft)
        section = _section_for_draft(draft) if isinstance(draft, dict) else None
        draft_id = draft.get("draft_id") if isinstance(draft, dict) else f"index-{index}"
        apply_id = f"post-promotion-{_safe_id(draft_id, f'draft-{index}')}"
        if blocker is not None:
            outcomes.append(_outcome(
                draft=draft,
                status="blocked",
                ok=False,
                dry_run=dry_run,
                apply_id=apply_id,
                section=section,
                error={"type": blocker.replace("-", "_"), "message": blocker},
            ))
            continue
        if section is None:
            outcomes.append(_outcome(
                draft=draft,
                status="blocked",
                ok=False,
                dry_run=dry_run,
                apply_id=apply_id,
                section=None,
                error={"type": "unsupported_policy_section", "message": "unsupported policy section"},
            ))
            continue

        path = config_path / POLICY_DRAFT_SECTION_FILES[section]
        payload = _load_section_payload(path)
        mutated, error = _mutate_payload(payload, draft, apply_id=apply_id)
        if error is not None or mutated is None:
            outcomes.append(_outcome(
                draft=draft,
                status="blocked",
                ok=False,
                dry_run=dry_run,
                apply_id=apply_id,
                section=section,
                error=error,
            ))
            continue

        stage = await stage_policy_draft(
            mutated,
            section=section,
            draft_id=apply_id,
            workspace=workspace_path,
            metadata={
                "schema": "agentflow.post_promotion_policy_apply_stage_metadata.v1",
                "source": "post-promotion-policy-draft-apply",
                "post_promotion_draft_id": draft.get("draft_id"),
                "draft_action": draft.get("draft_action"),
                "target_candidate_id": draft.get("target_candidate_id"),
                "dry_run": bool(dry_run),
                "provider_calls_made": False,
                "managed_server_calls_made": False,
            },
        )
        if not stage.get("ok"):
            outcomes.append(_outcome(
                draft=draft,
                status="blocked",
                ok=False,
                dry_run=dry_run,
                apply_id=apply_id,
                section=section,
                stage=stage,
                error=stage.get("error") if isinstance(stage.get("error"), dict) else {"type": "stage_failed", "message": "policy draft stage failed"},
            ))
            continue
        if dry_run:
            outcomes.append(_outcome(
                draft=draft,
                status="dry-run",
                ok=True,
                dry_run=True,
                apply_id=apply_id,
                section=section,
                stage=stage,
            ))
            continue

        applied = await apply_validated_policy_draft(
            str(stage.get("draft_id") or apply_id),
            workspace=workspace_path,
            config_dir=config_path,
            sections=[section],
            reload_policy_state=reload_policy_state,
            apply_id=apply_id,
            event_source="post-promotion-policy-draft-apply",
        )
        outcomes.append(_outcome(
            draft=draft,
            status="applied" if applied.get("ok") else str(applied.get("status") or "blocked"),
            ok=bool(applied.get("ok")),
            dry_run=False,
            apply_id=apply_id,
            section=section,
            stage=stage,
            apply=applied,
            error=applied.get("error") if isinstance(applied.get("error"), dict) else None,
        ))

    applied_count = sum(1 for outcome in outcomes if outcome.get("status") == "applied" and outcome.get("ok"))
    dry_run_count = sum(1 for outcome in outcomes if outcome.get("status") == "dry-run" and outcome.get("ok"))
    blocked_count = sum(1 for outcome in outcomes if not outcome.get("ok"))
    ok = (dry_run_count > 0 if dry_run else applied_count > 0) and blocked_count == 0
    return {
        "schema": SCHEMA,
        "ok": ok,
        "status": "dry-run" if dry_run and dry_run_count else "applied" if applied_count else "blocked" if blocked_count else "no-op",
        "generated_at": utc_now(),
        "source_report_schema": policy_draft_report.get("schema"),
        "source_report_generated_at": policy_draft_report.get("generated_at"),
        "summary": {
            "draft_count": len(drafts),
            "processed_count": len(outcomes),
            "applied_count": applied_count,
            "dry_run_count": dry_run_count,
            "blocked_count": blocked_count,
            "target_local_rule_files": sorted({
                POLICY_DRAFT_SECTION_FILES[str(outcome.get("section"))]
                for outcome in outcomes
                if outcome.get("section") in POLICY_DRAFT_SECTION_FILES
            }),
        },
        "outcomes": outcomes,
        "wrote_active_policy_files": bool(applied_count),
        "wrote_local_policy_files": bool(applied_count),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_enforced": False,
        "privacy": _privacy(wrote=bool(applied_count), dry_run=dry_run),
        "error": None if ok else {"type": "post_promotion_apply_blocked", "message": "one or more post-promotion policy drafts were not applied"},
    }
