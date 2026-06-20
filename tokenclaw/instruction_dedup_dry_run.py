from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from tokenclaw.instruction_dedup_report import (
    MIN_SECTION_CHARS as REPORT_MIN_SECTION_CHARS,
    _app_family,
    _as_float,
    _as_int,
    _body_instruction_features,
    _endpoint,
    _instruction_sections_from_body,
    _json_obj,
    _model_family,
    _privacy_summary,
    _source_surface,
    _text_bucket,
    _workflow_phase,
)
from tokenclaw.limiter import model_tier
from tokenclaw.policy_files import policy_file_status
from tokenclaw.pricing import estimate_cost
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


SCHEMA = "agentflow.instruction_dedup_dry_run.v1"
PLAN_SCHEMA = "agentflow.instruction_dedup_plan.v1"
TOKEN_CHARS = 4


def _hash_json(value: Any, *, length: int = 16) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _public_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{_hash_json(value, length=20)}"


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _breakdown(counter: dict[str, int], key_name: str = "value") -> list[dict[str, Any]]:
    return [
        {key_name: key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    text = public_label(key, "unknown")
    counter[text] = counter.get(text, 0) + amount


def _candidate_id(value: Any, fallback: str = "unknown") -> str | None:
    if value in (None, ""):
        return None
    return public_id(value, prefix="instruction-dedup-candidate", fallback=fallback)


def _canary_validation_errors(raw_canary: Any) -> list[str]:
    if raw_canary is None:
        return []
    if not isinstance(raw_canary, dict):
        return ["invalid-canary-configuration"]
    errors: list[str] = []

    def parse_unit_interval(key: str) -> float | None:
        if raw_canary.get(key) is None:
            return None
        try:
            value = float(raw_canary.get(key))
        except (TypeError, ValueError):
            errors.append(f"invalid-canary-{key.replace('_', '-')}")
            return None
        if value < 0.0 or value > 1.0:
            errors.append(f"invalid-canary-{key.replace('_', '-')}")
        return value

    fraction = parse_unit_interval("fraction")
    if fraction is None:
        fraction = parse_unit_interval("canary_fraction")
    if fraction is None:
        fraction = parse_unit_interval("rollout_fraction")
    holdout_fraction = parse_unit_interval("holdout_fraction")
    if fraction is not None and holdout_fraction is not None and 0.0 <= fraction <= 1.0 and 0.0 <= holdout_fraction <= 1.0:
        if fraction + holdout_fraction > 1.0:
            errors.append("invalid-canary-fraction-sum")
    return sorted(set(errors))


def _body_has_type(value: Any, names: set[str]) -> bool:
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").strip().lower()
        if block_type in names:
            return True
        return any(_body_has_type(child, names) for child in value.values())
    if isinstance(value, list):
        return any(_body_has_type(item, names) for item in value)
    return False


def _body_has_key(value: Any, names: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in names:
                return True
            if _body_has_key(child, names):
                return True
    elif isinstance(value, list):
        return any(_body_has_key(item, names) for item in value)
    return False


def _has_tool_protocol(body: dict[str, Any]) -> bool:
    if body.get("tools") or body.get("tool_choice"):
        return True
    if _body_has_type(body, {"tool_use", "tool_result", "function_call", "function_call_output"}):
        return True
    return _body_has_key(body, {"tool_calls", "tool_call_id", "function_call", "tool_use_id"})


def _has_thinking(body: dict[str, Any]) -> bool:
    if body.get("thinking"):
        return True
    return _body_has_type(body, {"thinking"})


def _source_allowed(value: str, allowed: list[Any]) -> bool:
    if not allowed:
        return True
    return value in {str(item) for item in allowed}


def _matches_rule(plan_basis: dict[str, Any], section_fingerprint: str, rule: dict[str, Any]) -> bool:
    if not _safe_bool(rule.get("enabled"), True):
        return False
    fingerprints = {str(item) for item in rule.get("instruction_section_fingerprints") or []}
    if fingerprints and section_fingerprint not in fingerprints:
        return False
    if not _source_allowed(str(plan_basis.get("source_surface") or ""), rule.get("source_surfaces") or []):
        return False
    if not _source_allowed(str(plan_basis.get("category") or ""), rule.get("categories") or []):
        return False
    if not _source_allowed(str(plan_basis.get("workflow_phase") or ""), rule.get("workflow_phases") or []):
        return False
    return True


def _base_rule(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(policy.get("rule_id") or "instruction-section-dedup-policy"),
        "enabled": _safe_bool(policy.get("enabled"), False),
        "policy_source": str(policy.get("policy_source") or "local-default"),
        "source_surfaces": [str(item) for item in policy.get("source_surfaces") or []],
        "categories": [str(item) for item in policy.get("categories") or []],
        "workflow_phases": [str(item) for item in policy.get("workflow_phases") or []],
        "instruction_section_fingerprints": [],
        "min_section_chars": _as_int(policy.get("min_section_chars"), 700),
        "min_repeated_count": _as_int(policy.get("min_repeated_count"), 2),
        "keep_recent_sections": _as_int(policy.get("keep_recent_sections"), 1),
        "replacement_notice": str(policy.get("replacement_notice") or "[repeated instruction section omitted by AgentFlow]"),
        "max_replacements": _as_int(policy.get("max_replacements"), 0),
        "block_tool_protocol": _safe_bool(policy.get("block_tool_protocol"), True),
        "block_tool_payloads": _safe_bool(policy.get("block_tool_payloads"), True),
        "block_responses": _safe_bool(policy.get("block_responses"), True),
        "block_thinking": _safe_bool(policy.get("block_thinking"), True),
        "canary": copy.deepcopy(policy.get("canary") if isinstance(policy.get("canary"), dict) else {}),
        "safety_stop": copy.deepcopy(policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {}),
    }


def _select_rule(policy: dict[str, Any], plan_basis: dict[str, Any], section_fingerprint: str) -> dict[str, Any] | None:
    for rule in policy.get("rules") or []:
        if isinstance(rule, dict) and _matches_rule(plan_basis, section_fingerprint, rule):
            return copy.deepcopy(rule)
    base = _base_rule(policy)
    if not _matches_rule(plan_basis, section_fingerprint, base):
        return None
    return base


def _cohort(rule: dict[str, Any], section_fingerprint: str, plan_basis: dict[str, Any], *, local_salt: str | None) -> dict[str, Any]:
    raw_canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    enabled = _safe_bool(raw_canary.get("enabled"), True)
    validation_errors = _canary_validation_errors(raw_canary)
    fraction = max(0.0, min(1.0, _as_float(raw_canary.get("fraction"), 0.0)))
    holdout_raw = raw_canary.get("holdout_fraction")
    holdout_fraction = None if holdout_raw is None else max(0.0, min(1.0, _as_float(holdout_raw)))
    holdout = holdout_fraction if holdout_fraction is not None else 0.0
    if fraction + holdout > 1.0:
        fraction = max(0.0, 1.0 - holdout)
    basis = {
        "salt": local_salt or str(raw_canary.get("salt") or ""),
        "rule_id": rule.get("id"),
        "unit": raw_canary.get("unit") or "instruction_section_fingerprint",
        "fingerprint": section_fingerprint,
        "surface": plan_basis.get("source_surface"),
        "category": plan_basis.get("category"),
        "phase": plan_basis.get("workflow_phase"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if validation_errors:
        cohort = "invalid"
        selected = False
        is_holdout = False
    elif not enabled:
        cohort = "disabled"
        selected = False
        is_holdout = False
    elif score < holdout:
        cohort = "holdout"
        selected = False
        is_holdout = True
    elif score < holdout + fraction:
        cohort = "canary"
        selected = True
        is_holdout = False
    else:
        cohort = "not_selected"
        selected = False
        is_holdout = False
    return {
        "enabled": enabled,
        "cohort": cohort,
        "selected": selected,
        "holdout": is_holdout,
        "canary_fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "holdout_configured": holdout_fraction is not None,
        "cohort_key_hash": digest[:16],
        "cohort_score": round(score, 12),
        "cohort_basis": "public-metadata-plus-hidden-instruction-fingerprint",
        "salt_included": False,
        "fingerprint_included": False,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
    }


def _coordinator_compatibility(row_meta: dict[str, Any]) -> dict[str, Any]:
    decision = row_meta.get("optimization_coordinator")
    if not isinstance(decision, dict):
        return {
            "status": "unknown",
            "compatible": True,
            "reason_codes": [],
            "selected_family": None,
        }
    selected_raw = str(decision.get("selected_family") or decision.get("selected_action_family") or "none")
    selected = public_label(selected_raw, "unknown")
    reasons = [public_label(item, "sanitized-reason") for item in decision.get("reason_codes") or [] if str(item)]
    suppressed = decision.get("suppressed_families") if isinstance(decision.get("suppressed_families"), list) else []
    for item in suppressed:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "")
        if family in {"pattern_crunch", "prompt_role", "instruction_section_deduplication"} or "prompt_role" in family:
            reasons.extend(public_label(reason, "sanitized-reason") for reason in item.get("reason_codes") or [] if str(reason))
    if selected_raw not in {"none", "pattern_crunch", "prompt_role", "instruction_section_deduplication"}:
        reasons.append("conflicts-with-coordinator-selection")
        return {
            "status": "conflict",
            "compatible": False,
            "reason_codes": sorted(set(reasons)),
            "selected_family": selected,
        }
    if "coordinator-holdout" in reasons or "coordinator-canary-not-selected" in reasons:
        return {
            "status": "blocked",
            "compatible": False,
            "reason_codes": sorted(set(reasons)),
            "selected_family": selected,
        }
    return {
        "status": "compatible",
        "compatible": True,
        "reason_codes": sorted(set(reasons)),
        "selected_family": selected,
    }


def _replacement_preview(rule: dict[str, Any], before_chars: int, after_chars: int, saved_chars: int) -> dict[str, Any]:
    return {
        "redacted": True,
        "source_text_included": False,
        "replacement_text": "[redacted replacement notice]",
        "before_chars": before_chars,
        "after_chars": after_chars,
        "saved_chars": saved_chars,
    }


def _row_public_basis(row: dict[str, Any], *, provider: str, body: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    path = str(row.get("path") or "")
    source_surface = str(row.get("source_surface") or _source_surface(provider, path))
    endpoint = str(row.get("endpoint") or _endpoint(provider, path))
    requested_model = row.get("requested_model") or body.get("model")
    routed_model = row.get("routed_model") or requested_model
    category = str(row.get("category") or routing.get("category") or "unknown")
    phase = _workflow_phase(routing, category)
    input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    text_chars = _as_int(routing.get("text_chars")) or input_tokens * TOKEN_CHARS
    return {
        "provider": provider,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "app_family": _app_family(provider, requested_model, path, source_surface),
        "category": category,
        "workflow_phase": phase,
        "requested_model_family": _model_family(provider, requested_model, row.get("requested_model_family")),
        "routed_model_tier": model_tier(str(routed_model or requested_model or "")),
        "stream": bool(_as_int(row.get("stream"))),
        "text_bucket": _text_bucket(text_chars),
        "fingerprint_source": features.get("fingerprint_source"),
        "fingerprint_present": bool(features.get("fingerprint_present")),
    }


def _plan_for_section(
    *,
    row: dict[str, Any],
    provider: str,
    body: dict[str, Any],
    section: dict[str, Any],
    repeated_count: int,
    policy: dict[str, Any],
    reload_required: bool,
    local_salt: str | None,
) -> dict[str, Any]:
    features = _body_instruction_features(row.get("request_json"), provider=provider)
    basis = _row_public_basis(row, provider=provider, body=body, features=features)
    rule = _select_rule(policy, basis, str(section.get("fingerprint") or ""))
    before_chars = _as_int(section.get("chars"))
    replacement_notice = str((rule or policy).get("replacement_notice") or "[repeated instruction section omitted by AgentFlow]")
    after_chars = len(replacement_notice)
    saved_chars = max(0, before_chars - after_chars)
    saved_tokens = saved_chars // TOKEN_CHARS
    model = row.get("routed_model") or row.get("requested_model") or body.get("model") or ""
    projected_usd = estimate_cost(str(model), saved_tokens, 0, provider=provider if provider in {"anthropic", "openai"} else "openai") or 0.0
    blockers: set[str] = set()
    if not policy.get("enabled"):
        blockers.add("instruction-dedup-policy-disabled")
    if rule is None:
        blockers.add("no-matching-instruction-dedup-rule")
        rule = _base_rule(policy)
    if not _source_allowed(str(basis.get("source_surface") or ""), policy.get("source_surfaces") or []):
        blockers.add("unsafe-source-surface")
    if repeated_count < _as_int(rule.get("min_repeated_count"), _as_int(policy.get("min_repeated_count"), 2)):
        blockers.add("insufficient-repeated-instruction-rows")
    min_section_chars = _as_int(rule.get("min_section_chars"), _as_int(policy.get("min_section_chars"), REPORT_MIN_SECTION_CHARS))
    if before_chars < min_section_chars:
        blockers.add("instruction-section-below-min-chars")
    if saved_chars <= 0:
        blockers.add("no-instruction-dedup-savings-projected")
    if _as_int(row.get("status_code")) >= 400:
        blockers.add("error-response")
    if reload_required:
        blockers.add("stale-policy-reload-required")
    has_tools = _has_tool_protocol(body)
    has_thinking = _has_thinking(body)
    if has_tools and (_safe_bool(rule.get("block_tool_protocol"), True) or _safe_bool(rule.get("block_tool_payloads"), True)):
        blockers.add("tool-protocol-risk")
    if has_thinking and _safe_bool(rule.get("block_thinking"), True):
        blockers.add("thinking-content-risk")
    canary = _cohort(rule, str(section.get("fingerprint") or ""), basis, local_salt=local_salt)
    if not canary["holdout_configured"]:
        blockers.add("missing-holdout-configuration")
    if canary["holdout"]:
        blockers.add("instruction-dedup-holdout")
    if not canary.get("valid", True):
        blockers.add("invalid-canary-configuration")
    elif canary["cohort"] == "not_selected":
        blockers.add("instruction-dedup-canary-not-selected")
    elif canary["cohort"] == "disabled":
        blockers.add("instruction-dedup-canary-disabled")
    routing = _json_obj(row.get("routing_json"))
    compatibility = _coordinator_compatibility(routing)
    if not compatibility["compatible"]:
        blockers.add("coordinator-conflict")
    status = "actionable" if not blockers and canary["selected"] else "blocked"
    if status == "actionable" and _as_int(rule.get("max_replacements"), 0) <= 0:
        blockers.add("max-replacements-disabled")
        status = "blocked"
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": _public_id("instruction-dedup-plan", {
            "row": row.get("id"),
            "section": section.get("fingerprint"),
            "rule": rule.get("id"),
        }),
        "status": status,
        "blockers": sorted(blockers),
        "selected_rule_id": public_label(rule.get("id") or "instruction-section-dedup-policy", "instruction-section-dedup-policy"),
        "policy_source": public_label(rule.get("policy_source") or policy.get("policy_source") or "local-default", "unknown"),
        "candidate_id": _candidate_id(rule.get("candidate_id")),
        "source_surface": public_label(basis["source_surface"], "unknown"),
        "provider": public_label(provider, "unknown"),
        "endpoint": public_label(basis["endpoint"], "unknown"),
        "app_family": public_label(basis["app_family"], "unknown"),
        "category": public_label(basis["category"], "unknown"),
        "workflow_phase": public_label(basis["workflow_phase"], "unknown"),
        "requested_model_family": public_label(basis["requested_model_family"], "unknown"),
        "routed_model_tier": public_label(basis["routed_model_tier"], "unknown"),
        "text_bucket": basis["text_bucket"],
        "instruction_section": {
            "source_field": str(section.get("source_field") or "unknown"),
            "fingerprint_present": True,
            "fingerprint_included": False,
            "repeated_count": repeated_count,
            "raw_text_included": False,
        },
        "counts": {
            "before_chars": before_chars,
            "after_chars": after_chars,
            "saved_chars": saved_chars,
            "saved_tokens_est": saved_tokens,
            "projected_saved_usd": round(projected_usd, 6),
        },
        "canary": canary,
        "coordinator_compatibility": compatibility,
        "replacement_preview": _replacement_preview(rule, before_chars, after_chars, saved_chars),
        "mutation": {
            "dry_run_only": True,
            "request_body_changed": False,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "policy_file_changed": False,
        },
        "privacy": {
            "metadata_only_output": True,
            "raw_instruction_text_included": False,
            "instruction_section_fingerprint_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }


def _blocked_bodyless_plan(
    *,
    row: dict[str, Any],
    provider: str,
    policy: dict[str, Any],
    reload_required: bool,
) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    path = str(row.get("path") or "")
    source_surface = str(row.get("source_surface") or _source_surface(provider, path))
    endpoint = str(row.get("endpoint") or _endpoint(provider, path))
    category = str(row.get("category") or routing.get("category") or "unknown")
    blockers = {"request-body-unavailable"}
    if not policy.get("enabled"):
        blockers.add("instruction-dedup-policy-disabled")
    if not _source_allowed(source_surface, policy.get("source_surfaces") or []):
        blockers.add("unsafe-source-surface")
    if reload_required:
        blockers.add("stale-policy-reload-required")
    if _as_int(row.get("status_code")) >= 400:
        blockers.add("error-response")
    canary_policy = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    if canary_policy.get("holdout_fraction") is None:
        blockers.add("missing-holdout-configuration")
    if _canary_validation_errors(canary_policy):
        blockers.add("invalid-canary-configuration")
    compatibility = _coordinator_compatibility(routing)
    if not compatibility["compatible"]:
        blockers.add("coordinator-conflict")
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": _public_id("instruction-dedup-plan", {"row": row.get("id"), "body": "missing"}),
        "status": "blocked",
        "blockers": sorted(blockers),
        "selected_rule_id": None,
        "policy_source": public_label(policy.get("policy_source") or "local-default", "unknown"),
        "source_surface": public_label(source_surface, "unknown"),
        "provider": public_label(provider, "unknown"),
        "endpoint": public_label(endpoint, "unknown"),
        "app_family": public_label(_app_family(provider, row.get("requested_model"), path, source_surface), "unknown"),
        "category": public_label(category, "unknown"),
        "workflow_phase": public_label(_workflow_phase(routing, category), "unknown"),
        "instruction_section": {
            "source_field": None,
            "fingerprint_present": False,
            "fingerprint_included": False,
            "repeated_count": 0,
            "raw_text_included": False,
        },
        "counts": {
            "before_chars": 0,
            "after_chars": 0,
            "saved_chars": 0,
            "saved_tokens_est": 0,
            "projected_saved_usd": 0.0,
        },
        "canary": {
            "enabled": _safe_bool(canary_policy.get("enabled"), True),
            "cohort": "not_applicable",
            "selected": False,
            "holdout": False,
            "canary_fraction": _as_float(canary_policy.get("fraction"), 0.0),
            "holdout_fraction": canary_policy.get("holdout_fraction"),
            "holdout_configured": canary_policy.get("holdout_fraction") is not None,
            "cohort_key_hash": None,
            "cohort_score": None,
            "cohort_basis": "unavailable-without-request-body",
            "salt_included": False,
            "fingerprint_included": False,
            "valid": not _canary_validation_errors(canary_policy),
            "validation_errors": _canary_validation_errors(canary_policy),
        },
        "coordinator_compatibility": compatibility,
        "replacement_preview": {
            "redacted": True,
            "source_text_included": False,
            "replacement_text": None,
        },
        "mutation": {
            "dry_run_only": True,
            "request_body_changed": False,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "policy_file_changed": False,
        },
        "privacy": {
            "metadata_only_output": True,
            "raw_instruction_text_included": False,
            "instruction_section_fingerprint_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }


def _load_rows(store_obj: Any, limit: int) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 1000), 10_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, status_code,
                   input_tokens_est, actual_input_tokens, cost_est_usd, cost_baseline_usd,
                   category, crunch_json, routing_json, cache_json, request_json, session_id
            from calls
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]
    codex_rows = [
        {
            "id": row.get("id"),
            "created_at": row.get("created_at"),
            "path": "codex_app_server",
            "provider": "codex_app",
            "source_surface": "codex_app_server",
            "endpoint": "app_server",
            "requested_model": "codex-app",
            "routed_model": "codex-app",
            "status_code": 500 if row.get("error_code") is not None else 200,
            "input_tokens_est": max(0, _as_int(row.get("input_text_chars")) // TOKEN_CHARS),
            "actual_input_tokens": 0,
            "cost_est_usd": 0.0,
            "cost_baseline_usd": 0.0,
            "category": _json_obj(row.get("routing_json")).get("category") or row.get("method") or "unknown",
            "routing_json": row.get("routing_json"),
            "crunch_json": row.get("crunch_json"),
            "cache_json": row.get("cache_json"),
            "request_json": None,
            "session_id": row.get("session_id"),
        }
        for row in store_obj.conn.execute(
            """
            select id, created_at, direction, method, input_text_chars, params_chars,
                   error_code, routing_json, crunch_json, cache_json, metadata_json, session_id
            from codex_app_events
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]
    return rows + codex_rows


def _policy_reload_state(rule_path: str | None) -> dict[str, Any]:
    if not rule_path:
        return {"reload_required": False, "path_included": False}
    try:
        from tokenclaw import crunch

        status = policy_file_status(
            rule_path,
            loaded_at=crunch.CRUNCH_RULES_LOADED_AT,
            loaded_snapshot=crunch.CRUNCH_RULES_LOADED_FILE,
        )
        return {
            "reload_required": bool(status.get("reload_required")),
            "path_included": False,
            "loaded": {
                "exists": bool((status.get("loaded") or {}).get("exists")),
                "sha256_included": False,
            },
            "current": {
                "exists": bool((status.get("current") or {}).get("exists")),
                "sha256_included": False,
            },
        }
    except Exception:
        return {"reload_required": False, "path_included": False, "status_error": True}


def build_instruction_dedup_dry_run(
    store_obj: Any,
    *,
    limit: int = 1000,
    examples: int = 20,
    policy: dict[str, Any] | None = None,
    policy_source: str | None = None,
    rule_path: str | None = None,
    local_salt: str | None = None,
) -> dict[str, Any]:
    from tokenclaw import crunch

    active_policy = copy.deepcopy(policy if isinstance(policy, dict) else crunch.INSTRUCTION_SECTION_DEDUP_POLICY)
    active_source = str(policy_source or active_policy.get("policy_source") or crunch.CRUNCH_POLICY_SOURCE)
    active_rule_path = rule_path if rule_path is not None else getattr(crunch, "CRUNCH_RULES_PATH", None)
    reload_state = _policy_reload_state(active_rule_path)
    reload_required = bool(reload_state.get("reload_required"))
    rows = _load_rows(store_obj, limit)
    body_rows: list[tuple[dict[str, Any], str, dict[str, Any], list[dict[str, Any]]]] = []
    fingerprint_counts: dict[str, int] = {}
    for row in rows:
        provider = str(row.get("provider") or "anthropic").lower()
        if provider not in {"anthropic", "openai"}:
            continue
        body = _json_obj(row.get("request_json"))
        if not body:
            continue
        sections = _instruction_sections_from_body(body, provider=provider)
        body_rows.append((row, provider, body, sections))
        for section in sections:
            fingerprint = str(section.get("fingerprint") or "")
            if fingerprint:
                fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1

    plans: list[dict[str, Any]] = []
    for row, provider, body, sections in body_rows:
        for section in sections:
            plans.append(
                _plan_for_section(
                    row=row,
                    provider=provider,
                    body=body,
                    section=section,
                    repeated_count=fingerprint_counts.get(str(section.get("fingerprint") or ""), 0),
                    policy=active_policy,
                    reload_required=reload_required,
                    local_salt=local_salt,
                )
            )
    bodyless_rows = [row for row in rows if not _json_obj(row.get("request_json"))]
    for row in bodyless_rows[: max(0, int(examples or 20))]:
        provider = str(row.get("provider") or "anthropic").lower()
        plans.append(
            _blocked_bodyless_plan(
                row=row,
                provider=provider,
                policy=active_policy,
                reload_required=reload_required,
            )
        )

    plans.sort(
        key=lambda plan: (
            plan.get("status") == "actionable",
            _as_float((plan.get("counts") or {}).get("projected_saved_usd")),
            _as_int((plan.get("counts") or {}).get("saved_chars")),
        ),
        reverse=True,
    )
    sample_limit = max(1, min(int(examples or 20), 200))
    sample_plans = plans[:sample_limit]
    blocker_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    coordinator_counts: dict[str, int] = {}
    for plan in plans:
        _increment(status_counts, plan.get("status"))
        _increment(provider_counts, plan.get("provider"))
        _increment(source_counts, plan.get("source_surface"))
        _increment(cohort_counts, (plan.get("canary") or {}).get("cohort") or "not_applicable")
        _increment(coordinator_counts, (plan.get("coordinator_compatibility") or {}).get("status") or "unknown")
        for blocker in plan.get("blockers") or []:
            _increment(blocker_counts, blocker)

    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "lookback_call_limit": max(1, min(int(limit or 1000), 10_000)),
        "summary": {
            "scanned_row_count": len(rows),
            "body_on_row_count": len(body_rows),
            "body_off_row_count": len(bodyless_rows),
            "plan_count": len(plans),
            "actionable_plan_count": sum(1 for plan in plans if plan.get("status") == "actionable"),
            "blocked_plan_count": sum(1 for plan in plans if plan.get("status") != "actionable"),
            "projected_saved_chars": sum(_as_int((plan.get("counts") or {}).get("saved_chars")) for plan in plans),
            "projected_saved_tokens": sum(_as_int((plan.get("counts") or {}).get("saved_tokens_est")) for plan in plans),
            "projected_saved_usd": round(sum(_as_float((plan.get("counts") or {}).get("projected_saved_usd")) for plan in plans), 6),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "request_bodies_modified": False,
            "policy_files_changed": False,
        },
        "policy": {
            "enabled": bool(active_policy.get("enabled")),
            "policy_source": active_source,
            "rule_path_included": False,
            "reload_required": reload_required,
            "file": reload_state,
            "rule_count": len(active_policy.get("rules") or []),
            "source_surfaces": [public_label(item, "unknown") for item in active_policy.get("source_surfaces") or []],
            "min_section_chars": _as_int(active_policy.get("min_section_chars"), 700),
            "min_repeated_count": _as_int(active_policy.get("min_repeated_count"), 2),
            "max_replacements": _as_int(active_policy.get("max_replacements"), 0),
            "canary": {
                "enabled": _safe_bool((active_policy.get("canary") or {}).get("enabled"), True)
                if isinstance(active_policy.get("canary"), dict) else True,
                "fraction": _as_float((active_policy.get("canary") or {}).get("fraction"), 0.0)
                if isinstance(active_policy.get("canary"), dict) else 0.0,
                "holdout_fraction": (active_policy.get("canary") or {}).get("holdout_fraction")
                if isinstance(active_policy.get("canary"), dict) else None,
                "salt_included": False,
            },
        },
        "status_breakdown": _breakdown(status_counts, "status"),
        "provider_breakdown": _breakdown(provider_counts, "provider"),
        "source_surface_breakdown": _breakdown(source_counts, "source_surface"),
        "cohort_breakdown": _breakdown(cohort_counts, "cohort"),
        "coordinator_breakdown": _breakdown(coordinator_counts, "status"),
        "blocker_reason_breakdown": _breakdown(blocker_counts, "reason"),
        "plans": sample_plans,
        "privacy": {
            **_privacy_summary(),
            "metadata_only_output": True,
            "raw_bodies_read_locally": True,
            "raw_body_values_emitted": False,
            "raw_instruction_text_included": False,
            "instruction_section_fingerprints_included": False,
            "replacement_preview_redacted": True,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_changed": False,
            "request_bodies_modified": False,
        },
    }
