from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.local_compaction_canary_ramp.v1"
DECISION_SCHEMA = "tokenclaw.local_compaction_canary_ramp_decision.v1"
MANAGED_TREATMENT_SCHEMA = "tokenclaw.managed_thinking_compaction_treatment_apply.v1"
CRUNCH_RULES_FILE = "crunch_rules.yaml"
THINKING_SECTION = "anthropic_thinking_history_compaction"
THINKING_TARGET_CANDIDATE = "repeated-context-thinking-tool-result-gte-128k"
OLD_CONTEXT_SECTION = "old_context_summarization"


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 6)


def _privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_responses_included": False,
        "raw_thinking_text_included": False,
        "generated_summaries_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "managed_server_calls_made": False,
        "provider_calls_made": False,
    }


def _default_rules_path(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir).expanduser() / CRUNCH_RULES_FILE
    return Path.cwd() / "config" / CRUNCH_RULES_FILE


def _managed_rules_path(config_dir: str | Path | None = None, rules_path: str | Path | None = None) -> Path:
    if rules_path is not None:
        return Path(rules_path).expanduser()
    env_path = os.getenv("TOKENCLAW_CRUNCH_RULES")
    if env_path:
        return Path(env_path).expanduser()
    return _default_rules_path(config_dir)


def _package_rules_path() -> Path:
    return Path(__file__).parent / CRUNCH_RULES_FILE


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None, str]:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text) or {}
        return (parsed if isinstance(parsed, dict) else {}), text, str(path)
    package_path = _package_rules_path()
    text = package_path.read_text(encoding="utf-8") if package_path.exists() else ""
    parsed = yaml.safe_load(text) or {}
    return (parsed if isinstance(parsed, dict) else {}), None, str(package_path)


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def _diff(before: str | None, after: str, path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{path}.before",
            tofile=f"{path}.after",
        )
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _normalized_treatment(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "apply": "live",
        "applied": "live",
        "enforced": "live",
        "selected": "canary",
        "canary_applied": "canary",
        "dry_run": "observe",
        "dry-run": "observe",
        "observe_only": "observe",
        "noop": "none",
        "no_op": "none",
        "rolled_back": "rollback",
        "rollback_required": "rollback",
    }
    return aliases.get(text, text or "hold")


def _source_crunch_section(decision: dict[str, Any]) -> dict[str, Any]:
    section = decision.get("crunch")
    return section if isinstance(section, dict) else {}


def _managed_fraction(crunch: dict[str, Any], decision: dict[str, Any], key: str) -> float | None:
    for source in (crunch, decision):
        value = source.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        try:
            return _bounded_fraction(value)
        except (TypeError, ValueError):
            continue
    return None


def _rule_holdout_fraction(rule: dict[str, Any] | None, section: dict[str, Any]) -> float:
    for source in (rule, section):
        canary = source.get("canary") if isinstance(source, dict) and isinstance(source.get("canary"), dict) else {}
        if canary.get("holdout_fraction") is not None:
            return _bounded_fraction(canary.get("holdout_fraction"), 0.10)
    return 0.10


def _managed_controller_meta(
    *,
    decision: dict[str, Any],
    treatment: str,
    current_fraction: float,
    recommended_fraction: float,
    holdout_fraction: float,
    now: str,
) -> dict[str, Any]:
    crunch = _source_crunch_section(decision)
    reason_codes = crunch.get("reason_codes") if isinstance(crunch.get("reason_codes"), list) else decision.get("reason_codes")
    return {
        "schema": "tokenclaw.managed_thinking_compaction_treatment_metadata.v1",
        "updated_at": now,
        "decision": treatment,
        "reason_codes": [str(item) for item in reason_codes] if isinstance(reason_codes, list) else [],
        "previous_fraction": current_fraction,
        "recommended_fraction": recommended_fraction,
        "recommended_holdout_fraction": holdout_fraction,
        "policy_id": decision.get("policy_id") or crunch.get("policy_id"),
        "decision_id": decision.get("decision_id"),
        "candidate_id": crunch.get("candidate_id") or THINKING_TARGET_CANDIDATE,
        "policy_source": "managed-recommended",
        "server_traffic_treatment": treatment,
        "local_only": True,
        "metadata_only": True,
        "raw_content_included": False,
    }


def _apply_managed_thinking_edit(
    data: dict[str, Any],
    *,
    decision: dict[str, Any],
    treatment: str,
    recommended_fraction: float,
    holdout_fraction: float,
    now: str,
) -> None:
    rule, section = _current_thinking_rule(data)
    if rule is None:
        return
    current_fraction = _current_fraction(section, rule=rule)
    enabled = recommended_fraction > 0.0
    section["enabled"] = enabled
    section["policy_source"] = "managed-recommended"
    section["managed_controller"] = _managed_controller_meta(
        decision=decision,
        treatment=treatment,
        current_fraction=current_fraction,
        recommended_fraction=recommended_fraction,
        holdout_fraction=holdout_fraction,
        now=now,
    )
    parent_canary = section.setdefault("canary", {})
    if isinstance(parent_canary, dict):
        parent_canary["enabled"] = True
        parent_canary["canary_fraction"] = recommended_fraction
        parent_canary["holdout_fraction"] = holdout_fraction
        parent_canary.setdefault("canary_salt", "managed-thinking-compaction-treatment-v1")
        parent_canary.setdefault("canary_unit", "thinking_block_local_fingerprint")
    rule["enabled"] = enabled
    rule["policy_source"] = "managed-recommended"
    canary = rule.setdefault("canary", {})
    if isinstance(canary, dict):
        canary["enabled"] = True
        canary["canary_fraction"] = recommended_fraction
        canary["holdout_fraction"] = holdout_fraction
        canary.setdefault("canary_salt", "managed-thinking-compaction-treatment-v1")
        canary.setdefault("canary_unit", "thinking_block_local_fingerprint")
    safety = rule.setdefault("safety_stop", {})
    if isinstance(safety, dict) and treatment == "rollback":
        safety["last_managed_rollback_reason"] = "server-rollback"
        safety["last_managed_rollback_at"] = now
        safety["last_ramp_stop_reason"] = "server-rollback"
        safety["last_ramp_stop_at"] = now
    rule["managed_controller"] = _managed_controller_meta(
        decision=decision,
        treatment=treatment,
        current_fraction=current_fraction,
        recommended_fraction=recommended_fraction,
        holdout_fraction=holdout_fraction,
        now=now,
    )


def apply_managed_thinking_compaction_treatment(
    decision: dict[str, Any],
    *,
    apply: bool = False,
    config_dir: str | Path | None = None,
    rules_path: str | Path | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Apply a server-owned traffic treatment to the local thinking compaction rule."""
    target_path = _managed_rules_path(config_dir=config_dir, rules_path=rules_path)
    data, original_text, loaded_from = _load_yaml(target_path)
    proposed = copy.deepcopy(data)
    timestamp = now or utc_now()
    crunch = _source_crunch_section(decision)
    candidate_id = str(crunch.get("candidate_id") or decision.get("candidate_id") or "")
    treatment = _normalized_treatment(
        crunch.get("traffic_treatment")
        or crunch.get("server_traffic_treatment")
        or decision.get("traffic_treatment")
        or decision.get("server_traffic_treatment")
    )
    rule, section = _current_thinking_rule(proposed)
    current_fraction = _current_fraction(section, rule=rule) if rule is not None else 0.0
    current_holdout = _rule_holdout_fraction(rule, section)

    result = {
        "schema": MANAGED_TREATMENT_SCHEMA,
        "generated_at": timestamp,
        "ok": True,
        "apply": bool(apply),
        "target_rule_file": CRUNCH_RULES_FILE,
        "target_path": str(target_path),
        "loaded_from": loaded_from,
        "target_file_existed": original_text is not None,
        "candidate_id": candidate_id or THINKING_TARGET_CANDIDATE,
        "server_traffic_treatment": treatment,
        "current_fraction": current_fraction,
        "current_holdout_fraction": current_holdout,
        "recommended_fraction": current_fraction,
        "recommended_holdout_fraction": current_holdout,
        "changed": False,
        "status": "no-change",
        "reason": "server-held",
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
        "diff": "",
    }
    if candidate_id and candidate_id != THINKING_TARGET_CANDIDATE:
        result.update({"ok": False, "status": "unsupported", "reason": "unsupported-crunch-candidate"})
        return result
    if rule is None:
        result.update({"ok": False, "status": "unsupported", "reason": "target-rule-not-found"})
        return result
    if treatment in {"hold", "held", "observe", "none", "shadow", "holdout"}:
        result["reason"] = f"server-{treatment}"
        return result

    holdout = current_holdout
    if treatment in {"canary", "widen"}:
        fraction = _managed_fraction(crunch, decision, "canary_fraction")
        if fraction is None:
            fraction = current_fraction
        reason = "managed-crunch-canary-fraction"
    elif treatment == "live":
        fraction = _managed_fraction(crunch, decision, "canary_fraction")
        if fraction is None:
            fraction = 1.0
        server_holdout = _managed_fraction(crunch, decision, "holdout_fraction")
        holdout = 0.0 if server_holdout is None else server_holdout
        reason = "managed-crunch-live-treatment"
    elif treatment == "rollback":
        fraction = 0.0
        reason = "managed-crunch-rollback"
    else:
        result.update({"ok": False, "status": "unsupported", "reason": "unsupported-traffic-treatment"})
        return result

    if fraction + holdout > 1.0:
        holdout = max(0.0, 1.0 - fraction)
    result["recommended_fraction"] = fraction
    result["recommended_holdout_fraction"] = holdout
    result["reason"] = reason
    _apply_managed_thinking_edit(
        proposed,
        decision=decision,
        treatment=treatment,
        recommended_fraction=fraction,
        holdout_fraction=holdout,
        now=timestamp,
    )
    proposed_text = _dump_yaml(proposed)
    changed = proposed_text != (original_text or "")
    result["changed"] = changed
    result["diff"] = _diff(original_text, proposed_text, target_path) if changed else ""
    if apply and changed:
        _write_atomic(target_path, proposed_text)
        result["status"] = "applied"
        result["wrote_policy_files"] = True
    elif changed:
        result["status"] = "planned"
    else:
        result["status"] = "no-change"
    return result


def _rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    where = "where created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since, capped) if since else (capped,)
    sql = f"""
        select id, created_at, status_code, retry_count, latency_ms,
               cost_est_usd, cost_baseline_usd, actual_input_tokens,
               actual_output_tokens, input_tokens_est, cache_read_input_tokens,
               crunch_json, routing_json, cache_json
        from calls
        {where}
        order by created_at desc
        limit ?
    """
    return [dict(row) for row in reversed(store_obj.conn.execute(sql, params).fetchall())]


def _thinking_meta(crunch: dict[str, Any]) -> dict[str, Any]:
    meta = crunch.get(THINKING_SECTION)
    if not isinstance(meta, dict):
        return {}
    candidate = str(meta.get("candidate_id") or "")
    rule = str(meta.get("rule_id") or "")
    evaluated = meta.get("evaluated_rules") if isinstance(meta.get("evaluated_rules"), list) else []
    matched_target = candidate == THINKING_TARGET_CANDIDATE or any(
        isinstance(item, dict)
        and item.get("candidate_id") == THINKING_TARGET_CANDIDATE
        and item.get("status") == "matched"
        for item in evaluated
    )
    if matched_target or rule == "local-repeated-context-thinking-tool-result-canary":
        return meta
    return {}


def _old_context_meta(crunch: dict[str, Any]) -> dict[str, Any]:
    meta = crunch.get(OLD_CONTEXT_SECTION)
    return meta if isinstance(meta, dict) else {}


def _cohort(family: str, meta: dict[str, Any]) -> str:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    lifecycle = meta.get("lifecycle_feedback") if isinstance(meta.get("lifecycle_feedback"), dict) else {}
    status = str(meta.get("status") or lifecycle.get("status") or "").strip().lower()
    reason = str(meta.get("reason") or "").strip().lower()
    raw = str(canary.get("cohort") or canary.get("status") or lifecycle.get("cohort") or "").strip().lower()
    if raw in {"canary_applied", "applied"} or status == "applied" or bool(meta.get("applied")):
        return "applied"
    if raw in {"canary_holdout", "holdout"} or status == "holdout" or reason == "canary_holdout":
        return "holdout"
    if "safety-stop" in reason or status in {"safety_stop", "safety-stopped"} or meta.get("safety_stop_state") == "stopped":
        return "safety_stop"
    if family == OLD_CONTEXT_SECTION and status in {"summary_failed", "error"}:
        return "safety_stop"
    return "skipped"


def _realized_crunch_savings(crunch: dict[str, Any], meta: dict[str, Any], cohort: str) -> float:
    for value in (
        meta.get("realized_crunch_savings_usd"),
        (meta.get("realized_savings") or {}).get("realized_crunch_savings_usd") if isinstance(meta.get("realized_savings"), dict) else None,
        crunch.get("realized_crunch_savings_usd"),
        (crunch.get("realized_savings") or {}).get("realized_crunch_savings_usd") if isinstance(crunch.get("realized_savings"), dict) else None,
        (crunch.get("realized_savings_attribution") or {}).get("realized_crunch_savings_usd") if isinstance(crunch.get("realized_savings_attribution"), dict) else None,
    ):
        if value is not None:
            return max(0.0, _as_float(value))
    if cohort == "applied":
        return max(
            _as_float(meta.get("estimated_net_savings_usd")),
            _as_float(meta.get("estimated_gross_savings_usd")) - _as_float(meta.get("summary_cost_est_usd")),
            0.0,
        )
    return 0.0


def _planned_savings(meta: dict[str, Any]) -> float:
    return max(
        _as_float(meta.get("projected_savings_usd")),
        _as_float(meta.get("estimated_gross_savings_usd")) - _as_float(meta.get("summary_cost_est_usd")),
        _as_float(meta.get("estimated_net_savings_usd")),
        0.0,
    )


def _extract_similarity_values(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if "similarity" in lowered or "output_match" in lowered:
                number = _as_float(item, -1.0)
                if 0.0 <= number <= 1.0:
                    values.append(number)
                    continue
            values.extend(_extract_similarity_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_extract_similarity_values(item))
    return values


def _empty_family(family: str) -> dict[str, Any]:
    return {
        "family": family,
        "observed": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "safety_stop_count": 0,
        "applied_errors": 0,
        "holdout_errors": 0,
        "applied_retries": 0,
        "holdout_retries": 0,
        "applied_cost_usd": 0.0,
        "holdout_cost_usd": 0.0,
        "applied_realized_savings_usd": 0.0,
        "holdout_projected_savings_usd": 0.0,
        "similarity_samples": [],
        "min_similarity": None,
    }


def _add_sample(family: dict[str, Any], row: dict[str, Any], crunch: dict[str, Any], meta: dict[str, Any], cohort: str) -> None:
    family["observed"] += 1
    errored = _as_int(row.get("status_code"), -1) >= 400
    retried = _as_int(row.get("retry_count")) > 0
    cost = _as_float(row.get("cost_est_usd"))
    if cohort == "applied":
        family["applied_count"] += 1
        family["applied_errors"] += int(errored)
        family["applied_retries"] += int(retried)
        family["applied_cost_usd"] += cost
        family["applied_realized_savings_usd"] += _realized_crunch_savings(crunch, meta, cohort)
        values = _extract_similarity_values(meta)
        family["similarity_samples"].extend(values)
        if values:
            minimum = min(values)
            current = family.get("min_similarity")
            family["min_similarity"] = minimum if current is None else min(float(current), minimum)
    elif cohort == "holdout":
        family["holdout_count"] += 1
        family["holdout_errors"] += int(errored)
        family["holdout_retries"] += int(retried)
        family["holdout_cost_usd"] += cost
        family["holdout_projected_savings_usd"] += _planned_savings(meta)
    elif cohort == "safety_stop":
        family["safety_stop_count"] += 1
    else:
        family["skipped_count"] += 1


def _finalize_family(family: dict[str, Any]) -> dict[str, Any]:
    applied = _as_int(family.get("applied_count"))
    holdout = _as_int(family.get("holdout_count"))
    observed = _as_int(family.get("observed"))
    result = dict(family)
    result.pop("similarity_samples", None)
    result.update({
        "applied_error_rate": round(_as_int(family.get("applied_errors")) / applied, 6) if applied else 0.0,
        "holdout_error_rate": round(_as_int(family.get("holdout_errors")) / holdout, 6) if holdout else 0.0,
        "applied_retry_rate": round(_as_int(family.get("applied_retries")) / applied, 6) if applied else 0.0,
        "holdout_retry_rate": round(_as_int(family.get("holdout_retries")) / holdout, 6) if holdout else 0.0,
        "applied_cost_avg_usd": round(_as_float(family.get("applied_cost_usd")) / applied, 8) if applied else 0.0,
        "holdout_cost_avg_usd": round(_as_float(family.get("holdout_cost_usd")) / holdout, 8) if holdout else 0.0,
        "applied_realized_savings_avg_usd": round(_as_float(family.get("applied_realized_savings_usd")) / applied, 8) if applied else 0.0,
        "holdout_projected_savings_avg_usd": round(_as_float(family.get("holdout_projected_savings_usd")) / holdout, 8) if holdout else 0.0,
        "applied_rate": round(applied / observed, 6) if observed else 0.0,
        "holdout_rate": round(holdout / observed, 6) if observed else 0.0,
        "similarity_sample_count": len(family.get("similarity_samples") or []),
    })
    result["applied_minus_holdout_cost_avg_usd"] = round(result["applied_cost_avg_usd"] - result["holdout_cost_avg_usd"], 8)
    result["applied_minus_holdout_error_rate"] = round(result["applied_error_rate"] - result["holdout_error_rate"], 6)
    result["applied_minus_holdout_retry_rate"] = round(result["applied_retry_rate"] - result["holdout_retry_rate"], 6)
    for key in ("applied_cost_usd", "holdout_cost_usd", "applied_realized_savings_usd", "holdout_projected_savings_usd"):
        result[key] = round(_as_float(result.get(key)), 8)
    return result


def _evidence(store_obj: Any, *, limit: int, since: str | None) -> dict[str, dict[str, Any]]:
    families = {
        THINKING_SECTION: _empty_family(THINKING_SECTION),
        OLD_CONTEXT_SECTION: _empty_family(OLD_CONTEXT_SECTION),
    }
    for row in _rows(store_obj, limit=limit, since=since):
        crunch = _json_obj(row.get("crunch_json"))
        for family, extractor in ((THINKING_SECTION, _thinking_meta), (OLD_CONTEXT_SECTION, _old_context_meta)):
            meta = extractor(crunch)
            if not meta:
                continue
            cohort = _cohort(family, meta)
            _add_sample(families[family], row, crunch, meta, cohort)
    return {family: _finalize_family(value) for family, value in families.items()}


def _current_thinking_rule(data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    section = data.setdefault(THINKING_SECTION, {})
    if not isinstance(section, dict):
        data[THINKING_SECTION] = {}
        section = data[THINKING_SECTION]
    rules = section.setdefault("rules", [])
    if not isinstance(rules, list):
        section["rules"] = []
        rules = section["rules"]
    for rule in rules:
        if isinstance(rule, dict) and rule.get("candidate_id") == THINKING_TARGET_CANDIDATE:
            return rule, section
    return None, section


def _current_fraction(section: dict[str, Any], *, rule: dict[str, Any] | None = None) -> float:
    canary = (rule or section).get("canary") if isinstance((rule or section).get("canary"), dict) else {}
    return _bounded_fraction(canary.get("canary_fraction", canary.get("fraction", 0.0)))


def _decision(
    family: str,
    evidence: dict[str, Any],
    current_fraction: float,
    *,
    initial_fraction: float,
    ramp_step: float,
    max_fraction: float,
    holdout_fraction: float,
    min_applied_samples: int,
    min_holdout_samples: int,
    max_error_rate: float,
    max_error_rate_delta: float,
    max_retry_rate_delta: float,
    similarity_floor: float,
) -> dict[str, Any]:
    applied = _as_int(evidence.get("applied_count"))
    holdout = _as_int(evidence.get("holdout_count"))
    reasons: list[str] = []
    action = "hold"
    recommended = current_fraction

    if _as_int(evidence.get("safety_stop_count")) > 0:
        action = "stop"
        reasons.append("safety-stop-observed")
    if _as_float(evidence.get("applied_error_rate")) >= max_error_rate and applied:
        action = "stop"
        reasons.append("applied-error-rate-above-threshold")
    if _as_float(evidence.get("applied_minus_holdout_error_rate")) >= max_error_rate_delta and applied and holdout:
        action = "stop"
        reasons.append("applied-error-rate-regression")
    if _as_float(evidence.get("applied_minus_holdout_retry_rate")) >= max_retry_rate_delta and applied and holdout:
        action = "stop"
        reasons.append("applied-retry-rate-regression")
    min_similarity = evidence.get("min_similarity")
    if min_similarity is not None and _as_float(min_similarity, 1.0) < similarity_floor:
        action = "stop"
        reasons.append("output-similarity-floor-breach")

    if action == "stop":
        recommended = 0.0
    elif applied >= min_applied_samples and holdout >= min_holdout_samples:
        cheaper = (
            _as_float(evidence.get("applied_minus_holdout_cost_avg_usd")) < 0.0
            or _as_float(evidence.get("applied_realized_savings_avg_usd")) > 0.0
        )
        if cheaper:
            action = "widen"
            reasons.append("realized-canary-advantage")
            recommended = max(initial_fraction, current_fraction + ramp_step)
        else:
            action = "hold"
            reasons.append("missing-realized-canary-advantage")
    elif current_fraction <= 0.0 and holdout >= min_holdout_samples and _as_float(evidence.get("holdout_projected_savings_avg_usd")) > 0.0:
        action = "widen"
        reasons.append("initial-canary-enable-from-holdout-projection")
        recommended = initial_fraction
    else:
        if applied < min_applied_samples:
            reasons.append("insufficient-applied-samples")
        if holdout < min_holdout_samples:
            reasons.append("insufficient-holdout-samples")

    recommended = _bounded_fraction(min(max_fraction, recommended))
    if action == "hold":
        recommended = current_fraction
    if action == "stop":
        recommended = 0.0
    if recommended + holdout_fraction > 1.0:
        holdout_fraction = max(0.0, 1.0 - recommended)

    return {
        "schema": DECISION_SCHEMA,
        "family": family,
        "action": action,
        "reason_codes": reasons or ["no-op"],
        "current_fraction": round(current_fraction, 6),
        "recommended_fraction": round(recommended, 6),
        "recommended_holdout_fraction": round(holdout_fraction, 6),
        "changed": recommended != current_fraction,
        "evidence": evidence,
        "thresholds": {
            "initial_fraction": initial_fraction,
            "ramp_step": ramp_step,
            "max_fraction": max_fraction,
            "holdout_fraction": holdout_fraction,
            "min_applied_samples": min_applied_samples,
            "min_holdout_samples": min_holdout_samples,
            "max_error_rate": max_error_rate,
            "max_error_rate_delta": max_error_rate_delta,
            "max_retry_rate_delta": max_retry_rate_delta,
            "similarity_floor": similarity_floor,
        },
        "privacy": _privacy(),
    }


def _controller_meta(decision: dict[str, Any], *, now: str) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.local_compaction_canary_ramp_metadata.v1",
        "updated_at": now,
        "decision": decision["action"],
        "reason_codes": decision["reason_codes"],
        "previous_fraction": decision["current_fraction"],
        "recommended_fraction": decision["recommended_fraction"],
        "evidence": {
            "applied_count": decision["evidence"]["applied_count"],
            "holdout_count": decision["evidence"]["holdout_count"],
            "safety_stop_count": decision["evidence"]["safety_stop_count"],
            "applied_realized_savings_usd": decision["evidence"]["applied_realized_savings_usd"],
            "applied_minus_holdout_cost_avg_usd": decision["evidence"]["applied_minus_holdout_cost_avg_usd"],
            "applied_minus_holdout_error_rate": decision["evidence"]["applied_minus_holdout_error_rate"],
            "min_similarity": decision["evidence"].get("min_similarity"),
        },
        "local_only": True,
        "metadata_only": True,
        "raw_content_included": False,
    }


def _apply_thinking_edit(data: dict[str, Any], decision: dict[str, Any], *, now: str) -> None:
    rule, section = _current_thinking_rule(data)
    if rule is None:
        return
    fraction = float(decision["recommended_fraction"])
    section["enabled"] = bool(fraction > 0.0)
    section["ramp_controller"] = _controller_meta(decision, now=now)
    parent_canary = section.setdefault("canary", {})
    if isinstance(parent_canary, dict):
        parent_canary["enabled"] = True
        parent_canary["canary_fraction"] = fraction
        parent_canary["holdout_fraction"] = float(decision["recommended_holdout_fraction"])
        parent_canary.setdefault("canary_salt", "local-thinking-compaction-ramp-v1")
        parent_canary.setdefault("canary_unit", "thinking_block_local_fingerprint")
    rule["enabled"] = bool(fraction > 0.0)
    rule["policy_source"] = rule.get("policy_source") or section.get("policy_source") or "local-manual"
    canary = rule.setdefault("canary", {})
    if isinstance(canary, dict):
        canary["enabled"] = True
        canary["canary_fraction"] = fraction
        canary["holdout_fraction"] = float(decision["recommended_holdout_fraction"])
        canary.setdefault("canary_salt", "local-thinking-compaction-ramp-v1")
        canary.setdefault("canary_unit", "thinking_block_local_fingerprint")
    safety = rule.setdefault("safety_stop", {})
    if isinstance(safety, dict) and decision["action"] == "stop":
        safety["last_ramp_stop_reason"] = decision["reason_codes"][0]
        safety["last_ramp_stop_at"] = now
    rule["ramp_controller"] = _controller_meta(decision, now=now)


def _apply_old_context_edit(data: dict[str, Any], decision: dict[str, Any], *, now: str) -> None:
    section = data.get(OLD_CONTEXT_SECTION)
    if not isinstance(section, dict):
        return
    fraction = float(decision["recommended_fraction"])
    section["enabled"] = bool(fraction > 0.0)
    canary = section.setdefault("canary", {})
    if isinstance(canary, dict):
        canary["enabled"] = True
        canary["fraction"] = fraction
        canary.setdefault("salt", "local-old-context-summary-ramp-v1")
        canary.setdefault("unit", "source_hash")
    safety = section.setdefault("safety_stop", {})
    if isinstance(safety, dict) and decision["action"] == "stop":
        safety["last_ramp_stop_reason"] = decision["reason_codes"][0]
        safety["last_ramp_stop_at"] = now
    section["ramp_controller"] = _controller_meta(decision, now=now)


def build_local_compaction_canary_ramp(
    store_obj: Any,
    *,
    config_dir: str | Path | None = None,
    rules_path: str | Path | None = None,
    apply: bool = False,
    limit: int = 500,
    since: str | None = None,
    initial_fraction: float = 0.05,
    ramp_step: float = 0.05,
    max_fraction: float = 0.50,
    holdout_fraction: float = 0.10,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_error_rate: float = 0.10,
    max_error_rate_delta: float = 0.05,
    max_retry_rate_delta: float = 0.10,
    similarity_floor: float = 0.98,
    now: str | None = None,
) -> dict[str, Any]:
    target_path = Path(rules_path).expanduser() if rules_path is not None else _default_rules_path(config_dir)
    data, original_text, loaded_from = _load_yaml(target_path)
    proposed = copy.deepcopy(data)
    evidence = _evidence(store_obj, limit=limit, since=since)
    timestamp = now or utc_now()

    thinking_rule, thinking_section = _current_thinking_rule(proposed)
    current_thinking = _current_fraction(thinking_section, rule=thinking_rule) if thinking_rule is not None else 0.0
    old_section = proposed.get(OLD_CONTEXT_SECTION) if isinstance(proposed.get(OLD_CONTEXT_SECTION), dict) else {}
    current_old = _current_fraction(old_section)
    decisions = [
        _decision(
            THINKING_SECTION,
            evidence[THINKING_SECTION],
            current_thinking,
            initial_fraction=_bounded_fraction(initial_fraction, 0.05),
            ramp_step=_bounded_fraction(ramp_step, 0.05),
            max_fraction=_bounded_fraction(max_fraction, 0.50),
            holdout_fraction=_bounded_fraction(holdout_fraction, 0.10),
            min_applied_samples=max(0, _as_int(min_applied_samples)),
            min_holdout_samples=max(0, _as_int(min_holdout_samples)),
            max_error_rate=max(0.0, _as_float(max_error_rate, 0.10)),
            max_error_rate_delta=max(0.0, _as_float(max_error_rate_delta, 0.05)),
            max_retry_rate_delta=max(0.0, _as_float(max_retry_rate_delta, 0.10)),
            similarity_floor=_bounded_fraction(similarity_floor, 0.98),
        ),
        _decision(
            OLD_CONTEXT_SECTION,
            evidence[OLD_CONTEXT_SECTION],
            current_old,
            initial_fraction=_bounded_fraction(initial_fraction, 0.05),
            ramp_step=_bounded_fraction(ramp_step, 0.05),
            max_fraction=_bounded_fraction(max_fraction, 0.50),
            holdout_fraction=0.0,
            min_applied_samples=max(0, _as_int(min_applied_samples)),
            min_holdout_samples=max(0, _as_int(min_holdout_samples)),
            max_error_rate=max(0.0, _as_float(max_error_rate, 0.10)),
            max_error_rate_delta=max(0.0, _as_float(max_error_rate_delta, 0.05)),
            max_retry_rate_delta=max(0.0, _as_float(max_retry_rate_delta, 0.10)),
            similarity_floor=_bounded_fraction(similarity_floor, 0.98),
        ),
    ]

    for decision in decisions:
        if not decision["changed"]:
            continue
        if decision["family"] == THINKING_SECTION:
            _apply_thinking_edit(proposed, decision, now=timestamp)
        elif decision["family"] == OLD_CONTEXT_SECTION:
            _apply_old_context_edit(proposed, decision, now=timestamp)

    proposed_text = _dump_yaml(proposed)
    changed = proposed_text != (original_text or "")
    if apply and changed:
        _write_atomic(target_path, proposed_text)

    return {
        "schema": SCHEMA,
        "generated_at": timestamp,
        "ok": True,
        "apply": bool(apply),
        "status": "applied" if apply and changed else "planned" if changed else "no-change",
        "target_rule_file": CRUNCH_RULES_FILE,
        "target_path": str(target_path),
        "loaded_from": loaded_from,
        "target_file_existed": original_text is not None,
        "lookback_limit": max(1, min(int(limit or 500), 10_000)),
        "since": since,
        "changed": changed,
        "decisions": decisions,
        "summary": {
            "changed_decision_count": sum(1 for decision in decisions if decision.get("changed")),
            "widen_count": sum(1 for decision in decisions if decision.get("action") == "widen"),
            "stop_count": sum(1 for decision in decisions if decision.get("action") == "stop"),
            "hold_count": sum(1 for decision in decisions if decision.get("action") == "hold"),
        },
        "diff": _diff(original_text, proposed_text, target_path) if changed else "",
        "wrote_policy_files": bool(apply and changed),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
    }


def local_compaction_canary_ramp_config_hash(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
