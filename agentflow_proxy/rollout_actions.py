from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.pattern_rollout import PATTERN_ROLLOUT_SCHEMA, normalize_pattern_rollout
from agentflow_proxy.policy_bundle import (
    MANAGED_POLICY_VERIFICATION_SECRET_ENV,
    POLICY_BUNDLE_PROVENANCE_SCHEMA,
    _hmac_signature,
    _normalize_signature,
    _secret_for_key_id,
)
from agentflow_proxy.store import utc_now


PATTERN_ROLLOUT_ACTION_SCHEMA = "agentflow.pattern_rollout_action.v1"
PATTERN_ROLLOUT_ACTIONS_SCHEMA = "agentflow.pattern_rollout_actions.v1"
PATTERN_ROLLOUT_ACTION_REVIEW_SCHEMA = "agentflow.pattern_rollout_actions_review.v1"
PATTERN_ROLLOUT_ACTION_APPLY_SCHEMA = "agentflow.pattern_rollout_actions_apply.v1"
PATTERN_ROLLOUT_ACTION_DRY_RUN_SCHEMA = "agentflow.pattern_rollout_actions_dry_run.v1"
PATTERN_ROLLOUT_ACTION_VALIDATION_SCHEMA = "agentflow.pattern_rollout_actions_validation.v1"
PATTERN_ROLLOUT_ACTION_PROVENANCE_VERIFICATION_SCHEMA = "agentflow.pattern_rollout_actions_provenance_verification.v1"

ROLLOUT_ACTION_TYPES = {
    "widen",
    "hold",
    "rollback",
    "retire",
    "disable",
    "more-samples",
    "request-more-samples",
}
_POLICY_SECTION_FILES = {
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
}
_SAFE_POLICY_SOURCES = {"managed-recommended"}
_RAW_LIKE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "command",
    "content",
    "credential",
    "file_content",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_pattern",
    "raw_request",
    "raw_response",
    "secret",
    "system",
    "tool_payload",
    "transcript",
)
_ALLOWED_RAW_KEYS = {
    "raw_body_storage",
    "raw_payloads_returned",
    "raw_prompts_included",
    "raw_params_included",
    "raw_responses_included",
    "raw_tool_payloads_included",
    "raw_provider_bodies_included",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _action_bundle_payload_for_hash(bundle: dict[str, Any]) -> dict[str, Any]:
    payload = _json_roundtrip(bundle)
    if isinstance(payload, dict):
        payload.pop("provenance", None)
    return payload if isinstance(payload, dict) else {}


def canonical_rollout_action_bundle_hash(bundle: Any) -> str | None:
    if not isinstance(bundle, dict):
        return None
    payload = _action_bundle_payload_for_hash(bundle)
    return f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"


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


def _add_warning(warnings: list[dict[str, str]], path: str, message: str) -> None:
    warnings.append({"path": path, "message": message})


def _normalize_pattern_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    digest = text.removeprefix("sha256:") if text.startswith("sha256:") else text
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return f"sha256:{digest}"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _truthy(value: Any) -> bool:
    if value in (None, False, 0, "", [], {}):
        return False
    return True


def _scan_raw_like(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    _add_error(errors, child_path, "raw or prompt-like rollout action payloads are not accepted")
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:200]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


def verify_rollout_action_provenance(bundle: Any) -> dict[str, Any]:
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    managed_bundle = isinstance(bundle, dict) and bundle.get("schema") == PATTERN_ROLLOUT_ACTIONS_SCHEMA
    secret, configured = _secret_for_key_id(provenance.get("key_id") if isinstance(provenance, dict) else None)
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_PROVENANCE_VERIFICATION_SCHEMA,
        "status": "missing",
        "ok": True,
        "managed_bundle": managed_bundle,
        "verification_configured": configured,
        "signature_required": bool(managed_bundle and configured),
        "algorithm": None,
        "issuer": None,
        "server_id": None,
        "key_id": None,
        "generated_at": None,
        "bundle_hash": None,
        "computed_bundle_hash": canonical_rollout_action_bundle_hash(bundle),
        "signature_present": False,
        "errors": [],
        "warnings": [],
    }

    if not configured:
        result["status"] = "not-configured"
        if managed_bundle:
            result["warnings"].append({
                "path": "$.provenance",
                "message": f"managed rollout action provenance was not verified because {MANAGED_POLICY_VERIFICATION_SECRET_ENV} is not configured",
            })
        return result

    if not isinstance(provenance, dict):
        result["ok"] = False
        result["status"] = "missing"
        result["errors"].append({
            "path": "$.provenance",
            "message": "managed rollout action bundle is missing provenance required by configured verification",
        })
        return result

    for key in ("algorithm", "issuer", "server_id", "key_id", "generated_at", "bundle_hash"):
        result[key] = provenance.get(key)
    result["signature_present"] = bool(provenance.get("signature"))

    errors: list[dict[str, str]] = []
    if provenance.get("schema") != POLICY_BUNDLE_PROVENANCE_SCHEMA:
        _add_error(errors, "$.provenance.schema", f"expected {POLICY_BUNDLE_PROVENANCE_SCHEMA}")
    if provenance.get("algorithm") != "hmac-sha256":
        _add_error(errors, "$.provenance.algorithm", "expected hmac-sha256")
    for key in ("issuer", "server_id", "key_id"):
        if not isinstance(provenance.get(key), str) or not str(provenance.get(key)).strip():
            _add_error(errors, f"$.provenance.{key}", "expected non-empty string")
    if not _is_iso_datetime(provenance.get("generated_at")):
        _add_error(errors, "$.provenance.generated_at", "expected ISO-8601 timestamp string")
    if provenance.get("bundle_hash") != canonical_rollout_action_bundle_hash(bundle):
        _add_error(errors, "$.provenance.bundle_hash", "bundle hash does not match canonical payload")
    signature = _normalize_signature(provenance.get("signature"))
    if not signature:
        _add_error(errors, "$.provenance.signature", "expected HMAC signature")
    elif secret is None:
        _add_error(errors, "$.provenance.key_id", "no configured verification secret for key_id")
    elif not hmac.compare_digest(signature, _hmac_signature(provenance, secret)):
        _add_error(errors, "$.provenance.signature", "HMAC signature does not match provenance metadata")

    result["errors"] = errors
    if errors:
        result["status"] = "invalid"
        result["ok"] = False
    else:
        result["status"] = "verified"
        result["ok"] = True
    return result


def attach_rollout_action_provenance(
    bundle: dict[str, Any],
    *,
    secret: str,
    issuer: str,
    server_id: str,
    key_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    signed = _json_roundtrip(bundle)
    signed.pop("provenance", None)
    provenance = {
        "schema": POLICY_BUNDLE_PROVENANCE_SCHEMA,
        "algorithm": "hmac-sha256",
        "generated_at": generated_at or utc_now(),
        "issuer": issuer,
        "server_id": server_id,
        "key_id": key_id,
        "bundle_hash": canonical_rollout_action_bundle_hash(signed),
    }
    provenance["signature"] = _hmac_signature(provenance, secret)
    signed["provenance"] = provenance
    return signed


def validate_rollout_action_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_rollout_action_provenance(bundle)

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "rollout action bundle must be a JSON object")
        return {
            "schema": PATTERN_ROLLOUT_ACTION_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
        }
    if bundle.get("schema") != PATTERN_ROLLOUT_ACTIONS_SCHEMA:
        _add_error(errors, "$.schema", f"expected {PATTERN_ROLLOUT_ACTIONS_SCHEMA}")
    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")
    actions = bundle.get("actions")
    if not isinstance(actions, list):
        _add_error(errors, "$.actions", "expected list")
    else:
        for index, action in enumerate(actions):
            _validate_rollout_action(action, f"$.actions[{index}]", errors)
    _scan_raw_like(bundle, "$", errors)
    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            _add_warning(warnings, str(warning.get("path") or "$.provenance"), str(warning.get("message") or "provenance was not verified"))
    return {
        "schema": PATTERN_ROLLOUT_ACTION_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle.get("schema"),
        "action_count": len(actions) if isinstance(actions, list) else 0,
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
    }


def _validate_rollout_action(action: Any, path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(action, dict):
        _add_error(errors, path, "expected rollout action object")
        return
    if action.get("schema") != PATTERN_ROLLOUT_ACTION_SCHEMA:
        _add_error(errors, f"{path}.schema", f"expected {PATTERN_ROLLOUT_ACTION_SCHEMA}")
    action_type = str(action.get("action_type") or "").strip()
    if action_type not in ROLLOUT_ACTION_TYPES:
        _add_error(errors, f"{path}.action_type", "expected widen, hold, rollback, retire, disable, or more-samples")
    if action.get("policy_section") not in _POLICY_SECTION_FILES:
        _add_error(errors, f"{path}.policy_section", "expected crunch or cache")
    if not isinstance(action.get("target_candidate_id"), str) or not action.get("target_candidate_id").strip():
        _add_error(errors, f"{path}.target_candidate_id", "expected non-empty string")
    if action.get("target_rule_id") is not None and (not isinstance(action.get("target_rule_id"), str) or not action.get("target_rule_id").strip()):
        _add_error(errors, f"{path}.target_rule_id", "expected non-empty string when present")
    if _normalize_pattern_hash(action.get("pattern_hash")) is None:
        _add_error(errors, f"{path}.pattern_hash", "expected sha256 pattern hash")
    for key in ("current_fraction", "recommended_fraction", "confidence"):
        try:
            number = float(action.get(key))
        except (TypeError, ValueError):
            _add_error(errors, f"{path}.{key}", "expected numeric value")
            continue
        if number < 0 or number > 1:
            _add_error(errors, f"{path}.{key}", "expected number between 0 and 1")
    if action.get("required_local_review") is not True:
        _add_error(errors, f"{path}.required_local_review", "expected true")
    if action.get("managed_enforced") is not False:
        _add_error(errors, f"{path}.managed_enforced", "expected false")


def _load_policy_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    return parsed if isinstance(parsed, dict) else {}, text


def _rule_hashes(rule: dict[str, Any]) -> set[str]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    values = conditions.get("pattern_hashes", conditions.get("pattern_hash"))
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return set()
    return {normalized for value in values if (normalized := _normalize_pattern_hash(value))}


def _find_rule(rules: list[Any], action: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    pattern_hash = _normalize_pattern_hash(action.get("pattern_hash"))
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        candidate_id = str(rule.get("candidate_id") or rule.get("recommendation_id") or rule.get("policy_id") or "").strip()
        rule_id_match = not target_rule_id or rule_id == target_rule_id
        candidate_id_match = not target_candidate_id or candidate_id == target_candidate_id
        hash_match = pattern_hash in _rule_hashes(rule)
        if rule_id_match and candidate_id_match and hash_match:
            return index, rule
    return None, None


def _rule_rollout(rule: dict[str, Any]) -> dict[str, Any]:
    rollout = normalize_pattern_rollout(rule.get("rollout"))
    if rollout is None:
        rollout = {
            "schema": PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "canary",
            "canary_enabled": True,
            "canary_fraction": 1.0,
            "canary_salt": "",
            "canary_unit": "request_fingerprint",
        }
    return rollout


def _plan_rule_edit(rule: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("action_type") or "")
    if action_type == "request-more-samples":
        action_type = "more-samples"
    current_rollout = _rule_rollout(rule)
    current_fraction = _as_float(current_rollout.get("canary_fraction"), 1.0)
    recommended_fraction = _as_float(action.get("recommended_fraction"), current_fraction)
    disable = action_type in {"rollback", "retire", "disable"}
    proposed_enabled = False if disable else bool(rule.get("enabled", True))
    proposed_rollout = dict(current_rollout)
    proposed_rollout["schema"] = str(proposed_rollout.get("schema") or PATTERN_ROLLOUT_SCHEMA)
    proposed_rollout["canary_fraction"] = 0.0 if disable else recommended_fraction
    proposed_rollout["canary_enabled"] = False if disable else bool(proposed_rollout.get("canary_enabled", True))
    proposed_rollout["recommendation_mode"] = "disabled-by-rollout-action" if disable else str(
        proposed_rollout.get("recommendation_mode") or "canary"
    )
    changed = bool(rule.get("enabled", True)) != proposed_enabled or normalize_pattern_rollout(rule.get("rollout")) != normalize_pattern_rollout(proposed_rollout)
    return {
        "action_type": action_type,
        "disable": disable,
        "current_fraction": current_fraction,
        "recommended_fraction": recommended_fraction,
        "proposed_enabled": proposed_enabled,
        "proposed_rollout": proposed_rollout,
        "changed": changed,
    }


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_bucket(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _status_bucket(status_code: Any) -> str:
    status = _as_int(status_code)
    if status <= 0:
        return "unknown"
    if status < 200:
        return "lt_2xx"
    if status < 300:
        return "2xx"
    if status < 400:
        return "3xx"
    if status < 500:
        return "4xx"
    return "5xx"


def _saving_bucket(value: Any) -> str:
    amount = _as_float(value, 0.0)
    if amount <= 0:
        return "zero"
    if amount < 0.001:
        return "lt_0_001"
    if amount < 0.01:
        return "0_001_0_01"
    if amount < 0.05:
        return "0_01_0_05"
    return "gte_0_05"


def _cohort_for_summary(summary: dict[str, Any]) -> str:
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    status = str(summary.get("status") or "")
    outcome = str(summary.get("outcome") or "")
    reason = str(summary.get("reason") or "")
    cohort = str(summary.get("cohort") or canary.get("cohort") or "")
    if outcome == "holdout" or status == "holdout" or cohort == "canary_holdout" or canary.get("status") == "holdout":
        return "canary_holdout"
    if outcome == "bypassed" or status in {"bypass", "bypassed"} or "bypass" in reason or "disabled" in reason:
        return "bypassed"
    if _as_int(summary.get("applied_count")) > 0 and (canary.get("enabled") or cohort == "canary_applied"):
        return "canary_applied"
    if _as_int(summary.get("applied_count")) > 0 or outcome == "applied" or status == "applied":
        return "applied"
    return "received"


def _codex_estimated_cost(input_text_chars: Any, result_chars: Any) -> float:
    # Rollout action dry-runs are metadata-only; Codex app rows do not have provider-reported
    # billing fields yet, so expose count/risk impact without inventing model-specific spend.
    _ = input_text_chars, result_chars
    return 0.0


def _traffic_pattern_summaries(store_obj: Any, *, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from agentflow_proxy.recommendations import pattern_decision_summaries

    capped_limit = max(1, min(int(limit or 500), 5000))
    conn = store_obj.conn
    summaries: list[dict[str, Any]] = []
    unknowns = {
        "provider_rows_considered": 0,
        "codex_turn_rows_considered": 0,
        "rows_without_pattern_decisions": 0,
        "summaries_missing_candidate_id": 0,
        "summaries_missing_rule_id": 0,
        "summaries_missing_pattern_hash": 0,
        "summaries_missing_canary_cohort": 0,
    }

    provider_rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   cost_est_usd, cost_baseline_usd, crunch_json, routing_json,
                   cache_json, category
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    unknowns["provider_rows_considered"] = len(provider_rows)
    for row in provider_rows:
        routing = _json_obj(row.get("routing_json"))
        rows = pattern_decision_summaries(
            provider=str(row.get("provider") or "anthropic"),
            path=str(row.get("path") or ""),
            requested_model=row.get("requested_model"),
            routed_model=row.get("routed_model"),
            status_code=_as_int(row.get("status_code")) if row.get("status_code") is not None else None,
            cost_est_usd=_as_float(row.get("cost_est_usd")) if row.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(row.get("cost_baseline_usd")) if row.get("cost_baseline_usd") is not None else None,
            cache_meta=_json_obj(row.get("cache_json")),
            crunch_meta=_json_obj(row.get("crunch_json")),
            routing_meta=routing,
            category=row.get("category") or routing.get("category"),
        )
        if not rows:
            unknowns["rows_without_pattern_decisions"] += 1
        for summary in rows:
            if not isinstance(summary, dict):
                continue
            item = dict(summary)
            item.update({
                "traffic_row_id": row.get("id"),
                "created_at": row.get("created_at"),
                "status_code": row.get("status_code"),
                "latency_ms": row.get("latency_ms"),
                "cost_est_usd": row.get("cost_est_usd"),
                "traffic_kind": "provider_call",
            })
            summaries.append(item)

    codex_rows = [
        dict(row)
        for row in conn.execute(
            """
            select s.id as start_event_id,
                   s.created_at,
                   s.request_id,
                   s.thread_id,
                   s.session_id,
                   s.input_text_chars,
                   s.routing_json,
                   s.crunch_json,
                   s.cache_json,
                   (
                       select r.id from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_event_id,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars,
                   (
                       select r.error_code from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_error_code,
                   (
                       select r.latency_ms from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_latency_ms
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    unknowns["codex_turn_rows_considered"] = len(codex_rows)
    for row in codex_rows:
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        status_code = 500 if row.get("response_error_code") is not None else (200 if row.get("response_event_id") else None)
        rows = pattern_decision_summaries(
            provider="openai",
            path="codex-app://turn/start",
            requested_model=routing.get("requested_model") or routing.get("requested_model_value"),
            routed_model=routing.get("routed_model") or routing.get("target_model") or routing.get("requested_model"),
            status_code=status_code,
            cost_est_usd=_codex_estimated_cost(row.get("input_text_chars"), row.get("response_result_chars")),
            cost_baseline_usd=_codex_estimated_cost(row.get("input_text_chars"), row.get("response_result_chars")),
            cache_meta=cache,
            crunch_meta=_json_obj(row.get("crunch_json")),
            routing_meta=routing,
            category=routing.get("category") or "codex_turn",
        )
        if not rows:
            unknowns["rows_without_pattern_decisions"] += 1
        for summary in rows:
            if not isinstance(summary, dict):
                continue
            item = dict(summary)
            item.update({
                "traffic_row_id": row.get("start_event_id"),
                "created_at": row.get("created_at"),
                "source_surface": "codex_turn",
                "app_family": "codex",
                "status_code": status_code,
                "latency_ms": row.get("response_latency_ms"),
                "cost_est_usd": 0.0,
                "traffic_kind": "codex_turn",
            })
            summaries.append(item)

    for summary in summaries:
        if not summary.get("candidate_id"):
            unknowns["summaries_missing_candidate_id"] += 1
        if not summary.get("rule_id"):
            unknowns["summaries_missing_rule_id"] += 1
        if not str(summary.get("pattern_hash") or "").startswith("sha256:"):
            unknowns["summaries_missing_pattern_hash"] += 1
        canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
        if not summary.get("cohort") and not canary.get("cohort") and not canary.get("status"):
            unknowns["summaries_missing_canary_cohort"] += 1

    return summaries, unknowns


def _summary_matches_action(summary: dict[str, Any], action: dict[str, Any]) -> bool:
    if str(summary.get("decision_type") or "") != str(action.get("policy_section") or ""):
        return False
    pattern_hash = _normalize_pattern_hash(action.get("pattern_hash"))
    summary_hashes = {
        normalized
        for value in [summary.get("pattern_hash"), *(summary.get("pattern_hashes") or [])]
        if (normalized := _normalize_pattern_hash(value))
    }
    if pattern_hash and pattern_hash not in summary_hashes:
        return False
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    if target_candidate_id and str(summary.get("candidate_id") or "").strip() != target_candidate_id:
        return False
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    if target_rule_id and str(summary.get("rule_id") or "").strip() != target_rule_id:
        return False
    return True


def _projected_counts(*, matched_count: int, applied_count: int, holdout_count: int, edit: dict[str, Any]) -> dict[str, Any]:
    proposed_fraction = _as_float(edit.get("recommended_fraction"), _as_float(edit.get("current_fraction"), 1.0))
    disable = bool(edit.get("disable"))
    if disable:
        projected_applied = 0
        projected_holdout = 0
        projected_disabled = matched_count
    else:
        projected_applied = max(applied_count, round(matched_count * proposed_fraction))
        projected_applied = min(projected_applied, matched_count)
        projected_holdout = max(0, matched_count - projected_applied)
        projected_disabled = 0
    return {
        "current_fraction": _as_float(edit.get("current_fraction"), 0.0),
        "projected_fraction": 0.0 if disable else proposed_fraction,
        "current_canary_applied_count": applied_count,
        "current_canary_holdout_count": holdout_count,
        "projected_canary_applied_count": projected_applied,
        "projected_canary_holdout_count": projected_holdout,
        "projected_local_bypass_or_disable_count": projected_disabled,
        "projected_additional_applied_count": max(0, projected_applied - applied_count),
    }


def dry_run_rollout_actions(
    bundle: Any,
    *,
    store_obj: Any,
    config_dir: str | Path,
    sections: list[str] | tuple[str, ...] | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    review = plan_rollout_actions(bundle, config_dir=config_dir, sections=sections)
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_DRY_RUN_SCHEMA,
        "ok": False,
        "dry_run": True,
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "config_dir": str(config_path),
        "lookback_limit": max(1, min(int(limit or 500), 5000)),
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "actions": [],
        "summary": {},
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "policy_files_written": False,
            "store_written": False,
            "basis": "stored pattern decision metadata, hashes, canary cohorts, status codes, latency, and size-derived savings only",
        },
        "errors": review.get("errors", []),
        "warnings": review.get("warnings", []),
    }
    if not review.get("validation", {}).get("ok"):
        result["error"] = {"type": "validation_failed", "message": "rollout action bundle failed validation"}
        return result

    summaries, unknowns = _traffic_pattern_summaries(store_obj, limit=result["lookback_limit"])
    action_results: list[dict[str, Any]] = []
    for planned in review.get("actions", []):
        action_index = int(str(planned.get("path") or "$.actions[0]").split("[")[-1].split("]")[0]) if "[" in str(planned.get("path") or "") else 0
        raw_action = (bundle.get("actions") or [])[action_index] if isinstance(bundle, dict) and isinstance(bundle.get("actions"), list) and action_index < len(bundle["actions"]) else {}
        matched = [summary for summary in summaries if _summary_matches_action(summary, raw_action)]
        status_counts: dict[str, int] = {}
        savings_counts: dict[str, int] = {}
        traffic_counts: dict[str, int] = {}
        bypass_reasons: dict[str, int] = {}
        applied_count = 0
        holdout_count = 0
        bypass_count = 0
        saved_chars = 0
        tokens_saved = 0
        cost_savings = 0.0
        for summary in matched:
            cohort = _cohort_for_summary(summary)
            if cohort == "canary_applied":
                applied_count += 1
            elif cohort == "canary_holdout":
                holdout_count += 1
            elif cohort == "bypassed":
                bypass_count += 1
                reason = str(summary.get("reason") or "unknown")
                bypass_reasons[reason] = bypass_reasons.get(reason, 0) + 1
            status_counts[_status_bucket(summary.get("status_code"))] = status_counts.get(_status_bucket(summary.get("status_code")), 0) + 1
            savings_counts[_saving_bucket(summary.get("estimated_cost_savings_usd"))] = savings_counts.get(_saving_bucket(summary.get("estimated_cost_savings_usd")), 0) + 1
            traffic_kind = str(summary.get("traffic_kind") or "unknown")
            traffic_counts[traffic_kind] = traffic_counts.get(traffic_kind, 0) + 1
            saved_chars += _as_int(summary.get("saved_chars"))
            tokens_saved += _as_int(summary.get("tokens_saved_est"))
            cost_savings += _as_float(summary.get("estimated_cost_savings_usd"), 0.0)
        edit = planned.get("proposed_edit") if isinstance(planned.get("proposed_edit"), dict) else {}
        projected = _projected_counts(
            matched_count=len(matched),
            applied_count=applied_count,
            holdout_count=holdout_count,
            edit=edit,
        )
        unknown_action = {
            "matched_summaries_missing_candidate_id": sum(1 for item in matched if not item.get("candidate_id")),
            "matched_summaries_missing_rule_id": sum(1 for item in matched if not item.get("rule_id")),
            "matched_summaries_missing_pattern_hash": sum(1 for item in matched if not str(item.get("pattern_hash") or "").startswith("sha256:")),
            "matched_summaries_missing_canary_cohort": sum(
                1
                for item in matched
                if not item.get("cohort")
                and not ((item.get("canary") if isinstance(item.get("canary"), dict) else {}) or {}).get("cohort")
                and not ((item.get("canary") if isinstance(item.get("canary"), dict) else {}) or {}).get("status")
            ),
        }
        action_results.append({
            "path": planned.get("path"),
            "status": planned.get("status"),
            "reason": planned.get("reason"),
            "policy_section": raw_action.get("policy_section") or planned.get("policy_section"),
            "action_type": raw_action.get("action_type") or planned.get("action_type"),
            "target_candidate_id": raw_action.get("target_candidate_id") or planned.get("target_candidate_id"),
            "target_rule_id": raw_action.get("target_rule_id") or planned.get("target_rule_id"),
            "rule_id": planned.get("rule_id"),
            "pattern_hash": _normalize_pattern_hash(raw_action.get("pattern_hash")) or planned.get("pattern_hash"),
            "affected_metadata_row_count": len(matched),
            "affected_provider_call_count": traffic_counts.get("provider_call", 0),
            "affected_codex_turn_count": traffic_counts.get("codex_turn", 0),
            "current_bypassed_or_disabled_count": bypass_count,
            **projected,
            "historical_tokens_saved_est": tokens_saved,
            "historical_saved_chars": saved_chars,
            "historical_estimated_cost_savings_usd": round(cost_savings, 8),
            "savings_buckets": _count_bucket(savings_counts),
            "status_risk_buckets": _count_bucket(status_counts),
            "local_bypass_reasons": _count_bucket(bypass_reasons),
            "unknowns": unknown_action,
            "proposed_edit": planned.get("proposed_edit"),
        })

    total_affected = sum(_as_int(action.get("affected_metadata_row_count")) for action in action_results)
    result.update({
        "ok": bool(review.get("ok")),
        "actions": action_results,
        "summary": {
            "sampled_provider_calls": unknowns["provider_rows_considered"],
            "sampled_codex_turns": unknowns["codex_turn_rows_considered"],
            "sampled_metadata_rows": unknowns["provider_rows_considered"] + unknowns["codex_turn_rows_considered"],
            "pattern_decision_summary_count": len(summaries),
            "affected_metadata_row_count": total_affected,
            "affected_provider_call_count": sum(_as_int(action.get("affected_provider_call_count")) for action in action_results),
            "affected_codex_turn_count": sum(_as_int(action.get("affected_codex_turn_count")) for action in action_results),
            "projected_additional_applied_count": sum(_as_int(action.get("projected_additional_applied_count")) for action in action_results),
            "projected_local_bypass_or_disable_count": sum(_as_int(action.get("projected_local_bypass_or_disable_count")) for action in action_results),
            "historical_tokens_saved_est": sum(_as_int(action.get("historical_tokens_saved_est")) for action in action_results),
            "historical_estimated_cost_savings_usd": round(sum(_as_float(action.get("historical_estimated_cost_savings_usd")) for action in action_results), 8),
            "unknowns": unknowns,
        },
    })
    if not review.get("ok"):
        result["error"] = {"type": "review_failed", "message": "rollout actions are invalid or target unknown local rules"}
    return result


def plan_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    validation = validate_rollout_action_bundle(bundle)
    requested_sections = set(sections or _POLICY_SECTION_FILES)
    config_path = Path(config_dir).expanduser()
    actions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    files: dict[str, dict[str, Any]] = {}

    if validation["ok"] and (invalid := sorted(requested_sections - set(_POLICY_SECTION_FILES))):
        for section in invalid:
            _add_error(errors, f"$.sections.{section}", "unknown rollout action policy section")

    if validation["ok"] and not errors:
        for index, action in enumerate(bundle.get("actions") or []):
            section = str(action.get("policy_section"))
            if section not in requested_sections:
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "skipped",
                    "reason": "not-requested",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            file_plan = files.get(section)
            if file_plan is None:
                path = config_path / _POLICY_SECTION_FILES[section]
                data, old_text = _load_policy_yaml(path)
                rules = data.get("pattern_rules")
                if not isinstance(rules, list):
                    rules = []
                    data["pattern_rules"] = rules
                file_plan = {"section": section, "path": path, "data": data, "old_text": old_text, "rules": rules}
                files[section] = file_plan

            rule_index, rule = _find_rule(file_plan["rules"], action)
            if rule is None or rule_index is None:
                _add_error(errors, f"$.actions[{index}]", "rollout action targets an unknown local pattern rule")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unknown-rule",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                    "pattern_hash": action.get("pattern_hash"),
                })
                continue
            policy_source = str(rule.get("policy_source") or "")
            if policy_source not in _SAFE_POLICY_SOURCES:
                _add_error(errors, f"$.actions[{index}]", "rollout action targets a rule with an unsafe policy source")
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unsafe-policy-source",
                    "policy_section": section,
                    "rule_id": rule.get("id") or rule.get("rule_id"),
                    "policy_source": policy_source,
                })
                continue
            edit = _plan_rule_edit(rule, action)
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": section,
                "action_type": edit["action_type"],
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "rule_id": rule.get("id") or rule.get("rule_id"),
                "rule_index": rule_index,
                "pattern_hash": _normalize_pattern_hash(action.get("pattern_hash")),
                "confidence": action.get("confidence"),
                "blockers": action.get("blockers") if isinstance(action.get("blockers"), list) else [],
                "rationale": action.get("rationale"),
                "current_rule": {
                    "enabled": bool(rule.get("enabled", True)),
                    "policy_source": policy_source,
                    "rollout": normalize_pattern_rollout(rule.get("rollout")),
                },
                "proposed_edit": {
                    "changed": edit["changed"],
                    "disable": edit["disable"],
                    "current_fraction": edit["current_fraction"],
                    "recommended_fraction": edit["recommended_fraction"],
                    "enabled": edit["proposed_enabled"],
                    "rollout": edit["proposed_rollout"],
                },
            })

    ok = bool(validation["ok"] and not errors)
    return {
        "schema": PATTERN_ROLLOUT_ACTION_REVIEW_SCHEMA,
        "ok": ok,
        "config_dir": str(config_path),
        "validation": validation,
        "provenance": validation.get("provenance"),
        "action_count": validation.get("action_count", 0),
        "planned_action_count": sum(1 for action in actions if action.get("status") == "planned"),
        "rejected_action_count": sum(1 for action in actions if action.get("status") == "rejected") + len(errors),
        "changed_action_count": sum(
            1 for action in actions
            if action.get("status") == "planned" and (action.get("proposed_edit") or {}).get("changed")
        ),
        "actions": actions,
        "errors": [*validation.get("errors", []), *errors],
        "warnings": validation.get("warnings", []),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def apply_rollout_actions(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    review = plan_rollout_actions(bundle, config_dir=config_dir, sections=sections)
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": PATTERN_ROLLOUT_ACTION_APPLY_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "review": review,
        "validation": review.get("validation"),
        "provenance": review.get("provenance"),
        "applied_sections": [],
        "files": [],
        "actions": review.get("actions", []),
        "error": None,
    }
    if not review.get("ok"):
        result["error"] = {"type": "validation_failed", "message": "rollout actions are invalid or target unknown local rules"}
        return result

    file_sections = sorted({
        action["policy_section"]
        for action in review.get("actions", [])
        if action.get("status") == "planned"
    })
    for section in file_sections:
        path = config_path / _POLICY_SECTION_FILES[section]
        data, old_text = _load_policy_yaml(path)
        rules = data.get("pattern_rules")
        if not isinstance(rules, list):
            rules = []
            data["pattern_rules"] = rules
        section_actions = [
            action for action in review.get("actions", [])
            if action.get("status") == "planned" and action.get("policy_section") == section
        ]
        for planned in section_actions:
            rule_index = int(planned["rule_index"])
            if rule_index < 0 or rule_index >= len(rules) or not isinstance(rules[rule_index], dict):
                result["error"] = {"type": "plan_mismatch", "message": "local policy file changed after rollout action review"}
                return result
            edit = planned.get("proposed_edit") or {}
            rules[rule_index]["enabled"] = bool(edit.get("enabled"))
            rules[rule_index]["rollout"] = edit.get("rollout")
            rules[rule_index]["rollout_action"] = {
                "schema": PATTERN_ROLLOUT_ACTION_SCHEMA,
                "action_type": planned.get("action_type"),
                "target_candidate_id": planned.get("target_candidate_id"),
                "pattern_hash": planned.get("pattern_hash"),
                "confidence": planned.get("confidence"),
                "blockers": planned.get("blockers", []),
                "rationale": planned.get("rationale"),
                "reviewed_at": utc_now(),
            }
        text = yaml.safe_dump(data, sort_keys=False)
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
