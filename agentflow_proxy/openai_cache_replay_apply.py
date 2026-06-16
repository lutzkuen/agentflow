from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy.cache import _parse_pattern_hashes
from agentflow_proxy.openai_cache_replay_dry_run import build_openai_cache_replay_dry_run
from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_impact_report
from agentflow_proxy.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report
from agentflow_proxy.openai_cache_replay_report import _as_float, _as_int, _json_obj
from agentflow_proxy.pattern_rollout import PATTERN_ROLLOUT_SCHEMA
from agentflow_proxy.request_shape_rollups import build_request_shape_rollups_report
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_cache_replay_apply.v1"
PLAN_SCHEMA = "agentflow.openai_cache_replay_apply_plan.v1"
POLICY_SCHEMA = "agentflow.openai_cache_replay_canary_policy.v1"
CACHE_CANARY_POLICY_FILE = "cache_canary_policy.yaml"

_READY_VERDICTS = {"widen", "promote", "ready"}
_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_REQUEST_SHAPE_PATTERN_WILDCARD = "sha256:*"


def _bounded_fraction(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _public_id(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if text and _PUBLIC_ID_RE.match(text) and not any(part in text.lower() for part in ("raw", "secret", "cache_key", "request_id", "session_id")):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _backup_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _read_openai_cache_rows(store_obj: Any, limit: int) -> list[dict[str, Any]]:
    capped = max(1, min(_as_int(limit) or 1000, 10_000))
    return [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, input_tokens_est, actual_input_tokens, cost_est_usd,
                   cost_baseline_usd, category, routing_json, cache_json
            from calls
            where coalesce(provider, 'anthropic') = 'openai'
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]


def _feature_unit(routing: dict[str, Any]) -> dict[str, Any]:
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        value = routing.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _rule_candidates_from_cache(cache: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if isinstance(cache.get("pattern_rule"), dict):
        rules.append(cache["pattern_rule"])
    pattern_rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), dict) else {}
    for key in ("rules", "skip_reasons"):
        for item in pattern_rules.get(key) or []:
            if isinstance(item, dict):
                rules.append(item)
    return rules


def _pattern_hashes_from_rule(rule: dict[str, Any]) -> list[str]:
    hashes: list[Any] = []
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    hashes.extend(conditions.get("pattern_hashes") or [])
    hashes.extend(rule.get("matched_hashes") or [])
    hashes.extend(rule.get("pattern_hashes") or [])
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    hashes.extend(canary.get("pattern_hashes") or [])
    return _parse_pattern_hashes(hashes)


def _candidate_ids_from_rule(rule: dict[str, Any], cache: dict[str, Any]) -> set[str]:
    canary = cache.get("cache_replay_canary") if isinstance(cache.get("cache_replay_canary"), dict) else {}
    nested_canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    ids = {
        rule.get("candidate_id"),
        rule.get("rule_id"),
        canary.get("candidate_id"),
        canary.get("rule_id"),
        nested_canary.get("candidate_id"),
        nested_canary.get("rule_id"),
    }
    return {_public_id(value, "candidate-id") for value in ids if value}


def _existing_rule_index(store_obj: Any, *, limit: int) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_openai_cache_rows(store_obj, limit):
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        feature = _feature_unit(routing)
        for rule in _rule_candidates_from_cache(cache):
            hashes = _pattern_hashes_from_rule(rule)
            if not hashes:
                continue
            rule_id = _public_id(rule.get("rule_id") or rule.get("id") or rule.get("candidate_id"), "rule-id")
            candidate_id = _public_id(rule.get("candidate_id") or rule_id, "candidate-id")
            entry = indexed.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "rule_id": rule_id,
                    "pattern_hashes": set(),
                    "source_surface": row.get("source_surface") or feature.get("source_surface"),
                    "endpoint": row.get("endpoint") or feature.get("endpoint"),
                    "category": row.get("category") or feature.get("category"),
                    "workflow_phase": feature.get("workflow_phase") or row.get("category"),
                    "stream": bool(_as_int(row.get("stream"))),
                    "has_tools": bool(routing.get("has_tools")),
                    "allow_tool_calls": bool(rule.get("allow_tool_calls")),
                    "safe_invalidation": bool(rule.get("safe_invalidation")),
                    "scope": str(rule.get("scope") or "session"),
                    "replayability_level": rule.get("replayability_level"),
                },
            )
            entry["pattern_hashes"].update(hashes)
            entry["allow_tool_calls"] = bool(entry["allow_tool_calls"] or rule.get("allow_tool_calls"))
            entry["safe_invalidation"] = bool(entry["safe_invalidation"] or rule.get("safe_invalidation"))
            for alternate in _candidate_ids_from_rule(rule, cache):
                indexed.setdefault(alternate, entry)
    return indexed


def _rule_from_candidate(
    candidate: dict[str, Any],
    indexed: dict[str, Any],
    *,
    holdout_fraction: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    pattern_hashes = sorted(indexed.get("pattern_hashes") or [])
    if not pattern_hashes:
        blockers.append("pattern-hashes-missing")
    if not indexed.get("rule_id"):
        blockers.append("rule-id-missing")
    if blockers:
        return None, blockers

    candidate_id = str(indexed.get("candidate_id") or candidate.get("candidate_id"))
    rule_id = str(indexed.get("rule_id") or candidate.get("rule_id") or candidate_id)
    has_tools = bool(indexed.get("has_tools"))
    allow_tool_calls = bool(indexed.get("allow_tool_calls") or has_tools)
    safe_invalidation = bool(indexed.get("safe_invalidation") or not has_tools)
    conditions = {
        "pattern_hashes": pattern_hashes,
        "source_surface": indexed.get("source_surface") or candidate.get("source_surface"),
        "endpoint": indexed.get("endpoint") or candidate.get("endpoint"),
        "category": indexed.get("category") or candidate.get("category"),
        "workflow_phase": indexed.get("workflow_phase") or candidate.get("workflow_phase"),
        "has_tools": has_tools,
        "stream": bool(indexed.get("stream")),
        "replayability_levels": [indexed.get("replayability_level") or "local-exact-response"],
    }
    conditions = {key: value for key, value in conditions.items() if value not in (None, "", [])}
    canary_fraction = round(1.0 - holdout_fraction, 6)
    return {
        "id": rule_id,
        "enabled": True,
        "policy_source": "managed-recommended",
        "candidate_id": candidate_id,
        "description": "Local OpenAI cache replay canary graduated from readiness evidence.",
        "conditions": conditions,
        "action": {
            "type": "exact_cache_pattern",
            "allow_tool_calls": allow_tool_calls,
            "safe_invalidation": safe_invalidation,
            "streaming": bool(indexed.get("stream")),
            "scope": str(indexed.get("scope") or "session"),
        },
        "rollout": {
            "schema": PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "openai-cache-replay-local-canary",
            "canary_enabled": True,
            "canary_fraction": canary_fraction,
            "canary_salt": candidate_id,
            "canary_unit": "request_fingerprint",
            "local_feedback_fields": [
                "cache_hit",
                "status_code",
                "retry_count",
                "latency_ms",
                "cost_est_usd",
                "cost_baseline_usd",
                "invalidation_reason",
            ],
        },
        "graduation": {
            "schema": "agentflow.openai_cache_replay_graduation.v1",
            "source_verdict": candidate.get("verdict"),
            "observed_savings_usd": candidate.get("observed_savings_usd"),
            "projected_savings_usd": candidate.get("projected_savings_usd"),
            "sample_count": candidate.get("sample_count"),
            "applied_count": candidate.get("applied_count"),
            "holdout_count": candidate.get("holdout_count"),
            "graduated_at": utc_now(),
        },
    }, []


def _shape_candidate_id(cohort: dict[str, Any]) -> str:
    basis = {
        "schema": "agentflow.openai_cache_replay_shape_candidate_basis.v1",
        "provider_family": cohort.get("provider_family"),
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "stream": bool(cohort.get("stream")),
        "has_tools": bool(cohort.get("has_tools")),
        "cache_status": cohort.get("cache_status"),
        "routing_status": cohort.get("routing_status"),
        "text_bucket": cohort.get("text_bucket"),
        "token_bucket": cohort.get("token_bucket"),
    }
    digest = hashlib.sha256(
        yaml.safe_dump(basis, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    endpoint = str(cohort.get("endpoint") or "unknown").replace("_", "-")
    category = str(cohort.get("category") or "unknown").replace("_", "-")
    return _public_id(f"request-shape-cache:{endpoint}:{category}:{digest}", "shape-candidate")


def _rule_from_request_shape_cohort(
    cohort: dict[str, Any],
    *,
    holdout_fraction: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    if str(cohort.get("readiness") or "") != "replay-ready":
        blockers.append("not-replay-ready")
    if str(cohort.get("provider_family") or "").lower() != "openai":
        blockers.append("non-openai-provider")
    if bool(cohort.get("has_tools")):
        blockers.append("tools-present")
    if bool(cohort.get("stream")):
        blockers.append("streaming-replay-not-supported")
    if _as_int(cohort.get("projected_hits")) <= 0:
        blockers.append("non-positive-projected-hits")
    if _as_float(cohort.get("projected_savings_usd")) <= 0:
        blockers.append("non-positive-projected-savings")
    if blockers:
        return None, blockers

    candidate_id = _shape_candidate_id(cohort)
    rule_id = _public_id(candidate_id.replace("request-shape-cache:", "openai-cache-shape-"), "shape-rule")
    canary_fraction = round(1.0 - holdout_fraction, 6)
    cohort_bucket = "/".join(
        str(value or "unknown").replace("/", "_")
        for value in (
            cohort.get("source_surface"),
            cohort.get("endpoint"),
            cohort.get("category"),
            cohort.get("workflow_phase"),
            cohort.get("text_bucket"),
            cohort.get("token_bucket"),
        )
    )
    conditions = {
        "pattern_hashes": [_REQUEST_SHAPE_PATTERN_WILDCARD],
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "text_bucket": cohort.get("text_bucket"),
        # token_bucket is intentionally omitted: the OpenAI feature unit does not
        # populate a bare "token_bucket" key in pattern_features, so the pattern
        # matcher in cache.py would always fail this condition.
        "has_tools": False,
        "stream": False,
        "replayability_levels": ["features_only", "local-exact-response"],
    }
    conditions = {key: value for key, value in conditions.items() if value not in (None, "", [])}
    return {
        "id": rule_id,
        "enabled": True,
        "policy_source": "managed-recommended",
        "candidate_id": candidate_id,
        "description": "Local OpenAI exact-cache canary staged from replay-ready request-shape evidence.",
        "conditions": conditions,
        "action": {
            "type": "exact_cache_pattern",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "streaming": False,
            "scope": "session",
            "min_call_count": 2,
        },
        "rollout": {
            "schema": PATTERN_ROLLOUT_SCHEMA,
            "recommendation_mode": "openai-cache-replay-request-shape-canary",
            "canary_enabled": True,
            "canary_fraction": canary_fraction,
            "canary_salt": candidate_id,
            "canary_unit": "request_fingerprint",
            "local_feedback_fields": [
                "cache_hit",
                "status_code",
                "retry_count",
                "latency_ms",
                "cost_est_usd",
                "cost_baseline_usd",
                "cache_replay_canary",
            ],
        },
        "graduation": {
            "schema": "agentflow.openai_cache_replay_shape_activation.v1",
            "source_schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
            "source_reason": cohort.get("reason"),
            "cohort_bucket": cohort_bucket,
            "source_surface": cohort.get("source_surface"),
            "endpoint": cohort.get("endpoint"),
            "category": cohort.get("category"),
            "workflow_phase": cohort.get("workflow_phase"),
            "text_bucket": cohort.get("text_bucket"),
            "token_bucket": cohort.get("token_bucket"),
            "projected_hits": cohort.get("projected_hits"),
            "projected_savings_usd": cohort.get("projected_savings_usd"),
            "sample_count": cohort.get("row_count"),
            "aggregate_only": True,
            "graduated_at": utc_now(),
        },
    }, []


def _safety_stop_count(dry_run: dict[str, Any]) -> int:
    count = 0
    for row in dry_run.get("reason_breakdown") or []:
        if isinstance(row, dict) and str(row.get("value") or "") == "local-canary-safety-stop":
            count += _as_int(row.get("count"))
    for row in dry_run.get("blocker_breakdown") or []:
        if isinstance(row, dict) and str(row.get("value") or "") == "local-canary-safety-stop":
            count += _as_int(row.get("count"))
    return count


def build_openai_cache_replay_apply_plan(
    store_obj: Any,
    *,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
    min_observed_savings_usd: float = 0.0,
    holdout_fraction: float = 0.20,
    max_candidates: int = 10,
) -> dict[str, Any]:
    readiness = build_openai_cache_replay_readiness_report(
        store_obj,
        opportunity_limit=opportunity_limit,
        impact_limit=impact_limit,
    )
    impact = build_openai_cache_replay_impact_report(store_obj, limit=impact_limit)
    request_shape = build_request_shape_rollups_report(
        store_obj,
        limit=opportunity_limit,
        persist=False,
        run_id="openai-cache-replay-apply-shape-evidence",
    )
    shape_replay = (
        request_shape.get("cache_replayability_dry_run")
        if isinstance(request_shape.get("cache_replayability_dry_run"), dict)
        else {}
    )
    indexed = _existing_rule_index(store_obj, limit=max(opportunity_limit, impact_limit))
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    holdout = _bounded_fraction(holdout_fraction, 0.20)

    candidates = [
        item
        for item in readiness.get("candidates") or []
        if isinstance(item, dict)
    ]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("sample_count")),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    for candidate in candidates:
        verdict = str(candidate.get("verdict") or "").strip().lower()
        candidate_id = _public_id(candidate.get("candidate_id"), "candidate-id")
        if verdict not in _READY_VERDICTS:
            skipped.append({
                "candidate_id": candidate_id,
                "verdict": verdict or "unknown",
                "reason_codes": ["not-ready-verdict"],
            })
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        observed = _as_float(candidate.get("observed_savings_usd"))
        if observed < float(min_observed_savings_usd):
            skipped.append({
                "candidate_id": candidate_id,
                "verdict": verdict,
                "reason_codes": ["observed-savings-below-threshold"],
                "observed_savings_usd": observed,
            })
            continue
        indexed_rule = indexed.get(candidate_id)
        if not indexed_rule:
            skipped.append({
                "candidate_id": candidate_id,
                "verdict": verdict,
                "reason_codes": ["local-rule-evidence-missing"],
            })
            continue
        rule, blockers = _rule_from_candidate(candidate, indexed_rule, holdout_fraction=holdout)
        if rule is None:
            skipped.append({
                "candidate_id": candidate_id,
                "verdict": verdict,
                "reason_codes": blockers,
            })
            continue
        rules.append(rule)
        accepted.append({
            "candidate_id": candidate_id,
            "rule_id": rule["id"],
            "verdict": verdict,
            "observed_savings_usd": candidate.get("observed_savings_usd"),
            "projected_savings_usd": candidate.get("projected_savings_usd"),
            "sample_count": candidate.get("sample_count"),
            "applied_count": candidate.get("applied_count"),
            "holdout_count": candidate.get("holdout_count"),
            "canary_fraction": rule["rollout"]["canary_fraction"],
            "holdout_fraction": holdout,
            "pattern_hash_count": len(rule["conditions"].get("pattern_hashes") or []),
            "pattern_hashes_included": False,
        })
        if len(accepted) >= max(1, _as_int(max_candidates) or 10):
            break

    shape_cohorts = [
        item
        for item in shape_replay.get("cohorts") or []
        if isinstance(item, dict)
    ]
    shape_cohorts.sort(
        key=lambda item: (
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("projected_hits")),
            _as_int(item.get("row_count")),
        ),
        reverse=True,
    )
    for cohort in shape_cohorts:
        if len(accepted) >= max(1, _as_int(max_candidates) or 10):
            break
        candidate_id = _shape_candidate_id(cohort)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        rule, blockers = _rule_from_request_shape_cohort(cohort, holdout_fraction=holdout)
        if rule is None:
            skipped.append({
                "candidate_id": candidate_id,
                "verdict": str(cohort.get("readiness") or "unknown"),
                "source_schema": shape_replay.get("schema"),
                "reason_codes": blockers,
            })
            continue
        rules.append(rule)
        accepted.append({
            "candidate_id": candidate_id,
            "rule_id": rule["id"],
            "verdict": "ready",
            "source_schema": shape_replay.get("schema"),
            "reason": cohort.get("reason"),
            "cohort_bucket": rule["graduation"].get("cohort_bucket"),
            "projected_hits": cohort.get("projected_hits"),
            "projected_savings_usd": cohort.get("projected_savings_usd"),
            "sample_count": cohort.get("row_count"),
            "canary_fraction": rule["rollout"]["canary_fraction"],
            "holdout_fraction": holdout,
            "pattern_hash_count": 0,
            "pattern_hashes_included": False,
            "aggregate_only": True,
        })

    policy = {
        "schema": POLICY_SCHEMA,
        "generated_at": utc_now(),
        "policy_source": "managed-recommended",
        "pattern_rules": rules,
    }
    canary_dry_run = build_openai_cache_replay_dry_run(store_obj, policy, limit=opportunity_limit) if rules else {}
    canary_summary = canary_dry_run.get("summary") if isinstance(canary_dry_run.get("summary"), dict) else {}
    projected_hits = sum(_as_int(item.get("projected_hits")) for item in accepted)
    projected_savings = sum(_as_float(item.get("projected_savings_usd")) for item in accepted)
    return {
        "schema": PLAN_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(accepted),
        "read_only": True,
        "wrote_local_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "summary": {
            "candidate_count": len(candidates),
            "accepted_candidate_count": len(accepted),
            "skipped_candidate_count": len(skipped),
            "policy_rule_count": len(rules),
            "holdout_fraction": holdout,
            "canary_fraction": round(1.0 - holdout, 6),
            "min_observed_savings_usd": float(min_observed_savings_usd),
            "projected_hits": projected_hits,
            "projected_savings_usd": round(projected_savings, 6),
            "applied_count": _as_int(canary_summary.get("projected_applied_rows")),
            "holdout_count": _as_int(canary_summary.get("holdout_rows")),
            "skipped_count": max(
                0,
                _as_int(canary_summary.get("openai_rows_considered"))
                - _as_int(canary_summary.get("matched_rows")),
            ),
            "safety_stop_count": _safety_stop_count(canary_dry_run),
        },
        "readiness": {
            "schema": readiness.get("schema"),
            "state": readiness.get("state"),
            "state_reason": readiness.get("state_reason"),
            "summary": readiness.get("summary") or {},
        },
        "request_shape_evidence": {
            "schema": request_shape.get("schema"),
            "summary": request_shape.get("summary") or {},
            "cache_replayability_dry_run": {
                "schema": shape_replay.get("schema"),
                "status": shape_replay.get("status"),
                "summary": shape_replay.get("summary") or {},
                "privacy": shape_replay.get("privacy") or {},
            },
        },
        "activation_dry_run": {
            "schema": canary_dry_run.get("schema"),
            "summary": canary_summary,
            "status_breakdown": canary_dry_run.get("status_breakdown") or [],
            "reason_breakdown": canary_dry_run.get("reason_breakdown") or [],
            "blocker_breakdown": canary_dry_run.get("blocker_breakdown") or [],
            "rows": canary_dry_run.get("rows") or [],
            "privacy": canary_dry_run.get("privacy") or {},
        },
        "impact": {
            "schema": impact.get("schema"),
            "summary": impact.get("summary") or {},
            "quality_gate": impact.get("quality_gate") or {},
        },
        "accepted_candidates": accepted,
        "skipped_candidates": skipped[:50],
        "policy": policy,
        "privacy": {
            "metadata_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "request_fingerprints_included": False,
            "pattern_hashes_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }


def _redact_policy_from_result(result: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(result)
    policy = redacted.pop("policy", None)
    if isinstance(policy, dict):
        redacted["policy"] = {
            "schema": policy.get("schema"),
            "generated_at": policy.get("generated_at"),
            "policy_source": policy.get("policy_source"),
            "pattern_rule_count": len(policy.get("pattern_rules") or []),
            "pattern_hashes_included": False,
        }
    return redacted


def apply_openai_cache_replay_candidates(
    store_obj: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
    min_observed_savings_usd: float = 0.0,
    holdout_fraction: float = 0.20,
    max_candidates: int = 10,
) -> dict[str, Any]:
    plan = build_openai_cache_replay_apply_plan(
        store_obj,
        opportunity_limit=opportunity_limit,
        impact_limit=impact_limit,
        min_observed_savings_usd=min_observed_savings_usd,
        holdout_fraction=holdout_fraction,
        max_candidates=max_candidates,
    )
    config_path = Path(config_dir).expanduser()
    path = config_path / CACHE_CANARY_POLICY_FILE
    policy = plan.get("policy") if isinstance(plan.get("policy"), dict) else {}
    text = yaml.safe_dump(policy, sort_keys=False)
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    changed = old_text != text
    backup_path = None
    if plan.get("ok") and changed and not dry_run:
        backup_path = _write_policy_file(path, text)

    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(plan.get("ok")),
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "plan": _redact_policy_from_result(plan),
        "summary": plan.get("summary") or {},
        "accepted_candidates": plan.get("accepted_candidates") or [],
        "skipped_candidates": plan.get("skipped_candidates") or [],
        "files": [{
            "section": "cache",
            "path": str(path),
            "changed": bool(changed and plan.get("ok")),
            "backup_path": backup_path,
            "sha256_before": _sha256_text(old_text) if old_text is not None else None,
            "sha256_after": _sha256_text(text) if plan.get("ok") else None,
            "bytes_after": len(text.encode("utf-8")) if plan.get("ok") else 0,
            "reason": None if plan.get("ok") else "no-ready-openai-cache-replay-candidates",
        }],
        "wrote_policy_files": bool(plan.get("ok") and changed and not dry_run),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": plan.get("privacy") or {},
    }
    return result
