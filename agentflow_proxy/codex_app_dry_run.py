from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentflow_proxy.codex_app_policy import CODEX_APP_SOURCE_SURFACE, canonical_source_surface
from agentflow_proxy.crunch import TOKEN_CHARS
from agentflow_proxy.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from agentflow_proxy.store import Store


CODEX_APP_DRY_RUN_SCHEMA = "agentflow.codex_app_policy_dry_run.v1"
SUPPORTED_CONDITIONS = {
    "app_family",
    "granularity",
    "workflow_phase",
    "model_field_state",
    "input_size_bucket",
    "cache_eligible",
    "cache_status",
    "replayability_level",
    "has_action_like_params",
    "supported_action_family",
}


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _input_size_bucket(chars: int) -> str:
    if chars < 2_000:
        return "small"
    if chars < 8_000:
        return "medium"
    return "large"


def _candidate_id(rule: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "recommendation_id", "policy_id", "id"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    managed = rule.get("managed_recommendation")
    if isinstance(managed, dict):
        for key in ("candidate_id", "recommendation_id", "policy_id"):
            value = managed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"codex-app-rule-{index + 1}"


def _workflow_phase(window: dict[str, Any], routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any]) -> str:
    for source in (window, cache, crunch, routing):
        value = source.get("workflow_phase") if isinstance(source, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _recent_row_features(row: Any, index: int) -> dict[str, Any]:
    routing = _json_obj(row["routing_json"])
    crunch = _json_obj(row["crunch_json"])
    cache = _json_obj(row["cache_json"])
    window = _json_obj(row["event_window_json"])
    input_chars = _as_int(row["input_text_chars"]) or _as_int(window.get("input_text_chars"))
    result_chars = _as_int(row["result_chars"]) or _as_int(window.get("result_chars"))
    requested_model = (
        routing.get("requested_model")
        or routing.get("routed_model")
        or (window.get("model_state") or {}).get("normalized_model")
        or codex_app_model()
    )
    cache_status = str(cache.get("status") or "unknown")
    cache_eligible = _as_bool(cache.get("eligible"))
    if cache_eligible is None:
        cache_eligible = cache_status in {"miss", "hit", "holdout"}
    has_action_like_params = any(
        str(source.get("reason") or "") == "action-like-params"
        for source in (routing, crunch, cache)
        if isinstance(source, dict)
    )
    return {
        "source": "recent_codex_event",
        "row_index": index,
        "created_at": row["created_at"],
        "app_family": "codex",
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "workflow_phase": _workflow_phase(window, routing, crunch, cache),
        "model_field_state": str(window.get("model_field_state") or "unknown"),
        "input_size_bucket": _input_size_bucket(input_chars),
        "input_text_chars": input_chars,
        "result_chars": result_chars,
        "cache_eligible": bool(cache_eligible),
        "cache_status": cache_status,
        "replayability_level": str(cache.get("replayability_level") or "turn-metadata-only"),
        "has_action_like_params": bool(has_action_like_params),
        "supported_action_family": ["routing", "crunch", "cache"],
        "requested_model": str(requested_model),
        "cache_hit": cache_status == "hit",
        "error_count": _as_int(window.get("error_count")),
        "privacy": {
            "metadata_only": True,
            "raw_payload_included": False,
            "raw_ids_included": False,
        },
    }


def _synthetic_features() -> list[dict[str, Any]]:
    return [
        {
            "source": "synthetic_fixture",
            "row_index": 0,
            "app_family": "codex",
            "source_surface": CODEX_APP_SOURCE_SURFACE,
            "granularity": "agent_turn",
            "workflow_phase": "summary",
            "model_field_state": "derived_present",
            "input_size_bucket": "small",
            "input_text_chars": 640,
            "result_chars": 160,
            "cache_eligible": False,
            "cache_status": "skipped",
            "replayability_level": "turn-metadata-only",
            "has_action_like_params": False,
            "supported_action_family": ["routing", "crunch", "cache"],
            "requested_model": codex_app_model(),
            "cache_hit": False,
            "error_count": 0,
            "privacy": {
                "metadata_only": True,
                "raw_payload_included": False,
                "raw_ids_included": False,
            },
        }
    ]


def load_codex_app_fixture_features(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("features") or payload.get("event_windows") or []
    else:
        rows = payload
    features: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return features
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        window = _json_obj(item.get("event_window") or item.get("event_window_json") or item)
        routing = _json_obj(item.get("routing") or item.get("routing_json"))
        crunch = _json_obj(item.get("crunch") or item.get("crunch_json"))
        cache = _json_obj(item.get("cache") or item.get("cache_json"))
        input_chars = _as_int(item.get("input_text_chars")) or _as_int(window.get("input_text_chars"))
        result_chars = _as_int(item.get("result_chars")) or _as_int(window.get("result_chars"))
        cache_status = str(item.get("cache_status") or cache.get("status") or "unknown")
        cache_eligible = _as_bool(item.get("cache_eligible"))
        if cache_eligible is None:
            cache_eligible = _as_bool(cache.get("eligible"))
        features.append({
            "source": "fixture",
            "row_index": index,
            "app_family": str(item.get("app_family") or "codex"),
            "source_surface": canonical_source_surface(item.get("source_surface") or CODEX_APP_SOURCE_SURFACE),
            "granularity": str(item.get("granularity") or "agent_turn"),
            "workflow_phase": str(item.get("workflow_phase") or _workflow_phase(window, routing, crunch, cache)),
            "model_field_state": str(item.get("model_field_state") or window.get("model_field_state") or "unknown"),
            "input_size_bucket": str(item.get("input_size_bucket") or _input_size_bucket(input_chars)),
            "input_text_chars": input_chars,
            "result_chars": result_chars,
            "cache_eligible": bool(cache_eligible) if cache_eligible is not None else False,
            "cache_status": cache_status,
            "replayability_level": str(item.get("replayability_level") or cache.get("replayability_level") or "turn-metadata-only"),
            "has_action_like_params": bool(_as_bool(item.get("has_action_like_params")) or False),
            "supported_action_family": item.get("supported_action_family") or ["routing", "crunch", "cache"],
            "requested_model": str(item.get("requested_model") or routing.get("requested_model") or codex_app_model()),
            "cache_hit": cache_status == "hit",
            "error_count": _as_int(item.get("error_count") or window.get("error_count")),
            "privacy": {
                "metadata_only": True,
                "raw_payload_included": False,
                "raw_ids_included": False,
            },
        })
    return features


def load_recent_codex_app_features(store: Store, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = store.conn.execute(
        """
        select id,
               created_at,
               input_text_chars,
               result_chars,
               routing_json,
               crunch_json,
               cache_json,
               event_window_json,
               metadata_json
        from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
        order by created_at desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    return [_recent_row_features(row, index) for index, row in enumerate(rows)]


def _condition_value_matches(expected: Any, actual: Any) -> bool:
    expected_bool = _as_bool(expected)
    if expected_bool is not None:
        actual_bool = _as_bool(actual)
        return actual_bool is not None and actual_bool == expected_bool
    return str(actual or "").strip().lower().replace("-", "_") == str(expected or "").strip().lower().replace("-", "_")


def _match_conditions(rule: dict[str, Any], features: dict[str, Any]) -> tuple[bool, list[str]]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    blockers: list[str] = []
    for key in sorted(set(conditions) - SUPPORTED_CONDITIONS):
        blockers.append(f"unsupported-condition:{key}")
    for key, expected in conditions.items():
        if key not in SUPPORTED_CONDITIONS:
            continue
        if key not in features or features.get(key) in {None, ""}:
            blockers.append(f"insufficient-metadata:{key}")
            continue
        if not _condition_value_matches(expected, features.get(key)):
            blockers.append(f"condition-mismatch:{key}")
    return not blockers, blockers


def _canary_cohort(rule: dict[str, Any], candidate_id: str, features: dict[str, Any]) -> tuple[str, str]:
    canary = rule.get("canary")
    if not isinstance(canary, dict):
        managed = rule.get("managed_recommendation")
        canary = managed.get("canary") if isinstance(managed, dict) and isinstance(managed.get("canary"), dict) else {}
    enabled = _as_bool(canary.get("enabled")) if isinstance(canary, dict) else None
    if enabled is False:
        return "applied", "no-canary"
    fraction = float(canary.get("fraction", canary.get("canary_fraction", 1.0)) or 1.0) if isinstance(canary, dict) else 1.0
    holdout = float(canary.get("holdout_fraction", 0.0) or 0.0) if isinstance(canary, dict) else 0.0
    fraction = min(max(fraction, 0.0), 1.0)
    holdout = min(max(holdout, 0.0), 1.0)
    if holdout <= 0.0 and fraction >= 1.0:
        return "applied", "no-holdout-full-fraction"
    material = f"{candidate_id}:{features.get('source')}:{features.get('row_index')}"
    salt = str(canary.get("salt") or "codex-app-policy-dry-run") if isinstance(canary, dict) else "codex-app-policy-dry-run"
    digest = hashlib.sha256(f"{salt}\0{material}".encode("utf-8")).hexdigest()
    sample = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if sample < holdout:
        return "holdout", "canary-holdout"
    if sample < min(1.0, holdout + fraction):
        return "applied", "canary-applied"
    return "skipped", "canary-not-selected"


def _project_savings(rule: dict[str, Any], features: dict[str, Any]) -> float:
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    input_tokens = max(0, _as_int(features.get("input_text_chars")) // TOKEN_CHARS)
    output_tokens = max(0, _as_int(features.get("result_chars")) // TOKEN_CHARS)
    requested_model = str(features.get("requested_model") or codex_app_model())
    baseline = estimate_cost(
        requested_model,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    if baseline is None:
        return 0.0
    projected = float(baseline)
    target_model = action.get("recommended_model") or action.get("model_hint")
    if isinstance(target_model, str) and target_model.strip() and target_model != requested_model:
        routed = estimate_cost(
            target_model,
            input_tokens,
            output_tokens,
            provider="openai",
            processing_mode=codex_app_processing_mode(),
        )
        if routed is not None:
            projected = min(projected, float(routed))
    if _as_bool(action.get("cache_eligible")) is True and not features.get("cache_hit"):
        projected = 0.0
    return max(0.0, round(float(baseline) - projected, 8))


def _cache_row_count(store: Store) -> int | None:
    try:
        row = store.conn.execute("select count(*) as c from cache").fetchone()
        return int(row["c"]) if row is not None else 0
    except Exception:
        return None


def dry_run_codex_app_policy(
    bundle: dict[str, Any],
    *,
    store: Store | None = None,
    recent_limit: int = 200,
    fixture_features: list[dict[str, Any]] | None = None,
    include_synthetic: bool = True,
) -> dict[str, Any]:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    policy = policies.get("codex_app") if isinstance(policies.get("codex_app"), dict) else {}
    rules = policy.get("rules") if isinstance(policy.get("rules"), list) else []
    features: list[dict[str, Any]] = []
    if include_synthetic:
        features.extend(_synthetic_features())
    features.extend(fixture_features or [])
    if store is not None:
        features.extend(load_recent_codex_app_features(store, limit=recent_limit))
    cache_rows_before = _cache_row_count(store) if store is not None else None

    candidates: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    total_applied = 0
    total_holdout = 0
    total_skipped = 0
    total_savings = 0.0

    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        candidate_id = _candidate_id(rule, index)
        projected = {
            "rule_id": str(rule.get("id") or rule.get("rule_id") or candidate_id),
            "candidate_id": candidate_id,
            "policy_source": rule.get("policy_source") or policy.get("policy_source") or "managed-recommended",
            "condition_keys": sorted((rule.get("conditions") or {}).keys()) if isinstance(rule.get("conditions"), dict) else [],
            "action_keys": sorted((rule.get("action") or {}).keys()) if isinstance(rule.get("action"), dict) else [],
            "matched_count": 0,
            "projected_applied_count": 0,
            "projected_holdout_count": 0,
            "projected_skip_count": 0,
            "projected_savings_usd": 0.0,
            "blockers": [],
        }
        local_blockers: dict[str, int] = {}
        for row in features:
            matched, blockers = _match_conditions(rule, row)
            if not matched:
                projected["projected_skip_count"] += 1
                for blocker in blockers:
                    local_blockers[blocker] = local_blockers.get(blocker, 0) + 1
                    blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
                continue
            cohort, reason = _canary_cohort(rule, candidate_id, row)
            projected["matched_count"] += 1
            if cohort == "holdout":
                projected["projected_holdout_count"] += 1
            elif cohort == "applied":
                projected["projected_applied_count"] += 1
                projected["projected_savings_usd"] += _project_savings(rule, row)
            else:
                projected["projected_skip_count"] += 1
                local_blockers[reason] = local_blockers.get(reason, 0) + 1
                blocker_counts[reason] = blocker_counts.get(reason, 0) + 1
        projected["projected_savings_usd"] = round(float(projected["projected_savings_usd"]), 8)
        projected["blockers"] = [
            {"reason": reason, "count": count}
            for reason, count in sorted(local_blockers.items())
        ]
        total_applied += int(projected["projected_applied_count"])
        total_holdout += int(projected["projected_holdout_count"])
        total_skipped += int(projected["projected_skip_count"])
        total_savings += float(projected["projected_savings_usd"])
        candidates.append(projected)

    cache_rows_after = _cache_row_count(store) if store is not None else None
    return {
        "schema": CODEX_APP_DRY_RUN_SCHEMA,
        "ok": True,
        "dry_run": True,
        "applied": False,
        "wrote_local_policy_files": False,
        "cache_table_mutated": bool(
            cache_rows_before is not None
            and cache_rows_after is not None
            and cache_rows_before != cache_rows_after
        ),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy": {
            "surface": canonical_source_surface(policy.get("surface", CODEX_APP_SOURCE_SURFACE)),
            "policy_source": policy.get("policy_source"),
            "review_only": bool(policy.get("review_only", True)),
            "rule_count": len(candidates),
        },
        "summary": {
            "synthetic_rows": len(_synthetic_features()) if include_synthetic else 0,
            "fixture_rows": len(fixture_features or []),
            "recent_rows": max(0, len(features) - len(fixture_features or []) - (len(_synthetic_features()) if include_synthetic else 0)),
            "evaluated_rows": len(features),
            "candidate_count": len(candidates),
            "projected_applied_count": total_applied,
            "projected_holdout_count": total_holdout,
            "projected_skip_count": total_skipped,
            "projected_savings_usd": round(total_savings, 8),
            "cache_rows_before": cache_rows_before,
            "cache_rows_after": cache_rows_after,
        },
        "candidates": candidates,
        "blocker_breakdown": [
            {"reason": reason, "count": count}
            for reason, count in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "privacy": {
            "metadata_only": True,
            "raw_payloads_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "raw_params_included": False,
            "raw_tool_payloads_included": False,
            "raw_session_ids_included": False,
            "raw_request_ids_included": False,
            "cache_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
