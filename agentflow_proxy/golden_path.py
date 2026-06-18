from __future__ import annotations

import copy
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from agentflow_proxy.cache import cache_decision_meta
from agentflow_proxy.optimization.openai_pipeline import (
    execute_openai_local_policy,
    extract_openai_preflight_features,
    parse_openai_request_body,
)
from agentflow_proxy.pricing import estimate_cost, pricing_basis
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


SCHEMA = "agentflow.golden_path_summary.v1"
FIXTURE_SURFACE = "openai_responses"
FIXTURE_ENDPOINT = "responses"
FIXTURE_MODEL = "gpt-5.4-mini"


def _privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_provider_bodies_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _fixture_request() -> dict[str, Any]:
    repeated = (
        "AgentFlow fixture repeated context. "
        "This paragraph is intentionally duplicated to prove local crunching. "
        "No user prompt or provider response content is emitted in the summary. "
    ) * 35
    return {
        "model": FIXTURE_MODEL,
        "stream": False,
        "input": [
            {"role": "user", "content": repeated},
            {"role": "user", "content": repeated},
        ],
        "_agentflow_source_surface": FIXTURE_SURFACE,
        "_agentflow_endpoint": FIXTURE_ENDPOINT,
    }


def _fixture_policy_decision() -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_decision.v1",
        "feature_only": True,
        "locally_executed": True,
        "managed_enforced": False,
        "server_content_processing": False,
        "provider_forwarding": False,
        "routing": {"status": "not-present"},
        "crunch": {"status": "not-present"},
        "cache": {"status": "not-present"},
        "omitted_actions": [{"action": "managed-policy", "reason": "demo-local-only"}],
        "privacy_summary": _privacy(),
    }


def _round_usd(value: float | None) -> float:
    return round(float(value or 0.0), 8)


def _input_savings_usd(model: str, tokens_saved: int) -> float:
    basis = pricing_basis(model, provider="openai")
    input_per_m = float(basis.get("input_usd_per_million") or 0.0)
    return (max(0, int(tokens_saved)) / 1_000_000.0) * input_per_m


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_fixture_summary() -> dict[str, Any]:
    raw_body = _fixture_request()
    parsed = parse_openai_request_body(copy.deepcopy(raw_body), [FIXTURE_MODEL])
    preflight = extract_openai_preflight_features(parsed, path="/v1/responses")
    policy_decision = _fixture_policy_decision()

    with tempfile.TemporaryDirectory(prefix="agentflow-golden-path-") as tmpdir:
        store = SQLiteStore(str(Path(tmpdir) / "agentflow.sqlite3"))
        try:
            local_stage = execute_openai_local_policy(
                raw_body=copy.deepcopy(parsed.body),
                path="/v1/responses",
                requested_model=parsed.requested_model,
                category=parsed.category,
                stream=parsed.stream,
                session_id=None,
                preflight=preflight,
                policy_decision=policy_decision,
                store_obj=store,
            )
            output_tokens = 24
            before_tokens = int(local_stage.crunch_meta.get("tokens_before_est") or parsed.input_tokens_est or 0)
            after_tokens = int(local_stage.crunch_meta.get("tokens_after_est") or local_stage.input_tokens_est or 0)
            tokens_saved = max(0, before_tokens - after_tokens)
            actual_cost = estimate_cost(
                local_stage.routed_model,
                after_tokens,
                output_tokens,
                provider="openai",
            ) or 0.0
            baseline_cost = estimate_cost(
                parsed.requested_model,
                before_tokens,
                output_tokens,
                provider="openai",
            ) or actual_cost
            estimated_savings = max(0.0, baseline_cost - actual_cost)
            if estimated_savings <= 0.0 and tokens_saved > 0:
                estimated_savings = _input_savings_usd(parsed.requested_model, tokens_saved)

            cache_meta = cache_decision_meta("skipped", "demo-no-provider-cache")
            store.log_call(
                id=f"golden-path:{uuid.uuid4().hex}",
                created_at=utc_now(),
                path="/v1/responses",
                requested_model=parsed.requested_model,
                routed_model=local_stage.routed_model,
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=0,
                input_tokens_est=before_tokens,
                output_tokens_est=output_tokens,
                actual_input_tokens=after_tokens,
                actual_output_tokens=output_tokens,
                cost_est_usd=actual_cost,
                cost_baseline_usd=baseline_cost,
                crunch_json=stable_json(local_stage.crunch_meta),
                routing_json=stable_json(local_stage.routing_meta),
                cache_json=stable_json(cache_meta),
                error=None,
                request_json=None,
                response_json=None,
                session_id=None,
                category=parsed.category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="openai",
                source_surface=FIXTURE_SURFACE,
                endpoint=FIXTURE_ENDPOINT,
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
            )
            row_count = int(store.conn.execute("select count(*) as c from calls").fetchone()["c"])
        finally:
            store.conn.close()

    changed = bool(local_stage.crunch_meta.get("changed"))
    local_action_family = "crunch" if changed else "none"
    return {
        "surface": FIXTURE_SURFACE,
        "provider": "openai",
        "endpoint": FIXTURE_ENDPOINT,
        "local_action_family": local_action_family,
        "decision_status": "demo_applied" if changed else "demo_noop",
        "managed_server_required": False,
        "estimated_agentflow_savings_usd": _round_usd(estimated_savings),
        "mocked_provider_response": True,
        "provider_calls_made": False,
        "outcome_evidence_written": row_count == 1,
        "outcome_evidence_store": "ephemeral-fixture",
        "metadata_capture": {
            "preflight_stage": preflight.stage,
            "local_policy_stage": local_stage.stage,
            "feature_summary_schema": preflight.feature_summary.get("schema"),
            "feature_summary_included": True,
            "raw_feature_unit_included": False,
        },
        "local_outcome": {
            "status_code": 200,
            "requested_model": parsed.requested_model,
            "routed_model": local_stage.routed_model,
            "tokens_before_est": before_tokens,
            "tokens_after_est": after_tokens,
            "tokens_saved_est": tokens_saved,
            "crunch_changed": changed,
            "crunch_ratio": local_stage.crunch_meta.get("crunch_ratio"),
            "cache_status": "skipped",
            "cache_reason": "demo-no-provider-cache",
            "request_body_logged": False,
            "response_body_logged": False,
        },
        "privacy": _privacy(),
    }


def _live_evidence_summary(store: Any | None, *, limit: int = 1000) -> dict[str, Any]:
    if store is None:
        return {
            "status": "not_checked",
            "reason": "db-not-provided",
            "surface": FIXTURE_SURFACE,
            "managed_server_required": False,
            "privacy": _privacy(),
        }

    capped = max(1, min(int(limit or 1000), 5000))
    try:
        rows = [
            dict(row)
            for row in store.conn.execute(
                """
                select provider, source_surface, endpoint, requested_model, routed_model,
                       cost_est_usd, cost_baseline_usd, crunch_json, routing_json, cache_json,
                       status_code
                from calls
                where coalesce(provider, 'anthropic') = 'openai'
                order by created_at desc
                limit ?
                """,
                (capped,),
            ).fetchall()
        ]
    except Exception:
        rows = []

    routing_applied = 0
    crunch_changed = 0
    tokens_saved = 0
    estimated_savings = 0.0
    surfaces: set[str] = set()
    endpoints: set[str] = set()
    for row in rows:
        surface = str(row.get("source_surface") or "")
        endpoint = str(row.get("endpoint") or "")
        if surface:
            surfaces.add(surface)
        if endpoint:
            endpoints.add(endpoint)
        requested = str(row.get("requested_model") or "")
        routed = str(row.get("routed_model") or "")
        if requested and routed and requested != routed:
            routing_applied += 1
        baseline = row.get("cost_baseline_usd")
        actual = row.get("cost_est_usd")
        try:
            estimated_savings += max(0.0, float(baseline or 0.0) - float(actual or 0.0))
        except (TypeError, ValueError):
            pass
        crunch_meta = _safe_json_loads(row.get("crunch_json"))
        if crunch_meta.get("changed"):
            crunch_changed += 1
        tokens_saved += int(crunch_meta.get("tokens_saved_est") or 0)

    active = bool(routing_applied or crunch_changed or estimated_savings > 0.0)
    family = "routing" if routing_applied else ("crunch" if crunch_changed else "none")
    return {
        "status": "active" if active else "inactive",
        "surface": FIXTURE_SURFACE,
        "rows_scanned": len(rows),
        "source_surfaces": sorted(surfaces)[:8],
        "endpoints": sorted(endpoints)[:8],
        "local_action_family": family,
        "routing_applied_count": routing_applied,
        "crunch_changed_count": crunch_changed,
        "crunch_tokens_saved_est": max(0, tokens_saved),
        "estimated_agentflow_savings_usd": _round_usd(estimated_savings),
        "managed_server_required": False,
        "routing_coverage": {
            "status": "openai_api_only",
            "covered_surfaces": ["openai_responses", "openai_chat_completions"],
            "codex_coverage": "via-openai-compatible-traffic-when-configured",
            "anthropic_routing_included": False,
        },
        "privacy": _privacy(),
    }


def build_golden_path_summary(
    *,
    store: Any | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    fixture = _build_fixture_summary()
    live = _live_evidence_summary(store, limit=limit)
    live_active = live.get("status") == "active"
    local_action_family = str(live.get("local_action_family") or fixture["local_action_family"])
    if local_action_family == "none":
        local_action_family = fixture["local_action_family"]
    savings = max(
        float(fixture.get("estimated_agentflow_savings_usd") or 0.0),
        float(live.get("estimated_agentflow_savings_usd") or 0.0),
    )
    return {
        "schema": SCHEMA,
        "ok": True,
        "generated_at": utc_now(),
        "surface": FIXTURE_SURFACE,
        "local_action_family": local_action_family,
        "decision_status": "active" if live_active else fixture["decision_status"],
        "estimated_agentflow_savings_usd": _round_usd(savings),
        "managed_server_required": False,
        "provider_calls_made": False,
        "mocked_provider_response": True,
        "fixture": fixture,
        "live_evidence": live,
        "routing_coverage": live.get("routing_coverage")
        or {
            "status": "openai_api_only",
            "covered_surfaces": ["openai_responses", "openai_chat_completions"],
            "codex_coverage": "via-openai-compatible-traffic-when-configured",
            "anthropic_routing_included": False,
        },
        "privacy": _privacy(),
    }
