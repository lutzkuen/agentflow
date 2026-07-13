from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import contextmanager
import json
import os
import tempfile
from typing import Any, Iterator, Sequence

from tokenclaw.action_executor import ActionExecutor
from tokenclaw.cli_common import default_db_path, write_json_output
from tokenclaw.managed_action_outcome_feedback import record_managed_action_feedback
from tokenclaw.managed_egress import ManagedEgressBlocked, assert_managed_egress_safe, managed_egress_blocked_meta
from tokenclaw.managed_mode import managed_product_mode
from tokenclaw.policy_files import utc_now
from tokenclaw.store import Store


MANAGED_ACTIVATION_PROOF_SCHEMA = "tokenclaw.managed_activation_proof.v1"
DEFAULT_CANDIDATE = "thinking-tail-compaction"
DEFAULT_SOURCE_SURFACE = "anthropic_messages"
DEFAULT_APP_FAMILY = "claude_code"


@contextmanager
def _temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _stage(name: str, **values: Any) -> dict[str, Any]:
    result = {
        "schema": "tokenclaw.managed_activation_proof_stage.v1",
        "stage": name,
        "metadata_only": True,
    }
    result.update({key: value for key, value in values.items() if value is not None})
    return result


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _queue_counts(store_obj: Store) -> dict[str, int]:
    if not hasattr(store_obj, "managed_outcome_feedback_summary"):
        return {}
    return {
        str(row.get("status") or "unknown"): int(row.get("count") or 0)
        for row in store_obj.managed_outcome_feedback_summary()
    }


def _safe_decision_meta(decision: dict[str, Any]) -> dict[str, Any]:
    crunch = decision.get("crunch") if isinstance(decision.get("crunch"), dict) else {}
    client_contract = decision.get("client_contract") if isinstance(decision.get("client_contract"), dict) else {}
    return {
        "status": decision.get("status"),
        "reason": decision.get("reason"),
        "policy_id": decision.get("policy_id"),
        "decision_id": decision.get("decision_id"),
        "policy_decision_schema": decision.get("policy_decision_schema") or decision.get("schema"),
        "status_code": decision.get("status_code"),
        "latency_ms": decision.get("latency_ms"),
        "fallback": decision.get("fallback"),
        "applied": bool(decision.get("applied")),
        "replacement_prompt_present": bool(decision.get("replacement_prompt_present")),
        "crunch_status": crunch.get("status"),
        "crunch_profile": crunch.get("profile"),
        "candidate_id": crunch.get("candidate_id") or decision.get("candidate_id"),
        "traffic_treatment": (
            crunch.get("traffic_treatment")
            or crunch.get("server_traffic_treatment")
            or decision.get("server_traffic_treatment")
        ),
        "client_contract": {
            "status": client_contract.get("status"),
            "reason": client_contract.get("reason"),
            "active": bool(client_contract.get("active")),
            "contract_id": client_contract.get("contract_id"),
            "cache_status": client_contract.get("cache_status"),
            "filtered": bool(client_contract.get("filtered")),
            "included_path_count": client_contract.get("included_path_count"),
            "metadata_only": True,
        },
        "metadata_only": True,
        "raw_payload_included": False,
    }


def _safe_action_meta(result: dict[str, Any]) -> dict[str, Any]:
    crunch = result.get("crunch") if isinstance(result.get("crunch"), dict) else {}
    return {
        "status": result.get("status"),
        "apply_reason": result.get("apply_reason"),
        "applied": bool(result.get("applied")),
        "applied_families": sorted(result.get("applied_families") or []),
        "supported_local_action_families": sorted(result.get("supported_local_action_families") or []),
        "enabled_local_action_families": sorted(result.get("enabled_local_action_families") or []),
        "product_mode_enforced": bool(result.get("product_mode_enforced")),
        "server_traffic_treatment": result.get("server_traffic_treatment"),
        "canary_fraction": result.get("canary_fraction"),
        "holdout_fraction": result.get("holdout_fraction"),
        "crunch": {
            "status": crunch.get("status"),
            "applied": bool(crunch.get("applied")),
            "apply_reason": crunch.get("apply_reason"),
            "candidate_id": crunch.get("candidate_id"),
            "server_traffic_treatment": crunch.get("server_traffic_treatment"),
            "metadata_only": True,
        },
        "unsupported_action_count": len(result.get("unsupported_actions") or []),
        "metadata_only": True,
        "raw_payload_included": False,
    }


def _proof_unit(*, family: str, candidate: str, queue_counts: dict[str, int]) -> dict[str, Any]:
    requested_actions = [family]
    return {
        "schema": "tokenclaw.managed_activation_proof_feature_unit.v1",
        "feature_schema_version": "tokenclaw.optimization_unit_features.v1",
        "source_surface": DEFAULT_SOURCE_SURFACE,
        "granularity": "provider_request",
        "app_family": DEFAULT_APP_FAMILY,
        "requested_model": "claude-sonnet-4-5-20240620",
        "candidate_target_model": "claude-sonnet-4-5-20240620",
        "input_features": {
            "api_endpoint": "v1_messages",
            "provider_family": "anthropic",
            "category": "tool-result",
            "workflow_phase": "verification",
            "text_chars": 64000,
            "context_token_bucket": "large",
            "requested_local_actions": requested_actions,
            "candidate_id": candidate,
            "queued_feedback_count": int(queue_counts.get("queued", 0)),
            "retryable_feedback_count": int(queue_counts.get("retryable-error", 0)),
        },
        "tool_features": {
            "has_tools": True,
            "tool_count": 1,
        },
        "outcome_features": {
            "recent_status_bucket": "dev-proof",
            "recent_quality_bucket": "unknown",
        },
        "privacy_summary": {
            "metadata_only": True,
            "raw_payload_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "file_paths_included": False,
        },
        "replayability_level": "features_only",
    }


async def build_managed_activation_proof(
    *,
    db_path: str,
    server_url: str | None,
    mode: str | None,
    family: str = "crunch",
    candidate: str = DEFAULT_CANDIDATE,
    drain_limit: int = 25,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    now = utc_now()
    env_updates = {
        "TOKENCLAW_MANAGED": "1" if mode else None,
        "TOKENCLAW_MANAGED_MODE": mode,
        "TOKENCLAW_MANAGED_CRUNCH": "1",
        "TOKENCLAW_MANAGED_ROUTING": "1",
        "TOKENCLAW_MANAGED_CACHE": "1",
        "TOKENCLAW_RECOMMENDATIONS_ENABLED": "1" if mode else None,
        "TOKENCLAW_POLICY_DECISIONS_ENABLED": "1" if mode else None,
        "TOKENCLAW_RECOMMENDATION_SERVER_URL": server_url,
    }
    if timeout_seconds is not None:
        env_updates["TOKENCLAW_RECOMMENDATION_TIMEOUT_SECONDS"] = str(timeout_seconds)

    with _temporary_env(env_updates):
        store_obj = Store(db_path)
        try:
            product_mode = managed_product_mode()
            mode_stage = _stage(
                "mode",
                status="ready" if product_mode.server_calls_enabled and bool(server_url) else "blocked",
                reason=None if product_mode.server_calls_enabled and bool(server_url) else "managed-mode-or-server-url-absent",
                product_mode=product_mode.public_meta(),
                server_url_configured=bool(server_url),
                requested_family=family,
                candidate_id=candidate,
            )
            stages: dict[str, Any] = {"mode": mode_stage}
            if mode_stage["status"] == "blocked":
                report = {
                    "schema": MANAGED_ACTIVATION_PROOF_SCHEMA,
                    "generated_at": now,
                    "status": "blocked",
                    "reason": mode_stage["reason"],
                    "candidate_id": candidate,
                    "local_action_family": family,
                    "stages": stages,
                    "privacy_summary": _privacy_summary(),
                }
                _assert_report_safe(report)
                return report

            from tokenclaw import recommendations
            from tokenclaw.optimization.feedback import managed_feedback_activation_drain_result

            queue_before = _queue_counts(store_obj)
            activation_drain = await managed_feedback_activation_drain_result(
                store_obj,
                limit=max(1, min(int(drain_limit or 1), 100)),
                per_family_limit=max(1, min(int(drain_limit or 1), 100)),
                max_age_seconds=7 * 24 * 60 * 60,
            )
            drain_results = activation_drain.get("results") if isinstance(activation_drain.get("results"), list) else []
            queue_after = _queue_counts(store_obj)
            stages["drain"] = _stage(
                "drain",
                status="drained" if drain_results else activation_drain.get("status") if activation_drain.get("status") != "completed" else "no-due-feedback",
                attempted=True,
                limit=max(1, min(int(drain_limit or 1), 100)),
                per_family_limit=max(1, min(int(drain_limit or 1), 100)),
                result_counts=_status_counts(drain_results),
                queue_before=queue_before,
                queue_after=queue_after,
                expired=activation_drain.get("expired"),
                exhausted_dropped=activation_drain.get("exhausted_dropped"),
                recovered_stale_sending=activation_drain.get("recovered_stale_sending"),
                family_freshness_after=activation_drain.get("family_freshness_after"),
                managed_server_calls_made=bool(drain_results),
            )

            unit = _proof_unit(family=family, candidate=candidate, queue_counts=queue_after)
            decision = await recommendations.fetch_policy_decision(unit)
            decision_meta = _safe_decision_meta(decision)
            contract_status = decision_meta["client_contract"].get("status") or "unknown"
            if decision_meta["client_contract"].get("active") is True and contract_status == "active":
                contract_status = "received"
            stages["contract"] = _stage(
                "contract",
                status=contract_status,
                reason=decision_meta["client_contract"].get("reason"),
                active=decision_meta["client_contract"].get("active"),
                contract_id=decision_meta["client_contract"].get("contract_id"),
                cache_status=decision_meta["client_contract"].get("cache_status"),
            )
            stages["decision"] = _stage(
                "decision",
                **decision_meta,
            )

            action_input = {
                "model": "claude-sonnet-4-5-20240620",
                "max_tokens": 1024,
            }
            routing_meta = {
                "requested_model": "claude-sonnet-4-5-20240620",
                "routed_model": "claude-sonnet-4-5-20240620",
                "category": "tool-result",
            }
            with tempfile.TemporaryDirectory(prefix="tokenclaw-managed-proof-") as tmpdir:
                action_result = ActionExecutor(
                    provider="anthropic",
                    config_dir=tmpdir,
                    store_obj=store_obj,
                    session_id="managed-activation-proof",
                ).execute(
                    body=action_input,
                    routing_meta=routing_meta,
                    decision=copy.deepcopy(decision),
                    application_enabled=mode in {"canary", "live"},
                    shadow_only=mode == "dry_run",
                    source_surface=DEFAULT_SOURCE_SURFACE,
                )
            stages["local_action"] = _stage("local_action", **_safe_action_meta(action_result))

            feedback_meta = await record_managed_action_feedback(
                store_obj,
                action_result,
                source_surface=DEFAULT_SOURCE_SURFACE,
                app_family=DEFAULT_APP_FAMILY,
                contract_id=stages["contract"].get("contract_id"),
                provider="anthropic",
                outcome_metrics={"status_code": 200, "latency_ms": 0},
                flush_immediately=True,
            )
            stages["feedback"] = _stage(
                "feedback",
                status=feedback_meta.get("status"),
                reason=feedback_meta.get("reason"),
                endpoint=feedback_meta.get("endpoint"),
                queue_id_present=bool(feedback_meta.get("queue_id")),
                attempts=feedback_meta.get("attempts"),
                local_result=feedback_meta.get("local_result"),
                payload_included=False,
            )

            status = "ok" if decision_meta.get("status") == "received" else "blocked"
            report = {
                "schema": MANAGED_ACTIVATION_PROOF_SCHEMA,
                "generated_at": now,
                "status": status,
                "reason": None if status == "ok" else decision_meta.get("reason"),
                "candidate_id": candidate,
                "local_action_family": family,
                "stages": stages,
                "summary": {
                    "mode_ready": stages["mode"].get("status") == "ready",
                    "contract_active": stages["contract"].get("active") is True,
                    "decision_received": stages["decision"].get("status") == "received",
                    "local_action_status": stages["local_action"].get("status"),
                    "feedback_status": stages["feedback"].get("status"),
                    "metadata_only": True,
                },
                "privacy_summary": _privacy_summary(),
            }
            report = {key: value for key, value in report.items() if value is not None}
            _assert_report_safe(report)
            return report
        finally:
            store_obj.conn.close()


def _privacy_summary() -> dict[str, Any]:
    return {
        "schema": "tokenclaw.managed_activation_proof_privacy.v1",
        "metadata_only": True,
        "feature_only": True,
        "raw_payload_included": False,
        "raw_prompts_included": False,
        "raw_responses_included": False,
        "provider_bodies_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
        "secrets_included": False,
        "api_key_value_included": False,
        "provider_calls_made": False,
        "prod_port_touched": False,
    }


def _assert_report_safe(report: dict[str, Any]) -> None:
    try:
        assert_managed_egress_safe(report)
    except ManagedEgressBlocked as exc:
        meta = managed_egress_blocked_meta(endpoint=None, violations=exc.violations)
        raise ValueError(json.dumps(meta, sort_keys=True)) from exc


def managed_activation_proof_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Emit a metadata-only managed activation proof bundle")
    parser.add_argument("--family", default="crunch", choices=["routing", "crunch", "cache"])
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--mode", choices=["observe_only", "dry_run", "canary", "live"])
    parser.add_argument("--server-url")
    parser.add_argument("--db", default=default_db_path())
    parser.add_argument("--drain-limit", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--push",
        action="store_true",
        help="POST the proof to the managed server's /v1/managed-activation-proofs ingest.",
    )
    args = parser.parse_args(list(argv or []))
    stdout = stdout or os.sys.stdout
    stderr = stderr or os.sys.stderr

    if not args.mode or not args.server_url:
        report = {
            "schema": MANAGED_ACTIVATION_PROOF_SCHEMA,
            "generated_at": utc_now(),
            "status": "blocked",
            "reason": "managed-mode-or-server-url-absent",
            "candidate_id": args.candidate,
            "local_action_family": args.family,
            "stages": {
                "mode": _stage(
                    "mode",
                    status="blocked",
                    reason="managed-mode-or-server-url-absent",
                    server_url_configured=bool(args.server_url),
                    mode_configured=bool(args.mode),
                )
            },
            "privacy_summary": _privacy_summary(),
        }
        write_json_output(stdout, report, pretty=args.pretty)
        return 2

    try:
        report = asyncio.run(
            build_managed_activation_proof(
                db_path=args.db,
                server_url=args.server_url,
                mode=args.mode,
                family=args.family,
                candidate=args.candidate,
                drain_limit=args.drain_limit,
                timeout_seconds=args.timeout,
            )
        )
    except Exception as exc:
        stderr.write(f"managed activation proof failed: {exc}\n")
        return 1
    if args.push:
        # The proof only feeds the server's widen/promotion gates once it lands
        # in the managed-history rollups; building it locally and printing it —
        # what the retired orchestrator used to pipe onward — leaves those gates
        # starved. The report is metadata-only by construction and re-validated
        # server-side by the ingest schema's feature-boundary check.
        report["push"] = asyncio.run(
            push_managed_activation_proof(
                report,
                server_url=args.server_url,
                timeout_seconds=args.timeout,
            )
        )
    write_json_output(stdout, report, pretty=args.pretty)
    return 0 if report.get("status") == "ok" else 2


async def push_managed_activation_proof(
    report: dict[str, Any],
    *,
    server_url: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    from tokenclaw.http_client import async_client
    from tokenclaw.recommendations import _managed_headers

    def _without_endpoint_paths(value: Any) -> Any:
        # Stage diagnostics carry endpoint *paths* ("/v1/..."), which the ingest
        # schema's identifier-boundary check rightly rejects as path-like. They
        # are local diagnostics the server has no use for; drop them from the
        # pushed copy only.
        if isinstance(value, dict):
            return {
                key: _without_endpoint_paths(item)
                for key, item in value.items()
                if key != "endpoint"
            }
        if isinstance(value, list):
            return [_without_endpoint_paths(item) for item in value]
        return value

    payload = {
        key: _without_endpoint_paths(report[key])
        for key in (
            "schema",
            "generated_at",
            "status",
            "reason",
            "candidate_id",
            "local_action_family",
            "stages",
            "summary",
            "privacy_summary",
        )
        if key in report
    }
    url = server_url.rstrip("/") + "/v1/managed-activation-proofs"
    try:
        async with async_client(timeout=timeout_seconds or 10.0) as client:
            response = await client.post(url, json=payload, headers=_managed_headers())
        try:
            body = response.json()
        except Exception:
            body = None
        return {
            "schema": "tokenclaw.managed_activation_proof_push.v1",
            "status": "sent" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "ingest_status": body.get("status") if isinstance(body, dict) else None,
            "metadata_only": True,
        }
    except Exception as exc:
        return {
            "schema": "tokenclaw.managed_activation_proof_push.v1",
            "status": "error",
            "error_class": type(exc).__name__,
            "metadata_only": True,
        }
