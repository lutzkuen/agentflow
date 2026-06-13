from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import httpx

from agentflow_proxy.cli_commands.policy_bundle import (
    _default_policy_reload_url,
    _is_loopback_url,
    _policy_review_read_error_result,
    _read_policy_json_arg,
    _validation_result_error,
)
from agentflow_proxy.optimization.cli_support import redact_secret as _redact_secret
from agentflow_proxy.optimization.cli_support import write_json as _write_json
from agentflow_proxy.upstream_url import redact_url as _redact_url


POLICY_BUNDLE_RECOMMENDATION_URL_ENV = "AGENTFLOW_POLICY_BUNDLE_RECOMMENDATION_URL"
MANAGED_POLICY_API_KEY_ENV = "AGENTFLOW_MANAGED_API_KEY"


def _policy_fetch_review_error_result(
    *,
    error_type: str,
    message: str,
    url: str | None,
    auth_configured: bool,
    reason: str,
    status_code: int | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    fetch: dict[str, Any] = {
        "status": "skipped" if status_code is None else "error",
        "reason": reason,
        "url": _redact_url(url),
        "auth_configured": bool(auth_configured),
        "status_code": status_code,
    }
    if body is not None:
        fetch["body"] = body[:500]
    return {
        "schema": "agentflow.policy_bundle_fetch_review.v1",
        "ok": False,
        "applied": False,
        "wrote_local_files": False,
        "fetch": fetch,
        "validation": None,
        "review": None,
        "recommendation": {},
        "bundle": None,
        "next_manual_command": None,
        "error": {"type": error_type, "message": message},
    }


def _managed_policy_auth(args: argparse.Namespace) -> tuple[dict[str, str], bool, str]:
    api_key = args.api_key
    source = "argument" if api_key else ""
    if not api_key and args.api_key_env:
        api_key = os.getenv(args.api_key_env)
        if api_key:
            source = f"env:{args.api_key_env}"

    headers: dict[str, str] = {}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    if args.tenant:
        headers["x-agentflow-tenant"] = args.tenant
    if args.account:
        headers["x-agentflow-account"] = args.account
    if api_key:
        return headers, True, source
    if args.allow_unauthenticated:
        return headers, False, "unauthenticated-explicit"
    return headers, False, ""


def _managed_policy_query(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "min_samples": args.min_samples,
        "max_error_rate": args.max_error_rate,
        "limit": args.limit,
    }
    for key in ("source_surface", "app_family", "category"):
        value = getattr(args, key)
        if value:
            params[key] = value
    if _openai_optimization_review_requested(args):
        for key in (
            "provider_endpoint",
            "requested_model_family",
            "max_retry_rate",
            "max_latency_regression_ms",
            "max_invalidation_rate",
        ):
            value = getattr(args, key, None)
            if value is not None and value != "":
                params[key] = value
        params["supported_local_action_families"] = _supported_openai_optimization_action_families(args)
    return params


def _openai_optimization_review_requested(args: argparse.Namespace) -> bool:
    url = str(getattr(args, "url", "") or "")
    if "openai-optimization-review-bundle" in url:
        return True
    for key in (
        "provider_endpoint",
        "requested_model_family",
        "max_retry_rate",
        "max_latency_regression_ms",
        "max_invalidation_rate",
    ):
        value = getattr(args, key, None)
        if value is not None and value != "":
            return True
    return bool(getattr(args, "supported_local_action_families", None))


def _supported_openai_optimization_action_families(args: argparse.Namespace) -> list[str]:
    configured = getattr(args, "supported_local_action_families", None) or []
    if configured:
        return sorted({str(value).strip() for value in configured if str(value).strip()})
    return ["cache", "old_context_summarization", "routing"]


def _managed_policy_capability_headers(args: argparse.Namespace) -> dict[str, str]:
    if not _openai_optimization_review_requested(args):
        return {}
    from agentflow_proxy import __version__

    families = _supported_openai_optimization_action_families(args)
    return {
        "x-agentflow-local-version": __version__,
        "x-agentflow-supported-local-action-families": ",".join(families),
    }


def _count_openai_review_actions_by_family(openai_review: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for decision, key in (
        ("selected", "selected_actions"),
        ("suppressed", "suppressed_actions"),
        ("omitted", "omitted_actions"),
    ):
        actions = openai_review.get(key) if isinstance(openai_review.get(key), list) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            family = str(action.get("action_family") or "unknown")
            row = counts.setdefault(family, {"selected": 0, "suppressed": 0, "omitted": 0})
            row[decision] += 1
    return dict(sorted(counts.items()))


def _openai_review_action_summary(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    compatibility = action.get("local_executor_compatibility") if isinstance(action.get("local_executor_compatibility"), dict) else {}
    surface = action.get("local_policy_surface") if isinstance(action.get("local_policy_surface"), dict) else {}
    return {
        "action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "action_family": action.get("action_family"),
        "candidate_family": action.get("candidate_family"),
        "policy_section": action.get("policy_section"),
        "decision": action.get("decision"),
        "reason_codes": action.get("reason_codes", []),
        "compatible": compatibility.get("compatible"),
        "compatibility_reason_codes": compatibility.get("reason_codes", []),
        "policy_file": surface.get("policy_file"),
        "expected_impact": action.get("expected_impact") if isinstance(action.get("expected_impact"), dict) else {},
    }


def _openai_optimization_review_summary(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return {"schema": "agentflow.openai_optimization_review_summary.v1", "status": "missing"}
    openai_review = bundle.get("openai_optimization") if isinstance(bundle.get("openai_optimization"), dict) else {}
    recommendation = bundle.get("recommendation") if isinstance(bundle.get("recommendation"), dict) else {}
    if not openai_review:
        return {"schema": "agentflow.openai_optimization_review_summary.v1", "status": "missing"}
    selected = openai_review.get("selected_actions") if isinstance(openai_review.get("selected_actions"), list) else []
    suppressed = openai_review.get("suppressed_actions") if isinstance(openai_review.get("suppressed_actions"), list) else []
    omitted = openai_review.get("omitted_actions") if isinstance(openai_review.get("omitted_actions"), list) else []
    all_actions = [*selected, *suppressed, *omitted]
    local_gaps: list[dict[str, Any]] = []
    for action in all_actions:
        if not isinstance(action, dict):
            continue
        compatibility = action.get("local_executor_compatibility") if isinstance(action.get("local_executor_compatibility"), dict) else {}
        reason_codes = [
            str(reason)
            for reason in [
                *(action.get("reason_codes") if isinstance(action.get("reason_codes"), list) else []),
                *(compatibility.get("reason_codes") if isinstance(compatibility.get("reason_codes"), list) else []),
            ]
            if reason
        ]
        if compatibility.get("compatible") is False or any("unsupported" in reason for reason in reason_codes):
            local_gaps.append({
                "target_candidate_id": action.get("target_candidate_id"),
                "action_family": action.get("action_family"),
                "decision": action.get("decision"),
                "reason_codes": sorted(set(reason_codes)),
            })
    return {
        "schema": "agentflow.openai_optimization_review_summary.v1",
        "status": "present",
        "review_bundle_schema": openai_review.get("schema") or recommendation.get("openai_optimization_schema"),
        "selected_action_count": len(selected),
        "suppressed_action_count": len(suppressed),
        "omitted_action_count": len(omitted),
        "counts_by_family": _count_openai_review_actions_by_family(openai_review),
        "conflict_summary": recommendation.get("conflict_summary") if isinstance(recommendation.get("conflict_summary"), dict) else {},
        "local_capability_gaps": local_gaps,
        "selected_actions": [_openai_review_action_summary(action) for action in selected],
        "suppressed_actions": [_openai_review_action_summary(action) for action in suppressed],
        "omitted_actions": [_openai_review_action_summary(action) for action in omitted],
    }


def _managed_recommendation_summary(bundle: Any) -> dict[str, Any]:
    from agentflow_proxy.recommendation_health import summarize_recommendation_health

    if not isinstance(bundle, dict):
        return {}
    recommendation = bundle.get("recommendation")
    if not isinstance(recommendation, dict):
        recommendation = {}
    policies = bundle.get("policies")
    routing = policies.get("routing") if isinstance(policies, dict) and isinstance(policies.get("routing"), dict) else {}
    codex_app = policies.get("codex_app") if isinstance(policies, dict) and isinstance(policies.get("codex_app"), dict) else {}
    crunch = policies.get("crunch") if isinstance(policies, dict) and isinstance(policies.get("crunch"), dict) else {}
    cache = policies.get("cache") if isinstance(policies, dict) and isinstance(policies.get("cache"), dict) else {}
    routing_recommendation = (
        routing.get("recommendation")
        if isinstance(routing, dict) and isinstance(routing.get("recommendation"), dict)
        else {}
    )
    rules = routing.get("rules") if isinstance(routing, dict) and isinstance(routing.get("rules"), list) else []
    candidates: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        managed = rule.get("managed_recommendation")
        if not isinstance(managed, dict):
            continue
        candidates.append({
            key: managed.get(key)
            for key in (
                "candidate_id",
                "confidence",
                "sample_count",
                "success_count",
                "error_count",
                "error_rate",
                "estimated_savings_usd",
                "baseline_sample_count",
                "requested_model",
                "recommended_target_model",
                "source_surface",
                "app_family",
                "category",
                "text_bucket",
                "token_bucket",
            )
            if key in managed
        })

    codex_app_summary: dict[str, Any] = {}
    if codex_app:
        from agentflow_proxy.policy_bundle import codex_app_policy_review_summary

        codex_app_summary = codex_app_policy_review_summary(codex_app)
    pattern_summaries: dict[str, Any] = {}
    old_context_summary: dict[str, Any] = {}
    if crunch or cache:
        from agentflow_proxy.policy_bundle import old_context_summary_policy_review_summary, pattern_policy_review_summary

        if crunch:
            pattern_summaries["crunch"] = pattern_policy_review_summary(crunch, section="crunch")
            old_context_summary = old_context_summary_policy_review_summary(crunch)
        if cache:
            pattern_summaries["cache"] = pattern_policy_review_summary(cache, section="cache")
    pattern_candidate_ids = [
        candidate_id
        for summary in pattern_summaries.values()
        for candidate_id in summary.get("candidate_ids", [])
    ]

    return {
        "schema": recommendation.get("schema"),
        "policy_source": recommendation.get("policy_source"),
        "candidate_ids": recommendation.get("candidate_ids", []),
        "candidate_count": recommendation.get("candidate_count", len(candidates)),
        "routing_rule_count": recommendation.get("routing_rule_count", len(candidates)),
        "codex_app_candidate_ids": codex_app_summary.get("candidate_ids", []),
        "codex_app_candidate_count": codex_app_summary.get("candidate_count", 0),
        "codex_app_review_only": codex_app_summary.get("review_only", False),
        "codex_app_application_status": (codex_app_summary.get("application") or {}).get("status"),
        "pattern_candidate_ids": pattern_candidate_ids,
        "pattern_candidate_count": sum(summary.get("candidate_count", 0) for summary in pattern_summaries.values()),
        "crunch_pattern_candidate_count": pattern_summaries.get("crunch", {}).get("candidate_count", 0),
        "cache_pattern_candidate_count": pattern_summaries.get("cache", {}).get("candidate_count", 0),
        "pattern_review_only_candidate_count": sum(summary.get("review_only_candidate_count", 0) for summary in pattern_summaries.values()),
        "pattern_omitted_candidate_count": sum(summary.get("omitted_candidate_count", 0) for summary in pattern_summaries.values()),
        "old_context_summary_candidate_ids": old_context_summary.get("candidate_ids", []),
        "old_context_summary_candidate_count": old_context_summary.get("candidate_count", 0),
        "old_context_summary_application_status": (old_context_summary.get("application") or {}).get("status"),
        "old_context_summary_warning_codes": old_context_summary.get("warning_codes", []),
        "omitted_candidate_count": recommendation.get(
            "omitted_candidate_count",
            routing_recommendation.get("omitted_candidate_count", 0),
        ),
        "filters": recommendation.get("filters", {}),
        "candidates": candidates,
        "codex_app": codex_app_summary,
        "patterns": pattern_summaries,
        "old_context_summarization": old_context_summary,
        "health": summarize_recommendation_health(bundle),
    }


def _write_policy_fetch_review_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def policy_fetch_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a managed AgentFlow policy bundle recommendation and review it without applying it"
    )
    parser.add_argument(
        "--url",
        default=os.getenv(POLICY_BUNDLE_RECOMMENDATION_URL_ENV),
        help=f"Full managed policy bundle recommendation URL. May also be set with {POLICY_BUNDLE_RECOMMENDATION_URL_ENV}.",
    )
    parser.add_argument(
        "--api-key",
        help=f"Managed optimizer API key. Prefer --api-key-env or {MANAGED_POLICY_API_KEY_ENV} for shell history safety.",
    )
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Fetch without an API key. Intended only for local/dev managed servers.",
    )
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum candidates to request.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument("--provider-endpoint", help="Optional OpenAI optimization provider endpoint filter, such as responses or chat_completions.")
    parser.add_argument("--requested-model-family", help="Optional OpenAI optimization requested model family filter.")
    parser.add_argument("--max-retry-rate", type=float, default=None, help="Maximum OpenAI optimization retry rate to request.")
    parser.add_argument(
        "--max-latency-regression-ms",
        type=float,
        default=None,
        help="Maximum OpenAI optimization latency regression in milliseconds to request.",
    )
    parser.add_argument("--max-invalidation-rate", type=float, default=None, help="Maximum OpenAI cache replay invalidation rate to request.")
    parser.add_argument(
        "--supported-local-action-families",
        action="append",
        choices=("routing", "crunch", "cache", "old_context_summarization"),
        help="Local OpenAI optimization action family supported by this executor. Repeat to send multiple values.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="SQLite metadata database path for local impact simulation, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Maximum recent calls to scan for metadata-only impact simulation, default: 1000.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print fetch/review JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    headers, auth_configured, auth_source = _managed_policy_auth(args)
    headers.update(_managed_policy_capability_headers(args))
    safe_url = _redact_url(args.url)
    from agentflow_proxy.policy_events import log_policy_event

    if not args.url:
        result = _policy_fetch_review_error_result(
            error_type="missing_url",
            message=f"set --url or {POLICY_BUNDLE_RECOMMENDATION_URL_ENV} to enable managed fetch/review",
            url=None,
            auth_configured=auth_configured,
            reason="missing-url",
        )
        log_policy_event(
            "fetch-review",
            ok=False,
            details={"source": "cli", "url": None, "auth_configured": auth_configured, "exit_code": 2},
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 2
    if not auth_configured and not args.allow_unauthenticated:
        result = _policy_fetch_review_error_result(
            error_type="missing_auth",
            message=f"set --api-key, --api-key-env, {MANAGED_POLICY_API_KEY_ENV}, or --allow-unauthenticated",
            url=args.url,
            auth_configured=False,
            reason="missing-auth",
        )
        log_policy_event(
            "fetch-review",
            ok=False,
            details={"source": "cli", "url": safe_url, "auth_configured": False, "exit_code": 2},
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 2

    started = time.time()
    secret = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    try:
        response = httpx.get(args.url, headers=headers, params=_managed_policy_query(args), timeout=args.timeout)
        latency_ms = int((time.time() - started) * 1000)
    except httpx.HTTPError as exc:
        result = _policy_fetch_review_error_result(
            error_type=exc.__class__.__name__,
            message=_redact_secret(str(exc), secret),
            url=args.url,
            auth_configured=auth_configured,
            reason="request-failed",
        )
        result["fetch"]["latency_ms"] = int((time.time() - started) * 1000)
        log_policy_event(
            "fetch-review",
            ok=False,
            details={
                "source": "cli",
                "url": safe_url,
                "auth_configured": auth_configured,
                "auth_source": auth_source,
                "error_type": exc.__class__.__name__,
                "exit_code": 1,
            },
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 1

    if response.status_code >= 400:
        result = _policy_fetch_review_error_result(
            error_type="server_error",
            message="managed server returned an error response",
            url=args.url,
            auth_configured=auth_configured,
            reason="server-error",
            status_code=response.status_code,
            body=_redact_secret(response.text, secret),
        )
        result["fetch"]["latency_ms"] = latency_ms
        log_policy_event(
            "fetch-review",
            ok=False,
            details={
                "source": "cli",
                "url": safe_url,
                "auth_configured": auth_configured,
                "auth_source": auth_source,
                "status_code": response.status_code,
                "exit_code": 1,
            },
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 1

    try:
        bundle = response.json()
    except ValueError as exc:
        validation = _validation_result_error(f"invalid JSON: {exc}", path="$")
        result = _policy_fetch_review_error_result(
            error_type="invalid_json",
            message=f"managed server response was not valid JSON: {exc}",
            url=args.url,
            auth_configured=auth_configured,
            reason="invalid-json",
            status_code=response.status_code,
        )
        result["fetch"]["latency_ms"] = latency_ms
        result["validation"] = validation
        result["review"] = _policy_review_read_error_result(validation)
        log_policy_event(
            "fetch-review",
            ok=False,
            details={
                "source": "cli",
                "url": safe_url,
                "auth_configured": auth_configured,
                "auth_source": auth_source,
                "status_code": response.status_code,
                "proposed_error_count": len(validation["errors"]),
                "exit_code": 1,
            },
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 1

    from agentflow_proxy.policy_bundle import build_policy_bundle, review_policy_bundle, validate_policy_bundle
    from agentflow_proxy.recommendation_health import strip_raw_payload_fields

    validation = validate_policy_bundle(bundle)
    current = asyncio.run(build_policy_bundle())
    review = review_policy_bundle(current, bundle, impact_db_path=args.db, impact_limit=max(0, args.impact_limit))
    recommendation = _managed_recommendation_summary(bundle)
    openai_review = _openai_optimization_review_summary(bundle)
    next_manual_commands = ["agentflow-policy-apply reviewed-bundle.json --dry-run --pretty"]
    if openai_review.get("status") == "present":
        next_manual_commands = [
            "agentflow-policy-draft-stage reviewed-bundle.json --pretty",
            "agentflow-policy-draft-validate <draft-id> --pretty",
        ]
    ok = bool(validation["ok"] and review["ok"])
    result = {
        "schema": "agentflow.policy_bundle_fetch_review.v1",
        "ok": ok,
        "applied": False,
        "wrote_local_files": False,
        "fetch": {
            "status": "received",
            "reason": "ok",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "query": _managed_policy_query(args),
        },
        "validation": validation,
        "review": review,
        "provenance": validation.get("provenance"),
        "recommendation": recommendation,
        "openai_optimization_review": openai_review,
        "bundle": strip_raw_payload_fields(bundle),
        "next_manual_command": next_manual_commands[0],
        "next_manual_commands": next_manual_commands,
        "error": None if ok else {"type": "validation_failed", "message": "managed policy bundle is invalid"},
    }
    result = _redact_secret(result, secret)
    log_policy_event(
        "fetch-review",
        ok=ok,
        details={
            "source": "cli",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "changed": review.get("changed"),
            "changed_sections": review.get("changed_sections", []),
            "change_count": review.get("change_count", 0),
            "safety_warning_count": review.get("safety_warning_count", 0),
            "provenance_status": (validation.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (validation.get("provenance") or {}).get("managed_bundle"),
            "recommendation_health": recommendation.get("health", {}),
            "impact_status": (review.get("impact_summary") or {}).get("status"),
            "proposed_error_count": len(validation.get("errors", [])),
            "candidate_ids": result.get("recommendation", {}).get("candidate_ids", []),
            "candidate_count": result.get("recommendation", {}).get("candidate_count", 0),
            "openai_optimization_review": {
                "status": openai_review.get("status"),
                "selected_action_count": openai_review.get("selected_action_count", 0),
                "suppressed_action_count": openai_review.get("suppressed_action_count", 0),
                "omitted_action_count": openai_review.get("omitted_action_count", 0),
                "local_capability_gap_count": len(openai_review.get("local_capability_gaps", [])),
            },
            "exit_code": 0 if ok else 1,
        },
    )
    _write_policy_fetch_review_result(stdout if ok else stderr, result, pretty=args.pretty)
    return 0 if ok else 1


def _policy_apply_read_error_result(read_error: dict[str, Any], *, config_dir: str, dry_run: bool) -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_apply.v1",
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": config_dir,
        "applied_sections": [],
        "skipped_sections": [],
        "files": [],
        "validation": read_error,
        "safety_warning_count": 0,
        "safety_warnings": [],
        "error": {"type": "read_failed", "message": "policy bundle could not be read"},
    }


def policy_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply an AgentFlow policy bundle to local YAML rule files offline")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Apply only one policy section. Repeat to apply multiple sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the files that would change without writing them.",
    )
    parser.add_argument(
        "--allow-risky",
        action="store_true",
        help="Apply bundles with safety warnings. The warnings are still included in the JSON result.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print apply JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _policy_apply_read_error_result(read_error, config_dir=args.config_dir, dry_run=args.dry_run)
    else:
        from agentflow_proxy.policy_bundle import apply_policy_bundle

        result = apply_policy_bundle(
            bundle,
            config_dir=args.config_dir,
            dry_run=args.dry_run,
            allow_risky=args.allow_risky,
            sections=args.section,
        )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "apply",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "allow_risky": args.allow_risky,
            "applied_sections": result.get("applied_sections", []),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "safety_warning_count": result.get("safety_warning_count", 0),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (result.get("provenance") or {}).get("managed_bundle"),
            "old_context_summarization": result.get("old_context_summarization"),
            "codex_app": result.get("codex_app"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_apply_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def policy_draft_stage_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Stage a local AgentFlow policy draft and return structured diffs without touching active rules"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON/YAML path, section YAML path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--section",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Treat input as one local policy section payload and patch it into the current policy bundle.",
    )
    parser.add_argument(
        "--draft-id",
        help="Optional local draft ID. Unsafe path characters are stripped.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print draft JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.path == "-":
        raw = stdin.read()
    else:
        try:
            raw = Path(args.path).read_text(encoding="utf-8")
        except OSError as exc:
            result = {
                "schema": "agentflow.policy_draft_stage.v1",
                "ok": False,
                "draft": None,
                "draft_id": args.draft_id,
                "workspace": args.workspace,
                "wrote_active_policy_files": False,
                "reloaded_modules": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "diff": None,
                "sections": [],
                "error": {
                    "type": "read_failed",
                    "message": str(exc),
                    "errors": [{"path": args.path, "message": str(exc)}],
                },
            }
            _write_policy_draft_stage_result(stdout, result, pretty=args.pretty)
            return 1

    from agentflow_proxy.openai_optimization_drafts import (
        is_openai_optimization_review_payload,
        stage_openai_optimization_review_draft,
    )
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.policy_files import parse_policy_payload, stage_policy_draft

    payload, parse_error = parse_policy_payload(raw)
    if parse_error:
        result = {
            "schema": "agentflow.policy_draft_stage.v1",
            "ok": False,
            "draft": None,
            "draft_id": args.draft_id,
            "workspace": args.workspace,
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "diff": None,
            "sections": [],
            "error": {"type": "parse_failed", "message": "policy draft payload could not be parsed", "errors": parse_error["errors"]},
        }
    elif is_openai_optimization_review_payload(payload):
        result = asyncio.run(stage_openai_optimization_review_draft(
            payload,
            draft_id=args.draft_id,
            workspace=args.workspace,
        ))
    else:
        result = asyncio.run(stage_policy_draft(
            payload,
            section=args.section,
            draft_id=args.draft_id,
            workspace=args.workspace,
        ))

    log_policy_event(
        "draft-stage",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.path,
            "section": args.section,
            "draft_id": result.get("draft_id"),
            "workspace": result.get("workspace"),
            "changed": (result.get("diff") or {}).get("changed") if isinstance(result.get("diff"), dict) else None,
            "changed_sections": (result.get("diff") or {}).get("changed_sections", []) if isinstance(result.get("diff"), dict) else [],
            "change_count": (result.get("diff") or {}).get("change_count", 0) if isinstance(result.get("diff"), dict) else 0,
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_policy_draft_stage_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def policy_draft_validate_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and dry-run a staged local AgentFlow policy draft before apply"
    )
    parser.add_argument(
        "draft",
        help="Staged draft ID, draft directory, draft.json path, or policy_bundle.json path.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files used by the dry-run apply projection, default: ~/.agentflow.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only impact simulation.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Recent provider metadata rows to inspect for impact simulation, default: 1000.",
    )
    parser.add_argument(
        "--codex-recent-limit",
        type=int,
        default=200,
        help="Recent Codex app metadata rows to inspect for Codex app dry-run projection, default: 200.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print validation JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.policy_workbench import validate_staged_policy_draft

    result = asyncio.run(validate_staged_policy_draft(
        args.draft,
        workspace=args.workspace,
        config_dir=args.config_dir,
        db_path=args.db,
        impact_limit=args.impact_limit,
        codex_recent_limit=args.codex_recent_limit,
    ))

    log_policy_event(
        "draft-validate",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "draft": args.draft,
            "workspace": args.workspace,
            "config_dir": args.config_dir,
            "db": args.db,
            "status": result.get("status"),
            "can_apply": result.get("can_apply"),
            "apply_blocked": result.get("apply_blocked"),
            "changed_sections": (result.get("draft") or {}).get("changed_sections", []) if isinstance(result.get("draft"), dict) else [],
            "section_verdicts": {
                section.get("section"): section.get("verdict")
                for section in result.get("sections", [])
                if isinstance(section, dict)
            },
            "blocker_reason_codes": (result.get("apply_prerequisites") or {}).get("blocker_reason_codes", [])
            if isinstance(result.get("apply_prerequisites"), dict)
            else [],
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_policy_draft_validate_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


async def _reload_policy_state_via_url(url: str, *, timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url)
    try:
        payload = response.json()
    except ValueError:
        payload = {
            "ok": response.is_success,
            "status_code": response.status_code,
            "body": response.text,
            "url": url,
        }
    if isinstance(payload, dict):
        payload.setdefault("status_code", response.status_code)
        payload.setdefault("url", url)
        if not response.is_success:
            payload.setdefault("ok", False)
        return payload
    return {
        "ok": response.is_success,
        "status_code": response.status_code,
        "response": payload,
        "url": url,
    }


def policy_draft_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a validated staged AgentFlow policy draft transactionally, with backup, reload, and verification"
    )
    parser.add_argument(
        "draft",
        help="Staged draft ID, draft directory, draft.json path, or policy_bundle.json path.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only impact simulation.",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Apply only one policy section. Repeat to apply multiple sections.",
    )
    parser.add_argument(
        "--reload-url",
        default=os.getenv("AGENTFLOW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_ADMIN_TIMEOUT", "10")),
        help="HTTP timeout in seconds for the loopback reload call, default: 10.",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow posting reload to a non-loopback URL. Use only for explicit trusted tunnels.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Recent provider metadata rows to inspect for impact simulation, default: 1000.",
    )
    parser.add_argument(
        "--codex-recent-limit",
        type=int,
        default=200,
        help="Recent Codex app metadata rows to inspect for Codex app dry-run projection, default: 200.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print apply JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if not args.allow_non_loopback and not _is_loopback_url(args.reload_url):
        from agentflow_proxy.policy_events import log_policy_event
        from agentflow_proxy.policy_workbench import POLICY_DRAFT_APPLY_SCHEMA

        result = {
            "schema": POLICY_DRAFT_APPLY_SCHEMA,
            "ok": False,
            "status": "blocked",
            "draft_id": args.draft,
            "apply_id": None,
            "backup_id": None,
            "config_dir": args.config_dir,
            "requested_sections": args.section or ["routing", "crunch", "cache", "routing_experiments", "codex_app"],
            "applied_sections": [],
            "changed_sections": [],
            "files": [],
            "backups": [],
            "reloaded_modules": False,
            "reload": None,
            "verification": None,
            "validation": None,
            "restored": False,
            "restore": None,
            "rollback_command": None,
            "privacy": {"provider_calls_made": False, "managed_server_calls_made": False, "loopback_admin_calls_made": False},
            "error": {
                "type": "unsafe_url",
                "message": "policy draft apply only posts reloads to loopback URLs unless --allow-non-loopback is set",
                "url": args.reload_url,
            },
        }
        log_policy_event(
            "draft-apply",
            ok=False,
            details={"source": "cli", "draft_id": args.draft, "error_type": "unsafe_url", "exit_code": 2},
        )
        _write_policy_draft_apply_result(stderr, result, pretty=args.pretty)
        return 2

    from agentflow_proxy.policy_workbench import apply_validated_policy_draft

    async def reload_state() -> dict[str, Any]:
        return await _reload_policy_state_via_url(args.reload_url, timeout=args.timeout)

    result = asyncio.run(apply_validated_policy_draft(
        args.draft,
        workspace=args.workspace,
        config_dir=args.config_dir,
        db_path=args.db,
        impact_limit=args.impact_limit,
        codex_recent_limit=args.codex_recent_limit,
        sections=args.section,
        reload_policy_state=reload_state,
        event_source="cli",
        loopback_admin_calls_made=True,
    ))

    _write_policy_draft_apply_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def _write_policy_apply_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_draft_stage_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_draft_validate_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_draft_apply_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


