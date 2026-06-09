from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlparse, urlunparse

import httpx


POLICY_RELOAD_PATH = "/agentflow/admin/reload-policies"
POLICY_BUNDLE_RECOMMENDATION_URL_ENV = "AGENTFLOW_POLICY_BUNDLE_RECOMMENDATION_URL"
PATTERN_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_PATTERN_ROLLOUT_ACTIONS_URL"
MANAGED_POLICY_API_KEY_ENV = "AGENTFLOW_MANAGED_API_KEY"


def _default_policy_reload_url() -> str:
    port = os.getenv("AGENTFLOW_ADMIN_PORT") or os.getenv("AGENTFLOW_PORT", "4000")
    return f"http://127.0.0.1:{port}{POLICY_RELOAD_PATH}"


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def _write_json(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _redact_url(url: str | None) -> str | None:
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _redact_secret(value: Any, secret: str | None) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[redacted]")
    if isinstance(value, dict):
        return {key: _redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    return value


def _open_store_for_db(db_arg: str) -> Any:
    from agentflow_proxy.store import Store

    old_database_url = os.environ.get("AGENTFLOW_DATABASE_URL")
    try:
        if db_arg.startswith(("postgresql://", "postgres://")):
            os.environ["AGENTFLOW_DATABASE_URL"] = db_arg
            return Store()
        os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        return Store(db_arg)
    finally:
        if old_database_url is None:
            os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        else:
            os.environ["AGENTFLOW_DATABASE_URL"] = old_database_url


def policy_reload_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Reload local AgentFlow policy files through the loopback admin API")
    parser.add_argument(
        "--url",
        default=os.getenv("AGENTFLOW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_ADMIN_TIMEOUT", "10")),
        help="HTTP timeout in seconds, default: 10",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow posting to a non-loopback URL. Use only for explicit trusted tunnels.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if not args.allow_non_loopback and not _is_loopback_url(args.url):
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "reload",
            ok=False,
            details={"source": "cli", "url": args.url, "error_type": "unsafe_url", "exit_code": 2},
        )
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {
                    "type": "unsafe_url",
                    "message": "policy reload CLI only posts to loopback URLs unless --allow-non-loopback is set",
                },
                "url": args.url,
            },
        )
        return 2

    try:
        response = httpx.post(args.url, timeout=args.timeout)
    except httpx.HTTPError as exc:
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "reload",
            ok=False,
            details={"source": "cli", "url": args.url, "error_type": exc.__class__.__name__, "exit_code": 1},
        )
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "url": args.url,
            },
        )
        return 1

    try:
        payload = response.json()
    except ValueError:
        payload = {
            "ok": response.is_success,
            "status_code": response.status_code,
            "body": response.text,
            "url": args.url,
        }

    if response.is_success:
        from agentflow_proxy.policy_events import log_policy_event

        details = {
            "source": "cli",
            "url": args.url,
            "status_code": response.status_code,
            "exit_code": 0,
        }
        if isinstance(payload, dict):
            details["reloaded_modules"] = payload.get("reloaded_modules", [])
            details["policies"] = payload.get("policies")
        log_policy_event("reload", ok=True, details=details)
        _write_json(stdout, payload if isinstance(payload, dict) else {"ok": True, "response": payload})
        return 0

    error_payload = payload if isinstance(payload, dict) else {"ok": False, "response": payload}
    error_payload.setdefault("ok", False)
    error_payload.setdefault("status_code", response.status_code)
    error_payload.setdefault("url", args.url)
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "reload",
        ok=False,
        details={"source": "cli", "url": args.url, "status_code": response.status_code, "exit_code": 1},
    )
    _write_json(stderr, error_payload)
    return 1


def policy_export_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export the effective local AgentFlow policy bundle as JSON")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.policy_bundle import build_policy_bundle
    from agentflow_proxy.policy_events import log_policy_event

    bundle = asyncio.run(build_policy_bundle())
    log_policy_event("export", ok=True, details={"source": "cli", "exit_code": 0, "policies": bundle.get("policies")})
    if args.pretty:
        stdout.write(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, bundle)
    return 0


def _validation_result_error(message: str, *, path: str = "$") -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_validation.v1",
        "ok": False,
        "bundle_schema": None,
        "errors": [{"path": path, "message": message}],
        "warnings": [],
    }


def policy_validate_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Validate an AgentFlow policy bundle JSON file offline")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print validation JSON instead of emitting one compact line.",
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
            result = _validation_result_error(str(exc), path=args.path)
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "validate",
                ok=False,
                details={"source": "cli", "path": args.path, "error_count": 1, "warning_count": 0, "exit_code": 1},
            )
            _write_validation_result(stdout, result, pretty=args.pretty)
            return 1

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        result = _validation_result_error(f"invalid JSON: {exc}", path="$")
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "validate",
            ok=False,
            details={"source": "cli", "path": args.path, "error_count": 1, "warning_count": 0, "exit_code": 1},
        )
        _write_validation_result(stdout, result, pretty=args.pretty)
        return 1

    from agentflow_proxy.policy_bundle import validate_policy_bundle
    from agentflow_proxy.policy_events import log_policy_event

    result = validate_policy_bundle(payload)
    log_policy_event(
        "validate",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "bundle_schema": result.get("bundle_schema"),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (result.get("provenance") or {}).get("managed_bundle"),
            "error_count": len(result.get("errors", [])),
            "warning_count": len(result.get("warnings", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_validation_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _read_policy_json_arg(path: str, *, stdin: Any, stdin_used: bool) -> tuple[Any, dict[str, Any] | None, bool]:
    if path == "-":
        if stdin_used:
            return None, _validation_result_error("stdin can only be used for one policy bundle input"), stdin_used
        raw = stdin.read()
        stdin_used = True
    else:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return None, _validation_result_error(str(exc), path=path), stdin_used

    try:
        return json.loads(raw), None, stdin_used
    except ValueError as exc:
        return None, _validation_result_error(f"invalid JSON: {exc}", path="$"), stdin_used


def _policy_diff_error_result(
    before_validation: dict[str, Any],
    after_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_diff.v1",
        "ok": False,
        "changed": False,
        "changed_sections": [],
        "change_count": 0,
        "changes": [],
        "before_validation": before_validation,
        "after_validation": after_validation,
    }


def policy_diff_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Compare two AgentFlow policy bundle JSON files offline")
    parser.add_argument("before", help="Earlier policy bundle JSON path, or '-' for stdin.")
    parser.add_argument("after", help="Later policy bundle JSON path, or '-' for stdin.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print diff JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    before, before_error, stdin_used = _read_policy_json_arg(args.before, stdin=stdin, stdin_used=False)
    after, after_error, _stdin_used = _read_policy_json_arg(args.after, stdin=stdin, stdin_used=stdin_used)

    if before_error or after_error:
        result = _policy_diff_error_result(
            before_error or _validation_result_error("not validated because the other input could not be read"),
            after_error or _validation_result_error("not validated because the other input could not be read"),
        )
    else:
        from agentflow_proxy.policy_bundle import compare_policy_bundles

        result = compare_policy_bundles(before, after)

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "diff",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "before": args.before,
            "after": args.after,
            "changed": result.get("changed"),
            "changed_sections": result.get("changed_sections", []),
            "change_count": result.get("change_count", 0),
            "before_error_count": len(result.get("before_validation", {}).get("errors", [])),
            "after_error_count": len(result.get("after_validation", {}).get("errors", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_diff_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _policy_review_read_error_result(proposed_validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_review.v1",
        "ok": False,
        "changed": False,
        "changed_sections": [],
        "change_count": 0,
        "safety_warning_count": 0,
        "safety_warnings": [],
        "current_validation": None,
        "proposed_validation": proposed_validation,
        "diff": {
            "schema": "agentflow.policy_bundle_diff.v1",
            "ok": False,
            "changed": False,
            "changed_sections": [],
            "change_count": 0,
            "changes": [],
        },
    }


def policy_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review a proposed AgentFlow policy bundle against current local policy")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Proposed policy bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print review JSON instead of emitting one compact line.",
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
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    proposed, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _policy_review_read_error_result(read_error)
    else:
        from agentflow_proxy.policy_bundle import build_policy_bundle, review_policy_bundle

        current = asyncio.run(build_policy_bundle())
        result = review_policy_bundle(current, proposed, impact_db_path=args.db, impact_limit=max(0, args.impact_limit))

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "review",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "changed": result.get("changed"),
            "changed_sections": result.get("changed_sections", []),
            "change_count": result.get("change_count", 0),
            "safety_warning_count": result.get("safety_warning_count", 0),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (result.get("provenance") or {}).get("managed_bundle"),
            "impact_status": (result.get("impact_summary") or {}).get("status"),
            "proposed_error_count": len((result.get("proposed_validation") or {}).get("errors", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _attach_old_context_summary_lifecycle_feedback(result, command="review", db_path=str(args.db))
    _write_policy_review_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


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
    return params


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
        "bundle": strip_raw_payload_fields(bundle),
        "next_manual_command": "agentflow-policy-apply reviewed-bundle.json --dry-run --pretty",
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
        choices=["routing", "crunch", "cache", "routing_experiments"],
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
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_apply_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _read_rollout_actions_from_url(args: argparse.Namespace) -> tuple[Any, dict[str, Any] | None, int | None]:
    headers, auth_configured, auth_source = _managed_policy_auth(args)
    safe_url = _redact_url(args.url)
    if not auth_configured and not args.allow_unauthenticated:
        return None, {
            "status": "skipped",
            "reason": "missing-auth",
            "url": safe_url,
            "auth_configured": False,
            "auth_source": "",
            "error": {"type": "missing_auth", "message": f"set --api-key, --api-key-env, {MANAGED_POLICY_API_KEY_ENV}, or --allow-unauthenticated"},
        }, 2

    started = time.time()
    secret = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    try:
        response = httpx.get(args.url, headers=headers, params=_managed_policy_query(args), timeout=args.timeout)
        latency_ms = int((time.time() - started) * 1000)
    except httpx.HTTPError as exc:
        return None, {
            "status": "error",
            "reason": "request-failed",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "latency_ms": int((time.time() - started) * 1000),
            "error": {"type": exc.__class__.__name__, "message": _redact_secret(str(exc), secret)},
        }, 1
    if response.status_code >= 400:
        return None, {
            "status": "error",
            "reason": "server-error",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "body": _redact_secret(response.text[:500], secret),
            "error": {"type": "server_error", "message": "managed server returned an error response"},
        }, 1
    try:
        return response.json(), {
            "status": "received",
            "reason": "ok",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "query": _managed_policy_query(args),
        }, None
    except ValueError as exc:
        return None, {
            "status": "error",
            "reason": "invalid-json",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "error": {"type": "invalid_json", "message": f"managed server response was not valid JSON: {exc}"},
        }, 1


def managed_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed pattern rollout actions against local crunch/cache rules")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(PATTERN_ROLLOUT_ACTIONS_URL_ENV),
        help=f"Managed pattern rollout actions URL. May also be set with {PATTERN_ROLLOUT_ACTIONS_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum actions to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["crunch", "cache"],
        help="Review only one policy section. Repeat to review multiple sections.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print review JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    if args.url:
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            from agentflow_proxy.policy_events import log_policy_event

            result = {
                "schema": "agentflow.pattern_rollout_actions_fetch_review.v1",
                "ok": False,
                "fetch": fetch,
                "review": None,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            log_policy_event(
                "rollout-actions-review",
                ok=False,
                details={"source": "cli", "url": _redact_url(args.url), "fetch_status": fetch.get("status"), "exit_code": fetch_exit},
            )
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.rollout_actions import plan_rollout_actions

    review = plan_rollout_actions(bundle, config_dir=args.config_dir, sections=args.section)
    if fetch:
        review["fetch"] = fetch
    log_policy_event(
        "rollout-actions-review",
        ok=bool(review["ok"]),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "config_dir": args.config_dir,
            "action_count": review.get("action_count", 0),
            "planned_action_count": review.get("planned_action_count", 0),
            "changed_action_count": review.get("changed_action_count", 0),
            "provenance_status": (review.get("provenance") or {}).get("status"),
            "error_count": len(review.get("errors", [])),
            "exit_code": 0 if review["ok"] else 1,
        },
    )
    _attach_rollout_lifecycle_feedback(review, command="review", db_path=str(args.db))
    _write_rollout_actions_result(stdout if review["ok"] else stderr, review, pretty=args.pretty)
    return 0 if review["ok"] else 1


def managed_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed managed pattern rollout actions to local YAML rule files")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["crunch", "cache"],
        help="Apply only one policy section. Repeat to apply multiple sections.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report exact YAML edits without writing files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.pattern_rollout_actions_apply.v1",
            "ok": False,
            "dry_run": bool(args.dry_run),
            "config_dir": args.config_dir,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "rollout action bundle could not be read"},
            "files": [],
            "actions": [],
        }
    else:
        from agentflow_proxy.rollout_actions import apply_rollout_actions

        result = apply_rollout_actions(
            bundle,
            config_dir=args.config_dir,
            dry_run=args.dry_run,
            sections=args.section,
        )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "rollout-actions-apply",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "applied_sections": result.get("applied_sections", []),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _attach_rollout_lifecycle_feedback(result, command="apply", db_path=str(args.db))
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def managed_rollout_actions_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run managed pattern rollout actions against recent local traffic metadata")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls and Codex turns to inspect, default: 500.")
    parser.add_argument(
        "--section",
        action="append",
        choices=["crunch", "cache"],
        help="Dry-run only one policy section. Repeat to include multiple sections.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.pattern_rollout_actions_dry_run.v1",
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "config_dir": args.config_dir,
            "db_path": args.db,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "rollout action bundle could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.rollout_actions import dry_run_rollout_actions

        store = _open_store_for_db(args.db)
        try:
            result = dry_run_rollout_actions(
                bundle,
                store_obj=store,
                config_dir=args.config_dir,
                sections=args.section,
                limit=args.limit,
            )
            result["db_path"] = args.db
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "rollout-actions-dry-run",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "config_dir": args.config_dir,
            "db_path": args.db,
            "dry_run": True,
            "action_count": len(result.get("actions", [])),
            "affected_metadata_row_count": (result.get("summary") or {}).get("affected_metadata_row_count"),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _attach_rollout_lifecycle_feedback(result, command="dry-run", db_path=str(args.db))
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def old_context_summary_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run old-context summarization against recent local traffic")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional policy bundle JSON path, '-' for stdin, or omit to use the currently loaded local crunch policy.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle_or_policy: Any = None
    read_error: dict[str, Any] | None = None
    if args.path is not None:
        bundle_or_policy, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)

    if read_error:
        from agentflow_proxy.old_context_summary_dry_run import OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA

        result = {
            "schema": OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA,
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "db_path": args.db,
            "lookback_call_limit": max(0, args.limit),
            "validation": read_error,
            "error": {"type": "read_failed", "message": "policy bundle could not be read"},
            "groups": [],
        }
    else:
        from agentflow_proxy.old_context_summary_dry_run import dry_run_old_context_summary

        result = dry_run_old_context_summary(
            bundle_or_policy,
            db_path=str(args.db),
            limit=max(0, args.limit),
        )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.path or "current-policy",
            "db_path": args.db,
            "dry_run": True,
            "eligible_call_count": (result.get("summary") or {}).get("eligible_call_count"),
            "projected_saved_tokens": (result.get("summary") or {}).get("projected_saved_tokens"),
            "projected_net_savings_usd": (result.get("summary") or {}).get("projected_net_savings_usd"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _attach_old_context_summary_lifecycle_feedback(result, command="dry-run", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def old_context_summary_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Measure post-apply old-context summarization canary impact against a dry-run projection")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Old-context summary dry-run or policy review JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500.")
    parser.add_argument(
        "--since",
        help="Only count metadata at or after this ISO-8601 post-apply timestamp. Defaults to dry-run generated_at.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print impact JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        from agentflow_proxy.old_context_summary_impact import OLD_CONTEXT_SUMMARY_IMPACT_SCHEMA

        result = {
            "schema": OLD_CONTEXT_SUMMARY_IMPACT_SCHEMA,
            "ok": False,
            "read_only": True,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary dry-run or review report could not be read"},
            "summary": {},
            "privacy": {
                "metadata_only": True,
                "raw_old_context_included": False,
                "generated_summaries_included": False,
                "raw_request_bodies_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "tenant_ids_included": False,
                "local_session_ids_included": False,
                "cache_keys_included": False,
            },
        }
    else:
        from agentflow_proxy.old_context_summary_impact import measure_old_context_summary_impact

        store = _open_store_for_db(args.db)
        try:
            result = measure_old_context_summary_impact(
                report,
                store_obj=store,
                limit=args.limit,
                since=args.since,
            )
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    log_policy_event(
        "old-context-summary-impact",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "input_source": "stdin" if args.path == "-" else "file",
            "db_configured": bool(args.db),
            "since": args.since,
            "projected_affected_metadata_row_count": summary.get("projected_affected_metadata_row_count"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "quality_gate_verdict": quality_gate.get("verdict"),
            "quality_gate_reason_codes": quality_gate.get("reason_codes"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _attach_old_context_summary_lifecycle_feedback(result, command="impact", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def managed_rollout_actions_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Measure post-apply rollout-action impact against a dry-run projection")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action dry-run JSON report path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls and Codex turns to inspect, default: 500.")
    parser.add_argument(
        "--since",
        help="Only count provider/Codex metadata at or after this ISO-8601 post-apply timestamp. Defaults to dry-run generated_at.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print impact JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    dry_run_report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.pattern_rollout_actions_impact.v1",
            "ok": False,
            "read_only": True,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "rollout action dry-run report could not be read"},
            "actions": [],
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "raw_responses_included": False,
                "tool_payloads_included": False,
                "request_ids_included": False,
                "local_session_ids_included": False,
                "file_paths_included": False,
                "yaml_contents_included": False,
            },
        }
    else:
        from agentflow_proxy.rollout_actions import measure_rollout_action_impact

        store = _open_store_for_db(args.db)
        try:
            result = measure_rollout_action_impact(
                dry_run_report,
                store_obj=store,
                limit=args.limit,
                since=args.since,
            )
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "rollout-actions-impact",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "input_source": "stdin" if args.path == "-" else "file",
            "db_configured": bool(args.db),
            "since": args.since,
            "action_count": len(result.get("actions", [])),
            "projected_affected_metadata_row_count": summary.get("projected_affected_metadata_row_count"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_matched_provider_call_count": summary.get("actual_matched_provider_call_count"),
            "actual_matched_codex_turn_count": summary.get("actual_matched_codex_turn_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_estimated_cost_savings_usd": summary.get("actual_estimated_cost_savings_usd"),
            "actions_without_post_apply_matches": summary.get("actions_without_post_apply_matches"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def policy_rollback_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Rollback local AgentFlow policy YAML files from apply backups")
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments"],
        help="Rollback only one policy section. Repeat to rollback multiple sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the backups that would be restored without writing files.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print rollback JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.policy_bundle import rollback_policy_files
    from agentflow_proxy.policy_events import log_policy_event

    result = rollback_policy_files(config_dir=args.config_dir, dry_run=args.dry_run, sections=args.section)
    log_policy_event(
        "rollback",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "restored_sections": result.get("restored_sections", []),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "old_context_summarization": result.get("old_context_summarization"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_rollback_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _parse_utc_iso(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _seconds_since(raw: Any, now: datetime) -> int | None:
    parsed = _parse_utc_iso(raw)
    if parsed is None:
        return None
    return max(0, int((now - parsed).total_seconds()))


def _breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _public_feedback_row(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    optimization_unit_id = row.get("optimization_unit_id")
    if optimization_unit_id in (0, "0"):
        optimization_unit_id = None
    return {
        "queue_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "optimization_unit_id": optimization_unit_id,
        "status": row.get("status"),
        "attempts": row.get("attempts") or 0,
        "next_attempt_at": row.get("next_attempt_at"),
        "last_status_code": row.get("last_status_code"),
        "sent_at": row.get("sent_at"),
        "age_seconds": _seconds_since(row.get("created_at"), now),
        "payload_included": False,
    }


def _managed_feedback_config() -> dict[str, Any]:
    from agentflow_proxy import recommendations

    return {
        "enabled": recommendations.recommendations_enabled(),
        "server_url": _redact_url(recommendations.recommendation_server_url()),
        "server_configured": recommendations.recommendation_server_configured(),
        "timeout_seconds": recommendations.recommendation_timeout_seconds(),
        "failure_mode": recommendations.recommendation_failure_mode(),
        "queue_max_attempts": recommendations.outcome_feedback_queue_max_attempts(),
        "queue_retry_delay_seconds": recommendations.outcome_feedback_queue_retry_delay_seconds(),
        "auth_configured": recommendations.managed_auth_configured(),
        "api_key_value_included": False,
    }


def _managed_feedback_status_result(
    store: Any,
    *,
    source_surface: str | None,
    sample_limit: int,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    generated_at = now.isoformat()
    status_counts = {
        str(row.get("status") or "unknown"): int(row.get("count") or 0)
        for row in store.managed_outcome_feedback_summary(source_surface=source_surface)
    } if hasattr(store, "managed_outcome_feedback_summary") else {}
    rows = (
        store.managed_outcome_feedback_rows(source_surface=source_surface, limit=10000)
        if hasattr(store, "managed_outcome_feedback_rows")
        else []
    )
    due_rows = (
        store.due_managed_outcome_feedback(limit=max(1, sample_limit), source_surface=source_surface)
        if hasattr(store, "due_managed_outcome_feedback")
        else []
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_surface") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    pending_rows = [
        row
        for row in rows
        if row.get("status") in {"queued", "retryable-error"}
    ]
    oldest_pending = min(
        pending_rows,
        key=lambda row: _parse_utc_iso(row.get("created_at")) or now,
        default=None,
    )
    dropped = status_counts.get("dropped-after-limit", 0)
    summary = {
        "total": sum(status_counts.values()),
        "queued": status_counts.get("queued", 0),
        "retryable_error": status_counts.get("retryable-error", 0),
        "sending": status_counts.get("sending", 0),
        "sent": status_counts.get("sent", 0),
        "dropped_after_limit": dropped,
        "error": status_counts.get("error", 0),
        "due": len(due_rows),
        "oldest_pending_age_seconds": _seconds_since(oldest_pending.get("created_at"), now) if oldest_pending else None,
        "retry_limit_drops": dropped,
    }
    return {
        "schema": "agentflow.managed_feedback_status.v1",
        "ok": True,
        "generated_at": generated_at,
        "source_surface": source_surface,
        "managed_feedback": _managed_feedback_config(),
        "summary": summary,
        "status_breakdown": _breakdown_from_counts(status_counts),
        "source_surface_breakdown": _breakdown_from_counts(source_counts),
        "oldest_pending": _public_feedback_row(oldest_pending, now=now) if oldest_pending else None,
        "due_samples": [
            _public_feedback_row(row, now=now)
            for row in due_rows[:max(0, sample_limit)]
        ],
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }


def _safe_managed_feedback_flush_result(result: dict[str, Any]) -> dict[str, Any]:
    secret = os.getenv(MANAGED_POLICY_API_KEY_ENV)
    safe = _redact_secret(result, secret)
    if isinstance(safe.get("managed_feedback"), dict):
        safe["managed_feedback"]["server_url"] = _redact_url(safe["managed_feedback"].get("server_url"))
    for item in safe.get("results", []) if isinstance(safe.get("results"), list) else []:
        if isinstance(item, dict) and "server_url" in item:
            item["server_url"] = _redact_url(item.get("server_url"))
    return safe


def managed_feedback_status_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report local managed outcome feedback queue status")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--source-surface", help="Optional queue source surface filter, for example codex_turn.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum due queue samples to include, default: 20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    store = _open_store_for_db(str(args.db))
    try:
        result = _managed_feedback_status_result(
            store,
            source_surface=args.source_surface,
            sample_limit=max(0, min(args.limit, 100)),
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def managed_feedback_flush_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Flush due local managed outcome feedback queue rows in bounded batches")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--source-surface", help="Optional queue source surface filter, for example codex_turn.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum due rows to flush, default: 5, max: 100.")
    parser.add_argument("--dry-run", action="store_true", help="Report due rows without claiming or sending them.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    limit = max(1, min(args.limit, 100))
    store = _open_store_for_db(str(args.db))
    try:
        before = _managed_feedback_status_result(store, source_surface=args.source_surface, sample_limit=limit)
        if args.dry_run:
            results = [
                {**row, "status": "would-send"}
                for row in before.get("due_samples", [])
            ]
            flush_status = "dry-run"
            reason = "dry-run"
        else:
            from agentflow_proxy import recommendations

            if recommendations.recommendations_enabled():
                results = asyncio.run(
                    recommendations.flush_queued_outcome_feedback(
                        store,
                        limit=limit,
                        source_surface=args.source_surface,
                    )
                )
                flush_status = "completed"
                reason = "ok"
            else:
                results = []
                flush_status = "skipped"
                reason = "managed-feedback-disabled"
        after = _managed_feedback_status_result(store, source_surface=args.source_surface, sample_limit=limit)
    finally:
        store.conn.close()

    result_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        result_counts[status] = result_counts.get(status, 0) + 1
    result = {
        "schema": "agentflow.managed_feedback_flush.v1",
        "ok": True,
        "dry_run": bool(args.dry_run),
        "source_surface": args.source_surface,
        "limit": limit,
        "flush": {
            "status": flush_status,
            "reason": reason,
            "attempted": len(results) if not args.dry_run else 0,
            "would_attempt": len(results) if args.dry_run else 0,
            "sent": result_counts.get("sent", 0),
            "retryable_error": result_counts.get("retryable-error", 0),
            "dropped_after_limit": result_counts.get("dropped-after-limit", 0),
        },
        "managed_feedback": _managed_feedback_config(),
        "before": before["summary"],
        "after": after["summary"],
        "result_breakdown": _breakdown_from_counts(result_counts),
        "results": results,
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }
    result = _safe_managed_feedback_flush_result(result)
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def codex_diagnose_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Codex app-server routing, crunching, and cache effectiveness from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent turn/start rows to inspect, default: 500, max: 5000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_codex_effectiveness

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_codex_effectiveness(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def managed_pattern_rollups_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export metadata-only managed pattern canary cohort outcome rollups")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls and Codex turn/start rows to inspect per surface, default: 500, max: 5000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required before a cohort bucket is marked ready, default: 10",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_managed_pattern_rollups

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_managed_pattern_rollups(store, limit=args.limit, min_samples=args.min_samples))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _write_validation_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_diff_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_review_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_apply_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_rollout_actions_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _rollout_lifecycle_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rollout_action_id(action: dict[str, Any]) -> str:
    basis = {
        "policy_section": action.get("policy_section"),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "pattern_hash": action.get("pattern_hash"),
        "action_type": action.get("action_type"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"rollout-action:{digest[:24]}"


def _rollout_action_snapshot(action: dict[str, Any]) -> dict[str, Any]:
    edit = action.get("proposed_edit") if isinstance(action.get("proposed_edit"), dict) else {}
    current_rule = action.get("current_rule") if isinstance(action.get("current_rule"), dict) else {}
    snapshot = {
        "action_id": _rollout_action_id(action),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "policy_section": action.get("policy_section"),
        "policy_source": current_rule.get("policy_source"),
        "pattern_hash": action.get("pattern_hash"),
        "action_type": action.get("action_type"),
        "current_fraction": edit.get("current_fraction") if edit else action.get("current_fraction"),
        "recommended_fraction": edit.get("recommended_fraction") if edit else action.get("projected_fraction"),
        "confidence": action.get("confidence"),
        "blockers": action.get("blockers") if isinstance(action.get("blockers"), list) else [],
        "required_local_review": True,
        "managed_enforced": False,
    }
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, "", [], {})
    }


def _rollout_lifecycle_event_type(command: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return "rejected"
    if command == "review":
        return "reviewed"
    if command == "dry-run" or result.get("dry_run"):
        return "dry-run"
    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    action_types = {str(item.get("action_type") or "") for item in actions}
    if action_types & {"rollback", "retire", "disable"}:
        return "rollback"
    return "applied"


def _rollout_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy import __version__

    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    snapshots = [_rollout_action_snapshot(item) for item in actions]
    action_ids = sorted({str(item["action_id"]) for item in snapshots if item.get("action_id")})
    candidate_ids = sorted({str(item.get("target_candidate_id")) for item in snapshots if item.get("target_candidate_id")})
    rule_ids = sorted({str(item.get("target_rule_id")) for item in snapshots if item.get("target_rule_id")})
    pattern_hashes = sorted({str(item.get("pattern_hash")) for item in snapshots if str(item.get("pattern_hash") or "").startswith("sha256:")})
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    files = [item for item in result.get("files", []) if isinstance(item, dict)]
    event_type = _rollout_lifecycle_event_type(command, result)
    metadata: dict[str, Any] = {
        "schema": "agentflow.rollout_action_lifecycle_metadata.v1",
        "lifecycle_kind": "pattern_rollout_actions",
        "command": f"rollout-actions-{command}",
        "local_result_status": "ok" if result.get("ok") else "error",
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only")),
        "action_count": len(actions) or result.get("action_count") or validation.get("action_count") or 0,
        "planned_action_count": result.get("planned_action_count") or review.get("planned_action_count") or 0,
        "changed_action_count": result.get("changed_action_count") or review.get("changed_action_count") or 0,
        "rejected_action_count": result.get("rejected_action_count") or review.get("rejected_action_count") or 0,
        "action_type_counts": _rollout_lifecycle_counts(actions, "action_type"),
        "policy_section_counts": _rollout_lifecycle_counts(actions, "policy_section"),
        "local_status_counts": _rollout_lifecycle_counts(actions, "status"),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "pattern_hashes": pattern_hashes,
        "rollout_action_snapshots": snapshots,
        "validation_error_count": len(validation.get("errors", []) if isinstance(validation.get("errors"), list) else []),
        "validation_warning_count": len(validation.get("warnings", []) if isinstance(validation.get("warnings"), list) else []),
        "review_error_count": len(review.get("errors", []) if isinstance(review.get("errors"), list) else []),
        "review_warning_count": len(review.get("warnings", []) if isinstance(review.get("warnings"), list) else []),
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "provenance_status": provenance.get("status"),
        "provenance_bundle_hash": provenance.get("bundle_hash"),
        "computed_bundle_hash": provenance.get("computed_bundle_hash"),
        "changed_file_count": sum(1 for item in files if item.get("changed")),
        "changed_sections": sorted({str(item.get("section")) for item in files if item.get("changed") and item.get("section")}),
        "affected_metadata_row_count": summary.get("affected_metadata_row_count"),
        "affected_provider_call_count": summary.get("affected_provider_call_count"),
        "affected_codex_turn_count": summary.get("affected_codex_turn_count"),
        "projected_additional_applied_count": summary.get("projected_additional_applied_count"),
        "projected_local_bypass_or_disable_count": summary.get("projected_local_bypass_or_disable_count"),
        "historical_tokens_saved_est": summary.get("historical_tokens_saved_est"),
        "historical_estimated_cost_savings_usd": summary.get("historical_estimated_cost_savings_usd"),
        "safety_stop_reason_counts": _rollout_lifecycle_counts(
            [
                reason
                for action in actions
                for reason in (action.get("local_bypass_reasons") or [])
                if isinstance(reason, dict)
            ],
            "value",
        ),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "raw_params_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": action_ids[0] if len(action_ids) == 1 else None,
        "bundle_hash": provenance.get("computed_bundle_hash") or provenance.get("bundle_hash"),
        "policy_sections": sorted(_rollout_lifecycle_counts(actions, "policy_section")),
        "validation_warning_count": metadata.get("validation_warning_count", 0),
        "review_warning_count": metadata.get("review_warning_count", 0),
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _public_lifecycle_feedback_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "enabled": meta.get("enabled"),
            "server_url": _redact_url(meta.get("server_url")),
            "endpoint": meta.get("endpoint"),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "status_code": meta.get("status_code"),
            "latency_ms": meta.get("latency_ms"),
            "auth_configured": meta.get("auth_configured"),
            "api_key_value_included": False,
            "payload_included": False,
        }.items()
        if value is not None
    }


def _attach_rollout_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _rollout_lifecycle_payload(command, result)
    if not recommendations.recommendations_enabled():
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            **recommendations.disabled_outcome_feedback_meta(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "disabled",
        })
        return

    store = _open_store_for_db(str(db_path))
    try:
        meta = asyncio.run(recommendations.queue_policy_event_feedback(store, payload))
    except Exception as exc:
        meta = {
            "enabled": True,
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if command in {"dry-run", "impact"} and public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def _old_context_summary_lifecycle_result(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if command == "review":
        impact = result.get("impact_summary") if isinstance(result.get("impact_summary"), dict) else {}
        sections = impact.get("sections") if isinstance(impact.get("sections"), dict) else {}
        crunch = sections.get("crunch") if isinstance(sections.get("crunch"), dict) else {}
        dry_run = crunch.get("old_context_summary_dry_run")
        return dry_run if isinstance(dry_run, dict) else None
    return result


def _old_context_summary_lifecycle_event_type(command: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return "rejected"
    if command == "impact":
        return "impact"
    if command == "review":
        return "reviewed"
    return "dry-run"


def _old_context_summary_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    from agentflow_proxy import __version__

    dry_run = _old_context_summary_lifecycle_result(command, result)
    if not isinstance(dry_run, dict):
        return None
    if command == "impact":
        dry_run_meta = dry_run.get("dry_run") if isinstance(dry_run.get("dry_run"), dict) else {}
        policy = dry_run_meta.get("policy") if isinstance(dry_run_meta.get("policy"), dict) else {}
        projection = dry_run_meta.get("projection") if isinstance(dry_run_meta.get("projection"), dict) else {}
        summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), dict) else {}
        actual = dry_run.get("actual") if isinstance(dry_run.get("actual"), dict) else {}
        delta = dry_run.get("delta") if isinstance(dry_run.get("delta"), dict) else {}
        quality_gate = dry_run.get("quality_gate") if isinstance(dry_run.get("quality_gate"), dict) else {}
        basis = {
            "command": command,
            "rule_id": policy.get("rule_id"),
            "candidate_id": policy.get("candidate_id"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
        }
        digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
        metadata = {
            "schema": "agentflow.old_context_summary_lifecycle_metadata.v1",
            "lifecycle_kind": "old_context_summarization",
            "command": "old-context-summary-impact",
            "local_result_status": "ok" if dry_run.get("ok") else "error",
            "dry_run": False,
            "read_only": bool(dry_run.get("read_only", True)),
            "policy_source": policy.get("policy_source"),
            "rule_id": policy.get("rule_id"),
            "candidate_id": policy.get("candidate_id"),
            "model": policy.get("model"),
            "canary_enabled": ((policy.get("canary") or {}).get("enabled") if isinstance(policy.get("canary"), dict) else None),
            "canary_fraction": ((policy.get("canary") or {}).get("fraction") if isinstance(policy.get("canary"), dict) else None),
            "safety_stop_enabled": ((policy.get("safety_stop") or {}).get("enabled") if isinstance(policy.get("safety_stop"), dict) else None),
            "projected_affected_metadata_row_count": projection.get("projected_affected_metadata_row_count"),
            "projected_canary_applied_count": projection.get("projected_canary_applied_count"),
            "projected_canary_holdout_count": projection.get("projected_canary_holdout_count"),
            "projected_saved_tokens": projection.get("projected_saved_tokens"),
            "projected_net_savings_usd": projection.get("projected_net_savings_usd"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "error_rate": summary.get("error_rate"),
            "retry_rate": summary.get("retry_rate"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_gross_savings_usd": summary.get("actual_gross_savings_usd"),
            "actual_summary_model_cost_usd": summary.get("actual_summary_model_cost_usd"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "net_savings_vs_projection_delta_usd": summary.get("net_savings_vs_projection_delta_usd"),
            "latency": actual.get("latency") if isinstance(actual.get("latency"), dict) else None,
            "status_buckets": actual.get("status_buckets"),
            "summary_decision_status_buckets": actual.get("summary_decision_status_buckets"),
            "summary_reason_buckets": actual.get("summary_reason_buckets"),
            "summary_cache_buckets": actual.get("summary_cache_buckets"),
            "safety_stop_buckets": actual.get("safety_stop_buckets"),
            "delta": delta,
            "quality_gate": {
                "schema": quality_gate.get("schema"),
                "verdict": quality_gate.get("verdict"),
                "reason_codes": quality_gate.get("reason_codes"),
                "warning_codes": quality_gate.get("warning_codes"),
                "metrics": quality_gate.get("metrics"),
                "thresholds": quality_gate.get("thresholds"),
            } if quality_gate else None,
            "error_type": (dry_run.get("error") or {}).get("type") if isinstance(dry_run.get("error"), dict) else None,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_old_turns_included": False,
                "raw_summaries_included": False,
                "provider_bodies_included": False,
                "raw_session_ids_included": False,
                "request_ids_included": False,
                "tenant_ids_included": False,
                "cache_keys_included": False,
                "file_paths_included": False,
                "db_path_included": False,
            },
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
        return {
            "event_type": _old_context_summary_lifecycle_event_type(command, dry_run),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "recommendation_id": f"old-context-summary:{digest}",
            "bundle_hash": None,
            "policy_sections": ["crunch"],
            "validation_warning_count": 0,
            "review_warning_count": 0,
            "applied_files": [],
            "local_tool_version": __version__,
            "metadata": metadata,
        }

    policy = dry_run.get("policy") if isinstance(dry_run.get("policy"), dict) else {}
    summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), dict) else {}
    groups = [item for item in dry_run.get("groups", []) if isinstance(item, dict)]
    eligible_groups = [item for item in groups if item.get("blocker") == "eligible"]
    group_counts: dict[str, int] = {}
    for group in groups:
        blocker = str(group.get("blocker") or "unknown")
        group_counts[blocker] = group_counts.get(blocker, 0) + int(group.get("call_count") or 0)
    basis = {
        "command": command,
        "rule_id": policy.get("rule_id"),
        "candidate_id": policy.get("candidate_id"),
        "eligible_call_count": summary.get("eligible_call_count"),
        "projected_saved_tokens": summary.get("projected_saved_tokens"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    event_type = _old_context_summary_lifecycle_event_type(command, dry_run)
    metadata = {
        "schema": "agentflow.old_context_summary_lifecycle_metadata.v1",
        "lifecycle_kind": "old_context_summarization",
        "command": "policy-review" if command == "review" else "old-context-summary-dry-run",
        "local_result_status": "ok" if dry_run.get("ok") else "error",
        "dry_run": True,
        "read_only": bool(dry_run.get("read_only", True)),
        "policy_source": policy.get("policy_source"),
        "rule_id": policy.get("rule_id"),
        "candidate_id": policy.get("candidate_id"),
        "model": policy.get("model"),
        "placement": policy.get("placement"),
        "canary_enabled": ((policy.get("canary") or {}).get("enabled") if isinstance(policy.get("canary"), dict) else None),
        "canary_fraction": ((policy.get("canary") or {}).get("fraction") if isinstance(policy.get("canary"), dict) else None),
        "safety_stop_enabled": ((policy.get("safety_stop") or {}).get("enabled") if isinstance(policy.get("safety_stop"), dict) else None),
        "sampled_call_count": summary.get("sampled_call_count"),
        "sampled_provider_call_count": summary.get("sampled_provider_call_count"),
        "request_body_available_count": summary.get("request_body_available_count"),
        "request_body_replayed_count": summary.get("request_body_replayed_count"),
        "eligible_call_count": summary.get("eligible_call_count"),
        "summary_cache_hit_count": summary.get("summary_cache_hit_count"),
        "eligible_old_turns": summary.get("eligible_old_turns"),
        "eligible_chars": summary.get("eligible_chars"),
        "projected_saved_chars": summary.get("projected_saved_chars"),
        "projected_saved_tokens": summary.get("projected_saved_tokens"),
        "estimated_summary_cost_usd": summary.get("estimated_summary_cost_usd"),
        "projected_gross_savings_usd": summary.get("projected_gross_savings_usd"),
        "projected_net_savings_usd": summary.get("projected_net_savings_usd"),
        "eligible_group_count": len(eligible_groups),
        "blocker_counts": dict(sorted(group_counts.items())),
        "reload_required": dry_run.get("reload_required"),
        "error_type": (dry_run.get("error") or {}).get("type") if isinstance(dry_run.get("error"), dict) else None,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_old_turns_included": False,
            "raw_summaries_included": False,
            "provider_bodies_included": False,
            "raw_session_ids_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "db_path_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": f"old-context-summary:{digest}",
        "bundle_hash": None,
        "policy_sections": ["crunch"],
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _attach_old_context_summary_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _old_context_summary_lifecycle_payload(command, result)
    if payload is None:
        return
    if not recommendations.recommendations_enabled():
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            **recommendations.disabled_outcome_feedback_meta(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "disabled",
        })
        return

    store = None
    try:
        store = _open_store_for_db(str(db_path))
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=recommendations.OLD_CONTEXT_SUMMARY_LIFECYCLE_SOURCE_SURFACE,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": True,
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        if store is not None:
            store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if command in {"dry-run", "impact"} and public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def _write_policy_rollback_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def proxy_main() -> None:
    # The provider proxy forwards real API credentials and request bodies upstream.
    # Keep installed CLI defaults localhost-only unless the user explicitly opts in
    # to a different bind address through AGENTFLOW_HOST or --host.
    os.environ.setdefault("AGENTFLOW_HOST", "127.0.0.1")

    from agentflow_proxy.server import main

    main()


def policy_reload_main() -> None:
    raise SystemExit(policy_reload_cli())


def policy_export_main() -> None:
    raise SystemExit(policy_export_cli())


def policy_validate_main() -> None:
    raise SystemExit(policy_validate_cli())


def policy_diff_main() -> None:
    raise SystemExit(policy_diff_cli())


def policy_review_main() -> None:
    raise SystemExit(policy_review_cli())


def policy_fetch_review_main() -> None:
    raise SystemExit(policy_fetch_review_cli())


def policy_apply_main() -> None:
    raise SystemExit(policy_apply_cli())


def managed_rollout_actions_review_main() -> None:
    raise SystemExit(managed_rollout_actions_review_cli())


def managed_rollout_actions_apply_main() -> None:
    raise SystemExit(managed_rollout_actions_apply_cli())


def managed_rollout_actions_dry_run_main() -> None:
    raise SystemExit(managed_rollout_actions_dry_run_cli())


def old_context_summary_dry_run_main() -> None:
    raise SystemExit(old_context_summary_dry_run_cli())


def old_context_summary_impact_main() -> None:
    raise SystemExit(old_context_summary_impact_cli())


def managed_rollout_actions_impact_main() -> None:
    raise SystemExit(managed_rollout_actions_impact_cli())


def policy_rollback_main() -> None:
    raise SystemExit(policy_rollback_cli())


def codex_diagnose_main() -> None:
    raise SystemExit(codex_diagnose_cli())


def managed_pattern_rollups_main() -> None:
    raise SystemExit(managed_pattern_rollups_cli())


def managed_feedback_status_main() -> None:
    raise SystemExit(managed_feedback_status_cli())


def managed_feedback_flush_main() -> None:
    raise SystemExit(managed_feedback_flush_cli())
