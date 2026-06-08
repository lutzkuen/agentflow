from __future__ import annotations

import argparse
import asyncio
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
            "impact_status": (result.get("impact_summary") or {}).get("status"),
            "proposed_error_count": len((result.get("proposed_validation") or {}).get("errors", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
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
    if not isinstance(bundle, dict):
        return {}
    recommendation = bundle.get("recommendation")
    if not isinstance(recommendation, dict):
        recommendation = {}
    policies = bundle.get("policies")
    routing = policies.get("routing") if isinstance(policies, dict) and isinstance(policies.get("routing"), dict) else {}
    codex_app = policies.get("codex_app") if isinstance(policies, dict) and isinstance(policies.get("codex_app"), dict) else {}
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
        "omitted_candidate_count": recommendation.get(
            "omitted_candidate_count",
            routing_recommendation.get("omitted_candidate_count", 0),
        ),
        "filters": recommendation.get("filters", {}),
        "candidates": candidates,
        "codex_app": codex_app_summary,
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

    validation = validate_policy_bundle(bundle)
    current = asyncio.run(build_policy_bundle())
    review = review_policy_bundle(current, bundle, impact_db_path=args.db, impact_limit=max(0, args.impact_limit))
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
        "recommendation": _managed_recommendation_summary(bundle),
        "bundle": bundle,
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
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_apply_result(stdout, result, pretty=args.pretty)
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
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_rollback_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


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
    from agentflow_proxy.store import Store

    old_database_url = os.environ.get("AGENTFLOW_DATABASE_URL")
    db_arg = str(args.db)
    try:
        if db_arg.startswith(("postgresql://", "postgres://")):
            os.environ["AGENTFLOW_DATABASE_URL"] = db_arg
            store = Store()
        else:
            os.environ.pop("AGENTFLOW_DATABASE_URL", None)
            store = Store(db_arg)
    finally:
        if old_database_url is None:
            os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        else:
            os.environ["AGENTFLOW_DATABASE_URL"] = old_database_url
    try:
        result = asyncio.run(stats_codex_effectiveness(store, limit=args.limit))
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


def policy_rollback_main() -> None:
    raise SystemExit(policy_rollback_cli())


def codex_diagnose_main() -> None:
    raise SystemExit(codex_diagnose_cli())
