from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import httpx

from tokenclaw.cli_common import (
    default_db_path,
    default_loopback_url,
    is_loopback_url,
    open_store_for_db as _open_store_for_db,
    write_json as _write_json,
)
from tokenclaw.env import env, env_float
from tokenclaw.upstream_url import redact_url as _redact_url


POLICY_RELOAD_PATH = "/tokenclaw/admin/reload-policies"


def _default_policy_reload_url() -> str:
    return default_loopback_url(POLICY_RELOAD_PATH)


def _is_loopback_url(url: str) -> bool:
    return is_loopback_url(url)


def policy_reload_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Reload local AgentFlow policy files through the loopback admin API")
    parser.add_argument(
        "--url",
        default=env("TOKENCLAW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=env_float("TOKENCLAW_ADMIN_TIMEOUT", 10.0),
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
        from tokenclaw.policy_events import log_policy_event

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
        from tokenclaw.policy_events import log_policy_event

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
        from tokenclaw.policy_events import log_policy_event

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
    from tokenclaw.policy_events import log_policy_event

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

    from tokenclaw.policy_bundle import build_policy_bundle
    from tokenclaw.policy_events import log_policy_event

    bundle = asyncio.run(build_policy_bundle())
    log_policy_event("export", ok=True, details={"source": "cli", "exit_code": 0, "policies": bundle.get("policies")})
    if args.pretty:
        stdout.write(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, bundle)
    return 0

def _validation_result_error(message: str, *, path: str = "$") -> dict[str, Any]:
    return {
        "schema": "tokenclaw.policy_bundle_validation.v1",
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
            from tokenclaw.policy_events import log_policy_event

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
        from tokenclaw.policy_events import log_policy_event

        log_policy_event(
            "validate",
            ok=False,
            details={"source": "cli", "path": args.path, "error_count": 1, "warning_count": 0, "exit_code": 1},
        )
        _write_validation_result(stdout, result, pretty=args.pretty)
        return 1

    from tokenclaw.policy_bundle import validate_policy_bundle
    from tokenclaw.policy_events import log_policy_event

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
        "schema": "tokenclaw.policy_bundle_diff.v1",
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
        from tokenclaw.policy_bundle import compare_policy_bundles

        result = compare_policy_bundles(before, after)

    from tokenclaw.policy_events import log_policy_event

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
        "schema": "tokenclaw.policy_bundle_review.v1",
        "ok": False,
        "changed": False,
        "changed_sections": [],
        "change_count": 0,
        "safety_warning_count": 0,
        "safety_warnings": [],
        "current_validation": None,
        "proposed_validation": proposed_validation,
        "diff": {
            "schema": "tokenclaw.policy_bundle_diff.v1",
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
        default=default_db_path(),
        help="SQLite metadata database path for local impact simulation, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
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
        from tokenclaw.policy_bundle import build_policy_bundle, review_policy_bundle

        current = asyncio.run(build_policy_bundle())
        result = review_policy_bundle(current, proposed, impact_db_path=args.db, impact_limit=max(0, args.impact_limit))

    from tokenclaw.policy_events import log_policy_event

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
    if command == "quality-gate":
        return "quality-gate"
    if command == "impact":
        return "impact"
    if command == "review":
        return "reviewed"
    return "dry-run"

def _old_context_summary_metadata_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    text_l = text.lower()
    unsafe_terms = {
        "account",
        "apikey",
        "api_key",
        "authorization",
        "body",
        "cache_key",
        "content",
        "file",
        "message",
        "path",
        "payload",
        "prompt",
        "request",
        "response",
        "secret",
        "session",
        "summary_text",
        "tenant",
        "tool",
        "transcript",
    }
    if (
        len(text) > 128
        or any(char.isspace() for char in text)
        or any(char in text for char in ("/", "\\", "{", "}", "[", "]", "\"", "'"))
        or any(term in text_l for term in unsafe_terms)
    ):
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text

def _old_context_summary_quality_gate_feedback(
    *,
    quality_gate: dict[str, Any],
    policy: dict[str, Any],
    summary: dict[str, Any],
    actual: dict[str, Any],
    delta: dict[str, Any],
    local_tool_version: str,
) -> dict[str, Any] | None:
    if not isinstance(quality_gate, dict) or not quality_gate:
        return None

    metrics = quality_gate.get("metrics") if isinstance(quality_gate.get("metrics"), dict) else {}
    cohorts = quality_gate.get("cohorts") if isinstance(quality_gate.get("cohorts"), dict) else {}
    applied = cohorts.get("canary_applied") if isinstance(cohorts.get("canary_applied"), dict) else {}
    holdout = cohorts.get("canary_holdout") if isinstance(cohorts.get("canary_holdout"), dict) else {}
    bypassed = cohorts.get("bypassed_or_disabled") if isinstance(cohorts.get("bypassed_or_disabled"), dict) else {}
    matched_count = int(metrics.get("matched_metadata_row_count") or summary.get("actual_matched_metadata_row_count") or 0)
    summary_failure_count = int(metrics.get("summary_failure_count") or summary.get("summary_failure_count") or 0)
    safety_stop_count = int(applied.get("safety_stop_count") or 0) + int(bypassed.get("safety_stop_count") or 0)
    latency = actual.get("latency") if isinstance(actual.get("latency"), dict) else {}
    privacy = quality_gate.get("privacy") if isinstance(quality_gate.get("privacy"), dict) else {}

    return {
        "schema": "tokenclaw.old_context_summary_quality_gate_feedback.v1",
        "quality_gate_schema": quality_gate.get("schema"),
        "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
        "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
        "policy_source": policy.get("policy_source"),
        "local_tool_version": local_tool_version,
        "verdict": quality_gate.get("verdict"),
        "reason_codes": quality_gate.get("reason_codes") or [],
        "warning_codes": quality_gate.get("warning_codes") or [],
        "thresholds": quality_gate.get("thresholds") if isinstance(quality_gate.get("thresholds"), dict) else {},
        "cohorts": {
            "canary_applied": applied,
            "canary_holdout": holdout,
            "bypassed_or_disabled": bypassed,
        },
        "cohort_counts": {
            "matched": matched_count,
            "canary_applied": int(metrics.get("canary_applied_count") or summary.get("actual_canary_applied_count") or 0),
            "canary_holdout": int(metrics.get("canary_holdout_count") or summary.get("actual_canary_holdout_count") or 0),
            "bypassed_or_disabled": int(metrics.get("bypassed_or_disabled_count") or summary.get("actual_bypassed_or_disabled_count") or 0),
        },
        "aggregate_rates": {
            "error_rate": summary.get("error_rate"),
            "retry_rate": summary.get("retry_rate"),
            "summary_failure_rate": round(summary_failure_count / matched_count, 6) if matched_count else 0.0,
        },
        "aggregate_deltas": {
            "applied_minus_holdout_error_rate": metrics.get("applied_minus_holdout_error_rate"),
            "applied_minus_holdout_retry_rate": metrics.get("applied_minus_holdout_retry_rate"),
            "applied_minus_holdout_latency_avg_ms": metrics.get("applied_minus_holdout_latency_avg_ms"),
            "latency_applied_minus_holdout_avg_ms": latency.get("applied_minus_holdout_avg_ms"),
            "matched_vs_projected_affected_delta": delta.get("matched_vs_projected_affected_delta"),
            "applied_vs_projected_delta": delta.get("applied_vs_projected_delta"),
            "holdout_vs_projected_delta": delta.get("holdout_vs_projected_delta"),
            "bypass_or_disabled_vs_projected_delta": delta.get("bypass_or_disabled_vs_projected_delta"),
            "net_savings_vs_projection_delta_usd": delta.get("net_savings_vs_projection_delta_usd") or summary.get("net_savings_vs_projection_delta_usd"),
        },
        "savings": {
            "net_savings_usd": metrics.get("net_savings_usd") or summary.get("actual_net_savings_usd"),
            "gross_savings_usd": metrics.get("gross_savings_usd") or summary.get("actual_gross_savings_usd"),
            "summary_model_cost_usd": metrics.get("summary_model_cost_usd") or summary.get("actual_summary_model_cost_usd"),
            "payback_ratio": metrics.get("payback_ratio"),
            "projection_realization_ratio": metrics.get("projection_realization_ratio"),
        },
        "safety": {
            "summary_failure_count": summary_failure_count,
            "summary_failure_rate": round(summary_failure_count / matched_count, 6) if matched_count else 0.0,
            "safety_stop_count": safety_stop_count,
            "applied_safety_stop_count": int(applied.get("safety_stop_count") or 0),
            "bypassed_safety_stop_count": int(bypassed.get("safety_stop_count") or 0),
        },
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": bool(privacy.get("raw_old_context_included", False)),
            "generated_summaries_included": bool(privacy.get("generated_summaries_included", False)),
            "summary_prompts_included": False,
            "raw_messages_included": bool(privacy.get("raw_messages_included", False)),
            "raw_transcripts_included": bool(privacy.get("raw_transcripts_included", False)),
            "provider_bodies_included": bool(privacy.get("provider_bodies_included", False)),
            "file_contents_included": False,
            "request_ids_included": bool(privacy.get("request_ids_included", False)),
            "tenant_ids_included": bool(privacy.get("tenant_ids_included", False)),
            "local_session_ids_included": bool(privacy.get("local_session_ids_included", False)),
            "cache_keys_included": bool(privacy.get("cache_keys_included", False)),
            "raw_payload_strings_included": False,
        },
    }

def _old_context_summary_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    from tokenclaw import __version__

    dry_run = _old_context_summary_lifecycle_result(command, result)
    if not isinstance(dry_run, dict):
        return None
    if command in {"impact", "quality-gate"}:
        dry_run_meta = dry_run.get("dry_run") if isinstance(dry_run.get("dry_run"), dict) else {}
        if command == "quality-gate":
            policy = {
                "rule_id": dry_run.get("rule_id"),
                "candidate_id": dry_run.get("candidate_id"),
                "policy_source": dry_run.get("policy_source"),
                "model": dry_run.get("model"),
                "canary": dry_run.get("canary") if isinstance(dry_run.get("canary"), dict) else {},
                "safety_gates": dry_run.get("safety_gates") if isinstance(dry_run.get("safety_gates"), dict) else {},
            }
            projection = dry_run.get("projection") if isinstance(dry_run.get("projection"), dict) else {}
        else:
            policy = dry_run_meta.get("policy") if isinstance(dry_run_meta.get("policy"), dict) else {}
            projection = dry_run_meta.get("projection") if isinstance(dry_run_meta.get("projection"), dict) else {}
        summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), dict) else {}
        actual = dry_run.get("actual") if isinstance(dry_run.get("actual"), dict) else {}
        delta = dry_run.get("delta") if isinstance(dry_run.get("delta"), dict) else {}
        quality_gate = dry_run.get("quality_gate") if isinstance(dry_run.get("quality_gate"), dict) else {}
        if command == "quality-gate" and not quality_gate:
            quality_gate = dry_run
        quality_gate_feedback = _old_context_summary_quality_gate_feedback(
            quality_gate=quality_gate,
            policy=policy,
            summary=summary,
            actual=actual,
            delta=delta,
            local_tool_version=__version__,
        )
        basis = {
            "command": command,
            "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
            "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
        }
        digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
        metadata = {
            "schema": "tokenclaw.old_context_summary_lifecycle_metadata.v1",
            "lifecycle_kind": "old_context_summarization",
            "command": "old-context-summary-quality-gate" if command == "quality-gate" else "old-context-summary-impact",
            "local_result_status": "ok" if dry_run.get("ok") else "error",
            "dry_run": False,
            "read_only": bool(dry_run.get("read_only", True)),
            "policy_source": policy.get("policy_source"),
            "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
            "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
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
            "old_context_summary_quality_gate": quality_gate_feedback,
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
        "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
        "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
        "eligible_call_count": summary.get("eligible_call_count"),
        "projected_saved_tokens": summary.get("projected_saved_tokens"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    event_type = _old_context_summary_lifecycle_event_type(command, dry_run)
    metadata = {
        "schema": "tokenclaw.old_context_summary_lifecycle_metadata.v1",
        "lifecycle_kind": "old_context_summarization",
        "command": "policy-review" if command == "review" else "old-context-summary-dry-run",
        "local_result_status": "ok" if dry_run.get("ok") else "error",
        "dry_run": True,
        "read_only": bool(dry_run.get("read_only", True)),
        "policy_source": policy.get("policy_source"),
        "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
        "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
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
    from tokenclaw import recommendations

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
