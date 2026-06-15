from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import httpx

from agentflow_proxy.cli_common import (
    open_store_for_db as _open_store_for_db,
    redact_secret as _redact_secret,
    write_json as _write_json,
)
from agentflow_proxy.upstream_url import redact_url as _redact_url

from agentflow_proxy.cli_commands.policy_bundle import (
    _attach_old_context_summary_lifecycle_feedback,
    _old_context_summary_lifecycle_payload,
    _read_policy_json_arg,
)
from agentflow_proxy.cli_commands.policy_workbench import (
    MANAGED_POLICY_API_KEY_ENV,
    POLICY_BUNDLE_RECOMMENDATION_URL_ENV,
    _managed_policy_auth,
    _managed_policy_query,
    _write_policy_draft_apply_result,
)

PATTERN_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_PATTERN_ROLLOUT_ACTIONS_URL"
OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_OPTIMIZATION_ROLLOUT_ACTIONS_URL"
SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_SCAFFOLD_ROLLOUT_ACTIONS_URL"
PROMOTION_BLOCKER_RECOMMENDATIONS_URL_ENV = "AGENTFLOW_PROMOTION_BLOCKER_RECOMMENDATIONS_URL"
POST_PROMOTION_PRIORITY_DELTAS_URL_ENV = "AGENTFLOW_POST_PROMOTION_PRIORITY_DELTAS_URL"


def openai_optimization_draft_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run a staged managed OpenAI optimization draft through the local governor"
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
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only OpenAI dry-run rows.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Recent local call metadata rows to inspect, default: 1000.")
    parser.add_argument(
        "--canary-fraction",
        type=float,
        default=1.0,
        help="Projected per-action and governor canary fraction for the dry-run, default: 1.0.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Projected per-action and governor holdout fraction for the dry-run, default: 0.0.",
    )
    parser.add_argument(
        "--queue-feedback",
        action="store_true",
        help="Queue sanitized dry-run lifecycle feedback in the local metadata queue.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_optimization_draft_dry_run import dry_run_openai_optimization_draft
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.store import Store

    store = None
    try:
        db_path = Path(args.db).expanduser()
        if db_path.exists():
            store = Store(str(db_path))
        result = asyncio.run(
            dry_run_openai_optimization_draft(
                args.draft,
                workspace=args.workspace,
                store_obj=store,
                limit=args.limit,
                canary_fraction=args.canary_fraction,
                holdout_fraction=args.holdout_fraction,
                queue_feedback=args.queue_feedback,
            )
        )
    finally:
        if store is not None:
            store.conn.close()

    log_policy_event(
        "openai-optimization-draft-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "draft": args.draft,
            "workspace": args.workspace,
            "db": args.db,
            "limit": args.limit,
            "queue_feedback": bool(args.queue_feedback),
            "openai_rows_considered": (result.get("summary") or {}).get("openai_rows_considered")
            if isinstance(result.get("summary"), dict)
            else None,
            "applied_if_enabled_total": (result.get("summary") or {}).get("applied_if_enabled_total")
            if isinstance(result.get("summary"), dict)
            else None,
            "suppressed_total": (result.get("summary") or {}).get("suppressed_total")
            if isinstance(result.get("summary"), dict)
            else None,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "active_policy_files_written": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    stdout.write(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    return 0 if result.get("ok") else 1


def openai_optimization_draft_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a staged OpenAI optimization draft as bounded local canaries, with backups and rollback metadata"
    )
    parser.add_argument("draft", help="Staged OpenAI optimization draft ID, draft directory, draft.json path, or policy_bundle.json path.")
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
        help="Local SQLite DB path for metadata-only dry-run projection and lifecycle feedback.",
    )
    parser.add_argument("--section", action="append", choices=["routing", "crunch", "cache"], help="Apply only one policy section. Repeat to apply multiple sections.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview canary policy edits without writing local YAML files.")
    parser.add_argument("--write", dest="dry_run", action="store_false", help="Write canary edits to local YAML policy files.")
    parser.add_argument("--canary-fraction", type=float, default=0.10, help="Deterministic applied canary fraction, default: 0.10.")
    parser.add_argument("--holdout-fraction", type=float, default=0.10, help="Deterministic holdout fraction, default: 0.10.")
    parser.add_argument("--impact-limit", type=int, default=1000, help="Recent OpenAI metadata rows to inspect for governor conflicts, default: 1000.")
    parser.add_argument(
        "--require-verified-provenance",
        action="store_true",
        help="Require staged draft provenance to verify with the configured managed policy secret before applying.",
    )
    parser.add_argument("--queue-feedback", action="store_true", help="Queue sanitized apply lifecycle feedback in the local metadata queue.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from agentflow_proxy.openai_optimization_draft_apply import apply_openai_optimization_draft
    from agentflow_proxy.store import Store

    store = None
    try:
        db_path = Path(args.db).expanduser()
        if db_path.exists():
            store = Store(str(db_path))
        result = asyncio.run(
            apply_openai_optimization_draft(
                args.draft,
                workspace=args.workspace,
                config_dir=args.config_dir,
                store_obj=store,
                dry_run=args.dry_run,
                canary_fraction=args.canary_fraction,
                holdout_fraction=args.holdout_fraction,
                impact_limit=args.impact_limit,
                sections=args.section,
                require_verified_provenance=args.require_verified_provenance,
                queue_feedback=args.queue_feedback,
            )
        )
    finally:
        if store is not None:
            store.conn.close()

    _write_policy_draft_apply_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


async def _queue_codex_app_dry_run_lifecycle_events(store: Any, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy.recommendations import queue_policy_event_feedback

    queued: dict[str, Any] = {
        "schema": "agentflow.codex_app_canary_lifecycle_queue_meta.v1",
        "source_surface": "codex_app_canary_lifecycle",
        "event_phase": "dry_run",
        "results": {},
        "payload_included": False,
    }
    candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        rule_id = str(candidate.get("rule_id") or candidate.get("candidate_id") or "unknown-codex-app-rule")
        candidate_id = str(candidate.get("candidate_id") or rule_id)
        event = {
            "schema": "agentflow.codex_app_canary_lifecycle_feedback.v1",
            "event_type": "codex_app_canary_lifecycle",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "source_surface": "codex_turn",
            "app_family": "codex",
            "lifecycle_kind": "codex_app_canary",
            "lifecycle_phase": "dry_run",
            "action_family": "codex_app",
            "policy_id": candidate.get("policy_id") or candidate_id,
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "canary_cohort": "projected",
            "decision": {
                "status": "dry_run",
                "reason": "codex-app-policy-dry-run",
                "policy_source": candidate.get("policy_source") or "managed-recommended",
                "applied": False,
                "rule_path_included": False,
                "cache_key_included": False,
            },
            "outcome": {
                "status": "dry_run",
                "status_class": "metadata",
                "projected_applied_count": candidate.get("projected_applied_count"),
                "projected_holdout_count": candidate.get("projected_holdout_count"),
                "projected_skip_count": candidate.get("projected_skip_count"),
                "projected_savings_usd": candidate.get("projected_savings_usd"),
            },
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_params_included": False,
                "raw_transcripts_included": False,
                "raw_commands_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "thread_ids_included": False,
                "local_session_ids_included": False,
                "cache_keys_included": False,
                "rule_path_included": False,
            },
        }
        meta = await queue_policy_event_feedback(
            store,
            event,
            source_surface="codex_app_canary_lifecycle",
            queue_when_disabled=True,
            flush_immediately=False,
        )
        queued["results"][candidate_id] = {
            "enabled": bool(meta.get("enabled")),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "endpoint": meta.get("endpoint"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "payload_included": False,
        }
    return queued


def codex_app_policy_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run managed Codex app policy rules against local metadata-only turn windows")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(POLICY_BUNDLE_RECOMMENDATION_URL_ENV),
        help=f"Managed policy bundle recommendation URL. May also be set with {POLICY_BUNDLE_RECOMMENDATION_URL_ENV}.",
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
    parser.add_argument("--limit", type=int, default=50, help="Maximum candidates to request when fetching.")
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
        help="SQLite metadata database path for recent Codex turn windows, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument("--recent-limit", type=int, default=200, help="Maximum recent Codex turn/start rows to project, default: 200.")
    parser.add_argument("--fixture", help="Optional JSON fixture rows/features to include in the projection.")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Do not include the built-in synthetic summary-turn fixture row.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    managed_server_calls_made = False
    if args.url:
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        managed_server_calls_made = fetch.get("status") not in {"skipped", None} if isinstance(fetch, dict) else False
        if fetch_exit is not None:
            result = {
                "schema": "agentflow.codex_app_policy_dry_run.v1",
                "ok": False,
                "dry_run": True,
                "applied": False,
                "wrote_local_policy_files": False,
                "cache_table_mutated": False,
                "provider_calls_made": False,
                "managed_server_calls_made": managed_server_calls_made,
                "fetch": fetch,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        fetch = {"status": "skipped", "reason": "local-input"}
        if read_error:
            result = {
                "schema": "agentflow.codex_app_policy_dry_run.v1",
                "ok": False,
                "dry_run": True,
                "applied": False,
                "wrote_local_policy_files": False,
                "cache_table_mutated": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "fetch": fetch,
                "validation": read_error,
                "error": {"type": "read_failed", "message": "policy bundle could not be read"},
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return 1

    from agentflow_proxy.codex_app_dry_run import dry_run_codex_app_policy, load_codex_app_fixture_features
    from agentflow_proxy.policy_bundle import validate_policy_bundle
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.store import Store

    validation = validate_policy_bundle(bundle)
    if not validation["ok"]:
        result = {
            "schema": "agentflow.codex_app_policy_dry_run.v1",
            "ok": False,
            "dry_run": True,
            "applied": False,
            "wrote_local_policy_files": False,
            "cache_table_mutated": False,
            "provider_calls_made": False,
            "managed_server_calls_made": managed_server_calls_made,
            "fetch": fetch,
            "validation": validation,
            "error": {"type": "validation_failed", "message": "policy bundle is invalid"},
        }
        log_policy_event(
            "codex-app-policy-dry-run",
            ok=False,
            details={"source": "cli", "validation_error_count": len(validation.get("errors", [])), "exit_code": 1},
        )
        _write_rollout_actions_result(stderr, result, pretty=args.pretty)
        return 1

    fixture_features = load_codex_app_fixture_features(args.fixture) if args.fixture else []
    store = Store(args.db)
    try:
        result = dry_run_codex_app_policy(
            bundle,
            store=store,
            recent_limit=max(0, args.recent_limit),
            fixture_features=fixture_features,
            include_synthetic=not args.no_synthetic,
        )
        result["managed_lifecycle_feedback"] = asyncio.run(
            _queue_codex_app_dry_run_lifecycle_events(store, result)
        )
    finally:
        store.conn.close()
    result["validation"] = validation
    result["fetch"] = fetch
    result["managed_server_calls_made"] = managed_server_calls_made
    result["privacy"]["managed_server_calls_made"] = managed_server_calls_made
    log_policy_event(
        "codex-app-policy-dry-run",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "candidate_count": result.get("summary", {}).get("candidate_count", 0),
            "projected_applied_count": result.get("summary", {}).get("projected_applied_count", 0),
            "projected_holdout_count": result.get("summary", {}).get("projected_holdout_count", 0),
            "projected_skip_count": result.get("summary", {}).get("projected_skip_count", 0),
            "projected_savings_usd": result.get("summary", {}).get("projected_savings_usd", 0.0),
            "provider_calls_made": False,
            "managed_server_calls_made": managed_server_calls_made,
            "exit_code": 0,
        },
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0


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

    if isinstance(bundle, dict) and bundle.get("schema") == "agentflow.openai_cache_replay_rollout_actions.v1":
        from agentflow_proxy.openai_cache_replay_rollout_actions import review_openai_cache_replay_rollout_actions

        review = review_openai_cache_replay_rollout_actions(bundle, config_dir=args.config_dir)
    else:
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
    _attach_terminal_output_compaction_lifecycle_feedback(
        review,
        command="review",
        db_path=str(args.db),
        result_key="managed_terminal_output_compaction_lifecycle_feedback",
    )
    _write_rollout_actions_result(stdout if review["ok"] else stderr, review, pretty=args.pretty)
    return 0 if review["ok"] else 1


def optimization_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed optimization rollout actions before local apply")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV),
        help=f"Managed optimization rollout actions URL. May also be set with {OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV}.",
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
                "schema": "agentflow.optimization_rollout_actions_fetch_review.v1",
                "ok": False,
                "fetch": fetch,
                "review": None,
                "wrote_local_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            log_policy_event(
                "optimization-rollout-actions-review",
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

    from agentflow_proxy.optimization_rollout_review import review_optimization_rollout_actions
    from agentflow_proxy.policy_events import log_policy_event

    review = review_optimization_rollout_actions(bundle)
    if fetch:
        review["fetch"] = fetch
    log_policy_event(
        "optimization-rollout-actions-review",
        ok=bool(review["ok"]),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "action_count": review.get("summary", {}).get("action_count", 0),
            "accepted_action_count": review.get("summary", {}).get("accepted_action_count", 0),
            "provenance_status": (review.get("provenance") or {}).get("status"),
            "error_count": len(review.get("errors", [])),
            "wrote_local_policy_files": False,
            "exit_code": 0 if review["ok"] else 1,
        },
    )
    _write_rollout_actions_result(stdout if review["ok"] else stderr, review, pretty=args.pretty)
    return 0 if review["ok"] else 1


def optimization_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply managed optimization rollout actions as local canary policy files")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help=f"Managed optimization rollout actions URL. May also be set with {OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV}.",
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
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache"],
        help="Policy section to apply. May be repeated. Default: routing, crunch, and cache.",
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without writing local YAML files.")
    parser.add_argument("--write", dest="dry_run", action="store_false", help="Write reviewed canary edits to local YAML policy files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    fetch_url = args.url or os.getenv(OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV)
    if fetch_url:
        args.url = fetch_url
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            result = {
                "schema": "agentflow.optimization_rollout_actions_apply.v1",
                "ok": False,
                "dry_run": bool(args.dry_run),
                "fetch": fetch,
                "actions": [],
                "files": [],
                "wrote_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.optimization_promotion_canary import apply_optimization_promotion_canaries

    result = apply_optimization_promotion_canaries(
        bundle,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
        sections=args.section,
    )
    result["schema"] = "agentflow.optimization_rollout_actions_apply.v1"
    result["source_command"] = "agentflow-optimization-rollout-actions-apply"
    if fetch:
        result["fetch"] = fetch
    _attach_optimization_promotion_lifecycle_feedback(
        result,
        command="dry-run" if args.dry_run else "apply",
        db_path=str(args.db),
    )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "optimization-rollout-actions-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": None if fetch_url else args.path,
            "url": _redact_url(fetch_url),
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "planned_action_count": (result.get("summary") or {}).get("planned_action_count"),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def scaffold_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed repeated-scaffold rollout actions before local canary apply")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV),
        help=f"Managed scaffold rollout actions URL. May also be set with {SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV}.",
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
                "schema": "agentflow.scaffold_rollout_actions_fetch_review.v1",
                "ok": False,
                "fetch": fetch,
                "review": None,
                "wrote_local_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            log_policy_event(
                "scaffold-rollout-actions-review",
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
    from agentflow_proxy.scaffold_rollout_review import review_scaffold_rollout_actions

    review = review_scaffold_rollout_actions(bundle)
    if fetch:
        review["fetch"] = fetch
        review["managed_server_calls_made"] = fetch.get("status") != "skipped"
    log_policy_event(
        "scaffold-rollout-actions-review",
        ok=bool(review.get("ok")),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "fetch_status": fetch.get("status") if isinstance(fetch, dict) else ("skipped" if not args.url else None),
            "action_count": review.get("action_count", 0),
            "accepted_action_count": review.get("accepted_action_count", 0),
            "provenance_status": (review.get("provenance") or {}).get("status"),
            "error_count": len(review.get("errors", [])),
            "wrote_local_policy_files": False,
            "exit_code": 0 if review.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if review.get("ok") else stderr, review, pretty=args.pretty)
    return 0 if review.get("ok") else 1


def scaffold_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply managed repeated-scaffold rollout actions as a local crunch canary overlay")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help=f"Managed scaffold rollout actions URL. May also be set with {SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV}.",
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
        default=os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local AgentFlow policy overlays, default: AGENTFLOW_CONFIG_DIR, AGENTFLOW_POLICY_CONFIG_DIR, or ~/.agentflow",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview the scaffold overlay without writing YAML.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    fetch_url = args.url or os.getenv(SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV)
    if fetch_url:
        args.url = fetch_url
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            result = {
                "schema": "agentflow.scaffold_rollout_actions_apply.v1",
                "ok": False,
                "dry_run": bool(args.dry_run),
                "fetch": fetch,
                "actions": [],
                "files": [],
                "wrote_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.scaffold_rollout_review import apply_scaffold_rollout_actions

    result = apply_scaffold_rollout_actions(bundle, config_dir=args.config_dir, dry_run=args.dry_run)
    if fetch:
        result["fetch"] = fetch
        result["managed_server_calls_made"] = fetch.get("status") != "skipped"
    log_policy_event(
        "scaffold-rollout-actions-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": None if fetch_url else args.path,
            "url": _redact_url(fetch_url),
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "fetch_status": fetch.get("status") if isinstance(fetch, dict) else ("skipped" if not fetch_url else None),
            "action_count": result.get("action_count", 0),
            "accepted_action_count": result.get("accepted_action_count", 0),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


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
        if isinstance(bundle, dict) and bundle.get("schema") == "agentflow.openai_cache_replay_rollout_actions.v1":
            from agentflow_proxy.openai_cache_replay_rollout_actions import apply_openai_cache_replay_rollout_actions

            result = apply_openai_cache_replay_rollout_actions(
                bundle,
                config_dir=args.config_dir,
                dry_run=args.dry_run,
            )
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
    _attach_terminal_output_compaction_lifecycle_feedback(
        result,
        command="apply",
        db_path=str(args.db),
        result_key="managed_terminal_output_compaction_lifecycle_feedback",
    )
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
        if isinstance(bundle, dict) and bundle.get("schema") == "agentflow.openai_cache_replay_rollout_actions.v1":
            from agentflow_proxy.openai_cache_replay_rollout_actions import dry_run_openai_cache_replay_rollout_actions

            result = dry_run_openai_cache_replay_rollout_actions(bundle, config_dir=args.config_dir)
            result["db_path"] = args.db
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
    parser.add_argument(
        "--profile",
        choices=("current-policy", "tool-protocol-aware"),
        default="current-policy",
        help=(
            "Dry-run profile to simulate. tool-protocol-aware enables a read-only overlay that "
            "summarizes only old non-tool text turns while preserving tool protocol messages."
        ),
    )
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
            profile=args.profile,
        )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.path or "current-policy",
            "profile": args.profile,
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


def old_context_summary_quality_gate_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local old-context summary canary quality gate")
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
    parser.add_argument("--pretty", action="store_true", help="Pretty-print quality gate JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        from agentflow_proxy.old_context_summary_impact import OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA

        result = {
            "schema": OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA,
            "ok": False,
            "read_only": True,
            "wrote_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "verdict": "hold",
            "reason_codes": ["input-read-failed"],
            "warning_codes": [],
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary dry-run or review report could not be read"},
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
                "file_paths_included": False,
            },
        }
    else:
        from agentflow_proxy.old_context_summary_impact import build_old_context_summary_quality_gate

        store = _open_store_for_db(args.db)
        try:
            result = build_old_context_summary_quality_gate(
                report,
                store_obj=store,
                limit=args.limit,
                since=args.since,
            )
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "old-context-summary-quality-gate",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "input_source": "stdin" if args.path == "-" else "file",
            "db_configured": bool(args.db),
            "since": args.since,
            "candidate_id": result.get("candidate_id"),
            "rule_id": result.get("rule_id"),
            "verdict": result.get("verdict"),
            "reason_codes": result.get("reason_codes"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else result.get("source_error_type"),
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if result.get("ok"):
        _attach_old_context_summary_lifecycle_feedback(result, command="quality-gate", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def _summary_rollout_event_details(command: str, result: dict[str, Any], *, path: str | None, config_dir: str | None = None, db_path: str | None = None) -> dict[str, Any]:
    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    files = [item for item in result.get("files", []) if isinstance(item, dict)]

    snapshots = []
    for action in actions:
        edit = action.get("proposed_edit") if isinstance(action.get("proposed_edit"), dict) else {}
        gate = action.get("quality_gate") if isinstance(action.get("quality_gate"), dict) else {}
        snapshots.append({
            key: value
            for key, value in {
                "action_id": action.get("action_id"),
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
                "action_type": action.get("action_type"),
                "current_fraction": edit.get("current_fraction") if edit else action.get("current_fraction"),
                "recommended_fraction": edit.get("recommended_fraction") if edit else action.get("projected_fraction"),
                "quality_gate_verdict": gate.get("verdict"),
                "quality_gate_reason_codes": gate.get("reason_codes"),
                "confidence": action.get("confidence"),
                "required_local_review": True,
                "managed_enforced": False,
            }.items()
            if value not in (None, "", [], {})
        })

    return {
        "source": "cli",
        "command": f"old-context-summary-rollout-actions-{command}",
        "input_source": "stdin" if path == "-" else "file",
        "config_dir_configured": bool(config_dir),
        "db_configured": bool(db_path),
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only")),
        "action_count": len(actions) or result.get("action_count") or validation.get("action_count") or 0,
        "planned_action_count": result.get("planned_action_count") or review.get("planned_action_count") or 0,
        "changed_action_count": result.get("changed_action_count") or review.get("changed_action_count") or 0,
        "rejected_action_count": result.get("rejected_action_count") or review.get("rejected_action_count") or 0,
        "action_snapshots": snapshots,
        "affected_metadata_row_count": summary.get("affected_metadata_row_count") or summary.get("actual_matched_metadata_row_count"),
        "projected_additional_applied_count": summary.get("projected_additional_applied_count"),
        "projected_local_bypass_or_disable_count": summary.get("projected_local_bypass_or_disable_count"),
        "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
        "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
        "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
        "changed_file_count": sum(1 for item in files if item.get("changed")),
        "changed_sections": sorted({str(item.get("section")) for item in files if item.get("changed") and item.get("section")}),
        "provenance_status": provenance.get("status"),
        "computed_bundle_hash": provenance.get("computed_bundle_hash"),
        "validation_error_count": len(validation.get("errors", []) if isinstance(validation.get("errors"), list) else []),
        "validation_warning_count": len(validation.get("warnings", []) if isinstance(validation.get("warnings"), list) else []),
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "exit_code": 0 if result.get("ok") else 1,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "yaml_contents_included": False,
        },
    }


def old_context_summary_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed old-context summary rollout actions against local crunch rules")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action bundle JSON path, or '-' for stdin.")
    parser.add_argument("--config-dir", default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_review.v1",
            "ok": False,
            "config_dir": args.config_dir,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action bundle could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import plan_summary_rollout_actions

        result = plan_summary_rollout_actions(bundle, config_dir=args.config_dir)
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-review",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("review", result, path=args.path, config_dir=args.config_dir),
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def old_context_summary_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply managed old-context summary rollout actions to local crunch rules")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action bundle JSON path, or '-' for stdin.")
    parser.add_argument("--config-dir", default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_apply.v1",
            "ok": False,
            "dry_run": bool(args.dry_run),
            "config_dir": args.config_dir,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action bundle could not be read"},
            "files": [],
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import apply_summary_rollout_actions

        result = apply_summary_rollout_actions(bundle, config_dir=args.config_dir, dry_run=args.dry_run)
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-apply",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("apply", result, path=args.path, config_dir=args.config_dir),
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def old_context_summary_rollout_actions_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run managed old-context summary rollout actions against recent local metadata")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action bundle JSON path, or '-' for stdin.")
    parser.add_argument("--config-dir", default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")))
    parser.add_argument("--db", default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_dry_run.v1",
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "config_dir": args.config_dir,
            "db_path": args.db,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action bundle could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import dry_run_summary_rollout_actions

        store = _open_store_for_db(args.db)
        try:
            result = dry_run_summary_rollout_actions(
                bundle,
                store_obj=store,
                config_dir=args.config_dir,
                limit=args.limit,
            )
            result["db_path"] = args.db
        finally:
            store.conn.close()
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-dry-run",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("dry-run", result, path=args.path, config_dir=args.config_dir, db_path=args.db),
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def old_context_summary_rollout_actions_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Measure post-apply old-context summary rollout-action impact against a dry-run projection")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action dry-run JSON path, or '-' for stdin.")
    parser.add_argument("--db", default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--since")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_impact.v1",
            "ok": False,
            "read_only": True,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action dry-run report could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import measure_summary_rollout_action_impact

        store = _open_store_for_db(args.db)
        try:
            result = measure_summary_rollout_action_impact(report, store_obj=store, limit=args.limit, since=args.since)
        finally:
            store.conn.close()
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-impact",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("impact", result, path=args.path, db_path=args.db),
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
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


def _phase_routing_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if command != "dry-run" or not isinstance(result, dict) or result.get("schema") != "agentflow.phase_routing_dry_run.v1":
        return None
    from agentflow_proxy import __version__

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    rules = [item for item in result.get("rules", []) if isinstance(item, dict)]
    candidate_ids = [
        str(item.get("candidate_id") or item.get("rule_id"))
        for item in rules
        if item.get("candidate_id") or item.get("rule_id")
    ]
    basis = {
        "command": command,
        "candidate_ids": candidate_ids,
        "matched_count": summary.get("matched_count"),
        "projected_candidate_count": summary.get("projected_candidate_count"),
        "projected_savings_usd": summary.get("projected_savings_usd"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    excluded: dict[str, int] = {}
    for item in summary.get("excluded_count_by_reason") or []:
        if isinstance(item, dict):
            reason = str(item.get("reason") or "unknown")
            excluded[reason] = excluded.get(reason, 0) + int(item.get("count") or 0)
    metadata = {
        "schema": "agentflow.phase_routing_lifecycle_metadata.v1",
        "lifecycle_kind": "phase_routing",
        "command": "phase-routing-dry-run",
        "local_result_status": "ok",
        "dry_run": True,
        "read_only": True,
        "wrote_local_files": bool(result.get("wrote_local_files")),
        "altered_provider_routing": bool(result.get("altered_provider_routing")),
        "policy_source": result.get("policy_source"),
        "sampled_call_count": result.get("sampled_call_count"),
        "rule_count": result.get("rule_count"),
        "matched_count": summary.get("matched_count"),
        "projected_candidate_count": summary.get("projected_candidate_count"),
        "excluded_count": summary.get("excluded_count"),
        "projected_savings_usd": summary.get("projected_savings_usd"),
        "projected_target_cost_usd": summary.get("projected_target_cost_usd"),
        "risk_warning_count": summary.get("risk_warning_count"),
        "candidate_rule_ids": candidate_ids,
        "excluded_count_by_reason": excluded,
        "settings": result.get("settings") if isinstance(result.get("settings"), dict) else {},
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "db_path_included": False,
            "policy_file_contents_included": False,
            "secrets_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": "dry-run" if result.get("ok", True) else "rejected",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": f"phase-routing:{digest}",
        "bundle_hash": None,
        "policy_sections": ["routing"],
        "validation_warning_count": 0,
        "review_warning_count": int(summary.get("risk_warning_count") or 0),
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _attach_phase_routing_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _phase_routing_lifecycle_payload(command, result)
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
                source_surface=recommendations.PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE,
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
    if public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


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


def codex_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report managed Codex app canary impact and lifecycle evidence by rule")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent turn/start rows to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_codex_canary_impact

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_codex_canary_impact(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_scoreboard_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.optimization.scoreboard import openai_scoreboard_cli as _openai_scoreboard_cli

    return _openai_scoreboard_cli(argv, stdout=stdout)


def openai_routing_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OpenAI local routing opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_routing_report import build_openai_routing_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_routing_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_routing_canary_stage_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Stage default-off OpenAI large-to-mini routing canary drafts from local metadata")
    parser.add_argument(
        "routing_report",
        nargs="?",
        help="OpenAI routing report JSON path, or '-' for stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--promotion-blocker-review",
        help="Promotion blocker review/recommendation JSON path, or '-' for stdin. When set, stage OpenAI canary drafts from reviewed canary lifecycle recommendations.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Recent OpenAI provider calls to inspect when building a fresh report, default: 1000.")
    parser.add_argument("--draft-id", help="Optional local draft ID. When multiple candidates are staged, a numeric suffix is added.")
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument("--canary-fraction", type=float, default=0.05, help="Proposed deterministic canary fraction, default: 0.05.")
    parser.add_argument("--holdout-fraction", type=float, default=0.10, help="Proposed deterministic holdout fraction, default: 0.10.")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum candidate samples before staging, default: 5.")
    parser.add_argument("--top-candidates", type=int, default=1, help="Maximum ranked eligible candidates to stage, default: 1.")
    parser.add_argument("--queue-feedback", action="store_true", help="Queue sanitized activation lifecycle feedback in the local metadata queue.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.routing_report and args.promotion_blocker_review:
        _write_json(
            stdout,
            {
                "ok": False,
                "schema": "agentflow.openai_routing_canary_stage_error.v1",
                "error": {
                    "type": "conflicting_inputs",
                    "message": "routing_report and --promotion-blocker-review are mutually exclusive",
                },
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_active_policy_files": False,
            },
        )
        return 2

    if args.promotion_blocker_review:
        from agentflow_proxy.promotion_blocker_review import build_promotion_blocker_recommendation_review

        report = _read_json_input(str(args.promotion_blocker_review), stdin=stdin)
        if report.get("schema") != "agentflow.promotion_blocker_recommendation_review.v1":
            report = build_promotion_blocker_recommendation_review(report, limit=int(args.limit or 25))
    elif args.routing_report:
        report = _read_json_input(str(args.routing_report), stdin=stdin)
    else:
        from agentflow_proxy.openai_routing_report import build_openai_routing_report
        from agentflow_proxy.orchestrator_research import _pass_through_routing_report

        store = _open_store_for_db(str(args.db))
        try:
            rows = [
                dict(row)
                for row in store.conn.execute(
                    """
                    select coalesce(provider, 'anthropic') as provider,
                           coalesce(source_surface, 'unknown') as source_surface,
                           coalesce(endpoint, 'unknown') as endpoint,
                           requested_model,
                           routed_model,
                           coalesce(category, json_extract(routing_json, '$.category'), 'unknown') as category,
                           count(*) as c,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') = 'canary_applied' then 1 else 0 end) as openai_canary_applied_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') = 'canary_holdout' then 1 else 0 end) as openai_canary_holdout_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') = 'safety_stopped'
                                     or json_extract(routing_json, '$.openai_canary.status') = 'safety_stopped'
                                    then 1 else 0 end) as openai_canary_safety_stopped_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') = 'skipped' then 1 else 0 end) as openai_canary_skipped_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') = 'bypassed_or_disabled' then 1 else 0 end) as openai_canary_bypassed_or_disabled_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') is not null
                                      and json_extract(routing_json, '$.openai_canary.cohort') not in (
                                          'canary_applied',
                                          'canary_holdout',
                                          'safety_stopped',
                                          'skipped',
                                          'bypassed_or_disabled'
                                      )
                                    then 1 else 0 end) as openai_canary_unknown_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') is not null and status_code >= 400 then 1 else 0 end) as openai_canary_error_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.cohort') is not null and coalesce(retry_count, 0) > 0 then 1 else 0 end) as openai_canary_retry_count,
                           sum(case when json_extract(routing_json, '$.openai_canary.fallback_reason') is not null
                                      or json_extract(routing_json, '$.fallback_reason') is not null
                                    then 1 else 0 end) as openai_canary_fallback_count,
                           max(case when json_extract(routing_json, '$.openai_canary.cohort') is not null then created_at else null end) as openai_canary_latest_observed_at
                    from (
                        select provider, source_surface, endpoint, requested_model, routed_model, category,
                               created_at, status_code, retry_count, routing_json
                        from calls
                        order by created_at desc
                        limit ?
                    )
                    group by coalesce(provider, 'anthropic'), coalesce(source_surface, 'unknown'),
                             coalesce(endpoint, 'unknown'), requested_model, routed_model,
                             coalesce(category, json_extract(routing_json, '$.category'), 'unknown')
                    order by c desc
                    limit 50
                    """,
                    (max(1, min(int(args.limit or 1000), 10_000)),),
                ).fetchall()
            ]
            report = _pass_through_routing_report(rows, limit=20) or build_openai_routing_report(store, limit=args.limit)
        finally:
            store.conn.close()

    from agentflow_proxy.openai_routing_canary_stage import stage_openai_routing_canary_drafts
    from agentflow_proxy.policy_events import log_policy_event

    result = asyncio.run(stage_openai_routing_canary_drafts(
        report,
        draft_id=args.draft_id,
        workspace=args.workspace,
        canary_fraction=args.canary_fraction,
        holdout_fraction=args.holdout_fraction,
        min_samples=args.min_samples,
        top_candidates=args.top_candidates,
    ))
    if args.queue_feedback:
        from agentflow_proxy.activation_lifecycle_feedback import queue_activation_staged_lifecycle_feedback

        store = _open_store_for_db(str(args.db))
        try:
            result["lifecycle_feedback"] = asyncio.run(
                queue_activation_staged_lifecycle_feedback(
                    store,
                    result,
                    event_phase="stage",
                    command="openai-routing-canary-stage",
                )
            )
        finally:
            store.conn.close()
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "openai-routing-canary-stage",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.promotion_blocker_review or args.routing_report,
            "workspace": args.workspace,
            "candidate_count": summary.get("candidate_count", 0),
            "eligible_candidate_count": summary.get("eligible_candidate_count", 0),
            "staged_count": summary.get("staged_count", 0),
            "omitted_count": summary.get("omitted_count", 0),
            "queue_feedback": bool(args.queue_feedback),
            "feedback_status": (result.get("lifecycle_feedback") or {}).get("status") if isinstance(result.get("lifecycle_feedback"), dict) else None,
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def openai_old_context_summary_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OpenAI old-context summary opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_old_context_summary_report import build_openai_old_context_summary_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_old_context_summary_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_cache_replay_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OpenAI cache replay opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_report import build_openai_cache_replay_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_cache_replay_blocker_outcomes_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export aggregate OpenAI cache replay blocker outcomes after local dependency checks")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--opportunity-limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for replay opportunity, default: 1000, max: 10000.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=500,
        help="Recent OpenAI calls to scan for replay impact evidence, default: 500, max: 10000.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_blocker_outcomes import build_openai_cache_replay_blocker_outcomes_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_blocker_outcomes_report(
            store,
            opportunity_limit=args.opportunity_limit,
            impact_limit=args.impact_limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def crunch_blocker_outcomes_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export aggregate crunch lifecycle outcomes from local opportunity and promotion blocker data")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--rollup-limit",
        type=int,
        default=1000,
        help="Recent calls to scan for crunch opportunity rollups, default: 1000, max: 10000.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.crunch_blocker_outcomes import build_crunch_blocker_outcomes_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_crunch_blocker_outcomes_report(
            store,
            rollup_limit=args.rollup_limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def activation_safety_stop_burndown_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export aggregate activation safety-stop burn-down groups from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--plan-json",
        help="Optional AgentFlow orchestrator research plan JSON to include repeated safety-stop diagnostics.",
    )
    parser.add_argument("--limit", type=int, default=10000, help="Queued lifecycle feedback rows to scan, default: 10000.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from agentflow_proxy.activation_lifecycle_feedback import activation_safety_stop_burndown_report

    research_plan = None
    if args.plan_json:
        try:
            with Path(args.plan_json).expanduser().open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError, TypeError) as exc:
            _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
            return 1
        if not isinstance(loaded, dict):
            _write_json(stderr, {"ok": False, "error": {"type": "invalid_plan_json", "message": "plan JSON must be an object"}})
            return 1
        research_plan = loaded

    store = _open_store_for_db(str(args.db))
    try:
        result = activation_safety_stop_burndown_report(
            store,
            research_plan=research_plan,
            limit=args.limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_action_ledger_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize cross-family optimization eligibility from local call metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.optimization_action_ledger import build_optimization_action_ledger_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_optimization_action_ledger_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def optimization_coordinator_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run cross-family optimization coordinator decisions against local metadata")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional managed rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument("--provider", help="Only inspect rows for this provider, for example openai or anthropic.")
    parser.add_argument("--source-surface", help="Only inspect rows for this source surface, for example openai_responses.")
    parser.add_argument("--canary-fraction", type=float, help="Coordinator canary fraction override for the dry run.")
    parser.add_argument("--holdout-fraction", type=float, help="Coordinator holdout fraction override for the dry run.")
    parser.add_argument(
        "--local-salt",
        help="Coordinator cohort salt override. The salt is used for hashing but is not emitted in the report.",
    )
    parser.add_argument("--examples", type=int, default=20, help="Maximum sanitized example decisions to include, default: 20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    rollout_actions = None
    if args.path is not None:
        rollout_actions, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            result = {
                "schema": "agentflow.optimization_coordinator_dry_run.v1",
                "ok": False,
                "dry_run": True,
                "read_only": True,
                "db_path": args.db,
                "validation": read_error,
                "error": {"type": "read_failed", "message": "managed rollout action bundle could not be read"},
                "privacy": {
                    "metadata_only": True,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "policy_files_changed": False,
                    "provider_body_changed": False,
                },
            }
            if args.pretty:
                stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            else:
                _write_json(stdout, result)
            return 1

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.optimization_coordinator_dry_run import build_optimization_coordinator_dry_run

    store = open_store_for_db(str(args.db))
    try:
        result = build_optimization_coordinator_dry_run(
            store,
            rollout_actions=rollout_actions,
            limit=args.limit,
            provider=args.provider,
            source_surface=args.source_surface,
            local_salt=args.local_salt,
            canary_fraction=args.canary_fraction,
            holdout_fraction=args.holdout_fraction,
            examples=args.examples,
        )
    finally:
        store.conn.close()
    result["db_path"] = args.db
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "optimization-coordinator-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "db_path": args.db,
            "path": args.path,
            "dry_run": True,
            "sampled_call_count": result.get("sampled_call_count"),
            "decision_count": result.get("decision_count"),
            "projected_savings_usd_est": result.get("projected_savings_usd_est"),
            "exit_code": 0,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def provider_tool_adoption_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.provider_adoption import provider_tool_adoption_report_cli as _provider_tool_adoption_report_cli

    return _provider_tool_adoption_report_cli(argv, stdout=stdout)


def repeated_scaffold_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure repeated provider-message scaffolding crunch opportunity")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-repeated-rows",
        type=int,
        default=2,
        help="Minimum rows per hidden scaffold fingerprint before a group is considered repeated, default: 2",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.repeated_scaffold_report import build_repeated_scaffold_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_repeated_scaffold_opportunity_report(
            store,
            limit=args.limit,
            min_repeated_rows=args.min_repeated_rows,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def instruction_dedup_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure instruction-section deduplication opportunity")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls and Codex app events to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-repeated-rows",
        type=int,
        default=2,
        help="Minimum rows per hidden instruction fingerprint before a group is considered repeated, default: 2",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.instruction_dedup_report import build_instruction_dedup_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_instruction_dedup_opportunity_report(
            store,
            limit=args.limit,
            min_repeated_rows=args.min_repeated_rows,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def instruction_dedup_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run instruction-section deduplication plans")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls and Codex app events to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=20,
        help="Maximum dry-run plan samples to emit, default: 20, max: 200",
    )
    parser.add_argument(
        "--local-salt",
        default=None,
        help="Optional local cohort salt for deterministic canary/holdout assignment; never emitted.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.instruction_dedup_dry_run import build_instruction_dedup_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_instruction_dedup_dry_run(
            store,
            limit=args.limit,
            examples=args.examples,
            local_salt=args.local_salt,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def instruction_dedup_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report instruction-section deduplication canary impact and lifecycle gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500, max: 10000")
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error rollback threshold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Applied-minus-holdout retry rollback threshold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument("--min-net-savings-usd", type=float, default=0.0, help="Minimum applied net savings for widen, default: 0.")
    parser.add_argument(
        "--max-negative-net-savings-rate",
        type=float,
        default=0.0,
        help="Applied negative net savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.instruction_dedup_impact import build_instruction_dedup_impact_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_instruction_dedup_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            min_net_savings_usd=args.min_net_savings_usd,
            max_negative_net_savings_rate=args.max_negative_net_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_instruction_dedup_lifecycle_feedback(result, store=store)
        from agentflow_proxy.instruction_dedup_feedback import SOURCE_SURFACE as INSTRUCTION_DEDUP_LIFECYCLE_SOURCE_SURFACE
        from agentflow_proxy.optimization.feedback import managed_feedback_status_result

        queue_status = managed_feedback_status_result(
            store,
            source_surface=INSTRUCTION_DEDUP_LIFECYCLE_SOURCE_SURFACE,
            sample_limit=5,
        )
        result["managed_lifecycle_feedback_queue"] = queue_status
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def terminal_output_compaction_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure terminal-output compaction opportunity for plateaued tool-result sessions")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum adjacent call text size for plateau detection, default: 8000",
    )
    parser.add_argument(
        "--max-plateau-delta-ratio",
        type=float,
        default=0.03,
        help="Maximum adjacent text size delta ratio for plateau detection, default: 0.03",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.terminal_compaction_report import build_terminal_output_compaction_opportunity_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_terminal_output_compaction_opportunity_report(
            store,
            limit=args.limit,
            min_text_chars=args.min_text_chars,
            max_plateau_delta_ratio=args.max_plateau_delta_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def codex_terminal_transcript_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Codex terminal-transcript compaction opportunity from local event windows")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent Codex turn/start rows to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-input-chars",
        type=int,
        default=8000,
        help="Minimum Codex turn input text size for candidate rows, default: 8000",
    )
    parser.add_argument(
        "--min-terminal-chars",
        type=int,
        default=2000,
        help="Minimum estimated terminal-transcript chars for candidate rows, default: 2000",
    )
    parser.add_argument(
        "--compaction-ratio",
        type=float,
        default=0.65,
        help="Projected removable fraction of terminal transcript chars after preserving diagnostics, default: 0.65",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.codex_terminal_compaction_report import build_codex_terminal_transcript_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_codex_terminal_transcript_opportunity_report(
            store,
            limit=args.limit,
            min_input_chars=args.min_input_chars,
            min_terminal_chars=args.min_terminal_chars,
            compaction_ratio=args.compaction_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def anthropic_thinking_compaction_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Anthropic thinking-session compaction opportunity from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum thinking-session input size for candidate rows, default: 8000",
    )
    parser.add_argument(
        "--max-plateau-delta-ratio",
        type=float,
        default=0.03,
        help="Maximum adjacent text size delta ratio for plateau detection, default: 0.03",
    )
    parser.add_argument(
        "--metadata-compaction-ratio",
        type=float,
        default=0.20,
        help="Projected removable fraction of input chars when raw bodies are unavailable, default: 0.20",
    )
    parser.add_argument(
        "--body-compaction-ratio",
        type=float,
        default=0.65,
        help="Projected removable fraction of detected thinking-history chars when local bodies are available, default: 0.65",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.anthropic_thinking_compaction_report import build_anthropic_thinking_compaction_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_anthropic_thinking_compaction_opportunity_report(
            store,
            limit=args.limit,
            min_text_chars=args.min_text_chars,
            max_plateau_delta_ratio=args.max_plateau_delta_ratio,
            metadata_compaction_ratio=args.metadata_compaction_ratio,
            body_compaction_ratio=args.body_compaction_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def anthropic_thinking_compaction_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Anthropic thinking-history compaction canary impact from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", default=None, help="Optional inclusive UTC timestamp lower bound.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied samples for budget feedback.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout samples for budget feedback.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.anthropic_thinking_compaction_impact import build_anthropic_thinking_compaction_impact_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_anthropic_thinking_compaction_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def anthropic_thinking_compaction_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Anthropic thinking-history compaction plans from local request metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=50,
        help="Maximum plans to include in output, default: 50, max: 500",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum request text size for candidate rows, default: 8000",
    )
    parser.add_argument(
        "--min-block-chars",
        type=int,
        default=2000,
        help="Minimum thinking block size for a compaction target, default: 2000",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Near-duplicate shingle similarity threshold, default: 0.95",
    )
    parser.add_argument(
        "--canary-fraction",
        type=float,
        default=1.0,
        help="Deterministic dry-run canary fraction, default: 1.0",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Deterministic dry-run holdout fraction, default: 0.0",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.anthropic_thinking_compaction_dry_run import build_anthropic_thinking_compaction_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_anthropic_thinking_compaction_dry_run(
            store,
            limit=args.limit,
            examples=args.examples,
            min_text_chars=args.min_text_chars,
            min_block_chars=args.min_block_chars,
            similarity_threshold=args.similarity_threshold,
            canary_fraction=args.canary_fraction,
            holdout_fraction=args.holdout_fraction,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def codex_terminal_transcript_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Codex terminal-transcript compaction plans from local event windows")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent Codex turn/start rows to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.codex_terminal_compaction_dry_run import build_codex_terminal_transcript_compaction_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_codex_terminal_transcript_compaction_dry_run(
            store,
            limit=args.limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def codex_terminal_transcript_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Codex terminal-transcript compaction canary impact and lifecycle gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent Codex turn/start rows to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect rows at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error rollback threshold, default: 0.05.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument("--min-net-savings-usd", type=float, default=0.0, help="Minimum applied net savings for promote, default: 0.")
    parser.add_argument(
        "--max-negative-savings-rate",
        type=float,
        default=0.0,
        help="Applied negative savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.codex_terminal_compaction_impact import build_codex_terminal_transcript_compaction_impact_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_codex_terminal_transcript_compaction_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            min_net_savings_usd=args.min_net_savings_usd,
            max_negative_savings_rate=args.max_negative_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_codex_terminal_transcript_lifecycle_feedback(result, store=store)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def terminal_output_compaction_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run terminal-output compaction plans for Anthropic tool-result history")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--keep-recent-turns",
        type=int,
        default=2,
        help="Newest message turns to preserve untouched, default: 2",
    )
    parser.add_argument(
        "--min-block-chars",
        type=int,
        default=2000,
        help="Minimum terminal/log text block size eligible for compaction, default: 2000",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum adjacent call text size for plateau detection, default: 8000",
    )
    parser.add_argument(
        "--max-plateau-delta-ratio",
        type=float,
        default=0.03,
        help="Maximum adjacent text size delta ratio for plateau detection, default: 0.03",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.terminal_compaction_dry_run import build_terminal_output_compaction_dry_run

    store = open_store_for_db(str(args.db))
    try:
        result = build_terminal_output_compaction_dry_run(
            store,
            limit=args.limit,
            keep_recent_turns=args.keep_recent_turns,
            min_block_chars=args.min_block_chars,
            min_text_chars=args.min_text_chars,
            max_plateau_delta_ratio=args.max_plateau_delta_ratio,
        )
        _attach_terminal_output_compaction_lifecycle_feedback(result, command="review", store=store)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def terminal_output_compaction_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report terminal-output compaction canary impact and rollback gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error rollback threshold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Applied-minus-holdout retry rollback threshold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument("--min-net-savings-usd", type=float, default=0.0, help="Minimum applied net savings for promote, default: 0.")
    parser.add_argument(
        "--max-non-positive-savings-rate",
        type=float,
        default=0.0,
        help="Applied non-positive savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.terminal_compaction_impact import build_terminal_output_compaction_impact_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_terminal_output_compaction_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            min_net_savings_usd=args.min_net_savings_usd,
            max_non_positive_savings_rate=args.max_non_positive_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_terminal_output_compaction_lifecycle_feedback(result, command="impact", store=store)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def repeated_scaffold_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report repeated-scaffold crunch canary impact and rollback gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Evidence staleness threshold, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error regression rollback threshold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Applied-minus-holdout retry regression rollback threshold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument(
        "--max-non-positive-savings-rate",
        type=float,
        default=0.0,
        help="Applied non-positive savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument(
        "--flush-feedback",
        action="store_true",
        help="Flush due repeated-scaffold lifecycle feedback after queueing, bounded by --feedback-limit.",
    )
    parser.add_argument(
        "--feedback-dry-run",
        action="store_true",
        help="Preview due repeated-scaffold lifecycle feedback flush rows without claiming or sending.",
    )
    parser.add_argument("--feedback-limit", type=int, default=5, help="Maximum repeated-scaffold feedback rows to flush or sample, default: 5, max: 100.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.repeated_scaffold_impact import build_repeated_scaffold_impact_report
    from agentflow_proxy.repeated_scaffold_feedback import (
        build_repeated_scaffold_lifecycle_feedback_status,
        flush_repeated_scaffold_lifecycle_feedback,
    )

    store = open_store_for_db(str(args.db))
    try:
        result = build_repeated_scaffold_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            max_non_positive_savings_rate=args.max_non_positive_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_repeated_scaffold_lifecycle_feedback(result, store)
        feedback_limit = max(1, min(int(args.feedback_limit or 5), 100))
        if args.flush_feedback or args.feedback_dry_run:
            result["managed_lifecycle_feedback_flush"] = asyncio.run(
                flush_repeated_scaffold_lifecycle_feedback(
                    store,
                    limit=feedback_limit,
                    dry_run=bool(args.feedback_dry_run),
                )
            )
        result["managed_lifecycle_feedback_queue"] = build_repeated_scaffold_lifecycle_feedback_status(
            store,
            sample_limit=feedback_limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def repeated_scaffold_activation_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Anthropic repeated-scaffold policy-decision activation coverage")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent Anthropic provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.repeated_scaffold_activation import build_repeated_scaffold_activation_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_repeated_scaffold_activation_report(
            store,
            limit=args.limit,
            since=args.since,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def _attach_repeated_scaffold_lifecycle_feedback(result: dict[str, Any], store: Any) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.repeated_scaffold_feedback import (
        SOURCE_SURFACE as REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
        build_repeated_scaffold_lifecycle_feedback,
    )

    payload = build_repeated_scaffold_lifecycle_feedback(result)
    if payload is None:
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-repeated-scaffold-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        })
        return

    try:
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }

    result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta(meta)


def _attach_instruction_dedup_lifecycle_feedback(result: dict[str, Any], store: Any) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.instruction_dedup_feedback import (
        build_instruction_dedup_lifecycle_feedback,
        queue_instruction_dedup_lifecycle_feedback,
    )

    payload = build_instruction_dedup_lifecycle_feedback(result)
    if payload is None:
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-instruction-dedup-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        })
        return

    try:
        meta = asyncio.run(
            queue_instruction_dedup_lifecycle_feedback(
                store,
                result,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }

    result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta(meta)


def _attach_terminal_output_compaction_lifecycle_feedback(
    result: dict[str, Any],
    *,
    command: str,
    store: Any | None = None,
    db_path: str | None = None,
    result_key: str = "managed_lifecycle_feedback",
) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.terminal_compaction_feedback import (
        build_terminal_output_compaction_lifecycle_feedback,
        queue_terminal_output_compaction_lifecycle_feedback,
    )

    payload = build_terminal_output_compaction_lifecycle_feedback(result, command=command)
    if payload is None:
        return

    close_store = False
    local_store = store
    if local_store is None:
        if not db_path:
            return
        local_store = _open_store_for_db(str(db_path))
        close_store = True
    try:
        meta = asyncio.run(
            queue_terminal_output_compaction_lifecycle_feedback(
                local_store,
                result,
                command=command,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }
    finally:
        if close_store and local_store is not None:
            local_store.conn.close()

    result[result_key] = _public_lifecycle_feedback_meta(meta)


def _attach_codex_terminal_transcript_lifecycle_feedback(result: dict[str, Any], *, store: Any) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.codex_terminal_compaction_feedback import (
        build_codex_terminal_transcript_lifecycle_feedback,
        queue_codex_terminal_transcript_lifecycle_feedback,
    )

    payload = build_codex_terminal_transcript_lifecycle_feedback(result)
    if payload is None:
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-codex-terminal-transcript-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        })
        return

    try:
        meta = asyncio.run(
            queue_codex_terminal_transcript_lifecycle_feedback(
                store,
                result,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }

    result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta(meta)


def _attach_openai_cache_replay_lifecycle_feedback(result: dict[str, Any], *, db_path: str) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_lifecycle_feedback

    payload = build_openai_cache_replay_lifecycle_feedback(result)
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
                source_surface=recommendations.CACHE_REPLAY_LIFECYCLE_SOURCE_SURFACE,
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
    if public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def openai_cache_replay_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report OpenAI cache replay canary impact and safety gates from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent OpenAI calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied cache replay samples before promotion, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout cache replay samples before promotion, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate before hold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout retry-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression before hold, default: 2000.")
    parser.add_argument("--max-invalidation-rate", type=float, default=0.02, help="Invalidation rate that triggers rollback, default: 0.02.")
    parser.add_argument("--min-cache-hit-rate", type=float, default=0.01, help="Minimum applied cache hit rate before promotion, default: 0.01.")
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Applied error rate that triggers rollback, default: 0.20.")
    parser.add_argument("--min-savings-realization-ratio", type=float, default=0.50, help="Minimum observed/projected savings ratio before hold, default: 0.50.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_impact_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            max_invalidation_rate=args.max_invalidation_rate,
            min_cache_hit_rate=args.min_cache_hit_rate,
            rollback_error_rate=args.rollback_error_rate,
            min_savings_realization_ratio=args.min_savings_realization_ratio,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()
    _attach_openai_cache_replay_lifecycle_feedback(result, db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_cache_replay_readiness_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Show OpenAI cache replay readiness from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--opportunity-limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for replay opportunity, default: 1000, max: 10000.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=500,
        help="Recent OpenAI calls to scan for replay impact evidence, default: 500, max: 10000.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_readiness_report(
            store,
            opportunity_limit=args.opportunity_limit,
            impact_limit=args.impact_limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def local_promotion_candidates_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Rank local promotion candidates from measured cache, crunch, and routing canaries")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent local metadata rows to inspect, default: 1000, max: 10000.",
    )
    parser.add_argument(
        "--since",
        help="Optional ISO timestamp lower bound for canary impact reports.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.local_promotion_candidates import build_local_promotion_candidates_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_local_promotion_candidates_report(
            store,
            limit=args.limit,
            since=args.since,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def crunch_promotion_draft_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run guarded crunch rule drafts from local promotion candidates")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Local promotion candidates report JSON path, or '-' for stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent local metadata rows to inspect when building a fresh report, default: 1000, max: 10000.",
    )
    parser.add_argument("--since", help="Optional ISO timestamp lower bound when building fresh canary impact reports.")
    parser.add_argument(
        "--initial-canary-fraction",
        type=float,
        default=0.10,
        help="Deterministic applied canary fraction for drafted crunch rule, default: 0.10.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Deterministic holdout fraction for drafted crunch rule, default: 0.10.",
    )
    parser.add_argument(
        "--max-evidence-age-hours",
        type=float,
        default=168.0,
        help="Maximum age for promotion evidence before the dry-run no-ops, default: 168.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.promotion_report:
        report = _read_json_input(str(args.promotion_report), stdin=stdin)
    else:
        from agentflow_proxy.local_promotion_candidates import build_local_promotion_candidates_report

        store = _open_store_for_db(str(args.db))
        try:
            report = build_local_promotion_candidates_report(store, limit=args.limit, since=args.since)
        finally:
            store.conn.close()

    from agentflow_proxy.crunch_promotion_drafts import dry_run_crunch_promotion_drafts

    result = dry_run_crunch_promotion_drafts(
        report,
        initial_canary_fraction=args.initial_canary_fraction,
        holdout_fraction=args.holdout_fraction,
        max_evidence_age_hours=args.max_evidence_age_hours,
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    if (result.get("error") or {}).get("type") in {"invalid_report", "raw_payload_rejected"}:
        return 1
    return 0


def cache_promotion_draft_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run guarded exact-cache rule drafts from local promotion candidates")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Local promotion candidates report JSON path, or '-' for stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent local metadata rows to inspect when building a fresh report, default: 1000, max: 10000.",
    )
    parser.add_argument("--since", help="Optional ISO timestamp lower bound when building fresh canary impact reports.")
    parser.add_argument(
        "--initial-canary-fraction",
        type=float,
        default=0.10,
        help="Deterministic applied canary fraction for drafted cache rule, default: 0.10.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Deterministic holdout fraction for drafted cache rule, default: 0.10.",
    )
    parser.add_argument(
        "--max-evidence-age-hours",
        type=float,
        default=168.0,
        help="Maximum age for promotion evidence before the dry-run no-ops, default: 168.",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=86_400,
        help="Proposed exact-cache TTL for drafted rules, default: 86400.",
    )
    parser.add_argument(
        "--max-drafts",
        type=int,
        default=10,
        help="Maximum number of cache rule drafts to emit, default: 10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.promotion_report:
        report = _read_json_input(str(args.promotion_report), stdin=stdin)
    else:
        from agentflow_proxy.local_promotion_candidates import build_local_promotion_candidates_report

        store = _open_store_for_db(str(args.db))
        try:
            report = build_local_promotion_candidates_report(store, limit=args.limit, since=args.since)
        finally:
            store.conn.close()

    from agentflow_proxy.cache_promotion_drafts import dry_run_cache_promotion_drafts

    result = dry_run_cache_promotion_drafts(
        report,
        initial_canary_fraction=args.initial_canary_fraction,
        holdout_fraction=args.holdout_fraction,
        max_evidence_age_hours=args.max_evidence_age_hours,
        ttl_seconds=args.ttl_seconds,
        max_drafts=args.max_drafts,
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    if (result.get("error") or {}).get("type") in {"invalid_report", "raw_payload_rejected"}:
        return 1
    return 0


def routing_promotion_draft_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run guarded Anthropic routing rule drafts from local promotion candidates")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Local promotion candidates report JSON path, or '-' for stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent local metadata rows to inspect when building a fresh report, default: 1000, max: 10000.",
    )
    parser.add_argument("--since", help="Optional ISO timestamp lower bound when building fresh canary impact reports.")
    parser.add_argument(
        "--initial-canary-fraction",
        type=float,
        default=0.10,
        help="Deterministic applied canary fraction for drafted routing rule, default: 0.10.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Deterministic holdout fraction for drafted routing rule, default: 0.10.",
    )
    parser.add_argument(
        "--max-evidence-age-hours",
        type=float,
        default=72.0,
        help="Maximum age for routing lifecycle evidence before the dry-run no-ops, default: 72.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.promotion_report:
        report = _read_json_input(str(args.promotion_report), stdin=stdin)
    else:
        from agentflow_proxy.local_promotion_candidates import build_local_promotion_candidates_report

        store = _open_store_for_db(str(args.db))
        try:
            report = build_local_promotion_candidates_report(store, limit=args.limit, since=args.since)
        finally:
            store.conn.close()

    from agentflow_proxy.routing_promotion_rule_drafts import dry_run_routing_promotion_drafts

    result = dry_run_routing_promotion_drafts(
        report,
        initial_canary_fraction=args.initial_canary_fraction,
        holdout_fraction=args.holdout_fraction,
        max_evidence_age_hours=args.max_evidence_age_hours,
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    if (result.get("error") or {}).get("type") in {"invalid_report", "raw_payload_rejected"}:
        return 1
    return 0


def openai_cache_replay_apply_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Graduate ready OpenAI cache replay candidates into a local cache canary overlay")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local AgentFlow policy overlays, default: AGENTFLOW_CONFIG_DIR, AGENTFLOW_POLICY_CONFIG_DIR, or ~/.agentflow",
    )
    parser.add_argument(
        "--opportunity-limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for replay opportunity, default: 1000, max: 10000.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=500,
        help="Recent OpenAI calls to scan for replay impact evidence, default: 500, max: 10000.",
    )
    parser.add_argument(
        "--min-observed-savings-usd",
        type=float,
        default=0.0,
        help="Minimum observed applied savings required before writing a canary, default: 0.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
        help="Fraction of matching traffic to keep as holdout, default: 0.20.",
    )
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum ready candidates to write, default: 10.")
    parser.add_argument("--dry-run", action="store_true", help="Preview cache canary policy without writing YAML.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_apply import apply_openai_cache_replay_candidates
    from agentflow_proxy.policy_events import log_policy_event

    store = _open_store_for_db(str(args.db))
    try:
        result = apply_openai_cache_replay_candidates(
            store,
            config_dir=args.config_dir,
            dry_run=args.dry_run,
            opportunity_limit=args.opportunity_limit,
            impact_limit=args.impact_limit,
            min_observed_savings_usd=args.min_observed_savings_usd,
            holdout_fraction=args.holdout_fraction,
            max_candidates=args.max_candidates,
        )
    finally:
        store.conn.close()
    log_policy_event(
        "openai-cache-replay-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "db": str(args.db),
            "config_dir": str(args.config_dir),
            "dry_run": bool(args.dry_run),
            "accepted_candidate_count": (result.get("summary") or {}).get("accepted_candidate_count"),
            "policy_rule_count": (result.get("summary") or {}).get("policy_rule_count"),
            "wrote_policy_files": bool(result.get("wrote_policy_files")),
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def _openai_cache_replay_dry_run_read_error_result(read_error: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.openai_cache_replay_dry_run.v1",
        "ok": False,
        "summary": {
            "openai_rows_considered": 0,
            "policy_rule_count": 0,
            "projected_applied_rows": 0,
            "holdout_rows": 0,
            "blocked_rows": 0,
            "projected_hits": 0,
            "projected_savings_usd": 0.0,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
        },
        "read_error": read_error,
        "rows": [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "pattern_hashes_included": False,
        },
    }


def openai_cache_replay_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run OpenAI cache replay pattern rules against local metadata")
    parser.add_argument(
        "path",
        help="Proposed cache policy JSON path, policy bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent OpenAI provider calls to inspect and aggregate rows to return, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    proposed, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _openai_cache_replay_dry_run_read_error_result(read_error)
        if args.pretty:
            stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _write_json(stdout, result)
        return 1

    from agentflow_proxy.openai_cache_replay_dry_run import build_openai_cache_replay_dry_run

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_dry_run(store, proposed, limit=args.limit)
    finally:
        store.conn.close()
    result["ok"] = True
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_old_context_summary_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run OpenAI old-context summary plans with protocol preservation checks")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_old_context_summary_dry_run import build_openai_old_context_summary_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_old_context_summary_dry_run(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report OpenAI local routing canary impact and promotion verdicts from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent OpenAI calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate before hold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout retry-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-fallback-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout fallback-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression before hold, default: 2000.")
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Applied error rate that triggers rollback, default: 0.20.")
    parser.add_argument("--min-projection-realization-ratio", type=float, default=0.50, help="Minimum observed/projected savings ratio before hold, default: 0.50.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_canary_impact import build_openai_canary_impact_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_canary_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_fallback_rate_delta=args.max_fallback_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            rollback_error_rate=args.rollback_error_rate,
            min_projection_realization_ratio=args.min_projection_realization_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _extract_anthropic_routing_report(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") == "agentflow.pass_through_routing_activation_candidates.v1":
        return payload
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    report = evidence.get("pass_through_routing_report")
    if isinstance(report, dict):
        return report
    stats_summary = evidence.get("stats_summary") if isinstance(evidence.get("stats_summary"), dict) else {}
    report = stats_summary.get("pass_through_routing_report")
    if isinstance(report, dict):
        return report
    report = payload.get("pass_through_routing_report")
    if isinstance(report, dict):
        return report
    return payload


def anthropic_routing_canary_stage_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Stage default-off Anthropic Sonnet-to-Haiku tool-result routing canary drafts from aggregate local metadata")
    parser.add_argument(
        "routing_report",
        help="Pass-through routing activation report JSON, research plan JSON, or '-' for stdin.",
    )
    parser.add_argument("--draft-id", help="Optional local draft ID. When multiple candidates are staged, a numeric suffix is added.")
    parser.add_argument("--canary-fraction", type=float, default=0.05, help="Proposed deterministic canary fraction, default: 0.05.")
    parser.add_argument("--holdout-fraction", type=float, default=0.10, help="Proposed deterministic holdout fraction, default: 0.10.")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum candidate samples before staging, default: 5.")
    parser.add_argument("--top-candidates", type=int, default=1, help="Maximum ranked eligible candidates to stage, default: 1.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        payload = _read_json_input(str(args.routing_report), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.anthropic_routing_canary_stage_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_active_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.anthropic_routing_canary_stage import build_anthropic_routing_canary_stage_report

    result = build_anthropic_routing_canary_stage_report(
        _extract_anthropic_routing_report(payload),
        canary_fraction=args.canary_fraction,
        holdout_fraction=args.holdout_fraction,
        min_samples=args.min_samples,
        top_candidates=args.top_candidates,
        draft_id=args.draft_id,
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def claude_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Claude local routing canary impact and promotion verdicts from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Claude calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate before hold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout retry-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-fallback-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout fallback-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-rate-limit-fallback-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout rate-limit fallback-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression before hold, default: 2000.")
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Applied error rate that triggers rollback, default: 0.20.")
    parser.add_argument("--rollback-fallback-rate", type=float, default=0.50, help="Applied fallback rate that triggers rollback, default: 0.50.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_claude_canary_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_fallback_rate_delta=args.max_fallback_rate_delta,
            max_rate_limit_fallback_rate_delta=args.max_rate_limit_fallback_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            rollback_error_rate=args.rollback_error_rate,
            rollback_fallback_rate=args.rollback_fallback_rate,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def anthropic_routing_lifecycle_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Anthropic routing canary lifecycle evidence from local phase_canary metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Anthropic calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.claude_canary_impact import build_anthropic_routing_canary_lifecycle_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_anthropic_routing_canary_lifecycle_report(
            store,
            limit=args.limit,
            since=args.since,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def claude_canary_actions_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Emit reviewable local Claude routing canary widen, hold, rollback, and more-samples actions")
    parser.add_argument(
        "impact_report",
        nargs="?",
        help="Optional Claude canary impact report JSON path, or '-' to read from stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh impact report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Claude calls to scan when building a fresh report, default: 500.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp when building a fresh report.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--widen-step", type=float, default=0.25, help="Fraction added when widening an existing Claude canary, default: 0.25.")
    parser.add_argument("--max-canary-fraction", type=float, default=1.0, help="Maximum Claude canary fraction, default: 1.0.")
    parser.add_argument("--preserved-holdout-fraction", type=float, default=0.10, help="Minimum holdout fraction preserved while widening, default: 0.10.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        if args.impact_report:
            report = _read_json_input(str(args.impact_report), stdin=stdin)
        else:
            from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report

            store = _open_store_for_db(str(args.db))
            try:
                report = build_claude_canary_impact_report(
                    store,
                    limit=args.limit,
                    since=args.since,
                    min_applied_samples=args.min_applied_samples,
                    min_holdout_samples=args.min_holdout_samples,
                    max_evidence_age_hours=args.max_evidence_age_hours,
                )
            finally:
                store.conn.close()
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.claude_canary_rollout_actions_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.claude_canary_actions import build_claude_canary_actions

    result = build_claude_canary_actions(
        report,
        widen_step=args.widen_step,
        max_canary_fraction=args.max_canary_fraction,
        preserved_holdout_fraction=args.preserved_holdout_fraction,
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def claude_canary_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed Claude routing canary actions to the local routing policy file")
    parser.add_argument(
        "actions",
        nargs="?",
        default="-",
        help="Claude canary rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without writing local YAML files.")
    parser.add_argument("--write", dest="dry_run", action="store_false", help="Write reviewed Claude canary edits to the local routing YAML file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        bundle = _read_json_input(str(args.actions), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.claude_canary_rollout_actions_apply_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.claude_canary_actions import apply_claude_canary_actions
    from agentflow_proxy.policy_events import log_policy_event

    result = apply_claude_canary_actions(
        bundle,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
    )
    event = log_policy_event(
        "claude-canary-actions-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "action_count": (result.get("summary") or {}).get("action_count") if isinstance(result.get("summary"), dict) else None,
            "planned_action_count": (result.get("summary") or {}).get("planned_action_count") if isinstance(result.get("summary"), dict) else None,
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "wrote_policy_files": bool(result.get("wrote_policy_files")),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if event is not None:
        result["policy_event"] = {
            "id": event.get("id"),
            "action": event.get("action"),
            "created_at": event.get("created_at"),
            "ok": event.get("ok"),
        }
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def routing_canary_promote_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Promote winning Claude routing canaries to permanent local routing rules")
    parser.add_argument(
        "impact_report",
        nargs="?",
        help="Optional Claude canary impact report JSON path, or '-' to read from stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh impact report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Claude calls to scan when building a fresh report, default: 500.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp when building a fresh report.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples required for promotion, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples required for promotion, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours when building a fresh report, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate for promotion, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta for promotion, default: 0.05.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression for promotion, default: 2000.")
    parser.add_argument("--apply", action="store_true", help="Write permanent rules and disable matching local canary entries.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        if args.impact_report:
            report = _read_json_input(str(args.impact_report), stdin=stdin)
        else:
            from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report

            store = _open_store_for_db(str(args.db))
            try:
                report = build_claude_canary_impact_report(
                    store,
                    limit=args.limit,
                    since=args.since,
                    min_applied_samples=args.min_applied_samples,
                    min_holdout_samples=args.min_holdout_samples,
                    max_evidence_age_hours=args.max_evidence_age_hours,
                    max_error_rate=args.max_error_rate,
                    max_error_rate_delta=args.max_error_rate_delta,
                    max_latency_regression_ms=args.max_latency_regression_ms,
                )
            finally:
                store.conn.close()
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.routing_canary_promotion_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.routing_canary_promote import (
        apply_routing_canary_promotion_plan,
        build_routing_canary_promotion_plan,
    )

    plan = build_routing_canary_promotion_plan(
        report,
        config_dir=args.config_dir,
        min_applied_samples=args.min_applied_samples,
        min_holdout_samples=args.min_holdout_samples,
        max_error_rate=args.max_error_rate,
        max_error_rate_delta=args.max_error_rate_delta,
        max_latency_regression_ms=args.max_latency_regression_ms,
    )
    result = (
        apply_routing_canary_promotion_plan(plan, config_dir=args.config_dir, dry_run=not args.apply)
        if plan.get("ok")
        else plan
    )
    event = log_policy_event(
        "routing-canary-promote",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "config_dir": args.config_dir,
            "dry_run": not args.apply,
            "action_count": (result.get("summary") or {}).get("action_count") if isinstance(result.get("summary"), dict) else None,
            "planned_action_count": (result.get("summary") or {}).get("planned_action_count") if isinstance(result.get("summary"), dict) else None,
            "promotion_action_count": (plan.get("summary") or {}).get("promotion_action_count") if isinstance(plan.get("summary"), dict) else None,
            "omitted_count": (plan.get("summary") or {}).get("omitted_count") if isinstance(plan.get("summary"), dict) else None,
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "wrote_policy_files": bool(result.get("wrote_policy_files")),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if event is not None:
        result["policy_event"] = {
            "id": event.get("id"),
            "action": event.get("action"),
            "created_at": event.get("created_at"),
            "ok": event.get("ok"),
        }
    result["promotion_plan"] = {
        "schema": plan.get("schema"),
        "ok": plan.get("ok"),
        "summary": plan.get("summary"),
    }
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def routing_experiment_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report local budgeted routing A/B experiment results from metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum candidate rows to include, default: 20.")
    parser.add_argument("--since", help="Only include post-fix shadow yield rows at or after this ISO timestamp.")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Rolling post-fix shadow-yield window when --since is omitted, default: 24. Use 0 for all history.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.routing_experiments import build_routing_experiment_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_routing_experiment_report(
            store,
            limit=args.limit,
            since=args.since,
            window_hours=args.window_hours,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def routing_promotion_draft_stage_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Stage local policy drafts from promoted routing experiment candidates")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Routing experiment promotion report JSON path, or '-' for stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum candidate rows when building a fresh report, default: 20.")
    parser.add_argument(
        "--draft-id",
        help="Optional local draft ID. When multiple promoted candidates are staged, a numeric suffix is added.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--initial-canary-fraction",
        type=float,
        default=0.10,
        help="Deterministic applied canary fraction for staged routing canaries, default: 0.10.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Deterministic holdout fraction for staged routing canaries, default: 0.10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.promotion_report:
        report = _read_json_input(str(args.promotion_report), stdin=stdin)
    else:
        from agentflow_proxy.routing_experiments import build_routing_experiment_report

        store = _open_store_for_db(str(args.db))
        try:
            report = build_routing_experiment_report(store, limit=args.limit)
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.routing_promotion_drafts import stage_routing_promotion_drafts

    result = asyncio.run(stage_routing_promotion_drafts(
        report,
        draft_id=args.draft_id,
        workspace=args.workspace,
        initial_canary_fraction=args.initial_canary_fraction,
        holdout_fraction=args.holdout_fraction,
    ))
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "routing-promotion-draft-stage",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.promotion_report,
            "workspace": args.workspace,
            "candidate_count": summary.get("candidate_count", 0),
            "promoted_candidate_count": summary.get("promoted_candidate_count", 0),
            "staged_count": summary.get("staged_count", 0),
            "omitted_count": summary.get("omitted_count", 0),
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def cache_replayability_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure replay-safe cache opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Replayability groups to return, default: 25",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_cache_replayability

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_cache_replayability(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def request_shape_rollups_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build and persist privacy-safe request-shape rollups from local call metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument("--run-id", help="Optional stable rollup run id. Defaults to a generated local id.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the report without writing request_shape_rollups rows.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.request_shape_rollups import build_request_shape_rollups_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_request_shape_rollups_report(
            store,
            limit=args.limit,
            persist=not args.dry_run,
            run_id=args.run_id,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def request_shape_crunch_canary_stage_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Stage a read-only repeated-context crunch canary action from request-shape metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument("--run-id", help="Optional stable rollup run id. Defaults to a generated local id.")
    parser.add_argument(
        "--persist-rollups",
        action="store_true",
        help="Persist the underlying request_shape_rollups rows. The staged canary action itself remains dry-run/read-only.",
    )
    parser.add_argument(
        "--rollout-fraction",
        type=float,
        default=0.10,
        help="Candidate canary rollout fraction to encode in the staged action, default: 0.10",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Candidate holdout fraction to encode in the staged action, default: 0.10",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the staged canary action to a local crunch rules YAML file. Default is read-only.",
    )
    parser.add_argument(
        "--rules-path",
        help="Crunch rules YAML file to update when --apply is used. Defaults to AGENTFLOW_CRUNCH_RULES or ~/.agentflow/crunch_rules.yaml.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.paths import agentflow_config_path
    from agentflow_proxy.request_shape_rollups import (
        apply_request_shape_crunch_canary_action,
        build_request_shape_crunch_canary_stage_report,
    )

    store = _open_store_for_db(str(args.db))
    try:
        result = build_request_shape_crunch_canary_stage_report(
            store,
            limit=args.limit,
            run_id=args.run_id,
            persist_rollups=args.persist_rollups,
            rollout_fraction=args.rollout_fraction,
            holdout_fraction=args.holdout_fraction,
        )
    finally:
        store.conn.close()
    if args.apply:
        rules_path = Path(args.rules_path or os.getenv("AGENTFLOW_CRUNCH_RULES") or agentflow_config_path("crunch_rules.yaml"))
        action = result.get("top_stage_action") if isinstance(result.get("top_stage_action"), dict) else None
        if action is None:
            apply_result = {
                "schema": "agentflow.request_shape_crunch_canary_apply.v1",
                "ok": False,
                "dry_run": False,
                "wrote_policy_files": False,
                "rules_path_included": False,
                "errors": [{"path": "$.top_stage_action", "message": "no stageable request-shape crunch canary action"}],
            }
        else:
            apply_result = apply_request_shape_crunch_canary_action(action, rules_path=rules_path)
        result["apply_result"] = apply_result
        result["read_only"] = False
        result["dry_run"] = False
        result["ok"] = bool(result.get("ok")) and bool(apply_result.get("ok"))
        if apply_result.get("ok"):
            result["status"] = "staged-and-applied"
            result["reason"] = "applied-local-repeated-context-crunch-canary"
            result["next_action"] = "measure-repeated-context-crunch-canary-impact"
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def request_shape_crunch_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure request-shape repeated-context crunch canary impact from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--max-evidence-age-hours",
        type=float,
        default=72.0,
        help="Maximum age for canary impact evidence before it is treated as stale, default: 72.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.request_shape_rollups import (
        build_request_shape_crunch_canary_impact_report,
        build_request_shape_rollups_report,
    )

    store = _open_store_for_db(str(args.db))
    try:
        rollups = build_request_shape_rollups_report(
            store,
            limit=args.limit,
            persist=False,
            max_crunch_canary_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()
    result = rollups.get("crunch_canary_impact")
    if not isinstance(result, dict):
        result = build_request_shape_crunch_canary_impact_report(
            [],
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    else:
        result = dict(result)
        result["max_evidence_age_hours"] = max(0.0, float(args.max_evidence_age_hours))
    result["source_report"] = {
        "schema": rollups.get("schema"),
        "summary": {
            "rows_considered": (rollups.get("summary") or {}).get("rows_considered")
            if isinstance(rollups.get("summary"), dict)
            else None,
            "rollup_count": (rollups.get("summary") or {}).get("rollup_count")
            if isinstance(rollups.get("summary"), dict)
            else None,
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
        },
    }
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def request_shape_cache_replay_canary_stage_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Stage a read-only OpenAI Responses exact-cache replay canary action from request-shape metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument("--run-id", help="Optional stable rollup run id. Defaults to a generated local id.")
    parser.add_argument(
        "--persist-rollups",
        action="store_true",
        help="Persist the underlying request_shape_rollups rows. The staged canary action itself remains dry-run/read-only.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local AgentFlow policy overlays when --apply is used, default: AGENTFLOW_CONFIG_DIR, AGENTFLOW_POLICY_CONFIG_DIR, or ~/.agentflow",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the selected request-shape cache replay canary to cache_canary_policy.yaml.",
    )
    parser.add_argument(
        "--rollout-fraction",
        type=float,
        default=0.10,
        help="Candidate canary rollout fraction to encode in the staged action, default: 0.10",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Candidate holdout fraction to encode in the staged action, default: 0.10",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.request_shape_rollups import build_request_shape_cache_replay_canary_stage_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_request_shape_cache_replay_canary_stage_report(
            store,
            limit=args.limit,
            run_id=args.run_id,
            persist_rollups=args.persist_rollups,
            rollout_fraction=args.rollout_fraction,
            holdout_fraction=args.holdout_fraction,
        )
    finally:
        store.conn.close()
    if args.apply:
        from agentflow_proxy.request_shape_rollups import apply_request_shape_cache_replay_canary_action

        action = result.get("top_stage_action") if isinstance(result.get("top_stage_action"), dict) else {}
        apply_result = apply_request_shape_cache_replay_canary_action(
            action,
            rules_path=Path(args.config_dir).expanduser() / "cache_canary_policy.yaml",
            dry_run=False,
        )
        result = {
            **result,
            "dry_run": False,
            "read_only": False,
            "apply_result": apply_result,
            "wrote_policy_files": bool(apply_result.get("wrote_policy_files")),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "cache_entries_written": False,
        }
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if not args.apply or bool((result.get("apply_result") or {}).get("ok")) else 1


def request_shape_cache_replay_evidence_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report aggregate local evidence for staged request-shape cache replay canaries")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Recent provider calls to inspect, default: 10000",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing cache_canary_policy.yaml, default: AGENTFLOW_CONFIG_DIR, AGENTFLOW_POLICY_CONFIG_DIR, or ~/.agentflow",
    )
    parser.add_argument(
        "--rules-path",
        help="Explicit cache_canary_policy.yaml path. Overrides --config-dir.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=72.0,
        help="Mark evidence stale after this many hours without observed canary traffic, default: 72",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.request_shape_rollups import build_request_shape_cache_replay_evidence_report

    rules_path = Path(args.rules_path).expanduser() if args.rules_path else Path(args.config_dir).expanduser() / "cache_canary_policy.yaml"
    store = _open_store_for_db(str(args.db))
    try:
        result = build_request_shape_cache_replay_evidence_report(
            store,
            rules_path=rules_path,
            limit=args.limit,
            max_age_hours=args.max_age_hours,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def cache_replay_cohorts_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Rank replay-ready plateau cohorts from local cache metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="Recent provider and Codex rows per surface to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Ranked cohorts to return, default: 25",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_cache_replay_cohort_ranking

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            stats_cache_replay_cohort_ranking(
                store,
                limit=args.limit,
                row_limit=args.scan_limit,
            )
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def cache_smoke_diagnostic_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether the local exact cache can serve hits from metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Newest cache rows to summarize, default: 10",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=5000,
        help="Recent call rows to inspect for cache decisions, default: 5000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.cache_smoke import build_cache_smoke_diagnostic

    store = _open_store_for_db(str(args.db))
    try:
        result = build_cache_smoke_diagnostic(store, limit=args.limit, scan_limit=args.scan_limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _cache_replay_dry_run_read_error_result(read_error: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.cache_replay_dry_run.v1",
        "ok": False,
        "summary": {
            "rows_considered": 0,
            "policy_rule_count": 0,
            "candidate_rows": 0,
            "projected_exact_hits": 0,
            "projected_streaming_hits": 0,
            "estimated_saved_cost_usd": 0.0,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
        },
        "read_error": read_error,
        "rows": [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
        },
    }


def cache_replay_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run proposed cache replay pattern rules against recent local metadata")
    parser.add_argument(
        "path",
        help="Proposed cache policy JSON path, policy bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="Recent provider and Codex rows per surface to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Dry-run aggregate rows to return, default: 50",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    proposed, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _cache_replay_dry_run_read_error_result(read_error)
        if args.pretty:
            stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _write_json(stdout, result)
        return 1

    from agentflow_proxy.stats import stats_cache_replay_dry_run

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            stats_cache_replay_dry_run(
                store,
                proposed,
                limit=args.limit,
                row_limit=args.scan_limit,
            )
        )
    finally:
        store.conn.close()
    result["ok"] = True
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def phase_routing_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure or dry-run Anthropic phase-routing policy from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent Anthropic provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    parser.add_argument(
        "--dry-run-policy",
        help="Proposed routing YAML or policy bundle JSON/YAML to simulate without writing local policy files.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum matched safe samples required before projecting savings in dry-run mode, default: 1.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Historical matched error-rate warning threshold for dry-run mode, default: 0.05.",
    )
    parser.add_argument(
        "--stale-hours",
        type=int,
        default=168,
        help="Treat evidence older than this as stale in dry-run mode, default: 168 hours.",
    )
    parser.add_argument(
        "--require-shadow-support",
        action="store_true",
        help="Exclude streaming calls when the proposed phase policy requires shadow/holdout comparison support.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.phase_routing_report import (
        build_phase_routing_dry_run,
        build_phase_routing_report,
        load_phase_routing_policy,
    )

    store = _open_store_for_db(str(args.db))
    try:
        if args.dry_run_policy:
            proposed = load_phase_routing_policy(args.dry_run_policy)
            result = build_phase_routing_dry_run(
                store,
                proposed,
                limit=args.limit,
                min_samples=args.min_samples,
                max_error_rate=args.max_error_rate,
                stale_hours=args.stale_hours,
                require_shadow_support=args.require_shadow_support,
            )
        else:
            result = build_phase_routing_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.dry_run_policy:
        _attach_phase_routing_lifecycle_feedback(result, command="dry-run", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def session_phase_memory_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build metadata-only session phase memory rollups from local calls")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Recent calls per session to include in each memory window, default: 20, max: 200",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.session_phase_memory import build_session_phase_memory

    store = _open_store_for_db(str(args.db))
    try:
        result = build_session_phase_memory(
            store,
            limit=args.limit,
            window_size=args.window_size,
        )
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


def optimization_eval_plan_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export family-agnostic optimization eval plans from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider and Codex rows to inspect per source report, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples for managed pattern readiness normalization, default: 1",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization_eval_plan import build_optimization_eval_plan

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            build_optimization_eval_plan(
                store,
                limit=args.limit,
                min_samples=args.min_samples,
            )
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _read_json_input(path: str, *, stdin: Any = None) -> dict[str, Any]:
    if path == "-":
        source = stdin if stdin is not None else sys.stdin
        return json.loads(source.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def optimization_shadow_eval_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run opt-in local metadata-only shadow evals from an optimization eval plan")
    parser.add_argument(
        "plan",
        help="Optimization eval plan JSON path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for result records, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--results-jsonl",
        help="Optional path to write sanitized per-candidate result records as JSONL.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to provider-call execution. Requires --budget-usd > 0; rows without replayable local inputs remain blocked.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=0.0,
        help="Maximum provider-call spend allowed when --execute is set. Default 0, which disables execution.",
    )
    parser.add_argument(
        "--min-output-similarity",
        type=float,
        default=0.9,
        help="Minimum offline fixture output similarity/quality score required for a pass verdict, default: 0.9",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum plan rows to evaluate, default: 100, max: 1000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if args.execute and args.budget_usd <= 0:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_shadow_eval_error.v1",
                "error": {
                    "type": "missing_budget_cap",
                    "message": "--execute requires --budget-usd greater than 0",
                },
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 2

    from agentflow_proxy.optimization_shadow_eval import run_optimization_shadow_eval

    try:
        plan = _read_json_input(str(args.plan), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_shadow_eval_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    store = _open_store_for_db(str(args.db))
    try:
        result = run_optimization_shadow_eval(
            plan,
            store=store,
            execute=bool(args.execute),
            budget_usd=float(args.budget_usd or 0.0),
            min_output_similarity=float(args.min_output_similarity or 0.9),
            max_candidates=int(args.max_candidates or 100),
            results_jsonl_path=args.results_jsonl,
        )
    finally:
        store.conn.close()

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_eval_queue_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded batch from the local optimization eval queue")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queue selection and result records, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--family",
        help="Optional optimization family filter, for example phase_routing or cache_replayability.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum queue candidates to evaluate, default: 25, max: 1000",
    )
    parser.add_argument(
        "--plan-limit",
        type=int,
        default=500,
        help="Recent provider and Codex rows to inspect while building the queue, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples for managed pattern readiness normalization, default: 1",
    )
    parser.add_argument(
        "--max-candidate-age-hours",
        type=int,
        help="Record candidates older than this many hours as blocked with candidate-stale.",
    )
    parser.add_argument(
        "--results-jsonl",
        help="Optional path to write sanitized per-candidate result records as JSONL.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to provider-call execution. Requires --budget-usd > 0; rows without replayable local inputs remain blocked.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=0.0,
        help="Maximum provider-call spend allowed when --execute is set. Default 0, which disables execution.",
    )
    parser.add_argument(
        "--min-output-similarity",
        type=float,
        default=0.9,
        help="Minimum offline fixture output similarity/quality score required for a pass verdict, default: 0.9",
    )
    parser.add_argument(
        "--promotion-report",
        help="Optimization promotion report JSON path, or '-' for stdin. When set, backfill eval tasks from needs_eval promotion blockers.",
    )
    parser.add_argument(
        "--promotion-blocker-review",
        help="Promotion blocker review/recommendation JSON path, or '-' for stdin. When set, queue local shadow eval tasks from managed needs-eval recommendations.",
    )
    parser.add_argument(
        "--apply-backfill",
        action="store_true",
        help="With --promotion-report, write metadata-only queued eval task rows. Without this flag, emit a dry-run only.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    stdin = stdin if stdin is not None else sys.stdin

    if args.execute and args.budget_usd <= 0:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_eval_queue_error.v1",
                "error": {
                    "type": "missing_budget_cap",
                    "message": "--execute requires --budget-usd greater than 0",
                },
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 2

    if args.promotion_report and args.promotion_blocker_review:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_eval_queue_error.v1",
                "error": {
                    "type": "conflicting_inputs",
                    "message": "--promotion-report and --promotion-blocker-review are mutually exclusive",
                },
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
                "wrote_eval_queue_rows": False,
            },
        )
        return 2

    if args.promotion_report:
        from agentflow_proxy.optimization_eval_queue import backfill_promotion_eval_tasks

        try:
            promotion_report = _read_json_input(str(args.promotion_report), stdin=stdin)
        except (OSError, json.JSONDecodeError) as exc:
            _write_json(
                stderr,
                {
                    "ok": False,
                    "schema": "agentflow.optimization_promotion_eval_backfill_error.v1",
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    "provider_calls_made": False,
                    "wrote_local_policy_files": False,
                    "wrote_eval_queue_rows": False,
                },
            )
            return 1

        store = _open_store_for_db(str(args.db))
        try:
            result = backfill_promotion_eval_tasks(
                store,
                promotion_report,
                family=args.family,
                limit=int(args.limit or 25),
                apply=bool(args.apply_backfill),
            )
        finally:
            store.conn.close()

        if args.pretty:
            stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _write_json(stdout, result)
        return 0

    if args.promotion_blocker_review:
        from agentflow_proxy.optimization_eval_queue import queue_promotion_recommendation_eval_tasks
        from agentflow_proxy.promotion_blocker_review import build_promotion_blocker_recommendation_review

        try:
            recommendation_payload = _read_json_input(str(args.promotion_blocker_review), stdin=stdin)
        except (OSError, json.JSONDecodeError) as exc:
            _write_json(
                stderr,
                {
                    "ok": False,
                    "schema": "agentflow.promotion_recommendation_eval_queue_error.v1",
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    "provider_calls_made": False,
                    "wrote_local_policy_files": False,
                    "wrote_eval_queue_rows": False,
                },
            )
            return 1

        if recommendation_payload.get("schema") != "agentflow.promotion_blocker_recommendation_review.v1":
            recommendation_payload = build_promotion_blocker_recommendation_review(recommendation_payload, limit=int(args.limit or 25))

        store = _open_store_for_db(str(args.db))
        try:
            result = queue_promotion_recommendation_eval_tasks(
                store,
                recommendation_payload,
                family=args.family,
                limit=int(args.limit or 25),
                apply=bool(args.apply_backfill),
            )
        finally:
            store.conn.close()

        if args.pretty:
            stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _write_json(stdout, result)
        return 0

    from agentflow_proxy.optimization_eval_queue import run_optimization_eval_queue

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            run_optimization_eval_queue(
                store,
                family=args.family,
                limit=int(args.limit or 25),
                max_candidate_age_hours=args.max_candidate_age_hours,
                execute=bool(args.execute),
                budget_usd=float(args.budget_usd or 0.0),
                min_output_similarity=float(args.min_output_similarity or 0.9),
                plan_limit=int(args.plan_limit or 500),
                min_samples=int(args.min_samples or 1),
                results_jsonl_path=args.results_jsonl,
            )
        )
    finally:
        store.conn.close()

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_promotion_report_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Score local eval, canary, and holdout evidence into optimization promotion verdicts")
    parser.add_argument(
        "plan",
        nargs="?",
        help="Optional optimization eval plan JSON path, or '-' to read from stdin. If omitted, a fresh plan is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--evidence-report",
        action="append",
        default=[],
        help="Optional sanitized post-apply impact or feedback report JSON to merge by candidate_id. May be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum candidates to score and recent eval result rows to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples when building a fresh eval plan, default: 1",
    )
    parser.add_argument(
        "--min-eval-pass-count",
        type=int,
        default=1,
        help="Minimum passing local eval results required before widening, default: 1",
    )
    parser.add_argument(
        "--min-canary-applied-samples",
        type=int,
        default=2,
        help="Minimum applied canary samples required before widening, default: 2",
    )
    parser.add_argument(
        "--min-canary-holdout-samples",
        type=int,
        default=1,
        help="Minimum holdout samples required before widening, default: 1",
    )
    parser.add_argument(
        "--max-evidence-age-hours",
        type=int,
        default=168,
        help="Mark eval evidence stale after this many hours, default: 168",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        plan = _read_json_input(str(args.plan), stdin=stdin) if args.plan else None
        evidence_reports = [_read_json_input(str(path), stdin=stdin) for path in args.evidence_report]
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_report_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_report import build_optimization_promotion_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_optimization_promotion_report(
            store,
            plan=plan,
            evidence_reports=evidence_reports,
            limit=args.limit,
            min_samples=args.min_samples,
            min_eval_pass_count=args.min_eval_pass_count,
            min_canary_applied_samples=args.min_canary_applied_samples,
            min_canary_holdout_samples=args.min_canary_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_promotion_actions_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Emit local rollout actions from optimization promotion verdicts")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Optional optimization promotion report JSON path, or '-' to read from stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh promotion report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum candidates to score when building a fresh promotion report, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples when building a fresh eval plan for a fresh promotion report, default: 1",
    )
    parser.add_argument("--initial-canary-fraction", type=float, default=0.10, help="Canary fraction for first local rollout actions, default: 0.10")
    parser.add_argument("--widen-step", type=float, default=0.25, help="Fraction added when widening an existing canary, default: 0.25")
    parser.add_argument("--max-canary-fraction", type=float, default=1.0, help="Maximum recommended canary fraction, default: 1.0")
    parser.add_argument("--holdout-fraction", type=float, default=0.10, help="Deterministic holdout fraction to preserve, default: 0.10")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        if args.promotion_report:
            report = _read_json_input(str(args.promotion_report), stdin=stdin)
        else:
            from agentflow_proxy.optimization_promotion_report import build_optimization_promotion_report

            store = _open_store_for_db(str(args.db))
            try:
                report = build_optimization_promotion_report(
                    store,
                    limit=args.limit,
                    min_samples=args.min_samples,
                )
            finally:
                store.conn.close()
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_rollout_actions_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_actions import build_optimization_promotion_actions

    result = build_optimization_promotion_actions(
        report,
        initial_canary_fraction=args.initial_canary_fraction,
        widen_step=args.widen_step,
        max_canary_fraction=args.max_canary_fraction,
        holdout_fraction=args.holdout_fraction,
    )

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _fetch_promotion_blocker_recommendations(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    safe_url = _redact_url(args.url)
    headers, auth_configured, auth_source = _managed_policy_auth(args)
    secret = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    if not auth_configured and not args.allow_unauthenticated:
        return {}, {
            "status": "skipped",
            "reason": "missing-auth",
            "url": safe_url,
            "auth_configured": False,
            "auth_source": "",
            "error": {
                "type": "missing_auth",
                "message": f"set --api-key, --api-key-env, {MANAGED_POLICY_API_KEY_ENV}, or --allow-unauthenticated",
            },
        }

    started = time.time()
    try:
        response = httpx.get(
            args.url,
            headers=headers,
            params={
                key: value
                for key, value in {
                    "limit": args.source_limit,
                    "max_recommendations": args.limit,
                }.items()
                if value is not None
            },
            timeout=args.timeout,
        )
        latency_ms = int((time.time() - started) * 1000)
    except httpx.HTTPError as exc:
        return {}, {
            "status": "unavailable",
            "reason": "request-failed",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "latency_ms": int((time.time() - started) * 1000),
            "error": {"type": exc.__class__.__name__, "message": _redact_secret(str(exc), secret)},
        }
    if response.status_code >= 400:
        return {}, {
            "status": "unavailable",
            "reason": "server-error",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "body": _redact_secret(response.text[:500], secret),
            "error": {"type": "server_error", "message": "managed server returned an error response"},
        }
    try:
        return response.json(), {
            "status": "received",
            "reason": "ok",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    except ValueError as exc:
        return {}, {
            "status": "unavailable",
            "reason": "invalid-json",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "error": {"type": "invalid_json", "message": f"managed server response was not valid JSON: {exc}"},
        }


def optimization_promotion_blocker_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed promotion blocker next-action recommendations as local candidates")
    parser.add_argument(
        "recommendations",
        nargs="?",
        help="Promotion blocker recommendation JSON path, or '-' to read from stdin. If omitted, --url or AGENTFLOW_PROMOTION_BLOCKER_RECOMMENDATIONS_URL is used when configured.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(PROMOTION_BLOCKER_RECOMMENDATIONS_URL_ENV),
        help=f"Managed promotion blocker recommendations URL. May also be set with {PROMOTION_BLOCKER_RECOMMENDATIONS_URL_ENV}.",
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
    parser.add_argument("--limit", type=int, default=20, help="Maximum recommendations to include in the local review, default: 20.")
    parser.add_argument("--source-limit", type=int, default=500, help="Maximum managed rollups to scan when fetching, default: 500.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print review JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    try:
        if args.recommendations:
            payload = _read_json_input(str(args.recommendations), stdin=stdin)
            fetch = {"status": "skipped", "reason": "local-input"}
        elif args.url:
            payload, fetch = _fetch_promotion_blocker_recommendations(args)
        else:
            payload = {}
            fetch = {"status": "skipped", "reason": "no-managed-url-configured"}
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.promotion_blocker_recommendation_review_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.promotion_blocker_review import build_promotion_blocker_recommendation_review

    result = build_promotion_blocker_recommendation_review(payload, limit=args.limit, fetch=fetch)
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "promotion-blocker-recommendations-review",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.recommendations,
            "url": _redact_url(args.url),
            "fetch_status": fetch.get("status") if isinstance(fetch, dict) else None,
            "review_candidate_count": summary.get("review_candidate_count"),
            "group_count": summary.get("group_count"),
            "recommended_count": summary.get("recommended_count"),
            "noop_count": summary.get("noop_count"),
            "wrote_local_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": result.get("managed_server_calls_made"),
            "exit_code": 0,
        },
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0


def optimization_promotion_canary_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply optimization promotion canaries to local routing, crunch, and cache policy files")
    parser.add_argument(
        "promotion_actions",
        nargs="?",
        default="-",
        help="Optimization promotion rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache"],
        help="Policy section to apply. May be repeated. Default: routing, crunch, and cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without writing local YAML files.",
    )
    parser.add_argument(
        "--write",
        dest="dry_run",
        action="store_false",
        help="Write reviewed promotion canary edits to local YAML policy files.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        bundle = _read_json_input(str(args.promotion_actions), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_canary_apply_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_canary import apply_optimization_promotion_canaries

    result = apply_optimization_promotion_canaries(
        bundle,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
        sections=args.section,
    )
    _attach_optimization_promotion_lifecycle_feedback(
        result,
        command="dry-run" if args.dry_run else "apply",
        db_path=str(args.db),
    )
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def optimization_promotion_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Report post-apply optimization promotion canary impact from local metadata")
    parser.add_argument(
        "promotion_actions",
        nargs="?",
        default="-",
        help="Optimization promotion rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp. Defaults to the action bundle generated_at.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument(
        "--no-record-outcome-feedback",
        action="store_true",
        help="Do not append the metadata-only local promotion outcome feedback ledger.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        bundle = _read_json_input(str(args.promotion_actions), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_impact_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
                "wrote_store": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_impact import measure_optimization_promotion_impact

    store = _open_store_for_db(str(args.db))
    try:
        result = measure_optimization_promotion_impact(
            bundle,
            store_obj=store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
        if result.get("ok") and not args.no_record_outcome_feedback:
            from agentflow_proxy.promotion_outcome_feedback import record_promotion_outcome_feedback

            feedback = record_promotion_outcome_feedback(result, store_obj=store)
            result["promotion_outcome_feedback"] = feedback
            result["wrote_store"] = bool(feedback.get("wrote_store"))
    finally:
        store.conn.close()

    _attach_optimization_promotion_lifecycle_feedback(result, command="impact", db_path=str(args.db))
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


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


def _rollout_lifecycle_nested_counts(items: list[dict[str, Any]], object_key: str, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        nested = item.get(object_key) if isinstance(item.get(object_key), dict) else {}
        value = nested.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rollout_lifecycle_rejection_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        reason = str(action.get("reason") or "")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
        family_validation = action.get("family_validation") if isinstance(action.get("family_validation"), dict) else {}
        for error in family_validation.get("errors") or []:
            if isinstance(error, dict):
                message = str(error.get("message") or "family-specific-validation-failed")
                counts[message] = counts.get(message, 0) + 1
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
    family_validation = action.get("family_validation") if isinstance(action.get("family_validation"), dict) else {}
    snapshot = {
        "action_id": _rollout_action_id(action),
        "status": action.get("status"),
        "reason": action.get("reason"),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "policy_section": action.get("policy_section"),
        "policy_source": current_rule.get("policy_source"),
        "pattern_family": family_validation.get("family"),
        "policy_profile": family_validation.get("policy_profile"),
        "family_validation_status": family_validation.get("status"),
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
        "pattern_family_counts": _rollout_lifecycle_nested_counts(actions, "family_validation", "family"),
        "family_validation_status_counts": _rollout_lifecycle_nested_counts(actions, "family_validation", "status"),
        "rejection_reason_counts": _rollout_lifecycle_rejection_counts(actions),
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


def _optimization_promotion_event_type(command: str, result: dict[str, Any]) -> str:
    if command == "impact":
        return "impact"
    if command == "dry-run" or result.get("dry_run"):
        return "dry-run"
    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    action_types = {str(item.get("action_type") or "") for item in actions}
    if action_types & {"rollback", "retire", "disable"}:
        return "rollback"
    return "apply"


def _optimization_promotion_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _optimization_promotion_reason_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        for key in ("reason",):
            value = action.get(key)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
        next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
        for key in ("reason_codes", "warning_codes"):
            for value in next_step.get(key) or []:
                if isinstance(value, str) and value:
                    counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _optimization_promotion_action_snapshot(action: dict[str, Any]) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    projection = action.get("projection") if isinstance(action.get("projection"), dict) else {}
    actual = action.get("actual") if isinstance(action.get("actual"), dict) else {}
    next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
    cohorts = actual.get("cohorts") if isinstance(actual.get("cohorts"), dict) else {}
    evidence = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
    evidence_counts = evidence.get("cohort_counts") if isinstance(evidence.get("cohort_counts"), dict) else {}
    policy_source = (
        action.get("policy_source")
        or local_update.get("policy_source")
        or "managed-recommended"
    )
    requested_model_family = (
        action.get("requested_model_family")
        or action.get("requested_model")
        or local_update.get("model_pattern")
    )
    routed_model_family = (
        action.get("routed_model_family")
        or action.get("candidate_target_model")
        or action.get("target_model")
        or local_update.get("candidate_target_model")
        or local_update.get("target_model")
    )
    actual_counts = {
        "canary_applied": actual.get("actual_canary_applied_count"),
        "canary_holdout": actual.get("actual_canary_holdout_count"),
        "skipped": actual.get("actual_skipped_count"),
        "bypassed_or_disabled": actual.get("actual_bypassed_or_disabled_count"),
        "safety_stopped": actual.get("actual_safety_stopped_count"),
    }
    projected_counts = {
        "canary_applied": projection.get("current_canary_applied_count", evidence_counts.get("canary_applied")),
        "canary_holdout": projection.get("current_canary_holdout_count", evidence_counts.get("canary_holdout")),
        "bypassed_or_disabled": projection.get("current_bypassed_or_disabled_count", evidence_counts.get("bypassed_or_disabled")),
    }
    snapshot = {
        "action_id": action.get("action_id"),
        "action_type": action.get("action_type"),
        "status": action.get("status"),
        "reason": action.get("reason"),
        "action_family": action.get("action_family"),
        "optimization_family": action.get("optimization_family"),
        "source_surface": action.get("source_surface"),
        "app_family": action.get("app_family"),
        "policy_section": action.get("policy_section"),
        "policy_source": policy_source,
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "requested_model_family": requested_model_family,
        "routed_model_family": routed_model_family,
        "model_family_pair": f"{requested_model_family or 'unknown'}->{routed_model_family}" if routed_model_family else None,
        "target_model": routed_model_family,
        "canary_fraction": action.get("canary_fraction") or projection.get("target_canary_fraction"),
        "holdout_fraction": action.get("holdout_fraction") or projection.get("target_holdout_fraction"),
        "projected_savings_usd": projection.get("projected_savings_usd"),
        "observed_savings_usd": actual.get("observed_savings_usd"),
        "projected_cohort_counts": {key: value for key, value in projected_counts.items() if value not in (None, "", [], {})},
        "actual_cohort_counts": {key: value for key, value in actual_counts.items() if value not in (None, "", [], {})},
        "actual_canary_applied_count": actual.get("actual_canary_applied_count"),
        "actual_canary_holdout_count": actual.get("actual_canary_holdout_count"),
        "actual_skipped_count": actual.get("actual_skipped_count"),
        "actual_bypassed_or_disabled_count": actual.get("actual_bypassed_or_disabled_count"),
        "actual_safety_stopped_count": actual.get("actual_safety_stopped_count"),
        "error_rate_delta": actual.get("applied_minus_holdout_error_rate"),
        "retry_rate_delta": actual.get("applied_minus_holdout_retry_rate"),
        "latency_avg_delta_ms": actual.get("applied_minus_holdout_latency_avg_ms"),
        "applied_error_rate": (cohorts.get("canary_applied") or {}).get("error_rate") if isinstance(cohorts.get("canary_applied"), dict) else None,
        "holdout_error_rate": (cohorts.get("canary_holdout") or {}).get("error_rate") if isinstance(cohorts.get("canary_holdout"), dict) else None,
        "applied_retry_rate": (cohorts.get("canary_applied") or {}).get("retry_rate") if isinstance(cohorts.get("canary_applied"), dict) else None,
        "holdout_retry_rate": (cohorts.get("canary_holdout") or {}).get("retry_rate") if isinstance(cohorts.get("canary_holdout"), dict) else None,
        "status_buckets": actual.get("status_buckets") or [],
        "reason_buckets": actual.get("reason_buckets") or [],
        "error_buckets": actual.get("error_buckets") or [],
        "latency_buckets": actual.get("latency_buckets") or [],
        "next_step_verdict": next_step.get("verdict"),
        "next_step_reason_codes": next_step.get("reason_codes") or [],
        "next_step_warning_codes": next_step.get("warning_codes") or [],
    }
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, "", [], {})
    }


def _optimization_promotion_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy import __version__

    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    snapshots = [_optimization_promotion_action_snapshot(item) for item in actions]
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    source_bundle = result.get("source_action_bundle") if isinstance(result.get("source_action_bundle"), dict) else {}
    action_ids = sorted({str(item.get("action_id")) for item in snapshots if item.get("action_id")})
    candidate_ids = sorted({str(item.get("target_candidate_id")) for item in snapshots if item.get("target_candidate_id")})
    rule_ids = sorted({str(item.get("target_rule_id")) for item in snapshots if item.get("target_rule_id")})
    policy_sections = sorted(_optimization_promotion_counts(actions, "policy_section"))
    source_surface_counts = _optimization_promotion_counts(snapshots, "source_surface")
    model_family_pair_counts = _optimization_promotion_counts(snapshots, "model_family_pair")
    event_type = _optimization_promotion_event_type(command, result)
    basis = {
        "command": command,
        "generated_at": result.get("generated_at"),
        "source_generated_at": source_bundle.get("generated_at"),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "status": result.get("status"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    metadata: dict[str, Any] = {
        "schema": "agentflow.optimization_promotion_lifecycle_feedback.v1",
        "lifecycle_kind": "optimization_promotion_canary",
        "command": f"optimization-promotion-{command}",
        "local_result_status": "ok" if result.get("ok") else "error",
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only")),
        "wrote_policy_files": bool(result.get("wrote_policy_files")),
        "action_count": len(actions) or summary.get("action_count") or source_bundle.get("action_count") or 0,
        "planned_action_count": summary.get("planned_action_count"),
        "skipped_action_count": summary.get("skipped_action_count"),
        "error_count": summary.get("error_count"),
        "action_type_counts": _optimization_promotion_counts(actions, "action_type"),
        "policy_section_counts": _optimization_promotion_counts(actions, "policy_section"),
        "source_surface_counts": source_surface_counts,
        "model_family_pair_counts": model_family_pair_counts,
        "local_status_counts": _optimization_promotion_counts(actions, "status"),
        "reason_code_counts": _optimization_promotion_reason_counts(actions),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "action_snapshots": snapshots,
        "sampled_call_count": summary.get("sampled_call_count"),
        "observed_promotion_metadata_row_count": summary.get("observed_promotion_metadata_row_count"),
        "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
        "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
        "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
        "actual_skipped_count": summary.get("actual_skipped_count"),
        "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
        "actual_safety_stopped_count": summary.get("actual_safety_stopped_count"),
        "observed_savings_usd": summary.get("observed_savings_usd"),
        "next_step_counts": summary.get("next_step_counts"),
        "recommendation_counts": summary.get("recommendation_counts"),
        "family_impacts": result.get("family_impacts") if command == "impact" else None,
        "stale_evidence_action_count": summary.get("stale_evidence_action_count"),
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
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
            "db_path_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": action_ids[0] if len(action_ids) == 1 else f"optimization-promotion:{digest}",
        "bundle_hash": f"sha256:{hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
        "policy_sections": policy_sections,
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _attach_optimization_promotion_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _optimization_promotion_lifecycle_payload(command, result)
    store = None
    try:
        store = _open_store_for_db(str(db_path))
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=recommendations.OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE,
                queue_when_disabled=True,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
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
    if public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def _fetch_post_promotion_priority_deltas(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    safe_url = _redact_url(args.url)
    headers, auth_configured, auth_source = _managed_policy_auth(args)
    secret = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    if not auth_configured and not args.allow_unauthenticated:
        return {}, {
            "status": "skipped",
            "reason": "missing-auth",
            "url": safe_url,
            "auth_configured": False,
            "auth_source": "",
            "error": {
                "type": "missing_auth",
                "message": f"set --api-key, --api-key-env, {MANAGED_POLICY_API_KEY_ENV}, or --allow-unauthenticated",
            },
        }

    started = time.time()
    try:
        response = httpx.get(
            args.url,
            headers=headers,
            params={
                key: value
                for key, value in {
                    "limit": args.source_limit,
                    "max_deltas": args.limit,
                }.items()
                if value is not None
            },
            timeout=args.timeout,
        )
        latency_ms = int((time.time() - started) * 1000)
    except httpx.HTTPError as exc:
        return {}, {
            "status": "unavailable",
            "reason": "request-failed",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "latency_ms": int((time.time() - started) * 1000),
            "error": {"type": exc.__class__.__name__, "message": _redact_secret(str(exc), secret)},
        }
    if response.status_code >= 400:
        return {}, {
            "status": "unavailable",
            "reason": "server-error",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "body": _redact_secret(response.text[:500], secret),
            "error": {"type": "server_error", "message": "managed server returned an error response"},
        }
    try:
        return response.json(), {
            "status": "received",
            "reason": "ok",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    except ValueError as exc:
        return {}, {
            "status": "unavailable",
            "reason": "invalid-json",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "error": {"type": "invalid_json", "message": f"managed server response was not valid JSON: {exc}"},
        }


def post_promotion_priority_delta_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Review managed post-promotion policy priority deltas as local widen/rollback/keep-blocked candidates"
    )
    parser.add_argument(
        "deltas",
        nargs="?",
        help="Post-promotion priority delta JSON path, or '-' to read from stdin. If omitted, --url or AGENTFLOW_POST_PROMOTION_PRIORITY_DELTAS_URL is used when configured.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(POST_PROMOTION_PRIORITY_DELTAS_URL_ENV),
        help=f"Managed post-promotion priority deltas URL. May also be set with {POST_PROMOTION_PRIORITY_DELTAS_URL_ENV}.",
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
    parser.add_argument("--limit", type=int, default=20, help="Maximum deltas to include in the local review, default: 20.")
    parser.add_argument("--source-limit", type=int, default=500, help="Maximum managed rollups to scan when fetching, default: 500.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print review JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    try:
        if args.deltas:
            payload = _read_json_input(str(args.deltas), stdin=stdin)
            fetch = {"status": "skipped", "reason": "local-input"}
        elif args.url:
            payload, fetch = _fetch_post_promotion_priority_deltas(args)
        else:
            payload = {}
            fetch = {"status": "skipped", "reason": "no-managed-url-configured"}
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.post_promotion_priority_delta_review_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.post_promotion_priority_delta_review import build_post_promotion_priority_delta_review

    result = build_post_promotion_priority_delta_review(payload, limit=args.limit, fetch=fetch)
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "post-promotion-priority-delta-review",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.deltas,
            "url": _redact_url(args.url),
            "fetch_status": fetch.get("status") if isinstance(fetch, dict) else None,
            "review_candidate_count": summary.get("review_candidate_count"),
            "group_count": summary.get("group_count"),
            "recommended_count": summary.get("recommended_count"),
            "noop_count": summary.get("noop_count"),
            "wrote_local_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": result.get("managed_server_calls_made"),
            "exit_code": 0,
        },
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0


def post_promotion_policy_draft_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run local policy widen/rollback drafts from a post-promotion priority review report"
    )
    parser.add_argument(
        "priority_review",
        nargs="?",
        help="Post-promotion priority review JSON path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--widen-fraction",
        type=float,
        default=0.05,
        help="Maximum activation fraction delta for widen drafts, default: 0.05.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Minimum holdout fraction to preserve in widen drafts, default: 0.10.",
    )
    parser.add_argument(
        "--max-drafts",
        type=int,
        default=20,
        help="Maximum widen/rollback drafts to emit, default: 20.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print draft JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if not args.priority_review:
        result = {
            "schema": "agentflow.post_promotion_policy_draft_dry_run.v1",
            "ok": False,
            "status": "invalid",
            "error": {
                "type": "missing_input",
                "message": "provide a post-promotion priority review JSON path or '-' for stdin",
            },
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "wrote_local_policy_files": False,
        }
        _write_json(stderr, result)
        return 1

    try:
        report = _read_json_input(str(args.priority_review), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.post_promotion_policy_draft_dry_run_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.post_promotion_policy_drafts import build_post_promotion_policy_drafts

    result = build_post_promotion_policy_drafts(
        report,
        widen_fraction=args.widen_fraction,
        holdout_fraction=args.holdout_fraction,
        max_drafts=args.max_drafts,
    )
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "post-promotion-policy-draft-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.priority_review,
            "candidate_count": summary.get("candidate_count"),
            "draft_count": summary.get("draft_count"),
            "widen_draft_count": summary.get("widen_draft_count"),
            "rollback_draft_count": summary.get("rollback_draft_count"),
            "omitted_count": summary.get("omitted_count"),
            "wrote_local_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    if (result.get("error") or {}).get("type") in {"invalid_report", "raw_payload_rejected", "impact_gate_blocked"}:
        return 1
    return 0


def post_promotion_policy_draft_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Apply gated post-promotion local policy drafts to file-backed AgentFlow rule files"
    )
    parser.add_argument(
        "policy_draft_report",
        nargs="?",
        help="Post-promotion policy draft dry-run JSON path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local rule files, default: ~/.agentflow.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument("--max-apply", type=int, default=20, help="Maximum gated drafts to process, default: 20.")
    parser.add_argument("--dry-run", action="store_true", help="Stage and validate generated policy drafts without writing active rule files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if not args.policy_draft_report:
        result = {
            "schema": "agentflow.post_promotion_policy_draft_apply.v1",
            "ok": False,
            "status": "invalid",
            "error": {
                "type": "missing_input",
                "message": "provide a post-promotion policy draft dry-run JSON path or '-' for stdin",
            },
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "wrote_local_policy_files": False,
        }
        _write_json(stderr, result)
        return 1

    try:
        report = _read_json_input(str(args.policy_draft_report), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.post_promotion_policy_draft_apply_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.post_promotion_policy_apply import apply_post_promotion_policy_drafts

    result = asyncio.run(apply_post_promotion_policy_drafts(
        report,
        config_dir=args.config_dir,
        workspace=args.workspace,
        dry_run=bool(args.dry_run),
        max_apply=int(args.max_apply or 0),
    ))
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "post-promotion-policy-draft-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.policy_draft_report,
            "config_dir": args.config_dir,
            "workspace": args.workspace,
            "dry_run": bool(args.dry_run),
            "draft_count": summary.get("draft_count"),
            "processed_count": summary.get("processed_count"),
            "applied_count": summary.get("applied_count"),
            "blocked_count": summary.get("blocked_count"),
            "wrote_local_policy_files": bool(result.get("wrote_local_policy_files")),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1
