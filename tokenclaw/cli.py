from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from tokenclaw.cli_common import (
    default_config_dir as _default_config_dir,
    default_db_path as _default_db_path,
    is_loopback_url as _is_loopback_url,
    open_metadata_report_store_for_db as _open_metadata_report_store_for_db,
    open_store_for_db as _open_store_for_db,
    redact_secret as _redact_secret,
    write_json as _write_json,
)
from tokenclaw.env import env, env_float
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.managed_activation_proof import managed_activation_proof_cli
from tokenclaw.upstream_url import redact_url as _redact_url

from tokenclaw.cli_commands.policy_bundle import (
    POLICY_RELOAD_PATH,
    _attach_old_context_summary_lifecycle_feedback,
    _default_policy_reload_url,
    _old_context_summary_lifecycle_payload,
    _read_policy_json_arg,
    _validation_result_error,
    policy_diff_cli,
    policy_export_cli,
    policy_reload_cli,
    policy_review_cli,
    policy_validate_cli,
)
from tokenclaw.cli_commands.policy_workbench import (
    MANAGED_POLICY_API_KEY_ENV,
    POLICY_BUNDLE_RECOMMENDATION_URL_ENV,
    _managed_policy_auth,
    _managed_policy_query,
    _reload_policy_state_via_url,
    _write_policy_draft_apply_result,
    managed_activation_bundle_apply_cli,
    managed_activation_bundle_stage_cli,
    policy_apply_cli,
    policy_draft_apply_cli,
    policy_draft_stage_cli,
    policy_draft_validate_cli,
    policy_fetch_review_cli,
)


from tokenclaw.cli_commands.onboarding import (
    DEFAULT_STATS_URL,
    ONBOARDING_TARGETS,
    RUN_TARGETS,
    UNSUPPORTED_ONBOARDING_TARGETS,
    _CODEX_OPENAI_BASE_URL_RE,
    _activation_config_error_result,
    _activation_doctor_result,
    _activation_stats_result,
    _codex_openai_base_url_from_toml,
    _decode_toml_string,
    _doctor_activation_target,
    _doctor_claude_desktop_target,
    _doctor_claude_vscode_target,
    _doctor_codex_target,
    _doctor_provider_target,
    _env_file_value,
    _fetch_tokenclaw_stats,
    _profile_for_target,
    _selected_activation_targets,
    _target_activation_base,
    _write_activation_config_error,
    _write_activation_doctor_summary,
    _write_activation_stats_summary,
    _write_activation_summary,
    _write_doctor_summary,
    _write_savings_report_summary,
    _write_stats_summary,
    tokenclaw_cli as _onboarding_tokenclaw_cli,
)

from tokenclaw.cli_commands.optimization_reports import (
    PATTERN_ROLLOUT_ACTIONS_URL_ENV,
    OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV,
    SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV,
    openai_optimization_draft_dry_run_cli,
    openai_optimization_draft_apply_cli,
    codex_app_policy_dry_run_cli,
    managed_rollout_actions_review_cli,
    optimization_rollout_actions_review_cli,
    optimization_rollout_actions_apply_cli,
    scaffold_rollout_actions_review_cli,
    scaffold_rollout_actions_apply_cli,
    managed_rollout_actions_apply_cli,
    managed_rollout_actions_dry_run_cli,
    old_context_summary_dry_run_cli,
    old_context_summary_impact_cli,
    old_context_summary_quality_gate_cli,
    old_context_summary_rollout_actions_review_cli,
    old_context_summary_rollout_actions_apply_cli,
    old_context_summary_rollout_actions_dry_run_cli,
    old_context_summary_rollout_actions_impact_cli,
    managed_rollout_actions_impact_cli,
    codex_diagnose_cli,
    codex_canary_impact_cli,
    openai_scoreboard_cli,
    openai_routing_report_cli,
    routing_coverage_report_cli,
    openai_routing_narrow_canary_review_cli,
    anthropic_routing_canary_stage_cli,
    openai_old_context_summary_report_cli,
    openai_cache_replay_report_cli,
    openai_cache_replay_blocker_outcomes_cli,
    crunch_blocker_outcomes_cli,
    activation_safety_stop_burndown_cli,
    anthropic_routing_safety_stop_unblock_drill_cli,
    optimization_action_ledger_cli,
    optimization_coordinator_dry_run_cli,
    provider_tool_adoption_report_cli,
    repeated_scaffold_opportunity_cli,
    instruction_dedup_opportunity_cli,
    instruction_dedup_dry_run_cli,
    instruction_dedup_impact_cli,
    terminal_output_compaction_opportunity_cli,
    codex_terminal_transcript_opportunity_cli,
    anthropic_thinking_compaction_opportunity_cli,
    anthropic_thinking_compaction_impact_cli,
    local_compaction_canary_ramp_cli,
    anthropic_thinking_compaction_dry_run_cli,
    codex_terminal_transcript_dry_run_cli,
    codex_terminal_transcript_impact_cli,
    terminal_output_compaction_dry_run_cli,
    terminal_output_compaction_impact_cli,
    repeated_scaffold_impact_cli,
    repeated_scaffold_activation_cli,
    openai_cache_replay_impact_cli,
    openai_cache_replay_readiness_cli,
    local_promotion_candidates_cli,
    crunch_promotion_draft_dry_run_cli,
    cache_promotion_draft_dry_run_cli,
    routing_promotion_draft_dry_run_cli,
    openai_cache_replay_apply_cli,
    openai_cache_replay_dry_run_cli,
    openai_old_context_summary_dry_run_cli,
    openai_canary_impact_cli,
    claude_canary_impact_cli,
    anthropic_routing_lifecycle_report_cli,
    claude_canary_actions_cli,
    claude_canary_actions_apply_cli,
    routing_canary_promote_cli,
    routing_experiment_report_cli,
    routing_promotion_draft_stage_cli,
    phase_routing_report_cli,
    session_phase_memory_cli,
    cache_replayability_report_cli,
    request_shape_rollups_cli,
    request_shape_crunch_canary_impact_cli,
    request_shape_crunch_policy_decision_cli,
    request_shape_crunch_canary_stage_cli,
    request_shape_cache_replay_canary_stage_cli,
    request_shape_cache_replay_evidence_cli,
    request_shape_cache_replay_policy_decision_cli,
    managed_recommendation_handoff_cli,
    local_activation_outcome_summary_cli,
    managed_activation_bundle_apply_outcomes_cli,
    cache_replay_cohorts_cli,
    cache_smoke_diagnostic_cli,
    cache_replay_dry_run_cli,
    managed_pattern_rollups_cli,
    optimization_eval_plan_cli,
    optimization_shadow_eval_cli,
    optimization_eval_queue_cli,
    optimization_promotion_report_cli,
    optimization_promotion_actions_cli,
    optimization_promotion_blocker_review_cli,
    optimization_promotion_canary_apply_cli,
    optimization_promotion_impact_cli,
    post_promotion_priority_delta_review_cli,
    post_promotion_policy_draft_dry_run_cli,
    post_promotion_policy_draft_apply_cli,
)


def _env_enabled(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_cli_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)



def policy_rollback_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Rollback local TokenClaw policy YAML files from apply backups")
    parser.add_argument(
        "--apply-id",
        help="Rollback the exact policy workbench apply transaction with this apply ID.",
    )
    parser.add_argument(
        "--config-dir",
        default=env("TOKENCLAW_POLICY_CONFIG_DIR", _default_config_dir()),
        help="Directory for local rule files, default: ~/.tokenclaw",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Rollback only one policy section. Repeat to rollback multiple sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the backups that would be restored without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow CLI-only partial recovery when an apply event is missing or backup metadata is incomplete.",
    )
    parser.add_argument(
        "--reload-url",
        default=env("TOKENCLAW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL for --apply-id rollback, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=env_float("TOKENCLAW_ADMIN_TIMEOUT", 10.0),
        help="HTTP timeout in seconds for the loopback reload call, default: 10.",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow posting reload to a non-loopback URL. Use only for explicit trusted tunnels.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print rollback JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    if args.apply_id:
        from tokenclaw.policy_events import log_policy_event
        from tokenclaw.policy_workbench import POLICY_DRAFT_ROLLBACK_SCHEMA, rollback_policy_apply

        if not args.dry_run and not args.allow_non_loopback and not _is_loopback_url(args.reload_url):
            result = {
                "schema": POLICY_DRAFT_ROLLBACK_SCHEMA,
                "ok": False,
                "status": "blocked",
                "apply_id": args.apply_id,
                "backup_id": args.apply_id,
                "dry_run": bool(args.dry_run),
                "force": bool(args.force),
                "config_dir": args.config_dir,
                "manifest_source": None,
                "apply_event_found": None,
                "requested_sections": args.section or ["routing", "crunch", "cache", "routing_experiments", "codex_app"],
                "restored_sections": [],
                "skipped_sections": [],
                "files": [],
                "current_backups": [],
                "reloaded_modules": False,
                "reload": None,
                "verification": None,
                "privacy": {"provider_calls_made": False, "managed_server_calls_made": False, "loopback_admin_calls_made": False},
                "error": {
                    "type": "unsafe_url",
                    "message": "policy apply rollback only posts reloads to loopback URLs unless --allow-non-loopback is set",
                    "url": args.reload_url,
                },
            }
            log_policy_event(
                "rollback",
                ok=False,
                details={"source": "cli", "apply_id": args.apply_id, "error_type": "unsafe_url", "exit_code": 2},
            )
            _write_policy_rollback_result(stdout, result, pretty=args.pretty)
            return 2

        async def reload_state() -> dict[str, Any]:
            return await _reload_policy_state_via_url(args.reload_url, timeout=args.timeout)

        result = asyncio.run(rollback_policy_apply(
            args.apply_id,
            config_dir=args.config_dir,
            sections=args.section,
            dry_run=args.dry_run,
            force=args.force,
            reload_policy_state=reload_state,
            event_source="cli",
            loopback_admin_calls_made=not args.dry_run,
        ))
        _write_policy_rollback_result(stdout, result, pretty=args.pretty)
        return 0 if result.get("ok") else 1

    from tokenclaw.policy_bundle import rollback_policy_files
    from tokenclaw.policy_events import log_policy_event

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
            "codex_app": result.get("codex_app"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_rollback_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def managed_feedback_status_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from tokenclaw.optimization.feedback import managed_feedback_status_cli as _managed_feedback_status_cli

    return _managed_feedback_status_cli(argv, stdout=stdout)


def managed_feedback_flush_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from tokenclaw.optimization.feedback import managed_feedback_flush_cli as _managed_feedback_flush_cli

    return _managed_feedback_flush_cli(argv, stdout=stdout)


def sqlite_maintenance_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run local SQLite retention maintenance for TokenClaw metadata")
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="TokenClaw SQLite DB path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Retention window in days. Defaults to TOKENCLAW_SQLITE_RETENTION_DAYS or 7.",
    )
    parser.add_argument("--disable-retention", action="store_true", help="Record disabled maintenance without deleting rows.")
    parser.add_argument("--dry-run", action="store_true", help="Report rows that would be purged without deleting them.")
    parser.add_argument("--no-analyze", action="store_true", help="Skip ANALYZE after a purge.")
    parser.add_argument("--no-optimize", action="store_true", help="Skip PRAGMA optimize after maintenance.")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after the purge. Use only during an explicit maintenance window.")
    args = parser.parse_args(argv)
    store = _open_store_for_db(str(args.db))
    try:
        result = store.run_sqlite_maintenance(
            retention_days=None if args.retention_days is None and not args.disable_retention else (0 if args.disable_retention else args.retention_days),
            dry_run=bool(args.dry_run),
            analyze=not bool(args.no_analyze),
            optimize=not bool(args.no_optimize),
            vacuum=bool(args.vacuum),
        )
    finally:
        try:
            store.conn.close()
        except Exception:
            pass
    _write_json(stdout or sys.stdout, result)
    return 0


def db_adopt_legacy_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from tokenclaw.db_adoption import adopt_legacy_sqlite_evidence

    parser = argparse.ArgumentParser(description="Adopt local legacy AgentFlow SQLite evidence into the canonical TokenClaw DB")
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="Canonical TokenClaw SQLite DB path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
    )
    parser.add_argument(
        "--from",
        dest="legacy_db",
        default=None,
        help="Legacy AgentFlow SQLite DB path, default: sibling agentflow.sqlite3 next to the canonical DB.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report rows that would be adopted without writing them.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    result = adopt_legacy_sqlite_evidence(
        canonical_db=str(args.db),
        legacy_db=str(args.legacy_db) if args.legacy_db else None,
        dry_run=bool(args.dry_run),
    )
    stream = stdout or sys.stdout
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def savings_loop_bottlenecks_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report local savings-loop stall bottlenecks from metadata-only evidence")
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="Canonical TokenClaw SQLite DB path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
    )
    parser.add_argument(
        "--legacy-db",
        default=None,
        help="Legacy AgentFlow SQLite DB path, default: sibling agentflow.sqlite3 next to the canonical DB.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("TOKENCLAW_CONFIG_DIR") or os.getenv("TOKENCLAW_POLICY_CONFIG_DIR", str(Path.home() / ".tokenclaw")),
        help="Directory containing local policy rule files, default: TOKENCLAW_CONFIG_DIR, TOKENCLAW_POLICY_CONFIG_DIR, or ~/.tokenclaw.",
    )
    parser.add_argument("--active-window-hours", type=float, default=24.0, help="Source traffic window, default: 24.")
    parser.add_argument("--activation-min-source-rows", type=int, default=10, help="Minimum source rows before activation is considered alive, default: 10.")
    parser.add_argument("--rollup-max-age-hours", type=float, default=72.0, help="Maximum rollup/snapshot evidence age, default: 72.")
    parser.add_argument("--policy-max-age-hours", type=float, default=72.0, help="Maximum staged policy evidence age, default: 72.")
    parser.add_argument("--policy-scan-limit", type=int, default=1000, help="Bounded recent metadata rows to inspect for policy traffic, default: 1000.")
    parser.add_argument(
        "--adopt-legacy-preflight",
        action="store_true",
        help="Before reporting, adopt richer sibling legacy SQLite metadata into the canonical DB.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    from tokenclaw.savings_loop_bottlenecks import build_savings_loop_bottlenecks_report

    stream = stdout or sys.stdout
    store = _open_metadata_report_store_for_db(str(args.db))
    try:
        result = build_savings_loop_bottlenecks_report(
            store,
            db_path=str(args.db),
            legacy_db=str(args.legacy_db) if args.legacy_db else None,
            config_dir=args.config_dir,
            active_window_hours=args.active_window_hours,
            activation_min_source_rows=args.activation_min_source_rows,
            rollup_max_age_hours=args.rollup_max_age_hours,
            policy_max_age_hours=args.policy_max_age_hours,
            policy_scan_limit=args.policy_scan_limit,
            adopt_legacy_preflight=bool(args.adopt_legacy_preflight),
        )
    finally:
        store.conn.close()
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0




def _write_policy_rollback_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def orchestrator_research_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a sanitized data-backed AgentFlow research-mode backlog plan")
    parser.add_argument(
        "--issues-json",
        required=True,
        help="Path to a GitHub issue-list JSON array. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--stats-json",
        help="Optional path to local stats JSON from the dashboard/API.",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Recent orchestrator log path to inspect. Can be repeated.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(os.getenv("TOKENCLAW_RESEARCH_BACKLOG_THRESHOLD", "3")),
        help="Minimum status:ready actionable issue count before research mode is skipped.",
    )
    parser.add_argument(
        "--trusted-author",
        default=os.getenv("TOKENCLAW_GITHUB_TRUSTED_AUTHOR", "lutzkuen"),
        help="Only issues from this GitHub author are considered unattended-actionable.",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=14,
        help="Age in days before a blocked issue is treated as stale for research comments.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.orchestrator_research import build_research_plan, load_json_file, write_json

    try:
        if args.issues_json == "-":
            issues = json.loads(sys.stdin.read())
        else:
            issues = load_json_file(args.issues_json)
        stats = load_json_file(args.stats_json) if args.stats_json else None
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1

    if not isinstance(issues, list):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_issues_json", "message": "issues JSON must be an array"}})
        return 1
    if stats is not None and not isinstance(stats, dict):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_stats_json", "message": "stats JSON must be an object"}})
        return 1
    issues = _attach_recent_closed_github_issues_for_research(issues, trusted_author=args.trusted_author)
    stats = _attach_request_shape_rollups_for_research(stats)

    plan = build_research_plan(
        issues=issues,
        stats=stats,
        log_sources=args.log,
        threshold=args.threshold,
        trusted_author=args.trusted_author,
        stale_days=args.stale_days,
    )
    write_json(stdout, plan, pretty=args.pretty)
    return 0


def _issue_repo_for_research(issue: Any) -> str | None:
    if not isinstance(issue, dict):
        return None
    repo = str(issue.get("repo") or issue.get("repository") or "").strip()
    if repo.count("/") == 1:
        return repo
    url = str(issue.get("url") or issue.get("html_url") or "").strip()
    marker = "github.com/"
    if marker in url:
        parts = url.split(marker, 1)[1].split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
    return None


def _attach_recent_closed_github_issues_for_research(
    issues: list[Any],
    *,
    trusted_author: str,
) -> list[Any]:
    if os.getenv("TOKENCLAW_RESEARCH_FETCH_CLOSED_ISSUES", "1").strip().lower() in {"0", "false", "no", "off"}:
        return issues
    gh = shutil.which("gh")
    if not gh:
        return issues
    repos = sorted({repo for repo in (_issue_repo_for_research(issue) for issue in issues) if repo})
    if not repos:
        env_repos = os.getenv("TOKENCLAW_RESEARCH_GITHUB_REPOS", "")
        repos = sorted({repo.strip() for repo in env_repos.split(",") if repo.strip().count("/") == 1})
    if not repos:
        return issues
    limit = os.getenv("TOKENCLAW_RESEARCH_CLOSED_ISSUE_LIMIT", "200")
    try:
        limit_value = str(max(1, min(200, int(limit))))
    except ValueError:
        limit_value = "50"
    merged = list(issues)
    seen = {
        (_issue_repo_for_research(issue) or "", str(issue.get("number") or ""))
        for issue in issues
        if isinstance(issue, dict)
    }
    for repo in repos:
        cmd = [
            gh,
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "closed",
            "--author",
            trusted_author,
            "--limit",
            limit_value,
            "--json",
            "number,title,closedAt,updatedAt,labels,url",
        ]
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode != 0:
            continue
        try:
            rows = json.loads(completed.stdout)
        except (TypeError, ValueError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = (repo, str(row.get("number") or ""))
            if key in seen:
                continue
            seen.add(key)
            enriched = dict(row)
            enriched["repo"] = repo
            enriched["state"] = "CLOSED"
            enriched["author"] = {"login": trusted_author}
            merged.append(enriched)
    return merged


def _attach_request_shape_rollups_for_research(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if stats is None:
        stats = {}
    if not isinstance(stats, dict):
        return stats

    db_arg, db_source = _research_db_source(stats)
    if db_source == "stats-without-db":
        return stats
    if not db_arg:
        enriched = dict(stats)
        enriched["request_shape_research_source"] = _research_db_source_report(
            db_arg=None,
            source=db_source,
            status="unavailable",
            reason="canonical-tokenclaw-db-unconfigured",
        )
        enriched.setdefault(
            "request_shape_rollups",
            _request_shape_source_unavailable_report("canonical-tokenclaw-db-unconfigured"),
        )
        return enriched
    db_available = db_arg.startswith(("postgresql://", "postgres://")) or Path(db_arg).expanduser().exists()
    if not db_available:
        enriched = dict(stats)
        enriched["request_shape_research_source"] = _research_db_source_report(
            db_arg=db_arg,
            source=db_source,
            status="unavailable",
            reason="canonical-tokenclaw-db-missing",
        )
        enriched.setdefault(
            "request_shape_rollups",
            _request_shape_source_unavailable_report("canonical-tokenclaw-db-missing"),
        )
        return enriched

    try:
        limit = max(1, min(int(os.getenv("TOKENCLAW_RESEARCH_REQUEST_SHAPE_LIMIT", "1000")), 10_000))
    except ValueError:
        limit = 1000

    from tokenclaw.request_shape_rollups import (
        build_request_shape_cache_replay_evidence_report,
        build_request_shape_cache_replay_policy_decision_report,
        latest_request_shape_rollup_snapshot_report,
        build_request_shape_rollups_report,
    )
    from tokenclaw.cache_smoke import build_isolated_cache_replay_hit_recovery_smoke

    needs_rollups = not any(
        isinstance(stats.get(key), dict)
        for key in ("request_shape_rollups", "request_shape_rollup_report", "request_shape_rollup_candidates_report")
    )
    needs_cache_replay_evidence = not isinstance(stats.get("request_shape_cache_replay_evidence"), dict)
    needs_cache_replay_policy_decision = not isinstance(stats.get("request_shape_cache_replay_policy_decision"), dict)
    needs_managed_preview_outcomes = not any(
        isinstance(stats.get(key), dict)
        for key in (
            "managed_activation_preview_outcomes",
            "managed_preview_outcomes",
            "managed_activation_preview_outcome_summary",
        )
    )
    if (
        not needs_rollups
        and not needs_cache_replay_evidence
        and not needs_cache_replay_policy_decision
        and not needs_managed_preview_outcomes
    ):
        return stats

    enriched = dict(stats)
    enriched["db"] = db_arg
    enriched["request_shape_research_source"] = _research_db_source_report(
        db_arg=db_arg,
        source=db_source,
        status="available",
        reason="canonical-tokenclaw-db-selected",
    )
    store = _open_store_for_db(db_arg)
    try:
        managed_preview_outcomes = (
            enriched.get("managed_activation_preview_outcomes")
            if isinstance(enriched.get("managed_activation_preview_outcomes"), dict)
            else enriched.get("managed_preview_outcomes")
            if isinstance(enriched.get("managed_preview_outcomes"), dict)
            else None
        )
        if needs_managed_preview_outcomes:
            from tokenclaw.managed_activation_preview_outcomes import (
                build_managed_activation_preview_outcomes_report,
            )

            managed_preview_outcomes = build_managed_activation_preview_outcomes_report(
                store,
                limit=limit,
            )
            enriched["managed_activation_preview_outcomes"] = managed_preview_outcomes
        if needs_rollups:
            rollups_report = build_request_shape_rollups_report(
                store,
                limit=limit,
                persist=True,
                run_id="orchestrator-research-dry-run",
                managed_preview_outcomes=managed_preview_outcomes,
            )
            if int((rollups_report.get("summary") or {}).get("rows_considered") or 0) <= 0:
                try:
                    max_age = float(os.getenv("TOKENCLAW_RESEARCH_REQUEST_SHAPE_SNAPSHOT_MAX_AGE_HOURS", "72"))
                except ValueError:
                    max_age = 72.0
                snapshot_report = latest_request_shape_rollup_snapshot_report(
                    store,
                    max_age_hours=max_age,
                )
                if snapshot_report is not None:
                    rollups_report = snapshot_report
            enriched["request_shape_rollups"] = rollups_report
            if isinstance(rollups_report.get("rollup_snapshot"), dict):
                enriched["request_shape_rollup_snapshot"] = rollups_report["rollup_snapshot"]
        cache_replay_evidence = (
            enriched.get("request_shape_cache_replay_evidence")
            if isinstance(enriched.get("request_shape_cache_replay_evidence"), dict)
            else None
        )
        if needs_cache_replay_evidence:
            rules_path = (
                Path(env("TOKENCLAW_CACHE_CANARY_POLICY")).expanduser()
                if env("TOKENCLAW_CACHE_CANARY_POLICY")
                else Path(_default_config_dir()).expanduser()
                / "cache_canary_policy.yaml"
            )
            cache_replay_evidence = build_request_shape_cache_replay_evidence_report(
                store,
                rules_path=rules_path,
                limit=max(limit, 1000),
            )
            enriched["request_shape_cache_replay_evidence"] = cache_replay_evidence
        if needs_cache_replay_policy_decision and isinstance(cache_replay_evidence, dict):
            enriched["request_shape_cache_replay_policy_decision"] = build_request_shape_cache_replay_policy_decision_report(
                cache_replay_evidence,
                hit_recovery_report=build_isolated_cache_replay_hit_recovery_smoke(),
            )
        preview_outcomes_empty = (
            isinstance(managed_preview_outcomes, dict)
            and int((managed_preview_outcomes.get("summary") or {}).get("stored_preview_outcome_count") or 0) == 0
        )
        if preview_outcomes_empty:
            from tokenclaw.managed_activation_preview_outcomes import (
                persist_unavailable_managed_activation_preview_outcomes,
            )
            from tokenclaw.orchestrator_research import build_local_activation_next_action_queue

            activation_queue = (
                enriched.get("local_activation_next_action_queue")
                if isinstance(enriched.get("local_activation_next_action_queue"), dict)
                else build_local_activation_next_action_queue(enriched)
            )
            has_activation_rows = (
                isinstance(activation_queue, dict)
                and bool(activation_queue.get("entries") or activation_queue.get("successor_actions"))
            )
            if has_activation_rows:
                enriched["local_activation_next_action_queue"] = activation_queue
                managed_preview_outcomes = persist_unavailable_managed_activation_preview_outcomes(
                    store,
                    activation_queue,
                    reason="managed-preview-refresh-not-configured",
                )
                enriched["managed_activation_preview_outcomes"] = managed_preview_outcomes
    finally:
        try:
            store.conn.close()
        except Exception:
            pass

    return enriched


def _research_db_source(stats: dict[str, Any]) -> tuple[str | None, str]:
    db = stats.get("db")
    if isinstance(db, str) and db.strip():
        return db.strip(), "stats-db"
    if not _stats_need_canonical_db_fallback(stats):
        return None, "stats-without-db"
    configured = env("TOKENCLAW_DATABASE_URL") or env("TOKENCLAW_DB")
    if isinstance(configured, str) and configured.strip():
        return configured.strip(), "environment"
    default_db = _default_db_path()
    if isinstance(default_db, str) and default_db.strip():
        return default_db.strip(), "default-tokenclaw-db"
    return None, "unconfigured"


def _stats_need_canonical_db_fallback(stats: dict[str, Any]) -> bool:
    if not stats:
        return True
    status = str(stats.get("status") or stats.get("stats") or stats.get("error") or "").strip().lower()
    if status in {"unavailable", "stats-unavailable", "error"}:
        return True
    if stats.get("stats_unavailable") is True:
        return True
    return False


def _research_db_path_class(db_arg: str | None) -> str:
    if not db_arg:
        return "unknown"
    if db_arg.startswith(("postgresql://", "postgres://")):
        return "configured-postgres-url"
    expanded = os.path.abspath(os.path.expanduser(db_arg.removeprefix("sqlite:///")))
    home = os.path.abspath(os.path.expanduser("~"))
    if expanded.startswith(os.path.join(home, ".tokenclaw") + os.sep):
        return "local-tokenclaw-home"
    if expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        return "local-temp"
    return "local-path"


def _research_db_source_report(
    *,
    db_arg: str | None,
    source: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.request_shape_research_source.v1",
        "status": status,
        "reason": reason,
        "source": source,
        "db_path_included": False,
        "db_path_class": _research_db_path_class(db_arg),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "absolute_paths_included": False,
            "provider_bodies_included": False,
            "raw_prompts_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
        },
    }


def _request_shape_source_unavailable_report(reason: str) -> dict[str, Any]:
    privacy = {
        "metadata_only": True,
        "aggregate_only": True,
        "absolute_paths_included": False,
        "provider_bodies_included": False,
        "raw_prompts_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    follow_up = {
        "schema": "tokenclaw.request_shape_follow_up_candidates.v1",
        "status": "source-unavailable",
        "summary": {
            "rows_considered": 0,
            "rollup_count": 0,
            "ranked_candidate_count": 0,
            "top_next_action": "restore-tokenclaw-db-source",
            "top_local_action_family": "cohort-ranking",
            "top_readiness_state": "blocked",
            "no_source_traffic_reason": reason,
            "provider_calls_made": 0,
            "managed_server_calls_made": 0,
            "policy_files_written": False,
        },
        "top_candidate": None,
        "top_blocker_cohort": None,
        "candidates": [],
        "blocker_cohorts": [],
        "missing_measurements": [reason],
        "privacy": privacy,
    }
    return {
        "schema": "tokenclaw.request_shape_rollups.v1",
        "status": "source-unavailable",
        "summary": {
            "rows_considered": 0,
            "rollup_count": 0,
            "follow_up_candidate_count": 0,
            "top_next_action": "restore-tokenclaw-db-source",
            "top_local_action_family": "cohort-ranking",
            "no_source_traffic_reason": reason,
        },
        "follow_up_candidates": follow_up,
        "rollups": [],
        "privacy": privacy,
    }


def evidence_to_activation_burndown_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a metadata-only AgentFlow evidence-to-activation burn-down report")
    parser.add_argument(
        "--plan-json",
        required=True,
        help="Path to an AgentFlow orchestrator research plan JSON.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.orchestrator_research import build_evidence_to_activation_burndown, load_json_file, write_json

    try:
        plan = load_json_file(args.plan_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(plan, dict):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_plan_json", "message": "plan JSON must be an object"}})
        return 1

    report = build_evidence_to_activation_burndown(plan)
    write_json(stdout, report, pretty=args.pretty)
    return 0


def activation_burndown_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the unified metadata-only AgentFlow local activation burndown report")
    parser.add_argument(
        "--plan-json",
        required=True,
        help="Path to an AgentFlow orchestrator research plan JSON.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.orchestrator_research import build_activation_burndown_report, load_json_file, write_json

    try:
        plan = load_json_file(args.plan_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(plan, dict):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_plan_json", "message": "plan JSON must be an object"}})
        return 1

    report = build_activation_burndown_report(plan)
    write_json(stdout, report, pretty=args.pretty)
    return 0


def local_activation_executor_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a dry-run metadata-only AgentFlow local activation executor plan")
    parser.add_argument(
        "--plan-json",
        required=True,
        help="Path to an AgentFlow orchestrator research plan, activation queue, ledger, or burndown JSON.",
    )
    parser.add_argument(
        "--managed-handoff",
        action="store_true",
        help="Emit feature-only managed handoff rows for the local executor outcomes.",
    )
    parser.add_argument(
        "--review-bundle",
        action="store_true",
        help="Emit exactly one review-only local executor bundle for a ranked activation action.",
    )
    parser.add_argument(
        "--bundle-rank",
        type=int,
        default=1,
        help="Activation action rank to emit with --review-bundle, default: 1.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.local_activation_executor import (
        build_local_activation_executor_bundle,
        build_local_activation_executor_managed_handoff,
        build_local_activation_executor_plan,
    )
    from tokenclaw.orchestrator_research import load_json_file, write_json

    try:
        plan = load_json_file(args.plan_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(plan, dict):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_plan_json", "message": "plan JSON must be an object"}})
        return 1

    if args.review_bundle:
        report = build_local_activation_executor_bundle(plan, action_rank=args.bundle_rank)
    elif args.managed_handoff:
        report = build_local_activation_executor_managed_handoff(plan)
    else:
        report = build_local_activation_executor_plan(plan)
    write_json(stdout, report, pretty=args.pretty)
    return 0


def _attach_managed_activation_preview_outcome_persistence(
    result: dict[str, Any],
    *,
    db: str,
    stale_after_hours: float,
) -> dict[str, Any]:
    from tokenclaw.managed_activation_preview_outcomes import (
        persist_managed_activation_preview_outcomes,
    )

    store = _open_store_for_db(str(db))
    try:
        return persist_managed_activation_preview_outcomes(
            store,
            result,
            stale_after_hours=float(stale_after_hours),
        )
    finally:
        store.conn.close()


def _managed_routing_pathway_outcomes_url(
    *,
    explicit_url: str,
    managed_preview_url: str,
) -> str:
    url = str(explicit_url or "").strip()
    if url:
        return url
    preview_url = str(managed_preview_url or "").strip()
    if not preview_url:
        return ""
    parts = urlsplit(preview_url)
    if not parts.scheme or not parts.netloc:
        return ""
    path = parts.path or ""
    if "/v1/" in path:
        prefix = path.split("/v1/", 1)[0]
        next_path = f"{prefix}/v1/managed-routing-pathway-outcomes"
    else:
        base = path.rsplit("/", 1)[0] if "/" in path.rstrip("/") else ""
        next_path = f"{base}/managed-routing-pathway-outcomes"
    return urlunsplit((parts.scheme, parts.netloc, next_path, "", ""))


def _routing_pathway_outcome_batch_result(
    *,
    status: str,
    reason: str,
    outcome_count: int = 0,
    url: str | None = None,
    managed_server_calls_made: bool = False,
    status_code: int | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "tokenclaw.routing_pathway_outcome_batch_preflight.v1",
        "status": status,
        "reason": reason,
        "outcome_count": int(outcome_count),
        "submitted_outcome_count": int(outcome_count) if status == "submitted" else 0,
        "managed_dependency": "optional",
        "review_only": True,
        "feature_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": bool(managed_server_calls_made),
        "privacy": {
            "schema": "tokenclaw.routing_pathway_outcome_batch_preflight_privacy.v1",
            "feature_only": True,
            "metadata_only": True,
            "aggregate_only": True,
            "review_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_response_bodies_included": False,
            "provider_bodies_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "individual_candidate_ids_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": False,
            "policy_files_written": False,
            "managed_server_calls_made": bool(managed_server_calls_made),
        },
    }
    if url:
        result["url"] = _redact_url(url)
    if status_code is not None:
        result["status_code"] = int(status_code)
    if latency_ms is not None:
        result["latency_ms"] = int(latency_ms)
    if error:
        result["error"] = str(error)[:500]
    return result


def _managed_routing_pathway_outcome_ingest_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.managed_routing_pathway_outcomes.v1",
        "generated_at": report.get("generated_at"),
        "status": report.get("status") or "tracked",
        "read_only": True,
        "review_only": True,
        "managed_dependency": "optional",
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
        "source_schema": report.get("schema"),
        "summary": report.get("summary") or {},
        "outcomes": report.get("outcomes") or [],
        "privacy": report.get("privacy") or {},
        "egress_guard": report.get("egress_guard") or {},
    }


def _submit_routing_pathway_outcome_batch(
    *,
    source_json: str,
    db: str,
    limit: int,
    stale_after_hours: float,
    managed_preview_url: str,
    managed_routing_pathway_outcomes_url: str,
    allow_insecure_managed_url: bool,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    source_path = str(source_json or "").strip()
    if not source_path:
        return _routing_pathway_outcome_batch_result(
            status="skipped",
            reason="routing-pathway-outcome-batch-input-not-configured",
        )

    from tokenclaw.managed_routing_pathway_outcomes import (
        build_local_routing_pathway_outcome_feedback,
    )
    from tokenclaw.orchestrator_research import load_json_file

    try:
        source = load_json_file(source_path)
    except (OSError, ValueError, TypeError) as exc:
        return _routing_pathway_outcome_batch_result(
            status="error",
            reason="routing-pathway-outcome-batch-input-read-failed",
            error=f"{exc.__class__.__name__}: {exc}",
        )
    if not isinstance(source, dict):
        return _routing_pathway_outcome_batch_result(
            status="error",
            reason="routing-pathway-outcome-batch-input-not-object",
        )

    store = _open_store_for_db(str(db))
    try:
        report = build_local_routing_pathway_outcome_feedback(
            store,
            source,
            limit=int(limit),
            stale_after_hours=float(stale_after_hours),
        )
    finally:
        store.conn.close()

    outcome_count = len([row for row in report.get("outcomes") or [] if isinstance(row, dict)])
    if bool((report.get("egress_guard") or {}).get("blocked")):
        return _routing_pathway_outcome_batch_result(
            status="blocked",
            reason="unsafe-routing-pathway-outcome-batch",
            outcome_count=outcome_count,
        )
    if outcome_count <= 0:
        return _routing_pathway_outcome_batch_result(
            status="no-data",
            reason="routing-pathway-outcome-batch-empty",
            outcome_count=0,
        )

    url = _managed_routing_pathway_outcomes_url(
        explicit_url=managed_routing_pathway_outcomes_url,
        managed_preview_url=managed_preview_url,
    )
    if not url:
        return _routing_pathway_outcome_batch_result(
            status="skipped",
            reason="managed-routing-pathway-outcomes-url-not-configured",
            outcome_count=outcome_count,
        )
    if url.startswith("http://") and not _is_loopback_url(url) and not allow_insecure_managed_url:
        return _routing_pathway_outcome_batch_result(
            status="blocked",
            reason="insecure-managed-routing-pathway-outcomes-url",
            outcome_count=outcome_count,
            url=url,
        )

    payload = _managed_routing_pathway_outcome_ingest_payload(report)
    violations = managed_egress_violations(payload)
    if violations:
        return _routing_pathway_outcome_batch_result(
            status="blocked",
            reason="unsafe-routing-pathway-outcome-batch",
            outcome_count=outcome_count,
            url=url,
            error=f"egress_violation_count={len(violations)}",
        )

    started = time.time()
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=float(timeout))
        latency_ms = int((time.time() - started) * 1000)
        if response.status_code >= 400:
            return _routing_pathway_outcome_batch_result(
                status="error",
                reason="managed-routing-pathway-outcomes-server-error",
                outcome_count=outcome_count,
                url=url,
                managed_server_calls_made=True,
                status_code=response.status_code,
                latency_ms=latency_ms,
                error=response.text,
            )
        return _routing_pathway_outcome_batch_result(
            status="submitted",
            reason="managed-routing-pathway-outcome-batch-submitted",
            outcome_count=outcome_count,
            url=url,
            managed_server_calls_made=True,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
    except httpx.HTTPError as exc:
        return _routing_pathway_outcome_batch_result(
            status="error",
            reason=exc.__class__.__name__,
            outcome_count=outcome_count,
            url=url,
            error=str(exc),
        )


def _attach_routing_pathway_outcome_batch(
    result: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    result["routing_pathway_outcome_batch"] = batch
    summary = result.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["routing_pathway_outcome_batch_status"] = batch.get("status")
        summary["routing_pathway_outcome_batch_reason"] = batch.get("reason")
        summary["routing_pathway_outcome_batch_outcome_count"] = int(batch.get("outcome_count") or 0)
        summary["routing_pathway_outcome_batch_submitted_count"] = int(batch.get("submitted_outcome_count") or 0)
    return result


def managed_activation_preview_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description="Opt in to a feature-only managed preview for local activation executor handoff rows"
    )
    parser.add_argument(
        "--plan-json",
        required=True,
        help="Path to an AgentFlow orchestrator research plan, activation queue, ledger, burndown, or handoff JSON.",
    )
    parser.add_argument(
        "--managed-preview-url",
        default=os.getenv("TOKENCLAW_MANAGED_ACTIVATION_PREVIEW_URL", ""),
        help="Full managed preview endpoint URL. If omitted, no managed server call is made.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("TOKENCLAW_MANAGED_ACTIVATION_PREVIEW_TIMEOUT", "10")),
        help="HTTP timeout in seconds for the opt-in managed preview call, default: 10.",
    )
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing a bearer token, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--allow-insecure-managed-url",
        action="store_true",
        help="Allow non-loopback HTTP managed preview URLs. HTTPS and loopback HTTP are allowed by default.",
    )
    parser.add_argument(
        "--persist-outcomes",
        action="store_true",
        default=_env_enabled("TOKENCLAW_MANAGED_ACTIVATION_PREVIEW_PERSIST_OUTCOMES", False),
        help="Persist sanitized managed preview decisions as review-only local outcomes.",
    )
    parser.add_argument(
        "--top-preview-successors",
        type=int,
        default=int(os.getenv("TOKENCLAW_MANAGED_ACTIVATION_PREVIEW_TOP_SUCCESSORS", "0")),
        help=(
            "Submit only the top N preview-required, unverified activation successors. "
            "Default 0 keeps the full legacy handoff batch."
        ),
    )
    parser.add_argument(
        "--routing-pathway-outcomes-json",
        default=os.getenv("TOKENCLAW_ROUTING_PATHWAY_OUTCOMES_JSON", ""),
        help=(
            "Optional managed policy decision, routing_pathway_matrix, or pathway candidate JSON "
            "used to submit local routing pathway outcome feedback before the activation preview."
        ),
    )
    parser.add_argument(
        "--managed-routing-pathway-outcomes-url",
        default=os.getenv("TOKENCLAW_MANAGED_ROUTING_PATHWAY_OUTCOMES_URL", ""),
        help=(
            "Managed routing pathway outcome ingest endpoint. If omitted, it is derived from "
            "--managed-preview-url when possible."
        ),
    )
    parser.add_argument(
        "--routing-pathway-outcome-limit",
        type=int,
        default=int(os.getenv("TOKENCLAW_ROUTING_PATHWAY_OUTCOME_LIMIT", "1000")),
        help="Local routing pathway evidence rows to inspect before preview, default: 1000.",
    )
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="TokenClaw database URL or SQLite path for --persist-outcomes.",
    )
    parser.add_argument(
        "--preview-stale-after-hours",
        type=float,
        default=float(os.getenv("TOKENCLAW_MANAGED_ACTIVATION_PREVIEW_STALE_AFTER_HOURS", "72")),
        help="Classify persisted preview outcomes as stale after this many hours, default: 72.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.local_activation_executor import (
        build_managed_activation_preview_request,
        build_managed_activation_preview_result,
    )
    from tokenclaw.managed_egress import managed_egress_violations
    from tokenclaw.orchestrator_research import load_json_file, write_json

    try:
        plan = load_json_file(args.plan_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(plan, dict):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_plan_json", "message": "plan JSON must be an object"}})
        return 1

    request_payload = build_managed_activation_preview_request(
        plan,
        top_successor_count=int(args.top_preview_successors or 0),
    )
    routing_pathway_batch = _routing_pathway_outcome_batch_result(
        status="skipped",
        reason="routing-pathway-outcome-batch-input-not-configured",
    )
    violations = managed_egress_violations(request_payload)
    if violations:
        result = build_managed_activation_preview_result(
            request_payload,
            fetch={
                "status": "blocked",
                "reason": "unsafe-egress-payload",
                "managed_server_calls_made": False,
                "violation_count": len(violations),
                "blocked_keys": sorted({item.get("key", "unknown") for item in violations}),
            },
        )
        _attach_routing_pathway_outcome_batch(result, routing_pathway_batch)
        if args.persist_outcomes:
            result["stored_preview_outcomes"] = _attach_managed_activation_preview_outcome_persistence(
                result,
                db=str(args.db),
                stale_after_hours=float(args.preview_stale_after_hours),
            )
        write_json(stdout, result, pretty=args.pretty)
        return 1

    url = str(args.managed_preview_url or "").strip()
    if not url:
        result = build_managed_activation_preview_result(
            request_payload,
            fetch={
                "status": "skipped",
                "reason": "managed-preview-url-not-configured",
                "managed_server_calls_made": False,
            },
        )
        _attach_routing_pathway_outcome_batch(result, routing_pathway_batch)
        if args.persist_outcomes:
            result["stored_preview_outcomes"] = _attach_managed_activation_preview_outcome_persistence(
                result,
                db=str(args.db),
                stale_after_hours=float(args.preview_stale_after_hours),
            )
        write_json(stdout, result, pretty=args.pretty)
        return 0

    if url.startswith("http://") and not _is_loopback_url(url) and not args.allow_insecure_managed_url:
        result = build_managed_activation_preview_result(
            request_payload,
            fetch={
                "status": "blocked",
                "reason": "insecure-managed-preview-url",
                "url": _redact_url(url),
                "managed_server_calls_made": False,
            },
        )
        _attach_routing_pathway_outcome_batch(result, routing_pathway_batch)
        if args.persist_outcomes:
            result["stored_preview_outcomes"] = _attach_managed_activation_preview_outcome_persistence(
                result,
                db=str(args.db),
                stale_after_hours=float(args.preview_stale_after_hours),
            )
        write_json(stdout, result, pretty=args.pretty)
        return 2

    headers = {"content-type": "application/json"}
    token = os.getenv(str(args.api_key_env or ""))
    if token:
        headers["authorization"] = f"Bearer {token}"

    routing_pathway_batch = _submit_routing_pathway_outcome_batch(
        source_json=str(args.routing_pathway_outcomes_json),
        db=str(args.db),
        limit=int(args.routing_pathway_outcome_limit),
        stale_after_hours=float(args.preview_stale_after_hours),
        managed_preview_url=url,
        managed_routing_pathway_outcomes_url=str(args.managed_routing_pathway_outcomes_url),
        allow_insecure_managed_url=bool(args.allow_insecure_managed_url),
        headers=headers,
        timeout=float(args.timeout),
    )

    started = time.time()
    response_payload: Any | None = None
    fetch: dict[str, Any] = {
        "status": "ok",
        "url": _redact_url(url),
        "timeout_seconds": float(args.timeout),
        "managed_server_calls_made": True,
        "provider_calls_made": False,
        "policy_files_written": False,
        "api_key_value_included": False,
    }
    try:
        response = httpx.post(url, json=request_payload, headers=headers, timeout=float(args.timeout))
        fetch["latency_ms"] = int((time.time() - started) * 1000)
        fetch["status_code"] = response.status_code
        try:
            response_payload = response.json()
            fetch["response_json"] = True
        except ValueError:
            response_payload = None
            fetch["response_json"] = False
        if response.status_code >= 400:
            fetch.update({
                "status": "error",
                "reason": "managed-preview-server-error",
                "error": response.text[:500],
            })
    except httpx.HTTPError as exc:
        fetch.update({
            "status": "error",
            "reason": exc.__class__.__name__,
            "error": str(exc)[:500],
        })

    result = build_managed_activation_preview_result(
        request_payload,
        response_payload=response_payload,
        fetch=fetch,
    )
    _attach_routing_pathway_outcome_batch(result, routing_pathway_batch)
    if args.persist_outcomes:
        result["stored_preview_outcomes"] = _attach_managed_activation_preview_outcome_persistence(
            result,
            db=str(args.db),
            stale_after_hours=float(args.preview_stale_after_hours),
        )
    write_json(stdout, result, pretty=args.pretty)
    return 0 if fetch.get("status") == "ok" else 1


def managed_activation_preview_outcomes_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Report review-only managed activation preview outcomes persisted in the local TokenClaw DB"
    )
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="TokenClaw database URL or SQLite path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Stored preview outcomes to inspect, default: 1000.")
    parser.add_argument(
        "--preview-stale-after-hours",
        type=float,
        default=float(os.getenv("TOKENCLAW_MANAGED_ACTIVATION_PREVIEW_STALE_AFTER_HOURS", "72")),
        help="Classify preview outcomes as stale after this many hours, default: 72.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from tokenclaw.managed_activation_preview_outcomes import (
        build_managed_activation_preview_outcomes_report,
    )

    store = _open_store_for_db(str(args.db))
    try:
        result = build_managed_activation_preview_outcomes_report(
            store,
            limit=int(args.limit),
            stale_after_hours=float(args.preview_stale_after_hours),
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if not bool((result.get("egress_guard") or {}).get("blocked")) else 1


def managed_routing_pathway_candidates_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Convert managed routing_pathway_matrix rows into review-only local shadow routing candidates"
    )
    parser.add_argument(
        "--decision-json",
        required=True,
        help="Path to a managed policy decision or routing_pathway_matrix JSON file.",
    )
    parser.add_argument(
        "--preview-stale-after-hours",
        type=float,
        default=float(os.getenv("TOKENCLAW_MANAGED_ROUTING_PATHWAY_STALE_AFTER_HOURS", "72")),
        help="Classify pathway matrix rows as stale after this many hours, default: 72.",
    )
    parser.add_argument("--now", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.managed_routing_pathway_candidates import (
        build_managed_routing_pathway_shadow_candidates,
    )
    from tokenclaw.orchestrator_research import load_json_file, write_json

    try:
        source = load_json_file(args.decision_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(source, dict):
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {"type": "invalid_decision_json", "message": "decision JSON must be an object"},
            },
        )
        return 1

    result = build_managed_routing_pathway_shadow_candidates(
        source,
        now=_parse_cli_datetime(args.now),
        stale_after_hours=float(args.preview_stale_after_hours),
    )
    write_json(stdout, result, pretty=args.pretty)
    return 0 if not bool((result.get("egress_guard") or {}).get("blocked")) else 1


def managed_routing_pathway_outcomes_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Record metadata-only local outcome feedback for routing pathway matrix candidates"
    )
    parser.add_argument(
        "--decision-json",
        required=True,
        help="Path to a managed policy decision, routing_pathway_matrix, or pathway candidate JSON file.",
    )
    parser.add_argument(
        "--db",
        default=_default_db_path(),
        help="TokenClaw database URL or SQLite path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Local evidence rows to inspect, default: 1000.")
    parser.add_argument(
        "--preview-stale-after-hours",
        type=float,
        default=float(os.getenv("TOKENCLAW_MANAGED_ROUTING_PATHWAY_STALE_AFTER_HOURS", "72")),
        help="Classify pathway matrix rows as stale after this many hours, default: 72.",
    )
    parser.add_argument("--now", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.managed_routing_pathway_outcomes import (
        build_local_routing_pathway_outcome_feedback,
    )
    from tokenclaw.orchestrator_research import load_json_file, write_json

    try:
        source = load_json_file(args.decision_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(source, dict):
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {"type": "invalid_decision_json", "message": "decision JSON must be an object"},
            },
        )
        return 1

    store = _open_store_for_db(str(args.db))
    try:
        result = build_local_routing_pathway_outcome_feedback(
            store,
            source,
            limit=int(args.limit),
            stale_after_hours=float(args.preview_stale_after_hours),
            now=_parse_cli_datetime(args.now),
        )
    finally:
        store.conn.close()
    write_json(stdout, result, pretty=args.pretty)
    return 0 if not bool((result.get("egress_guard") or {}).get("blocked")) else 1


def managed_routing_canary_action_drafts_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Convert managed routing pathway outcome scores into review-only local routing canary action drafts"
    )
    parser.add_argument(
        "--scores-json",
        required=True,
        help="Path to managed routing outcome scores, managed-history rollups, or scored pathway outcome JSON.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from tokenclaw.managed_routing_canary_action_drafts import (
        build_managed_routing_canary_action_drafts,
    )
    from tokenclaw.orchestrator_research import load_json_file, write_json

    try:
        source = load_json_file(args.scores_json)
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1
    if not isinstance(source, dict):
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {"type": "invalid_scores_json", "message": "scores JSON must be an object"},
            },
        )
        return 1

    result = build_managed_routing_canary_action_drafts(source)
    write_json(stdout, result, pretty=args.pretty)
    return 0 if not bool((result.get("egress_guard") or {}).get("blocked")) else 1


def proxy_main() -> None:
    # The provider proxy forwards real API credentials and request bodies upstream.
    # Keep installed CLI defaults localhost-only unless the user explicitly opts in
    # to a different bind address through TOKENCLAW_HOST or --host.
    if "TOKENCLAW_HOST" not in os.environ:
        os.environ["TOKENCLAW_HOST"] = "127.0.0.1"

    from tokenclaw.server import main

    main()


def _internal_proxy_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    if "TOKENCLAW_HOST" not in os.environ:
        os.environ["TOKENCLAW_HOST"] = "127.0.0.1"

    from tokenclaw.server import main

    main(list(argv or []))
    return 0


def _internal_dashboard_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    from tokenclaw import dashboard

    old_argv = sys.argv
    try:
        sys.argv = ["tokenclaw internal dashboard", *list(argv or [])]
        dashboard.main()
    finally:
        sys.argv = old_argv
    return 0


def _internal_command_names() -> list[str]:
    names = {
        name[:-4].replace("_", "-")
        for name, value in globals().items()
        if name.endswith("_cli")
        and not name.startswith("_")
        and name not in {"internal_cli", "tokenclaw_cli"}
        and callable(value)
    }
    names.update({"proxy", "dashboard"})
    return sorted(names)


def _internal_command_handler(command: str) -> Any | None:
    command = command.strip()
    if command.startswith("tokenclaw-"):
        command = command.removeprefix("tokenclaw-")
    if command == "proxy":
        return _internal_proxy_cli
    if command == "dashboard":
        return _internal_dashboard_cli
    name = command.replace("-", "_") + "_cli"
    handler = globals().get(name)
    return handler if callable(handler) else None


def _invoke_internal_command(
    handler: Any,
    argv: Sequence[str],
    *,
    stdout: Any,
    stderr: Any,
) -> int:
    parameters = inspect.signature(handler).parameters
    kwargs: dict[str, Any] = {"stdout": stdout}
    if "stderr" in parameters:
        kwargs["stderr"] = stderr
    return int(handler(list(argv), **kwargs))


def internal_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="tokenclaw internal",
        description="Advanced TokenClaw development, policy, report, and diagnostic commands.",
        epilog="Use `tokenclaw internal --list` to print available internal command names.",
    )
    parser.add_argument("--list", action="store_true", help="List internal commands and exit.")
    parser.add_argument("command", nargs="?", help="Internal command name, without the tokenclaw- prefix.")
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    if not args:
        parser.print_help(stdout)
        return 0
    if args[0] in {"-h", "--help"}:
        parser.print_help(stdout)
        return 0
    if args[0] == "--list":
        for name in _internal_command_names():
            stdout.write(name + "\n")
        return 0

    parsed = parser.parse_args(args)
    if parsed.list:
        for name in _internal_command_names():
            stdout.write(name + "\n")
        return 0
    if not parsed.command:
        parser.print_help(stdout)
        return 0

    handler = _internal_command_handler(parsed.command)
    if handler is None:
        stderr.write(f"unknown internal command: {parsed.command}\n")
        stderr.write("Run `tokenclaw internal --list` to see available commands.\n")
        return 2
    return _invoke_internal_command(handler, parsed.command_args, stdout=stdout, stderr=stderr)


def tokenclaw_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "internal":
        return internal_cli(args[1:], stdout=stdout, stderr=stderr)
    return _onboarding_tokenclaw_cli(args, stdout=stdout, stderr=stderr)


def tokenclaw_main() -> None:
    raise SystemExit(tokenclaw_cli())


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


def policy_draft_stage_main() -> None:
    raise SystemExit(policy_draft_stage_cli())


def policy_draft_validate_main() -> None:
    raise SystemExit(policy_draft_validate_cli())


def managed_activation_bundle_stage_main() -> None:
    raise SystemExit(managed_activation_bundle_stage_cli())


def managed_activation_bundle_apply_main() -> None:
    raise SystemExit(managed_activation_bundle_apply_cli())


def managed_activation_proof_main() -> None:
    raise SystemExit(managed_activation_proof_cli())


def openai_optimization_draft_dry_run_main() -> None:
    raise SystemExit(openai_optimization_draft_dry_run_cli())


def openai_optimization_draft_apply_main() -> None:
    raise SystemExit(openai_optimization_draft_apply_cli())


def policy_draft_apply_main() -> None:
    raise SystemExit(policy_draft_apply_cli())


def codex_app_policy_dry_run_main() -> None:
    raise SystemExit(codex_app_policy_dry_run_cli())


def managed_rollout_actions_review_main() -> None:
    raise SystemExit(managed_rollout_actions_review_cli())


def optimization_rollout_actions_review_main() -> None:
    raise SystemExit(optimization_rollout_actions_review_cli())


def optimization_rollout_actions_apply_main() -> None:
    raise SystemExit(optimization_rollout_actions_apply_cli())


def scaffold_rollout_actions_review_main() -> None:
    raise SystemExit(scaffold_rollout_actions_review_cli())


def scaffold_rollout_actions_apply_main() -> None:
    raise SystemExit(scaffold_rollout_actions_apply_cli())


def managed_rollout_actions_apply_main() -> None:
    raise SystemExit(managed_rollout_actions_apply_cli())


def managed_rollout_actions_dry_run_main() -> None:
    raise SystemExit(managed_rollout_actions_dry_run_cli())


def old_context_summary_dry_run_main() -> None:
    raise SystemExit(old_context_summary_dry_run_cli())


def old_context_summary_impact_main() -> None:
    raise SystemExit(old_context_summary_impact_cli())


def old_context_summary_quality_gate_main() -> None:
    raise SystemExit(old_context_summary_quality_gate_cli())


def old_context_summary_rollout_actions_review_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_review_cli())


def old_context_summary_rollout_actions_apply_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_apply_cli())


def old_context_summary_rollout_actions_dry_run_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_dry_run_cli())


def old_context_summary_rollout_actions_impact_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_impact_cli())


def managed_rollout_actions_impact_main() -> None:
    raise SystemExit(managed_rollout_actions_impact_cli())


def policy_rollback_main() -> None:
    raise SystemExit(policy_rollback_cli())


def codex_diagnose_main() -> None:
    raise SystemExit(codex_diagnose_cli())


def codex_canary_impact_main() -> None:
    raise SystemExit(codex_canary_impact_cli())


def openai_scoreboard_main() -> None:
    raise SystemExit(openai_scoreboard_cli())


def openai_routing_report_main() -> None:
    raise SystemExit(openai_routing_report_cli())


def routing_coverage_report_main() -> None:
    raise SystemExit(routing_coverage_report_cli())




def openai_routing_narrow_canary_review_main() -> None:
    raise SystemExit(openai_routing_narrow_canary_review_cli())


def anthropic_routing_canary_stage_main() -> None:
    raise SystemExit(anthropic_routing_canary_stage_cli())


def openai_old_context_summary_report_main() -> None:
    raise SystemExit(openai_old_context_summary_report_cli())


def openai_cache_replay_report_main() -> None:
    raise SystemExit(openai_cache_replay_report_cli())


def openai_cache_replay_blocker_outcomes_main() -> None:
    raise SystemExit(openai_cache_replay_blocker_outcomes_cli())


def crunch_blocker_outcomes_main() -> None:
    raise SystemExit(crunch_blocker_outcomes_cli())


def activation_safety_stop_burndown_main() -> None:
    raise SystemExit(activation_safety_stop_burndown_cli())


def anthropic_routing_safety_stop_unblock_drill_main() -> None:
    raise SystemExit(anthropic_routing_safety_stop_unblock_drill_cli())


def optimization_action_ledger_main() -> None:
    raise SystemExit(optimization_action_ledger_cli())


def optimization_coordinator_dry_run_main() -> None:
    raise SystemExit(optimization_coordinator_dry_run_cli())


def provider_tool_adoption_report_main() -> None:
    raise SystemExit(provider_tool_adoption_report_cli())


def repeated_scaffold_opportunity_main() -> None:
    raise SystemExit(repeated_scaffold_opportunity_cli())


def instruction_dedup_opportunity_main() -> None:
    raise SystemExit(instruction_dedup_opportunity_cli())


def instruction_dedup_dry_run_main() -> None:
    raise SystemExit(instruction_dedup_dry_run_cli())


def instruction_dedup_impact_main() -> None:
    raise SystemExit(instruction_dedup_impact_cli())


def terminal_output_compaction_opportunity_main() -> None:
    raise SystemExit(terminal_output_compaction_opportunity_cli())


def codex_terminal_transcript_opportunity_main() -> None:
    raise SystemExit(codex_terminal_transcript_opportunity_cli())


def anthropic_thinking_compaction_opportunity_main() -> None:
    raise SystemExit(anthropic_thinking_compaction_opportunity_cli())


def anthropic_thinking_compaction_impact_main() -> None:
    raise SystemExit(anthropic_thinking_compaction_impact_cli())


def anthropic_thinking_compaction_dry_run_main() -> None:
    raise SystemExit(anthropic_thinking_compaction_dry_run_cli())


def codex_terminal_transcript_dry_run_main() -> None:
    raise SystemExit(codex_terminal_transcript_dry_run_cli())


def codex_terminal_transcript_impact_main() -> None:
    raise SystemExit(codex_terminal_transcript_impact_cli())


def terminal_output_compaction_dry_run_main() -> None:
    raise SystemExit(terminal_output_compaction_dry_run_cli())


def terminal_output_compaction_impact_main() -> None:
    raise SystemExit(terminal_output_compaction_impact_cli())


def repeated_scaffold_impact_main() -> None:
    raise SystemExit(repeated_scaffold_impact_cli())


def repeated_scaffold_activation_main() -> None:
    raise SystemExit(repeated_scaffold_activation_cli())


def openai_cache_replay_impact_main() -> None:
    raise SystemExit(openai_cache_replay_impact_cli())


def openai_cache_replay_readiness_main() -> None:
    raise SystemExit(openai_cache_replay_readiness_cli())


def local_promotion_candidates_main() -> None:
    raise SystemExit(local_promotion_candidates_cli())


def crunch_promotion_draft_dry_run_main() -> None:
    raise SystemExit(crunch_promotion_draft_dry_run_cli())


def cache_promotion_draft_dry_run_main() -> None:
    raise SystemExit(cache_promotion_draft_dry_run_cli())


def routing_promotion_draft_dry_run_main() -> None:
    raise SystemExit(routing_promotion_draft_dry_run_cli())


def openai_cache_replay_apply_main() -> None:
    raise SystemExit(openai_cache_replay_apply_cli())


def openai_cache_replay_dry_run_main() -> None:
    raise SystemExit(openai_cache_replay_dry_run_cli())


def openai_old_context_summary_dry_run_main() -> None:
    raise SystemExit(openai_old_context_summary_dry_run_cli())


def openai_canary_impact_main() -> None:
    raise SystemExit(openai_canary_impact_cli())


def claude_canary_impact_main() -> None:
    raise SystemExit(claude_canary_impact_cli())


def anthropic_routing_lifecycle_report_main() -> None:
    raise SystemExit(anthropic_routing_lifecycle_report_cli())


def claude_canary_actions_main() -> None:
    raise SystemExit(claude_canary_actions_cli())


def claude_canary_actions_apply_main() -> None:
    raise SystemExit(claude_canary_actions_apply_cli())


def routing_canary_promote_main() -> None:
    raise SystemExit(routing_canary_promote_cli())


def routing_experiment_report_main() -> None:
    raise SystemExit(routing_experiment_report_cli())


def routing_promotion_draft_stage_main() -> None:
    raise SystemExit(routing_promotion_draft_stage_cli())


def phase_routing_report_main() -> None:
    raise SystemExit(phase_routing_report_cli())


def session_phase_memory_main() -> None:
    raise SystemExit(session_phase_memory_cli())


def cache_replayability_report_main() -> None:
    raise SystemExit(cache_replayability_report_cli())


def request_shape_rollups_main() -> None:
    raise SystemExit(request_shape_rollups_cli())


def request_shape_crunch_canary_impact_main() -> None:
    raise SystemExit(request_shape_crunch_canary_impact_cli())


def request_shape_crunch_policy_decision_main() -> None:
    raise SystemExit(request_shape_crunch_policy_decision_cli())


def request_shape_crunch_canary_stage_main() -> None:
    raise SystemExit(request_shape_crunch_canary_stage_cli())


def request_shape_cache_replay_canary_stage_main() -> None:
    raise SystemExit(request_shape_cache_replay_canary_stage_cli())


def request_shape_cache_replay_evidence_main() -> None:
    raise SystemExit(request_shape_cache_replay_evidence_cli())


def request_shape_cache_replay_policy_decision_main() -> None:
    raise SystemExit(request_shape_cache_replay_policy_decision_cli())


def managed_recommendation_handoff_main() -> None:
    raise SystemExit(managed_recommendation_handoff_cli())


def local_activation_outcome_summary_main() -> None:
    raise SystemExit(local_activation_outcome_summary_cli())


def managed_activation_bundle_apply_outcomes_main() -> None:
    raise SystemExit(managed_activation_bundle_apply_outcomes_cli())


def cache_replay_cohorts_main() -> None:
    raise SystemExit(cache_replay_cohorts_cli())


def cache_smoke_diagnostic_main() -> None:
    raise SystemExit(cache_smoke_diagnostic_cli())


def cache_replay_dry_run_main() -> None:
    raise SystemExit(cache_replay_dry_run_cli())


def managed_pattern_rollups_main() -> None:
    raise SystemExit(managed_pattern_rollups_cli())


def optimization_eval_plan_main() -> None:
    raise SystemExit(optimization_eval_plan_cli())


def optimization_shadow_eval_main() -> None:
    raise SystemExit(optimization_shadow_eval_cli())


def optimization_eval_queue_main() -> None:
    raise SystemExit(optimization_eval_queue_cli())


def optimization_promotion_report_main() -> None:
    raise SystemExit(optimization_promotion_report_cli())


def optimization_promotion_actions_main() -> None:
    raise SystemExit(optimization_promotion_actions_cli())


def optimization_promotion_blocker_review_main() -> None:
    raise SystemExit(optimization_promotion_blocker_review_cli())


def optimization_promotion_canary_apply_main() -> None:
    raise SystemExit(optimization_promotion_canary_apply_cli())


def optimization_promotion_impact_main() -> None:
    raise SystemExit(optimization_promotion_impact_cli())


def managed_feedback_status_main() -> None:
    raise SystemExit(managed_feedback_status_cli())


def managed_feedback_flush_main() -> None:
    raise SystemExit(managed_feedback_flush_cli())


def sqlite_maintenance_main() -> None:
    raise SystemExit(sqlite_maintenance_cli())


def db_adopt_legacy_main() -> None:
    raise SystemExit(db_adopt_legacy_cli())


def savings_loop_bottlenecks_main() -> None:
    raise SystemExit(savings_loop_bottlenecks_cli())


def orchestrator_research_main() -> None:
    raise SystemExit(orchestrator_research_cli())


def evidence_to_activation_burndown_main() -> None:
    raise SystemExit(evidence_to_activation_burndown_cli())


def activation_burndown_main() -> None:
    raise SystemExit(activation_burndown_cli())


def local_activation_executor_main() -> None:
    raise SystemExit(local_activation_executor_cli())


def managed_activation_preview_main() -> None:
    raise SystemExit(managed_activation_preview_cli())


def managed_activation_preview_outcomes_main() -> None:
    raise SystemExit(managed_activation_preview_outcomes_cli())


def managed_routing_pathway_candidates_main() -> None:
    raise SystemExit(managed_routing_pathway_candidates_cli())


def managed_routing_pathway_outcomes_main() -> None:
    raise SystemExit(managed_routing_pathway_outcomes_cli())


def managed_routing_canary_action_drafts_main() -> None:
    raise SystemExit(managed_routing_canary_action_drafts_cli())


def post_promotion_priority_delta_review_main() -> None:
    raise SystemExit(post_promotion_priority_delta_review_cli())


def post_promotion_policy_draft_dry_run_main() -> None:
    raise SystemExit(post_promotion_policy_draft_dry_run_cli())


def post_promotion_policy_draft_apply_main() -> None:
    raise SystemExit(post_promotion_policy_draft_apply_cli())
