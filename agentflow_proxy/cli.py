from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import httpx

from agentflow_proxy.cli_common import (
    is_loopback_url as _is_loopback_url,
    open_store_for_db as _open_store_for_db,
    redact_secret as _redact_secret,
    write_json as _write_json,
)
from agentflow_proxy.upstream_url import redact_url as _redact_url

from agentflow_proxy.cli_commands.policy_bundle import (
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
from agentflow_proxy.cli_commands.policy_workbench import (
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


from agentflow_proxy.cli_commands.onboarding import (
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
    _doctor_claude_vscode_target,
    _doctor_codex_target,
    _doctor_provider_target,
    _env_file_value,
    _fetch_agentflow_stats,
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
    agentflow_cli,
)

from agentflow_proxy.cli_commands.optimization_reports import (
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
    openai_routing_canary_stage_cli,
    anthropic_routing_canary_stage_cli,
    openai_old_context_summary_report_cli,
    openai_cache_replay_report_cli,
    openai_cache_replay_blocker_outcomes_cli,
    crunch_blocker_outcomes_cli,
    activation_safety_stop_burndown_cli,
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



def policy_rollback_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Rollback local AgentFlow policy YAML files from apply backups")
    parser.add_argument(
        "--apply-id",
        help="Rollback the exact policy workbench apply transaction with this apply ID.",
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
        default=os.getenv("AGENTFLOW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL for --apply-id rollback, default: {_default_policy_reload_url()}",
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
        "--pretty",
        action="store_true",
        help="Pretty-print rollback JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    if args.apply_id:
        from agentflow_proxy.policy_events import log_policy_event
        from agentflow_proxy.policy_workbench import POLICY_DRAFT_ROLLBACK_SCHEMA, rollback_policy_apply

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
            "codex_app": result.get("codex_app"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_rollback_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def managed_feedback_status_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.optimization.feedback import managed_feedback_status_cli as _managed_feedback_status_cli

    return _managed_feedback_status_cli(argv, stdout=stdout)


def managed_feedback_flush_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.optimization.feedback import managed_feedback_flush_cli as _managed_feedback_flush_cli

    return _managed_feedback_flush_cli(argv, stdout=stdout)


def sqlite_maintenance_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run local SQLite retention maintenance for AgentFlow metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow SQLite DB path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Retention window in days. Defaults to AGENTFLOW_SQLITE_RETENTION_DAYS or 7.",
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
        default=int(os.getenv("AGENTFLOW_RESEARCH_BACKLOG_THRESHOLD", "3")),
        help="Minimum status:ready actionable issue count before research mode is skipped.",
    )
    parser.add_argument(
        "--trusted-author",
        default=os.getenv("AGENTFLOW_GITHUB_TRUSTED_AUTHOR", "lutzkuen"),
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

    from agentflow_proxy.orchestrator_research import build_research_plan, load_json_file, write_json

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
    if os.getenv("AGENTFLOW_RESEARCH_FETCH_CLOSED_ISSUES", "1").strip().lower() in {"0", "false", "no", "off"}:
        return issues
    gh = shutil.which("gh")
    if not gh:
        return issues
    repos = sorted({repo for repo in (_issue_repo_for_research(issue) for issue in issues) if repo})
    if not repos:
        env_repos = os.getenv("AGENTFLOW_RESEARCH_GITHUB_REPOS", "")
        repos = sorted({repo.strip() for repo in env_repos.split(",") if repo.strip().count("/") == 1})
    if not repos:
        return issues
    limit = os.getenv("AGENTFLOW_RESEARCH_CLOSED_ISSUE_LIMIT", "50")
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
    if not isinstance(stats, dict):
        return stats

    db = stats.get("db")
    if not isinstance(db, str) or not db.strip():
        return stats
    db_arg = db.strip()
    if not db_arg.startswith(("postgresql://", "postgres://")) and not Path(db_arg).expanduser().exists():
        return stats

    try:
        limit = max(1, min(int(os.getenv("AGENTFLOW_RESEARCH_REQUEST_SHAPE_LIMIT", "1000")), 10_000))
    except ValueError:
        limit = 1000

    from agentflow_proxy.request_shape_rollups import (
        build_request_shape_cache_replay_evidence_report,
        build_request_shape_cache_replay_policy_decision_report,
        build_request_shape_rollups_report,
    )
    from agentflow_proxy.cache_smoke import build_isolated_cache_replay_hit_recovery_smoke

    needs_rollups = not any(
        isinstance(stats.get(key), dict)
        for key in ("request_shape_rollups", "request_shape_rollup_report", "request_shape_rollup_candidates_report")
    )
    needs_cache_replay_evidence = not isinstance(stats.get("request_shape_cache_replay_evidence"), dict)
    needs_cache_replay_policy_decision = not isinstance(stats.get("request_shape_cache_replay_policy_decision"), dict)
    if not needs_rollups and not needs_cache_replay_evidence and not needs_cache_replay_policy_decision:
        return stats

    enriched = dict(stats)
    store = _open_store_for_db(db_arg)
    try:
        if needs_rollups:
            enriched["request_shape_rollups"] = build_request_shape_rollups_report(
                store,
                limit=limit,
                persist=False,
                run_id="orchestrator-research-dry-run",
            )
        cache_replay_evidence = (
            enriched.get("request_shape_cache_replay_evidence")
            if isinstance(enriched.get("request_shape_cache_replay_evidence"), dict)
            else None
        )
        if needs_cache_replay_evidence:
            rules_path = (
                Path(os.getenv("AGENTFLOW_CACHE_CANARY_POLICY")).expanduser()
                if os.getenv("AGENTFLOW_CACHE_CANARY_POLICY")
                else Path(
                    os.getenv(
                        "AGENTFLOW_CONFIG_DIR",
                        os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
                    )
                ).expanduser()
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
    finally:
        try:
            store.conn.close()
        except Exception:
            pass

    return enriched


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

    from agentflow_proxy.orchestrator_research import build_evidence_to_activation_burndown, load_json_file, write_json

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


def proxy_main() -> None:
    # The provider proxy forwards real API credentials and request bodies upstream.
    # Keep installed CLI defaults localhost-only unless the user explicitly opts in
    # to a different bind address through AGENTFLOW_HOST or --host.
    os.environ.setdefault("AGENTFLOW_HOST", "127.0.0.1")

    from agentflow_proxy.server import main

    main()


def agentflow_main() -> None:
    raise SystemExit(agentflow_cli())


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


def openai_routing_canary_stage_main() -> None:
    raise SystemExit(openai_routing_canary_stage_cli())


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


def orchestrator_research_main() -> None:
    raise SystemExit(orchestrator_research_cli())


def evidence_to_activation_burndown_main() -> None:
    raise SystemExit(evidence_to_activation_burndown_cli())


def post_promotion_priority_delta_review_main() -> None:
    raise SystemExit(post_promotion_priority_delta_review_cli())


def post_promotion_policy_draft_dry_run_main() -> None:
    raise SystemExit(post_promotion_policy_draft_dry_run_cli())


def post_promotion_policy_draft_apply_main() -> None:
    raise SystemExit(post_promotion_policy_draft_apply_cli())
