from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import ipaddress
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import yaml

from tokenclaw.cache_smoke import build_cache_smoke_diagnostic
from tokenclaw.codex_turn_policy import (
    CODEX_APP_SOURCE_SURFACE,
    canonical_source_surface,
    codex_app_bundle_policy_state,
    codex_app_surface_policy_state,
    is_codex_turn_source_surface,
)
from tokenclaw.limiter import model_tier
from tokenclaw.policy_files import policy_file_status
from tokenclaw.pricing import (
    codex_app_pricing_basis,
    estimate_blended_input_savings,
    estimate_cost,
    provider_prompt_cache_accounting,
)
from tokenclaw.golden_path import build_golden_path_summary
from tokenclaw.managed_mode import managed_mode_public_meta, managed_product_mode
from tokenclaw.paths import tokenclaw_config_path
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.quality import (
    derive_codex_turn_quality_signals,
    derive_provider_quality_signals,
    summarize_quality_signals,
)
from tokenclaw.recommendations import (
    OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
    OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE,
    PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE,
    PHASE_ROUTING_OUTCOME_SOURCE_SURFACE,
    pattern_decision_summaries,
)
from tokenclaw.routing_experiments import build_post_fix_shadow_yield_report, build_routing_experiment_report
from tokenclaw.savings_loop_bottlenecks import build_savings_loop_bottlenecks_report
from tokenclaw.savings_attribution import realized_savings_attribution
from tokenclaw.session_phase_memory import build_session_phase_memory
from tokenclaw.store import utc_now

CODEX_APP_PRICING_BASIS = codex_app_pricing_basis()
CODEX_APP_MODEL = str(CODEX_APP_PRICING_BASIS["model"])
CODEX_APP_COST_BASIS = str(CODEX_APP_PRICING_BASIS["cost_basis"])
CODEX_APP_PROCESSING_MODE = str(CODEX_APP_PRICING_BASIS["processing_mode"])
CODEX_APP_COST_KNOWN = bool(CODEX_APP_PRICING_BASIS["cost_known"])
CODEX_APP_TELEMETRY_ONLY_REASON = "codex-app-telemetry-only"
TOKEN_CHARS = 4
MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT = 200
MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT = 20


def _utc_today_start_iso() -> str:
    return f"{utc_now()[:10]}T00:00:00+00:00"


def _utc_day_window(days: int = 7) -> list[str]:
    today = date.fromisoformat(utc_now()[:10])
    first = today - timedelta(days=max(1, days) - 1)
    return [(first + timedelta(days=i)).isoformat() for i in range(max(1, days))]


def _json_obj(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _json_obj_has_value(raw: Any) -> bool:
    if not raw:
        return False
    if isinstance(raw, dict):
        return bool(raw)
    try:
        value = json.loads(raw)
    except Exception:
        return True
    return bool(value) if isinstance(value, dict) else True














def _metadata_only_privacy() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "tool_payloads_included": False,
    }


def _crunch_canary_lifecycle_for_stats(crunch: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "request_shape_repeated_context_canary",
        "repeated_context_crunch_canary",
        "request_shape_crunch_canary",
        "crunch_canary",
    ):
        meta = crunch.get(key)
        if isinstance(meta, dict):
            status = public_label(meta.get("status") or meta.get("lifecycle") or meta.get("cohort"), "unknown")
            cohort = public_label(meta.get("cohort") or status, "unknown")
            if status in {"canary-applied", "canary_applied"}:
                status = "applied"
            elif status in {"canary-holdout", "canary_holdout"}:
                status = "holdout"
            return {
                "status": status,
                "cohort": cohort.replace("-", "_"),
                "policy_id": public_label(meta.get("policy_id"), "unknown"),
                "cohort_id": public_label(meta.get("cohort_id"), "unknown"),
                "rule_group": public_label(meta.get("rule_group") or meta.get("candidate_rule") or "repeated-context-conservative", "repeated-context-conservative"),
            }
    return None


def _crunch_canary_captured_savings(conn: Any, *, limit: int = 10_000) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        rows = conn.execute(
            """
            select crunch_json, cost_est_usd, cost_baseline_usd
            from calls
            where crunch_json is not null
            order by created_at desc
            limit ?
            """,
            (max(1, min(int(limit or 1), 50_000)),),
        ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        crunch = _json_obj(row["crunch_json"] if hasattr(row, "keys") else row[0])
        lifecycle = _crunch_canary_lifecycle_for_stats(crunch)
        if lifecycle is None:
            continue
        status = str(lifecycle.get("status") or "")
        cohort = str(lifecycle.get("cohort") or "")
        is_applied = status == "applied" or cohort == "canary_applied"
        is_holdout = status == "holdout" or cohort == "canary_holdout"
        if not (is_applied or is_holdout):
            continue
        key = (
            str(lifecycle.get("policy_id") or "unknown"),
            str(lifecycle.get("cohort_id") or "unknown"),
            str(lifecycle.get("rule_group") or "repeated-context-conservative"),
        )
        group = groups.setdefault(
            key,
            {
                "policy_id": key[0],
                "cohort_id": key[1],
                "rule_group": key[2],
                "applied_count": 0,
                "holdout_count": 0,
                "applied_cost_delta_usd": 0.0,
                "holdout_cost_delta_usd": 0.0,
                "applied_saved_tokens": 0,
                "holdout_saved_tokens": 0,
            },
        )
        cost = _as_float(row["cost_est_usd"] if hasattr(row, "keys") else row[1])
        baseline = _as_float(row["cost_baseline_usd"] if hasattr(row, "keys") else row[2]) or cost
        delta = baseline - cost
        saved_tokens = _as_int(crunch.get("tokens_saved_est") or crunch.get("saved_tokens"))
        if is_applied:
            group["applied_count"] += 1
            group["applied_cost_delta_usd"] += delta
            group["applied_saved_tokens"] += saved_tokens
        else:
            group["holdout_count"] += 1
            group["holdout_cost_delta_usd"] += delta
            group["holdout_saved_tokens"] += saved_tokens

    captured_rows: list[dict[str, Any]] = []
    for group in groups.values():
        applied_count = _as_int(group.get("applied_count"))
        holdout_count = _as_int(group.get("holdout_count"))
        holdout_avg_delta = _as_float(group.get("holdout_cost_delta_usd")) / holdout_count if holdout_count else 0.0
        captured_usd = max(0.0, _as_float(group.get("applied_cost_delta_usd")) - (holdout_avg_delta * applied_count))
        holdout_avg_tokens = _as_int(group.get("holdout_saved_tokens")) / holdout_count if holdout_count else 0.0
        captured_tokens = max(0, int(round(_as_int(group.get("applied_saved_tokens")) - (holdout_avg_tokens * applied_count))))
        captured_rows.append({
            "policy_id": group["policy_id"],
            "cohort_id": group["cohort_id"],
            "rule_group": group["rule_group"],
            "status": "captured" if applied_count > 0 and holdout_count > 0 and captured_usd > 0 else "no-captured-savings",
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "captured_saved_tokens": captured_tokens,
            "captured_saved_usd": round(captured_usd, 8),
            "applied_cost_delta_usd": round(_as_float(group.get("applied_cost_delta_usd")), 8),
            "holdout_cost_delta_usd": round(_as_float(group.get("holdout_cost_delta_usd")), 8),
            "holdout_avg_cost_delta_usd": round(holdout_avg_delta, 8),
        })
    captured_rows.sort(
        key=lambda item: (_as_float(item.get("captured_saved_usd")), _as_int(item.get("captured_saved_tokens"))),
        reverse=True,
    )
    total_captured = sum(_as_float(row.get("captured_saved_usd")) for row in captured_rows)
    return {
        "schema": "tokenclaw.crunch_canary_captured_savings_stats.v1",
        "status": "captured" if total_captured > 0 else "no-captured-savings",
        "summary": {
            "cohort_count": len(captured_rows),
            "captured_cohort_count": sum(1 for row in captured_rows if row.get("status") == "captured"),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in captured_rows),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in captured_rows),
            "captured_saved_tokens": sum(_as_int(row.get("captured_saved_tokens")) for row in captured_rows),
            "captured_saved_usd": round(total_captured, 8),
        },
        "rows": captured_rows[:50],
        "privacy": _metadata_only_privacy(),
    }


def _crunch_rule_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("TOKENCLAW_CRUNCH_RULES")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / "crunch_rules.yaml")
    candidates.append(tokenclaw_config_path("crunch_rules.yaml"))
    candidates.append(Path(__file__).parent / "crunch_rules.yaml")
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _active_crunch_rule_coverage() -> dict[str, Any] | None:
    loaded_path: Path | None = None
    loaded: dict[str, Any] | None = None
    for path in _crunch_rule_candidate_paths():
        if not path.exists():
            continue
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "schema": "tokenclaw.active_crunch_rule_coverage.v1",
                "status": "unreadable-rule-file",
                "rule_file": "crunch_rules.yaml",
                "target_local_policy": "crunch_rules",
                "target_local_policy_section": "crunch.rules",
                "summary": {
                    "active_rule_count": 0,
                    "widened_rule_count": 0,
                    "applied_count": 0,
                    "holdout_count": 0,
                    "skipped_count": 0,
                    "blocked_count": 0,
                    "observed_saved_chars": 0,
                    "observed_saved_tokens": 0,
                    "observed_saved_usd": 0.0,
                    "no_op_reason": "unreadable-crunch-rule-file",
                    "next_action": "inspect-crunch-rule-file",
                },
                "rules": [],
                "missing_measurements": ["active-crunch-rule-coverage"],
                "privacy": _metadata_only_privacy(),
            }
        if isinstance(value, dict):
            loaded_path = path
            loaded = value
            break
    if loaded is None:
        return None

    section = loaded.get("request_shape_repeated_context_canaries")
    raw_rules = section.get("rules") if isinstance(section, dict) and isinstance(section.get("rules"), list) else []
    rules: list[dict[str, Any]] = []
    policy_sources: dict[str, int] = {}
    decisions: dict[str, int] = {}
    applied_count = 0
    holdout_count = 0
    skipped_count = 0
    blocked_count = 0
    observed_saved_chars = 0
    observed_saved_tokens = 0
    observed_saved_usd = 0.0

    for item in raw_rules:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        decision = item.get("policy_decision") if isinstance(item.get("policy_decision"), dict) else {}
        decision_value = str(decision.get("decision") or "").strip()
        if decision.get("schema") != "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1":
            continue
        source = public_label(item.get("policy_source") or "local-manual", "local-manual")
        policy_sources[source] = policy_sources.get(source, 0) + 1
        decisions[decision_value or "unknown"] = decisions.get(decision_value or "unknown", 0) + 1
        rule_applied = _as_int(decision.get("applied_count"))
        rule_holdout = _as_int(decision.get("holdout_count"))
        rule_skipped = _as_int(decision.get("skipped_count"))
        rule_blocked = _as_int(decision.get("blocked_count"))
        rule_tokens = _as_int(decision.get("observed_saved_tokens"))
        rule_chars = _as_int(decision.get("observed_saved_chars"))
        applied_count += rule_applied
        holdout_count += rule_holdout
        skipped_count += rule_skipped
        blocked_count += rule_blocked
        observed_saved_tokens += rule_tokens
        observed_saved_chars += rule_chars
        observed_saved_usd += _as_float(decision.get("observed_saved_usd"))
        rollout = item.get("rollout") if isinstance(item.get("rollout"), dict) else {}
        rules.append({
            "rank": len(rules) + 1,
            "rule_id": public_label(item.get("id") or f"request-shape-crunch-rule-{len(rules) + 1}", "request-shape-crunch-rule"),
            "rule_ref": public_id(item.get("id") or f"request-shape-crunch-rule-{len(rules) + 1}", prefix="rule"),
            "policy_source": source,
            "decision": public_label(decision_value or "unknown", "unknown"),
            "graduation_decision": public_label(decision.get("graduation_decision") or decision_value or "unknown", "unknown"),
            "decision_id": public_label(decision.get("decision_id") or "unknown", "unknown"),
            "source_evidence_schema": public_label(decision.get("source_evidence_schema") or "unknown", "unknown"),
            "applied_count": rule_applied,
            "holdout_count": rule_holdout,
            "skipped_count": rule_skipped,
            "blocked_count": rule_blocked,
            "observed_saved_chars": rule_chars,
            "observed_saved_tokens": rule_tokens,
            "observed_saved_usd": round(_as_float(decision.get("observed_saved_usd")), 8),
            "canary_fraction": round(_as_float(rollout.get("canary_fraction") or rollout.get("fraction")), 6),
            "holdout_fraction": round(_as_float(rollout.get("holdout_fraction")), 6),
            "metadata_only": True,
            "aggregate_only": True,
        })

    active_rule_count = len(rules)
    widened_rule_count = sum(1 for item in rules if item.get("decision") == "widen")
    has_applied = applied_count > 0 or observed_saved_tokens > 0 or observed_saved_chars > 0 or observed_saved_usd > 0
    status = "observed" if has_applied else "no-applied-coverage"
    no_op_reason = None if has_applied else "no-applied-coverage"
    missing = [] if has_applied else ["no-applied-coverage"]
    return {
        "schema": "tokenclaw.active_crunch_rule_coverage.v1",
        "status": status,
        "rule_file": "crunch_rules.yaml",
        "rules_path_included": False,
        "target_local_policy": "crunch_rules",
        "target_local_policy_section": "crunch.rules",
        "summary": {
            "active_rule_count": active_rule_count,
            "widened_rule_count": widened_rule_count,
            "rule_count": len(raw_rules),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "blocked_count": blocked_count,
            "observed_saved_chars": observed_saved_chars,
            "observed_saved_tokens": observed_saved_tokens,
            "observed_saved_usd": round(observed_saved_usd, 8),
            "policy_source_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(policy_sources.items(), key=lambda pair: (-pair[1], pair[0]))
            ],
            "decision_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(decisions.items(), key=lambda pair: (-pair[1], pair[0]))
            ],
            "policy_source": rules[0]["policy_source"] if rules else None,
            "target_local_rule_file": "crunch_rules.yaml",
            "target_local_policy_section": "crunch.rules",
            "no_op_reason": no_op_reason,
            "next_action": "rank-observed-crunch-family-follow-up" if has_applied else "inspect-active-crunch-rule-coverage",
        },
        "rules": rules[:10],
        "missing_measurements": missing,
        "privacy": _metadata_only_privacy(),
        "loaded_rule_file": loaded_path.name if loaded_path is not None else "crunch_rules.yaml",
    }


def _copy_policy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_utc_datetime(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _seconds_since_iso(raw: Any, now: datetime) -> int | None:
    parsed = _parse_utc_datetime(raw)
    if parsed is None:
        return None
    return max(0, int((now - parsed).total_seconds()))


def _host_is_loopback(host: str | None) -> bool | None:
    if not host:
        return None
    cleaned = str(host).strip().strip("[]").lower()
    if not cleaned:
        return None
    if cleaned == "localhost":
        return True
    if cleaned in {"0.0.0.0", "::", "*"}:
        return False
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def _redact_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return None
    parsed = urlparse(raw_url)
    host = parsed.hostname or ""
    netloc = host
    if parsed.port is not None:
        netloc = f"{host}:{parsed.port}"
    redacted_query = urlencode(
        [(key, "[redacted]") for key, _value in parse_qsl(parsed.query, keep_blank_values=True)]
    ).replace("%5Bredacted%5D", "[redacted]")
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, redacted_query, parsed.fragment))


def _url_host_state(raw_url: str | None) -> dict[str, Any]:
    if not raw_url:
        return {
            "configured": False,
            "scheme": None,
            "host": None,
            "host_loopback": None,
            "redacted_url": None,
        }
    parsed = urlparse(raw_url)
    host = parsed.hostname
    return {
        "configured": True,
        "scheme": parsed.scheme or None,
        "host": host,
        "host_loopback": _host_is_loopback(host),
        "redacted_url": _redact_url(raw_url),
    }


def _db_path_class(default_db: str | None) -> str:
    raw = os.getenv("TOKENCLAW_DATABASE_URL") or default_db or ""
    lowered = raw.lower()
    if "://" in raw:
        if lowered.startswith("sqlite://"):
            return "sqlite-url"
        return "external-database-url"
    expanded = os.path.abspath(os.path.expanduser(raw)) if raw else ""
    home = os.path.abspath(os.path.expanduser("~"))
    if not expanded:
        return "unknown"
    if expanded.startswith(os.path.join(home, ".tokenclaw") + os.sep):
        return "local-tokenclaw-home"
    if expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        return "local-temp"
    return "local-path"


def _policy_events_path_class() -> str:
    raw = os.getenv("TOKENCLAW_POLICY_EVENTS_LOG", "~/.tokenclaw/policy_events.jsonl")
    expanded = os.path.abspath(os.path.expanduser(raw))
    home = os.path.abspath(os.path.expanduser("~"))
    if expanded.startswith(os.path.join(home, ".tokenclaw") + os.sep):
        return "local-tokenclaw-home"
    if expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        return "local-temp"
    return "local-path"


def _promotion_blocker_review_path() -> Path:
    raw = os.getenv("TOKENCLAW_PROMOTION_BLOCKER_REVIEW_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("promotion_blocker_recommendation_review.json")


def _post_promotion_priority_review_path() -> Path:
    raw = os.getenv("TOKENCLAW_POST_PROMOTION_PRIORITY_REVIEW_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("post_promotion_priority_delta_review.json")


def _post_promotion_policy_draft_dry_run_path() -> Path:
    raw = os.getenv("TOKENCLAW_POST_PROMOTION_POLICY_DRAFT_DRY_RUN_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("post_promotion_policy_draft_dry_run.json")


def _evidence_to_activation_plan_candidate_paths(package_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON"):
        raw = os.getenv(name)
        if raw:
            candidates.append(Path(raw).expanduser())
            return candidates
    ops_root = os.getenv("TOKENCLAW_OPS_ROOT")
    if ops_root:
        candidates.append(Path(ops_root).expanduser() / "runs" / "research" / "latest.plan.json")
        return candidates
    root = package_root or Path(__file__).resolve().parents[1]
    candidates.append(root.parent / "runs" / "research" / "latest.plan.json")
    for parent in (root, *root.parents):
        candidates.append(parent / "tokenclaw_ops" / "runs" / "research" / "latest.plan.json")
    candidates.append(tokenclaw_config_path("research/latest.plan.json"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _evidence_to_activation_plan_path() -> Path:
    candidates = _evidence_to_activation_plan_candidate_paths()
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _post_promotion_outcome_flush_status_path() -> Path:
    raw = os.getenv("TOKENCLAW_POST_PROMOTION_OUTCOME_FLUSH_STATUS_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("post_promotion_outcome_flush_status.json")


def _local_path_class(raw: str | os.PathLike[str] | None) -> str:
    expanded = os.path.abspath(os.path.expanduser(str(raw or ""))) if raw else ""
    home = os.path.abspath(os.path.expanduser("~"))
    if not expanded:
        return "unknown"
    if expanded.startswith(os.path.join(home, ".tokenclaw") + os.sep):
        return "local-tokenclaw-home"
    if expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        return "local-temp"
    return "local-path"


def _breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _read_workbench_draft_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _public_workbench_draft(manifest: dict[str, Any], *, mtime: float | None = None) -> dict[str, Any]:
    sections = manifest.get("sections") if isinstance(manifest.get("sections"), list) else []
    return {
        "draft_id": manifest.get("draft_id"),
        "created_at": manifest.get("created_at"),
        "mtime_seconds": mtime,
        "requested_section": manifest.get("requested_section"),
        "changed": bool(manifest.get("changed")),
        "changed_sections": manifest.get("changed_sections") if isinstance(manifest.get("changed_sections"), list) else [],
        "change_count": _as_int(manifest.get("change_count")),
        "section_count": len([section for section in sections if isinstance(section, dict)]),
        "workspace_path_included": False,
        "bundle_path_included": False,
        "raw_payload_included": False,
        "privacy": {
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }


def _workbench_staged_drafts(limit: int = 10) -> tuple[list[dict[str, Any]], int]:
    from tokenclaw.policy_files import _draft_workspace_root

    raw_workspace = os.getenv("TOKENCLAW_POLICY_DRAFT_DIR")
    workspace = _draft_workspace_root(raw_workspace)
    if not workspace.exists() or not workspace.is_dir():
        return [], 0

    rows: list[tuple[float, dict[str, Any]]] = []
    unreadable = 0
    try:
        children = list(workspace.iterdir())
    except OSError:
        return [], 1
    for child in children:
        manifest_path = child / "draft.json" if child.is_dir() else child
        if not manifest_path.exists() or manifest_path.name != "draft.json":
            continue
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        manifest = _read_workbench_draft_manifest(manifest_path)
        if manifest is None:
            unreadable += 1
            continue
        rows.append((mtime, _public_workbench_draft(manifest, mtime=mtime)))
    rows.sort(key=lambda item: ((item[1].get("created_at") or ""), item[0]), reverse=True)
    return [row for _mtime, row in rows[: max(1, min(int(limit), 50))]], unreadable


def _latest_policy_event(events: list[dict[str, Any]], *actions: str) -> dict[str, Any] | None:
    wanted = set(actions)
    for event in events:
        if isinstance(event, dict) and event.get("action") in wanted:
            return event
    return None


def _public_workbench_event(event: dict[str, Any] | None, *, now: datetime) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    action = str(event.get("action") or "")
    changed_sections = details.get("changed_sections") if isinstance(details.get("changed_sections"), list) else []
    requested_sections = details.get("requested_sections") if isinstance(details.get("requested_sections"), list) else []
    restored_sections = details.get("restored_sections") if isinstance(details.get("restored_sections"), list) else []
    reloaded_modules = details.get("reloaded_modules") if isinstance(details.get("reloaded_modules"), list) else []
    blocker_codes = details.get("blocker_reason_codes") if isinstance(details.get("blocker_reason_codes"), list) else []
    section_verdicts = details.get("section_verdicts") if isinstance(details.get("section_verdicts"), dict) else {}
    return {
        "created_at": event.get("created_at"),
        "age_seconds": _seconds_since_iso(event.get("created_at"), now),
        "action": action,
        "ok": bool(event.get("ok")),
        "source": details.get("source"),
        "status": details.get("status"),
        "draft_id": details.get("draft_id") or details.get("draft"),
        "apply_id": details.get("apply_id"),
        "backup_id": details.get("backup_id"),
        "can_apply": details.get("can_apply"),
        "apply_blocked": details.get("apply_blocked"),
        "dry_run": details.get("dry_run"),
        "force": details.get("force"),
        "manifest_source": details.get("manifest_source"),
        "apply_event_found": details.get("apply_event_found"),
        "changed_sections": changed_sections,
        "requested_sections": requested_sections,
        "restored_sections": restored_sections,
        "reloaded_module_count": len(reloaded_modules),
        "reloaded_modules": reloaded_modules,
        "change_count": _as_int(details.get("change_count")),
        "validation_error_count": _as_int(details.get("error_count")),
        "section_verdicts": dict(section_verdicts),
        "blocker_reason_codes": [str(code) for code in blocker_codes if isinstance(code, str)],
        "error_type": details.get("error_type"),
        "exit_code": details.get("exit_code"),
        "provider_calls_made": bool(details.get("provider_calls_made")),
        "managed_server_calls_made": bool(details.get("managed_server_calls_made")),
        "loopback_admin_calls_made": bool(details.get("loopback_admin_calls_made")),
        "file_paths_included": False,
        "raw_payload_included": False,
    }


def _workbench_readiness_state(
    *,
    reload_required_sections: list[str],
    latest_validation: dict[str, Any] | None,
    latest_apply: dict[str, Any] | None,
    latest_rollback: dict[str, Any] | None,
    staged_draft_count: int,
) -> tuple[str, str]:
    if reload_required_sections:
        return "reload-required", "active policy files changed after load"
    if latest_apply and not latest_apply.get("ok"):
        return "attention", "latest apply failed or was blocked"
    if latest_rollback and not latest_rollback.get("ok"):
        return "attention", "latest rollback failed or was blocked"
    if latest_validation:
        if latest_validation.get("can_apply") is True:
            return "draft-ready", "latest validation can be applied from CLI or loopback admin API"
        if latest_validation.get("apply_blocked") is True or latest_validation.get("ok") is False:
            return "draft-blocked", "latest validation blocks apply"
    if staged_draft_count:
        return "draft-needs-validation", "staged drafts exist without a passing latest validation"
    return "loaded", "no staged draft is waiting"


async def stats_policy_workbench_readiness(policy_state: dict[str, Any] | None = None) -> dict[str, Any]:
    from tokenclaw.policy_events import policy_events_enabled, recent_policy_events

    state = policy_state if isinstance(policy_state, dict) else await stats_policies()
    summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    reload_sections = summary.get("reload_required_sections") if isinstance(summary.get("reload_required_sections"), list) else []
    staged_drafts, unreadable_drafts = _workbench_staged_drafts(limit=10)
    events_payload = recent_policy_events(limit=200)
    raw_events = events_payload.get("events") if isinstance(events_payload.get("events"), list) else []
    events = [event for event in raw_events if isinstance(event, dict)]
    now = datetime.now(timezone.utc)

    latest_stage = _public_workbench_event(_latest_policy_event(events, "draft-stage"), now=now)
    latest_validation = _public_workbench_event(_latest_policy_event(events, "draft-validate"), now=now)
    latest_apply = _public_workbench_event(_latest_policy_event(events, "draft-apply"), now=now)
    latest_rollback = _public_workbench_event(_latest_policy_event(events, "rollback"), now=now)
    latest_reload = _public_workbench_event(_latest_policy_event(events, "reload"), now=now)
    public_events = [
        row
        for row in (
            _public_workbench_event(event, now=now)
            for event in events
            if event.get("action") in {"draft-stage", "draft-validate", "draft-apply", "rollback", "reload"}
        )
        if row is not None
    ][:25]
    failures = [row for row in public_events if not row.get("ok")]
    last_failure = failures[0] if failures else None
    backup_ids = []
    for row in public_events:
        backup_id = row.get("backup_id")
        if isinstance(backup_id, str) and backup_id and backup_id not in backup_ids:
            backup_ids.append(backup_id)
        if len(backup_ids) >= 5:
            break
    status, reason = _workbench_readiness_state(
        reload_required_sections=[str(section) for section in reload_sections],
        latest_validation=latest_validation,
        latest_apply=latest_apply,
        latest_rollback=latest_rollback,
        staged_draft_count=len(staged_drafts),
    )

    return {
        "schema": "tokenclaw.policy_workbench_readiness.v1",
        "status": status,
        "status_reason": reason,
        "generated_at": utc_now(),
        "read_only": True,
        "mutating_dashboard_endpoints": False,
        "loopback_admin_only": True,
        "workspace": {
            "configured": bool(os.getenv("TOKENCLAW_POLICY_DRAFT_DIR")),
            "path_class": _local_path_class(os.getenv("TOKENCLAW_POLICY_DRAFT_DIR") or "~/.tokenclaw/policy_drafts"),
            "raw_path_included": False,
        },
        "staged_drafts": {
            "count": len(staged_drafts),
            "changed_count": sum(1 for draft in staged_drafts if draft.get("changed")),
            "unreadable_count": unreadable_drafts,
            "latest": staged_drafts[0] if staged_drafts else None,
            "recent": staged_drafts,
        },
        "validation": {
            "latest": latest_validation,
            "can_apply": latest_validation.get("can_apply") if latest_validation else None,
            "status": latest_validation.get("status") if latest_validation else None,
            "blocker_reason_codes": latest_validation.get("blocker_reason_codes") if latest_validation else [],
        },
        "apply": {
            "latest": latest_apply,
            "last_backup_ids": backup_ids,
        },
        "rollback": {
            "latest": latest_rollback,
        },
        "reload": {
            "required": bool(reload_sections),
            "required_sections": [str(section) for section in reload_sections],
            "latest": latest_reload,
        },
        "events": {
            "enabled": policy_events_enabled(),
            "path_class": _policy_events_path_class(),
            "raw_path_included": False,
            "latest_stage": latest_stage,
            "latest_validation": latest_validation,
            "latest_apply": latest_apply,
            "latest_rollback": latest_rollback,
            "latest_reload": latest_reload,
            "latest_failure": last_failure,
            "recent": public_events,
        },
        "operator_labels": {
            "stage": "tokenclaw-policy-draft-stage",
            "validate": "tokenclaw-policy-draft-validate",
            "apply": "tokenclaw-policy-draft-apply",
            "rollback": "tokenclaw-policy-rollback",
            "reload": "tokenclaw-policy-reload",
            "admin_reload_path": "/tokenclaw/admin/reload-policies",
        },
        "privacy": {
            "local_only": True,
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_tool_payloads_included": False,
            "raw_session_ids_included": False,
            "raw_request_ids_included": False,
            "cache_keys_included": False,
            "draft_bundle_contents_included": False,
            "policy_file_contents_included": False,
            "absolute_paths_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "dashboard_mutations_available": False,
        },
    }
































async def stats_safety(
    *,
    store_obj: Any | None = None,
    default_db: str,
    proxy_host: str | None = None,
    dashboard_host: str | None = None,
    dashboard_read_only: bool = True,
) -> dict[str, Any]:
    from tokenclaw.recommendations import (
        managed_auth_configured,
        recommendation_failure_mode,
        recommendation_server_configured,
        recommendation_server_url,
        recommendation_timeout_seconds,
        recommendations_enabled,
    )

    product_mode = managed_product_mode()
    recommendation_enabled = recommendations_enabled()
    recommendation_url = recommendation_server_url()
    policy_bundle_url = os.getenv("TOKENCLAW_POLICY_BUNDLE_RECOMMENDATION_URL")
    auth_configured = managed_auth_configured()
    log_bodies_enabled = _env_bool("TOKENCLAW_LOG_BODIES", False)
    policy_events_enabled = _env_bool("TOKENCLAW_POLICY_EVENTS", True)
    proxy_host_value = proxy_host or os.getenv("TOKENCLAW_PROXY_HOST") or os.getenv("TOKENCLAW_HOST")
    proxy_loopback = _host_is_loopback(proxy_host_value)
    dashboard_loopback = _host_is_loopback(dashboard_host) if dashboard_host else None
    db_class = _db_path_class(default_db)
    recommendation_state = _url_host_state(recommendation_url if recommendation_enabled or os.getenv("TOKENCLAW_RECOMMENDATION_SERVER_URL") else None)
    policy_bundle_state = _url_host_state(policy_bundle_url)
    feedback_queue = _managed_feedback_queue_health(store_obj)
    feedback_summary = feedback_queue.get("summary") or {}
    feedback_drain = feedback_queue.get("drain") if isinstance(feedback_queue.get("drain"), dict) else {}

    warnings: list[dict[str, Any]] = []

    def warn(code: str, severity: str, message: str) -> None:
        warnings.append({"code": code, "severity": severity, "message": message})

    if proxy_loopback is False:
        warn(
            "proxy-bind-non-loopback",
            "critical",
            "Provider proxy host is not loopback; provider credentials and request bodies can be exposed on the network.",
        )
    elif proxy_loopback is None:
        warn(
            "proxy-bind-unknown",
            "info",
            "Provider proxy host was not supplied to this dashboard process.",
        )
    if log_bodies_enabled:
        warn(
            "body-logging-enabled",
            "critical",
            "TOKENCLAW_LOG_BODIES is enabled; raw request and response bodies may be stored locally for debugging.",
        )
    if recommendation_enabled and not auth_configured:
        warn(
            "managed-recommendation-unauthenticated",
            "high",
            "Managed recommendation and outcome feedback are enabled without a configured managed API key.",
        )
    if recommendation_enabled and not recommendation_server_configured():
        warn(
            "managed-recommendation-server-unconfigured",
            "medium",
            "Managed recommendations are enabled but no recommendation server URL is configured; local policy will remain authoritative.",
        )
    if product_mode.mode != "local_only" and not any(product_mode.family_enabled.values()):
        warn(
            "managed-actions-all-locally-disabled",
            "info",
            "Managed mode is enabled but routing, crunching, and cache actions are all locally disabled.",
        )
    if _as_int(feedback_summary.get("due")) > 0:
        warn(
            "managed-feedback-due-queue",
            "medium",
            "Managed outcome feedback has retryable rows due for flush; the managed feedback loop may be stuck.",
        )
    if _as_int(feedback_summary.get("due")) > 0 and feedback_drain.get("blocked_reason"):
        warn(
            "managed-feedback-drain-blocked",
            "medium",
            "Managed outcome feedback is due but the drain loop is blocked by current managed opt-in configuration.",
        )
    if _as_int(feedback_summary.get("retryable_error")) > 0:
        warn(
            "managed-feedback-retryable-errors",
            "medium",
            "Managed outcome feedback has rows waiting after retryable delivery errors.",
        )
    if _as_int(feedback_summary.get("stale_sending")) > 0:
        warn(
            "managed-feedback-stale-sending",
            "medium",
            "Managed outcome feedback has stale in-flight rows that should be recovered for retry.",
        )
    if _as_int(feedback_summary.get("dropped_after_limit")) > 0:
        warn(
            "managed-feedback-dropped-after-limit",
            "high",
            "Managed outcome feedback rows were dropped after reaching the retry limit.",
        )
    if policy_bundle_url and not auth_configured:
        warn(
            "managed-policy-fetch-unauthenticated",
            "medium",
            "Managed policy bundle URL is configured without a managed API key in the environment.",
        )
    if db_class == "external-database-url":
        warn(
            "external-database-url-configured",
            "medium",
            "AgentFlow is configured with a non-SQLite database URL; verify the database remains inside the intended privacy boundary.",
        )
    if not policy_events_enabled:
        warn(
            "policy-events-disabled",
            "medium",
            "Policy event logging is disabled, so local policy review/apply audit history will not be recorded.",
        )
    if not dashboard_read_only:
        warn(
            "dashboard-not-read-only",
            "critical",
            "Dashboard read-only mode is disabled.",
        )

    severity_rank = {"critical": 4, "high": 3, "medium": 2, "info": 1}
    worst = "ok"
    if warnings:
        worst = max((row["severity"] for row in warnings), key=lambda value: severity_rank.get(value, 0))

    return {
        "schema": "tokenclaw.safety_privacy.v1",
        "generated_at": utc_now(),
        "status": "warning" if any(row["severity"] != "info" for row in warnings) else "ok",
        "highest_severity": worst,
        "summary": {
            "warning_count": len([row for row in warnings if row["severity"] != "info"]),
            "info_count": len([row for row in warnings if row["severity"] == "info"]),
            "proxy_loopback": proxy_loopback,
            "body_logging_enabled": log_bodies_enabled,
            "managed_communication_enabled": bool(recommendation_enabled or policy_bundle_url),
            "managed_mode": product_mode.mode,
            "managed_server_calls_enabled": product_mode.server_calls_enabled,
            "managed_local_application_enabled": product_mode.local_application_enabled,
            "managed_auth_configured": auth_configured,
            "managed_feedback_due": _as_int(feedback_summary.get("due")),
            "managed_feedback_retryable_error": _as_int(feedback_summary.get("retryable_error")),
            "managed_feedback_dropped_after_limit": _as_int(feedback_summary.get("dropped_after_limit")),
            "dashboard_read_only": bool(dashboard_read_only),
            "db_path_class": db_class,
            "policy_events_enabled": policy_events_enabled,
        },
        "checks": {
            "provider_proxy": {
                "host_configured": bool(proxy_host_value),
                "host": proxy_host_value,
                "loopback": proxy_loopback,
            },
            "dashboard": {
                "read_only": bool(dashboard_read_only),
                "host_configured": bool(dashboard_host),
                "host": dashboard_host,
                "loopback": dashboard_loopback,
            },
            "body_logging": {
                "enabled": log_bodies_enabled,
                "raw_request_bodies_included_in_payload": False,
                "raw_response_bodies_included_in_payload": False,
            },
            "managed": {
                "recommendations_enabled": recommendation_enabled,
                "mode": product_mode.mode,
                "product_mode": product_mode.public_meta(),
                "recommendation_server": recommendation_state,
                "recommendation_server_configured": recommendation_server_configured(),
                "recommendation_timeout_seconds": recommendation_timeout_seconds(),
                "recommendation_failure_mode": recommendation_failure_mode(),
                "policy_bundle_recommendation": policy_bundle_state,
                "auth_configured": auth_configured,
                "api_key_value_included": False,
                "feedback_queue": feedback_queue,
            },
            "database": {
                "path_class": db_class,
                "raw_path_included": False,
            },
            "policy_events": {
                "enabled": policy_events_enabled,
                "path_class": _policy_events_path_class(),
                "raw_path_included": False,
            },
        },
        "privacy": {
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_response_bodies_included": False,
            "managed_feedback_payload_json_included": False,
            "secrets_included": False,
            "url_credentials_redacted": True,
            "sensitive_query_values_redacted": True,
        },
        "warnings": warnings,
    }


async def stats_policies() -> dict[str, Any]:
    from tokenclaw import cache, crunch, router, routing_experiments

    state = {
        "schema": "tokenclaw.policy_state.v1",
        "routing": {
            "enabled": bool(router.ROUTING_ENABLED),
            "policy_source": router.ROUTING_RULES_SOURCE,
            "rule_path": router.ROUTING_RULES_PATH,
            "file": policy_file_status(
                router.ROUTING_RULES_PATH,
                loaded_at=router.ROUTING_RULES_LOADED_AT,
                loaded_snapshot=router.ROUTING_RULES_LOADED_FILE,
            ),
            "rules": _copy_policy(router.ROUTING_RULES),
            "phase_canary": _copy_policy(router.ROUTING_PHASE_CANARY),
            "defaults": {
                "haiku": router.HAIKU_DEFAULT,
                "sonnet": router.SONNET_DEFAULT,
                "opus": router.OPUS_DEFAULT,
            },
            "openai": {
                "enabled": bool(router.OPENAI_ROUTING_ENABLED or router.ROUTING_OPENAI_CANARY.get("enabled")),
                "env_routing_enabled": bool(router.OPENAI_ROUTING_ENABLED),
                "policy_source": router.ROUTING_OPENAI_CANARY.get("policy_source") or router.ROUTING_RULES_SOURCE,
                "rule_path": router.ROUTING_RULES_PATH,
                "large": router.OPENAI_LARGE_DEFAULT,
                "small": router.OPENAI_SMALL_DEFAULT,
                "tiny": router.OPENAI_TINY_DEFAULT,
                "canary": _copy_policy(router.ROUTING_OPENAI_CANARY),
            },
            "strip_thinking_history": bool(router.STRIP_THINKING_HISTORY),
        },
        "crunch": {
            "enabled": bool(crunch.CRUNCH_ENABLED),
            "policy_source": crunch.CRUNCH_POLICY_SOURCE,
            "rule_path": crunch.CRUNCH_RULES_PATH,
            "file": policy_file_status(
                crunch.CRUNCH_RULES_PATH,
                loaded_at=crunch.CRUNCH_RULES_LOADED_AT,
                loaded_snapshot=crunch.CRUNCH_RULES_LOADED_FILE,
            ),
            "threshold_chars": crunch.CRUNCH_THRESHOLD_CHARS,
            "prompt_cache": {
                "enabled": bool(crunch.PROMPT_CACHE_ENABLED),
                "min_chars": crunch.PROMPT_CACHE_MIN_CHARS,
            },
            "old_context_summarization": _copy_policy(crunch.OLD_CONTEXT_SUMMARY_POLICY),
            "thinking_deduplication": _copy_policy(crunch.THINKING_DEDUP_POLICY),
            "anthropic_thinking_history_compaction": (
                crunch.anthropic_thinking_compaction_effective_policy()
                if hasattr(crunch, "anthropic_thinking_compaction_effective_policy")
                else _copy_policy(getattr(crunch, "ANTHROPIC_THINKING_COMPACTION_POLICY", {}))
            ),
            "instruction_section_deduplication": _copy_policy(crunch.INSTRUCTION_SECTION_DEDUP_POLICY),
            "pattern_rules": _copy_policy(crunch.PATTERN_RULES),
            "repeated_provider_scaffolding": _copy_policy(crunch.REPEATED_PROVIDER_SCAFFOLDING_POLICY),
            "codex_repeated_scaffolding": _copy_policy(crunch.CODEX_REPEATED_SCAFFOLDING_POLICY),
        },
        "cache": {
            "enabled": bool(cache.CACHE_ENABLED or cache.SEMANTIC_CACHE_ENABLED),
            "policy_source": cache.CACHE_POLICY_SOURCE,
            "rule_path": cache.CACHE_RULES_PATH,
            "file": policy_file_status(
                cache.CACHE_RULES_PATH,
                loaded_at=cache.CACHE_RULES_LOADED_AT,
                loaded_snapshot=cache.CACHE_RULES_LOADED_FILE,
            ),
            "exact_cache": {
                "enabled": bool(cache.CACHE_ENABLED),
                "cache_tool_calls": bool(cache.CACHE_TOOL_CALLS),
            },
            "semantic_cache": {
                "enabled": bool(cache.SEMANTIC_CACHE_ENABLED),
                "threshold": cache.SEMANTIC_CACHE_THRESHOLD,
            },
            "file_watch": {
                "enabled": bool(cache.CACHE_FILE_WATCH_ENABLED),
                "root": cache.CACHE_FILE_WATCH_ROOT,
                "max_paths": cache.CACHE_FILE_WATCH_MAX_PATHS,
            },
        },
        "routing_experiments": {
            "enabled": bool(routing_experiments.ROUTING_EXPERIMENT_ENABLED),
            "policy_source": routing_experiments.ROUTING_EXPERIMENT_POLICY_SOURCE,
            "rule_path": routing_experiments.ROUTING_EXPERIMENT_RULES_PATH,
            "file": policy_file_status(
                routing_experiments.ROUTING_EXPERIMENT_RULES_PATH,
                loaded_at=routing_experiments.ROUTING_EXPERIMENT_RULES_LOADED_AT,
                loaded_snapshot=routing_experiments.ROUTING_EXPERIMENT_RULES_LOADED_FILE,
            ),
            "policy": _copy_policy(routing_experiments.ROUTING_EXPERIMENT_POLICY),
        },
        "codex_app": codex_app_bundle_policy_state(),
    }
    sections = ("routing", "crunch", "cache", "routing_experiments", "codex_app")
    file_backed_sections = ("routing", "crunch", "cache", "routing_experiments", "codex_app")
    state["source_surfaces"] = {
        CODEX_APP_SOURCE_SURFACE: codex_app_surface_policy_state(state),
    }
    reload_required_sections = [
        section
        for section in file_backed_sections
        if bool((state.get(section, {}).get("file") or {}).get("reload_required"))
    ]
    state["summary"] = {
        "policy_count": len(sections),
        "loaded_file_count": sum(
            1
            for section in file_backed_sections
            if bool(
                (((state.get(section, {}).get("file") or {}).get("loaded") or {}).get("exists"))
            )
        ),
        "manual_policy_count": sum(
            1
            for section in sections
            if state.get(section, {}).get("policy_source") == "local-manual"
        ),
        "local_default_policy_count": sum(
            1
            for section in sections
            if state.get(section, {}).get("policy_source") == "local-default"
        ),
        "reload_required": bool(reload_required_sections),
        "reload_required_sections": reload_required_sections,
        "source_surface_policy_count": len(state["source_surfaces"]),
    }
    state["workbench"] = await stats_policy_workbench_readiness(state)
    return state


async def stats_policy_events(limit: int = 50) -> dict[str, Any]:
    from tokenclaw.policy_events import recent_policy_events

    return recent_policy_events(limit=limit)


def _promotion_blocker_dashboard_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "absolute_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
    }


def _post_promotion_priority_handoff_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "absolute_paths_included": False,
        "file_paths_included": False,
        "individual_candidate_ids_included": False,
        "individual_action_ids_included": False,
        "individual_rule_ids_included": False,
        "artifact_payloads_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
    }


def _post_promotion_artifact_source(
    *,
    kind: str,
    path: Path,
    env_name: str,
    payload: dict[str, Any] | None,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "configured": bool(os.getenv(env_name)),
        "available": payload is not None,
        "status": status,
        "reason": reason,
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
        "path_class": _local_path_class(path),
        "path_included": False,
        "payload_included": False,
    }


def _read_post_promotion_artifact(path: Path) -> tuple[dict[str, Any] | None, str, str]:
    if not path.exists():
        return None, "missing", "artifact-not-found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, "unavailable", f"artifact-unreadable:{exc.__class__.__name__}"
    if not isinstance(payload, dict):
        return None, "unavailable", "artifact-not-json-object"
    return payload, "available", "loaded-local-artifact"


def _safe_count_breakdown(rows: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows[: max(0, limit)]:
        if not isinstance(row, dict):
            continue
        result.append({
            "value": public_label(row.get("value"), "unknown"),
            "count": _as_int(row.get("count")),
        })
    return result


def _post_promotion_action_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "widen-local-policy": 0,
        "collect-holdout-evidence": 0,
        "rollback-local-policy": 0,
        "keep-blocked": 0,
    }
    for candidate in candidates:
        action = public_label(candidate.get("next_action"), "keep-blocked")
        if action not in counts:
            action = "keep-blocked"
        counts[action] += 1
    return counts


def _post_promotion_status_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        status = public_label(candidate.get("status"), "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _post_promotion_noop_reasons(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        reasons = candidate.get("no_op_reasons") if isinstance(candidate.get("no_op_reasons"), list) else []
        for reason in reasons:
            label = public_label(reason, "unknown")
            counts[label] = counts.get(label, 0) + 1
    return counts


def _post_promotion_handoff_freshness(generated_values: list[Any], *, now: datetime) -> dict[str, Any]:
    parsed_values = [_parse_utc_datetime(value) for value in generated_values if value]
    parsed = [value for value in parsed_values if value is not None]
    latest = max(parsed, default=None)
    if latest is None:
        return {
            "latest_artifact_at": None,
            "age_seconds": None,
            "state": "no-artifacts",
        }
    age_seconds = max(0, int((now - latest).total_seconds()))
    if age_seconds <= 6 * 60 * 60:
        state = "fresh"
    elif age_seconds <= 24 * 60 * 60:
        state = "aging"
    else:
        state = "stale"
    return {
        "latest_artifact_at": latest.isoformat(),
        "age_seconds": age_seconds,
        "state": state,
    }


def _post_promotion_next_safe_command(
    *,
    review_payload: dict[str, Any] | None,
    draft_payload: dict[str, Any] | None,
    flush_payload: dict[str, Any] | None,
    top_next_action: str | None,
) -> dict[str, Any]:
    if review_payload is None:
        return {
            "label": "fetch managed priority deltas",
            "command": "tokenclaw-post-promotion-priority-delta-review --pretty",
            "read_only": True,
            "reason": "priority-review-missing",
        }
    if top_next_action == "collect-holdout-evidence":
        return {
            "label": "inspect holdout evidence successor",
            "command": "tokenclaw-post-promotion-priority-delta-review --pretty",
            "read_only": True,
            "reason": "holdout-evidence-required",
        }
    if draft_payload is None and top_next_action in {"widen-local-policy", "rollback-local-policy", "keep-blocked"}:
        return {
            "label": "dry-run local policy handoff",
            "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
            "read_only": True,
            "reason": "policy-draft-dry-run-missing",
        }
    draft_status = public_label(draft_payload.get("status"), "unknown") if isinstance(draft_payload, dict) else "missing"
    if draft_status == "blocked":
        return {
            "label": "inspect dry-run impact gate blockers",
            "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
            "read_only": True,
            "reason": "impact-gate-blocked",
        }
    if flush_payload is None:
        return {
            "label": "dry-run post-promotion outcome flush",
            "command": "tokenclaw-managed-feedback-status --post-promotion-action-outcomes --dry-run --pretty",
            "read_only": True,
            "reason": "outcome-flush-status-missing",
        }
    return {
        "label": "review handoff status",
        "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
        "read_only": True,
        "reason": "handoff-artifacts-present",
    }


async def stats_post_promotion_priority_handoff() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    review_path = _post_promotion_priority_review_path()
    draft_path = _post_promotion_policy_draft_dry_run_path()
    flush_path = _post_promotion_outcome_flush_status_path()
    review_payload, review_status, review_reason = _read_post_promotion_artifact(review_path)
    draft_payload, draft_status, draft_reason = _read_post_promotion_artifact(draft_path)
    flush_payload, flush_status, flush_reason = _read_post_promotion_artifact(flush_path)

    if review_payload is not None and review_payload.get("schema") != "tokenclaw.post_promotion_priority_delta_review.v1":
        review_payload = None
        review_status = "unavailable"
        review_reason = "unexpected-priority-review-schema"
    if draft_payload is not None and draft_payload.get("schema") != "tokenclaw.post_promotion_policy_draft_dry_run.v1":
        draft_payload = None
        draft_status = "unavailable"
        draft_reason = "unexpected-policy-draft-schema"
    if flush_payload is not None and flush_payload.get("schema") not in {
        "tokenclaw.managed_feedback_flush.v1",
        "tokenclaw.post_promotion_action_outcome_rollup_flush_status.v1",
    }:
        flush_payload = None
        flush_status = "unavailable"
        flush_reason = "unexpected-outcome-flush-schema"

    review_summary = review_payload.get("summary") if isinstance(review_payload, dict) and isinstance(review_payload.get("summary"), dict) else {}
    review_candidates = review_payload.get("candidates") if isinstance(review_payload, dict) and isinstance(review_payload.get("candidates"), list) else []
    review_candidates = [item for item in review_candidates if isinstance(item, dict)]
    action_counts = _post_promotion_action_counts(review_candidates)
    status_counts = _post_promotion_status_counts(review_candidates)
    noop_reasons = _post_promotion_noop_reasons(review_candidates)
    top_next_action = public_label(review_summary.get("top_next_action"), "none") if review_payload else None
    if top_next_action in {None, "none"} and review_candidates:
        top_next_action = public_label(review_candidates[0].get("next_action"), "keep-blocked")

    draft_summary = draft_payload.get("summary") if isinstance(draft_payload, dict) and isinstance(draft_payload.get("summary"), dict) else {}
    impact_gate_status = "missing"
    if draft_payload is not None:
        blocked = _as_int(draft_summary.get("impact_gate_blocked_count"))
        passed = _as_int(draft_summary.get("impact_gate_pass_count"))
        if blocked:
            impact_gate_status = "blocked"
        elif passed or _as_int(draft_summary.get("impact_gate_count")):
            impact_gate_status = "passed"
        else:
            impact_gate_status = public_label(draft_payload.get("status"), "unknown")

    flush_nested = (
        flush_payload.get("post_promotion_action_outcome_rollups")
        if isinstance(flush_payload, dict) and isinstance(flush_payload.get("post_promotion_action_outcome_rollups"), dict)
        else flush_payload
        if isinstance(flush_payload, dict) and flush_payload.get("schema") == "tokenclaw.post_promotion_action_outcome_rollup_flush_status.v1"
        else {}
    )
    flush_summary = flush_payload.get("flush") if isinstance(flush_payload, dict) and isinstance(flush_payload.get("flush"), dict) else {}
    flush_status_label = (
        public_label(flush_nested.get("status"), "unknown")
        if isinstance(flush_nested, dict) and flush_nested
        else public_label(flush_summary.get("status"), "missing")
        if isinstance(flush_summary, dict) and flush_summary
        else "missing"
    )
    freshness = _post_promotion_handoff_freshness(
        [
            review_payload.get("generated_at") if isinstance(review_payload, dict) else None,
            draft_payload.get("generated_at") if isinstance(draft_payload, dict) else None,
            flush_payload.get("generated_at") if isinstance(flush_payload, dict) else None,
        ],
        now=now,
    )
    command = _post_promotion_next_safe_command(
        review_payload=review_payload,
        draft_payload=draft_payload,
        flush_payload=flush_payload,
        top_next_action=top_next_action,
    )
    available_count = sum(1 for payload in (review_payload, draft_payload, flush_payload) if payload is not None)
    overall_status = "available" if review_payload is not None else "no-data" if available_count == 0 else "partial"
    return {
        "schema": "tokenclaw.post_promotion_priority_handoff_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": overall_status,
        "status_reason": "priority handoff artifacts loaded" if review_payload is not None else "post-promotion priority review artifact not found",
        "summary": {
            "artifact_count": available_count,
            "priority_review_status": review_status,
            "priority_review_candidate_count": _as_int(review_summary.get("review_candidate_count")) or len(review_candidates),
            "recommended_count": _as_int(review_summary.get("recommended_count")),
            "noop_count": _as_int(review_summary.get("noop_count")),
            "top_next_action": top_next_action,
            "widen_count": action_counts["widen-local-policy"],
            "collect_holdout_evidence_count": action_counts["collect-holdout-evidence"],
            "rollback_count": action_counts["rollback-local-policy"],
            "keep_blocked_count": action_counts["keep-blocked"],
            "policy_draft_status": public_label(draft_payload.get("status"), draft_status) if isinstance(draft_payload, dict) else draft_status,
            "draft_count": _as_int(draft_summary.get("draft_count")),
            "widen_draft_count": _as_int(draft_summary.get("widen_draft_count")),
            "rollback_draft_count": _as_int(draft_summary.get("rollback_draft_count")),
            "omitted_count": _as_int(draft_summary.get("omitted_count")),
            "impact_gate_status": impact_gate_status,
            "impact_gate_blocked_count": _as_int(draft_summary.get("impact_gate_blocked_count")),
            "outcome_flush_status": flush_status_label,
            "outcome_rollup_count": _as_int(flush_nested.get("rollup_count")) if isinstance(flush_nested, dict) else 0,
            "outcome_flush_reason": public_label(flush_nested.get("reason"), "none") if isinstance(flush_nested, dict) and flush_nested else public_label(flush_reason, "none"),
            "freshness_state": freshness["state"],
            "latest_artifact_at": freshness["latest_artifact_at"],
            "latest_artifact_age_seconds": freshness["age_seconds"],
            "next_safe_command": command["command"],
            "next_command_reason": command["reason"],
        },
        "status_counts": _breakdown_from_counts(status_counts),
        "next_action_counts": _breakdown_from_counts(action_counts),
        "no_op_reason_counts": _breakdown_from_counts(noop_reasons)[:10],
        "impact_gate_blocker_reason_counts": _safe_count_breakdown(draft_summary.get("impact_gate_blocker_reason_counts")),
        "sources": {
            "priority_review": _post_promotion_artifact_source(
                kind="priority-review-report",
                path=review_path,
                env_name="TOKENCLAW_POST_PROMOTION_PRIORITY_REVIEW_PATH",
                payload=review_payload,
                status=review_status,
                reason=review_reason,
            ),
            "policy_draft_dry_run": _post_promotion_artifact_source(
                kind="policy-draft-dry-run-report",
                path=draft_path,
                env_name="TOKENCLAW_POST_PROMOTION_POLICY_DRAFT_DRY_RUN_PATH",
                payload=draft_payload,
                status=draft_status,
                reason=draft_reason,
            ),
            "outcome_flush_status": _post_promotion_artifact_source(
                kind="outcome-flush-status-report",
                path=flush_path,
                env_name="TOKENCLAW_POST_PROMOTION_OUTCOME_FLUSH_STATUS_PATH",
                payload=flush_payload,
                status=flush_status,
                reason=flush_reason,
            ),
        },
        "commands": [
            command,
            {
                "label": "fetch managed priority deltas",
                "command": "tokenclaw-post-promotion-priority-delta-review --pretty",
                "read_only": True,
            },
            {
                "label": "dry-run local policy handoff",
                "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
                "read_only": True,
            },
            {
                "label": "dry-run post-promotion outcome flush",
                "command": "tokenclaw-managed-feedback-status --post-promotion-action-outcomes --dry-run --pretty",
                "read_only": True,
            },
        ],
        "privacy": _post_promotion_priority_handoff_privacy(),
    }


def _promotion_blocker_no_data(reason: str, *, source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or _promotion_blocker_review_path()
    return {
        "schema": "tokenclaw.promotion_blocker_next_actions_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": "no-data",
        "status_reason": reason,
        "source": {
            "kind": "local-review-report",
            "configured": bool(os.getenv("TOKENCLAW_PROMOTION_BLOCKER_REVIEW_PATH")),
            "available": False,
            "path_class": _local_path_class(path),
            "path_included": False,
        },
        "summary": {
            "source_recommendation_count": 0,
            "review_candidate_count": 0,
            "group_count": 0,
            "recommended_count": 0,
            "noop_count": 0,
            "stale_evidence_count": 0,
            "projected_savings_usd": 0.0,
            "top_local_action_family": None,
            "top_blocker_reason": None,
            "top_next_action": None,
            "top_expected_local_executor": None,
        },
        "family_counts": [],
        "top_blocker_reasons": [],
        "expected_local_executors": [],
        "next_actions": [],
        "groups": [],
        "commands": [
            {
                "label": "review local promotion blocker recommendations",
                "command": "tokenclaw-optimization-promotion-blocker-review recommendations.json --pretty",
                "read_only": True,
            }
        ],
        "privacy": _promotion_blocker_dashboard_privacy(),
    }


def _promotion_blocker_reason_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        reasons = candidate.get("blocker_reason_codes") if isinstance(candidate.get("blocker_reason_codes"), list) else []
        for reason in reasons:
            key = public_label(reason, "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts


def _promotion_blocker_stale_count(candidates: list[dict[str, Any]]) -> int:
    total = 0
    for candidate in candidates:
        reasons = candidate.get("blocker_reason_codes") if isinstance(candidate.get("blocker_reason_codes"), list) else []
        noops = candidate.get("no_op_reasons") if isinstance(candidate.get("no_op_reasons"), list) else []
        values = [str(item).lower() for item in [*reasons, *noops]]
        if any("stale" in value for value in values):
            total += 1
    return total


def _promotion_blocker_public_group(group: dict[str, Any], *, candidate_limit: int = 3) -> dict[str, Any]:
    recommendations = group.get("recommendations") if isinstance(group.get("recommendations"), list) else []
    public_candidates: list[dict[str, Any]] = []
    for candidate in recommendations[: max(0, candidate_limit)]:
        if not isinstance(candidate, dict):
            continue
        file_backed = candidate.get("file_backed_policy_representation")
        public_candidates.append(
            {
                "rank": _as_int(candidate.get("rank")),
                "status": public_label(candidate.get("status"), "unknown"),
                "recommendation_type": public_label(candidate.get("recommendation_type"), "unknown"),
                "candidate_family": public_label(candidate.get("candidate_family"), "unknown"),
                "blocker_family": public_label(candidate.get("blocker_family"), "unknown"),
                "blocker_reason_codes": [
                    public_label(reason, "unknown")
                    for reason in (candidate.get("blocker_reason_codes") if isinstance(candidate.get("blocker_reason_codes"), list) else [])[:5]
                ],
                "next_action": public_label(candidate.get("next_action"), "unknown"),
                "safety_stop_reason_code": public_label(candidate.get("safety_stop_reason_code"), "none") if candidate.get("safety_stop_reason_code") else None,
                "recommended_blocker_state": public_label(candidate.get("recommended_blocker_state"), "unknown") if candidate.get("recommended_blocker_state") else None,
                "recommended_unblock_action": public_label(candidate.get("recommended_unblock_action"), "unknown") if candidate.get("recommended_unblock_action") else None,
                "expected_local_executor": public_label(candidate.get("expected_local_executor"), "none"),
                "projected_savings_usd": _money(candidate.get("projected_savings_usd")),
                "file_backed_policy_exists": bool(file_backed.get("exists")) if isinstance(file_backed, dict) else False,
                "required_local_review": True,
            }
        )
    reason_counts = group.get("blocker_reason_code_counts") if isinstance(group.get("blocker_reason_code_counts"), list) else []
    safety_reason_counts = group.get("safety_stop_reason_counts") if isinstance(group.get("safety_stop_reason_counts"), list) else []
    top_reason = reason_counts[0].get("value") if reason_counts and isinstance(reason_counts[0], dict) else None
    top_safety_reason = safety_reason_counts[0].get("value") if safety_reason_counts and isinstance(safety_reason_counts[0], dict) else None
    return {
        "rank": _as_int(group.get("rank")),
        "local_action_family": public_label(group.get("local_action_family"), "unknown"),
        "candidate_count": _as_int(group.get("candidate_count")),
        "recommended_count": _as_int(group.get("recommended_count")),
        "noop_count": _as_int(group.get("noop_count")),
        "projected_savings_usd": _money(group.get("projected_savings_usd")),
        "top_next_action": public_label(group.get("top_next_action"), "unknown"),
        "top_blocker_reason": public_label(top_reason, "none") if top_reason else None,
        "top_safety_stop_reason": public_label(top_safety_reason or group.get("top_safety_stop_reason"), "none") if (top_safety_reason or group.get("top_safety_stop_reason")) else None,
        "blocker_reason_code_counts": [
            {"value": public_label(row.get("value"), "unknown"), "count": _as_int(row.get("count"))}
            for row in reason_counts[:5]
            if isinstance(row, dict)
        ],
        "safety_stop_reason_counts": [
            {"value": public_label(row.get("value"), "unknown"), "count": _as_int(row.get("count"))}
            for row in safety_reason_counts[:5]
            if isinstance(row, dict)
        ],
        "sample_recommendations": public_candidates,
    }


async def stats_promotion_blocker_next_actions(limit: int = 20) -> dict[str, Any]:
    path = _promotion_blocker_review_path()
    if not path.exists():
        return _promotion_blocker_no_data("local promotion blocker review report not found", source_path=path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result = _promotion_blocker_no_data(f"local promotion blocker review report unreadable: {exc.__class__.__name__}", source_path=path)
        result["status"] = "unavailable"
        return result
    if not isinstance(payload, dict) or payload.get("schema") != "tokenclaw.promotion_blocker_recommendation_review.v1":
        result = _promotion_blocker_no_data("local report is not a promotion blocker recommendation review", source_path=path)
        result["status"] = "unavailable"
        return result

    bounded_limit = max(0, min(_as_int(limit), 50))
    source_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    candidates = [item for item in candidates if isinstance(item, dict)]
    groups = payload.get("groups") if isinstance(payload.get("groups"), list) else []
    public_groups = [
        _promotion_blocker_public_group(group)
        for group in groups[:bounded_limit]
        if isinstance(group, dict)
    ]
    reason_counts = _promotion_blocker_reason_counts(candidates)
    family_counts: dict[str, int] = {}
    executor_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    for candidate in candidates:
        family = public_label(candidate.get("local_action_family"), "unknown")
        executor = public_label(candidate.get("expected_local_executor"), "none")
        action = public_label(candidate.get("next_action"), "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        executor_counts[executor] = executor_counts.get(executor, 0) + 1
        next_action_counts[action] = next_action_counts.get(action, 0) + 1
    top_reasons = _breakdown_from_counts(reason_counts)[:10]
    safety_counts = source_summary.get("safety_stop_reason_counts") if isinstance(source_summary.get("safety_stop_reason_counts"), list) else []
    top_reason = top_reasons[0]["value"] if top_reasons else None
    top_executors = _breakdown_from_counts(executor_counts)[:10]
    top_actions = _breakdown_from_counts(next_action_counts)[:10]
    top_candidate = candidates[0] if candidates else {}
    generated_at = payload.get("generated_at") if isinstance(payload.get("generated_at"), str) else None
    return {
        "schema": "tokenclaw.promotion_blocker_next_actions_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": "available" if candidates else "no-data",
        "status_reason": "loaded local promotion blocker review report" if candidates else "local review report has no candidates",
        "source": {
            "kind": "local-review-report",
            "configured": bool(os.getenv("TOKENCLAW_PROMOTION_BLOCKER_REVIEW_PATH")),
            "available": True,
            "generated_at": generated_at,
            "source_schema": payload.get("source_schema"),
            "path_class": _local_path_class(path),
            "path_included": False,
        },
        "summary": {
            "source_recommendation_count": _as_int(source_summary.get("source_recommendation_count")),
            "review_candidate_count": _as_int(source_summary.get("review_candidate_count")),
            "group_count": _as_int(source_summary.get("group_count")),
            "recommended_count": _as_int(source_summary.get("recommended_count")),
            "noop_count": _as_int(source_summary.get("noop_count")),
            "stale_evidence_count": _promotion_blocker_stale_count(candidates),
            "projected_savings_usd": _money(source_summary.get("projected_savings_usd")),
            "top_local_action_family": public_label(source_summary.get("top_local_action_family"), "none"),
            "top_blocker_reason": top_reason,
            "top_safety_stop_reason": public_label(source_summary.get("top_safety_stop_reason"), "none") if source_summary.get("top_safety_stop_reason") else None,
            "safety_stop_reason_count": _as_int(source_summary.get("safety_stop_reason_count")),
            "top_next_action": public_label(source_summary.get("top_next_action"), "none"),
            "top_expected_local_executor": public_label(top_candidate.get("expected_local_executor"), "none") if top_candidate else None,
        },
        "family_counts": _breakdown_from_counts(family_counts)[:10],
        "top_blocker_reasons": top_reasons,
        "top_safety_stop_reasons": [
            {"value": public_label(row.get("value"), "unknown"), "count": _as_int(row.get("count"))}
            for row in safety_counts[:10]
            if isinstance(row, dict)
        ],
        "expected_local_executors": top_executors,
        "next_actions": top_actions,
        "groups": public_groups,
        "commands": [
            {
                "label": "review local promotion blocker recommendations",
                "command": "tokenclaw-optimization-promotion-blocker-review recommendations.json --pretty",
                "read_only": True,
            },
            {
                "label": "queue local shadow eval tasks",
                "command": "tokenclaw-optimization-eval-next --promotion-blocker-review review.json --dry-run --pretty",
                "read_only": True,
            },
            {
                "label": "inspect promotion funnel",
                "command": "tokenclaw-optimization-promotion-report --pretty",
                "read_only": True,
            },
        ],
        "privacy": _promotion_blocker_dashboard_privacy(),
    }


async def stats_sqlite_maintenance(store_obj: Any) -> dict[str, Any]:
    if hasattr(store_obj, "sqlite_retention_status"):
        status = store_obj.sqlite_retention_status()
    else:
        status = {
            "schema": "tokenclaw.sqlite_retention_status.v1",
            "backend": getattr(store_obj, "backend", "unknown"),
            "enabled": False,
            "retention_days": None,
            "default_retention_days": 7,
            "last_run": None,
            "request_path_maintenance": "not-available",
        }
    return {
        "schema": "tokenclaw.sqlite_maintenance_dashboard.v1",
        "generated_at": utc_now(),
        "status": status,
        "summary": {
            "enabled": bool(status.get("enabled")),
            "retention_days": status.get("retention_days"),
            "default_retention_days": status.get("default_retention_days"),
            "last_purge_at": (status.get("last_run") or {}).get("created_at") if isinstance(status.get("last_run"), dict) else None,
            "last_deleted_rows": _as_int((status.get("last_run") or {}).get("total_deleted_rows")) if isinstance(status.get("last_run"), dict) else 0,
            "provider_request_path_delayed": False,
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "queue_payloads_included": False,
        },
    }



SCAFFOLD_ROLLOUT_ACTIONS = {
    "scaffold-rollout-actions-review",
    "scaffold-rollout-actions-apply",
}
SCAFFOLD_CANARY_POLICY_FILENAME = "scaffold_canary_policy.yaml"














def _scaffold_canary_policy_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("TOKENCLAW_SCAFFOLD_CANARY_POLICY")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "config" / SCAFFOLD_CANARY_POLICY_FILENAME)
    candidates.append(tokenclaw_config_path(SCAFFOLD_CANARY_POLICY_FILENAME))
    return candidates


def _scaffold_canary_policy_health() -> dict[str, Any]:
    selected = next((path for path in _scaffold_canary_policy_candidates() if path.exists()), None)
    if selected is None:
        return {
            "available": False,
            "status": "missing",
            "enabled": False,
            "active_rule_count": 0,
            "policy_source": None,
            "rule_path_included": False,
            "rule_path_class": _local_path_class(os.getenv("TOKENCLAW_SCAFFOLD_CANARY_POLICY") or "~/.tokenclaw/scaffold_canary_policy.yaml"),
            "yaml_contents_included": False,
        }

    try:
        raw = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    except Exception:
        return {
            "available": True,
            "status": "unreadable",
            "enabled": False,
            "active_rule_count": 0,
            "policy_source": None,
            "rule_path_included": False,
            "rule_path_class": _local_path_class(str(selected)),
            "yaml_contents_included": False,
        }

    if not isinstance(raw, dict):
        raw = {}
    provider = raw.get("repeated_provider_scaffolding") if isinstance(raw.get("repeated_provider_scaffolding"), dict) else {}
    rules = provider.get("rules") if isinstance(provider.get("rules"), list) else []
    enabled = bool(provider.get("enabled") and rules)
    active_rule_count = sum(
        1
        for rule in rules
        if isinstance(rule, dict) and rule.get("enabled", True) is not False
    ) if enabled else 0
    return {
        "available": True,
        "status": "active" if active_rule_count else "empty",
        "enabled": enabled,
        "active_rule_count": active_rule_count,
        "policy_source": raw.get("policy_source") or provider.get("policy_source"),
        "rule_path_included": False,
        "rule_path_class": _local_path_class(str(selected)),
        "yaml_contents_included": False,
    }


def _public_scaffold_rollout_event(event: dict[str, Any] | None, *, now: datetime) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    action = str(event.get("action") or "")
    fetch_status = details.get("fetch_status")
    if not fetch_status:
        fetch_status = "ok" if event.get("ok") else "failed"
    return {
        "created_at": event.get("created_at"),
        "age_seconds": _seconds_since_iso(event.get("created_at"), now),
        "action": action,
        "ok": bool(event.get("ok")),
        "source": details.get("source"),
        "fetch_status": fetch_status,
        "dry_run": bool(details.get("dry_run")),
        "action_count": _as_int(details.get("action_count")),
        "accepted_action_count": _as_int(details.get("accepted_action_count")),
        "changed_file_count": len(details.get("changed_files") if isinstance(details.get("changed_files"), list) else []),
        "error_type": details.get("error_type"),
        "exit_code": details.get("exit_code"),
        "managed_server_calls_made": details.get("url") not in (None, ""),
        "payload_included": False,
        "raw_payload_included": False,
        "file_paths_included": False,
        "yaml_contents_included": False,
    }


def _scaffold_rollout_status(
    *,
    latest_fetch: dict[str, Any] | None,
    latest_apply: dict[str, Any] | None,
    active_rule_count: int,
    safety_stop_count: int,
) -> tuple[str, str]:
    if latest_fetch and latest_fetch.get("ok") is False:
        return "fetch-blocked", "latest scaffold rollout fetch/review failed"
    if latest_apply and latest_apply.get("ok") is False:
        return "apply-blocked", "latest scaffold rollout apply failed"
    if safety_stop_count > 0:
        return "safety-stop", "recent scaffold canary evidence has safety-stop rows"
    if active_rule_count > 0:
        return "canary-active", "local scaffold canary overlay has active rules"
    if latest_fetch and latest_fetch.get("ok"):
        return "bundle-reviewed", "managed scaffold rollout actions were fetched and reviewed"
    return "local-only", "no scaffold rollout metadata found"




ROLLOUT_ACTION_STAGES = {
    "rollout-actions-review": "review",
    "rollout-actions-dry-run": "dry_run",
    "rollout-actions-impact": "impact",
    "rollout-actions-apply": "apply",
    "pattern-canary-safety-stop": "safety_stop",
}


def _nonzero(value: Any) -> int | float | None:
    number = _as_float(value)
    if number == 0:
        return None
    integer = _as_int(value)
    if float(integer) == number:
        return integer
    return round(number, 8)


def _rollout_count_breakdown(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if key is None:
            continue
        counts[str(key)] = counts.get(str(key), 0) + _as_int(value)
    return _managed_breakdown(counts)


def _rollout_details_counts(details: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_count": _as_int(details.get("action_count")),
        "planned_action_count": _as_int(details.get("planned_action_count")),
        "changed_action_count": _as_int(details.get("changed_action_count")),
        "rejected_action_count": _as_int(details.get("rejected_action_count")),
        "changed_file_count": _as_int(details.get("changed_file_count")),
        "affected_metadata_row_count": _as_int(details.get("affected_metadata_row_count")),
        "projected_affected_metadata_row_count": _as_int(details.get("projected_affected_metadata_row_count")),
        "actual_matched_metadata_row_count": _as_int(details.get("actual_matched_metadata_row_count")),
        "actual_matched_provider_call_count": _as_int(details.get("actual_matched_provider_call_count")),
        "actual_matched_codex_turn_count": _as_int(details.get("actual_matched_codex_turn_count")),
        "actual_canary_applied_count": _as_int(details.get("actual_canary_applied_count")),
        "actual_canary_holdout_count": _as_int(details.get("actual_canary_holdout_count")),
        "actual_bypassed_or_disabled_count": _as_int(details.get("actual_bypassed_or_disabled_count")),
        "actual_tokens_saved_est": _as_int(details.get("actual_tokens_saved_est")),
        "actual_estimated_cost_savings_usd": _as_float(details.get("actual_estimated_cost_savings_usd")),
        "actions_without_post_apply_matches": _as_int(details.get("actions_without_post_apply_matches")),
        "validation_error_count": _as_int(details.get("validation_error_count") or details.get("error_count")),
        "validation_warning_count": _as_int(details.get("validation_warning_count")),
        "review_error_count": _as_int(details.get("review_error_count")),
        "review_warning_count": _as_int(details.get("review_warning_count")),
    }


def _public_rollout_policy_event(event: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    stage = ROLLOUT_ACTION_STAGES.get(str(event.get("action") or ""), "unknown")
    counts = _rollout_details_counts(details)
    return {
        "created_at": event.get("created_at"),
        "age_seconds": _seconds_since_iso(event.get("created_at"), now),
        "action": event.get("action"),
        "stage": stage,
        "ok": bool(event.get("ok")),
        "dry_run": bool(details.get("dry_run") or stage == "dry_run"),
        "source": details.get("source"),
        "provenance_status": details.get("provenance_status"),
        "status_code": details.get("status_code"),
        "error_type": details.get("error_type"),
        "exit_code": details.get("exit_code"),
        "counts": {key: value for key, value in counts.items() if value not in (None, 0)},
        "payload_included": False,
        "raw_payload_included": False,
        "file_paths_included": False,
        "yaml_contents_included": False,
    }


def _rollout_lifecycle_rows(store_obj: Any, *, limit: int) -> list[dict[str, Any]]:
    if store_obj is None or not hasattr(store_obj, "conn"):
        return []
    capped = max(1, min(int(limit or 500), 5000))
    try:
        rows = store_obj.conn.execute(
            """
            select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                   payload_json, status, attempts, next_attempt_at, last_error,
                   last_status_code, sent_at
            from managed_outcome_feedback_queue
            where source_surface = ?
            order by created_at desc
            limit ?
            """,
            ("rollout_action_lifecycle", capped),
        ).fetchall()
    except Exception:
        return []
    return [dict(row) for row in rows]


def _public_rollout_lifecycle_row(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    payload = _json_obj(row.get("payload_json"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    missing: list[str] = []
    if _as_int(metadata.get("action_count")) > 0 and not isinstance(metadata.get("action_type_counts"), dict):
        missing.append("action_type_counts")
    if _as_int(metadata.get("action_count")) > 0 and not isinstance(metadata.get("policy_section_counts"), dict):
        missing.append("policy_section_counts")
    if metadata.get("affected_metadata_row_count") is None and str(payload.get("event_type") or "") == "dry-run":
        missing.append("affected_metadata_row_count")

    projected = {
        "affected_metadata_row_count": _nonzero(metadata.get("affected_metadata_row_count")),
        "affected_provider_call_count": _nonzero(metadata.get("affected_provider_call_count")),
        "affected_codex_turn_count": _nonzero(metadata.get("affected_codex_turn_count")),
        "projected_additional_applied_count": _nonzero(metadata.get("projected_additional_applied_count")),
        "projected_local_bypass_or_disable_count": _nonzero(metadata.get("projected_local_bypass_or_disable_count")),
        "historical_tokens_saved_est": _nonzero(metadata.get("historical_tokens_saved_est")),
        "historical_estimated_cost_savings_usd": _nonzero(metadata.get("historical_estimated_cost_savings_usd")),
    }
    return {
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "age_seconds": _seconds_since_iso(row.get("created_at"), now),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "feedback_status": row.get("status"),
        "attempts": _as_int(row.get("attempts")),
        "next_attempt_at": row.get("next_attempt_at"),
        "sent_at": row.get("sent_at"),
        "last_status_code": row.get("last_status_code"),
        "last_error_class": _managed_feedback_error_class(row),
        "event_type": payload.get("event_type"),
        "occurred_at": payload.get("occurred_at"),
        "command": metadata.get("command"),
        "local_result_status": metadata.get("local_result_status"),
        "dry_run": bool(metadata.get("dry_run")),
        "read_only": bool(metadata.get("read_only")),
        "action_count": _as_int(metadata.get("action_count")),
        "planned_action_count": _as_int(metadata.get("planned_action_count")),
        "changed_action_count": _as_int(metadata.get("changed_action_count")),
        "rejected_action_count": _as_int(metadata.get("rejected_action_count")),
        "action_type_counts": _rollout_count_breakdown(metadata.get("action_type_counts")),
        "policy_section_counts": _rollout_count_breakdown(metadata.get("policy_section_counts")),
        "local_status_counts": _rollout_count_breakdown(metadata.get("local_status_counts")),
        "validation_error_count": _as_int(metadata.get("validation_error_count")),
        "validation_warning_count": _as_int(metadata.get("validation_warning_count")),
        "review_error_count": _as_int(metadata.get("review_error_count")),
        "review_warning_count": _as_int(metadata.get("review_warning_count")),
        "changed_file_count": _as_int(metadata.get("changed_file_count")),
        "projected_impact": {key: value for key, value in projected.items() if value is not None},
        "safety_stop_reason_counts": _rollout_count_breakdown(metadata.get("safety_stop_reason_counts")),
        "missing_metadata": missing,
        "bundle_hash_present": bool(payload.get("bundle_hash") or metadata.get("computed_bundle_hash") or metadata.get("provenance_bundle_hash")),
        "payload_included": False,
        "raw_payload_included": False,
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
            "cache_keys_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
        },
    }


def _rollout_feedback_queue_summary(rows: list[dict[str, Any]], *, now: datetime) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    pending_count = 0
    due_count = 0
    oldest_due_age: int | None = None
    for row in rows:
        status = str(row.get("status") or "unknown")
        _increment_count(status_counts, status)
        _increment_count(endpoint_counts, str(row.get("endpoint") or "unknown"))
        if status in {"queued", "retryable-error"}:
            pending_count += 1
            next_attempt = _parse_utc_datetime(row.get("next_attempt_at"))
            if next_attempt is None or next_attempt <= now:
                due_count += 1
                age = _seconds_since_iso(row.get("next_attempt_at") or row.get("created_at"), now)
                if age is not None:
                    oldest_due_age = age if oldest_due_age is None else max(oldest_due_age, age)
    return {
        "available": True,
        "summary": {
            "total": len(rows),
            "pending": pending_count,
            "due": due_count,
            "queued": status_counts.get("queued", 0),
            "retryable_error": status_counts.get("retryable-error", 0),
            "sent": status_counts.get("sent", 0),
            "dropped_after_limit": status_counts.get("dropped-after-limit", 0),
            "oldest_due_age_seconds": oldest_due_age,
        },
        "status_breakdown": _managed_breakdown(status_counts),
        "endpoint_breakdown": _managed_breakdown(endpoint_counts),
        "payload_included": False,
    }


def _latest_rollout_event(events: list[dict[str, Any]], action: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("action") == action:
            return event
    return None


def _latest_lifecycle(
    public_rows: list[dict[str, Any]],
    event_types: set[str],
    commands: set[str] | None = None,
) -> dict[str, Any] | None:
    for row in public_rows:
        if str(row.get("event_type") or "") in event_types and (
            commands is None or str(row.get("command") or "") in commands
        ):
            return row
    return None


def _rollout_safety_stop_state(events: list[dict[str, Any]], public_rows: list[dict[str, Any]], *, now: datetime) -> dict[str, Any]:
    latest_event = _latest_rollout_event(events, "pattern-canary-safety-stop")
    latest_public = _public_rollout_policy_event(latest_event, now=now) if latest_event else None
    reason_counts: dict[str, int] = {}
    if latest_event:
        details = latest_event.get("details") if isinstance(latest_event.get("details"), dict) else {}
        reason = details.get("reason")
        if reason:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    for row in public_rows:
        for item in row.get("safety_stop_reason_counts") or []:
            reason_counts[str(item.get("value") or "unknown")] = reason_counts.get(str(item.get("value") or "unknown"), 0) + _as_int(item.get("count"))
    return {
        "active": bool(latest_event or reason_counts),
        "latest": latest_public,
        "reason_breakdown": _managed_breakdown(reason_counts),
        "payload_included": False,
    }


def _next_rollout_read_only_command(
    *,
    latest_review: dict[str, Any] | None,
    latest_dry_run: dict[str, Any] | None,
) -> str | None:
    if not latest_review or not latest_review.get("ok", True):
        return "tokenclaw-managed-rollout-actions-review actions.json --pretty"
    if not latest_dry_run or not latest_dry_run.get("ok", True):
        return "tokenclaw-managed-rollout-actions-dry-run actions.json --db ~/.tokenclaw/tokenclaw.sqlite3 --pretty"
    return "tokenclaw-managed-rollout-actions-apply actions.json --config-dir ~/.tokenclaw --dry-run --pretty"


async def stats_rollout_actions_readiness(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.policy_events import recent_policy_events

    capped_limit = max(1, min(int(limit or 500), 5000))
    now = datetime.now(timezone.utc)
    events = [
        event
        for event in recent_policy_events(limit=500).get("events", [])
        if isinstance(event, dict) and str(event.get("action") or "") in ROLLOUT_ACTION_STAGES
    ]
    public_events = [_public_rollout_policy_event(event, now=now) for event in events]
    lifecycle_rows = _rollout_lifecycle_rows(store_obj, limit=capped_limit)
    public_lifecycle = [_public_rollout_lifecycle_row(row, now=now) for row in lifecycle_rows]

    review_event = _latest_rollout_event(events, "rollout-actions-review")
    dry_run_event = _latest_rollout_event(events, "rollout-actions-dry-run")
    impact_event = _latest_rollout_event(events, "rollout-actions-impact")
    apply_event = _latest_rollout_event(events, "rollout-actions-apply")
    latest_review = _latest_lifecycle(public_lifecycle, {"reviewed", "rejected"}, {"rollout-actions-review"}) or (
        _public_rollout_policy_event(review_event, now=now) if review_event else None
    )
    latest_dry_run = _latest_lifecycle(public_lifecycle, {"dry-run"}, {"rollout-actions-dry-run"}) or (
        _public_rollout_policy_event(dry_run_event, now=now) if dry_run_event else None
    )
    latest_apply_or_rollback = _latest_lifecycle(
        public_lifecycle,
        {"applied", "rollback", "rejected"},
        {"rollout-actions-apply"},
    ) or (
        _public_rollout_policy_event(apply_event, now=now) if apply_event else None
    )
    latest_impact = _public_rollout_policy_event(impact_event, now=now) if impact_event else None
    latest_lifecycle = public_lifecycle[0] if public_lifecycle else None
    action_counts = (latest_lifecycle or {}).get("action_type_counts") or []
    latest_projected = (latest_dry_run or {}).get("projected_impact") if isinstance(latest_dry_run, dict) else None
    if not isinstance(latest_projected, dict):
        latest_projected = ((latest_dry_run or {}).get("counts") if isinstance(latest_dry_run, dict) else {}) or {}
    feedback_queue = _rollout_feedback_queue_summary(lifecycle_rows, now=now)

    warning_count = 0
    if isinstance(latest_lifecycle, dict):
        warning_count += _as_int(latest_lifecycle.get("validation_warning_count"))
        warning_count += _as_int(latest_lifecycle.get("review_warning_count"))
    if isinstance(latest_review, dict):
        warning_count += _as_int((latest_review.get("counts") or {}).get("validation_warning_count"))
        warning_count += _as_int((latest_review.get("counts") or {}).get("review_warning_count"))

    missing_metadata = sorted(
        {
            str(item)
            for row in public_lifecycle[:10]
            for item in (row.get("missing_metadata") or [])
            if item
        }
    )
    ready = bool(
        latest_review
        and latest_review.get("ok", True)
        and latest_dry_run
        and latest_dry_run.get("ok", True)
        and not _as_int(feedback_queue.get("summary", {}).get("due"))
    )
    return {
        "schema": "tokenclaw.rollout_actions_readiness.v1",
        "generated_at": utc_now(),
        "status": "ready" if ready else "needs-review",
        "limit": capped_limit,
        "summary": {
            "policy_event_count": len(public_events),
            "lifecycle_feedback_count": len(public_lifecycle),
            "latest_action_count": sum(_as_int(row.get("count")) for row in action_counts),
            "latest_warning_count": warning_count,
            "pending_lifecycle_feedback_count": _as_int(feedback_queue.get("summary", {}).get("pending")),
            "due_lifecycle_feedback_count": _as_int(feedback_queue.get("summary", {}).get("due")),
            "affected_metadata_row_count": _as_int(latest_projected.get("affected_metadata_row_count")),
            "projected_additional_applied_count": _as_int(latest_projected.get("projected_additional_applied_count")),
            "projected_local_bypass_or_disable_count": _as_int(latest_projected.get("projected_local_bypass_or_disable_count")),
            "historical_tokens_saved_est": _as_int(latest_projected.get("historical_tokens_saved_est")),
            "historical_estimated_cost_savings_usd": _as_float(latest_projected.get("historical_estimated_cost_savings_usd")),
            "missing_metadata_count": len(missing_metadata),
        },
        "latest_review": latest_review,
        "latest_dry_run": latest_dry_run,
        "latest_impact": latest_impact,
        "latest_apply_or_rollback": latest_apply_or_rollback,
        "latest_lifecycle_feedback": latest_lifecycle,
        "action_type_counts": action_counts,
        "policy_section_counts": (latest_lifecycle or {}).get("policy_section_counts") or [],
        "local_status_counts": (latest_lifecycle or {}).get("local_status_counts") or [],
        "dry_run_impact": latest_projected,
        "post_apply_impact": (latest_impact or {}).get("counts") or {},
        "missing_metadata": missing_metadata,
        "safety_stop": _rollout_safety_stop_state(events, public_lifecycle, now=now),
        "lifecycle_feedback_queue": feedback_queue,
        "recent_events": public_events[:25],
        "recent_lifecycle_feedback": public_lifecycle[:25],
        "next_read_only_command": _next_rollout_read_only_command(
            latest_review=latest_review,
            latest_dry_run=latest_dry_run,
        ),
        "privacy": {
            "metadata_only": True,
            "raw_action_payloads_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
            "local_session_ids_included": False,
            "payload_json_included": False,
            "basis": "local policy-event metadata plus queued rollout lifecycle feedback aggregates only",
        },
    }


def _source_surface(provider: str, path: str) -> str:
    provider_l = (provider or "").lower()
    path_l = (path or "").lower()
    if provider_l in {"codex-app", "codex_app"}:
        return CODEX_APP_SOURCE_SURFACE
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        if "chat/completions" in path_l:
            return "openai_chat"
        return "openai_responses"
    return "unknown"


def _endpoint_label(provider: str, path: str) -> str:
    provider_l = (provider or "").lower()
    path_l = (path or "").lower()
    if provider_l in {"codex-app", "codex_app"}:
        return "codex_app_turn"
    if "chat/completions" in path_l:
        return "chat_completions"
    if "responses" in path_l:
        return "responses"
    if "messages" in path_l:
        return "messages"
    return "unknown"


def _app_family_for_call(provider: str, requested_model: Any, path: str) -> str:
    provider_l = (provider or "").lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in (path or "").lower():
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _money(value: Any) -> float:
    return round(_as_float(value), 8)


def estimate_tokens_from_text_chars(chars: Any) -> int:
    char_count = max(_as_int(chars), 0)
    if char_count <= 0:
        return 0
    return max(1, int(char_count / TOKEN_CHARS))


def _old_context_summary_skip_reason(meta: dict[str, Any]) -> str:
    reason = str(meta.get("reason") or "unknown")
    if reason == "eligible-context-too-small" and _as_int(meta.get("eligible_turns")) <= 0:
        return "tool/protocol-context-only"
    return reason


def _old_context_summary_tokens_saved(meta: dict[str, Any], *, planned: bool) -> int:
    tokens = _as_int(meta.get("tokens_saved_est"))
    if tokens > 0:
        return tokens
    if not planned:
        return 0
    try:
        from tokenclaw import crunch

        max_summary_chars = _as_int(getattr(crunch, "OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS", 4000))
    except Exception:
        max_summary_chars = 4000
    eligible_chars = _as_int(meta.get("eligible_chars"))
    if eligible_chars <= max_summary_chars:
        return 0
    return max(0, (eligible_chars - max_summary_chars) // TOKEN_CHARS)


def _old_context_summary_call_cost(meta: dict[str, Any], *, planned: bool) -> tuple[int, int, float]:
    recorded_cost = _as_float(meta.get("summary_cost_est_usd"))
    recorded_input = _as_int(meta.get("summary_input_tokens"))
    recorded_output = _as_int(meta.get("summary_output_tokens"))
    if recorded_cost > 0 or recorded_input > 0 or recorded_output > 0:
        return recorded_input, recorded_output, recorded_cost
    if not planned:
        return 0, 0, 0.0
    try:
        from tokenclaw import crunch

        model = str(getattr(crunch, "OLD_CONTEXT_SUMMARY_MODEL", "claude-haiku-4-5-20251001"))
        max_summary_chars = _as_int(getattr(crunch, "OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS", 4000))
    except Exception:
        model = "claude-haiku-4-5-20251001"
        max_summary_chars = 4000
    input_tokens = estimate_tokens_from_text_chars(meta.get("eligible_chars"))
    output_tokens = max(256, max_summary_chars // TOKEN_CHARS)
    return input_tokens, output_tokens, estimate_cost(model, input_tokens, output_tokens, provider="anthropic") or 0.0


def _old_context_summary_quality_cohort(meta: dict[str, Any]) -> str:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    cohort = str(canary.get("cohort") or canary.get("status") or "")
    status = str(meta.get("status") or "")
    reason = str(meta.get("reason") or "")
    if cohort == "canary_applied" or status == "applied":
        return "canary_applied"
    if cohort == "canary_holdout" or reason == "canary_holdout":
        return "canary_holdout"
    if status in {"bypass", "disabled"} or reason in {"disabled", "safety-stop"} or "safety-stop" in reason:
        return "bypassed_or_disabled"
    if status == "skipped":
        return "bypassed_or_disabled"
    return "unknown"


def _old_context_summary_failed(meta: dict[str, Any]) -> bool:
    return (
        _as_int(meta.get("summary_status_code")) >= 400
        or bool(meta.get("summary_error"))
        or str(meta.get("status") or "") in {"summary_failed", "error"}
        or str(meta.get("reason") or "") in {"summary-error", "summary-model-error"}
    )


def _old_context_summary_safety_stopped(meta: dict[str, Any]) -> bool:
    safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
    return str(meta.get("safety_stop_state") or "") == "stopped" or bool(safety.get("stopped"))


def _old_context_summary_quality_thresholds(meta: dict[str, Any]) -> dict[str, Any]:
    gate = meta.get("quality_gate") if isinstance(meta.get("quality_gate"), dict) else {}
    gate_thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
    safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    min_samples = _as_int(
        gate_thresholds.get("min_matched_samples")
        or safety.get("min_matched_samples")
        or safety.get("min_outcome_samples")
        or 5
    )
    min_applied = _as_int(
        gate_thresholds.get("min_canary_applied_samples")
        or safety.get("min_canary_applied_samples")
        or max(1, min_samples // 2)
    )
    min_holdout = _as_int(
        gate_thresholds.get("min_canary_holdout_samples")
        or safety.get("min_canary_holdout_samples")
        or (max(1, min_samples // 2) if bool(canary.get("enabled")) else 0)
    )
    return {
        "min_matched_samples": min_samples,
        "min_canary_applied_samples": min_applied,
        "min_canary_holdout_samples": min_holdout,
        "min_net_savings_usd": round(_as_float(gate_thresholds.get("min_net_savings_usd")), 8),
        "min_payback_ratio": round(_as_float(gate_thresholds.get("min_payback_ratio") or 1.0), 6),
        "max_error_rate": round(_as_float(gate_thresholds.get("max_error_rate") or safety.get("max_error_rate") or 0.1), 6),
        "max_error_rate_delta": round(_as_float(gate_thresholds.get("max_error_rate_delta") or safety.get("max_error_rate_delta") or 0.05), 6),
        "max_retry_rate": round(_as_float(gate_thresholds.get("max_retry_rate") or safety.get("max_retry_rate") or 0.25), 6),
        "max_retry_rate_delta": round(_as_float(gate_thresholds.get("max_retry_rate_delta") or 0.05), 6),
        "max_summary_failure_rate": round(
            _as_float(gate_thresholds.get("max_summary_failure_rate") or safety.get("max_summary_failure_rate") or 0.1),
            6,
        ),
        "max_safety_stop_count": _as_int(gate_thresholds.get("max_safety_stop_count")),
        "max_latency_regression_ms": _as_int(gate_thresholds.get("max_latency_regression_ms") or 2000),
        "rollback_error_rate": round(_as_float(gate_thresholds.get("rollback_error_rate") or 0.4), 6),
        "rollback_summary_failure_rate": round(_as_float(gate_thresholds.get("rollback_summary_failure_rate") or 0.2), 6),
        "rollback_safety_stop_count": _as_int(gate_thresholds.get("rollback_safety_stop_count") or 1),
        "rollback_negative_net_savings_usd": round(_as_float(gate_thresholds.get("rollback_negative_net_savings_usd")), 8),
    }


def _new_old_context_summary_quality_bucket(row: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    return {
        "candidate_id": meta.get("candidate_id"),
        "rule_id": meta.get("rule_id"),
        "policy_source": meta.get("policy_source"),
        "summary_model": meta.get("model"),
        "canary_fraction": canary.get("fraction"),
        "canary_unit": canary.get("unit"),
        "canary_enabled": bool(canary.get("enabled")),
        "last_decision_at": row.get("created_at"),
        "enabled_rows": 0,
        "disabled_rows": 0,
        "matched_metadata_row_count": 0,
        "canary_applied_count": 0,
        "canary_holdout_count": 0,
        "bypassed_or_disabled_count": 0,
        "unknown_cohort_count": 0,
        "summary_failure_count": 0,
        "safety_stop_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "actual_tokens_saved_est": 0,
        "actual_gross_savings_usd": 0.0,
        "actual_summary_model_cost_usd": 0.0,
        "actual_net_savings_usd": 0.0,
        "cohorts": {
            "canary_applied": {"count": 0, "error_count": 0, "retry_count": 0, "summary_failure_count": 0, "safety_stop_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "canary_holdout": {"count": 0, "error_count": 0, "retry_count": 0, "summary_failure_count": 0, "safety_stop_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "bypassed_or_disabled": {"count": 0, "error_count": 0, "retry_count": 0, "summary_failure_count": 0, "safety_stop_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "unknown": {"count": 0, "error_count": 0, "retry_count": 0, "summary_failure_count": 0, "safety_stop_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
        },
        "thresholds": _old_context_summary_quality_thresholds(meta),
    }


def _finalize_old_context_summary_quality_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    matched = _as_int(bucket.get("matched_metadata_row_count"))
    applied = _as_int(bucket.get("canary_applied_count"))
    holdout = _as_int(bucket.get("canary_holdout_count"))
    disabled_rows = _as_int(bucket.get("disabled_rows"))
    gross = _as_float(bucket.get("actual_gross_savings_usd"))
    summary_cost = _as_float(bucket.get("actual_summary_model_cost_usd"))
    net = _as_float(bucket.get("actual_net_savings_usd"))
    thresholds = bucket["thresholds"]

    cohorts: dict[str, Any] = {}
    for name, cohort in bucket["cohorts"].items():
        count = _as_int(cohort.get("count"))
        latency_samples = _as_int(cohort.get("latency_sample_count"))
        latency_avg = (
            round(_as_int(cohort.get("latency_ms_total")) / latency_samples, 2)
            if latency_samples
            else None
        )
        cohorts[name] = {
            "count": count,
            "error_count": _as_int(cohort.get("error_count")),
            "retry_count": _as_int(cohort.get("retry_count")),
            "summary_failure_count": _as_int(cohort.get("summary_failure_count")),
            "safety_stop_count": _as_int(cohort.get("safety_stop_count")),
            "error_rate": round(_as_int(cohort.get("error_count")) / count, 6) if count else 0.0,
            "retry_rate": round(_as_int(cohort.get("retry_count")) / count, 6) if count else 0.0,
            "summary_failure_rate": round(_as_int(cohort.get("summary_failure_count")) / count, 6) if count else 0.0,
            "latency_avg_ms": latency_avg,
        }

    applied_cohort = cohorts["canary_applied"]
    holdout_cohort = cohorts["canary_holdout"]
    latency_delta = None
    if applied_cohort["latency_avg_ms"] is not None and holdout_cohort["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied_cohort["latency_avg_ms"]) - _as_float(holdout_cohort["latency_avg_ms"]), 2)
    error_delta = round(_as_float(applied_cohort["error_rate"]) - _as_float(holdout_cohort["error_rate"]), 6)
    retry_delta = round(_as_float(applied_cohort["retry_rate"]) - _as_float(holdout_cohort["retry_rate"]), 6)
    payback_ratio = round(gross / summary_cost, 6) if summary_cost > 0 else None

    blockers: list[str] = []
    warnings: list[str] = []
    rollback_reasons: list[str] = []
    if matched <= 0:
        blockers.append("no-observed-rows")
    if matched < _as_int(thresholds["min_matched_samples"]):
        blockers.append("insufficient-matched-samples")
    if applied < _as_int(thresholds["min_canary_applied_samples"]):
        blockers.append("insufficient-canary-applied-samples")
    if bucket.get("canary_enabled") and holdout < _as_int(thresholds["min_canary_holdout_samples"]):
        blockers.append("insufficient-canary-holdout-samples")
    if matched and net <= _as_float(thresholds["min_net_savings_usd"]):
        blockers.append("non-positive-net-savings")
    if payback_ratio is not None and payback_ratio < _as_float(thresholds["min_payback_ratio"]):
        blockers.append("summary-cost-payback-below-threshold")
    if _as_float(applied_cohort["error_rate"]) > _as_float(thresholds["max_error_rate"]):
        blockers.append("applied-error-rate-above-threshold")
    if error_delta > _as_float(thresholds["max_error_rate_delta"]):
        blockers.append("applied-error-rate-regression")
    if _as_float(applied_cohort["retry_rate"]) > _as_float(thresholds["max_retry_rate"]):
        blockers.append("applied-retry-rate-above-threshold")
    if retry_delta > _as_float(thresholds["max_retry_rate_delta"]):
        blockers.append("applied-retry-rate-regression")
    if applied and _as_float(applied_cohort["summary_failure_rate"]) > _as_float(thresholds["max_summary_failure_rate"]):
        blockers.append("summary-failure-rate-above-threshold")
    if _as_int(bucket.get("safety_stop_count")) > _as_int(thresholds["max_safety_stop_count"]):
        blockers.append("safety-stop-events-present")
    if latency_delta is not None and latency_delta > _as_int(thresholds["max_latency_regression_ms"]):
        warnings.append("latency-regression-above-threshold")

    if _as_float(applied_cohort["error_rate"]) >= _as_float(thresholds["rollback_error_rate"]):
        rollback_reasons.append("rollback-error-rate")
    if applied and _as_float(applied_cohort["summary_failure_rate"]) >= _as_float(thresholds["rollback_summary_failure_rate"]):
        rollback_reasons.append("rollback-summary-failure-rate")
    if _as_int(bucket.get("safety_stop_count")) >= _as_int(thresholds["rollback_safety_stop_count"]):
        rollback_reasons.append("rollback-safety-stop")
    if net < -abs(_as_float(thresholds["rollback_negative_net_savings_usd"])):
        rollback_reasons.append("rollback-negative-net-savings")

    if matched == 0 and disabled_rows:
        verdict = "disabled"
        reason_codes = ["old-context-summary-disabled"]
    elif rollback_reasons:
        verdict = "rollback"
        reason_codes = rollback_reasons
    elif any(code.startswith("insufficient-") for code in blockers) or "no-observed-rows" in blockers:
        verdict = "insufficient-evidence"
        reason_codes = blockers
    elif blockers:
        verdict = "hold"
        reason_codes = blockers
    else:
        verdict = "promote"
        reason_codes = ["quality-gate-passed"]

    return {
        "schema": "tokenclaw.old_context_summary_dashboard_quality_gate.v1",
        "candidate_id": bucket.get("candidate_id"),
        "rule_id": bucket.get("rule_id"),
        "policy_source": bucket.get("policy_source"),
        "summary_model": bucket.get("summary_model"),
        "canary_fraction": bucket.get("canary_fraction"),
        "canary_unit": bucket.get("canary_unit"),
        "canary_enabled": bool(bucket.get("canary_enabled")),
        "last_decision_at": bucket.get("last_decision_at"),
        "verdict": verdict,
        "reason_codes": reason_codes,
        "warning_codes": warnings,
        "thresholds": thresholds,
        "metrics": {
            "matched_metadata_row_count": matched,
            "canary_applied_count": applied,
            "canary_holdout_count": holdout,
            "bypassed_or_disabled_count": _as_int(bucket.get("bypassed_or_disabled_count")),
            "unknown_cohort_count": _as_int(bucket.get("unknown_cohort_count")),
            "summary_failure_count": _as_int(bucket.get("summary_failure_count")),
            "safety_stop_count": _as_int(bucket.get("safety_stop_count")),
            "error_count": _as_int(bucket.get("error_count")),
            "retry_count": _as_int(bucket.get("retry_count")),
            "actual_tokens_saved_est": _as_int(bucket.get("actual_tokens_saved_est")),
            "actual_gross_savings_usd": round(gross, 8),
            "actual_summary_model_cost_usd": round(summary_cost, 8),
            "actual_net_savings_usd": round(net, 8),
            "payback_ratio": payback_ratio,
            "applied_minus_holdout_error_rate": error_delta,
            "applied_minus_holdout_retry_rate": retry_delta,
            "applied_minus_holdout_latency_avg_ms": latency_delta,
        },
        "cohorts": cohorts,
        "read_only": True,
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _old_context_summary_rollout_status(
    *,
    summary: dict[str, Any],
    policy: dict[str, Any],
    quality_gate_summary: dict[str, Any],
    canary_applied_rows: int,
    canary_holdout_rows: int,
    safety_stop_rows: int,
) -> str:
    observed = _as_int(summary.get("observed_rows"))
    if not bool(policy.get("enabled")) and observed <= 0:
        return "disabled"
    if observed <= 0:
        return "not-deployed-yet"
    if safety_stop_rows > 0 or _as_int(quality_gate_summary.get("rollback_count")) > 0:
        return "safety-stopped"
    if _as_int(summary.get("applied_rows")) <= 0 and canary_holdout_rows > 0:
        return "no-applied-canary-rows"
    if canary_applied_rows > 0 and canary_holdout_rows > 0:
        return "canary-observed"
    if _as_int(summary.get("applied_rows")) > 0:
        return "applied-observed"
    if _as_int(summary.get("planned_rows")) > 0:
        return "planned-only"
    return "observed-no-rollout"


def _old_context_summary_policy_health(policy_state: dict[str, Any]) -> dict[str, Any]:
    crunch_state = policy_state.get("crunch") if isinstance(policy_state.get("crunch"), dict) else {}
    policy = crunch_state.get("old_context_summarization")
    if not isinstance(policy, dict):
        policy = {}
    file_state = crunch_state.get("file") if isinstance(crunch_state.get("file"), dict) else {}
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    safety_stop = policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {}
    return {
        "enabled": bool(policy.get("enabled")),
        "policy_source": policy.get("policy_source") or crunch_state.get("policy_source"),
        "rule_id": policy.get("rule_id"),
        "candidate_id": policy.get("candidate_id"),
        "summary_model": policy.get("model"),
        "rule_path": crunch_state.get("rule_path"),
        "reload_required": bool(file_state.get("reload_required")),
        "loaded_at": file_state.get("loaded_at"),
        "canary": {
            "enabled": bool(canary.get("enabled")),
            "fraction": _as_float(canary.get("fraction")),
            "unit": canary.get("unit"),
        },
        "safety_stop": {
            "enabled": bool(safety_stop.get("enabled")),
            "window": _as_int(safety_stop.get("window")),
            "min_outcome_samples": _as_int(safety_stop.get("min_outcome_samples")),
            "max_error_rate": _as_float(safety_stop.get("max_error_rate")),
            "max_retry_rate": _as_float(safety_stop.get("max_retry_rate")),
            "max_summary_failure_rate": _as_float(safety_stop.get("max_summary_failure_rate")),
        },
    }


def _breakdown_count(rows: list[dict[str, Any]], value: str) -> int:
    for row in rows:
        if str(row.get("value") or "") == value:
            return _as_int(row.get("count"))
    return 0
















async def stats_old_context_summary(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
    today_start = _utc_today_start_iso()
    rows = [
        dict(row)
        for row in conn.execute("""
            select created_at,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model, requested_model) as model,
                   coalesce(actual_input_tokens, input_tokens_est, 0) as input_tokens,
                   coalesce(cache_read_input_tokens, 0) as cache_read_tokens,
                   status_code,
                   latency_ms,
                   retry_count,
                   session_id,
                   coalesce(category, 'unknown') as category,
                   routing_json,
                   crunch_json
            from calls
            where crunch_json is not null
        """).fetchall()
    ]

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    model_rows: dict[tuple[str, str], dict[str, Any]] = {}
    observed_rows = 0
    today_observed_rows = 0
    eligible_count = 0
    today_eligible_count = 0
    ineligible_count = 0
    skipped_count = 0
    planned_count = 0
    applied_count = 0
    today_applied_count = 0
    summary_created_count = 0
    summary_cache_hits = 0
    summary_empty_count = 0
    error_count = 0
    gross_saved_tokens = 0
    today_gross_saved_tokens = 0
    gross_savings_usd = 0.0
    today_gross_savings_usd = 0.0
    summary_input_tokens = 0
    summary_output_tokens = 0
    summary_cost_usd = 0.0
    today_summary_cost_usd = 0.0
    quality_gate_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    readiness_state_counts: dict[str, int] = {
        "disabled": 0,
        "eligible": 0,
        "applied": 0,
        "holdout": 0,
        "safety_stop": 0,
        "rollback": 0,
    }
    affected_sessions: set[str] = set()
    plateau_text_chars: list[int] = []
    category_rows: dict[str, dict[str, Any]] = {}

    for row in rows:
        crunch = _json_obj(row.get("crunch_json"))
        meta = crunch.get("old_context_summarization")
        if not isinstance(meta, dict) or not meta.get("status"):
            continue
        observed_rows += 1
        is_today = str(row.get("created_at") or "") >= today_start
        if is_today:
            today_observed_rows += 1
        status = str(meta.get("status") or "unknown")
        reason = str(meta.get("reason") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "skipped":
            skipped_count += 1
            normalized_reason = _old_context_summary_skip_reason(meta)
            reason_counts[normalized_reason] = reason_counts.get(normalized_reason, 0) + 1
        elif reason and reason != "eligible":
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        is_planned = status == "planned"
        is_applied = status == "applied"
        is_eligible = is_planned or is_applied
        if is_eligible:
            eligible_count += 1
            if is_today:
                today_eligible_count += 1
        else:
            ineligible_count += 1
        if is_planned:
            planned_count += 1
        if is_applied:
            applied_count += 1
            if is_today:
                today_applied_count += 1
        if reason == "summary-created":
            summary_created_count += 1
        if bool(meta.get("summary_cache_hit")):
            summary_cache_hits += 1
        if reason == "summary-empty":
            summary_empty_count += 1
        if _as_int(meta.get("summary_status_code")) >= 400 or meta.get("summary_error"):
            error_count += 1

        canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
        cohort = _old_context_summary_quality_cohort(meta)
        safety_stopped = _old_context_summary_safety_stopped(meta)
        disabled_state = (
            status == "disabled"
            or reason == "disabled"
            or (status == "skipped" and not bool(meta.get("enabled")) and reason in {"disabled", "unknown"})
        )
        readiness_state_counts["disabled"] += int(disabled_state)
        readiness_state_counts["eligible"] += int(is_eligible)
        readiness_state_counts["applied"] += int(is_applied)
        readiness_state_counts["holdout"] += int(cohort == "canary_holdout")
        readiness_state_counts["safety_stop"] += int(safety_stopped)

        tokens_saved = _old_context_summary_tokens_saved(meta, planned=is_planned)
        input_tokens, output_tokens, summary_cost = _old_context_summary_call_cost(meta, planned=is_planned)
        provider = str(row.get("provider") or "anthropic")
        model = str(row.get("model") or "unknown")
        gross_savings = estimate_blended_input_savings(
            model,
            tokens_saved=tokens_saved,
            input_tokens=_as_int(row.get("input_tokens")),
            cache_read_tokens=_as_int(row.get("cache_read_tokens")),
            provider=provider,
        ) or 0.0
        gross_saved_tokens += tokens_saved
        gross_savings_usd += gross_savings
        summary_input_tokens += input_tokens
        summary_output_tokens += output_tokens
        summary_cost_usd += summary_cost
        if is_today:
            today_gross_saved_tokens += tokens_saved
            today_gross_savings_usd += gross_savings
            today_summary_cost_usd += summary_cost

        session_id = str(row.get("session_id") or "")
        if session_id:
            affected_sessions.add(session_id)
        routing = _json_obj(row.get("routing_json"))
        text_chars = _as_int(
            meta.get("eligible_chars")
            or routing.get("text_chars")
            or (_as_int(row.get("input_tokens")) * TOKEN_CHARS)
        )
        if text_chars > 0:
            plateau_text_chars.append(text_chars)
        category = str(meta.get("category") or row.get("category") or "unknown")
        category_bucket = category_rows.setdefault(
            category,
            {
                "category": category,
                "observed_rows": 0,
                "eligible_rows": 0,
                "applied_rows": 0,
                "holdout_rows": 0,
                "disabled_rows": 0,
                "safety_stop_rows": 0,
                "projected_saved_tokens_est": 0,
                "projected_gross_savings_usd": 0.0,
                "projected_summary_cost_usd": 0.0,
                "projected_net_savings_usd": 0.0,
                "applied_saved_tokens_est": 0,
                "applied_gross_savings_usd": 0.0,
                "applied_summary_cost_usd": 0.0,
                "applied_net_savings_usd": 0.0,
            },
        )
        category_bucket["observed_rows"] += 1
        category_bucket["eligible_rows"] += int(is_eligible)
        category_bucket["applied_rows"] += int(is_applied)
        category_bucket["holdout_rows"] += int(cohort == "canary_holdout")
        category_bucket["disabled_rows"] += int(disabled_state)
        category_bucket["safety_stop_rows"] += int(safety_stopped)
        projected = is_eligible or cohort == "canary_holdout"
        if projected:
            projected_tokens = _old_context_summary_tokens_saved(meta, planned=not is_applied)
            projected_input, projected_output, projected_cost = _old_context_summary_call_cost(meta, planned=not is_applied)
            projected_gross = estimate_blended_input_savings(
                model,
                tokens_saved=projected_tokens,
                input_tokens=_as_int(row.get("input_tokens")),
                cache_read_tokens=_as_int(row.get("cache_read_tokens")),
                provider=provider,
            ) or 0.0
            category_bucket["projected_saved_tokens_est"] += projected_tokens
            category_bucket["projected_gross_savings_usd"] += projected_gross
            category_bucket["projected_summary_cost_usd"] += projected_cost
            category_bucket["projected_net_savings_usd"] += projected_gross - projected_cost
        if is_applied:
            category_bucket["applied_saved_tokens_est"] += tokens_saved
            category_bucket["applied_gross_savings_usd"] += gross_savings
            category_bucket["applied_summary_cost_usd"] += summary_cost
            category_bucket["applied_net_savings_usd"] += gross_savings - summary_cost

        quality_key = (
            str(meta.get("candidate_id") or "local-old-context-summary"),
            str(meta.get("rule_id") or "unknown"),
            str(meta.get("policy_source") or "unknown"),
        )
        quality_bucket = quality_gate_rows.setdefault(quality_key, _new_old_context_summary_quality_bucket(row, meta))
        if str(row.get("created_at") or "") > str(quality_bucket.get("last_decision_at") or ""):
            quality_bucket["last_decision_at"] = row.get("created_at")
        if quality_bucket.get("candidate_id") is None and meta.get("candidate_id") is not None:
            quality_bucket["candidate_id"] = meta.get("candidate_id")
        if quality_bucket.get("rule_id") is None and meta.get("rule_id") is not None:
            quality_bucket["rule_id"] = meta.get("rule_id")
        if quality_bucket.get("policy_source") is None and meta.get("policy_source") is not None:
            quality_bucket["policy_source"] = meta.get("policy_source")
        if quality_bucket.get("summary_model") is None and meta.get("model") is not None:
            quality_bucket["summary_model"] = meta.get("model")
        if quality_bucket.get("canary_fraction") is None and canary.get("fraction") is not None:
            quality_bucket["canary_fraction"] = canary.get("fraction")
        if quality_bucket.get("canary_unit") is None and canary.get("unit") is not None:
            quality_bucket["canary_unit"] = canary.get("unit")
        quality_bucket["canary_enabled"] = bool(quality_bucket.get("canary_enabled") or canary.get("enabled"))
        quality_bucket["enabled_rows"] += int(bool(meta.get("enabled")))
        quality_bucket["disabled_rows"] += int(not bool(meta.get("enabled")))
        if bool(meta.get("enabled")):
            quality_bucket["matched_metadata_row_count"] += 1
        if cohort == "canary_applied":
            quality_bucket["canary_applied_count"] += 1
        elif cohort == "canary_holdout":
            quality_bucket["canary_holdout_count"] += 1
        elif cohort == "bypassed_or_disabled":
            quality_bucket["bypassed_or_disabled_count"] += 1
        else:
            quality_bucket["unknown_cohort_count"] += 1
        failed = _old_context_summary_failed(meta)
        errored = _as_int(row.get("status_code")) >= 400
        retried = _as_int(row.get("retry_count")) > 0
        quality_bucket["summary_failure_count"] += int(failed)
        quality_bucket["safety_stop_count"] += int(safety_stopped)
        quality_bucket["error_count"] += int(errored)
        quality_bucket["retry_count"] += int(retried)
        quality_bucket["actual_tokens_saved_est"] += tokens_saved
        quality_bucket["actual_gross_savings_usd"] += gross_savings
        quality_bucket["actual_summary_model_cost_usd"] += summary_cost
        quality_bucket["actual_net_savings_usd"] += _as_float(meta.get("estimated_net_savings_usd") or (gross_savings - summary_cost))
        cohort_bucket = quality_bucket["cohorts"].get(cohort, quality_bucket["cohorts"]["unknown"])
        cohort_bucket["count"] += 1
        cohort_bucket["error_count"] += int(errored)
        cohort_bucket["retry_count"] += int(retried)
        cohort_bucket["summary_failure_count"] += int(failed)
        cohort_bucket["safety_stop_count"] += int(safety_stopped)
        latency = _as_int(row.get("latency_ms"))
        if latency > 0:
            cohort_bucket["latency_ms_total"] += latency
            cohort_bucket["latency_sample_count"] += 1

        model_key = (provider, model)
        model_bucket = model_rows.setdefault(model_key, {
            "provider": provider,
            "model": model,
            "observed_rows": 0,
            "eligible_rows": 0,
            "applied_rows": 0,
            "gross_saved_tokens": 0,
            "gross_savings_usd": 0.0,
            "summary_cost_usd": 0.0,
            "net_savings_usd": 0.0,
        })
        model_bucket["observed_rows"] += 1
        if is_eligible:
            model_bucket["eligible_rows"] += 1
        if is_applied:
            model_bucket["applied_rows"] += 1
        model_bucket["gross_saved_tokens"] += tokens_saved
        model_bucket["gross_savings_usd"] += gross_savings
        model_bucket["summary_cost_usd"] += summary_cost
        model_bucket["net_savings_usd"] += gross_savings - summary_cost

    net_savings_usd = gross_savings_usd - summary_cost_usd
    today_net_savings_usd = today_gross_savings_usd - today_summary_cost_usd
    model_breakdown = []
    for bucket in model_rows.values():
        bucket["gross_savings_usd"] = round(float(bucket["gross_savings_usd"]), 6)
        bucket["summary_cost_usd"] = round(float(bucket["summary_cost_usd"]), 6)
        bucket["net_savings_usd"] = round(float(bucket["net_savings_usd"]), 6)
        model_breakdown.append(bucket)
    model_breakdown.sort(key=lambda item: (item["net_savings_usd"], item["eligible_rows"]), reverse=True)
    quality_gates = [_finalize_old_context_summary_quality_bucket(bucket) for bucket in quality_gate_rows.values()]
    quality_gates.sort(key=lambda item: (str(item.get("last_decision_at") or ""), item["metrics"]["matched_metadata_row_count"]), reverse=True)
    verdict_counts: dict[str, int] = {}
    reason_code_counts: dict[str, int] = {}
    warning_code_counts: dict[str, int] = {}
    for gate in quality_gates:
        verdict_counts[str(gate.get("verdict") or "unknown")] = verdict_counts.get(str(gate.get("verdict") or "unknown"), 0) + 1
        for code in gate.get("reason_codes") or []:
            reason_code_counts[str(code or "unknown")] = reason_code_counts.get(str(code or "unknown"), 0) + 1
        for code in gate.get("warning_codes") or []:
            warning_code_counts[str(code or "unknown")] = warning_code_counts.get(str(code or "unknown"), 0) + 1
    quality_gate_summary = {
        "status": "observed" if quality_gates else "no-observed-rows",
        "decision_count": len(quality_gates),
        "promote_count": verdict_counts.get("promote", 0),
        "hold_count": verdict_counts.get("hold", 0),
        "rollback_count": verdict_counts.get("rollback", 0),
        "insufficient_evidence_count": verdict_counts.get("insufficient-evidence", 0),
        "disabled_count": verdict_counts.get("disabled", 0),
        "verdict_breakdown": _count_breakdown(verdict_counts),
        "reason_code_breakdown": _count_breakdown(reason_code_counts),
        "warning_code_breakdown": _count_breakdown(warning_code_counts),
    }
    readiness_state_counts["rollback"] = quality_gate_summary["rollback_count"]
    readiness = {
        "schema": "tokenclaw.old_context_summary_dashboard_readiness.v1",
        "status": "observed" if observed_rows else "no-observed-rows",
        "latest_quality_gate_verdict": quality_gates[0].get("verdict") if quality_gates else None,
        "state_breakdown": _count_breakdown(readiness_state_counts),
        "blocker_breakdown": quality_gate_summary["reason_code_breakdown"],
        "read_only": True,
        "wrote_policy_files": False,
        "provider_calls_made": False,
    }
    category_breakdown = []
    for bucket in category_rows.values():
        for money_field in (
            "projected_gross_savings_usd",
            "projected_summary_cost_usd",
            "projected_net_savings_usd",
            "applied_gross_savings_usd",
            "applied_summary_cost_usd",
            "applied_net_savings_usd",
        ):
            bucket[money_field] = round(float(bucket[money_field]), 6)
        category_breakdown.append(bucket)
    category_breakdown.sort(
        key=lambda item: (
            item["applied_net_savings_usd"] + item["projected_net_savings_usd"],
            item["observed_rows"],
        ),
        reverse=True,
    )
    plateau_session_context = {
        "schema": "tokenclaw.old_context_summary_plateau_session_context.v1",
        "affected_session_count": len(affected_sessions),
        "observed_large_context_rows": len(plateau_text_chars),
        "median_text_chars": _median_int(plateau_text_chars),
        "p90_text_chars": _percentile_int(plateau_text_chars, 0.9),
        "category_breakdown": category_breakdown,
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "local_session_ids_included": False,
        },
    }
    summary = {
        "observed_rows": observed_rows,
        "today_observed_rows": today_observed_rows,
        "eligible_rows": eligible_count,
        "today_eligible_rows": today_eligible_count,
        "ineligible_rows": ineligible_count,
        "skipped_rows": skipped_count,
        "planned_rows": planned_count,
        "applied_rows": applied_count,
        "today_applied_rows": today_applied_count,
        "summary_created_rows": summary_created_count,
        "cached_summary_hit_rows": summary_cache_hits,
        "summary_empty_rows": summary_empty_count,
        "error_rows": error_count,
        "eligibility_rate": round(eligible_count / observed_rows, 4) if observed_rows else 0.0,
        "applied_rate": round(applied_count / observed_rows, 4) if observed_rows else 0.0,
        "summary_cache_hit_rate": round(summary_cache_hits / applied_count, 4) if applied_count else 0.0,
        "gross_saved_tokens_est": int(gross_saved_tokens),
        "today_gross_saved_tokens_est": int(today_gross_saved_tokens),
        "summary_model_input_tokens_est": int(summary_input_tokens),
        "summary_model_output_tokens_est": int(summary_output_tokens),
        "gross_savings_usd": round(gross_savings_usd, 6),
        "today_gross_savings_usd": round(today_gross_savings_usd, 6),
        "summary_model_cost_usd": round(summary_cost_usd, 6),
        "today_summary_model_cost_usd": round(today_summary_cost_usd, 6),
        "net_savings_usd": round(net_savings_usd, 6),
        "today_net_savings_usd": round(today_net_savings_usd, 6),
        "payback_ratio": round(gross_savings_usd / summary_cost_usd, 4) if summary_cost_usd > 0 else None,
        "today_payback_ratio": round(today_gross_savings_usd / today_summary_cost_usd, 4) if today_summary_cost_usd > 0 else None,
    }
    policy_state = await stats_policies()
    policy_health = _old_context_summary_policy_health(policy_state)
    canary_applied_rows = sum(_as_int((row.get("metrics") or {}).get("canary_applied_count")) for row in quality_gates)
    canary_holdout_rows = sum(_as_int((row.get("metrics") or {}).get("canary_holdout_count")) for row in quality_gates)
    bypassed_or_disabled_rows = sum(_as_int((row.get("metrics") or {}).get("bypassed_or_disabled_count")) for row in quality_gates)
    safety_stop_rows = sum(_as_int((row.get("metrics") or {}).get("safety_stop_count")) for row in quality_gates)
    rollout_status = _old_context_summary_rollout_status(
        summary=summary,
        policy=policy_health,
        quality_gate_summary=quality_gate_summary,
        canary_applied_rows=canary_applied_rows,
        canary_holdout_rows=canary_holdout_rows,
        safety_stop_rows=safety_stop_rows,
    )
    latest_gate = quality_gates[0] if quality_gates else {}
    managed_feedback_queue = _managed_feedback_queue_health(
        store_obj,
        sample_limit=5,
        source_surface=OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
    )
    skip_breakdown = _count_breakdown(reason_counts)
    rollout_health = {
        "schema": "tokenclaw.old_context_summary_rollout_health.v1",
        "status": rollout_status,
        "state_flags": {
            "disabled": rollout_status == "disabled",
            "not_deployed_yet": rollout_status == "not-deployed-yet",
            "no_observed_rows": observed_rows <= 0,
            "no_applied_canary_rows": canary_holdout_rows > 0 and canary_applied_rows <= 0,
            "safety_stopped": rollout_status == "safety-stopped",
            "read_only": True,
        },
        "policy": policy_health,
        "latest": {
            "candidate_id": latest_gate.get("candidate_id") or policy_health.get("candidate_id"),
            "rule_id": latest_gate.get("rule_id") or policy_health.get("rule_id"),
            "policy_source": latest_gate.get("policy_source") or policy_health.get("policy_source"),
            "summary_model": latest_gate.get("summary_model") or policy_health.get("summary_model"),
            "last_decision_at": latest_gate.get("last_decision_at"),
            "quality_gate_verdict": latest_gate.get("verdict"),
        },
        "rollout_counts": {
            "observed_rows": observed_rows,
            "today_observed_rows": today_observed_rows,
            "disabled_rows": _breakdown_count(skip_breakdown, "disabled"),
            "planned_rows": planned_count,
            "applied_rows": applied_count,
            "today_applied_rows": today_applied_count,
            "canary_applied_rows": canary_applied_rows,
            "canary_holdout_rows": canary_holdout_rows,
            "bypassed_or_disabled_rows": bypassed_or_disabled_rows,
            "safety_stop_rows": safety_stop_rows,
            "summary_failure_rows": sum(_as_int((row.get("metrics") or {}).get("summary_failure_count")) for row in quality_gates),
        },
        "economics": {
            "gross_saved_tokens_est": summary["gross_saved_tokens_est"],
            "today_gross_saved_tokens_est": summary["today_gross_saved_tokens_est"],
            "summary_model_input_tokens_est": summary["summary_model_input_tokens_est"],
            "summary_model_output_tokens_est": summary["summary_model_output_tokens_est"],
            "summary_cache_hit_rate": summary["summary_cache_hit_rate"],
            "gross_savings_usd": summary["gross_savings_usd"],
            "today_gross_savings_usd": summary["today_gross_savings_usd"],
            "summary_model_cost_usd": summary["summary_model_cost_usd"],
            "today_summary_model_cost_usd": summary["today_summary_model_cost_usd"],
            "net_savings_usd": summary["net_savings_usd"],
            "today_net_savings_usd": summary["today_net_savings_usd"],
            "payback_ratio": summary["payback_ratio"],
            "today_payback_ratio": summary["today_payback_ratio"],
        },
        "managed_feedback_queue": managed_feedback_queue,
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
        },
    }

    return {
        "schema": "tokenclaw.old_context_summarization_opportunity.v1",
        "generated_at": utc_now(),
        "summary": summary,
        "rollout_health": rollout_health,
        "readiness": readiness,
        "plateau_session_context": plateau_session_context,
        "status_breakdown": _count_breakdown(status_counts),
        "skip_reason_breakdown": skip_breakdown,
        "model_breakdown": model_breakdown,
        "quality_gate_summary": quality_gate_summary,
        "quality_gates": quality_gates,
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "file_contents_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
        },
    }










def _sanitize_error_sample(error: Any, limit: int = 180) -> str | None:
    if not error:
        return None
    text = str(error)
    try:
        body = json.loads(text)
    except Exception:
        body = None
    if isinstance(body, dict):
        error_body = body.get("error")
        if isinstance(error_body, dict):
            text = str(error_body.get("message") or error_body.get("code") or error_body.get("type") or text)
        elif error_body:
            text = str(error_body)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "..."
    return text or None


def _error_type(status_code: Any, error: Any) -> str:
    status = _as_int(status_code)
    sample = (_sanitize_error_sample(error, limit=500) or "").lower()
    if sample.startswith("temporarily limiting requests"):
        return "local_rate_limit"
    if status in (429, 529) or "rate_limit" in sample or "rate limit" in sample:
        return "upstream_rate_limit"
    if "does not support the effort parameter" in sample:
        return "model_incompatible_param"
    if "adaptive thinking is not supported" in sample:
        return "model_incompatible_thinking"
    if "connecterror" in sample or "temporary failure in name resolution" in sample:
        return "network_connect_error"
    if "readtimeout" in sample or "timeout" in sample:
        return "network_timeout"
    if status in (401, 403) or "invalid_api_key" in sample or "incorrect api key" in sample:
        return "auth_error"
    if status:
        return f"http_{status}"
    return "unknown_error"


def _error_breakdown(rows: list[dict[str, Any]], *, today_only: bool = False, limit: int = 30) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        status_code = _as_int(row.get("status_code"))
        error_type = _error_type(status_code, row.get("error"))
        error_sample = _sanitize_error_sample(row.get("error")) or f"HTTP {status_code}"
        model = str(row.get("model") or "")
        tier = model_tier(model)
        key = (
            row.get("provider") or "anthropic",
            status_code,
            tier,
            row.get("requested_model"),
            row.get("routed_model"),
            error_type,
            error_sample,
        )
        bucket = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "status_code": status_code,
                "tier": tier,
                "requested_model": row.get("requested_model"),
                "routed_model": row.get("routed_model"),
                "model": row.get("model"),
                "error_type": error_type,
                "error_sample": error_sample,
                "count": 0,
                "last_seen_at": row.get("created_at"),
            },
        )
        bucket["count"] += 1
        if str(row.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = row.get("created_at")

    breakdown = list(grouped.values())
    breakdown.sort(key=lambda r: (r["count"], str(r.get("last_seen_at") or "")), reverse=True)
    return breakdown[:limit]


def _crunch_rule_breakdowns(
    rows: list[dict[str, Any]],
    *,
    today_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_rule: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    by_group: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        crunch = _json_obj(row.get("crunch_json"))
        decisions: list[dict[str, Any]] = []
        for key, status in (("applied_rules", "applied"), ("skipped_rules", "skipped")):
            values = crunch.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        decision = dict(item)
                        decision.setdefault("status", status)
                        decisions.append(decision)
        if not decisions and crunch.get("changed"):
            decisions.append({
                "rule_id": str(crunch.get("rule_id") or "legacy_crunch"),
                "rule_group": str(crunch.get("rule_group") or "legacy"),
                "lossiness_class": str(crunch.get("lossiness_class") or "unknown"),
                "canary_state": str(crunch.get("canary_state") or "active"),
                "policy_source": str(crunch.get("policy_source") or "unknown"),
                "status": "applied",
                "reason": "legacy-crunch-metadata",
                "count": 1,
                "saved_chars": _as_int(crunch.get("saved_chars")),
                "tokens_saved_est": _as_int(crunch.get("tokens_saved_est")),
            })
        for item in decisions:
            status = str(item.get("status") or "unknown")
            rule_id = str(item.get("rule_id") or "unknown")
            rule_group = str(item.get("rule_group") or "unknown")
            lossiness = str(item.get("lossiness_class") or "unknown")
            canary_state = str(item.get("canary_state") or "unknown")
            policy_source = str(item.get("policy_source") or crunch.get("policy_source") or "unknown")
            reason = str(item.get("reason") or "unknown")
            count = max(0, _as_int(item.get("count")) or 1)
            saved_chars = _as_int(item.get("saved_chars"))
            tokens_saved = _as_int(item.get("tokens_saved_est"))
            rule_key = (rule_id, rule_group, lossiness, canary_state, status, policy_source)
            rule_bucket = by_rule.setdefault(
                rule_key,
                {
                    "rule_id": rule_id,
                    "rule_group": rule_group,
                    "lossiness_class": lossiness,
                    "canary_state": canary_state,
                    "status": status,
                    "policy_source": policy_source,
                    "count": 0,
                    "decision_count": 0,
                    "saved_chars": 0,
                    "tokens_saved_est": 0,
                    "reason_counts": {},
                },
            )
            rule_bucket["count"] += count
            rule_bucket["decision_count"] += 1
            rule_bucket["saved_chars"] += saved_chars
            rule_bucket["tokens_saved_est"] += tokens_saved
            reasons = rule_bucket["reason_counts"]
            reasons[reason] = reasons.get(reason, 0) + count

            group_key = (rule_group, lossiness, canary_state, status)
            group_bucket = by_group.setdefault(
                group_key,
                {
                    "rule_group": rule_group,
                    "lossiness_class": lossiness,
                    "canary_state": canary_state,
                    "status": status,
                    "count": 0,
                    "decision_count": 0,
                    "saved_chars": 0,
                    "tokens_saved_est": 0,
                    "rule_ids": set(),
                },
            )
            group_bucket["count"] += count
            group_bucket["decision_count"] += 1
            group_bucket["saved_chars"] += saved_chars
            group_bucket["tokens_saved_est"] += tokens_saved
            group_bucket["rule_ids"].add(rule_id)

    rule_rows = []
    for row in by_rule.values():
        reasons = row.pop("reason_counts", {})
        calls_affected = _as_int(row.get("count"))
        saved_chars = _as_int(row.get("saved_chars"))
        row["calls_affected"] = calls_affected
        row["total_chars_saved"] = saved_chars
        row["avg_chars_saved"] = round(saved_chars / calls_affected, 2) if calls_affected else 0.0
        row["reason_breakdown"] = [
            {"value": key, "count": reasons[key]}
            for key in sorted(reasons, key=lambda key: (-reasons[key], key))
        ]
        rule_rows.append(row)
    group_rows = []
    for row in by_group.values():
        calls_affected = _as_int(row.get("count"))
        saved_chars = _as_int(row.get("saved_chars"))
        row["calls_affected"] = calls_affected
        row["total_chars_saved"] = saved_chars
        row["avg_chars_saved"] = round(saved_chars / calls_affected, 2) if calls_affected else 0.0
        row["rule_ids"] = sorted(row["rule_ids"])
        group_rows.append(row)
    rule_rows.sort(key=lambda row: (row["status"] != "applied", -row["total_chars_saved"], -row["count"], row["rule_id"]))
    group_rows.sort(key=lambda row: (row["status"] != "applied", -row["total_chars_saved"], -row["count"], row["rule_group"]))
    return rule_rows, group_rows


def _pattern_decision_breakdown(rows: list[dict[str, Any]], *, today_only: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        provider = str(row.get("provider") or "anthropic")
        path = str(row.get("path") or "")
        summaries = pattern_decision_summaries(
            provider=provider,
            path=path,
            requested_model=row.get("requested_model"),
            routed_model=row.get("routed_model"),
            status_code=_as_int(row.get("status_code")) if row.get("status_code") is not None else None,
            cost_est_usd=_as_float(row.get("cost_est_usd")) if row.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(row.get("cost_baseline_usd")) if row.get("cost_baseline_usd") is not None else None,
            cache_meta=cache,
            crunch_meta=crunch,
            routing_meta=routing,
            category=row.get("category") or routing.get("category"),
        )
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            key = (
                str(summary.get("source_surface") or _source_surface(provider, path)),
                str(summary.get("app_family") or _app_family_for_call(provider, row.get("requested_model"), path)),
                str(summary.get("category") or row.get("category") or routing.get("category") or "unknown"),
                str(summary.get("workflow_phase") or summary.get("category") or "unknown"),
                str(summary.get("decision_type") or "unknown"),
                str(summary.get("policy_source") or "unknown"),
                str(summary.get("rule_id") or "unknown"),
                str(summary.get("pattern_hash") or ""),
                str(summary.get("outcome") or "unknown"),
            )
            bucket = grouped.setdefault(
                key,
                {
                    "source_surface": key[0],
                    "app_family": key[1],
                    "category": key[2],
                    "workflow_phase": key[3],
                    "decision_type": key[4],
                    "policy_source": key[5],
                    "rule_id": key[6],
                    "candidate_id": summary.get("candidate_id"),
                    "pattern_hash": key[7] or None,
                    "outcome": key[8],
                    "status": summary.get("status"),
                    "reason": summary.get("reason"),
                    "hit_type": summary.get("hit_type"),
                    "count": 0,
                    "applied_count": 0,
                    "error_count": 0,
                    "saved_chars": 0,
                    "tokens_saved_est": 0,
                    "estimated_cost_savings_usd": 0.0,
                    "raw_payload_included": False,
                },
            )
            bucket["count"] += 1
            bucket["applied_count"] += _as_int(summary.get("applied_count"))
            if key[8] == "errored":
                bucket["error_count"] += 1
            bucket["saved_chars"] += _as_int(summary.get("saved_chars"))
            bucket["tokens_saved_est"] += _as_int(summary.get("tokens_saved_est"))
            bucket["estimated_cost_savings_usd"] += _as_float(summary.get("estimated_cost_savings_usd"))
            if bucket.get("candidate_id") is None and summary.get("candidate_id") is not None:
                bucket["candidate_id"] = summary.get("candidate_id")
            if bucket.get("reason") is None and summary.get("reason") is not None:
                bucket["reason"] = summary.get("reason")

    result = []
    for bucket in grouped.values():
        bucket["estimated_cost_savings_usd"] = round(float(bucket["estimated_cost_savings_usd"]), 8)
        count = _as_int(bucket.get("count"))
        bucket["error_rate"] = round(_as_int(bucket.get("error_count")) / count, 4) if count else 0.0
        result.append(bucket)
    result.sort(
        key=lambda r: (
            _as_float(r.get("estimated_cost_savings_usd")),
            _as_int(r.get("saved_chars")),
            _as_int(r.get("count")),
        ),
        reverse=True,
    )
    return result


LOCAL_PATTERN_COVERAGE_FAMILIES = (
    "terminal_logs",
    "tool_results",
    "diffs",
    "generated_artifacts",
    "tabular_data",
    "cacheability",
)
MANAGED_PATTERN_SUPPORTED_FAMILIES = set(LOCAL_PATTERN_COVERAGE_FAMILIES)


def _empty_local_pattern_family_bucket(family: str, *, registered: bool, supports_local_crunch: bool) -> dict[str, Any]:
    return {
        "family": family,
        "registered": bool(registered),
        "managed_supported": family in MANAGED_PATTERN_SUPPORTED_FAMILIES,
        "supports_local_crunch": bool(supports_local_crunch),
        "detected_call_count": 0,
        "feature_call_count": 0,
        "fingerprint_row_count": 0,
        "fingerprint_count": 0,
        "action_families_seen": set(),
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "bypassed_count": 0,
        "safety_stop_count": 0,
        "disabled_count": 0,
        "status_counts": {},
        "skip_reasons": {},
        "raw_content_included": False,
        "latest_seen_at": None,
    }


def _count_dict_increment(values: dict[str, int], key: Any, amount: int = 1) -> None:
    label = str(key or "unknown")
    values[label] = values.get(label, 0) + max(0, int(amount))


def _local_pattern_family_eligibility(
    row: dict[str, Any],
    *,
    recommendations_configured: bool,
    min_samples: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if row["disabled_count"] and not row["detected_call_count"]:
        reasons.append("local-module-disabled")
    if row["detected_call_count"] <= 0:
        reasons.append("no-detected-samples")
    if row["fingerprint_count"] <= 0:
        reasons.append("no-fingerprints")
    if not row["managed_supported"]:
        reasons.append("unsupported-family")
    if row["detected_call_count"] < min_samples:
        reasons.append("insufficient-samples")
    if not recommendations_configured:
        reasons.append("recommendation-fetch-disabled")

    if not reasons:
        status = "ready"
        reasons = ["clean-ready"]
    elif "local-module-disabled" in reasons:
        status = "disabled"
    elif "unsupported-family" in reasons:
        status = "unsupported-family"
    elif "recommendation-fetch-disabled" in reasons:
        status = "recommendation-fetch-disabled"
    elif "no-detected-samples" in reasons:
        status = "no-samples"
    elif "no-fingerprints" in reasons:
        status = "no-fingerprints"
    else:
        status = "insufficient-samples"
    return {
        "status": status,
        "reasons": reasons,
        "min_samples": min_samples,
        "recommendation_fetch_enabled": bool(recommendations_configured),
    }


async def stats_local_pattern_coverage(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    """Aggregate local pattern-module coverage without exposing raw request content."""
    from tokenclaw.pattern_modules import registered_pattern_modules
    from tokenclaw.recommendations import recommendations_enabled, recommendation_server_configured

    conn = store_obj.conn
    row_limit = max(1, min(int(limit or 1000), 10000))
    min_samples = max(1, _as_int(os.getenv("TOKENCLAW_PATTERN_COVERAGE_MIN_SAMPLES")) or 10)
    recommendations_configured = bool(recommendations_enabled() and recommendation_server_configured())
    registered = {item["family"]: item for item in registered_pattern_modules() if isinstance(item, dict)}
    families = sorted(set(LOCAL_PATTERN_COVERAGE_FAMILIES) | set(registered))
    grouped: dict[str, dict[str, Any]] = {
        family: _empty_local_pattern_family_bucket(
            family,
            registered=family in registered,
            supports_local_crunch=bool((registered.get(family) or {}).get("supports_local_crunch")),
        )
        for family in families
    }

    rows = [
        dict(r)
        for r in conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   retry_count, cost_est_usd, cost_baseline_usd,
                   crunch_json, routing_json, cache_json, category
            from calls
            order by created_at desc
            limit ?
            """,
            (row_limit,),
        ).fetchall()
    ]

    for db_row in rows:
        crunch_meta = _json_obj(db_row.get("crunch_json"))
        routing_meta = _json_obj(db_row.get("routing_json"))
        cache_meta = _json_obj(db_row.get("cache_json"))
        pattern_meta = (
            crunch_meta.get("pattern_modules")
            if isinstance(crunch_meta.get("pattern_modules"), dict)
            else {}
        )
        modules = pattern_meta.get("modules") if isinstance(pattern_meta, dict) else []
        if not isinstance(modules, list):
            modules = []
        emitted_families: set[str] = set()
        server_features = pattern_meta.get("server_features") if isinstance(pattern_meta, dict) else {}
        for feature in (server_features.get("features") if isinstance(server_features, dict) else []) or []:
            if isinstance(feature, dict) and feature.get("family"):
                emitted_families.add(str(feature["family"]))

        for module in modules:
            if not isinstance(module, dict):
                continue
            family = str(module.get("family") or "unknown")
            if family not in grouped:
                grouped[family] = _empty_local_pattern_family_bucket(
                    family,
                    registered=family in registered,
                    supports_local_crunch=bool((registered.get(family) or {}).get("supports_local_crunch")),
                )
            bucket = grouped[family]
            if module.get("detected"):
                bucket["detected_call_count"] += 1
                bucket["latest_seen_at"] = max(
                    [item for item in (bucket["latest_seen_at"], db_row.get("created_at")) if item],
                    default=None,
                )
            if family in emitted_families or module.get("features_emitted"):
                bucket["feature_call_count"] += 1
                bucket["action_families_seen"].add("feature")
            status = str(module.get("status") or "unknown")
            reason = str(module.get("reason") or "unknown")
            _count_dict_increment(bucket["status_counts"], status)
            if status == "applied":
                bucket["applied_count"] += 1
                bucket["action_families_seen"].add("local_crunch")
            elif status == "skipped":
                bucket["skipped_count"] += 1
                _count_dict_increment(bucket["skip_reasons"], reason)
            elif status in {"bypass", "bypassed"}:
                bucket["bypassed_count"] += 1
                _count_dict_increment(bucket["skip_reasons"], reason)
            if module.get("enabled") is False:
                bucket["disabled_count"] += 1
            privacy = module.get("privacy_guard") if isinstance(module.get("privacy_guard"), dict) else {}
            summary = module.get("feature_summary") if isinstance(module.get("feature_summary"), dict) else {}
            bucket["raw_content_included"] = bool(
                bucket["raw_content_included"]
                or module.get("raw_content_included")
                or summary.get("raw_content_included")
                or not privacy.get("safe", True)
            )

        diagnostics = (
            routing_meta.get("managed_pattern_features")
            if isinstance(routing_meta.get("managed_pattern_features"), dict)
            else {}
        )
        diagnostic_families = diagnostics.get("local_pattern_module_families") if isinstance(diagnostics, dict) else []
        if not isinstance(diagnostic_families, list):
            diagnostic_families = []
        if diagnostics.get("present"):
            hash_count = _as_int(diagnostics.get("pattern_hash_count"))
            for family in {str(item) for item in diagnostic_families if item}:
                if family not in grouped:
                    grouped[family] = _empty_local_pattern_family_bucket(
                        family,
                        registered=family in registered,
                        supports_local_crunch=bool((registered.get(family) or {}).get("supports_local_crunch")),
                    )
                grouped[family]["fingerprint_row_count"] += 1
                grouped[family]["fingerprint_count"] += hash_count
                grouped[family]["action_families_seen"].add("fingerprint")

        for summary in pattern_decision_summaries(
            provider=str(db_row.get("provider") or "anthropic"),
            path=str(db_row.get("path") or ""),
            requested_model=db_row.get("requested_model"),
            routed_model=db_row.get("routed_model"),
            status_code=_as_int(db_row.get("status_code")) if db_row.get("status_code") is not None else None,
            cost_est_usd=_as_float(db_row.get("cost_est_usd")) if db_row.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(db_row.get("cost_baseline_usd")) if db_row.get("cost_baseline_usd") is not None else None,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=db_row.get("category") or routing_meta.get("category"),
        ):
            summary_families = summary.get("local_pattern_module_families") if isinstance(summary, dict) else []
            if not isinstance(summary_families, list):
                summary_families = []
            action = str(summary.get("decision_type") or summary.get("pattern_family") or "unknown")
            status = str(summary.get("status") or "")
            outcome = str(summary.get("outcome") or "")
            reason = str(summary.get("reason") or "")
            for family in {str(item) for item in summary_families if item}:
                if family not in grouped:
                    continue
                bucket = grouped[family]
                if action in {"routing", "crunch", "cache", "local_pattern_fingerprint"}:
                    bucket["action_families_seen"].add("fingerprint" if action == "local_pattern_fingerprint" else action)
                if status == "holdout" or outcome == "holdout":
                    bucket["holdout_count"] += 1
                if status in {"bypass", "bypassed"} or outcome == "bypassed":
                    bucket["bypassed_count"] += 1
                    if reason:
                        _count_dict_increment(bucket["skip_reasons"], reason)
                if reason == "local-canary-safety-stop" or summary.get("safety_stop"):
                    bucket["safety_stop_count"] += 1

    output_rows: list[dict[str, Any]] = []
    for family, bucket in grouped.items():
        public_row = {
            "family": family,
            "registered": bucket["registered"],
            "managed_supported": bucket["managed_supported"],
            "supports_local_crunch": bucket["supports_local_crunch"],
            "detected_call_count": bucket["detected_call_count"],
            "feature_call_count": bucket["feature_call_count"],
            "fingerprint_row_count": bucket["fingerprint_row_count"],
            "fingerprint_count": bucket["fingerprint_count"],
            "action_families_seen": sorted(bucket["action_families_seen"]),
            "applied_count": bucket["applied_count"],
            "holdout_count": bucket["holdout_count"],
            "skipped_count": bucket["skipped_count"],
            "bypassed_count": bucket["bypassed_count"],
            "safety_stop_count": bucket["safety_stop_count"],
            "disabled_count": bucket["disabled_count"],
            "status_counts": _breakdown_from_counts(bucket["status_counts"]),
            "top_skip_reasons": _breakdown_from_counts(bucket["skip_reasons"])[:5],
            "raw_content_included": bool(bucket["raw_content_included"]),
            "latest_seen_at": bucket["latest_seen_at"],
        }
        public_row["managed_eligibility"] = _local_pattern_family_eligibility(
            public_row,
            recommendations_configured=recommendations_configured,
            min_samples=min_samples,
        )
        output_rows.append(public_row)

    output_rows.sort(key=lambda item: (item["family"] not in LOCAL_PATTERN_COVERAGE_FAMILIES, item["family"]))
    return {
        "schema": "tokenclaw.local_pattern_coverage.v1",
        "generated_at": utc_now(),
        "sampled_call_limit": row_limit,
        "sampled_call_count": len(rows),
        "summary": {
            "family_count": len(output_rows),
            "families_with_detections": sum(1 for row in output_rows if row["detected_call_count"] > 0),
            "families_with_fingerprints": sum(1 for row in output_rows if row["fingerprint_count"] > 0),
            "ready_family_count": sum(1 for row in output_rows if row["managed_eligibility"]["status"] == "ready"),
            "recommendation_fetch_enabled": recommendations_configured,
            "min_samples_for_managed_eligibility": min_samples,
        },
        "families": output_rows,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "pattern_hashes_included": False,
        },
    }


def _status_code_bucket(status_code: Any) -> str:
    if status_code is None:
        return "unknown"
    code = _as_int(status_code)
    if code <= 0:
        return "unknown"
    if code < 200:
        return "lt_2xx"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _latency_bucket(latency_ms: Any) -> str:
    latency = _as_int(latency_ms)
    if latency <= 0:
        return "unknown"
    if latency < 1_000:
        return "lt_1s"
    if latency < 5_000:
        return "1s_5s"
    if latency < 15_000:
        return "5s_15s"
    return "gte_15s"


def _usd_bucket(value: Any) -> str:
    amount = _as_float(value)
    if amount <= 0:
        return "zero"
    if amount < 0.001:
        return "lt_0_001"
    if amount < 0.01:
        return "0_001_0_01"
    if amount < 0.05:
        return "0_01_0_05"
    return "gte_0_05"


def _managed_pattern_lifecycle_bucket(summary: dict[str, Any]) -> str | None:
    text = " ".join(
        str(summary.get(key) or "").lower()
        for key in ("status", "outcome", "reason", "action")
    )
    if "rollback" in text or "rolled_back" in text or "rolled-back" in text:
        return "rolled_back"
    if "reject" in text:
        return "rejected"
    return None


def _managed_pattern_add_summary(
    grouped: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, Any]],
    *,
    summary: dict[str, Any],
    created_at: Any,
    status_code: Any,
    latency_ms: Any,
    cost_est_usd: Any,
    min_samples: int,
) -> None:
    pattern_hash = str(summary.get("pattern_hash") or "")
    if not pattern_hash.startswith("sha256:"):
        return
    policy_source = str(summary.get("policy_source") or "")
    if (
        not policy_source.startswith("managed-")
        and summary.get("candidate_id") is None
        and not isinstance(summary.get("canary"), dict)
        and not summary.get("evidence_only")
    ):
        return
    cohort = str(summary.get("cohort") or "non_canary")
    key = (
        str(summary.get("decision_type") or "unknown"),
        str(summary.get("candidate_id") or "unknown"),
        str(summary.get("rule_id") or "unknown"),
        pattern_hash,
        str(summary.get("source_surface") or "unknown"),
        str(summary.get("app_family") or "unknown"),
        str(summary.get("workflow_phase") or summary.get("category") or "unknown"),
        str(summary.get("category") or "unknown"),
        cohort,
    )
    bucket = grouped.setdefault(
        key,
        {
            "schema": "tokenclaw.managed_pattern_canary_cohort_bucket.v1",
            "policy_section": key[0],
            "candidate_id": None if key[1] == "unknown" else key[1],
            "rule_id": None if key[2] == "unknown" else key[2],
            "pattern_hash": key[3],
            "source_surface": key[4],
            "app_family": key[5],
            "workflow_phase": key[6],
            "category": key[7],
            "canary_cohort": key[8],
            "policy_source": summary.get("policy_source"),
            "sample_count": 0,
            "success_count": 0,
            "error_count": 0,
            "holdout_count": 0,
            "bypassed_count": 0,
            "applied_count": 0,
            "saved_chars": 0,
            "tokens_saved_est": 0,
            "estimated_cost_savings_usd": 0.0,
            "cost_est_usd": 0.0,
            "status_code_counts": {},
            "latency_buckets": {},
            "cost_buckets": {},
            "savings_buckets": {},
            "local_bypass_reasons": {},
            "lifecycle_counts": {"rolled_back": 0, "rejected": 0},
            "first_seen_at": None,
            "last_seen_at": None,
            "canary": None,
            "evidence_only": bool(summary.get("evidence_only")),
            "pattern_family": summary.get("pattern_family"),
            "pattern_types": summary.get("pattern_types") if isinstance(summary.get("pattern_types"), list) else [],
            "local_pattern_module_families": (
                summary.get("local_pattern_module_families")
                if isinstance(summary.get("local_pattern_module_families"), list)
                else []
            ),
            "local_pattern_module_count": _as_int(summary.get("local_pattern_module_count")),
            "raw_payload_included": False,
        },
    )
    bucket["sample_count"] += 1
    if _as_int(summary.get("applied_count")) > 0:
        bucket["applied_count"] += _as_int(summary.get("applied_count"))
    if str(summary.get("outcome") or "") == "holdout" or key[8] == "canary_holdout":
        bucket["holdout_count"] += 1
    if _managed_pattern_is_bypass(summary):
        bucket["bypassed_count"] += 1
        reason = str(summary.get("reason") or "unknown")
        bucket["local_bypass_reasons"][reason] = bucket["local_bypass_reasons"].get(reason, 0) + 1
    if status_code is not None and _as_int(status_code) >= 400:
        bucket["error_count"] += 1
    elif status_code is not None:
        bucket["success_count"] += 1

    bucket["saved_chars"] += _as_int(summary.get("saved_chars"))
    bucket["tokens_saved_est"] += _as_int(summary.get("tokens_saved_est"))
    bucket["estimated_cost_savings_usd"] += _as_float(summary.get("estimated_cost_savings_usd"))
    bucket["cost_est_usd"] += _as_float(cost_est_usd)
    for counts_key, counts_value in (
        ("status_code_counts", _status_code_bucket(status_code)),
        ("latency_buckets", _latency_bucket(latency_ms)),
        ("cost_buckets", _usd_bucket(cost_est_usd)),
        ("savings_buckets", _usd_bucket(summary.get("estimated_cost_savings_usd"))),
    ):
        counts = bucket[counts_key]
        counts[counts_value] = counts.get(counts_value, 0) + 1

    lifecycle = _managed_pattern_lifecycle_bucket(summary)
    if lifecycle:
        bucket["lifecycle_counts"][lifecycle] = bucket["lifecycle_counts"].get(lifecycle, 0) + 1

    seen_at = str(created_at or "")
    if seen_at:
        if not bucket["first_seen_at"] or seen_at < bucket["first_seen_at"]:
            bucket["first_seen_at"] = seen_at
        if not bucket["last_seen_at"] or seen_at > bucket["last_seen_at"]:
            bucket["last_seen_at"] = seen_at

    canary = summary.get("canary")
    if isinstance(canary, dict) and bucket["canary"] is None:
        bucket["canary"] = {
            key: canary.get(key)
            for key in ("enabled", "status", "cohort", "fraction", "unit", "threshold")
            if canary.get(key) is not None
        }
    bucket["minimum_sample_readiness"] = {
        "min_samples": min_samples,
        "ready": bucket["sample_count"] >= min_samples,
        "remaining": max(0, min_samples - bucket["sample_count"]),
    }


def _managed_pattern_finalize_buckets(grouped: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        sample_count = _as_int(bucket.get("sample_count"))
        error_count = _as_int(bucket.get("error_count"))
        success_count = _as_int(bucket.get("success_count"))
        bucket["estimated_cost_savings_usd"] = round(_as_float(bucket.get("estimated_cost_savings_usd")), 8)
        bucket["cost_est_usd"] = round(_as_float(bucket.get("cost_est_usd")), 8)
        bucket["error_rate"] = round(error_count / sample_count, 4) if sample_count else 0.0
        bucket["success_rate"] = round(success_count / sample_count, 4) if sample_count else 0.0
        for key in ("status_code_counts", "latency_buckets", "cost_buckets", "savings_buckets", "local_bypass_reasons", "lifecycle_counts"):
            bucket[key] = _count_breakdown(bucket.get(key) or {})
        rows.append(bucket)
    rows.sort(
        key=lambda item: (
            _as_int(item.get("sample_count")),
            _as_float(item.get("estimated_cost_savings_usd")),
            _as_int(item.get("tokens_saved_est")),
        ),
        reverse=True,
    )
    return rows


async def stats_managed_pattern_rollups(store_obj: Any, *, limit: int = 500, min_samples: int = 10) -> dict[str, Any]:
    """Return metadata-only managed pattern canary cohort rollups for export/review."""
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 500), 5000))
    sample_floor = max(1, min(int(min_samples or 10), 10_000))
    grouped: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, Any]] = {}

    provider_rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   input_tokens_est, output_tokens_est, actual_input_tokens,
                   actual_output_tokens, cost_est_usd, cost_baseline_usd,
                   crunch_json, routing_json, cache_json, category
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    for row in provider_rows:
        routing = _json_obj(row.get("routing_json"))
        summaries = pattern_decision_summaries(
            provider=str(row.get("provider") or "anthropic"),
            path=str(row.get("path") or ""),
            requested_model=row.get("requested_model"),
            routed_model=row.get("routed_model"),
            status_code=_as_int(row.get("status_code")) if row.get("status_code") is not None else None,
            cost_est_usd=_as_float(row.get("cost_est_usd")) if row.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(row.get("cost_baseline_usd")) if row.get("cost_baseline_usd") is not None else None,
            cache_meta=_json_obj(row.get("cache_json")),
            crunch_meta=_json_obj(row.get("crunch_json")),
            routing_meta=routing,
            category=row.get("category") or routing.get("category"),
        )
        for summary in summaries:
            if isinstance(summary, dict):
                _managed_pattern_add_summary(
                    grouped,
                    summary=summary,
                    created_at=row.get("created_at"),
                    status_code=row.get("status_code"),
                    latency_ms=row.get("latency_ms"),
                    cost_est_usd=row.get("cost_est_usd"),
                    min_samples=sample_floor,
                )

    codex_rows = [
        dict(row)
        for row in conn.execute(
            """
            select s.id as start_event_id,
                   s.created_at,
                   s.request_id,
                   s.thread_id,
                   s.session_id,
                   s.input_text_chars,
                   s.routing_json,
                   s.crunch_json,
                   s.cache_json,
                   (
                       select r.id from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_event_id,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars,
                   (
                       select r.error_code from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_error_code,
                   (
                       select r.latency_ms from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_latency_ms
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    for row in codex_rows:
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), row.get("response_result_chars"), cache)
        status_code = 500 if row.get("response_error_code") is not None else (200 if row.get("response_event_id") else None)
        summaries = pattern_decision_summaries(
            provider="openai",
            path="codex-app://turn/start",
            requested_model=routing.get("requested_model") or CODEX_APP_MODEL,
            routed_model=routing.get("routed_model") or routing.get("requested_model") or CODEX_APP_MODEL,
            status_code=status_code,
            cost_est_usd=_as_float(estimates.get("cost_est_usd")) if estimates.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(estimates.get("baseline_cost_est_usd")) if estimates.get("baseline_cost_est_usd") is not None else None,
            cache_meta=cache,
            crunch_meta=_json_obj(row.get("crunch_json")),
            routing_meta=routing,
            category=routing.get("category") or "codex_turn",
        )
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            summary = dict(summary)
            summary["source_surface"] = CODEX_APP_SOURCE_SURFACE
            summary["app_family"] = "codex"
            summary["category"] = routing.get("category") or summary.get("category") or "codex_turn"
            summary["workflow_phase"] = routing.get("workflow_phase") or summary.get("workflow_phase") or summary["category"]
            _managed_pattern_add_summary(
                grouped,
                summary=summary,
                created_at=row.get("created_at"),
                status_code=status_code,
                latency_ms=row.get("response_latency_ms"),
                cost_est_usd=estimates.get("cost_est_usd"),
                min_samples=sample_floor,
            )

    cohorts = _managed_pattern_finalize_buckets(grouped)
    return {
        "schema": "tokenclaw.managed_pattern_canary_cohort_rollups.v1",
        "generated_at": utc_now(),
        "limit": capped_limit,
        "min_samples": sample_floor,
        "summary": {
            "provider_rows_considered": len(provider_rows),
            "codex_turn_rows_considered": len(codex_rows),
            "cohort_bucket_count": len(cohorts),
            "ready_bucket_count": sum(1 for row in cohorts if (row.get("minimum_sample_readiness") or {}).get("ready")),
            "total_samples": sum(_as_int(row.get("sample_count")) for row in cohorts),
            "error_samples": sum(_as_int(row.get("error_count")) for row in cohorts),
            "holdout_samples": sum(_as_int(row.get("holdout_count")) for row in cohorts),
            "bypassed_samples": sum(_as_int(row.get("bypassed_count")) for row in cohorts),
            "rolled_back_events": sum(
                _as_int(item.get("count"))
                for row in cohorts
                for item in row.get("lifecycle_counts", [])
                if item.get("value") == "rolled_back"
            ),
            "rejected_events": sum(
                _as_int(item.get("count"))
                for row in cohorts
                for item in row.get("lifecycle_counts", [])
                if item.get("value") == "rejected"
            ),
        },
        "cohorts": cohorts,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "tenant_ids_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "basis": "stored decision metadata, hashes, status codes, latency, cost, and size buckets only",
        },
    }






MANAGED_PATTERN_ADOPTION_STAGES = (
    "received",
    "reviewed",
    "dry_run",
    "applied",
    "canary_applied",
    "canary_holdout",
    "bypassed",
    "errored",
    "rolled_back",
    "rejected",
)


































OPENAI_GOVERNOR_SCHEMA = "tokenclaw.openai_optimization_governor.v1"
OPENAI_GOVERNOR_FAMILIES = ("routing", "old_context_summary", "cache_replay")














































































async def stats_openai_cache_replay_report(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.openai_cache_replay_report import build_openai_cache_replay_report

    return build_openai_cache_replay_report(store_obj, limit=limit)


async def stats_streaming_tool_cache_invalidation_drill(
    store_obj: Any,
    *,
    window_hours: int = 24,
    max_rows: int = 10000,
    max_cohorts: int = 50,
) -> dict[str, Any]:
    from tokenclaw.streaming_tool_cache_invalidation_drill import (
        build_streaming_tool_cache_invalidation_drill_from_store,
    )

    return build_streaming_tool_cache_invalidation_drill_from_store(
        store_obj,
        window_hours=window_hours,
        max_rows=max_rows,
        max_cohorts=max_cohorts,
    )


async def stats_repeated_scaffold_opportunity(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_repeated_rows: int = 2,
) -> dict[str, Any]:
    from tokenclaw.repeated_scaffold_report import build_repeated_scaffold_opportunity_report

    return build_repeated_scaffold_opportunity_report(
        store_obj,
        limit=limit,
        min_repeated_rows=min_repeated_rows,
    )


async def stats_instruction_dedup_opportunity(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_repeated_rows: int = 2,
) -> dict[str, Any]:
    from tokenclaw.instruction_dedup_report import build_instruction_dedup_opportunity_report

    return build_instruction_dedup_opportunity_report(
        store_obj,
        limit=limit,
        min_repeated_rows=min_repeated_rows,
    )


async def stats_instruction_dedup_impact(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    from tokenclaw.instruction_dedup_feedback import SOURCE_SURFACE
    from tokenclaw.instruction_dedup_impact import build_instruction_dedup_impact_report

    result = build_instruction_dedup_impact_report(
        store_obj,
        limit=limit,
        since=since,
    )
    result["managed_lifecycle_feedback_queue"] = _managed_feedback_queue_health(
        store_obj,
        sample_limit=5,
        source_surface=SOURCE_SURFACE,
    )
    return result


async def stats_terminal_output_compaction_opportunity(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_text_chars: int = 8000,
    max_plateau_delta_ratio: float = 0.03,
) -> dict[str, Any]:
    from tokenclaw.terminal_compaction_report import build_terminal_output_compaction_opportunity_report

    return build_terminal_output_compaction_opportunity_report(
        store_obj,
        limit=limit,
        min_text_chars=min_text_chars,
        max_plateau_delta_ratio=max_plateau_delta_ratio,
    )


def _terminal_output_compaction_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "dashboard_read_only": True,
        "raw_terminal_lines_included": False,
        "raw_terminal_text_included": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_tool_payloads_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "filesystem_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "policy_file_contents_included": False,
        "yaml_contents_included": False,
        "rule_path_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_files": False,
        "wrote_store": False,
        "basis": "terminal-output compaction policy metadata plus sanitized local opportunity and canary impact aggregates only",
    }


def _terminal_output_compaction_state(
    *,
    policy_enabled: bool,
    opportunity_summary: dict[str, Any],
    impact_summary: dict[str, Any],
    impact_candidates: list[dict[str, Any]],
) -> tuple[str, str]:
    verdicts = {str(row.get("verdict") or "") for row in impact_candidates if isinstance(row, dict)}
    candidate_count = _as_int(opportunity_summary.get("candidate_count"))
    projected_saved_tokens = _as_int(opportunity_summary.get("projected_saved_tokens"))
    applied_count = _as_int(impact_summary.get("applied_count"))
    holdout_count = _as_int(impact_summary.get("holdout_count"))
    safety_stop_count = _as_int(impact_summary.get("safety_stop_count"))
    net_savings = _as_float(impact_summary.get("net_savings_usd"))
    rollback_actions = _as_int(impact_summary.get("rollback_action_count"))

    if rollback_actions or "rollback" in verdicts:
        return "rollback", "local impact gates recommend review-only rollback"
    if safety_stop_count:
        return "safety-stopped", "local safety-stop metadata was observed"
    if applied_count and net_savings > 0 and "promote" in verdicts:
        return "saving", "applied canary evidence is positive"
    if applied_count or holdout_count:
        return "canarying", "applied or holdout canary metadata is present"
    if not policy_enabled:
        return "disabled", "terminal-output compaction policy is disabled"
    if candidate_count and projected_saved_tokens > 0:
        return "ready", "candidate cohorts have projected savings"
    if candidate_count:
        return "blocked", "candidate cohorts exist but blockers or low savings prevent readiness"
    return "no-candidates", "no terminal-output compaction candidate cohorts found in the sampled window"


def _terminal_output_compaction_policy_state() -> dict[str, Any]:
    from tokenclaw import crunch

    policy = getattr(crunch, "TERMINAL_OUTPUT_COMPACTION_POLICY", {}) or {}
    effective_policy = (
        crunch.terminal_output_compaction_effective_policy()
        if hasattr(crunch, "terminal_output_compaction_effective_policy")
        else {}
    )
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    safety = policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {}
    file_state = policy_file_status(
        getattr(crunch, "CRUNCH_RULES_PATH", None),
        loaded_at=getattr(crunch, "CRUNCH_RULES_LOADED_AT", None),
        loaded_snapshot=getattr(crunch, "CRUNCH_RULES_LOADED_FILE", None),
    )
    raw_effective_rules = effective_policy.get("rules") if isinstance(effective_policy.get("rules"), list) else []
    effective_rules: list[dict[str, Any]] = []
    for rule in raw_effective_rules:
        if not isinstance(rule, dict):
            continue
        rule_canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
        effective_rules.append({
            "enabled": bool(rule.get("enabled")),
            "policy_source": str(rule.get("policy_source") or "unknown"),
            "rule_id": rule.get("rule_id"),
            "candidate_id": rule.get("candidate_id"),
            "action_id": rule.get("action_id"),
            "conditions": rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {},
            "action": rule.get("action") if isinstance(rule.get("action"), dict) else {},
            "canary": {
                "enabled": bool(rule_canary.get("enabled", True)),
                "fraction": _as_float(rule_canary.get("fraction")),
                "holdout_fraction": _as_float(rule_canary.get("holdout_fraction")),
                "unit": str(rule_canary.get("unit") or "request_fingerprint"),
                "salt_configured": bool(rule_canary.get("salt_configured")),
            },
            "safety_stop": rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {},
            "provenance": rule.get("provenance") if isinstance(rule.get("provenance"), dict) else None,
        })
    return {
        "enabled": bool(policy.get("enabled")),
        "policy_source": str(policy.get("policy_source") or getattr(crunch, "CRUNCH_POLICY_SOURCE", "unknown")),
        "rule_id": str(policy.get("rule_id") or "local-terminal-output-compaction-canary"),
        "candidate_id_configured": policy.get("candidate_id") is not None,
        "conditions": effective_policy.get("conditions") if isinstance(effective_policy.get("conditions"), dict) else {},
        "action": effective_policy.get("action") if isinstance(effective_policy.get("action"), dict) else {},
        "rule_count": len(effective_rules),
        "rules": effective_rules,
        "rule_file": {
            "configured": bool(getattr(crunch, "CRUNCH_RULES_PATH", None)),
            "path_class": _local_path_class(getattr(crunch, "CRUNCH_RULES_PATH", None)),
            "reload_required": bool(file_state.get("reload_required")),
            "rule_path_included": False,
            "policy_file_contents_included": False,
            "yaml_contents_included": False,
        },
        "canary": {
            "enabled": bool(canary.get("enabled", True)),
            "fraction": _as_float(canary.get("fraction")),
            "holdout_fraction": _as_float(canary.get("holdout_fraction")),
            "unit": str(canary.get("unit") or "request_fingerprint"),
            "salt_configured": bool(canary.get("salt")),
        },
        "safety_stop": {
            "enabled": bool(safety.get("enabled", True)),
            "min_outcome_samples": _as_int(safety.get("min_outcome_samples")),
            "window": _as_int(safety.get("window")),
            "max_error_rate": _as_float(safety.get("max_error_rate")),
            "max_retry_rate": _as_float(safety.get("max_retry_rate")),
            "max_negative_savings_rate": _as_float(safety.get("max_negative_savings_rate")),
            "max_error_rate_delta": _as_float(safety.get("max_error_rate_delta")),
        },
    }


def _terminal_output_compaction_public_rule(rule: dict[str, Any]) -> dict[str, Any]:
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {}
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    return {
        "enabled": bool(rule.get("enabled")),
        "policy_source": public_label(rule.get("policy_source") or "unknown", "unknown"),
        "rule_id": public_id(rule.get("rule_id"), prefix="terminal-compaction-rule", fallback="unknown"),
        "candidate_id": public_id(rule.get("candidate_id"), prefix="terminal-compaction-candidate"),
        "action_id": public_id(rule.get("action_id"), prefix="terminal-compaction-action"),
        "action_type": public_label(action.get("type") or "compact_terminal_output", "compact_terminal_output"),
        "canary": {
            "enabled": bool(canary.get("enabled", True)),
            "fraction": _as_float(canary.get("fraction")),
            "holdout_fraction": _as_float(canary.get("holdout_fraction")),
            "unit": public_label(canary.get("unit") or "request_fingerprint", "request_fingerprint"),
            "salt_configured": bool(canary.get("salt_configured")),
        },
        "safety_stop": {
            "enabled": bool(safety.get("enabled", True)),
            "min_outcome_samples": _as_int(safety.get("min_outcome_samples")),
            "window": _as_int(safety.get("window")),
        },
        "conditions_included": False,
        "policy_file_contents_included": False,
    }


def _terminal_output_compaction_count_lookup(rows: Any) -> dict[str, int]:
    if not isinstance(rows, list):
        return {}
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        result[str(row.get("value") or "unknown")] = _as_int(row.get("count"))
    return result


def _terminal_output_compaction_lifecycle_summary(lifecycle: dict[str, Any]) -> dict[str, Any]:
    queue_counts = _terminal_output_compaction_count_lookup(lifecycle.get("queue_state_breakdown"))
    lifecycle_counts = _terminal_output_compaction_count_lookup(lifecycle.get("lifecycle_status_breakdown"))
    cohort_counts = _terminal_output_compaction_count_lookup(lifecycle.get("cohort_count_breakdown"))
    raw_reason_counts = lifecycle.get("reason_code_breakdown") if isinstance(lifecycle.get("reason_code_breakdown"), list) else []
    reason_counts = [
        {"value": public_label(row.get("value"), "sanitized-reason"), "count": _as_int(row.get("count"))}
        for row in raw_reason_counts
        if isinstance(row, dict)
    ]
    return {
        "schema": lifecycle.get("schema"),
        "queue_rows": _as_int(lifecycle.get("queue_rows")),
        "action_count": _as_int(lifecycle.get("action_count")),
        "queued_count": queue_counts.get("queued", 0),
        "sent_count": queue_counts.get("sent", 0),
        "retryable_error_count": queue_counts.get("retryable-error", 0),
        "dropped_count": queue_counts.get("dropped-after-limit", 0),
        "reviewed_count": lifecycle_counts.get("reviewed", 0) + lifecycle_counts.get("review", 0),
        "applied_count": lifecycle_counts.get("applied", 0) + lifecycle_counts.get("canary-applied", 0),
        "rejected_count": lifecycle_counts.get("rejected", 0),
        "holdout_count": cohort_counts.get("canary_holdout", 0) + cohort_counts.get("holdout", 0),
        "safety_stop_count": lifecycle_counts.get("safety-stop", 0) + lifecycle_counts.get("safety_stopped", 0),
        "rollback_count": lifecycle_counts.get("rollback", 0),
        "queue_state_breakdown": lifecycle.get("queue_state_breakdown") if isinstance(lifecycle.get("queue_state_breakdown"), list) else [],
        "event_type_breakdown": lifecycle.get("event_type_breakdown") if isinstance(lifecycle.get("event_type_breakdown"), list) else [],
        "lifecycle_status_breakdown": lifecycle.get("lifecycle_status_breakdown") if isinstance(lifecycle.get("lifecycle_status_breakdown"), list) else [],
        "cohort_count_breakdown": lifecycle.get("cohort_count_breakdown") if isinstance(lifecycle.get("cohort_count_breakdown"), list) else [],
        "reason_code_breakdown": reason_counts,
        "payload_json_included": False,
    }


def _terminal_output_compaction_activation_status(
    *,
    policy: dict[str, Any],
    readiness_summary: dict[str, Any],
    lifecycle_summary: dict[str, Any],
) -> tuple[str, str]:
    if _as_int(readiness_summary.get("rollback_action_count")) or _as_int(lifecycle_summary.get("rollback_count")):
        return "rollback-ready", "rollback metadata or rollback action readiness is present"
    if _as_int(readiness_summary.get("safety_stop_count")) or _as_int(lifecycle_summary.get("safety_stop_count")):
        return "safety-stopped", "terminal-output compaction safety-stop metadata is present"
    if _as_int(readiness_summary.get("applied_count")):
        return "applied", "local terminal-output compaction canary has applied rows"
    if _as_int(readiness_summary.get("holdout_count")):
        return "holdout", "local terminal-output compaction has holdout rows"
    if bool(policy.get("enabled")) and (_as_int(policy.get("rule_count")) or bool(policy.get("candidate_id_configured"))):
        return "staged", "terminal-output compaction policy is staged locally"
    if bool(policy.get("enabled")):
        return "available", "terminal-output compaction policy is enabled with bundled defaults"
    return "disabled", "terminal-output compaction policy is disabled"


def _terminal_output_compaction_latest_safety_reason(readiness_summary: dict[str, Any], lifecycle_summary: dict[str, Any]) -> str | None:
    reason_counts = readiness_summary.get("reason_code_counts")
    if isinstance(reason_counts, list):
        for row in reason_counts:
            if isinstance(row, dict) and row.get("value"):
                return public_label(row.get("value"), "sanitized-reason")
    lifecycle_reasons = lifecycle_summary.get("reason_code_breakdown")
    if isinstance(lifecycle_reasons, list):
        for row in lifecycle_reasons:
            if isinstance(row, dict) and row.get("value"):
                return public_label(row.get("value"), "sanitized-reason")
    recent = readiness_summary.get("recent_reason_codes")
    if isinstance(recent, list) and recent:
        return public_label(recent[0], "sanitized-reason")
    return None


def _terminal_output_compaction_public_lifecycle_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": public_id(row.get("candidate_id"), prefix="terminal-compaction-candidate", fallback="unknown"),
        "queue_rows": _as_int(row.get("queue_rows")),
        "action_count": _as_int(row.get("action_count")),
        "net_savings_usd": round(_as_float(row.get("net_savings_usd")), 8),
        "projected_saved_tokens": _as_int(row.get("projected_saved_tokens")),
        "rule_id_breakdown": [
            {"value": public_id(item.get("value"), prefix="terminal-compaction-rule", fallback="unknown"), "count": _as_int(item.get("count"))}
            for item in row.get("rule_id_breakdown") or []
            if isinstance(item, dict)
        ],
        "queue_state_breakdown": row.get("queue_state_breakdown") if isinstance(row.get("queue_state_breakdown"), list) else [],
        "event_type_breakdown": row.get("event_type_breakdown") if isinstance(row.get("event_type_breakdown"), list) else [],
        "lifecycle_status_breakdown": row.get("lifecycle_status_breakdown") if isinstance(row.get("lifecycle_status_breakdown"), list) else [],
        "cohort_count_breakdown": row.get("cohort_count_breakdown") if isinstance(row.get("cohort_count_breakdown"), list) else [],
        "reason_code_breakdown": [
            {"value": public_label(item.get("value"), "sanitized-reason"), "count": _as_int(item.get("count"))}
            for item in row.get("reason_code_breakdown") or []
            if isinstance(item, dict)
        ],
        "payload_json_included": False,
    }


async def stats_terminal_output_compaction_activation(
    store_obj: Any,
    *,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
) -> dict[str, Any]:
    from tokenclaw.optimization.feedback import managed_feedback_status_result
    from tokenclaw.terminal_compaction_feedback import SOURCE_SURFACE as TERMINAL_COMPACTION_LIFECYCLE_SOURCE_SURFACE

    readiness = await stats_terminal_output_compaction_readiness(
        store_obj,
        opportunity_limit=opportunity_limit,
        impact_limit=impact_limit,
    )
    policy = readiness.get("policy") if isinstance(readiness.get("policy"), dict) else {}
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), dict) else {}
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    feedback = managed_feedback_status_result(
        store_obj,
        source_surface=TERMINAL_COMPACTION_LIFECYCLE_SOURCE_SURFACE,
        sample_limit=5,
    )
    lifecycle = feedback.get("terminal_output_compaction_lifecycle") if isinstance(feedback.get("terminal_output_compaction_lifecycle"), dict) else {}
    lifecycle_summary = _terminal_output_compaction_lifecycle_summary(lifecycle)
    status, status_reason = _terminal_output_compaction_activation_status(
        policy=policy,
        readiness_summary=summary,
        lifecycle_summary=lifecycle_summary,
    )
    safety_stop_count = _as_int(summary.get("safety_stop_count")) + _as_int(lifecycle_summary.get("safety_stop_count"))
    rollback_action_count = _as_int(summary.get("rollback_action_count"))
    latest_safety_reason = _terminal_output_compaction_latest_safety_reason(summary, lifecycle_summary) if safety_stop_count else None
    rules = [
        _terminal_output_compaction_public_rule(rule)
        for rule in policy.get("rules") or []
        if isinstance(rule, dict)
    ]
    if not rules and policy:
        rules = [_terminal_output_compaction_public_rule({
            "enabled": policy.get("enabled"),
            "policy_source": policy.get("policy_source"),
            "rule_id": policy.get("rule_id"),
            "candidate_id": None,
            "action_id": None,
            "canary": canary,
            "safety_stop": policy.get("safety_stop"),
            "action": policy.get("action"),
        })]

    impact_rows: list[dict[str, Any]] = []
    for row in readiness.get("impact_gates") or []:
        if not isinstance(row, dict):
            continue
        impact_rows.append({
            "candidate_id": public_id(row.get("candidate_id"), prefix="terminal-compaction-candidate", fallback="unknown"),
            "rule_id": public_id(row.get("rule_id"), prefix="terminal-compaction-rule", fallback="unknown"),
            "policy_source": public_label(row.get("policy_source") or "unknown", "unknown"),
            "activation_state": row.get("verdict") or "observed",
            "applied_count": _as_int(row.get("applied_count")),
            "holdout_count": _as_int(row.get("holdout_count")),
            "bypass_count": _as_int(row.get("safety_stop_count")),
            "safety_stop_count": _as_int(row.get("safety_stop_count")),
            "projected_saved_tokens": _as_int(row.get("planned_saved_tokens")),
            "observed_saved_tokens": _as_int(row.get("applied_saved_tokens")),
            "projected_savings_usd": round(_as_float(row.get("projected_holdout_savings_usd")), 8),
            "observed_savings_usd": round(_as_float(row.get("net_savings_usd")), 8),
            "rollback_action_ready": str(row.get("verdict") or "") == "rollback",
            "latest_safety_stop_reason": latest_safety_reason if _as_int(row.get("safety_stop_count")) else None,
            "reason_codes": [
                public_label(value, "sanitized-reason")
                for value in (row.get("reason_codes") or [])
            ] if isinstance(row.get("reason_codes"), list) else [],
        })

    lifecycle_candidates = [
        _terminal_output_compaction_public_lifecycle_candidate(row)
        for row in lifecycle.get("candidate_breakdown") or []
        if isinstance(row, dict)
    ]

    states = [
        {"state": "staged", "active": status in {"staged", "available", "applied", "holdout", "safety-stopped", "rollback-ready"}, "count": len(rules)},
        {"state": "applied", "active": _as_int(summary.get("applied_count")) > 0, "count": _as_int(summary.get("applied_count"))},
        {"state": "holdout", "active": _as_int(summary.get("holdout_count")) > 0, "count": _as_int(summary.get("holdout_count"))},
        {"state": "safety-stopped", "active": safety_stop_count > 0, "count": safety_stop_count},
        {"state": "rollback-ready", "active": rollback_action_count > 0, "count": rollback_action_count},
    ]

    return {
        "schema": "tokenclaw.terminal_output_compaction_activation.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "status": status,
        "status_reason": status_reason,
        "summary": {
            "local_rule_staged_count": len(rules),
            "active_rule_count": sum(1 for rule in rules if rule.get("enabled")),
            "policy_source": public_label(policy.get("policy_source") or "unknown", "unknown"),
            "policy_enabled": bool(policy.get("enabled")),
            "canary_fraction": _as_float(canary.get("fraction")),
            "holdout_fraction": _as_float(canary.get("holdout_fraction")),
            "applied_count": _as_int(summary.get("applied_count")),
            "holdout_count": _as_int(summary.get("holdout_count")),
            "bypass_count": _as_int(summary.get("skipped_count")) + safety_stop_count,
            "safety_stop_count": safety_stop_count,
            "rollback_action_count": rollback_action_count,
            "rollback_action_ready": rollback_action_count > 0 or status == "rollback-ready",
            "latest_safety_stop_reason": latest_safety_reason,
            "projected_saved_tokens": _as_int(summary.get("projected_saved_tokens")),
            "projected_savings_usd": round(_as_float(summary.get("projected_saved_usd")), 8),
            "observed_saved_tokens": _as_int(summary.get("applied_saved_tokens")),
            "observed_savings_usd": round(_as_float(summary.get("net_savings_usd")), 8),
            "managed_lifecycle_feedback_rows": _as_int(lifecycle_summary.get("queue_rows")),
            "managed_lifecycle_feedback_queued": _as_int(lifecycle_summary.get("queued_count")),
            "managed_lifecycle_feedback_retryable_error": _as_int(lifecycle_summary.get("retryable_error_count")),
            "managed_lifecycle_feedback_dropped": _as_int(lifecycle_summary.get("dropped_count")),
        },
        "states": states,
        "policy": {
            "enabled": bool(policy.get("enabled")),
            "policy_source": public_label(policy.get("policy_source") or "unknown", "unknown"),
            "rule_id": public_id(policy.get("rule_id"), prefix="terminal-compaction-rule", fallback="unknown"),
            "candidate_id_configured": bool(policy.get("candidate_id_configured")),
            "rule_count": _as_int(policy.get("rule_count")),
            "rule_file": policy.get("rule_file") if isinstance(policy.get("rule_file"), dict) else {},
            "canary": {
                "enabled": bool(canary.get("enabled", True)),
                "fraction": _as_float(canary.get("fraction")),
                "holdout_fraction": _as_float(canary.get("holdout_fraction")),
                "unit": public_label(canary.get("unit") or "request_fingerprint", "request_fingerprint"),
                "salt_configured": bool(canary.get("salt_configured")),
            },
            "rules": rules,
            "conditions_included": False,
            "action_contents_included": False,
            "policy_file_contents_included": False,
        },
        "activation_candidates": impact_rows,
        "lifecycle_feedback": lifecycle_summary,
        "lifecycle_candidates": lifecycle_candidates,
        "privacy": _terminal_output_compaction_privacy(),
    }


async def stats_terminal_output_compaction_readiness(
    store_obj: Any,
    *,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
    min_text_chars: int = 8000,
    max_plateau_delta_ratio: float = 0.03,
) -> dict[str, Any]:
    from tokenclaw.terminal_compaction_impact import build_terminal_output_compaction_impact_report

    policy = _terminal_output_compaction_policy_state()
    opportunity = await stats_terminal_output_compaction_opportunity(
        store_obj,
        limit=opportunity_limit,
        min_text_chars=min_text_chars,
        max_plateau_delta_ratio=max_plateau_delta_ratio,
    )
    impact = build_terminal_output_compaction_impact_report(store_obj, limit=impact_limit)
    opportunity_summary = opportunity.get("summary") if isinstance(opportunity.get("summary"), dict) else {}
    impact_summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    impact_candidates = [row for row in impact.get("candidates") or [] if isinstance(row, dict)]
    state, state_reason = _terminal_output_compaction_state(
        policy_enabled=bool(policy.get("enabled")),
        opportunity_summary=opportunity_summary,
        impact_summary=impact_summary,
        impact_candidates=impact_candidates,
    )
    latest = impact_candidates[0] if impact_candidates else {}
    latest_cohorts = latest.get("cohorts") if isinstance(latest.get("cohorts"), dict) else {}
    latest_applied = latest_cohorts.get("applied") if isinstance(latest_cohorts.get("applied"), dict) else {}

    summary = {
        "state": state,
        "state_reason": state_reason,
        "opportunity_candidate_count": _as_int(opportunity_summary.get("candidate_count")),
        "matched_count": _as_int(opportunity_summary.get("matched_count")),
        "plateau_pair_count": _as_int(opportunity_summary.get("plateau_pair_count")),
        "terminal_signal_rows": _as_int(opportunity_summary.get("terminal_signal_rows")),
        "projected_saved_tokens": _as_int(opportunity_summary.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(opportunity_summary.get("projected_saved_usd")), 8),
        "impact_candidate_count": _as_int(impact_summary.get("candidate_group_count")),
        "observed_terminal_output_compaction_metadata_row_count": _as_int(
            impact_summary.get("observed_terminal_output_compaction_metadata_row_count")
        ),
        "applied_count": _as_int(impact_summary.get("applied_count")),
        "holdout_count": _as_int(impact_summary.get("holdout_count")),
        "skipped_count": _as_int(impact_summary.get("skipped_count")),
        "safety_stop_count": _as_int(impact_summary.get("safety_stop_count")),
        "applied_saved_tokens": sum(
            _as_int(((row.get("cohorts") or {}).get("applied") or {}).get("tokens_saved_est"))
            for row in impact_candidates
            if isinstance(row.get("cohorts"), dict)
        ),
        "net_savings_usd": round(_as_float(impact_summary.get("net_savings_usd")), 8),
        "projected_holdout_savings_usd": round(_as_float(impact_summary.get("projected_holdout_savings_usd")), 8),
        "rollback_action_count": _as_int(impact_summary.get("rollback_action_count")),
        "recent_verdict": latest.get("verdict") if latest else None,
        "recent_reason_codes": latest.get("reason_codes") if isinstance(latest.get("reason_codes"), list) else [],
        "recent_error_rate_delta": (latest.get("deltas") or {}).get("error_rate_delta") if isinstance(latest.get("deltas"), dict) else None,
        "recent_retry_rate_delta": (latest.get("deltas") or {}).get("retry_rate_delta") if isinstance(latest.get("deltas"), dict) else None,
        "recent_latency_avg_ms_delta": (latest.get("deltas") or {}).get("latency_avg_ms_delta") if isinstance(latest.get("deltas"), dict) else None,
        "recent_applied_error_rate": latest_applied.get("error_rate"),
        "recent_applied_retry_rate": latest_applied.get("retry_rate"),
        "verdict_counts": impact_summary.get("verdict_counts") if isinstance(impact_summary.get("verdict_counts"), list) else [],
        "reason_code_counts": impact_summary.get("reason_code_counts") if isinstance(impact_summary.get("reason_code_counts"), list) else [],
        "top_blockers": opportunity.get("blocker_reason_breakdown") if isinstance(opportunity.get("blocker_reason_breakdown"), list) else [],
    }

    readiness_candidates: list[dict[str, Any]] = []
    for row in (opportunity.get("candidates") or [])[:25]:
        if not isinstance(row, dict):
            continue
        readiness_candidates.append({
            "candidate_id": row.get("candidate_id"),
            "provider": row.get("provider"),
            "source_surface": row.get("source_surface"),
            "category": row.get("category"),
            "workflow_phase": row.get("workflow_phase"),
            "requested_model_family": row.get("requested_model_family"),
            "routed_model_family": row.get("routed_model_family"),
            "matched_count": _as_int(row.get("matched_count")),
            "plateau_pair_count": _as_int(row.get("plateau_pair_count")),
            "terminal_signal_rows": _as_int(row.get("terminal_signal_rows")),
            "body_rows": _as_int(row.get("body_rows")),
            "metadata_only_rows": _as_int(row.get("metadata_only_rows")),
            "projected_saved_tokens": _as_int(row.get("projected_saved_tokens")),
            "projected_saved_usd": round(_as_float(row.get("projected_saved_usd")), 8),
            "blocker_reason_breakdown": row.get("blocker_reason_breakdown") if isinstance(row.get("blocker_reason_breakdown"), list) else [],
            "privacy": row.get("privacy") if isinstance(row.get("privacy"), dict) else _terminal_output_compaction_privacy(),
        })

    impact_rows: list[dict[str, Any]] = []
    for row in impact_candidates[:25]:
        cohorts = row.get("cohorts") if isinstance(row.get("cohorts"), dict) else {}
        applied = cohorts.get("applied") if isinstance(cohorts.get("applied"), dict) else {}
        holdout = cohorts.get("holdout") if isinstance(cohorts.get("holdout"), dict) else {}
        safety = cohorts.get("safety_stop") if isinstance(cohorts.get("safety_stop"), dict) else {}
        impact_rows.append({
            "candidate_id": row.get("candidate_id"),
            "rule_id": row.get("rule_id"),
            "policy_source": row.get("policy_source"),
            "provider": row.get("provider"),
            "source_surface": row.get("source_surface"),
            "category": row.get("category"),
            "workflow_phase": row.get("workflow_phase"),
            "requested_model_family": row.get("requested_model_family"),
            "routed_model_family": row.get("routed_model_family"),
            "verdict": row.get("verdict"),
            "reason_codes": row.get("reason_codes") if isinstance(row.get("reason_codes"), list) else [],
            "applied_count": _as_int(applied.get("count")),
            "holdout_count": _as_int(holdout.get("count")),
            "safety_stop_count": _as_int(safety.get("count")),
            "applied_saved_tokens": _as_int(applied.get("tokens_saved_est")),
            "planned_saved_tokens": _as_int(applied.get("planned_saved_tokens")) + _as_int(holdout.get("planned_saved_tokens")),
            "net_savings_usd": round(_as_float(row.get("net_savings_usd")), 8),
            "projected_holdout_savings_usd": round(_as_float(row.get("projected_holdout_savings_usd")), 8),
            "error_rate_delta": (row.get("deltas") or {}).get("error_rate_delta") if isinstance(row.get("deltas"), dict) else None,
            "retry_rate_delta": (row.get("deltas") or {}).get("retry_rate_delta") if isinstance(row.get("deltas"), dict) else None,
            "latency_avg_ms_delta": (row.get("deltas") or {}).get("latency_avg_ms_delta") if isinstance(row.get("deltas"), dict) else None,
            "privacy": row.get("privacy") if isinstance(row.get("privacy"), dict) else _terminal_output_compaction_privacy(),
        })

    return {
        "schema": "tokenclaw.terminal_output_compaction_readiness.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "state": state,
        "state_reason": state_reason,
        "policy": policy,
        "summary": summary,
        "safety_stop": {
            "enabled": bool((policy.get("safety_stop") or {}).get("enabled")),
            "active": state in {"safety-stopped", "rollback"} or _as_int(summary.get("safety_stop_count")) > 0,
            "observed_count": _as_int(summary.get("safety_stop_count")),
            "rollback_action_count": _as_int(summary.get("rollback_action_count")),
            "reason_code_counts": summary["reason_code_counts"],
        },
        "provider_breakdown": opportunity.get("provider_breakdown") if isinstance(opportunity.get("provider_breakdown"), list) else [],
        "category_breakdown": opportunity.get("category_breakdown") if isinstance(opportunity.get("category_breakdown"), list) else [],
        "source_surface_breakdown": opportunity.get("source_surface_breakdown") if isinstance(opportunity.get("source_surface_breakdown"), list) else [],
        "blocker_reason_breakdown": opportunity.get("blocker_reason_breakdown") if isinstance(opportunity.get("blocker_reason_breakdown"), list) else [],
        "opportunity": {
            "schema": opportunity.get("schema"),
            "limit": opportunity.get("limit"),
            "projection_policy": opportunity.get("projection_policy"),
            "summary": opportunity_summary,
        },
        "impact": {
            "schema": impact.get("schema"),
            "status": impact.get("status"),
            "lookback_limit": impact.get("lookback_limit"),
            "thresholds": impact.get("thresholds"),
            "summary": impact_summary,
        },
        "candidate_cohorts": readiness_candidates,
        "impact_gates": impact_rows,
        "privacy": _terminal_output_compaction_privacy(),
    }


async def stats_anthropic_thinking_compaction_impact(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    from tokenclaw.anthropic_thinking_compaction_impact import build_anthropic_thinking_compaction_impact_report

    return build_anthropic_thinking_compaction_impact_report(
        store_obj,
        limit=limit,
        since=since,
    )


async def stats_repeated_scaffold_impact(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.repeated_scaffold_impact import build_repeated_scaffold_impact_report

    return build_repeated_scaffold_impact_report(store_obj, limit=limit)


async def stats_repeated_scaffold_activation(
    store_obj: Any,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    from tokenclaw.repeated_scaffold_activation import build_repeated_scaffold_activation_report

    return build_repeated_scaffold_activation_report(store_obj, limit=limit, since=since)


async def stats_scaffold_rollout_health(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.policy_events import recent_policy_events

    capped_limit = max(1, min(int(limit or 500), 5000))
    now = datetime.now(timezone.utc)
    events = [
        event
        for event in recent_policy_events(limit=500).get("events", [])
        if isinstance(event, dict) and str(event.get("action") or "") in SCAFFOLD_ROLLOUT_ACTIONS
    ]
    latest_fetch = _public_scaffold_rollout_event(
        _latest_policy_event(events, "scaffold-rollout-actions-review", "scaffold-rollout-actions-apply"),
        now=now,
    )
    latest_apply = _public_scaffold_rollout_event(
        _latest_policy_event(events, "scaffold-rollout-actions-apply"),
        now=now,
    )
    public_events = [
        item
        for item in (_public_scaffold_rollout_event(event, now=now) for event in events)
        if item is not None
    ]
    policy = _scaffold_canary_policy_health()
    impact: dict[str, Any] = {}
    try:
        impact = await stats_repeated_scaffold_impact(store_obj, limit=capped_limit)
    except Exception:
        impact = {}
    impact_summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    applied_count = _as_int(impact_summary.get("applied_count"))
    holdout_count = _as_int(impact_summary.get("holdout_count"))
    safety_stop_count = _as_int(impact_summary.get("safety_stop_count"))
    active_rule_count = _as_int(policy.get("active_rule_count"))
    status, status_reason = _scaffold_rollout_status(
        latest_fetch=latest_fetch,
        latest_apply=latest_apply,
        active_rule_count=active_rule_count,
        safety_stop_count=safety_stop_count,
    )
    summary = {
        "policy_event_count": len(public_events),
        "last_fetch_at": (latest_fetch or {}).get("created_at"),
        "last_fetch_status": (latest_fetch or {}).get("fetch_status"),
        "action_count": _as_int((latest_fetch or {}).get("action_count")),
        "accepted_action_count": _as_int((latest_fetch or {}).get("accepted_action_count")),
        "active_rule_count": active_rule_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "safety_stop_count": safety_stop_count,
        "observed_repeated_scaffold_metadata_row_count": _as_int(impact_summary.get("observed_repeated_scaffold_metadata_row_count")),
        "candidate_group_count": _as_int(impact_summary.get("candidate_group_count")),
        "estimated_saved_tokens": _as_int(impact_summary.get("estimated_saved_tokens")),
        "estimated_savings_usd": _as_float(impact_summary.get("estimated_savings_usd")),
    }
    return {
        "schema": "tokenclaw.scaffold_rollout_health.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "read_only": True,
        "limit": capped_limit,
        "summary": summary,
        "latest_fetch": latest_fetch,
        "latest_apply": latest_apply,
        "active_policy": policy,
        "canary_health": {
            "impact_status": impact.get("status"),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "safety_stop_count": safety_stop_count,
            "verdict_counts": impact_summary.get("verdict_counts") if isinstance(impact_summary.get("verdict_counts"), list) else [],
            "reason_code_counts": impact_summary.get("reason_code_counts") if isinstance(impact_summary.get("reason_code_counts"), list) else [],
        },
        "recent_events": public_events[:25],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "dashboard_read_only": True,
            "raw_action_payloads_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "local_session_ids_included": False,
            "payload_json_included": False,
            "yaml_contents_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local scaffold policy-event metadata, local canary policy file status, and repeated-scaffold impact aggregates only",
        },
    }


async def stats_openai_cache_replay_impact(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.openai_cache_replay_impact import build_openai_cache_replay_impact_report

    return build_openai_cache_replay_impact_report(store_obj, limit=limit)


async def stats_openai_cache_replay_readiness(
    store_obj: Any,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
) -> dict[str, Any]:
    from tokenclaw.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report

    return build_openai_cache_replay_readiness_report(
        store_obj,
        opportunity_limit=opportunity_limit,
        impact_limit=impact_limit,
    )


async def stats_openai_tool_cache_invalidation_burndown(
    store_obj: Any,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
    row_limit: int = 25,
) -> dict[str, Any]:
    from tokenclaw.openai_cache_replay_blocker_outcomes import (
        build_openai_cache_replay_blocker_outcomes_report,
    )

    capped_opportunity_limit = max(1, min(int(opportunity_limit or 1000), 10_000))
    capped_impact_limit = max(1, min(int(impact_limit or 500), 10_000))
    capped_row_limit = max(1, min(int(row_limit or 25), 100))
    report = build_openai_cache_replay_blocker_outcomes_report(
        store_obj,
        opportunity_limit=capped_opportunity_limit,
        impact_limit=capped_impact_limit,
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for raw_row in report.get("cohorts") or []:
        if not isinstance(raw_row, dict):
            continue
        outcome = str(raw_row.get("outcome") or "unknown")
        reason = str(raw_row.get("reason") or "unknown")
        sample_count = _as_int(raw_row.get("sample_count"))
        key = (
            "openai",
            str(raw_row.get("source_surface") or "unknown"),
            str(raw_row.get("endpoint") or "unknown"),
            str(raw_row.get("category") or "unknown"),
            str(raw_row.get("workflow_phase") or "unknown"),
        )
        row = grouped.setdefault(
            key,
            {
                "provider": "openai",
                "source_surface": key[1],
                "endpoint": key[2],
                "category": key[3],
                "workflow_phase": key[4],
                "sample_count": 0,
                "missing_dependency_evidence_count": 0,
                "safe_dependency_evidence_count": 0,
                "stale_dependency_count": 0,
                "unsafe_dependency_count": 0,
                "unknown_dependency_count": 0,
                "noop_count": 0,
                "projected_hits": 0,
                "projected_savings_usd": 0.0,
                "top_next_action": raw_row.get("next_action"),
                "reason_counts": {},
            },
        )
        row["sample_count"] += sample_count
        row["projected_hits"] += _as_int(raw_row.get("projected_hits"))
        row["projected_savings_usd"] += _as_float(raw_row.get("projected_savings_usd"))
        row["top_next_action"] = row.get("top_next_action") or raw_row.get("next_action")
        if outcome == "missing-invalidation":
            row["missing_dependency_evidence_count"] += sample_count
        elif outcome == "replay-ready":
            row["safe_dependency_evidence_count"] += sample_count
        elif outcome == "stale-dependency":
            row["stale_dependency_count"] += sample_count
        elif outcome == "unsafe-dependency":
            row["unsafe_dependency_count"] += sample_count
        elif outcome == "unknown-dependency":
            row["unknown_dependency_count"] += sample_count
        elif outcome == "noop":
            row["noop_count"] += sample_count
        row["reason_counts"][reason] = _as_int(row["reason_counts"].get(reason)) + sample_count
        reason_counts[reason] = _as_int(reason_counts.get(reason)) + sample_count

    blocker_rows: list[dict[str, Any]] = []
    for row in grouped.values():
        row["projected_savings_usd"] = round(_as_float(row.get("projected_savings_usd")), 8)
        row["reason_breakdown"] = _managed_breakdown(row.pop("reason_counts", {}))[:6]
        row["privacy"] = {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        }
        blocker_rows.append(row)
    blocker_rows.sort(
        key=lambda item: (
            _as_int(item.get("missing_dependency_evidence_count")),
            _as_int(item.get("unsafe_dependency_count")),
            _as_int(item.get("unknown_dependency_count")),
            _as_int(item.get("stale_dependency_count")),
            _as_int(item.get("safe_dependency_evidence_count")),
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("sample_count")),
        ),
        reverse=True,
    )
    blocker_rows = blocker_rows[:capped_row_limit]
    for rank, row in enumerate(blocker_rows, start=1):
        row["rank"] = rank

    return {
        "schema": "tokenclaw.openai_tool_cache_invalidation_burndown.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "opportunity_limit": capped_opportunity_limit,
        "impact_limit": capped_impact_limit,
        "row_limit": capped_row_limit,
        "status": report.get("status") or "unknown",
        "top_next_action": report.get("top_next_action"),
        "summary": {
            "openai_call_count": _as_int(summary.get("openai_call_count")),
            "opportunity_candidate_count": _as_int(summary.get("opportunity_candidate_count")),
            "impact_candidate_count": _as_int(summary.get("impact_candidate_count")),
            "observed_replay_metadata_rows": _as_int(summary.get("observed_replay_metadata_rows")),
            "missing_dependency_evidence_count": _as_int(summary.get("missing_invalidation_count")),
            "safe_dependency_evidence_count": _as_int(summary.get("replay_ready_count")),
            "stale_dependency_count": _as_int(summary.get("stale_dependency_count")),
            "unsafe_dependency_count": _as_int(summary.get("unsafe_dependency_count")),
            "unknown_dependency_count": _as_int(summary.get("unknown_dependency_count")),
            "staged_canary_count": _as_int(summary.get("staged_canary_count")),
            "staged_canary_policy_status": summary.get("staged_canary_policy_status"),
            "applied_count": _as_int(summary.get("applied_count")),
            "holdout_count": _as_int(summary.get("holdout_count")),
            "exact_hit_count": _as_int(summary.get("exact_hit_count")),
            "safety_stop_count": _as_int(summary.get("safety_stop_count")),
            "invalidated_count": _as_int(summary.get("invalidated_count")),
            "projected_savings_usd": round(_as_float(summary.get("projected_savings_usd")), 8),
            "observed_savings_usd": round(_as_float(summary.get("observed_savings_usd")), 8),
            "ranked_blocker_count": len(blocker_rows),
            "top_next_action": report.get("top_next_action"),
            "dependency_evidence_classes": [
                "stable-dependency-evidence",
                "stale-dependency-evidence",
                "unsafe-dependency-evidence",
                "unknown-dependency-evidence",
                "missing-dependency-evidence",
            ],
            "tool_cache_replay_enabled": False,
            "streaming_replay_enabled": False,
            "cache_apply_action_count": 0,
            "cache_entries_written": 0,
            "policy_files_written": False,
        },
        "blockers": blocker_rows,
        "reason_breakdown": _managed_breakdown(reason_counts),
        "outcome_breakdown": report.get("outcome_breakdown") if isinstance(report.get("outcome_breakdown"), list) else [],
        "source_reports": {
            "blocker_outcomes_schema": report.get("schema"),
            "opportunity_limit": capped_opportunity_limit,
            "impact_limit": capped_impact_limit,
            "raw_source_reports_included": False,
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "dashboard_read_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "absolute_paths_included": False,
            "file_paths_included": False,
            "filesystem_paths_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "bounded OpenAI cache replay blocker, invalidation, and canary lifecycle aggregates only",
        },
    }


async def stats_openai_old_context_summary_report(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.openai_old_context_summary_report import build_openai_old_context_summary_report
    from tokenclaw.openai_old_context_summary import load_openai_old_context_summary_policy

    report = build_openai_old_context_summary_report(store_obj, limit=limit)
    try:
        policy = load_openai_old_context_summary_policy()
    except Exception as exc:
        policy = {
            "enabled": False,
            "policy_source": "unavailable",
            "summary_provider": None,
            "summary_model": None,
            "rule_id": None,
            "rule_path": None,
            "load_error_class": type(exc).__name__,
        }
    try:
        policy_state = await stats_policies()
    except Exception:
        policy_state = {}
    crunch_state = policy_state.get("crunch") if isinstance(policy_state.get("crunch"), dict) else {}
    crunch_file = crunch_state.get("file") if isinstance(crunch_state.get("file"), dict) else {}
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    local_policy = {
        "schema": "tokenclaw.openai_old_context_summary_dashboard_policy.v1",
        "enabled": bool(policy.get("enabled")),
        "policy_source": policy.get("policy_source") or "unknown",
        "summary_provider": policy.get("summary_provider"),
        "summary_model": policy.get("summary_model"),
        "rule_id": policy.get("rule_id"),
        "supported_endpoints": policy.get("supported_endpoints") if isinstance(policy.get("supported_endpoints"), list) else [],
        "blocked_categories": policy.get("blocked_categories") if isinstance(policy.get("blocked_categories"), list) else [],
        "canary": {
            "enabled": bool(canary.get("enabled")),
            "canary_fraction": _as_float(canary.get("canary_fraction")),
            "holdout_fraction": _as_float(canary.get("holdout_fraction")),
            "cohort_basis": canary.get("cohort_basis") or "deterministic-candidate-id-hash",
        },
        "rule_file": {
            "configured": bool(policy.get("rule_path")),
            "path_class": _local_path_class(policy.get("rule_path")),
            "rule_path_included": False,
            "reload_required": bool(crunch_file.get("reload_required")),
            "loaded": bool(((crunch_file.get("loaded") or {}) if isinstance(crunch_file.get("loaded"), dict) else {}).get("exists")),
        },
        "read_only": True,
        "dashboard_mutations_available": False,
        "provider_calls_made": False,
        "load_error_class": policy.get("load_error_class"),
    }
    report["local_policy"] = local_policy
    measurement = report.get("measurement_policy") if isinstance(report.get("measurement_policy"), dict) else {}
    measurement.update({
        "policy_enabled": local_policy["enabled"],
        "policy_source": local_policy["policy_source"],
        "policy_reload_required": local_policy["rule_file"]["reload_required"],
        "dashboard_mutations_available": False,
    })
    report["measurement_policy"] = measurement
    privacy = report.get("privacy") if isinstance(report.get("privacy"), dict) else {}
    privacy.setdefault("raw_request_bodies_included", False)
    privacy.setdefault("file_paths_included", False)
    privacy.setdefault("provider_calls_made", False)
    privacy.setdefault("managed_server_calls_made", False)
    report["privacy"] = privacy
    return report


def _optimization_eval_reason_codes(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if len(text) > 80 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-" for ch in text):
            result.add("unsanitized-reason-code")
        else:
            result.add(text)
    return sorted(result)


def _optimization_eval_is_privacy_blocked(reason_codes: list[str], replayability_level: Any) -> bool:
    text = " ".join(reason_codes + [str(replayability_level or "")]).lower()
    return any(
        marker in text
        for marker in (
            "privacy",
            "raw",
            "prompt",
            "body",
            "payload",
            "secret",
            "identifier",
            "filesystem",
            "file-path",
            "api-key",
            "egress",
        )
    )


def _optimization_eval_status(evidence: dict[str, Any]) -> str:
    result_count = _as_int(evidence.get("result_count"))
    if result_count <= 0:
        return "missing"
    if _as_int(evidence.get("fail_count")):
        return "fail"
    if _as_int(evidence.get("blocked_count")) and not _as_int(evidence.get("pass_count")):
        return "blocked"
    if _as_int(evidence.get("unknown_count")) and not _as_int(evidence.get("pass_count")):
        return "unknown"
    if _as_int(evidence.get("pass_count")):
        return "pass"
    return "mixed"


def _optimization_eval_candidate_row(plan_row: dict[str, Any], verdict_row: dict[str, Any]) -> dict[str, Any]:
    evidence = verdict_row.get("eval_evidence") if isinstance(verdict_row.get("eval_evidence"), dict) else {}
    blockers = _optimization_eval_reason_codes(plan_row.get("blocker_reason_codes"))
    reasons = _optimization_eval_reason_codes(verdict_row.get("reason_codes"))
    privacy_blocked = _optimization_eval_is_privacy_blocked(blockers + reasons, plan_row.get("replayability_level"))
    eval_status = _optimization_eval_status(evidence)
    verdict = str(verdict_row.get("verdict") or "needs_eval")
    queue_status = "privacy_blocked" if privacy_blocked else verdict
    return {
        "candidate_id": str(plan_row.get("candidate_id") or verdict_row.get("candidate_id") or "unknown"),
        "action_family": str(plan_row.get("action_family") or verdict_row.get("action_family") or "unknown"),
        "optimization_family": str(plan_row.get("optimization_family") or verdict_row.get("optimization_family") or "unknown"),
        "source_surface": str(plan_row.get("source_surface") or verdict_row.get("source_surface") or "unknown"),
        "app_family": str(plan_row.get("app_family") or verdict_row.get("app_family") or "unknown"),
        "workflow_phase": str(plan_row.get("workflow_phase") or "unknown"),
        "category": str(plan_row.get("category") or "unknown"),
        "candidate_target_model": plan_row.get("candidate_target_model") or verdict_row.get("candidate_target_model"),
        "candidate_profile": plan_row.get("candidate_profile") or verdict_row.get("candidate_profile"),
        "projected_savings_usd": round(_as_float(plan_row.get("projected_savings_usd") or verdict_row.get("projected_savings_usd")), 8),
        "sample_count": _as_int(plan_row.get("sample_count") or verdict_row.get("sample_count")),
        "current_canary_count": _as_int(plan_row.get("current_canary_count")),
        "holdout_count": _as_int(plan_row.get("holdout_count")),
        "recommended_eval_mode": str(plan_row.get("recommended_eval_mode") or "collect-baseline-evidence"),
        "replayability_level": str(plan_row.get("replayability_level") or "metadata_only"),
        "eval_status": eval_status,
        "eval_result_count": _as_int(evidence.get("result_count")),
        "eval_pass_count": _as_int(evidence.get("pass_count")),
        "eval_fail_count": _as_int(evidence.get("fail_count")),
        "eval_blocked_count": _as_int(evidence.get("blocked_count")),
        "last_evidence_at": evidence.get("latest_result_at"),
        "eval_evidence_stale": bool(evidence.get("stale")),
        "verdict": verdict,
        "queue_status": queue_status,
        "privacy_blocked": privacy_blocked,
        "blocker_reason_codes": blockers,
        "reason_codes": reasons,
        "next_action": str(verdict_row.get("next_action") or "run_local_shadow_eval_or_collect_canary_holdout_evidence"),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
        },
    }


async def stats_post_fix_shadow_yield(
    store_obj: Any,
    limit: int = 50,
    since: str | None = None,
    window_hours: float = 24.0,
) -> dict[str, Any]:
    return build_post_fix_shadow_yield_report(
        store_obj,
        limit=limit,
        since=since,
        window_hours=window_hours,
    )


async def stats_optimization_eval_queue(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.optimization_eval_plan import build_optimization_eval_plan
    from tokenclaw.optimization_promotion_report import build_optimization_promotion_report

    capped_limit = max(1, min(int(limit or 500), 10_000))
    plan = await build_optimization_eval_plan(store_obj, limit=capped_limit, min_samples=1)
    promotion = build_optimization_promotion_report(store_obj, plan=plan, limit=capped_limit)
    verdicts = {
        str(row.get("candidate_id") or "unknown"): row
        for row in promotion.get("candidates", [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for plan_row in plan.get("plans") or []:
        if not isinstance(plan_row, dict):
            continue
        candidate_id = str(plan_row.get("candidate_id") or "unknown")
        verdict_row = verdicts.get(candidate_id, {"candidate_id": candidate_id, "verdict": "needs_eval"})
        rows.append(_optimization_eval_candidate_row(plan_row, verdict_row))

    rows.sort(
        key=lambda row: (
            str(row.get("action_family") or ""),
            str(row.get("optimization_family") or ""),
            str(row.get("candidate_id") or ""),
        )
    )

    verdict_counts: dict[str, int] = {}
    queue_status_counts: dict[str, int] = {}
    eval_status_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    for row in rows:
        _increment_count(verdict_counts, row.get("verdict"))
        _increment_count(queue_status_counts, row.get("queue_status"))
        _increment_count(eval_status_counts, row.get("eval_status"))
        _increment_count(action_counts, row.get("action_family"))
        for blocker in row.get("blocker_reason_codes") or []:
            _increment_count(blocker_counts, blocker)

    return {
        "schema": "tokenclaw.optimization_eval_queue.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            "candidate_count": len(rows),
            "needs_eval_count": queue_status_counts.get("needs_eval", 0),
            "hold_count": queue_status_counts.get("hold", 0),
            "widen_count": queue_status_counts.get("widen", 0),
            "rollback_count": queue_status_counts.get("rollback", 0),
            "privacy_blocked_count": queue_status_counts.get("privacy_blocked", 0),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in rows), 8),
            "last_evidence_at": max((str(row.get("last_evidence_at")) for row in rows if row.get("last_evidence_at")), default=None),
        },
        "verdict_counts": _count_breakdown(verdict_counts),
        "queue_status_counts": _count_breakdown(queue_status_counts),
        "eval_status_counts": _count_breakdown(eval_status_counts),
        "action_family_counts": _count_breakdown(action_counts),
        "blocker_counts": _count_breakdown(blocker_counts),
        "candidates": rows,
        "source_reports": {
            "eval_plan_schema": plan.get("schema"),
            "promotion_report_schema": promotion.get("schema"),
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "optimization eval plan metadata plus sanitized promotion verdicts only",
        },
    }


def _promotion_privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_transcripts_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "api_keys_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "basis": "optimization eval plan metadata, sanitized promotion verdicts, policy-event summaries, and stored canary decision metadata only",
    }


def _promotion_canary_meta_from_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("phase_canary", "promotion_canary", "optimization_promotion_canary"):
        value = decision.get(key)
        if isinstance(value, dict) and (value.get("target_candidate_id") or value.get("promotion_action_id") or value.get("action_id")):
            return value
    canary = decision.get("canary")
    if isinstance(canary, dict) and (canary.get("target_candidate_id") or canary.get("promotion_action_id") or canary.get("action_id")):
        return canary
    return None


def _promotion_new_observed_bucket(candidate_id: str, action_id: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "action_id": action_id,
        "policy_section": None,
        "policy_source": None,
        "canary_fraction": None,
        "holdout_fraction": None,
        "observed_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "bypassed_count": 0,
        "safety_stop_count": 0,
        "applied_error_count": 0,
        "holdout_error_count": 0,
        "applied_retry_count": 0,
        "holdout_retry_count": 0,
        "applied_latency_ms_total": 0,
        "holdout_latency_ms_total": 0,
        "applied_latency_count": 0,
        "holdout_latency_count": 0,
        "observed_savings_usd": 0.0,
        "last_observed_at": None,
        "reason_counts": {},
        "source_surface_counts": {},
    }


def _promotion_observed_canary_rows(store_obj: Any, limit: int) -> dict[str, dict[str, Any]]:
    rows = store_obj.conn.execute(
        """
        select created_at, source_surface, status_code, latency_ms, retry_count,
               cost_est_usd, cost_baseline_usd, routing_json, crunch_json, cache_json
        from calls
        where routing_json is not null or crunch_json is not null or cache_json is not null
        order by created_at desc
        limit ?
        """,
        (max(1, min(int(limit or 500), 10_000)),),
    ).fetchall()
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        decisions = (
            _json_obj(row.get("routing_json")),
            _json_obj(row.get("crunch_json")),
            _json_obj(row.get("cache_json")),
        )
        meta = next((item for item in (_promotion_canary_meta_from_decision(decision) for decision in decisions) if item), None)
        if not meta:
            continue
        candidate_id = str(meta.get("target_candidate_id") or meta.get("candidate_id") or "unknown")
        if candidate_id == "unknown":
            continue
        action_id = meta.get("promotion_action_id") or meta.get("action_id")
        bucket = buckets.setdefault(candidate_id, _promotion_new_observed_bucket(candidate_id, str(action_id) if action_id else None))
        if action_id and not bucket.get("action_id"):
            bucket["action_id"] = str(action_id)
        for key in ("policy_section", "policy_source", "canary_fraction", "holdout_fraction"):
            if bucket.get(key) is None and meta.get(key) is not None:
                bucket[key] = meta.get(key)
        source_surface = str(row.get("source_surface") or meta.get("source_surface") or "unknown")
        _increment_count(bucket["source_surface_counts"], source_surface)

        status = str(meta.get("status") or "")
        cohort = str(meta.get("cohort") or "")
        reason = str(meta.get("reason") or "")
        if reason:
            _increment_count(bucket["reason_counts"], reason)
        safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
        for code in safety.get("reason_codes") or []:
            _increment_count(bucket["reason_counts"], code)

        bucket["observed_count"] += 1
        if status == "applied" or cohort == "canary_applied":
            bucket["applied_count"] += 1
            if _as_int(row.get("status_code")) >= 400:
                bucket["applied_error_count"] += 1
            if _as_int(row.get("retry_count")) > 0:
                bucket["applied_retry_count"] += 1
            latency = _as_int(row.get("latency_ms"))
            if latency > 0:
                bucket["applied_latency_ms_total"] += latency
                bucket["applied_latency_count"] += 1
            savings = _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd"))
            if savings > 0:
                bucket["observed_savings_usd"] += savings
        elif status == "holdout" or cohort == "canary_holdout":
            bucket["holdout_count"] += 1
            if _as_int(row.get("status_code")) >= 400:
                bucket["holdout_error_count"] += 1
            if _as_int(row.get("retry_count")) > 0:
                bucket["holdout_retry_count"] += 1
            latency = _as_int(row.get("latency_ms"))
            if latency > 0:
                bucket["holdout_latency_ms_total"] += latency
                bucket["holdout_latency_count"] += 1
        elif status == "safety_stopped" or reason in {"local-canary-safety-stop", "safety-stop-tripped"} or safety.get("tripped"):
            bucket["safety_stop_count"] += 1
            bucket["bypassed_count"] += 1
        elif status in {"skipped", "not_selected"} or cohort == "skipped":
            bucket["skipped_count"] += 1
        else:
            bucket["bypassed_count"] += 1
        created_at = row.get("created_at")
        if created_at and (not bucket.get("last_observed_at") or str(created_at) > str(bucket.get("last_observed_at"))):
            bucket["last_observed_at"] = str(created_at)
    return buckets


def _promotion_lifecycle_rows(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if not details:
            continue
        text = " ".join(str(value or "") for value in (event.get("action"), details.get("command"), details.get("lifecycle_kind"), details.get("schema")))
        if "optimization" not in text and "promotion" not in text:
            continue
        candidate_ids = details.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate = details.get("target_candidate_id") or details.get("candidate_id")
            candidate_ids = [candidate] if candidate else []
        action_ids = details.get("action_ids")
        if not isinstance(action_ids, list):
            action = details.get("promotion_action_id") or details.get("action_id")
            action_ids = [action] if action else []
        for candidate in candidate_ids:
            candidate_id = str(candidate or "")
            if not candidate_id:
                continue
            bucket = buckets.setdefault(candidate_id, {
                "candidate_id": candidate_id,
                "latest_event_at": None,
                "event_count": 0,
                "action_ids": set(),
                "applied_count": 0,
                "holdout_count": 0,
                "skipped_count": 0,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "reason_counts": {},
            })
            bucket["event_count"] += 1
            if event.get("created_at") and (not bucket.get("latest_event_at") or str(event.get("created_at")) > str(bucket.get("latest_event_at"))):
                bucket["latest_event_at"] = str(event.get("created_at"))
            for action_id in action_ids:
                if action_id:
                    bucket["action_ids"].add(str(action_id))
            bucket["applied_count"] += _as_int(details.get("actual_canary_applied_count") or details.get("canary_applied_count") or details.get("applied_count"))
            bucket["holdout_count"] += _as_int(details.get("actual_canary_holdout_count") or details.get("canary_holdout_count") or details.get("holdout_count"))
            bucket["skipped_count"] += _as_int(details.get("skipped_count"))
            bucket["safety_stop_count"] += _as_int(details.get("safety_stop_count"))
            if str(event.get("action") or "").endswith("rollback") or str(details.get("event_type") or "") == "rollback":
                bucket["rollback_count"] += 1
            for key in ("reason", "status", "event_type", "local_result_status"):
                if details.get(key):
                    _increment_count(bucket["reason_counts"], details.get(key))
            reason_counts = details.get("reason_counts") or details.get("reason_code_counts")
            if isinstance(reason_counts, dict):
                for reason, count in reason_counts.items():
                    bucket["reason_counts"][str(reason)] = bucket["reason_counts"].get(str(reason), 0) + _as_int(count)
    result: dict[str, dict[str, Any]] = {}
    for candidate_id, bucket in buckets.items():
        public = dict(bucket)
        public["action_ids"] = sorted(public["action_ids"])
        public["reason_counts"] = _count_breakdown(public["reason_counts"])
        result[candidate_id] = public
    return result


def _promotion_finalize_observed(bucket: dict[str, Any] | None) -> dict[str, Any]:
    if not bucket:
        return {
            "observed_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "skipped_count": 0,
            "bypassed_count": 0,
            "safety_stop_count": 0,
            "observed_savings_usd": 0.0,
            "applied_error_rate": 0.0,
            "holdout_error_rate": 0.0,
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "latency_delta_ms": None,
            "last_observed_at": None,
            "reason_counts": [],
            "source_surface_counts": [],
        }
    applied = _as_int(bucket.get("applied_count"))
    holdout = _as_int(bucket.get("holdout_count"))
    applied_latency = None
    holdout_latency = None
    if _as_int(bucket.get("applied_latency_count")):
        applied_latency = _as_float(bucket.get("applied_latency_ms_total")) / _as_int(bucket.get("applied_latency_count"))
    if _as_int(bucket.get("holdout_latency_count")):
        holdout_latency = _as_float(bucket.get("holdout_latency_ms_total")) / _as_int(bucket.get("holdout_latency_count"))
    applied_error_rate = (_as_int(bucket.get("applied_error_count")) / applied) if applied else 0.0
    holdout_error_rate = (_as_int(bucket.get("holdout_error_count")) / holdout) if holdout else 0.0
    applied_retry_rate = (_as_int(bucket.get("applied_retry_count")) / applied) if applied else 0.0
    holdout_retry_rate = (_as_int(bucket.get("holdout_retry_count")) / holdout) if holdout else 0.0
    return {
        "action_id": bucket.get("action_id"),
        "policy_section": bucket.get("policy_section"),
        "policy_source": bucket.get("policy_source"),
        "canary_fraction": bucket.get("canary_fraction"),
        "holdout_fraction": bucket.get("holdout_fraction"),
        "observed_count": _as_int(bucket.get("observed_count")),
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": _as_int(bucket.get("skipped_count")),
        "bypassed_count": _as_int(bucket.get("bypassed_count")),
        "safety_stop_count": _as_int(bucket.get("safety_stop_count")),
        "observed_savings_usd": round(_as_float(bucket.get("observed_savings_usd")), 8),
        "applied_error_rate": round(applied_error_rate, 6),
        "holdout_error_rate": round(holdout_error_rate, 6),
        "error_rate_delta": round(applied_error_rate - holdout_error_rate, 6),
        "applied_retry_rate": round(applied_retry_rate, 6),
        "holdout_retry_rate": round(holdout_retry_rate, 6),
        "retry_rate_delta": round(applied_retry_rate - holdout_retry_rate, 6),
        "latency_delta_ms": round(applied_latency - holdout_latency, 2) if applied_latency is not None and holdout_latency is not None else None,
        "last_observed_at": bucket.get("last_observed_at"),
        "reason_counts": _count_breakdown(bucket.get("reason_counts") or {}),
        "source_surface_counts": _count_breakdown(bucket.get("source_surface_counts") or {}),
    }


def _promotion_primary_state(verdict: str, eval_evidence: dict[str, Any], observed: dict[str, Any], lifecycle: dict[str, Any] | None) -> str:
    if _as_int(observed.get("safety_stop_count")) or _as_int((lifecycle or {}).get("safety_stop_count")):
        return "safety-stopped"
    if verdict == "rollback" or _as_int((lifecycle or {}).get("rollback_count")):
        return "rollback-recommended"
    if verdict == "widen":
        return "widening-eligible"
    if _as_int(observed.get("applied_count")) or _as_int(observed.get("holdout_count")):
        return "canary-active"
    if _as_int(eval_evidence.get("pass_count")):
        return "eval-passed"
    return "needs-eval"


def _promotion_policy_section(row: dict[str, Any]) -> str:
    values = [
        str(row.get("action_family") or ""),
        str(row.get("optimization_family") or ""),
    ]
    normalized = [value.strip().lower().replace("_", "-") for value in values]
    joined = " ".join(normalized)
    if "old-context" in joined or "summarization" in joined or "summary" in joined:
        return "old_context_summarization"
    if "routing" in joined:
        return "routing"
    if "cache" in joined:
        return "cache"
    if "crunch" in joined or "pattern" in joined:
        return "crunch"
    return "unsupported"


def _promotion_target_local_policy_section(policy_section: str) -> str | None:
    if policy_section == "routing":
        return "routing.rules"
    if policy_section == "cache":
        return "cache.rules"
    if policy_section == "crunch":
        return "crunch.rules"
    if policy_section == "old_context_summarization":
        return "crunch.old_context_summarization"
    return None


def _promotion_next_command(status: str) -> tuple[str, str]:
    if status in {"pending-lifecycle-feedback", "impact-stale", "needs-more-samples"}:
        return "promotion-impact", "tokenclaw-optimization-promotion-impact promotion-actions.json --pretty"
    if status == "supported":
        return "promotion-canaries-apply --dry-run", "tokenclaw-optimization-promotion-canaries-apply promotion-actions.json --dry-run --pretty"
    return "promotion-actions", "tokenclaw-optimization-promotion-actions --pretty"


def _promotion_impact_stale(last_evidence_at: Any, *, max_age_hours: int = 168) -> bool:
    parsed = _parse_utc_datetime(last_evidence_at)
    if parsed is None:
        return False
    age = datetime.now(timezone.utc) - parsed
    return age.total_seconds() > max(1, max_age_hours) * 3600


def _promotion_reason_has_any(reasons: list[str], *needles: str) -> bool:
    haystack = " ".join(str(reason or "").lower() for reason in reasons)
    return any(needle in haystack for needle in needles)


def _promotion_pending_lifecycle_rows(store_obj: Any) -> dict[str, dict[str, Any]]:
    if not hasattr(store_obj, "managed_outcome_feedback_payload_rows"):
        return {}
    try:
        rows = store_obj.managed_outcome_feedback_payload_rows(
            source_surface=OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE,
            limit=1000,
        )
    except Exception:
        return {}

    pending_statuses = {"queued", "retryable-error", "claimed"}
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = str(row.get("status") or "")
        if status not in pending_statuses:
            continue
        payload = _json_obj(row.get("payload_json"))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        candidate_ids = metadata.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        action_ids = metadata.get("action_ids")
        if not isinstance(action_ids, list):
            action_ids = []
        for candidate in candidate_ids:
            candidate_id = str(candidate or "")
            if not candidate_id:
                continue
            bucket = buckets.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "pending_count": 0,
                    "statuses": {},
                    "commands": {},
                    "action_ids": set(),
                    "latest_queued_at": None,
                },
            )
            bucket["pending_count"] += 1
            _increment_count(bucket["statuses"], status)
            _increment_count(bucket["commands"], metadata.get("command") or payload.get("event_type") or "unknown")
            for action_id in action_ids:
                if action_id:
                    bucket["action_ids"].add(str(action_id))
            created_at = row.get("created_at") or row.get("updated_at") or payload.get("occurred_at")
            if created_at and (not bucket.get("latest_queued_at") or str(created_at) > str(bucket.get("latest_queued_at"))):
                bucket["latest_queued_at"] = str(created_at)
    result: dict[str, dict[str, Any]] = {}
    for candidate_id, bucket in buckets.items():
        public = dict(bucket)
        public["statuses"] = _count_breakdown(public["statuses"])
        public["commands"] = _count_breakdown(public["commands"])
        public["action_ids"] = sorted(public["action_ids"])
        result[candidate_id] = public
    return result


def _promotion_executor_readiness(
    *,
    verdict_row: dict[str, Any],
    primary_state: str,
    observed_row: dict[str, Any],
    lifecycle_row: dict[str, Any] | None,
    pending_lifecycle_row: dict[str, Any] | None,
    last_evidence_at: Any,
) -> dict[str, Any]:
    policy_section = _promotion_policy_section(verdict_row)
    supported = policy_section in {"routing", "crunch", "cache", "old_context_summarization"}
    reasons = _optimization_eval_reason_codes(verdict_row.get("reason_codes"))
    status = "supported"
    detail_reasons: list[str] = []
    if not supported:
        status = "unsupported"
        detail_reasons.append("unsupported-local-policy-section")
    elif policy_section == "cache" and _promotion_reason_has_any(reasons, "invalidation", "stale-risk"):
        status = "missing-invalidation-evidence"
        detail_reasons.append("cache-invalidation-evidence-required")
    elif primary_state in {"rollback-recommended", "safety-stopped"} or str(verdict_row.get("verdict") or "") == "rollback":
        status = "rollback-recommended"
    elif pending_lifecycle_row:
        status = "pending-lifecycle-feedback"
    elif primary_state == "canary-active" and _promotion_impact_stale(last_evidence_at):
        status = "impact-stale"
    elif primary_state == "widening-eligible" or str(verdict_row.get("verdict") or "") == "widen":
        status = "widening-eligible"
    elif primary_state in {"needs-eval", "canary-active"} and (
        _promotion_reason_has_any(reasons, "insufficient-")
        or _as_int((observed_row or {}).get("applied_count"))
        or _as_int((observed_row or {}).get("holdout_count"))
    ):
        status = "needs-more-samples"
    elif primary_state == "needs-eval":
        status = "missing-local-evidence"
    elif primary_state == "eval-passed":
        status = "supported"

    command_kind, command = _promotion_next_command(status)
    pending = pending_lifecycle_row or {}
    return {
        "status": status,
        "supported": supported,
        "policy_section": policy_section,
        "target_local_policy_section": _promotion_target_local_policy_section(policy_section),
        "next_command_kind": command_kind,
        "next_command": command,
        "reason_codes": sorted(set(detail_reasons + reasons)),
        "pending_lifecycle_feedback_count": _as_int(pending.get("pending_count")),
        "pending_lifecycle_feedback_statuses": pending.get("statuses") or [],
        "pending_lifecycle_feedback_commands": pending.get("commands") or [],
        "impact_stale": status == "impact-stale",
        "privacy": _promotion_privacy_summary(),
    }


def _promotion_action_dashboard_row(action: dict[str, Any], *, rank: int) -> dict[str, Any]:
    evidence = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
    cohorts = evidence.get("cohort_counts") if isinstance(evidence.get("cohort_counts"), dict) else {}
    local_review = action.get("local_review") if isinstance(action.get("local_review"), dict) else {}
    return {
        "rank": rank,
        "status": str(action.get("status") or "planned"),
        "action_type": str(action.get("action_type") or "unknown"),
        "verdict": str(action.get("verdict") or "unknown"),
        "action_family": str(action.get("action_family") or "unknown"),
        "optimization_family": str(action.get("optimization_family") or "unknown"),
        "source_surface": str(action.get("source_surface") or "unknown"),
        "app_family": str(action.get("app_family") or "unknown"),
        "policy_section": str(action.get("policy_section") or "unknown"),
        "target_local_policy_section": action.get("target_local_policy_section"),
        "projected_savings_usd": round(_as_float(evidence.get("projected_savings_usd")), 8),
        "sample_count": _as_int(evidence.get("sample_count")),
        "canary_applied_count": _as_int(cohorts.get("canary_applied")),
        "canary_holdout_count": _as_int(cohorts.get("canary_holdout")),
        "bypassed_or_disabled_count": _as_int(cohorts.get("bypassed_or_disabled")),
        "eval_result_count": _as_int(evidence.get("eval_result_count")),
        "eval_pass_count": _as_int(evidence.get("eval_pass_count")),
        "eval_fail_count": _as_int(evidence.get("eval_fail_count")),
        "eval_blocked_count": _as_int(evidence.get("eval_blocked_count")),
        "latest_eval_result_at": evidence.get("latest_eval_result_at"),
        "eval_evidence_stale": bool(evidence.get("eval_evidence_stale")),
        "current_canary_fraction": round(_as_float(action.get("current_canary_fraction")), 6),
        "canary_fraction": round(_as_float(action.get("canary_fraction")), 6),
        "holdout_fraction": round(_as_float(action.get("holdout_fraction")), 6),
        "review_command": str(local_review.get("review_command") or ""),
        "apply_preview_command": str(local_review.get("apply_preview_command") or ""),
        "privacy": _promotion_privacy_summary(),
    }


def _promotion_omission_dashboard_bucket(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": _as_int(row.get("rank")),
        "reason": str(row.get("reason") or "unknown"),
        "action_family": str(row.get("action_family") or "unknown"),
        "candidate_count": _as_int(row.get("candidate_count")),
        "projected_savings_usd": round(_as_float(row.get("projected_savings_usd")), 8),
        "next_action": str(row.get("next_action") or "unknown"),
        "reason_codes": _optimization_eval_reason_codes(row.get("reason_codes")),
        "privacy": _promotion_privacy_summary(),
    }


def _post_promotion_privacy_summary() -> dict[str, Any]:
    privacy = {
        **_promotion_privacy_summary(),
        "content_free": True,
        "local_only": True,
        "individual_candidate_ids_included": False,
        "individual_action_ids_included": False,
        "individual_rule_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "provider_bodies_included": False,
        "policy_file_contents_included": False,
    }
    return privacy


def _post_promotion_family(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("_", "-")
    if text in {"cache", "cache-replay"}:
        return "cache"
    if text in {"routing", "provider-routing", "phase-routing"}:
        return "routing"
    if text in {"crunch", "old-context-summary", "old-context-summarization"}:
        return "crunch"
    return public_label(text or "unknown", "unknown")


def _post_promotion_state(entry: dict[str, Any]) -> str:
    status = str(entry.get("status") or "").strip().lower().replace("_", "-")
    recommendation = str(entry.get("recommendation") or "").strip().lower().replace("_", "-")
    if entry.get("rollback_needed") or "rollback" in status or recommendation == "rollback":
        return "blocked"
    if "safety" in status or status in {"regression-flagged", "keep-blocked"}:
        return "blocked"
    if status in {"needs-more-samples", "needs-more-evidence", "needs-review"}:
        return "needs-evidence"
    if status in {"positive", "promoted", "widened"} or recommendation in {"promote", "widen"}:
        return "improving"
    if _as_float(entry.get("observed_savings_usd")) > 0:
        return "measured"
    return "observed"


def _post_promotion_next_action(state: str, latest: dict[str, Any], blocker: str | None) -> str:
    status = str(latest.get("status") or "").strip().lower().replace("_", "-")
    recommendation = str(latest.get("recommendation") or "").strip().lower().replace("_", "-")
    if state == "blocked":
        if "safety" in status or (blocker and "safety" in blocker):
            return "review-post-promotion-safety-blocker"
        if "rollback" in status or recommendation == "rollback":
            return "rollback-or-keep-promotion-blocked"
        return "review-post-promotion-regression"
    if state == "needs-evidence":
        return "collect-post-promotion-holdout-evidence"
    if recommendation in {"promote", "widen"}:
        return "widen-local-promotion"
    if state in {"improving", "measured"}:
        return "continue-measuring-post-promotion-impact"
    return "inspect-post-promotion-feedback"


def _post_promotion_status(states: list[str], latest: dict[str, Any] | None) -> str:
    if "blocked" in states:
        return "blocked"
    if "needs-evidence" in states:
        return "needs-evidence"
    if "improving" in states:
        return "improving"
    if "measured" in states:
        return "measured"
    return "observed" if latest else "no-feedback"


def _post_promotion_latest_blocker(entries: list[dict[str, Any]], safety_groups: list[dict[str, Any]]) -> str | None:
    for group in sorted(safety_groups, key=lambda row: str(row.get("rank") or ""), reverse=False):
        for key in ("keep_blocked_reason", "blocker_code", "safety_stop_reason"):
            value = str(group.get(key) or "").strip()
            if value:
                return public_label(value, "post-promotion-blocker")
    for entry in sorted(entries, key=lambda row: str(row.get("created_at") or row.get("impact_generated_at") or ""), reverse=True):
        codes = []
        for key in ("reason_codes", "warning_codes"):
            value = entry.get(key)
            if isinstance(value, list):
                codes.extend(_optimization_eval_reason_codes(value))
        if codes:
            return codes[0]
        if entry.get("rollback_needed"):
            return "rollback-needed"
        status = str(entry.get("status") or "").strip()
        if status and status not in {"positive", "observed"}:
            return public_label(status, "post-promotion-blocker")
    return None


def _post_promotion_delta_row(
    family: str,
    entries: list[dict[str, Any]],
    safety_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = max(entries, key=lambda row: str(row.get("created_at") or row.get("impact_generated_at") or ""), default=None)
    states = [_post_promotion_state(entry) for entry in entries]
    if safety_groups and "blocked" not in states:
        states.append("blocked")
    status = _post_promotion_status(states, latest)
    applied = sum(_as_int(entry.get("applied_count")) for entry in entries) + sum(_as_int(group.get("applied_count")) for group in safety_groups)
    holdout = sum(_as_int(entry.get("holdout_count")) for entry in entries) + sum(_as_int(group.get("holdout_count")) for group in safety_groups)
    safety_stops = sum(_as_int(entry.get("safety_stop_count")) for entry in entries) + sum(_as_int(group.get("safety_stop_count")) for group in safety_groups)
    observed = sum(_as_float(entry.get("observed_savings_usd")) for entry in entries)
    projected = sum(_as_float(entry.get("projected_savings_usd")) for entry in entries) + sum(_as_float(group.get("savings_estimate_usd")) for group in safety_groups)
    latest_blocker = _post_promotion_latest_blocker(entries, safety_groups)
    recommendation_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for entry in entries:
        recommendation = public_label(entry.get("recommendation"), "none")
        status_label = public_label(entry.get("status"), "unknown")
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        status_counts[status_label] = status_counts.get(status_label, 0) + 1
    return {
        "schema": "tokenclaw.post_promotion_blocker_delta.v1",
        "local_action_family": family,
        "status": status,
        "entry_count": len(entries),
        "safety_stop_group_count": len(safety_groups),
        "applied_count": applied,
        "holdout_count": holdout,
        "safety_stop_count": safety_stops,
        "latest_blocker_reason": latest_blocker,
        "observed_savings_usd": round(observed, 8),
        "projected_savings_usd": round(projected, 8),
        "savings_delta_usd": round(observed - projected, 8),
        "next_action": _post_promotion_next_action(status, latest or {}, latest_blocker),
        "latest_feedback_at": (latest or {}).get("created_at") or (latest or {}).get("impact_generated_at"),
        "status_counts": _breakdown_from_counts(status_counts),
        "recommendation_counts": _breakdown_from_counts(recommendation_counts),
        "privacy": _post_promotion_privacy_summary(),
    }


async def stats_post_promotion_deltas(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.activation_lifecycle_feedback import activation_safety_stop_burndown_report
    from tokenclaw.promotion_outcome_feedback import promotion_outcome_feedback_summary

    capped_limit = max(1, min(int(limit or 1000), 10_000))
    feedback = promotion_outcome_feedback_summary(store_obj, limit=capped_limit)
    safety = activation_safety_stop_burndown_report(store_obj, limit=capped_limit)
    entries_by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in feedback.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        family = _post_promotion_family(entry.get("action_family") or entry.get("policy_section"))
        entries_by_family.setdefault(family, []).append(entry)

    safety_by_family: dict[str, list[dict[str, Any]]] = {}
    for group in safety.get("groups") or []:
        if not isinstance(group, dict):
            continue
        family = _post_promotion_family(group.get("action_family"))
        safety_by_family.setdefault(family, []).append(group)

    families = sorted(set(entries_by_family) | set(safety_by_family))
    deltas = [
        _post_promotion_delta_row(
            family,
            entries_by_family.get(family, []),
            safety_by_family.get(family, []),
        )
        for family in families
    ]
    deltas.sort(
        key=lambda row: (
            row.get("status") != "blocked",
            -_as_int(row.get("safety_stop_count")),
            -abs(_as_float(row.get("savings_delta_usd"))),
            str(row.get("local_action_family") or ""),
        )
    )
    top = deltas[0] if deltas else {}
    privacy = _post_promotion_privacy_summary()
    return {
        "schema": "tokenclaw.post_promotion_blocker_deltas_dashboard.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "status": "available" if deltas else "no-feedback",
        "summary": {
            "family_count": len(deltas),
            "entry_count": sum(_as_int(row.get("entry_count")) for row in deltas),
            "blocked_family_count": sum(1 for row in deltas if row.get("status") == "blocked"),
            "needs_evidence_family_count": sum(1 for row in deltas if row.get("status") == "needs-evidence"),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in deltas),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in deltas),
            "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in deltas),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in deltas), 8),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in deltas), 8),
            "savings_delta_usd": round(sum(_as_float(row.get("savings_delta_usd")) for row in deltas), 8),
            "top_local_action_family": top.get("local_action_family"),
            "top_status": top.get("status"),
            "top_blocker_reason": top.get("latest_blocker_reason"),
            "top_next_action": top.get("next_action"),
        },
        "deltas": deltas,
        "source_reports": {
            "promotion_outcome_feedback_schema": feedback.get("schema"),
            "activation_safety_stop_burndown_schema": safety.get("schema"),
            "feedback_entry_count": feedback.get("entry_count"),
            "safety_stop_group_count": (safety.get("summary") or {}).get("ranked_group_count") if isinstance(safety.get("summary"), dict) else None,
        },
        "privacy": privacy,
    }


async def stats_optimization_promotion_actions(store_obj: Any, limit: int = 50) -> dict[str, Any]:
    from tokenclaw.optimization_eval_plan import build_optimization_eval_plan
    from tokenclaw.optimization_promotion_actions import build_optimization_promotion_actions
    from tokenclaw.optimization_promotion_report import build_optimization_promotion_report

    capped_limit = max(1, min(int(limit or 50), 100))
    plan = await build_optimization_eval_plan(store_obj, limit=capped_limit, min_samples=1)
    promotion = build_optimization_promotion_report(store_obj, plan=plan, limit=capped_limit)
    promotion_actions = build_optimization_promotion_actions(promotion)
    raw_actions = [
        row
        for row in promotion_actions.get("actions", [])
        if isinstance(row, dict)
    ][:capped_limit]
    actions = [
        _promotion_action_dashboard_row(row, rank=index)
        for index, row in enumerate(raw_actions, start=1)
    ]
    omitted = [
        row
        for row in promotion_actions.get("omission_buckets", [])
        if isinstance(row, dict)
    ][:capped_limit]
    omission_buckets = [_promotion_omission_dashboard_bucket(row) for row in omitted]
    summary = promotion_actions.get("summary") if isinstance(promotion_actions.get("summary"), dict) else {}
    return {
        "schema": "tokenclaw.optimization_promotion_actions_dashboard.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            "candidate_count": _as_int(summary.get("candidate_count")),
            "action_count": _as_int(summary.get("action_count")),
            "displayed_action_count": len(actions),
            "omitted_count": _as_int(summary.get("omitted_count")),
            "displayed_omission_bucket_count": len(omission_buckets),
            "policy_section_counts": summary.get("policy_section_counts") if isinstance(summary.get("policy_section_counts"), list) else [],
            "action_family_counts": summary.get("action_family_counts") if isinstance(summary.get("action_family_counts"), list) else [],
            "omission_reason_counts": summary.get("omission_reason_counts") if isinstance(summary.get("omission_reason_counts"), list) else [],
            "top_omission_next_action": summary.get("top_omission_next_action"),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in actions), 8),
            "canary_applied_count": sum(_as_int(row.get("canary_applied_count")) for row in actions),
            "canary_holdout_count": sum(_as_int(row.get("canary_holdout_count")) for row in actions),
            "latest_eval_result_at": max((str(row.get("latest_eval_result_at")) for row in actions if row.get("latest_eval_result_at")), default=None),
        },
        "actions": actions,
        "omission_buckets": omission_buckets,
        "source_reports": {
            "eval_plan_schema": plan.get("schema") if isinstance(plan, dict) else None,
            "promotion_report_schema": promotion.get("schema") if isinstance(promotion, dict) else None,
            "promotion_actions_schema": promotion_actions.get("schema") if isinstance(promotion_actions, dict) else None,
        },
        "privacy": {
            **_promotion_privacy_summary(),
            "individual_candidate_ids_included": False,
            "individual_action_ids_included": False,
        },
    }


async def stats_optimization_promotion_funnel(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.optimization_eval_plan import build_optimization_eval_plan
    from tokenclaw.optimization_promotion_actions import build_optimization_promotion_actions
    from tokenclaw.optimization_promotion_report import build_optimization_promotion_report
    from tokenclaw.policy_events import recent_policy_events

    capped_limit = max(1, min(int(limit or 500), 10_000))
    plan = await build_optimization_eval_plan(store_obj, limit=capped_limit, min_samples=1)
    promotion = build_optimization_promotion_report(store_obj, plan=plan, limit=capped_limit)
    promotion_actions = build_optimization_promotion_actions(promotion)
    observed = _promotion_observed_canary_rows(store_obj, capped_limit * 5)
    events = recent_policy_events(limit=500).get("events", [])
    lifecycle = _promotion_lifecycle_rows(events if isinstance(events, list) else [])
    pending_lifecycle = _promotion_pending_lifecycle_rows(store_obj)

    candidates: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    policy_section_counts: dict[str, int] = {}
    readiness_policy_counts: dict[str, int] = {}
    action_family_policy_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    all_candidate_ids = {
        str(row.get("candidate_id"))
        for row in promotion.get("candidates", [])
        if isinstance(row, dict) and row.get("candidate_id")
    } | set(observed) | set(lifecycle) | set(pending_lifecycle)
    verdicts = {
        str(row.get("candidate_id")): row
        for row in promotion.get("candidates", [])
        if isinstance(row, dict) and row.get("candidate_id")
    }
    for candidate_id in sorted(all_candidate_ids):
        verdict_row = verdicts.get(candidate_id, {"candidate_id": candidate_id, "verdict": "needs_eval", "eval_evidence": {}})
        observed_row = _promotion_finalize_observed(observed.get(candidate_id))
        lifecycle_row = lifecycle.get(candidate_id)
        pending_lifecycle_row = pending_lifecycle.get(candidate_id)
        verdict = str(verdict_row.get("verdict") or "needs_eval")
        eval_evidence = verdict_row.get("eval_evidence") if isinstance(verdict_row.get("eval_evidence"), dict) else {}
        primary_state = _promotion_primary_state(verdict, eval_evidence, observed_row, lifecycle_row)
        last_evidence_at = max(
            (
                str(value)
                for value in (
                    eval_evidence.get("latest_result_at"),
                    observed_row.get("last_observed_at"),
                    (lifecycle_row or {}).get("latest_event_at"),
                    (pending_lifecycle_row or {}).get("latest_queued_at"),
                )
                if value
            ),
            default=None,
        )
        executor_readiness = _promotion_executor_readiness(
            verdict_row=verdict_row,
            primary_state=primary_state,
            observed_row=observed_row,
            lifecycle_row=lifecycle_row,
            pending_lifecycle_row=pending_lifecycle_row,
            last_evidence_at=last_evidence_at,
        )
        _increment_count(state_counts, primary_state)
        _increment_count(readiness_counts, executor_readiness["status"])
        _increment_count(policy_section_counts, executor_readiness["policy_section"])
        _increment_count(readiness_policy_counts, f"{executor_readiness['policy_section']}:{executor_readiness['status']}")
        _increment_count(action_family_policy_counts, f"{verdict_row.get('action_family') or 'unknown'}:{executor_readiness['policy_section']}")
        row_reasons = _optimization_eval_reason_codes(verdict_row.get("reason_codes"))
        for reason in row_reasons:
            _increment_count(reason_counts, reason)
        for reason in (observed_row.get("reason_counts") or [])[:5]:
            _increment_count(reason_counts, reason.get("value"))
        candidates.append({
            "candidate_id": candidate_id,
            "action_id": observed_row.get("action_id") or ((lifecycle_row or {}).get("action_ids") or [None])[0],
            "action_family": str(verdict_row.get("action_family") or "unknown"),
            "optimization_family": str(verdict_row.get("optimization_family") or "unknown"),
            "source_surface": str(verdict_row.get("source_surface") or "unknown"),
            "app_family": str(verdict_row.get("app_family") or "unknown"),
            "candidate_target_model": verdict_row.get("candidate_target_model"),
            "candidate_profile": verdict_row.get("candidate_profile"),
            "projected_savings_usd": round(_as_float(verdict_row.get("projected_savings_usd")), 8),
            "observed_savings_usd": observed_row.get("observed_savings_usd", 0.0),
            "verdict": verdict,
            "primary_state": primary_state,
            "policy_section": executor_readiness["policy_section"],
            "target_local_policy_section": executor_readiness["target_local_policy_section"],
            "executor_readiness": executor_readiness,
            "next_command_kind": executor_readiness["next_command_kind"],
            "next_command": executor_readiness["next_command"],
            "eval_pass_count": _as_int(eval_evidence.get("pass_count")),
            "eval_fail_count": _as_int(eval_evidence.get("fail_count")),
            "eval_result_count": _as_int(eval_evidence.get("result_count")),
            "eval_evidence_stale": bool(eval_evidence.get("stale")),
            "canary": observed_row,
            "lifecycle": lifecycle_row or {},
            "pending_lifecycle_feedback": pending_lifecycle_row or {},
            "reason_codes": row_reasons,
            "top_reason_counts": observed_row.get("reason_counts") or [],
            "last_evidence_at": last_evidence_at,
            "privacy": _promotion_privacy_summary(),
        })

    candidates.sort(key=lambda row: (str(row.get("primary_state")), str(row.get("action_family")), str(row.get("candidate_id"))))
    return {
        "schema": "tokenclaw.optimization_promotion_funnel.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            "candidate_count": len(candidates),
            "needs_eval_count": state_counts.get("needs-eval", 0),
            "eval_passed_count": state_counts.get("eval-passed", 0),
            "canary_active_count": state_counts.get("canary-active", 0),
            "widening_eligible_count": state_counts.get("widening-eligible", 0),
            "rollback_recommended_count": state_counts.get("rollback-recommended", 0),
            "safety_stopped_count": state_counts.get("safety-stopped", 0),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in candidates), 8),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in candidates), 8),
            "canary_applied_count": sum(_as_int((row.get("canary") or {}).get("applied_count")) for row in candidates),
            "canary_holdout_count": sum(_as_int((row.get("canary") or {}).get("holdout_count")) for row in candidates),
            "pending_lifecycle_feedback_count": sum(_as_int((row.get("executor_readiness") or {}).get("pending_lifecycle_feedback_count")) for row in candidates),
            "promotion_action_count": _as_int((promotion_actions.get("summary") or {}).get("action_count")),
            "promotion_omitted_count": _as_int((promotion_actions.get("summary") or {}).get("omitted_count")),
            "promotion_omission_bucket_count": _as_int((promotion_actions.get("summary") or {}).get("omission_bucket_count")),
            "top_promotion_omission_next_action": (promotion_actions.get("summary") or {}).get("top_omission_next_action"),
            "last_evidence_at": max((str(row.get("last_evidence_at")) for row in candidates if row.get("last_evidence_at")), default=None),
        },
        "state_counts": _count_breakdown(state_counts),
        "executor_readiness_counts": _count_breakdown(readiness_counts),
        "policy_section_counts": _count_breakdown(policy_section_counts),
        "executor_readiness_by_policy_section": _count_breakdown(readiness_policy_counts),
        "action_family_policy_section_counts": _count_breakdown(action_family_policy_counts),
        "reason_counts": _count_breakdown(reason_counts),
        "omission_buckets": promotion_actions.get("omission_buckets") if isinstance(promotion_actions.get("omission_buckets"), list) else [],
        "candidates": candidates,
        "source_reports": {
            "eval_plan_schema": plan.get("schema") if isinstance(plan, dict) else None,
            "promotion_report_schema": promotion.get("schema") if isinstance(promotion, dict) else None,
            "promotion_actions_schema": promotion_actions.get("schema") if isinstance(promotion_actions, dict) else None,
            "policy_event_count": len(events) if isinstance(events, list) else 0,
        },
        "privacy": _promotion_privacy_summary(),
    }


def _increment_count(grouped: dict[str, int], key: Any) -> None:
    label = str(key or "unknown")
    grouped[label] = grouped.get(label, 0) + 1


def _count_breakdown(grouped: dict[str, int]) -> list[dict[str, Any]]:
    rows = [{"value": key, "count": count} for key, count in grouped.items()]
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows


def _decision_breakdown(rows: list[dict[str, Any]], decision_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        decision = row.get(f"{decision_key}_normalized")
        if not isinstance(decision, dict):
            decision = _json_obj(row.get(decision_key))
        status = str(decision.get("status") or "missing")
        reason = str(decision.get("reason") or "unknown")
        policy_source = str(decision.get("policy_source") or "unknown")
        key = (status, reason, policy_source)
        bucket = grouped.setdefault(
            key,
            {
                "status": status,
                "reason": reason,
                "policy_source": policy_source,
                "count": 0,
            },
        )
        bucket["count"] += 1
    result = list(grouped.values())
    result.sort(key=lambda r: r["count"], reverse=True)
    return result














































































def _avg_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    return round(sum(values) / len(values))


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return int(round((sorted_values[mid - 1] + sorted_values[mid]) / 2))


def _percentile_int(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, math.ceil((len(sorted_values) - 1) * percentile))
    return sorted_values[idx]


































def _openai_provider_prompt_cache_discount(store_obj: Any, *, limit: int) -> float:
    try:
        rows = store_obj.conn.execute(
            """
            select coalesce(routed_model, requested_model) as model,
                   sum(coalesce(cache_creation_input_tokens, 0)) as creation_tok,
                   sum(coalesce(cache_read_input_tokens, 0)) as read_tok
            from calls
            where coalesce(provider, 'anthropic') = 'openai'
              and (coalesce(cache_creation_input_tokens, 0) > 0 or coalesce(cache_read_input_tokens, 0) > 0)
            group by coalesce(routed_model, requested_model)
            limit ?
            """,
            (max(1, min(int(limit or 1000), 5000)),),
        ).fetchall()
    except Exception:
        return 0.0
    discount = 0.0
    for row in rows:
        accounting = provider_prompt_cache_accounting(
            str(row["model"] or ""),
            provider="openai",
            cache_creation_tokens=_as_int(row["creation_tok"]),
            cache_read_tokens=_as_int(row["read_tok"]),
        )
        discount += _as_float(accounting.get("read_discount_usd"))
    return round(discount, 8)








def _usage_bucket_identity(app_family: str, session_id: Any) -> dict[str, Any]:
    engineer = os.getenv("TOKENCLAW_ENGINEER") or None
    app = os.getenv("TOKENCLAW_APP") or app_family or "unknown"
    session = str(session_id or "")
    sid = session[:8] if session else None
    if engineer:
        bucket_id = f"engineer:{engineer}|app:{app}"
        label = f"{engineer} / {app}"
        bucket_kind = "engineer_app"
    elif session:
        bucket_id = f"app:{app}|session:{session}"
        label = f"{app} / session {sid}"
        bucket_kind = "app_session"
    else:
        bucket_id = f"app:{app}|session:unknown"
        label = f"{app} / unknown session"
        bucket_kind = "app_unknown_session"
    return {
        "bucket_id": bucket_id,
        "bucket_label": label,
        "bucket_kind": bucket_kind,
        "engineer": engineer,
        "app": app,
        "app_family": app_family or "unknown",
        "session_id": session or None,
        "sid": sid,
        "label_sources": {
            "engineer": "env:TOKENCLAW_ENGINEER" if engineer else None,
            "app": "env:TOKENCLAW_APP" if os.getenv("TOKENCLAW_APP") else "inferred_app_family",
            "session": "stored_session_id" if session else None,
        },
    }


def _new_usage_bucket(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        **identity,
        "provider_calls": 0,
        "codex_turns": 0,
        "turns": 0,
        "provider_input_tokens": 0,
        "provider_output_tokens": 0,
        "provider_total_tokens": 0,
        "codex_input_text_chars": 0,
        "codex_result_chars": 0,
        "codex_input_tokens_est": 0,
        "codex_output_tokens_est": 0,
        "codex_total_tokens_est": 0,
        "codex_cost_est_usd": 0.0,
        "codex_baseline_cost_est_usd": 0.0,
        "codex_hard_floor_usd": 0.0,
        "codex_exact_cache_savings_usd": 0.0,
        "codex_cost_estimated": False,
        "spend_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "baseline_cost_usd": 0.0,
        "routing_savings_usd": 0.0,
        "crunch_savings_usd": 0.0,
        "cache_savings_usd": 0.0,
        "token_basis": "unknown",
        "cost_basis": "unknown",
        "source_surfaces": [],
        "baseline_provider_cost_usd": 0.0,
        "captured_savings_usd": 0.0,
        "hard_floor_usd": None,
        "provider_cost_known": False,
        "codex_cost_known": False,
        "excludes_unknown_codex_app_cost": False,
        "codex_mutation_safe_turns": 0,
        "codex_telemetry_only_turns": 0,
        "optimized_calls": 0,
        "routed_calls": 0,
        "crunched_calls": 0,
        "local_cache_hits": 0,
        "prompt_cache_read_tokens": 0,
        "prompt_cache_creation_tokens": 0,
        "prompt_cache_read_savings_usd": 0.0,
        "prompt_cache_creation_cost_usd": 0.0,
        "thinking_tokens": 0,
        "thinking_cost_usd": 0.0,
        "errors": 0,
        "rate_limited": 0,
        "unrouted_high_cost_calls": 0,
        "large_tool_result_calls": 0,
        "context_plateau_pairs": 0,
        "_prev_text_chars_by_session": {},
        "_hint_codes": set(),
        "_token_bases": set(),
        "_cost_bases": set(),
        "_source_surface_counts": {},
        "remaining_saving_potential_hints": [],
    }


def _add_accounting_to_usage_bucket(bucket: dict[str, Any], unit: dict[str, Any]) -> None:
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        bucket[field] += _as_int(unit.get(field))
    for field in (
        "baseline_cost_usd",
        "routing_savings_usd",
        "crunch_savings_usd",
        "cache_savings_usd",
    ):
        bucket[field] += _as_float(unit.get(field))
    bucket["_token_bases"].add(str(unit.get("token_basis") or "unknown"))
    bucket["_cost_bases"].add(str(unit.get("cost_basis") or "unknown"))
    source_surface = str(unit.get("source_surface") or "unknown")
    surface_counts = bucket["_source_surface_counts"]
    surface_counts[source_surface] = surface_counts.get(source_surface, 0) + 1


def _add_usage_hint(bucket: dict[str, Any], code: str, label: str, detail: str) -> None:
    if code in bucket["_hint_codes"]:
        return
    bucket["_hint_codes"].add(code)
    bucket["remaining_saving_potential_hints"].append({
        "code": code,
        "label": label,
        "detail": detail,
    })


def _routing_candidate_target_from_meta(*items: Any) -> str | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in (
            "target_model_normalized",
            "target_model",
            "would_route_model",
            "route_to",
            "routed_model",
            "managed_route_candidate_model",
            "local_route_candidate_model",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _activity_routing_candidate_state(candidate: dict[str, Any], routing: dict[str, Any]) -> dict[str, Any]:
    state = dict(candidate or {})
    experiment = routing.get("routing_experiment") if isinstance(routing.get("routing_experiment"), dict) else {}
    shadow_collecting = (
        isinstance(experiment, dict)
        and experiment.get("mode") == "shadow_candidate_pass_through"
        and bool(experiment.get("sampled"))
    )
    if state.get("covered") and not shadow_collecting:
        state.setdefault("provenance", "local-rule")
        state.setdefault("cell_status", "covered")
        return state

    managed = routing.get("managed_recommendation") if isinstance(routing.get("managed_recommendation"), dict) else {}
    managed_target = _routing_candidate_target_from_meta(managed)
    if isinstance(managed, dict) and managed.get("enabled") and managed_target:
        state.update({
            "status": "proposed",
            "cell_status": "proposed",
            "covered": False,
            "actionable": False,
            "reason": "managed-recommendation-proposed",
            "provenance": "managed",
            "proposed_routed_model": managed_target,
            "suggested_routed_model": None,
            "add_payload": None,
        })
        return state

    if shadow_collecting:
        shadow_target = _routing_candidate_target_from_meta(experiment, routing)
        state.update({
            "status": "shadow-collecting",
            "cell_status": "shadow-collecting",
            "covered": False,
            "actionable": False,
            "reason": "shadow-candidate-pass-through",
            "provenance": "managed-shadow" if experiment.get("managed_route_candidate_model") else "local-shadow",
            "shadow_routed_model": shadow_target,
            "suggested_routed_model": None,
            "add_payload": None,
        })
        return state

    if state.get("covered"):
        state.setdefault("provenance", "local-rule")
        state.setdefault("cell_status", "covered")
        return state

    state.update({
        "status": "routing-off",
        "cell_status": "routing-off",
        "covered": False,
        "actionable": False,
        "reason": "no-backed-routing",
        "provenance": "none",
        "suggested_routed_model": None,
        "add_payload": None,
    })
    return state


def _activity_routing_cell(candidate: dict[str, Any]) -> dict[str, Any]:
    status = str(candidate.get("cell_status") or candidate.get("status") or "").strip()
    provenance = str(candidate.get("provenance") or candidate.get("policy_source") or "").strip()
    normalized_provenance = "managed" if provenance.startswith("managed") else "local"
    if bool(candidate.get("covered")):
        return {
            "schema": "tokenclaw.dashboard_routing_cell.v1",
            "state": "covered",
            "routed_model": (
                candidate.get("suggested_routed_model")
                or candidate.get("routed_model")
                or candidate.get("target_model")
            ),
            "provenance": normalized_provenance,
            "add_payload": None,
        }
    if status == "proposed":
        return {
            "schema": "tokenclaw.dashboard_routing_cell.v1",
            "state": "proposed",
            "routed_model": candidate.get("proposed_routed_model") or candidate.get("target_model"),
            "provenance": "managed",
            "add_payload": None,
        }
    if status == "shadow-collecting":
        return {
            "schema": "tokenclaw.dashboard_routing_cell.v1",
            "state": "shadow",
            "routed_model": candidate.get("shadow_routed_model") or candidate.get("target_model"),
            "provenance": normalized_provenance,
            "add_payload": None,
        }
    if bool(candidate.get("actionable")) and isinstance(candidate.get("add_payload"), dict):
        return {
            "schema": "tokenclaw.dashboard_routing_cell.v1",
            "state": "uncovered_actionable",
            "routed_model": (
                (candidate.get("add_payload") or {}).get("routed_model")
                or candidate.get("suggested_routed_model")
            ),
            "provenance": normalized_provenance,
            "add_payload": dict(candidate.get("add_payload") or {}),
        }
    return {
        "schema": "tokenclaw.dashboard_routing_cell.v1",
        "state": "none",
        "routed_model": None,
        "provenance": "none",
        "add_payload": None,
    }


def _provider_activity_unit(
    row: sqlite3.Row | dict[str, Any],
    *,
    provider_adoption_windows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    r = dict(row)
    routing = _json_obj(r.get("routing_json"))
    crunch = _json_obj(r.get("crunch_json"))
    cache = _json_obj(r.get("cache_json"))
    provider = str(r.get("provider") or "anthropic")
    requested_model = r.get("requested_model")
    routed_model = r.get("routed_model")
    target_model = routed_model or requested_model
    input_tokens = r.get("actual_input_tokens")
    if input_tokens is None:
        input_tokens = r.get("input_tokens_est")
    output_tokens = r.get("actual_output_tokens")
    if output_tokens is None:
        output_tokens = r.get("output_tokens_est")
    source_surface = str(r.get("source_surface") or _source_surface(provider, str(r.get("path") or "")))
    endpoint = str(r.get("endpoint") or str(r.get("path") or ""))
    category = r.get("category") or routing.get("category")
    workflow_phase = routing.get("workflow_phase")
    text_chars = routing.get("text_chars")
    quality_signals = derive_provider_quality_signals(
        source_surface=source_surface,
        status_code=r.get("status_code"),
        retry_count=r.get("retry_count") or 0,
        latency_ms=r.get("latency_ms"),
        error=r.get("error"),
        requested_model=requested_model,
        routed_model=routed_model,
        cache_hit=bool(r.get("cache_hit")),
        routing_meta=routing,
        crunch_meta=crunch,
        cache_meta=cache,
        provider_adoption_windows=provider_adoption_windows,
    )
    try:
        from tokenclaw.routing_experiments import routing_candidate_coverage

        routing_candidate = routing_candidate_coverage(
            requested_model=requested_model,
            provider=provider,
            source_surface=source_surface,
            category=category,
            workflow_phase=workflow_phase,
            stream=bool(r.get("stream")),
            text_chars=text_chars or 0,
            input_tokens=input_tokens or 0,
        )
    except Exception as exc:
        routing_candidate = {
            "schema": "tokenclaw.routing_candidate_coverage.v1",
            "status": "coverage-error",
            "covered": False,
            "actionable": False,
            "reason": exc.__class__.__name__,
            "eligible_candidate_count": 0,
            "eligible_candidate_ids": [],
            "add_payload": None,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }
    routing_candidate = _activity_routing_candidate_state(routing_candidate, routing)
    routing_cell = _activity_routing_cell(routing_candidate)
    return {
        "feature_schema_version": "tokenclaw.optimization_unit_features.v1",
        "unit_id": f"provider_call:{r.get('id')}",
        "created_at": r.get("created_at"),
        "routing_cell": routing_cell,
        "source_surface": source_surface,
        "granularity": "provider_request",
        "app_family": _app_family_for_call(provider, requested_model, str(r.get("path") or "")),
        "requested_model": requested_model,
        "candidate_target_model": target_model,
        "target_model": target_model,
        "routed_model": routed_model,
        "provider": provider,
        "endpoint": endpoint,
        "requested_model_family": r.get("requested_model_family"),
        "routed_model_family": r.get("routed_model_family"),
        "input_features": {
            "path": r.get("path"),
            "endpoint": endpoint,
            "source_surface": source_surface,
            "stream": bool(r.get("stream")),
            "category": category,
            "workflow_phase": workflow_phase,
            "text_chars": text_chars,
            "input_tokens": input_tokens,
            "input_tokens_est": r.get("input_tokens_est"),
            "actual_input_tokens": r.get("actual_input_tokens"),
            "cache_creation_input_tokens": r.get("cache_creation_input_tokens") or 0,
            "cache_read_input_tokens": r.get("cache_read_input_tokens") or 0,
        },
        "tool_features": {
            "has_tools": routing.get("has_tools"),
            "category": category,
            "thinking_history_stripped": routing.get("thinking_history_stripped"),
            "stripped_params": routing.get("stripped_params") or [],
        },
        "optimization_features": {
            "routing": routing,
            "routing_candidate": routing_candidate,
            "routing_cell": routing_cell,
            "crunch": crunch,
            "cache": cache,
            "policy_sources": sorted({
                str(source)
                for source in (
                    routing.get("policy_source"),
                    routing.get("final_policy_source"),
                    crunch.get("policy_source"),
                    cache.get("policy_source"),
                )
                if source
            }),
        },
        "outcome_features": {
            "status_code": r.get("status_code"),
            "latency_ms": r.get("latency_ms"),
            "cache_hit": bool(r.get("cache_hit")),
            "retry_count": r.get("retry_count") or 0,
            "output_tokens": output_tokens,
            "thinking_output_tokens": r.get("thinking_output_tokens") or 0,
            "cost_est_usd": r.get("cost_est_usd"),
            "cost_baseline_usd": r.get("cost_baseline_usd"),
            "error": r.get("error"),
            "quality_signals": quality_signals,
        },
        "quality_signals": quality_signals,
        "replayability_level": "raw_body_opt_in" if r.get("request_json") else "features_only",
        "privacy_summary": {
            "telemetry_profile": "metadata-only",
            "raw_body_storage": bool(r.get("request_json")),
            "metadata_only": not bool(r.get("request_json")),
            "aggregate_only": False,
        },
        "local_ids": {
            "calls_id": r.get("id"),
            "session_id": r.get("session_id"),
        },
    }




def _policy_sources_from(*decisions: dict[str, Any]) -> list[str]:
    return sorted({
        str(source)
        for decision in decisions
        for source in (
            decision.get("policy_source"),
            decision.get("final_policy_source"),
        )
        if source
    })


def _provider_accounting_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    routing = _json_obj(r.get("routing_json"))
    crunch = _json_obj(r.get("crunch_json"))
    cache = _json_obj(r.get("cache_json"))
    provider = str(r.get("provider") or "anthropic").lower()
    path = str(r.get("path") or "")
    requested_model = r.get("requested_model")
    routed_model = r.get("routed_model")
    target_model = routed_model or requested_model
    base_input_tokens = _as_int(
        r.get("actual_input_tokens")
        if r.get("actual_input_tokens") is not None
        else r.get("input_tokens_est")
    )
    output_tokens = _as_int(
        r.get("actual_output_tokens")
        if r.get("actual_output_tokens") is not None
        else r.get("output_tokens_est")
    )
    cache_creation_tokens = _as_int(r.get("cache_creation_input_tokens"))
    cache_read_tokens = _as_int(r.get("cache_read_input_tokens"))
    input_tokens = base_input_tokens + cache_creation_tokens + cache_read_tokens
    cost = _as_float(r.get("cost_est_usd"))
    baseline = _as_float(r.get("cost_baseline_usd")) or cost
    attribution = realized_savings_attribution(
        requested_model=requested_model,
        routed_model=routed_model,
        provider=provider,
        actual_input_tokens=r.get("actual_input_tokens"),
        input_tokens_est=r.get("input_tokens_est"),
        actual_output_tokens=r.get("actual_output_tokens"),
        output_tokens_est=r.get("output_tokens_est"),
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
        cost_est_usd=r.get("cost_est_usd"),
        cost_baseline_usd=r.get("cost_baseline_usd"),
        crunch_json=crunch,
        routing_json=routing,
        cache_json=cache,
        cache_hit=r.get("cache_hit"),
    )
    routing_savings = _as_float(routing.get("realized_routing_savings_usd"))
    if routing_savings <= 0:
        routing_savings = _as_float(attribution.get("realized_routing_savings_usd"))

    crunch_savings = _as_float(crunch.get("realized_crunch_savings_usd"))
    if crunch_savings <= 0 and isinstance(crunch.get("realized_savings"), dict):
        crunch_savings = _as_float(crunch["realized_savings"].get("realized_crunch_savings_usd"))
    if crunch_savings <= 0:
        crunch_savings = _as_float(attribution.get("realized_crunch_savings_usd"))

    cache_savings = _as_float(cache.get("realized_cache_savings_usd"))
    if cache_savings <= 0:
        cache_savings = _as_float(attribution.get("realized_cache_savings_usd"))

    token_basis = "provider-reported"
    if r.get("actual_input_tokens") is None and r.get("actual_output_tokens") is None:
        token_basis = "estimated-from-request"

    return {
        "source_surface": _source_surface(provider, path),
        "granularity": "provider_request",
        "app_family": _app_family_for_call(provider, requested_model, path),
        "session_id": r.get("session_id"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "token_basis": token_basis,
        "cost_est_usd": cost,
        "cost_basis": "provider-reported",
        "baseline_cost_usd": baseline,
        "routing_savings_usd": routing_savings,
        "crunch_savings_usd": crunch_savings,
        "cache_savings_usd": cache_savings,
        "provider_prompt_cache_discount_usd": _as_float(attribution.get("provider_prompt_cache_discount_usd")),
        "provider_prompt_cache_net_discount_usd": _as_float(attribution.get("provider_prompt_cache_net_discount_usd")),
        "hard_floor_usd": cost,
        "policy_sources": _policy_sources_from(routing, crunch, cache),
        "is_today": bool(r.get("is_today")),
    }




def _mixed_label(values: set[str], default: str = "unknown") -> str:
    clean = sorted(value for value in values if value)
    if not clean:
        return default
    if len(clean) == 1:
        return clean[0]
    return "mixed"


def _accounting_rollup(units: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "units": 0,
        "provider_calls": 0,
        "codex_turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "routing_savings_usd": 0.0,
        "crunch_savings_usd": 0.0,
        "cache_savings_usd": 0.0,
        "provider_prompt_cache_discount_usd": 0.0,
        "provider_prompt_cache_net_discount_usd": 0.0,
        "hard_floor_usd": 0.0,
        "_token_bases": set(),
        "_cost_bases": set(),
        "_policy_sources": set(),
    }
    by_surface: dict[str, dict[str, Any]] = {}
    savings_by_surface: dict[tuple[str, str], dict[str, Any]] = {}

    def add_common(bucket: dict[str, Any], unit: dict[str, Any]) -> None:
        bucket["units"] += 1
        if unit["granularity"] == "provider_request":
            bucket["provider_calls"] += 1
        if is_codex_turn_source_surface(unit["source_surface"]):
            bucket["codex_turns"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            bucket[field] += _as_int(unit.get(field))
        for field in (
            "cost_est_usd",
            "baseline_cost_usd",
            "routing_savings_usd",
            "crunch_savings_usd",
            "cache_savings_usd",
            "provider_prompt_cache_discount_usd",
            "provider_prompt_cache_net_discount_usd",
            "hard_floor_usd",
        ):
            bucket[field] += _as_float(unit.get(field))
        bucket["_token_bases"].add(str(unit.get("token_basis") or "unknown"))
        bucket["_cost_bases"].add(str(unit.get("cost_basis") or "unknown"))
        for source in unit.get("policy_sources") or []:
            bucket["_policy_sources"].add(str(source))

    for unit in units:
        add_common(total, unit)
        source_surface = str(unit.get("source_surface") or "unknown")
        bucket = by_surface.setdefault(
            source_surface,
            {
                "source_surface": source_surface,
                "granularities": set(),
                "app_families": set(),
                **{
                    key: 0 for key in (
                        "units",
                        "provider_calls",
                        "codex_turns",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                },
                "cost_est_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "routing_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "cache_savings_usd": 0.0,
                "provider_prompt_cache_discount_usd": 0.0,
                "provider_prompt_cache_net_discount_usd": 0.0,
                "hard_floor_usd": 0.0,
                "_token_bases": set(),
                "_cost_bases": set(),
                "_policy_sources": set(),
            },
        )
        bucket["granularities"].add(str(unit.get("granularity") or "unknown"))
        bucket["app_families"].add(str(unit.get("app_family") or "unknown"))
        add_common(bucket, unit)
        for optimization_type, field in (
            ("routing", "routing_savings_usd"),
            ("crunching", "crunch_savings_usd"),
            ("cache", "cache_savings_usd"),
            ("provider_prompt_cache", "provider_prompt_cache_discount_usd"),
        ):
            savings = _as_float(unit.get(field))
            if savings <= 0:
                continue
            key = (source_surface, optimization_type)
            row = savings_by_surface.setdefault(
                key,
                {
                    "source_surface": source_surface,
                    "optimization_type": optimization_type,
                    "savings_usd": 0.0,
                },
            )
            row["savings_usd"] += savings

    def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
        finalized = dict(bucket)
        finalized["token_basis"] = _mixed_label(finalized.pop("_token_bases"))
        finalized["cost_basis"] = _mixed_label(finalized.pop("_cost_bases"))
        finalized["policy_sources"] = sorted(finalized.pop("_policy_sources"))
        if isinstance(finalized.get("granularities"), set):
            finalized["granularities"] = sorted(finalized["granularities"])
        if isinstance(finalized.get("app_families"), set):
            finalized["app_families"] = sorted(finalized["app_families"])
        for field in (
            "cost_est_usd",
            "baseline_cost_usd",
            "routing_savings_usd",
            "crunch_savings_usd",
            "cache_savings_usd",
            "provider_prompt_cache_discount_usd",
            "provider_prompt_cache_net_discount_usd",
            "hard_floor_usd",
        ):
            finalized[field] = round(float(finalized[field]), 6)
        return finalized

    savings_rows = []
    for row in savings_by_surface.values():
        savings_rows.append({
            **row,
            "savings_usd": round(float(row["savings_usd"]), 6),
        })
    savings_rows.sort(key=lambda row: (row["source_surface"], row["optimization_type"]))

    source_rows = [finalize(bucket) for bucket in by_surface.values()]
    source_rows.sort(key=lambda row: row["source_surface"])
    return {
        **finalize(total),
        "source_surfaces": source_rows,
        "savings_by_source_surface": savings_rows,
    }



async def stats(store_obj: Any, default_db: str) -> dict[str, Any]:
    conn = store_obj.conn
    today_start = _utc_today_start_iso()
    activation_burndown = await stats_local_activation_next_action_queue(limit=5, store_obj=store_obj)
    closed_loop_activation = _closed_loop_activation_readiness(activation_burndown)
    managed_feed_state = _managed_feed_state()
    managed_feed_today = _managed_feed_decision_summary(conn, since=today_start)

    def scalar(sql: str, params: tuple[Any, ...] = ()) -> Any:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None

    calls = int(scalar("select count(*) from calls") or 0)
    cache_hits = int(scalar("select count(*) from calls where cache_hit = 1") or 0)
    today_summary = dict(conn.execute(
        """
        select count(*) as calls,
               coalesce(sum(coalesce(cost_est_usd, 0)), 0) as cost_est_usd,
               coalesce(sum(coalesce(cost_baseline_usd, cost_est_usd, 0)), 0) as cost_baseline_usd,
               coalesce(sum(case
                 when coalesce(cost_baseline_usd, 0) > coalesce(cost_est_usd, 0)
                 then coalesce(cost_baseline_usd, 0) - coalesce(cost_est_usd, 0)
                 else 0 end), 0) as savings_usd,
               coalesce(sum(case
                 when routed_model is not null and requested_model != routed_model
                 then 1 else 0 end), 0) as routed_count,
               coalesce(sum(case when cache_hit = 1 then 1 else 0 end), 0) as cache_hits,
               coalesce(sum(case when status_code >= 400 then 1 else 0 end), 0) as errors,
               avg(latency_ms) as avg_latency_ms,
               coalesce(sum(coalesce(actual_input_tokens, input_tokens_est, 0)), 0) as input_tokens,
               coalesce(sum(coalesce(actual_output_tokens, output_tokens_est, 0)), 0) as output_tokens,
               coalesce(sum(coalesce(actual_input_tokens, input_tokens_est, 0)
                            + coalesce(actual_output_tokens, output_tokens_est, 0)
                            + coalesce(cache_creation_input_tokens, 0)
                            + coalesce(cache_read_input_tokens, 0)), 0) as total_tokens,
               coalesce(sum(coalesce(json_extract(crunch_json, '$.saved_chars'), 0)), 0) as crunch_chars_saved,
               coalesce(sum(coalesce(json_extract(crunch_json, '$.tokens_saved_est'), 0)), 0) as crunch_tokens_saved,
               coalesce(sum(case when json_extract(crunch_json, '$.changed') = 1 then 1 else 0 end), 0) as crunched_count
        from calls
        where created_at >= ?
        """,
        (today_start,),
    ).fetchone())
    total_summary = dict(conn.execute(
        """
        select coalesce(sum(coalesce(cost_est_usd, 0)), 0) as cost_est_usd,
               coalesce(sum(coalesce(cost_baseline_usd, cost_est_usd, 0)), 0) as cost_baseline_usd,
               coalesce(sum(case
                 when coalesce(cost_baseline_usd, 0) > coalesce(cost_est_usd, 0)
                 then coalesce(cost_baseline_usd, 0) - coalesce(cost_est_usd, 0)
                 else 0 end), 0) as savings_usd,
               coalesce(sum(case when status_code >= 400 then 1 else 0 end), 0) as errors,
               avg(latency_ms) as avg_latency_ms
        from calls
        """
    ).fetchone())
    today_volume = [
        dict(row)
        for row in conn.execute(
            """
            select coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model_family, requested_model_family, routed_model, requested_model, 'unknown') as model_family,
                   count(*) as calls,
                   coalesce(sum(coalesce(cost_est_usd, 0)), 0) as cost_est_usd,
                   coalesce(sum(coalesce(cost_baseline_usd, cost_est_usd, 0)), 0) as cost_baseline_usd,
                   coalesce(sum(case
                     when coalesce(cost_baseline_usd, 0) > coalesce(cost_est_usd, 0)
                     then coalesce(cost_baseline_usd, 0) - coalesce(cost_est_usd, 0)
                     else 0 end), 0) as savings_usd,
                   coalesce(sum(coalesce(actual_input_tokens, input_tokens_est, 0)
                                + coalesce(actual_output_tokens, output_tokens_est, 0)
                                + coalesce(cache_creation_input_tokens, 0)
                                + coalesce(cache_read_input_tokens, 0)), 0) as total_tokens
            from calls
            where created_at >= ?
            group by coalesce(provider, 'anthropic'),
                     coalesce(routed_model_family, requested_model_family, routed_model, requested_model, 'unknown')
            order by calls desc
            limit 20
            """,
            (today_start,),
        ).fetchall()
    ]
    routed = conn.execute("""
        select coalesce(provider, 'anthropic') as provider,
               coalesce(source_surface, 'unknown') as source_surface,
               coalesce(endpoint, 'unknown') as endpoint,
               requested_model,
               routed_model,
               coalesce(category, json_extract(routing_json, '$.category'), 'unknown') as category,
               count(*) c,
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
               max(case when json_extract(routing_json, '$.openai_canary.cohort') is not null then created_at else null end) as openai_canary_latest_observed_at,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') = 'canary_applied' then 1 else 0 end) as anthropic_canary_applied_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') = 'canary_holdout' then 1 else 0 end) as anthropic_canary_holdout_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') = 'safety_stopped'
                         or json_extract(routing_json, '$.phase_canary.status') = 'safety_stopped'
                        then 1 else 0 end) as anthropic_canary_safety_stopped_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') = 'skipped' then 1 else 0 end) as anthropic_canary_skipped_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') = 'bypassed_or_disabled' then 1 else 0 end) as anthropic_canary_bypassed_or_disabled_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') is not null
                          and json_extract(routing_json, '$.phase_canary.cohort') not in (
                              'canary_applied',
                              'canary_holdout',
                              'safety_stopped',
                              'skipped',
                              'bypassed_or_disabled'
                          )
                        then 1 else 0 end) as anthropic_canary_unknown_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') is not null and status_code >= 400 then 1 else 0 end) as anthropic_canary_error_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') is not null and coalesce(retry_count, 0) > 0 then 1 else 0 end) as anthropic_canary_retry_count,
               sum(case when json_extract(routing_json, '$.phase_canary.cohort') is not null
                          and (
                              json_extract(routing_json, '$.phase_canary.fallback_reason') is not null
                              or json_extract(routing_json, '$.fallback_reason') is not null
                          )
                        then 1 else 0 end) as anthropic_canary_fallback_count,
               max(case when json_extract(routing_json, '$.phase_canary.cohort') is not null then created_at else null end) as anthropic_canary_latest_observed_at
        from calls
        group by coalesce(provider, 'anthropic'), coalesce(source_surface, 'unknown'),
                 coalesce(endpoint, 'unknown'), requested_model, routed_model,
                 coalesce(category, json_extract(routing_json, '$.category'), 'unknown')
        order by c desc
        limit 20
    """).fetchall()
    today_provider_rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model_family, requested_model_family, routed_model, requested_model, 'unknown') as model_family,
                   requested_model, routed_model, stream, cache_hit, status_code,
                   latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, crunch_json, routing_json, cache_json, error,
                   request_json, response_json, session_id, category,
                   cache_creation_input_tokens, cache_read_input_tokens, retry_count,
                   thinking_output_tokens,
                   1 as is_today
            from calls
            where created_at >= ?
            """,
            (today_start,),
        ).fetchall()
    ]
    today_accounting_units = [_provider_accounting_unit(row) for row in today_provider_rows]
    today_accounting = _accounting_rollup(today_accounting_units)
    today_tokenclaw_savings = (
        _as_float(today_accounting.get("routing_savings_usd"))
        + _as_float(today_accounting.get("crunch_savings_usd"))
        + _as_float(today_accounting.get("cache_savings_usd"))
    )
    today_provider_prompt_cache_discount = _as_float(today_accounting.get("provider_prompt_cache_discount_usd"))
    today_volume_savings: dict[tuple[str, str], float] = {}
    for row, unit in zip(today_provider_rows, today_accounting_units):
        key = (str(row.get("provider") or "anthropic"), str(row.get("model_family") or "unknown"))
        today_volume_savings[key] = today_volume_savings.get(key, 0.0) + (
            _as_float(unit.get("routing_savings_usd"))
            + _as_float(unit.get("crunch_savings_usd"))
            + _as_float(unit.get("cache_savings_usd"))
        )
    for row in today_volume:
        key = (str(row.get("provider") or "anthropic"), str(row.get("model_family") or "unknown"))
        row["savings_usd"] = round(today_volume_savings.get(key, 0.0), 8)

    recent = [
        dict(row)
        for row in conn.execute("""
        select coalesce(provider, 'anthropic') as provider,
               path,
               created_at,
               requested_model,
               routed_model,
               cache_hit,
               status_code,
               latency_ms,
               input_tokens_est,
               output_tokens_est,
               actual_input_tokens,
               actual_output_tokens,
               coalesce(actual_input_tokens, input_tokens_est, 0) as tokens_in,
               coalesce(actual_output_tokens, output_tokens_est, 0) as tokens_out,
               cost_est_usd,
               cost_baseline_usd,
               crunch_json,
               routing_json,
               cache_json,
               cache_creation_input_tokens,
               cache_read_input_tokens,
               coalesce(category, json_extract(routing_json, '$.category'), 'unknown') as category
        from calls
        order by created_at desc
        limit 50
    """).fetchall()
    ]
    for row in recent:
        unit = _provider_accounting_unit({**row, "is_today": True})
        row["saved_usd"] = round(
            _as_float(unit.get("routing_savings_usd"))
            + _as_float(unit.get("crunch_savings_usd"))
            + _as_float(unit.get("cache_savings_usd")),
            8,
        )
    today_calls = int(today_summary["calls"] or 0)
    today_cost = float(today_summary["cost_est_usd"] or 0.0)
    total_cost = float(total_summary["cost_est_usd"] or 0.0)
    today_savings = today_tokenclaw_savings
    today_routed = int(today_summary["routed_count"] or 0)
    today_cache_hits = int(today_summary["cache_hits"] or 0)
    today_crunch_tokens_saved = int(today_summary["crunch_tokens_saved"] or 0)
    today_crunch_savings = _as_float(today_accounting.get("crunch_savings_usd"))
    return {
        "schema": "tokenclaw.lightweight_dashboard_stats.v1",
        "generated_at": utc_now(),
        "calls": calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": (cache_hits / calls) if calls else 0,
        "today_calls": today_calls,
        "today_cost_usd": round(today_cost, 8),
        "today_savings_usd": round(today_savings, 8),
        "today_volume": today_volume,
        "today_routing_hit_rate": round(today_routed / today_calls, 4) if today_calls else 0.0,
        "today_cache_hit_rate": round(today_cache_hits / today_calls, 4) if today_calls else 0.0,
        "today_crunch_chars_saved": int(today_summary["crunch_chars_saved"] or 0),
        "today_crunch_tokens_saved": today_crunch_tokens_saved,
        "today_crunch_savings_usd": round(today_crunch_savings, 8),
        "summary": {
            "total_calls": calls,
            "today_calls": today_calls,
            "total_cost_usd": round(total_cost, 8),
            "today_cost_usd": round(today_cost, 8),
            "today_savings_usd": round(today_savings, 8),
            "today_routed_count": today_routed,
            "today_cache_hits": today_cache_hits,
            "today_crunched_count": int(today_summary["crunched_count"] or 0),
            "errors": int(total_summary["errors"] or 0),
            "today_errors": int(today_summary["errors"] or 0),
            "avg_latency_ms": round(float(total_summary["avg_latency_ms"] or 0), 2),
            "closed_loop_activation": closed_loop_activation.get("summary", {}),
        },
        "executive_summary": {
            "accounting_today": {
                "total_tokens": int(today_summary["total_tokens"] or 0),
                "cost_est_usd": round(today_cost, 8),
                "captured_savings_usd": round(today_savings, 8),
                "routing_savings_usd": round(_as_float(today_accounting.get("routing_savings_usd")), 8),
                "crunch_savings_usd": round(_as_float(today_accounting.get("crunch_savings_usd")), 8),
                "cache_savings_usd": round(_as_float(today_accounting.get("cache_savings_usd")), 8),
                "provider_prompt_cache_discount_usd": round(today_provider_prompt_cache_discount, 8),
                "source_surfaces": today_accounting.get("source_surfaces", []),
            },
            "accounting_total": {
                "cost_est_usd": round(total_cost, 8),
            },
            "tokens_today": {
                "provider_input_tokens": int(today_summary["input_tokens"] or 0),
                "provider_output_tokens": int(today_summary["output_tokens"] or 0),
                "provider_total_tokens": int(today_summary["total_tokens"] or 0),
                "total_tokens": int(today_summary["total_tokens"] or 0),
                "codex_app_turns": 0,
                "codex_app_total_tokens_est": 0,
                "codex_app_input_text_chars": 0,
            },
            "spend": {
                "today_calculated_spend_usd": round(today_cost, 8),
                "calculated_spend_usd": round(total_cost, 8),
                "today_provider_spend_usd": round(today_cost, 8),
                "today_codex_app_estimated_spend_usd": 0.0,
            },
            "savings": {
                "today_tokenclaw_generated_savings_usd": round(today_savings, 8),
                "today_tokenclaw_generated_buckets": {
                    "routing_usd": round(_as_float(today_accounting.get("routing_savings_usd")), 8),
                    "crunching_usd": round(_as_float(today_accounting.get("crunch_savings_usd")), 8),
                    "exact_local_cache_usd": round(_as_float(today_accounting.get("cache_savings_usd")), 8),
                },
                "today_provider_prompt_cache_discount_usd": round(today_provider_prompt_cache_discount, 8),
                "today_buckets": {
                    "routing_usd": round(_as_float(today_accounting.get("routing_savings_usd")), 8),
                    "crunching_usd": round(_as_float(today_accounting.get("crunch_savings_usd")), 8),
                    "exact_local_cache_usd": round(_as_float(today_accounting.get("cache_savings_usd")), 8),
                    "provider_prompt_cache_discount_usd": round(today_provider_prompt_cache_discount, 8),
                },
            },
            "health": {
                "errors": int(total_summary["errors"] or 0),
                "errors_today": int(today_summary["errors"] or 0),
                "avg_latency_ms": round(float(today_summary["avg_latency_ms"] or total_summary["avg_latency_ms"] or 0), 2),
            },
        },
        "db": default_db,
        "closed_loop_activation": closed_loop_activation,
        "managed_feed": {
            "schema": "tokenclaw.managed_feed_dashboard_summary.v1",
            "state": managed_feed_state,
            "today": {
                "schema": "tokenclaw.managed_feed_window_summary.v1",
                "window": "today",
                "window_start": today_start,
                "total_calls": managed_feed_today["total_calls"],
                "policy_decision_calls": managed_feed_today["policy_decision_calls"],
                "backing_counts": managed_feed_today["backing_counts"],
                "privacy": _metadata_only_privacy(),
            },
            "privacy": _metadata_only_privacy(),
        },
        "routing": [dict(r) for r in routed],
        "recent": [dict(r) for r in recent],
    }


async def stats_limiter(store_obj: Any, tier_status: Any, limiter_config: dict[str, Any]) -> dict[str, Any]:
    conn = store_obj.conn
    recent_rows = conn.execute("""
        select created_at,
               status_code,
               coalesce(routed_model, requested_model) as model,
               coalesce(provider, 'anthropic') as provider,
               retry_count,
               latency_ms,
               error
        from calls
        where status_code in (429, 529)
           or error like 'temporarily limiting requests%'
        order by created_at desc
        limit 50
    """).fetchall()
    recent = []
    last_upstream_by_tier: dict[str, Optional[str]] = {
        "haiku": None,
        "sonnet": None,
        "opus": None,
    }
    local_throttled_recent = 0
    upstream_limited_recent = 0
    for row in recent_rows:
        error = row["error"] or ""
        tier = model_tier(str(row["model"] or ""))
        local_throttled = error.startswith("temporarily limiting requests")
        if local_throttled:
            local_throttled_recent += 1
        else:
            upstream_limited_recent += 1
            if last_upstream_by_tier.get(tier) is None:
                last_upstream_by_tier[tier] = row["created_at"]
        recent.append({
            "created_at": row["created_at"],
            "tier": tier,
            "provider": row["provider"],
            "model": row["model"],
            "status_code": row["status_code"],
            "retry_count": row["retry_count"] or 0,
            "latency_ms": row["latency_ms"],
            "local_throttled": local_throttled,
            "error": error[:240] if error else None,
        })

    tiers = tier_status()
    for tier in tiers:
        tier["last_upstream_429_at"] = last_upstream_by_tier.get(tier["tier"])

    return {
        "generated_at": utc_now(),
        "config": {
            "min_request_interval_ms": limiter_config["min_request_interval_ms"],
            "max_tier_backoff_wait_s": limiter_config["max_tier_backoff_wait_s"],
            "max_concurrent_per_tier": limiter_config["max_concurrent_per_tier"],
        },
        "tiers": tiers,
        "recent_rate_limits": recent,
        "summary": {
            "active_cooldowns": sum(1 for tier in tiers if tier["active"]),
            "local_throttled_recent": local_throttled_recent,
            "upstream_limited_recent": upstream_limited_recent,
        },
    }


def _phase_memory_session_key(session_id: Any) -> str:
    value = str(session_id or "").strip()
    if not value:
        return "missing-session"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


_PHASE_MEMORY_DECISION_STATUSES = {"blocked", "ignored", "used", "evaluating", "unknown"}
_PHASE_MEMORY_DECISION_REASONS = {
    "blocked_phase_present",
    "condition-disabled",
    "db-missing",
    "dominant_phase_mismatch",
    "dominant_phase_window_too_short",
    "matched",
    "memory-missing",
    "missing-session-id",
    "model_family_floor_not_met",
    "non-sqlite-db",
    "phase_not_stable",
    "recent_errors",
    "recent_retries",
    "recent_routing_fallback",
    "stable_window_too_small",
    "thinking_present",
    "unknown",
}


def _phase_memory_decision_label(value: Any, allowed: set[str], *, default: str = "unknown") -> str:
    label = str(value or "").strip().lower()
    if not label:
        return default
    if label.startswith("db-error:"):
        return "db-error"
    return label if label in allowed else default


def _session_phase_memory_decision_usage(store_obj: Any, *, limit: int = 5000) -> dict[str, Any]:
    rows = store_obj.conn.execute(
        """
        SELECT created_at, session_id, routing_json
        FROM calls
        WHERE routing_json LIKE '%session_phase_memory%'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (max(1, min(int(limit or 5000), 10000)),),
    ).fetchall()
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    per_session: dict[str, dict[str, Any]] = {}
    for row in rows:
        routing = _json_obj(row["routing_json"])
        memory = routing.get("session_phase_memory")
        if not isinstance(memory, dict):
            continue
        status = _phase_memory_decision_label(memory.get("status"), _PHASE_MEMORY_DECISION_STATUSES)
        reason = _phase_memory_decision_label(memory.get("reason"), _PHASE_MEMORY_DECISION_REASONS)
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        session_key = _phase_memory_session_key(row["session_id"])
        bucket = per_session.setdefault(
            session_key,
            {
                "decision_count": 0,
                "status_counts": {},
                "reason_counts": {},
                "last_decision_at": None,
                "last_status": None,
                "last_reason": None,
            },
        )
        bucket["decision_count"] += 1
        bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
        bucket["reason_counts"][reason] = bucket["reason_counts"].get(reason, 0) + 1
        if bucket["last_decision_at"] is None:
            bucket["last_decision_at"] = row["created_at"]
            bucket["last_status"] = status
            bucket["last_reason"] = reason

    for bucket in per_session.values():
        bucket["status_counts"] = _breakdown_from_counts(bucket["status_counts"])
        bucket["reason_counts"] = _breakdown_from_counts(bucket["reason_counts"])

    return {
        "decision_count": sum(status_counts.values()),
        "status_counts": _breakdown_from_counts(status_counts),
        "reason_counts": _breakdown_from_counts(reason_counts),
        "per_session": per_session,
    }


async def stats_session_phase_memory(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    memory = build_session_phase_memory(store_obj, limit=limit)
    decision_usage = _session_phase_memory_decision_usage(store_obj)
    per_session_decisions = decision_usage.pop("per_session")
    sessions = []
    for session in memory.get("sessions", []):
        row = dict(session)
        usage = per_session_decisions.get(row.get("session_key")) or {
            "decision_count": 0,
            "status_counts": [],
            "reason_counts": [],
            "last_decision_at": None,
            "last_status": None,
            "last_reason": None,
        }
        row["decision_usage"] = usage
        sessions.append(row)

    summary = dict(memory.get("summary") or {})
    session_count = int(summary.get("session_count") or 0)
    ready_count = int(summary.get("memory_ready_session_count") or 0)
    blocked_count = int(summary.get("blocked_session_count") or 0)
    summary.update({
        "memory_ready_session_count": ready_count,
        "blocked_session_count": blocked_count,
        "memory_ready_rate": round(ready_count / session_count, 4) if session_count else 0.0,
        "readiness_counts": _breakdown_from_counts({
            "ready": ready_count,
            "blocked": blocked_count,
        }),
        "top_blocker_reasons": summary.get("blocker_counts") or [],
        "decision_usage": decision_usage,
    })

    return {
        "schema": "tokenclaw.session_phase_memory_dashboard.v1",
        "memory_schema": memory.get("schema"),
        "generated_at": memory.get("generated_at") or utc_now(),
        "lookback": memory.get("lookback") or {},
        "summary": summary,
        "sessions": sessions,
        "privacy": {
            **(memory.get("privacy") or {}),
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "session_ids_hashed": True,
            "cache_keys_included": False,
            "provider_calls_made": False,
        },
    }


async def stats_activity(store_obj: Any, limit: int = 100) -> dict[str, Any]:
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 100), 500))

    provider_rows = conn.execute("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, stream, cache_hit, status_code,
               latency_ms, input_tokens_est, output_tokens_est,
               actual_input_tokens, actual_output_tokens, cost_est_usd,
               cost_baseline_usd, crunch_json, routing_json, cache_json, error,
               request_json, response_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens, retry_count,
               thinking_output_tokens
        from calls
        order by created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    adoption_by_call: dict[str, list[dict[str, Any]]] = {}
    if hasattr(store_obj, "provider_tool_adoption_windows_for_call_ids"):
        adoption_by_call = store_obj.provider_tool_adoption_windows_for_call_ids(
            [str(dict(row).get("id") or "") for row in provider_rows]
        )
    provider_units = [
        _provider_activity_unit(
            row,
            provider_adoption_windows=adoption_by_call.get(str(dict(row).get("id") or "")),
        )
        for row in provider_rows
    ]

    codex_rows = conn.execute("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.message_chars,
               s.params_chars,
               s.input_items,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
               s.cache_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms
        from codex_app_events s
        where s.direction = 'client_to_server' and s.method = 'turn/start'
        order by s.created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    codex_units = [_codex_turn_activity_unit(row) for row in codex_rows]

    units = sorted(
        provider_units + codex_units,
        key=lambda unit: str(unit.get("created_at") or ""),
        reverse=True,
    )[:capped_limit]

    def counts_by(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in units:
            value = str(unit.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    return {
        "generated_at": utc_now(),
        "schema": "tokenclaw.optimization_activity.v1",
        "summary": {
            "units": len(units),
            "provider_request_units": sum(1 for unit in units if unit["granularity"] == "provider_request"),
            "codex_turn_units": sum(1 for unit in units if is_codex_turn_source_surface(unit["source_surface"])),
            "codex_app_turn_units": sum(1 for unit in units if is_codex_turn_source_surface(unit["source_surface"])),
            "by_source_surface": counts_by("source_surface"),
            "by_granularity": counts_by("granularity"),
            "by_app_family": counts_by("app_family"),
            "by_replayability_level": counts_by("replayability_level"),
            "quality_signal_summary": summarize_quality_signals(units),
        },
        "units": units,
    }


async def stats_quality_signals(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    activity = await stats_activity(store_obj, limit=limit)
    return {
        "generated_at": utc_now(),
        "schema": "tokenclaw.quality_signal_report.v1",
        "privacy": {
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "basis": "derived local metadata only",
        },
        "summary": activity["summary"]["quality_signal_summary"],
        "recent": [
            {
                "unit_id": unit.get("unit_id"),
                "created_at": unit.get("created_at"),
                "source_surface": unit.get("source_surface"),
                "granularity": unit.get("granularity"),
                "app_family": unit.get("app_family"),
                "quality_signals": unit.get("quality_signals"),
            }
            for unit in activity["units"]
        ],
    }


PROVIDER_ADOPTION_RISK_STATUSES = {"abandoned", "orphan_result", "unknown"}


def _provider_adoption_label(value: Any, *, fallback: str = "unknown") -> str:
    return public_label(value, fallback=fallback)


def _provider_adoption_model_family(row: dict[str, Any]) -> str:
    family = row.get("routed_model_family") or row.get("requested_model_family")
    if family:
        return _provider_adoption_label(family)
    model = row.get("routed_model") or row.get("requested_model")
    return _provider_adoption_label(model_tier(str(model or "")), fallback="unknown")


def _provider_adoption_normalized_cohort(value: Any) -> str:
    label = _provider_adoption_label(value, fallback="")
    if label in {"applied", "canary_applied", "active"}:
        return "applied"
    if label in {"holdout", "canary_holdout"}:
        return "holdout"
    if label in {"safety_stopped", "safety-stop", "safety_stopped"}:
        return "safety_stopped"
    return label or "unknown"


def _provider_adoption_policy_id(value: Any) -> str | None:
    return public_id(value, prefix="policy")


def _provider_adoption_cohort_entries(row: dict[str, Any]) -> list[dict[str, str]]:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    entries: list[dict[str, str]] = []

    def add(family: str, meta: Any, *, nested_key: str | None = None) -> None:
        candidate = meta.get(nested_key) if isinstance(meta, dict) and nested_key else meta
        if not isinstance(candidate, dict):
            return
        cohort = _provider_adoption_normalized_cohort(
            candidate.get("cohort") or candidate.get("canary_cohort") or candidate.get("status")
        )
        if cohort == "unknown" and candidate.get("applied") is not True:
            return
        if cohort == "unknown" and candidate.get("applied") is True:
            cohort = "applied"
        item: dict[str, str] = {
            "optimization_family": _provider_adoption_label(family),
            "cohort": cohort,
            "policy_source": _provider_adoption_label(candidate.get("policy_source") or row.get("policy_source")),
        }
        policy_id = _provider_adoption_policy_id(
            candidate.get("policy_id") or candidate.get("candidate_id") or candidate.get("rule_id")
        )
        if policy_id:
            item["policy_id"] = policy_id
        if item not in entries:
            entries.append(item)

    add("routing_experiment", routing, nested_key="routing_experiment")
    add("phase_routing", routing, nested_key="phase_canary")
    add("openai_routing", routing, nested_key="openai_canary")
    managed = routing.get("managed_recommendation")
    if isinstance(managed, dict):
        add("managed_recommendation", managed)
        add("managed_recommendation", managed, nested_key="canary")
    add("old_context_summarization", crunch, nested_key="old_context_summarization")
    add("cache_replay", cache, nested_key="cache_replay_canary")
    if cache.get("status") in {"holdout", "applied", "canary_holdout", "canary_applied"}:
        add("cache", cache)

    if entries:
        return entries[:20]

    optimized = bool(
        row.get("cache_hit")
        or cache.get("status") == "hit"
        or crunch.get("changed")
        or routing.get("applied")
        or (
            row.get("requested_model")
            and row.get("routed_model")
            and row.get("requested_model") != row.get("routed_model")
        )
    )
    return [{
        "optimization_family": "overall",
        "cohort": "optimized" if optimized else "baseline",
        "policy_source": _provider_adoption_label(row.get("policy_source")),
    }]


def _new_provider_adoption_bucket(key: dict[str, str]) -> dict[str, Any]:
    return {
        **key,
        "window_count": 0,
        "fulfilled_count": 0,
        "pending_count": 0,
        "abandoned_count": 0,
        "orphan_result_count": 0,
        "unknown_count": 0,
        "risk_window_count": 0,
        "tool_use_count": 0,
        "tool_result_count": 0,
        "_blocker_counts": {},
    }


def _add_provider_adoption_row(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    status = _provider_adoption_label(row.get("status"))
    reason = _provider_adoption_label(row.get("reason"))
    bucket["window_count"] += 1
    bucket["tool_use_count"] += _as_int(row.get("tool_use_count"))
    bucket["tool_result_count"] += _as_int(row.get("tool_result_count"))
    if status == "fulfilled":
        bucket["fulfilled_count"] += 1
    elif status == "pending":
        bucket["pending_count"] += 1
    elif status == "abandoned":
        bucket["abandoned_count"] += 1
    elif status == "orphan_result":
        bucket["orphan_result_count"] += 1
    else:
        bucket["unknown_count"] += 1
    if status in PROVIDER_ADOPTION_RISK_STATUSES:
        bucket["risk_window_count"] += 1
    if status != "fulfilled":
        counts = bucket["_blocker_counts"]
        counts[reason] = counts.get(reason, 0) + 1


def _finalize_provider_adoption_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    total = max(_as_int(bucket.get("window_count")), 1)
    blockers = _breakdown_from_counts(bucket.pop("_blocker_counts", {}))
    bucket["adoption_rate"] = bucket["fulfilled_count"] / total
    bucket["risk_rate"] = bucket["risk_window_count"] / total
    bucket["pending_rate"] = bucket["pending_count"] / total
    bucket["top_blockers"] = blockers[:5]
    bucket["health"] = (
        "healthy"
        if bucket["window_count"] and bucket["risk_window_count"] == 0
        else "watch"
        if bucket["window_count"]
        else "no-data"
    )
    return bucket


async def stats_provider_adoption_health(store_obj: Any, limit: int = 5000) -> dict[str, Any]:
    if hasattr(store_obj, "abandon_stale_provider_tool_adoption_windows"):
        store_obj.abandon_stale_provider_tool_adoption_windows(now=utc_now())
    if hasattr(store_obj, "provider_tool_adoption_health_rows"):
        rows = store_obj.provider_tool_adoption_health_rows(limit=limit)
    else:
        rows = store_obj.provider_tool_adoption_window_rows(limit=limit)

    summary = _new_provider_adoption_bucket({
        "scope": "all",
        "optimization_family": "all",
        "cohort": "all",
        "policy_source": "all",
    })
    cohort_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    dimension_buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}

    for raw in rows:
        row = dict(raw)
        status = _provider_adoption_label(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        _add_provider_adoption_row(summary, row)

        base = {
            "source_surface": _provider_adoption_label(row.get("source_surface")),
            "app_family": _provider_adoption_label(row.get("app_family")),
            "model_family": _provider_adoption_model_family(row),
            "category": _provider_adoption_label(row.get("category")),
            "workflow_phase": _provider_adoption_label(row.get("workflow_phase")),
        }
        for cohort in _provider_adoption_cohort_entries(row):
            cohort_key = (
                cohort["optimization_family"],
                cohort["cohort"],
                cohort["policy_source"],
            )
            cohort_bucket = cohort_buckets.setdefault(
                cohort_key,
                _new_provider_adoption_bucket({
                    "optimization_family": cohort["optimization_family"],
                    "cohort": cohort["cohort"],
                    "policy_source": cohort["policy_source"],
                }),
            )
            _add_provider_adoption_row(cohort_bucket, row)

            dim_key = (
                base["source_surface"],
                base["app_family"],
                base["model_family"],
                base["category"],
                base["workflow_phase"],
                cohort["optimization_family"],
                cohort["cohort"],
                cohort["policy_source"],
            )
            dim_bucket = dimension_buckets.setdefault(
                dim_key,
                _new_provider_adoption_bucket({
                    **base,
                    "optimization_family": cohort["optimization_family"],
                    "cohort": cohort["cohort"],
                    "policy_source": cohort["policy_source"],
                }),
            )
            _add_provider_adoption_row(dim_bucket, row)

    cohort_rows = [_finalize_provider_adoption_bucket(bucket) for bucket in cohort_buckets.values()]
    cohort_rows.sort(key=lambda item: (-item["window_count"], item["optimization_family"], item["cohort"]))
    dimension_rows = [_finalize_provider_adoption_bucket(bucket) for bucket in dimension_buckets.values()]
    dimension_rows.sort(key=lambda item: (-item["window_count"], item["source_surface"], item["optimization_family"], item["cohort"]))

    by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for row in cohort_rows:
        by_family.setdefault(row["optimization_family"], {})[row["cohort"]] = row
    comparisons: list[dict[str, Any]] = []
    for family, cohorts in sorted(by_family.items()):
        applied = cohorts.get("applied")
        holdout = cohorts.get("holdout")
        if not applied and not holdout:
            continue
        applied_rate = applied.get("adoption_rate") if applied else None
        holdout_rate = holdout.get("adoption_rate") if holdout else None
        comparisons.append({
            "optimization_family": family,
            "applied_windows": _as_int(applied.get("window_count")) if applied else 0,
            "holdout_windows": _as_int(holdout.get("window_count")) if holdout else 0,
            "applied_adoption_rate": applied_rate,
            "holdout_adoption_rate": holdout_rate,
            "applied_minus_holdout_adoption_rate": (
                applied_rate - holdout_rate
                if applied_rate is not None and holdout_rate is not None
                else None
            ),
            "applied_risk_rate": applied.get("risk_rate") if applied else None,
            "holdout_risk_rate": holdout.get("risk_rate") if holdout else None,
            "health": (
                "healthy"
                if applied and holdout and applied.get("risk_rate", 0) <= holdout.get("risk_rate", 0)
                else "watch"
            ),
        })

    finalized_summary = _finalize_provider_adoption_bucket(summary)
    return {
        "generated_at": utc_now(),
        "schema": "tokenclaw.provider_adoption_dashboard_health.v1",
        "window_schema": "tokenclaw.provider_tool_adoption_window.v1",
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "summary": finalized_summary,
        "cohort_health": cohort_rows[:50],
        "cohort_comparisons": comparisons[:50],
        "dimension_breakdown": dimension_rows[:100],
        "blocker_reason_breakdown": finalized_summary.get("top_blockers", []),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "tool_ids_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "provider_bodies_included": False,
            "correlation_digests_included": False,
        },
    }


async def stats_usage_by_owner(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
    buckets: dict[str, dict[str, Any]] = {}

    def bucket_for(app_family: str, session_id: Any) -> dict[str, Any]:
        identity = _usage_bucket_identity(app_family, session_id)
        bucket = buckets.get(identity["bucket_id"])
        if bucket is None:
            bucket = _new_usage_bucket(identity)
            buckets[identity["bucket_id"]] = bucket
        return bucket

    provider_rows = conn.execute("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, status_code, input_tokens_est,
               output_tokens_est, actual_input_tokens, actual_output_tokens,
               cost_est_usd, cost_baseline_usd, cache_hit, crunch_json,
               routing_json, cache_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens,
               thinking_output_tokens
        from calls
        where date(created_at) = date('now')
        order by coalesce(session_id, ''), created_at
    """).fetchall()

    min_plateau_chars = 8_000
    max_plateau_delta_ratio = 0.03
    high_cost_unrouted_usd = 0.01

    for row in provider_rows:
        r = dict(row)
        provider = str(r.get("provider") or "anthropic").lower()
        requested_model = r.get("requested_model")
        routed_model = r.get("routed_model")
        target_model = routed_model or requested_model
        app_family = _app_family_for_call(provider, requested_model, str(r.get("path") or ""))
        bucket = bucket_for(app_family, r.get("session_id"))
        routing = _json_obj(r.get("routing_json"))
        crunch = _json_obj(r.get("crunch_json"))
        cache = _json_obj(r.get("cache_json"))

        input_tokens = _as_int(r.get("actual_input_tokens") if r.get("actual_input_tokens") is not None else r.get("input_tokens_est"))
        output_tokens = _as_int(r.get("actual_output_tokens") if r.get("actual_output_tokens") is not None else r.get("output_tokens_est"))
        cache_creation_tokens = _as_int(r.get("cache_creation_input_tokens"))
        cache_read_tokens = _as_int(r.get("cache_read_input_tokens"))
        provider_input_tokens = input_tokens + cache_creation_tokens + cache_read_tokens
        cost = _as_float(r.get("cost_est_usd"))
        baseline = _as_float(r.get("cost_baseline_usd")) or cost
        status_code = _as_int(r.get("status_code"))
        category = r.get("category") or routing.get("category") or "unknown"
        text_chars = _as_int(routing.get("text_chars")) or input_tokens * 4
        thinking_tokens = _as_int(r.get("thinking_output_tokens"))
        accounting_unit = _provider_accounting_unit({**r, "is_today": True})
        _add_accounting_to_usage_bucket(bucket, accounting_unit)

        bucket["provider_calls"] += 1
        bucket["turns"] += 1
        bucket["provider_input_tokens"] += provider_input_tokens
        bucket["provider_output_tokens"] += output_tokens
        bucket["provider_total_tokens"] += provider_input_tokens + output_tokens
        bucket["spend_usd"] += cost
        bucket["baseline_provider_cost_usd"] += baseline
        bucket["captured_savings_usd"] += (
            _as_float(accounting_unit.get("routing_savings_usd"))
            + _as_float(accounting_unit.get("crunch_savings_usd"))
            + _as_float(accounting_unit.get("cache_savings_usd"))
        )
        bucket["provider_cost_known"] = True
        bucket["hard_floor_usd"] = _as_float(bucket["hard_floor_usd"]) + cost
        bucket["prompt_cache_creation_tokens"] += cache_creation_tokens
        bucket["prompt_cache_read_tokens"] += cache_read_tokens
        bucket["thinking_tokens"] += thinking_tokens

        if status_code >= 400:
            bucket["errors"] += 1
        if status_code in (429, 529):
            bucket["rate_limited"] += 1
        if routed_model and requested_model != routed_model:
            bucket["routed_calls"] += 1
        if crunch.get("changed"):
            bucket["crunched_calls"] += 1
        if r.get("cache_hit"):
            bucket["local_cache_hits"] += 1
        if routed_model and requested_model != routed_model or crunch.get("changed") or r.get("cache_hit") or cache_read_tokens:
            bucket["optimized_calls"] += 1
        if (not routed_model or requested_model == routed_model) and cost >= high_cost_unrouted_usd:
            bucket["unrouted_high_cost_calls"] += 1
        if category == "tool-result" and text_chars >= min_plateau_chars:
            bucket["large_tool_result_calls"] += 1

        session_key = str(r.get("session_id") or f"call:{r.get('id')}")
        prev_text = bucket["_prev_text_chars_by_session"].get(session_key)
        if (
            prev_text is not None
            and prev_text >= min_plateau_chars
            and text_chars >= min_plateau_chars
            and abs(text_chars - prev_text) / max(prev_text, 1) <= max_plateau_delta_ratio
        ):
            bucket["context_plateau_pairs"] += 1
        bucket["_prev_text_chars_by_session"][session_key] = text_chars

        if cache_creation_tokens:
            bucket["prompt_cache_creation_cost_usd"] += estimate_cost(
                target_model,
                0,
                0,
                cache_creation=cache_creation_tokens,
                provider=provider,
            ) or 0.0
        if cache_read_tokens:
            full_read_cost = estimate_cost(target_model, cache_read_tokens, 0, provider=provider) or 0.0
            cached_read_input_tokens = cache_read_tokens if provider == "openai" else 0
            cached_read_cost = estimate_cost(
                target_model,
                cached_read_input_tokens,
                0,
                cache_read=cache_read_tokens,
                provider=provider,
            ) or 0.0
            bucket["prompt_cache_read_savings_usd"] += max(full_read_cost - cached_read_cost, 0.0)
        if thinking_tokens:
            bucket["thinking_cost_usd"] += estimate_cost(target_model, 0, thinking_tokens, provider=provider) or 0.0

    codex_rows = conn.execute("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.method,
               s.message_chars,
               s.params_chars,
               s.input_items,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
               s.cache_json,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code
               ,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms
        from codex_app_events s
        where s.direction = 'client_to_server'
          and s.method = 'turn/start'
          and date(s.created_at) = date('now')
        order by coalesce(s.session_id, ''), s.created_at
    """).fetchall()

    for row in codex_rows:
        r = dict(row)
        unit = _codex_turn_activity_unit(r)
        input_features = unit["input_features"]
        outcome_features = unit["outcome_features"]
        bucket = bucket_for("codex", r.get("session_id"))
        accounting_unit = _codex_accounting_unit({**r, "is_today": True})
        _add_accounting_to_usage_bucket(bucket, accounting_unit)
        bucket["codex_turns"] += 1
        bucket["turns"] += 1
        bucket["codex_input_text_chars"] += _as_int(r.get("input_text_chars"))
        bucket["codex_result_chars"] += _as_int(r.get("response_result_chars"))
        bucket["codex_input_tokens_est"] += _as_int(input_features.get("input_tokens_est"))
        bucket["codex_output_tokens_est"] += _as_int(outcome_features.get("output_tokens_est"))
        bucket["codex_total_tokens_est"] += _as_int(outcome_features.get("total_tokens_est"))
        bucket["codex_cost_est_usd"] += _as_float(outcome_features.get("cost_est_usd"))
        bucket["codex_baseline_cost_est_usd"] += _as_float(outcome_features.get("cost_baseline_usd"))
        bucket["codex_hard_floor_usd"] += _as_float(outcome_features.get("hard_floor_usd"))
        turn_cost_known = bool(outcome_features.get("cost_known"))
        if bucket["codex_turns"] == 1:
            bucket["codex_cost_known"] = turn_cost_known
            bucket["codex_cost_estimated"] = turn_cost_known
        else:
            bucket["codex_cost_known"] = bool(bucket["codex_cost_known"]) and turn_cost_known
            bucket["codex_cost_estimated"] = bool(bucket["codex_cost_estimated"]) and turn_cost_known
        bucket["excludes_unknown_codex_app_cost"] = not bool(bucket["codex_cost_known"])
        bucket["spend_usd"] += _as_float(outcome_features.get("cost_est_usd"))
        codex_saved = max(
            _as_float(outcome_features.get("cost_baseline_usd")) - _as_float(outcome_features.get("cost_est_usd")),
            0.0,
        )
        bucket["captured_savings_usd"] += (
            _as_float(accounting_unit.get("routing_savings_usd"))
            + _as_float(accounting_unit.get("crunch_savings_usd"))
            + _as_float(accounting_unit.get("cache_savings_usd"))
        )
        cache_decision = unit["optimization_features"]["cache"]
        if cache_decision.get("status") == "hit":
            bucket["local_cache_hits"] += 1
            bucket["codex_exact_cache_savings_usd"] += codex_saved
        if (
            unit["optimization_features"]["routing"].get("applied")
            or unit["optimization_features"]["crunch"].get("changed")
            or cache_decision.get("status") == "hit"
        ):
            bucket["optimized_calls"] += 1
        bucket["hard_floor_usd"] = _as_float(bucket["hard_floor_usd"]) + _as_float(outcome_features.get("hard_floor_usd"))
        if unit.get("mutation_safe"):
            bucket["codex_mutation_safe_turns"] += 1
        if unit["optimization_features"]["routing"].get("reason") == CODEX_APP_TELEMETRY_ONLY_REASON:
            bucket["codex_telemetry_only_turns"] += 1
        if r.get("response_error_code") is not None:
            bucket["errors"] += 1

    rows = []
    for bucket in buckets.values():
        if bucket["context_plateau_pairs"]:
            _add_usage_hint(
                bucket,
                "context_plateau",
                "Repeated context plateau",
                f"{bucket['context_plateau_pairs']} adjacent large-context turns stayed within 3% size.",
            )
        if bucket["thinking_tokens"]:
            _add_usage_hint(
                bucket,
                "thinking_output",
                "High thinking output",
                f"{bucket['thinking_tokens']:,} thinking tokens cost about ${bucket['thinking_cost_usd']:.4f}.",
            )
        if bucket["prompt_cache_creation_cost_usd"] > bucket["prompt_cache_read_savings_usd"] and bucket["prompt_cache_creation_tokens"]:
            _add_usage_hint(
                bucket,
                "cache_warmup",
                "Cache warmup not recouped",
                "Provider prompt-cache writes cost more than reads saved in this bucket today.",
            )
        if bucket["unrouted_high_cost_calls"]:
            _add_usage_hint(
                bucket,
                "unrouted_high_cost",
                "Unrouted high-cost calls",
                f"{bucket['unrouted_high_cost_calls']} provider calls cost at least ${high_cost_unrouted_usd:.2f} and stayed on the requested model.",
            )
        if bucket["large_tool_result_calls"]:
            _add_usage_hint(
                bucket,
                "large_tool_result_context",
                "Large tool-result context",
                f"{bucket['large_tool_result_calls']} tool-result turns carried at least {min_plateau_chars:,} chars.",
            )
        if bucket["rate_limited"]:
            _add_usage_hint(
                bucket,
                "rate_limited",
                "Rate-limit pressure",
                f"{bucket['rate_limited']} turns hit 429/529 responses.",
            )
        elif bucket["errors"]:
            _add_usage_hint(
                bucket,
                "errors",
                "Error signal",
                f"{bucket['errors']} turns returned errors.",
            )
        if bucket["provider_calls"] and not bucket["prompt_cache_read_tokens"] and bucket["provider_input_tokens"] >= 50_000:
            _add_usage_hint(
                bucket,
                "low_prompt_cache_reads",
                "Low prompt-cache reuse",
                "High provider input tokens had no prompt-cache reads today.",
            )

        bucket["spend_usd"] = round(float(bucket["spend_usd"]), 6)
        bucket["baseline_cost_usd"] = round(float(bucket["baseline_cost_usd"]), 6)
        bucket["routing_savings_usd"] = round(float(bucket["routing_savings_usd"]), 6)
        bucket["crunch_savings_usd"] = round(float(bucket["crunch_savings_usd"]), 6)
        bucket["cache_savings_usd"] = round(float(bucket["cache_savings_usd"]), 6)
        bucket["token_basis"] = _mixed_label(bucket["_token_bases"])
        if bucket["provider_calls"] and bucket["codex_turns"]:
            bucket["cost_basis"] = CODEX_APP_COST_BASIS
        elif bucket["provider_calls"]:
            bucket["cost_basis"] = "provider-reported"
        else:
            bucket["cost_basis"] = "codex-estimated-from-chars"
        bucket["source_surfaces"] = [
            {"source_surface": source_surface, "units": count}
            for source_surface, count in sorted(bucket["_source_surface_counts"].items())
        ]
        bucket["baseline_provider_cost_usd"] = round(float(bucket["baseline_provider_cost_usd"]), 6)
        bucket["captured_savings_usd"] = round(float(bucket["captured_savings_usd"]), 6)
        bucket["hard_floor_usd"] = round(float(bucket["hard_floor_usd"]), 6) if bucket["provider_cost_known"] or bucket["codex_cost_known"] else None
        bucket["codex_cost_est_usd"] = round(float(bucket["codex_cost_est_usd"]), 6)
        bucket["codex_baseline_cost_est_usd"] = round(float(bucket["codex_baseline_cost_est_usd"]), 6)
        bucket["codex_hard_floor_usd"] = round(float(bucket["codex_hard_floor_usd"]), 6)
        bucket["codex_exact_cache_savings_usd"] = round(float(bucket["codex_exact_cache_savings_usd"]), 6)
        bucket["prompt_cache_read_savings_usd"] = round(float(bucket["prompt_cache_read_savings_usd"]), 6)
        bucket["prompt_cache_creation_cost_usd"] = round(float(bucket["prompt_cache_creation_cost_usd"]), 6)
        bucket["thinking_cost_usd"] = round(float(bucket["thinking_cost_usd"]), 6)
        bucket["optimization_rate"] = round(bucket["optimized_calls"] / bucket["provider_calls"], 4) if bucket["provider_calls"] else None
        bucket["error_rate"] = round(bucket["errors"] / bucket["turns"], 4) if bucket["turns"] else 0.0
        bucket["potential_hint_count"] = len(bucket["remaining_saving_potential_hints"])
        bucket.pop("_prev_text_chars_by_session", None)
        bucket.pop("_hint_codes", None)
        bucket.pop("_token_bases", None)
        bucket.pop("_cost_bases", None)
        bucket.pop("_source_surface_counts", None)
        rows.append(bucket)

    rows.sort(
        key=lambda row: (
            row["spend_usd"] if row["provider_cost_known"] else -1.0,
            row["provider_total_tokens"],
            row["codex_turns"],
        ),
        reverse=True,
    )

    return {
        "generated_at": utc_now(),
        "schema": "tokenclaw.usage_by_owner.v1",
        "scope": "today",
        "grouping": {
            "display_name": "By source",
            "priority": ["TOKENCLAW_ENGINEER", "TOKENCLAW_APP", "app_family", "session_id"],
            "primary_fields": ["TOKENCLAW_ENGINEER", "TOKENCLAW_APP", "app_family"],
            "fallback_fields": ["session_id"],
            "description": (
                "Usage is grouped by configured engineer/app source labels when present, "
                "then inferred app family, with stored session_id only as the fallback "
                "separator for unlabeled local traffic."
            ),
            "cost_unknown_for": [],
            "raw_prompt_logging": False,
            "codex_cost_basis": CODEX_APP_COST_BASIS,
            "codex_app_model": CODEX_APP_MODEL,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
        },
        "summary": {
            "buckets": len(rows),
            "provider_calls": sum(row["provider_calls"] for row in rows),
            "codex_turns": sum(row["codex_turns"] for row in rows),
            "known_provider_spend_usd": round(
                sum(row["spend_usd"] - row["codex_cost_est_usd"] for row in rows),
                6,
            ),
            "provider_reported_spend_usd": round(
                sum(row["spend_usd"] - row["codex_cost_est_usd"] for row in rows),
                6,
            ),
            "codex_estimated_spend_usd": round(sum(row["codex_cost_est_usd"] for row in rows), 6),
            "codex_exact_cache_savings_usd": round(sum(row["codex_exact_cache_savings_usd"] for row in rows), 6),
            "calculated_spend_usd": round(sum(row["spend_usd"] for row in rows), 6),
            "captured_savings_usd": round(sum(row["captured_savings_usd"] for row in rows), 6),
            "hard_floor_usd": round(sum(row["hard_floor_usd"] or 0.0 for row in rows), 6),
            "codex_cost_unknown": False,
            "cost_basis": CODEX_APP_COST_BASIS if any(row["codex_turns"] for row in rows) else "provider-reported",
        },
        "buckets": rows,
    }


async def stats_full(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
    today_start = _utc_today_start_iso()
    old_context_summary_opportunity = await stats_old_context_summary(store_obj)
    sqlite_maintenance = await stats_sqlite_maintenance(store_obj)
    active_crunch_rule_coverage = _active_crunch_rule_coverage()
    activation_burndown = await stats_local_activation_next_action_queue(limit=5, store_obj=store_obj)
    closed_loop_activation = _closed_loop_activation_readiness(activation_burndown)
    activation_successor_queue_health = build_activation_successor_queue_health(limit=5)
    savings_loop_bottlenecks = build_savings_loop_bottlenecks_report(
        store_obj,
        db_path=getattr(store_obj, "path", None),
        config_dir=os.getenv("TOKENCLAW_CONFIG_DIR") or os.getenv("TOKENCLAW_POLICY_CONFIG_DIR"),
        activation_burndown=activation_burndown,
        policy_scan_limit=1000,
        persist_outcome_feedback=False,
    )
    managed_mode = managed_mode_public_meta()

    def q(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def s(sql: str, params: tuple = ()) -> Any:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None

    total_calls = s("select count(*) from calls") or 0
    today_calls = s("select count(*) from calls where date(created_at) = date('now')") or 0
    total_cost = s("select sum(cost_est_usd) from calls") or 0.0
    today_cost = s("select sum(cost_est_usd) from calls where date(created_at) = date('now')") or 0.0
    cache_hits = s("select count(*) from calls where cache_hit = 1") or 0
    cache_cost_saved = s("select count(*) * 0.003 from calls where cache_hit = 1") or 0.0  # rough avg
    avg_latency = s("select avg(latency_ms) from calls where latency_ms is not null") or 0
    routed_count = s("select count(*) from calls where requested_model != routed_model and routed_model is not null") or 0
    crunched_count = s("select count(*) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    errors = s("select count(*) from calls where status_code >= 400") or 0
    today_errors = s("select count(*) from calls where status_code >= 400 and date(created_at) = date('now')") or 0

    # Estimate routing savings: calls where model was downgraded, cost diff
    routing_savings = 0.0
    today_routing_savings = 0.0
    downgraded = q("""
        select coalesce(provider, 'anthropic') as provider, requested_model, routed_model,
               coalesce(actual_input_tokens, input_tokens_est, 0) as in_tok,
               coalesce(actual_output_tokens, output_tokens_est, 0) as out_tok,
               (date(created_at) = date('now')) as is_today
        from calls where requested_model != routed_model and routed_model is not null
    """)
    for row in downgraded:
        req_cost = estimate_cost(row["requested_model"], row["in_tok"], row["out_tok"], provider=row["provider"]) or 0
        act_cost = estimate_cost(row["routed_model"], row["in_tok"], row["out_tok"], provider=row["provider"]) or 0
        delta = max(0.0, req_cost - act_cost)
        routing_savings += delta
        if row["is_today"]:
            today_routing_savings += delta

    today_cache_savings = s("select count(*) * 0.003 from calls where cache_hit = 1 and date(created_at) = date('now')") or 0.0

    crunch_chars_saved = s("select sum(json_extract(crunch_json, '$.saved_chars')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    crunch_tokens_saved = s("select sum(json_extract(crunch_json, '$.tokens_saved_est')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    avg_crunch_ratio = s("select avg(json_extract(crunch_json, '$.crunch_ratio')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    crunch_savings = 0.0
    today_crunch_savings = 0.0
    crunch_by_model = q("""
        select coalesce(provider, 'anthropic') as provider,
               coalesce(routed_model, requested_model) as model,
               sum(coalesce(json_extract(crunch_json, '$.tokens_saved_est'), 0)) as saved_tok,
               sum(coalesce(actual_input_tokens, input_tokens_est, 0)) as input_tok,
               sum(coalesce(cache_read_input_tokens, 0)) as cache_read_tok,
               sum(case when date(created_at) = date('now') then coalesce(json_extract(crunch_json, '$.tokens_saved_est'), 0) else 0 end) as today_saved_tok,
               sum(case when date(created_at) = date('now') then coalesce(actual_input_tokens, input_tokens_est, 0) else 0 end) as today_input_tok,
               sum(case when date(created_at) = date('now') then coalesce(cache_read_input_tokens, 0) else 0 end) as today_cache_read_tok
        from calls
        where json_extract(crunch_json, '$.changed') = 1
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in crunch_by_model:
        crunch_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=int(row["saved_tok"] or 0),
            input_tokens=int(row["input_tok"] or 0),
            cache_read_tokens=int(row["cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0
        today_crunch_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=int(row["today_saved_tok"] or 0),
            input_tokens=int(row["today_input_tok"] or 0),
            cache_read_tokens=int(row["today_cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0

    crunch_canary_captured_savings = _crunch_canary_captured_savings(conn)
    crunch_canary_captured_summary = crunch_canary_captured_savings.get("summary", {})

    summary_applied_count = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
    """) or 0)
    today_summary_applied_count = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
          and date(created_at) = date('now')
    """) or 0)
    summary_created_count = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.reason') = 'summary-created'
    """) or 0)
    summary_cache_hits = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.summary_cache_hit') = 1
    """) or 0)
    summary_extra_cost = float(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.summary_cost_est_usd'), 0))
        from calls
    """) or 0.0)
    today_summary_extra_cost = float(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.summary_cost_est_usd'), 0))
        from calls
        where date(created_at) = date('now')
    """) or 0.0)
    summary_chars_saved = int(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.saved_chars'), 0))
        from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
    """) or 0)
    summary_tokens_saved = int(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.tokens_saved_est'), 0))
        from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
    """) or 0)
    summary_savings = 0.0
    today_summary_savings = 0.0
    summary_by_model = q("""
        select coalesce(provider, 'anthropic') as provider,
               coalesce(routed_model, requested_model) as model,
               sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.tokens_saved_est'), 0)) as saved_tok,
               sum(coalesce(actual_input_tokens, input_tokens_est, 0)) as input_tok,
               sum(coalesce(cache_read_input_tokens, 0)) as cache_read_tok,
               sum(case when date(created_at) = date('now') then coalesce(json_extract(crunch_json, '$.old_context_summarization.tokens_saved_est'), 0) else 0 end) as today_saved_tok,
               sum(case when date(created_at) = date('now') then coalesce(actual_input_tokens, input_tokens_est, 0) else 0 end) as today_input_tok,
               sum(case when date(created_at) = date('now') then coalesce(cache_read_input_tokens, 0) else 0 end) as today_cache_read_tok
        from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in summary_by_model:
        summary_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=max(0, int(row["saved_tok"] or 0)),
            input_tokens=int(row["input_tok"] or 0),
            cache_read_tokens=int(row["cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0
        today_summary_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=max(0, int(row["today_saved_tok"] or 0)),
            input_tokens=int(row["today_input_tok"] or 0),
            cache_read_tokens=int(row["today_cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0
    prompt_cache_creation_tokens = s("select sum(cache_creation_input_tokens) from calls") or 0
    prompt_cache_read_tokens = s("select sum(cache_read_input_tokens) from calls") or 0
    prompt_cache_hits = s("select count(*) from calls where cache_read_input_tokens > 0") or 0
    prompt_cache_hit_rate = round(prompt_cache_hits / total_calls, 4) if total_calls else 0

    prompt_cache_savings = 0.0
    today_prompt_cache_savings = 0.0
    prompt_cache_cached_read_cost = 0.0
    today_prompt_cache_cached_read_cost = 0.0
    prompt_cache_creation_cost = 0.0
    today_prompt_cache_creation_cost = 0.0
    prompt_cache_creation_premium = 0.0
    today_prompt_cache_creation_premium = 0.0
    prompt_cache_net_discount = 0.0
    today_prompt_cache_net_discount = 0.0
    prompt_cache_accounting_by_model: list[dict[str, Any]] = []
    cache_read_by_model = q("""
        select coalesce(routed_model, requested_model) as model,
               sum(coalesce(cache_creation_input_tokens, 0)) as creation_tok,
               sum(cache_read_input_tokens) as read_tok,
               sum(case when date(created_at) = date('now') then coalesce(cache_creation_input_tokens, 0) else 0 end) as today_creation_tok,
               sum(case when date(created_at) = date('now') then cache_read_input_tokens else 0 end) as today_read_tok,
               coalesce(provider, 'anthropic') as provider
        from calls
        where coalesce(cache_read_input_tokens, 0) > 0
           or coalesce(cache_creation_input_tokens, 0) > 0
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in cache_read_by_model:
        accounting = provider_prompt_cache_accounting(
            row["model"],
            provider=row["provider"],
            cache_creation_tokens=_as_int(row.get("creation_tok")),
            cache_read_tokens=_as_int(row.get("read_tok")),
        )
        today_accounting = provider_prompt_cache_accounting(
            row["model"],
            provider=row["provider"],
            cache_creation_tokens=_as_int(row.get("today_creation_tok")),
            cache_read_tokens=_as_int(row.get("today_read_tok")),
        )
        prompt_cache_savings += _as_float(accounting.get("read_discount_usd"))
        today_prompt_cache_savings += _as_float(today_accounting.get("read_discount_usd"))
        prompt_cache_cached_read_cost += _as_float(accounting.get("actual_cached_read_cost_usd"))
        today_prompt_cache_cached_read_cost += _as_float(today_accounting.get("actual_cached_read_cost_usd"))
        prompt_cache_creation_cost += _as_float(accounting.get("creation_cost_usd"))
        today_prompt_cache_creation_cost += _as_float(today_accounting.get("creation_cost_usd"))
        prompt_cache_creation_premium += _as_float(accounting.get("creation_premium_usd"))
        today_prompt_cache_creation_premium += _as_float(today_accounting.get("creation_premium_usd"))
        prompt_cache_net_discount += _as_float(accounting.get("net_provider_cache_discount_usd"))
        today_prompt_cache_net_discount += _as_float(today_accounting.get("net_provider_cache_discount_usd"))
        prompt_cache_accounting_by_model.append({
            "provider": accounting["provider"],
            "model": accounting["model"],
            "matched_model": accounting.get("matched_model"),
            "processing_mode": accounting.get("processing_mode"),
            "pricing_source": accounting.get("pricing_source"),
            "pricing_version": accounting.get("pricing_version"),
            "cost_known": accounting.get("cost_known"),
            "read_tokens": accounting["read_tokens"],
            "creation_tokens": accounting["creation_tokens"],
            "read_discount_usd": round(_as_float(accounting.get("read_discount_usd")), 8),
            "actual_cached_read_cost_usd": round(_as_float(accounting.get("actual_cached_read_cost_usd")), 8),
            "creation_cost_usd": round(_as_float(accounting.get("creation_cost_usd")), 8),
            "creation_premium_usd": round(_as_float(accounting.get("creation_premium_usd")), 8),
            "net_provider_cache_discount_usd": round(_as_float(accounting.get("net_provider_cache_discount_usd")), 8),
            "today": {
                "read_tokens": today_accounting["read_tokens"],
                "creation_tokens": today_accounting["creation_tokens"],
                "read_discount_usd": round(_as_float(today_accounting.get("read_discount_usd")), 8),
                "actual_cached_read_cost_usd": round(_as_float(today_accounting.get("actual_cached_read_cost_usd")), 8),
                "creation_cost_usd": round(_as_float(today_accounting.get("creation_cost_usd")), 8),
                "creation_premium_usd": round(_as_float(today_accounting.get("creation_premium_usd")), 8),
                "net_provider_cache_discount_usd": round(_as_float(today_accounting.get("net_provider_cache_discount_usd")), 8),
            },
            "pricing_basis": accounting.get("pricing_basis"),
        })

    prompt_cache_accounting = {
        "schema": "tokenclaw.provider_prompt_cache_accounting_rollup.v1",
        "label": "provider prompt-cache discount/economics",
        "boundary": "provider-side prompt-cache pricing; separate from AgentFlow local exact-cache replay savings",
        "totals": {
            "read_tokens": int(prompt_cache_read_tokens),
            "creation_tokens": int(prompt_cache_creation_tokens),
            "full_price_equivalent_read_cost_usd": round(prompt_cache_savings + prompt_cache_cached_read_cost, 8),
            "actual_cached_read_cost_usd": round(prompt_cache_cached_read_cost, 8),
            "read_discount_usd": round(prompt_cache_savings, 8),
            "creation_cost_usd": round(prompt_cache_creation_cost, 8),
            "creation_premium_usd": round(prompt_cache_creation_premium, 8),
            "net_provider_cache_discount_usd": round(prompt_cache_net_discount, 8),
        },
        "today": {
            "read_tokens": int(s("select sum(cache_read_input_tokens) from calls where date(created_at) = date('now')") or 0),
            "creation_tokens": int(s("select sum(cache_creation_input_tokens) from calls where date(created_at) = date('now')") or 0),
            "full_price_equivalent_read_cost_usd": round(today_prompt_cache_savings + today_prompt_cache_cached_read_cost, 8),
            "actual_cached_read_cost_usd": round(today_prompt_cache_cached_read_cost, 8),
            "read_discount_usd": round(today_prompt_cache_savings, 8),
            "creation_cost_usd": round(today_prompt_cache_creation_cost, 8),
            "creation_premium_usd": round(today_prompt_cache_creation_premium, 8),
            "net_provider_cache_discount_usd": round(today_prompt_cache_net_discount, 8),
        },
        "by_model": sorted(
            prompt_cache_accounting_by_model,
            key=lambda item: (str(item.get("provider") or ""), str(item.get("model") or "")),
        ),
    }

    thinking_output_tokens = int(s("select sum(thinking_output_tokens) from calls") or 0)
    today_thinking_output_tokens = int(s("select sum(thinking_output_tokens) from calls where date(created_at) = date('now')") or 0)
    thinking_cost = 0.0
    today_thinking_cost = 0.0
    thinking_by_model = q("""
        select coalesce(routed_model, requested_model) as model,
               sum(thinking_output_tokens) as think_tok,
               sum(case when date(created_at) = date('now') then coalesce(thinking_output_tokens, 0) else 0 end) as today_think_tok,
               coalesce(provider, 'anthropic') as provider
        from calls where thinking_output_tokens > 0
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in thinking_by_model:
        thinking_cost += estimate_cost(row["model"], 0, row["think_tok"] or 0, provider=row["provider"]) or 0
        today_thinking_cost += estimate_cost(row["model"], 0, row["today_think_tok"] or 0, provider=row["provider"]) or 0

    codex_app_total_events = int(s("select count(*) from codex_app_events") or 0)
    codex_app_today_events = int(s(
        "select count(*) from codex_app_events where created_at >= ?",
        (today_start,),
    ) or 0)
    codex_app_sessions = int(s("select count(distinct session_id) from codex_app_events where session_id is not null") or 0)
    codex_app_turns = int(s("select count(*) from codex_app_events where direction = 'server_to_client' and method = 'turn/completed'") or 0)
    codex_app_today_turns = int(s("""
        select count(*) from codex_app_events
        where direction = 'server_to_client'
          and method = 'turn/completed'
          and created_at >= ?
    """, (today_start,)) or 0)
    codex_app_last_event_at = s("select max(created_at) from codex_app_events")
    codex_app_input_text_chars = int(s("select sum(input_text_chars) from codex_app_events where direction = 'client_to_server' and method = 'turn/start'") or 0)
    codex_app_today_input_text_chars = int(s("""
        select sum(input_text_chars) from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
          and created_at >= ?
    """, (today_start,)) or 0)
    codex_app_avg_latency = s("select avg(latency_ms) from codex_app_events where latency_ms is not null") or 0
    codex_turn_rows = q("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.message_chars,
               s.params_chars,
               s.input_items,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
               s.cache_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms,
               (s.created_at >= ?) as is_today
        from codex_app_events s
        where s.direction = 'client_to_server'
          and s.method = 'turn/start'
    """, (today_start,))
    codex_input_tokens_est = 0
    codex_output_tokens_est = 0
    codex_cost_est = 0.0
    codex_cache_savings = 0.0
    codex_cost_known = True
    today_codex_input_tokens_est = 0
    today_codex_output_tokens_est = 0
    today_codex_cost_est = 0.0
    today_codex_cache_savings = 0.0
    today_codex_cost_known = True
    for row in codex_turn_rows:
        cache = _json_obj(row.get("cache_json"))
        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), row.get("response_result_chars"), cache)
        codex_input_tokens_est += estimates["input_tokens_est"]
        codex_output_tokens_est += estimates["output_tokens_est"]
        codex_cost_est += _as_float(estimates["cost_est_usd"])
        codex_cache_savings += estimates["cache_savings_usd"]
        codex_cost_known = codex_cost_known and bool(estimates["cost_known"])
        if row.get("is_today"):
            today_codex_input_tokens_est += estimates["input_tokens_est"]
            today_codex_output_tokens_est += estimates["output_tokens_est"]
            today_codex_cost_est += _as_float(estimates["cost_est_usd"])
            today_codex_cache_savings += estimates["cache_savings_usd"]
            today_codex_cost_known = today_codex_cost_known and bool(estimates["cost_known"])
    if not codex_turn_rows:
        codex_cost_known = CODEX_APP_COST_KNOWN
        today_codex_cost_known = CODEX_APP_COST_KNOWN

    provider_input_tokens = int(s("""
        select sum(
            coalesce(actual_input_tokens, input_tokens_est, 0)
            + coalesce(cache_creation_input_tokens, 0)
            + coalesce(cache_read_input_tokens, 0)
        ) from calls
    """) or 0)
    provider_output_tokens = int(s("""
        select sum(coalesce(actual_output_tokens, output_tokens_est, 0))
        from calls
    """) or 0)
    today_provider_input_tokens = int(s("""
        select sum(
            coalesce(actual_input_tokens, input_tokens_est, 0)
            + coalesce(cache_creation_input_tokens, 0)
            + coalesce(cache_read_input_tokens, 0)
        ) from calls
        where date(created_at) = date('now')
    """) or 0)
    today_provider_output_tokens = int(s("""
        select sum(coalesce(actual_output_tokens, output_tokens_est, 0))
        from calls
        where date(created_at) = date('now')
    """) or 0)
    provider_accounting_rows = q("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, stream, cache_hit, status_code,
               latency_ms, input_tokens_est, output_tokens_est,
               actual_input_tokens, actual_output_tokens, cost_est_usd,
               cost_baseline_usd, crunch_json, routing_json, cache_json, error,
               request_json, response_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens, retry_count,
               thinking_output_tokens,
               (date(created_at) = date('now')) as is_today
        from calls
    """)
    accounting_units = (
        [_provider_accounting_unit(row) for row in provider_accounting_rows]
        + [_codex_accounting_unit(row) for row in codex_turn_rows]
    )
    accounting_total = _accounting_rollup(accounting_units)
    accounting_today = _accounting_rollup([unit for unit in accounting_units if unit.get("is_today")])
    realized_routing_savings = _as_float(accounting_total.get("routing_savings_usd"))
    today_realized_routing_savings = _as_float(accounting_today.get("routing_savings_usd"))
    realized_crunch_savings = _as_float(accounting_total.get("crunch_savings_usd"))
    today_realized_crunch_savings = _as_float(accounting_today.get("crunch_savings_usd"))
    realized_cache_savings = _as_float(accounting_total.get("cache_savings_usd"))
    today_realized_cache_savings = _as_float(accounting_today.get("cache_savings_usd"))
    today_savings_buckets = {
        "routing_usd": round(today_realized_routing_savings, 6),
        "crunching_usd": round(today_realized_crunch_savings, 6),
        "exact_local_cache_usd": round(today_realized_cache_savings, 6),
        "provider_exact_local_cache_usd": round(today_cache_savings, 6),
        "codex_app_exact_local_cache_usd": round(today_codex_cache_savings, 6),
        "provider_prompt_cache_discount_usd": round(today_prompt_cache_savings, 6),
    }
    savings_buckets = {
        "routing_usd": round(realized_routing_savings, 6),
        "crunching_usd": round(realized_crunch_savings, 6),
        "exact_local_cache_usd": round(realized_cache_savings, 6),
        "provider_exact_local_cache_usd": round(cache_cost_saved, 6),
        "codex_app_exact_local_cache_usd": round(codex_cache_savings, 6),
        "provider_prompt_cache_discount_usd": round(prompt_cache_savings, 6),
    }
    # AgentFlow-controlled savings keys only (routing, crunching, local exact-cache).
    # provider_prompt_cache_discount_usd is provider-side billing economics, not AgentFlow-generated.
    # sub-buckets provider_exact_local_cache_usd / codex_app_exact_local_cache_usd are already
    # counted inside exact_local_cache_usd, so exclude them from the sum to avoid double-counting.
    _TOKENCLAW_SAVINGS_KEYS = ("routing_usd", "crunching_usd", "exact_local_cache_usd")
    today_tokenclaw_generated_savings = sum(float(today_savings_buckets[k] or 0.0) for k in _TOKENCLAW_SAVINGS_KEYS)
    tokenclaw_generated_savings = sum(float(savings_buckets[k] or 0.0) for k in _TOKENCLAW_SAVINGS_KEYS)
    today_total_savings = sum(float(value or 0.0) for value in today_savings_buckets.values())
    total_savings = sum(float(value or 0.0) for value in savings_buckets.values())
    today_observed_baseline = today_cost + today_tokenclaw_generated_savings
    observed_baseline = total_cost + tokenclaw_generated_savings
    today_calculated_spend = today_cost + today_codex_cost_est
    calculated_spend = total_cost + codex_cost_est
    today_observed_baseline_with_codex = today_observed_baseline + today_codex_cost_est
    observed_baseline_with_codex = observed_baseline + codex_cost_est
    today_hard_floor = today_calculated_spend
    hard_floor = calculated_spend
    today_provider_spend_rounded = round(today_cost, 6)
    provider_spend_rounded = round(total_cost, 6)
    today_codex_spend_rounded = round(today_codex_cost_est, 6)
    codex_spend_rounded = round(codex_cost_est, 6)
    executive_summary = {
        "schema": "tokenclaw.executive_summary.v1",
        "accounting_today": accounting_today,
        "accounting_total": accounting_total,
        "tokens_today": {
            "total_tokens": today_provider_input_tokens + today_provider_output_tokens + today_codex_input_tokens_est + today_codex_output_tokens_est,
            "provider_total_tokens": today_provider_input_tokens + today_provider_output_tokens,
            "provider_input_tokens": today_provider_input_tokens,
            "provider_output_tokens": today_provider_output_tokens,
            "codex_app_turns": codex_app_today_turns,
            "codex_app_input_text_chars": codex_app_today_input_text_chars,
            "codex_app_input_tokens_est": today_codex_input_tokens_est,
            "codex_app_output_tokens_est": today_codex_output_tokens_est,
            "codex_app_total_tokens_est": today_codex_input_tokens_est + today_codex_output_tokens_est,
            "codex_app_cost_known": today_codex_cost_known,
            "codex_app_cost_estimated": today_codex_cost_known,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "tokens_total": {
            "total_tokens": provider_input_tokens + provider_output_tokens + codex_input_tokens_est + codex_output_tokens_est,
            "provider_total_tokens": provider_input_tokens + provider_output_tokens,
            "provider_input_tokens": provider_input_tokens,
            "provider_output_tokens": provider_output_tokens,
            "codex_app_turns": codex_app_turns,
            "codex_app_input_text_chars": codex_app_input_text_chars,
            "codex_app_input_tokens_est": codex_input_tokens_est,
            "codex_app_output_tokens_est": codex_output_tokens_est,
            "codex_app_total_tokens_est": codex_input_tokens_est + codex_output_tokens_est,
            "codex_app_cost_known": codex_cost_known,
            "codex_app_cost_estimated": codex_cost_known,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "spend": {
            "today_calculated_spend_usd": round(today_provider_spend_rounded + today_codex_spend_rounded, 6),
            "calculated_spend_usd": round(provider_spend_rounded + codex_spend_rounded, 6),
            "today_provider_spend_usd": today_provider_spend_rounded,
            "total_provider_spend_usd": provider_spend_rounded,
            "today_codex_app_estimated_spend_usd": today_codex_spend_rounded,
            "codex_app_estimated_spend_usd": codex_spend_rounded,
            "today_baseline_provider_cost_usd": round(today_observed_baseline, 6),
            "baseline_provider_cost_usd": round(observed_baseline, 6),
            "today_baseline_calculated_cost_usd": round(today_observed_baseline_with_codex, 6),
            "baseline_calculated_cost_usd": round(observed_baseline_with_codex, 6),
            "thinking_cost_today_usd": round(today_thinking_cost, 6),
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "savings": {
            "today_tokenclaw_generated_savings_usd": round(today_tokenclaw_generated_savings, 6),
            "tokenclaw_generated_savings_usd": round(tokenclaw_generated_savings, 6),
            "today_tokenclaw_generated_buckets": {k: today_savings_buckets[k] for k in (*_TOKENCLAW_SAVINGS_KEYS, "provider_exact_local_cache_usd", "codex_app_exact_local_cache_usd")},
            "tokenclaw_generated_buckets": {k: savings_buckets[k] for k in (*_TOKENCLAW_SAVINGS_KEYS, "provider_exact_local_cache_usd", "codex_app_exact_local_cache_usd")},
            "provider_prompt_cache_discount_usd": round(prompt_cache_savings, 6),
            "today_provider_prompt_cache_discount_usd": round(today_prompt_cache_savings, 6),
            "provider_prompt_cache_economics": {
                "today_read_discount_usd": round(today_prompt_cache_savings, 6),
                "read_discount_usd": round(prompt_cache_savings, 6),
                "today_cached_read_cost_usd": round(today_prompt_cache_cached_read_cost, 6),
                "cached_read_cost_usd": round(prompt_cache_cached_read_cost, 6),
                "today_creation_cost_usd": round(today_prompt_cache_creation_cost, 6),
                "creation_cost_usd": round(prompt_cache_creation_cost, 6),
                "today_creation_premium_usd": round(today_prompt_cache_creation_premium, 6),
                "creation_premium_usd": round(prompt_cache_creation_premium, 6),
                "today_net_discount_usd": round(today_prompt_cache_net_discount, 6),
                "net_discount_usd": round(prompt_cache_net_discount, 6),
                "label": "provider billing efficiency",
                "boundary": "provider-side pricing; separate from AgentFlow local exact-cache replay savings",
            },
            # backward-compat: these include provider_prompt_cache_discount_usd; prefer tokenclaw_generated_savings_usd
            "today_total_savings_usd": round(today_total_savings, 6),
            "total_savings_usd": round(total_savings, 6),
            "today_buckets": today_savings_buckets,
            "buckets": savings_buckets,
        },
        "hard_floor": {
            "today_unavoidable_provider_spend_usd": round(today_hard_floor, 6),
            "unavoidable_provider_spend_usd": round(hard_floor, 6),
            "today_unavoidable_calculated_spend_usd": round(today_hard_floor, 6),
            "unavoidable_calculated_spend_usd": round(hard_floor, 6),
            "today_baseline_minus_feasible_savings_usd": round(today_observed_baseline_with_codex - today_tokenclaw_generated_savings, 6),
            "excludes_unknown_codex_app_cost": not today_codex_cost_known,
            "codex_app_cost_estimated": today_codex_cost_known,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "health": {
            "errors": today_errors,
            "errors_today": today_errors,
            "total_errors": errors,
            "avg_latency_ms": round(avg_latency),
            "rate_limit_cooldowns": None,
        },
    }

    recent = q("""
        select id, coalesce(provider, 'anthropic') as provider, created_at, requested_model, routed_model, stream, cache_hit,
               status_code, latency_ms,
               coalesce(actual_input_tokens, input_tokens_est) as input_tokens,
               coalesce(actual_output_tokens, output_tokens_est) as output_tokens,
               cost_est_usd,
               json_extract(crunch_json, '$.changed') as crunched,
               json_extract(crunch_json, '$.saved_chars') as crunch_saved_chars,
               json_extract(routing_json, '$.reason') as routing_reason,
               error
        from calls order by created_at desc limit 50
    """)

    routing_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, requested_model, routed_model, count(*) as count
        from calls group by coalesce(provider, 'anthropic'), requested_model, routed_model order by count desc limit 15
    """)

    category_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, coalesce(category, 'unknown') as category, count(*) as count,
               round(sum(coalesce(cost_est_usd, 0)), 6) as cost_usd,
               sum(case when requested_model != routed_model and routed_model is not null then 1 else 0 end) as routed_count
        from calls group by coalesce(provider, 'anthropic'), coalesce(category, 'unknown') order by count desc
    """)

    cache_rows = q("""
        select created_at, stream, cache_hit, status_code, cache_json, routing_json,
               path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint,
               (created_at >= ?) as is_today
        from calls
        union all
        select created_at, 0 as stream,
               case when json_extract(cache_json, '$.status') = 'hit' then 1 else 0 end as cache_hit,
               null as status_code,
               cache_json,
               routing_json,
               'codex-app://turn/start' as path,
               'codex-app' as provider,
               ? as source_surface,
               'turn_start' as endpoint,
               (created_at >= ?) as is_today
        from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
    """, (today_start, CODEX_APP_SOURCE_SURFACE, today_start))
    cache_decision_breakdown = _cache_decision_breakdown(cache_rows)
    today_cache_decision_breakdown = _cache_decision_breakdown(cache_rows, today_only=True)
    crunch_rule_rows = q("""
        select created_at,
               crunch_json,
               coalesce(provider, 'anthropic') as provider,
               source_surface,
               category,
               (created_at >= ?) as is_today
        from calls
        where crunch_json is not null
    """, (today_start,))
    crunch_rule_breakdown, crunch_rule_group_breakdown = _crunch_rule_breakdowns(crunch_rule_rows)
    today_crunch_rule_breakdown, today_crunch_rule_group_breakdown = _crunch_rule_breakdowns(crunch_rule_rows, today_only=True)
    crunch_rule_savings_breakdown = [
        row for row in crunch_rule_breakdown
        if row.get("status") == "applied" and _as_int(row.get("total_chars_saved")) > 0
    ][:5]
    today_crunch_rule_savings_breakdown = [
        row for row in today_crunch_rule_breakdown
        if row.get("status") == "applied" and _as_int(row.get("total_chars_saved")) > 0
    ][:5]
    cache_ladder_rows = q("""
        select created_at, stream, cache_hit, status_code, cache_json, routing_json,
               path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint
        from calls
        order by created_at desc
        limit ?
    """, (_CACHE_BLOCKER_SCAN_LIMIT,)) + q("""
        select created_at, 0 as stream,
               case when json_extract(cache_json, '$.status') = 'hit' then 1 else 0 end as cache_hit,
               null as status_code,
               cache_json,
               routing_json,
               'codex-app://turn/start' as path,
               'codex-app' as provider,
               ? as source_surface,
               'turn_start' as endpoint
        from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
        order by created_at desc
        limit ?
    """, (CODEX_APP_SOURCE_SURFACE, _CACHE_BLOCKER_SCAN_LIMIT))
    cache_ladder_rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    cache_zero_hit_blocker_ladder = _cache_zero_hit_blocker_ladder(
        cache_ladder_rows,
        scan_limit=_CACHE_BLOCKER_SCAN_LIMIT,
    )
    cache_replayability = await stats_cache_replayability(store_obj, limit=20)
    cache_replay_cohort_ranking = await stats_cache_replay_cohort_ranking(store_obj, limit=20, row_limit=1000)
    cache_replay_confidence = await stats_cache_replay_confidence(store_obj, limit=50)
    cache_replay_readiness = await stats_cache_replay_readiness(store_obj, limit=50)
    cache_replay_activation_health = await stats_cache_replay_activation_health(store_obj, limit=50, scan_limit=1000)
    streaming_cache_hit_recovery = await stats_streaming_cache_hit_recovery(store_obj, limit=50, scan_limit=1000)
    cache_effectiveness = await stats_cache_effectiveness(store_obj, limit=5, scan_limit=5000)
    pattern_decision_breakdown = _pattern_decision_breakdown(provider_accounting_rows)
    today_pattern_decision_breakdown = _pattern_decision_breakdown(provider_accounting_rows, today_only=True)

    error_rows = q("""
        select created_at,
               coalesce(provider, 'anthropic') as provider,
               status_code,
               requested_model,
               routed_model,
               coalesce(routed_model, requested_model) as model,
               error,
               (date(created_at) = date('now')) as is_today
        from calls
        where status_code >= 400
        order by created_at desc
    """)
    error_breakdown = _error_breakdown(error_rows)
    today_error_breakdown = _error_breakdown(error_rows, today_only=True)

    routing_experiment_report = build_routing_experiment_report(store_obj, limit=20)
    routing_experiment_summary = routing_experiment_report["candidates"]
    routing_experiment_report_summary = routing_experiment_report["summary"]

    provider_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, count(*) as count,
               round(sum(coalesce(cost_est_usd, 0)), 6) as cost_usd,
               sum(case when requested_model != routed_model and routed_model is not null then 1 else 0 end) as routed_count
        from calls group by coalesce(provider, 'anthropic') order by count desc
    """)
    if codex_app_total_events:
        provider_breakdown.append({
            "provider": "codex-app",
            "count": codex_app_turns,
            "cost_usd": round(codex_cost_est, 6),
            "routed_count": 0,
            "events": codex_app_total_events,
            "tokens_est": codex_input_tokens_est + codex_output_tokens_est,
            "cost_basis": "codex-estimated-from-chars",
        })

    codex_app_methods = q("""
        select direction, coalesce(method, '(response)') as method, count(*) as count,
               round(avg(latency_ms)) as avg_latency_ms,
               sum(coalesce(input_text_chars, 0)) as input_text_chars
        from codex_app_events
        group by direction, coalesce(method, '(response)')
        order by count desc
        limit 20
    """)
    codex_app_recent = q("""
        select created_at, direction, coalesce(method, '(response)') as method,
               request_id, thread_id, message_chars, input_items, input_text_chars,
               result_chars, error_code, error_message, latency_ms, session_id
        from codex_app_events
        order by created_at desc
        limit 50
    """)

    return {
        "executive_summary": executive_summary,
        "managed_mode": managed_mode,
        "source_surface_accounting": accounting_total["source_surfaces"],
        "today_source_surface_accounting": accounting_today["source_surfaces"],
        "savings_by_source_surface": accounting_total["savings_by_source_surface"],
        "today_savings_by_source_surface": accounting_today["savings_by_source_surface"],
        "summary": {
            "total_calls": total_calls,
            "today_calls": today_calls,
            "total_cost_usd": round(total_cost, 6),
            "today_cost_usd": round(today_cost, 6),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_calls, 4) if total_calls else 0,
            "routing_savings_usd": round(realized_routing_savings, 6),
            "today_routing_savings_usd": round(today_realized_routing_savings, 6),
            "cache_savings_usd": round(realized_cache_savings, 6),
            "today_cache_savings_usd": round(today_realized_cache_savings, 6),
            "provider_cache_savings_usd": round(cache_cost_saved, 6),
            "today_provider_cache_savings_usd": round(today_cache_savings, 6),
            "codex_app_cache_savings_usd": round(codex_cache_savings, 6),
            "today_codex_app_cache_savings_usd": round(today_codex_cache_savings, 6),
            "total_savings_usd": round(realized_routing_savings + realized_crunch_savings + realized_cache_savings, 6),
            "avg_latency_ms": round(avg_latency),
            "routed_count": routed_count,
            "crunched_count": crunched_count,
            "crunch_chars_saved": crunch_chars_saved,
            "crunch_tokens_saved": int(crunch_tokens_saved),
            "crunch_savings_usd": round(realized_crunch_savings, 6),
            "today_crunch_savings_usd": round(today_realized_crunch_savings, 6),
            "crunch_captured_savings_usd": round(_as_float(crunch_canary_captured_summary.get("captured_saved_usd")), 6),
            "crunch_captured_saved_tokens": _as_int(crunch_canary_captured_summary.get("captured_saved_tokens")),
            "crunch_captured_applied_count": _as_int(crunch_canary_captured_summary.get("applied_count")),
            "crunch_captured_holdout_count": _as_int(crunch_canary_captured_summary.get("holdout_count")),
            "avg_crunch_ratio": round(avg_crunch_ratio, 4),
            "active_crunch_rule_coverage": active_crunch_rule_coverage,
            "old_context_summary_applied_count": summary_applied_count,
            "today_old_context_summary_applied_count": today_summary_applied_count,
            "old_context_summary_created_count": summary_created_count,
            "old_context_summary_cache_hits": summary_cache_hits,
            "old_context_summary_cache_hit_rate": round(summary_cache_hits / summary_applied_count, 4) if summary_applied_count else 0,
            "old_context_summary_chars_saved": summary_chars_saved,
            "old_context_summary_tokens_saved": summary_tokens_saved,
            "old_context_summary_cost_usd": round(summary_extra_cost, 6),
            "today_old_context_summary_cost_usd": round(today_summary_extra_cost, 6),
            "old_context_summary_savings_usd": round(summary_savings, 6),
            "today_old_context_summary_savings_usd": round(today_summary_savings, 6),
            "today_old_context_summary_net_usd": round(today_summary_savings - today_summary_extra_cost, 6),
            "errors": errors,
            "prompt_cache_creation_tokens": int(prompt_cache_creation_tokens),
            "prompt_cache_read_tokens": int(prompt_cache_read_tokens),
            "prompt_cache_hit_rate": prompt_cache_hit_rate,
            "prompt_cache_savings_usd": round(prompt_cache_savings, 6),
            "today_prompt_cache_savings_usd": round(today_prompt_cache_savings, 6),
            "provider_prompt_cache_label": "provider prompt-cache discount/economics",
            "provider_prompt_cache_boundary": "provider-side pricing; separate from AgentFlow local exact-cache replay savings",
            "provider_prompt_cache_discount_usd": round(prompt_cache_savings, 6),
            "today_provider_prompt_cache_discount_usd": round(today_prompt_cache_savings, 6),
            "provider_prompt_cache_cached_read_cost_usd": round(prompt_cache_cached_read_cost, 6),
            "today_provider_prompt_cache_cached_read_cost_usd": round(today_prompt_cache_cached_read_cost, 6),
            "provider_prompt_cache_creation_cost_usd": round(prompt_cache_creation_cost, 6),
            "today_provider_prompt_cache_creation_cost_usd": round(today_prompt_cache_creation_cost, 6),
            "provider_prompt_cache_creation_premium_usd": round(prompt_cache_creation_premium, 6),
            "today_provider_prompt_cache_creation_premium_usd": round(today_prompt_cache_creation_premium, 6),
            "provider_prompt_cache_net_discount_usd": round(prompt_cache_net_discount, 6),
            "today_provider_prompt_cache_net_discount_usd": round(today_prompt_cache_net_discount, 6),
            "thinking_output_tokens": thinking_output_tokens,
            "today_thinking_output_tokens": today_thinking_output_tokens,
            "thinking_cost_usd": round(thinking_cost, 6),
            "today_thinking_cost_usd": round(today_thinking_cost, 6),
            "codex_app_total_events": codex_app_total_events,
            "codex_app_today_events": codex_app_today_events,
            "codex_app_sessions": codex_app_sessions,
            "codex_app_turns": codex_app_turns,
            "codex_app_today_turns": codex_app_today_turns,
            "codex_app_last_event_at": codex_app_last_event_at,
            "codex_app_input_text_chars": codex_app_input_text_chars,
            "codex_app_input_tokens_est": codex_input_tokens_est,
            "codex_app_output_tokens_est": codex_output_tokens_est,
            "codex_app_total_tokens_est": codex_input_tokens_est + codex_output_tokens_est,
            "codex_app_cost_est_usd": round(codex_cost_est, 6),
            "codex_app_cache_savings_usd": round(codex_cache_savings, 6),
            "today_codex_app_input_tokens_est": today_codex_input_tokens_est,
            "today_codex_app_output_tokens_est": today_codex_output_tokens_est,
            "today_codex_app_total_tokens_est": today_codex_input_tokens_est + today_codex_output_tokens_est,
            "today_codex_app_cost_est_usd": round(today_codex_cost_est, 6),
            "today_codex_app_cache_savings_usd": round(today_codex_cache_savings, 6),
            "codex_app_cost_basis": CODEX_APP_COST_BASIS,
            "codex_app_model": CODEX_APP_MODEL,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
            "codex_app_avg_latency_ms": round(codex_app_avg_latency),
            "routing_experiment_samples": routing_experiment_report_summary["sample_count"],
            "routing_experiment_compared_samples": routing_experiment_report_summary["comparison_count"],
            "routing_experiment_avg_similarity": routing_experiment_report_summary["avg_similarity"],
            "routing_experiment_pass_rate": routing_experiment_report_summary["pass_rate"],
            "routing_experiment_cost_delta_usd": routing_experiment_report_summary["cost_delta_usd"],
            "routing_experiment_avg_latency_delta_ms": routing_experiment_report_summary["avg_latency_delta_ms"],
            "routing_experiment_daily_budget_exhausted": routing_experiment_report["policy"]["daily_budget_exhausted"],
            "routing_experiment_feedback_status_counts": routing_experiment_report_summary["feedback_status_counts"],
            "routing_experiment_sample_mode_counts": routing_experiment_report_summary["sample_mode_counts"],
            "routing_experiment_decisions": routing_experiment_report_summary["decision_count"],
            "routing_experiment_decision_status_counts": routing_experiment_report_summary["decision_status_counts"],
            "routing_experiment_applied_routed_down_samples": routing_experiment_report_summary["applied_routed_down_samples"],
            "routing_experiment_shadow_candidate_pass_through_samples": routing_experiment_report_summary["shadow_candidate_pass_through_samples"],
            "routing_experiment_promotion_verdict_counts": routing_experiment_report_summary["promotion_verdict_counts"],
            "routing_experiment_promotion_reason_counts": routing_experiment_report_summary["promotion_reason_counts"],
            "routing_experiment_promotion_ready_candidates": routing_experiment_report_summary["promotion_ready_candidates"],
            "activation_burndown": activation_burndown.get("summary", {}),
            "closed_loop_activation": closed_loop_activation.get("summary", {}),
            "activation_successor_burndown": (activation_burndown.get("successor_burndown") or {}).get("summary", {}),
            "activation_preview_agreement_burndown": (
                activation_burndown.get("activation_preview_agreement_burndown") or {}
            ).get("summary", {}),
            "savings_loop_bottlenecks": savings_loop_bottlenecks.get("summary", {}),
            "activation_successor_queue_health": activation_successor_queue_health.get("summary", {}),
            "managed_preview_coverage": (activation_burndown.get("managed_preview_coverage") or {}).get("summary", {}),
            "managed_mode": managed_mode,
        },
        "savings_loop_bottlenecks": savings_loop_bottlenecks,
        "activation_burndown": activation_burndown,
        "closed_loop_activation": closed_loop_activation,
        "activation_successor_queue_health": activation_successor_queue_health,
        "managed_preview_coverage": activation_burndown.get("managed_preview_coverage"),
        "recent": recent,
        "routing_breakdown": routing_breakdown,
        "category_breakdown": category_breakdown,
        "cache_decision_breakdown": cache_decision_breakdown,
        "today_cache_decision_breakdown": today_cache_decision_breakdown,
        "crunch_rule_savings_breakdown": crunch_rule_savings_breakdown,
        "today_crunch_rule_savings_breakdown": today_crunch_rule_savings_breakdown,
        "crunch_rule_breakdown": crunch_rule_breakdown,
        "today_crunch_rule_breakdown": today_crunch_rule_breakdown,
        "crunch_rule_group_breakdown": crunch_rule_group_breakdown,
        "today_crunch_rule_group_breakdown": today_crunch_rule_group_breakdown,
        "crunch_canary_captured_savings": crunch_canary_captured_savings,
        "cache_zero_hit_blocker_ladder": cache_zero_hit_blocker_ladder,
        "cache_effectiveness": cache_effectiveness,
        "cache_replayability": cache_replayability,
        "cache_replay_cohort_ranking": cache_replay_cohort_ranking,
        "cache_replay_confidence": cache_replay_confidence,
        "cache_replay_readiness": cache_replay_readiness,
        "cache_replay_activation_health": cache_replay_activation_health,
        "streaming_cache_hit_recovery": streaming_cache_hit_recovery,
        "old_context_summary_opportunity": old_context_summary_opportunity,
        "active_crunch_rule_coverage": active_crunch_rule_coverage,
        "provider_prompt_cache_accounting": prompt_cache_accounting,
        "sqlite_maintenance": sqlite_maintenance,
        "pattern_decision_breakdown": pattern_decision_breakdown,
        "today_pattern_decision_breakdown": today_pattern_decision_breakdown,
        "error_breakdown": error_breakdown,
        "today_error_breakdown": today_error_breakdown,
        "routing_experiment_summary": routing_experiment_summary,
        "routing_experiment_report": routing_experiment_report,
        "provider_breakdown": provider_breakdown,
        "codex_app_methods": codex_app_methods,
        "codex_app_recent": codex_app_recent,
    }


async def stats_weekly(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
    generated_at = utc_now()
    day_keys = _utc_day_window(7)
    first_day = day_keys[0]
    first_day_start = f"{first_day}T00:00:00+00:00"
    managed_feed_state = _managed_feed_state()
    managed_feed_week = _managed_feed_decision_summary(conn, since=first_day_start, day_field=True)

    def new_day(day: str) -> dict[str, Any]:
        return {
            "day": day,
            "total_calls": 0,
            "successful_calls": 0,
            "errors": 0,
            "cache_hits": 0,
            "avg_latency_ms": None,
            "cost_est_usd": 0.0,
            "cost_baseline_usd": 0.0,
            "savings_usd": 0.0,
            "provider_prompt_cache_discount_usd": 0.0,
            "provider_calls": 0,
            "codex_turns": 0,
            "total_units": 0,
            "provider_tokens": 0,
            "codex_tokens_est": 0,
            "total_tokens": 0,
            "codex_cost_est_usd": 0.0,
            "cost_basis": "provider-reported + codex-estimated-from-chars",
            "_latency_sum": 0,
            "_latency_count": 0,
        }

    days_by_key = {day: new_day(day) for day in day_keys}

    provider_rows = conn.execute("""
        select
            date(created_at) as day,
            count(*) as provider_calls,
            sum(case when status_code = 200 then 1 else 0 end) as successful_calls,
            sum(case when status_code >= 400 then 1 else 0 end) as errors,
            sum(cache_hit) as cache_hits,
            sum(case when latency_ms is not null then latency_ms else 0 end) as latency_sum,
            count(latency_ms) as latency_count,
            round(sum(coalesce(cost_est_usd, 0)), 6) as cost_est_usd,
            round(sum(coalesce(cost_baseline_usd, 0)), 6) as cost_baseline_usd,
            sum(
                coalesce(actual_input_tokens, input_tokens_est, 0)
                + coalesce(cache_creation_input_tokens, 0)
                + coalesce(cache_read_input_tokens, 0)
                + coalesce(actual_output_tokens, output_tokens_est, 0)
            ) as provider_tokens
        from calls
        where created_at >= ?
        group by date(created_at)
        order by day asc
    """, (first_day_start,)).fetchall()
    for raw in provider_rows:
        r = dict(raw)
        day = str(r.get("day") or "")
        row = days_by_key.get(day)
        if row is None:
            continue
        provider_calls = _as_int(r.get("provider_calls"))
        row["provider_calls"] += provider_calls
        row["total_calls"] += provider_calls
        row["total_units"] += provider_calls
        row["successful_calls"] += _as_int(r.get("successful_calls"))
        row["errors"] += _as_int(r.get("errors"))
        row["cache_hits"] += _as_int(r.get("cache_hits"))
        row["_latency_sum"] += _as_int(r.get("latency_sum"))
        row["_latency_count"] += _as_int(r.get("latency_count"))
        row["cost_est_usd"] += _as_float(r.get("cost_est_usd"))
        row["cost_baseline_usd"] += _as_float(r.get("cost_baseline_usd"))
        row["provider_tokens"] += _as_int(r.get("provider_tokens"))

    weekly_provider_accounting_rows = conn.execute("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, stream, cache_hit, status_code,
               latency_ms, input_tokens_est, output_tokens_est,
               actual_input_tokens, actual_output_tokens, cost_est_usd,
               cost_baseline_usd, crunch_json, routing_json, cache_json, error,
               request_json, response_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens, retry_count,
               thinking_output_tokens,
               (date(created_at) = date('now')) as is_today
        from calls
        where created_at >= ?
    """, (first_day_start,)).fetchall()
    for raw in weekly_provider_accounting_rows:
        r = dict(raw)
        day = str(r.get("created_at") or "")[:10]
        row = days_by_key.get(day)
        if row is None:
            continue
        unit = _provider_accounting_unit(r)
        row["savings_usd"] += (
            _as_float(unit.get("routing_savings_usd"))
            + _as_float(unit.get("crunch_savings_usd"))
            + _as_float(unit.get("cache_savings_usd"))
        )
        row["provider_prompt_cache_discount_usd"] += _as_float(unit.get("provider_prompt_cache_discount_usd"))

    codex_rows = conn.execute("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.input_text_chars,
               s.cache_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms
        from codex_app_events s
        where s.direction = 'client_to_server'
          and s.method = 'turn/start'
          and s.created_at >= ?
        order by s.created_at asc
    """, (first_day_start,)).fetchall()
    for raw in codex_rows:
        r = dict(raw)
        day = str(r.get("created_at") or "")[:10]
        row = days_by_key.get(day)
        if row is None:
            continue
        cache = _json_obj(r.get("cache_json"))
        estimates = _codex_estimates_with_cache(
            r.get("input_text_chars"),
            r.get("response_result_chars"),
            cache,
        )
        row["codex_turns"] += 1
        row["total_calls"] += 1
        row["total_units"] += 1
        if r.get("response_event_id") is not None and r.get("response_error_code") is None:
            row["successful_calls"] += 1
        if r.get("response_error_code") is not None:
            row["errors"] += 1
        if cache.get("status") == "hit":
            row["cache_hits"] += 1
        latency = r.get("response_latency_ms")
        if latency is not None:
            row["_latency_sum"] += _as_int(latency)
            row["_latency_count"] += 1
        cost = _as_float(estimates.get("cost_est_usd"))
        baseline = _as_float(estimates.get("baseline_cost_est_usd"))
        row["cost_est_usd"] += cost
        row["cost_baseline_usd"] += baseline
        row["savings_usd"] += max(baseline - cost, 0.0)
        row["codex_cost_est_usd"] += cost
        row["codex_tokens_est"] += _as_int(estimates.get("total_tokens_est"))

    total_latency_sum = 0
    total_latency_count = 0
    days = []
    for day in day_keys:
        row = days_by_key[day]
        managed_day = managed_feed_week["by_day"].get(day, {})
        row["managed_feed"] = {
            "schema": "tokenclaw.managed_feed_window_summary.v1",
            "window": "day",
            "window_start": f"{day}T00:00:00+00:00",
            "total_calls": row["provider_calls"],
            "policy_decision_calls": managed_day.get("policy_decision_calls")
            or {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": 0},
            "backing_counts": managed_day.get("backing_counts")
            or {
                "local-manual": 0,
                "managed-recommended": 0,
                "managed-enforced": 0,
                "off/pass-through": 0,
            },
            "privacy": _metadata_only_privacy(),
        }
        row["total_tokens"] = row["provider_tokens"] + row["codex_tokens_est"]
        if row["_latency_count"]:
            row["avg_latency_ms"] = round(row["_latency_sum"] / row["_latency_count"])
        total_latency_sum += _as_int(row.get("_latency_sum"))
        total_latency_count += _as_int(row.get("_latency_count"))
        row["cost_est_usd"] = round(row["cost_est_usd"], 6)
        row["cost_baseline_usd"] = round(row["cost_baseline_usd"], 6)
        row["codex_cost_est_usd"] = round(row["codex_cost_est_usd"], 6)
        row["savings_usd"] = round(row["savings_usd"], 6)
        row["provider_prompt_cache_discount_usd"] = round(row["provider_prompt_cache_discount_usd"], 6)
        row.pop("_latency_sum", None)
        row.pop("_latency_count", None)
        days.append(row)

    totals = {
        "day": "Total",
        "total_calls": sum(r["total_calls"] for r in days),
        "successful_calls": sum(r["successful_calls"] or 0 for r in days),
        "errors": sum(r["errors"] or 0 for r in days),
        "cache_hits": sum(r["cache_hits"] or 0 for r in days),
        "avg_latency_ms": round(total_latency_sum / total_latency_count) if total_latency_count else None,
        "cost_est_usd": round(sum(r["cost_est_usd"] or 0 for r in days), 6),
        "cost_baseline_usd": round(sum(r["cost_baseline_usd"] or 0 for r in days), 6),
        "savings_usd": round(sum(r["savings_usd"] for r in days), 6),
        "provider_prompt_cache_discount_usd": round(sum(r["provider_prompt_cache_discount_usd"] for r in days), 6),
        "provider_calls": sum(r["provider_calls"] for r in days),
        "codex_turns": sum(r["codex_turns"] for r in days),
        "total_units": sum(r["total_units"] for r in days),
        "provider_tokens": sum(r["provider_tokens"] for r in days),
        "codex_tokens_est": sum(r["codex_tokens_est"] for r in days),
        "total_tokens": sum(r["total_tokens"] for r in days),
        "codex_cost_est_usd": round(sum(r["codex_cost_est_usd"] or 0 for r in days), 6),
        "cost_basis": "provider-reported + codex-estimated-from-chars",
        "managed_feed": {
            "schema": "tokenclaw.managed_feed_window_summary.v1",
            "window": "last_7_days",
            "window_start": first_day_start,
            "total_calls": managed_feed_week["total_calls"],
            "policy_decision_calls": managed_feed_week["policy_decision_calls"],
            "backing_counts": managed_feed_week["backing_counts"],
            "privacy": _metadata_only_privacy(),
        },
    }
    return {
        "generated_at": generated_at,
        "schema": "tokenclaw.weekly_activity.v1",
        "source_surfaces": ["anthropic_messages", "openai_responses", "openai_chat", CODEX_APP_SOURCE_SURFACE],
        "cost_basis": "provider-reported + codex-estimated-from-chars",
        "managed_feed": {
            "schema": "tokenclaw.managed_feed_dashboard_summary.v1",
            "state": managed_feed_state,
            "last_7_days": totals["managed_feed"],
            "privacy": _metadata_only_privacy(),
        },
        "days": days,
        "totals": totals,
    }


async def stats_sessions(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
    sessions_by_key: dict[str, dict[str, Any]] = {}

    def session_bucket(session_key: Any, *, basis: str, source_surface: str, app_family: str, units: int = 1) -> dict[str, Any]:
        key = str(session_key or "unknown")
        bucket = sessions_by_key.get(key)
        if bucket is None:
            bucket = {
                "sid": key[:8],
                "session_id": key,
                "session_key_basis": basis,
                "source_surface": source_surface,
                "app_family": app_family,
                "calls": 0,
                "turns": 0,
                "provider_calls": 0,
                "codex_turns": 0,
                "cost_usd": 0.0,
                "tool_result": 0,
                "tool_heavy": 0,
                "short_completion": 0,
                "code_gen": 0,
                "chat": 0,
                "other": 0,
                "codex_input_text_chars": 0,
                "codex_result_chars": 0,
                "codex_input_tokens_est": 0,
                "codex_output_tokens_est": 0,
                "codex_total_tokens_est": 0,
                "codex_cost_est_usd": 0.0,
                "codex_baseline_cost_est_usd": 0.0,
                "codex_hard_floor_usd": 0.0,
                "codex_exact_cache_savings_usd": 0.0,
                "codex_routed_turns": 0,
                "codex_crunched_turns": 0,
                "codex_cache_hits": 0,
                "codex_optimized_turns": 0,
                "codex_errors": 0,
                "codex_cost_basis": CODEX_APP_COST_BASIS,
                "codex_app_model": CODEX_APP_MODEL,
                "codex_workflow_grouping": None,
                "_source_surface_counts": {},
                "_app_family_counts": {},
                "_codex_method_counts": {},
                "_codex_phase_counts": {},
                "_codex_original_key_basis_counts": {},
                "_codex_original_keys": set(),
            }
            sessions_by_key[key] = bucket
        bucket["_source_surface_counts"][source_surface] = bucket["_source_surface_counts"].get(source_surface, 0) + int(units)
        bucket["_app_family_counts"][app_family] = bucket["_app_family_counts"].get(app_family, 0) + int(units)
        return bucket

    provider_rows = conn.execute("""
        SELECT session_id,
            coalesce(provider, 'anthropic') as provider,
            requested_model,
            path,
            COUNT(*) as calls,
            ROUND(SUM(cost_est_usd),6) as cost_usd,
            SUM(CASE WHEN category='tool-result' THEN 1 ELSE 0 END) as tool_result,
            SUM(CASE WHEN category='tool-heavy' THEN 1 ELSE 0 END) as tool_heavy,
            SUM(CASE WHEN category='short-completion' THEN 1 ELSE 0 END) as short_completion,
            SUM(CASE WHEN category='code-gen' THEN 1 ELSE 0 END) as code_gen,
            SUM(CASE WHEN category='chat' THEN 1 ELSE 0 END) as chat,
            SUM(CASE WHEN category IS NULL OR category NOT IN ('tool-result','tool-heavy','short-completion','code-gen','chat') THEN 1 ELSE 0 END) as other
        FROM calls
        WHERE DATE(created_at) = DATE('now') AND session_id IS NOT NULL
        GROUP BY session_id, coalesce(provider, 'anthropic'), requested_model, path
    """).fetchall()
    for row in provider_rows:
        source_surface = _source_surface(row["provider"], str(row["path"] or ""))
        app_family = _app_family_for_call(str(row["provider"] or ""), row["requested_model"], str(row["path"] or ""))
        calls = int(row["calls"] or 0)
        bucket = session_bucket(
            row["session_id"],
            basis="session_id",
            source_surface=source_surface,
            app_family=app_family,
            units=calls,
        )
        bucket["calls"] += calls
        bucket["turns"] += calls
        bucket["provider_calls"] += calls
        bucket["cost_usd"] += float(row["cost_usd"] or 0.0)
        for field in ("tool_result", "tool_heavy", "short_completion", "code_gen", "chat", "other"):
            bucket[field] += int(row[field] or 0)

    plateau_rows = conn.execute("""
        SELECT session_id,
               created_at,
               CAST(coalesce(
                   json_extract(routing_json, '$.text_chars'),
                   coalesce(actual_input_tokens, input_tokens_est, 0) * 4,
                   0
               ) AS INTEGER) as text_chars,
               coalesce(provider, 'anthropic') as provider,
               path,
               coalesce(routed_model, requested_model) as model,
               coalesce(cost_est_usd, 0) as cost_usd,
               coalesce(cache_read_input_tokens, 0) as cache_read_tokens,
               coalesce(json_extract(crunch_json, '$.saved_chars'), 0) as crunch_saved_chars
        FROM calls
        WHERE DATE(created_at) = DATE('now') AND session_id IS NOT NULL
        ORDER BY session_id, created_at
    """).fetchall()
    plateau_by_session: dict[str, dict[str, Any]] = {}
    prev_by_session: dict[str, int] = {}
    min_plateau_chars = 8_000
    max_plateau_delta_ratio = 0.03
    flagged_plateau_pairs = 50

    def median_int(values: list[int]) -> int:
        if not values:
            return 0
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return sorted_values[mid]
        return int(round((sorted_values[mid - 1] + sorted_values[mid]) / 2))

    def percentile_int(values: list[int], percentile: float) -> int:
        if not values:
            return 0
        sorted_values = sorted(values)
        idx = min(len(sorted_values) - 1, math.ceil((len(sorted_values) - 1) * percentile))
        return sorted_values[idx]

    def plateau_bucket(session_key: Any, *, basis: str, source_surface: str, app_family: str) -> dict[str, Any]:
        key = str(session_key or "unknown")
        bucket = plateau_by_session.setdefault(
            key,
            {
                "session_id": key,
                "sid": key[:8],
                "session_key_basis": basis,
                "source_surface": source_surface,
                "app_family": app_family,
                "calls": 0,
                "cost_usd": 0.0,
                "plateau_pairs": 0,
                "large_text_values": [],
                "cache_read_savings_usd": 0.0,
                "crunch_saved_chars": 0,
                "_source_surface_counts": {},
                "_app_family_counts": {},
            },
        )
        bucket["_source_surface_counts"][source_surface] = bucket["_source_surface_counts"].get(source_surface, 0) + 1
        bucket["_app_family_counts"][app_family] = bucket["_app_family_counts"].get(app_family, 0) + 1
        return bucket

    def add_plateau_observation(
        session_key: Any,
        *,
        basis: str,
        source_surface: str,
        app_family: str,
        text_chars: int,
        cost_usd: float,
        cache_read_savings_usd: float = 0.0,
        crunch_saved_chars: int = 0,
    ) -> None:
        key = str(session_key or "unknown")
        bucket = plateau_bucket(key, basis=basis, source_surface=source_surface, app_family=app_family)
        bucket["calls"] += 1
        bucket["cost_usd"] += float(cost_usd or 0.0)
        if text_chars >= min_plateau_chars:
            bucket["large_text_values"].append(text_chars)
        prev_text = prev_by_session.get(key)
        if (
            prev_text is not None
            and prev_text >= min_plateau_chars
            and text_chars >= min_plateau_chars
            and abs(text_chars - prev_text) / max(prev_text, 1) <= max_plateau_delta_ratio
        ):
            bucket["plateau_pairs"] += 1
        prev_by_session[key] = text_chars
        bucket["cache_read_savings_usd"] += float(cache_read_savings_usd or 0.0)
        bucket["crunch_saved_chars"] += int(crunch_saved_chars or 0)

    for row in plateau_rows:
        sid = row["session_id"]
        text_chars = int(row["text_chars"] or 0)
        read_tokens = int(row["cache_read_tokens"] or 0)
        cache_read_savings = 0.0
        if read_tokens:
            provider = str(row["provider"] or "anthropic").lower()
            full_read_cost = estimate_cost(row["model"], read_tokens, 0, provider=provider) or 0.0
            cached_read_input_tokens = read_tokens if provider == "openai" else 0
            cached_read_cost = estimate_cost(
                row["model"],
                cached_read_input_tokens,
                0,
                cache_read=read_tokens,
                provider=provider,
            ) or 0.0
            cache_read_savings = max(full_read_cost - cached_read_cost, 0.0)
        source_surface = _source_surface(row["provider"], str(row["path"] or ""))
        app_family = _app_family_for_call(str(row["provider"] or ""), None, str(row["path"] or ""))
        add_plateau_observation(
            sid,
            basis="session_id",
            source_surface=source_surface,
            app_family=app_family,
            text_chars=text_chars,
            cost_usd=float(row["cost_usd"] or 0.0),
            cache_read_savings_usd=cache_read_savings,
            crunch_saved_chars=int(row["crunch_saved_chars"] or 0),
        )

    codex_rows = conn.execute("""
        SELECT s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.method,
               s.message_chars,
               s.params_chars,
               s.input_items,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
               s.cache_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms
        from codex_app_events s
        where s.direction = 'client_to_server'
          and s.method = 'turn/start'
          and date(s.created_at) = date('now')
        order by s.created_at
    """).fetchall()

    codex_dict_rows = [dict(row) for row in codex_rows]
    codex_workflow_groups = _codex_metadata_workflow_groups(codex_dict_rows)

    for r in codex_dict_rows:
        raw_key, raw_basis = _codex_original_session_key(r)
        group = codex_workflow_groups.get(str(r.get("start_event_id") or ""))
        if group:
            key, basis = str(group["key"]), str(group["basis"])
        else:
            key, basis = raw_key, raw_basis
        unit = _codex_turn_activity_unit(r)
        input_features = unit["input_features"]
        outcome_features = unit["outcome_features"]
        optimization_features = unit["optimization_features"]
        routing = optimization_features["routing"]
        crunch = optimization_features["crunch"]
        cache = optimization_features["cache"]
        bucket = session_bucket(
            key,
            basis=basis,
            source_surface=CODEX_APP_SOURCE_SURFACE,
            app_family="codex",
        )
        bucket["codex_workflow_grouping"] = {
            "basis": basis,
            "derived_key": key,
            "idle_gap_seconds": group.get("idle_gap_seconds") if group else None,
            "group_start_at": group.get("group_start_at") if group else None,
            "original_key_count": 0,
            "original_key_basis_counts": {},
            "model_state_counts": group.get("model_state_counts", {}) if group else {},
            "raw_keys_included": False,
        }
        bucket["_codex_original_keys"].add(f"{raw_basis}:{raw_key}")
        bucket["_codex_original_key_basis_counts"][raw_basis] = (
            bucket["_codex_original_key_basis_counts"].get(raw_basis, 0) + 1
        )
        bucket["calls"] += 1
        bucket["turns"] += 1
        bucket["codex_turns"] += 1
        bucket["codex_input_text_chars"] += _as_int(r.get("input_text_chars"))
        bucket["codex_result_chars"] += _as_int(r.get("response_result_chars"))
        bucket["codex_input_tokens_est"] += _as_int(input_features.get("input_tokens_est"))
        bucket["codex_output_tokens_est"] += _as_int(outcome_features.get("output_tokens_est"))
        bucket["codex_total_tokens_est"] += _as_int(outcome_features.get("total_tokens_est"))
        cost = _as_float(outcome_features.get("cost_est_usd"))
        baseline = _as_float(outcome_features.get("cost_baseline_usd"))
        hard_floor = _as_float(outcome_features.get("hard_floor_usd"))
        cache_savings = _as_float(outcome_features.get("cache_savings_usd"))
        bucket["cost_usd"] += cost
        bucket["codex_cost_est_usd"] += cost
        bucket["codex_baseline_cost_est_usd"] += baseline
        bucket["codex_hard_floor_usd"] += hard_floor
        bucket["codex_exact_cache_savings_usd"] += cache_savings if cache.get("status") == "hit" else 0.0
        bucket["_codex_method_counts"][str(r.get("method") or "unknown")] = (
            bucket["_codex_method_counts"].get(str(r.get("method") or "unknown"), 0) + 1
        )
        phase = _codex_phase_from_decision_metadata(routing, crunch, cache)
        bucket["_codex_phase_counts"][phase] = bucket["_codex_phase_counts"].get(phase, 0) + 1
        optimized = False
        if routing.get("applied"):
            bucket["codex_routed_turns"] += 1
            optimized = True
        if crunch.get("changed") or crunch.get("applied"):
            bucket["codex_crunched_turns"] += 1
            optimized = True
        if cache.get("status") == "hit":
            bucket["codex_cache_hits"] += 1
            optimized = True
        if optimized:
            bucket["codex_optimized_turns"] += 1
        if r.get("response_error_code") is not None:
            bucket["codex_errors"] += 1
        add_plateau_observation(
            key,
            basis=basis,
            source_surface=CODEX_APP_SOURCE_SURFACE,
            app_family="codex",
            text_chars=_as_int(r.get("input_text_chars")),
            cost_usd=cost,
            cache_read_savings_usd=cache_savings if cache.get("status") == "hit" else 0.0,
            crunch_saved_chars=_as_int(crunch.get("saved_chars")),
        )

    all_plateau_metrics = []
    for bucket in plateau_by_session.values():
        large_text_values = bucket.pop("large_text_values")
        bucket["median_text_chars"] = median_int(large_text_values)
        bucket["p90_text_chars"] = percentile_int(large_text_values, 0.9)
        bucket["cost_usd"] = round(float(bucket["cost_usd"]), 6)
        bucket["cache_read_savings_usd"] = round(float(bucket["cache_read_savings_usd"]), 6)
        bucket["flagged"] = int(bucket["plateau_pairs"]) > flagged_plateau_pairs
        source_counts = bucket.pop("_source_surface_counts", {})
        app_counts = bucket.pop("_app_family_counts", {})
        bucket["source_surfaces"] = [
            {"source_surface": source, "units": count}
            for source, count in sorted(source_counts.items())
        ]
        bucket["source_surface"] = next(iter(source_counts), "unknown") if len(source_counts) == 1 else "mixed"
        bucket["app_family"] = next(iter(app_counts), "unknown") if len(app_counts) == 1 else "mixed"
        all_plateau_metrics.append(bucket)
    context_plateaus = [
        bucket for bucket in all_plateau_metrics
        if bucket["plateau_pairs"] > 0
    ]
    context_plateaus.sort(key=lambda r: (r["flagged"], r["plateau_pairs"], r["cost_usd"]), reverse=True)
    context_plateaus = context_plateaus[:20]
    plateau_metrics_by_session = {
        row["session_id"]: row
        for row in all_plateau_metrics
    }
    sessions = list(sessions_by_key.values())
    session_ids = [row["session_id"] for row in sessions]
    thinking_by_session: dict[str, dict[str, float | int]] = {
        sid: {"thinking_tokens": 0, "thinking_cost_usd": 0.0}
        for sid in session_ids
    }
    prompt_cache_by_session: dict[str, dict[str, float | int]] = {
        sid: {
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_cost_usd": 0.0,
            "cache_read_savings_usd": 0.0,
        }
        for sid in session_ids
    }
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        thinking_rows = conn.execute(f"""
            SELECT session_id,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model, requested_model) as model,
                   SUM(coalesce(thinking_output_tokens, 0)) as thinking_tokens
            FROM calls
            WHERE DATE(created_at) = DATE('now')
              AND session_id IN ({placeholders})
              AND coalesce(thinking_output_tokens, 0) > 0
            GROUP BY session_id, coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
        """, tuple(session_ids)).fetchall()
        for row in thinking_rows:
            sid = row["session_id"]
            tokens = int(row["thinking_tokens"] or 0)
            thinking_by_session[sid]["thinking_tokens"] = int(thinking_by_session[sid]["thinking_tokens"]) + tokens
            thinking_by_session[sid]["thinking_cost_usd"] = float(thinking_by_session[sid]["thinking_cost_usd"]) + (
                estimate_cost(row["model"], 0, tokens, provider=row["provider"]) or 0.0
            )
        prompt_cache_rows = conn.execute(f"""
            SELECT session_id,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model, requested_model) as model,
                   SUM(coalesce(cache_creation_input_tokens, 0)) as cache_creation_tokens,
                   SUM(coalesce(cache_read_input_tokens, 0)) as cache_read_tokens
            FROM calls
            WHERE DATE(created_at) = DATE('now')
              AND session_id IN ({placeholders})
              AND (
                  coalesce(cache_creation_input_tokens, 0) > 0
                  OR coalesce(cache_read_input_tokens, 0) > 0
              )
            GROUP BY session_id, coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
        """, tuple(session_ids)).fetchall()
        for row in prompt_cache_rows:
            sid = row["session_id"]
            creation_tokens = int(row["cache_creation_tokens"] or 0)
            read_tokens = int(row["cache_read_tokens"] or 0)
            bucket = prompt_cache_by_session[sid]
            bucket["cache_creation_tokens"] = int(bucket["cache_creation_tokens"]) + creation_tokens
            bucket["cache_read_tokens"] = int(bucket["cache_read_tokens"]) + read_tokens

            creation_cost = estimate_cost(
                row["model"],
                0,
                0,
                cache_creation=creation_tokens,
                provider=row["provider"],
            ) or 0.0
            provider = str(row["provider"]).lower()
            full_read_cost = estimate_cost(row["model"], read_tokens, 0, provider=provider) or 0.0
            cached_read_input_tokens = read_tokens if provider == "openai" else 0
            cached_read_cost = estimate_cost(
                row["model"],
                cached_read_input_tokens,
                0,
                cache_read=read_tokens,
                provider=provider,
            ) or 0.0
            bucket["cache_creation_cost_usd"] = float(bucket["cache_creation_cost_usd"]) + creation_cost
            bucket["cache_read_savings_usd"] = float(bucket["cache_read_savings_usd"]) + max(
                full_read_cost - cached_read_cost,
                0.0,
            )
    for row in sessions:
        thinking = thinking_by_session.get(row["session_id"], {})
        row["thinking_tokens"] = int(thinking.get("thinking_tokens", 0) or 0)
        row["thinking_cost_usd"] = round(float(thinking.get("thinking_cost_usd", 0.0) or 0.0), 6)
        prompt_cache = prompt_cache_by_session.get(row["session_id"], {})
        creation_tokens = int(prompt_cache.get("cache_creation_tokens", 0) or 0)
        read_tokens = int(prompt_cache.get("cache_read_tokens", 0) or 0)
        creation_cost = float(prompt_cache.get("cache_creation_cost_usd", 0.0) or 0.0)
        read_savings = float(prompt_cache.get("cache_read_savings_usd", 0.0) or 0.0)
        row["cache_creation_tokens"] = creation_tokens
        row["cache_read_tokens"] = read_tokens
        row["cache_write_read_token_ratio"] = round(creation_tokens / read_tokens, 3) if read_tokens else None
        row["cache_creation_cost_usd"] = round(creation_cost, 6)
        row["cache_read_savings_usd"] = round(read_savings, 6)
        row["cache_warmup_payback_ratio"] = round(creation_cost / read_savings, 3) if read_savings else None
        plateau = plateau_metrics_by_session.get(row["session_id"], {})
        row["plateau_pairs"] = int(plateau.get("plateau_pairs", 0) or 0)
        row["median_text_chars"] = int(plateau.get("median_text_chars", 0) or 0)
        row["p90_text_chars"] = int(plateau.get("p90_text_chars", 0) or 0)
        source_counts = row.pop("_source_surface_counts", {})
        app_counts = row.pop("_app_family_counts", {})
        method_counts = row.pop("_codex_method_counts", {})
        phase_counts = row.pop("_codex_phase_counts", {})
        original_basis_counts = row.pop("_codex_original_key_basis_counts", {})
        original_keys = row.pop("_codex_original_keys", set())
        grouping = row.get("codex_workflow_grouping")
        if isinstance(grouping, dict):
            grouping["original_key_count"] = len(original_keys)
            grouping["original_key_basis_counts"] = dict(original_basis_counts)
        row["source_surfaces"] = [
            {"source_surface": source, "units": count}
            for source, count in sorted(source_counts.items())
        ]
        row["source_surface"] = next(iter(source_counts), "unknown") if len(source_counts) == 1 else "mixed"
        row["app_family"] = next(iter(app_counts), "unknown") if len(app_counts) == 1 else "mixed"
        row["codex_method_counts"] = [
            {"method": method, "turns": count}
            for method, count in sorted(method_counts.items())
        ]
        row["codex_workflow_phase_counts"] = [
            {"phase": phase, "turns": count}
            for phase, count in sorted(phase_counts.items())
        ]
        row["cost_usd"] = round(float(row["cost_usd"]), 6)
        for money_field in (
            "codex_cost_est_usd",
            "codex_baseline_cost_est_usd",
            "codex_hard_floor_usd",
            "codex_exact_cache_savings_usd",
        ):
            row[money_field] = round(float(row[money_field]), 6)
    sessions.sort(key=lambda row: (row["cost_usd"], row["calls"], row["codex_turns"]), reverse=True)
    sessions = sessions[:20]
    return {
        "sessions": sessions,
        "context_plateaus": context_plateaus,
        "context_plateau_policy": {
            "min_text_chars": min_plateau_chars,
            "max_delta_ratio": max_plateau_delta_ratio,
            "flagged_plateau_pairs": flagged_plateau_pairs,
        },
    }


_COORDINATOR_STALE_OR_MISSING_EVIDENCE = {
    "missing-evidence",
    "missing-lifecycle-evidence",
    "missing-canary-evidence",
    "missing-dependency-evidence",
    "missing-dependency-freshness-evidence",
    "stale-evidence",
    "stale-lifecycle-evidence",
    "stale-canary-evidence",
    "aggregate-only-evidence",
    "insufficient-evidence",
}
_COORDINATOR_SAFETY_REASONS = {
    "rollback",
    "rollback-required",
    "safety-stop",
    "safety-stop-tripped",
    "safety-stopped",
    "safety-regression",
    "quality-regression",
    "safety-stop-priority",
}


def _increment_count(counts: dict[str, int], key: Any, amount: int = 1) -> None:
    label = public_label(key, "unknown")
    counts[label] = counts.get(label, 0) + amount


def _increment_tuple_count(counts: dict[tuple[str, ...], int], key: tuple[Any, ...], amount: int = 1) -> None:
    labels = tuple(public_label(item, "unknown") for item in key)
    counts[labels] = counts.get(labels, 0) + amount


def _tuple_breakdown(counts: dict[tuple[str, ...], int], names: tuple[str, ...], *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        row = {name: value for name, value in zip(names, key)}
        row["count"] = count
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _coordinator_report_privacy() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "read_only": True,
        "raw_prompts_included": False,
        "raw_responses_included": False,
        "provider_bodies_included": False,
        "raw_request_bodies_included": False,
        "terminal_lines_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "secrets_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "provider_body_changed": False,
        "policy_files_changed": False,
    }


def _coordinator_meta_from_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    decision: dict[str, Any] = {}
    enforcement: dict[str, Any] = {}
    for key in ("routing_json", "crunch_json", "cache_json"):
        meta = _json_obj(row.get(key))
        if not decision and isinstance(meta.get("optimization_coordinator"), dict):
            decision = meta["optimization_coordinator"]
        if not enforcement and isinstance(meta.get("optimization_coordinator_enforcement"), dict):
            enforcement = meta["optimization_coordinator_enforcement"]
    return decision, enforcement


def _coordinator_state(
    *,
    enforcement_enabled: bool,
    runtime_decision_count: int,
    rows_with_entries: int,
    selected_count: int,
    conflict_count: int,
    safety_stop_count: int,
    missing_metadata_count: int,
) -> str:
    if safety_stop_count > 0:
        return "safety-stop"
    if conflict_count > 0:
        return "conflict-observed"
    if selected_count > 0:
        return "active-selection"
    if runtime_decision_count == 0 and rows_with_entries > 0:
        return "dry-run-only"
    if not enforcement_enabled and runtime_decision_count == 0:
        return "disabled"
    if missing_metadata_count > 0:
        return "missing-metadata"
    return "no-coordinator-metadata"


async def stats_optimization_coordinator_dashboard(store_obj: Any, *, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.optimization_coordinator_dry_run import build_optimization_coordinator_dry_run
    from tokenclaw.optimization_coordinator_enforcement import optimization_coordinator_enforcement_enabled

    capped = max(1, min(int(limit or 1), 10_000))
    dry_run = build_optimization_coordinator_dry_run(store_obj, limit=capped, examples=0)
    rows = []
    if hasattr(store_obj, "optimization_action_ledger_rows"):
        rows = [dict(row) for row in store_obj.optimization_action_ledger_rows(limit=capped)]

    selected_counts: dict[str, int] = {}
    holdout_counts: dict[str, int] = {}
    suppressed_counts: dict[str, int] = {}
    reason_counts: dict[tuple[str, str], int] = {}
    conflict_counts: dict[tuple[str, str, str], int] = {}
    dimension_counts: dict[tuple[str, str, str, str, str, str], int] = {}
    status_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    retry_bucket_counts: dict[str, int] = {}
    savings_by_family: dict[str, float] = {}
    cost_by_family: dict[str, float] = {}
    safety_stop_count = 0
    stale_or_missing_evidence_count = 0
    selected_count = 0
    conflict_count = 0
    rows_with_errors = 0
    sample_decisions: list[dict[str, Any]] = []

    runtime_decision_count = 0
    runtime_enforcement_count = 0
    for index, row in enumerate(rows):
        decision, enforcement = _coordinator_meta_from_row(row)
        if not decision and not enforcement:
            continue
        if decision:
            runtime_decision_count += 1
        if enforcement:
            runtime_enforcement_count += 1

        selected_family = public_label(
            decision.get("selected_action_family") or decision.get("selected_family") or enforcement.get("selected_family"),
            "none",
        )
        _increment_count(selected_counts, selected_family)
        if selected_family != "none":
            selected_count += 1
            row_savings = max(0.0, _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd")))
            savings_by_family[selected_family] = savings_by_family.get(selected_family, 0.0) + row_savings
            cost_by_family[selected_family] = cost_by_family.get(selected_family, 0.0) + _as_float(row.get("cost_est_usd"))

        canary = decision.get("canary") if isinstance(decision.get("canary"), dict) else {}
        cohort = public_label(canary.get("cohort"), "unknown")
        _increment_count(cohort_counts, cohort)
        if bool(canary.get("holdout")):
            _increment_count(holdout_counts, selected_family)

        status = public_label(enforcement.get("status") or "observed", "observed")
        _increment_count(status_counts, status)
        retry_count = _as_int(row.get("retry_count"))
        retry_bucket = "0" if retry_count <= 0 else "1" if retry_count == 1 else "2" if retry_count == 2 else "gte_3"
        _increment_count(retry_bucket_counts, retry_bucket)
        if _as_int(row.get("status_code")) >= 400 or bool(row.get("error_present")):
            rows_with_errors += 1

        provider = public_label(row.get("provider") or decision.get("provider_family"), "unknown")
        source_surface = public_label(row.get("source_surface") or decision.get("source_surface"), "unknown")
        category = public_label(row.get("category") or decision.get("category"), "unknown")
        phase = public_label(decision.get("phase"), "unknown")
        public_session_bucket = public_label(decision.get("public_session_bucket"), "unknown")
        _increment_tuple_count(
            dimension_counts,
            (provider, source_surface, category, phase, public_session_bucket, selected_family),
        )

        for item in decision.get("suppressed_families") or []:
            if not isinstance(item, dict):
                continue
            family = public_label(item.get("family"), "unknown")
            _increment_count(suppressed_counts, family)
            reasons = item.get("reason_codes") if isinstance(item.get("reason_codes"), list) else []
            if not reasons:
                reasons = ["unknown"]
            for raw_reason in reasons:
                reason = public_label(raw_reason, "unknown")
                _increment_tuple_count(reason_counts, (family, reason))
                if reason == "conflicts-with-selected-family":
                    conflict_count += 1
                    _increment_tuple_count(conflict_counts, (selected_family, family, reason))
                if reason in _COORDINATOR_STALE_OR_MISSING_EVIDENCE:
                    stale_or_missing_evidence_count += 1
                if reason in _COORDINATOR_SAFETY_REASONS:
                    safety_stop_count += 1

        decision_reasons = decision.get("reason_codes") if isinstance(decision.get("reason_codes"), list) else []
        if any(public_label(reason, "unknown") in _COORDINATOR_SAFETY_REASONS for reason in decision_reasons):
            safety_stop_count += 1
        if selected_family in {"rollback", "safety-stop", "safety-stopped"}:
            safety_stop_count += 1

        if len(sample_decisions) < 10:
            sample_decisions.append({
                "example_id": public_id(
                    {
                        "index": index,
                        "created_at": row.get("created_at"),
                        "selected_family": selected_family,
                        "cohort": cohort,
                    },
                    prefix="coordinator-row",
                    fallback="coordinator-row",
                ),
                "created_at": row.get("created_at"),
                "provider": provider,
                "source_surface": source_surface,
                "endpoint": public_label(row.get("endpoint") or decision.get("endpoint"), "unknown"),
                "category": category,
                "workflow_phase": phase,
                "public_session_bucket": public_session_bucket,
                "selected_family": selected_family,
                "cohort": cohort,
                "candidate_count": _as_int(decision.get("candidate_count")),
                "suppressed_family_count": len(decision.get("suppressed_families") or []),
                "status": status,
            })

    rows_with_entries = _as_int(dry_run.get("rows_with_ledger_entries"))
    missing_metadata_count = max(0, rows_with_entries - runtime_decision_count)
    enforcement_enabled = optimization_coordinator_enforcement_enabled()
    state = _coordinator_state(
        enforcement_enabled=enforcement_enabled,
        runtime_decision_count=runtime_decision_count,
        rows_with_entries=rows_with_entries,
        selected_count=selected_count,
        conflict_count=conflict_count,
        safety_stop_count=safety_stop_count,
        missing_metadata_count=missing_metadata_count,
    )
    observed_by_family = [
        {
            "family": family,
            "observed_savings_usd_est": _money(amount),
            "observed_cost_usd_est": _money(cost_by_family.get(family, 0.0)),
        }
        for family, amount in sorted(savings_by_family.items(), key=lambda item: (-item[1], item[0]))
    ]
    runtime_error_rate = rows_with_errors / runtime_decision_count if runtime_decision_count else None
    return {
        "schema": "tokenclaw.optimization_coordinator_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "state": state,
        "summary": {
            "sampled_call_count": len(rows),
            "runtime_decision_count": runtime_decision_count,
            "runtime_enforcement_count": runtime_enforcement_count,
            "rows_with_ledger_entries": rows_with_entries,
            "missing_runtime_metadata_count": missing_metadata_count,
            "selected_count": selected_count,
            "holdout_count": sum(holdout_counts.values()),
            "suppressed_family_count": sum(suppressed_counts.values()),
            "conflict_count": conflict_count,
            "safety_stop_count": safety_stop_count,
            "stale_or_missing_evidence_blocker_count": stale_or_missing_evidence_count,
            "rows_with_errors": rows_with_errors,
            "runtime_error_rate": runtime_error_rate,
            "observed_savings_usd_est": _money(sum(savings_by_family.values())),
            "observed_cost_usd_est": _money(sum(cost_by_family.values())),
        },
        "capabilities": {
            "enforcement_enabled": enforcement_enabled,
            "dry_run_capable": True,
            "runtime_metadata_observed": runtime_decision_count > 0,
            "dry_run_only": runtime_decision_count == 0 and rows_with_entries > 0,
            "missing_metadata": missing_metadata_count > 0,
        },
        "selected_family_counts": _breakdown_from_counts(selected_counts),
        "holdout_family_counts": _breakdown_from_counts(holdout_counts),
        "suppressed_family_counts": _breakdown_from_counts(suppressed_counts),
        "top_suppression_reason_codes": _tuple_breakdown(reason_counts, ("family", "reason"), limit=20),
        "conflict_buckets": _tuple_breakdown(conflict_counts, ("selected_family", "suppressed_family", "reason"), limit=20),
        "dimension_breakdown": _tuple_breakdown(
            dimension_counts,
            ("provider", "source_surface", "category", "workflow_phase", "public_session_bucket", "selected_family"),
            limit=50,
        ),
        "status_counts": _breakdown_from_counts(status_counts),
        "cohort_counts": _breakdown_from_counts(cohort_counts),
        "retry_count_buckets": _breakdown_from_counts(retry_bucket_counts),
        "observed_savings_by_family": observed_by_family,
        "dry_run_summary": {
            "sampled_call_count": dry_run.get("sampled_call_count", 0),
            "rows_with_ledger_entries": rows_with_entries,
            "selected_family_counts": dry_run.get("selected_family_counts", []),
            "suppressed_family_counts": dry_run.get("suppressed_family_counts", []),
            "top_suppression_reason_codes": dry_run.get("top_suppression_reason_codes", []),
            "suppression_opportunity_buckets": dry_run.get("suppression_opportunity_buckets", []),
            "top_suppression_next_action": dry_run.get("top_suppression_next_action"),
            "projected_savings_usd_est": dry_run.get("projected_savings_usd_est", 0.0),
            "projected_savings_by_family": dry_run.get("projected_savings_by_family", []),
        },
        "sample_decisions": sample_decisions,
        "privacy": _coordinator_report_privacy(),
    }

_EVIDENCE_NEXT_ACTION_ENTRY_FIELDS = {
    "rank",
    "lever",
    "local_action_family",
    "current_status",
    "state",
    "next_action",
    "blocker_codes",
    "sample_count",
    "applied_count",
    "holdout_count",
    "projected_hits",
    "actual_hits",
    "actual_saved_cost_usd",
    "projected_saved_usd",
    "savings_per_1000_calls_usd",
    "evidence_schema",
    "cohort_bucket",
    "issue_worthy_status",
    "expected_savings_path",
    "legacy_issue_title",
    "requested_model",
    "candidate_target_model",
    "omitted_reason",
    "follow_up_owner",
    "managed_dependency",
    "local_handoff_reason",
    "local_file_backed_representation",
}


def _public_evidence_next_action_entry(entry: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: _copy_policy(value)
        for key, value in entry.items()
        if key in _EVIDENCE_NEXT_ACTION_ENTRY_FIELDS and value not in (None, "", [])
    }
    if not isinstance(public.get("blocker_codes"), list):
        public["blocker_codes"] = []
    public["rank"] = _as_int(public.get("rank"))
    public["sample_count"] = _as_int(public.get("sample_count"))
    public["applied_count"] = _as_int(public.get("applied_count"))
    public["holdout_count"] = _as_int(public.get("holdout_count"))
    public["projected_hits"] = _as_int(public.get("projected_hits"))
    public["actual_hits"] = _as_int(public.get("actual_hits"))
    public["actual_saved_cost_usd"] = round(_as_float(public.get("actual_saved_cost_usd")), 8)
    public["projected_saved_usd"] = round(_as_float(public.get("projected_saved_usd")), 8)
    public["savings_per_1000_calls_usd"] = round(_as_float(public.get("savings_per_1000_calls_usd")), 8)
    issue = entry.get("prior_issue") if isinstance(entry.get("prior_issue"), dict) else None
    if issue:
        public["prior_issue"] = {
            key: issue.get(key)
            for key in ("number", "state", "title", "url")
            if issue.get(key) not in (None, "", [])
        }
    return public


def _empty_evidence_next_actions_payload(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_evidence_to_activation_next_actions.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "summary": {
            "tracked_entry_count": 0,
            "top_lever": None,
            "top_current_status": None,
            "top_next_action": None,
            "top_blocker_codes": [],
            "top_expected_savings_path": None,
            "status_counts": [],
        },
        "source": source,
        "entries": [],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "dashboard_read_only": True,
            "artifact_path_included": False,
        },
    }


async def stats_evidence_to_activation_next_actions(limit: int = 20) -> dict[str, Any]:
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_evidence_next_actions_payload(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _empty_evidence_next_actions_payload(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
        )
    if not isinstance(payload, dict):
        return _empty_evidence_next_actions_payload(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
        )

    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    ledger = evidence.get("evidence_to_activation_next_action_ledger")
    if not isinstance(ledger, dict):
        ledger = payload.get("evidence_to_activation_next_action_ledger")
    stats_summary = evidence.get("stats_summary") if isinstance(evidence.get("stats_summary"), dict) else {}
    if not isinstance(ledger, dict):
        ledger = stats_summary.get("evidence_to_activation_next_action_ledger")
    if not isinstance(ledger, dict):
        return _empty_evidence_next_actions_payload(
            status="no-ledger",
            status_reason="latest research plan does not contain an evidence-to-activation next-action ledger",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )

    capped = max(1, min(int(limit or 20), 100))
    entries = [
        _public_evidence_next_action_entry(entry)
        for entry in ledger.get("entries") or []
        if isinstance(entry, dict)
    ][:capped]
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
    public_summary = {
        key: _copy_policy(summary.get(key))
        for key in (
            "tracked_entry_count",
            "closed_issue_seen_count",
            "top_lever",
            "top_current_status",
            "top_next_action",
            "top_blocker_codes",
            "top_expected_savings_path",
            "status_counts",
            "issue_status_counts",
        )
        if summary.get(key) not in (None, "", [])
    }
    public_summary["tracked_entry_count"] = _as_int(public_summary.get("tracked_entry_count")) or len(entries)
    if not isinstance(public_summary.get("top_blocker_codes"), list):
        public_summary["top_blocker_codes"] = []
    if not isinstance(public_summary.get("status_counts"), list):
        public_summary["status_counts"] = []
    if not isinstance(public_summary.get("issue_status_counts"), list):
        public_summary["issue_status_counts"] = []

    ledger_privacy = ledger.get("privacy") if isinstance(ledger.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.dashboard_evidence_to_activation_next_actions.v1",
        "generated_at": utc_now(),
        "status": "tracked" if entries else "empty",
        "status_reason": "latest research plan ledger loaded" if entries else "latest research plan ledger has no entries",
        "ledger_schema": ledger.get("schema"),
        "ledger_status": ledger.get("status"),
        "summary": public_summary,
        "source": {**source, "plan_generated_at": payload.get("generated_at")},
        "entries": entries,
        "privacy": {
            "metadata_only": ledger_privacy.get("metadata_only", True) is True,
            "aggregate_only": ledger_privacy.get("aggregate_only", True) is True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "dashboard_read_only": True,
            "artifact_path_included": False,
        },
    }


def _local_activation_queue_privacy(source_privacy: dict[str, Any] | None = None) -> dict[str, bool]:
    source_privacy = source_privacy if isinstance(source_privacy, dict) else {}
    return {
        "metadata_only": source_privacy.get("metadata_only", True) is True,
        "aggregate_only": source_privacy.get("aggregate_only", True) is True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
        "dashboard_read_only": True,
        "artifact_path_included": False,
    }


def _managed_preview_coverage_privacy(*, managed_server_calls_made: bool = False) -> dict[str, bool]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": bool(managed_server_calls_made),
        "dashboard_read_only": True,
    }


def _empty_managed_preview_coverage(*, status: str, status_reason: str) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_managed_activation_preview_coverage.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "preview_data_status": status,
        "lookback_limit": MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
        "sample_limit": MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT,
        "summary": {
            "stored_preview_outcome_count": 0,
            "sample_outcome_count": 0,
            "fresh_count": 0,
            "stale_count": 0,
            "missing_preview_decision_count": 0,
            "omission_count": 0,
            "failed_closed_count": 0,
            "agreement_count": 0,
            "disagreement_count": 0,
            "latest_preview_age_hours": None,
            "classification_counts": [],
            "local_action_family_counts": [],
        },
        "family_coverage": [],
        "sample_outcomes": [],
        "privacy": _managed_preview_coverage_privacy(),
    }


def _managed_preview_reason(outcome: dict[str, Any]) -> str:
    for key in ("omitted_reason", "no_op_reason"):
        value = str(outcome.get(key) or "").strip()
        if value:
            return value
    reason_codes = outcome.get("reason_codes")
    if isinstance(reason_codes, list):
        for value in reason_codes:
            text = str(value or "").strip()
            if text:
                return text
    return str(outcome.get("classification") or "unknown").strip() or "unknown"


def _managed_preview_public_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    public = {
        "local_action_family": outcome.get("local_action_family") or "unknown",
        "evidence_schema": outcome.get("evidence_schema"),
        "classification": outcome.get("classification") or "unknown",
        "decision": outcome.get("decision"),
        "decision_status": outcome.get("decision_status"),
        "next_action": outcome.get("next_action"),
        "omitted_reason": outcome.get("omitted_reason"),
        "no_op_reason": outcome.get("no_op_reason"),
        "reason_codes": outcome.get("reason_codes") if isinstance(outcome.get("reason_codes"), list) else [],
        "preview_age_hours": outcome.get("preview_age_hours"),
        "stale": bool(outcome.get("stale")),
        "missing_preview_decision": bool(outcome.get("missing_preview_decision")),
        "failed_closed": bool(outcome.get("failed_closed")),
        "disagrees_with_local_evidence": bool(outcome.get("disagrees_with_local_evidence")),
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_preview_policy_files_written": bool(outcome.get("managed_preview_policy_files_written")),
        "managed_preview_provider_calls_made": bool(outcome.get("managed_preview_provider_calls_made")),
        "privacy": _managed_preview_coverage_privacy(
            managed_server_calls_made=bool(outcome.get("managed_server_calls_made"))
        ),
    }
    return {key: _copy_policy(value) for key, value in public.items() if value not in (None, "", [])}


def _managed_preview_family_row(
    *,
    family: str,
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    ages = []
    for outcome in outcomes:
        reason = _managed_preview_reason(outcome)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        classification = str(outcome.get("classification") or "unknown")
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        if outcome.get("preview_age_hours") is not None:
            ages.append(_as_float(outcome.get("preview_age_hours")))
    agreement_count = sum(
        1
        for outcome in outcomes
        if not bool(outcome.get("stale"))
        and not bool(outcome.get("missing_preview_decision"))
        and not bool(outcome.get("failed_closed"))
        and not bool(outcome.get("disagrees_with_local_evidence"))
    )
    top_reason = _breakdown_from_counts(reason_counts)[0]["value"] if reason_counts else None
    return {
        "local_action_family": family,
        "stored_preview_outcome_count": len(outcomes),
        "fresh_count": agreement_count,
        "stale_count": sum(1 for outcome in outcomes if outcome.get("stale")),
        "missing_preview_decision_count": sum(1 for outcome in outcomes if outcome.get("missing_preview_decision")),
        "omission_count": sum(1 for outcome in outcomes if outcome.get("omitted_reason")),
        "failed_closed_count": sum(1 for outcome in outcomes if outcome.get("failed_closed")),
        "agreement_count": agreement_count,
        "disagreement_count": sum(1 for outcome in outcomes if outcome.get("disagrees_with_local_evidence")),
        "latest_preview_age_hours": min(ages) if ages else None,
        "top_omitted_or_blocker_reason": top_reason,
        "reason_counts": _breakdown_from_counts(reason_counts)[:10],
        "classification_counts": _breakdown_from_counts(classification_counts),
    }


def _managed_preview_data_status(outcomes: list[dict[str, Any]]) -> str:
    if not outcomes:
        return "missing"
    fresh = [
        outcome
        for outcome in outcomes
        if not bool(outcome.get("stale"))
        and not bool(outcome.get("missing_preview_decision"))
        and not bool(outcome.get("failed_closed"))
    ]
    if fresh:
        return "fresh"
    if any(bool(outcome.get("stale")) for outcome in outcomes):
        return "stale"
    return "missing"


def _managed_preview_coverage_for_family(coverage: dict[str, Any] | None, family: str | None) -> dict[str, Any] | None:
    if not isinstance(coverage, dict) or not family:
        return None
    for row in coverage.get("family_coverage") or []:
        if isinstance(row, dict) and str(row.get("local_action_family") or "") == str(family):
            return {
                "schema": "tokenclaw.dashboard_managed_activation_preview_family_coverage.v1",
                "status": coverage.get("status"),
                "preview_data_status": coverage.get("preview_data_status"),
                "stored_preview_outcome_count": row.get("stored_preview_outcome_count", 0),
                "fresh_count": row.get("fresh_count", 0),
                "stale_count": row.get("stale_count", 0),
                "missing_preview_decision_count": row.get("missing_preview_decision_count", 0),
                "omission_count": row.get("omission_count", 0),
                "failed_closed_count": row.get("failed_closed_count", 0),
                "agreement_count": row.get("agreement_count", 0),
                "disagreement_count": row.get("disagreement_count", 0),
                "latest_preview_age_hours": row.get("latest_preview_age_hours"),
                "top_omitted_or_blocker_reason": row.get("top_omitted_or_blocker_reason"),
            }
    return {
        "schema": "tokenclaw.dashboard_managed_activation_preview_family_coverage.v1",
        "status": coverage.get("status"),
        "preview_data_status": coverage.get("preview_data_status"),
        "stored_preview_outcome_count": 0,
        "fresh_count": 0,
        "stale_count": 0,
        "missing_preview_decision_count": 0,
        "omission_count": 0,
        "failed_closed_count": 0,
        "agreement_count": 0,
        "disagreement_count": 0,
        "latest_preview_age_hours": None,
        "top_omitted_or_blocker_reason": None,
    }


def _managed_activation_preview_coverage(store_obj: Any | None) -> dict[str, Any]:
    if store_obj is None:
        return _empty_managed_preview_coverage(
            status="disabled",
            status_reason="local store was not provided for managed preview coverage",
        )
    try:
        from tokenclaw.managed_activation_preview_outcomes import (
            DEFAULT_STALE_AFTER_HOURS,
            build_managed_activation_preview_outcomes_report,
        )

        report = build_managed_activation_preview_outcomes_report(
            store_obj,
            limit=MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
            stale_after_hours=DEFAULT_STALE_AFTER_HOURS,
        )
    except sqlite3.Error:
        return _empty_managed_preview_coverage(
            status="missing",
            status_reason="managed preview outcome table is unavailable",
        )
    outcomes = [row for row in report.get("outcomes") or [] if isinstance(row, dict)]
    if not outcomes:
        return _empty_managed_preview_coverage(
            status="missing",
            status_reason="no managed preview outcome rows have been recorded",
        )
    classification_counts: dict[str, int] = {}
    family_groups: dict[str, list[dict[str, Any]]] = {}
    ages = []
    for outcome in outcomes:
        classification = str(outcome.get("classification") or "unknown")
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        family = str(outcome.get("local_action_family") or "unknown")
        family_groups.setdefault(family, []).append(outcome)
        if outcome.get("preview_age_hours") is not None:
            ages.append(_as_float(outcome.get("preview_age_hours")))
    agreement_count = sum(
        1
        for outcome in outcomes
        if not bool(outcome.get("stale"))
        and not bool(outcome.get("missing_preview_decision"))
        and not bool(outcome.get("failed_closed"))
        and not bool(outcome.get("disagrees_with_local_evidence"))
    )
    managed_calls_made = bool(report.get("managed_server_calls_made"))
    preview_data_status = _managed_preview_data_status(outcomes)
    return {
        "schema": "tokenclaw.dashboard_managed_activation_preview_coverage.v1",
        "generated_at": utc_now(),
        "status": "tracked",
        "status_reason": "bounded local managed preview outcomes loaded",
        "preview_data_status": preview_data_status,
        "lookback_limit": MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
        "sample_limit": MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT,
        "summary": {
            "stored_preview_outcome_count": len(outcomes),
            "sample_outcome_count": min(len(outcomes), MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT),
            "fresh_count": agreement_count,
            "stale_count": sum(1 for outcome in outcomes if outcome.get("stale")),
            "missing_preview_decision_count": sum(1 for outcome in outcomes if outcome.get("missing_preview_decision")),
            "omission_count": sum(1 for outcome in outcomes if outcome.get("omitted_reason")),
            "failed_closed_count": sum(1 for outcome in outcomes if outcome.get("failed_closed")),
            "agreement_count": agreement_count,
            "disagreement_count": sum(1 for outcome in outcomes if outcome.get("disagrees_with_local_evidence")),
            "latest_preview_age_hours": min(ages) if ages else None,
            "classification_counts": _breakdown_from_counts(classification_counts),
            "local_action_family_counts": _breakdown_from_counts({
                family: len(rows) for family, rows in family_groups.items()
            }),
        },
        "family_coverage": [
            _managed_preview_family_row(family=family, outcomes=rows)
            for family, rows in sorted(family_groups.items())
        ],
        "sample_outcomes": [
            _managed_preview_public_outcome(outcome)
            for outcome in outcomes[:MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT]
        ],
        "source_report_schema": report.get("schema"),
        "source_report_status": report.get("status"),
        "privacy": _managed_preview_coverage_privacy(managed_server_calls_made=managed_calls_made),
    }


def _empty_local_activation_queue_payload(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
    managed_preview_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_local_activation_next_action_queue.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "queue_schema": None,
        "queue_status": None,
        "source_schema": None,
        "source": source,
        "summary": {
            "queued_action_count": 0,
            "top_lever": None,
            "top_state": None,
            "top_current_status": None,
            "top_next_action": None,
            "top_unblock_reason": None,
            "top_realized_savings_usd": 0.0,
            "top_projected_savings_usd": 0.0,
            "total_realized_savings_usd": 0.0,
            "total_projected_savings_usd": 0.0,
            "lever_counts": [],
            "status_counts": [],
            "unblock_reason_counts": [],
        },
        "entries": [],
        "successor_burndown": _activation_successor_burndown([]),
        "activation_preview_agreement_burndown": _activation_preview_agreement_burndown({}),
        "managed_preview_coverage": managed_preview_coverage or _empty_managed_preview_coverage(
            status="disabled",
            status_reason="managed preview coverage was not requested",
        ),
        "privacy": _local_activation_queue_privacy(),
    }


def _empty_preview_gated_activation_issue_queue_payload(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_preview_gated_activation_issue_queue.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "queue_schema": None,
        "queue_status": None,
        "source_schema": None,
        "source": source,
        "summary": {
            "successor_decision_count": 0,
            "issue_proposal_count": 0,
            "ready_count": 0,
            "blocked_count": 0,
            "stale_or_no_data_count": 0,
            "suppressed_count": 0,
            "decision_counts": [],
            "issue_status_counts": [],
            "preview_agreement_status_counts": [],
            "issue_queue_status_counts": [],
            "local_action_family_counts": [],
            "top_reason_counts": [],
            "top_ready_issue": None,
            "top_blocked_issue": None,
            "top_stale_or_no_data_issue": None,
        },
        "successor_decisions": [],
        "issue_proposals": [],
        "privacy": _local_activation_queue_privacy(),
    }


def _public_local_activation_queue_summary(summary: dict[str, Any], entry_count: int) -> dict[str, Any]:
    public = {
        key: _copy_policy(summary.get(key))
        for key in (
            "queued_action_count",
            "top_lever",
            "top_state",
            "top_current_status",
            "top_next_action",
            "top_unblock_reason",
            "top_blocking_reason",
            "top_freshness_state",
            "top_savings_per_1000_calls_usd",
            "top_freshness_adjusted_savings_per_1000_calls_usd",
            "top_rank_basis",
            "top_realized_savings_usd",
            "top_projected_savings_usd",
            "total_realized_savings_usd",
            "total_projected_savings_usd",
            "lever_counts",
            "status_counts",
            "unblock_reason_counts",
        )
        if summary.get(key) not in (None, "", [])
    }
    public["queued_action_count"] = _as_int(public.get("queued_action_count")) or entry_count
    public["top_realized_savings_usd"] = round(_as_float(public.get("top_realized_savings_usd")), 8)
    public["top_projected_savings_usd"] = round(_as_float(public.get("top_projected_savings_usd")), 8)
    public["top_savings_per_1000_calls_usd"] = round(_as_float(public.get("top_savings_per_1000_calls_usd")), 8)
    public["top_freshness_adjusted_savings_per_1000_calls_usd"] = round(
        _as_float(public.get("top_freshness_adjusted_savings_per_1000_calls_usd")),
        8,
    )
    public["total_realized_savings_usd"] = round(_as_float(public.get("total_realized_savings_usd")), 8)
    public["total_projected_savings_usd"] = round(_as_float(public.get("total_projected_savings_usd")), 8)
    for key in ("lever_counts", "status_counts", "unblock_reason_counts"):
        if not isinstance(public.get(key), list):
            public[key] = []
    return public


ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES = (
    "source-traffic-acquisition",
    "cache-reobserve",
    "crunch-canary",
)


def _activation_successor_family(row: dict[str, Any]) -> str | None:
    family = str(row.get("local_action_family") or row.get("lever") or "").strip()
    text_parts = [
        family,
        str(row.get("lever") or ""),
        str(row.get("next_action") or ""),
        str(row.get("unblock_reason") or ""),
        str(row.get("blocking_reason") or ""),
        str(row.get("current_status") or ""),
        str(row.get("state") or ""),
        str(row.get("evidence_schema") or ""),
    ]
    text_parts.extend(str(code or "") for code in row.get("blocker_codes") or [])
    text = " ".join(text_parts).lower()
    if family == "source-traffic-acquisition" or "source-traffic" in text or "no-source-traffic-for-request-shape-rollups" in text:
        return "source-traffic-acquisition"
    if family == "cache-reobserve" or "reobserve" in text or "rollback-cache-replay" in text or "cache-replay" in text:
        return "cache-reobserve"
    if family == "crunch-canary" or family == "crunch" or "crunch" in text:
        return "crunch-canary"
    return None


def _activation_status_bucket(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "freshness_state",
            "current_status",
            "state",
            "issue_worthy_status",
            "next_action",
            "unblock_reason",
            "blocking_reason",
            "duplicate_suppression_status",
        )
    ).lower()
    text += " " + " ".join(str(code or "").lower() for code in row.get("blocker_codes") or [])
    if "suppressed" in text or "duplicate" in text:
        return "suppressed"
    if "retired" in text or "superseded" in text:
        return "retired"
    if "stale" in text or "evidence-older-than-max-age" in text:
        return "stale"
    if "rollback" in text:
        return "rollback"
    if _as_int(row.get("applied_count")) > 0 or "applied" in text or "active" in text or "full-rollout" in text:
        return "applied"
    if _as_int(row.get("holdout_count")) > 0 or "holdout" in text:
        return "held-out"
    if "ready" in text or "review" in text:
        return "ready"
    if "no-data" in text or "missing" in text or "no-source-traffic" in text:
        return "missing"
    if "blocked" in text or "keep-blocked" in text:
        return "blocked"
    return "unknown"


def _activation_top_blocker(row: dict[str, Any]) -> str:
    blockers = row.get("blocker_codes") if isinstance(row.get("blocker_codes"), list) else []
    for value in blockers:
        text = str(value or "").strip()
        if text:
            return public_label(text, "unknown")
    for key in ("blocking_reason", "unblock_reason"):
        text = str(row.get(key) or "").strip()
        if text:
            return public_label(text, "unknown")
    return "none"


def _status_count_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    ordered = [
        "missing",
        "stale",
        "ready",
        "applied",
        "held-out",
        "blocked",
        "rollback",
        "retired",
        "suppressed",
        "unknown",
    ]
    rows = [{"value": status, "count": int(counter.get(status, 0))} for status in ordered if counter.get(status, 0)]
    rows.extend(
        {"value": status, "count": int(count)}
        for status, count in sorted(counter.items())
        if status not in ordered
    )
    return rows


def _activation_successor_burndown(entries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {family: [] for family in ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES}
    for row in entries:
        family = _activation_successor_family(row)
        if family is not None:
            grouped[family].append(row)

    rows: list[dict[str, Any]] = []
    for family in ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES:
        family_rows = grouped[family]
        status_counter: Counter[str] = Counter(_activation_status_bucket(row) for row in family_rows)
        next_counter: Counter[str] = Counter(
            public_label(row.get("next_action") or "inspect-local-evidence", "inspect-local-evidence")
            for row in family_rows
        )
        blocker_counter: Counter[str] = Counter(_activation_top_blocker(row) for row in family_rows)
        top = sorted(
            family_rows,
            key=lambda row: (
                _as_int(row.get("rank")) or 9999,
                -_as_float(row.get("realized_savings_usd")),
                -_as_float(row.get("projected_savings_usd")),
                -_as_int(row.get("sample_count")),
            ),
        )
        top_row = top[0] if top else {}
        rows.append(
            {
                "family": family,
                "status": "tracked" if family_rows else "missing",
                "row_count": len(family_rows),
                "status_counts": _status_count_rows(status_counter),
                "top_next_action": public_label(
                    (next_counter.most_common(1)[0][0] if next_counter else None)
                    or top_row.get("next_action")
                    or "none",
                    "none",
                ),
                "top_blocker": public_label(
                    (blocker_counter.most_common(1)[0][0] if blocker_counter else None)
                    or _activation_top_blocker(top_row)
                    or "none",
                    "none",
                ),
                "sample_count": sum(_as_int(row.get("sample_count")) for row in family_rows),
                "applied_count": sum(_as_int(row.get("applied_count")) for row in family_rows),
                "holdout_count": sum(_as_int(row.get("holdout_count")) for row in family_rows),
                "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in family_rows),
                "rollback_count": sum(_as_int(row.get("rollback_count")) for row in family_rows),
                "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in family_rows), 8),
                "realized_savings_usd": round(sum(_as_float(row.get("realized_savings_usd")) for row in family_rows), 8),
                "top_entry_rank": _as_int(top_row.get("rank")) if top_row else 0,
                "privacy": _local_activation_queue_privacy(),
            }
        )

    tracked_rows = [row for row in rows if row["row_count"] > 0]
    status_counter = Counter(row["status"] for row in rows)
    return {
        "schema": "tokenclaw.dashboard_activation_successor_burndown.v1",
        "status": "tracked" if tracked_rows else "missing",
        "summary": {
            "tracked_family_count": len(tracked_rows),
            "expected_family_count": len(ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES),
            "tracked_row_count": sum(row["row_count"] for row in rows),
            "status_counts": _status_count_rows(status_counter),
            "total_projected_savings_usd": round(sum(row["projected_savings_usd"] for row in rows), 8),
            "total_realized_savings_usd": round(sum(row["realized_savings_usd"] for row in rows), 8),
            "top_family": tracked_rows[0]["family"] if tracked_rows else None,
            "top_next_action": tracked_rows[0]["top_next_action"] if tracked_rows else None,
            "top_blocker": tracked_rows[0]["top_blocker"] if tracked_rows else None,
        },
        "families": rows,
        "privacy": _local_activation_queue_privacy(),
    }


CLOSED_LOOP_ACTIVATION_FAMILIES = ("cache", "crunch", "routing")

CLOSED_LOOP_ACTIVATION_STATES = (
    "preview_missing",
    "preview_agreed",
    "draft_ready",
    "applied_waiting_observation",
    "realized_savings",
    "retired_no_repeat",
    "rollback_required",
    "safety_stopped",
    "keep_blocked",
)


def _closed_loop_activation_family(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "cache" in text:
        return "cache"
    if "crunch" in text:
        return "crunch"
    if "routing" in text or "route" in text:
        return "routing"
    return text if text in CLOSED_LOOP_ACTIVATION_FAMILIES else None


def _closed_loop_activation_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "state",
        "current_status",
        "issue_worthy_status",
        "next_action",
        "unblock_reason",
        "blocking_reason",
        "freshness_state",
        "duplicate_suppression_status",
        "evidence_schema",
        "promotion_readiness",
        "promotion_decision",
        "decision",
    ):
        parts.append(str(row.get(key) or ""))
    parts.extend(str(code or "") for code in row.get("blocker_codes") or [])
    coverage = row.get("managed_preview_coverage") if isinstance(row.get("managed_preview_coverage"), dict) else {}
    for key in ("status", "preview_data_status", "top_omitted_or_blocker_reason"):
        parts.append(str(coverage.get(key) or ""))
    return " ".join(parts).lower()


def _closed_loop_activation_entry_states(row: dict[str, Any]) -> set[str]:
    text = _closed_loop_activation_text(row)
    states: set[str] = set()
    realized = _as_float(row.get("realized_savings_usd") or row.get("actual_saved_cost_usd") or row.get("observed_saved_usd"))
    applied = _as_int(row.get("applied_count"))
    safety_stops = _as_int(row.get("safety_stop_count"))
    rollbacks = _as_int(row.get("rollback_count"))
    preview_status = str(
        row.get("preview_verification_status")
        or (row.get("managed_preview_coverage") or {}).get("preview_data_status")
        or (row.get("managed_preview_coverage") or {}).get("status")
        or ""
    ).lower()

    if "no-data" in preview_status or "missing" in preview_status or "not-previewed" in preview_status:
        states.add("preview_missing")
    if row.get("preview_verified") or "preview-verified" in preview_status or "agreed" in text:
        states.add("preview_agreed")
    if (
        row.get("policy_write_candidate")
        or row.get("required_local_executor")
        or row.get("local_policy_patch")
        or "draft" in text
        or "stage" in text
    ):
        states.add("draft_ready")
    if safety_stops > 0 or "safety" in text and "stop" in text:
        states.add("safety_stopped")
    if rollbacks > 0 or row.get("rollback_required") or "rollback" in text:
        states.add("rollback_required")
    if "retire" in text or "retired" in text or "no-repeat" in text or "superseded" in text:
        states.add("retired_no_repeat")
    if realized > 0:
        states.add("realized_savings")
    if applied > 0 and realized <= 0 and not states.intersection({"rollback_required", "safety_stopped", "retired_no_repeat"}):
        states.add("applied_waiting_observation")
    if "keep-blocked" in text or "blocked" in text:
        states.add("keep_blocked")
    if not states:
        states.add("preview_missing")
    return states


def _closed_loop_entry_stale_age_hours(row: dict[str, Any]) -> float | None:
    rank_basis = row.get("rank_basis") if isinstance(row.get("rank_basis"), dict) else {}
    for value in (
        rank_basis.get("evidence_age_hours"),
        row.get("evidence_age_hours"),
        (row.get("managed_preview_coverage") or {}).get("latest_preview_age_hours")
        if isinstance(row.get("managed_preview_coverage"), dict)
        else None,
    ):
        if value is not None:
            return round(_as_float(value), 3)
    return None


def _closed_loop_activation_readiness(activation_burndown: dict[str, Any]) -> dict[str, Any]:
    family_rows: dict[str, dict[str, Any]] = {}
    for family in CLOSED_LOOP_ACTIVATION_FAMILIES:
        family_rows[family] = {
            "family": family,
            "row_count": 0,
            "state_counts": {state: 0 for state in CLOSED_LOOP_ACTIVATION_STATES},
            "top_next_action": None,
            "top_blocker": None,
            "top_state": None,
            "top_stale_evidence_age_hours": None,
            "projected_savings_usd": 0.0,
            "realized_savings_usd": 0.0,
            "sample_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "safety_stop_count": 0,
            "rollback_count": 0,
            "privacy": _local_activation_queue_privacy(),
        }

    entries = [row for row in activation_burndown.get("entries") or [] if isinstance(row, dict)]
    top_candidates: dict[str, dict[str, Any]] = {}
    for entry in entries:
        family = _closed_loop_activation_family(entry.get("local_action_family") or entry.get("lever"))
        if family not in family_rows:
            continue
        row = family_rows[family]
        states = _closed_loop_activation_entry_states(entry)
        row["row_count"] += 1
        for state in states:
            if state in row["state_counts"]:
                row["state_counts"][state] += 1
        row["projected_savings_usd"] += _as_float(entry.get("projected_savings_usd"))
        row["realized_savings_usd"] += _as_float(entry.get("realized_savings_usd"))
        row["sample_count"] += _as_int(entry.get("sample_count"))
        row["applied_count"] += _as_int(entry.get("applied_count"))
        row["holdout_count"] += _as_int(entry.get("holdout_count"))
        row["safety_stop_count"] += _as_int(entry.get("safety_stop_count"))
        row["rollback_count"] += _as_int(entry.get("rollback_count"))
        current_top = top_candidates.get(family)
        sort_key = (_as_int(entry.get("rank")) or 9999, -_as_float(entry.get("projected_savings_usd")))
        current_sort = (
            (_as_int(current_top.get("rank")) or 9999, -_as_float(current_top.get("projected_savings_usd")))
            if current_top
            else (999999, 0.0)
        )
        if sort_key < current_sort:
            top_candidates[family] = entry

    preview_burndown = activation_burndown.get("activation_preview_agreement_burndown")
    preview_families = preview_burndown.get("families") if isinstance(preview_burndown, dict) else []
    for preview in preview_families or []:
        if not isinstance(preview, dict):
            continue
        family = _closed_loop_activation_family(preview.get("local_action_family"))
        if family not in family_rows:
            continue
        row = family_rows[family]
        row["state_counts"]["preview_agreed"] += _as_int(preview.get("agreed_count"))
        row["state_counts"]["preview_missing"] += _as_int(preview.get("missing_count")) + _as_int(preview.get("not_previewed_count"))
        row["state_counts"]["draft_ready"] += _as_int(preview.get("dry_run_drafted_count"))
        if row["top_next_action"] is None and preview.get("top_next_action"):
            row["top_next_action"] = public_label(preview.get("top_next_action"), "none")
        if row["top_blocker"] is None and preview.get("top_reason_code"):
            row["top_blocker"] = public_label(preview.get("top_reason_code"), "none")

    for family, entry in top_candidates.items():
        row = family_rows[family]
        row["top_next_action"] = public_label(entry.get("next_action") or "inspect-local-evidence", "inspect-local-evidence")
        row["top_blocker"] = _activation_top_blocker(entry)
        ranked_states = [
            state
            for state in CLOSED_LOOP_ACTIVATION_STATES
            if row["state_counts"].get(state)
        ]
        row["top_state"] = ranked_states[0] if ranked_states else "preview_missing"
        row["top_stale_evidence_age_hours"] = _closed_loop_entry_stale_age_hours(entry)

    public_families: list[dict[str, Any]] = []
    total_state_counts: Counter[str] = Counter()
    for family in CLOSED_LOOP_ACTIVATION_FAMILIES:
        row = family_rows[family]
        state_counts = {state: _as_int(row["state_counts"].get(state)) for state in CLOSED_LOOP_ACTIVATION_STATES}
        total_state_counts.update(state_counts)
        row_count = _as_int(row.get("row_count"))
        status = "tracked" if row_count or any(state_counts.values()) else "missing"
        public_families.append(
            {
                "family": family,
                "status": status,
                "row_count": row_count,
                "state_counts": [{"state": state, "count": count} for state, count in state_counts.items()],
                "top_state": row.get("top_state") or ("preview_missing" if status == "tracked" else "missing"),
                "top_next_action": row.get("top_next_action") or "none",
                "top_blocker": row.get("top_blocker") or "none",
                "top_stale_evidence_age_hours": row.get("top_stale_evidence_age_hours"),
                "projected_savings_usd": round(_as_float(row.get("projected_savings_usd")), 8),
                "realized_savings_usd": round(_as_float(row.get("realized_savings_usd")), 8),
                "sample_count": _as_int(row.get("sample_count")),
                "applied_count": _as_int(row.get("applied_count")),
                "holdout_count": _as_int(row.get("holdout_count")),
                "safety_stop_count": _as_int(row.get("safety_stop_count")),
                "rollback_count": _as_int(row.get("rollback_count")),
                "privacy": row["privacy"],
            }
        )

    tracked = [row for row in public_families if row["status"] == "tracked"]
    top = sorted(
        tracked,
        key=lambda row: (
            -_as_float(row.get("projected_savings_usd")),
            -_as_float(row.get("realized_savings_usd")),
            -_as_int(row.get("sample_count")),
            str(row.get("family") or ""),
        ),
    )
    return {
        "schema": "tokenclaw.closed_loop_activation_readiness.v1",
        "generated_at": utc_now(),
        "status": "tracked" if tracked else "missing",
        "summary": {
            "family_count": len(public_families),
            "tracked_family_count": len(tracked),
            "row_count": sum(_as_int(row.get("row_count")) for row in public_families),
            "state_counts": [{"state": state, "count": int(total_state_counts.get(state, 0))} for state in CLOSED_LOOP_ACTIVATION_STATES],
            "top_family": top[0]["family"] if top else None,
            "top_state": top[0]["top_state"] if top else None,
            "top_next_action": top[0]["top_next_action"] if top else None,
            "top_blocker": top[0]["top_blocker"] if top else None,
            "top_projected_savings_usd": round(_as_float(top[0].get("projected_savings_usd")) if top else 0.0, 8),
            "top_realized_savings_usd": round(_as_float(top[0].get("realized_savings_usd")) if top else 0.0, 8),
            "top_stale_evidence_age_hours": top[0].get("top_stale_evidence_age_hours") if top else None,
            "total_projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in public_families), 8),
            "total_realized_savings_usd": round(sum(_as_float(row.get("realized_savings_usd")) for row in public_families), 8),
        },
        "families": public_families,
        "privacy": _local_activation_queue_privacy(
            activation_burndown.get("privacy") if isinstance(activation_burndown.get("privacy"), dict) else {}
        ),
    }


def _activation_successor_health_empty(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.activation_successor_queue_health.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "source": source,
        "summary": {
            "queued_action_count": 0,
            "successor_action_count": 0,
            "successor_decision_count": 0,
            "top_local_action_family": None,
            "top_state": None,
            "top_status": None,
            "top_next_action": None,
            "top_blocker": None,
            "top_blocker_codes": [],
            "top_preview_verification_status": None,
            "top_preview_verification_decision": None,
            "top_projected_savings_usd": 0.0,
            "top_realized_savings_usd": 0.0,
            "total_projected_savings_usd": 0.0,
            "total_realized_savings_usd": 0.0,
            "latest_preview_age_hours": None,
            "local_action_family_counts": [],
            "status_counts": [],
            "preview_gate_status_counts": [],
            "preview_gate_decision_counts": [],
            "blocker_counts": [],
            "next_action_counts": [],
        },
        "top_entries": [],
        "privacy": _local_activation_queue_privacy(),
    }


def _count_activation_values(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                counted = False
                for item in value:
                    text = str(item or "").strip()
                    if text:
                        counts[text] = counts.get(text, 0) + 1
                        counted = True
                if counted:
                    break
                continue
            text = str(value or "").strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
                break
    return _breakdown_from_counts(counts)


def _activation_successor_preview_gate(row: dict[str, Any]) -> dict[str, Any]:
    gate = row.get("managed_preview_gate") if isinstance(row.get("managed_preview_gate"), dict) else {}
    health_gate = gate.get("health_gate") if isinstance(gate.get("health_gate"), dict) else {}
    return {
        "status": row.get("preview_verification_status")
        or gate.get("status")
        or health_gate.get("status"),
        "decision": row.get("preview_verification_decision")
        or gate.get("decision"),
        "latest_preview_age_hours": row.get("latest_preview_age_hours")
        if row.get("latest_preview_age_hours") is not None
        else health_gate.get("latest_preview_age_hours"),
    }


def _activation_successor_top_entry(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: (_as_int(row.get("rank")) or 9999, _as_int(row.get("ledger_rank")) or 9999))[0]


def _public_activation_successor_health_entry(entry: dict[str, Any]) -> dict[str, Any]:
    gate = _activation_successor_preview_gate(entry)
    blocker_codes = entry.get("blocker_codes") if isinstance(entry.get("blocker_codes"), list) else []
    public = {
        "rank": _as_int(entry.get("rank")),
        "ledger_rank": _as_int(entry.get("ledger_rank")),
        "local_action_family": public_label(entry.get("local_action_family") or entry.get("lever") or "unknown", "unknown"),
        "state": public_label(entry.get("state") or "", ""),
        "current_status": public_label(entry.get("current_status") or entry.get("successor_status") or "", ""),
        "next_action": public_label(
            entry.get("next_action") or entry.get("recommended_next_action") or "",
            "",
        ),
        "top_blocker": public_label(
            blocker_codes[0] if blocker_codes else entry.get("unblock_reason") or "",
            "",
        ),
        "blocker_codes": [public_label(code, "unknown") for code in blocker_codes],
        "preview_verification_status": public_label(gate.get("status") or "", ""),
        "preview_verification_decision": public_label(gate.get("decision") or "", ""),
        "latest_preview_age_hours": gate.get("latest_preview_age_hours"),
        "projected_savings_usd": round(_as_float(entry.get("projected_savings_usd") or entry.get("projected_saved_usd")), 8),
        "realized_savings_usd": round(_as_float(entry.get("realized_savings_usd") or entry.get("actual_saved_cost_usd")), 8),
        "sample_count": _as_int(entry.get("sample_count")),
    }
    return {key: value for key, value in public.items() if value not in (None, "", [])}


def _activation_successor_health_from_queue(
    queue: dict[str, Any],
    *,
    source: dict[str, Any],
    plan_generated_at: Any = None,
    limit: int = 5,
) -> dict[str, Any]:
    entries = [row for row in queue.get("entries") or [] if isinstance(row, dict)]
    successor_actions = [row for row in queue.get("successor_actions") or [] if isinstance(row, dict)]
    successor_decisions = [row for row in queue.get("successor_decisions") or [] if isinstance(row, dict)]
    rows = entries or successor_actions
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    top = _activation_successor_top_entry(rows)
    top_gate = _activation_successor_preview_gate(top or {})
    top_blocker_codes = top.get("blocker_codes") if isinstance(top, dict) and isinstance(top.get("blocker_codes"), list) else []
    preview_ages = [
        _as_float(_activation_successor_preview_gate(row).get("latest_preview_age_hours"))
        for row in rows
        if _activation_successor_preview_gate(row).get("latest_preview_age_hours") is not None
    ]
    gate_status_counts = summary.get("preview_gate_status_counts")
    if not isinstance(gate_status_counts, list):
        gate_status_counts = _count_activation_values(
            [{**row, **_activation_successor_preview_gate(row)} for row in rows],
            "status",
        )
    gate_decision_counts = summary.get("preview_gate_decision_counts")
    if not isinstance(gate_decision_counts, list):
        gate_decision_counts = _count_activation_values(
            [{**row, **_activation_successor_preview_gate(row)} for row in rows],
            "decision",
        )
    capped = max(1, min(int(limit or 5), 20))
    return {
        "schema": "tokenclaw.activation_successor_queue_health.v1",
        "generated_at": utc_now(),
        "status": "ranked" if rows else "empty",
        "status_reason": "latest local activation successor queue health loaded" if rows else "latest queue has no entries",
        "queue_schema": queue.get("schema"),
        "queue_status": queue.get("status"),
        "source_schema": queue.get("source_schema"),
        "source": {**source, "plan_generated_at": plan_generated_at},
        "summary": {
            "queued_action_count": _as_int(summary.get("queued_action_count")) or len(entries),
            "successor_action_count": _as_int(summary.get("successor_action_count")) or len(successor_actions),
            "successor_decision_count": _as_int(summary.get("successor_decision_count")) or len(successor_decisions),
            "top_local_action_family": public_label(
                (top or {}).get("local_action_family") or (top or {}).get("lever") or summary.get("top_lever") or "",
                "",
            ),
            "top_state": public_label((top or {}).get("state") or summary.get("top_state") or "", ""),
            "top_status": public_label(
                (top or {}).get("current_status") or (top or {}).get("successor_status") or summary.get("top_current_status") or "",
                "",
            ),
            "top_next_action": public_label(
                (top or {}).get("next_action") or (top or {}).get("recommended_next_action") or summary.get("top_next_action") or "",
                "",
            ),
            "top_blocker": public_label(
                top_blocker_codes[0] if top_blocker_codes else (top or {}).get("unblock_reason") or summary.get("top_unblock_reason") or "",
                "",
            ),
            "top_blocker_codes": [public_label(code, "unknown") for code in top_blocker_codes],
            "top_preview_verification_status": public_label(top_gate.get("status") or "", ""),
            "top_preview_verification_decision": public_label(top_gate.get("decision") or "", ""),
            "top_projected_savings_usd": round(
                _as_float(summary.get("top_projected_savings_usd") or (top or {}).get("projected_savings_usd") or (top or {}).get("projected_saved_usd")),
                8,
            ),
            "top_realized_savings_usd": round(
                _as_float(summary.get("top_realized_savings_usd") or (top or {}).get("realized_savings_usd") or (top or {}).get("actual_saved_cost_usd")),
                8,
            ),
            "total_projected_savings_usd": round(_as_float(summary.get("total_projected_savings_usd")), 8),
            "total_realized_savings_usd": round(_as_float(summary.get("total_realized_savings_usd")), 8),
            "latest_preview_age_hours": min(preview_ages) if preview_ages else None,
            "local_action_family_counts": _count_activation_values(rows, "local_action_family", "lever"),
            "status_counts": summary.get("status_counts") if isinstance(summary.get("status_counts"), list) else _count_activation_values(rows, "current_status", "successor_status", "state"),
            "preview_gate_status_counts": gate_status_counts,
            "preview_gate_decision_counts": gate_decision_counts,
            "blocker_counts": _count_activation_values(rows, "blocker_codes", "unblock_reason"),
            "next_action_counts": _count_activation_values(rows, "next_action", "recommended_next_action"),
        },
        "top_entries": [_public_activation_successor_health_entry(row) for row in rows[:capped]],
        "privacy": _local_activation_queue_privacy(queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}),
    }


def build_activation_successor_queue_health(limit: int = 5) -> dict[str, Any]:
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _activation_successor_health_empty(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _activation_successor_health_empty(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
        )
    if not isinstance(payload, dict):
        return _activation_successor_health_empty(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
        )
    queue = _extract_local_activation_queue_from_plan(payload)
    if not isinstance(queue, dict):
        return _activation_successor_health_empty(
            status="no-queue",
            status_reason="latest research plan does not contain a local activation next-action queue",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )
    return _activation_successor_health_from_queue(
        queue,
        source=source,
        plan_generated_at=payload.get("generated_at"),
        limit=limit,
    )


def _activation_public_ref(value: Any, *, prefix: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return public_id(text, prefix=prefix, fallback=f"{prefix}:unknown")


def _activation_issue_queue_status(decision: dict[str, Any]) -> str:
    decision_text = str(decision.get("decision") or "").strip()
    issue_status = str(decision.get("issue_worthy_status") or "").strip()
    preview_status = str(decision.get("preview_agreement_status") or "").strip()
    preview_verified = bool(decision.get("preview_verified"))
    if issue_status == "suppressed" or decision_text in {"keep-current-rule", "suppress-duplicate"}:
        return "suppressed"
    if not preview_verified and preview_status in {
        "",
        "not-previewed",
        "missing-preview-decision",
        "no-data-preview-health",
        "stale-preview",
        "stale-preview-health",
        "incomplete-preview-health",
    }:
        return "stale/no-data"
    if issue_status == "blocked" or decision_text in {"keep-blocked", "review-stale-preview"}:
        return "blocked"
    if preview_verified and issue_status in {"ready", "review"} and decision_text in {"ready", "review", "review-only"}:
        return "ready"
    return issue_status or decision_text or "unknown"


def _activation_issue_queue_reason(decision: dict[str, Any]) -> str:
    for key in (
        "preview_omitted_reason",
        "preview_no_op_reason",
        "top_preview_omission_reason",
        "preview_agreement_status",
        "preview_verification_status",
        "decision",
    ):
        text = str(decision.get(key) or "").strip()
        if text:
            return text
    return "unknown"


def _public_preview_gated_successor_decision(decision: dict[str, Any]) -> dict[str, Any]:
    preview_requirement = str(decision.get("preview_requirement") or "").strip()
    managed_preview_required = bool(
        decision.get("managed_preview_required")
        or preview_requirement in {"required", "preview-required", "managed-preview-required"}
    )
    public = {
        "source_ref": _activation_public_ref(decision.get("source_fingerprint"), prefix="activation-ref"),
        "successor_action_ref": _activation_public_ref(
            decision.get("successor_action_fingerprint"),
            prefix="successor-ref",
        ),
        "local_action_family": public_label(decision.get("local_action_family") or "unknown", "unknown"),
        "decision": public_label(decision.get("decision") or "unknown", "unknown"),
        "recommended_next_action": public_label(
            decision.get("recommended_next_action") or "inspect-local-evidence",
            "inspect-local-evidence",
        ),
        "issue_worthy_status": public_label(decision.get("issue_worthy_status") or "unknown", "unknown"),
        "issue_queue_status": _activation_issue_queue_status(decision),
        "preview_agreement_status": public_label(
            decision.get("preview_agreement_status") or "not-previewed",
            "not-previewed",
        ),
        "preview_verified": bool(decision.get("preview_verified")),
        "preview_verification_status": public_label(decision.get("preview_verification_status") or "", ""),
        "preview_verification_decision": public_label(decision.get("preview_verification_decision") or "", ""),
        "preview_requirement": public_label(preview_requirement, ""),
        "managed_preview_required": managed_preview_required,
        "policy_write_candidate": bool(
            decision.get("policy_write_candidate")
            or decision.get("cache_apply_action_count")
            or decision.get("draft_action_count")
            or decision.get("dry_run_drafted")
        ),
        "preview_omitted_reason": public_label(decision.get("preview_omitted_reason") or "", ""),
        "preview_no_op_reason": public_label(decision.get("preview_no_op_reason") or "", ""),
        "top_preview_omission_reason": public_label(decision.get("top_preview_omission_reason") or "", ""),
        "privacy": _local_activation_queue_privacy(
            decision.get("privacy") if isinstance(decision.get("privacy"), dict) else {}
        ),
    }
    bool_keys = {"preview_verified", "managed_preview_required", "policy_write_candidate"}
    return {key: value for key, value in public.items() if value not in (None, "", []) or key in bool_keys}


PREVIEW_AGREEMENT_BUCKETS = (
    "agreed",
    "missing",
    "stale",
    "unsafe",
    "omitted",
    "blocked",
    "disagreed",
    "not_previewed",
)

PREVIEW_AGREEMENT_EXTRA_COUNTS = (
    "preview_optional",
    "preview_required",
    "dry_run_drafted",
)


def _preview_agreement_bucket(row: dict[str, Any]) -> str:
    status = str(row.get("preview_agreement_status") or "").strip().lower()
    verification_status = str(row.get("preview_verification_status") or "").strip().lower()
    decision = str(row.get("decision") or "").strip().lower()
    if row.get("preview_verified") or status == "agreed":
        return "agreed"
    if row.get("preview_omitted_reason") or row.get("top_preview_omission_reason") or "omitted" in status:
        return "omitted"
    if "unsafe" in status or "unsafe" in verification_status:
        return "unsafe"
    if "disagree" in status or "failed-closed" in status or "disagree" in verification_status:
        return "disagreed"
    if "stale" in status or "stale" in verification_status:
        return "stale"
    if "missing" in status or "no-data" in status or "missing" in verification_status or "no-data" in verification_status:
        return "missing"
    if status in {"", "not-previewed"}:
        return "not_previewed"
    if "blocked" in status or "blocked" in decision or "keep-blocked" in decision:
        return "blocked"
    return "blocked"


def _preview_agreement_extra_counts(row: dict[str, Any]) -> dict[str, int]:
    decision = str(row.get("preview_verification_decision") or row.get("decision") or "").strip().lower()
    requirement = str(row.get("preview_requirement") or "").strip().lower()
    managed_required = bool(row.get("managed_preview_required"))
    policy_write_candidate = bool(row.get("policy_write_candidate"))
    optional = (
        decision == "preview-optional"
        or requirement in {"optional", "preview-optional", "managed-preview-optional"}
    )
    required = managed_required or requirement in {"required", "preview-required", "managed-preview-required"}
    return {
        "preview_optional_count": int(bool(optional)),
        "preview_required_count": int(bool(required)),
        "dry_run_drafted_count": int(bool(policy_write_candidate or decision in {"draft", "dry-run-drafted"})),
    }


def _preview_agreement_reason(row: dict[str, Any]) -> str:
    for key in (
        "preview_omitted_reason",
        "preview_no_op_reason",
        "top_preview_omission_reason",
        "preview_verification_status",
        "preview_agreement_status",
        "decision",
    ):
        text = str(row.get(key) or "").strip()
        if text:
            return public_label(text, "unknown")
    return "unknown"


def _summary_preview_agreement_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("preview_agreement_by_local_action_family")
    if not isinstance(rows, list):
        return []
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("local_action_family") or "").strip()
        if not family:
            continue
        public: dict[str, Any] = {
            "local_action_family": public_label(family, "unknown"),
            "top_reason_code": public_label(row.get("top_reason_code") or row.get("top_reason") or "none", "none"),
            "top_next_action": public_label(row.get("top_next_action") or "none", "none"),
            "top_source_ref": _activation_public_ref(row.get("source_fingerprint") or row.get("top_source_fingerprint"), prefix="activation-ref"),
            "top_successor_action_ref": _activation_public_ref(
                row.get("successor_action_fingerprint") or row.get("top_successor_action_fingerprint"),
                prefix="successor-ref",
            ),
            "privacy": _local_activation_queue_privacy(row.get("privacy") if isinstance(row.get("privacy"), dict) else {}),
        }
        total = 0
        for bucket in PREVIEW_AGREEMENT_BUCKETS:
            count = _as_int(row.get(f"{bucket}_count"))
            if bucket == "unsafe" and not count:
                count = _as_int(row.get("request_shape_unsafe_count")) + _as_int(row.get("crunch_preview_quality_risk_count"))
            public[f"{bucket}_count"] = count
            total += count
        for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS:
            public[f"{bucket}_count"] = _as_int(row.get(f"{bucket}_count"))
        if not total:
            total = sum(_as_int(row.get(key)) for key in ("stored_preview_outcome_count", "row_count", "count"))
        public["total_count"] = total
        public_rows.append(public)
    return public_rows


def _activation_preview_agreement_burndown(queue: dict[str, Any]) -> dict[str, Any]:
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    summary_rows = _summary_preview_agreement_rows(summary)
    raw_decisions: list[dict[str, Any]] = []
    if isinstance(queue, dict) and queue:
        try:
            from tokenclaw.orchestrator_research import (
                build_local_activation_successor_actions,
                build_local_activation_successor_decisions,
            )

            actions = queue.get("successor_actions")
            if not isinstance(actions, list):
                actions = build_local_activation_successor_actions(queue)
            decisions = queue.get("successor_decisions")
            if not isinstance(decisions, list):
                decisions = build_local_activation_successor_decisions({"successor_actions": actions})
            raw_decisions = [row for row in decisions if isinstance(row, dict)]
        except Exception:
            raw_decisions = []
    public_decisions = [_public_preview_gated_successor_decision(row) for row in raw_decisions]
    if not public_decisions:
        total_by_bucket = {
            f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in summary_rows)
            for bucket in PREVIEW_AGREEMENT_BUCKETS
        }
        extra_totals = {
            f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in summary_rows)
            for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS
        }
        return {
            "schema": "tokenclaw.dashboard_activation_preview_agreement_burndown.v1",
            "status": "tracked" if summary_rows else "missing",
            "summary": {
                "local_action_family_count": len(summary_rows),
                "successor_decision_count": 0,
                "total_count": sum(_as_int(row.get("total_count")) for row in summary_rows),
                **total_by_bucket,
                **extra_totals,
            },
            "families": summary_rows,
            "privacy": _local_activation_queue_privacy(queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}),
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in public_decisions:
        family = public_label(row.get("local_action_family") or "unknown", "unknown")
        grouped.setdefault(family, []).append(row)
    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        bucket_counts = Counter(_preview_agreement_bucket(row) for row in rows)
        reason_counts = Counter(_preview_agreement_reason(row) for row in rows)
        top = rows[0]
        family_rows.append(
            {
                "local_action_family": family,
                "total_count": len(rows),
                **{f"{bucket}_count": int(bucket_counts.get(bucket, 0)) for bucket in PREVIEW_AGREEMENT_BUCKETS},
                **{
                    key: sum(_preview_agreement_extra_counts(row)[key] for row in rows)
                    for key in (f"{bucket}_count" for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS)
                },
                "top_preview_agreement_status": public_label(top.get("preview_agreement_status") or "not-previewed", "not-previewed"),
                "top_reason_code": reason_counts.most_common(1)[0][0] if reason_counts else "unknown",
                "top_next_action": public_label(top.get("recommended_next_action") or "inspect-local-evidence", "inspect-local-evidence"),
                "top_source_ref": top.get("source_ref"),
                "top_successor_action_ref": top.get("successor_action_ref"),
                "privacy": _local_activation_queue_privacy(top.get("privacy") if isinstance(top.get("privacy"), dict) else {}),
            }
        )
    total_counts = {
        f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in family_rows)
        for bucket in PREVIEW_AGREEMENT_BUCKETS
    }
    extra_totals = {
        f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in family_rows)
        for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS
    }
    return {
        "schema": "tokenclaw.dashboard_activation_preview_agreement_burndown.v1",
        "status": "tracked" if family_rows else "missing",
        "summary": {
            "local_action_family_count": len(family_rows),
            "successor_decision_count": len(public_decisions),
            "total_count": len(public_decisions),
            **total_counts,
            **extra_totals,
        },
        "families": family_rows,
        "privacy": _local_activation_queue_privacy(queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}),
    }


def _proposal_status_label(labels: Any) -> str:
    if not isinstance(labels, list):
        return "status:unknown"
    for label in labels:
        text = str(label or "").strip()
        if text.startswith("status:"):
            return text
    return "status:unknown"


def _public_preview_gated_issue_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    labels = [public_label(label, "unknown") for label in proposal.get("labels") or [] if str(label or "").strip()]
    public = {
        "repo": public_label(proposal.get("repo") or "lutzkuen/tokenclaw", "lutzkuen/tokenclaw"),
        "title": public_label(proposal.get("title") or "Untitled activation successor issue", "Untitled activation successor issue"),
        "labels": labels,
        "status_label": _proposal_status_label(labels),
        "proposal_source": public_label(proposal.get("proposal_source") or "activation-successor", "activation-successor"),
        "source_ref": _activation_public_ref(proposal.get("fingerprint"), prefix="activation-ref"),
        "successor_action_ref": _activation_public_ref(
            proposal.get("successor_action_fingerprint"),
            prefix="successor-ref",
        ),
        "expected_savings_path": public_label(proposal.get("expected_savings_path") or "", ""),
        "privacy": _local_activation_queue_privacy(
            proposal.get("privacy") if isinstance(proposal.get("privacy"), dict) else {}
        ),
    }
    return {key: value for key, value in public.items() if value not in (None, "", [])}


def _top_issue(rows: list[dict[str, Any]], status: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("issue_queue_status") == status:
            return {
                "source_ref": row.get("source_ref"),
                "local_action_family": row.get("local_action_family"),
                "decision": row.get("decision"),
                "recommended_next_action": row.get("recommended_next_action"),
                "preview_agreement_status": row.get("preview_agreement_status"),
                "reason": _activation_issue_queue_reason(row),
            }
    return None


def _preview_gated_activation_issue_queue_payload(
    *,
    payload: dict[str, Any],
    queue: dict[str, Any],
    source: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    try:
        from tokenclaw.orchestrator_research import (
            _proposals_from_activation_successor_decisions,
            build_local_activation_successor_actions,
            build_local_activation_successor_decisions,
        )
    except Exception:
        return _empty_preview_gated_activation_issue_queue_payload(
            status="builder-unavailable",
            status_reason="activation successor builders could not be imported",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )

    actions = queue.get("successor_actions")
    if not isinstance(actions, list):
        actions = build_local_activation_successor_actions(queue)
    decisions = queue.get("successor_decisions")
    if not isinstance(decisions, list):
        decisions = build_local_activation_successor_decisions({"successor_actions": actions})
    raw_decisions = [row for row in decisions if isinstance(row, dict)]
    public_decisions = [_public_preview_gated_successor_decision(row) for row in raw_decisions]

    stats_summary = {"local_activation_next_action_queue": {**queue, "successor_actions": actions, "successor_decisions": raw_decisions}}
    derived_proposals = _proposals_from_activation_successor_decisions(stats_summary)
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    backlog_changes = payload.get("backlog_changes") if isinstance(payload.get("backlog_changes"), dict) else {}
    extra_proposals = backlog_changes.get("create_issues") if isinstance(backlog_changes.get("create_issues"), list) else []
    if not extra_proposals and isinstance(evidence.get("backlog_changes"), dict):
        extra_proposals = evidence["backlog_changes"].get("create_issues") if isinstance(evidence["backlog_changes"].get("create_issues"), list) else []
    seen_proposals: set[tuple[str, str]] = set()
    public_proposals: list[dict[str, Any]] = []
    for proposal in [*derived_proposals, *[row for row in extra_proposals if isinstance(row, dict)]]:
        public = _public_preview_gated_issue_proposal(proposal)
        key = (str(public.get("repo") or ""), str(public.get("title") or ""))
        if key in seen_proposals:
            continue
        seen_proposals.add(key)
        public_proposals.append(public)

    status_counts = Counter(str(row.get("issue_queue_status") or "unknown") for row in public_decisions)
    decision_counts = Counter(str(row.get("decision") or "unknown") for row in public_decisions)
    issue_status_counts = Counter(str(row.get("issue_worthy_status") or "unknown") for row in public_decisions)
    preview_counts = Counter(str(row.get("preview_agreement_status") or "not-previewed") for row in public_decisions)
    family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in public_decisions)
    reason_counts = Counter(_activation_issue_queue_reason(row) for row in public_decisions)
    capped = max(1, min(int(limit or 20), 50))
    queue_privacy = queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.dashboard_preview_gated_activation_issue_queue.v1",
        "generated_at": utc_now(),
        "status": "ranked" if public_decisions else "empty",
        "status_reason": "latest preview-gated activation issue queue loaded" if public_decisions else "latest queue has no successor decisions",
        "queue_schema": queue.get("schema"),
        "queue_status": queue.get("status"),
        "source_schema": queue.get("source_schema"),
        "source": {**source, "plan_generated_at": payload.get("generated_at")},
        "summary": {
            "successor_decision_count": len(public_decisions),
            "issue_proposal_count": len(public_proposals),
            "ready_count": status_counts.get("ready", 0),
            "blocked_count": status_counts.get("blocked", 0),
            "stale_or_no_data_count": status_counts.get("stale/no-data", 0),
            "suppressed_count": status_counts.get("suppressed", 0),
            "decision_counts": _breakdown_from_counts(dict(decision_counts)),
            "issue_status_counts": _breakdown_from_counts(dict(issue_status_counts)),
            "preview_agreement_status_counts": _breakdown_from_counts(dict(preview_counts)),
            "issue_queue_status_counts": _breakdown_from_counts(dict(status_counts)),
            "local_action_family_counts": _breakdown_from_counts(dict(family_counts)),
            "top_reason_counts": _breakdown_from_counts(dict(reason_counts))[:10],
            "top_ready_issue": _top_issue(public_decisions, "ready"),
            "top_blocked_issue": _top_issue(public_decisions, "blocked"),
            "top_stale_or_no_data_issue": _top_issue(public_decisions, "stale/no-data"),
        },
        "successor_decisions": public_decisions[:capped],
        "issue_proposals": public_proposals[:capped],
        "privacy": _local_activation_queue_privacy(queue_privacy),
    }


def _public_local_activation_queue_entry(
    entry: dict[str, Any],
    *,
    managed_preview_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = {
        "rank",
        "ledger_rank",
        "lever",
        "local_action_family",
        "state",
        "current_status",
        "issue_worthy_status",
        "next_action",
        "unblock_reason",
        "blocking_reason",
        "freshness_state",
        "rank_bucket",
        "rank_basis",
        "blocker_codes",
        "sample_count",
        "applied_count",
        "holdout_count",
        "fallback_count",
        "safety_stop_count",
        "rollback_count",
        "realized_savings_usd",
        "projected_savings_usd",
        "savings_per_1000_calls_usd",
        "freshness_adjusted_savings_per_1000_calls_usd",
        "target_local_rule_file",
        "target_local_policy_section",
        "duplicate_suppression_status",
        "duplicate_suppression_reason",
        "evidence_schema",
        "expected_savings_path",
        "requested_model",
        "candidate_target_model",
        "required_local_executor",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
    }
    public = {
        key: _copy_policy(value)
        for key, value in entry.items()
        if key in allowed and value not in (None, "", [])
    }
    for key in (
        "rank",
        "ledger_rank",
        "sample_count",
        "applied_count",
        "holdout_count",
        "fallback_count",
        "safety_stop_count",
        "rollback_count",
        "rank_bucket",
    ):
        public[key] = _as_int(public.get(key))
    for key in (
        "realized_savings_usd",
        "projected_savings_usd",
        "savings_per_1000_calls_usd",
        "freshness_adjusted_savings_per_1000_calls_usd",
    ):
        public[key] = round(_as_float(public.get(key)), 8)
    if not isinstance(public.get("blocker_codes"), list):
        public["blocker_codes"] = []
    public["privacy"] = _local_activation_queue_privacy(entry.get("privacy") if isinstance(entry.get("privacy"), dict) else {})
    family_coverage = _managed_preview_coverage_for_family(
        managed_preview_coverage,
        str(public.get("local_action_family") or entry.get("local_action_family") or ""),
    )
    if family_coverage is not None:
        public["managed_preview_coverage"] = family_coverage
    return public


def _extract_local_activation_queue_from_plan(payload: dict[str, Any]) -> dict[str, Any] | None:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    stats_summary = evidence.get("stats_summary") if isinstance(evidence.get("stats_summary"), dict) else {}
    for container in (evidence, payload, stats_summary):
        queue = container.get("local_activation_next_action_queue") if isinstance(container, dict) else None
        if isinstance(queue, dict):
            return queue

    ledger = None
    for container in (evidence, payload, stats_summary):
        candidate = container.get("evidence_to_activation_next_action_ledger") if isinstance(container, dict) else None
        if isinstance(candidate, dict):
            ledger = candidate
            break
    if not isinstance(ledger, dict):
        return None
    try:
        from tokenclaw.orchestrator_research import build_local_activation_next_action_queue

        return build_local_activation_next_action_queue({"evidence_to_activation_next_action_ledger": ledger})
    except Exception:
        return None


async def stats_local_activation_next_action_queue(limit: int = 20, store_obj: Any | None = None) -> dict[str, Any]:
    managed_preview_coverage = _managed_activation_preview_coverage(store_obj)
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_local_activation_queue_payload(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
            managed_preview_coverage=managed_preview_coverage,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _empty_local_activation_queue_payload(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
            managed_preview_coverage=managed_preview_coverage,
        )
    if not isinstance(payload, dict):
        return _empty_local_activation_queue_payload(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
            managed_preview_coverage=managed_preview_coverage,
        )

    queue = _extract_local_activation_queue_from_plan(payload)
    if not isinstance(queue, dict):
        return _empty_local_activation_queue_payload(
            status="no-queue",
            status_reason="latest research plan does not contain a local activation next-action queue",
            source={**source, "plan_generated_at": payload.get("generated_at")},
            managed_preview_coverage=managed_preview_coverage,
        )

    capped = max(1, min(int(limit or 20), 50))
    all_entries = [
        _public_local_activation_queue_entry(entry, managed_preview_coverage=managed_preview_coverage)
        for entry in queue.get("entries") or []
        if isinstance(entry, dict)
    ]
    entries = all_entries[:capped]
    summary = _public_local_activation_queue_summary(
        queue.get("summary") if isinstance(queue.get("summary"), dict) else {},
        len(entries),
    )
    queue_privacy = queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.dashboard_local_activation_next_action_queue.v1",
        "generated_at": utc_now(),
        "status": "ranked" if entries else "empty",
        "status_reason": "latest local activation next-action queue loaded" if entries else "latest queue has no entries",
        "queue_schema": queue.get("schema"),
        "queue_status": queue.get("status"),
        "source_schema": queue.get("source_schema"),
        "source": {**source, "plan_generated_at": payload.get("generated_at")},
        "summary": summary,
        "entries": entries,
        "successor_burndown": _activation_successor_burndown(all_entries),
        "activation_preview_agreement_burndown": _activation_preview_agreement_burndown(queue),
        "managed_preview_coverage": managed_preview_coverage,
        "privacy": _local_activation_queue_privacy(queue_privacy),
    }


async def stats_activation_preview_burndown(limit: int = 20, store_obj: Any | None = None) -> dict[str, Any]:
    queue_payload = await stats_local_activation_next_action_queue(limit=limit, store_obj=store_obj)
    burndown = queue_payload.get("activation_preview_agreement_burndown")
    if not isinstance(burndown, dict):
        burndown = _activation_preview_agreement_burndown({})
    return {
        **burndown,
        "generated_at": utc_now(),
        "queue_status": queue_payload.get("status"),
        "queue_status_reason": queue_payload.get("status_reason"),
        "queue_schema": queue_payload.get("queue_schema"),
        "queue_source_schema": queue_payload.get("source_schema"),
        "source": queue_payload.get("source") if isinstance(queue_payload.get("source"), dict) else {},
        "managed_preview_coverage": queue_payload.get("managed_preview_coverage"),
    }


async def stats_preview_gated_activation_issue_queue(limit: int = 20, store_obj: Any | None = None) -> dict[str, Any]:
    del store_obj
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_preview_gated_activation_issue_queue_payload(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _empty_preview_gated_activation_issue_queue_payload(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
        )
    if not isinstance(payload, dict):
        return _empty_preview_gated_activation_issue_queue_payload(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
        )

    queue = _extract_local_activation_queue_from_plan(payload)
    if not isinstance(queue, dict):
        return _empty_preview_gated_activation_issue_queue_payload(
            status="no-queue",
            status_reason="latest research plan does not contain a local activation next-action queue",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )
    return _preview_gated_activation_issue_queue_payload(
        payload=payload,
        queue=queue,
        source=source,
        limit=limit,
    )


from tokenclaw.stats.dashboard_html import dashboard_html

from tokenclaw.stats.managed import (
    _claude_public_metadata_label,
    _claude_route_blocker_reason,
    _claude_route_count,
    _claude_route_finalize_candidate,
    _claude_route_group_key,
    _claude_route_increment_breakdown,
    _claude_route_is_eligible,
    _claude_route_public_label,
    _claude_route_stage_from_canary,
    _claude_route_touch,
    _claude_routing_funnel_privacy,
    _collect_action_family,
    _day_key,
    _finalize_openai_scoreboard_bucket,
    _finalize_phase_canary_bucket,
    _find_nested_dict,
    _managed_attempt_status,
    _managed_breakdown,
    _managed_decision_source,
    _managed_error_class,
    _managed_feed_decision_summary,
    _managed_feed_state,
    _managed_feedback_error_class,
    _managed_feedback_inferred_action_family,
    _managed_feedback_payload_action_families,
    _managed_feedback_queue_expectation,
    _managed_feedback_queue_health,
    _managed_feedback_safe_error_class,
    _managed_openai_activation_staged_drafts,
    _managed_openai_activation_status,
    _managed_pattern_add_adoption_row,
    _managed_pattern_adoption_from_store,
    _managed_pattern_blocker_reasons,
    _managed_pattern_finalize_adoption_rows,
    _managed_pattern_holdout_comparisons,
    _managed_pattern_is_bypass,
    _managed_pattern_summary_stage,
    _managed_policy_event_sections,
    _managed_policy_event_stage,
    _managed_policy_lifecycle_rows,
    _MANAGED_SUCCESS_STATUSES,
    _new_claude_route_funnel_candidate,
    _new_openai_scoreboard_bucket,
    _new_phase_canary_bucket,
    _normalize_feedback_dimension,
    _openai_activation_conflict_summary,
    _openai_activation_counts_by_family,
    _openai_activation_review_metadata,
    _openai_canary_candidate_rows,
    _openai_canary_policy_state,
    _openai_canary_readiness_state,
    _openai_canary_top_blockers,
    _openai_candidate_id,
    _openai_dashboard_dimension,
    _openai_dashboard_reason_codes,
    _openai_governor_endpoint,
    _openai_governor_from_metadata,
    _openai_quality_gate_outcome,
    _openai_recommendation_state,
    _phase_canary_cohort,
    _phase_routing_feedback_rows,
    _phase_routing_public_health_rows,
    _phase_routing_public_lifecycle_rows,
    _public_managed_feedback_row,
    _public_managed_openai_activation_event,
    _routing_candidate_lifecycle_privacy,
    _routing_candidate_lifecycle_source,
    _routing_candidate_stage_row,
    _safe_latest_managed_activation_proof,
    _safe_public_bool_env_configured,
    _safe_thinking_tail_loop_status,
    _safe_thinking_tail_readiness,
    _shadow_routing_candidate_id,
    _shadow_routing_candidate_rows,
    _shadow_routing_readiness_state,
    _shadow_routing_row_key,
    _shadow_routing_scoreboard_to_candidate_row,
    MANAGED_OPENAI_ACTIVATION_ACTIONS,
    MANAGED_OPENAI_SUPPORTED_ACTION_FAMILIES,
    ROUTING_CANDIDATE_LIFECYCLE_STATES,
    stats_claude_canary_impact,
    stats_claude_routing_promotion_funnel,
    stats_managed_activation_status,
    stats_managed_feedback_queue_freshness,
    stats_managed_openai_activation,
    stats_managed_recommendations,
    stats_openai_canary_readiness,
    stats_openai_optimization_readiness,
    stats_openai_routing_report,
    stats_openai_scoreboard,
    stats_phase_routing,
    stats_routing_candidate_lifecycle_burndown,
    stats_routing_coverage_report,
    stats_shadow_routing_promotion_readiness,
)

from tokenclaw.stats.codex import (
    _codex_turn_estimates,
    _codex_estimates_with_cache,
    _codex_not_applied_decision,
    _codex_turn_risk_features,
    _codex_summary_hint_status,
    _codex_summary_hint_estimated_savings,
    _new_codex_summary_hint_bucket,
    _add_codex_summary_hint_bucket,
    _finalize_codex_summary_hint_buckets,
    _codex_summary_hint_status_totals,
    _codex_summary_hint_canary_summary,
    _codex_crunch_pattern_breakdown,
    _codex_decision_metadata_state,
    _codex_missing_decision,
    _codex_normalized_decision,
    _codex_model_field_state,
    _codex_param_shape_category,
    _codex_phase_signal,
    _codex_phase_from_signal_counts,
    _codex_signal_counts_from_method_counts,
    _codex_public_event_window,
    _codex_same_scope,
    _codex_turn_bounds,
    _codex_count_bucket,
    _empty_codex_token_totals,
    _codex_public_scope_hash,
    _codex_public_token_usage,
    _codex_token_usage_delta,
    _codex_token_usage_cost,
    _codex_turn_token_estimates,
    _codex_turn_model,
    _codex_usage_matching_turn,
    _codex_usage_reconciliation_status,
    _codex_token_usage_reconciliation_state,
    _add_codex_token_totals,
    _codex_quota_token_usage_report,
    _codex_workflow_phase,
    _new_codex_phase_bucket,
    _finalize_codex_phase_bucket,
    _codex_plateau_scope,
    _codex_original_session_key,
    _codex_metadata_workflow_groups,
    _codex_phase_from_decision_metadata,
    _codex_meaningful_crunch,
    _codex_plateau_candidate_report,
    stats_codex_effectiveness,
    _codex_rule_report_meta,
    _codex_rule_report_cohort,
    _new_codex_rule_report_bucket,
    _finalize_codex_rule_report_bucket,
    stats_codex_canary_impact,
    _codex_cache_readiness_cohort,
    _codex_cache_readiness_cohorts,
    _codex_readiness_check,
    _openai_codex_blocker,
    stats_openai_codex_readiness,
    stats_codex_readiness,
    _codex_turn_activity_unit,
    _codex_accounting_unit,
    _breakdown_lookup,
    _token_drift_bucket,
)

from tokenclaw.stats.cache_replay import (
    _CACHE_BLOCKER_SCAN_LIMIT,
    _cache_decision_breakdown,
    _cache_replay_cohort_ranking_from_units,
    _cache_replayability_units_from_store,
    _cache_zero_hit_blocker_ladder,
    stats_cache_effectiveness,
    stats_cache_replay_activation_health,
    stats_cache_replay_cohort_ranking,
    stats_cache_replay_confidence,
    stats_cache_replay_dry_run,
    stats_cache_replay_readiness,
    stats_cache_replayability,
    stats_streaming_cache_hit_recovery,
)
