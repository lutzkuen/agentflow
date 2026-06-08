from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
import ipaddress
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from agentflow_proxy.codex_app_policy import (
    CODEX_APP_SOURCE_SURFACE,
    canonical_source_surface,
    codex_app_bundle_policy_state,
    codex_app_surface_policy_state,
    is_codex_turn_source_surface,
)
from agentflow_proxy.limiter import model_tier
from agentflow_proxy.policy_files import policy_file_status
from agentflow_proxy.pricing import codex_app_pricing_basis, estimate_blended_input_savings, estimate_cost
from agentflow_proxy.quality import (
    derive_codex_turn_quality_signals,
    derive_provider_quality_signals,
    summarize_quality_signals,
)
from agentflow_proxy.routing_experiments import ROUTING_EXPERIMENT_MIN_SAMPLES
from agentflow_proxy.store import utc_now

CODEX_APP_PRICING_BASIS = codex_app_pricing_basis()
CODEX_APP_MODEL = str(CODEX_APP_PRICING_BASIS["model"])
CODEX_APP_COST_BASIS = str(CODEX_APP_PRICING_BASIS["cost_basis"])
CODEX_APP_PROCESSING_MODE = str(CODEX_APP_PRICING_BASIS["processing_mode"])
CODEX_APP_COST_KNOWN = bool(CODEX_APP_PRICING_BASIS["cost_known"])
CODEX_APP_TELEMETRY_ONLY_REASON = "codex-app-telemetry-only"
TOKEN_CHARS = 4


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
    raw = os.getenv("AGENTFLOW_DATABASE_URL") or default_db or ""
    lowered = raw.lower()
    if "://" in raw:
        if lowered.startswith("sqlite://"):
            return "sqlite-url"
        return "external-database-url"
    expanded = os.path.abspath(os.path.expanduser(raw)) if raw else ""
    home = os.path.abspath(os.path.expanduser("~"))
    if not expanded:
        return "unknown"
    if expanded.startswith(os.path.join(home, ".agentflow") + os.sep):
        return "local-agentflow-home"
    if expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        return "local-temp"
    return "local-path"


def _policy_events_path_class() -> str:
    raw = os.getenv("AGENTFLOW_POLICY_EVENTS_LOG", "~/.agentflow/policy_events.jsonl")
    expanded = os.path.abspath(os.path.expanduser(raw))
    home = os.path.abspath(os.path.expanduser("~"))
    if expanded.startswith(os.path.join(home, ".agentflow") + os.sep):
        return "local-agentflow-home"
    if expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        return "local-temp"
    return "local-path"


def _breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _managed_feedback_error_class(row: dict[str, Any]) -> str | None:
    error = _sanitize_error_sample(row.get("last_error"), limit=240)
    if not error:
        status_code = _as_int(row.get("last_status_code"))
        return f"http_{status_code}" if status_code else None
    head = error.split(":", 1)[0].strip()
    return head[:80] if head else "error"


def _public_managed_feedback_row(row: dict[str, Any] | None, *, now: datetime) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "queue_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "optimization_unit_id": row.get("optimization_unit_id"),
        "status": row.get("status"),
        "attempts": _as_int(row.get("attempts")),
        "next_attempt_at": row.get("next_attempt_at"),
        "last_status_code": row.get("last_status_code"),
        "last_error_class": _managed_feedback_error_class(row),
        "sent_at": row.get("sent_at"),
        "age_seconds": _seconds_since_iso(row.get("created_at"), now),
        "due_age_seconds": _seconds_since_iso(row.get("next_attempt_at"), now)
        if row.get("status") in {"queued", "retryable-error"}
        else None,
        "payload_included": False,
    }


def _managed_feedback_queue_health(store_obj: Any | None, *, sample_limit: int = 5) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if store_obj is None or not hasattr(store_obj, "managed_outcome_feedback_rows"):
        return {
            "available": False,
            "summary": {
                "total": 0,
                "queued": 0,
                "due": 0,
                "retryable_error": 0,
                "dropped_after_limit": 0,
                "sent": 0,
                "oldest_due_age_seconds": None,
            },
            "status_breakdown": [],
            "source_surface_breakdown": [],
            "oldest_due": None,
            "last_successful_flush": None,
            "due_samples": [],
            "privacy": {
                "metadata_only": True,
                "payload_json_included": False,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "provider_bodies_included": False,
            },
        }

    try:
        rows = store_obj.managed_outcome_feedback_rows(limit=10000)
    except Exception:
        rows = []

    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    due_rows: list[dict[str, Any]] = []
    sent_rows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        source = str(row.get("source_surface") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if status in {"queued", "retryable-error"}:
            due_at = _parse_utc_datetime(row.get("next_attempt_at"))
            if due_at is not None and due_at <= now:
                due_rows.append(row)
        if status == "sent":
            sent_rows.append(row)

    due_rows.sort(key=lambda row: _parse_utc_datetime(row.get("next_attempt_at")) or now)
    oldest_due = due_rows[0] if due_rows else None
    sent_rows.sort(
        key=lambda row: _parse_utc_datetime(row.get("sent_at") or row.get("updated_at")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    last_successful = sent_rows[0] if sent_rows else None
    summary = {
        "total": sum(status_counts.values()),
        "queued": status_counts.get("queued", 0),
        "due": len(due_rows),
        "retryable_error": status_counts.get("retryable-error", 0),
        "sending": status_counts.get("sending", 0),
        "sent": status_counts.get("sent", 0),
        "dropped_after_limit": status_counts.get("dropped-after-limit", 0),
        "error": status_counts.get("error", 0),
        "oldest_due_age_seconds": _seconds_since_iso(oldest_due.get("next_attempt_at"), now) if oldest_due else None,
        "oldest_pending_age_seconds": _seconds_since_iso(
            min(
                (row for row in rows if row.get("status") in {"queued", "retryable-error"}),
                key=lambda row: _parse_utc_datetime(row.get("created_at")) or now,
                default={},
            ).get("created_at"),
            now,
        ),
    }
    return {
        "available": True,
        "summary": summary,
        "status_breakdown": _breakdown_from_counts(status_counts),
        "source_surface_breakdown": _breakdown_from_counts(source_counts),
        "oldest_due": _public_managed_feedback_row(oldest_due, now=now),
        "last_successful_flush": _public_managed_feedback_row(last_successful, now=now),
        "due_samples": [
            item
            for item in (
                _public_managed_feedback_row(row, now=now)
                for row in due_rows[: max(0, int(sample_limit or 0))]
            )
            if item is not None
        ],
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "secrets_included": False,
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
    recommendation_enabled = _env_bool("AGENTFLOW_RECOMMENDATION_ENABLED", False)
    recommendation_url = os.getenv("AGENTFLOW_RECOMMENDATION_SERVER_URL", "http://127.0.0.1:4100").rstrip("/")
    policy_bundle_url = os.getenv("AGENTFLOW_POLICY_BUNDLE_RECOMMENDATION_URL")
    managed_auth_configured = bool(os.getenv("AGENTFLOW_MANAGED_API_KEY"))
    log_bodies_enabled = _env_bool("AGENTFLOW_LOG_BODIES", False)
    policy_events_enabled = _env_bool("AGENTFLOW_POLICY_EVENTS", True)
    proxy_host_value = proxy_host or os.getenv("AGENTFLOW_PROXY_HOST") or os.getenv("AGENTFLOW_HOST")
    proxy_loopback = _host_is_loopback(proxy_host_value)
    dashboard_loopback = _host_is_loopback(dashboard_host) if dashboard_host else None
    db_class = _db_path_class(default_db)
    recommendation_state = _url_host_state(recommendation_url if recommendation_enabled or os.getenv("AGENTFLOW_RECOMMENDATION_SERVER_URL") else None)
    policy_bundle_state = _url_host_state(policy_bundle_url)
    feedback_queue = _managed_feedback_queue_health(store_obj)
    feedback_summary = feedback_queue.get("summary") or {}

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
            "AGENTFLOW_LOG_BODIES is enabled; raw request and response bodies may be stored locally for debugging.",
        )
    if recommendation_enabled and not managed_auth_configured:
        warn(
            "managed-recommendation-unauthenticated",
            "high",
            "Managed recommendation and outcome feedback are enabled without a configured managed API key.",
        )
    if _as_int(feedback_summary.get("due")) > 0:
        warn(
            "managed-feedback-due-queue",
            "medium",
            "Managed outcome feedback has retryable rows due for flush; the managed feedback loop may be stuck.",
        )
    if _as_int(feedback_summary.get("retryable_error")) > 0:
        warn(
            "managed-feedback-retryable-errors",
            "medium",
            "Managed outcome feedback has rows waiting after retryable delivery errors.",
        )
    if _as_int(feedback_summary.get("dropped_after_limit")) > 0:
        warn(
            "managed-feedback-dropped-after-limit",
            "high",
            "Managed outcome feedback rows were dropped after reaching the retry limit.",
        )
    if policy_bundle_url and not managed_auth_configured:
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
        "schema": "agentflow.safety_privacy.v1",
        "generated_at": utc_now(),
        "status": "warning" if any(row["severity"] != "info" for row in warnings) else "ok",
        "highest_severity": worst,
        "summary": {
            "warning_count": len([row for row in warnings if row["severity"] != "info"]),
            "info_count": len([row for row in warnings if row["severity"] == "info"]),
            "proxy_loopback": proxy_loopback,
            "body_logging_enabled": log_bodies_enabled,
            "managed_communication_enabled": bool(recommendation_enabled or policy_bundle_url),
            "managed_auth_configured": managed_auth_configured,
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
                "recommendation_server": recommendation_state,
                "policy_bundle_recommendation": policy_bundle_state,
                "auth_configured": managed_auth_configured,
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
    from agentflow_proxy import cache, crunch, router, routing_experiments

    state = {
        "schema": "agentflow.policy_state.v1",
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
            "defaults": {
                "haiku": router.HAIKU_DEFAULT,
                "sonnet": router.SONNET_DEFAULT,
                "opus": router.OPUS_DEFAULT,
            },
            "openai": {
                "enabled": bool(router.OPENAI_ROUTING_ENABLED),
                "large": router.OPENAI_LARGE_DEFAULT,
                "small": router.OPENAI_SMALL_DEFAULT,
                "tiny": router.OPENAI_TINY_DEFAULT,
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
    file_backed_sections = ("routing", "crunch", "cache", "routing_experiments")
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
    return state


async def stats_policy_events(limit: int = 50) -> dict[str, Any]:
    from agentflow_proxy.policy_events import recent_policy_events

    return recent_policy_events(limit=limit)


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


def estimate_tokens_from_text_chars(chars: Any) -> int:
    char_count = max(_as_int(chars), 0)
    if char_count <= 0:
        return 0
    return max(1, int(char_count / TOKEN_CHARS))


def _codex_turn_estimates(input_text_chars: Any, result_chars: Any) -> dict[str, Any]:
    input_tokens = estimate_tokens_from_text_chars(input_text_chars)
    output_tokens = estimate_tokens_from_text_chars(result_chars)
    cost = estimate_cost(
        CODEX_APP_MODEL,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=CODEX_APP_PROCESSING_MODE,
    )
    cost_known = cost is not None
    cost_value = float(cost) if cost_known else None
    return {
        "model": CODEX_APP_MODEL,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": input_tokens + output_tokens,
        "cost_est_usd": cost_value,
        "baseline_cost_est_usd": cost_value,
        "hard_floor_usd": cost_value,
        "cost_basis": CODEX_APP_COST_BASIS,
        "pricing_basis": CODEX_APP_PRICING_BASIS,
        "cost_known": cost_known,
        "cost_estimated": cost_known,
    }


def _codex_estimates_with_cache(input_text_chars: Any, result_chars: Any, cache: dict[str, Any]) -> dict[str, Any]:
    estimates = _codex_turn_estimates(input_text_chars, result_chars)
    if cache.get("status") == "hit":
        baseline = float(estimates["baseline_cost_est_usd"] or estimates["cost_est_usd"] or 0.0)
        estimates["cost_est_usd"] = 0.0
        estimates["hard_floor_usd"] = 0.0
        estimates["baseline_cost_est_usd"] = baseline
        estimates["cache_savings_usd"] = baseline
        estimates["cost_known"] = True
        estimates["cost_estimated"] = True
    else:
        estimates["cache_savings_usd"] = 0.0
    return estimates


def _codex_not_applied_decision(kind: str) -> dict[str, Any]:
    return {
        "status": "not-applied",
        "reason": CODEX_APP_TELEMETRY_ONLY_REASON,
        "policy_source": "local-default",
        "surface": CODEX_APP_SOURCE_SURFACE,
        "decision_type": kind,
        "applied": False,
    }


def _codex_turn_risk_features(row: dict[str, Any]) -> dict[str, Any]:
    input_items = _as_int(row.get("input_items"))
    input_text_chars = _as_int(row.get("input_text_chars"))
    params_chars = _as_int(row.get("params_chars"))
    method = str(row.get("method") or "turn/start")
    raw_prompt_logging_enabled = os.getenv("AGENTFLOW_LOG_BODIES", "0") == "1"
    return {
        "mutation_safe": False,
        "mutation_safe_reason": CODEX_APP_TELEMETRY_ONLY_REASON,
        "method": method,
        "params_shape": {
            "has_params": params_chars > 0,
            "params_chars": params_chars,
            "has_input": input_items > 0 or input_text_chars > 0,
            "input_items": input_items,
            "input_text_chars": input_text_chars,
        },
        "tool_or_approval_hints": {
            "captured": False,
            "tool_use_present": None,
            "approval_required": None,
            "reason": "raw-params-not-stored",
        },
        "raw_prompt_logging_enabled": raw_prompt_logging_enabled,
        "raw_prompt_stored": False,
        "raw_response_stored": False,
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


def _legacy_cache_decision(row: dict[str, Any]) -> dict[str, str]:
    status_code = _as_int(row.get("status_code"))
    source_surface = canonical_source_surface(
        row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
    )
    if _as_int(row.get("cache_hit")):
        return {
            "status": "hit",
            "reason": "legacy-cache-hit",
            "hit_type": "exact",
            "policy_source": "legacy-inferred",
            "source_surface": source_surface,
        }
    if _as_int(row.get("stream")):
        return {
            "status": "skipped",
            "reason": "legacy-streaming",
            "hit_type": "",
            "policy_source": "legacy-inferred",
            "source_surface": source_surface,
        }
    if status_code >= 400:
        return {
            "status": "skipped",
            "reason": "legacy-upstream-error",
            "hit_type": "",
            "policy_source": "legacy-inferred",
            "source_surface": source_surface,
        }
    return {
        "status": "missing",
        "reason": "legacy-unknown",
        "hit_type": "",
        "policy_source": "legacy-inferred",
        "source_surface": source_surface,
    }


def _cache_decision_for_breakdown(row: dict[str, Any]) -> dict[str, str]:
    cache = _json_obj(row.get("cache_json"))
    if cache:
        policy_source = str(cache.get("policy_source") or "unknown")
        source_surface = canonical_source_surface(
            cache.get("surface") or row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        )
        if not cache.get("status") and not cache.get("reason"):
            legacy_hit_type = str(cache.get("hit_type") or "")
            if legacy_hit_type == "skip-streaming":
                return {
                    "status": "skipped",
                    "reason": "legacy-streaming",
                    "hit_type": "",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                }
            if legacy_hit_type == "miss":
                return {
                    "status": "miss",
                    "reason": "legacy-exact-miss",
                    "hit_type": "",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                }
            if legacy_hit_type == "hit":
                return {
                    "status": "hit",
                    "reason": "legacy-cache-hit",
                    "hit_type": "exact",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                }
            return {
                "status": "missing",
                "reason": "legacy-partial-cache-json",
                "hit_type": legacy_hit_type,
                "policy_source": policy_source,
                "source_surface": source_surface,
            }
        return {
            "status": str(cache.get("status") or "missing"),
            "reason": str(cache.get("reason") or "unknown"),
            "hit_type": str(cache.get("hit_type") or ""),
            "policy_source": policy_source,
            "source_surface": source_surface,
        }
    return _legacy_cache_decision(row)


def _cache_decision_breakdown(rows: list[dict[str, Any]], *, today_only: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        decision = _cache_decision_for_breakdown(row)
        key = (
            decision["source_surface"],
            decision["status"],
            decision["reason"],
            decision["hit_type"],
            decision["policy_source"],
        )
        bucket = grouped.setdefault(
            key,
            {
                "source_surface": key[0],
                "status": key[1],
                "reason": key[2],
                "hit_type": key[3],
                "policy_source": key[4],
                "count": 0,
            },
        )
        bucket["count"] += 1

    breakdown = list(grouped.values())
    breakdown.sort(key=lambda r: r["count"], reverse=True)
    return breakdown


def _size_bucket(value: Any) -> str:
    n = _as_int(value)
    if n <= 0:
        return "0"
    if n < 2_000:
        return "1_2k"
    if n < 8_000:
        return "2k_8k"
    if n < 32_000:
        return "8k_32k"
    if n < 128_000:
        return "32k_128k"
    return "128k_plus"


def _short_session_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:8] if text else None


def _cache_replayability_blockers(unit: dict[str, Any]) -> list[str]:
    cache = unit["cache"]
    reason = str(unit["cache_reason"] or "")
    category = str(unit.get("category") or "")
    blockers: set[str] = set()
    if unit.get("stream") or "streaming" in reason:
        blockers.add("streaming")
    if "tools-disabled" in reason or (unit.get("has_tools") and not bool(cache.get("tool_cache_enabled"))):
        blockers.add("tool-call-disabled")
    if (
        not bool(cache.get("semantic_enabled"))
        and (
            unit["cache_status"] == "skipped"
            or "semantic" in reason
            or "cache-disabled" in reason
        )
    ):
        blockers.add("semantic-cache-disabled")
    tool_like = bool(unit.get("has_tools")) or category.startswith("tool")
    if tool_like and not bool(cache.get("file_watch_enabled")):
        blockers.add("file-dependency-unknown")
    if is_codex_turn_source_surface(str(unit.get("source_surface") or "")):
        blockers.add("turn-level-only")
    if unit["cache_status"] == "missing":
        blockers.add("missing-cache-metadata")
    return sorted(blockers)


def _cache_replayability_unit(row: dict[str, Any], *, source_surface: str, granularity: str) -> dict[str, Any] | None:
    decision = _cache_decision_for_breakdown({**row, "source_surface": source_surface})
    status = str(decision.get("status") or "missing")
    if status == "hit":
        return None
    cache = _json_obj(row.get("cache_json"))
    routing = _json_obj(row.get("routing_json"))
    text_chars = _as_int(row.get("text_chars") if row.get("text_chars") is not None else routing.get("text_chars"))
    input_tokens = _as_int(row.get("input_tokens") if row.get("input_tokens") is not None else row.get("input_tokens_est"))
    if not text_chars and input_tokens:
        text_chars = input_tokens * TOKEN_CHARS
    category = str(row.get("category") or routing.get("category") or "unknown")
    requested_model = str(row.get("requested_model") or "")
    routed_model = str(row.get("routed_model") or requested_model)
    replayability_level = str(cache.get("replayability_level") or ("features_only" if granularity == "agent_turn" else "metadata_shape"))
    unit = {
        "source_surface": canonical_source_surface(source_surface),
        "granularity": granularity,
        "created_at": row.get("created_at"),
        "session_id": row.get("session_id"),
        "stream": bool(_as_int(row.get("stream"))),
        "cache_status": status,
        "cache_reason": str(decision.get("reason") or "unknown"),
        "hit_type": str(decision.get("hit_type") or ""),
        "policy_source": str(decision.get("policy_source") or "unknown"),
        "category": category,
        "requested_tier": model_tier(requested_model) if requested_model else "unknown",
        "target_tier": model_tier(routed_model) if routed_model else "unknown",
        "has_tools": bool(routing.get("has_tools") or category.startswith("tool")),
        "eligible": bool(cache.get("eligible")),
        "replayability_level": replayability_level,
        "text_size_bucket": _size_bucket(text_chars),
        "input_items_bucket": _size_bucket(row.get("input_items")),
        "cost_est_usd": _as_float(row.get("cost_est_usd")),
        "baseline_cost_usd": _as_float(row.get("cost_baseline_usd")),
        "input_tokens": input_tokens,
        "text_chars": text_chars,
        "cache": cache,
    }
    unit["blockers"] = _cache_replayability_blockers(unit)
    return unit


def _cache_replayability_fingerprint_basis(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_surface": unit["source_surface"],
        "granularity": unit["granularity"],
        "cache_status": unit["cache_status"],
        "cache_reason": unit["cache_reason"],
        "category": unit["category"],
        "stream": unit["stream"],
        "has_tools": unit["has_tools"],
        "requested_tier": unit["requested_tier"],
        "target_tier": unit["target_tier"],
        "text_size_bucket": unit["text_size_bucket"],
        "input_items_bucket": unit["input_items_bucket"],
        "replayability_level": unit["replayability_level"],
        "eligible": unit["eligible"],
    }


def _cache_replayability_fingerprint(unit: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    basis = _cache_replayability_fingerprint_basis(unit)
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], basis


def _cache_replayability_report_from_units(units: list[dict[str, Any]], *, limit: int = 25) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    one_off_rows = 0
    for unit in units:
        fingerprint, basis = _cache_replayability_fingerprint(unit)
        bucket = grouped.setdefault(
            fingerprint,
            {
                "shape_fingerprint": fingerprint,
                "fingerprint_basis": basis,
                "source_surface": unit["source_surface"],
                "granularity": unit["granularity"],
                "cache_status": unit["cache_status"],
                "cache_reason": unit["cache_reason"],
                "category": unit["category"],
                "text_size_bucket": unit["text_size_bucket"],
                "input_items_bucket": unit["input_items_bucket"],
                "requested_tier": unit["requested_tier"],
                "target_tier": unit["target_tier"],
                "stream": unit["stream"],
                "has_tools": unit["has_tools"],
                "eligible": unit["eligible"],
                "replayability_level": unit["replayability_level"],
                "policy_source": unit["policy_source"],
                "count": 0,
                "sessions": set(),
                "example_sessions": [],
                "estimated_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "input_tokens": 0,
                "text_chars": 0,
                "first_seen_at": unit.get("created_at"),
                "last_seen_at": unit.get("created_at"),
                "_blockers": set(unit.get("blockers") or []),
            },
        )
        bucket["count"] += 1
        session = str(unit.get("session_id") or "")
        if session:
            bucket["sessions"].add(session)
            short = _short_session_id(session)
            if short and short not in bucket["example_sessions"] and len(bucket["example_sessions"]) < 3:
                bucket["example_sessions"].append(short)
        bucket["estimated_cost_usd"] += _as_float(unit.get("cost_est_usd"))
        bucket["baseline_cost_usd"] += _as_float(unit.get("baseline_cost_usd"))
        bucket["input_tokens"] += _as_int(unit.get("input_tokens"))
        bucket["text_chars"] += _as_int(unit.get("text_chars"))
        for blocker in unit.get("blockers") or []:
            bucket["_blockers"].add(str(blocker))
        if str(unit.get("created_at") or "") < str(bucket.get("first_seen_at") or unit.get("created_at") or ""):
            bucket["first_seen_at"] = unit.get("created_at")
        if str(unit.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = unit.get("created_at")

    groups = []
    blocker_counts: dict[str, dict[str, Any]] = {}
    repeated_groups = 0
    repeated_rows = 0
    repeated_cost = 0.0
    unsafe_repeated_rows = 0
    unsafe_repeated_cost = 0.0
    for bucket in grouped.values():
        session_count = len(bucket["sessions"])
        blockers = set(bucket.pop("_blockers"))
        if session_count > 1:
            blockers.add("session-context-changed")
        if bucket["count"] == 1 and not blockers and bucket["cache_status"] == "miss":
            blockers.add("true-one-off-miss")
            one_off_rows += 1
        elif bucket["count"] == 1:
            one_off_rows += 1
        blocker_list = sorted(blockers)
        repeated = bucket["count"] > 1
        if repeated:
            repeated_groups += 1
            repeated_rows += bucket["count"]
            repeated_cost += float(bucket["estimated_cost_usd"])
            if blocker_list:
                unsafe_repeated_rows += bucket["count"]
                unsafe_repeated_cost += float(bucket["estimated_cost_usd"])
        for blocker in blocker_list or ["none"]:
            row = blocker_counts.setdefault(
                blocker,
                {"blocker": blocker, "groups": 0, "calls": 0, "estimated_cost_usd": 0.0},
            )
            row["groups"] += 1
            row["calls"] += bucket["count"]
            row["estimated_cost_usd"] += float(bucket["estimated_cost_usd"])
        finalized = {
            **bucket,
            "sessions": session_count,
            "repeated": repeated,
            "replayability_blockers": blocker_list,
            "estimated_cost_usd": round(float(bucket["estimated_cost_usd"]), 6),
            "baseline_cost_usd": round(float(bucket["baseline_cost_usd"]), 6),
        }
        groups.append(finalized)

    groups.sort(key=lambda row: (row["count"], row["estimated_cost_usd"], str(row.get("last_seen_at") or "")), reverse=True)
    blocker_rows = [
        {
            **row,
            "estimated_cost_usd": round(float(row["estimated_cost_usd"]), 6),
        }
        for row in blocker_counts.values()
    ]
    blocker_rows.sort(key=lambda row: (row["calls"], row["estimated_cost_usd"]), reverse=True)
    return {
        "schema": "agentflow.cache_replayability.v1",
        "generated_at": utc_now(),
        "privacy": {
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "basis": "metadata-derived shape fingerprints only; request/response bodies are not inspected",
        },
        "summary": {
            "candidate_rows": len(units),
            "shape_groups": len(groups),
            "repeated_shape_groups": repeated_groups,
            "repeated_candidate_rows": repeated_rows,
            "one_off_candidate_rows": one_off_rows,
            "repeated_estimated_cost_usd": round(repeated_cost, 6),
            "unsafe_repeated_rows": unsafe_repeated_rows,
            "unsafe_repeated_estimated_cost_usd": round(unsafe_repeated_cost, 6),
            "no_repeated_shape_exists": repeated_groups == 0,
            "repeated_shape_exists_but_cache_is_unsafe": unsafe_repeated_rows > 0,
        },
        "blocker_breakdown": blocker_rows,
        "groups": groups[: max(1, int(limit or 25))],
    }


async def stats_cache_replayability(store_obj: Any, limit: int = 25) -> dict[str, Any]:
    conn = store_obj.conn
    provider_rows = [
        dict(row)
        for row in conn.execute("""
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, stream, cache_hit, status_code,
                   cache_json, routing_json, category, session_id,
                   input_tokens_est, actual_input_tokens as input_tokens,
                   cost_est_usd, cost_baseline_usd,
                   null as input_items,
                   null as text_chars
            from calls
            order by created_at desc
        """).fetchall()
    ]
    units: list[dict[str, Any]] = []
    for row in provider_rows:
        surface = _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))
        unit = _cache_replayability_unit(row, source_surface=surface, granularity="provider_request")
        if unit is not None:
            units.append(unit)

    codex_rows = [
        dict(row)
        for row in conn.execute("""
            select s.created_at,
                   s.session_id,
                   s.routing_json,
                   s.cache_json,
                   s.input_items,
                   s.input_text_chars as text_chars,
                   s.input_text_chars as input_tokens,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
        """).fetchall()
    ]
    for row in codex_rows:
        estimates = _codex_estimates_with_cache(row.get("text_chars"), row.get("response_result_chars"), _json_obj(row.get("cache_json")))
        prepared = {
            **row,
            "requested_model": CODEX_APP_MODEL,
            "routed_model": CODEX_APP_MODEL,
            "stream": 0,
            "cache_hit": 1 if _json_obj(row.get("cache_json")).get("status") == "hit" else 0,
            "status_code": None,
            "category": "codex-app-turn",
            "input_tokens": estimates.get("input_tokens_est"),
            "cost_est_usd": estimates.get("cost_est_usd"),
            "cost_baseline_usd": estimates.get("baseline_cost_est_usd"),
        }
        unit = _cache_replayability_unit(prepared, source_surface=CODEX_APP_SOURCE_SURFACE, granularity="agent_turn")
        if unit is not None:
            units.append(unit)

    return _cache_replayability_report_from_units(units, limit=limit)


def _managed_error_class(meta: dict[str, Any]) -> str | None:
    reason = str(meta.get("reason") or "")
    if reason:
        return reason
    status = str(meta.get("status") or "")
    if status in {"error", "invalid"}:
        return status
    error = _sanitize_error_sample(meta.get("error"), limit=240)
    if not error:
        return None
    head = error.split(":", 1)[0].strip()
    return head[:80] if head else "error"


def _managed_breakdown(grouped: dict[str, int]) -> list[dict[str, Any]]:
    rows = [{"value": key, "count": value} for key, value in grouped.items()]
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows


async def stats_managed_recommendations(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from agentflow_proxy.recommendations import recommendation_server_url, recommendations_enabled
    from agentflow_proxy.policy_events import recent_policy_events

    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 500), 5000))
    rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   routing_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    feedback_status_counts: dict[str, int] = {}
    feedback_reason_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    latency_values: list[int] = []
    feedback_latency_values: list[int] = []
    recent: list[dict[str, Any]] = []

    metadata_rows = 0
    historical_null_rows = 0
    enabled_count = 0
    disabled_count = 0
    received_count = 0
    applied_count = 0
    changed_model_count = 0
    server_error_count = 0
    invalid_count = 0
    feedback_present_count = 0
    feedback_sent_count = 0
    feedback_skipped_count = 0
    feedback_failed_count = 0
    feedback_sanitized_count = 0
    last_recommendation_error_class = None
    last_feedback_error_class = None

    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        managed = routing.get("managed_recommendation")
        if not isinstance(managed, dict) or not managed:
            historical_null_rows += 1
            if len(recent) < 50:
                recent.append({
                    "created_at": row.get("created_at"),
                    "provider": row.get("provider"),
                    "source_surface": _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or "")),
                    "requested_model": row.get("requested_model"),
                    "routed_model": row.get("routed_model"),
                    "status_code": row.get("status_code"),
                    "recommendation_status": "missing",
                    "recommendation_reason": "historical-null",
                    "recommendation_enabled": None,
                    "applied": False,
                    "changed_model": False,
                    "policy_id": None,
                    "target_model": None,
                    "fallback": None,
                    "latency_ms": None,
                    "feedback_status": "missing",
                    "feedback_reason": "historical-null",
                    "feedback_latency_ms": None,
                    "feedback_error_class": None,
                })
            continue

        metadata_rows += 1
        status = str(managed.get("status") or "missing")
        reason = str(managed.get("reason") or "unknown")
        _increment_count(status_counts, status)
        _increment_count(reason_counts, reason)
        if managed.get("enabled") is False:
            disabled_count += 1
        elif managed.get("enabled") is True:
            enabled_count += 1
        if status == "received":
            received_count += 1
        if bool(managed.get("applied")):
            applied_count += 1
        if bool(managed.get("changed_model")):
            changed_model_count += 1
        if status == "error" or reason == "server-error":
            server_error_count += 1
            if last_recommendation_error_class is None:
                last_recommendation_error_class = _managed_error_class(managed)
        if status == "invalid":
            invalid_count += 1
            if last_recommendation_error_class is None:
                last_recommendation_error_class = _managed_error_class(managed)
        fallback = managed.get("fallback")
        if fallback:
            _increment_count(fallback_counts, fallback)
        policy_id = managed.get("policy_id")
        if policy_id:
            _increment_count(policy_counts, policy_id)
        latency_ms = managed.get("latency_ms")
        if latency_ms is not None:
            latency_values.append(_as_int(latency_ms))

        feedback = managed.get("outcome_feedback")
        feedback_status = "missing"
        feedback_reason = "missing"
        feedback_error_class = None
        feedback_latency_ms = None
        if isinstance(feedback, dict) and feedback:
            feedback_present_count += 1
            feedback_sanitized_count += 1
            feedback_status = str(feedback.get("status") or "missing")
            feedback_reason = str(feedback.get("reason") or "unknown")
            _increment_count(feedback_status_counts, feedback_status)
            _increment_count(feedback_reason_counts, feedback_reason)
            if feedback_status == "sent":
                feedback_sent_count += 1
            elif feedback_status == "skipped":
                feedback_skipped_count += 1
            elif feedback_status in {"error", "invalid"}:
                feedback_failed_count += 1
                feedback_error_class = _managed_error_class(feedback)
                if last_feedback_error_class is None:
                    last_feedback_error_class = feedback_error_class
            feedback_latency_ms = feedback.get("latency_ms")
            if feedback_latency_ms is not None:
                feedback_latency_values.append(_as_int(feedback_latency_ms))
        else:
            _increment_count(feedback_status_counts, "missing")

        if len(recent) < 50:
            recent.append({
                "created_at": row.get("created_at"),
                "provider": row.get("provider"),
                "source_surface": _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or "")),
                "requested_model": row.get("requested_model"),
                "routed_model": row.get("routed_model"),
                "status_code": row.get("status_code"),
                "recommendation_status": status,
                "recommendation_reason": reason,
                "recommendation_enabled": managed.get("enabled"),
                "applied": bool(managed.get("applied")),
                "changed_model": bool(managed.get("changed_model")),
                "policy_id": managed.get("policy_id"),
                "target_model": managed.get("target_model"),
                "fallback": fallback,
                "latency_ms": _as_int(latency_ms) if latency_ms is not None else None,
                "feedback_status": feedback_status,
                "feedback_reason": feedback_reason,
                "feedback_latency_ms": _as_int(feedback_latency_ms) if feedback_latency_ms is not None else None,
                "feedback_error_class": feedback_error_class,
            })

    latest_fetch_review_health: dict[str, Any] | None = None
    for event in recent_policy_events(limit=100).get("events", []):
        if not isinstance(event, dict) or event.get("action") != "fetch-review":
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        health = details.get("recommendation_health")
        if isinstance(health, dict) and health:
            latest_fetch_review_health = {
                **health,
                "event_created_at": event.get("created_at"),
                "event_ok": bool(event.get("ok")),
            }
            break

    return {
        "schema": "agentflow.managed_recommendations.v1",
        "generated_at": utc_now(),
        "limit": capped_limit,
        "current_config": {
            "enabled": recommendations_enabled(),
            "server_url": recommendation_server_url(),
            "offline_state": (
                "managed recommendations disabled; local policy remains authoritative"
                if not recommendations_enabled()
                else "managed recommendation bridge enabled"
            ),
        },
        "privacy": {
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "basis": "stored recommendation and outcome-feedback metadata only",
        },
        "summary": {
            "window_calls": len(rows),
            "metadata_rows": metadata_rows,
            "historical_null_rows": historical_null_rows,
            "enabled_count": enabled_count,
            "disabled_count": disabled_count,
            "received_count": received_count,
            "applied_count": applied_count,
            "changed_model_count": changed_model_count,
            "server_error_count": server_error_count,
            "invalid_count": invalid_count,
            "fallback_count": sum(fallback_counts.values()),
            "avg_recommendation_latency_ms": _avg_or_none(latency_values),
            "feedback_present_count": feedback_present_count,
            "feedback_sent_count": feedback_sent_count,
            "feedback_skipped_count": feedback_skipped_count,
            "feedback_failed_count": feedback_failed_count,
            "feedback_sanitized_count": feedback_sanitized_count,
            "avg_feedback_latency_ms": _avg_or_none(feedback_latency_values),
            "last_recommendation_error_class": last_recommendation_error_class,
            "last_feedback_error_class": last_feedback_error_class,
            "policy_id_count": len(policy_counts),
        },
        "status_breakdown": _managed_breakdown(status_counts),
        "reason_breakdown": _managed_breakdown(reason_counts),
        "fallback_breakdown": _managed_breakdown(fallback_counts),
        "policy_ids": _managed_breakdown(policy_counts),
        "feedback_status_breakdown": _managed_breakdown(feedback_status_counts),
        "feedback_reason_breakdown": _managed_breakdown(feedback_reason_counts),
        "recommendation_health": {
            "latest_fetch_review": latest_fetch_review_health,
            "rows": (
                latest_fetch_review_health.get("rows", [])
                if isinstance(latest_fetch_review_health, dict)
                else []
            ),
        },
        "recent": recent,
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


def _codex_crunch_pattern_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        crunch = _json_obj(row.get("crunch_json"))
        patterns = crunch.get("codex_patterns")
        if not isinstance(patterns, list):
            codex_meta = crunch.get("codex_repeated_scaffolding")
            patterns = codex_meta.get("patterns") if isinstance(codex_meta, dict) else []
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            pattern_type = str(pattern.get("type") or "unknown")
            bucket = grouped.setdefault(
                pattern_type,
                {"type": pattern_type, "turns": 0, "count": 0, "saved_chars_est": 0},
            )
            bucket["turns"] += 1
            bucket["count"] += _as_int(pattern.get("count"))
            bucket["saved_chars_est"] += _as_int(pattern.get("saved_chars_est"))
    result = list(grouped.values())
    result.sort(key=lambda r: (r["saved_chars_est"], r["count"]), reverse=True)
    return result


_CODEX_DECISION_KEYS = ("routing_json", "crunch_json", "cache_json")


def _codex_decision_metadata_state(row: dict[str, Any]) -> str:
    present = sum(1 for key in _CODEX_DECISION_KEYS if _json_obj_has_value(row.get(key)))
    if present == len(_CODEX_DECISION_KEYS):
        return "complete"
    if present:
        return "not-instrumented"
    if _json_obj_has_value(row.get("event_window_json")):
        return "current-missing"
    return "historical-unavailable"


def _codex_missing_decision(decision_key: str, metadata_state: str) -> dict[str, Any]:
    decision_name = decision_key.replace("_json", "")
    if metadata_state == "historical-unavailable":
        return {
            "status": "historical-unavailable",
            "reason": f"{decision_name}-decision-metadata-historical-unavailable",
            "applied": False,
            "eligible": False,
            "policy_source": "unknown",
        }
    if metadata_state == "not-instrumented":
        return {
            "status": "not-instrumented",
            "reason": f"{decision_name}-decision-metadata-not-instrumented",
            "applied": False,
            "eligible": False,
            "policy_source": "unknown",
        }
    return {
        "status": "missing",
        "reason": f"{decision_name}-decision-metadata-current-missing",
        "applied": False,
        "eligible": False,
        "policy_source": "unknown",
    }


def _codex_normalized_decision(row: dict[str, Any], decision_key: str, metadata_state: str) -> dict[str, Any]:
    decision = _json_obj(row.get(decision_key))
    if decision:
        return decision
    return _codex_missing_decision(decision_key, metadata_state)


def _codex_model_field_state(routing: dict[str, Any], event_window_raw: Any = None) -> tuple[str, str | None]:
    field = routing.get("model_field")
    if field:
        return "present", str(field)
    reason = str(routing.get("reason") or "")
    if routing.get("requested_model") or routing.get("routed_model"):
        return "present_unknown_field", None
    event_window = _json_obj(event_window_raw)
    window_state = str(event_window.get("model_field_state") or "")
    if window_state in {"derived_present", "derived_absent"}:
        window_field = event_window.get("model_field")
        return window_state, str(window_field) if window_field else None
    model_state = event_window.get("model_state")
    if isinstance(model_state, dict):
        state = str(model_state.get("state") or "")
        if state in {"derived_present", "derived_absent"}:
            field = model_state.get("field")
            return state, str(field) if field else None
    if reason == "codex-turn-start-model-field-absent":
        return "absent", None
    return "unknown", None


def _codex_param_shape_category(row: dict[str, Any], routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any]) -> str:
    reasons = {
        str(decision.get("reason") or "")
        for decision in (routing, crunch, cache)
        if isinstance(decision, dict)
    }
    if "action-like-params" in reasons:
        return "action-like-params"
    if "unknown-param-shape" in reasons:
        return "unknown-param-shape"
    if "non-text-input" in reasons:
        return "non-text-input"
    if "params-not-object" in reasons:
        return "params-not-object"
    if "codex-app-cache-disabled" in reasons:
        if _as_int(row.get("input_text_chars")) > 0:
            return "text-input-cache-disabled"
        return "cache-disabled-unknown-shape"
    if _as_int(row.get("input_text_chars")) > 0:
        return "text-input"
    if _as_int(row.get("params_chars")) > 0:
        return "params-without-text"
    return "unknown"


def _codex_phase_signal(method: Any) -> str | None:
    method_l = str(method or "").replace("_", "").replace("-", "").lower()
    if not method_l:
        return None
    if method_l in {"initialize", "threadstart", "threadconfigure"}:
        return "idle_control"
    if "commandexecution" in method_l or "toolcall" in method_l or "toolresult" in method_l:
        return "tool_execution"
    if "diff" in method_l or "patch" in method_l:
        return "verification"
    if "plan" in method_l:
        return "planning"
    if "agentmessage" in method_l or "message/delta" in str(method or "").lower():
        return "summary"
    return None


def _codex_phase_from_signal_counts(
    signal_counts: dict[str, int],
    signal_methods: dict[str, list[str]],
    *,
    reason_prefix: str,
    source: str,
) -> dict[str, Any] | None:
    priority = ("tool_execution", "verification", "planning", "summary", "idle_control")
    for phase in priority:
        if signal_counts.get(phase):
            return {
                "phase": phase,
                "reason": f"{reason_prefix}:{phase}",
                "source": source,
                "signals": signal_methods.get(phase, [])[:5],
            }
    return None


def _codex_signal_counts_from_method_counts(method_counts: Any) -> tuple[dict[str, int], dict[str, list[str]]]:
    signal_counts: dict[str, int] = {}
    signal_methods: dict[str, list[str]] = {}
    if not isinstance(method_counts, dict):
        return signal_counts, signal_methods
    for method, count_raw in method_counts.items():
        signal = _codex_phase_signal(method)
        if not signal:
            continue
        count = _as_int(count_raw)
        if count <= 0:
            count = 1
        signal_counts[signal] = signal_counts.get(signal, 0) + count
        methods = signal_methods.setdefault(signal, [])
        method_s = str(method or "")
        if method_s and method_s not in methods:
            methods.append(method_s)
    return signal_counts, signal_methods


def _codex_public_event_window(raw: Any) -> dict[str, Any]:
    window = _json_obj(raw)
    if not window:
        return {}
    public: dict[str, Any] = {
        "schema": window.get("schema"),
        "event_count": _as_int(window.get("event_count")),
        "method_counts": dict(window.get("method_counts") or {}) if isinstance(window.get("method_counts"), dict) else {},
        "direction_counts": dict(window.get("direction_counts") or {}) if isinstance(window.get("direction_counts"), dict) else {},
        "first_event_delta_ms": _as_int(window.get("first_event_delta_ms")),
        "last_event_delta_ms": _as_int(window.get("last_event_delta_ms")),
        "input_items": window.get("input_items"),
        "input_text_chars": _as_int(window.get("input_text_chars")),
        "start_message_chars": _as_int(window.get("start_message_chars")),
        "start_params_chars": _as_int(window.get("start_params_chars")),
        "result_chars": _as_int(window.get("result_chars")),
        "server_message_chars": _as_int(window.get("server_message_chars")),
        "error_count": _as_int(window.get("error_count")),
        "model_field_state": window.get("model_field_state") or "unknown",
        "model_field": window.get("model_field"),
        "model_state": dict(window.get("model_state") or {}) if isinstance(window.get("model_state"), dict) else {},
        "request_id_present": bool(window.get("request_id")),
        "thread_id_present": bool(window.get("thread_id")),
        "session_id_present": bool(window.get("session_id")),
    }
    signal_counts, signal_methods = _codex_signal_counts_from_method_counts(public["method_counts"])
    public["phase_signal_counts"] = dict(signal_counts)
    public["phase_signal_methods"] = {
        phase: methods[:5]
        for phase, methods in signal_methods.items()
    }
    return public


def _codex_same_scope(event: dict[str, Any], row: dict[str, Any]) -> bool:
    row_session = str(row.get("session_id") or "")
    event_session = str(event.get("session_id") or "")
    row_thread = str(row.get("thread_id") or "")
    event_thread = str(event.get("thread_id") or "")
    if row_thread and event_thread:
        return row_thread == event_thread
    if row_session and event_session:
        return row_session == event_session
    return False


def _codex_turn_bounds(turn_rows: list[dict[str, Any]]) -> dict[str, str | None]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in turn_rows:
        key = (str(row.get("session_id") or ""), str(row.get("thread_id") or ""))
        grouped.setdefault(key, []).append(row)
    bounds: dict[str, str | None] = {}
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda item: str(item.get("created_at") or ""))
        for index, row in enumerate(ordered):
            next_row = ordered[index + 1] if index + 1 < len(ordered) else None
            bounds[str(row.get("start_event_id"))] = str(next_row.get("created_at")) if next_row else None
    return bounds


def _codex_workflow_phase(
    row: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    next_start_at: str | None,
    routing: dict[str, Any],
    crunch: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    for decision in (routing, crunch, cache):
        if isinstance(decision, dict) and decision.get("workflow_phase"):
            return {
                "phase": str(decision.get("workflow_phase") or "unknown"),
                "reason": str(decision.get("workflow_phase_reason") or "decision-metadata"),
                "source": "decision_metadata",
                "signals": list(decision.get("workflow_phase_signals") or []),
            }

    event_window = _json_obj(row.get("event_window_json"))
    if event_window:
        signal_counts, signal_methods = _codex_signal_counts_from_method_counts(event_window.get("method_counts"))
        phase = _codex_phase_from_signal_counts(
            signal_counts,
            signal_methods,
            reason_prefix="event-window-signal",
            source="event_window",
        )
        if phase:
            return phase

    start_at = str(row.get("created_at") or "")
    scoped: list[dict[str, Any]] = []
    for event in events:
        event_at = str(event.get("created_at") or "")
        if event_at < start_at:
            continue
        if next_start_at and event_at >= next_start_at:
            continue
        if _codex_same_scope(event, row):
            scoped.append(event)

    signal_counts: dict[str, int] = {}
    signal_methods: dict[str, list[str]] = {}
    for event in scoped:
        signal = _codex_phase_signal(event.get("method"))
        if not signal:
            continue
        _increment_count(signal_counts, signal)
        methods = signal_methods.setdefault(signal, [])
        method = str(event.get("method") or "")
        if method and method not in methods:
            methods.append(method)

    phase = _codex_phase_from_signal_counts(
        signal_counts,
        signal_methods,
        reason_prefix="event-method-signal",
        source="event_sequence",
    )
    if phase:
        return phase

    reasons = {
        str(decision.get("reason") or "")
        for decision in (routing, crunch, cache)
        if isinstance(decision, dict)
    }
    if "action-like-params" in reasons:
        return {
            "phase": "tool_execution",
            "reason": "decision-reason:action-like-params",
            "source": "decision_metadata",
            "signals": ["action-like-params"],
        }
    if _as_int(row.get("input_text_chars")) <= 0 and _as_int(row.get("params_chars")) > 0:
        return {
            "phase": "idle_control",
            "reason": "params-without-text-input",
            "source": "size_metadata",
            "signals": [],
        }
    return {
        "phase": "unknown",
        "reason": "insufficient-metadata",
        "source": "metadata_only_classifier",
        "signals": [],
    }


def _new_codex_phase_bucket(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "turns": 0,
        "completed": 0,
        "errors": 0,
        "pending": 0,
        "input_text_chars": 0,
        "result_chars": 0,
        "input_tokens_est": 0,
        "output_tokens_est": 0,
        "total_tokens_est": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_est_usd": 0.0,
        "hard_floor_usd": 0.0,
        "cost_known_turns": 0,
        "routing_applied": 0,
        "crunch_applied": 0,
        "cache_hits": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "latency_values": [],
        "reason_counts": {},
        "signal_methods": {},
    }


def _finalize_codex_phase_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    latency_values = list(bucket.pop("latency_values", []))
    reason_counts = dict(bucket.pop("reason_counts", {}))
    signal_methods = dict(bucket.pop("signal_methods", {}))
    turns = _as_int(bucket.get("turns"))
    errors = _as_int(bucket.get("errors"))
    bucket["error_rate"] = round(errors / turns, 4) if turns else 0
    bucket["avg_latency_ms"] = _avg_or_none(latency_values)
    bucket["phase_reasons"] = _count_breakdown(reason_counts)
    bucket["signal_methods"] = [
        {"method": method, "count": count}
        for method, count in sorted(signal_methods.items(), key=lambda item: item[1], reverse=True)
    ]
    return bucket


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


def _codex_plateau_scope(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("thread_id"):
        return str(row["thread_id"]), "thread_id"
    if row.get("session_id"):
        return str(row["session_id"]), "session_id"
    if row.get("request_id"):
        return f"request:{row['request_id']}", "request_id"
    return "unknown", "unknown"


def _codex_original_session_key(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("thread_id"):
        return str(row["thread_id"]), "thread_id"
    if row.get("session_id"):
        return str(row["session_id"]), "session_id"
    if row.get("request_id"):
        return f"request:{row['request_id']}", "request_id"
    return "codex:unknown", "unknown"


def _codex_metadata_workflow_groups(
    rows: list[dict[str, Any]],
    *,
    idle_gap_seconds: int = 30 * 60,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: str(item.get("created_at") or ""))
    groups_by_event: dict[str, dict[str, Any]] = {}
    group_index = 0
    current_group: dict[str, Any] | None = None
    previous_at: datetime | None = None

    def new_window(row: dict[str, Any], started_at: datetime | None) -> dict[str, Any]:
        nonlocal group_index
        group_index += 1
        started_text = started_at.isoformat() if started_at else str(row.get("created_at") or "unknown")
        model_state, _model_field = _codex_model_field_state(
            _json_obj(row.get("routing_json")),
            row.get("event_window_json"),
        )
        digest = hashlib.sha256(
            f"codex_turn|codex|{started_text}|{model_state}|{group_index}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "key": f"codex-workflow:{digest}",
            "basis": "workflow_window",
            "group_start_at": started_text,
            "group_index": group_index,
            "idle_gap_seconds": idle_gap_seconds,
            "model_state_counts": {},
            "original_key_basis_counts": {},
            "original_key_count": 0,
            "_original_keys": set(),
        }

    for row in ordered:
        created_at = _parse_utc_datetime(row.get("created_at"))
        thread_id = str(row.get("thread_id") or "").strip()
        if thread_id:
            digest = hashlib.sha256(f"codex_turn|codex|thread_id|{thread_id}".encode("utf-8")).hexdigest()[:16]
            current = {
                "key": f"codex-workflow:{digest}",
                "basis": "workflow_thread_id",
                "group_start_at": None,
                "group_index": None,
                "idle_gap_seconds": idle_gap_seconds,
                "model_state_counts": {},
                "original_key_basis_counts": {},
                "original_key_count": 0,
                "_original_keys": set(),
            }
        else:
            gap_seconds = (
                (created_at - previous_at).total_seconds()
                if created_at is not None and previous_at is not None
                else None
            )
            if current_group is None or (gap_seconds is not None and gap_seconds > idle_gap_seconds):
                current_group = new_window(row, created_at)
            current = current_group

        model_state, _model_field = _codex_model_field_state(
            _json_obj(row.get("routing_json")),
            row.get("event_window_json"),
        )
        original_key, original_basis = _codex_original_session_key(row)
        current["model_state_counts"][model_state] = current["model_state_counts"].get(model_state, 0) + 1
        current["original_key_basis_counts"][original_basis] = (
            current["original_key_basis_counts"].get(original_basis, 0) + 1
        )
        current["_original_keys"].add(f"{original_basis}:{original_key}")
        current["original_key_count"] = len(current["_original_keys"])
        event_id = str(row.get("start_event_id") or row.get("id") or row.get("request_id") or "")
        if event_id:
            groups_by_event[event_id] = current
        if not thread_id and created_at is not None:
            previous_at = created_at

    public: dict[str, dict[str, Any]] = {}
    for event_id, group in groups_by_event.items():
        public[event_id] = {
            "key": group["key"],
            "basis": group["basis"],
            "group_start_at": group.get("group_start_at"),
            "group_index": group.get("group_index"),
            "idle_gap_seconds": group["idle_gap_seconds"],
            "model_state_counts": dict(group.get("model_state_counts") or {}),
            "original_key_basis_counts": dict(group.get("original_key_basis_counts") or {}),
            "original_key_count": int(group.get("original_key_count") or 0),
        }
    return public


def _codex_phase_from_decision_metadata(
    routing: dict[str, Any],
    crunch: dict[str, Any],
    cache: dict[str, Any],
) -> str:
    for meta in (cache, crunch, routing):
        phase = meta.get("workflow_phase") if isinstance(meta, dict) else None
        if phase:
            return str(phase)
    return "unknown"


def _codex_meaningful_crunch(row: dict[str, Any], *, min_ratio: float) -> bool:
    input_chars = _as_int(row.get("input_text_chars"))
    if input_chars <= 0:
        return False
    return (_as_int(row.get("saved_chars")) / input_chars) >= min_ratio


def _codex_plateau_candidate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    min_input_chars = 8_000
    max_delta_ratio = 0.03
    min_candidate_pairs = 2
    meaningful_crunch_ratio = 0.05
    conservative_opportunity_ratio = 0.10

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        scope_id, scope_basis = _codex_plateau_scope(row)
        groups.setdefault((scope_basis, scope_id), []).append(row)

    candidates: list[dict[str, Any]] = []
    total_plateau_pairs = 0
    total_candidate_pairs = 0
    total_opportunity_chars = 0
    for (scope_basis, scope_id), scoped_rows in groups.items():
        ordered = sorted(scoped_rows, key=lambda item: str(item.get("created_at") or ""))
        input_values = [_as_int(row.get("input_text_chars")) for row in ordered]
        large_values = [value for value in input_values if value >= min_input_chars]
        if len(large_values) < min_candidate_pairs + 1:
            continue

        plateau_pairs = 0
        candidate_pairs = 0
        repeated_chars_est = 0
        candidate_repeated_chars_est = 0
        method_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        cache_status_counts: dict[str, int] = {}
        crunch_status_counts: dict[str, int] = {}
        for row in ordered:
            _increment_count(phase_counts, row.get("workflow_phase") or "unknown")
            _increment_count(cache_status_counts, row.get("cache_status") or "missing")
            _increment_count(crunch_status_counts, row.get("crunch_status") or "missing")
            event_window = _json_obj(row.get("event_window_json"))
            window_methods = event_window.get("method_counts") if isinstance(event_window, dict) else None
            if isinstance(window_methods, dict):
                for method, count in window_methods.items():
                    method_counts[str(method or "unknown")] = method_counts.get(str(method or "unknown"), 0) + max(1, _as_int(count))
            else:
                _increment_count(method_counts, row.get("method") or "turn/start")

        for previous, current in zip(ordered, ordered[1:]):
            previous_chars = _as_int(previous.get("input_text_chars"))
            current_chars = _as_int(current.get("input_text_chars"))
            if previous_chars < min_input_chars or current_chars < min_input_chars:
                continue
            delta_ratio = abs(current_chars - previous_chars) / max(previous_chars, 1)
            if delta_ratio > max_delta_ratio:
                continue
            plateau_pairs += 1
            repeated_chars = min(previous_chars, current_chars)
            repeated_chars_est += repeated_chars
            cache_hit = previous.get("cache_status") == "hit" or current.get("cache_status") == "hit"
            meaningful_crunch = (
                _codex_meaningful_crunch(previous, min_ratio=meaningful_crunch_ratio)
                or _codex_meaningful_crunch(current, min_ratio=meaningful_crunch_ratio)
            )
            if not cache_hit and not meaningful_crunch:
                candidate_pairs += 1
                candidate_repeated_chars_est += repeated_chars

        total_plateau_pairs += plateau_pairs
        total_candidate_pairs += candidate_pairs
        if candidate_pairs < min_candidate_pairs:
            continue

        current_saved_chars = sum(_as_int(row.get("saved_chars")) for row in ordered)
        current_saved_tokens = sum(_as_int(row.get("tokens_saved_est")) for row in ordered)
        opportunity_chars = max(
            int(candidate_repeated_chars_est * conservative_opportunity_ratio) - current_saved_chars,
            0,
        )
        opportunity_tokens = estimate_tokens_from_text_chars(opportunity_chars)
        opportunity_cost = estimate_cost(
            CODEX_APP_MODEL,
            opportunity_tokens,
            0,
            provider="openai",
            processing_mode=CODEX_APP_PROCESSING_MODE,
        )
        total_opportunity_chars += opportunity_chars
        candidates.append({
            "candidate_id": f"codex-context-plateau:{scope_basis}:{scope_id[:24]}",
            "scope_id": scope_id,
            "sid": scope_id[:8] if scope_id else None,
            "scope_basis": scope_basis,
            "turns": len(ordered),
            "large_turns": len(large_values),
            "plateau_count": plateau_pairs,
            "plateau_pairs": plateau_pairs,
            "candidate_pairs": candidate_pairs,
            "median_input_chars": _median_int(large_values),
            "p90_input_chars": _percentile_int(large_values, 0.9),
            "min_input_chars": min(large_values),
            "max_input_chars": max(large_values),
            "current_saved_chars": current_saved_chars,
            "current_saved_tokens_est": current_saved_tokens,
            "estimated_repeated_chars": repeated_chars_est,
            "estimated_candidate_repeated_chars": candidate_repeated_chars_est,
            "estimated_opportunity_saved_chars": opportunity_chars,
            "estimated_opportunity_tokens": opportunity_tokens,
            "estimated_opportunity_usd": round(float(opportunity_cost or 0.0), 6) if opportunity_cost is not None else None,
            "opportunity_basis": "10pct-of-unoptimized-adjacent-large-plateau-chars-minus-current-saved-chars",
            "cache_status_counts": _count_breakdown(cache_status_counts),
            "crunch_status_counts": _count_breakdown(crunch_status_counts),
            "workflow_phase_counts": _count_breakdown(phase_counts),
            "method_counts": [
                {"method": method, "count": count}
                for method, count in sorted(method_counts.items(), key=lambda item: item[1], reverse=True)[:10]
            ],
        })

    candidates.sort(
        key=lambda item: (
            item["estimated_opportunity_saved_chars"],
            item["candidate_pairs"],
            item["p90_input_chars"],
        ),
        reverse=True,
    )
    candidates = candidates[:20]
    return {
        "policy": {
            "min_input_chars": min_input_chars,
            "max_adjacent_delta_ratio": max_delta_ratio,
            "min_candidate_pairs": min_candidate_pairs,
            "meaningful_crunch_ratio": meaningful_crunch_ratio,
            "conservative_opportunity_ratio": conservative_opportunity_ratio,
            "privacy_basis": "metadata-only input sizes, decision status, event-window method counts, and scope IDs",
        },
        "summary": {
            "scopes_considered": len(groups),
            "plateau_pairs": total_plateau_pairs,
            "candidate_pairs": total_candidate_pairs,
            "candidate_count": len(candidates),
            "estimated_opportunity_saved_chars": total_opportunity_chars,
            "estimated_opportunity_tokens": estimate_tokens_from_text_chars(total_opportunity_chars),
        },
        "candidates": candidates,
    }


async def stats_codex_effectiveness(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 500), 5000))
    rows = conn.execute("""
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
               s.event_window_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
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
               ) as response_latency_ms,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars
        from codex_app_events s
        where s.direction = 'client_to_server'
          and s.method = 'turn/start'
        order by s.created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    turn_rows = [dict(row) for row in rows]
    min_start_at = min((str(row.get("created_at") or "") for row in turn_rows), default="")
    event_scan_limit = max(15000, min(200000, capped_limit * 1000))
    event_rows = [
        dict(row)
        for row in conn.execute(
            """
            select created_at, direction, method, request_id, thread_id, session_id
            from codex_app_events
            where created_at >= ?
            order by created_at asc
            limit ?
            """,
            (min_start_at or "0000-00-00T00:00:00+00:00", event_scan_limit),
        ).fetchall()
    ] if turn_rows else []
    events_by_thread: dict[str, list[dict[str, Any]]] = {}
    events_by_session_without_thread: dict[str, list[dict[str, Any]]] = {}
    events_by_session: dict[str, list[dict[str, Any]]] = {}
    for event in event_rows:
        session_key = str(event.get("session_id") or "")
        thread_key = str(event.get("thread_id") or "")
        if session_key:
            events_by_session.setdefault(session_key, []).append(event)
            if not thread_key:
                events_by_session_without_thread.setdefault(session_key, []).append(event)
        if thread_key:
            events_by_thread.setdefault(thread_key, []).append(event)
    turn_bounds = _codex_turn_bounds(turn_rows)

    model_field_counts: dict[str, int] = {}
    model_field_names: dict[str, int] = {}
    param_shape_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    phase_source_counts: dict[str, int] = {}
    phase_buckets: dict[str, dict[str, Any]] = {}
    optimized_latency: list[int] = []
    pass_through_latency: list[int] = []
    optimized_errors = 0
    pass_through_errors = 0
    optimized_count = 0
    pass_through_count = 0
    pending_count = 0
    error_count = 0
    success_count = 0
    total_saved_chars = 0
    total_saved_tokens = 0
    total_codex_scaffolding_saved_chars = 0
    action_like_skips = 0
    unknown_param_skips = 0
    non_text_skips = 0
    decision_metadata_counts: dict[str, int] = {}
    current_missing_decision_counts: dict[str, int] = {}
    not_instrumented_decision_counts: dict[str, int] = {}
    historical_unavailable_decision_counts: dict[str, int] = {}
    managed_status_counts: dict[str, int] = {}
    managed_feedback_status_counts: dict[str, int] = {}
    managed_feedback_reason_counts: dict[str, int] = {}
    managed_feedback_queue_counts: dict[str, int] = {}
    if hasattr(store_obj, "managed_outcome_feedback_summary"):
        try:
            for row in store_obj.managed_outcome_feedback_summary(source_surface=CODEX_APP_SOURCE_SURFACE):
                managed_feedback_queue_counts[str(row.get("status") or "unknown")] = _as_int(row.get("count"))
        except Exception:
            managed_feedback_queue_counts = {}

    recent_samples: list[dict[str, Any]] = []
    plateau_candidate_rows: list[dict[str, Any]] = []
    for row in turn_rows:
        metadata_state = _codex_decision_metadata_state(row)
        row["decision_metadata_state"] = metadata_state
        _increment_count(decision_metadata_counts, metadata_state)
        routing = _codex_normalized_decision(row, "routing_json", metadata_state)
        crunch = _codex_normalized_decision(row, "crunch_json", metadata_state)
        cache = _codex_normalized_decision(row, "cache_json", metadata_state)
        row["routing_json_normalized"] = routing
        row["crunch_json_normalized"] = crunch
        row["cache_json_normalized"] = cache
        for decision_key, decision in (
            ("routing", routing),
            ("crunch", crunch),
            ("cache", cache),
        ):
            status = str(decision.get("status") or "")
            if status == "missing":
                _increment_count(current_missing_decision_counts, decision_key)
            elif status == "not-instrumented":
                _increment_count(not_instrumented_decision_counts, decision_key)
            elif status == "historical-unavailable":
                _increment_count(historical_unavailable_decision_counts, decision_key)
        managed = routing.get("managed_recommendation") if isinstance(routing, dict) else None
        feedback = managed.get("outcome_feedback") if isinstance(managed, dict) else None
        if isinstance(managed, dict):
            _increment_count(managed_status_counts, managed.get("status") or "unknown")
            if isinstance(feedback, dict):
                _increment_count(managed_feedback_status_counts, feedback.get("status") or "unknown")
                _increment_count(managed_feedback_reason_counts, feedback.get("reason") or "unknown")
            else:
                _increment_count(managed_feedback_status_counts, "pending")
        model_state, model_field = _codex_model_field_state(routing, row.get("event_window_json"))
        _increment_count(model_field_counts, model_state)
        if model_field:
            _increment_count(model_field_names, model_field)
        shape = _codex_param_shape_category(row, routing, crunch, cache)
        _increment_count(param_shape_counts, shape)
        _increment_count(method_counts, row.get("method") or "turn/start")
        session_key = str(row.get("session_id") or "")
        thread_key = str(row.get("thread_id") or "")
        if thread_key:
            phase_events = list(events_by_thread.get(thread_key, []))
            if session_key:
                phase_events.extend(events_by_session_without_thread.get(session_key, []))
        elif session_key:
            phase_events = events_by_session.get(session_key, [])
        else:
            phase_events = event_rows
        phase_meta = _codex_workflow_phase(
            row,
            events=phase_events,
            next_start_at=turn_bounds.get(str(row.get("start_event_id"))),
            routing=routing,
            crunch=crunch,
            cache=cache,
        )
        phase = str(phase_meta.get("phase") or "unknown")
        _increment_count(phase_counts, phase)
        _increment_count(phase_source_counts, phase_meta.get("source") or "unknown")

        reasons = {
            str(decision.get("reason") or "")
            for decision in (routing, crunch, cache)
            if isinstance(decision, dict)
        }
        if "action-like-params" in reasons:
            action_like_skips += 1
        if "unknown-param-shape" in reasons:
            unknown_param_skips += 1
        if "non-text-input" in reasons:
            non_text_skips += 1

        saved_chars = _as_int(crunch.get("saved_chars"))
        saved_tokens = _as_int(crunch.get("tokens_saved_est"))
        total_saved_chars += saved_chars
        total_saved_tokens += saved_tokens
        codex_scaffolding = crunch.get("codex_repeated_scaffolding")
        if isinstance(codex_scaffolding, dict):
            total_codex_scaffolding_saved_chars += _as_int(codex_scaffolding.get("saved_chars"))

        optimized = bool(routing.get("applied") or crunch.get("applied") or cache.get("status") == "hit")
        has_response = bool(row.get("response_event_id"))
        has_error = row.get("response_error_code") is not None
        latency = _as_int(row.get("response_latency_ms"))
        result_chars = _as_int(row.get("response_result_chars"))
        if has_error:
            error_count += 1
        elif has_response:
            success_count += 1
        else:
            pending_count += 1

        if optimized:
            optimized_count += 1
            if has_error:
                optimized_errors += 1
            if latency:
                optimized_latency.append(latency)
        else:
            pass_through_count += 1
            if has_error:
                pass_through_errors += 1
            if latency:
                pass_through_latency.append(latency)

        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), result_chars, cache)
        phase_bucket = phase_buckets.setdefault(phase, _new_codex_phase_bucket(phase))
        phase_bucket["turns"] += 1
        phase_bucket["input_text_chars"] += _as_int(row.get("input_text_chars"))
        phase_bucket["result_chars"] += result_chars
        phase_bucket["input_tokens_est"] += _as_int(estimates.get("input_tokens_est"))
        phase_bucket["output_tokens_est"] += _as_int(estimates.get("output_tokens_est"))
        phase_bucket["total_tokens_est"] += _as_int(estimates.get("total_tokens_est"))
        phase_bucket["cost_est_usd"] += _as_float(estimates.get("cost_est_usd"))
        phase_bucket["baseline_cost_est_usd"] += _as_float(estimates.get("baseline_cost_est_usd"))
        phase_bucket["hard_floor_usd"] += _as_float(estimates.get("hard_floor_usd"))
        if estimates.get("cost_known"):
            phase_bucket["cost_known_turns"] += 1
        if routing.get("applied"):
            phase_bucket["routing_applied"] += 1
        if crunch.get("applied"):
            phase_bucket["crunch_applied"] += 1
        if cache.get("status") == "hit":
            phase_bucket["cache_hits"] += 1
        phase_bucket["saved_chars"] += saved_chars
        phase_bucket["tokens_saved_est"] += saved_tokens
        if has_error:
            phase_bucket["errors"] += 1
        elif has_response:
            phase_bucket["completed"] += 1
        else:
            phase_bucket["pending"] += 1
        if latency:
            phase_bucket["latency_values"].append(latency)
        _increment_count(phase_bucket["reason_counts"], phase_meta.get("reason") or "unknown")
        for signal_method in phase_meta.get("signals") or []:
            _increment_count(phase_bucket["signal_methods"], signal_method)

        if len(recent_samples) < 20:
            recent_samples.append({
                "created_at": row.get("created_at"),
                "method": row.get("method") or "turn/start",
                "workflow_phase": phase,
                "workflow_phase_reason": phase_meta.get("reason"),
                "workflow_phase_source": phase_meta.get("source"),
                "workflow_phase_signals": phase_meta.get("signals") or [],
                "event_window": _codex_public_event_window(row.get("event_window_json")),
                "decision_metadata_state": metadata_state,
                "model_field": model_state,
                "param_shape": shape,
                "routing_status": routing.get("status") or "missing",
                "routing_reason": routing.get("reason") or "unknown",
                "crunch_status": crunch.get("status") or "missing",
                "crunch_reason": crunch.get("reason") or "unknown",
                "codex_pattern_types": [
                    str(pattern.get("type"))
                    for pattern in (crunch.get("codex_patterns") or [])
                    if isinstance(pattern, dict) and pattern.get("type")
                ],
                "cache_status": cache.get("status") or "missing",
                "cache_reason": cache.get("reason") or "unknown",
                "managed_recommendation_status": (managed or {}).get("status") if isinstance(managed, dict) else "missing",
                "managed_feedback_status": (feedback or {}).get("status") if isinstance(feedback, dict) else ("pending" if isinstance(managed, dict) else "missing"),
                "managed_feedback_reason": (feedback or {}).get("reason") if isinstance(feedback, dict) else None,
                "input_text_chars": _as_int(row.get("input_text_chars")),
                "saved_chars": saved_chars,
                "tokens_saved_est": saved_tokens,
                "outcome": "error" if has_error else ("success" if has_response else "pending"),
                "latency_ms": latency or None,
                "error_code": row.get("response_error_code"),
            })
        plateau_candidate_rows.append({
            "created_at": row.get("created_at"),
            "method": row.get("method") or "turn/start",
            "request_id": row.get("request_id"),
            "thread_id": row.get("thread_id"),
            "session_id": row.get("session_id"),
            "input_text_chars": _as_int(row.get("input_text_chars")),
            "saved_chars": saved_chars,
            "tokens_saved_est": saved_tokens,
            "workflow_phase": phase,
            "routing_status": routing.get("status") or "missing",
            "crunch_status": crunch.get("status") or "missing",
            "cache_status": cache.get("status") or "missing",
            "event_window_json": row.get("event_window_json"),
        })

    total = len(turn_rows)
    plateau_candidate_report = _codex_plateau_candidate_report(plateau_candidate_rows)
    return {
        "schema": "agentflow.codex_app_effectiveness.v1",
        "generated_at": utc_now(),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "limit": capped_limit,
        "privacy": {
            "raw_prompts_included": False,
            "raw_params_included": False,
            "raw_responses_included": False,
            "basis": "stored metadata, sizes, hashes, and decision JSON only",
        },
        "event_scan": {
            "events_considered": len(event_rows),
            "event_scan_limit": event_scan_limit,
            "truncated": bool(turn_rows) and len(event_rows) >= event_scan_limit,
        },
        "summary": {
            "turn_start_rows": total,
            "completed_rows": success_count,
            "error_rows": error_count,
            "pending_rows": pending_count,
            "model_field_present": (
                model_field_counts.get("present", 0)
                + model_field_counts.get("present_unknown_field", 0)
                + model_field_counts.get("derived_present", 0)
            ),
            "model_field_derived": model_field_counts.get("derived_present", 0),
            "model_field_absent": model_field_counts.get("absent", 0) + model_field_counts.get("derived_absent", 0),
            "model_field_unknown": model_field_counts.get("unknown", 0),
            "decision_metadata_complete_rows": decision_metadata_counts.get("complete", 0),
            "decision_metadata_historical_unavailable_rows": decision_metadata_counts.get("historical-unavailable", 0),
            "decision_metadata_not_instrumented_rows": decision_metadata_counts.get("not-instrumented", 0),
            "decision_metadata_current_missing_rows": decision_metadata_counts.get("current-missing", 0),
            "current_missing_decisions": sum(current_missing_decision_counts.values()),
            "not_instrumented_decisions": sum(not_instrumented_decision_counts.values()),
            "historical_unavailable_decisions": sum(historical_unavailable_decision_counts.values()),
            "routing_applied": sum(1 for row in turn_rows if _json_obj(row.get("routing_json_normalized")).get("applied")),
            "crunch_applied": sum(1 for row in turn_rows if _json_obj(row.get("crunch_json_normalized")).get("applied")),
            "cache_hits": sum(1 for row in turn_rows if _json_obj(row.get("cache_json_normalized")).get("status") == "hit"),
            "cache_eligible": sum(1 for row in turn_rows if bool(_json_obj(row.get("cache_json_normalized")).get("eligible"))),
            "action_like_skips": action_like_skips,
            "unknown_param_skips": unknown_param_skips,
            "non_text_input_skips": non_text_skips,
            "workflow_phase_known": total - phase_counts.get("unknown", 0),
            "workflow_phase_unknown": phase_counts.get("unknown", 0),
            "total_input_text_chars": sum(_as_int(row.get("input_text_chars")) for row in turn_rows),
            "total_saved_chars": total_saved_chars,
            "total_saved_tokens_est": total_saved_tokens,
            "codex_repeated_scaffolding_saved_chars": total_codex_scaffolding_saved_chars,
            "optimized_rows": optimized_count,
            "pass_through_rows": pass_through_count,
            "optimized_error_rate": round(optimized_errors / optimized_count, 4) if optimized_count else 0,
            "pass_through_error_rate": round(pass_through_errors / pass_through_count, 4) if pass_through_count else 0,
            "optimized_avg_latency_ms": _avg_or_none(optimized_latency),
            "pass_through_avg_latency_ms": _avg_or_none(pass_through_latency),
            "managed_recommendation_rows": sum(managed_status_counts.values()),
            "managed_recommendation_enabled": sum(
                1
                for row in turn_rows
                if bool((_json_obj(row.get("routing_json_normalized")).get("managed_recommendation") or {}).get("enabled"))
            ),
            "managed_recommendation_disabled": sum(
                1
                for row in turn_rows
                if isinstance(_json_obj(row.get("routing_json_normalized")).get("managed_recommendation"), dict)
                and not bool((_json_obj(row.get("routing_json_normalized")).get("managed_recommendation") or {}).get("enabled"))
            ),
            "managed_feedback_sent": managed_feedback_status_counts.get("sent", 0),
            "managed_feedback_skipped": (
                managed_feedback_status_counts.get("skipped", 0)
                + managed_feedback_status_counts.get("disabled", 0)
            ),
            "managed_feedback_queued": managed_feedback_status_counts.get("queued", 0),
            "managed_feedback_error": (
                managed_feedback_status_counts.get("error", 0)
                + managed_feedback_status_counts.get("retryable-error", 0)
                + managed_feedback_status_counts.get("dropped-after-limit", 0)
            ),
            "managed_feedback_retryable_error": managed_feedback_status_counts.get("retryable-error", 0),
            "managed_feedback_dropped_after_limit": managed_feedback_status_counts.get("dropped-after-limit", 0),
            "managed_feedback_pending": managed_feedback_status_counts.get("pending", 0),
            "managed_feedback_queue_sent": managed_feedback_queue_counts.get("sent", 0),
            "managed_feedback_queue_queued": managed_feedback_queue_counts.get("queued", 0),
            "managed_feedback_queue_error": (
                managed_feedback_queue_counts.get("retryable-error", 0)
                + managed_feedback_queue_counts.get("dropped-after-limit", 0)
            ),
            "repeated_context_plateau_candidate_count": plateau_candidate_report["summary"]["candidate_count"],
            "repeated_context_plateau_pairs": plateau_candidate_report["summary"]["plateau_pairs"],
            "repeated_context_plateau_opportunity_chars": plateau_candidate_report["summary"]["estimated_opportunity_saved_chars"],
        },
        "decision_metadata_breakdown": _count_breakdown(decision_metadata_counts),
        "current_missing_decision_breakdown": _count_breakdown(current_missing_decision_counts),
        "not_instrumented_decision_breakdown": _count_breakdown(not_instrumented_decision_counts),
        "historical_unavailable_decision_breakdown": _count_breakdown(historical_unavailable_decision_counts),
        "model_field_breakdown": _count_breakdown(model_field_counts),
        "model_field_names": _count_breakdown(model_field_names),
        "method_breakdown": _count_breakdown(method_counts),
        "param_shape_breakdown": _count_breakdown(param_shape_counts),
        "workflow_phase_breakdown": [
            _finalize_codex_phase_bucket(bucket)
            for bucket in sorted(phase_buckets.values(), key=lambda item: item["turns"], reverse=True)
        ],
        "workflow_phase_counts": _count_breakdown(phase_counts),
        "workflow_phase_source_breakdown": _count_breakdown(phase_source_counts),
        "routing_breakdown": _decision_breakdown(turn_rows, "routing_json"),
        "crunch_breakdown": _decision_breakdown(turn_rows, "crunch_json"),
        "crunch_pattern_breakdown": _codex_crunch_pattern_breakdown(turn_rows),
        "cache_breakdown": _decision_breakdown(turn_rows, "cache_json"),
        "managed_recommendation_breakdown": _count_breakdown(managed_status_counts),
        "managed_feedback_breakdown": _count_breakdown(managed_feedback_status_counts),
        "managed_feedback_reason_breakdown": _count_breakdown(managed_feedback_reason_counts),
        "managed_feedback_queue_breakdown": _count_breakdown(managed_feedback_queue_counts),
        "repeated_context_plateau_candidates": plateau_candidate_report,
        "outcome_by_optimization": [
            {
                "bucket": "optimized",
                "count": optimized_count,
                "errors": optimized_errors,
                "error_rate": round(optimized_errors / optimized_count, 4) if optimized_count else 0,
                "avg_latency_ms": _avg_or_none(optimized_latency),
            },
            {
                "bucket": "pass_through",
                "count": pass_through_count,
                "errors": pass_through_errors,
                "error_rate": round(pass_through_errors / pass_through_count, 4) if pass_through_count else 0,
                "avg_latency_ms": _avg_or_none(pass_through_latency),
            },
        ],
        "recent_samples": recent_samples,
    }


def _usage_bucket_identity(app_family: str, session_id: Any) -> dict[str, Any]:
    engineer = os.getenv("AGENTFLOW_ENGINEER") or None
    app = os.getenv("AGENTFLOW_APP") or app_family or "unknown"
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
            "engineer": "env:AGENTFLOW_ENGINEER" if engineer else None,
            "app": "env:AGENTFLOW_APP" if os.getenv("AGENTFLOW_APP") else "inferred_app_family",
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


def _provider_activity_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
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
    source_surface = _source_surface(provider, str(r.get("path") or ""))
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
    )
    return {
        "unit_id": f"provider_call:{r.get('id')}",
        "created_at": r.get("created_at"),
        "source_surface": source_surface,
        "granularity": "provider_request",
        "app_family": _app_family_for_call(provider, requested_model, str(r.get("path") or "")),
        "requested_model": requested_model,
        "target_model": target_model,
        "routed_model": routed_model,
        "input_features": {
            "path": r.get("path"),
            "stream": bool(r.get("stream")),
            "category": r.get("category") or routing.get("category"),
            "text_chars": routing.get("text_chars"),
            "input_tokens": input_tokens,
            "input_tokens_est": r.get("input_tokens_est"),
            "actual_input_tokens": r.get("actual_input_tokens"),
            "cache_creation_input_tokens": r.get("cache_creation_input_tokens") or 0,
            "cache_read_input_tokens": r.get("cache_read_input_tokens") or 0,
        },
        "tool_features": {
            "has_tools": routing.get("has_tools"),
            "category": r.get("category") or routing.get("category"),
            "thinking_history_stripped": routing.get("thinking_history_stripped"),
            "stripped_params": routing.get("stripped_params") or [],
        },
        "optimization_features": {
            "routing": routing,
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
        "local_ids": {
            "calls_id": r.get("id"),
            "session_id": r.get("session_id"),
        },
    }


def _codex_turn_activity_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    error_code = r.get("response_error_code")
    response_event_id = r.get("response_event_id")
    status = "error" if error_code is not None else ("success" if response_event_id else "pending")
    routing = _json_obj(r.get("routing_json")) or _codex_not_applied_decision("routing")
    crunch = _json_obj(r.get("crunch_json")) or _codex_not_applied_decision("crunch")
    cache = _json_obj(r.get("cache_json")) or _codex_not_applied_decision("cache")
    estimates = _codex_estimates_with_cache(r.get("input_text_chars"), r.get("response_result_chars"), cache)
    requested_model = routing.get("requested_model") or estimates["model"]
    target_model = routing.get("routed_model") or requested_model
    if crunch.get("tokens_before_est") is not None:
        baseline_input_tokens = _as_int(crunch.get("tokens_before_est"))
        baseline_output_tokens = estimates["output_tokens_est"]
        baseline_cost = estimate_cost(
            requested_model,
            baseline_input_tokens,
            baseline_output_tokens,
            provider="openai",
        )
        if baseline_cost is not None:
            estimates["baseline_cost_est_usd"] = float(baseline_cost)
            if cache.get("status") == "hit":
                estimates["cache_savings_usd"] = float(baseline_cost)
    risk = _codex_turn_risk_features(r)
    quality_signals = derive_codex_turn_quality_signals(
        created_at=r.get("created_at"),
        response_event_id=response_event_id,
        error_code=error_code,
        error_message=r.get("response_error_message"),
        latency_ms=r.get("response_latency_ms"),
        routing_meta=routing,
        crunch_meta=crunch,
        cache_meta=cache,
    )
    policy_sources = sorted({
        str(source)
        for source in (
            routing.get("policy_source"),
            routing.get("final_policy_source"),
            crunch.get("policy_source"),
            cache.get("policy_source"),
        )
        if source
    }) or ["local-default"]
    return {
        "schema": "agentflow.optimization_unit.v1",
        "unit_id": f"codex_turn:{r.get('start_event_id')}",
        "created_at": r.get("created_at"),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "app_family": "codex",
        "requested_model": requested_model,
        "target_model": target_model,
        "routed_model": routing.get("routed_model") if routing.get("applied") else None,
        "model_basis": "estimated",
        "input_features": {
            "category": "codex-app-turn",
            "input_text_chars": r.get("input_text_chars") or 0,
            "input_tokens_est": estimates["input_tokens_est"],
            "total_tokens_est": estimates["total_tokens_est"],
            "input_items": r.get("input_items") or 0,
            "params_chars": r.get("params_chars"),
            "message_chars": r.get("message_chars"),
            "cost_basis": estimates["cost_basis"],
        },
        "tool_features": {
            "method": "turn/start",
            "thread_id": r.get("thread_id"),
            "category": "codex-app-turn",
            "tool_or_approval_hints": risk["tool_or_approval_hints"],
            "mutation_safe": risk["mutation_safe"],
            "mutation_safe_reason": risk["mutation_safe_reason"],
        },
        "optimization_features": {
            "routing": routing,
            "crunch": crunch,
            "cache": cache,
            "policy_sources": policy_sources,
            "mutation_safe": risk["mutation_safe"],
            "mutation_safe_reason": risk["mutation_safe_reason"],
        },
        "risk_features": risk,
        "mutation_safe": risk["mutation_safe"],
        "outcome_features": {
            "status": status,
            "latency_ms": r.get("response_latency_ms"),
            "result_chars": r.get("response_result_chars"),
            "output_tokens_est": estimates["output_tokens_est"],
            "total_tokens_est": estimates["total_tokens_est"],
            "cost_est_usd": estimates["cost_est_usd"],
            "cost_baseline_usd": estimates["baseline_cost_est_usd"],
            "hard_floor_usd": estimates["hard_floor_usd"],
            "cache_savings_usd": estimates["cache_savings_usd"],
            "cost_basis": estimates["cost_basis"],
            "pricing_basis": estimates["pricing_basis"],
            "cost_known": estimates["cost_known"],
            "cost_estimated": estimates["cost_estimated"],
            "error_code": error_code,
            "error_message": r.get("response_error_message"),
            "quality_signals": quality_signals,
        },
        "quality_signals": quality_signals,
        "replayability_level": str(cache.get("replayability_level") or "features_only"),
        "local_ids": {
            "codex_app_start_event_id": r.get("start_event_id"),
            "codex_app_response_event_id": response_event_id,
            "request_id": r.get("request_id"),
            "thread_id": r.get("thread_id"),
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
    routing_savings = 0.0
    if routed_model and requested_model != routed_model:
        requested_cost = estimate_cost(
            str(requested_model or ""),
            base_input_tokens,
            output_tokens,
            provider=provider,
        ) or 0.0
        routed_cost = estimate_cost(
            str(routed_model or ""),
            base_input_tokens,
            output_tokens,
            provider=provider,
        ) or 0.0
        routing_savings = max(requested_cost - routed_cost, 0.0)

    crunch_tokens_saved = _as_int(crunch.get("tokens_saved_est"))
    summary = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else {}
    if summary:
        crunch_tokens_saved += _as_int(summary.get("tokens_saved_est"))
    crunch_gross = estimate_blended_input_savings(
        str(target_model or ""),
        tokens_saved=crunch_tokens_saved,
        input_tokens=base_input_tokens,
        cache_read_tokens=cache_read_tokens,
        provider=provider,
    ) or 0.0
    crunch_savings = max(crunch_gross - _as_float(summary.get("summary_cost_est_usd")), 0.0)

    cache_savings = 0.003 if _as_int(r.get("cache_hit")) else 0.0
    if cache_read_tokens:
        full_read_cost = estimate_cost(str(target_model or ""), cache_read_tokens, 0, provider=provider) or 0.0
        cached_read_input_tokens = cache_read_tokens if provider == "openai" else 0
        cached_read_cost = estimate_cost(
            str(target_model or ""),
            cached_read_input_tokens,
            0,
            cache_read=cache_read_tokens,
            provider=provider,
        ) or 0.0
        cache_savings += max(full_read_cost - cached_read_cost, 0.0)

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
        "hard_floor_usd": cost,
        "policy_sources": _policy_sources_from(routing, crunch, cache),
        "is_today": bool(r.get("is_today")),
    }


def _codex_accounting_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    unit = _codex_turn_activity_unit(row)
    input_features = unit["input_features"]
    outcome_features = unit["outcome_features"]
    optimization_features = unit["optimization_features"]
    cost = _as_float(outcome_features.get("cost_est_usd"))
    baseline = _as_float(outcome_features.get("cost_baseline_usd")) or cost
    cache_savings = _as_float(outcome_features.get("cache_savings_usd"))
    remaining_savings = max(baseline - cost - cache_savings, 0.0)
    routing_savings = remaining_savings if optimization_features["routing"].get("applied") else 0.0
    crunch_savings = 0.0
    if not routing_savings and optimization_features["crunch"].get("changed"):
        crunch_savings = remaining_savings
    return {
        "source_surface": unit["source_surface"],
        "granularity": unit["granularity"],
        "app_family": unit["app_family"],
        "session_id": unit["local_ids"].get("session_id"),
        "input_tokens": _as_int(input_features.get("input_tokens_est")),
        "output_tokens": _as_int(outcome_features.get("output_tokens_est")),
        "total_tokens": _as_int(outcome_features.get("total_tokens_est")),
        "token_basis": "estimated-from-chars",
        "cost_est_usd": cost,
        "cost_basis": str(outcome_features.get("cost_basis") or CODEX_APP_COST_BASIS),
        "baseline_cost_usd": baseline,
        "routing_savings_usd": routing_savings,
        "crunch_savings_usd": crunch_savings,
        "cache_savings_usd": cache_savings,
        "hard_floor_usd": _as_float(outcome_features.get("hard_floor_usd")),
        "policy_sources": list(optimization_features.get("policy_sources") or []),
        "is_today": bool(dict(row).get("is_today")),
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
    calls = conn.execute("select count(*) c from calls").fetchone()["c"]
    cache_hits = conn.execute("select count(*) c from calls where cache_hit = 1").fetchone()["c"]
    routed = conn.execute("select coalesce(provider, 'anthropic') as provider, requested_model, routed_model, count(*) c from calls group by coalesce(provider, 'anthropic'), requested_model, routed_model order by c desc limit 20").fetchall()
    recent = conn.execute("select coalesce(provider, 'anthropic') as provider, created_at, requested_model, routed_model, cache_hit, status_code, latency_ms, cost_est_usd from calls order by created_at desc limit 20").fetchall()
    return {
        "calls": calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": (cache_hits / calls) if calls else 0,
        "db": default_db,
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
    provider_units = [_provider_activity_unit(row) for row in provider_rows]

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
        "schema": "agentflow.optimization_activity.v1",
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
        "schema": "agentflow.quality_signal_report.v1",
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
                "local_ids": unit.get("local_ids"),
            }
            for unit in activity["units"]
        ],
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
        _add_accounting_to_usage_bucket(bucket, _provider_accounting_unit({**r, "is_today": True}))

        bucket["provider_calls"] += 1
        bucket["turns"] += 1
        bucket["provider_input_tokens"] += provider_input_tokens
        bucket["provider_output_tokens"] += output_tokens
        bucket["provider_total_tokens"] += provider_input_tokens + output_tokens
        bucket["spend_usd"] += cost
        bucket["baseline_provider_cost_usd"] += baseline
        bucket["captured_savings_usd"] += max(baseline - cost, 0.0)
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
        _add_accounting_to_usage_bucket(bucket, _codex_accounting_unit({**r, "is_today": True}))
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
        bucket["captured_savings_usd"] += codex_saved
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
        "schema": "agentflow.usage_by_owner.v1",
        "scope": "today",
        "grouping": {
            "priority": ["AGENTFLOW_ENGINEER", "AGENTFLOW_APP", "app_family", "session_id"],
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
    cache_read_by_model = q("""
        select coalesce(routed_model, requested_model) as model,
               sum(cache_read_input_tokens) as read_tok,
               sum(case when date(created_at) = date('now') then cache_read_input_tokens else 0 end) as today_read_tok,
               coalesce(provider, 'anthropic') as provider
        from calls where cache_read_input_tokens > 0
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in cache_read_by_model:
        full_cost = estimate_cost(row["model"], row["read_tok"], 0, provider=row["provider"]) or 0
        prompt_cache_savings += 0.90 * full_cost
        today_full_cost = estimate_cost(row["model"], row["today_read_tok"] or 0, 0, provider=row["provider"]) or 0
        today_prompt_cache_savings += 0.90 * today_full_cost

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
    today_crunching_net_savings = today_crunch_savings + (today_summary_savings - today_summary_extra_cost)
    crunching_net_savings = crunch_savings + (summary_savings - summary_extra_cost)
    today_savings_buckets = {
        "routing_usd": round(today_routing_savings, 6),
        "crunching_usd": round(max(0.0, today_crunching_net_savings), 6),
        "exact_local_cache_usd": round(today_cache_savings + today_codex_cache_savings, 6),
        "provider_exact_local_cache_usd": round(today_cache_savings, 6),
        "codex_app_exact_local_cache_usd": round(today_codex_cache_savings, 6),
        "provider_prompt_cache_discount_usd": round(today_prompt_cache_savings, 6),
    }
    savings_buckets = {
        "routing_usd": round(routing_savings, 6),
        "crunching_usd": round(max(0.0, crunching_net_savings), 6),
        "exact_local_cache_usd": round(cache_cost_saved + codex_cache_savings, 6),
        "provider_exact_local_cache_usd": round(cache_cost_saved, 6),
        "codex_app_exact_local_cache_usd": round(codex_cache_savings, 6),
        "provider_prompt_cache_discount_usd": round(prompt_cache_savings, 6),
    }
    today_total_savings = sum(float(value or 0.0) for value in today_savings_buckets.values())
    total_savings = sum(float(value or 0.0) for value in savings_buckets.values())
    today_observed_baseline = today_cost + today_total_savings
    observed_baseline = total_cost + total_savings
    today_calculated_spend = today_cost + today_codex_cost_est
    calculated_spend = total_cost + codex_cost_est
    today_observed_baseline_with_codex = today_observed_baseline + today_codex_cost_est
    observed_baseline_with_codex = observed_baseline + codex_cost_est
    today_hard_floor = today_calculated_spend
    hard_floor = calculated_spend
    executive_summary = {
        "schema": "agentflow.executive_summary.v1",
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
            "today_calculated_spend_usd": round(today_calculated_spend, 6),
            "calculated_spend_usd": round(calculated_spend, 6),
            "today_provider_spend_usd": round(today_cost, 6),
            "total_provider_spend_usd": round(total_cost, 6),
            "today_codex_app_estimated_spend_usd": round(today_codex_cost_est, 6),
            "codex_app_estimated_spend_usd": round(codex_cost_est, 6),
            "today_baseline_provider_cost_usd": round(today_observed_baseline, 6),
            "baseline_provider_cost_usd": round(observed_baseline, 6),
            "today_baseline_calculated_cost_usd": round(today_observed_baseline_with_codex, 6),
            "baseline_calculated_cost_usd": round(observed_baseline_with_codex, 6),
            "thinking_cost_today_usd": round(today_thinking_cost, 6),
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "savings": {
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
            "today_baseline_minus_feasible_savings_usd": round(today_observed_baseline_with_codex - today_total_savings, 6),
            "excludes_unknown_codex_app_cost": not today_codex_cost_known,
            "codex_app_cost_estimated": today_codex_cost_known,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "health": {
            "errors": errors,
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
        select created_at, stream, cache_hit, status_code, cache_json,
               path, coalesce(provider, 'anthropic') as provider,
               null as source_surface,
               (created_at >= ?) as is_today
        from calls
        union all
        select created_at, 0 as stream,
               case when json_extract(cache_json, '$.status') = 'hit' then 1 else 0 end as cache_hit,
               null as status_code,
               cache_json,
               'codex-app://turn/start' as path,
               'codex-app' as provider,
               ? as source_surface,
               (created_at >= ?) as is_today
        from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
    """, (today_start, CODEX_APP_SOURCE_SURFACE, today_start))
    cache_decision_breakdown = _cache_decision_breakdown(cache_rows)
    today_cache_decision_breakdown = _cache_decision_breakdown(cache_rows, today_only=True)
    cache_replayability = await stats_cache_replayability(store_obj, limit=20)

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

    routing_experiment_rows = q("""
        select requested_model,
               routed_model,
               coalesce(category, 'unknown') as category,
               coalesce(routing_reason, 'unknown') as routing_reason,
               count(*) as samples,
               sum(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then 1 else 0 end) as compared_samples,
               avg(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then output_similarity else null end) as avg_similarity,
               avg(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then passed_threshold else null end) as pass_rate,
               round(sum(coalesce(primary_cost_est_usd, 0)), 6) as primary_cost_usd,
               round(sum(coalesce(shadow_cost_est_usd, 0)), 6) as shadow_cost_usd,
               max(created_at) as last_sample_at
        from routing_experiments
        group by requested_model, routed_model, coalesce(category, 'unknown'), coalesce(routing_reason, 'unknown')
        order by samples desc, last_sample_at desc
        limit 20
    """)
    routing_experiment_summary = []
    for row in routing_experiment_rows:
        compared_samples = int(row["compared_samples"] or 0)
        avg_similarity = row["avg_similarity"]
        pass_rate = row["pass_rate"]
        confidence_score = 0.0
        if avg_similarity is not None and compared_samples > 0:
            confidence_score = float(avg_similarity) * min(1.0, compared_samples / ROUTING_EXPERIMENT_MIN_SAMPLES)
        row["compared_samples"] = compared_samples
        row["avg_similarity"] = round(float(avg_similarity), 6) if avg_similarity is not None else None
        row["pass_rate"] = round(float(pass_rate), 4) if pass_rate is not None else None
        row["confidence_score"] = round(confidence_score, 6)
        row["min_samples_for_confidence"] = ROUTING_EXPERIMENT_MIN_SAMPLES
        routing_experiment_summary.append(row)
    routing_experiment_samples = int(s("select count(*) from routing_experiments") or 0)
    routing_experiment_compared = int(s("""
        select count(*) from routing_experiments
        where primary_status_code < 400
          and shadow_status_code < 400
          and output_similarity is not null
    """) or 0)
    routing_experiment_avg_similarity = s("""
        select avg(output_similarity) from routing_experiments
        where primary_status_code < 400
          and shadow_status_code < 400
          and output_similarity is not null
    """)
    routing_experiment_feedback_status_counts: dict[str, int] = {}
    for row in q("select experiment_json from routing_experiments where experiment_json is not null"):
        try:
            experiment = json.loads(row["experiment_json"])
        except Exception:
            status = "invalid-json"
        else:
            feedback = experiment.get("managed_feedback") if isinstance(experiment, dict) else None
            status = str((feedback or {}).get("status") or "not-exported") if isinstance(feedback, dict) else "not-exported"
        routing_experiment_feedback_status_counts[status] = routing_experiment_feedback_status_counts.get(status, 0) + 1

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
            "routing_savings_usd": round(routing_savings, 6),
            "today_routing_savings_usd": round(today_routing_savings, 6),
            "cache_savings_usd": round(cache_cost_saved + codex_cache_savings, 6),
            "today_cache_savings_usd": round(today_cache_savings + today_codex_cache_savings, 6),
            "provider_cache_savings_usd": round(cache_cost_saved, 6),
            "today_provider_cache_savings_usd": round(today_cache_savings, 6),
            "codex_app_cache_savings_usd": round(codex_cache_savings, 6),
            "today_codex_app_cache_savings_usd": round(today_codex_cache_savings, 6),
            "total_savings_usd": round(routing_savings + cache_cost_saved + codex_cache_savings, 6),
            "avg_latency_ms": round(avg_latency),
            "routed_count": routed_count,
            "crunched_count": crunched_count,
            "crunch_chars_saved": crunch_chars_saved,
            "crunch_tokens_saved": int(crunch_tokens_saved),
            "crunch_savings_usd": round(crunch_savings, 6),
            "today_crunch_savings_usd": round(today_crunch_savings, 6),
            "avg_crunch_ratio": round(avg_crunch_ratio, 4),
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
            "routing_experiment_samples": routing_experiment_samples,
            "routing_experiment_compared_samples": routing_experiment_compared,
            "routing_experiment_avg_similarity": (
                round(float(routing_experiment_avg_similarity), 6)
                if routing_experiment_avg_similarity is not None else None
            ),
            "routing_experiment_feedback_status_counts": routing_experiment_feedback_status_counts,
        },
        "recent": recent,
        "routing_breakdown": routing_breakdown,
        "category_breakdown": category_breakdown,
        "cache_decision_breakdown": cache_decision_breakdown,
        "today_cache_decision_breakdown": today_cache_decision_breakdown,
        "cache_replayability": cache_replayability,
        "error_breakdown": error_breakdown,
        "today_error_breakdown": today_error_breakdown,
        "routing_experiment_summary": routing_experiment_summary,
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
        row["codex_cost_est_usd"] += cost
        row["codex_tokens_est"] += _as_int(estimates.get("total_tokens_est"))

    total_latency_sum = 0
    total_latency_count = 0
    days = []
    for day in day_keys:
        row = days_by_key[day]
        row["total_tokens"] = row["provider_tokens"] + row["codex_tokens_est"]
        if row["_latency_count"]:
            row["avg_latency_ms"] = round(row["_latency_sum"] / row["_latency_count"])
        total_latency_sum += _as_int(row.get("_latency_sum"))
        total_latency_count += _as_int(row.get("_latency_count"))
        row["cost_est_usd"] = round(row["cost_est_usd"], 6)
        row["cost_baseline_usd"] = round(row["cost_baseline_usd"], 6)
        row["codex_cost_est_usd"] = round(row["codex_cost_est_usd"], 6)
        row["savings_usd"] = round(max(row["cost_baseline_usd"] - row["cost_est_usd"], 0.0), 6)
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
        "provider_calls": sum(r["provider_calls"] for r in days),
        "codex_turns": sum(r["codex_turns"] for r in days),
        "total_units": sum(r["total_units"] for r in days),
        "provider_tokens": sum(r["provider_tokens"] for r in days),
        "codex_tokens_est": sum(r["codex_tokens_est"] for r in days),
        "total_tokens": sum(r["total_tokens"] for r in days),
        "codex_cost_est_usd": round(sum(r["codex_cost_est_usd"] or 0 for r in days), 6),
        "cost_basis": "provider-reported + codex-estimated-from-chars",
    }
    return {
        "generated_at": generated_at,
        "schema": "agentflow.weekly_activity.v1",
        "source_surfaces": ["anthropic_messages", "openai_responses", "openai_chat", CODEX_APP_SOURCE_SURFACE],
        "cost_basis": "provider-reported + codex-estimated-from-chars",
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


def dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgentFlow</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:ui-monospace,monospace;background:#0d1117;color:#c9d1d9;font-size:13px}
  a{color:#58a6ff;text-decoration:none}
  header{background:#161b22;border-bottom:1px solid #30363d;padding:14px 24px;display:flex;align-items:center;gap:16px}
  header h1{font-size:16px;font-weight:600;color:#f0f6fc}
  header .sub{color:#8b949e;font-size:12px}
  .dot{width:8px;height:8px;border-radius:50%;background:#3fb950;display:inline-block;margin-right:6px;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .cards{display:flex;gap:12px;padding:16px 24px;flex-wrap:wrap}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;min-width:150px;flex:1}
  .card .label{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
  .card .value{font-size:22px;font-weight:600;color:#f0f6fc}
  .card .sub{color:#8b949e;font-size:11px;line-height:1.35;margin-top:3px}
  .card.green .value{color:#3fb950}
  .card.yellow .value{color:#d29922}
  .card.blue .value{color:#58a6ff}
  .tabs{display:flex;padding:0 24px;border-bottom:1px solid #30363d}
  .tab-btn{background:none;border:none;border-bottom:2px solid transparent;color:#8b949e;cursor:pointer;font-family:inherit;font-size:13px;margin-bottom:-1px;padding:10px 16px}
  .tab-btn.active{border-bottom-color:#58a6ff;color:#f0f6fc}
  .tab-panel{display:none}
  .tab-panel.active{display:block}
  .section{padding:0 24px 24px}
  .section h2{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#8b949e;margin-bottom:10px;padding-top:4px}
  .table-tools{display:flex;gap:8px;align-items:center;margin:0 0 8px;max-width:420px}
  .table-filter{background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-family:inherit;font-size:12px;min-width:180px;padding:6px 8px;width:100%}
  .table-filter:focus{border-color:#58a6ff;outline:none}
  .table-clear{background:#161b22;border:1px solid #30363d;border-radius:6px;color:#8b949e;cursor:pointer;font-family:inherit;font-size:12px;padding:6px 8px}
  .table-clear:hover{color:#f0f6fc;border-color:#58a6ff}
  .table-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse}
  .activity-table{min-width:1080px}
  th{text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:6px 10px;border-bottom:1px solid #21262d;font-weight:400}
  th[data-sort-type]{cursor:pointer;user-select:none}
  th[data-sort-type]:hover{color:#f0f6fc}
  th.sort-asc,th.sort-desc{color:#58a6ff}
  th.sort-asc::after{content:" ^";color:#58a6ff}
  th.sort-desc::after{content:" v";color:#58a6ff}
  td{padding:6px 10px;border-bottom:1px solid #161b22;vertical-align:middle;white-space:nowrap}
  tr:hover td{background:#161b22}
  .badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:500}
  .badge.hit{background:#1a3a1f;color:#3fb950}
  .badge.miss{background:#1c1c1c;color:#8b949e}
  .badge.stream{background:#1a2a3a;color:#58a6ff}
  .badge.err{background:#3a1a1a;color:#f85149}
  .badge.routed{background:#2d2208;color:#d29922}
  .badge.crunched{background:#1a1a3a;color:#79c0ff}
  .badge.provider{background:#20242b;color:#c9d1d9}
  .model{max-width:160px;overflow:hidden;text-overflow:ellipsis;color:#c9d1d9}
  .model.downgraded{color:#d29922}
  .cost{color:#3fb950;font-variant-numeric:tabular-nums}
  .latency{color:#8b949e;font-variant-numeric:tabular-nums}
  .tokens{color:#8b949e;font-variant-numeric:tabular-nums}
  .ts{color:#8b949e;font-size:11px}
  .flags{white-space:normal;min-width:170px}
  .err-row td{background:#1a0a0a}
  .totals-row td{border-top:1px solid #30363d;font-weight:600}
  .savings{color:#3fb950;font-variant-numeric:tabular-nums}
  .baseline{color:#8b949e;font-variant-numeric:tabular-nums}
  #status{margin-left:auto;font-size:11px;color:#8b949e}
  .arrow{color:#8b949e;margin:0 3px}
  @media (max-width:700px){
    header{padding:12px;gap:8px;flex-wrap:wrap}
    .cards{padding:12px;gap:8px}
    .card{min-width:130px;padding:12px}
    .tabs{padding:0 12px;overflow-x:auto}
    .tab-btn{padding:10px 12px;white-space:nowrap}
    .section{padding:0 12px 18px}
    .table-tools{max-width:none}
  }
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <h1>AgentFlow</h1>
  <span class="sub">provider-aware proxy · cost reduction dashboard</span>
  <span id="status">loading...</span>
</header>

<div class="cards" id="cards">
  <div class="card"><div class="label">Tokens today</div><div class="value" id="c-tokens-today">—</div><div class="sub" id="c-tokens-sub">— provider split</div><div class="sub" id="c-tokens-codex">— Codex telemetry</div></div>
  <div class="card"><div class="label">Calculated spend</div><div class="value" id="c-spend">—</div><div class="sub" id="c-spend-sub">— total</div></div>
  <div class="card green"><div class="label">Savings</div><div class="value" id="c-savings">—</div><div class="sub" id="c-savings-sub">— buckets</div></div>
  <div class="card yellow"><div class="label">Hard floor</div><div class="value" id="c-floor">—</div><div class="sub" id="c-floor-sub">— baseline minus feasible savings</div></div>
  <div class="card blue"><div class="label">Ops health</div><div class="value" id="c-health">—</div><div class="sub" id="c-health-sub">— latency</div><div class="sub" id="c-health-cooldown">— cooldowns</div></div>
</div>

<div class="tabs">
  <button class="tab-btn" onclick="showTab('safety')">Safety</button>
  <button class="tab-btn active" onclick="showTab('activity')">Recent calls</button>
  <button class="tab-btn" onclick="showTab('usage')">Usage by app / engineer</button>
  <button class="tab-btn" onclick="showTab('weekly')">7-day stats</button>
  <button class="tab-btn" onclick="showTab('categories')">By category</button>
  <button class="tab-btn" onclick="showTab('cache')">Cache</button>
  <button class="tab-btn" onclick="showTab('errors')">Errors</button>
  <button class="tab-btn" onclick="showTab('limiter')">Limiter</button>
  <button class="tab-btn" onclick="showTab('policies')">Policies</button>
  <button class="tab-btn" onclick="showTab('managed')">Managed</button>
  <button class="tab-btn" onclick="showTab('sessions')">Sessions</button>
</div>

<div class="tab-panel" id="tab-safety">
<div class="section">
  <h2>Safety / privacy status</h2>
  <table data-table-id="safety-summary" data-filter-label="Filter safety status">
    <thead><tr>
      <th data-sort-type="text">Check</th><th data-sort-type="text">Status</th><th data-sort-type="text">Details</th>
    </tr></thead>
    <tbody id="safety-summary-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Configuration warnings</h2>
  <table data-table-id="safety-warnings" data-filter-label="Filter safety warnings">
    <thead><tr>
      <th data-sort-type="text">Severity</th><th data-sort-type="text">Code</th><th data-sort-type="text">Warning</th>
    </tr></thead>
    <tbody id="safety-warnings-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Managed feedback queue safety</h2>
  <table data-table-id="safety-managed-feedback" data-filter-label="Filter safety managed feedback">
    <thead><tr>
      <th data-sort-type="time">Due age</th><th data-sort-type="text">Surface</th><th data-sort-type="text">Status</th><th data-sort-type="number">Attempts</th><th data-sort-type="number">Unit</th><th data-sort-type="number">Status code</th><th data-sort-type="text">Error class</th><th data-sort-type="text">Payload</th>
    </tr></thead>
    <tbody id="safety-managed-feedback-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel active" id="tab-activity">
<div class="section">
  <h2>Recent calls</h2>
  <div class="table-wrap">
  <table class="activity-table" data-table-id="activity" data-filter-label="Filter recent calls">
    <thead><tr>
      <th data-sort-type="time">Time</th><th data-sort-type="text">Surface</th><th data-sort-type="text">Granularity</th><th data-sort-type="text">App family</th><th data-sort-type="text">Requested</th><th data-sort-type="text">Target</th><th data-sort-type="number">Input</th><th data-sort-type="number">Output / status</th><th data-sort-type="latency">Latency</th><th data-sort-type="text">Flags</th>
    </tr></thead>
    <tbody id="activity-tbody"></tbody>
  </table>
  </div>
</div>
</div>

<div class="tab-panel" id="tab-usage">
<div class="section">
  <h2>Usage by app / engineer</h2>
  <div class="table-wrap">
  <table class="activity-table" data-table-id="usage" data-filter-label="Filter usage buckets">
    <thead><tr>
      <th data-sort-type="text">Bucket</th><th data-sort-type="number">Turns</th><th data-sort-type="number">Provider calls</th><th data-sort-type="number">Codex turns</th><th data-sort-type="number">Tokens</th><th data-sort-type="money">Spend</th><th data-sort-type="money">Captured savings</th><th data-sort-type="money">Hard floor</th><th data-sort-type="percent">Optimized</th><th data-sort-type="number">Errors</th><th data-sort-type="text">Remaining saving potential</th><th data-sort-type="text">Cost basis</th>
    </tr></thead>
    <tbody id="usage-tbody"></tbody>
  </table>
  </div>
</div>
</div>

<div class="tab-panel" id="tab-weekly">
<div class="section">
  <h2>7-day activity statistics</h2>
  <table data-table-id="weekly" data-filter-label="Filter 7-day stats">
    <thead><tr>
      <th data-sort-type="timestamp">Date</th><th data-sort-type="number">Units</th><th data-sort-type="number">Provider calls</th><th data-sort-type="number">Codex turns</th><th data-sort-type="number">Success</th><th data-sort-type="number">Errors</th><th data-sort-type="number">Cache hits</th><th data-sort-type="number">Tokens</th><th data-sort-type="latency">Avg latency</th><th data-sort-type="money">Spend</th><th data-sort-type="money">Baseline</th><th data-sort-type="money">Savings</th><th data-sort-type="text">Cost basis</th>
    </tr></thead>
    <tbody id="weekly-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-categories">
<div class="section">
  <h2>Calls by request category</h2>
  <table data-table-id="categories" data-filter-label="Filter categories">
    <thead><tr>
      <th data-sort-type="text">Category</th><th data-sort-type="number">Calls</th><th data-sort-type="money">Cost</th><th data-sort-type="number">Routed</th>
    </tr></thead>
    <tbody id="cat-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-cache">
<div class="section">
  <h2>Skipped cache replayability</h2>
  <table data-table-id="cache-replayability" data-filter-label="Filter cache replayability groups">
    <thead><tr>
      <th data-sort-type="text">Shape</th><th data-sort-type="text">Surface</th><th data-sort-type="text">Reason</th><th data-sort-type="number">Calls</th><th data-sort-type="number">Sessions</th><th data-sort-type="money">Cost</th><th data-sort-type="text">Blockers</th><th data-sort-type="text">Basis</th>
    </tr></thead>
    <tbody id="cache-replayability-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Cache decisions today</h2>
  <table data-table-id="cache-today" data-filter-label="Filter cache decisions today">
    <thead><tr>
      <th data-sort-type="text">Surface</th><th data-sort-type="text">Status</th><th data-sort-type="text">Reason</th><th data-sort-type="text">Hit type</th><th data-sort-type="text">Policy source</th><th data-sort-type="number">Calls</th>
    </tr></thead>
    <tbody id="cache-today-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Cache decisions all time</h2>
  <table data-table-id="cache-all" data-filter-label="Filter all cache decisions">
    <thead><tr>
      <th data-sort-type="text">Surface</th><th data-sort-type="text">Status</th><th data-sort-type="text">Reason</th><th data-sort-type="text">Hit type</th><th data-sort-type="text">Policy source</th><th data-sort-type="number">Calls</th>
    </tr></thead>
    <tbody id="cache-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-errors">
<div class="section">
  <h2>Errors today</h2>
  <table data-table-id="errors-today" data-filter-label="Filter errors today">
    <thead><tr>
      <th data-sort-type="text">Type</th><th data-sort-type="number">Status</th><th data-sort-type="text">Provider</th><th data-sort-type="text">Tier</th><th data-sort-type="text">Requested</th><th data-sort-type="text">Routed</th><th data-sort-type="number">Calls</th><th data-sort-type="time">Last seen</th><th data-sort-type="text">Sample</th>
    </tr></thead>
    <tbody id="errors-today-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Errors all time</h2>
  <table data-table-id="errors-all" data-filter-label="Filter all errors">
    <thead><tr>
      <th data-sort-type="text">Type</th><th data-sort-type="number">Status</th><th data-sort-type="text">Provider</th><th data-sort-type="text">Tier</th><th data-sort-type="text">Requested</th><th data-sort-type="text">Routed</th><th data-sort-type="number">Calls</th><th data-sort-type="time">Last seen</th><th data-sort-type="text">Sample</th>
    </tr></thead>
    <tbody id="errors-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-limiter">
<div class="section">
  <h2>Tier limiter state</h2>
  <table data-table-id="limiter-state" data-filter-label="Filter limiter tiers">
    <thead><tr>
      <th data-sort-type="text">Tier</th><th data-sort-type="text">Status</th><th data-sort-type="latency">Remaining</th><th data-sort-type="time">Cooldown until</th><th data-sort-type="number">Slots</th><th data-sort-type="number">Queued</th><th data-sort-type="time">Last upstream 429</th>
    </tr></thead>
    <tbody id="limiter-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Recent rate limits</h2>
  <table data-table-id="limiter-recent" data-filter-label="Filter recent rate limits">
    <thead><tr>
      <th data-sort-type="time">Time</th><th data-sort-type="text">Tier</th><th data-sort-type="text">Provider</th><th data-sort-type="number">Status</th><th data-sort-type="number">Retries</th><th data-sort-type="latency">Latency</th><th data-sort-type="text">Source</th><th data-sort-type="text">Error</th>
    </tr></thead>
    <tbody id="limiter-recent-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-policies">
<div class="section">
  <h2>Policy reload summary</h2>
  <table data-table-id="policy-summary" data-filter-label="Filter policy summary">
    <thead><tr>
      <th data-sort-type="text">Status</th><th data-sort-type="number">Policies</th><th data-sort-type="number">Loaded files</th><th data-sort-type="number">Manual</th><th data-sort-type="number">Local default</th><th data-sort-type="text">Reload needed</th>
    </tr></thead>
    <tbody id="policy-summary-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Effective policy files</h2>
  <table data-table-id="policies" data-filter-label="Filter policy files">
    <thead><tr>
      <th data-sort-type="text">Policy</th><th data-sort-type="text">Status</th><th data-sort-type="text">Source</th><th data-sort-type="text">Rule path</th><th data-sort-type="text">Effective settings</th>
    </tr></thead>
    <tbody id="policies-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Routing rules</h2>
  <table data-table-id="routing-rules" data-filter-label="Filter routing rules">
    <thead><tr>
      <th data-sort-type="number">#</th><th data-sort-type="text">Conditions</th><th data-sort-type="text">Action</th>
    </tr></thead>
    <tbody id="routing-rules-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Recent policy events</h2>
  <table data-table-id="policy-events" data-filter-label="Filter policy events">
    <thead><tr>
      <th data-sort-type="time">Time</th><th data-sort-type="text">Action</th><th data-sort-type="text">Status</th><th data-sort-type="text">Source</th><th data-sort-type="text">Details</th>
    </tr></thead>
    <tbody id="policy-events-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-managed">
<div class="section">
  <h2>Managed recommendation status</h2>
  <table data-table-id="managed-summary" data-filter-label="Filter managed summary">
    <thead><tr>
      <th data-sort-type="text">Bridge</th><th data-sort-type="number">Window calls</th><th data-sort-type="number">Metadata</th><th data-sort-type="number">Disabled</th><th data-sort-type="number">Received</th><th data-sort-type="number">Applied</th><th data-sort-type="number">Errors</th><th data-sort-type="latency">Avg recommendation</th><th data-sort-type="latency">Avg feedback</th><th data-sort-type="text">Last error</th>
    </tr></thead>
    <tbody id="managed-summary-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Managed recommendation health</h2>
  <table data-table-id="managed-health" data-filter-label="Filter managed health">
    <thead><tr>
      <th data-sort-type="time">Fetched</th><th data-sort-type="text">Status</th><th data-sort-type="text">Kind</th><th data-sort-type="text">Candidate</th><th data-sort-type="text">Code</th><th data-sort-type="text">Evidence</th>
    </tr></thead>
    <tbody id="managed-health-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Managed feedback status</h2>
  <table data-table-id="managed-feedback" data-filter-label="Filter managed feedback">
    <thead><tr>
      <th data-sort-type="number">Sent</th><th data-sort-type="number">Skipped</th><th data-sort-type="number">Failed</th><th data-sort-type="number">Sanitized</th><th data-sort-type="number">Historical null</th><th data-sort-type="text">Feedback reasons</th><th data-sort-type="text">Fallback reasons</th><th data-sort-type="text">Policy IDs</th>
    </tr></thead>
    <tbody id="managed-feedback-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Managed feedback queue</h2>
  <table data-table-id="managed-feedback-queue" data-filter-label="Filter managed feedback queue">
    <thead><tr>
      <th data-sort-type="number">Queued</th><th data-sort-type="number">Due</th><th data-sort-type="number">Retryable errors</th><th data-sort-type="number">Dropped</th><th data-sort-type="number">Sent</th><th data-sort-type="time">Oldest due</th><th data-sort-type="time">Last sent</th><th data-sort-type="text">Sources</th>
    </tr></thead>
    <tbody id="managed-feedback-queue-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Recent managed recommendation rows</h2>
  <table class="activity-table" data-table-id="managed-recent" data-filter-label="Filter managed recommendation rows">
    <thead><tr>
      <th data-sort-type="time">Time</th><th data-sort-type="text">Surface</th><th data-sort-type="text">Requested</th><th data-sort-type="text">Routed</th><th data-sort-type="text">Recommendation</th><th data-sort-type="text">Policy</th><th data-sort-type="text">Target</th><th data-sort-type="latency">Latency</th><th data-sort-type="text">Feedback</th><th data-sort-type="text">Fallback</th>
    </tr></thead>
    <tbody id="managed-recent-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-sessions">
<div class="section">
  <h2>Sessions today</h2>
  <table data-table-id="sessions" data-filter-label="Filter sessions">
    <thead><tr>
      <th data-sort-type="text">Surface</th><th data-sort-type="text">App</th><th data-sort-type="text">Session</th><th data-sort-type="number">Units</th><th data-sort-type="number">Provider calls</th><th data-sort-type="number">Codex turns</th><th data-sort-type="money">Cost</th><th data-sort-type="number">Codex input</th><th data-sort-type="number">Codex output</th><th data-sort-type="text">Codex opt</th><th data-sort-type="number">Thinking</th><th data-sort-type="money">Thinking cost</th><th data-sort-type="number">Cache write</th><th data-sort-type="number">Cache read</th><th data-sort-type="number">Write/read</th><th data-sort-type="money">Write cost</th><th data-sort-type="money">Read saved</th><th data-sort-type="number">Payback</th><th data-sort-type="number">tool-result</th><th data-sort-type="number">tool-heavy</th><th data-sort-type="number">short-comp</th><th data-sort-type="number">code-gen</th><th data-sort-type="number">chat</th><th data-sort-type="number">other</th>
    </tr></thead>
    <tbody id="sess-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Context plateaus today</h2>
  <table data-table-id="context-plateaus" data-filter-label="Filter context plateaus">
    <thead><tr>
      <th data-sort-type="text">Surface</th><th data-sort-type="text">Session</th><th data-sort-type="number">Units</th><th data-sort-type="money">Cost</th><th data-sort-type="number">Plateau pairs</th><th data-sort-type="number">Median chars</th><th data-sort-type="number">P90 chars</th><th data-sort-type="money">Cache read saved</th><th data-sort-type="number">Crunch saved chars</th><th data-sort-type="text">Flag</th>
    </tr></thead>
    <tbody id="plateau-tbody"></tbody>
  </table>
</div>
</div>

<script>
function fmt(n,d=4){if(n==null)return'—';return'$'+n.toFixed(d)}
function fmtMs(n){if(n==null)return'—';return n<1000?n+'ms':(n/1000).toFixed(1)+'s'}
function fmtSec(n){if(n==null)return'—';return n<60?n.toFixed(1)+'s':(n/60).toFixed(1)+'m'}
function fmtTok(n){if(n==null)return'?';if(n>=1000000)return(n/1000000).toFixed(1)+'M';return n>=1000?(n/1000).toFixed(1)+'k':String(n)}
function fmtRatio(n){if(n==null)return'—';return n.toFixed(2)+'x'}
function until(ts){
  if(!ts)return'—';
  const d=Math.ceil((new Date(ts).getTime()-Date.now())/1000);
  if(isNaN(d))return'—';
  if(d<=0)return'now';
  return fmtSec(d);
}
function ago(ts){
  if(!ts)return'—';
  const d=Math.floor((Date.now()-new Date(ts).getTime())/1000);
  if(isNaN(d))return'—';
  if(d<60)return d+'s';if(d<3600)return Math.floor(d/60)+'m';
  if(d<86400)return Math.floor(d/3600)+'h';return Math.floor(d/86400)+'d';
}
function shortModel(m){
  if(!m)return'—';
  return m.replace('claude-','').replace(/-20\\d{6}$/,'');
}
function shortProvider(p){
  if(!p)return'—';
  return p==='anthropic'?'Claude':p.charAt(0).toUpperCase()+p.slice(1);
}
function esc(v){
  return String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function shortSurface(s){
  const labels={anthropic_messages:'Claude',openai_chat:'OpenAI chat',openai_responses:'OpenAI',codex_app_turn:'Codex turn',codex_turn:'Codex turn'};
  return labels[s]||s||'unknown';
}
function activityInput(unit){
  const f=unit.input_features||{};
  if(unit.granularity==='provider_request'){
    const tok=f.input_tokens??f.input_tokens_est;
    const cache=f.cache_read_input_tokens||0;
    const text=f.text_chars;
    const parts=[];
    if(tok!=null)parts.push(fmtTok(tok)+' tok');
    if(text!=null)parts.push(fmtTok(text)+' chars');
    if(cache)parts.push(fmtTok(cache)+' cached');
    return parts.join(' · ')||'—';
  }
  const parts=[];
  parts.push(fmtTok(f.input_text_chars||0)+' text chars');
  if(f.input_items!=null)parts.push((f.input_items||0)+' items');
  return parts.join(' · ');
}
function activityOutcome(unit){
  const o=unit.outcome_features||{};
  if(unit.granularity==='provider_request'){
    const status=o.status_code??'—';
    const cls=Number(status)>=400?'err':'hit';
    const out=o.output_tokens!=null?fmtTok(o.output_tokens)+' out':'— out';
    const cost=o.cost_est_usd==null?'cost unknown':fmt(o.cost_est_usd,5);
    return `<span class="badge ${cls}">${status}</span> <span class="tokens">${out}</span> <span class="cost">${cost}</span>`;
  }
  const cls=o.status==='error'?'err':o.status==='pending'?'miss':'hit';
  const chars=o.result_chars!=null?fmtTok(o.result_chars)+' result chars':'turn-level';
  const cost=o.cost_est_usd==null?'cost pending':fmt(o.cost_est_usd,5)+' est';
  return `<span class="badge ${cls}">${esc(o.status||'pending')}</span> <span class="tokens">${chars}</span> <span class="cost">${cost}</span> <span class="badge miss">Codex estimated from chars</span>`;
}
function activityFlags(unit){
  const flags=[];
  const opt=unit.optimization_features||{};
  const routing=opt.routing||{};
  const crunch=opt.crunch||{};
  const cache=opt.cache||{};
  if(unit.granularity==='agent_turn')flags.push('<span class="badge miss">not provider-replayable</span>');
  if(unit.replayability_level)flags.push(`<span class="badge provider">${esc(unit.replayability_level)}</span>`);
  if(routing.routed_model&&routing.routed_model!==routing.requested_model)flags.push('<span class="badge routed">routed</span>');
  if(cache.status)flags.push(`<span class="badge ${cache.status==='hit'?'hit':cache.status==='miss'?'miss':'stream'}">cache ${esc(cache.status)}</span>`);
  if(crunch.changed)flags.push('<span class="badge crunched">crunched</span>');
  const category=(unit.input_features&&unit.input_features.category)||(unit.tool_features&&unit.tool_features.category);
  if(category)flags.push(`<span class="badge provider">${esc(category)}</span>`);
  return flags.join(' ')||'<span class="badge miss">observed</span>';
}
function usageHints(row){
  const hints=row.remaining_saving_potential_hints||[];
  if(!hints.length)return'<span class="badge hit">no obvious signal</span>';
  return hints.slice(0,3).map(h=>`<span class="badge routed" title="${esc(h.detail)}">${esc(h.label)}</span>`).join(' ');
}

const tableState={};
let applyingTableState=false;
function tableId(table){return table.dataset.tableId;}
function tableStateFor(table){
  const id=tableId(table);
  if(!tableState[id])tableState[id]={filter:'',sortIndex:null,sortDir:'asc'};
  return tableState[id];
}
function numericText(text){
  const raw=String(text||'').trim().toLowerCase();
  if(!raw||raw==='—')return Number.NEGATIVE_INFINITY;
  const first=(raw.match(/-?\\$?[\\d,.]+\\s*[mk%]?/)||[''])[0].replace(/[$,\\s]/g,'');
  if(!first)return Number.NEGATIVE_INFINITY;
  let multiplier=1;
  if(first.endsWith('m'))multiplier=1000000;
  if(first.endsWith('k'))multiplier=1000;
  const cleaned=first.replace(/[mk%]/g,'');
  const value=parseFloat(cleaned);
  return Number.isFinite(value)?value*multiplier:Number.NEGATIVE_INFINITY;
}
function durationSeconds(text){
  const raw=String(text||'').trim().toLowerCase();
  if(!raw||raw==='—')return Number.NEGATIVE_INFINITY;
  if(raw==='now')return 0;
  const value=parseFloat(raw.replace(/[$,]/g,''));
  if(!Number.isFinite(value))return Number.NEGATIVE_INFINITY;
  if(raw.includes('ms'))return value/1000;
  if(raw.endsWith('s'))return value;
  if(raw.endsWith('m'))return value*60;
  if(raw.endsWith('h'))return value*3600;
  if(raw.endsWith('d'))return value*86400;
  return value;
}
function sortValue(cell,type){
  const text=(cell&&cell.innerText||'').trim();
  if(type==='number'||type==='money'||type==='percent')return numericText(text);
  if(type==='latency'||type==='time')return durationSeconds(text);
  if(type==='timestamp'){
    const parsed=Date.parse(text);
    return Number.isNaN(parsed)?durationSeconds(text):parsed;
  }
  return text.toLowerCase();
}
function isPlaceholderRow(row,table){
  const cell=row.cells&&row.cells.length===1?row.cells[0]:null;
  const colCount=table.tHead&&table.tHead.rows[0]?table.tHead.rows[0].cells.length:1;
  return Boolean(cell&&cell.colSpan>=colCount&&cell.textContent.trim().toLowerCase().startsWith('no '));
}
function ensureTableControls(table){
  if(table.dataset.tableReady)return;
  table.dataset.tableReady='1';
  const state=tableStateFor(table);
  const tools=document.createElement('div');
  tools.className='table-tools';
  tools.dataset.forTable=tableId(table);
  const input=document.createElement('input');
  input.className='table-filter';
  input.type='search';
  input.placeholder=table.dataset.filterLabel||'Filter rows';
  input.value=state.filter;
  input.setAttribute('aria-label',input.placeholder);
  const clear=document.createElement('button');
  clear.className='table-clear';
  clear.type='button';
  clear.textContent='Clear';
  clear.addEventListener('click',()=>{
    state.filter='';
    input.value='';
    applyDataTableState(table);
  });
  input.addEventListener('input',()=>{
    state.filter=input.value;
    applyDataTableState(table);
  });
  tools.appendChild(input);
  tools.appendChild(clear);
  table.parentNode.insertBefore(tools,table);
  table.querySelectorAll('thead th[data-sort-type]').forEach((th,index)=>{
    th.tabIndex=0;
    th.setAttribute('role','button');
    th.setAttribute('aria-sort','none');
    th.addEventListener('click',()=>setTableSort(table,index));
    th.addEventListener('keydown',event=>{
      if(event.key==='Enter'||event.key===' '){
        event.preventDefault();
        setTableSort(table,index);
      }
    });
  });
}
function setTableSort(table,index){
  const state=tableStateFor(table);
  if(state.sortIndex===index){
    state.sortDir=state.sortDir==='asc'?'desc':'asc';
  }else{
    state.sortIndex=index;
    state.sortDir='asc';
  }
  applyDataTableState(table);
}
function applyDataTableState(table){
  if(!table||applyingTableState)return;
  applyingTableState=true;
  try{
    const state=tableStateFor(table);
    const tbody=table.tBodies[0];
    if(!tbody)return;
    tbody.querySelectorAll('tr.filter-empty-row').forEach(row=>row.remove());
    const rows=Array.from(tbody.rows);
    const dataRows=rows.filter(row=>!isPlaceholderRow(row,table));
    if(!dataRows.length){
      table.querySelectorAll('thead th[data-sort-type]').forEach(th=>{th.classList.remove('sort-asc','sort-desc');th.setAttribute('aria-sort','none');});
      return;
    }
    const filter=state.filter.trim().toLowerCase();
    let visibleRows=dataRows;
    dataRows.forEach(row=>{
      const match=!filter||row.innerText.toLowerCase().includes(filter);
      row.style.display=match?'':'none';
    });
    visibleRows=dataRows.filter(row=>row.style.display!=='none');
    if(state.sortIndex!==null){
      const header=table.tHead&&table.tHead.rows[0]&&table.tHead.rows[0].cells[state.sortIndex];
      const type=header?header.dataset.sortType||'text':'text';
      const dir=state.sortDir==='desc'?-1:1;
      visibleRows.sort((a,b)=>{
        const av=sortValue(a.cells[state.sortIndex],type);
        const bv=sortValue(b.cells[state.sortIndex],type);
        if(typeof av==='number'&&typeof bv==='number')return (av-bv)*dir;
        return String(av).localeCompare(String(bv))*dir;
      });
      visibleRows.forEach(row=>tbody.appendChild(row));
    }
    table.querySelectorAll('thead th[data-sort-type]').forEach((th,index)=>{
      const active=state.sortIndex===index;
      th.classList.toggle('sort-asc',active&&state.sortDir==='asc');
      th.classList.toggle('sort-desc',active&&state.sortDir==='desc');
      th.setAttribute('aria-sort',active?(state.sortDir==='asc'?'ascending':'descending'):'none');
    });
    if(filter&&!visibleRows.length){
      const colCount=table.tHead&&table.tHead.rows[0]?table.tHead.rows[0].cells.length:1;
      const empty=document.createElement('tr');
      empty.className='filter-empty-row';
      empty.innerHTML=`<td colspan="${colCount}" style="color:#8b949e">No matching rows</td>`;
      tbody.appendChild(empty);
    }
  }finally{
    applyingTableState=false;
  }
}
function applyAllDataTables(){
  document.querySelectorAll('table[data-table-id]').forEach(table=>{
    ensureTableControls(table);
    applyDataTableState(table);
  });
}
function initDataTables(){
  applyAllDataTables();
}

const FULL_STATS_TTL_MS=5000;
let fullStatsCache=null;
let fullStatsCacheAt=0;
let fullStatsInFlight=null;
async function loadFullStats(){
  const now=Date.now();
  if(fullStatsCache&&now-fullStatsCacheAt<FULL_STATS_TTL_MS)return fullStatsCache;
  if(fullStatsInFlight)return fullStatsInFlight;
  fullStatsInFlight=fetch('/agentflow/stats/full')
    .then(r=>{
      if(!r.ok)throw new Error('full stats HTTP '+r.status);
      return r.json();
    })
    .then(d=>{
      fullStatsCache=d;
      fullStatsCacheAt=Date.now();
      return d;
    })
    .finally(()=>{fullStatsInFlight=null;});
  return fullStatsInFlight;
}

function showTab(name){
  const tabs=['safety','activity','usage','weekly','categories','cache','errors','limiter','policies','managed','sessions'];
  tabs.forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===name);
  });
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    b.classList.toggle('active',tabs[i]===name);
  });
}

async function refreshUsage(){
  try{
    const r=await fetch('/agentflow/stats/usage');
    const d=await r.json();
    const tb=document.getElementById('usage-tbody');
    const rows=d.buckets||[];
    tb.innerHTML=rows.map(row=>{
      const optimized=row.optimization_rate==null?'—':Math.round(row.optimization_rate*100)+'%';
      const errors=(row.errors||0)>0
        ? `<span class="badge err">${row.errors} (${Math.round((row.error_rate||0)*100)}%)</span>`
        : '<span class="badge hit">0</span>';
      const codexCost=row.codex_cost_estimated?'<span class="badge miss">Codex estimated</span>':'';
      const totalTokens=(row.provider_total_tokens||0)+(row.codex_total_tokens_est||0);
      return `<tr>
        <td><span class="badge provider">${esc(row.bucket_label)}</span></td>
        <td>${(row.turns||0).toLocaleString()}</td>
        <td>${(row.provider_calls||0).toLocaleString()}</td>
        <td>${(row.codex_turns||0).toLocaleString()}</td>
        <td class="tokens">${fmtTok(totalTokens)} total · ${fmtTok(row.provider_total_tokens||0)} provider · ${fmtTok(row.codex_total_tokens_est||0)} Codex est</td>
        <td class="cost">${(row.provider_cost_known||row.codex_cost_known)?fmt(row.spend_usd||0,5):'—'}</td>
        <td class="savings">${fmt(row.captured_savings_usd||0,5)}</td>
        <td class="cost">${row.hard_floor_usd==null?'—':fmt(row.hard_floor_usd,5)}</td>
        <td class="tokens">${optimized}</td>
        <td>${errors}</td>
        <td class="flags">${usageHints(row)}</td>
        <td class="flags"><span class="badge provider">${esc(row.cost_basis)}</span> ${codexCost}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="12" style="color:#8b949e">No app or engineer usage today</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

async function refreshActivity(){
  try{
    const r=await fetch('/agentflow/stats/activity?limit=100');
    const d=await r.json();
    const tb=document.getElementById('activity-tbody');
    const rows=d.units||[];
    tb.innerHTML=rows.map(unit=>{
      const o=unit.outcome_features||{};
      const err=(o.status_code&&o.status_code>=400)||o.status==='error';
      const requested=unit.requested_model?shortModel(unit.requested_model):(unit.granularity==='agent_turn'?'turn-level':'—');
      const target=unit.target_model?shortModel(unit.target_model):(unit.granularity==='agent_turn'?'not provider-replayable':'—');
      return `<tr class="${err?'err-row':''}">
        <td class="ts">${ago(unit.created_at)}</td>
        <td><span class="badge provider">${esc(shortSurface(unit.source_surface))}</span></td>
        <td><span class="badge stream">${esc(unit.granularity||'unknown')}</span></td>
        <td><span class="badge provider">${esc(unit.app_family||'unknown')}</span></td>
        <td class="model">${esc(requested)}</td>
        <td class="model">${esc(target)}</td>
        <td class="tokens">${esc(activityInput(unit))}</td>
        <td>${activityOutcome(unit)}</td>
        <td class="latency">${fmtMs(o.latency_ms)}</td>
        <td class="flags">${activityFlags(unit)}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="10" style="color:#8b949e">No recent activity yet</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

async function refreshWeekly(){
  try{
    const r=await fetch('/agentflow/stats/weekly');
    const d=await r.json();
    const tb=document.getElementById('weekly-tbody');
    const rows=[...d.days,{...d.totals,_total:true}];
    tb.innerHTML=rows.map(row=>{
      const cls=row._total?' class="totals-row"':'';
      const errColor=row.errors?'color:#f85149':'color:#8b949e';
      return `<tr${cls}>
        <td class="ts">${row.day}</td>
        <td>${(row.total_units??row.total_calls??0).toLocaleString()}</td>
        <td>${(row.provider_calls??0).toLocaleString()}</td>
        <td>${(row.codex_turns??0).toLocaleString()}</td>
        <td style="color:#3fb950">${(row.successful_calls??0).toLocaleString()}</td>
        <td style="${errColor}">${(row.errors??0).toLocaleString()}</td>
        <td>${(row.cache_hits??0).toLocaleString()}</td>
        <td class="tokens">${fmtTok(row.total_tokens??((row.provider_tokens??0)+(row.codex_tokens_est??0)))} total · ${fmtTok(row.provider_tokens??0)} provider · ${fmtTok(row.codex_tokens_est??0)} Codex est</td>
        <td class="latency">${fmtMs(row.avg_latency_ms)}</td>
        <td class="cost">${fmt(row.cost_est_usd,5)}</td>
        <td class="baseline">${fmt(row.cost_baseline_usd,5)}</td>
        <td class="savings">${fmt(row.savings_usd,5)}</td>
        <td class="flags"><span class="badge provider">${esc(row.cost_basis||d.cost_basis||'provider-reported')}</span></td>
      </tr>`;
    }).join('');
    applyAllDataTables();
  }catch(e){}
}

async function refresh(){
  try{
    const d=await loadFullStats();
    const s=d.summary;
    const e=d.executive_summary||{};
    const acct=e.accounting_today||{};
    const acctTotal=e.accounting_total||{};
    const surfaces=acct.source_surfaces||[];
    const toks=e.tokens_today||{};
    const spend=e.spend||{};
    const savings=e.savings||{};
    const buckets=savings.today_buckets||{};
    const floor=e.hard_floor||{};
    const health=e.health||{};
    const sourceText=surfaces.length
      ? surfaces.map(row=>shortSurface(row.source_surface)+': '+fmtTok(row.total_tokens||0)+' '+(row.token_basis||'tokens')).join(' · ')
      : fmtTok(toks.provider_input_tokens||0)+' input · '+fmtTok(toks.provider_output_tokens||0)+' output provider tokens';
    const basisText=surfaces.length
      ? surfaces.map(row=>shortSurface(row.source_surface)+' '+(row.cost_basis||'unknown')).join(' · ')
      : (toks.codex_app_turns||0).toLocaleString()+' Codex turns · '+fmtTok(toks.codex_app_total_tokens_est||0)+' estimated tokens from '+fmtTok(toks.codex_app_input_text_chars||0)+' chars';

    document.getElementById('c-tokens-today').textContent=fmtTok(acct.total_tokens??toks.total_tokens??toks.provider_total_tokens??0);
    document.getElementById('c-tokens-sub').textContent=sourceText;
    document.getElementById('c-tokens-codex').textContent=basisText;
    document.getElementById('c-spend').textContent=fmt(acct.cost_est_usd??spend.today_calculated_spend_usd??spend.today_provider_spend_usd??0,4);
    document.getElementById('c-spend-sub').textContent=fmt(acctTotal.cost_est_usd??spend.calculated_spend_usd??spend.total_provider_spend_usd??0,4)+' total · '+fmt(spend.today_provider_spend_usd||0,4)+' provider reported · '+fmt(spend.today_codex_app_estimated_spend_usd||0,4)+' Codex est';
    document.getElementById('c-savings').textContent=fmt((acct.routing_savings_usd||0)+(acct.crunch_savings_usd||0)+(acct.cache_savings_usd||0)||savings.today_total_savings_usd||0,4);
    document.getElementById('c-savings-sub').textContent='routing '+fmt(acct.routing_savings_usd??buckets.routing_usd??0,4)+' · crunch '+fmt(acct.crunch_savings_usd??buckets.crunching_usd??0,4)+' · cache '+fmt(acct.cache_savings_usd??buckets.exact_local_cache_usd??0,4);
    document.getElementById('c-floor').textContent=fmt(acct.hard_floor_usd??floor.today_unavoidable_provider_spend_usd??0,4);
    document.getElementById('c-floor-sub').textContent='baseline '+fmt(spend.today_baseline_calculated_cost_usd??spend.today_baseline_provider_cost_usd??0,4)+' - feasible savings '+fmt(savings.today_total_savings_usd||0,4)+'; Codex estimated';
    document.getElementById('c-health').textContent=(health.errors||0).toLocaleString()+' errors';
    document.getElementById('c-health-sub').textContent='avg latency '+fmtMs(health.avg_latency_ms||0)+' · '+(s.today_calls||0).toLocaleString()+' provider calls today';

    document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString();
  }catch(e){
    document.getElementById('status').textContent='error: '+e.message;
  }
}

async function refreshCategories(){
  try{
    const d=await loadFullStats();
    const tb=document.getElementById('cat-tbody');
    const rows=d.category_breakdown||[];
    const total=rows.reduce((s,r)=>s+(r.count||0),0)||1;
    tb.innerHTML=rows.map(row=>{
      const pct=Math.round((row.count/total)*100);
      return `<tr>
        <td><span class="badge provider">${shortProvider(row.provider)}</span> <span class="badge miss">${row.category}</span></td>
        <td>${(row.count||0).toLocaleString()} <span style="color:#8b949e;font-size:11px">(${pct}%)</span></td>
        <td class="cost">${fmt(row.cost_usd,5)}</td>
        <td class="tokens">${(row.routed_count||0).toLocaleString()}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="4" style="color:#8b949e">No data yet</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

async function refreshCache(){
  try{
    const d=await loadFullStats();
    const replay=d.cache_replayability||{};
    const replayRows=replay.groups||[];
    document.getElementById('cache-replayability-tbody').innerHTML=replayRows.map(row=>{
      const blockers=(row.replayability_blockers||[]).map(b=>`<span class="badge ${b==='true-one-off-miss'?'miss':'routed'}">${esc(b)}</span>`).join(' ')||'<span class="badge hit">none</span>';
      const basis=row.fingerprint_basis||{};
      const basisText=[
        basis.granularity,
        basis.category,
        basis.stream?'stream':'non-stream',
        basis.has_tools?'tools':'no tools',
        basis.text_size_bucket,
        basis.replayability_level
      ].filter(Boolean).join(' · ');
      return `<tr>
        <td class="model">${esc(row.shape_fingerprint)}</td>
        <td><span class="badge provider">${esc(shortSurface(row.source_surface||'unknown'))}</span></td>
        <td class="model">${esc(row.cache_status||'unknown')} / ${esc(row.cache_reason||'unknown')}</td>
        <td>${(row.count||0).toLocaleString()}</td>
        <td>${(row.sessions||0).toLocaleString()}</td>
        <td class="cost">${fmt(row.estimated_cost_usd||0,5)}</td>
        <td class="flags">${blockers}</td>
        <td class="flags"><span class="badge provider">${esc(basisText||'metadata shape')}</span></td>
      </tr>`;
    }).join('')||'<tr><td colspan="8" style="color:#8b949e">No skipped or missed cache candidates recorded</td></tr>';
    const renderRows=(rows)=>rows.map(row=>`<tr>
      <td><span class="badge provider">${esc(shortSurface(row.source_surface||'unknown'))}</span></td>
      <td><span class="badge ${row.status==='hit'?'hit':row.status==='miss'?'miss':'stream'}">${row.status}</span></td>
      <td class="model">${row.reason||'unknown'}</td>
      <td class="tokens">${row.hit_type||'—'}</td>
      <td><span class="badge provider">${row.policy_source||'unknown'}</span></td>
      <td>${(row.count||0).toLocaleString()}</td>
    </tr>`).join('')||'<tr><td colspan="6" style="color:#8b949e">No cache decision data yet</td></tr>';
    document.getElementById('cache-today-tbody').innerHTML=renderRows(d.today_cache_decision_breakdown||[]);
    document.getElementById('cache-tbody').innerHTML=renderRows(d.cache_decision_breakdown||[]);
    applyAllDataTables();
  }catch(e){}
}

async function refreshErrors(){
  try{
    const d=await loadFullStats();
    const renderRows=(rows)=>rows.map(row=>`<tr>
      <td><span class="badge err">${esc(row.error_type||'unknown_error')}</span></td>
      <td>${row.status_code>=500?`<span class="badge err">${row.status_code}</span>`:`<span class="badge routed">${row.status_code}</span>`}</td>
      <td><span class="badge provider">${shortProvider(row.provider)}</span></td>
      <td><span class="badge provider">${esc(row.tier||'unknown')}</span></td>
      <td class="model">${esc(shortModel(row.requested_model))}</td>
      <td class="model">${esc(shortModel(row.routed_model))}</td>
      <td>${(row.count||0).toLocaleString()}</td>
      <td class="ts">${row.last_seen_at?ago(row.last_seen_at):'—'}</td>
      <td class="model" title="${esc(row.error_sample||'')}">${esc(row.error_sample||'—')}</td>
    </tr>`).join('')||'<tr><td colspan="9" style="color:#8b949e">No errors recorded</td></tr>';
    document.getElementById('errors-today-tbody').innerHTML=renderRows(d.today_error_breakdown||[]);
    document.getElementById('errors-tbody').innerHTML=renderRows(d.error_breakdown||[]);
    applyAllDataTables();
  }catch(e){}
}

async function refreshLimiter(){
  try{
    const r=await fetch('/agentflow/stats/limiter');
    const d=await r.json();
    const tiers=d.tiers||[];
    const active=tiers.filter(t=>t.active);
    const queued=tiers.reduce((sum,t)=>sum+(t.queued_count||0),0);
    const longest=active.reduce((max,t)=>Math.max(max,t.seconds_remaining||0),0);
    document.getElementById('c-health-cooldown').textContent=active.length
      ? active.length+' cooldowns · '+queued+' queued · longest '+fmtSec(longest)
      : 'cooldowns clear · '+queued+' queued';

    const tb=document.getElementById('limiter-tbody');
    tb.innerHTML=tiers.map(row=>{
      const badge=row.active
        ? `<span class="badge err">cooldown</span>`
        : `<span class="badge hit">clear</span>`;
      const slots=row.available_slots==null?'—':`${row.available_slots}/${row.max_concurrent}`;
      return `<tr>
        <td><span class="badge provider">${row.tier}</span></td>
        <td>${badge}</td>
        <td class="latency">${fmtSec(row.seconds_remaining||0)}</td>
        <td class="ts">${until(row.cooldown_until)}</td>
        <td class="tokens">${slots}</td>
        <td class="tokens">${row.queued_count||0}</td>
        <td class="ts">${row.last_upstream_429_at?ago(row.last_upstream_429_at):'—'}</td>
      </tr>`;
    }).join('');

    const rb=document.getElementById('limiter-recent-tbody');
    const recent=d.recent_rate_limits||[];
    rb.innerHTML=recent.map(row=>`<tr>
      <td class="ts">${ago(row.created_at)}</td>
      <td><span class="badge provider">${row.tier}</span></td>
      <td><span class="badge provider">${shortProvider(row.provider)}</span></td>
      <td>${row.status_code>=500?`<span class="badge err">${row.status_code}</span>`:`<span class="badge routed">${row.status_code}</span>`}</td>
      <td class="tokens">${row.retry_count||0}</td>
      <td class="latency">${fmtMs(row.latency_ms)}</td>
      <td>${row.local_throttled?'<span class="badge err">local cooldown</span>':'<span class="badge routed">upstream</span>'}</td>
      <td class="model">${row.error||'—'}</td>
    </tr>`).join('')||'<tr><td colspan="8" style="color:#8b949e">No recent rate-limit responses</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

function boolBadge(value,goodLabel,badLabel){
  if(value===true)return`<span class="badge hit">${esc(goodLabel||'yes')}</span>`;
  if(value===false)return`<span class="badge err">${esc(badLabel||'no')}</span>`;
  return'<span class="badge miss">unknown</span>';
}
function safetySeverityBadge(severity){
  if(severity==='critical'||severity==='high')return'err';
  if(severity==='medium')return'routed';
  if(severity==='info')return'miss';
  return'provider';
}
function queueStatusBadge(status){
  if(status==='sent')return'hit';
  if(status==='retryable-error'||status==='dropped-after-limit'||status==='error')return'err';
  if(status==='queued'||status==='sending')return'routed';
  return'provider';
}
async function refreshSafety(){
  try{
    const r=await fetch('/agentflow/stats/safety');
    const d=await r.json();
    const s=d.summary||{};
    const checks=d.checks||{};
    const managed=checks.managed||{};
    const recommendation=managed.recommendation_server||{};
    const policyBundle=managed.policy_bundle_recommendation||{};
    const bodyLogging=checks.body_logging||{};
    const database=checks.database||{};
    const policyEvents=checks.policy_events||{};
    const providerProxy=checks.provider_proxy||{};
    const dashboard=checks.dashboard||{};
    const feedbackQueue=managed.feedback_queue||{};
    const feedbackSummary=feedbackQueue.summary||{};
    const oldestDue=feedbackQueue.oldest_due||{};
    const lastSent=feedbackQueue.last_successful_flush||{};
    const rows=[
      {
        check:'Provider proxy bind',
        status:boolBadge(providerProxy.loopback,'loopback','non-loopback'),
        detail:providerProxy.host_configured?`host ${providerProxy.host||'unknown'}`:'host not supplied'
      },
      {
        check:'Body logging',
        status:bodyLogging.enabled?'<span class="badge err">enabled</span>':'<span class="badge hit">off</span>',
        detail:'payload includes no raw request or response bodies'
      },
      {
        check:'Managed communication',
        status:s.managed_communication_enabled?'<span class="badge routed">configured</span>':'<span class="badge hit">off</span>',
        detail:`auth ${managed.auth_configured?'configured':'not configured'} · recommendation ${recommendation.configured?recommendation.redacted_url:'off'} · policy bundle ${policyBundle.configured?policyBundle.redacted_url:'off'}`
      },
      {
        check:'Managed feedback queue',
        status:(feedbackSummary.due||feedbackSummary.retryable_error||feedbackSummary.dropped_after_limit)
          ? '<span class="badge err">attention</span>'
          : '<span class="badge hit">clear</span>',
        detail:`queued ${(feedbackSummary.queued||0).toLocaleString()} · due ${(feedbackSummary.due||0).toLocaleString()} · retryable ${(feedbackSummary.retryable_error||0).toLocaleString()} · dropped ${(feedbackSummary.dropped_after_limit||0).toLocaleString()} · oldest due ${fmtSec(feedbackSummary.oldest_due_age_seconds)} · last sent ${lastSent.sent_at?ago(lastSent.sent_at):'—'} · payload omitted`
      },
      {
        check:'Dashboard mode',
        status:dashboard.read_only?'<span class="badge hit">read-only</span>':'<span class="badge err">mutable</span>',
        detail:dashboard.host_configured?`dashboard host ${dashboard.host}`:'dashboard host not supplied'
      },
      {
        check:'Database',
        status:database.path_class==='external-database-url'?'<span class="badge routed">external URL</span>':'<span class="badge hit">local class</span>',
        detail:`${database.path_class||'unknown'} · raw path omitted`
      },
      {
        check:'Policy event log',
        status:policyEvents.enabled?'<span class="badge hit">enabled</span>':'<span class="badge routed">disabled</span>',
        detail:`${policyEvents.path_class||'unknown'} · raw path omitted`
      }
    ];
    document.getElementById('safety-summary-tbody').innerHTML=rows.map(row=>`<tr>
      <td><span class="badge provider">${esc(row.check)}</span></td>
      <td>${row.status}</td>
      <td class="flags">${esc(row.detail)}</td>
    </tr>`).join('');
    const warnings=d.warnings||[];
    document.getElementById('safety-warnings-tbody').innerHTML=warnings.map(row=>`<tr>
      <td><span class="badge ${safetySeverityBadge(row.severity)}">${esc(row.severity)}</span></td>
      <td class="model">${esc(row.code)}</td>
      <td class="flags">${esc(row.message)}</td>
    </tr>`).join('')||'<tr><td colspan="3" style="color:#8b949e">No safety or privacy warnings</td></tr>';
    const dueRows=feedbackQueue.due_samples||[];
    document.getElementById('safety-managed-feedback-tbody').innerHTML=dueRows.map(row=>`<tr>
      <td class="latency">${fmtSec(row.due_age_seconds)}</td>
      <td><span class="badge provider">${esc(shortSurface(row.source_surface))}</span></td>
      <td><span class="badge ${queueStatusBadge(row.status)}">${esc(row.status)}</span></td>
      <td class="tokens">${row.attempts||0}</td>
      <td class="tokens">${row.optimization_unit_id??'—'}</td>
      <td class="tokens">${row.last_status_code??'—'}</td>
      <td class="model">${esc(row.last_error_class||'—')}</td>
      <td>${row.payload_included?'<span class="badge err">included</span>':'<span class="badge hit">omitted</span>'}</td>
    </tr>`).join('')||`<tr><td colspan="8" style="color:#8b949e">No due managed feedback rows; ${oldestDue.queue_id?'oldest due '+esc(oldestDue.queue_id):'queue clear'}</td></tr>`;
    applyAllDataTables();
  }catch(e){}
}

function policyStatus(enabled){
  return enabled?'<span class="badge hit">enabled</span>':'<span class="badge miss">disabled</span>';
}
function policySource(source){
  const cls=source==='local-manual'?'routed':'provider';
  return `<span class="badge ${cls}">${esc(source||'unknown')}</span>`;
}
function compactSettings(items){
  return items.filter(Boolean).map(item=>`<span class="badge stream">${esc(item)}</span>`).join(' ');
}
function policyReloadSetting(file){
  if(!file) return '';
  return file.reload_required?'reload required':'loaded';
}
function policyReloadBadge(summary){
  if(summary&&summary.reload_required){
    return '<span class="badge err">reload required</span>';
  }
  return '<span class="badge hit">loaded</span>';
}
async function refreshPolicies(){
  try{
    const r=await fetch('/agentflow/stats/policies');
    const d=await r.json();
    const summary=d.summary||{};
    const stale=(summary.reload_required_sections||[]).join(', ');
    document.getElementById('policy-summary-tbody').innerHTML=`<tr>
      <td>${policyReloadBadge(summary)}</td>
      <td class="tokens">${summary.policy_count??'—'}</td>
      <td class="tokens">${summary.loaded_file_count??'—'}</td>
      <td class="tokens">${summary.manual_policy_count??'—'}</td>
      <td class="tokens">${summary.local_default_policy_count??'—'}</td>
      <td class="flags">${stale?`<span class="badge err">${esc(stale)}</span>`:'<span class="badge hit">none</span>'}</td>
    </tr>`;
    const rows=[
      {
        name:'Routing',
        enabled:d.routing&&d.routing.enabled,
        source:d.routing&&d.routing.policy_source,
        path:d.routing&&d.routing.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.routing&&d.routing.file),
          'rules '+((d.routing&&d.routing.rules)||[]).length,
          d.routing&&d.routing.strip_thinking_history?'strip thinking history':'keep thinking history',
          d.routing&&d.routing.openai&&d.routing.openai.enabled?'OpenAI routing on':'OpenAI routing off'
        ])
      },
      {
        name:'Crunch',
        enabled:d.crunch&&d.crunch.enabled,
        source:d.crunch&&d.crunch.policy_source,
        path:d.crunch&&d.crunch.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.crunch&&d.crunch.file),
          'threshold '+fmtTok(d.crunch&&d.crunch.threshold_chars),
          d.crunch&&d.crunch.prompt_cache&&d.crunch.prompt_cache.enabled?'prompt cache on':'prompt cache off',
          d.crunch&&d.crunch.old_context_summarization&&d.crunch.old_context_summarization.enabled?'old-context summary on':'old-context summary off',
          d.crunch&&d.crunch.thinking_deduplication&&d.crunch.thinking_deduplication.enabled?'thinking dedupe on':'thinking dedupe off'
        ])
      },
      {
        name:'Cache',
        enabled:d.cache&&d.cache.enabled,
        source:d.cache&&d.cache.policy_source,
        path:d.cache&&d.cache.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.cache&&d.cache.file),
          d.cache&&d.cache.exact_cache&&d.cache.exact_cache.enabled?'exact on':'exact off',
          d.cache&&d.cache.exact_cache&&d.cache.exact_cache.cache_tool_calls?'tool cache on':'tool cache off',
          d.cache&&d.cache.semantic_cache&&d.cache.semantic_cache.enabled?'semantic on':'semantic off',
          d.cache&&d.cache.file_watch&&d.cache.file_watch.enabled?'file watch on':'file watch off'
        ])
      },
      {
        name:'Routing experiments',
        enabled:d.routing_experiments&&d.routing_experiments.enabled,
        source:d.routing_experiments&&d.routing_experiments.policy_source,
        path:d.routing_experiments&&d.routing_experiments.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.routing_experiments&&d.routing_experiments.file),
          'sample '+(((d.routing_experiments&&d.routing_experiments.policy&&d.routing_experiments.policy.sample_rate)||0)*100).toFixed(1)+'%',
          'similarity '+((d.routing_experiments&&d.routing_experiments.policy&&d.routing_experiments.policy.similarity_threshold)||0)
        ])
      }
    ];
    const codexSurface=d.source_surfaces&&(d.source_surfaces.codex_turn||d.source_surfaces.codex_app_turn);
    const codexPolicy=d.codex_app||{};
    if(codexSurface||codexPolicy.rules){
      const codexRuntime=codexSurface||{};
      const codexCache=codexRuntime.cache||{};
      const codexExact=codexCache.exact_cache||{};
      const codexSafe=codexRuntime.safe_turn_params||{};
      const codexSkip=codexRuntime.action_like_skip_behavior||{};
      const codexRules=codexPolicy.rules||[];
      const codexApp=(codexPolicy.application||{});
      rows.push({
        name:'Codex app-server',
        enabled:codexPolicy.enabled??codexRuntime.enabled,
        source:codexPolicy.policy_source||codexRuntime.policy_source,
        path:codexExact.namespace?('namespace '+codexExact.namespace):codexExact.upstream,
        settings:compactSettings([
          codexRuntime.reload_required?'reload required':'loaded',
          codexPolicy.review_only?'review only':'',
          codexApp.status?('application '+codexApp.status):'not applied',
          'Codex rules '+codexRules.length,
          'conditions '+((codexPolicy.supported_conditions||[]).length),
          'actions '+((codexPolicy.supported_actions||[]).length),
          codexRuntime.optimization&&codexRuntime.optimization.enabled?'optimization on':'optimization off',
          codexCache.enabled?'Codex exact cache on':'Codex exact cache off',
          codexExact.upstream?('upstream '+codexExact.upstream):'',
          'safe keys '+(codexSafe.allowed_key_count??0),
          codexSkip.enabled?'action-like skip on':'action-like skip off'
        ])
      });
    }
    document.getElementById('policies-tbody').innerHTML=rows.map(row=>`<tr>
      <td><span class="badge provider">${esc(row.name)}</span></td>
      <td>${policyStatus(row.enabled)}</td>
      <td>${policySource(row.source)}</td>
      <td class="model" title="${esc(row.path)}">${esc(row.path)}</td>
      <td class="flags">${row.settings}</td>
    </tr>`).join('');

    const ruleRows=(d.routing&&d.routing.rules)||[];
    document.getElementById('routing-rules-tbody').innerHTML=ruleRows.map((rule,i)=>`<tr>
      <td class="tokens">${i+1}</td>
      <td class="flags">${esc(JSON.stringify(rule.conditions||{}))}</td>
      <td class="flags">${esc(JSON.stringify(rule.action||{}))}</td>
    </tr>`).join('')||'<tr><td colspan="3" style="color:#8b949e">No routing rules loaded</td></tr>';

    const er=await fetch('/agentflow/stats/policy-events?limit=20');
    const ed=await er.json();
    const events=ed.events||[];
    document.getElementById('policy-events-tbody').innerHTML=events.map(event=>{
      const details=event.details||{};
      const parts=[];
      if(details.status_code!=null)parts.push('HTTP '+details.status_code);
      if(details.exit_code!=null)parts.push('exit '+details.exit_code);
      if(details.changed_sections&&details.changed_sections.length)parts.push('changed '+details.changed_sections.join(', '));
      if(details.change_count!=null)parts.push(details.change_count+' changes');
      if(details.error_count!=null)parts.push(details.error_count+' validation errors');
      if(details.reloaded_modules)parts.push(details.reloaded_modules.length+' modules');
      return `<tr>
        <td class="ts">${ago(event.created_at)}</td>
        <td><span class="badge provider">${esc(event.action)}</span></td>
        <td>${event.ok?'<span class="badge hit">ok</span>':'<span class="badge err">failed</span>'}</td>
        <td><span class="badge stream">${esc(details.source||'unknown')}</span></td>
        <td class="flags">${parts.map(p=>`<span class="badge miss">${esc(p)}</span>`).join(' ')||'<span class="badge miss">recorded</span>'}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="5" style="color:#8b949e">No policy operator events recorded</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

function compactBreakdown(rows,emptyLabel){
  rows=rows||[];
  if(!rows.length)return`<span class="badge miss">${esc(emptyLabel||'none')}</span>`;
  return rows.slice(0,5).map(row=>`<span class="badge provider">${esc(row.value)} ${(row.count||0).toLocaleString()}</span>`).join(' ');
}
function recBadge(status){
  if(status==='received'||status==='sent')return'hit';
  if(status==='error'||status==='invalid')return'err';
  if(status==='skipped'||status==='missing')return'miss';
  return'provider';
}
function managedLastError(summary){
  const parts=[];
  if(summary.last_feedback_error_class)parts.push(`<span class="badge err">feedback ${esc(summary.last_feedback_error_class)}</span>`);
  if(summary.last_recommendation_error_class)parts.push(`<span class="badge err">recommendation ${esc(summary.last_recommendation_error_class)}</span>`);
  return parts.join(' ')||'<span class="badge hit">none</span>';
}
function managedHealthDetails(row){
  const details=row.details||{};
  const parts=[];
  ['sample_count','min_samples','error_rate','max_error_rate','last_seen_at','threshold','value','reason'].forEach(key=>{
    if(details[key]!=null)parts.push(`${key}: ${details[key]}`);
  });
  return parts.map(p=>`<span class="badge miss">${esc(p)}</span>`).join(' ')||'<span class="badge miss">metadata only</span>';
}
async function refreshManaged(){
  try{
    const [managedResponse,safetyResponse]=await Promise.all([
      fetch('/agentflow/stats/managed-recommendations?limit=500'),
      fetch('/agentflow/stats/safety')
    ]);
    const d=await managedResponse.json();
    const safety=await safetyResponse.json();
    const s=d.summary||{};
    const cfg=d.current_config||{};
    const queue=((((safety||{}).checks||{}).managed||{}).feedback_queue)||{};
    const queueSummary=queue.summary||{};
    const lastSent=queue.last_successful_flush||{};
    document.getElementById('managed-summary-tbody').innerHTML=`<tr>
      <td class="flags">${cfg.enabled?'<span class="badge hit">enabled</span>':'<span class="badge miss">offline / local-only</span>'} <span class="badge provider">${esc(cfg.server_url||'—')}</span></td>
      <td class="tokens">${(s.window_calls||0).toLocaleString()}</td>
      <td class="tokens">${(s.metadata_rows||0).toLocaleString()}</td>
      <td class="tokens">${(s.disabled_count||0).toLocaleString()}</td>
      <td class="tokens">${(s.received_count||0).toLocaleString()}</td>
      <td class="tokens">${(s.applied_count||0).toLocaleString()}</td>
      <td class="tokens">${((s.server_error_count||0)+(s.invalid_count||0)).toLocaleString()}</td>
      <td class="latency">${fmtMs(s.avg_recommendation_latency_ms)}</td>
      <td class="latency">${fmtMs(s.avg_feedback_latency_ms)}</td>
      <td class="flags">${managedLastError(s)}</td>
    </tr>`;
    document.getElementById('managed-feedback-tbody').innerHTML=`<tr>
      <td class="tokens">${(s.feedback_sent_count||0).toLocaleString()}</td>
      <td class="tokens">${(s.feedback_skipped_count||0).toLocaleString()}</td>
      <td class="tokens">${(s.feedback_failed_count||0).toLocaleString()}</td>
      <td class="tokens">${(s.feedback_sanitized_count||0).toLocaleString()}</td>
      <td class="tokens">${(s.historical_null_rows||0).toLocaleString()}</td>
      <td class="flags">${compactBreakdown(d.feedback_reason_breakdown,'no feedback')}</td>
      <td class="flags">${compactBreakdown(d.fallback_breakdown,'none')}</td>
      <td class="flags">${compactBreakdown(d.policy_ids,'none')}</td>
    </tr>`;
    document.getElementById('managed-feedback-queue-tbody').innerHTML=`<tr>
      <td class="tokens">${(queueSummary.queued||0).toLocaleString()}</td>
      <td class="tokens">${(queueSummary.due||0).toLocaleString()}</td>
      <td class="tokens">${(queueSummary.retryable_error||0).toLocaleString()}</td>
      <td class="tokens">${(queueSummary.dropped_after_limit||0).toLocaleString()}</td>
      <td class="tokens">${(queueSummary.sent||0).toLocaleString()}</td>
      <td class="latency">${fmtSec(queueSummary.oldest_due_age_seconds)}</td>
      <td class="ts">${lastSent.sent_at?ago(lastSent.sent_at):'—'}</td>
      <td class="flags">${compactBreakdown(queue.source_surface_breakdown,'none')} <span class="badge hit">payload omitted</span></td>
    </tr>`;
    const health=(d.recommendation_health||{}).latest_fetch_review||{};
    const healthRows=(d.recommendation_health||{}).rows||[];
    document.getElementById('managed-health-tbody').innerHTML=healthRows.map(row=>`<tr>
      <td class="ts">${ago(health.event_created_at||health.generated_at)}</td>
      <td class="flags"><span class="badge ${health.status==='warning'?'err':'hit'}">${esc(health.status||'available')}</span></td>
      <td><span class="badge provider">${esc(row.kind||'health')}</span></td>
      <td class="model">${esc(row.candidate_id||'—')}</td>
      <td class="flags"><span class="badge miss">${esc(row.code||'warning')}</span></td>
      <td class="flags">${managedHealthDetails(row)}</td>
    </tr>`).join('')||`<tr><td colspan="6" style="color:#8b949e">${health.status?'No weak evidence warnings in latest fetch-review event':'No managed bundle health event recorded'}</td></tr>`;
    const rows=d.recent||[];
    document.getElementById('managed-recent-tbody').innerHTML=rows.map(row=>`<tr>
      <td class="ts">${ago(row.created_at)}</td>
      <td><span class="badge provider">${esc(shortSurface(row.source_surface))}</span></td>
      <td class="model">${esc(shortModel(row.requested_model))}</td>
      <td class="model">${esc(shortModel(row.routed_model))}</td>
      <td class="flags"><span class="badge ${recBadge(row.recommendation_status)}">${esc(row.recommendation_status)}</span> <span class="badge miss">${esc(row.recommendation_reason||'unknown')}</span>${row.applied?' <span class="badge hit">applied</span>':''}${row.changed_model?' <span class="badge routed">changed model</span>':''}</td>
      <td class="model">${esc(row.policy_id||'—')}</td>
      <td class="model">${esc(shortModel(row.target_model))}</td>
      <td class="latency">${fmtMs(row.latency_ms)}</td>
      <td class="flags"><span class="badge ${recBadge(row.feedback_status)}">${esc(row.feedback_status)}</span> <span class="badge miss">${esc(row.feedback_reason||'unknown')}</span>${row.feedback_error_class?` <span class="badge err">${esc(row.feedback_error_class)}</span>`:''}</td>
      <td class="model">${esc(row.fallback||'—')}</td>
    </tr>`).join('')||'<tr><td colspan="10" style="color:#8b949e">No managed recommendation rows yet</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

async function refreshSessions(){
  try{
    const r=await fetch('/agentflow/stats/sessions');
    const d=await r.json();
    const tb=document.getElementById('sess-tbody');
    const pb=document.getElementById('plateau-tbody');
    const rows=d.sessions||[];
    tb.innerHTML=rows.map(row=>`<tr>
      <td class="flags"><span class="badge provider">${esc(shortSurface(row.source_surface))}</span></td>
      <td>${esc(row.app_family||'unknown')}</td>
      <td class="ts" title="${esc(row.session_key_basis||'')}">${row.sid}<div class="sub">${esc(row.session_key_basis||'')}</div></td>
      <td>${(row.turns||row.calls||0).toLocaleString()}</td>
      <td>${(row.provider_calls||0).toLocaleString()}</td>
      <td>${(row.codex_turns||0).toLocaleString()}</td>
      <td class="cost">${fmt(row.cost_usd,5)}</td>
      <td class="tokens">${fmtTok(row.codex_input_tokens_est||0)}</td>
      <td class="tokens">${fmtTok(row.codex_output_tokens_est||0)}</td>
      <td class="flags"><span class="badge routed">r ${(row.codex_routed_turns||0).toLocaleString()}</span> <span class="badge crunched">c ${(row.codex_crunched_turns||0).toLocaleString()}</span> <span class="badge hit">cache ${(row.codex_cache_hits||0).toLocaleString()}</span>${row.codex_errors?` <span class="badge err">err ${row.codex_errors}</span>`:''}</td>
      <td class="tokens">${fmtTok(row.thinking_tokens||0)}</td>
      <td class="cost">${fmt(row.thinking_cost_usd||0,5)}</td>
      <td class="tokens">${fmtTok(row.cache_creation_tokens||0)}</td>
      <td class="tokens">${fmtTok(row.cache_read_tokens||0)}</td>
      <td class="tokens">${fmtRatio(row.cache_write_read_token_ratio)}</td>
      <td class="cost">${fmt(row.cache_creation_cost_usd||0,5)}</td>
      <td class="savings">${fmt(row.cache_read_savings_usd||0,5)}</td>
      <td class="tokens">${fmtRatio(row.cache_warmup_payback_ratio)}</td>
      <td class="tokens">${row.tool_result||0}</td>
      <td class="tokens">${row.tool_heavy||0}</td>
      <td class="tokens">${row.short_completion||0}</td>
      <td class="tokens">${row.code_gen||0}</td>
      <td class="tokens">${row.chat||0}</td>
      <td class="tokens">${row.other||0}</td>
    </tr>`).join('')||'<tr><td colspan="24" style="color:#8b949e">No sessions today</td></tr>';
    const plateaus=d.context_plateaus||[];
    pb.innerHTML=plateaus.map(row=>`<tr>
      <td class="flags"><span class="badge provider">${esc(shortSurface(row.source_surface))}</span></td>
      <td class="ts" title="${esc(row.session_key_basis||'')}">${row.sid}<div class="sub">${esc(row.session_key_basis||'')}</div></td>
      <td>${(row.calls||0).toLocaleString()}</td>
      <td class="cost">${fmt(row.cost_usd,5)}</td>
      <td class="tokens">${(row.plateau_pairs||0).toLocaleString()}</td>
      <td class="tokens">${fmtTok(row.median_text_chars||0)}</td>
      <td class="tokens">${fmtTok(row.p90_text_chars||0)}</td>
      <td class="savings">${fmt(row.cache_read_savings_usd||0,5)}</td>
      <td class="tokens">${fmtTok(row.crunch_saved_chars||0)}</td>
      <td>${row.flagged?'<span class="badge err">flagged</span>':'<span class="badge miss">watch</span>'}</td>
    </tr>`).join('')||'<tr><td colspan="10" style="color:#8b949e">No repeated large-context plateaus today</td></tr>';
    applyAllDataTables();
  }catch(e){}
}

initDataTables();
refreshSafety();
refreshActivity();
refreshUsage();
refresh();
refreshWeekly();
refreshCategories();
refreshCache();
refreshErrors();
refreshLimiter();
refreshSafety();
refreshPolicies();
refreshManaged();
refreshSessions();
setInterval(refreshSafety,30000);
setInterval(refreshActivity,5000);
setInterval(refreshUsage,30000);
setInterval(refresh,5000);
setInterval(refreshWeekly,30000);
setInterval(refreshCategories,30000);
setInterval(refreshCache,30000);
setInterval(refreshErrors,30000);
setInterval(refreshLimiter,5000);
setInterval(refreshPolicies,30000);
setInterval(refreshManaged,30000);
setInterval(refreshSessions,30000);
</script>
</body>
</html>"""
