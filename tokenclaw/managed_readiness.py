from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import httpx

from tokenclaw import __version__ as TOKENCLAW_VERSION
from tokenclaw.cli_common import default_db_path
from tokenclaw.client_contract import ClientContractRequest, ContractClient, fetch_or_get_client_contract
from tokenclaw.managed_mode import ManagedProductMode, managed_product_mode
from tokenclaw.recommendations import (
    _managed_headers,
    managed_auth_configured,
    managed_auth_source,
    recommendation_server_url,
)
from tokenclaw.upstream_url import redact_url


MANAGED_READINESS_SCHEMA = "tokenclaw.managed_readiness.v1"
SUPPORTED_LOCAL_ACTION_FAMILIES = ("routing", "crunch", "cache", "old_context_summarization")
DEFAULT_QUEUE_STALE_SECONDS = 3600.0


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: Any, *, now: datetime) -> float | None:
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    return max(0.0, round((now - parsed).total_seconds(), 3))


def _queue_stale_seconds() -> float:
    raw = os.getenv("TOKENCLAW_MANAGED_READINESS_QUEUE_STALE_SECONDS", "")
    try:
        return max(0.0, float(raw)) if raw.strip() else DEFAULT_QUEUE_STALE_SECONDS
    except ValueError:
        return DEFAULT_QUEUE_STALE_SECONDS


def _feedback_queue_readiness(*, now: datetime, stale_seconds: float) -> dict[str, Any]:
    db_path = default_db_path()
    base: dict[str, Any] = {
        "schema": "tokenclaw.managed_feedback_queue_readiness.v1",
        "status": "ok",
        "db_configured": bool(db_path),
        "db_path_present": False,
        "status_counts": {},
        "pending_count": 0,
        "oldest_pending_created_at": None,
        "oldest_pending_age_seconds": None,
        "last_sent_at": None,
        "last_sent_age_seconds": None,
        "stale_after_seconds": stale_seconds,
        "reason_codes": [],
        "metadata_only": True,
        "raw_payload_included": False,
    }
    if not db_path:
        return base
    if not db_path.startswith(("postgresql://", "postgres://")) and not Path(db_path).expanduser().exists():
        return base

    try:
        from tokenclaw.store import Store

        store = Store() if db_path.startswith(("postgresql://", "postgres://")) else Store(db_path)
        try:
            summary_rows = store.managed_outcome_feedback_summary()
            rows = store.managed_outcome_feedback_rows(limit=10000)
        finally:
            store.conn.close()
    except Exception as exc:
        base.update({
            "status": "unavailable",
            "reason_codes": ["feedback-queue-unavailable"],
            "error_type": type(exc).__name__,
        })
        return base

    base["db_path_present"] = True
    status_counts = {
        str(row.get("status") or "unknown"): int(row.get("count") or 0)
        for row in summary_rows
        if isinstance(row, dict)
    }
    pending_statuses = {"queued", "retryable-error", "sending"}
    pending_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "") in pending_statuses
    ]
    sent_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "") == "sent" and row.get("sent_at")
    ]
    pending_datetimes = [parsed for row in pending_rows if (parsed := _parse_datetime(row.get("created_at"))) is not None]
    sent_datetimes = [parsed for row in sent_rows if (parsed := _parse_datetime(row.get("sent_at"))) is not None]
    oldest_pending = min(pending_datetimes, default=None)
    last_sent = max(sent_datetimes, default=None)
    oldest_pending_text = oldest_pending.isoformat().replace("+00:00", "Z") if oldest_pending else None
    last_sent_text = last_sent.isoformat().replace("+00:00", "Z") if last_sent else None
    oldest_age = _age_seconds(oldest_pending_text, now=now)
    base.update({
        "status_counts": dict(sorted(status_counts.items())),
        "pending_count": sum(int(status_counts.get(status, 0)) for status in pending_statuses),
        "oldest_pending_created_at": oldest_pending_text,
        "oldest_pending_age_seconds": oldest_age,
        "last_sent_at": last_sent_text,
        "last_sent_age_seconds": _age_seconds(last_sent_text, now=now),
    })
    if oldest_age is not None and oldest_age > stale_seconds:
        base["status"] = "stale"
        base["reason_codes"] = ["managed-feedback-queue-stale"]
    return base


def _server_reachability(*, server_url: str, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "skipped",
        "server_url": redact_url(server_url),
        "status_code": None,
        "latency_ms": None,
        "reason": None,
        "metadata_only": True,
        "raw_response_included": False,
    }
    if not server_url:
        result.update({"status": "blocked", "reason": "server-url-not-configured"})
        return result
    started = datetime.now(timezone.utc)
    try:
        response = httpx.get(server_url.rstrip("/") + "/health", timeout=timeout)
    except Exception as exc:
        result.update({"status": "blocked", "reason": "server-unreachable", "error_type": type(exc).__name__})
        return result
    result["latency_ms"] = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    result["status_code"] = response.status_code
    if response.status_code < 200 or response.status_code >= 300:
        result.update({"status": "blocked", "reason": "server-health-non-2xx"})
        return result
    result["status"] = "reachable"
    try:
        body = response.json()
    except ValueError:
        return result
    if isinstance(body, dict):
        result["server_ok"] = bool(body.get("ok", True))
        if body.get("schema"):
            result["schema"] = str(body.get("schema"))
    return result


async def _client_contract_readiness(*, server_url: str, timeout: float) -> dict[str, Any]:
    request = ClientContractRequest(
        provider="anthropic",
        source_surface="anthropic_messages",
        app_family="tokenclaw_doctor",
        client_version=TOKENCLAW_VERSION,
    )
    client = ContractClient(
        base_url=server_url,
        headers=_managed_headers(),
        timeout_seconds=timeout,
        async_client_factory=httpx.AsyncClient,
    )
    meta = await fetch_or_get_client_contract(
        request,
        enabled=True,
        server_url=server_url,
        auth_configured=managed_auth_configured(),
        auth_source=managed_auth_source(),
        client=client,
    )
    contract = meta.get("contract") if isinstance(meta.get("contract"), dict) else {}
    return {
        "schema": "tokenclaw.managed_client_contract_readiness.v1",
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "active": bool(meta.get("active")),
        "cache_status": meta.get("cache_status"),
        "contract_id": meta.get("contract_id"),
        "expires_at": meta.get("expires_at") or contract.get("expires_at"),
        "allowed_action_families": sorted(contract.get("allowed_action_families") or []),
        "auth_configured": bool(meta.get("auth_configured")),
        "auth_source": meta.get("auth_source"),
        "status_code": meta.get("status_code"),
        "latency_ms": meta.get("latency_ms"),
        "metadata_only": True,
        "raw_response_included": False,
    }


def _ready_state_for_mode(mode: str) -> str:
    if mode == "observe_only":
        return "observe_only_ready"
    if mode == "dry_run":
        return "dry_run_ready"
    if mode == "canary":
        return "canary_ready"
    if mode == "live":
        return "live_ready"
    return "local_only"


def _enabled_action_families(product_mode: ManagedProductMode) -> list[str]:
    return sorted(
        family
        for family, enabled in product_mode.family_enabled.items()
        if enabled
    )


def build_managed_readiness(*, timeout: float = 1.5, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    product_mode = managed_product_mode()
    server_url = recommendation_server_url()
    stale_seconds = _queue_stale_seconds()
    queue = _feedback_queue_readiness(now=now, stale_seconds=stale_seconds)
    reason_codes: list[str] = []
    checks = {
        "server_reachability": {"status": "skipped", "reason": "managed-server-calls-disabled"},
        "client_contract": {"status": "skipped", "reason": "managed-server-calls-disabled"},
        "feedback_queue": queue,
    }
    result: dict[str, Any] = {
        "schema": MANAGED_READINESS_SCHEMA,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "ok": True,
        "status": "ready",
        "state": "local_only",
        "mode": product_mode.mode,
        "managed_mode": product_mode.public_meta(),
        "server_url_configured": bool(server_url),
        "server_url": redact_url(server_url) if server_url else None,
        "supported_local_action_families": list(SUPPORTED_LOCAL_ACTION_FAMILIES),
        "enabled_local_action_families": _enabled_action_families(product_mode),
        "reason_codes": reason_codes,
        "checks": checks,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "raw_server_responses_included": False,
            "api_key_value_included": False,
            "managed_server_calls_made": False,
            "provider_calls_made": False,
        },
    }
    if product_mode.mode == "local_only" or not product_mode.server_calls_enabled:
        result["status"] = "local_only"
        result["state"] = "local_only"
        result["reason_codes"] = [product_mode.reason]
        return result
    if not server_url:
        reason_codes.append("server-url-not-configured")
    if not _enabled_action_families(product_mode):
        reason_codes.append("no-managed-action-families-enabled")

    server = _server_reachability(server_url=server_url, timeout=timeout)
    checks["server_reachability"] = server
    result["privacy"]["managed_server_calls_made"] = bool(server_url)
    if server.get("status") != "reachable":
        reason_codes.append(str(server.get("reason") or "server-unreachable"))

    if server.get("status") == "reachable":
        contract = asyncio.run(_client_contract_readiness(server_url=server_url, timeout=timeout))
        checks["client_contract"] = contract
        if not contract.get("active"):
            reason_codes.append(str(contract.get("reason") or "client-contract-not-active"))

    if queue.get("status") == "stale":
        reason_codes.extend(str(code) for code in queue.get("reason_codes") or ["managed-feedback-queue-stale"])
    elif queue.get("status") == "unavailable":
        reason_codes.extend(str(code) for code in queue.get("reason_codes") or ["feedback-queue-unavailable"])

    clean_reasons = sorted({code for code in reason_codes if code})
    result["reason_codes"] = clean_reasons
    if clean_reasons:
        result["ok"] = False
        result["status"] = "blocked"
        result["state"] = "blocked"
    else:
        result["status"] = "ready"
        result["state"] = _ready_state_for_mode(product_mode.mode)
    return result
