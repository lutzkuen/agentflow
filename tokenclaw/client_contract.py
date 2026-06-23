from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from tokenclaw.managed_egress import RAW_FEATURE_KEYS, assert_managed_egress_safe, managed_egress_violations


CLIENT_CONTRACT_PATH = "/v1/client-contract"
CLIENT_CONTRACT_SCHEMA = "tokenclaw.client_contract.v1"
CLIENT_CONTRACT_META_SCHEMA = "tokenclaw.client_contract_meta.v1"

_CONTRACT_CACHE: dict[tuple[str, str, str, str], dict[str, Any]] = {}

_PROTO_ROOT_KEYS = {
    "schema",
    "feature_schema_version",
    "request_facts_schema",
    "source_surface",
    "granularity",
    "app_family",
    "provider",
    "provider_family",
    "endpoint",
    "requested_model",
    "candidate_target_model",
    "replayability_level",
    "feature_only",
    "locally_executed",
    "server_content_processing",
    "provider_forwarding",
    "raw_payload_included",
    "raw_body_storage",
    "privacy_summary",
    "client_contract",
    "local_client_version",
}

_PROTO_SECTION_KEYS = {
    "schema",
}

_PRIVACY_FALSE_KEYS = {
    "raw_prompts",
    "raw_prompt",
    "raw_prompts_included",
    "raw_responses",
    "raw_response",
    "raw_responses_included",
    "raw_messages",
    "raw_messages_included",
    "provider_bodies",
    "provider_body",
    "provider_bodies_included",
    "file_paths",
    "file_path",
    "file_paths_included",
    "cache_keys",
    "cache_key",
    "cache_keys_included",
    "request_ids",
    "request_ids_included",
    "session_ids",
    "session_ids_included",
    "tenant_ids",
    "tenant_ids_included",
    "tool_payloads",
    "tool_payloads_included",
}


@dataclass(frozen=True)
class ClientContractRequest:
    provider: str
    source_surface: str
    app_family: str
    client_version: str


@dataclass(frozen=True)
class ContractClient:
    base_url: str
    headers: dict[str, str]
    timeout_seconds: float
    async_client_factory: Callable[..., Any] = httpx.AsyncClient

    async def fetch(self, request: ClientContractRequest) -> tuple[int, Any, int]:
        started = time.time()
        params = {
            "provider": request.provider,
            "source_surface": request.source_surface,
            "app_family": request.app_family,
            "client_version": request.client_version,
        }
        async with self.async_client_factory(timeout=self.timeout_seconds) as client:
            if hasattr(client, "get"):
                response = await client.get(
                    self.base_url.rstrip("/") + CLIENT_CONTRACT_PATH,
                    params=params,
                    headers=self.headers,
                )
            else:
                response = await client.post(
                    self.base_url.rstrip("/") + CLIENT_CONTRACT_PATH,
                    json=params,
                    headers=self.headers,
                )
        latency_ms = int((time.time() - started) * 1000)
        try:
            body: Any = response.json()
        except Exception:
            body = response.text[:500]
        return response.status_code, body, latency_ms


def clear_client_contract_cache() -> None:
    _CONTRACT_CACHE.clear()


def client_contract_base_meta(
    *,
    enabled: bool,
    provider: str,
    source_surface: str,
    app_family: str,
    client_version: str,
    server_url: str,
    auth_configured: bool,
    auth_source: str | None,
    reason: str | None = None,
) -> dict[str, Any]:
    meta = {
        "schema": CLIENT_CONTRACT_META_SCHEMA,
        "endpoint": CLIENT_CONTRACT_PATH,
        "enabled": bool(enabled),
        "provider": provider,
        "source_surface": source_surface,
        "app_family": app_family,
        "client_version": client_version,
        "server_configured": bool(server_url),
        "auth_configured": bool(auth_configured),
        "auth_source": auth_source,
        "status": "skipped",
        "reason": reason or "disabled",
        "cache_status": "none",
        "fallback": "local-policy",
        "active": False,
        "metadata_only": True,
        "raw_payload_included": False,
    }
    return meta


def _cache_key(request: ClientContractRequest) -> tuple[str, str, str, str]:
    return (
        request.provider.strip().lower(),
        request.source_surface.strip().lower(),
        request.app_family.strip().lower(),
        request.client_version.strip(),
    )


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _field_path(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return None
    for key in ("path", "field_path", "field", "name"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _path_parts(path: str) -> list[str]:
    return [part for part in path.replace("/", ".").split(".") if part]


def _path_is_safe(path: str) -> bool:
    parts = _path_parts(path)
    if not parts:
        return False
    for part in parts:
        if part.lower() in RAW_FEATURE_KEYS:
            return False
    return True


def _measurement_paths(plan: Any, stage: str) -> tuple[list[str], list[str]]:
    if not isinstance(plan, dict):
        return [], ["measurement-plan-not-object"]
    raw_fields = plan.get(stage)
    if raw_fields is None and stage == "preflight":
        raw_fields = plan.get("request") or plan.get("input")
    if raw_fields is None and stage == "outcome":
        raw_fields = plan.get("response") or plan.get("result")
    if raw_fields is None:
        raw_fields = []
    if not isinstance(raw_fields, list):
        return [], [f"{stage}-fields-not-list"]
    paths: list[str] = []
    errors: list[str] = []
    for item in raw_fields:
        path = _field_path(item)
        if not path:
            errors.append(f"{stage}-field-missing-path")
            continue
        if not _path_is_safe(path):
            errors.append(f"unsafe-field:{path}")
            continue
        paths.append(".".join(_path_parts(path)))
    return sorted(set(paths)), errors


def _privacy_allows_metadata_only(privacy: Any) -> bool:
    if not isinstance(privacy, dict):
        return False
    for key in _PRIVACY_FALSE_KEYS:
        if privacy.get(key) is True:
            return False
    if privacy.get("metadata_only") is False:
        return False
    if privacy.get("raw_body_storage") is True:
        return False
    return True


def normalize_client_contract(
    body: Any,
    request: ClientContractRequest,
    *,
    now: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(body, dict):
        return None, "contract-not-object"
    if body.get("schema") != CLIENT_CONTRACT_SCHEMA:
        return None, "unsupported-schema"
    expires_at = body.get("expires_at") or body.get("expiry")
    expires_ts = _parse_timestamp(expires_at)
    if expires_ts is None:
        return None, "missing-or-invalid-expiry"
    current = time.time() if now is None else now
    if expires_ts <= current:
        return None, "expired"
    for key, expected in (
        ("provider", request.provider),
        ("source_surface", request.source_surface),
        ("app_family", request.app_family),
    ):
        value = body.get(key)
        if isinstance(value, str) and value and value != expected:
            return None, f"{key}-mismatch"
    privacy = body.get("privacy") or body.get("privacy_summary")
    if not _privacy_allows_metadata_only(privacy):
        return None, "privacy-not-metadata-only"
    if body.get("provider_forwarding") is True or body.get("server_content_processing") is True:
        return None, "server-content-or-forwarding-not-allowed"
    measurement_plan = body.get("measurement_plan")
    preflight_paths, preflight_errors = _measurement_paths(measurement_plan, "preflight")
    outcome_paths, outcome_errors = _measurement_paths(measurement_plan, "outcome")
    errors = preflight_errors + outcome_errors
    if errors:
        return None, errors[0]
    allowed_actions = body.get("allowed_action_families")
    if not isinstance(allowed_actions, list):
        allowed_actions = []
    normalized = {
        "schema": CLIENT_CONTRACT_SCHEMA,
        "contract_id": str(body.get("contract_id") or "managed-client-contract"),
        "generated_at": body.get("generated_at"),
        "expires_at": str(expires_at),
        "expires_at_epoch": expires_ts,
        "provider": request.provider,
        "source_surface": request.source_surface,
        "app_family": request.app_family,
        "client_version": request.client_version,
        "client_version_min": body.get("client_version_min"),
        "measurement_plan": {
            "preflight": preflight_paths,
            "outcome": outcome_paths,
        },
        "allowed_action_families": sorted({str(item) for item in allowed_actions if isinstance(item, str) and item}),
        "local_executor_requirements": body.get("local_executor_requirements") if isinstance(body.get("local_executor_requirements"), list) else [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_requested": False,
            "raw_responses_requested": False,
            "provider_bodies_requested": False,
            "file_paths_requested": False,
            "cache_keys_requested": False,
        },
        "provenance": body.get("provenance") if isinstance(body.get("provenance"), dict) else {},
    }
    if managed_egress_violations(normalized):
        return None, "contract-egress-unsafe"
    return normalized, ""


async def fetch_or_get_client_contract(
    request: ClientContractRequest,
    *,
    enabled: bool,
    server_url: str,
    auth_configured: bool,
    auth_source: str | None,
    client: ContractClient | None = None,
) -> dict[str, Any]:
    meta = client_contract_base_meta(
        enabled=enabled,
        provider=request.provider,
        source_surface=request.source_surface,
        app_family=request.app_family,
        client_version=request.client_version,
        server_url=server_url,
        auth_configured=auth_configured,
        auth_source=auth_source,
    )
    if not enabled:
        return meta
    if not server_url:
        meta.update({"reason": "server-url-not-configured"})
        return meta
    if not auth_configured:
        meta.update({"reason": "managed-auth-not-configured"})
        return meta

    key = _cache_key(request)
    cached = _CONTRACT_CACHE.get(key)
    now = time.time()
    if cached and float(cached.get("expires_at_epoch") or 0) > now:
        meta.update({
            "status": "received",
            "reason": "cached",
            "cache_status": "hit",
            "active": True,
            "contract": copy.deepcopy(cached),
            "contract_id": cached.get("contract_id"),
            "expires_at": cached.get("expires_at"),
        })
        return meta
    if cached:
        _CONTRACT_CACHE.pop(key, None)

    if client is None:
        client = ContractClient(base_url=server_url, headers={}, timeout_seconds=1.5)
    try:
        status_code, body, latency_ms = await client.fetch(request)
        meta["latency_ms"] = latency_ms
        meta["status_code"] = status_code
        if status_code >= 400:
            meta.update({
                "status": "error",
                "reason": "server-error",
                "error": str(body)[:500],
                "cache_status": "miss",
            })
            return meta
        contract, error = normalize_client_contract(body, request, now=now)
        if contract is None:
            meta.update({
                "status": "invalid",
                "reason": "invalid-contract",
                "schema_error": error,
                "cache_status": "miss",
            })
            return meta
        assert_managed_egress_safe(contract)
        _CONTRACT_CACHE[key] = copy.deepcopy(contract)
        meta.update({
            "status": "received",
            "reason": "fetched",
            "cache_status": "stored",
            "active": True,
            "contract": contract,
            "contract_id": contract.get("contract_id"),
            "expires_at": contract.get("expires_at"),
        })
        return meta
    except httpx.TimeoutException as exc:
        meta.update({"status": "error", "reason": "timeout", "error": repr(exc), "cache_status": "miss"})
        return meta
    except httpx.NetworkError as exc:
        meta.update({"status": "error", "reason": "unreachable", "error": repr(exc), "cache_status": "miss"})
        return meta
    except Exception as exc:
        meta.update({"status": "error", "reason": "fetch-error", "error": repr(exc), "cache_status": "miss"})
        return meta


def _copy_path(source: Any, destination: dict[str, Any], parts: list[str]) -> bool:
    if not parts:
        return False
    current = source
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    target = destination
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            return False
    target[parts[-1]] = copy.deepcopy(current)
    return True


def filter_payload_by_client_contract(
    payload: dict[str, Any],
    contract_meta: dict[str, Any] | None,
    *,
    stage: str = "preflight",
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics = {
        "schema": CLIENT_CONTRACT_META_SCHEMA,
        "status": "skipped",
        "reason": "no-active-contract",
        "active": False,
        "stage": stage,
        "filtered": False,
        "raw_payload_included": False,
    }
    if isinstance(contract_meta, dict):
        diagnostics.update({
            key: copy.deepcopy(contract_meta.get(key))
            for key in (
                "enabled",
                "provider",
                "source_surface",
                "app_family",
                "client_version",
                "status",
                "reason",
                "cache_status",
                "fallback",
                "active",
                "contract_id",
                "expires_at",
                "schema_error",
            )
            if key in contract_meta
        })
    contract = contract_meta.get("contract") if isinstance(contract_meta, dict) else None
    if not isinstance(contract, dict) or contract_meta.get("active") is not True:
        return payload, diagnostics
    measurement_plan = contract.get("measurement_plan") if isinstance(contract.get("measurement_plan"), dict) else {}
    paths = measurement_plan.get(stage)
    if not isinstance(paths, list):
        paths = []
    safe_paths = [path for path in (str(item) for item in paths) if _path_is_safe(path)]
    filtered = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key in _PROTO_ROOT_KEYS
    }
    for section_key in ("input_features", "tool_features", "outcome_features", "request_facts", "grouping_identifiers"):
        section = payload.get(section_key)
        if isinstance(section, dict):
            filtered[section_key] = {
                key: copy.deepcopy(value)
                for key, value in section.items()
                if key in _PROTO_SECTION_KEYS
            }
    copied = 0
    for path in safe_paths:
        if _copy_path(payload, filtered, _path_parts(path)):
            copied += 1
    filtered = {key: value for key, value in filtered.items() if value not in ({}, [], None)}
    assert_managed_egress_safe(filtered)
    diagnostics.update({
        "status": "active",
        "reason": "contract-filtered",
        "active": True,
        "filtered": True,
        "contract_id": contract.get("contract_id"),
        "expires_at": contract.get("expires_at"),
        "allowed_field_count": len(safe_paths),
        "copied_field_count": copied,
        "allowed_action_families": contract.get("allowed_action_families") or [],
        "privacy": contract.get("privacy"),
    })
    return filtered, diagnostics
