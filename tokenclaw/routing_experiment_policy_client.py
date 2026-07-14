"""Fetch, validate, and cache the server-issued routing-experiment/canary policy.

Part of the migration described in
`docs/design/server-issued-routing-experiment-policy.md` (tracking issues #932/#934/#935):
the routing-experiment candidate matrix, sample fractions, budgets, eligibility, and safety
stops move from the local `routing_experiments.yaml` (loaded once at process start, requiring
a restart to change) into `tokenclaw_server`, delivered at runtime as a signed, TTL'd bundle.

This module owns *delivery* — fetch -> validate -> cache by (provider, source_surface,
app_family), mirroring the established `client_contract.py` pattern. Consumption lives in
routing_experiments.py (Phase 3, issue #935): the async proxies keep this cache warm via
`prefetch_server_experiment_policy`, and the synchronous `routing_experiment_decision` reads
it through `get_cached_routing_experiment_policy` under the server -> local guardrail -> off
precedence.

Trust note: unlike `/v1/client-contract` (a read-only measurement plan the server may issue
unsigned in local-dev mode), `/v1/routing-experiment-policy` drives live model-routing
decisions, so the server requires it to be signed (HTTP 503 otherwise). This client mirrors
that by rejecting any policy where `provenance.signed` is not `True`. It does **not** yet
cryptographically verify the HMAC signature against `TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET`
(mirroring `policy_bundle.verify_policy_bundle_provenance`) -- no such secret is configured
locally yet, so every fetched policy is marked `signature_verified: False,
verification_reason: "verification-secret-not-configured"`. Callers must not treat an
unverified policy as safe to enforce without that follow-up landing.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from tokenclaw.managed_egress import assert_managed_egress_safe, managed_egress_violations
from tokenclaw.policy_bundle import _hmac_signature, _secret_for_key_id, _verification_secrets


ROUTING_EXPERIMENT_POLICY_PATH = "/v1/routing-experiment-policy"
ROUTING_EXPERIMENT_POLICY_SCHEMA = "tokenclaw.routing_experiment_policy.v1"
ROUTING_EXPERIMENT_POLICY_META_SCHEMA = "tokenclaw.routing_experiment_policy_meta.v1"
ROUTING_EXPERIMENT_POLICY_LOCAL_CAPABILITY = "routing-experiment-policy"

_POLICY_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}

# Negative cache: after a failed fetch, skip re-fetching this scope until the
# deadline so a down server costs one timeout per backoff window, not one per
# provider request.
_FETCH_BACKOFF: dict[tuple[str, str, str], float] = {}
_FETCH_BACKOFF_SECONDS = 60.0

_REQUIRED_CONTROLS_KEYS = (
    "enabled",
    "kill_switch",
    "sample_rate",
    "holdout_rate",
    "daily_budget_usd",
    "min_text_chars",
    "max_text_chars",
)


@dataclass(frozen=True)
class RoutingExperimentPolicyRequest:
    provider: str
    source_surface: str
    app_family: str


@dataclass(frozen=True)
class RoutingExperimentPolicyClient:
    base_url: str
    headers: dict[str, str]
    timeout_seconds: float
    async_client_factory: Callable[..., Any] = httpx.AsyncClient

    async def fetch(self, request: RoutingExperimentPolicyRequest) -> tuple[int, Any, int]:
        started = time.time()
        params = {
            "provider": request.provider,
            "source_surface": request.source_surface,
            "app_family": request.app_family,
            "local_capability": ROUTING_EXPERIMENT_POLICY_LOCAL_CAPABILITY,
        }
        async with self.async_client_factory(timeout=self.timeout_seconds) as client:
            response = await client.get(
                self.base_url.rstrip("/") + ROUTING_EXPERIMENT_POLICY_PATH,
                params=params,
                headers=self.headers,
            )
        latency_ms = int((time.time() - started) * 1000)
        try:
            body: Any = response.json()
        except Exception:
            body = response.text[:500]
        return response.status_code, body, latency_ms


def clear_routing_experiment_policy_cache() -> None:
    _POLICY_CACHE.clear()
    _FETCH_BACKOFF.clear()


def get_cached_routing_experiment_policy(
    *, provider: str, source_surface: str, app_family: str
) -> dict[str, Any] | None:
    """Synchronous unexpired-cache lookup for the live decision path.

    The decision function is synchronous and must never wait on the network;
    the async prefetch in the proxy request handlers keeps this cache warm.
    """
    cached = _POLICY_CACHE.get((provider, source_surface, app_family))
    if cached and float(cached.get("expires_at_epoch") or 0) > time.time():
        return copy.deepcopy(cached)
    return None


def routing_experiment_policy_base_meta(
    *,
    enabled: bool,
    provider: str,
    source_surface: str,
    app_family: str,
    server_url: str,
    auth_configured: bool,
    auth_source: str | None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": ROUTING_EXPERIMENT_POLICY_META_SCHEMA,
        "endpoint": ROUTING_EXPERIMENT_POLICY_PATH,
        "enabled": bool(enabled),
        "provider": provider,
        "source_surface": source_surface,
        "app_family": app_family,
        "server_configured": bool(server_url),
        "auth_configured": bool(auth_configured),
        "auth_source": auth_source,
        "status": "skipped",
        "reason": reason or "disabled",
        "cache_status": "none",
        "fallback": "local-guardrails-only",
        "active": False,
        "metadata_only": True,
        "raw_payload_included": False,
        "hot_applied": False,
    }


def _cache_key(request: RoutingExperimentPolicyRequest) -> tuple[str, str, str]:
    return (request.provider, request.source_surface, request.app_family)


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_controls(controls: Any) -> str | None:
    if not isinstance(controls, dict):
        return "missing-or-invalid-controls"
    for key in _REQUIRED_CONTROLS_KEYS:
        if key not in controls:
            return f"controls-missing-{key}"
    if not (0.0 <= float(controls.get("sample_rate", -1)) <= 1.0):
        return "controls-sample-rate-out-of-range"
    if not (0.0 <= float(controls.get("holdout_rate", -1)) <= 1.0):
        return "controls-holdout-rate-out-of-range"
    if float(controls.get("daily_budget_usd", -1)) < 0:
        return "controls-negative-daily-budget"
    return None


def _validate_candidate(candidate: Any) -> str | None:
    if not isinstance(candidate, dict):
        return "candidate-not-object"
    for key in ("candidate_id", "requested_model", "routed_model", "provider", "source_surface"):
        if not isinstance(candidate.get(key), str) or not candidate.get(key):
            return f"candidate-missing-{key}"
    if candidate.get("feature_only") is not True:
        return "candidate-not-feature-only"
    if candidate.get("locally_executed") is not True:
        return "candidate-not-locally-executed"
    if candidate.get("managed_enforced") is not False:
        return "candidate-managed-enforced-not-allowed"
    if candidate.get("provider_forwarding") is not False:
        return "candidate-provider-forwarding-not-allowed"
    if candidate.get("server_content_processing") is not False:
        return "candidate-server-content-processing-not-allowed"
    sample_rate = candidate.get("sample_rate")
    if not _is_number(sample_rate) or not (0.0 <= float(sample_rate) <= 1.0):
        return "candidate-sample-rate-out-of-range"
    return None


def _validate_safety_stop(stop: Any) -> str | None:
    if not isinstance(stop, dict):
        return "safety-stop-not-object"
    for key in ("stop_id", "metric", "comparator", "action", "reason_code"):
        if not isinstance(stop.get(key), str) or not stop.get(key):
            return f"safety-stop-missing-{key}"
    if not _is_number(stop.get("threshold")):
        return "safety-stop-missing-threshold"
    return None


def _privacy_allows_metadata_only(privacy: Any) -> bool:
    if not isinstance(privacy, dict):
        return False
    if privacy.get("feature_only") is not True or privacy.get("metadata_only") is not True:
        return False
    for key in ("provider_bodies_included", "raw_prompts_included", "raw_responses_included"):
        if key in privacy and privacy.get(key) is not False:
            return False
    return True


def verify_routing_experiment_policy_signature(policy: dict[str, Any]) -> dict[str, Any]:
    """Best-effort HMAC check, mirroring policy_bundle.verify_policy_bundle_provenance.

    Only meaningful once TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET(S) is configured
    locally; until then this always reports verified=False with an explicit reason so
    callers cannot mistake "server claims signed" for "we checked the signature".
    """
    provenance = policy.get("provenance") if isinstance(policy, dict) else None
    if not isinstance(provenance, dict):
        return {"verified": False, "reason": "missing-provenance"}
    secrets = _verification_secrets()
    if not secrets:
        return {"verified": False, "reason": "verification-secret-not-configured"}
    secret, configured = _secret_for_key_id(provenance.get("key_id"))
    if not configured or not secret:
        return {"verified": False, "reason": "no-secret-for-key-id"}
    expected = _hmac_signature(provenance, secret)
    signature = provenance.get("signature")
    if not isinstance(signature, str) or not signature:
        return {"verified": False, "reason": "signature-missing"}
    if signature != expected:
        return {"verified": False, "reason": "signature-mismatch"}
    return {"verified": True, "reason": "signature-matched"}


def normalize_routing_experiment_policy(
    body: Any,
    request: RoutingExperimentPolicyRequest,
    *,
    now: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(body, dict):
        return None, "policy-not-object"
    if body.get("schema") != ROUTING_EXPERIMENT_POLICY_SCHEMA:
        return None, "unsupported-schema"
    expires_at = body.get("expires_at")
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
    if body.get("feature_only") is not True or body.get("locally_executed") is not True:
        return None, "policy-not-feature-only"
    if body.get("provider_forwarding") is True or body.get("server_content_processing") is True:
        return None, "server-content-or-forwarding-not-allowed"
    if not _privacy_allows_metadata_only(body.get("privacy_summary")):
        return None, "privacy-not-metadata-only"

    controls_error = _validate_controls(body.get("controls"))
    if controls_error:
        return None, controls_error

    candidates = body.get("candidates")
    if not isinstance(candidates, list):
        return None, "candidates-not-list"
    for candidate in candidates:
        error = _validate_candidate(candidate)
        if error:
            return None, error

    safety_stops = body.get("safety_stops")
    if not isinstance(safety_stops, list):
        return None, "safety-stops-not-list"
    for stop in safety_stops:
        error = _validate_safety_stop(stop)
        if error:
            return None, error

    provenance = body.get("provenance") if isinstance(body.get("provenance"), dict) else {}
    if provenance.get("signed") is not True:
        return None, "policy-not-signed"
    if provenance.get("algorithm") != "hmac-sha256":
        return None, "unsupported-signature-algorithm"
    if not isinstance(provenance.get("signature"), str) or not provenance.get("signature"):
        return None, "signature-missing"

    normalized = {
        "schema": ROUTING_EXPERIMENT_POLICY_SCHEMA,
        "policy_id": str(body.get("policy_id") or "managed-routing-experiment-policy"),
        "generated_at": body.get("generated_at"),
        "expires_at": str(expires_at),
        "expires_at_epoch": expires_ts,
        "provider": request.provider,
        "source_surface": request.source_surface,
        "app_family": request.app_family,
        "controls": copy.deepcopy(body.get("controls")),
        "candidates": copy.deepcopy(candidates),
        "safety_stops": copy.deepcopy(safety_stops),
        "provenance": copy.deepcopy(provenance),
        "signature_verification": verify_routing_experiment_policy_signature(body),
    }
    if managed_egress_violations(normalized):
        return None, "policy-egress-unsafe"
    return normalized, ""


async def fetch_or_get_routing_experiment_policy(
    request: RoutingExperimentPolicyRequest,
    *,
    enabled: bool,
    server_url: str,
    auth_configured: bool,
    auth_source: str | None,
    client: RoutingExperimentPolicyClient | None = None,
) -> dict[str, Any]:
    meta = routing_experiment_policy_base_meta(
        enabled=enabled,
        provider=request.provider,
        source_surface=request.source_surface,
        app_family=request.app_family,
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
    cached = _POLICY_CACHE.get(key)
    now = time.time()
    if cached and float(cached.get("expires_at_epoch") or 0) > now:
        meta.update({
            "status": "received",
            "reason": "cached",
            "cache_status": "hit",
            "active": True,
            "policy": copy.deepcopy(cached),
            "policy_id": cached.get("policy_id"),
            "expires_at": cached.get("expires_at"),
        })
        return meta
    if cached:
        _POLICY_CACHE.pop(key, None)
    backoff_until = _FETCH_BACKOFF.get(key, 0.0)
    if backoff_until > now:
        meta.update({
            "status": "skipped",
            "reason": "fetch-backoff",
            "cache_status": "miss",
            "backoff_remaining_seconds": round(backoff_until - now, 3),
        })
        return meta

    if client is None:
        client = RoutingExperimentPolicyClient(base_url=server_url, headers={}, timeout_seconds=1.5)
    try:
        status_code, body, latency_ms = await client.fetch(request)
        meta["latency_ms"] = latency_ms
        meta["status_code"] = status_code
        if status_code >= 400:
            _FETCH_BACKOFF[key] = now + _FETCH_BACKOFF_SECONDS
            meta.update({
                "status": "error",
                "reason": "server-error",
                "error": str(body)[:500],
                "cache_status": "miss",
            })
            return meta
        policy, error = normalize_routing_experiment_policy(body, request, now=now)
        if policy is None:
            meta.update({
                "status": "invalid",
                "reason": "invalid-policy",
                "schema_error": error,
                "cache_status": "miss",
            })
            return meta
        assert_managed_egress_safe(policy)
        _POLICY_CACHE[key] = copy.deepcopy(policy)
        meta.update({
            "status": "received",
            "reason": "fetched",
            "cache_status": "stored",
            "active": True,
            "policy": policy,
            "policy_id": policy.get("policy_id"),
            "expires_at": policy.get("expires_at"),
        })
        return meta
    except httpx.TimeoutException as exc:
        _FETCH_BACKOFF[key] = now + _FETCH_BACKOFF_SECONDS
        meta.update({"status": "error", "reason": "timeout", "error": repr(exc), "cache_status": "miss"})
        return meta
    except httpx.NetworkError as exc:
        _FETCH_BACKOFF[key] = now + _FETCH_BACKOFF_SECONDS
        meta.update({"status": "error", "reason": "unreachable", "error": repr(exc), "cache_status": "miss"})
        return meta
    except Exception as exc:
        _FETCH_BACKOFF[key] = now + _FETCH_BACKOFF_SECONDS
        meta.update({"status": "error", "reason": "fetch-error", "error": repr(exc), "cache_status": "miss"})
        return meta
