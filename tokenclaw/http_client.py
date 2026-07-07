"""Shared outbound HTTP client factory with corporate-TLS-aware trust config.

TokenClaw's proxy makes outbound HTTPS calls to provider APIs (and, when opted in,
a managed server). In environments with TLS interception ("SSL inspection") the
presented certificate is signed by a corporate root CA that the default trust store
does not know, so verification fails and the proxy surfaces the request as a 500.

This module centralizes how those clients establish trust so it can be configured
without code changes. Trust source precedence (resolved once, cached):

  1. ``TOKENCLAW_TLS_VERIFY=0``  -> disable verification entirely (insecure escape
     hatch; prefer a CA bundle).
  2. ``TOKENCLAW_CA_BUNDLE`` / ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` -> trust the
     PEM at that path *in addition to* the public roots (the corporate-CA case).
  3. ``TOKENCLAW_TLS_TRUST_STORE=system`` -> use the OS trust store via the optional
     ``truststore`` package (the corporate CA is usually already installed there).
  4. default -> the standard public roots (httpx/certifi).

Only proxy/server code imports this (it needs httpx, a ``[server]`` extra). The
server-free library makes no network calls and must not import it.
"""

from __future__ import annotations

import logging
import os
import ssl
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_FALSY = {"0", "false", "no", "off"}


def _augmented_ca_context(ca_path: str) -> ssl.SSLContext:
    """A default TLS context that trusts the public roots *and* ``ca_path``."""
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - certifi always ships with httpx
        context = ssl.create_default_context()
    context.load_verify_locations(cafile=ca_path)
    return context


@lru_cache(maxsize=1)
def tls_verify() -> Any:
    """Resolve the httpx ``verify`` value from the environment (cached once)."""
    if os.getenv("TOKENCLAW_TLS_VERIFY", "").strip().lower() in _FALSY:
        logger.warning(
            "TOKENCLAW_TLS_VERIFY=0: TLS certificate verification is DISABLED for "
            "outbound calls. Prefer TOKENCLAW_CA_BUNDLE=<corporate-ca.pem> instead."
        )
        return False

    for var in ("TOKENCLAW_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        path = os.getenv(var)
        if path and os.path.exists(path):
            return _augmented_ca_context(path)

    if os.getenv("TOKENCLAW_TLS_TRUST_STORE", "").strip().lower() == "system":
        try:
            import truststore

            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except Exception:
            logger.warning(
                "TOKENCLAW_TLS_TRUST_STORE=system requested but the 'truststore' "
                "package is unavailable; falling back to the default public roots. "
                "Install it (pip install truststore) or set TOKENCLAW_CA_BUNDLE."
            )

    return True


def async_client(**kwargs: Any) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` with TokenClaw's TLS trust configuration applied.

    Accepts the same keyword arguments as ``httpx.AsyncClient``; callers that pass
    an explicit ``verify`` keep control.
    """
    kwargs.setdefault("verify", tls_verify())
    return httpx.AsyncClient(**kwargs)
