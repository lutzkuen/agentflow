from __future__ import annotations

import json
import os
import ssl
from typing import Any


INTERNAL_PROXY_ERROR_MESSAGE = "Internal proxy error"
INTERNAL_PROXY_ERROR_TYPE = "tokenclaw_error"


def debug_proxy_errors_enabled() -> bool:
    return os.getenv("TOKENCLAW_DEBUG_PROXY_ERRORS", "0") == "1"


# Fixed, secret-free hint returned to the client verbatim when an outbound request
# fails TLS trust verification. It names only env knobs (all already handled by
# http_client.tls_verify and documented in the README), never any exception text or
# path, so surfacing it unconditionally cannot leak internal detail the way repr(exc)
# would. The knobs mirror README "Corporate TLS interception (SSL inspection)".
_TLS_TRUST_HINT = (
    "Outbound TLS certificate verification to the upstream provider failed: the "
    "provider's certificate is signed by a CA this proxy does not trust (typical of "
    "corporate TLS interception / 'SSL inspection'). No code change is needed — set "
    "one of: TOKENCLAW_CA_BUNDLE=/path/to/corporate-root-ca.pem to trust your "
    "corporate CA alongside the public roots, TOKENCLAW_TLS_TRUST_STORE=system to use "
    "the OS trust store (pip install truststore), or TOKENCLAW_TLS_VERIFY=0 to disable "
    "verification (insecure; last resort). See README 'Corporate TLS interception'."
)

# Substrings CPython/OpenSSL use for a trust-chain failure regardless of the wrapping
# exception type (httpx wraps ssl errors in ConnectError, so match down the chain).
_TLS_TRUST_MARKERS = (
    "certificate verify failed",
    "unable to get local issuer certificate",
    "self signed certificate",
    "self-signed certificate",
)


def _is_tls_trust_failure(exc: BaseException | None) -> bool:
    """True if exc (or any exception in its cause/context chain) is an upstream TLS
    trust-verification failure. Walks __cause__/__context__ with id()-dedup so a cyclic
    chain cannot loop forever."""
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, ssl.SSLCertVerificationError):
            return True
        text = str(node).lower()
        if isinstance(node, ssl.SSLError) and "certificate verify failed" in text:
            return True
        if any(marker in text for marker in _TLS_TRUST_MARKERS):
            return True
        node = node.__cause__ or node.__context__
    return False


def tls_trust_error_hint(exc: BaseException | None) -> str | None:
    """The actionable corporate-CA hint if exc is a TLS trust failure, else None."""
    if exc is not None and _is_tls_trust_failure(exc):
        return _TLS_TRUST_HINT
    return None


def public_proxy_error_message(exc: BaseException | None = None) -> str:
    # A TLS trust failure is self-inflicted config, not an internal bug, and the hint
    # is secret-free — so surface it verbatim even when debug output is off. This is the
    # only way an operator behind a corporate MITM proxy discovers the fix at runtime.
    hint = tls_trust_error_hint(exc)
    if hint is not None:
        return hint
    if exc is not None and debug_proxy_errors_enabled():
        return repr(exc)
    return INTERNAL_PROXY_ERROR_MESSAGE


def public_proxy_error_body(provider: str = "anthropic", exc: BaseException | None = None) -> dict[str, Any]:
    body = {
        "error": {
            "type": INTERNAL_PROXY_ERROR_TYPE,
            "message": public_proxy_error_message(exc),
        }
    }
    if provider == "anthropic":
        body["type"] = "error"
    return body


def upstream_error_text(raw: Any, status_code: int, limit: int = 1000) -> str:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, (dict, list)):
        text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    elif raw is None:
        text = ""
    else:
        text = str(raw)
    text = text.strip()
    if not text:
        text = f"upstream_error: status={status_code}"
    return text[:limit]
