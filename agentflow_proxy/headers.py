from __future__ import annotations

import gzip
import json
import os
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional runtime dependency
    zstd = None


ANTHROPIC_FORWARD_HEADERS = {
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "content-type",
    "accept",
    "user-agent",
}
OPENAI_FORWARD_HEADERS = {
    "authorization",
    "x-api-key",
    "content-type",
    "content-encoding",
    "accept",
    "user-agent",
    "openai-organization",
    "openai-project",
    "openai-beta",
}
OPENAI_WEBSOCKET_FORWARD_HEADERS = {
    "authorization",
    "x-api-key",
    "accept",
    "user-agent",
    "openai-organization",
    "openai-project",
    "openai-beta",
}
HTTP_HOP_BY_HOP_HEADERS = {
    "connection",
    "host",
    "content-length",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "proxy-authenticate",
    "keep-alive",
    "te",
    "trailer",
}
WEBSOCKET_HANDSHAKE_HEADERS = {
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
}


@dataclass(frozen=True)
class ClientJsonRequestError(Exception):
    message: str


def client_json_error_body(provider: str, message: str) -> dict[str, Any]:
    error = {
        "type": "invalid_request_error",
        "message": message,
    }
    if provider == "anthropic":
        return {"type": "error", "error": error}
    return {"error": error}


async def read_json_object_body(
    request: Any,
    *,
    allow_compressed: bool = False,
    passthrough_unsupported_encoding: bool = False,
) -> Optional[dict[str, Any]]:
    body = await request.body()
    encoding = (request.headers.get("content-encoding") or "").lower().strip()
    try:
        if allow_compressed and encoding:
            if encoding in {"zstd", "zstandard"}:
                if zstd is None:
                    if passthrough_unsupported_encoding:
                        return None
                    raise ClientJsonRequestError("Unsupported content encoding.")
                body = zstd.ZstdDecompressor().decompress(body)
            elif encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding in {"deflate", "zlib"}:
                body = zlib.decompress(body)
            elif passthrough_unsupported_encoding:
                return None
            else:
                raise ClientJsonRequestError("Unsupported content encoding.")
        parsed = json.loads(body)
    except ClientJsonRequestError:
        raise
    except Exception as exc:
        raise ClientJsonRequestError("Malformed JSON request body.") from exc
    if not isinstance(parsed, dict):
        raise ClientJsonRequestError("JSON request body must be an object.")
    return parsed


def _copy_allowed_headers(
    source: Iterable[tuple[str, str]],
    *,
    allowed: set[str],
    blocked: set[str],
    auth_mode: str = "client",
) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in source:
        lname = name.lower()
        if (
            lname in allowed
            and lname not in blocked
            and not (lname == "authorization" and auth_mode == "proxy")
        ):
            headers[name] = value
    return headers


def build_anthropic_forward_headers(
    request_headers: Mapping[str, str],
    *,
    anthropic_version: str | None = None,
) -> dict[str, str]:
    headers = _copy_allowed_headers(
        request_headers.items(),
        allowed=ANTHROPIC_FORWARD_HEADERS,
        blocked=set(),
    )
    headers.setdefault(
        "anthropic-version",
        anthropic_version or os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
    )
    headers["content-type"] = "application/json"
    return headers


def build_openai_forward_headers(
    request_headers: Mapping[str, str],
    *,
    auth_mode: str,
    api_key: str | None = None,
    force_json: bool = True,
) -> dict[str, str]:
    headers = _copy_allowed_headers(
        request_headers.items(),
        allowed=OPENAI_FORWARD_HEADERS,
        blocked=HTTP_HOP_BY_HOP_HEADERS,
        auth_mode=auth_mode,
    )
    if auth_mode == "proxy" or "authorization" not in {k.lower() for k in headers}:
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    if force_json:
        headers.pop("content-encoding", None)
        headers.pop("Content-Encoding", None)
        headers["content-type"] = "application/json"
    return headers


def build_openai_websocket_headers(
    websocket_headers: Mapping[str, str],
    *,
    auth_mode: str,
    api_key: str | None = None,
) -> dict[str, str]:
    headers = _copy_allowed_headers(
        websocket_headers.items(),
        allowed=OPENAI_WEBSOCKET_FORWARD_HEADERS,
        blocked=HTTP_HOP_BY_HOP_HEADERS | WEBSOCKET_HANDSHAKE_HEADERS,
        auth_mode=auth_mode,
    )
    if auth_mode == "proxy" or "authorization" not in {k.lower() for k in headers}:
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    return headers
