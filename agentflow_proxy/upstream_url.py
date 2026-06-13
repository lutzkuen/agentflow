from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


SENSITIVE_QUERY_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "client_secret",
    "code",
    "key",
    "sig",
    "signature",
    "token",
}


class UpstreamUrlError(ValueError):
    pass


def normalize_openai_upstream_base_url(raw_url: str | None) -> str:
    raw = (raw_url or "").strip()
    if not raw:
        raise UpstreamUrlError("OpenAI upstream base URL must not be empty.")
    if any(ch.isspace() for ch in raw):
        raise UpstreamUrlError("OpenAI upstream base URL must not contain whitespace.")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise UpstreamUrlError("OpenAI upstream base URL must start with http:// or https://.")
    if not parsed.netloc or not parsed.hostname:
        raise UpstreamUrlError("OpenAI upstream base URL must include a host.")
    if parsed.fragment:
        raise UpstreamUrlError("OpenAI upstream base URL must not include a fragment.")

    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def redact_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return raw_url
    parsed = urlparse(str(raw_url))
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in SENSITIVE_QUERY_KEYS:
            query_items.append((key, "[redacted]"))
        else:
            query_items.append((key, value))
    query = urlencode(query_items, doseq=True)
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, ""))


def _route_path_for_base(base_path: str, route_path: str, query: str) -> str:
    base_parts = [part.lower() for part in base_path.split("/") if part]
    route = "/" + route_path.lstrip("/")
    if route.startswith("/v1/"):
        if base_parts and base_parts[-1] == "v1":
            return route[len("/v1") :]
        if "api-version=" in query.lower() or (
            len(base_parts) >= 3 and base_parts[-3:-1] == ["openai", "deployments"]
        ):
            return route[len("/v1") :]
    return route


def join_openai_upstream_url(
    upstream_base_url: str,
    route_path: str,
    *,
    request_query: dict[str, str] | None = None,
) -> str:
    base = normalize_openai_upstream_base_url(upstream_base_url)
    parsed = urlparse(base)
    base_path = parsed.path.rstrip("/")
    route = _route_path_for_base(base_path, route_path, parsed.query)
    path = f"{base_path}{route}" if base_path else route
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if request_query:
        query_items.extend((str(key), str(value)) for key, value in request_query.items())
    query = urlencode(query_items, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def openai_websocket_url(upstream_base_url: str, route_path: str) -> str:
    joined = join_openai_upstream_url(upstream_base_url, route_path)
    parsed = urlparse(joined)
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        raise UpstreamUrlError("OpenAI WebSocket upstream URL must start with http:// or https://.")
    return urlunparse((scheme, parsed.netloc, parsed.path, "", parsed.query, ""))
