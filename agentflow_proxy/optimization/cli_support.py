from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from agentflow_proxy.paths import default_db_path as agentflow_default_db_path


def default_db_path() -> str:
    return os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv(
        "AGENTFLOW_DB",
        str(agentflow_default_db_path()),
    )


def write_json(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True) + "\n")


def redact_url(url: str | None) -> str | None:
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment))


def redact_secret(value: Any, secret: str | None) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[redacted]")
    if isinstance(value, dict):
        return {key: redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secret(item, secret) for item in value]
    return value


def open_store_for_db(db_arg: str) -> Any:
    from agentflow_proxy.store import Store

    old_database_url = os.environ.get("AGENTFLOW_DATABASE_URL")
    try:
        if db_arg.startswith(("postgresql://", "postgres://")):
            os.environ["AGENTFLOW_DATABASE_URL"] = db_arg
            return Store()
        os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        return Store(db_arg)
    finally:
        if old_database_url is None:
            os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        else:
            os.environ["AGENTFLOW_DATABASE_URL"] = old_database_url
