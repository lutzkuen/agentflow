from __future__ import annotations

import ipaddress
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from tokenclaw.env import env, env_any
from tokenclaw.paths import default_db_path as tokenclaw_default_db_path
from tokenclaw.paths import default_config_dir as tokenclaw_default_config_dir


def default_config_dir() -> str:
    return str(tokenclaw_default_config_dir())


def default_db_path() -> str:
    return env("TOKENCLAW_DATABASE_URL") or env("TOKENCLAW_DB", str(tokenclaw_default_db_path()))


def default_stats_url() -> str:
    return env("TOKENCLAW_STATS_URL", "http://127.0.0.1:4002/tokenclaw/stats")


def default_loopback_url(
    path: str,
    *,
    port_env_names: tuple[str, ...] = ("TOKENCLAW_ADMIN_PORT", "TOKENCLAW_PORT"),
    default_port: str = "4000",
) -> str:
    port = env_any(port_env_names, default_port)
    return f"http://127.0.0.1:{port}{path}"


def output_stream(stream: Any | None, default: Any | None = None) -> Any:
    if stream is not None:
        return stream
    return default if default is not None else sys.stdout


def write_json(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json_output(stream: Any, payload: dict[str, Any], *, pretty: bool = False) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    write_json(stream, payload)


def is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


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
    from tokenclaw.store import Store

    old_database_url = os.environ.get("TOKENCLAW_DATABASE_URL")
    try:
        if db_arg.startswith(("postgresql://", "postgres://")):
            os.environ["TOKENCLAW_DATABASE_URL"] = db_arg
            return Store()
        os.environ.pop("TOKENCLAW_DATABASE_URL", None)
        return Store(db_arg)
    finally:
        if old_database_url is None:
            os.environ.pop("TOKENCLAW_DATABASE_URL", None)
        else:
            os.environ["TOKENCLAW_DATABASE_URL"] = old_database_url


def open_metadata_report_store_for_db(db_arg: str) -> Any:
    old_warning = os.environ.get("TOKENCLAW_LEGACY_DB_WARNING")
    try:
        os.environ["TOKENCLAW_LEGACY_DB_WARNING"] = "0"
        return open_store_for_db(db_arg)
    finally:
        if old_warning is None:
            os.environ.pop("TOKENCLAW_LEGACY_DB_WARNING", None)
        else:
            os.environ["TOKENCLAW_LEGACY_DB_WARNING"] = old_warning
