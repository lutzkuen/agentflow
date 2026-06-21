from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tokenclaw.env import env


def safe_home_dir() -> Path:
    """Return a local home-like directory without raising when HOME cannot be resolved."""
    home = os.getenv("HOME")
    if home:
        return Path(home)
    try:
        return Path.home()
    except RuntimeError:
        return Path(tempfile.gettempdir())


def safe_expanduser(path: str | Path) -> Path:
    value = str(path)
    if value == "~":
        return safe_home_dir()
    if value.startswith("~/"):
        return safe_home_dir() / value[2:]
    try:
        return Path(value).expanduser()
    except RuntimeError:
        return Path(value)


def default_config_dir() -> Path:
    configured = env("TOKENCLAW_CONFIG_DIR") or env("TOKENCLAW_POLICY_CONFIG_DIR")
    if configured:
        return safe_expanduser(configured)
    target = safe_home_dir() / ".tokenclaw"
    return target


def tokenclaw_config_path(*parts: str) -> Path:
    return default_config_dir().joinpath(*parts)


def default_db_path() -> Path:
    configured = env("TOKENCLAW_DB")
    if configured:
        return safe_expanduser(configured.removeprefix("sqlite:///"))
    return tokenclaw_config_path("tokenclaw.sqlite3")
