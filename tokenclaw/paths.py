from __future__ import annotations

import os
import tempfile
from pathlib import Path


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
    configured = os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR")
    if configured:
        return safe_expanduser(configured)
    return safe_home_dir() / ".agentflow"


def agentflow_config_path(*parts: str) -> Path:
    return default_config_dir().joinpath(*parts)


def default_db_path() -> Path:
    configured = os.getenv("AGENTFLOW_DB")
    if configured:
        return safe_expanduser(configured.removeprefix("sqlite:///"))
    return agentflow_config_path("agentflow.sqlite3")
