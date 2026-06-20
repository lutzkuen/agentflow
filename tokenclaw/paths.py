from __future__ import annotations

import os
import shutil
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
    _copy_legacy_config_dir_if_needed(target)
    return target


def _copy_legacy_config_dir_if_needed(target: Path) -> None:
    legacy = safe_home_dir() / ".agentflow"
    if target.exists() or not legacy.exists() or not legacy.is_dir():
        return
    shutil.copytree(legacy, target, symlinks=True)
    message = f"Copied legacy AgentFlow config directory from {legacy} to {target}; old directory was left intact."
    import logging

    logging.getLogger("tokenclaw").warning(message)


def agentflow_config_path(*parts: str) -> Path:
    return default_config_dir().joinpath(*parts)


def default_db_path() -> Path:
    configured = env("TOKENCLAW_DB")
    if configured:
        return safe_expanduser(configured.removeprefix("sqlite:///"))
    return agentflow_config_path("tokenclaw.sqlite3")
