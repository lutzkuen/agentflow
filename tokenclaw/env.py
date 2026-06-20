from __future__ import annotations

import logging
import os
import warnings
from typing import Any


def legacy_name(new_name: str) -> str:
    if new_name.startswith("TOKENCLAW_"):
        return "AGENTFLOW_" + new_name[len("TOKENCLAW_") :]
    return new_name


def env(new_name: str, default: str | None = None, *, old_name: str | None = None) -> str | None:
    legacy = old_name or legacy_name(new_name)
    if new_name in os.environ:
        return os.environ[new_name]
    if legacy and legacy in os.environ:
        message = f"{legacy} is deprecated; use {new_name}"
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        logging.getLogger("tokenclaw").warning(message)
        return os.environ[legacy]
    return default


def env_int(new_name: str, default: int, *, old_name: str | None = None) -> int:
    raw = env(new_name, str(default), old_name=old_name)
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def env_float(new_name: str, default: float, *, old_name: str | None = None) -> float:
    raw = env(new_name, str(default), old_name=old_name)
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return default


def env_any(names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = env(name)
        if value is not None:
            return value
    return default
