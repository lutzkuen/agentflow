from __future__ import annotations

import os
from typing import Any


def env(new_name: str, default: str | None = None) -> str | None:
    if new_name in os.environ:
        return os.environ[new_name]
    return default


def env_int(new_name: str, default: int) -> int:
    raw = env(new_name, str(default))
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def env_float(new_name: str, default: float) -> float:
    raw = env(new_name, str(default))
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
