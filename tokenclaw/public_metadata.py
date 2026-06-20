from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any


_PUBLIC_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RAW_LABEL_HINT_RE = re.compile(
    r"[/\\]|\s|raw|secret|api[-_]?key|cache[-_]?key|request[-_]?id|session[-_]?id|tenant[-_]?id|thread[-_]?id|"
    r"provider[-_]?body|tool[-_]?payload|sha256:[0-9a-f]{32,}",
    re.IGNORECASE,
)


def public_label(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if text and _PUBLIC_LABEL_RE.match(text) and not _RAW_LABEL_HINT_RE.search(text):
        return text
    return fallback


def public_id(value: Any, *, prefix: str, fallback: str | None = None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return fallback
    if _PUBLIC_LABEL_RE.match(text) and not _RAW_LABEL_HINT_RE.search(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def public_path_state(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if path is None:
        return {
            "configured": False,
            "path_class": None,
            "path_included": False,
        }
    expanded = os.path.abspath(os.path.expanduser(str(path)))
    home = os.path.abspath(os.path.expanduser("~"))
    if expanded.startswith(os.path.join(home, ".agentflow") + os.sep):
        path_class = "local-agentflow-home"
    elif expanded.startswith("/tmp/") or expanded.startswith("/var/tmp/"):
        path_class = "local-temp"
    elif Path(expanded).is_absolute():
        path_class = "local-path"
    else:
        path_class = "relative-local-path"
    return {
        "configured": True,
        "path_class": path_class,
        "path_included": False,
    }
