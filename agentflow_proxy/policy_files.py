from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(mtime: float | None) -> str | None:
    if mtime is None:
        return None
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat()


def policy_file_snapshot(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    snapshot: dict[str, Any] = {
        "path": str(file_path),
        "exists": False,
        "is_file": False,
        "size": None,
        "mtime_ns": None,
        "mtime": None,
        "sha256": None,
    }
    try:
        stat = file_path.stat()
    except OSError as exc:
        snapshot["error"] = str(exc)
        return snapshot

    snapshot["exists"] = True
    snapshot["is_file"] = file_path.is_file()
    snapshot["size"] = int(stat.st_size)
    snapshot["mtime_ns"] = int(stat.st_mtime_ns)
    snapshot["mtime"] = _mtime_iso(stat.st_mtime)
    if snapshot["is_file"]:
        try:
            snapshot["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError as exc:
            snapshot["error"] = str(exc)
    return snapshot


def policy_file_status(
    path: str | Path,
    *,
    loaded_at: str,
    loaded_snapshot: dict[str, Any],
) -> dict[str, Any]:
    current = policy_file_snapshot(path)
    loaded_key = (
        bool(loaded_snapshot.get("exists")),
        bool(loaded_snapshot.get("is_file")),
        loaded_snapshot.get("size"),
        loaded_snapshot.get("mtime_ns"),
        loaded_snapshot.get("sha256"),
    )
    current_key = (
        bool(current.get("exists")),
        bool(current.get("is_file")),
        current.get("size"),
        current.get("mtime_ns"),
        current.get("sha256"),
    )
    return {
        "path": str(path),
        "loaded_at": loaded_at,
        "loaded": loaded_snapshot,
        "current": current,
        "reload_required": loaded_key != current_key,
    }
