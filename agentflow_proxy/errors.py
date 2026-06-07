from __future__ import annotations

import json
from typing import Any


def upstream_error_text(raw: Any, status_code: int, limit: int = 1000) -> str:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, (dict, list)):
        text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    elif raw is None:
        text = ""
    else:
        text = str(raw)
    text = text.strip()
    if not text:
        text = f"upstream_error: status={status_code}"
    return text[:limit]
