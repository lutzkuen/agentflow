from __future__ import annotations

import json
import os
from typing import Any


INTERNAL_PROXY_ERROR_MESSAGE = "Internal proxy error"
INTERNAL_PROXY_ERROR_TYPE = "agentflow_proxy_error"


def debug_proxy_errors_enabled() -> bool:
    return os.getenv("AGENTFLOW_DEBUG_PROXY_ERRORS", "0") == "1"


def public_proxy_error_message(exc: BaseException | None = None) -> str:
    if exc is not None and debug_proxy_errors_enabled():
        return repr(exc)
    return INTERNAL_PROXY_ERROR_MESSAGE


def public_proxy_error_body(provider: str = "anthropic", exc: BaseException | None = None) -> dict[str, Any]:
    body = {
        "error": {
            "type": INTERNAL_PROXY_ERROR_TYPE,
            "message": public_proxy_error_message(exc),
        }
    }
    if provider == "anthropic":
        body["type"] = "error"
    return body


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
