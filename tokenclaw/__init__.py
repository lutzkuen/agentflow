__version__ = "0.3.1"

# Public, server-free library API. Importing these pulls only the local crunch/
# cache/store modules (no fastapi/uvicorn/httpx) — see tokenclaw/library.py.
from tokenclaw.crunch import estimate_tokens_from_text
from tokenclaw.library import (
    CrunchResult,
    LocalCache,
    RouteResult,
    crunch_openai,
    crunch_request,
    route_openai,
    route_request,
)

__all__ = [
    "__version__",
    "CrunchResult",
    "LocalCache",
    "RouteResult",
    "crunch_openai",
    "crunch_request",
    "estimate_tokens_from_text",
    "route_openai",
    "route_request",
]
