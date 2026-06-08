from __future__ import annotations

import argparse
import os
from typing import Any, Callable

from fastapi import FastAPI

from agentflow_proxy.dashboard_app import create_dashboard_app
from agentflow_proxy.limiter import TierLimiter
from agentflow_proxy.store import Store

DEFAULT_DASHBOARD_HOST = os.getenv("AGENTFLOW_DASHBOARD_HOST", "0.0.0.0")
DEFAULT_DASHBOARD_PORT = int(os.getenv("AGENTFLOW_DASHBOARD_PORT", "4002"))
DEFAULT_PROXY_HOST = os.getenv("AGENTFLOW_PROXY_HOST") or os.getenv("AGENTFLOW_HOST")
DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", os.path.expanduser("~/.agentflow/agentflow.sqlite3"))
PROVIDER = os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()
ANTHROPIC_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
OPENAI_UPSTREAM = os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com")
DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
MIN_REQUEST_INTERVAL_MS = int(os.getenv("AGENTFLOW_MIN_REQUEST_INTERVAL_MS", "0"))
MAX_TIER_BACKOFF_WAIT = float(os.getenv("AGENTFLOW_MAX_TIER_BACKOFF_WAIT", "30"))
MAX_CONCURRENT_PER_TIER = int(os.getenv("AGENTFLOW_MAX_CONCURRENT_PER_TIER", "2"))


def _limiter_config() -> dict[str, Any]:
    return {
        "min_request_interval_ms": MIN_REQUEST_INTERVAL_MS,
        "max_tier_backoff_wait_s": MAX_TIER_BACKOFF_WAIT,
        "max_concurrent_per_tier": MAX_CONCURRENT_PER_TIER,
    }


def create_app() -> FastAPI:
    store = Store(DEFAULT_DB)
    limiter = TierLimiter(
        min_request_interval_ms=MIN_REQUEST_INTERVAL_MS,
        max_tier_backoff_wait=MAX_TIER_BACKOFF_WAIT,
        max_concurrent_per_tier=MAX_CONCURRENT_PER_TIER,
    )
    return create_dashboard_app(
        store_obj=store,
        default_db=DEFAULT_DB,
        upstream=DEFAULT_UPSTREAM,
        limiter_status=limiter.status,
        limiter_config=_limiter_config(),
        proxy_host=DEFAULT_PROXY_HOST,
        dashboard_host=DEFAULT_DASHBOARD_HOST,
    )


class LazyDashboardApp:
    def __init__(self, factory: Callable[[], FastAPI]) -> None:
        self._factory = factory
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            self._app = self._factory()
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)


app = LazyDashboardApp(create_app)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow read-only dashboard server")
    parser.add_argument("--host", default=DEFAULT_DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
