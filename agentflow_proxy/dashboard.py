from __future__ import annotations

import argparse
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from agentflow_proxy import stats as stats_views
from agentflow_proxy.limiter import TierLimiter
from agentflow_proxy.store import Store, utc_now

DEFAULT_DASHBOARD_HOST = os.getenv("AGENTFLOW_DASHBOARD_HOST", "0.0.0.0")
DEFAULT_DASHBOARD_PORT = int(os.getenv("AGENTFLOW_DASHBOARD_PORT", "4002"))
DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", os.path.expanduser("~/.agentflow/agentflow.sqlite3"))
PROVIDER = os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()
ANTHROPIC_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
OPENAI_UPSTREAM = os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com")
DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
MIN_REQUEST_INTERVAL_MS = int(os.getenv("AGENTFLOW_MIN_REQUEST_INTERVAL_MS", "0"))
MAX_TIER_BACKOFF_WAIT = float(os.getenv("AGENTFLOW_MAX_TIER_BACKOFF_WAIT", "30"))
MAX_CONCURRENT_PER_TIER = int(os.getenv("AGENTFLOW_MAX_CONCURRENT_PER_TIER", "2"))

store = Store(DEFAULT_DB)
limiter = TierLimiter(
    min_request_interval_ms=MIN_REQUEST_INTERVAL_MS,
    max_tier_backoff_wait=MAX_TIER_BACKOFF_WAIT,
    max_concurrent_per_tier=MAX_CONCURRENT_PER_TIER,
)

app = FastAPI(title="AgentFlow Dashboard", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "dashboard-read-only",
        "db": DEFAULT_DB,
        "upstream": DEFAULT_UPSTREAM,
        "time": utc_now(),
    }


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/agentflow/dashboard")


@app.get("/agentflow/stats")
async def stats() -> dict[str, Any]:
    return await stats_views.stats(store, DEFAULT_DB)


@app.get("/agentflow/stats/activity")
async def stats_activity(limit: int = 100) -> dict[str, Any]:
    return await stats_views.stats_activity(store, limit=limit)


@app.get("/agentflow/stats/full")
async def stats_full() -> dict[str, Any]:
    return await stats_views.stats_full(store)


@app.get("/agentflow/stats/limiter")
async def stats_limiter() -> dict[str, Any]:
    return await stats_views.stats_limiter(
        store,
        limiter.status,
        {
            "min_request_interval_ms": MIN_REQUEST_INTERVAL_MS,
            "max_tier_backoff_wait_s": MAX_TIER_BACKOFF_WAIT,
            "max_concurrent_per_tier": MAX_CONCURRENT_PER_TIER,
        },
    )


@app.get("/agentflow/stats/weekly")
async def stats_weekly() -> dict[str, Any]:
    return await stats_views.stats_weekly(store)


@app.get("/agentflow/stats/sessions")
async def stats_sessions() -> dict[str, Any]:
    return await stats_views.stats_sessions(store)


@app.get("/agentflow/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return stats_views.dashboard_html()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow read-only dashboard server")
    parser.add_argument("--host", default=DEFAULT_DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("agentflow_proxy.dashboard:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
