from __future__ import annotations

import argparse
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from agentflow_proxy.server import DEFAULT_DB, DEFAULT_UPSTREAM, dashboard, stats, stats_activity, stats_full, stats_limiter, stats_sessions, stats_weekly, utc_now

DEFAULT_DASHBOARD_HOST = os.getenv("AGENTFLOW_DASHBOARD_HOST", "0.0.0.0")
DEFAULT_DASHBOARD_PORT = int(os.getenv("AGENTFLOW_DASHBOARD_PORT", "4002"))

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


app.get("/agentflow/stats")(stats)
app.get("/agentflow/stats/activity")(stats_activity)
app.get("/agentflow/stats/full")(stats_full)
app.get("/agentflow/stats/limiter")(stats_limiter)
app.get("/agentflow/stats/weekly")(stats_weekly)
app.get("/agentflow/stats/sessions")(stats_sessions)
app.get("/agentflow/dashboard", response_class=HTMLResponse)(dashboard)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow read-only dashboard server")
    parser.add_argument("--host", default=DEFAULT_DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("agentflow_proxy.dashboard:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
