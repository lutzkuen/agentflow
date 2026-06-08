from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from agentflow_proxy import stats as stats_views
from agentflow_proxy.store import utc_now


LimiterStatus = Callable[[], list[dict[str, Any]]]
StoreSource = Any | Callable[[], Any]


def _store(store_source: StoreSource) -> Any:
    return store_source() if callable(store_source) else store_source


def create_dashboard_router(
    *,
    store_obj: StoreSource,
    default_db: str,
    limiter_status: LimiterStatus,
    limiter_config: dict[str, Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/agentflow/stats")
    async def stats() -> dict[str, Any]:
        return await stats_views.stats(_store(store_obj), default_db)

    @router.get("/agentflow/stats/activity")
    async def stats_activity(limit: int = 100) -> dict[str, Any]:
        return await stats_views.stats_activity(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/full")
    async def stats_full() -> dict[str, Any]:
        return await stats_views.stats_full(_store(store_obj))

    @router.get("/agentflow/stats/usage")
    async def stats_usage() -> dict[str, Any]:
        return await stats_views.stats_usage_by_owner(_store(store_obj))

    @router.get("/agentflow/stats/limiter")
    async def stats_limiter() -> dict[str, Any]:
        return await stats_views.stats_limiter(_store(store_obj), limiter_status, limiter_config)

    @router.get("/agentflow/stats/policies")
    async def stats_policies() -> dict[str, Any]:
        return await stats_views.stats_policies()

    @router.get("/agentflow/stats/policy-events")
    async def stats_policy_events(limit: int = 50) -> dict[str, Any]:
        return await stats_views.stats_policy_events(limit=limit)

    @router.get("/agentflow/stats/weekly")
    async def stats_weekly() -> dict[str, Any]:
        return await stats_views.stats_weekly(_store(store_obj))

    @router.get("/agentflow/stats/sessions")
    async def stats_sessions() -> dict[str, Any]:
        return await stats_views.stats_sessions(_store(store_obj))

    @router.get("/agentflow/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return stats_views.dashboard_html()

    return router


def create_dashboard_app(
    *,
    store_obj: StoreSource,
    default_db: str,
    upstream: str,
    limiter_status: LimiterStatus,
    limiter_config: dict[str, Any],
) -> FastAPI:
    app = FastAPI(title="AgentFlow Dashboard", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "mode": "dashboard-read-only",
            "db": default_db,
            "upstream": upstream,
            "time": utc_now(),
        }

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse("/agentflow/dashboard")

    app.include_router(
        create_dashboard_router(
            store_obj=store_obj,
            default_db=default_db,
            limiter_status=limiter_status,
            limiter_config=limiter_config,
        )
    )
    return app
