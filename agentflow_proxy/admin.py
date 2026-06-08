from __future__ import annotations

import importlib
import ipaddress
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agentflow_proxy.policy_events import log_policy_event


POLICY_RELOAD_MODULES = (
    "agentflow_proxy.router",
    "agentflow_proxy.crunch",
    "agentflow_proxy.cache",
    "agentflow_proxy.routing_experiments",
    "agentflow_proxy.anthropic_proxy",
    "agentflow_proxy.openai_proxy",
    "agentflow_proxy.stats",
)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def request_is_loopback(request: Request) -> bool:
    return _is_loopback_host(request.client.host if request.client else None)


async def reload_policy_modules(after_reload: Callable[[], None] | None = None) -> dict[str, Any]:
    reloaded: list[str] = []
    for module_name in POLICY_RELOAD_MODULES:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        reloaded.append(module_name)

    if after_reload is not None:
        after_reload()

    from agentflow_proxy import stats

    policy_state = await stats.stats_policies()
    payload = {
        "ok": True,
        "schema": "agentflow.policy_reload.v1",
        "reloaded_modules": reloaded,
        "policies": policy_state,
    }
    log_policy_event(
        "reload",
        ok=True,
        details={"source": "admin_api", "reloaded_modules": reloaded, "policies": policy_state},
    )
    return payload


def create_admin_router(after_reload: Callable[[], None] | None = None) -> APIRouter:
    router = APIRouter()

    @router.post("/agentflow/admin/reload-policies", response_model=None)
    async def reload_policies(request: Request) -> Any:
        if not request_is_loopback(request):
            log_policy_event(
                "reload",
                ok=False,
                details={
                    "source": "admin_api",
                    "client_host": request.client.host if request.client else None,
                    "error_type": "forbidden",
                },
            )
            return JSONResponse(
                {
                    "ok": False,
                    "error": {
                        "type": "forbidden",
                        "message": "policy reload is only available from loopback clients",
                    },
                },
                status_code=403,
            )
        return await reload_policy_modules(after_reload=after_reload)

    return router
