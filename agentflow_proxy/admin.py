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
    "agentflow_proxy.codex_app_policy",
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

    @router.post("/agentflow/admin/policy-drafts/stage", response_model=None)
    async def stage_policy_draft(request: Request) -> Any:
        if not request_is_loopback(request):
            log_policy_event(
                "draft-stage",
                ok=False,
                details={
                    "source": "admin_api",
                    "client_host": request.client.host if request.client else None,
                    "error_type": "forbidden",
                    "wrote_active_policy_files": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
            )
            return JSONResponse(
                {
                    "ok": False,
                    "error": {
                        "type": "forbidden",
                        "message": "policy draft staging is only available from loopback clients",
                    },
                    "wrote_active_policy_files": False,
                    "reloaded_modules": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
                status_code=403,
            )

        try:
            body = await request.json()
        except Exception as exc:
            log_policy_event(
                "draft-stage",
                ok=False,
                details={"source": "admin_api", "error_type": "invalid_json", "exit_code": 1},
            )
            return JSONResponse(
                {
                    "schema": "agentflow.policy_draft_stage.v1",
                    "ok": False,
                    "error": {
                        "type": "invalid_json",
                        "message": str(exc),
                    },
                    "wrote_active_policy_files": False,
                    "reloaded_modules": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
                status_code=400,
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {
                    "schema": "agentflow.policy_draft_stage.v1",
                    "ok": False,
                    "error": {
                        "type": "invalid_payload",
                        "message": "expected a JSON object with bundle or section/policy",
                    },
                    "wrote_active_policy_files": False,
                    "reloaded_modules": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
                status_code=400,
            )

        from agentflow_proxy.policy_files import stage_policy_draft as stage_policy_draft_payload

        section = body.get("section")
        payload = body.get("bundle") if "bundle" in body else body.get("policy", body.get("payload"))
        result = await stage_policy_draft_payload(
            payload,
            section=section if isinstance(section, str) else None,
            draft_id=body.get("draft_id") if isinstance(body.get("draft_id"), str) else None,
            workspace=body.get("workspace") if isinstance(body.get("workspace"), str) else None,
        )
        log_policy_event(
            "draft-stage",
            ok=bool(result.get("ok")),
            details={
                "source": "admin_api",
                "client_host": request.client.host if request.client else None,
                "section": section,
                "draft_id": result.get("draft_id"),
                "changed_sections": (result.get("diff") or {}).get("changed_sections", [])
                if isinstance(result.get("diff"), dict)
                else [],
                "change_count": (result.get("diff") or {}).get("change_count", 0)
                if isinstance(result.get("diff"), dict)
                else 0,
                "wrote_active_policy_files": False,
                "reloaded_modules": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            },
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    return router
