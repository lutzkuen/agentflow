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


def _forbidden_workbench_response(action: str, request: Request, message: str) -> JSONResponse:
    log_policy_event(
        action,
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
                "message": message,
            },
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        status_code=403,
    )


async def _json_object_request(request: Request, *, schema: str) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        body = await request.json()
    except Exception as exc:
        return None, JSONResponse(
            {
                "schema": schema,
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
        return None, JSONResponse(
            {
                "schema": schema,
                "ok": False,
                "error": {
                    "type": "invalid_payload",
                    "message": "expected a JSON object",
                },
                "wrote_active_policy_files": False,
                "reloaded_modules": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
            },
            status_code=400,
        )
    return body, None


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
            return _forbidden_workbench_response(
                "draft-stage",
                request,
                "policy draft staging is only available from loopback clients",
            )

        body, error_response = await _json_object_request(request, schema="agentflow.policy_draft_stage.v1")
        if error_response is not None:
            log_policy_event(
                "draft-stage",
                ok=False,
                details={"source": "admin_api", "error_type": "invalid_json", "exit_code": 1},
            )
            return error_response
        assert body is not None

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

    @router.post("/agentflow/admin/policy-drafts/validate", response_model=None)
    async def validate_policy_draft(request: Request) -> Any:
        if not request_is_loopback(request):
            return _forbidden_workbench_response(
                "draft-validate",
                request,
                "policy draft validation is only available from loopback clients",
            )

        body, error_response = await _json_object_request(request, schema="agentflow.policy_draft_validate.v1")
        if error_response is not None:
            return error_response
        assert body is not None
        draft = body.get("draft") or body.get("draft_id")
        if not isinstance(draft, str) or not draft.strip():
            return JSONResponse(
                {
                    "schema": "agentflow.policy_draft_validate.v1",
                    "ok": False,
                    "status": "fail",
                    "can_apply": False,
                    "apply_blocked": True,
                    "error": {"type": "invalid_payload", "message": "draft or draft_id is required"},
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
                status_code=400,
            )

        from agentflow_proxy.policy_workbench import validate_staged_policy_draft

        result = await validate_staged_policy_draft(
            draft,
            workspace=body.get("workspace") if isinstance(body.get("workspace"), str) else None,
            config_dir=body.get("config_dir") if isinstance(body.get("config_dir"), str) else None,
            db_path=body.get("db") if isinstance(body.get("db"), str) else None,
            impact_limit=int(body.get("impact_limit", 1000) or 0),
            codex_recent_limit=int(body.get("codex_recent_limit", 200) or 0),
        )
        log_policy_event(
            "draft-validate",
            ok=bool(result.get("ok")),
            details={
                "source": "admin_api",
                "client_host": request.client.host if request.client else None,
                "draft": draft,
                "status": result.get("status"),
                "can_apply": result.get("can_apply"),
                "apply_blocked": result.get("apply_blocked"),
                "changed_sections": (result.get("draft") or {}).get("changed_sections", [])
                if isinstance(result.get("draft"), dict)
                else [],
                "section_verdicts": {
                    section.get("section"): section.get("verdict")
                    for section in result.get("sections", [])
                    if isinstance(section, dict)
                },
                "blocker_reason_codes": (result.get("apply_prerequisites") or {}).get("blocker_reason_codes", [])
                if isinstance(result.get("apply_prerequisites"), dict)
                else [],
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
                "exit_code": 0 if result.get("ok") else 1,
            },
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @router.post("/agentflow/admin/policy-drafts/apply", response_model=None)
    async def apply_policy_draft(request: Request) -> Any:
        if not request_is_loopback(request):
            return _forbidden_workbench_response(
                "draft-apply",
                request,
                "policy draft apply is only available from loopback clients",
            )

        body, error_response = await _json_object_request(request, schema="agentflow.policy_draft_apply.v1")
        if error_response is not None:
            return error_response
        assert body is not None
        draft = body.get("draft") or body.get("draft_id")
        if not isinstance(draft, str) or not draft.strip():
            return JSONResponse(
                {
                    "schema": "agentflow.policy_draft_apply.v1",
                    "ok": False,
                    "status": "blocked",
                    "error": {"type": "invalid_payload", "message": "draft or draft_id is required"},
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
                status_code=400,
            )

        sections = body.get("sections")
        if sections is not None and not isinstance(sections, list):
            sections = None
        from agentflow_proxy.policy_workbench import apply_validated_policy_draft

        result = await apply_validated_policy_draft(
            draft,
            workspace=body.get("workspace") if isinstance(body.get("workspace"), str) else None,
            config_dir=body.get("config_dir") if isinstance(body.get("config_dir"), str) else None,
            db_path=body.get("db") if isinstance(body.get("db"), str) else None,
            impact_limit=int(body.get("impact_limit", 1000) or 0),
            codex_recent_limit=int(body.get("codex_recent_limit", 200) or 0),
            sections=[section for section in sections if isinstance(section, str)] if isinstance(sections, list) else None,
            reload_policy_state=reload_policy_modules,
            event_source="admin_api",
            loopback_admin_calls_made=True,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @router.post("/agentflow/admin/policy-drafts/rollback", response_model=None)
    async def rollback_policy_draft(request: Request) -> Any:
        if not request_is_loopback(request):
            return _forbidden_workbench_response(
                "rollback",
                request,
                "policy draft rollback is only available from loopback clients",
            )

        body, error_response = await _json_object_request(request, schema="agentflow.policy_draft_rollback.v1")
        if error_response is not None:
            return error_response
        assert body is not None
        apply_id = body.get("apply_id") or body.get("backup_id")
        if not isinstance(apply_id, str) or not apply_id.strip():
            return JSONResponse(
                {
                    "schema": "agentflow.policy_draft_rollback.v1",
                    "ok": False,
                    "status": "blocked",
                    "error": {"type": "invalid_payload", "message": "apply_id is required"},
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                },
                status_code=400,
            )

        sections = body.get("sections")
        if sections is not None and not isinstance(sections, list):
            sections = None
        dry_run = bool(body.get("dry_run"))
        from agentflow_proxy.policy_workbench import rollback_policy_apply

        result = await rollback_policy_apply(
            apply_id,
            config_dir=body.get("config_dir") if isinstance(body.get("config_dir"), str) else None,
            sections=[section for section in sections if isinstance(section, str)] if isinstance(sections, list) else None,
            dry_run=dry_run,
            force=bool(body.get("force")),
            reload_policy_state=reload_policy_modules,
            event_source="admin_api",
            loopback_admin_calls_made=not dry_run,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    return router
