from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket

load_dotenv()

PROVIDER = os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()
ANTHROPIC_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
OPENAI_UPSTREAM = os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com")
DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_PORT = int(os.getenv("AGENTFLOW_PORT", "4000"))
DEFAULT_HOST = os.getenv("AGENTFLOW_HOST", "0.0.0.0")

LOG_BODIES = os.getenv("AGENTFLOW_LOG_BODIES", "0") == "1"
HTTP_TIMEOUT = float(os.getenv("AGENTFLOW_HTTP_TIMEOUT", "600"))
MIN_REQUEST_INTERVAL_MS = int(os.getenv("AGENTFLOW_MIN_REQUEST_INTERVAL_MS", "0"))
MAX_TIER_BACKOFF_WAIT = float(os.getenv("AGENTFLOW_MAX_TIER_BACKOFF_WAIT", "30"))
# 0 = disabled (unlimited concurrency). Default 2 prevents burst collisions before global backoff coordinates.
MAX_CONCURRENT_PER_TIER = int(os.getenv("AGENTFLOW_MAX_CONCURRENT_PER_TIER", "2"))
OPENAI_MODEL_LIST = list(dict.fromkeys([
    os.getenv("AGENTFLOW_OPENAI_LARGE_MODEL", "gpt-5-codex"),
    os.getenv("AGENTFLOW_OPENAI_SMALL_MODEL", "gpt-5-mini"),
    os.getenv("AGENTFLOW_OPENAI_TINY_MODEL", "gpt-5-nano"),
    "gpt-5.5",
    "gpt-5.2-codex",
    "gpt-5-codex",
]))
OPENAI_AUTH_MODE = os.getenv("AGENTFLOW_OPENAI_AUTH_MODE", "client").lower()


def _tier_backoff_status(now: Optional[float] = None) -> list[dict[str, Any]]:
    return _limiter.status(now)


def _recent_session_spending_summary(hours: int = 24, limit: int = 10) -> list[dict[str, Any]]:
    rows = store.conn.execute("""
        SELECT session_id,
               coalesce(provider, 'anthropic') as provider,
               requested_model,
               routed_model,
               coalesce(routed_model, requested_model) as model,
               COUNT(*) as calls,
               SUM(coalesce(cost_est_usd, 0)) as cost_usd,
               SUM(coalesce(cost_baseline_usd, 0)) as baseline_usd,
               SUM(coalesce(actual_input_tokens, input_tokens_est, 0)) as input_tokens,
               SUM(coalesce(actual_output_tokens, output_tokens_est, 0)) as output_tokens,
               SUM(coalesce(thinking_output_tokens, 0)) as thinking_tokens,
               SUM(coalesce(cache_creation_input_tokens, 0)) as cache_creation_tokens,
               SUM(coalesce(cache_read_input_tokens, 0)) as cache_read_tokens
        FROM calls
        WHERE datetime(created_at) >= datetime('now', ?)
          AND session_id IS NOT NULL
        GROUP BY session_id,
                 coalesce(provider, 'anthropic'),
                 requested_model,
                 routed_model
    """, (f"-{int(hours)} hours",)).fetchall()

    by_session: dict[str, dict[str, Any]] = {}
    for row in rows:
        session_id = row["session_id"]
        bucket = by_session.setdefault(
            session_id,
            {
                "session_id": session_id,
                "sid": session_id[:8],
                "calls": 0,
                "cost_usd": 0.0,
                "baseline_savings_usd": 0.0,
                "routing_savings_usd": 0.0,
                "prompt_cache_savings_usd": 0.0,
                "thinking_tokens": 0,
                "thinking_cost_usd": 0.0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            },
        )
        calls = int(row["calls"] or 0)
        cost = float(row["cost_usd"] or 0.0)
        baseline = float(row["baseline_usd"] or 0.0)
        provider = str(row["provider"] or "anthropic").lower()
        requested_model = row["requested_model"]
        routed_model = row["routed_model"]
        model = row["model"]
        input_tokens = int(row["input_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        thinking_tokens = int(row["thinking_tokens"] or 0)
        cache_creation_tokens = int(row["cache_creation_tokens"] or 0)
        cache_read_tokens = int(row["cache_read_tokens"] or 0)

        bucket["calls"] += calls
        bucket["cost_usd"] += cost
        bucket["baseline_savings_usd"] += max(baseline - cost, 0.0)
        bucket["thinking_tokens"] += thinking_tokens
        bucket["cache_creation_tokens"] += cache_creation_tokens
        bucket["cache_read_tokens"] += cache_read_tokens

        if requested_model and routed_model and requested_model != routed_model:
            requested_cost = estimate_cost(requested_model, input_tokens, output_tokens, provider=provider) or 0.0
            routed_cost = estimate_cost(routed_model, input_tokens, output_tokens, provider=provider) or 0.0
            bucket["routing_savings_usd"] += max(requested_cost - routed_cost, 0.0)
        if cache_read_tokens:
            full_read_cost = estimate_cost(model, cache_read_tokens, 0, provider=provider) or 0.0
            bucket["prompt_cache_savings_usd"] += 0.90 * full_read_cost
        if thinking_tokens:
            bucket["thinking_cost_usd"] += estimate_cost(model, 0, thinking_tokens, provider=provider) or 0.0

    summaries = sorted(by_session.values(), key=lambda r: r["cost_usd"], reverse=True)[:limit]
    for row in summaries:
        for key in (
            "cost_usd",
            "baseline_savings_usd",
            "routing_savings_usd",
            "prompt_cache_savings_usd",
            "thinking_cost_usd",
        ):
            row[key] = round(float(row[key]), 6)
    return summaries


def _log_recent_session_spending_summary(event: str, hours: int = 24, limit: int = 10) -> None:
    rows = _recent_session_spending_summary(hours=hours, limit=limit)
    if not rows:
        print(f"agentflow_session_summary event={event} window={hours}h sessions=0", file=sys.stderr)
        return
    for row in rows:
        print(
            "agentflow_session_summary "
            f"event={event} window={hours}h session={row['sid']} calls={row['calls']} "
            f"cost_usd={row['cost_usd']:.4f} baseline_savings_usd={row['baseline_savings_usd']:.4f} "
            f"routing_savings_usd={row['routing_savings_usd']:.4f} "
            f"prompt_cache_savings_usd={row['prompt_cache_savings_usd']:.4f} "
            f"thinking_tokens={row['thinking_tokens']} thinking_cost_usd={row['thinking_cost_usd']:.4f} "
            f"cache_creation_tokens={row['cache_creation_tokens']} cache_read_tokens={row['cache_read_tokens']}",
            file=sys.stderr,
        )


from agentflow_proxy.store import Store, utc_now
from agentflow_proxy import anthropic_proxy, openai_proxy, provider_handlers
from agentflow_proxy.admin import create_admin_router
from agentflow_proxy.provider_context import ProviderContext
from agentflow_proxy.dashboard_app import create_dashboard_router
from agentflow_proxy.pricing import estimate_cost
from agentflow_proxy.limiter import (
    TierLimiter,
)
from agentflow_proxy.router import (
    HAIKU_DEFAULT, SONNET_DEFAULT, OPUS_DEFAULT,
)


_limiter = TierLimiter(
    min_request_interval_ms=MIN_REQUEST_INTERVAL_MS,
    max_tier_backoff_wait=MAX_TIER_BACKOFF_WAIT,
    max_concurrent_per_tier=MAX_CONCURRENT_PER_TIER,
)
_tier_backoff_until = _limiter.backoff_until

store = Store(DEFAULT_DB)
app = FastAPI(title=f"AgentFlow {PROVIDER.title()} Proxy", version="0.1.0")


def _provider_context() -> ProviderContext:
    return ProviderContext(
        provider=PROVIDER,
        anthropic_upstream=ANTHROPIC_UPSTREAM,
        openai_upstream=OPENAI_UPSTREAM,
        default_upstream=DEFAULT_UPSTREAM,
        openai_auth_mode=OPENAI_AUTH_MODE,
        openai_model_list=tuple(OPENAI_MODEL_LIST),
        store=store,
        limiter=_limiter,
        log_bodies=LOG_BODIES,
        http_timeout=HTTP_TIMEOUT,
        anthropic_messages_handler=anthropic_proxy.anthropic_messages,
        openai_optimized_handler=openai_proxy.openai_optimized,
        openai_passthrough_handler=openai_proxy.openai_passthrough,
        openai_responses_websocket_handler=openai_proxy.openai_responses_websocket,
    )


def _dashboard_limiter_config() -> dict[str, Any]:
    return {
        "min_request_interval_ms": MIN_REQUEST_INTERVAL_MS,
        "max_tier_backoff_wait_s": MAX_TIER_BACKOFF_WAIT,
        "max_concurrent_per_tier": MAX_CONCURRENT_PER_TIER,
    }


def _refresh_policy_module_bindings() -> None:
    from agentflow_proxy import router

    global HAIKU_DEFAULT, SONNET_DEFAULT, OPUS_DEFAULT
    HAIKU_DEFAULT = router.HAIKU_DEFAULT
    SONNET_DEFAULT = router.SONNET_DEFAULT
    OPUS_DEFAULT = router.OPUS_DEFAULT


app.include_router(
    create_dashboard_router(
        store_obj=lambda: store,
        default_db=DEFAULT_DB,
        limiter_status=_tier_backoff_status,
        limiter_config=_dashboard_limiter_config(),
        proxy_host=DEFAULT_HOST,
    )
)
app.include_router(create_admin_router(after_reload=_refresh_policy_module_bindings))


@app.on_event("startup")
async def _log_startup_session_spending_summary() -> None:
    _log_recent_session_spending_summary("startup")


@app.on_event("shutdown")
async def _log_shutdown_session_spending_summary() -> None:
    _log_recent_session_spending_summary("shutdown")


def configure_provider(
    provider: str,
    anthropic_upstream: str = ANTHROPIC_UPSTREAM,
    openai_upstream: str = OPENAI_UPSTREAM,
    openai_auth_mode: str = OPENAI_AUTH_MODE,
) -> None:
    global PROVIDER, ANTHROPIC_UPSTREAM, OPENAI_UPSTREAM, DEFAULT_UPSTREAM, OPENAI_AUTH_MODE
    provider = provider.lower()
    if provider not in {"anthropic", "openai"}:
        raise ValueError("provider must be 'anthropic' or 'openai'")
    openai_auth_mode = openai_auth_mode.lower()
    if openai_auth_mode not in {"client", "proxy"}:
        raise ValueError("openai auth mode must be 'client' or 'proxy'")
    PROVIDER = provider
    ANTHROPIC_UPSTREAM = anthropic_upstream
    OPENAI_UPSTREAM = openai_upstream
    OPENAI_AUTH_MODE = openai_auth_mode
    DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
    app.title = f"AgentFlow {PROVIDER.title()} Proxy"

def build_openai_forward_headers(request: Request, *, force_json: bool = True) -> dict[str, str]:
    return openai_proxy.build_forward_headers(_provider_context(), request, force_json=force_json)


def build_openai_websocket_headers(websocket: WebSocket) -> dict[str, str]:
    return openai_proxy.build_websocket_headers(_provider_context(), websocket)


def openai_websocket_url(path: str) -> str:
    return openai_proxy.websocket_url(_provider_context(), path)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "provider": PROVIDER,
        "db": DEFAULT_DB,
        "upstream": DEFAULT_UPSTREAM,
        "openai_auth_mode": OPENAI_AUTH_MODE if PROVIDER == "openai" else None,
        "time": utc_now(),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    if PROVIDER == "openai":
        return {"object": "list", "data": [{"id": model, "object": "model"} for model in OPENAI_MODEL_LIST]}
    return {
        "data": [
            {"id": HAIKU_DEFAULT, "type": "model"},
            {"id": SONNET_DEFAULT, "type": "model"},
            {"id": OPUS_DEFAULT, "type": "model"},
        ]
    }


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    return await provider_handlers.anthropic_messages(_provider_context(), request)



async def openai_optimized(request: Request, path: str) -> Response:
    return await provider_handlers.openai_optimized(_provider_context(), request, path)


async def openai_passthrough(request: Request, path: str) -> Response:
    return await provider_handlers.openai_passthrough(_provider_context(), request, path)


@app.post("/v1/responses")
async def openai_responses(request: Request) -> Response:
    return await openai_optimized(request, "/v1/responses")


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    return await openai_optimized(request, "/v1/chat/completions")


@app.websocket("/v1/responses")
async def openai_responses_websocket(websocket: WebSocket) -> None:
    await provider_handlers.openai_responses_websocket(_provider_context(), websocket)


@app.api_route("/v1/responses/{rest:path}", methods=["GET", "POST", "DELETE"])
async def openai_responses_passthrough(request: Request, rest: str) -> Response:
    return await openai_passthrough(request, f"/v1/responses/{rest}")


@app.api_route("/v1/files", methods=["GET", "POST", "DELETE"])
async def openai_files_root_passthrough(request: Request) -> Response:
    return await openai_passthrough(request, "/v1/files")


@app.api_route("/v1/files/{rest:path}", methods=["GET", "POST", "DELETE"])
async def openai_files_passthrough(request: Request, rest: str) -> Response:
    return await openai_passthrough(request, f"/v1/files/{rest}")


@app.api_route("/v1/uploads", methods=["GET", "POST", "DELETE"])
async def openai_uploads_root_passthrough(request: Request) -> Response:
    return await openai_passthrough(request, "/v1/uploads")


@app.api_route("/v1/uploads/{rest:path}", methods=["GET", "POST", "DELETE"])
async def openai_uploads_passthrough(request: Request, rest: str) -> Response:
    return await openai_passthrough(request, f"/v1/uploads/{rest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow provider-specific local proxy")
    parser.add_argument("--provider", choices=("anthropic", "openai"), default=PROVIDER)
    parser.add_argument("--anthropic-upstream", default=ANTHROPIC_UPSTREAM)
    parser.add_argument("--openai-upstream", default=OPENAI_UPSTREAM)
    parser.add_argument("--openai-auth-mode", choices=("client", "proxy"), default=OPENAI_AUTH_MODE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ["AGENTFLOW_PROVIDER"] = args.provider
    os.environ["AGENTFLOW_ANTHROPIC_UPSTREAM"] = args.anthropic_upstream
    os.environ["AGENTFLOW_OPENAI_UPSTREAM"] = args.openai_upstream
    os.environ["AGENTFLOW_OPENAI_AUTH_MODE"] = args.openai_auth_mode
    configure_provider(args.provider, args.anthropic_upstream, args.openai_upstream, args.openai_auth_mode)
    import uvicorn
    if args.reload:
        uvicorn.run("agentflow_proxy.server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
