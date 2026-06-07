from __future__ import annotations

from typing import Any

from agentflow_proxy.provider_context import ProviderContext


async def anthropic_messages(context: ProviderContext, request: Any) -> Any:
    return await context.anthropic_messages_handler(context, request)


async def openai_optimized(context: ProviderContext, request: Any, path: str) -> Any:
    return await context.openai_optimized_handler(context, request, path)


async def openai_passthrough(context: ProviderContext, request: Any, path: str) -> Any:
    return await context.openai_passthrough_handler(request, path)


async def openai_responses_websocket(context: ProviderContext, websocket: Any) -> None:
    await context.openai_responses_websocket_handler(websocket)
