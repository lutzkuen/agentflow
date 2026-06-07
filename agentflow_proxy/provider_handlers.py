from __future__ import annotations

from typing import Any


async def anthropic_messages(server_module: Any, request: Any) -> Any:
    return await server_module._anthropic_messages_impl(request)


async def openai_optimized(server_module: Any, request: Any, path: str) -> Any:
    return await server_module._openai_optimized_impl(request, path)


async def openai_passthrough(server_module: Any, request: Any, path: str) -> Any:
    return await server_module._openai_passthrough_impl(request, path)


async def openai_responses_websocket(server_module: Any, websocket: Any) -> None:
    await server_module._openai_responses_websocket_impl(websocket)
