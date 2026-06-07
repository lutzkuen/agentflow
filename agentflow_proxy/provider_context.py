from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


AnthropicMessagesHandler = Callable[[Any, Any], Awaitable[Any]]
OpenAIOptimizedHandler = Callable[[Any, Any, str], Awaitable[Any]]
OpenAIPassthroughHandler = Callable[[Any, str], Awaitable[Any]]
OpenAIWebSocketHandler = Callable[[Any], Awaitable[None]]


@dataclass(slots=True)
class ProviderContext:
    provider: str
    anthropic_upstream: str
    openai_upstream: str
    default_upstream: str
    openai_auth_mode: str
    openai_model_list: tuple[str, ...]
    store: Any
    limiter: Any
    log_bodies: bool
    http_timeout: float
    anthropic_messages_handler: AnthropicMessagesHandler
    openai_optimized_handler: OpenAIOptimizedHandler
    openai_passthrough_handler: OpenAIPassthroughHandler
    openai_responses_websocket_handler: OpenAIWebSocketHandler
