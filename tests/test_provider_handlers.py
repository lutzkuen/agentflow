import sys
import unittest

from agentflow_proxy.provider_context import ProviderContext


class ProviderHandlerContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_dispatch_uses_explicit_context_callbacks(self):
        old_server = sys.modules.pop("agentflow_proxy.server", None)
        try:
            from agentflow_proxy import provider_handlers

            calls = []

            async def anthropic_handler(context_arg, request):
                calls.append(("anthropic", context_arg.provider, request))
                return {"provider": "anthropic"}

            async def openai_optimized_handler(context_arg, request, path):
                calls.append(("openai_optimized", context_arg.provider, request, path))
                return {"path": path}

            async def openai_passthrough_handler(context_arg, request, path):
                calls.append(("openai_passthrough", context_arg.provider, request, path))
                return {"path": path}

            async def websocket_handler(context_arg, websocket):
                calls.append(("websocket", context_arg.provider, websocket))

            context = ProviderContext(
                provider="anthropic",
                anthropic_upstream="https://anthropic.test",
                openai_upstream="https://openai.test",
                default_upstream="https://anthropic.test",
                openai_auth_mode="client",
                openai_model_list=("gpt-test",),
                store=object(),
                limiter=object(),
                log_bodies=False,
                http_timeout=30.0,
                anthropic_messages_handler=anthropic_handler,
                openai_optimized_handler=openai_optimized_handler,
                openai_passthrough_handler=openai_passthrough_handler,
                openai_responses_websocket_handler=websocket_handler,
            )

            self.assertEqual(
                await provider_handlers.anthropic_messages(context, "request"),
                {"provider": "anthropic"},
            )
            self.assertEqual(
                await provider_handlers.openai_optimized(context, "request", "/v1/responses"),
                {"path": "/v1/responses"},
            )
            self.assertEqual(
                await provider_handlers.openai_passthrough(context, "request", "/v1/files"),
                {"path": "/v1/files"},
            )
            await provider_handlers.openai_responses_websocket(context, "websocket")

            self.assertEqual(
                calls,
                [
                    ("anthropic", "anthropic", "request"),
                    ("openai_optimized", "anthropic", "request", "/v1/responses"),
                    ("openai_passthrough", "anthropic", "request", "/v1/files"),
                    ("websocket", "anthropic", "websocket"),
                ],
            )
            self.assertNotIn("agentflow_proxy.server", sys.modules)
        finally:
            if old_server is not None:
                sys.modules["agentflow_proxy.server"] = old_server


if __name__ == "__main__":
    unittest.main()
