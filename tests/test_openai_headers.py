import asyncio
import importlib.util
import os
import unittest
from unittest.mock import patch

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient
    from starlette.datastructures import Headers

    from tokenclaw import openai_proxy, server
    from tokenclaw.provider_context import ProviderContext


class DummyRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = Headers(headers)


class DummyWebSocket:
    def __init__(self, headers: dict[str, str]):
        self.headers = Headers(headers)


class FakePassthroughResponse:
    status_code = 201
    content = b'{"ok":true}'
    headers = {
        "content-type": "application/json; charset=utf-8",
        "content-encoding": "gzip",
        "transfer-encoding": "chunked",
        "connection": "keep-alive",
        "x-upstream-header": "kept",
    }


class CapturingPassthroughClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, *, headers=None, content=None, params=None):
        CapturingPassthroughClient.calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers or {}),
            "content": content,
            "params": params,
        })
        return FakePassthroughResponse()


class RecordingWebSocket(DummyWebSocket):
    def __init__(self, headers: dict[str, str]):
        super().__init__(headers)
        self.accepted = False
        self.close_calls = []

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000, reason=None):
        self.close_calls.append({"code": code, "reason": reason})


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class OpenAIHeaderForwardingTests(unittest.TestCase):
    def setUp(self):
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        server.configure_provider("openai", openai_auth_mode="client")

    def tearDown(self):
        server.configure_provider(
            self.old_provider,
            anthropic_upstream=self.old_anthropic_upstream,
            openai_upstream=self.old_openai_upstream,
            openai_auth_mode=self.old_openai_auth_mode,
        )

    def test_http_forwarding_allows_required_openai_headers(self):
        request = DummyRequest({
            "Authorization": "Bearer client-key",
            "X-Api-Key": "client-x-key",
            "OpenAI-Organization": "org_123",
            "OpenAI-Project": "proj_123",
            "OpenAI-Beta": "realtime=v1",
            "Accept": "application/json",
            "User-Agent": "agentflow-test",
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        })

        headers = server.build_openai_forward_headers(request, force_json=False)

        self.assertEqual(headers["authorization"], "Bearer client-key")
        self.assertEqual(headers["x-api-key"], "client-x-key")
        self.assertEqual(headers["openai-organization"], "org_123")
        self.assertEqual(headers["openai-project"], "proj_123")
        self.assertEqual(headers["openai-beta"], "realtime=v1")
        self.assertEqual(headers["accept"], "application/json")
        self.assertEqual(headers["user-agent"], "agentflow-test")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["content-encoding"], "gzip")

    def test_http_forwarding_drops_unknown_x_headers(self):
        request = DummyRequest({
            "Authorization": "Bearer client-key",
            "X-Internal-User": "alice@example.test",
            "X-Forwarded-For": "10.0.0.5",
            "X-Trace-Secret": "local-secret",
            "Codex-Internal": "local-codex",
            "OAI-Debug": "local-oai",
        })

        headers = server.build_openai_forward_headers(request, force_json=False)
        forwarded_names = {name.lower() for name in headers}

        self.assertIn("authorization", forwarded_names)
        self.assertNotIn("x-internal-user", forwarded_names)
        self.assertNotIn("x-forwarded-for", forwarded_names)
        self.assertNotIn("x-trace-secret", forwarded_names)
        self.assertNotIn("codex-internal", forwarded_names)
        self.assertNotIn("oai-debug", forwarded_names)

    def test_proxy_auth_mode_replaces_client_authorization(self):
        server.configure_provider("openai", openai_auth_mode="proxy")
        request = DummyRequest({
            "Authorization": "Bearer client-key",
            "OpenAI-Project": "proj_123",
        })

        with patch.dict(os.environ, {"AGENTFLOW_OPENAI_API_KEY": "proxy-key"}, clear=False):
            headers = server.build_openai_forward_headers(request, force_json=False)

        self.assertEqual(headers["authorization"], "Bearer proxy-key")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["openai-project"], "proj_123")

    def test_force_json_strips_content_encoding(self):
        request = DummyRequest({
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        })

        headers = server.build_openai_forward_headers(request)

        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn("Content-Encoding", headers)
        self.assertNotIn("content-encoding", headers)

    def test_websocket_forwarding_uses_same_allowlist_principle(self):
        websocket = DummyWebSocket({
            "Authorization": "Bearer client-key",
            "X-Api-Key": "client-x-key",
            "OpenAI-Beta": "realtime=v1",
            "User-Agent": "agentflow-test",
            "X-Internal-User": "alice@example.test",
            "Sec-WebSocket-Key": "handshake",
            "Connection": "upgrade",
        })

        headers = server.build_openai_websocket_headers(websocket)
        forwarded_names = {name.lower() for name in headers}

        self.assertEqual(headers["authorization"], "Bearer client-key")
        self.assertEqual(headers["x-api-key"], "client-x-key")
        self.assertEqual(headers["openai-beta"], "realtime=v1")
        self.assertEqual(headers["user-agent"], "agentflow-test")
        self.assertNotIn("x-internal-user", forwarded_names)
        self.assertNotIn("sec-websocket-key", forwarded_names)
        self.assertNotIn("connection", forwarded_names)

    def test_passthrough_route_filters_forwarded_and_client_headers(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        CapturingPassthroughClient.calls = []

        with patch.object(server.httpx, "AsyncClient", CapturingPassthroughClient):
            response = TestClient(server.app).post(
                "/v1/files?purpose=assistants",
                content=b"file-bytes",
                headers={
                    "Authorization": "Bearer client-key",
                    "OpenAI-Project": "proj_123",
                    "X-Trace-Secret": "local-secret",
                    "Content-Type": "application/octet-stream",
                },
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.headers["x-upstream-header"], "kept")
        self.assertNotIn("content-encoding", response.headers)
        self.assertNotIn("transfer-encoding", response.headers)
        self.assertEqual(len(CapturingPassthroughClient.calls), 1)
        call = CapturingPassthroughClient.calls[0]
        forwarded = {name.lower(): value for name, value in call["headers"].items()}
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://openai.test/v1/files")
        self.assertEqual(call["content"], b"file-bytes")
        self.assertEqual(call["params"], {"purpose": "assistants"})
        self.assertEqual(forwarded["authorization"], "Bearer client-key")
        self.assertEqual(forwarded["openai-project"], "proj_123")
        self.assertEqual(forwarded["content-type"], "application/octet-stream")
        self.assertNotIn("x-trace-secret", forwarded)

    def test_websocket_provider_mismatch_closes_without_upstream_connection(self):
        async def handler(*args):
            return None

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
            anthropic_messages_handler=handler,
            openai_optimized_handler=handler,
            openai_passthrough_handler=handler,
            openai_responses_websocket_handler=handler,
        )
        websocket = RecordingWebSocket({})

        asyncio.run(openai_proxy.openai_responses_websocket(context, websocket))

        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.close_calls, [{"code": 1008, "reason": "provider mismatch"}])


if __name__ == "__main__":
    unittest.main()
