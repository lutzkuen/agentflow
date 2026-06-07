import importlib.util
import os
import unittest
from unittest.mock import patch

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from starlette.datastructures import Headers

    from agentflow_proxy import server


class DummyRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = Headers(headers)


class DummyWebSocket:
    def __init__(self, headers: dict[str, str]):
        self.headers = Headers(headers)


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


if __name__ == "__main__":
    unittest.main()
