import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from agentflow_proxy.errors import upstream_error_text


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import server
    from agentflow_proxy.store import Store


class UpstreamErrorTextTest(unittest.TestCase):
    def test_empty_body_falls_back_to_status(self):
        self.assertEqual(upstream_error_text("", 400), "upstream_error: status=400")
        self.assertEqual(upstream_error_text(None, 529), "upstream_error: status=529")

    def test_json_body_is_stable_and_nonblank(self):
        self.assertEqual(
            upstream_error_text({"error": {"message": "bad request", "type": "invalid"}}, 400),
            '{"error":{"message":"bad request","type":"invalid"}}',
        )

    def test_bytes_body_is_decoded_and_trimmed(self):
        self.assertEqual(upstream_error_text(b"  upstream failed  ", 500), "upstream failed")

    def test_error_text_is_limited(self):
        self.assertEqual(upstream_error_text("x" * 20, 500, limit=7), "xxxxxxx")


class FakeRaisingStreamClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        raise RuntimeError("secret stream failure from /tmp/agentflow-token")


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class PublicProxyErrorTest(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server.configure_provider(
            self.old_provider,
            anthropic_upstream=self.old_anthropic_upstream,
            openai_upstream=self.old_openai_upstream,
            openai_auth_mode=self.old_openai_auth_mode,
        )

    def test_anthropic_internal_exception_returns_generic_public_error(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Trigger internal error."}],
        }

        with patch.object(
            server,
            "crunch_body",
            side_effect=RuntimeError("secret anthropic failure from /tmp/agentflow-token"),
        ), self.assertLogs(level="ERROR") as logs:
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "agentflow_proxy_error")
        self.assertEqual(body["error"]["message"], "Internal proxy error")
        self.assertNotIn("agentflow-token", response.text)
        self.assertIn("agentflow-token", "\n".join(logs.output))
        [row] = server.store.conn.execute("select status_code, error from calls").fetchall()
        self.assertEqual(row["status_code"], 500)
        self.assertIn("agentflow-token", row["error"])

    def test_openai_internal_exception_returns_generic_public_error(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")
        request_body = {"model": "gpt-5-codex", "input": "Trigger internal error."}

        with patch.object(
            server,
            "crunch_body",
            side_effect=RuntimeError("secret openai failure from /tmp/agentflow-token"),
        ), self.assertLogs(level="ERROR"):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["error"]["type"], "agentflow_proxy_error")
        self.assertEqual(body["error"]["message"], "Internal proxy error")
        self.assertNotIn("agentflow-token", response.text)
        [row] = server.store.conn.execute("select provider, status_code, error from calls").fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 500)
        self.assertIn("agentflow-token", row["error"])

    def test_anthropic_streaming_internal_exception_returns_generic_event(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "Trigger stream error."}],
        }

        with patch.object(server.httpx, "AsyncClient", FakeRaisingStreamClient), self.assertLogs(level="ERROR"):
            with TestClient(server.app).stream("POST", "/v1/messages", json=request_body) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Internal proxy error", body)
        self.assertIn("agentflow_proxy_error", body)
        self.assertNotIn("agentflow-token", body)
        [row] = server.store.conn.execute("select status_code, error from calls").fetchall()
        self.assertEqual(row["status_code"], 500)
        self.assertIn("agentflow-token", row["error"])

    def test_openai_streaming_internal_exception_returns_generic_event(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")
        request_body = {"model": "gpt-5-codex", "stream": True, "input": "Trigger stream error."}

        with patch.object(server.httpx, "AsyncClient", FakeRaisingStreamClient), self.assertLogs(level="ERROR"):
            with TestClient(server.app).stream("POST", "/v1/responses", json=request_body) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Internal proxy error", body)
        self.assertIn("agentflow_proxy_error", body)
        self.assertNotIn("agentflow-token", body)
        [row] = server.store.conn.execute("select provider, status_code, error from calls").fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 500)
        self.assertIn("agentflow-token", row["error"])


if __name__ == "__main__":
    unittest.main()
