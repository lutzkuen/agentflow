import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import server
    from agentflow_proxy.store import Store


class FakeJsonResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.content = json.dumps(body).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self):
        return self._body


class CapturingAsyncClient:
    calls = []
    response_body = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, **kwargs):
        CapturingAsyncClient.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "kwargs": kwargs,
        })
        return FakeJsonResponse(CapturingAsyncClient.response_body)


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class SafetyRegressionRouteTests(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        self.old_log_bodies = server.LOG_BODIES
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)
        server.LOG_BODIES = False
        CapturingAsyncClient.calls = []

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server.LOG_BODIES = self.old_log_bodies
        server.configure_provider(
            self.old_provider,
            anthropic_upstream=self.old_anthropic_upstream,
            openai_upstream=self.old_openai_upstream,
            openai_auth_mode=self.old_openai_auth_mode,
        )

    def test_log_bodies_defaults_disabled_when_env_is_absent(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as db:
            env = os.environ.copy()
            env.pop("AGENTFLOW_LOG_BODIES", None)
            env["AGENTFLOW_DB"] = db.name
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from agentflow_proxy import server; print(int(server.LOG_BODIES))",
                ],
                cwd=os.getcwd(),
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertEqual(result.stdout.strip(), "0")

    def test_anthropic_route_forwards_allowlisted_headers_and_does_not_log_bodies(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        CapturingAsyncClient.response_body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "secret-body-value"}],
        }

        with patch.object(server.httpx, "AsyncClient", CapturingAsyncClient):
            response = TestClient(server.app).post(
                "/v1/messages",
                json=request_body,
                headers={
                    "Authorization": "Bearer client-key",
                    "Anthropic-Beta": "prompt-caching-2024-07-31",
                    "X-Trace-Secret": "local-secret",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingAsyncClient.calls), 1)
        forwarded = {name.lower(): value for name, value in CapturingAsyncClient.calls[0]["headers"].items()}
        self.assertEqual(forwarded["authorization"], "Bearer client-key")
        self.assertEqual(forwarded["anthropic-beta"], "prompt-caching-2024-07-31")
        self.assertEqual(forwarded["content-type"], "application/json")
        self.assertNotIn("x-trace-secret", forwarded)
        self.assertEqual(CapturingAsyncClient.calls[0]["json"]["messages"][0]["content"], "secret-body-value")

        [row] = server.store.conn.execute(
            "select status_code, request_json, response_json from calls"
        ).fetchall()
        self.assertEqual(row["status_code"], 200)
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])

    def test_openai_route_forwards_allowlisted_headers_and_does_not_log_bodies(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        CapturingAsyncClient.response_body = {
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5-codex",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
        request_body = {"model": "gpt-5-codex", "input": "secret-openai-body"}

        with patch.object(server.httpx, "AsyncClient", CapturingAsyncClient):
            response = TestClient(server.app).post(
                "/v1/responses",
                json=request_body,
                headers={
                    "Authorization": "Bearer client-key",
                    "OpenAI-Project": "proj_123",
                    "X-Trace-Secret": "local-secret",
                    "Codex-Internal": "local-only",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(CapturingAsyncClient.calls), 1)
        forwarded = {name.lower(): value for name, value in CapturingAsyncClient.calls[0]["headers"].items()}
        self.assertEqual(forwarded["authorization"], "Bearer client-key")
        self.assertEqual(forwarded["openai-project"], "proj_123")
        self.assertEqual(forwarded["content-type"], "application/json")
        self.assertNotIn("x-trace-secret", forwarded)
        self.assertNotIn("codex-internal", forwarded)
        self.assertEqual(CapturingAsyncClient.calls[0]["json"]["input"], "secret-openai-body")

        [row] = server.store.conn.execute(
            "select provider, status_code, request_json, response_json from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 200)
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])


if __name__ == "__main__":
    unittest.main()
