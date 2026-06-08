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

    from agentflow_proxy import openai_proxy, server
    from agentflow_proxy.limiter import TierBackoffActive
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


class SequencedAsyncClient:
    calls = []
    responses = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, **kwargs):
        SequencedAsyncClient.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "kwargs": kwargs,
        })
        return SequencedAsyncClient.responses.pop(0)


class RecordingSemaphore:
    def __init__(self, limiter, tier):
        self.limiter = limiter
        self.tier = tier

    async def __aenter__(self):
        self.limiter.entered.append(self.tier)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.limiter.exited.append(self.tier)
        return False


class RecordingLimiter:
    def __init__(self, *, raise_backoff=False):
        self.raise_backoff = raise_backoff
        self.awaited = []
        self.throttled = 0
        self.recorded_backoffs = []
        self.entered = []
        self.exited = []
        self.semaphores = {
            "haiku": RecordingSemaphore(self, "haiku"),
            "sonnet": RecordingSemaphore(self, "sonnet"),
            "opus": RecordingSemaphore(self, "opus"),
        }

    async def await_backoff(self, model):
        self.awaited.append(model)
        if self.raise_backoff:
            raise TierBackoffActive(tier="sonnet", remaining=45.0)

    async def throttle_forward(self):
        self.throttled += 1

    async def record_backoff(self, model, response_headers, default_seconds=60.0):
        self.recorded_backoffs.append({
            "model": model,
            "retry_after": response_headers.get("retry-after"),
            "default_seconds": default_seconds,
        })


async def noop_sleep(_delay):
    return None


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class SafetyRegressionRouteTests(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_limiter = server._limiter
        self.old_tier_backoff_until = server._tier_backoff_until
        self.old_provider = server.PROVIDER
        self.old_anthropic_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.old_openai_auth_mode = server.OPENAI_AUTH_MODE
        self.old_log_bodies = server.LOG_BODIES
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)
        server.LOG_BODIES = False
        CapturingAsyncClient.calls = []
        SequencedAsyncClient.calls = []
        SequencedAsyncClient.responses = []

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server._limiter = self.old_limiter
        server._tier_backoff_until = self.old_tier_backoff_until
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

    def test_openai_route_records_tier_backoff_on_retry_after(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        limiter = RecordingLimiter()
        server._limiter = limiter
        SequencedAsyncClient.responses = [
            FakeJsonResponse(
                {"error": {"message": "rate limited", "type": "rate_limit_error"}},
                status_code=429,
                headers={"retry-after": "7", "content-type": "application/json"},
            ),
            FakeJsonResponse(
                {
                    "id": "resp_1",
                    "object": "response",
                    "model": "gpt-5-codex",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            ),
        ]
        request_body = {"model": "gpt-5-codex", "input": "retry after test"}

        with patch.object(server.httpx, "AsyncClient", SequencedAsyncClient), patch.object(openai_proxy.asyncio, "sleep", noop_sleep):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(limiter.entered, ["sonnet"])
        self.assertEqual(limiter.exited, ["sonnet"])
        self.assertEqual(limiter.awaited, ["gpt-5-codex", "gpt-5-codex"])
        self.assertEqual(limiter.throttled, 2)
        self.assertEqual(limiter.recorded_backoffs, [{
            "model": "gpt-5-codex",
            "retry_after": "7",
            "default_seconds": 60.0,
        }])
        self.assertEqual(len(SequencedAsyncClient.calls), 2)
        [row] = server.store.conn.execute(
            "select provider, status_code, retry_count from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 200)
        self.assertEqual(row["retry_count"], 1)

    def test_openai_route_returns_local_429_during_long_tier_cooldown(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        limiter = RecordingLimiter(raise_backoff=True)
        server._limiter = limiter
        request_body = {"model": "gpt-5-codex", "input": "cooldown test"}

        with patch.object(server.httpx, "AsyncClient", SequencedAsyncClient):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "45")
        self.assertEqual(response.headers["x-agentflow-routed-model"], "gpt-5-codex")
        self.assertEqual(response.json()["error"]["type"], "rate_limit_error")
        self.assertEqual(limiter.entered, ["sonnet"])
        self.assertEqual(limiter.exited, ["sonnet"])
        self.assertEqual(limiter.awaited, ["gpt-5-codex"])
        self.assertEqual(SequencedAsyncClient.calls, [])
        [row] = server.store.conn.execute(
            "select provider, status_code, error from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 429)
        self.assertIn("temporarily limiting requests for sonnet tier", row["error"])

    def test_openai_stream_returns_local_rate_limit_event_during_long_tier_cooldown(self):
        server.configure_provider("openai", openai_upstream="https://openai.test", openai_auth_mode="client")
        limiter = RecordingLimiter(raise_backoff=True)
        server._limiter = limiter
        request_body = {"model": "gpt-5-codex", "stream": True, "input": "stream cooldown test"}

        with patch.object(server.httpx, "AsyncClient", SequencedAsyncClient):
            with TestClient(server.app).stream("POST", "/v1/responses", json=request_body) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", body)
        self.assertIn("rate_limit_error", body)
        self.assertIn("temporarily limiting requests for sonnet tier", body)
        self.assertEqual(limiter.entered, ["sonnet"])
        self.assertEqual(limiter.exited, ["sonnet"])
        self.assertEqual(limiter.awaited, ["gpt-5-codex"])
        self.assertEqual(SequencedAsyncClient.calls, [])
        [row] = server.store.conn.execute(
            "select provider, status_code, error, stream from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 429)
        self.assertEqual(row["stream"], 1)
        self.assertIn("temporarily limiting requests for sonnet tier", row["error"])

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
