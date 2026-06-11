import base64
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    import agentflow_proxy.cache as cache_module
    from agentflow_proxy.crunch import crunch_body, estimate_tokens_from_text
    from agentflow_proxy.recommendations import build_optimization_unit, pattern_feature_diagnostics
    from agentflow_proxy.router import categorize_request, extract_text, route_model
    from agentflow_proxy import server
    from agentflow_proxy.store import Store


STREAM_FRAMES = [
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":2}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


class FakeStreamResponse:
    status_code = 200
    headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        for frame in STREAM_FRAMES:
            yield frame


class FakeAsyncClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        FakeAsyncClient.calls += 1
        return FakeStreamResponse()


class FakeStreamErrorResponse:
    status_code = 400
    headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        yield b'{"type":"error","error":{"type":"invalid_request_error","message":"stream body failed"}}'


class FakeStreamErrorClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        return FakeStreamErrorResponse()


class FakeEmptyErrorResponse:
    status_code = 400
    headers = {"content-type": "text/plain"}
    content = b""
    text = ""

    def json(self):
        raise ValueError("not json")


class FakeEmptyErrorClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return FakeEmptyErrorResponse()


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class StreamingCacheTest(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_provider = server.PROVIDER
        self.old_upstream = server.ANTHROPIC_UPSTREAM
        self.old_openai_upstream = server.OPENAI_UPSTREAM
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        FakeAsyncClient.calls = 0

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server.configure_provider(
            self.old_provider,
            anthropic_upstream=self.old_upstream,
            openai_upstream=self.old_openai_upstream,
        )

    def _pattern_features_for_request(self, request_body):
        category = categorize_request(request_body)
        crunched, crunch_meta = crunch_body(request_body)
        routed_model, routing_meta = route_model(crunched)
        input_tokens = estimate_tokens_from_text(extract_text(crunched))
        unit = build_optimization_unit(
            provider="anthropic",
            path="/v1/messages",
            requested_model=str(crunched.get("model")),
            routed_model=str(routed_model),
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta={"status": "skipped", "reason": "not-evaluated"},
            category=category,
            stream=True,
            input_tokens_est=input_tokens,
            session_id="streaming-cache-test",
        )
        return pattern_feature_diagnostics(unit)

    def _streaming_cache_rule_for_request(self, request_body, *, canary_fraction=1.0):
        features = self._pattern_features_for_request(request_body)
        return cache_module.normalize_cache_pattern_rules([{
            "id": "reviewed-static-streaming-cache",
            "policy_source": "managed-recommended",
            "candidate_id": "streaming-static-candidate",
            "conditions": {
                "pattern_hashes": features["pattern_hashes"],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": features["category"],
                "stream": True,
                "cacheability_bucket": "high",
                "static_information_hint": True,
                "time_sensitive_hint": False,
                "user_specific_hint": False,
            },
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "canary-only",
                "canary_enabled": True,
                "canary_fraction": canary_fraction,
                "canary_salt": "streaming-cache-http-test",
                "canary_unit": "request_fingerprint",
            },
            "action": {
                "type": "exact_cache_pattern",
                "streaming": True,
                "allow_tool_calls": False,
            },
        }])[0]

    def test_streamed_non_tool_response_is_not_cached_without_explicit_rule(self):
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }

        with patch.object(server.httpx, "AsyncClient", FakeAsyncClient):
            client = TestClient(server.app)
            with client.stream("POST", "/v1/messages", json=request_body) as first:
                first_body = b"".join(first.iter_bytes())
            with client.stream("POST", "/v1/messages", json=request_body) as second:
                second_body = b"".join(second.iter_bytes())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.headers["x-agentflow-cache"], "skip-streaming")
        self.assertEqual(second.headers["x-agentflow-cache"], "skip-streaming")
        self.assertEqual(first_body, b"".join(STREAM_FRAMES))
        self.assertEqual(second_body, first_body)
        self.assertEqual(FakeAsyncClient.calls, 2)

        rows = server.store.conn.execute(
            "select stream, cache_hit, cache_json from calls order by created_at"
        ).fetchall()
        self.assertEqual([row["stream"] for row in rows], [1, 1])
        self.assertEqual([row["cache_hit"] for row in rows], [0, 0])
        first_cache = json.loads(rows[0]["cache_json"])
        second_cache = json.loads(rows[1]["cache_json"])
        self.assertEqual(first_cache["reason"], "streaming-pattern-rule-required")
        self.assertEqual(second_cache["reason"], "streaming-pattern-rule-required")

    def test_streamed_static_response_replays_only_with_explicit_canary_rule(self):
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        old_rules = cache_module.CACHE_PATTERN_RULES
        try:
            cache_module.CACHE_PATTERN_RULES = (self._streaming_cache_rule_for_request(request_body),)
            with patch.object(server.httpx, "AsyncClient", FakeAsyncClient):
                client = TestClient(server.app)
                with client.stream("POST", "/v1/messages", json=request_body) as first:
                    first_body = b"".join(first.iter_bytes())
                with client.stream("POST", "/v1/messages", json=request_body) as second:
                    second_body = b"".join(second.iter_bytes())

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.headers["x-agentflow-cache"], "miss")
            self.assertEqual(second.headers["x-agentflow-cache"], "hit")
            self.assertEqual(first_body, b"".join(STREAM_FRAMES))
            self.assertEqual(second_body, first_body)
            self.assertEqual(FakeAsyncClient.calls, 1)

            rows = server.store.conn.execute(
                "select stream, cache_hit, cache_json from calls order by created_at"
            ).fetchall()
            self.assertEqual([row["stream"] for row in rows], [1, 1])
            self.assertEqual([row["cache_hit"] for row in rows], [0, 1])
            first_cache = json.loads(rows[0]["cache_json"])
            second_cache = json.loads(rows[1]["cache_json"])
            self.assertEqual(first_cache["reason"], "streaming-exact-pattern-miss")
            self.assertEqual(first_cache["pattern_rule"]["canary"]["status"], "applied")
            self.assertEqual(second_cache["reason"], "streaming-exact-match")
            self.assertEqual(second_cache["hit_type"], "streaming-exact")
            self.assertEqual(second_cache["pattern_rule"]["candidate_id"], "streaming-static-candidate")
            self.assertEqual(second_cache["pattern_rule"]["canary"]["status"], "applied")
            self.assertGreater(second_cache["estimated_saved_cost_usd"], 0)
            self.assertEqual(second_cache["stream_replay"]["media_type"], "text/event-stream")
            self.assertEqual(second_cache["stream_replay"]["frame_count"], len(STREAM_FRAMES))
            self.assertTrue(second_cache["stream_replay"]["complete"])
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

    def test_malformed_stream_cache_entry_bypasses_and_refreshes_from_upstream(self):
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        old_rules = cache_module.CACHE_PATTERN_RULES
        session_id = "streaming-cache-test"
        try:
            cache_module.CACHE_PATTERN_RULES = (self._streaming_cache_rule_for_request(request_body),)
            crunched, _crunch_meta = crunch_body(json.loads(json.dumps(request_body)))
            routed_model, _routing_meta = route_model(crunched, session_id=session_id)
            crunched["model"] = routed_model
            key = cache_module.cache_key_for(
                crunched,
                "/v1/messages",
                provider="anthropic",
                upstream=server.ANTHROPIC_UPSTREAM,
                replay_scope="session",
                replay_scope_id=session_id,
            )
            server.store.set_cache(
                key,
                str(crunched.get("model")),
                len(json.dumps(crunched)),
                {
                    "agentflow_cache_type": "sse-stream",
                    "version": 1,
                    "provider": "anthropic",
                    "frames_b64": [base64.b64encode(b"cached garbage\n\n").decode("ascii")],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                    "output_text": "cached garbage",
                },
            )

            with patch.object(server.httpx, "AsyncClient", FakeAsyncClient):
                client = TestClient(server.app)
                with client.stream(
                    "POST",
                    "/v1/messages",
                    json=request_body,
                    headers={"x-session-id": session_id},
                ) as response:
                    body = b"".join(response.iter_bytes())

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-agentflow-cache"], "miss")
            self.assertEqual(body, b"".join(STREAM_FRAMES))
            self.assertEqual(FakeAsyncClient.calls, 1)

            [row] = server.store.conn.execute(
                "select stream, cache_hit, cache_json from calls"
            ).fetchall()
            self.assertEqual(row["stream"], 1)
            self.assertEqual(row["cache_hit"], 0)
            cache_meta = json.loads(row["cache_json"])
            self.assertEqual(cache_meta["status"], "bypassed")
            self.assertEqual(cache_meta["reason"], "malformed-stream-cache")
            self.assertEqual(cache_meta["malformed_stream_cache"]["reason"], "sse-data-missing")
            self.assertFalse(cache_meta["malformed_stream_cache"]["raw_payload_included"])
        finally:
            cache_module.CACHE_PATTERN_RULES = old_rules

    def test_streamed_tool_turns_are_not_cached_by_tool_cache_toggle(self):
        old_cache_tool_calls = cache_module.CACHE_TOOL_CALLS
        old_cwd = os.getcwd()
        cache_module.CACHE_TOOL_CALLS = True
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                os.makedirs("src")
                with open("src/example.py", "w", encoding="utf-8") as f:
                    f.write("print('old')\n")
                request_body = {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 32,
                    "stream": True,
                    "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
                    "messages": [{"role": "user", "content": "Read src/example.py"}],
                }

                with patch.object(server.httpx, "AsyncClient", FakeAsyncClient):
                    client = TestClient(server.app)
                    with client.stream("POST", "/v1/messages", json=request_body) as first:
                        first_body = b"".join(first.iter_bytes())
                    with client.stream("POST", "/v1/messages", json=request_body) as second:
                        second_body = b"".join(second.iter_bytes())
                    with open("src/example.py", "w", encoding="utf-8") as f:
                        f.write("print('newer content')\n")
                    with client.stream("POST", "/v1/messages", json=request_body) as third:
                        third_body = b"".join(third.iter_bytes())

                self.assertEqual(first.headers["x-agentflow-cache"], "skip-streaming")
                self.assertEqual(second.headers["x-agentflow-cache"], "skip-streaming")
                self.assertEqual(third.headers["x-agentflow-cache"], "skip-streaming")
                self.assertEqual(first_body, b"".join(STREAM_FRAMES))
                self.assertEqual(second_body, first_body)
                self.assertEqual(third_body, first_body)
                self.assertEqual(FakeAsyncClient.calls, 3)

                rows = server.store.conn.execute(
                    "select cache_hit, cache_json from calls order by created_at"
                ).fetchall()
                self.assertEqual([row["cache_hit"] for row in rows], [0, 0, 0])
                self.assertEqual(json.loads(rows[0]["cache_json"])["reason"], "streaming-tools-disabled")
                self.assertEqual(json.loads(rows[1]["cache_json"])["reason"], "streaming-tools-disabled")
                self.assertEqual(json.loads(rows[2]["cache_json"])["reason"], "streaming-tools-disabled")
        finally:
            os.chdir(old_cwd)
            cache_module.CACHE_TOOL_CALLS = old_cache_tool_calls

    def test_non_streaming_empty_error_body_logs_status_fallback(self):
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Trigger an error."}],
        }

        with patch.object(server.httpx, "AsyncClient", FakeEmptyErrorClient):
            client = TestClient(server.app)
            response = client.post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 400)
        [row] = server.store.conn.execute(
            "select status_code, error from calls"
        ).fetchall()
        self.assertEqual(row["status_code"], 400)
        self.assertEqual(row["error"], "upstream_error: status=400")

    def test_anthropic_streaming_error_logs_upstream_body(self):
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "Trigger a stream error."}],
        }

        with patch.object(server.httpx, "AsyncClient", FakeStreamErrorClient):
            client = TestClient(server.app)
            with client.stream("POST", "/v1/messages", json=request_body) as response:
                body = b"".join(response.iter_bytes())

        self.assertIn(b"stream body failed", body)
        [row] = server.store.conn.execute(
            "select status_code, error from calls"
        ).fetchall()
        self.assertEqual(row["status_code"], 400)
        self.assertIn("stream body failed", row["error"])

    def test_openai_streaming_error_logs_upstream_body(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")
        request_body = {
            "model": "gpt-5-codex",
            "stream": True,
            "input": "Trigger a stream error.",
        }

        with patch.object(server.httpx, "AsyncClient", FakeStreamErrorClient):
            client = TestClient(server.app)
            with client.stream("POST", "/v1/responses", json=request_body) as response:
                body = b"".join(response.iter_bytes())

        self.assertIn(b"stream body failed", body)
        [row] = server.store.conn.execute(
            "select provider, status_code, error from calls"
        ).fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 400)
        self.assertIn("stream body failed", row["error"])


if __name__ == "__main__":
    unittest.main()
