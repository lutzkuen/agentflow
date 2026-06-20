import asyncio
import gzip
import json
import unittest

from tokenclaw.headers import (
    ClientJsonRequestError,
    build_anthropic_forward_headers,
    build_anthropic_summary_headers,
    build_openai_forward_headers,
    build_openai_websocket_headers,
    read_json_object_body,
)


class DummyRequest:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self._body = body
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


class HeaderForwardingModuleTests(unittest.TestCase):
    def test_anthropic_forwarding_uses_allowlist_and_json_content_type(self):
        headers = build_anthropic_forward_headers({
            "Authorization": "Bearer client-key",
            "Anthropic-Beta": "prompt-caching-2024-07-31",
            "X-Internal-User": "alice@example.test",
            "Content-Type": "text/plain",
        })

        self.assertEqual(headers["Authorization"], "Bearer client-key")
        self.assertEqual(headers["Anthropic-Beta"], "prompt-caching-2024-07-31")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn("X-Internal-User", headers)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")

    def test_anthropic_summary_headers_do_not_inherit_client_beta_headers(self):
        headers = build_anthropic_summary_headers({
            "Authorization": "Bearer client-key",
            "Anthropic-Beta": "prompt-caching-2024-07-31",
            "User-Agent": "claude-code",
            "X-Internal-User": "alice@example.test",
            "Content-Type": "text/plain",
        })

        self.assertEqual(headers["Authorization"], "Bearer client-key")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("Anthropic-Beta", headers)
        self.assertNotIn("User-Agent", headers)
        self.assertNotIn("X-Internal-User", headers)

    def test_openai_proxy_auth_replaces_client_authorization(self):
        headers = build_openai_forward_headers(
            {
                "Authorization": "Bearer client-key",
                "OpenAI-Project": "proj_123",
                "X-Trace-Secret": "local-secret",
            },
            auth_mode="proxy",
            api_key="proxy-key",
            force_json=False,
        )

        self.assertEqual(headers["authorization"], "Bearer proxy-key")
        self.assertEqual(headers["OpenAI-Project"], "proj_123")
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("X-Trace-Secret", headers)

    def test_openai_force_json_strips_compression_header(self):
        headers = build_openai_forward_headers(
            {
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
            auth_mode="client",
            force_json=True,
        )

        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn("Content-Encoding", headers)
        self.assertNotIn("content-encoding", headers)

    def test_websocket_forwarding_blocks_handshake_headers(self):
        headers = build_openai_websocket_headers(
            {
                "Authorization": "Bearer client-key",
                "OpenAI-Beta": "realtime=v1",
                "Sec-WebSocket-Key": "handshake",
                "Connection": "upgrade",
            },
            auth_mode="client",
        )

        self.assertEqual(headers["Authorization"], "Bearer client-key")
        self.assertEqual(headers["OpenAI-Beta"], "realtime=v1")
        self.assertNotIn("Sec-WebSocket-Key", headers)
        self.assertNotIn("Connection", headers)

    def test_json_body_parser_handles_compressed_objects(self):
        payload = gzip.compress(json.dumps({"model": "gpt-5-codex"}).encode())
        parsed = asyncio.run(read_json_object_body(
            DummyRequest(payload, {"content-encoding": "gzip"}),
            allow_compressed=True,
        ))

        self.assertEqual(parsed, {"model": "gpt-5-codex"})

    def test_json_body_parser_rejects_non_object_json(self):
        with self.assertRaises(ClientJsonRequestError) as ctx:
            asyncio.run(read_json_object_body(DummyRequest(b'["not-object"]')))

        self.assertEqual(ctx.exception.message, "JSON request body must be an object.")


if __name__ == "__main__":
    unittest.main()
