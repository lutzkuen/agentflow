import importlib.util
import os
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import patch

from tokenclaw.errors import (
    public_proxy_error_body,
    public_proxy_error_message,
    tls_trust_error_hint,
    upstream_error_text,
)


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    import httpx
    from fastapi.testclient import TestClient

    from tokenclaw import anthropic_proxy, openai_proxy, server
    from tokenclaw.store import Store


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


class TlsTrustHintTest(unittest.TestCase):
    """The corporate-CA hint must fire for outbound TLS trust failures (surfacing the
    env knobs at runtime) yet never for ordinary internal errors, and never leak the
    exception text — the message it emits is a fixed, secret-free constant."""

    SECRET = "secret failure from /tmp/tokenclaw-token"

    def _cert_verify_error(self) -> ssl.SSLCertVerificationError:
        return ssl.SSLCertVerificationError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get "
            "local issuer certificate (_ssl.c:1006)"
        )

    def test_direct_cert_verification_error_yields_actionable_hint(self):
        hint = tls_trust_error_hint(self._cert_verify_error())
        self.assertIsNotNone(hint)
        self.assertIn("TOKENCLAW_CA_BUNDLE", hint)
        self.assertIn("TOKENCLAW_TLS_TRUST_STORE=system", hint)
        self.assertIn("TOKENCLAW_TLS_VERIFY=0", hint)

    def test_wrapped_ssl_error_in_cause_chain_is_detected(self):
        outer = RuntimeError("Connection failed")
        outer.__cause__ = ssl.SSLError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1006)"
        )
        self.assertIsNotNone(tls_trust_error_hint(outer))

    def test_marker_string_deep_in_context_chain_is_detected(self):
        inner = OSError("unable to get local issuer certificate")
        middle = RuntimeError("tls handshake")
        middle.__context__ = inner
        outer = RuntimeError("upstream connect error")
        outer.__context__ = middle
        self.assertIsNotNone(tls_trust_error_hint(outer))

    def test_cyclic_exception_chain_terminates(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__context__ = b
        b.__context__ = a
        # Neither is a TLS failure; the id()-dedup must stop the walk rather than loop.
        self.assertIsNone(tls_trust_error_hint(a))

    def test_non_tls_error_gets_no_hint_and_no_leak(self):
        exc = RuntimeError(self.SECRET)
        self.assertIsNone(tls_trust_error_hint(exc))
        message = public_proxy_error_message(exc)
        self.assertEqual(message, "Internal proxy error")
        self.assertNotIn("tokenclaw-token", message)

    def test_none_exception_gets_no_hint(self):
        self.assertIsNone(tls_trust_error_hint(None))
        self.assertEqual(public_proxy_error_message(None), "Internal proxy error")

    def test_hint_bypasses_debug_gate_and_stays_secret_free(self):
        exc = self._cert_verify_error()
        with patch.dict(os.environ, {"TOKENCLAW_DEBUG_PROXY_ERRORS": "0"}, clear=False):
            message = public_proxy_error_message(exc)
        self.assertIn("TOKENCLAW_CA_BUNDLE", message)
        # The fixed constant must not interpolate the exception's own text.
        self.assertNotIn("_ssl.c", message)

    def test_public_error_body_carries_hint_and_keeps_anthropic_shape(self):
        body = public_proxy_error_body(provider="anthropic", exc=self._cert_verify_error())
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "tokenclaw_error")
        self.assertIn("TOKENCLAW_CA_BUNDLE", body["error"]["message"])


class FakeRaisingStreamClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        raise RuntimeError("secret stream failure from /tmp/tokenclaw-token")


class FakeSuccessfulAnthropicClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
        )


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
            anthropic_proxy,
            "crunch_body",
            side_effect=RuntimeError("secret anthropic failure from /tmp/tokenclaw-token"),
        ), self.assertLogs(level="ERROR") as logs:
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "tokenclaw_error")
        self.assertEqual(body["error"]["message"], "Internal proxy error")
        self.assertNotIn("tokenclaw-token", response.text)
        self.assertIn("tokenclaw-token", "\n".join(logs.output))
        [row] = server.store.conn.execute("select status_code, error from calls").fetchall()
        self.assertEqual(row["status_code"], 500)
        self.assertIn("tokenclaw-token", row["error"])

    def test_anthropic_tls_trust_failure_returns_actionable_public_error(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Trigger a corporate-cert failure."}],
        }

        # A real corporate-MITM failure surfaces as httpx.ConnectError wrapping an
        # ssl.SSLCertVerificationError. Inject that shape at the pre-forward crunch step
        # (same fast injection point as the generic-error test) to assert the
        # client-facing body carries the actionable knob-naming hint end to end.
        # httpx copies the underlying SSL detail into the ConnectError message, so the
        # outer error an operator sees logged carries the full "CERTIFICATE_VERIFY_FAILED"
        # string; the cause is set too so detection works either way.
        ssl_detail = (
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get "
            "local issuer certificate (_ssl.c:1006)"
        )
        connect_error = httpx.ConnectError(ssl_detail)
        connect_error.__cause__ = ssl.SSLCertVerificationError(ssl_detail)

        with patch.object(anthropic_proxy, "crunch_body", side_effect=connect_error), self.assertLogs(level="ERROR"):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "tokenclaw_error")
        self.assertIn("TOKENCLAW_CA_BUNDLE", body["error"]["message"])
        self.assertNotEqual(body["error"]["message"], "Internal proxy error")
        # The server-side log still records the full repr for the operator.
        [row] = server.store.conn.execute("select status_code, error from calls").fetchall()
        self.assertEqual(row["status_code"], 500)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", row["error"])

    def test_anthropic_request_does_not_500_when_home_directory_unavailable(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        request_body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say ok."}],
        }

        config_dir = str(Path(self.tmp.name).parent / "tokenclaw-config")
        with patch.dict(
            os.environ,
            {"HOME": "", "TOKENCLAW_CONFIG_DIR": config_dir, "TOKENCLAW_DB": self.tmp.name},
            clear=False,
        ), patch.object(Path, "home", side_effect=RuntimeError("Could not determine home directory.")), patch.object(
            anthropic_proxy.httpx,
            "AsyncClient",
            FakeSuccessfulAnthropicClient,
        ):
            response = TestClient(server.app).post("/v1/messages", json=request_body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"][0]["text"], "ok")
        [row] = server.store.conn.execute("select status_code, error from calls").fetchall()
        self.assertEqual(row["status_code"], 200)
        self.assertIsNone(row["error"])

    def test_anthropic_malformed_json_returns_400_without_internal_log(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")
        payload = '{"model":"claude-sonnet-4-6","secret":"do-not-echo",'

        with patch.object(server.logging, "exception") as log_exception:
            response = TestClient(server.app).post(
                "/v1/messages",
                content=payload,
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "Malformed JSON request body.")
        self.assertLess(len(response.text), 200)
        self.assertNotIn("do-not-echo", response.text)
        log_exception.assert_not_called()
        [row] = server.store.conn.execute("select status_code, error, request_json from calls").fetchall()
        self.assertEqual(row["status_code"], 400)
        self.assertEqual(row["error"], "Malformed JSON request body.")
        self.assertIsNone(row["request_json"])

    def test_anthropic_non_object_json_returns_400(self):
        server.configure_provider("anthropic", anthropic_upstream="https://anthropic.test")

        with patch.object(server.logging, "exception") as log_exception:
            response = TestClient(server.app).post(
                "/v1/messages",
                content='["do-not-echo"]',
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "JSON request body must be an object.")
        self.assertNotIn("do-not-echo", response.text)
        log_exception.assert_not_called()
        [row] = server.store.conn.execute("select status_code, error, request_json from calls").fetchall()
        self.assertEqual(row["status_code"], 400)
        self.assertEqual(row["error"], "JSON request body must be an object.")
        self.assertIsNone(row["request_json"])

    def test_openai_internal_exception_returns_generic_public_error(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")
        request_body = {"model": "gpt-5-codex", "input": "Trigger internal error."}

        with patch.object(
            openai_proxy,
            "crunch_body",
            side_effect=RuntimeError("secret openai failure from /tmp/tokenclaw-token"),
        ), self.assertLogs(level="ERROR"):
            response = TestClient(server.app).post("/v1/responses", json=request_body)

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["error"]["type"], "tokenclaw_error")
        self.assertEqual(body["error"]["message"], "Internal proxy error")
        self.assertNotIn("tokenclaw-token", response.text)
        [row] = server.store.conn.execute("select provider, status_code, error from calls").fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 500)
        self.assertIn("tokenclaw-token", row["error"])

    def test_openai_malformed_json_returns_400_without_passthrough(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")
        payload = '{"model":"gpt-5-codex","secret":"do-not-echo",'

        with patch.object(server.logging, "exception") as log_exception, patch.object(server.httpx, "AsyncClient") as client:
            response = TestClient(server.app).post(
                "/v1/responses",
                content=payload,
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "Malformed JSON request body.")
        self.assertLess(len(response.text), 200)
        self.assertNotIn("do-not-echo", response.text)
        log_exception.assert_not_called()
        client.assert_not_called()
        [row] = server.store.conn.execute("select provider, status_code, error, request_json from calls").fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 400)
        self.assertEqual(row["error"], "Malformed JSON request body.")
        self.assertIsNone(row["request_json"])

    def test_openai_non_object_json_returns_400(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")

        with patch.object(server.logging, "exception") as log_exception, patch.object(server.httpx, "AsyncClient") as client:
            response = TestClient(server.app).post(
                "/v1/chat/completions",
                content='["do-not-echo"]',
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "JSON request body must be an object.")
        self.assertNotIn("do-not-echo", response.text)
        log_exception.assert_not_called()
        client.assert_not_called()
        [row] = server.store.conn.execute("select provider, status_code, error, request_json from calls").fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 400)
        self.assertEqual(row["error"], "JSON request body must be an object.")
        self.assertIsNone(row["request_json"])

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
        self.assertIn("tokenclaw_error", body)
        self.assertNotIn("tokenclaw-token", body)
        [row] = server.store.conn.execute("select status_code, error from calls").fetchall()
        self.assertEqual(row["status_code"], 500)
        self.assertIn("tokenclaw-token", row["error"])

    def test_openai_streaming_internal_exception_returns_generic_event(self):
        server.configure_provider("openai", openai_upstream="https://openai.test")
        request_body = {"model": "gpt-5-codex", "stream": True, "input": "Trigger stream error."}

        with patch.object(server.httpx, "AsyncClient", FakeRaisingStreamClient), self.assertLogs(level="ERROR"):
            with TestClient(server.app).stream("POST", "/v1/responses", json=request_body) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Internal proxy error", body)
        self.assertIn("tokenclaw_error", body)
        self.assertNotIn("tokenclaw-token", body)
        [row] = server.store.conn.execute("select provider, status_code, error from calls").fetchall()
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["status_code"], 500)
        self.assertIn("tokenclaw-token", row["error"])


if __name__ == "__main__":
    unittest.main()
