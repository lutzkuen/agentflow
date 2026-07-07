import os
import ssl
import unittest
from unittest import mock

import certifi

from tokenclaw import http_client


_TLS_ENV = (
    "TOKENCLAW_TLS_VERIFY",
    "TOKENCLAW_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "TOKENCLAW_TLS_TRUST_STORE",
)


class TlsVerifyTests(unittest.TestCase):
    def setUp(self):
        # Isolate from any ambient TLS env and the module-level cache.
        self._clean = {k: None for k in _TLS_ENV}
        self._patch = mock.patch.dict(os.environ, {}, clear=False)
        self._patch.start()
        for k in _TLS_ENV:
            os.environ.pop(k, None)
        http_client.tls_verify.cache_clear()

    def tearDown(self):
        self._patch.stop()
        http_client.tls_verify.cache_clear()

    def test_default_is_true(self):
        self.assertIs(http_client.tls_verify(), True)

    def test_disable_verification(self):
        os.environ["TOKENCLAW_TLS_VERIFY"] = "0"
        http_client.tls_verify.cache_clear()
        self.assertIs(http_client.tls_verify(), False)

    def test_ca_bundle_returns_augmented_context(self):
        os.environ["TOKENCLAW_CA_BUNDLE"] = certifi.where()
        http_client.tls_verify.cache_clear()
        verify = http_client.tls_verify()
        self.assertIsInstance(verify, ssl.SSLContext)

    def test_standard_env_ca_paths_are_honored(self):
        os.environ["SSL_CERT_FILE"] = certifi.where()
        http_client.tls_verify.cache_clear()
        self.assertIsInstance(http_client.tls_verify(), ssl.SSLContext)

    def test_nonexistent_ca_path_falls_through_to_default(self):
        os.environ["TOKENCLAW_CA_BUNDLE"] = "/no/such/corporate-ca.pem"
        http_client.tls_verify.cache_clear()
        self.assertIs(http_client.tls_verify(), True)

    def test_trust_store_system_without_truststore_falls_back(self):
        os.environ["TOKENCLAW_TLS_TRUST_STORE"] = "system"
        http_client.tls_verify.cache_clear()
        # truststore is not a hard dependency; absent -> safe fallback to defaults.
        self.assertIn(http_client.tls_verify(), (True,))

    def test_async_client_builds_and_defaults_verify(self):
        client = http_client.async_client(timeout=5.0)
        try:
            self.assertIsNotNone(client)
        finally:
            import asyncio

            asyncio.new_event_loop().run_until_complete(client.aclose())


if __name__ == "__main__":
    unittest.main()
