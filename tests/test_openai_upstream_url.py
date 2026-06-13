import unittest

from agentflow_proxy.upstream_url import (
    UpstreamUrlError,
    join_openai_upstream_url,
    normalize_openai_upstream_base_url,
    openai_websocket_url,
    redact_url,
)


class OpenAIUpstreamUrlTests(unittest.TestCase):
    def test_normalize_preserves_path_and_query_without_trailing_slash(self):
        self.assertEqual(
            normalize_openai_upstream_base_url("https://example.test/root/v1/?api-version=2024-10-21"),
            "https://example.test/root/v1?api-version=2024-10-21",
        )

    def test_normalize_rejects_invalid_or_ambiguous_urls(self):
        for raw in ("", "api.openai.com", "ftp://api.openai.com", "https://api.openai.com/#token"):
            with self.subTest(raw=raw):
                with self.assertRaises(UpstreamUrlError):
                    normalize_openai_upstream_base_url(raw)

    def test_join_default_openai_routes(self):
        self.assertEqual(
            join_openai_upstream_url("https://api.openai.com", "/v1/responses"),
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(
            join_openai_upstream_url("https://api.openai.com", "/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            join_openai_upstream_url("https://api.openai.com", "/v1/files"),
            "https://api.openai.com/v1/files",
        )
        self.assertEqual(
            join_openai_upstream_url("https://api.openai.com", "/v1/uploads"),
            "https://api.openai.com/v1/uploads",
        )

    def test_join_avoids_duplicate_v1_segments(self):
        self.assertEqual(
            join_openai_upstream_url("https://proxy.example/v1/", "/v1/responses"),
            "https://proxy.example/v1/responses",
        )

    def test_join_preserves_custom_path_components_and_query(self):
        self.assertEqual(
            join_openai_upstream_url("https://proxy.example/gateway/team?api-version=2024-10-21", "/v1/files"),
            "https://proxy.example/gateway/team/files?api-version=2024-10-21",
        )

    def test_join_azure_deployment_shape_drops_proxy_v1_prefix(self):
        self.assertEqual(
            join_openai_upstream_url(
                "https://resource.openai.azure.com/openai/deployments/my-deployment?api-version=2024-10-21",
                "/v1/responses",
            ),
            "https://resource.openai.azure.com/openai/deployments/my-deployment/responses?api-version=2024-10-21",
        )

    def test_join_passthrough_merges_request_query_with_upstream_query(self):
        self.assertEqual(
            join_openai_upstream_url(
                "https://resource.openai.azure.com/openai/deployments/my-deployment?api-version=2024-10-21",
                "/v1/files/file-1",
                request_query={"purpose": "assistants"},
            ),
            "https://resource.openai.azure.com/openai/deployments/my-deployment/files/file-1?api-version=2024-10-21&purpose=assistants",
        )

    def test_websocket_url_converts_http_schemes_and_preserves_query(self):
        self.assertEqual(
            openai_websocket_url("https://api.openai.com/v1?api-version=2024-10-21", "/v1/responses"),
            "wss://api.openai.com/v1/responses?api-version=2024-10-21",
        )
        self.assertEqual(
            openai_websocket_url("http://localhost:8080/openai/deployments/d?api-version=2024-10-21", "/v1/responses"),
            "ws://localhost:8080/openai/deployments/d/responses?api-version=2024-10-21",
        )

    def test_redact_url_removes_userinfo_and_sensitive_query_values(self):
        self.assertEqual(
            redact_url("https://user:pass@example.test/root?api-key=secret&api-version=2024-10-21&sig=abc"),
            "https://example.test/root?api-key=%5Bredacted%5D&api-version=2024-10-21&sig=%5Bredacted%5D",
        )


if __name__ == "__main__":
    unittest.main()
