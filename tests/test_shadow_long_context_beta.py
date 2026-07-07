import unittest

from tokenclaw.anthropic_proxy import (
    _model_supports_1m_context,
    _shadow_headers_for_model,
)


class ModelSupports1MContextTests(unittest.TestCase):
    def test_1m_capable_models(self):
        for model in ("claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-5"):
            self.assertTrue(_model_supports_1m_context(model), model)

    def test_non_1m_models(self):
        for model in ("claude-haiku-4-5-20251001", "claude-3-5-haiku", "claude-sonnet-4"):
            self.assertFalse(_model_supports_1m_context(model), model)


class ShadowHeadersForModelTests(unittest.TestCase):
    def test_strips_context_1m_for_non_1m_target_keeps_others(self):
        headers = {"anthropic-beta": "context-1m-2025-08-07,prompt-caching-2024-07-31", "x-api-key": "k"}
        out = _shadow_headers_for_model(headers, "claude-haiku-4-5-20251001")
        self.assertIsNot(out, headers)  # copied, original untouched
        self.assertEqual(out["anthropic-beta"], "prompt-caching-2024-07-31")
        self.assertEqual(out["x-api-key"], "k")
        # original unchanged
        self.assertIn("context-1m-2025-08-07", headers["anthropic-beta"])

    def test_removes_anthropic_beta_when_only_context_1m(self):
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        out = _shadow_headers_for_model(headers, "claude-haiku-4-5-20251001")
        self.assertNotIn("anthropic-beta", out)

    def test_keeps_beta_for_1m_capable_target(self):
        headers = {"anthropic-beta": "context-1m-2025-08-07,prompt-caching-2024-07-31"}
        out = _shadow_headers_for_model(headers, "claude-sonnet-5")
        self.assertIs(out, headers)  # unchanged identity — opus->sonnet-5 stays working

    def test_no_beta_header_is_passthrough(self):
        headers = {"x-api-key": "k"}
        self.assertIs(_shadow_headers_for_model(headers, "claude-haiku-4-5-20251001"), headers)

    def test_beta_without_context_1m_is_passthrough(self):
        headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        self.assertIs(_shadow_headers_for_model(headers, "claude-haiku-4-5-20251001"), headers)


if __name__ == "__main__":
    unittest.main()
