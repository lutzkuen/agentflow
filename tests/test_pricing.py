import unittest
from unittest.mock import patch

from agentflow_proxy.pricing import (
    blended_input_price_per_million,
    codex_app_pricing_basis,
    estimate_blended_input_savings,
    estimate_cost,
    pricing_basis,
)


class PricingTest(unittest.TestCase):
    def test_openai_cached_input_uses_provider_price(self):
        cost = estimate_cost(
            "gpt-5-codex",
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read=900_000,
            provider="openai",
        )

        self.assertAlmostEqual(cost, 1.2375, places=6)

    def test_gpt_5_3_codex_standard_pricing(self):
        cost = estimate_cost(
            "gpt-5.3-codex",
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read=900_000,
            provider="openai",
        )

        self.assertAlmostEqual(cost, 1.7325, places=6)

    def test_gpt_5_3_codex_alias_pricing(self):
        basis = pricing_basis("gpt-5.3-codex-latest", provider="openai")

        self.assertTrue(basis["cost_known"])
        self.assertEqual(basis["matched_model"], "gpt-5.3-codex")
        self.assertEqual(basis["input_usd_per_million"], 1.75)

    def test_gpt_5_3_codex_priority_pricing_basis(self):
        basis = pricing_basis("gpt-5.3-codex", provider="openai", processing_mode="priority")

        self.assertTrue(basis["cost_known"])
        self.assertEqual(basis["processing_mode"], "priority")
        self.assertEqual(basis["input_usd_per_million"], 3.50)
        self.assertEqual(basis["cached_input_usd_per_million"], 0.350)
        self.assertEqual(basis["output_usd_per_million"], 28.0)

    def test_openai_unknown_model_is_unpriced(self):
        self.assertIsNone(estimate_cost("not-a-model", 1000, 1000, provider="openai"))

    def test_openai_unknown_model_can_be_configured_with_env_price(self):
        override = '{"future-codex":{"input":2.0,"cached_input":0.2,"output":16.0}}'
        with patch.dict("os.environ", {"AGENTFLOW_OPENAI_MODEL_PRICES_JSON": override}):
            cost = estimate_cost(
                "future-codex",
                input_tokens=1_000_000,
                output_tokens=100_000,
                cache_read=500_000,
                provider="openai",
            )
            basis = pricing_basis("future-codex", provider="openai")

        self.assertAlmostEqual(cost, 2.7, places=6)
        self.assertTrue(basis["cost_known"])
        self.assertEqual(basis["source"], "env:AGENTFLOW_OPENAI_MODEL_PRICES_JSON")

    def test_default_codex_app_pricing_basis_uses_current_model(self):
        with patch.dict("os.environ", {}, clear=True):
            basis = codex_app_pricing_basis()

        self.assertEqual(basis["model"], "gpt-5.3-codex")
        self.assertEqual(basis["provider"], "openai")
        self.assertEqual(basis["processing_mode"], "standard")
        self.assertTrue(basis["cost_known"])
        self.assertEqual(basis["input_usd_per_million"], 1.75)
        self.assertEqual(basis["cached_input_usd_per_million"], 0.175)
        self.assertEqual(basis["output_usd_per_million"], 14.0)

    def test_codex_app_model_env_override_is_exposed_in_basis(self):
        with patch.dict("os.environ", {"AGENTFLOW_CODEX_APP_MODEL": "future-codex"}, clear=True):
            basis = codex_app_pricing_basis()

        self.assertEqual(basis["model"], "future-codex")
        self.assertFalse(basis["cost_known"])

    def test_blended_anthropic_input_price_uses_cache_read_discount(self):
        rate = blended_input_price_per_million(
            "claude-sonnet-4-6",
            input_tokens=1_000,
            cache_read_tokens=9_000,
        )

        self.assertAlmostEqual(rate, 0.57, places=6)

    def test_blended_openai_input_price_uses_cached_input_price(self):
        rate = blended_input_price_per_million(
            "gpt-5-codex",
            input_tokens=1_000,
            cache_read_tokens=9_000,
            provider="openai",
        )

        self.assertAlmostEqual(rate, 0.2375, places=6)

    def test_blended_input_savings_uses_observed_cache_mix(self):
        savings = estimate_blended_input_savings(
            "claude-sonnet-4-6",
            tokens_saved=1_000,
            input_tokens=1_000,
            cache_read_tokens=9_000,
        )

        self.assertAlmostEqual(savings, 0.00057, places=6)


if __name__ == "__main__":
    unittest.main()
