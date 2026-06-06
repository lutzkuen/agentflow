import unittest

from agentflow_proxy.pricing import blended_input_price_per_million, estimate_blended_input_savings, estimate_cost


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

    def test_openai_unknown_model_is_unpriced(self):
        self.assertIsNone(estimate_cost("not-a-model", 1000, 1000, provider="openai"))

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
