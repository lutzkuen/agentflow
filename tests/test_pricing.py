import unittest

from agentflow_proxy.pricing import estimate_cost


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


if __name__ == "__main__":
    unittest.main()
