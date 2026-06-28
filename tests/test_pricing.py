import unittest
from unittest.mock import patch

from tokenclaw.pricing import (
    blended_input_price_per_million,
    codex_app_pricing_basis,
    estimate_blended_input_savings,
    estimate_cost,
    pricing_basis,
    provider_prompt_cache_accounting,
)


class PricingTest(unittest.TestCase):
    def test_anthropic_prompt_cache_accounting_splits_read_discount_and_write_premium(self):
        accounting = provider_prompt_cache_accounting(
            "claude-sonnet-4-6",
            provider="anthropic",
            cache_creation_tokens=1_000,
            cache_read_tokens=2_000,
        )

        self.assertEqual(accounting["pricing_source"], "embedded-tokenclaw-defaults")
        self.assertEqual(accounting["pricing_version"], "2026-06-08")
        self.assertEqual(accounting["input_usd_per_million"], 3.0)
        self.assertAlmostEqual(accounting["cached_input_usd_per_million"], 0.3, places=8)
        self.assertEqual(accounting["cache_creation_input_usd_per_million"], 3.75)
        self.assertAlmostEqual(accounting["full_price_equivalent_read_cost_usd"], 0.006, places=8)
        self.assertAlmostEqual(accounting["actual_cached_read_cost_usd"], 0.0006, places=8)
        self.assertAlmostEqual(accounting["read_discount_usd"], 0.0054, places=8)
        self.assertAlmostEqual(accounting["creation_cost_usd"], 0.00375, places=8)
        self.assertAlmostEqual(accounting["creation_premium_usd"], 0.00075, places=8)
        self.assertAlmostEqual(accounting["net_provider_cache_discount_usd"], 0.00465, places=8)

    def test_openai_prompt_cache_accounting_uses_cached_input_tuple(self):
        accounting = provider_prompt_cache_accounting(
            "gpt-5-codex",
            provider="openai",
            cache_read_tokens=2_000,
        )

        self.assertEqual(accounting["pricing_source"], "https://developers.openai.com/api/docs/pricing")
        self.assertEqual(accounting["input_usd_per_million"], 1.25)
        self.assertEqual(accounting["cached_input_usd_per_million"], 0.125)
        self.assertAlmostEqual(accounting["full_price_equivalent_read_cost_usd"], 0.0025, places=8)
        self.assertAlmostEqual(accounting["actual_cached_read_cost_usd"], 0.00025, places=8)
        self.assertAlmostEqual(accounting["read_discount_usd"], 0.00225, places=8)

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

    def test_current_opus_generations_use_current_rate_not_legacy_opus4(self):
        # Regression: claude-opus-4-8 (the live production model) substring-matched
        # the retired "claude-opus-4" entry and was billed at $15/$75 instead of its
        # real $5/$25 — a 3x overstatement that corrupted the savings metric and
        # routing-savings math. Current Opus generations must price at $5/$25.
        for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6"):
            cost = estimate_cost(model, 1_000_000, 1_000_000)
            self.assertEqual(cost, 30.0, f"{model} should be $5/$25 -> $30 for 1M/1M")
        # Legacy Opus 4 / 4.1 stay at the original $15/$75.
        self.assertEqual(estimate_cost("claude-opus-4", 1_000_000, 1_000_000), 90.0)
        self.assertEqual(estimate_cost("claude-opus-4-1", 1_000_000, 1_000_000), 90.0)
        # Sonnet 4.6 unchanged at $3/$15.
        self.assertEqual(estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000), 18.0)

    def test_openai_unknown_model_is_unpriced(self):
        self.assertIsNone(estimate_cost("not-a-model", 1000, 1000, provider="openai"))

    def test_openai_unknown_model_can_be_configured_with_env_price(self):
        override = '{"future-codex":{"input":2.0,"cached_input":0.2,"output":16.0}}'
        with patch.dict("os.environ", {"TOKENCLAW_OPENAI_MODEL_PRICES_JSON": override}):
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
        self.assertEqual(basis["source"], "env:TOKENCLAW_OPENAI_MODEL_PRICES_JSON")

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
        with patch.dict("os.environ", {"TOKENCLAW_CODEX_APP_MODEL": "future-codex"}, clear=True):
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
