from __future__ import annotations

from typing import Optional

# Approximate public list prices in USD per million tokens. Update in config/env as needed.
ANTHROPIC_MODEL_PRICES = {
    "claude-opus-4.5": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4.5": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-haiku-4.5": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-3-5-haiku": (0.8, 4.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}

OPENAI_MODEL_PRICES = {
    "gpt-5.5": (5.0, 30.0, 0.50),
    "gpt-5.4-mini": (0.75, 4.50, 0.075),
    "gpt-5.4": (2.50, 15.0, 0.25),
    "gpt-5.2-codex": (1.75, 14.0, 0.175),
    "gpt-5.2": (1.75, 14.0, 0.175),
    "gpt-5.1-codex-max": (1.25, 10.0, 0.125),
    "gpt-5.1-codex": (1.25, 10.0, 0.125),
    "gpt-5-codex": (1.25, 10.0, 0.125),
    "gpt-5-mini": (0.25, 2.0, 0.025),
    "gpt-5-nano": (0.05, 0.40, 0.005),
    "gpt-5": (1.25, 10.0, 0.125),
    "gpt-4.1-mini": (0.40, 1.60, 0.10),
    "gpt-4.1-nano": (0.10, 0.40, 0.025),
    "gpt-4.1": (2.0, 8.0, 0.50),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4o": (2.50, 10.0, 1.25),
}

MODEL_PRICES = ANTHROPIC_MODEL_PRICES

MODEL_ALIASES = {
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20240620",
    "claude-opus-4.5": "claude-opus-4-5",
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
    provider: str = "anthropic",
) -> Optional[float]:
    provider = provider.lower()
    if provider == "openai":
        openai_prices = None
        ml = model.lower()
        for name, val in OPENAI_MODEL_PRICES.items():
            if name in ml:
                openai_prices = val
                break
        if not openai_prices:
            return None
        in_per_m, out_per_m, cached_in_per_m = openai_prices
        cached_tokens = min(max(cache_read, 0), max(input_tokens, 0))
        uncached_tokens = max(input_tokens - cached_tokens, 0)
        return (
            (uncached_tokens / 1_000_000) * in_per_m
            + (cached_tokens / 1_000_000) * cached_in_per_m
            + (output_tokens / 1_000_000) * out_per_m
        )

    prices = None
    ml = model.lower()
    for name, val in ANTHROPIC_MODEL_PRICES.items():
        if name in ml:
            prices = val
            break
    if not prices:
        return None
    in_per_m, out_per_m = prices
    return (
        (input_tokens / 1_000_000) * in_per_m
        + (output_tokens / 1_000_000) * out_per_m
        + (cache_creation / 1_000_000) * in_per_m * 1.25
        + (cache_read / 1_000_000) * in_per_m * 0.10
    )


def input_price_per_million(model: str, provider: str = "anthropic", cache_read: bool = False) -> Optional[float]:
    provider = provider.lower()
    ml = model.lower()
    if provider == "openai":
        for name, val in OPENAI_MODEL_PRICES.items():
            if name in ml:
                return val[2] if cache_read else val[0]
        return None

    for name, val in ANTHROPIC_MODEL_PRICES.items():
        if name in ml:
            in_per_m = val[0]
            return in_per_m * 0.10 if cache_read else in_per_m
    return None


def blended_input_price_per_million(
    model: str,
    input_tokens: int,
    cache_read_tokens: int,
    provider: str = "anthropic",
) -> Optional[float]:
    input_price = input_price_per_million(model, provider=provider, cache_read=False)
    cache_read_price = input_price_per_million(model, provider=provider, cache_read=True)
    if input_price is None or cache_read_price is None:
        return None

    input_tokens = max(input_tokens, 0)
    cache_read_tokens = max(cache_read_tokens, 0)
    total = input_tokens + cache_read_tokens
    if total <= 0:
        return input_price
    return ((input_tokens * input_price) + (cache_read_tokens * cache_read_price)) / total


def estimate_blended_input_savings(
    model: str,
    tokens_saved: int,
    input_tokens: int,
    cache_read_tokens: int,
    provider: str = "anthropic",
) -> Optional[float]:
    price = blended_input_price_per_million(
        model,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        provider=provider,
    )
    if price is None:
        return None
    return (max(tokens_saved, 0) / 1_000_000) * price
