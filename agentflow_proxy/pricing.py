from __future__ import annotations

from typing import Optional

# Approximate public list prices in USD per million tokens. Update in config/env as needed.
MODEL_PRICES = {
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

MODEL_ALIASES = {
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20240620",
    "claude-opus-4.5": "claude-opus-4-5",
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    prices = None
    ml = model.lower()
    for name, val in MODEL_PRICES.items():
        if name in ml:
            prices = val
            break
    if not prices:
        return None
    in_per_m, out_per_m = prices
    return (input_tokens / 1_000_000) * in_per_m + (output_tokens / 1_000_000) * out_per_m
