from __future__ import annotations

import json
import os
from typing import Any, Optional

# Approximate public list prices in USD per million tokens. Update in config/env as needed.
PRICING_SOURCE_OPENAI = "https://developers.openai.com/api/docs/pricing"
PRICING_VERSION_OPENAI = "2026-06-08"
PRICING_SOURCE_ANTHROPIC = "embedded-agentflow-defaults"
PRICING_VERSION_ANTHROPIC = "2026-06-08"
DEFAULT_CODEX_APP_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_APP_PROCESSING_MODE = "standard"

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
    "gpt-5.3-codex": (1.75, 14.0, 0.175),
    "gpt-5.3": (1.75, 14.0, 0.175),
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

OPENAI_PRIORITY_MODEL_PRICES = {
    "gpt-5.5": (12.50, 75.0, 1.25),
    "gpt-5.4-mini": (1.50, 9.0, 0.150),
    "gpt-5.4": (5.0, 30.0, 0.50),
    "gpt-5.3-codex": (3.50, 28.0, 0.350),
    "gpt-5.3": (3.50, 28.0, 0.350),
    "gpt-5.2-codex": (3.50, 28.0, 0.350),
    "gpt-5.2": (3.50, 28.0, 0.350),
    "gpt-5.1-codex-max": (2.50, 20.0, 0.250),
    "gpt-5.1-codex": (2.50, 20.0, 0.250),
    "gpt-5-codex": (2.50, 20.0, 0.250),
    "gpt-5-mini": (0.45, 3.60, 0.045),
    "gpt-5-nano": (0.10, 0.80, 0.010),
    "gpt-5": (2.50, 20.0, 0.250),
}

OPENAI_MODEL_ALIASES = {
    "gpt-5.3-codex-latest": "gpt-5.3-codex",
    "gpt-5.2-codex-latest": "gpt-5.2-codex",
    "gpt-5-codex-latest": "gpt-5-codex",
}

MODEL_PRICES = ANTHROPIC_MODEL_PRICES

MODEL_ALIASES = {
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20240620",
    "claude-opus-4.5": "claude-opus-4-5",
}


def codex_app_model() -> str:
    return (
        os.getenv("AGENTFLOW_CODEX_APP_MODEL")
        or os.getenv("AGENTFLOW_OPENAI_LARGE_MODEL")
        or DEFAULT_CODEX_APP_MODEL
    )


def codex_app_processing_mode() -> str:
    return (
        os.getenv("AGENTFLOW_CODEX_APP_PROCESSING_MODE")
        or os.getenv("AGENTFLOW_OPENAI_PROCESSING_MODE")
        or DEFAULT_CODEX_APP_PROCESSING_MODE
    ).strip().lower()


def _price_tuple_from_value(value: Any) -> Optional[tuple[float, float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        input_price = value.get("input", value.get("input_usd_per_million"))
        output_price = value.get("output", value.get("output_usd_per_million"))
        cached_price = value.get(
            "cached_input",
            value.get("cached_input_usd_per_million", value.get("cache_read")),
        )
        try:
            return float(input_price), float(output_price), float(cached_price)
        except (TypeError, ValueError):
            return None
    return None


def _openai_price_overrides(processing_mode: str) -> dict[str, tuple[float, float, float]]:
    raw = os.getenv("AGENTFLOW_OPENAI_MODEL_PRICES_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}

    overrides: dict[str, tuple[float, float, float]] = {}
    for model, value in parsed.items():
        prices = _price_tuple_from_value(value)
        if prices is None and isinstance(value, dict):
            mode_value = value.get(processing_mode) or value.get("standard")
            prices = _price_tuple_from_value(mode_value)
        if prices is not None:
            overrides[str(model).lower()] = prices
    return overrides


def _match_prices(
    model: str,
    prices_by_model: dict[str, tuple[float, ...]],
) -> tuple[str | None, tuple[float, ...] | None]:
    ml = model.lower()
    for name, val in prices_by_model.items():
        if name in ml:
            return name, val
    return None, None


def _openai_prices_for_model(
    model: str,
    processing_mode: str = DEFAULT_CODEX_APP_PROCESSING_MODE,
) -> tuple[str | None, tuple[float, float, float] | None, str]:
    processing_mode = (processing_mode or DEFAULT_CODEX_APP_PROCESSING_MODE).strip().lower()
    model = OPENAI_MODEL_ALIASES.get(model.lower(), model)
    overrides = _openai_price_overrides(processing_mode)
    matched, prices = _match_prices(model, overrides)
    if matched and prices:
        return matched, prices, "env:AGENTFLOW_OPENAI_MODEL_PRICES_JSON"

    table = OPENAI_PRIORITY_MODEL_PRICES if processing_mode == "priority" else OPENAI_MODEL_PRICES
    matched, prices = _match_prices(model, table)
    if matched and prices:
        return matched, prices, PRICING_SOURCE_OPENAI
    return None, None, PRICING_SOURCE_OPENAI


def pricing_basis(
    model: str,
    provider: str = "anthropic",
    processing_mode: str | None = None,
) -> dict[str, Any]:
    provider = provider.lower()
    processing_mode = (processing_mode or DEFAULT_CODEX_APP_PROCESSING_MODE).strip().lower()
    if provider == "openai":
        matched_model, prices, source = _openai_prices_for_model(model, processing_mode)
        in_per_m: float | None = None
        out_per_m: float | None = None
        cached_in_per_m: float | None = None
        if prices:
            in_per_m, out_per_m, cached_in_per_m = prices
        return {
            "schema": "agentflow.pricing_basis.v1",
            "provider": "openai",
            "model": model,
            "matched_model": matched_model,
            "processing_mode": processing_mode,
            "input_usd_per_million": in_per_m,
            "cached_input_usd_per_million": cached_in_per_m,
            "output_usd_per_million": out_per_m,
            "currency": "USD",
            "source": source,
            "version": PRICING_VERSION_OPENAI,
            "cost_known": prices is not None,
            "cost_basis": "provider-reported + codex-estimated-from-chars",
        }

    matched_model, prices = _match_prices(model, ANTHROPIC_MODEL_PRICES)
    in_per_m = prices[0] if prices else None
    out_per_m = prices[1] if prices else None
    return {
        "schema": "agentflow.pricing_basis.v1",
        "provider": "anthropic",
        "model": model,
        "matched_model": matched_model,
        "processing_mode": "standard",
        "input_usd_per_million": in_per_m,
        "cached_input_usd_per_million": (in_per_m * 0.10) if in_per_m is not None else None,
        "cache_creation_input_usd_per_million": (in_per_m * 1.25) if in_per_m is not None else None,
        "output_usd_per_million": out_per_m,
        "currency": "USD",
        "source": PRICING_SOURCE_ANTHROPIC,
        "version": PRICING_VERSION_ANTHROPIC,
        "cost_known": prices is not None,
        "cost_basis": "provider-reported",
    }


def _cost_from_per_million(tokens: int, price_per_million: Any) -> float | None:
    try:
        price = float(price_per_million)
    except (TypeError, ValueError):
        return None
    return (max(int(tokens or 0), 0) / 1_000_000) * price


def provider_prompt_cache_accounting(
    model: str,
    *,
    provider: str = "anthropic",
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    processing_mode: str | None = None,
) -> dict[str, Any]:
    """Return provider-side prompt-cache economics using the provider pricing table."""

    creation_tokens = max(int(cache_creation_tokens or 0), 0)
    read_tokens = max(int(cache_read_tokens or 0), 0)
    basis = pricing_basis(model, provider=provider, processing_mode=processing_mode)
    input_price = basis.get("input_usd_per_million")
    cached_input_price = basis.get("cached_input_usd_per_million")
    creation_price = basis.get("cache_creation_input_usd_per_million")

    full_read_cost = _cost_from_per_million(read_tokens, input_price)
    cached_read_cost = _cost_from_per_million(read_tokens, cached_input_price)
    read_discount = None
    if full_read_cost is not None and cached_read_cost is not None:
        read_discount = max(full_read_cost - cached_read_cost, 0.0)

    full_creation_cost = _cost_from_per_million(creation_tokens, input_price)
    creation_cost = _cost_from_per_million(
        creation_tokens,
        creation_price if creation_price is not None else input_price,
    )
    creation_premium = None
    if full_creation_cost is not None and creation_cost is not None:
        creation_premium = max(creation_cost - full_creation_cost, 0.0)

    net_discount = None
    if read_discount is not None and creation_premium is not None:
        net_discount = read_discount - creation_premium

    actual_provider_cache_cost = None
    if cached_read_cost is not None and creation_cost is not None:
        actual_provider_cache_cost = cached_read_cost + creation_cost

    return {
        "schema": "agentflow.provider_prompt_cache_accounting.v1",
        "provider": basis["provider"],
        "model": model,
        "matched_model": basis.get("matched_model"),
        "processing_mode": basis.get("processing_mode"),
        "pricing_source": basis.get("source"),
        "pricing_version": basis.get("version"),
        "pricing_basis": basis,
        "cost_known": bool(
            basis.get("cost_known")
            and input_price is not None
            and cached_input_price is not None
            and (creation_price is not None or input_price is not None)
        ),
        "read_tokens": read_tokens,
        "creation_tokens": creation_tokens,
        "input_usd_per_million": input_price,
        "cached_input_usd_per_million": cached_input_price,
        "cache_creation_input_usd_per_million": creation_price,
        "full_price_equivalent_read_cost_usd": full_read_cost or 0.0,
        "actual_cached_read_cost_usd": cached_read_cost or 0.0,
        "read_discount_usd": read_discount or 0.0,
        "full_price_equivalent_creation_cost_usd": full_creation_cost or 0.0,
        "creation_cost_usd": creation_cost or 0.0,
        "creation_premium_usd": creation_premium or 0.0,
        "actual_provider_cache_cost_usd": actual_provider_cache_cost or 0.0,
        "net_provider_cache_discount_usd": net_discount or 0.0,
        "net_provider_cache_economics_usd": net_discount or 0.0,
        "semantics": (
            "provider-side prompt-cache accounting; read_discount is gross cached-read "
            "discount versus full input price, creation_premium is the extra write cost "
            "above normal input price, and net_provider_cache_discount subtracts that premium"
        ),
    }


def codex_app_pricing_basis() -> dict[str, Any]:
    return pricing_basis(codex_app_model(), provider="openai", processing_mode=codex_app_processing_mode())


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
    provider: str = "anthropic",
    processing_mode: str | None = None,
) -> Optional[float]:
    provider = provider.lower()
    if provider == "openai":
        _matched, openai_prices, _source = _openai_prices_for_model(
            model,
            processing_mode or DEFAULT_CODEX_APP_PROCESSING_MODE,
        )
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

    _matched, prices = _match_prices(model, ANTHROPIC_MODEL_PRICES)
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
    if provider == "openai":
        _matched, prices, _source = _openai_prices_for_model(model)
        if prices:
            return prices[2] if cache_read else prices[0]
        return None

    _matched, prices = _match_prices(model, ANTHROPIC_MODEL_PRICES)
    if prices:
        in_per_m = prices[0]
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
