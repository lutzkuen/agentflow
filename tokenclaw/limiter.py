from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def model_tier(model: str) -> str:
    m = model.lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"


@dataclass(frozen=True)
class TierBackoffActive(Exception):
    tier: str
    remaining: float

    @property
    def retry_after(self) -> int:
        return max(1, int(self.remaining + 0.999))

    @property
    def message(self) -> str:
        return f"temporarily limiting requests for {self.tier} tier; retry after {self.retry_after}s"


class TierLimiter:
    def __init__(
        self,
        *,
        min_request_interval_ms: int = 0,
        max_tier_backoff_wait: float = 30.0,
        max_concurrent_per_tier: int = 2,
    ) -> None:
        self.min_request_interval_ms = min_request_interval_ms
        self.max_tier_backoff_wait = max_tier_backoff_wait
        self.max_concurrent_per_tier = max_concurrent_per_tier
        self.forward_lock = asyncio.Lock()
        self.last_forward_time = 0.0
        self.backoff_until: dict[str, float] = {}
        self.backoff_update_lock = asyncio.Lock()
        sem_value = max_concurrent_per_tier if max_concurrent_per_tier > 0 else 9999
        self.semaphores: dict[str, asyncio.Semaphore] = {
            "haiku": asyncio.Semaphore(sem_value),
            "sonnet": asyncio.Semaphore(sem_value),
            "opus": asyncio.Semaphore(sem_value),
        }

    def status(self, now: Optional[float] = None) -> list[dict[str, Any]]:
        now_ts = time.time() if now is None else now
        tiers = []
        for tier in ("haiku", "sonnet", "opus"):
            sem = self.semaphores[tier]
            until_ts = float(self.backoff_until.get(tier, 0.0) or 0.0)
            remaining = max(0.0, until_ts - now_ts)
            waiters = getattr(sem, "_waiters", None)
            queued = len(waiters) if waiters is not None else 0
            available = int(getattr(sem, "_value", 0))
            cooldown_until = (
                datetime.fromtimestamp(until_ts, timezone.utc).isoformat()
                if remaining > 0
                else None
            )
            tiers.append({
                "tier": tier,
                "active": remaining > 0,
                "cooldown_until": cooldown_until,
                "seconds_remaining": round(remaining, 1),
                "exceeds_max_wait": remaining > self.max_tier_backoff_wait,
                "max_concurrent": self.max_concurrent_per_tier,
                "available_slots": available if self.max_concurrent_per_tier > 0 else None,
                "queued_count": queued,
            })
        return tiers

    async def await_backoff(self, model: str) -> None:
        tier = model_tier(model)
        remaining = self.backoff_until.get(tier, 0.0) - time.time()
        if remaining <= 0:
            return
        if remaining > self.max_tier_backoff_wait:
            print(
                f"tier_backoff: tier={tier} remaining={remaining:.1f}s "
                f"exceeds_max_wait={self.max_tier_backoff_wait:.1f}s"
            )
            raise TierBackoffActive(tier=tier, remaining=remaining)
        print(f"tier_backoff: tier={tier} waiting={remaining:.1f}s")
        await asyncio.sleep(remaining)

    async def record_backoff(self, model: str, response_headers: Any, default_seconds: float = 60.0) -> None:
        tier = model_tier(model)
        raw = response_headers.get("retry-after")
        try:
            delay = float(raw) if raw else default_seconds
        except (ValueError, TypeError):
            delay = default_seconds
        new_until = time.time() + delay
        async with self.backoff_update_lock:
            if new_until > self.backoff_until.get(tier, 0.0):
                self.backoff_until[tier] = new_until

    async def throttle_forward(self) -> None:
        if self.min_request_interval_ms <= 0:
            return
        async with self.forward_lock:
            now = time.time()
            elapsed_ms = (now - self.last_forward_time) * 1000
            if elapsed_ms < self.min_request_interval_ms:
                await asyncio.sleep((self.min_request_interval_ms - elapsed_ms) / 1000)
            self.last_forward_time = time.time()


def tier_backoff_payload(exc: TierBackoffActive) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": exc.message,
        },
    }


def tier_backoff_headers(exc: TierBackoffActive, model: str) -> dict[str, str]:
    return {
        "retry-after": str(exc.retry_after),
        "x-tokenclaw-routed-model": model,
    }
