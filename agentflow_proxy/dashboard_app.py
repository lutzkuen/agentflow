from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Callable

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from agentflow_proxy import stats as stats_views
from agentflow_proxy.store import utc_now


LimiterStatus = Callable[[], list[dict[str, Any]]]
StoreSource = Any | Callable[[], Any]


def _store(store_source: StoreSource) -> Any:
    return store_source() if callable(store_source) else store_source


def _full_stats_ttl_s() -> float:
    raw = os.getenv("AGENTFLOW_DASHBOARD_FULL_STATS_TTL_SECONDS", "5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def create_dashboard_router(
    *,
    store_obj: StoreSource,
    default_db: str,
    limiter_status: LimiterStatus,
    limiter_config: dict[str, Any],
    proxy_host: str | None = None,
    dashboard_host: str | None = None,
    dashboard_read_only: bool = True,
    full_stats_ttl_s: float | None = None,
) -> APIRouter:
    router = APIRouter()
    stats_ttl_s = _full_stats_ttl_s() if full_stats_ttl_s is None else max(0.0, float(full_stats_ttl_s))
    full_stats_cache: dict[str, Any] | None = None
    full_stats_cache_at = 0.0
    full_stats_task: asyncio.Task[dict[str, Any]] | None = None
    full_stats_lock = asyncio.Lock()

    async def load_full_stats() -> dict[str, Any]:
        nonlocal full_stats_cache, full_stats_cache_at
        result = await stats_views.stats_full(_store(store_obj))
        full_stats_cache = result
        full_stats_cache_at = time.monotonic()
        return result

    def consume_task_exception(task: asyncio.Task[dict[str, Any]]) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"agentflow_dashboard_full_stats_refresh_error: {exc}", file=sys.stderr)
            return
        if exc is not None:
            print(f"agentflow_dashboard_full_stats_refresh_error: {exc}", file=sys.stderr)

    @router.get("/agentflow/stats")
    async def stats() -> dict[str, Any]:
        return await stats_views.stats(_store(store_obj), default_db)

    @router.get("/agentflow/stats/activity")
    async def stats_activity(limit: int = 100) -> dict[str, Any]:
        return await stats_views.stats_activity(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/quality-signals")
    async def stats_quality_signals(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_quality_signals(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/provider-adoption-health")
    async def stats_provider_adoption_health(limit: int = 5000) -> dict[str, Any]:
        return await stats_views.stats_provider_adoption_health(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/full")
    async def stats_full() -> dict[str, Any]:
        nonlocal full_stats_cache, full_stats_cache_at, full_stats_task
        if stats_ttl_s <= 0:
            return await stats_views.stats_full(_store(store_obj))

        now = time.monotonic()
        if full_stats_cache is not None and now - full_stats_cache_at < stats_ttl_s:
            return full_stats_cache

        async with full_stats_lock:
            now = time.monotonic()
            if full_stats_cache is not None and now - full_stats_cache_at < stats_ttl_s:
                return full_stats_cache
            if full_stats_task is None or full_stats_task.done():
                full_stats_task = asyncio.create_task(load_full_stats())
                full_stats_task.add_done_callback(consume_task_exception)
            task = full_stats_task
            stale = full_stats_cache

        if stale is not None:
            return stale
        return await task

    @router.get("/agentflow/stats/usage")
    async def stats_usage() -> dict[str, Any]:
        return await stats_views.stats_usage_by_owner(_store(store_obj))

    @router.get("/agentflow/stats/limiter")
    async def stats_limiter() -> dict[str, Any]:
        return await stats_views.stats_limiter(_store(store_obj), limiter_status, limiter_config)

    @router.get("/agentflow/stats/codex-effectiveness")
    async def stats_codex_effectiveness(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_codex_effectiveness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/codex-readiness")
    async def stats_codex_readiness(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_codex_readiness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/codex-canary-impact")
    async def stats_codex_canary_impact(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_codex_canary_impact(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/cache-replayability")
    async def stats_cache_replayability(limit: int = 25) -> dict[str, Any]:
        return await stats_views.stats_cache_replayability(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/cache-replay-cohorts")
    async def stats_cache_replay_cohorts(limit: int = 25, scan_limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_cache_replay_cohort_ranking(_store(store_obj), limit=limit, row_limit=scan_limit)

    @router.get("/agentflow/stats/cache-replay-confidence")
    async def stats_cache_replay_confidence(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_cache_replay_confidence(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/cache-replay-readiness")
    async def stats_cache_replay_readiness(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_cache_replay_readiness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/cache-replay-activation-health")
    async def stats_cache_replay_activation_health(limit: int = 1000, scan_limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_cache_replay_activation_health(
            _store(store_obj),
            limit=limit,
            scan_limit=scan_limit,
        )

    @router.get("/agentflow/stats/cache-effectiveness")
    async def stats_cache_effectiveness(limit: int = 10, scan_limit: int = 5000) -> dict[str, Any]:
        return await stats_views.stats_cache_effectiveness(_store(store_obj), limit=limit, scan_limit=scan_limit)

    @router.get("/agentflow/stats/old-context-summary")
    async def stats_old_context_summary() -> dict[str, Any]:
        return await stats_views.stats_old_context_summary(_store(store_obj))

    @router.get("/agentflow/stats/phase-routing")
    async def stats_phase_routing(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_phase_routing(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/session-phase-memory")
    async def stats_session_phase_memory(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_session_phase_memory(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/policies")
    async def stats_policies() -> dict[str, Any]:
        return await stats_views.stats_policies()

    @router.get("/agentflow/stats/policy-workbench")
    async def stats_policy_workbench() -> dict[str, Any]:
        return await stats_views.stats_policy_workbench_readiness()

    @router.get("/agentflow/stats/policy-events")
    async def stats_policy_events(limit: int = 50) -> dict[str, Any]:
        return await stats_views.stats_policy_events(limit=limit)

    @router.get("/agentflow/stats/managed-recommendations")
    async def stats_managed_recommendations(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_managed_recommendations(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-scoreboard")
    async def stats_openai_scoreboard(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_openai_scoreboard(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-optimization-readiness")
    async def stats_openai_optimization_readiness(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_openai_optimization_readiness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/managed-openai-activation")
    async def stats_managed_openai_activation(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_managed_openai_activation(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-canary-readiness")
    async def stats_openai_canary_readiness(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_openai_canary_readiness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/claude-canary-impact")
    async def stats_claude_canary_impact(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_claude_canary_impact(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/claude-routing-promotion-funnel")
    async def stats_claude_routing_promotion_funnel(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_claude_routing_promotion_funnel(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-old-context-summary")
    async def stats_openai_old_context_summary(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_openai_old_context_summary_report(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-cache-replay")
    async def stats_openai_cache_replay(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_openai_cache_replay_report(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-cache-replay-impact")
    async def stats_openai_cache_replay_impact(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_openai_cache_replay_impact(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/openai-cache-replay-readiness")
    async def stats_openai_cache_replay_readiness(
        opportunity_limit: int = 1000,
        impact_limit: int = 500,
    ) -> dict[str, Any]:
        return await stats_views.stats_openai_cache_replay_readiness(
            _store(store_obj),
            opportunity_limit=opportunity_limit,
            impact_limit=impact_limit,
        )

    @router.get("/agentflow/stats/repeated-scaffold-opportunity")
    async def stats_repeated_scaffold_opportunity(
        limit: int = 1000,
        min_repeated_rows: int = 2,
    ) -> dict[str, Any]:
        return await stats_views.stats_repeated_scaffold_opportunity(
            _store(store_obj),
            limit=limit,
            min_repeated_rows=min_repeated_rows,
        )

    @router.get("/agentflow/stats/instruction-dedup-opportunity")
    async def stats_instruction_dedup_opportunity(
        limit: int = 1000,
        min_repeated_rows: int = 2,
    ) -> dict[str, Any]:
        return await stats_views.stats_instruction_dedup_opportunity(
            _store(store_obj),
            limit=limit,
            min_repeated_rows=min_repeated_rows,
        )

    @router.get("/agentflow/stats/terminal-output-compaction")
    async def stats_terminal_output_compaction(
        opportunity_limit: int = 1000,
        impact_limit: int = 500,
        limit: int | None = None,
        min_text_chars: int = 8000,
        max_plateau_delta_ratio: float = 0.03,
    ) -> dict[str, Any]:
        if limit is not None:
            opportunity_limit = limit
        return await stats_views.stats_terminal_output_compaction_readiness(
            _store(store_obj),
            opportunity_limit=opportunity_limit,
            impact_limit=impact_limit,
            min_text_chars=min_text_chars,
            max_plateau_delta_ratio=max_plateau_delta_ratio,
        )

    @router.get("/agentflow/stats/terminal-output-compaction-activation")
    async def stats_terminal_output_compaction_activation(
        opportunity_limit: int = 1000,
        impact_limit: int = 500,
        limit: int | None = None,
    ) -> dict[str, Any]:
        if limit is not None:
            impact_limit = limit
        return await stats_views.stats_terminal_output_compaction_activation(
            _store(store_obj),
            opportunity_limit=opportunity_limit,
            impact_limit=impact_limit,
        )

    @router.get("/agentflow/stats/repeated-scaffold-impact")
    async def stats_repeated_scaffold_impact(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_repeated_scaffold_impact(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/repeated-scaffold-activation")
    async def stats_repeated_scaffold_activation(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_repeated_scaffold_activation(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/scaffold-rollout-health")
    async def stats_scaffold_rollout_health(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_scaffold_rollout_health(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/shadow-routing-promotion-readiness")
    async def stats_shadow_routing_promotion_readiness(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_shadow_routing_promotion_readiness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/optimization-eval-queue")
    async def stats_optimization_eval_queue(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_optimization_eval_queue(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/optimization-coordinator")
    async def stats_optimization_coordinator(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_optimization_coordinator_dashboard(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/optimization-promotion-funnel")
    async def stats_optimization_promotion_funnel(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_optimization_promotion_funnel(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/rollout-actions/readiness")
    async def stats_rollout_actions_readiness(limit: int = 500) -> dict[str, Any]:
        return await stats_views.stats_rollout_actions_readiness(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/local-pattern-coverage")
    async def stats_local_pattern_coverage(limit: int = 1000) -> dict[str, Any]:
        return await stats_views.stats_local_pattern_coverage(_store(store_obj), limit=limit)

    @router.get("/agentflow/stats/safety")
    async def stats_safety() -> dict[str, Any]:
        return await stats_views.stats_safety(
            store_obj=_store(store_obj),
            default_db=default_db,
            proxy_host=proxy_host,
            dashboard_host=dashboard_host,
            dashboard_read_only=dashboard_read_only,
        )

    @router.get("/agentflow/stats/weekly")
    async def stats_weekly() -> dict[str, Any]:
        return await stats_views.stats_weekly(_store(store_obj))

    @router.get("/agentflow/stats/sessions")
    async def stats_sessions() -> dict[str, Any]:
        return await stats_views.stats_sessions(_store(store_obj))

    @router.get("/agentflow/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return stats_views.dashboard_html()

    return router


def create_dashboard_app(
    *,
    store_obj: StoreSource,
    default_db: str,
    upstream: str,
    limiter_status: LimiterStatus,
    limiter_config: dict[str, Any],
    proxy_host: str | None = None,
    dashboard_host: str | None = None,
    full_stats_ttl_s: float | None = None,
) -> FastAPI:
    app = FastAPI(title="AgentFlow Dashboard", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "mode": "dashboard-read-only",
            "db": default_db,
            "upstream": upstream,
            "time": utc_now(),
        }

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse("/agentflow/dashboard")

    app.include_router(
        create_dashboard_router(
            store_obj=store_obj,
            default_db=default_db,
            limiter_status=limiter_status,
            limiter_config=limiter_config,
            proxy_host=proxy_host,
            dashboard_host=dashboard_host,
            dashboard_read_only=True,
            full_stats_ttl_s=full_stats_ttl_s,
        )
    )
    return app
