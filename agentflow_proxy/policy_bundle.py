from __future__ import annotations

from typing import Any

from agentflow_proxy import __version__
from agentflow_proxy.store import utc_now


async def build_policy_bundle() -> dict[str, Any]:
    from agentflow_proxy import stats

    policy_state = await stats.stats_policies()
    return {
        "schema": "agentflow.policy_bundle.v1",
        "generated_at": utc_now(),
        "generator": {
            "name": "agentflow-proxy",
            "version": __version__,
            "mode": "local-offline",
        },
        "managed_optimizer": {
            "enabled": False,
            "note": "Export only. No managed optimizer communication is performed by this command.",
        },
        "policies": policy_state,
    }
