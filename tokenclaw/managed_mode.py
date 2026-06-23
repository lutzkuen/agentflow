from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenclaw.env import env


MANAGED_MODE_SCHEMA = "tokenclaw.managed_product_mode.v1"
MANAGED_ENV = "TOKENCLAW_MANAGED"
MANAGED_MODE_ENV = "TOKENCLAW_MANAGED_MODE"
LOCAL_RULES_ONLY_ENV = "TOKENCLAW_LOCAL_RULES_ONLY"
MANAGED_ROUTING_ENV = "TOKENCLAW_MANAGED_ROUTING"
MANAGED_CRUNCH_ENV = "TOKENCLAW_MANAGED_CRUNCH"
MANAGED_CACHE_ENV = "TOKENCLAW_MANAGED_CACHE"

VALID_MANAGED_MODES = {"local_only", "observe_only", "dry_run", "canary", "live"}


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return default


def _normalize_mode(value: Any) -> str:
    raw = str(value or "observe_only").strip().lower().replace("-", "_")
    aliases = {
        "off": "local_only",
        "disabled": "local_only",
        "local": "local_only",
        "local_rules": "local_only",
        "observe": "observe_only",
        "observeonly": "observe_only",
        "dryrun": "dry_run",
        "dry": "dry_run",
        "shadow": "canary",
        "apply": "live",
        "active": "live",
        "enforced": "live",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in VALID_MANAGED_MODES else "observe_only"


def _explicit_bool(name: str) -> bool | None:
    raw = env(name)
    if raw is None:
        return None
    return _bool_value(raw, False)


def legacy_managed_enabled() -> bool:
    for name in (
        "TOKENCLAW_RECOMMENDATIONS_ENABLED",
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_POLICY_DECISIONS_ENABLED",
        "TOKENCLAW_POLICY_DECISION_ENABLED",
    ):
        raw = env(name)
        if raw is not None and _bool_value(raw, False):
            return True
    return False


@dataclass(frozen=True)
class ManagedProductMode:
    mode: str
    configured: bool
    managed_enabled: bool
    local_rules_only: bool
    server_calls_enabled: bool
    local_application_enabled: bool
    family_enabled: dict[str, bool]
    reason: str

    def public_meta(self) -> dict[str, Any]:
        return {
            "schema": MANAGED_MODE_SCHEMA,
            "mode": self.mode,
            "configured": self.configured,
            "managed_enabled": self.managed_enabled,
            "local_rules_only": self.local_rules_only,
            "server_calls_enabled": self.server_calls_enabled,
            "local_application_enabled": self.local_application_enabled,
            "families": dict(self.family_enabled),
            "reason": self.reason,
            "env": {
                "global": MANAGED_ENV,
                "mode": MANAGED_MODE_ENV,
                "local_rules_only": LOCAL_RULES_ONLY_ENV,
                "routing": MANAGED_ROUTING_ENV,
                "crunch": MANAGED_CRUNCH_ENV,
                "cache": MANAGED_CACHE_ENV,
            },
            "raw_values_included": False,
            "api_key_value_included": False,
        }


def managed_product_mode() -> ManagedProductMode:
    local_rules_only = _bool_value(env(LOCAL_RULES_ONLY_ENV), False)
    explicit_global = _explicit_bool(MANAGED_ENV)
    configured = explicit_global is not None or env(MANAGED_MODE_ENV) is not None
    if local_rules_only:
        mode = "local_only"
        managed_enabled = False
        reason = "local-rules-only"
    elif explicit_global is False:
        mode = "local_only"
        managed_enabled = False
        reason = "managed-disabled"
    else:
        managed_enabled = explicit_global is True or legacy_managed_enabled()
        mode = _normalize_mode(env(MANAGED_MODE_ENV, "observe_only"))
        if managed_enabled and not configured:
            mode = "live"
        if not managed_enabled:
            mode = "local_only"
            reason = "managed-not-enabled"
        else:
            reason = f"managed-{mode}"

    server_calls_enabled = managed_enabled and mode != "local_only"
    local_application_enabled = server_calls_enabled and mode in {"canary", "live"}
    family_enabled = {
        "routing": _bool_value(env(MANAGED_ROUTING_ENV), True) if server_calls_enabled else False,
        "crunch": _bool_value(env(MANAGED_CRUNCH_ENV), True) if server_calls_enabled else False,
        "cache": _bool_value(env(MANAGED_CACHE_ENV), True) if server_calls_enabled else False,
    }
    if not any(family_enabled.values()):
        local_application_enabled = False
    return ManagedProductMode(
        mode=mode,
        configured=configured,
        managed_enabled=managed_enabled,
        local_rules_only=local_rules_only,
        server_calls_enabled=server_calls_enabled,
        local_application_enabled=local_application_enabled,
        family_enabled=family_enabled,
        reason=reason,
    )


def managed_mode_public_meta() -> dict[str, Any]:
    return managed_product_mode().public_meta()
