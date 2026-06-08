from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentflow_proxy.store import utc_now


POLICY_EVENT_SCHEMA = "agentflow.policy_event.v1"
POLICY_EVENTS_SCHEMA = "agentflow.policy_events.v1"


def policy_events_enabled() -> bool:
    return os.getenv("AGENTFLOW_POLICY_EVENTS", "1") != "0"


def policy_events_log_path() -> Path:
    return Path(os.getenv("AGENTFLOW_POLICY_EVENTS_LOG", "~/.agentflow/policy_events.jsonl")).expanduser()


def summarize_policy_state(policies: Any) -> dict[str, Any]:
    if not isinstance(policies, dict):
        return {}

    summary: dict[str, Any] = {}
    for section in ("routing", "crunch", "cache", "routing_experiments"):
        value = policies.get(section)
        if not isinstance(value, dict):
            continue
        file_status = value.get("file") if isinstance(value.get("file"), dict) else {}
        summary[section] = {
            "enabled": value.get("enabled"),
            "policy_source": value.get("policy_source"),
            "rule_path": value.get("rule_path"),
            "reload_required": file_status.get("reload_required"),
        }
    return summary


def _compact_details(details: dict[str, Any]) -> dict[str, Any]:
    compact = dict(details)
    if "policies" in compact:
        compact["policies"] = summarize_policy_state(compact["policies"])
    return compact


def log_policy_event(action: str, *, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not policy_events_enabled():
        return None

    event = {
        "schema": POLICY_EVENT_SCHEMA,
        "id": uuid4().hex,
        "created_at": utc_now(),
        "action": action,
        "ok": bool(ok),
        "details": _compact_details(details or {}),
    }
    path = policy_events_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
        return None
    return event


def recent_policy_events(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(int(limit), 500))
    path = policy_events_log_path()
    events: list[dict[str, Any]] = []
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in reversed(lines):
            if len(events) >= limit:
                break
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                events.append(payload)

    return {
        "schema": POLICY_EVENTS_SCHEMA,
        "enabled": policy_events_enabled(),
        "path": str(path),
        "limit": limit,
        "events": events,
    }
