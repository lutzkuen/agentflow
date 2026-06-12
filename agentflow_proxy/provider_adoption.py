from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

from agentflow_proxy.optimization.cli_support import default_db_path, open_store_for_db, write_json
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.provider_tool_adoption_report.v1"
WINDOW_SCHEMA = "agentflow.provider_tool_adoption_window.v1"
ADOPTION_TTL_SECONDS = int(os.getenv("AGENTFLOW_PROVIDER_ADOPTION_TTL_SECONDS", "3600"))
TOKEN_NAMESPACE = "agentflow.provider_tool_adoption.v1"


def _parse_utc(raw: Any) -> datetime | None:
    if raw is None:
        return None
    try:
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _age_bucket(created_at: Any, *, now: datetime | None = None) -> str:
    parsed = _parse_utc(created_at)
    if parsed is None:
        return "unknown"
    now = now or datetime.now(timezone.utc)
    seconds = max(0, int((now - parsed).total_seconds()))
    if seconds < 60:
        return "0_1m"
    if seconds < 300:
        return "1_5m"
    if seconds < 1800:
        return "5_30m"
    if seconds < 3600:
        return "30_60m"
    if seconds < 21600:
        return "1_6h"
    return "6h_plus"


def _digest(provider: str, session_id: str | None, tool_id: str) -> str:
    raw = f"{TOKEN_NAMESPACE}|{provider}|{session_id or 'unknown-session'}|{tool_id}"
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_surface(provider: str, path: str | None) -> str:
    lowered = (path or "").lower()
    if provider == "anthropic":
        return "anthropic_messages"
    if "chat/completions" in lowered:
        return "openai_chat_completions"
    if "responses" in lowered:
        return "openai_responses"
    return f"{provider}_unknown"


def _endpoint(path: str | None) -> str:
    lowered = (path or "").lower()
    if "chat/completions" in lowered:
        return "chat_completions"
    if "responses" in lowered:
        return "responses"
    if "messages" in lowered:
        return "messages"
    return "unknown"


def _workflow_phase(category: str | None, routing_meta: dict[str, Any] | None) -> str | None:
    meta = routing_meta or {}
    for key in ("workflow_phase", "phase", "category"):
        value = meta.get(key)
        if value:
            return str(value)
    return category


def _app_family(provider: str, source_surface: str, routing_meta: dict[str, Any] | None) -> str:
    meta = routing_meta or {}
    value = meta.get("app_family")
    if value:
        return str(value)
    if source_surface.startswith("openai"):
        return "generic_openai"
    if provider == "anthropic":
        return "claude"
    return provider


def _collect_policy_sources(*metas: dict[str, Any] | None) -> str:
    sources: set[str] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        source = value.get("policy_source")
        if source:
            sources.add(str(source))
        for nested_key in (
            "routing_experiment",
            "openai_canary",
            "old_context_summarization",
            "optimization_governor",
            "cache_replay_canary",
        ):
            visit(value.get(nested_key))

    for meta in metas:
        visit(meta)
    return "+".join(sorted(sources)) if sources else "unknown"


def _collect_policy_ids(*metas: dict[str, Any] | None) -> list[str]:
    ids: set[str] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key in ("policy_id", "rule_id", "candidate_id", "policy_version"):
            item = value.get(key)
            if item:
                ids.add(str(item)[:128])
        for nested_key in (
            "routing_experiment",
            "openai_canary",
            "old_context_summarization",
            "optimization_governor",
            "cache_replay_canary",
        ):
            visit(value.get(nested_key))

    for meta in metas:
        visit(meta)
    return sorted(ids)[:20]


def _anthropic_tool_results(body: dict[str, Any] | None) -> tuple[list[str], int]:
    ids: list[str] = []
    missing = 0
    if not isinstance(body, dict):
        return ids, missing
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                if tool_id:
                    ids.append(str(tool_id))
                else:
                    missing += 1
    return ids, missing


def _anthropic_tool_uses(body: dict[str, Any] | None) -> tuple[list[str], int]:
    ids: list[str] = []
    missing = 0
    if not isinstance(body, dict):
        return ids, missing
    content = body.get("content")
    if not isinstance(content, list):
        return ids, missing
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_id = block.get("id")
            if tool_id:
                ids.append(str(tool_id))
            else:
                missing += 1
    return ids, missing


def _openai_request_tool_results(body: dict[str, Any] | None) -> tuple[list[str], int]:
    ids: list[str] = []
    missing = 0
    if not isinstance(body, dict):
        return ids, missing

    def add_tool_result(value: Any) -> None:
        nonlocal missing
        if not isinstance(value, dict):
            return
        kind = str(value.get("type") or value.get("role") or "")
        if kind in {"function_call_output", "tool", "tool_result"}:
            tool_id = value.get("call_id") or value.get("tool_call_id") or value.get("id")
            if tool_id:
                ids.append(str(tool_id))
            else:
                missing += 1

    input_value = body.get("input")
    if isinstance(input_value, list):
        for item in input_value:
            add_tool_result(item)
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            add_tool_result(message)
    tool_outputs = body.get("tool_outputs")
    if isinstance(tool_outputs, list):
        for item in tool_outputs:
            if isinstance(item, dict):
                tool_id = item.get("tool_call_id") or item.get("call_id") or item.get("id")
                if tool_id:
                    ids.append(str(tool_id))
                else:
                    missing += 1
    return ids, missing


def _openai_response_tool_uses(body: dict[str, Any] | None) -> tuple[list[str], int]:
    ids: list[str] = []
    missing = 0
    if not isinstance(body, dict):
        return ids, missing

    def add_call(value: Any) -> None:
        nonlocal missing
        if not isinstance(value, dict):
            return
        kind = str(value.get("type") or "")
        if kind in {"function_call", "tool_call", "function"}:
            tool_id = value.get("call_id") or value.get("id")
            if tool_id:
                ids.append(str(tool_id))
            else:
                missing += 1

    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            add_call(item)
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if isinstance(message, dict):
            for call in message.get("tool_calls") or []:
                add_call(call)
            function_call = message.get("function_call")
            if isinstance(function_call, dict):
                tool_id = function_call.get("call_id") or function_call.get("id") or function_call.get("name")
                if tool_id:
                    ids.append(str(tool_id))
                else:
                    missing += 1
    return ids, missing


def openai_stream_tool_use_ids(event: dict[str, Any]) -> tuple[list[str], int]:
    ids, missing = _openai_response_tool_uses(event.get("response") if isinstance(event.get("response"), dict) else event)
    item = event.get("item")
    if isinstance(item, dict):
        item_ids, item_missing = _openai_response_tool_uses({"output": [item]})
        ids.extend(item_ids)
        missing += item_missing
    return ids, missing


def _provider_extractors(provider: str):
    if provider == "anthropic":
        return _anthropic_tool_results, _anthropic_tool_uses
    if provider == "openai":
        return _openai_request_tool_results, _openai_response_tool_uses
    return None, None


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def capture_provider_tool_adoption(
    store: Any,
    *,
    provider: str,
    path: str,
    call_id: str,
    session_id: str | None,
    request_body: dict[str, Any] | None,
    response_body: dict[str, Any] | None = None,
    response_tool_use_ids: Sequence[str] | None = None,
    response_tool_use_missing_ids: int = 0,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    category: str | None,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
    now: str | None = None,
) -> None:
    request_extractor, response_extractor = _provider_extractors(provider)
    if request_extractor is None or response_extractor is None:
        return
    now = now or utc_now()
    store.abandon_stale_provider_tool_adoption_windows(now=now, ttl_seconds=ADOPTION_TTL_SECONDS)

    surface = _source_surface(provider, path)
    common = {
        "provider": provider,
        "source_surface": surface,
        "endpoint": _endpoint(path),
        "app_family": _app_family(provider, surface, routing_meta),
        "requested_model": requested_model,
        "routed_model": routed_model,
        "category": category,
        "workflow_phase": _workflow_phase(category, routing_meta),
        "policy_source": _collect_policy_sources(routing_meta, crunch_meta, cache_meta),
        "policy_ids_json": json.dumps(_collect_policy_ids(routing_meta, crunch_meta, cache_meta), sort_keys=True),
        "session_digest": _digest(provider, session_id, "session"),
        "age_bucket": "0_1m",
    }

    result_ids, missing_result_ids = request_extractor(request_body)
    result_ids = _unique(result_ids)
    for tool_id in result_ids:
        correlation = _digest(provider, session_id, tool_id)
        matched = store.fulfill_provider_tool_adoption_window(
            correlation_digest=correlation,
            fulfilled_call_id=call_id,
            updated_at=now,
            age_bucket="0_1m",
        )
        if not matched:
            store.log_provider_tool_adoption_window(
                id=str(uuid.uuid4()),
                created_at=now,
                updated_at=now,
                call_id=call_id,
                status="orphan_result",
                reason="no-pending-tool-use-window",
                correlation_digest=correlation,
                tool_use_count=0,
                tool_result_count=1,
                **common,
            )
    if missing_result_ids:
        store.log_provider_tool_adoption_window(
            id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            call_id=call_id,
            status="unknown",
            reason="unsupported-result-shape-missing-tool-id",
            correlation_digest=None,
            tool_use_count=0,
            tool_result_count=missing_result_ids,
            **common,
        )

    if status_code is not None and status_code >= 400:
        return
    if response_tool_use_ids is None:
        response_tool_use_ids, response_tool_use_missing_ids = response_extractor(response_body)
    for tool_id in _unique(response_tool_use_ids):
        store.log_provider_tool_adoption_window(
            id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            call_id=call_id,
            status="pending",
            reason="assistant-tool-use-observed",
            correlation_digest=_digest(provider, session_id, tool_id),
            tool_use_count=1,
            tool_result_count=0,
            **common,
        )
    if response_tool_use_missing_ids:
        store.log_provider_tool_adoption_window(
            id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            call_id=call_id,
            status="unknown",
            reason="unsupported-tool-use-shape-missing-tool-id",
            correlation_digest=None,
            tool_use_count=response_tool_use_missing_ids,
            tool_result_count=0,
            **common,
        )


def _bump(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    counter[str(key or "unknown")] = int(counter.get(str(key or "unknown"), 0)) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_provider_tool_adoption_report(
    store: Any,
    *,
    limit: int = 5000,
    now: str | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    store.abandon_stale_provider_tool_adoption_windows(now=now, ttl_seconds=ADOPTION_TTL_SECONDS)
    now_dt = _parse_utc(now) or datetime.now(timezone.utc)
    rows = store.provider_tool_adoption_window_rows(limit=limit)
    status_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    app_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    age_counts: Counter[str] = Counter()
    windows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "unknown")
        age = _age_bucket(row.get("created_at"), now=now_dt)
        status_counts[status] += 1
        _bump(provider_counts, row.get("provider"))
        _bump(surface_counts, row.get("source_surface"))
        _bump(app_counts, row.get("app_family"))
        _bump(model_counts, row.get("routed_model") or row.get("requested_model"))
        _bump(category_counts, row.get("category"))
        _bump(phase_counts, row.get("workflow_phase"))
        _bump(policy_counts, row.get("policy_source"))
        _bump(age_counts, age)
        windows.append(
            {
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "provider": row.get("provider"),
                "source_surface": row.get("source_surface"),
                "endpoint": row.get("endpoint"),
                "app_family": row.get("app_family"),
                "requested_model": row.get("requested_model"),
                "routed_model": row.get("routed_model"),
                "category": row.get("category"),
                "workflow_phase": row.get("workflow_phase"),
                "policy_source": row.get("policy_source"),
                "status": status,
                "reason": row.get("reason"),
                "age_bucket": age,
                "tool_use_count": row.get("tool_use_count") or 0,
                "tool_result_count": row.get("tool_result_count") or 0,
            }
        )
    return {
        "schema": SCHEMA,
        "generated_at": now,
        "window_schema": WINDOW_SCHEMA,
        "ttl_seconds": ADOPTION_TTL_SECONDS,
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "breakdowns": {
            "provider": _breakdown(provider_counts),
            "source_surface": _breakdown(surface_counts),
            "app_family": _breakdown(app_counts),
            "model": _breakdown(model_counts),
            "category": _breakdown(category_counts),
            "workflow_phase": _breakdown(phase_counts),
            "policy_source": _breakdown(policy_counts),
            "age_bucket": _breakdown(age_counts),
        },
        "recent_windows": windows[:100],
        "privacy": {
            "raw_prompt_included": False,
            "raw_response_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tool_ids_included": False,
            "cache_keys_included": False,
            "provider_bodies_included": False,
            "correlation_digests_included": False,
        },
    }


def provider_tool_adoption_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report provider tool-use adoption windows from local metadata only"
    )
    parser.add_argument("--db", default=default_db_path(), help="AgentFlow database URL or SQLite path")
    parser.add_argument("--limit", type=int, default=5000, help="Recent adoption windows to inspect")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)
    stdout = stdout if stdout is not None else sys.stdout
    store = open_store_for_db(str(args.db))
    try:
        report = build_provider_tool_adoption_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, report)
    return 0
