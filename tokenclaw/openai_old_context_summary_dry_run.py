from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from tokenclaw.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from tokenclaw.pricing import estimate_cost, pricing_basis
from tokenclaw.store import utc_now


SCHEMA = "agentflow.openai_old_context_summary_dry_run.v1"
TOKEN_CHARS = 4

DEFAULT_SUMMARY_PROVIDER = "openai"
DEFAULT_SUMMARY_MODEL = "gpt-5-mini"
DEFAULT_MIN_REQUEST_CHARS = 32_000
DEFAULT_MIN_SOURCE_CHARS = 8_000
DEFAULT_MAX_SOURCE_CHARS = 80_000
DEFAULT_KEEP_RECENT_ITEMS = 4
DEFAULT_MAX_SUMMARY_CHARS = 4_000
DEFAULT_SUMMARY_COMPRESSION_RATIO = 0.125
DEFAULT_MAX_SUMMARY_COST_USD = 0.02
DEFAULT_SUPPORTED_ENDPOINTS = ("responses", "chat_completions")
DEFAULT_BLOCKED_CATEGORIES = ("tool-heavy", "tool-result")

TOOL_ITEM_TYPES = {
    "function_call",
    "function_call_output",
    "tool_call",
    "tool_result",
    "computer_call",
    "computer_call_output",
    "file_search_call",
    "web_search_call",
}
INSTRUCTION_ROLES = {"system", "developer"}


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    return max(0, _as_int(os.getenv(name), default))


def _env_float(name: str, default: float) -> float:
    return max(0.0, _as_float(os.getenv(name), default))


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _summary_provider_configured() -> bool:
    if _env_bool("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER_CONFIGURED", False):
        return True
    return bool(os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _policy(
    *,
    summary_provider_configured: bool | None = None,
    summary_model: str | None = None,
) -> dict[str, Any]:
    provider_ready = _summary_provider_configured() if summary_provider_configured is None else bool(summary_provider_configured)
    canary_fraction = min(1.0, _env_float("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_CANARY_FRACTION", 0.0))
    holdout_fraction = min(1.0, _env_float("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_HOLDOUT_FRACTION", 1.0))
    return {
        "schema": "agentflow.openai_old_context_summary_dry_run_policy.v1",
        "read_only": True,
        "provider_calls_made": False,
        "summary_provider": os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER") or DEFAULT_SUMMARY_PROVIDER,
        "summary_provider_configured": provider_ready,
        "summary_model": summary_model or os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MODEL") or DEFAULT_SUMMARY_MODEL,
        "min_request_chars": _env_int("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS", DEFAULT_MIN_REQUEST_CHARS),
        "min_source_chars": _env_int("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MIN_SOURCE_CHARS", DEFAULT_MIN_SOURCE_CHARS),
        "max_source_chars": _env_int("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS", DEFAULT_MAX_SOURCE_CHARS),
        "keep_recent_items": _env_int("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_KEEP_RECENT_ITEMS", DEFAULT_KEEP_RECENT_ITEMS),
        "max_summary_chars": _env_int("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS", DEFAULT_MAX_SUMMARY_CHARS),
        "summary_compression_ratio": _env_float(
            "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_COMPRESSION_RATIO",
            DEFAULT_SUMMARY_COMPRESSION_RATIO,
        ),
        "max_summary_cost_usd": _env_float(
            "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SUMMARY_COST_USD",
            DEFAULT_MAX_SUMMARY_COST_USD,
        ),
        "supported_endpoints": list(_env_csv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_ENDPOINTS", DEFAULT_SUPPORTED_ENDPOINTS)),
        "blocked_categories": list(_env_csv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_BLOCKED_CATEGORIES", DEFAULT_BLOCKED_CATEGORIES)),
        "canary": {
            "enabled": canary_fraction > 0.0,
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "cohort_basis": "deterministic-candidate-id-hash",
        },
    }


def _metadata_text_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_metadata_text_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(_metadata_text_chars(item) for item in value.values())
    return 0


def _has_tool_protocol(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_tool_protocol(item) for item in value)
    if not isinstance(value, dict):
        return False
    item_type = str(value.get("type") or "").lower()
    if item_type in TOOL_ITEM_TYPES:
        return True
    if value.get("role") == "tool" or value.get("tool_call_id"):
        return True
    if isinstance(value.get("tool_calls"), list) and value["tool_calls"]:
        return True
    return any(_has_tool_protocol(item) for item in value.values())


def _has_file_reference(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_file_reference(item) for item in value)
    if not isinstance(value, dict):
        return False
    item_type = str(value.get("type") or "").lower()
    if item_type in {"input_file", "file", "file_path", "file_reference"}:
        return True
    if any(key in value for key in ("file_id", "filename", "file_path", "attachments")):
        return True
    return any(_has_file_reference(item) for item in value.values())


def _body_text_chars(body: dict[str, Any], routing: dict[str, Any], row: dict[str, Any]) -> int:
    text_chars = _as_int(routing.get("text_chars"))
    if text_chars > 0:
        return text_chars
    raw = row.get("request_json")
    if raw:
        return len(str(raw))
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * TOKEN_CHARS)


def _source_surface(row: dict[str, Any], feature: dict[str, Any]) -> str:
    return str(feature.get("source_surface") or row.get("source_surface") or openai_source_surface(str(row.get("path") or "")))


def _endpoint(row: dict[str, Any], feature: dict[str, Any]) -> str:
    return str(feature.get("endpoint") or row.get("endpoint") or openai_endpoint(str(row.get("path") or "")))


def _feature_summary(routing: dict[str, Any]) -> dict[str, Any]:
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        value = routing.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _candidate_id(row: dict[str, Any], endpoint: str, source_chars: int, source_items: int) -> str:
    basis = "|".join(
        str(part)
        for part in (
            row.get("id") or "",
            row.get("created_at") or "",
            row.get("requested_model") or "",
            endpoint,
            source_chars,
            source_items,
        )
    )
    return "openai-old-context-summary-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def _cohort(candidate_id: str, policy: dict[str, Any]) -> str:
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    canary_fraction = _as_float(canary.get("canary_fraction"))
    if canary_fraction <= 0:
        return "dry_run_only"
    value = int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < canary_fraction:
        return "canary_candidate"
    holdout_fraction = _as_float(canary.get("holdout_fraction"), 1.0)
    if value < canary_fraction + holdout_fraction:
        return "holdout_candidate"
    return "not_selected"


def _chat_windows(body: dict[str, Any], keep_recent: int) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return {"supported": False, "reason": "unsupported_request_shape"}
    old_limit = max(0, len(messages) - keep_recent)
    source: list[Any] = []
    preserved_instructions = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return {"supported": False, "reason": "unsupported_request_shape"}
        if str(message.get("role") or "").lower() in INSTRUCTION_ROLES:
            preserved_instructions += 1
            continue
        if index < old_limit:
            source.append(message)
    return {
        "supported": True,
        "shape": "chat_messages",
        "conversation_item_count": len(messages),
        "source_items": source,
        "source_item_count": len(source),
        "recent_item_count": min(keep_recent, len(messages)),
        "preserved_instruction_count": preserved_instructions,
        "top_level_instruction_present": False,
        "response_format_present": bool(body.get("response_format")),
    }


def _responses_windows(body: dict[str, Any], keep_recent: int) -> dict[str, Any]:
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return {"supported": False, "reason": "unsupported_request_shape"}
    old_limit = max(0, len(input_items) - keep_recent)
    source: list[Any] = []
    preserved_instructions = 0
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            return {"supported": False, "reason": "unsupported_request_shape"}
        role = str(item.get("role") or "").lower()
        if role in INSTRUCTION_ROLES:
            preserved_instructions += 1
            continue
        if index < old_limit:
            source.append(item)
    return {
        "supported": True,
        "shape": "responses_input_items",
        "conversation_item_count": len(input_items),
        "source_items": source,
        "source_item_count": len(source),
        "recent_item_count": min(keep_recent, len(input_items)),
        "preserved_instruction_count": preserved_instructions,
        "top_level_instruction_present": bool(body.get("instructions")),
        "response_format_present": bool(body.get("text") or body.get("response_format")),
    }


def _window_for_endpoint(body: dict[str, Any], endpoint: str, keep_recent: int) -> dict[str, Any]:
    if endpoint == "chat_completions":
        return _chat_windows(body, keep_recent)
    if endpoint == "responses":
        return _responses_windows(body, keep_recent)
    return {"supported": False, "reason": "unsupported_endpoint"}


def _projection(
    *,
    requested_model: str,
    endpoint: str,
    source_chars: int,
    source_item_count: int,
    policy: dict[str, Any],
) -> dict[str, Any]:
    bounded_source_chars = min(source_chars, _as_int(policy.get("max_source_chars"), DEFAULT_MAX_SOURCE_CHARS))
    summary_chars = min(
        _as_int(policy.get("max_summary_chars"), DEFAULT_MAX_SUMMARY_CHARS),
        max(400, int(bounded_source_chars * _as_float(policy.get("summary_compression_ratio"), DEFAULT_SUMMARY_COMPRESSION_RATIO))),
    )
    saved_chars = max(0, bounded_source_chars - summary_chars)
    saved_tokens = max(0, saved_chars // TOKEN_CHARS)
    summary_input_tokens = max(1, bounded_source_chars // TOKEN_CHARS)
    summary_output_tokens = max(1, summary_chars // TOKEN_CHARS)
    summary_cost = estimate_cost(
        str(policy.get("summary_model") or ""),
        summary_input_tokens,
        summary_output_tokens,
        provider="openai",
    ) or 0.0
    basis = pricing_basis(requested_model or "", provider="openai")
    input_price = _as_float(basis.get("input_usd_per_million"))
    gross = (saved_tokens / 1_000_000.0) * input_price
    return {
        "source_chars": bounded_source_chars,
        "source_tokens_est": summary_input_tokens,
        "source_item_count": source_item_count,
        "summary_chars": summary_chars,
        "summary_output_tokens_est": summary_output_tokens,
        "expected_saved_chars": saved_chars,
        "expected_saved_tokens": saved_tokens,
        "estimated_summary_cost_usd": round(summary_cost, 8),
        "projected_gross_savings_usd": round(gross, 8),
        "projected_net_savings_usd": round(gross - summary_cost, 8),
        "summary_request_shape": {
            "schema": "agentflow.openai_old_context_summary_request_shape.v1",
            "provider": policy.get("summary_provider"),
            "model": policy.get("summary_model"),
            "endpoint": endpoint,
            "source_item_count": source_item_count,
            "source_chars": bounded_source_chars,
            "source_tokens_est": summary_input_tokens,
            "max_summary_chars": summary_chars,
            "output_format": "plain_text_old_context_summary",
            "raw_source_included": False,
            "provider_calls_made": False,
        },
    }


def _plan_row(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("provider") or "").lower() != "openai":
        return None
    routing = _json_obj(row.get("routing_json"))
    feature = _feature_summary(routing)
    endpoint = _endpoint(row, feature)
    source_surface = _source_surface(row, feature)
    requested_model = str(row.get("requested_model") or routing.get("requested_model") or "")
    category = str(row.get("category") or feature.get("category") or routing.get("category") or "unknown")
    raw = row.get("request_json")
    body = _json_obj(raw)
    text_chars = _body_text_chars(body, routing, row)
    model_family = str(feature.get("requested_model_family") or row.get("requested_model_family") or openai_model_family(requested_model) or "unknown")

    reason_codes: list[str] = []
    if endpoint not in set(policy.get("supported_endpoints") or []):
        reason_codes.append("unsupported_endpoint")
    if category in set(policy.get("blocked_categories") or []):
        reason_codes.append("blocked_category")
    if not policy.get("summary_provider_configured"):
        reason_codes.append("summary_provider_not_configured")
    if not body:
        reason_codes.append("raw_body_unavailable")

    window = {"supported": False, "reason": "raw_body_unavailable"}
    if body:
        window = _window_for_endpoint(body, endpoint, _as_int(policy.get("keep_recent_items"), DEFAULT_KEEP_RECENT_ITEMS))
        if not window.get("supported"):
            reason_codes.append(str(window.get("reason") or "unsupported_request_shape"))

    source_items = window.get("source_items") if isinstance(window.get("source_items"), list) else []
    source_chars = _metadata_text_chars(source_items)
    if body and text_chars < _as_int(policy.get("min_request_chars"), DEFAULT_MIN_REQUEST_CHARS):
        reason_codes.append("request_below_min_chars")
    if body and source_chars < _as_int(policy.get("min_source_chars"), DEFAULT_MIN_SOURCE_CHARS):
        reason_codes.append("source_below_min_chars")
    if _has_tool_protocol(source_items) or _has_tool_protocol(body.get("tools")) or _has_tool_protocol(body.get("functions")):
        reason_codes.append("tool_function_protocol_ambiguous")
    if _has_file_reference(source_items):
        reason_codes.append("file_reference_in_source_window")

    projection = _projection(
        requested_model=requested_model,
        endpoint=endpoint,
        source_chars=source_chars,
        source_item_count=_as_int(window.get("source_item_count")),
        policy=policy,
    )
    if projection["estimated_summary_cost_usd"] > _as_float(policy.get("max_summary_cost_usd"), DEFAULT_MAX_SUMMARY_COST_USD):
        reason_codes.append("summary_cost_over_budget")

    reason_codes = sorted(set(reason_codes))
    status = "eligible" if not reason_codes else "blocked"
    candidate_id = _candidate_id(row, endpoint, projection["source_chars"], projection["source_item_count"])
    return {
        "candidate_id": candidate_id,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "requested_model_family": model_family,
        "category": category,
        "workflow_phase": str(feature.get("workflow_phase") or routing.get("workflow_phase") or "unknown"),
        "status": status,
        "canary_eligible": status == "eligible",
        "projected_cohort": _cohort(candidate_id, policy) if status == "eligible" else "blocked",
        "reason_codes": reason_codes or ["eligible"],
        "request_shape": {
            "shape": window.get("shape") or "unknown",
            "conversation_item_count": _as_int(window.get("conversation_item_count")),
            "source_item_count": projection["source_item_count"],
            "recent_item_count": _as_int(window.get("recent_item_count")),
            "text_chars": text_chars,
            "stream": bool(_as_int(row.get("stream")) or feature.get("stream")),
        },
        "preservation_checks": {
            "system_developer_instructions_preserved": True,
            "preserved_instruction_count": _as_int(window.get("preserved_instruction_count")),
            "top_level_instructions_preserved": bool(window.get("top_level_instruction_present")),
            "recent_items_preserved": _as_int(window.get("recent_item_count")),
            "tool_function_protocol_preserved": "tool_function_protocol_ambiguous" not in reason_codes,
            "response_format_constraints_preserved": bool(window.get("response_format_present")),
            "attachments_file_references_preserved": "file_reference_in_source_window" not in reason_codes,
            "streaming_compatible": True,
        },
        "expected_chars_saved": projection["expected_saved_chars"],
        "expected_tokens_saved": projection["expected_saved_tokens"],
        "estimated_summary_cost_usd": projection["estimated_summary_cost_usd"],
        "projected_gross_savings_usd": projection["projected_gross_savings_usd"],
        "projected_net_savings_usd": projection["projected_net_savings_usd"],
        "summary_request_shape": projection["summary_request_shape"],
        "privacy": {
            "raw_body_included": False,
            "raw_source_included": False,
            "tool_payloads_included": False,
            "function_arguments_included": False,
            "file_paths_included": False,
            "session_id_included": False,
            "request_id_included": False,
        },
    }


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [{"value": key, "count": value} for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def build_openai_old_context_summary_dry_run(
    store_obj: Any,
    limit: int = 1000,
    *,
    summary_provider_configured: bool | None = None,
    summary_model: str | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    policy = _policy(summary_provider_configured=summary_provider_configured, summary_model=summary_model)
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, input_tokens_est, actual_input_tokens, category,
                   routing_json, cache_json, request_json, session_id
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    plans: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    endpoints: dict[str, int] = {}
    surfaces: dict[str, int] = {}
    raw_available = 0
    for row in rows:
        plan = _plan_row(row, policy)
        if plan is None:
            continue
        if row.get("request_json"):
            raw_available += 1
        plans.append(plan)
        _increment(endpoints, str(plan["endpoint"]))
        _increment(surfaces, str(plan["source_surface"]))
        if plan["status"] != "eligible":
            for reason in plan["reason_codes"]:
                _increment(blockers, str(reason))

    eligible = [plan for plan in plans if plan["status"] == "eligible"]
    blocked = [plan for plan in plans if plan["status"] != "eligible"]
    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "openai_call_count": len(plans),
            "request_body_available_count": raw_available,
            "eligible_count": len(eligible),
            "blocked_count": len(blocked),
            "canary_eligible_count": sum(1 for plan in eligible if plan.get("canary_eligible")),
            "expected_chars_saved": sum(_as_int(plan.get("expected_chars_saved")) for plan in eligible),
            "expected_tokens_saved": sum(_as_int(plan.get("expected_tokens_saved")) for plan in eligible),
            "estimated_summary_cost_usd": round(sum(_as_float(plan.get("estimated_summary_cost_usd")) for plan in eligible), 8),
            "projected_gross_savings_usd": round(sum(_as_float(plan.get("projected_gross_savings_usd")) for plan in eligible), 8),
            "projected_net_savings_usd": round(sum(_as_float(plan.get("projected_net_savings_usd")) for plan in eligible), 8),
        },
        "policy": policy,
        "endpoint_breakdown": _breakdown(endpoints),
        "source_surface_breakdown": _breakdown(surfaces),
        "blocker_reason_breakdown": _breakdown(blockers),
        "plans": sorted(plans, key=lambda item: (item["status"] != "eligible", item["candidate_id"])),
        "privacy": {
            "metadata_only_output": True,
            "raw_bodies_read_locally": raw_available > 0,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "function_arguments_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "session_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
