from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentflow_proxy.public_metadata import public_id, public_label


LEDGER_SCHEMA = "agentflow.optimization_action_ledger.v1"
ENTRY_SCHEMA = "agentflow.optimization_action_ledger_entry.v1"
REPORT_SCHEMA = "agentflow.optimization_action_ledger_report.v1"

FAMILIES = {
    "routing",
    "old_context_summary",
    "cache_replay",
    "repeated_scaffold_crunch",
    "terminal_output_compaction",
    "anthropic_thinking_history_compaction",
}

RAW_REASON_HINTS = {
    "api",
    "apikey",
    "authorization",
    "body",
    "cache-key",
    "cache_key",
    "content",
    "file",
    "message",
    "path",
    "payload",
    "prompt",
    "request",
    "response",
    "secret",
    "session",
    "tenant",
    "thread",
    "tool-payload",
    "tool_payload",
}


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_bucket(chars: Any) -> str:
    value = _as_int(chars)
    if value <= 0:
        return "unknown"
    if value < 1_500:
        return "lt_1_5k"
    if value < 8_000:
        return "1_5k_8k"
    if value < 30_000:
        return "8k_30k"
    if value < 120_000:
        return "30k_120k"
    return "gte_120k"


def _token_bucket(tokens: Any) -> str:
    value = _as_int(tokens)
    if value <= 0:
        return "unknown"
    if value < 500:
        return "lt_500"
    if value < 2_000:
        return "500_2k"
    if value < 8_000:
        return "2k_8k"
    if value < 30_000:
        return "8k_30k"
    return "gte_30k"


def _money_bucket(value: Any) -> str:
    amount = _as_float(value)
    if amount is None:
        return "unknown"
    if amount <= 0:
        return "none"
    if amount < 0.001:
        return "lt_0_001"
    if amount < 0.01:
        return "0_001_0_01"
    if amount < 0.10:
        return "0_01_0_10"
    return "gte_0_10"


def _public_reason(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    if (
        len(text) > 96
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-.:/" for char in text)
        or "/" in text
        or "\\" in text
        or any(term in text for term in RAW_REASON_HINTS)
    ):
        return public_id(text, prefix="reason", fallback="redacted-reason")
    return text


def _reason_codes(*values: Any) -> list[str]:
    codes: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                code = _public_reason(item)
                if code:
                    codes.append(code)
        else:
            code = _public_reason(value)
            if code:
                codes.append(code)
    return sorted(set(codes))


def _policy_source(*metas: dict[str, Any]) -> str:
    for source in ("managed-enforced", "managed-recommended", "local-manual", "local-default"):
        for meta in metas:
            if meta.get("policy_source") == source:
                return source
    return "local-default"


def _source_surface(row: dict[str, Any], routing_meta: dict[str, Any]) -> str:
    feature_unit = routing_meta.get("openai_feature_unit") if isinstance(routing_meta.get("openai_feature_unit"), dict) else {}
    return public_label(
        row.get("source_surface")
        or routing_meta.get("source_surface")
        or feature_unit.get("source_surface")
        or "provider_request",
        "provider_request",
    )


def _provider_family(row: dict[str, Any], routing_meta: dict[str, Any]) -> str:
    provider = row.get("provider") or routing_meta.get("provider")
    if provider:
        return public_label(provider, "unknown")
    surface = _source_surface(row, routing_meta)
    if surface.startswith("openai"):
        return "openai"
    if surface.startswith("anthropic"):
        return "anthropic"
    return "unknown"


def _phase(row: dict[str, Any], routing_meta: dict[str, Any]) -> str | None:
    feature_unit = routing_meta.get("openai_feature_unit") if isinstance(routing_meta.get("openai_feature_unit"), dict) else {}
    value = routing_meta.get("workflow_phase") or routing_meta.get("phase") or feature_unit.get("workflow_phase")
    return public_label(value, "unknown") if value else None


def _category(row: dict[str, Any], routing_meta: dict[str, Any]) -> str | None:
    feature_unit = routing_meta.get("openai_feature_unit") if isinstance(routing_meta.get("openai_feature_unit"), dict) else {}
    value = row.get("category") or routing_meta.get("category") or feature_unit.get("category")
    return public_label(value, "unknown") if value else None


def _local_policy_change_required(meta: dict[str, Any]) -> bool:
    if bool(meta.get("requires_local_policy_change") or meta.get("requires_local_policy_file_change")):
        return True
    return any(
        key in meta
        for key in (
            "local_policy_update",
            "policy_file_update",
            "recommended_local_policy_patch",
            "recommended_policy_patch",
        )
    )


def _ids(meta: dict[str, Any], *extra_metas: dict[str, Any]) -> dict[str, str]:
    merged = [meta, *extra_metas]
    identifiers: dict[str, str] = {}
    for key, prefix in (
        ("policy_id", "policy"),
        ("rule_id", "rule"),
        ("candidate_id", "candidate"),
        ("target_candidate_id", "candidate"),
        ("action_id", "action"),
        ("promotion_action_id", "action"),
    ):
        for item in merged:
            value = item.get(key)
            if value not in (None, ""):
                public = public_id(value, prefix=prefix)
                if public:
                    identifiers[key] = public
                break
    return identifiers


def _status_from_meta(meta: dict[str, Any], *, applied_default: bool = False) -> str:
    raw_status = str(meta.get("status") or "").strip().lower().replace("_", "-")
    if raw_status in {"applied", "hit", "selected", "recommended"} or meta.get("applied") is True:
        return "applied"
    if raw_status in {"holdout", "canary-holdout"}:
        return "holdout"
    if raw_status in {"bypass", "bypassed", "skipped", "suppressed", "not-selected", "noop", "safety-stopped"}:
        return "suppressed"
    if raw_status in {"disabled", "ineligible", "rejected"}:
        return "ineligible"
    if applied_default:
        return "applied"
    if meta:
        return "eligible"
    return "unknown"


def _base_entry(row: dict[str, Any], routing_meta: dict[str, Any], family: str) -> dict[str, Any]:
    text_chars = (
        routing_meta.get("text_chars")
        or row.get("text_chars")
        or (_as_int(row.get("actual_input_tokens") or row.get("input_tokens_est")) * 4)
    )
    input_tokens = row.get("actual_input_tokens") or row.get("input_tokens_est")
    cost = _as_float(row.get("cost_est_usd"))
    baseline = _as_float(row.get("cost_baseline_usd"))
    projected_savings = None
    if cost is not None and baseline is not None:
        projected_savings = max(0.0, baseline - cost)
    return {
        "schema": ENTRY_SCHEMA,
        "family": family,
        "source_surface": _source_surface(row, routing_meta),
        "provider_family": _provider_family(row, routing_meta),
        "endpoint": public_label(row.get("endpoint") or routing_meta.get("endpoint") or row.get("path"), "unknown"),
        "category": _category(row, routing_meta),
        "phase": _phase(row, routing_meta),
        "text_bucket": _text_bucket(text_chars),
        "input_token_bucket": _token_bucket(input_tokens),
        "policy_source": "local-default",
        "status": "unknown",
        "reason_codes": [],
        "projected_cost_bucket": _money_bucket(cost),
        "projected_savings_bucket": _money_bucket(projected_savings),
        "requires_local_policy_change": False,
    }


def _finalize_entry(entry: dict[str, Any], meta: dict[str, Any], *extra_metas: dict[str, Any]) -> dict[str, Any]:
    entry["policy_source"] = _policy_source(meta, *extra_metas)
    entry.update(_ids(meta, *extra_metas))
    entry["requires_local_policy_change"] = any(_local_policy_change_required(item) for item in (meta, *extra_metas))
    for item in (meta, *extra_metas):
        if not isinstance(item, dict):
            continue
        for source_key, target_key in (
            ("projected_savings_usd", "projected_savings_usd"),
            ("projected_saved_usd", "projected_savings_usd"),
            ("estimated_saved_cost_usd", "projected_savings_usd"),
            ("estimated_gross_savings_usd", "projected_savings_usd"),
            ("gross_savings_usd", "projected_savings_usd"),
            ("projected_holdout_savings_usd", "projected_savings_usd"),
            ("net_savings_usd", "projected_savings_usd"),
            ("tokens_saved_est", "projected_saved_tokens"),
            ("saved_tokens_est", "projected_saved_tokens"),
            ("projected_saved_tokens", "projected_saved_tokens"),
            ("saved_chars", "projected_saved_chars"),
            ("planned_saved_chars", "projected_saved_chars"),
            ("projected_saved_chars", "projected_saved_chars"),
        ):
            if target_key in entry or source_key not in item:
                continue
            value = item.get(source_key)
            if value in (None, ""):
                continue
            if target_key == "projected_savings_usd":
                entry[target_key] = round(max(0.0, _as_float(value)), 8)
            else:
                entry[target_key] = max(0, _as_int(value))
    return {key: value for key, value in entry.items() if value not in (None, "", [])}


def _governor_entries(
    row: dict[str, Any],
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    governor = {}
    for meta in (routing_meta, crunch_meta, cache_meta):
        raw = meta.get("openai_optimization_governor")
        if isinstance(raw, dict):
            governor = raw
            break
    family_status = governor.get("family_status") if isinstance(governor.get("family_status"), dict) else {}
    if not family_status:
        return []
    suppressed = {
        str(item.get("family")): item
        for item in governor.get("suppressed_families", [])
        if isinstance(item, dict) and item.get("family")
    }
    selected = set(str(item) for item in governor.get("selected_action_families", []) if item)
    entries: list[dict[str, Any]] = []
    for family, status_meta in sorted(family_status.items()):
        if not isinstance(status_meta, dict):
            continue
        if family not in {"routing", "old_context_summary", "cache_replay"}:
            continue
        entry = _base_entry(row, routing_meta, family)
        reason_meta = suppressed.get(str(family), {})
        eligible = bool(status_meta.get("eligible"))
        is_selected = bool(status_meta.get("selected")) or str(family) in selected
        if is_selected:
            status = "applied"
        elif reason_meta:
            status = "suppressed"
        elif eligible:
            status = "eligible"
        else:
            status = "ineligible"
        entry.update({
            "status": status,
            "reason_codes": _reason_codes(reason_meta.get("reason_codes"), status_meta.get("reason_codes")),
        })
        entries.append(_finalize_entry(entry, status_meta, governor, reason_meta))
    return entries


def _routing_entry(row: dict[str, Any], routing_meta: dict[str, Any]) -> dict[str, Any] | None:
    if not routing_meta:
        return None
    requested = str(row.get("requested_model") or routing_meta.get("requested_model") or "")
    routed = str(row.get("routed_model") or routing_meta.get("routed_model") or requested)
    canary = routing_meta.get("openai_canary") if isinstance(routing_meta.get("openai_canary"), dict) else {}
    if not (routing_meta.get("enabled") or canary or (requested and routed and requested != routed)):
        return None
    meta = canary or routing_meta
    entry = _base_entry(row, routing_meta, "routing")
    status = _status_from_meta(meta, applied_default=bool(requested and routed and requested != routed))
    if status == "eligible" and requested and routed and requested != routed:
        status = "applied"
    entry.update({
        "status": status,
        "reason_codes": _reason_codes(meta.get("reason_codes"), meta.get("reason"), routing_meta.get("reason")),
    })
    return _finalize_entry(entry, meta, routing_meta)


def _old_context_entry(row: dict[str, Any], routing_meta: dict[str, Any], crunch_meta: dict[str, Any]) -> dict[str, Any] | None:
    meta = crunch_meta.get("old_context_summarization")
    if not isinstance(meta, dict) or not meta:
        return None
    entry = _base_entry(row, routing_meta, "old_context_summary")
    entry.update({
        "status": _status_from_meta(meta),
        "reason_codes": _reason_codes(meta.get("reason_codes"), meta.get("reason"), meta.get("status")),
    })
    return _finalize_entry(entry, meta)


def _cache_replay_entry(row: dict[str, Any], routing_meta: dict[str, Any], cache_meta: dict[str, Any]) -> dict[str, Any] | None:
    if not cache_meta:
        return None
    canary = cache_meta.get("cache_replay_canary") if isinstance(cache_meta.get("cache_replay_canary"), dict) else {}
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    replay_like = bool(canary or pattern_rule or cache_meta.get("replayability_level") or cache_meta.get("safe_invalidation_evidence"))
    if not replay_like:
        return None
    meta = canary or pattern_rule or cache_meta
    entry = _base_entry(row, routing_meta, "cache_replay")
    applied = bool(row.get("cache_hit")) or str(cache_meta.get("status") or "").lower() == "hit"
    entry.update({
        "status": _status_from_meta(meta, applied_default=applied),
        "reason_codes": _reason_codes(
            meta.get("reason_codes"),
            meta.get("reason"),
            cache_meta.get("reason"),
            cache_meta.get("status"),
        ),
    })
    return _finalize_entry(entry, meta, pattern_rule, cache_meta)


def _repeated_scaffold_entry(row: dict[str, Any], routing_meta: dict[str, Any], crunch_meta: dict[str, Any]) -> dict[str, Any] | None:
    meta = crunch_meta.get("codex_repeated_scaffolding")
    if not isinstance(meta, dict) or not meta:
        return None
    entry = _base_entry(row, routing_meta, "repeated_scaffold_crunch")
    entry.update({
        "status": _status_from_meta(meta, applied_default=bool(meta.get("saved_chars") or meta.get("tokens_saved_est"))),
        "reason_codes": _reason_codes(meta.get("reason_codes"), meta.get("reason"), meta.get("status")),
    })
    return _finalize_entry(entry, meta)


def _terminal_compaction_entry(row: dict[str, Any], routing_meta: dict[str, Any], crunch_meta: dict[str, Any]) -> dict[str, Any] | None:
    meta = crunch_meta.get("terminal_output_compaction")
    if not isinstance(meta, dict) or not meta:
        return None
    entry = _base_entry(row, routing_meta, "terminal_output_compaction")
    entry.update({
        "status": _status_from_meta(meta, applied_default=bool(meta.get("applied") or meta.get("saved_chars"))),
        "reason_codes": _reason_codes(meta.get("reason_codes"), meta.get("reason"), meta.get("status")),
    })
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    return _finalize_entry(entry, meta, canary)


def _anthropic_thinking_compaction_entry(row: dict[str, Any], routing_meta: dict[str, Any], crunch_meta: dict[str, Any]) -> dict[str, Any] | None:
    meta = crunch_meta.get("anthropic_thinking_history_compaction")
    if not isinstance(meta, dict) or not meta:
        return None
    entry = _base_entry(row, routing_meta, "anthropic_thinking_history_compaction")
    entry.update({
        "status": _status_from_meta(meta, applied_default=bool(meta.get("applied") or meta.get("saved_chars"))),
        "reason_codes": _reason_codes(meta.get("reason_codes"), meta.get("reason"), meta.get("status")),
    })
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    return _finalize_entry(entry, meta, canary)


def _pattern_crunch_entries(row: dict[str, Any], routing_meta: dict[str, Any]) -> list[dict[str, Any]]:
    features = routing_meta.get("managed_pattern_features")
    if not isinstance(features, dict):
        return []
    families = features.get("local_pattern_module_families")
    if not isinstance(families, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_family in families[:10]:
        public_family = public_label(raw_family, "pattern")
        family = f"pattern_crunch:{public_family}"
        entry = _base_entry(row, routing_meta, family)
        entry.update({
            "status": "eligible",
            "reason_codes": _reason_codes(features.get("reason_codes"), "pattern-family-observed"),
        })
        entries.append(_finalize_entry(entry, features))
    return entries


def build_optimization_action_ledger(
    *,
    row: dict[str, Any] | None = None,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_row = dict(row or {})
    routing = _json_obj(routing_meta if routing_meta is not None else local_row.get("routing_json"))
    crunch = _json_obj(crunch_meta if crunch_meta is not None else local_row.get("crunch_json"))
    cache = _json_obj(cache_meta if cache_meta is not None else local_row.get("cache_json"))
    entries = _governor_entries(local_row, routing, crunch, cache)
    if not entries:
        for entry in (
            _routing_entry(local_row, routing),
            _old_context_entry(local_row, routing, crunch),
            _cache_replay_entry(local_row, routing, cache),
            _repeated_scaffold_entry(local_row, routing, crunch),
            _terminal_compaction_entry(local_row, routing, crunch),
            _anthropic_thinking_compaction_entry(local_row, routing, crunch),
        ):
            if entry is not None:
                entries.append(entry)
    else:
        for entry in (
            _repeated_scaffold_entry(local_row, routing, crunch),
            _terminal_compaction_entry(local_row, routing, crunch),
            _anthropic_thinking_compaction_entry(local_row, routing, crunch),
        ):
            if entry is not None:
                entries.append(entry)
    entries.extend(_pattern_crunch_entries(local_row, routing))
    return {
        "schema": LEDGER_SCHEMA,
        "entry_count": len(entries),
        "entries": entries,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "terminal_lines_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }


def build_optimization_action_ledger_report(store_obj: Any, *, limit: int = 1000) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 10_000))
    if hasattr(store_obj, "optimization_action_ledger_rows"):
        rows = store_obj.optimization_action_ledger_rows(limit=capped)
    else:
        rows = []
    entries: list[dict[str, Any]] = []
    sampled_rows = 0
    rows_with_entries = 0
    for row in rows:
        sampled_rows += 1
        ledger = build_optimization_action_ledger(row=dict(row))
        row_entries = ledger["entries"]
        if row_entries:
            rows_with_entries += 1
            entries.extend(row_entries)

    by_family_status = Counter((entry["family"], entry["status"]) for entry in entries)
    by_family_reason = Counter(
        (entry["family"], reason)
        for entry in entries
        for reason in entry.get("reason_codes", ["none"])
    )
    by_surface_family = Counter((entry.get("source_surface", "unknown"), entry["family"]) for entry in entries)
    return {
        "schema": REPORT_SCHEMA,
        "read_only": True,
        "sampled_call_count": sampled_rows,
        "rows_with_ledger_entries": rows_with_entries,
        "entry_count": len(entries),
        "family_status_counts": [
            {"family": family, "status": status, "count": count}
            for (family, status), count in sorted(by_family_status.items(), key=lambda item: (-item[1], item[0]))
        ],
        "family_reason_counts": [
            {"family": family, "reason": reason, "count": count}
            for (family, reason), count in sorted(by_family_reason.items(), key=lambda item: (-item[1], item[0]))
        ],
        "surface_family_counts": [
            {"source_surface": surface, "family": family, "count": count}
            for (surface, family), count in sorted(by_surface_family.items(), key=lambda item: (-item[1], item[0]))
        ],
        "sample_entries": entries[: min(50, len(entries))],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "terminal_lines_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
