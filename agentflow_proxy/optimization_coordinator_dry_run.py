from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from agentflow_proxy.optimization_action_ledger import build_optimization_action_ledger
from agentflow_proxy.optimization_coordinator import build_optimization_coordinator
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.optimization_coordinator_dry_run.v1"
ENTRY_SCHEMA = "agentflow.optimization_action_ledger_entry.v1"

_ACTIONABLE_TYPES = {"apply", "canary", "promote", "review", "widen"}
_HOLD_TYPES = {"hold", "more-samples", "request-more-samples"}
_SAFETY_TYPES = {"disable", "retire", "rollback", "safety-stop", "suppress"}
_RAW_PRIVACY_FLAGS = {
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_provider_bodies_included",
    "provider_bodies_returned",
    "request_ids_returned",
    "tenant_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
    "tool_payloads_returned",
}
_RAW_REASON_HINTS = {
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
    "provider-body",
    "provider_body",
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


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _money(value: float) -> float:
    return round(max(0.0, value), 8)


def _public_hash(value: Any, *, prefix: str) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _counter_rows(counter: Counter[Any], *names: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        values = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(names, values)}
        row["count"] = count
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _family_from_action(action: dict[str, Any]) -> str | None:
    family = str(action.get("action_family") or action.get("policy_section") or "").strip().lower()
    candidate = str(action.get("candidate_family") or action.get("target_candidate_id") or "").strip().lower()
    if family == "routing" or "routing" in candidate:
        return "routing"
    if family == "cache" or "cache" in candidate:
        return "cache_replay"
    if family == "old_context_summarization" or "old-context" in candidate or "summary" in candidate:
        return "old_context_summary"
    if family == "crunch" and ("terminal" in candidate or "compaction" in candidate):
        return "terminal_output_compaction"
    if family == "crunch" and ("scaffold" in candidate or "repeated" in candidate):
        return "repeated_scaffold_crunch"
    if family == "crunch":
        return "repeated_scaffold_crunch"
    return None


def _nested_dict(value: Any, *path: str) -> dict[str, Any]:
    item = value
    for key in path:
        if not isinstance(item, dict):
            return {}
        item = item.get(key)
    return item if isinstance(item, dict) else {}


def _reason_codes(action: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("reason", "reason_code", "reason_codes", "status", "action_type"):
        raw = action.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw not in (None, ""):
            values.append(raw)
    for path in (
        ("evidence_summary", "rollout_decision", "reason_codes"),
        ("evidence_summary", "rollout_gate", "reason_codes"),
        ("evidence_summary", "local_eval_verdict", "reason_codes"),
    ):
        raw = _nested_dict(action, *path[:-1]).get(path[-1])
        if isinstance(raw, list):
            values.extend(raw)
    codes: list[str] = []
    for value in values:
        text = public_label(str(value).strip().lower().replace("_", "-").replace(" ", "-"), "")
        if text:
            if any(hint.replace("_", "-") in text for hint in _RAW_REASON_HINTS):
                public = public_id(text, prefix="reason", fallback="redacted-reason")
                codes.append(public or "redacted-reason")
            else:
                codes.append(text)
    return sorted(set(codes))


def _action_status(action: dict[str, Any]) -> str:
    raw = str(action.get("action_type") or action.get("next_action") or action.get("status") or "").strip().lower().replace("_", "-")
    if raw in _ACTIONABLE_TYPES:
        return "eligible"
    if raw in _HOLD_TYPES:
        return "holdout"
    if raw in _SAFETY_TYPES:
        return "rollback" if raw in {"disable", "retire", "rollback"} else "safety-stop"
    return "eligible"


def _action_matches_row(action: dict[str, Any], row: dict[str, Any]) -> bool:
    routing = _json_obj(row.get("routing_json"))
    action_surface = str(action.get("source_surface") or "").strip()
    row_surface = str(row.get("source_surface") or routing.get("source_surface") or "")
    if action_surface and row_surface and action_surface != row_surface:
        return False
    action_endpoint = str(action.get("provider_endpoint") or action.get("endpoint") or "").strip()
    row_endpoint = str(row.get("endpoint") or routing.get("endpoint") or "").strip()
    if action_endpoint and row_endpoint and action_endpoint != row_endpoint:
        return False
    return True


def _safe_rollout_actions(bundle: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(bundle, dict):
        return [], {"present": False, "action_count": 0, "schema": None}
    actions = [item for item in bundle.get("actions", []) if isinstance(item, dict)]
    omitted = [item for item in bundle.get("omitted_actions", []) if isinstance(item, dict)]
    unsafe_flags: list[str] = []
    for scope in (bundle.get("privacy_summary"), *(action.get("privacy_summary") for action in actions)):
        if not isinstance(scope, dict):
            continue
        for key in sorted(_RAW_PRIVACY_FLAGS):
            if bool(scope.get(key)):
                unsafe_flags.append(key)
    compatible = _nested_dict(bundle, "local_executor_compatibility").get("compatible")
    summary = {
        "present": True,
        "schema": public_label(bundle.get("schema"), "unknown"),
        "action_count": len(actions),
        "omitted_action_count": len(omitted),
        "compatible": compatible if isinstance(compatible, bool) else None,
        "unsafe_privacy_flag_count": len(set(unsafe_flags)),
        "actions_included": False,
        "raw_rollout_payload_included": False,
    }
    if unsafe_flags:
        summary["unsafe_privacy_flags"] = sorted(set(public_label(flag, "unsafe-privacy-flag") for flag in unsafe_flags))
        return [], summary
    return actions, {key: value for key, value in summary.items() if value is not None}


def _rollout_entry(action: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    family = _family_from_action(action)
    if not family:
        return None
    routing = _json_obj(row.get("routing_json"))
    cost = _as_float(row.get("cost_est_usd"))
    baseline = _as_float(row.get("cost_baseline_usd"))
    text_chars = _as_int(routing.get("text_chars")) or (_as_int(row.get("actual_input_tokens") or row.get("input_tokens_est")) * 4)
    entry = {
        "schema": ENTRY_SCHEMA,
        "family": family,
        "status": _action_status(action),
        "policy_source": "managed-recommended",
        "source_surface": public_label(action.get("source_surface") or row.get("source_surface") or routing.get("source_surface"), "provider_request"),
        "provider_family": public_label(row.get("provider") or routing.get("provider"), "unknown"),
        "endpoint": public_label(action.get("provider_endpoint") or action.get("endpoint") or row.get("endpoint") or routing.get("endpoint"), "unknown"),
        "category": public_label(row.get("category") or routing.get("category"), "unknown"),
        "phase": public_label(routing.get("workflow_phase") or routing.get("phase"), "unknown"),
        "text_bucket": _text_bucket(text_chars),
        "input_token_bucket": _token_bucket(row.get("actual_input_tokens") or row.get("input_tokens_est")),
        "projected_savings_bucket": _money_bucket(max(0.0, baseline - cost)),
        "reason_codes": _reason_codes(action),
        "requires_local_policy_change": True,
    }
    for key, prefix in (
        ("action_id", "action"),
        ("target_candidate_id", "candidate"),
        ("candidate_id", "candidate"),
        ("target_rule_id", "rule"),
        ("rule_id", "rule"),
    ):
        value = action.get(key)
        if value in (None, "") and key in {"target_rule_id", "rule_id"}:
            value = _nested_dict(action, "action").get(key) or _nested_dict(action, "action", "proposed_edit").get(key)
        if value not in (None, ""):
            entry[key] = public_id(value, prefix=prefix)
    if family == "cache_replay" and bool(action.get("safe_invalidation_evidence")):
        entry["reason_codes"] = sorted(set([*entry["reason_codes"], "dependency-stable"]))
    return {key: value for key, value in entry.items() if value not in (None, "", [])}


def _text_bucket(chars: Any) -> str:
    value = _as_int(chars)
    if value <= 0:
        return "unknown"
    if value < 1500:
        return "lt_1_5k"
    if value < 8000:
        return "1_5k_8k"
    if value < 30000:
        return "8k_30k"
    if value < 120000:
        return "30k_120k"
    return "gte_120k"


def _token_bucket(tokens: Any) -> str:
    value = _as_int(tokens)
    if value <= 0:
        return "unknown"
    if value < 500:
        return "lt_500"
    if value < 2000:
        return "500_2k"
    if value < 8000:
        return "2k_8k"
    if value < 30000:
        return "8k_30k"
    return "gte_30k"


def _money_bucket(value: Any) -> str:
    amount = _as_float(value)
    if amount <= 0:
        return "none"
    if amount < 0.001:
        return "lt_0_001"
    if amount < 0.01:
        return "0_001_0_01"
    if amount < 0.10:
        return "0_01_0_10"
    return "gte_0_10"


def _filtered_rows(store_obj: Any, *, limit: int, provider: str | None, source_surface: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 1), 10_000))
    if not hasattr(store_obj, "optimization_action_ledger_rows"):
        return []
    rows = [dict(row) for row in store_obj.optimization_action_ledger_rows(limit=capped)]
    if provider:
        wanted = provider.strip().lower()
        rows = [row for row in rows if str(row.get("provider") or "").lower() == wanted]
    if source_surface:
        wanted_surface = source_surface.strip()
        rows = [
            row
            for row in rows
            if str(row.get("source_surface") or _json_obj(row.get("routing_json")).get("source_surface") or "") == wanted_surface
        ]
    return rows


def build_optimization_coordinator_dry_run(
    store_obj: Any,
    *,
    rollout_actions: Any = None,
    limit: int = 1000,
    provider: str | None = None,
    source_surface: str | None = None,
    local_salt: str | None = None,
    canary_fraction: float | None = None,
    holdout_fraction: float | None = None,
    examples: int = 20,
) -> dict[str, Any]:
    rows = _filtered_rows(store_obj, limit=limit, provider=provider, source_surface=source_surface)
    actions, managed_summary = _safe_rollout_actions(rollout_actions)

    selected_counts: Counter[str] = Counter()
    suppressed_counts: Counter[str] = Counter()
    reason_counts: Counter[tuple[str, str]] = Counter()
    conflict_counts: Counter[tuple[str, str, str]] = Counter()
    status_counts: Counter[int] = Counter()
    error_counts: Counter[str] = Counter()
    retry_counts: Counter[str] = Counter()
    surface_counts: Counter[tuple[str, str]] = Counter()
    savings_by_family: defaultdict[str, float] = defaultdict(float)
    cost_by_family: defaultdict[str, float] = defaultdict(float)
    samples: list[dict[str, Any]] = []
    rows_with_entries = 0
    rows_with_rollout_entries = 0
    rows_with_errors = 0
    holdout_count = 0
    noop_count = 0
    total_projected_savings = 0.0
    total_projected_cost = 0.0

    for index, row in enumerate(rows):
        ledger = build_optimization_action_ledger(row=row)
        entries = list(ledger.get("entries") or [])
        rollout_entries = [
            entry
            for action in actions
            if _action_matches_row(action, row)
            for entry in [_rollout_entry(action, row)]
            if entry is not None
        ]
        if rollout_entries:
            rows_with_rollout_entries += 1
            entries.extend(rollout_entries)
        if entries:
            rows_with_entries += 1

        decision = build_optimization_coordinator(
            ledger={
                "schema": "agentflow.optimization_action_ledger.v1",
                "entry_count": len(entries),
                "entries": entries,
                "privacy": ledger.get("privacy") or {},
            },
            local_salt=local_salt,
            canary_fraction=canary_fraction,
            holdout_fraction=holdout_fraction,
        )
        selected_family = str(decision.get("selected_family") or "none")
        selected_counts[selected_family] += 1
        if selected_family == "none":
            if decision.get("candidate_count") and (decision.get("canary") or {}).get("holdout"):
                holdout_count += 1
            elif decision.get("candidate_count"):
                noop_count += 1
            else:
                noop_count += 1

        status = _as_int(row.get("status_code"))
        status_counts[status] += 1
        if status >= 400 or bool(row.get("error_present")):
            rows_with_errors += 1
        if status >= 400:
            error_counts[str(status)] += 1
        retries = _as_int(row.get("retry_count"))
        if retries <= 0:
            retry_counts["0"] += 1
        elif retries == 1:
            retry_counts["1"] += 1
        elif retries == 2:
            retry_counts["2"] += 1
        else:
            retry_counts["gte_3"] += 1

        row_cost = _as_float(row.get("cost_est_usd"))
        row_savings = max(0.0, _as_float(row.get("cost_baseline_usd")) - row_cost)
        if selected_family != "none":
            savings_by_family[selected_family] += row_savings
            cost_by_family[selected_family] += row_cost
            total_projected_savings += row_savings
            total_projected_cost += row_cost

        for item in decision.get("suppressed_families", []):
            family = str(item.get("family") or "unknown")
            suppressed_counts[family] += 1
            for reason in item.get("reason_codes") or ["unknown"]:
                reason_counts[(family, str(reason))] += 1
                if str(reason) == "conflicts-with-selected-family":
                    conflict_counts[(selected_family, family, str(reason))] += 1

        surface_counts[(str(decision.get("source_surface") or "unknown"), selected_family)] += 1

        if len(samples) < max(0, min(int(examples or 0), 100)):
            samples.append({
                "example_id": _public_hash(
                    {
                        "index": index,
                        "created_at": row.get("created_at"),
                        "provider": row.get("provider"),
                        "source_surface": decision.get("source_surface"),
                        "selected_family": selected_family,
                        "decision_hash": decision.get("decision_hash"),
                    },
                    prefix="example",
                ),
                "selected_family": selected_family,
                "suppressed_family_count": len(decision.get("suppressed_families") or []),
                "candidate_count": decision.get("candidate_count", 0),
                "source_surface": decision.get("source_surface", "unknown"),
                "provider_family": decision.get("provider_family", "unknown"),
                "endpoint": decision.get("endpoint", "unknown"),
                "category": decision.get("category", "unknown"),
                "phase": decision.get("phase", "unknown"),
                "text_bucket": decision.get("text_bucket", "unknown"),
                "input_token_bucket": decision.get("input_token_bucket", "unknown"),
                "decision_hash": decision.get("decision_hash"),
                "selected_candidate": decision.get("selected_candidate"),
                "suppressed_families": decision.get("suppressed_families", [])[:5],
            })

    projected = [
        {
            "family": family,
            "projected_savings_usd_est": _money(amount),
            "projected_cost_usd_est": _money(cost_by_family[family]),
        }
        for family, amount in sorted(savings_by_family.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "sampled_call_count": len(rows),
        "rows_with_ledger_entries": rows_with_entries,
        "rows_with_rollout_action_candidates": rows_with_rollout_entries,
        "decision_count": len(rows),
        "filters": {
            "limit": max(1, min(int(limit or 1), 10_000)),
            "provider": public_label(provider, "any") if provider else None,
            "source_surface": public_label(source_surface, "any") if source_surface else None,
        },
        "managed_rollout_actions": managed_summary,
        "selected_family_counts": _counter_rows(selected_counts, "family"),
        "suppressed_family_counts": _counter_rows(suppressed_counts, "family"),
        "conflict_buckets": _counter_rows(conflict_counts, "selected_family", "suppressed_family", "reason"),
        "top_suppression_reason_codes": _counter_rows(reason_counts, "family", "reason", limit=20),
        "surface_selection_counts": _counter_rows(surface_counts, "source_surface", "selected_family"),
        "holdout_count": holdout_count,
        "noop_count": noop_count,
        "projected_savings_usd_est": _money(total_projected_savings),
        "projected_cost_usd_est": _money(total_projected_cost),
        "projected_savings_by_family": projected,
        "status_counts": _counter_rows(status_counts, "status_code"),
        "rows_with_errors": rows_with_errors,
        "error_status_counts": _counter_rows(error_counts, "status_code"),
        "retry_count_buckets": _counter_rows(retry_counts, "retry_count_bucket"),
        "sample_decisions": samples,
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
            "rollout_action_payloads_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_changed": False,
            "provider_body_changed": False,
        },
    }
