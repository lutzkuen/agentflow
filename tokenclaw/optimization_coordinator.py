from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from tokenclaw.optimization_action_ledger import build_optimization_action_ledger
from tokenclaw.public_metadata import public_id, public_label


SCHEMA = "tokenclaw.optimization_coordinator.v1"
PRIORITY = (
    "cache_replay",
    "routing",
    "old_context_summary",
    "terminal_output_compaction",
    "anthropic_thinking_history_compaction",
    "repeated_scaffold_crunch",
)
ACTIONABLE_STATUSES = {"eligible", "applied", "hit", "recommended", "selected"}
BLOCKED_STATUSES = {"holdout", "suppressed", "ineligible", "not-eligible", "unknown"}
STALE_OR_MISSING_EVIDENCE = {
    "missing-evidence",
    "missing-lifecycle-evidence",
    "missing-canary-evidence",
    "missing-dependency-evidence",
    "missing-dependency-freshness-evidence",
    "stale-evidence",
    "stale-lifecycle-evidence",
    "stale-canary-evidence",
    "aggregate-only-evidence",
    "insufficient-evidence",
}
DEPENDENCY_FRESHNESS_REASONS = {
    "dependency-stable",
    "dependency-fresh",
    "dependency-freshness-proven",
    "fresh-dependency-evidence",
    "safe-invalidation-proven",
    "file-dependency-stable",
}
SAFETY_REASONS = {
    "rollback",
    "rollback-required",
    "safety-stop",
    "safety-stop-tripped",
    "safety-stopped",
    "safety-regression",
    "quality-regression",
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


def _float_0_1(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _hash_value(parts: list[str]) -> tuple[str, float]:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return "sha256:" + digest, score


def _stable_hash(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _reason_code(value: Any) -> str | None:
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
        or any(term in text for term in ("api", "body", "cache-key", "content", "file", "message", "path", "prompt", "request", "response", "secret", "session", "tenant", "tool"))
    ):
        return public_id(text, prefix="reason", fallback="redacted-reason")
    return text


def _reason_codes(*values: Any) -> list[str]:
    codes: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                code = _reason_code(item)
                if code:
                    codes.append(code)
        else:
            code = _reason_code(value)
            if code:
                codes.append(code)
    return sorted(set(codes))


def _status(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("_", "-")


def _policy_source(entries: list[dict[str, Any]], selected: dict[str, Any] | None = None) -> str:
    if selected:
        source = selected.get("policy_source")
        if source:
            return public_label(source, "local-default")
    for source in ("managed-enforced", "managed-recommended", "local-manual", "local-default"):
        if any(entry.get("policy_source") == source for entry in entries):
            return source
    return "local-default"


def _entry_ids(entry: dict[str, Any]) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for key, prefix in (
        ("policy_id", "policy"),
        ("rule_id", "rule"),
        ("candidate_id", "candidate"),
        ("target_candidate_id", "candidate"),
        ("action_id", "action"),
        ("promotion_action_id", "action"),
    ):
        value = entry.get(key)
        if value not in (None, ""):
            public = public_id(value, prefix=prefix)
            if public:
                identifiers[key] = public
    return identifiers


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    reason_codes = _reason_codes(entry.get("reason_codes"), entry.get("reason"), entry.get("status"))
    result = {
        "family": public_label(entry.get("family"), "unknown"),
        "status": _status(entry.get("status")),
        "policy_source": public_label(entry.get("policy_source"), "local-default"),
        "source_surface": public_label(entry.get("source_surface"), "provider_request"),
        "provider_family": public_label(entry.get("provider_family"), "unknown"),
        "endpoint": public_label(entry.get("endpoint"), "unknown"),
        "category": public_label(entry.get("category"), "unknown") if entry.get("category") else None,
        "phase": public_label(entry.get("phase"), "unknown") if entry.get("phase") else None,
        "text_bucket": public_label(entry.get("text_bucket"), "unknown"),
        "input_token_bucket": public_label(entry.get("input_token_bucket"), "unknown"),
        "projected_savings_bucket": public_label(entry.get("projected_savings_bucket"), "unknown"),
        "reason_codes": reason_codes,
        "requires_local_policy_change": bool(entry.get("requires_local_policy_change")),
        **_entry_ids(entry),
    }
    return {key: value for key, value in result.items() if value not in (None, "", [])}


def _is_safety_entry(entry: dict[str, Any]) -> bool:
    status = _status(entry.get("status"))
    reasons = set(_reason_codes(entry.get("reason_codes"), entry.get("reason"), status))
    return status in {"rollback", "safety-stop", "safety-stopped"} or bool(reasons & SAFETY_REASONS)


def _suppression_for_entry(entry: dict[str, Any]) -> list[str]:
    family = str(entry.get("family") or "")
    status = _status(entry.get("status"))
    reasons = set(_reason_codes(entry.get("reason_codes"), entry.get("reason")))
    if status in BLOCKED_STATUSES:
        return [f"entry-status-{status}"]
    if reasons & STALE_OR_MISSING_EVIDENCE:
        return sorted(reasons & STALE_OR_MISSING_EVIDENCE)
    if family == "cache_replay" and status == "eligible":
        has_freshness = bool(reasons & DEPENDENCY_FRESHNESS_REASONS) or bool(
            entry.get("dependency_freshness_proven") or entry.get("safe_invalidation_evidence")
        )
        if not has_freshness:
            return ["missing-dependency-freshness-evidence"]
    return []


def _priority_index(entry: dict[str, Any]) -> tuple[int, str]:
    family = str(entry.get("family") or "")
    if _is_safety_entry(entry):
        return (-1, family)
    if family in PRIORITY:
        return (PRIORITY.index(family), family)
    if family.startswith("pattern_crunch:"):
        return (len(PRIORITY), family)
    return (len(PRIORITY) + 1, family)


def _cohort(
    *,
    ledger: dict[str, Any],
    local_salt: str | None,
    canary_fraction: float | None,
    holdout_fraction: float | None,
) -> dict[str, Any]:
    entries = ledger.get("entries") if isinstance(ledger.get("entries"), list) else []
    first = entries[0] if entries and isinstance(entries[0], dict) else {}
    salt = local_salt if local_salt is not None else os.getenv("TOKENCLAW_OPTIMIZATION_COORDINATOR_SALT", "tokenclaw-optimization-coordinator-v1")
    canary = _float_0_1(
        canary_fraction if canary_fraction is not None else os.getenv("TOKENCLAW_OPTIMIZATION_COORDINATOR_CANARY_FRACTION"),
        1.0,
    )
    holdout = _float_0_1(
        holdout_fraction if holdout_fraction is not None else os.getenv("TOKENCLAW_OPTIMIZATION_COORDINATOR_HOLDOUT_FRACTION"),
        0.0,
    )
    if canary + holdout > 1.0:
        canary = max(0.0, 1.0 - holdout)
    families = ",".join(sorted(public_label(entry.get("family"), "unknown") for entry in entries if isinstance(entry, dict)))
    key_hash, score = _hash_value([
        salt,
        public_label(first.get("source_surface"), "provider_request"),
        public_label(first.get("provider_family"), "unknown"),
        public_label(first.get("endpoint"), "unknown"),
        public_label(first.get("category"), "unknown"),
        public_label(first.get("phase"), "unknown"),
        public_label(first.get("text_bucket"), "unknown"),
        public_label(first.get("input_token_bucket"), "unknown"),
        families,
    ])
    if score < holdout:
        cohort = "coordinator_holdout"
        selected = False
        is_holdout = True
    elif score < holdout + canary:
        cohort = "coordinator_canary"
        selected = True
        is_holdout = False
    else:
        cohort = "coordinator_not_selected"
        selected = False
        is_holdout = False
    return {
        "cohort": cohort,
        "selected": selected,
        "holdout": is_holdout,
        "cohort_key_hash": key_hash,
        "cohort_score": round(score, 12),
        "canary_fraction": canary,
        "holdout_fraction": holdout,
        "cohort_basis": "public-metadata-hash",
        "salt_included": False,
    }


def build_optimization_coordinator(
    *,
    ledger: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
    local_salt: str | None = None,
    canary_fraction: float | None = None,
    holdout_fraction: float | None = None,
) -> dict[str, Any]:
    source_ledger = ledger if isinstance(ledger, dict) else build_optimization_action_ledger(
        row=row,
        routing_meta=_json_obj(routing_meta),
        crunch_meta=_json_obj(crunch_meta),
        cache_meta=_json_obj(cache_meta),
    )
    entries = [
        _public_entry(entry)
        for entry in source_ledger.get("entries", [])
        if isinstance(entry, dict)
    ]
    cohort = _cohort(
        ledger={"entries": entries},
        local_salt=local_salt,
        canary_fraction=canary_fraction,
        holdout_fraction=holdout_fraction,
    )

    suppressed: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        reasons = _suppression_for_entry(entry)
        if reasons:
            suppressed.append({
                "family": entry["family"],
                "status": entry["status"],
                "reason_codes": _reason_codes(reasons),
                **_entry_ids(entry),
            })
            continue
        if _is_safety_entry(entry) or entry["status"] in ACTIONABLE_STATUSES:
            candidates.append(entry)

    selected: dict[str, Any] | None = None
    if cohort["selected"] and candidates:
        selected = sorted(candidates, key=_priority_index)[0]

    if not cohort["selected"] and candidates:
        reason = "coordinator-holdout" if cohort["holdout"] else "coordinator-canary-not-selected"
        for entry in candidates:
            suppressed.append({
                "family": entry["family"],
                "status": entry["status"],
                "reason_codes": [reason],
                **_entry_ids(entry),
            })
    elif selected:
        selected_family = selected["family"]
        for entry in candidates:
            if entry["family"] == selected_family:
                continue
            suppressed.append({
                "family": entry["family"],
                "status": entry["status"],
                "reason_codes": ["conflicts-with-selected-family"],
                **_entry_ids(entry),
            })

    first = entries[0] if entries else {}
    selected_family = selected["family"] if selected else "none"
    family_status = {}
    suppressed_by_family = {item["family"]: item for item in suppressed}
    for entry in entries:
        family = entry["family"]
        family_status[family] = {
            "eligible": entry in candidates or family == selected_family,
            "selected": family == selected_family,
            "status": entry["status"],
            "policy_source": entry.get("policy_source", "local-default"),
            "reason_codes": suppressed_by_family.get(family, {}).get("reason_codes", entry.get("reason_codes", [])),
        }

    reason_codes: list[str] = []
    if selected is None:
        if not candidates:
            reason_codes.append("no-eligible-families")
        elif cohort["holdout"]:
            reason_codes.append("coordinator-holdout")
        else:
            reason_codes.append("coordinator-canary-not-selected")
    if any(_is_safety_entry(entry) for entry in candidates):
        reason_codes.append("safety-stop-priority")

    decision = {
        "schema": SCHEMA,
        "selected_family": selected_family,
        "selected_action_family": selected_family,
        "selected_candidate": ({**_entry_ids(selected), "status": selected.get("status"), "policy_source": selected.get("policy_source")} if selected else None),
        "suppressed_families": suppressed,
        "family_status": family_status,
        "eligible_families": [entry["family"] for entry in candidates],
        "candidate_count": len(candidates),
        "entry_count": len(entries),
        "reason_codes": _reason_codes(reason_codes),
        "policy_source": _policy_source(entries, selected),
        "source_surface": public_label(first.get("source_surface"), "provider_request"),
        "provider_family": public_label(first.get("provider_family"), "unknown"),
        "endpoint": public_label(first.get("endpoint"), "unknown"),
        "category": public_label(first.get("category"), "unknown") if first.get("category") else None,
        "phase": public_label(first.get("phase"), "unknown") if first.get("phase") else None,
        "text_bucket": public_label(first.get("text_bucket"), "unknown"),
        "input_token_bucket": public_label(first.get("input_token_bucket"), "unknown"),
        "canary": cohort,
        "conservative_single_mutation": True,
        "provider_body_changed": False,
        "policy_files_changed": False,
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
            "local_salt_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
    decision = {key: value for key, value in decision.items() if value not in (None, "", [])}
    hash_payload = {key: value for key, value in decision.items() if key != "decision_hash"}
    decision["decision_hash"] = _stable_hash(hash_payload)
    return decision
