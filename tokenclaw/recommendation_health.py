from __future__ import annotations

from typing import Any


RAW_HEALTH_KEYS = {
    "arguments",
    "api_key",
    "apikey",
    "authorization",
    "body",
    "command",
    "commands",
    "completion",
    "content",
    "developer",
    "file_content",
    "input",
    "local_file",
    "message",
    "messages",
    "output",
    "params",
    "policy_yaml",
    "prompt",
    "provider_body",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "response",
    "system",
    "system_prompt",
    "tool_input",
    "tool_output",
    "tool_payload",
    "tool_payloads",
    "transcript",
    "transcripts",
}

HEALTH_WARNING_CODES = {
    "stale_evidence": "managed-recommendation-stale-evidence",
    "insufficient_samples": "managed-recommendation-insufficient-samples",
    "threshold_failure": "managed-recommendation-threshold-failure",
    "omitted_candidate": "managed-recommendation-omitted-candidate",
    "privacy_profile": "managed-recommendation-privacy-profile",
}

_SAFE_SCALAR_KEYS = {
    "candidate_id",
    "candidate_ids",
    "code",
    "confidence",
    "count",
    "error_rate",
    "field",
    "generated_at",
    "last_seen_at",
    "max_error_rate",
    "min_samples",
    "observed_at",
    "policy_id",
    "reason",
    "recommendation_id",
    "required",
    "sample_count",
    "seen",
    "source_surface",
    "status",
    "threshold",
    "timestamp",
    "type",
    "value",
}


def _raw_key(key: Any) -> bool:
    lowered = str(key).lower()
    return lowered in RAW_HEALTH_KEYS or any(
        part in lowered
        for part in (
            "api_key",
            "apikey",
            "authorization",
            "command",
            "policy_yaml",
            "prompt",
            "provider_body",
            "raw",
            "raw_request",
            "raw_response",
            "secret",
            "transcript",
        )
    )


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    return None


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            if _raw_key(key_s):
                continue
            if key_s not in _SAFE_SCALAR_KEYS and not key_s.endswith(("_at", "_count", "_rate", "_id", "_ids")):
                continue
            cleaned = _safe_metadata(item)
            if cleaned is not None:
                safe[key_s] = cleaned
        return safe
    if isinstance(value, list):
        cleaned_items = [_safe_metadata(item) for item in value[:50]]
        return [item for item in cleaned_items if item is not None]
    return _safe_scalar(value)


def strip_raw_payload_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_raw_payload_fields(item)
            for key, item in value.items()
            if not _raw_key(key)
        }
    if isinstance(value, list):
        return [strip_raw_payload_fields(item) for item in value]
    return value


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, False, "", 0):
        return []
    return [value]


def _add_rows(rows: list[dict[str, Any]], kind: str, value: Any, *, default_code: str, path: str) -> None:
    for index, item in enumerate(_as_list(value)):
        if isinstance(item, dict):
            details = _safe_metadata(item)
            code = str(item.get("code") or default_code)
            candidate_id = item.get("candidate_id") or item.get("policy_id") or item.get("recommendation_id")
        else:
            details = {"value": _safe_scalar(item)}
            code = default_code
            candidate_id = None
        if not details:
            details = {}
        row = {
            "kind": kind,
            "code": code,
            "path": f"{path}[{index}]",
            "candidate_id": str(candidate_id) if candidate_id is not None else details.get("candidate_id"),
            "details": details,
        }
        rows.append(row)


def _health_sources(bundle: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(bundle, dict):
        return []
    sources: list[tuple[str, dict[str, Any]]] = []
    recommendation = bundle.get("recommendation")
    if isinstance(recommendation, dict):
        sources.append(("$.recommendation", recommendation))
        for key in ("health", "recommendation_health", "audit", "evidence_audit"):
            value = recommendation.get(key)
            if isinstance(value, dict):
                sources.append((f"$.recommendation.{key}", value))
    for key in ("recommendation_health", "policy_recommendation_audit", "evidence_audit", "audit"):
        value = bundle.get(key)
        if isinstance(value, dict):
            sources.append((f"$.{key}", value))
    policies = bundle.get("policies")
    routing = policies.get("routing") if isinstance(policies, dict) and isinstance(policies.get("routing"), dict) else {}
    routing_recommendation = routing.get("recommendation")
    if isinstance(routing_recommendation, dict):
        sources.append(("$.policies.routing.recommendation", routing_recommendation))
    return sources


def summarize_recommendation_health(bundle: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    privacy_profiles: list[dict[str, Any]] = []
    generated_at = None
    min_samples = None
    max_error_rate = None

    for path, source in _health_sources(bundle):
        generated_at = generated_at or source.get("generated_at") or source.get("audit_generated_at")
        min_samples = min_samples if min_samples is not None else source.get("min_samples")
        max_error_rate = max_error_rate if max_error_rate is not None else source.get("max_error_rate")
        filters = source.get("filters")
        if isinstance(filters, dict):
            min_samples = min_samples if min_samples is not None else filters.get("min_samples")
            max_error_rate = max_error_rate if max_error_rate is not None else filters.get("max_error_rate")

        _add_rows(rows, "stale_evidence", source.get("stale_evidence") or source.get("stale_candidates"), default_code="stale-evidence", path=f"{path}.stale_evidence")
        _add_rows(rows, "insufficient_samples", source.get("insufficient_samples") or source.get("insufficient_sample_candidates"), default_code="insufficient-samples", path=f"{path}.insufficient_samples")
        _add_rows(rows, "threshold_failure", source.get("threshold_failures") or source.get("failed_thresholds"), default_code="threshold-failure", path=f"{path}.threshold_failures")
        _add_rows(rows, "omitted_candidate", source.get("omitted_candidates"), default_code="omitted-candidate", path=f"{path}.omitted_candidates")
        privacy = source.get("privacy_summary") or source.get("privacy_profile")
        if isinstance(privacy, dict):
            safe_privacy = _safe_metadata(privacy)
            if isinstance(safe_privacy, dict) and safe_privacy:
                privacy_profiles.append(safe_privacy)

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row["kind"], str(row.get("candidate_id")), str(row.get("details")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    counts: dict[str, int] = {}
    for row in deduped:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1

    warning_codes = sorted({HEALTH_WARNING_CODES.get(row["kind"], f"managed-recommendation-{row['kind']}") for row in deduped})
    status = "warning" if warning_codes else ("available" if _health_sources(bundle) else "missing")
    return {
        "schema": "tokenclaw.recommendation_health.v1",
        "status": status,
        "generated_at": generated_at,
        "min_samples": min_samples,
        "max_error_rate": max_error_rate,
        "warning_count": len(deduped),
        "warning_codes": warning_codes,
        "counts": counts,
        "rows": deduped,
        "privacy_profiles": privacy_profiles,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_response_bodies_included": False,
            "raw_params_included": False,
            "raw_tool_payloads_included": False,
        },
    }
