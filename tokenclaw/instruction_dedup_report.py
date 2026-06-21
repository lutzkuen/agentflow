from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from tokenclaw.limiter import model_tier
from tokenclaw.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from tokenclaw.pricing import estimate_cost
from tokenclaw.public_metadata import public_label
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.instruction_dedup_opportunity.v1"
TOKEN_CHARS = 4
MIN_SECTION_CHARS = 80

_LONG_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")
_PATH_RE = re.compile(r"(?:^|\s)(?:/|\.{1,2}/|[A-Za-z]:\\)[^\s]{2,}")
_SPACE_RE = re.compile(r"\s+")


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


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    text = public_label(key, "unknown")
    counter[text] = counter.get(text, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = _PATH_RE.sub(" <path> ", text)
    text = _LONG_HEX_RE.sub("<hex>", text)
    text = _NUMBER_RE.sub("<n>", text)
    return _SPACE_RE.sub(" ", text).strip()


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _public_digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _source_surface(provider: str, path: str) -> str:
    provider_l = provider.lower()
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        return openai_source_surface(path)
    return "unknown"


def _endpoint(provider: str, path: str) -> str:
    provider_l = provider.lower()
    if provider_l == "anthropic":
        return "messages" if "messages" in path else (path.strip("/") or "unknown")
    if provider_l == "openai":
        return openai_endpoint(path)
    return path.strip("/") or "unknown"


def _app_family(provider: str, requested_model: Any, path: str, source_surface: str) -> str:
    provider_l = provider.lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in path.lower():
        return "claude_code"
    if "codex" in model_l or source_surface == "codex_app_server":
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def _model_family(provider: str, model: Any, stored_family: Any = None) -> str:
    if stored_family:
        return str(stored_family)
    model_s = str(model or "")
    if provider.lower() == "openai":
        return openai_model_family(model_s) or "unknown"
    tier = model_tier(model_s)
    return tier if tier != "other" else "unknown"


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _status_bucket(status_code: Any) -> str:
    code = _as_int(status_code, -1)
    if code < 0:
        return "unknown"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _extract_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text(item))
        return parts
    if not isinstance(value, dict):
        return []
    block_type = str(value.get("type") or "").strip().lower()
    if block_type and block_type not in {"text", "input_text", "message", "content"}:
        return []
    parts: list[str] = []
    for key in ("text", "content", "input", "instructions"):
        if key in value:
            parts.extend(_extract_text(value.get(key)))
    return parts


def _role_instruction_texts(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    sections: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role in {"system", "developer"}:
            sections.extend(_extract_text(message.get("content")))
    return sections


def _instruction_sections_from_body(body: dict[str, Any], *, provider: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    def add(source_field: str, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        normalized = _normalize_text(text)
        if len(normalized) < MIN_SECTION_CHARS:
            return
        sections.append({
            "source_field": source_field,
            "chars": len(text),
            "normalized": normalized,
            "fingerprint": _hash_text(normalized),
        })

    provider_l = provider.lower()
    if provider_l == "anthropic":
        for text in _extract_text(body.get("system")):
            add("anthropic.system", text)
        for text in _role_instruction_texts(body.get("messages")):
            add("anthropic.messages.system_or_developer", text)
        return sections

    if provider_l == "openai":
        for text in _extract_text(body.get("instructions")):
            add("openai.instructions", text)
        for text in _role_instruction_texts(body.get("messages")):
            add("openai.messages.system_or_developer", text)
        for text in _role_instruction_texts(body.get("input")):
            add("openai.input.system_or_developer", text)
        return sections

    return sections


def _body_instruction_features(raw_request_json: Any, *, provider: str) -> dict[str, Any]:
    body = _json_obj(raw_request_json)
    if not body:
        return {
            "body_available": False,
            "fingerprint_source": "none",
            "fingerprint": None,
            "fingerprint_present": False,
            "instruction_section_count": 0,
            "instruction_section_chars": 0,
            "source_fields": [],
        }
    sections = _instruction_sections_from_body(body, provider=provider)
    section_hashes = sorted({str(section["fingerprint"]) for section in sections})
    return {
        "body_available": True,
        "fingerprint_source": "request_body_instruction_sections" if section_hashes else "none",
        "fingerprint": _hash_text("|".join(section_hashes)) if section_hashes else None,
        "fingerprint_present": bool(section_hashes),
        "instruction_section_count": len(sections),
        "instruction_section_chars": sum(_as_int(section.get("chars")) for section in sections),
        "source_fields": sorted({str(section["source_field"]) for section in sections}),
    }


_PATTERN_HASH_KEYS = {
    "instruction_fingerprint",
    "instruction_fingerprints",
    "instruction_section_fingerprint",
    "instruction_section_fingerprints",
    "instruction_section_hash",
    "instruction_section_hashes",
    "pattern_hash",
    "pattern_hashes",
    "normalized_pattern_hash",
    "crunch_pattern_hash",
}


def _collect_pattern_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            instruction_key = "instruction" in key_l and key_l.endswith(("hash", "hashes", "fingerprint", "fingerprints", "sha256"))
            pattern_key = key_l in _PATTERN_HASH_KEYS or key_l.endswith(("_pattern_hash", "_pattern_hashes", "_pattern_sha256"))
            if instruction_key or pattern_key:
                if isinstance(item, str) and item.startswith("sha256:"):
                    hashes.append(item)
                elif isinstance(item, list):
                    hashes.extend(str(child) for child in item if isinstance(child, str) and child.startswith("sha256:"))
            elif isinstance(item, (dict, list)):
                hashes.extend(_collect_pattern_hashes(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            hashes.extend(_collect_pattern_hashes(item))
    return sorted(set(hashes))


def _metadata_instruction_features(*values: Any, basis: dict[str, Any]) -> dict[str, Any]:
    pattern_hashes = _collect_pattern_hashes(values)
    if pattern_hashes:
        return {
            "fingerprint_source": "metadata_instruction_or_pattern_hash",
            "fingerprint": _hash_text("|".join(pattern_hashes)),
            "fingerprint_present": True,
            "pattern_hash_count": len(pattern_hashes),
        }
    return {
        "fingerprint_source": "metadata_shape",
        "fingerprint": None,
        "fingerprint_present": False,
        "pattern_hash_count": 0,
    }


def _workflow_phase(routing: dict[str, Any], category: str) -> str:
    for key in ("workflow_phase", "phase", "session_phase"):
        value = routing.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        unit = routing.get(key)
        if isinstance(unit, dict):
            input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
            value = unit.get("workflow_phase") or input_features.get("workflow_phase") or input_features.get("phase")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return category or "unknown"


def _row_blockers(*, body_available: bool, fingerprint_present: bool, repeated_rows: int, projected_saved_chars: int, status_code: Any) -> list[str]:
    blockers: set[str] = set()
    if not body_available:
        blockers.add("request-body-unavailable")
    if not fingerprint_present:
        blockers.add("instruction-section-fingerprint-missing")
    if repeated_rows < 2:
        blockers.add("insufficient-repeated-instruction-rows")
    if projected_saved_chars <= 0:
        blockers.add("no-instruction-dedup-savings-projected")
    if _as_int(status_code) >= 400:
        blockers.add("error-response")
    return sorted(blockers) or ["ready-for-dry-run-review"]


def _candidate_id(basis: dict[str, Any]) -> str:
    surface = str(basis.get("source_surface") or "unknown").replace("_", "-")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"instruction-dedup:{surface}:{category}:{_public_digest(basis)}"


def _new_group(public_basis: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(public_basis),
        **public_basis,
        "matched_count": 0,
        "successful_count": 0,
        "error_count": 0,
        "body_rows": 0,
        "metadata_only_rows": 0,
        "fingerprint_rows": 0,
        "pattern_hash_rows": 0,
        "pattern_hash_count": 0,
        "instruction_section_count": 0,
        "instruction_section_chars": 0,
        "estimated_input_chars": 0,
        "estimated_input_tokens": 0,
        "estimated_cost_usd": 0.0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "projected_saved_usd": 0.0,
        "status_counts": {},
        "blocker_counts": {},
        "source_field_counts": {},
        "_fingerprints": {},
        "_cost_rows": [],
    }


def _finalize_group(group: dict[str, Any], *, min_repeated_rows: int) -> dict[str, Any]:
    fingerprints = group.pop("_fingerprints", {})
    cost_rows = group.pop("_cost_rows", [])
    repeated_fingerprint_groups = sum(1 for rows in fingerprints.values() if len(rows) >= min_repeated_rows)
    repeated_fingerprint_rows = sum(len(rows) for rows in fingerprints.values() if len(rows) >= min_repeated_rows)
    projected_saved_chars = 0
    projected_saved_usd = 0.0
    for rows in fingerprints.values():
        if len(rows) < min_repeated_rows:
            continue
        ordered = sorted(rows, key=lambda row: row.get("created_at") or "")
        for row in ordered[1:]:
            projected_saved_chars += _as_int(row.get("instruction_chars") or row.get("fallback_chars"))
            projected_saved_usd += _as_float(row.get("saved_usd"))
    if projected_saved_chars <= 0 and repeated_fingerprint_rows:
        # Body-off fallback stays blocked; it only sizes the opportunity for repeated metadata fingerprints.
        projected_saved_chars = sum(max(0, _as_int(row.get("text_chars"))) // 50 for row in cost_rows)
        projected_saved_usd = sum(_as_float(row.get("saved_usd")) for row in cost_rows)

    group["projected_saved_chars"] = projected_saved_chars
    group["projected_saved_tokens"] = projected_saved_chars // TOKEN_CHARS
    group["projected_saved_usd"] = round(projected_saved_usd, 6)

    blockers = _row_blockers(
        body_available=group["body_rows"] > 0,
        fingerprint_present=group["fingerprint_rows"] > 0,
        repeated_rows=repeated_fingerprint_rows or group["matched_count"],
        projected_saved_chars=projected_saved_chars,
        status_code=200 if group["error_count"] == 0 else 500,
    )
    for blocker in blockers:
        _increment(group["blocker_counts"], blocker)

    group["repeated_fingerprint_groups"] = repeated_fingerprint_groups
    group["repeated_fingerprint_rows"] = repeated_fingerprint_rows
    group["estimated_cost_usd"] = round(_as_float(group["estimated_cost_usd"]), 6)
    group["status_breakdown"] = _breakdown(group.pop("status_counts", {}))
    group["blocker_reason_breakdown"] = _breakdown(group.pop("blocker_counts", {}))
    group["source_field_breakdown"] = _breakdown(group.pop("source_field_counts", {}))
    group["blockers"] = [item["value"] for item in group["blocker_reason_breakdown"]]
    group["instruction_section_fingerprint"] = {
        "present": bool(group["fingerprint_rows"]),
        "source": group["fingerprint_source"],
        "included": False,
        "distinct_count": len(fingerprints),
        "repeated_group_count": repeated_fingerprint_groups,
    }
    group["privacy"] = _privacy_summary()
    return group


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_instruction_text_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "terminal_output_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "thread_ids_included": False,
        "cache_keys_included": False,
        "policy_file_contents_included": False,
        "instruction_section_fingerprints_included": False,
        "pattern_hashes_included": False,
        "secrets_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _add_row_to_group(
    *,
    groups: dict[str, dict[str, Any]],
    public_basis: dict[str, Any],
    features: dict[str, Any],
    created_at: Any,
    status_code: Any,
    text_chars: int,
    input_tokens: int,
    estimated_cost: float,
    requested_model: Any,
    routed_model: Any,
    provider: str,
    source_fields: list[str],
) -> list[str]:
    group_key_basis = {**public_basis, "fingerprint_presence": bool(features.get("fingerprint_present"))}
    group_key = _candidate_id(group_key_basis)
    group = groups.setdefault(group_key, _new_group(group_key_basis))
    instruction_chars = _as_int(features.get("instruction_section_chars"))
    fallback_chars = max(0, text_chars // 50)
    saved_tokens = max(instruction_chars, fallback_chars) // TOKEN_CHARS
    saved_usd = estimate_cost(str(routed_model or requested_model or ""), saved_tokens, 0, provider=provider) or 0.0

    group["matched_count"] += 1
    group["successful_count"] += int(_as_int(status_code) < 400)
    group["error_count"] += int(_as_int(status_code) >= 400)
    group["body_rows"] += int(bool(features.get("body_available")))
    group["metadata_only_rows"] += int(not features.get("body_available"))
    group["fingerprint_rows"] += int(bool(features.get("fingerprint_present")))
    group["pattern_hash_rows"] += int(_as_int(features.get("pattern_hash_count")) > 0)
    group["pattern_hash_count"] += _as_int(features.get("pattern_hash_count"))
    group["instruction_section_count"] += _as_int(features.get("instruction_section_count"))
    group["instruction_section_chars"] += instruction_chars
    group["estimated_input_chars"] += text_chars
    group["estimated_input_tokens"] += input_tokens or max(0, text_chars // TOKEN_CHARS)
    group["estimated_cost_usd"] += estimated_cost
    _increment(group["status_counts"], _status_bucket(status_code))
    for source_field in source_fields:
        _increment(group["source_field_counts"], source_field)
    fingerprint = features.get("fingerprint")
    if fingerprint and features.get("fingerprint_present"):
        group["_fingerprints"].setdefault(str(fingerprint), []).append({
            "created_at": created_at,
            "instruction_chars": instruction_chars,
            "fallback_chars": fallback_chars,
            "text_chars": text_chars,
            "saved_usd": saved_usd,
        })
    group["_cost_rows"].append({"text_chars": text_chars, "saved_usd": saved_usd})
    row_blockers = _row_blockers(
        body_available=bool(features.get("body_available")),
        fingerprint_present=bool(features.get("fingerprint_present")),
        repeated_rows=2,
        projected_saved_chars=instruction_chars,
        status_code=status_code,
    )
    for blocker in row_blockers:
        _increment(group["blocker_counts"], blocker)
    return row_blockers


def build_instruction_dedup_opportunity_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_repeated_rows: int = 2,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    repeated_floor = max(2, min(int(min_repeated_rows or 2), 1000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, status_code,
                   input_tokens_est, actual_input_tokens, cost_est_usd, cost_baseline_usd,
                   category, crunch_json, routing_json, cache_json, request_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    codex_rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select created_at, direction, method, message_chars, params_chars,
                   input_items, input_text_chars, result_chars, error_code,
                   routing_json, crunch_json, cache_json, event_window_json, metadata_json
            from codex_app_events
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[str, dict[str, Any]] = {}
    provider_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    model_family_counts: dict[str, int] = {}
    blocker_totals: dict[str, int] = {}
    fingerprint_source_counts: dict[str, int] = {}
    body_rows = 0
    bodyless_rows = 0
    fingerprint_rows = 0
    pattern_hash_rows = 0

    for row in rows:
        provider = str(row.get("provider") or "anthropic").lower()
        if provider not in {"anthropic", "openai"}:
            continue
        path = str(row.get("path") or "")
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        source_surface = str(row.get("source_surface") or _source_surface(provider, path))
        endpoint = str(row.get("endpoint") or _endpoint(provider, path))
        requested_model = row.get("requested_model")
        routed_model = row.get("routed_model") or requested_model
        category = str(row.get("category") or routing.get("category") or "unknown")
        phase = _workflow_phase(routing, category)
        input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        text_chars = _as_int(routing.get("text_chars")) or input_tokens * TOKEN_CHARS
        stream = bool(_as_int(row.get("stream")))
        model_family = _model_family(provider, requested_model, row.get("requested_model_family"))
        tier = model_tier(str(routed_model or requested_model or ""))
        app_family = _app_family(provider, requested_model, path, source_surface)

        features = _body_instruction_features(row.get("request_json"), provider=provider)
        if features["body_available"]:
            body_rows += 1
        else:
            bodyless_rows += 1

        public_basis = {
            "provider": public_label(provider, "unknown"),
            "source_surface": public_label(source_surface, "unknown"),
            "endpoint": public_label(endpoint, "unknown"),
            "app_family": public_label(app_family, "unknown"),
            "category": public_label(category, "unknown"),
            "workflow_phase": public_label(phase, "unknown"),
            "requested_model_family": public_label(model_family, "unknown"),
            "routed_model_tier": public_label(tier, "unknown"),
            "stream": stream,
            "text_bucket": _text_bucket(text_chars),
            "fingerprint_source": features["fingerprint_source"],
            "fingerprint_present": bool(features["fingerprint_present"]),
        }
        if not features["fingerprint_present"]:
            metadata_features = _metadata_instruction_features(routing, crunch, cache, basis=public_basis)
            features.update(metadata_features)
            public_basis["fingerprint_source"] = metadata_features["fingerprint_source"]
            public_basis["fingerprint_present"] = bool(metadata_features["fingerprint_present"])
        else:
            features["pattern_hash_count"] = 0

        fingerprint_rows += int(bool(features.get("fingerprint_present")))
        pattern_hash_rows += int(_as_int(features.get("pattern_hash_count")) > 0)
        row_blockers = _add_row_to_group(
            groups=groups,
            public_basis=public_basis,
            features=features,
            created_at=row.get("created_at"),
            status_code=row.get("status_code"),
            text_chars=text_chars,
            input_tokens=input_tokens,
            estimated_cost=_as_float(row.get("cost_est_usd")) or _as_float(row.get("cost_baseline_usd")),
            requested_model=requested_model,
            routed_model=routed_model,
            provider=provider,
            source_fields=[str(item) for item in features.get("source_fields") or []],
        )
        for blocker in row_blockers:
            _increment(blocker_totals, blocker)
        _increment(provider_counts, provider)
        _increment(surface_counts, source_surface)
        _increment(category_counts, category)
        _increment(phase_counts, phase)
        _increment(model_family_counts, model_family)
        _increment(fingerprint_source_counts, public_basis["fingerprint_source"])

    for row in codex_rows:
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        event_window = _json_obj(row.get("event_window_json"))
        metadata = _json_obj(row.get("metadata_json"))
        method = str(row.get("method") or "unknown")
        direction = str(row.get("direction") or "unknown")
        category = str(routing.get("category") or event_window.get("category") or method.replace("/", "_") or "unknown")
        phase = str(routing.get("workflow_phase") or event_window.get("workflow_phase") or category)
        text_chars = _as_int(row.get("input_text_chars")) or _as_int(row.get("params_chars")) or _as_int(row.get("message_chars"))
        status_code = 500 if row.get("error_code") is not None else 200
        source_surface = "codex_app_server"
        features = {
            "body_available": False,
            "fingerprint_source": "none",
            "fingerprint": None,
            "fingerprint_present": False,
            "instruction_section_count": 0,
            "instruction_section_chars": 0,
            "source_fields": [],
        }
        public_basis = {
            "provider": "codex_app",
            "source_surface": source_surface,
            "endpoint": "app_server",
            "app_family": "codex",
            "category": public_label(category, "unknown"),
            "workflow_phase": public_label(phase, "unknown"),
            "requested_model_family": public_label(routing.get("requested_model_family") or "codex", "unknown"),
            "routed_model_tier": public_label(routing.get("routed_model_tier") or "codex", "unknown"),
            "stream": False,
            "text_bucket": _text_bucket(text_chars),
            "direction": public_label(direction, "unknown"),
            "method": public_label(method, "unknown"),
            "fingerprint_source": features["fingerprint_source"],
            "fingerprint_present": False,
        }
        metadata_features = _metadata_instruction_features(routing, crunch, cache, event_window, metadata, basis=public_basis)
        features.update(metadata_features)
        public_basis["fingerprint_source"] = metadata_features["fingerprint_source"]
        public_basis["fingerprint_present"] = bool(metadata_features["fingerprint_present"])

        bodyless_rows += 1
        fingerprint_rows += int(bool(features.get("fingerprint_present")))
        pattern_hash_rows += int(_as_int(features.get("pattern_hash_count")) > 0)
        row_blockers = _add_row_to_group(
            groups=groups,
            public_basis=public_basis,
            features=features,
            created_at=row.get("created_at"),
            status_code=status_code,
            text_chars=text_chars,
            input_tokens=max(0, text_chars // TOKEN_CHARS),
            estimated_cost=0.0,
            requested_model="codex-app",
            routed_model="codex-app",
            provider="openai",
            source_fields=[],
        )
        for blocker in row_blockers:
            _increment(blocker_totals, blocker)
        _increment(provider_counts, "codex_app")
        _increment(surface_counts, source_surface)
        _increment(category_counts, category)
        _increment(phase_counts, phase)
        _increment(model_family_counts, public_basis["requested_model_family"])
        _increment(fingerprint_source_counts, public_basis["fingerprint_source"])

    candidates = [_finalize_group(group, min_repeated_rows=repeated_floor) for group in groups.values()]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("projected_saved_chars")),
            _as_int(item.get("repeated_fingerprint_rows")),
            _as_int(item.get("matched_count")),
        ),
        reverse=True,
    )

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "limit": capped_limit,
        "min_repeated_rows": repeated_floor,
        "summary": {
            "scanned_provider_call_count": len(rows),
            "scanned_codex_event_count": len(codex_rows),
            "candidate_count": len(candidates),
            "matched_count": sum(_as_int(item.get("matched_count")) for item in candidates),
            "body_rows": body_rows,
            "body_logging_off_rows": bodyless_rows,
            "instruction_fingerprint_rows": fingerprint_rows,
            "metadata_pattern_hash_rows": pattern_hash_rows,
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in candidates),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in candidates),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in candidates), 6),
        },
        "projection_policy": {
            "schema": "tokenclaw.instruction_dedup_projection_policy.v1",
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "raw_body_required": False,
            "default_apply": False,
            "method": "body-on rows hash local instruction-bearing fields only; body-off rows use metadata instruction/pattern hashes or blocked shape buckets",
            "body_off_projection": "2pct input-character fallback only for repeated metadata fingerprints; body-off candidates remain blocked",
        },
        "provider_breakdown": _breakdown(provider_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "category_breakdown": _breakdown(category_counts),
        "workflow_phase_breakdown": _breakdown(phase_counts),
        "requested_model_family_breakdown": _breakdown(model_family_counts),
        "fingerprint_source_breakdown": _breakdown(fingerprint_source_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "candidates": candidates,
        "privacy": {
            **_privacy_summary(),
            "basis": "local call metadata, optional local-only request_json instruction-field scanning, and sanitized Codex app telemetry; raw text and hashes are never emitted",
        },
    }
