from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from tokenclaw.limiter import model_tier
from tokenclaw.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from tokenclaw.pricing import estimate_cost
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.repeated_scaffold_opportunity.v1"
TOKEN_CHARS = 4
MIN_LINE_CHARS = 48
MIN_SEGMENT_CHARS = 220

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
    text = str(key or "unknown")
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


def _source_surface(provider: str, path: str) -> str:
    provider_l = provider.lower()
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        return openai_source_surface(path)
    return "unknown"


def _endpoint(provider: str, path: str) -> str:
    if provider.lower() == "anthropic":
        return "messages" if "messages" in path else (path.strip("/") or "unknown")
    if provider.lower() == "openai":
        return openai_endpoint(path)
    return path.strip("/") or "unknown"


def _app_family(provider: str, requested_model: Any, path: str) -> str:
    provider_l = provider.lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in path.lower():
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
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
    block_type = str(value.get("type") or "")
    if block_type and block_type not in {"text", "input_text", "message", "content"}:
        # Tool payloads and binary/file blocks are intentionally excluded.
        return []
    parts = []
    for key in ("text", "content", "input", "system", "instructions"):
        if key in value:
            parts.extend(_extract_text(value.get(key)))
    return parts


def _provider_text_segments(body: dict[str, Any]) -> list[str]:
    segments: list[str] = []
    for key in ("system", "instructions"):
        if key in body:
            segments.extend(_extract_text(body.get(key)))
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                segments.extend(_extract_text(message.get("content")))
    if "input" in body:
        segments.extend(_extract_text(body.get("input")))
    return [segment for segment in segments if isinstance(segment, str) and segment.strip()]


def _body_scaffold_features(raw_request_json: Any) -> dict[str, Any]:
    body = _json_obj(raw_request_json)
    if not body:
        return {
            "body_available": False,
            "fingerprint_source": "none",
            "fingerprint": None,
            "fingerprint_present": False,
            "repeated_scaffold_chars": 0,
            "repeated_line_count": 0,
            "segment_count": 0,
            "normalized_line_count": 0,
        }
    segments = _provider_text_segments(body)
    segment_counts: dict[str, dict[str, int]] = {}
    line_counts: dict[str, dict[str, int]] = {}
    ordered_hashes: list[str] = []
    for segment in segments:
        normalized_segment = _normalize_text(segment)
        if len(normalized_segment) >= MIN_SEGMENT_CHARS:
            digest = _hash_text(normalized_segment)
            bucket = segment_counts.setdefault(digest, {"count": 0, "chars": len(segment)})
            bucket["count"] += 1
            bucket["chars"] = max(bucket["chars"], len(segment))
            ordered_hashes.append(digest)
        for line in segment.splitlines():
            normalized_line = _normalize_text(line)
            if len(normalized_line) < MIN_LINE_CHARS:
                continue
            digest = _hash_text(normalized_line)
            bucket = line_counts.setdefault(digest, {"count": 0, "chars": len(line)})
            bucket["count"] += 1
            bucket["chars"] = max(bucket["chars"], len(line))
            ordered_hashes.append(digest)

    repeated_segment_chars = sum((item["count"] - 1) * item["chars"] for item in segment_counts.values() if item["count"] > 1)
    repeated_line_chars = sum((item["count"] - 1) * item["chars"] for item in line_counts.values() if item["count"] > 1)
    repeated_line_count = sum(item["count"] - 1 for item in line_counts.values() if item["count"] > 1)
    unique_hashes = sorted(set(ordered_hashes))
    fingerprint = _hash_text("|".join(unique_hashes[:64])) if unique_hashes else None
    return {
        "body_available": True,
        "fingerprint_source": "request_body" if fingerprint else "none",
        "fingerprint": fingerprint,
        "fingerprint_present": bool(fingerprint),
        "repeated_scaffold_chars": max(repeated_segment_chars, repeated_line_chars),
        "repeated_line_count": repeated_line_count,
        "segment_count": len(segments),
        "normalized_line_count": len(line_counts),
    }


_PATTERN_HASH_KEYS = {
    "pattern_hash",
    "normalized_pattern_hash",
    "crunch_pattern_hash",
    "cache_pattern_hash",
    "pattern_hashes",
    "hashes",
}


def _collect_pattern_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in _PATTERN_HASH_KEYS or key_l.endswith(("_pattern_hash", "_pattern_hashes", "_pattern_sha256")):
                if isinstance(item, str) and item.startswith("sha256:"):
                    hashes.append(item)
                elif isinstance(item, list):
                    hashes.extend(str(child) for child in item if isinstance(child, str) and child.startswith("sha256:"))
            elif isinstance(item, (dict, list)):
                hashes.extend(_collect_pattern_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(_collect_pattern_hashes(item))
    return sorted(set(hashes))


def _metadata_scaffold_features(routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any], basis: dict[str, Any]) -> dict[str, Any]:
    pattern_hashes = _collect_pattern_hashes({"routing": routing, "crunch": crunch, "cache": cache})
    if pattern_hashes:
        return {
            "fingerprint_source": "metadata_pattern_hash",
            "fingerprint": _hash_text("|".join(pattern_hashes)),
            "fingerprint_present": True,
            "pattern_hash_count": len(pattern_hashes),
        }
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    return {
        "fingerprint_source": "metadata_shape",
        "fingerprint": _hash_text(raw),
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


def _row_blockers(*, body_available: bool, fingerprint_present: bool, repeated_chars: int, count: int, status_code: Any, category: str) -> list[str]:
    blockers: set[str] = set()
    if not body_available:
        blockers.add("request-body-unavailable")
    if not fingerprint_present:
        blockers.add("normalized-scaffold-fingerprint-missing")
    if count < 2:
        blockers.add("insufficient-repeated-rows")
    if repeated_chars <= 0:
        blockers.add("no-intra-request-repeat-measured")
    if _as_int(status_code) >= 400:
        blockers.add("error-response")
    if str(category or "").startswith("tool"):
        blockers.add("tool-protocol-safety-review-required")
    return sorted(blockers) or ["ready-for-canary-review"]


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    surface = str(basis.get("source_surface") or "unknown").replace("_", "-")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"repeated-scaffold:{surface}:{category}:{digest}"


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
        "has_tools_count": 0,
        "estimated_input_chars": 0,
        "estimated_input_tokens": 0,
        "estimated_cost_usd": 0.0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "projected_saved_usd": 0.0,
        "repeated_line_count": 0,
        "status_counts": {},
        "blocker_counts": {},
        "_fingerprints": {},
        "_cost_rows": [],
    }


def _finalize_group(group: dict[str, Any], *, min_repeated_rows: int) -> dict[str, Any]:
    fingerprints = group.pop("_fingerprints", {})
    cost_rows = group.pop("_cost_rows", [])
    repeated_fingerprint_groups = sum(1 for rows in fingerprints.values() if len(rows) >= min_repeated_rows)
    repeated_fingerprint_rows = sum(len(rows) for rows in fingerprints.values() if len(rows) >= min_repeated_rows)

    if group["projected_saved_chars"] <= 0 and repeated_fingerprint_rows:
        # Body-off fallback: estimate a small, explicit fraction for repeated metadata shapes.
        group["projected_saved_chars"] = sum(max(0, _as_int(row.get("text_chars"))) // 50 for row in cost_rows)
        group["projected_saved_tokens"] = group["projected_saved_chars"] // TOKEN_CHARS
        group["projected_saved_usd"] = sum(_as_float(row.get("saved_usd")) for row in cost_rows)

    blockers = _row_blockers(
        body_available=group["body_rows"] > 0,
        fingerprint_present=group["fingerprint_rows"] > 0,
        repeated_chars=group["projected_saved_chars"],
        count=repeated_fingerprint_rows or group["matched_count"],
        status_code=200 if group["error_count"] == 0 else 500,
        category=str(group.get("category") or ""),
    )
    for blocker in blockers:
        _increment(group["blocker_counts"], blocker)

    group["repeated_fingerprint_groups"] = repeated_fingerprint_groups
    group["repeated_fingerprint_rows"] = repeated_fingerprint_rows
    group["estimated_cost_usd"] = round(_as_float(group["estimated_cost_usd"]), 6)
    group["projected_saved_usd"] = round(_as_float(group["projected_saved_usd"]), 6)
    group["status_breakdown"] = _breakdown(group.pop("status_counts", {}))
    group["blocker_reason_breakdown"] = _breakdown(group.pop("blocker_counts", {}))
    group["blockers"] = [item["value"] for item in group["blocker_reason_breakdown"]]
    group["normalized_scaffold_fingerprint"] = {
        "present": bool(group["fingerprint_rows"]),
        "source": group["fingerprint_source"],
        "included": False,
        "distinct_count": len(fingerprints),
        "repeated_group_count": repeated_fingerprint_groups,
    }
    group["privacy"] = {
        "metadata_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_responses_included": False,
        "normalized_scaffold_fingerprints_included": False,
        "pattern_hashes_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
    }
    return group


def build_repeated_scaffold_opportunity_report(
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

    groups: dict[str, dict[str, Any]] = {}
    provider_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    blocker_totals: dict[str, int] = {}
    fingerprint_source_counts: dict[str, int] = {}
    scanned_provider_rows = 0
    body_rows = 0
    bodyless_rows = 0
    fingerprint_rows = 0
    pattern_hash_rows = 0

    for row in rows:
        provider = str(row.get("provider") or "anthropic").lower()
        if provider not in {"anthropic", "openai"}:
            continue
        scanned_provider_rows += 1
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
        has_tools = bool(routing.get("has_tools") or category.startswith("tool"))
        model_family = _model_family(provider, requested_model, row.get("requested_model_family"))
        tier = model_tier(str(routed_model or requested_model or ""))

        body_features = _body_scaffold_features(row.get("request_json"))
        if body_features["body_available"]:
            body_rows += 1
        else:
            bodyless_rows += 1

        public_basis = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "app_family": _app_family(provider, requested_model, path),
            "category": category,
            "workflow_phase": phase,
            "requested_model_family": model_family,
            "routed_model_tier": tier,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": _text_bucket(text_chars),
            "fingerprint_source": body_features["fingerprint_source"],
            "fingerprint_present": bool(body_features["fingerprint_present"]),
        }

        if not body_features["fingerprint_present"]:
            metadata_features = _metadata_scaffold_features(routing, crunch, cache, public_basis)
            body_features.update(metadata_features)
            public_basis["fingerprint_source"] = metadata_features["fingerprint_source"]
            public_basis["fingerprint_present"] = bool(metadata_features["fingerprint_present"])
        else:
            body_features["pattern_hash_count"] = 0

        if body_features["fingerprint_present"]:
            fingerprint_rows += 1
        if _as_int(body_features.get("pattern_hash_count")):
            pattern_hash_rows += 1

        group_key_basis = {
            **public_basis,
            "fingerprint_presence": bool(body_features["fingerprint_present"]),
        }
        group_key = _candidate_id(group_key_basis)
        group = groups.setdefault(group_key, _new_group(group_key_basis))

        repeated_chars = _as_int(body_features.get("repeated_scaffold_chars"))
        saved_tokens = repeated_chars // TOKEN_CHARS
        saved_usd = estimate_cost(str(routed_model or requested_model or ""), saved_tokens, 0, provider=provider) or 0.0
        if repeated_chars <= 0 and not body_features["body_available"]:
            fallback_saved_tokens = max(0, text_chars // TOKEN_CHARS // 50)
            saved_usd = estimate_cost(str(routed_model or requested_model or ""), fallback_saved_tokens, 0, provider=provider) or 0.0

        group["matched_count"] += 1
        group["successful_count"] += int(_as_int(row.get("status_code")) < 400)
        group["error_count"] += int(_as_int(row.get("status_code")) >= 400)
        group["body_rows"] += int(bool(body_features["body_available"]))
        group["metadata_only_rows"] += int(not body_features["body_available"])
        group["fingerprint_rows"] += int(bool(body_features["fingerprint_present"]))
        group["pattern_hash_rows"] += int(_as_int(body_features.get("pattern_hash_count")) > 0)
        group["pattern_hash_count"] += _as_int(body_features.get("pattern_hash_count"))
        group["has_tools_count"] += int(has_tools)
        group["estimated_input_chars"] += text_chars
        group["estimated_input_tokens"] += input_tokens or max(0, text_chars // TOKEN_CHARS)
        group["estimated_cost_usd"] += _as_float(row.get("cost_est_usd")) or _as_float(row.get("cost_baseline_usd"))
        group["projected_saved_chars"] += repeated_chars
        group["projected_saved_tokens"] += saved_tokens
        group["projected_saved_usd"] += saved_usd if repeated_chars > 0 else 0.0
        group["repeated_line_count"] += _as_int(body_features.get("repeated_line_count"))
        _increment(group["status_counts"], _status_bucket(row.get("status_code")))
        fingerprint = body_features.get("fingerprint")
        if fingerprint:
            group["_fingerprints"].setdefault(str(fingerprint), []).append({"text_chars": text_chars, "saved_usd": saved_usd})
        group["_cost_rows"].append({"text_chars": text_chars, "saved_usd": saved_usd})

        row_blockers = _row_blockers(
            body_available=bool(body_features["body_available"]),
            fingerprint_present=bool(body_features["fingerprint_present"]),
            repeated_chars=repeated_chars,
            count=repeated_floor,
            status_code=row.get("status_code"),
            category=category,
        )
        for blocker in row_blockers:
            _increment(group["blocker_counts"], blocker)
            _increment(blocker_totals, blocker)
        _increment(provider_counts, provider)
        _increment(surface_counts, source_surface)
        _increment(category_counts, category)
        _increment(phase_counts, phase)
        _increment(fingerprint_source_counts, body_features["fingerprint_source"])

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
        "limit": capped_limit,
        "min_repeated_rows": repeated_floor,
        "summary": {
            "scanned_call_count": len(rows),
            "provider_call_count": scanned_provider_rows,
            "candidate_count": len(candidates),
            "matched_count": sum(_as_int(item.get("matched_count")) for item in candidates),
            "body_rows": body_rows,
            "body_logging_off_rows": bodyless_rows,
            "normalized_scaffold_fingerprint_rows": fingerprint_rows,
            "metadata_pattern_hash_rows": pattern_hash_rows,
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in candidates),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in candidates),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in candidates), 6),
        },
        "projection_policy": {
            "schema": "tokenclaw.repeated_scaffold_projection_policy.v1",
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "raw_body_required": False,
            "default_apply": False,
            "method": "body-on rows measure repeated normalized provider-message text locally; body-off rows use metadata pattern hashes or repeated shape buckets with explicit blockers",
            "body_off_projection": "2pct input-token estimate only for repeated metadata groups; reported as blocked until body evidence or pattern review exists",
        },
        "provider_breakdown": _breakdown(provider_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "category_breakdown": _breakdown(category_counts),
        "workflow_phase_breakdown": _breakdown(phase_counts),
        "fingerprint_source_breakdown": _breakdown(fingerprint_source_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "normalized_scaffold_fingerprints_included": False,
            "pattern_hashes_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local calls metadata plus optional local-only request_json scanning; raw text and hashes are never emitted",
        },
    }
