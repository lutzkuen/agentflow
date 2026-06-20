from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

SCHEMA = "agentflow.session_phase_memory.v1"
TOKEN_CHARS = 4
DEFAULT_LIMIT = 1000
DEFAULT_WINDOW_SIZE = 20
DEFAULT_MIN_PLATEAU_CHARS = 8_000
DEFAULT_MAX_PLATEAU_DELTA_RATIO = 0.03
DEFAULT_PLATEAU_PAIRS_FOR_CLASSIFICATION = 2
MODEL_FAMILY_RANK = {"unknown": 0, "other": 0, "haiku": 1, "gpt": 1, "sonnet": 2, "opus": 3}
SAFE_PHASES = {"planning", "tool-execution", "verification", "summary", "thinking", "unknown"}
SAFE_CATEGORIES = {
    "chat",
    "code-gen",
    "long-context",
    "short-completion",
    "summary",
    "thinking",
    "tool-heavy",
    "tool-light",
    "tool-result",
    "unknown",
}
SAFE_SOURCE_SURFACES = {
    "anthropic_messages",
    "codex_app",
    "codex_turn",
    "openai_chat",
    "openai_responses",
    "unknown",
}
SAFE_CACHE_STATUSES = {"bypass", "disabled", "error", "hit", "miss", "skipped", "unknown"}
SAFE_CACHE_REASONS = {
    "cache-disabled",
    "disabled",
    "exact-hit",
    "exact-match",
    "exact-miss",
    "legacy-cache-hit",
    "legacy-exact-miss",
    "legacy-streaming",
    "legacy-unknown",
    "missing-cache-json",
    "semantic-match",
    "streaming",
    "tool-call-cache-disabled",
    "unknown",
}


def _json_obj(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_hash(session_id: Any) -> str | None:
    if session_id is None or str(session_id).strip() == "":
        return None
    digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _model_family(model: Any) -> str:
    value = str(model or "").lower()
    if "haiku" in value:
        return "haiku"
    if "sonnet" in value:
        return "sonnet"
    if "opus" in value:
        return "opus"
    if value.startswith("gpt-"):
        return "gpt"
    if not value:
        return "unknown"
    return "other"


def _safe_label(value: Any, allowed: set[str], *, default: str = "unknown") -> str:
    label = str(value or "").strip().lower().replace("_", "-")
    if not label:
        return default
    return label if label in allowed else default


def _text_bucket(chars: int) -> str:
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _token_bucket(tokens: int) -> str:
    if tokens < 1_000:
        return "lt_1k_tokens"
    if tokens < 4_000:
        return "1k_4k_tokens"
    if tokens < 16_000:
        return "4k_16k_tokens"
    if tokens < 64_000:
        return "16k_64k_tokens"
    return "gte_64k_tokens"


def _cost_bucket(cost: float) -> str:
    if cost <= 0:
        return "zero"
    if cost < 0.01:
        return "lt_1c"
    if cost < 0.10:
        return "1c_10c"
    if cost < 1.0:
        return "10c_1usd"
    if cost < 5.0:
        return "1_5usd"
    return "gte_5usd"


def _savings_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "none"
    if tokens < 1_000:
        return "lt_1k_tokens"
    if tokens < 10_000:
        return "1k_10k_tokens"
    if tokens < 100_000:
        return "10k_100k_tokens"
    return "gte_100k_tokens"


def _model_family_floor(*counters: Counter[str]) -> str:
    floor = "unknown"
    floor_rank = 0
    for counter in counters:
        for family in counter:
            rank = MODEL_FAMILY_RANK.get(family, 0)
            if rank > floor_rank:
                floor = family
                floor_rank = rank
    return floor


def _status_bucket(status_code: int) -> str:
    if status_code <= 0:
        return "unknown"
    if status_code < 400:
        return "ok"
    if status_code == 429:
        return "rate_limited"
    if status_code == 529:
        return "overloaded"
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "auth_error"
    if status_code >= 500:
        return "server_error"
    return f"http_{status_code}"


def _derive_phase(row: dict[str, Any], routing: dict[str, Any]) -> tuple[str, str]:
    explicit = str(routing.get("workflow_phase") or "").strip().lower()
    explicit_map = {
        "tool-result": "tool-execution",
        "tool_execution": "tool-execution",
        "tool-execution": "tool-execution",
        "summary": "summary",
        "planning": "planning",
        "verification": "verification",
        "thinking": "thinking",
        "chat": "unknown",
    }
    if explicit:
        return _safe_label(explicit_map.get(explicit, explicit), SAFE_PHASES), "routing_json.workflow_phase"

    reason = str(routing.get("reason") or "").lower()
    category = _safe_label(row.get("category") or routing.get("category"), SAFE_CATEGORIES)
    if "thinking" in reason:
        return "thinking", "routing_reason"
    if category == "tool-result":
        return "tool-execution", "category"
    if category in {"short-completion", "summary"}:
        return "summary", "category"
    if category in {"code-gen", "tool-heavy", "tool-light"}:
        return "verification", "category"
    if category == "long-context":
        return "planning", "category"
    return "unknown", "fallback"


def _dominant(counter: Counter[str]) -> str:
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _breakdown(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _row_features(row: dict[str, Any]) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    output_tokens = _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))
    text_chars = _as_int(routing.get("text_chars")) or input_tokens * TOKEN_CHARS
    phase, phase_source = _derive_phase(row, routing)
    category = _safe_label(row.get("category") or routing.get("category"), SAFE_CATEGORIES)
    status_code = _as_int(row.get("status_code"))
    tokens_saved = _as_int(crunch.get("tokens_saved_est"))
    if not tokens_saved:
        tokens_saved = max(0, _as_int(crunch.get("saved_chars")) // TOKEN_CHARS)
    return {
        "created_at": row.get("created_at"),
        "session_id": row.get("session_id"),
        "provider": str(row.get("provider") or "anthropic"),
        "source_surface": _safe_label(row.get("source_surface"), SAFE_SOURCE_SURFACES)
        if row.get("source_surface")
        else _source_surface(row),
        "app_family": _app_family(row),
        "phase": phase,
        "phase_source": phase_source,
        "category": category,
        "requested_model_family": _model_family(row.get("requested_model")),
        "routed_model_family": _model_family(row.get("routed_model") or row.get("requested_model")),
        "text_chars": text_chars,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "text_bucket": _text_bucket(text_chars),
        "token_bucket": _token_bucket(input_tokens),
        "status_bucket": _status_bucket(status_code),
        "retry_count": _as_int(row.get("retry_count")),
        "fallback": bool(routing.get("fallback_reason")),
        "cache_status": _safe_label(cache.get("status"), SAFE_CACHE_STATUSES),
        "cache_reason": _safe_label(cache.get("reason"), SAFE_CACHE_REASONS, default="other"),
        "crunch_changed": bool(crunch.get("changed") or crunch.get("applied")),
        "tokens_saved_est": tokens_saved,
        "cost_est_usd": _as_float(row.get("cost_est_usd")),
    }


def _source_surface(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "").lower()
    path = str(row.get("path") or "")
    if provider == "openai" and "chat/completions" in path:
        return "openai_chat"
    if provider == "openai":
        return "openai_responses"
    return "anthropic_messages"


def _app_family(row: dict[str, Any]) -> str:
    model = str(row.get("requested_model") or "").lower()
    path = str(row.get("path") or "")
    if "codex" in model:
        return "codex"
    if "/v1/messages" in path:
        return "claude_code"
    if str(row.get("provider") or "").lower() == "openai":
        return "generic_openai"
    return "unknown"


def _memory_blockers(features: list[dict[str, Any]], plateau_pairs: int, *, plateau_pairs_for_classification: int) -> list[str]:
    blockers: list[str] = []
    if not features:
        return ["no_recent_calls"]
    if any(feature["status_bucket"] != "ok" for feature in features):
        blockers.append("recent_errors")
    if any(feature["retry_count"] > 0 for feature in features):
        blockers.append("recent_retries")
    if any(feature["fallback"] for feature in features):
        blockers.append("recent_routing_fallback")
    if any(feature["phase"] == "thinking" for feature in features):
        blockers.append("thinking_phase_present")
    if plateau_pairs >= plateau_pairs_for_classification:
        blockers.append("context_plateau_active")
    if len(features) < 3:
        blockers.append("small_sample")
    return blockers


def _session_memory(
    key: str,
    features: list[dict[str, Any]],
    *,
    window_size: int,
    min_plateau_chars: int,
    max_plateau_delta_ratio: float,
    plateau_pairs_for_classification: int,
) -> dict[str, Any]:
    recent = features[-window_size:]
    phase_counts = Counter(feature["phase"] for feature in recent)
    category_counts = Counter(feature["category"] for feature in recent)
    source_counts = Counter(feature["source_surface"] for feature in recent)
    app_counts = Counter(feature["app_family"] for feature in recent)
    requested_counts = Counter(feature["requested_model_family"] for feature in recent)
    routed_counts = Counter(feature["routed_model_family"] for feature in recent)
    text_buckets = Counter(feature["text_bucket"] for feature in recent)
    token_buckets = Counter(feature["token_bucket"] for feature in recent)
    status_buckets = Counter(feature["status_bucket"] for feature in recent)
    cache_statuses = Counter(feature["cache_status"] for feature in recent)
    cache_reasons = Counter(feature["cache_reason"] for feature in recent)
    phase_sources = Counter(feature["phase_source"] for feature in recent)

    plateau_pairs = 0
    previous_text_chars: int | None = None
    for feature in recent:
        text_chars = int(feature["text_chars"])
        if (
            previous_text_chars is not None
            and previous_text_chars >= min_plateau_chars
            and text_chars >= min_plateau_chars
            and abs(text_chars - previous_text_chars) / max(previous_text_chars, 1) <= max_plateau_delta_ratio
        ):
            plateau_pairs += 1
        previous_text_chars = text_chars

    dominant_phase = _dominant(phase_counts)
    classifications = [dominant_phase]
    if plateau_pairs >= plateau_pairs_for_classification:
        classifications.append("plateau")

    total_cost = sum(float(feature["cost_est_usd"]) for feature in recent)
    tokens_saved = sum(int(feature["tokens_saved_est"]) for feature in recent)
    retry_count = sum(int(feature["retry_count"]) for feature in recent)
    error_count = sum(1 for feature in recent if feature["status_bucket"] != "ok")
    fallback_count = sum(1 for feature in recent if feature["fallback"])
    crunch_changed_count = sum(1 for feature in recent if feature["crunch_changed"])
    raw_session_hash = _session_hash(key) if key != "__missing_session__" else None
    session_key = raw_session_hash or "missing-session"
    dominant_phase_count = phase_counts.get(dominant_phase, 0)
    phase_stability = round(dominant_phase_count / len(recent), 4) if recent else 0.0
    blocker_reasons = _memory_blockers(
        recent,
        plateau_pairs,
        plateau_pairs_for_classification=plateau_pairs_for_classification,
    )

    return {
        "session_key": session_key,
        "session_key_kind": "sha256_session_id" if raw_session_hash else "missing_session",
        "raw_session_id_included": False,
        "window": {
            "call_count": len(recent),
            "first_seen_at": recent[0]["created_at"] if recent else None,
            "last_seen_at": recent[-1]["created_at"] if recent else None,
            "window_size": window_size,
        },
        "source_surface": _dominant(source_counts),
        "source_surface_counts": _breakdown(source_counts),
        "app_family": _dominant(app_counts),
        "readiness": "ready" if recent and not blocker_reasons else "blocked",
        "dominant_phase": dominant_phase,
        "dominant_phase_count": dominant_phase_count,
        "phase_stability": phase_stability,
        "classifications": classifications,
        "phase_counts": _breakdown(phase_counts),
        "phase_source_counts": _breakdown(phase_sources),
        "category_counts": _breakdown(category_counts),
        "requested_model_family_counts": _breakdown(requested_counts),
        "routed_model_family_counts": _breakdown(routed_counts),
        "model_family_floor": _model_family_floor(requested_counts, routed_counts),
        "text_bucket_counts": _breakdown(text_buckets),
        "token_bucket_counts": _breakdown(token_buckets),
        "status_bucket_counts": _breakdown(status_buckets),
        "cache_status_counts": _breakdown(cache_statuses),
        "cache_reason_counts": _breakdown(cache_reasons),
        "retry_count": retry_count,
        "retry_rate": round(retry_count / len(recent), 4) if recent else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(recent), 4) if recent else 0.0,
        "error_count": error_count,
        "error_rate": round(error_count / len(recent), 4) if recent else 0.0,
        "crunch_changed_count": crunch_changed_count,
        "crunch_savings_bucket": _savings_bucket(tokens_saved),
        "cost_bucket": _cost_bucket(total_cost),
        "projected_savings_bucket": _savings_bucket(tokens_saved),
        "context_plateau": {
            "active": plateau_pairs >= plateau_pairs_for_classification,
            "pairs": plateau_pairs,
            "min_text_chars": min_plateau_chars,
            "max_delta_ratio": max_plateau_delta_ratio,
        },
        "blocker_reasons": blocker_reasons,
    }


def build_session_phase_memory(
    store_obj: Any,
    *,
    limit: int = DEFAULT_LIMIT,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_plateau_chars: int = DEFAULT_MIN_PLATEAU_CHARS,
    max_plateau_delta_ratio: float = DEFAULT_MAX_PLATEAU_DELTA_RATIO,
    plateau_pairs_for_classification: int = DEFAULT_PLATEAU_PAIRS_FOR_CLASSIFICATION,
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 10_000))
    window_size = max(1, min(int(window_size), 200))
    rows = store_obj.conn.execute(
        """
        SELECT created_at,
               session_id,
               coalesce(provider, 'anthropic') as provider,
               source_surface,
               path,
               requested_model,
               routed_model,
               stream,
               status_code,
               input_tokens_est,
               output_tokens_est,
               actual_input_tokens,
               actual_output_tokens,
               cost_est_usd,
               cost_baseline_usd,
               crunch_json,
               routing_json,
               cache_json,
               category,
               cache_creation_input_tokens,
               cache_read_input_tokens,
               retry_count,
               thinking_output_tokens
        FROM calls
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    newest_first = [dict(row) for row in rows]
    chronological = list(reversed(newest_first))
    by_session: dict[str, list[dict[str, Any]]] = {}
    unknown_session_count = 0
    for row in chronological:
        key = str(row.get("session_id") or "").strip()
        if not key:
            key = "__missing_session__"
            unknown_session_count += 1
        by_session.setdefault(key, []).append(_row_features(row))

    memories = [
        _session_memory(
            key,
            features,
            window_size=window_size,
            min_plateau_chars=min_plateau_chars,
            max_plateau_delta_ratio=max_plateau_delta_ratio,
            plateau_pairs_for_classification=plateau_pairs_for_classification,
        )
        for key, features in by_session.items()
    ]
    memories.sort(
        key=lambda item: (
            item["window"]["last_seen_at"] or "",
            item["window"]["call_count"],
        ),
        reverse=True,
    )
    return {
        "schema": SCHEMA,
        "generated_at": _generated_at(),
        "lookback": {
            "row_limit": limit,
            "window_size": window_size,
            "sampled_call_count": len(chronological),
        },
        "summary": {
            "session_count": len(memories),
            "hashed_session_count": sum(1 for item in memories if item["session_key_kind"] == "sha256_session_id"),
            "unknown_session_call_count": unknown_session_count,
            "plateau_session_count": sum(1 for item in memories if item["context_plateau"]["active"]),
            "memory_ready_session_count": sum(1 for item in memories if item["readiness"] == "ready"),
            "blocked_session_count": sum(1 for item in memories if item["readiness"] == "blocked"),
            "dominant_phase_counts": _breakdown(Counter(item["dominant_phase"] for item in memories)),
            "blocker_counts": _breakdown(
                Counter(reason for item in memories for reason in item["blocker_reasons"])
            ),
        },
        "sessions": memories,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "session_ids_hashed": True,
            "request_json_read": False,
            "response_json_read": False,
            "error_text_included": False,
        },
    }


def build_session_phase_memory_for_session(
    store_obj: Any,
    session_id: str,
    *,
    limit: int = DEFAULT_WINDOW_SIZE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_plateau_chars: int = DEFAULT_MIN_PLATEAU_CHARS,
    max_plateau_delta_ratio: float = DEFAULT_MAX_PLATEAU_DELTA_RATIO,
    plateau_pairs_for_classification: int = DEFAULT_PLATEAU_PAIRS_FOR_CLASSIFICATION,
) -> dict[str, Any] | None:
    """Build the metadata-only recent phase memory for one raw local session id."""
    session_id = str(session_id or "").strip()
    if not session_id:
        return None
    limit = max(1, min(int(limit), 10_000))
    window_size = max(1, min(int(window_size), 200))
    rows = store_obj.conn.execute(
        """
        SELECT created_at,
               session_id,
               coalesce(provider, 'anthropic') as provider,
               source_surface,
               path,
               requested_model,
               routed_model,
               stream,
               status_code,
               input_tokens_est,
               output_tokens_est,
               actual_input_tokens,
               actual_output_tokens,
               cost_est_usd,
               cost_baseline_usd,
               crunch_json,
               routing_json,
               cache_json,
               category,
               cache_creation_input_tokens,
               cache_read_input_tokens,
               retry_count,
               thinking_output_tokens
        FROM calls
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    if not rows:
        return None
    features = [_row_features(dict(row)) for row in reversed(rows)]
    return _session_memory(
        session_id,
        features,
        window_size=window_size,
        min_plateau_chars=min_plateau_chars,
        max_plateau_delta_ratio=max_plateau_delta_ratio,
        plateau_pairs_for_classification=plateau_pairs_for_classification,
    )
