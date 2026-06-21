from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tokenclaw.policy_files import policy_file_status
from tokenclaw.pricing import estimate_cost, pricing_basis
from tokenclaw.store import stable_json, utc_now


OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA = "tokenclaw.old_context_summary_dry_run.v1"
TOKEN_CHARS = 4
CURRENT_POLICY_PROFILE = "current-policy"
TOOL_PROTOCOL_AWARE_PROFILE = "tool-protocol-aware"
PLATEAU_MIN_TEXT_CHARS = 8_000
PLATEAU_MAX_DELTA_RATIO = 0.03


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _sqlite_db_path(db_path: str) -> Path | None:
    if db_path.startswith("sqlite:///"):
        return Path(db_path.removeprefix("sqlite:///")).expanduser()
    if "://" in db_path:
        return None
    return Path(db_path).expanduser()


def _source_surface(provider: Any, path: Any) -> str:
    provider_text = str(provider or "anthropic")
    path_text = str(path or "")
    if provider_text == "anthropic" and path_text.endswith("/v1/messages"):
        return "anthropic_messages"
    if provider_text == "openai":
        return "openai"
    return provider_text


def _model_tier(model: Any) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    if "gpt" in text:
        return "openai"
    return "unknown"


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


def _session_key(value: Any) -> str | None:
    if value in (None, ""):
        return None
    import hashlib

    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _summary_policy_from_bundle_or_current(bundle_or_policy: Any) -> tuple[dict[str, Any], str, str | None]:
    from tokenclaw import crunch

    if isinstance(bundle_or_policy, dict):
        policies = bundle_or_policy.get("policies") if isinstance(bundle_or_policy.get("policies"), dict) else None
        crunch_policy = policies.get("crunch") if isinstance(policies, dict) and isinstance(policies.get("crunch"), dict) else None
        if crunch_policy is None and (
            "old_context_summarization" in bundle_or_policy or "enabled" in bundle_or_policy
        ):
            crunch_policy = bundle_or_policy
        if isinstance(crunch_policy, dict):
            base = copy.deepcopy(crunch.CRUNCH_POLICY)
            summary = crunch_policy.get("old_context_summarization")
            if isinstance(summary, dict):
                crunch._apply_summary_policy_yaml(base, summary)
            policy_source = str(crunch_policy.get("policy_source") or crunch.CRUNCH_POLICY_SOURCE)
            return base["old_context_summarization"], policy_source, None

    return copy.deepcopy(crunch.OLD_CONTEXT_SUMMARY_POLICY), str(crunch.CRUNCH_POLICY_SOURCE), str(crunch.CRUNCH_RULES_PATH)


def _apply_dry_run_profile(policy: dict[str, Any], profile: str | None) -> dict[str, Any]:
    profile_name = profile or CURRENT_POLICY_PROFILE
    if profile_name == CURRENT_POLICY_PROFILE:
        out = copy.deepcopy(policy)
        out["dry_run_profile"] = CURRENT_POLICY_PROFILE
        return out
    if profile_name != TOOL_PROTOCOL_AWARE_PROFILE:
        raise ValueError(f"unsupported old-context summary dry-run profile: {profile_name}")

    out = copy.deepcopy(policy)
    out["enabled"] = True
    out["dry_run_profile"] = TOOL_PROTOCOL_AWARE_PROFILE
    out["rule_id"] = str(out.get("rule_id") or "local-old-context-summarization")
    excluded = {str(item) for item in out.get("excluded_categories") or []}
    excluded.difference_update({"tool-heavy", "tool-result"})
    out["excluded_categories"] = sorted(excluded)
    out["block_tool_protocol"] = False
    out["block_thinking"] = True
    out["tool_protocol_handling"] = {
        "summary_source": "non-tool-text-turns-only",
        "forwarded_request": "preserve-tool-use-and-tool-result-messages",
        "runtime_enabled": False,
    }
    return out


@contextmanager
def _patched_summary_policy(policy: dict[str, Any], policy_source: str, rule_path: str | None) -> Iterator[None]:
    from tokenclaw import crunch

    names = (
        "CRUNCH_POLICY_SOURCE",
        "CRUNCH_RULES_PATH",
        "OLD_CONTEXT_SUMMARY_POLICY",
        "OLD_CONTEXT_SUMMARY_ENABLED",
        "OLD_CONTEXT_SUMMARY_MODEL",
        "OLD_CONTEXT_SUMMARY_RULE_ID",
        "OLD_CONTEXT_SUMMARY_CANDIDATE_ID",
        "OLD_CONTEXT_SUMMARY_PLACEMENT",
        "OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS",
        "OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS",
        "OLD_CONTEXT_SUMMARY_MAX_TURNS",
        "OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS",
        "OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS",
        "OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS",
        "OLD_CONTEXT_SUMMARY_MAX_COST_USD",
        "OLD_CONTEXT_SUMMARY_EXCLUDED_CATEGORIES",
        "OLD_CONTEXT_SUMMARY_BLOCK_TOOL_PROTOCOL",
        "OLD_CONTEXT_SUMMARY_BLOCK_THINKING",
    )
    previous = {name: getattr(crunch, name) for name in names}
    try:
        crunch.CRUNCH_POLICY_SOURCE = policy_source
        if rule_path is not None:
            crunch.CRUNCH_RULES_PATH = rule_path
        crunch.OLD_CONTEXT_SUMMARY_POLICY = policy
        crunch.OLD_CONTEXT_SUMMARY_ENABLED = bool(policy.get("enabled"))
        crunch.OLD_CONTEXT_SUMMARY_MODEL = str(policy.get("model") or "claude-haiku-4-5-20251001")
        crunch.OLD_CONTEXT_SUMMARY_RULE_ID = str(policy.get("rule_id") or "local-old-context-summarization")
        crunch.OLD_CONTEXT_SUMMARY_CANDIDATE_ID = policy.get("candidate_id")
        crunch.OLD_CONTEXT_SUMMARY_PLACEMENT = str(policy.get("placement") or "system")
        crunch.OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS = _as_int(policy.get("min_request_chars"), 32000)
        crunch.OLD_CONTEXT_SUMMARY_MIN_SUMMARIZED_CHARS = _as_int(policy.get("min_summarized_chars"), 12000)
        crunch.OLD_CONTEXT_SUMMARY_MAX_TURNS = _as_int(policy.get("max_turns"), 6)
        crunch.OLD_CONTEXT_SUMMARY_KEEP_RECENT_TURNS = _as_int(policy.get("keep_recent_turns"), 4)
        crunch.OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS = _as_int(policy.get("max_summary_chars"), 4000)
        crunch.OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS = _as_int(policy.get("max_source_chars"), 80000)
        crunch.OLD_CONTEXT_SUMMARY_MAX_COST_USD = _as_float(policy.get("max_summary_cost_usd"), 0.02)
        crunch.OLD_CONTEXT_SUMMARY_EXCLUDED_CATEGORIES = {
            str(item) for item in policy.get("excluded_categories") or []
        }
        crunch.OLD_CONTEXT_SUMMARY_BLOCK_TOOL_PROTOCOL = _enabled(policy.get("block_tool_protocol"), True)
        crunch.OLD_CONTEXT_SUMMARY_BLOCK_THINKING = _enabled(policy.get("block_thinking"), True)
        yield
    finally:
        for name, value in previous.items():
            setattr(crunch, name, value)


def _load_recent_rows(db_path: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = _sqlite_db_path(db_path)
    if path is None:
        return [], {
            "reason": "unsupported-db-url",
            "message": "old-context dry-run currently reads local SQLite traffic only",
        }
    if not path.exists():
        return [], {"reason": "db-not-found", "message": f"AgentFlow SQLite database not found: {path}"}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select id, created_at, provider, path, requested_model, routed_model, stream,
                   status_code, latency_ms, input_tokens_est, actual_input_tokens,
                   output_tokens_est, actual_output_tokens, cost_est_usd, cost_baseline_usd,
                   crunch_json, routing_json, cache_json, request_json, session_id,
                   category, retry_count, thinking_output_tokens
            from calls
            order by datetime(created_at) desc
            limit ?
            """,
            (max(0, int(limit)),),
        ).fetchall()
        return [dict(row) for row in rows], None
    except sqlite3.Error as exc:
        return [], {"reason": "db-query-failed", "message": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _row_text_chars(row: dict[str, Any]) -> int:
    routing = _json_obj(row.get("routing_json"))
    text_chars = _as_int(routing.get("text_chars"))
    if text_chars > 0:
        return text_chars
    raw = row.get("request_json")
    if raw:
        return len(str(raw))
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * TOKEN_CHARS)


def _plateau_row_ids(rows: list[dict[str, Any]]) -> set[str]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session = row.get("session_id")
        row_id = row.get("id")
        if session in (None, "") or row_id in (None, ""):
            continue
        by_session.setdefault(str(session), []).append(row)

    plateau_ids: set[str] = set()
    for session_rows in by_session.values():
        previous: dict[str, Any] | None = None
        for row in sorted(session_rows, key=lambda item: str(item.get("created_at") or "")):
            text_chars = _row_text_chars(row)
            if previous is not None:
                previous_text = _row_text_chars(previous)
                if (
                    previous_text >= PLATEAU_MIN_TEXT_CHARS
                    and text_chars >= PLATEAU_MIN_TEXT_CHARS
                    and abs(text_chars - previous_text) / max(previous_text, 1) <= PLATEAU_MAX_DELTA_RATIO
                ):
                    plateau_ids.add(str(previous.get("id")))
                    plateau_ids.add(str(row.get("id")))
            previous = row
    return plateau_ids


def _plateau_status(row: dict[str, Any], plateau_ids: set[str]) -> str:
    if row.get("session_id") in (None, ""):
        return "no-session"
    if str(row.get("id") or "") in plateau_ids:
        return "plateau-adjacent"
    if _row_text_chars(row) >= PLATEAU_MIN_TEXT_CHARS:
        return "large-not-plateaued"
    return "below-plateau-threshold"


def _old_messages(body: dict[str, Any], keep_recent_turns: int) -> list[dict[str, Any]]:
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return []
    old_limit = max(0, len(messages) - max(0, keep_recent_turns))
    return [msg for msg in messages[:old_limit] if isinstance(msg, dict)]


def _message_has_tool_protocol(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"} for block in content)


def _message_has_thinking(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "thinking" for block in content)


def _context_blocker_states(body: dict[str, Any], policy: dict[str, Any], reason: str) -> tuple[str, str]:
    old_messages = _old_messages(body, _as_int(policy.get("keep_recent_turns"), 4))
    has_tool_protocol = any(_message_has_tool_protocol(msg) for msg in old_messages)
    has_thinking = any(_message_has_thinking(msg) for msg in old_messages)
    if reason == "tool-protocol-context-blocked":
        tool_protocol = "blocked"
    elif has_tool_protocol:
        tool_protocol = "preserved"
    else:
        tool_protocol = "none"
    if reason == "thinking-context-blocked":
        thinking = "blocked"
    elif has_thinking:
        thinking = "present"
    else:
        thinking = "clear"
    return thinking, tool_protocol


def _cache_key_exists(db_path: str, cache_key: str) -> bool:
    path = _sqlite_db_path(db_path)
    if path is None or not path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        row = conn.execute("select 1 from cache where cache_key = ? limit 1", (cache_key,)).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _group_key(
    row: dict[str, Any],
    category: str,
    blocker: str,
    plateau_status: str,
    thinking_blocker: str,
    tool_protocol_blocker: str,
) -> tuple[Any, ...]:
    return (
        _source_surface(row.get("provider"), row.get("path")),
        category or "unknown",
        _model_tier(row.get("requested_model")),
        bool(row.get("stream")),
        plateau_status,
        thinking_blocker,
        tool_protocol_blocker,
        blocker,
    )


def _empty_group(
    row: dict[str, Any],
    category: str,
    blocker: str,
    plateau_status: str,
    thinking_blocker: str,
    tool_protocol_blocker: str,
) -> dict[str, Any]:
    (
        source_surface,
        group_category,
        model_tier,
        stream,
        plateau,
        thinking,
        tool_protocol,
        blocker_reason,
    ) = _group_key(row, category, blocker, plateau_status, thinking_blocker, tool_protocol_blocker)
    return {
        "source_surface": source_surface,
        "category": group_category,
        "model_tier": model_tier,
        "stream": stream,
        "plateau_status": plateau,
        "thinking_blocker": thinking,
        "tool_protocol_blocker": tool_protocol,
        "blocker": blocker_reason,
        "call_count": 0,
        "session_count": 0,
        "_sessions": set(),
        "eligible_call_count": 0,
        "summary_cache_hit_count": 0,
        "eligible_old_turns": 0,
        "eligible_chars": 0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "estimated_summary_cost_usd": 0.0,
        "projected_gross_savings_usd": 0.0,
        "projected_net_savings_usd": 0.0,
    }


def _input_savings_usd(model: str, tokens_saved: int) -> float:
    basis = pricing_basis(model or "claude-sonnet-4-6", provider="anthropic")
    input_price = float(basis.get("input_usd_per_million") or 0.0)
    return (max(0, int(tokens_saved)) / 1_000_000.0) * input_price


def _summary_settings(policy: dict[str, Any], policy_source: str, rule_path: str | None) -> dict[str, Any]:
    return {
        "enabled": bool(policy.get("enabled")),
        "dry_run_profile": str(policy.get("dry_run_profile") or CURRENT_POLICY_PROFILE),
        "policy_source": policy_source,
        "rule_path": rule_path,
        "rule_id": policy.get("rule_id"),
        "candidate_id": policy.get("candidate_id"),
        "model": policy.get("model"),
        "placement": policy.get("placement"),
        "min_request_chars": _as_int(policy.get("min_request_chars")),
        "min_summarized_chars": _as_int(policy.get("min_summarized_chars")),
        "max_turns": _as_int(policy.get("max_turns")),
        "keep_recent_turns": _as_int(policy.get("keep_recent_turns")),
        "max_summary_chars": _as_int(policy.get("max_summary_chars")),
        "max_source_chars": _as_int(policy.get("max_source_chars")),
        "max_summary_cost_usd": _as_float(policy.get("max_summary_cost_usd")),
        "excluded_categories": [str(item) for item in policy.get("excluded_categories") or []],
        "block_tool_protocol": _enabled(policy.get("block_tool_protocol"), True),
        "block_thinking": _enabled(policy.get("block_thinking"), True),
        "tool_protocol_handling": copy.deepcopy(policy.get("tool_protocol_handling") or {}),
        "canary": copy.deepcopy(policy.get("canary") or {}),
        "safety_stop": copy.deepcopy(policy.get("safety_stop") or {}),
    }


def dry_run_old_context_summary(
    bundle_or_policy: Any = None,
    *,
    db_path: str,
    limit: int = 500,
    profile: str = CURRENT_POLICY_PROFILE,
) -> dict[str, Any]:
    from tokenclaw import crunch

    policy, policy_source, rule_path = _summary_policy_from_bundle_or_current(bundle_or_policy)
    try:
        policy = _apply_dry_run_profile(policy, profile)
    except ValueError as exc:
        policy = _apply_dry_run_profile(policy, CURRENT_POLICY_PROFILE)
        settings = _summary_settings(policy, policy_source, rule_path)
        return {
            "schema": OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA,
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "generated_at": utc_now(),
            "status": "invalid-profile",
            "reason": "unsupported-profile",
            "message": str(exc),
            "db_path": db_path,
            "lookback_call_limit": limit,
            "policy": settings,
            "groups": [],
            "summary": {},
            "privacy": {
                "metadata_only_output": True,
                "raw_bodies_read_locally": False,
                "raw_prompts_included": False,
                "raw_request_bodies_included": False,
                "raw_session_ids_included": False,
                "cache_keys_included": False,
            },
        }
    rows, unavailable = _load_recent_rows(db_path, limit)
    settings = _summary_settings(policy, policy_source, rule_path)
    reload_status = None
    if rule_path is not None:
        reload_status = policy_file_status(
            rule_path,
            loaded_at=crunch.CRUNCH_RULES_LOADED_AT,
            loaded_snapshot=crunch.CRUNCH_RULES_LOADED_FILE,
        )
    if unavailable:
        return {
            "schema": OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA,
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "generated_at": utc_now(),
            "status": "unavailable",
            **unavailable,
            "db_path": db_path,
            "lookback_call_limit": limit,
            "policy": settings,
            "reload_required": bool((reload_status or {}).get("reload_required", False)),
            "groups": [],
            "summary": {},
            "privacy": {
                "metadata_only_output": True,
                "raw_bodies_read_locally": False,
                "raw_prompts_included": False,
                "raw_request_bodies_included": False,
                "raw_session_ids_included": False,
                "cache_keys_included": False,
            },
        }

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    plateau_ids = _plateau_row_ids(rows)
    sampled = len(rows)
    raw_available = 0
    raw_used = 0
    provider_rows = 0
    with _patched_summary_policy(policy, policy_source, rule_path):
        for row in rows:
            plateau = _plateau_status(row, plateau_ids)
            thinking_blocker = "unknown"
            tool_protocol_blocker = "unknown"
            provider = row.get("provider") or "anthropic"
            if provider != "anthropic" or not str(row.get("path") or "").endswith("/v1/messages"):
                category = row.get("category") or _json_obj(row.get("routing_json")).get("category") or "unknown"
                blocker = "unsupported-source-surface"
                group = groups.setdefault(
                    _group_key(row, category, blocker, plateau, thinking_blocker, tool_protocol_blocker),
                    _empty_group(row, category, blocker, plateau, thinking_blocker, tool_protocol_blocker),
                )
                group["call_count"] += 1
                session = _session_key(row.get("session_id"))
                if session:
                    group["_sessions"].add(session)
                continue
            provider_rows += 1
            routing = _json_obj(row.get("routing_json"))
            category = row.get("category") or routing.get("category") or "unknown"
            raw = row.get("request_json")
            if raw:
                raw_available += 1
            body = _json_obj(raw)
            if not body:
                blocker = "request-body-unavailable"
                group = groups.setdefault(
                    _group_key(row, category, blocker, plateau, thinking_blocker, tool_protocol_blocker),
                    _empty_group(row, category, blocker, plateau, thinking_blocker, tool_protocol_blocker),
                )
                group["call_count"] += 1
                session = _session_key(row.get("session_id"))
                if session:
                    group["_sessions"].add(session)
                continue
            raw_used += 1
            plan, meta = crunch.old_context_summary_plan(body, exact_cache_enabled=None)
            blocker = "eligible" if plan is not None else str(meta.get("reason") or "not-eligible")
            category = str(meta.get("category") or category or "unknown")
            thinking_blocker, tool_protocol_blocker = _context_blocker_states(body, policy, blocker)
            group = groups.setdefault(
                _group_key(row, category, blocker, plateau, thinking_blocker, tool_protocol_blocker),
                _empty_group(row, category, blocker, plateau, thinking_blocker, tool_protocol_blocker),
            )
            group["call_count"] += 1
            session = _session_key(row.get("session_id"))
            if session:
                group["_sessions"].add(session)
            if plan is None:
                continue

            estimated_summary_chars = min(
                _as_int(policy.get("max_summary_chars"), 4000),
                max(400, _as_int(plan.get("eligible_chars")) // 8),
            )
            placeholder = "x" * max(1, estimated_summary_chars)
            summarized = crunch.apply_old_context_summary(body, plan, placeholder)
            after_chars = len(stable_json(summarized))
            saved_chars = max(0, _as_int(plan.get("before_chars")) - after_chars)
            saved_tokens = max(0, saved_chars // TOKEN_CHARS)
            summary_request_chars = len(stable_json(plan.get("summary_request") or {}))
            summary_input_tokens = max(1, summary_request_chars // TOKEN_CHARS)
            summary_output_tokens = max(1, estimated_summary_chars // TOKEN_CHARS)
            summary_cost = estimate_cost(str(policy.get("model") or ""), summary_input_tokens, summary_output_tokens) or 0.0
            gross = _input_savings_usd(str(body.get("model") or row.get("requested_model") or ""), saved_tokens)
            cache_hit = _cache_key_exists(db_path, str(plan.get("cache_key") or ""))

            group["eligible_call_count"] += 1
            group["summary_cache_hit_count"] += int(cache_hit)
            group["eligible_old_turns"] += _as_int(plan.get("eligible_turns"))
            group["eligible_chars"] += _as_int(plan.get("eligible_chars"))
            group["projected_saved_chars"] += saved_chars
            group["projected_saved_tokens"] += saved_tokens
            group["estimated_summary_cost_usd"] += 0.0 if cache_hit else summary_cost
            group["projected_gross_savings_usd"] += gross
            group["projected_net_savings_usd"] += gross - (0.0 if cache_hit else summary_cost)

    output_groups: list[dict[str, Any]] = []
    for group in groups.values():
        group["session_count"] = len(group.pop("_sessions"))
        for key in ("estimated_summary_cost_usd", "projected_gross_savings_usd", "projected_net_savings_usd"):
            group[key] = round(float(group[key]), 8)
        output_groups.append(group)
    output_groups.sort(key=lambda item: (
        str(item["source_surface"]),
        str(item["category"]),
        str(item["model_tier"]),
        str(item["stream"]),
        str(item["plateau_status"]),
        str(item["thinking_blocker"]),
        str(item["tool_protocol_blocker"]),
        str(item["blocker"]),
    ))
    eligible_groups = [group for group in output_groups if group["blocker"] == "eligible"]
    skip_reasons: dict[str, int] = {}
    for group in output_groups:
        if group["blocker"] != "eligible":
            skip_reasons[group["blocker"]] = skip_reasons.get(group["blocker"], 0) + int(group["call_count"])
    return {
        "schema": OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": "simulated",
        "db_path": db_path,
        "lookback_call_limit": limit,
        "policy": settings,
        "reload_required": bool((reload_status or {}).get("reload_required", False)),
        "reload_status": reload_status,
        "summary": {
            "sampled_call_count": sampled,
            "sampled_provider_call_count": provider_rows,
            "request_body_available_count": raw_available,
            "request_body_replayed_count": raw_used,
            "eligible_call_count": sum(group["eligible_call_count"] for group in eligible_groups),
            "summary_cache_hit_count": sum(group["summary_cache_hit_count"] for group in eligible_groups),
            "eligible_old_turns": sum(group["eligible_old_turns"] for group in eligible_groups),
            "eligible_chars": sum(group["eligible_chars"] for group in eligible_groups),
            "projected_saved_chars": sum(group["projected_saved_chars"] for group in eligible_groups),
            "projected_saved_tokens": sum(group["projected_saved_tokens"] for group in eligible_groups),
            "estimated_summary_cost_usd": round(sum(float(group["estimated_summary_cost_usd"]) for group in eligible_groups), 8),
            "projected_gross_savings_usd": round(sum(float(group["projected_gross_savings_usd"]) for group in eligible_groups), 8),
            "projected_net_savings_usd": round(sum(float(group["projected_net_savings_usd"]) for group in eligible_groups), 8),
            "skip_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(skip_reasons.items())
            ],
            "plateau_policy": {
                "min_text_chars": PLATEAU_MIN_TEXT_CHARS,
                "max_delta_ratio": PLATEAU_MAX_DELTA_RATIO,
            },
        },
        "groups": output_groups,
        "privacy": {
            "metadata_only_output": True,
            "raw_bodies_read_locally": raw_used > 0,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_session_ids_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "source_hashes_included": False,
        },
    }
