from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable

from agentflow_proxy.pricing import estimate_cost, pricing_basis


SCHEMA = "agentflow.orchestrator_research_plan.v1"

SENSITIVE_VALUE = "[REDACTED]"
SENSITIVE_ID = "[REDACTED_ID]"
SENSITIVE_PATH = "[REDACTED_PATH]"
SENSITIVE_SECRET = "[REDACTED_SECRET]"

_PATH_RE = re.compile(r"(?<![\w:])/(?:home|tmp|var|private|Users|mnt|workspace|srv|opt)/[^\s\"'`),;]+")
_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|api[_-]?key[=:][^\s\"']+)",
    re.IGNORECASE,
)
_ID_ASSIGNMENT_RE = re.compile(
    r"\b((?:request|session|thread|tenant|cache|candidate|run|trace)[_-]?id)\s*[:=]\s*[\"']?([A-Za-z0-9_.:/@-]{6,})",
    re.IGNORECASE,
)
_RAW_FIELD_NAMES = {
    "content",
    "file_path",
    "file_paths",
    "messages",
    "path",
    "prompt",
    "raw_body",
    "raw_content",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request_body",
    "request_json",
    "response_body",
    "response_json",
}
_ID_FIELD_RE = re.compile(
    r"(?:(?:^|_)(?:request|session|thread|tenant|trace|candidate|cohort|proposal|policy|rule)_?(?:id|key|fingerprint)$)"
    r"|(?:(?:^|_)cache_?(?:id|key|fingerprint)$)"
    r"|(?:^|_)pattern_hash(?:es)?$"
)
_DIAGNOSTIC_RE = re.compile(
    r"(?:skip[_ -]?reason|omitted[_ -]?reason|blocker|blocked|reason|verdict)\s*[:=]\s*[\"']?([A-Za-z0-9_.:-]+(?:[ -][A-Za-z0-9_.:-]+){0,5})",
    re.IGNORECASE,
)
_KNOWN_DIAGNOSTIC_TERMS = (
    "need-more-samples",
    "missing dependency evidence",
    "missing-dependency-evidence",
    "missing lifecycle feedback",
    "missing-lifecycle-feedback",
    "stale quality evidence",
    "stale-quality-evidence",
    "stale lifecycle evidence",
    "stale-lifecycle-evidence",
    "high retry rate",
    "high-retry-rate",
    "high error rate",
    "high-error-rate",
    "holdout regression",
    "holdout-regression",
    "non-positive savings",
    "non-positive-savings",
    "privacy-blocked",
    "aggregate-only",
    "safety-stop",
    "provider capability mismatch",
    "provider-capability-mismatch",
    "unsupported local executor",
    "unsupported-local-executor",
    "no local representation",
    "no-local-representation",
)

_PASS_DIAGNOSTIC_REASONS = {
    "pass",
    "passed",
    "verdict-pass",
    "test-verdict-pass",
    "quality-gate-passed",
    "offline-fixture-passed",
    "eval-pass-threshold-met",
    "canary-holdout-thresholds-met",
    "promotion-thresholds-met",
}

_DIAGNOSTIC_TAXONOMY: tuple[dict[str, Any], ...] = (
    {
        "class": "safety-stop",
        "priority": 10,
        "aliases": ("safety-stop", "safety-stopped", "safety_stop"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Review the safety stop and either resolve the safe bypass condition or keep the affected activation blocked with a narrow reason.",
        "acceptance_check": "The next report shows the safety-stop count reduced for the affected cohort or records an explicit keep-blocked reason.",
    },
    {
        "class": "regression",
        "priority": 15,
        "aliases": ("regression", "regressed", "holdout-regression", "quality-regression", "error-regression"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Rollback, disable, or narrow the candidate until applied and holdout evidence no longer shows a regression.",
        "acceptance_check": "A bounded canary or eval report shows no applied-vs-holdout regression before activation is reconsidered.",
    },
    {
        "class": "high-retry-error-rate",
        "priority": 20,
        "aliases": ("high-retry-rate", "high-error-rate", "retry-rate-above-threshold", "error-rate-above-threshold"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Identify the provider, local action, or canary cohort causing elevated retry/error pressure and reduce it before widening.",
        "acceptance_check": "The affected cohort reports retry and error rates below the activation threshold in the next metadata window.",
    },
    {
        "class": "missing-lifecycle-feedback",
        "priority": 30,
        "aliases": (
            "missing-lifecycle-feedback",
            "lifecycle-feedback-missing",
            "missing-feedback",
            "missing-canary-lifecycle",
        ),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Emit applied, holdout, fallback, retry, error, and savings lifecycle feedback for the affected activation path.",
        "acceptance_check": "Lifecycle feedback includes applied and holdout cohort counts plus savings and omission reason fields.",
    },
    {
        "class": "stale-evidence",
        "priority": 35,
        "aliases": ("stale-evidence", "stale-quality-evidence", "stale-lifecycle-evidence", "stale-canary-evidence"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Refresh the canary, eval, or rollout evidence inside the configured evidence window.",
        "acceptance_check": "A later report uses fresh evidence timestamps before creating an activation or widening issue.",
    },
    {
        "class": "aggregate-only",
        "priority": 40,
        "aliases": ("aggregate-only", "aggregate_only", "aggregate-only-feedback"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Add privacy-safe candidate/cohort-level lifecycle fields without exposing prompts, provider bodies, request IDs, session IDs, or raw identifiers.",
        "acceptance_check": "The affected report can name the candidate/cohort state needed for review while preserving metadata-only privacy.",
    },
    {
        "class": "unsupported-provider-action",
        "priority": 45,
        "aliases": (
            "unsupported-provider-action",
            "unsupported-action",
            "unsupported-local-executor",
            "provider-capability-mismatch",
            "capability-mismatch",
        ),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Map the recommendation to a supported local executor capability or keep the provider/action pair explicitly omitted.",
        "acceptance_check": "The provider capability matrix reports supported local execution or a machine-readable omitted action.",
    },
    {
        "class": "no-local-representation",
        "priority": 50,
        "aliases": ("no-local-representation", "local-representation-missing", "not-representable-locally"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Define the file-backed local rule, dry-run, canary, or review-only bundle representation for the recommendation.",
        "acceptance_check": "A local rule or review artifact can represent the recommendation without managed enforcement or provider body rewrites.",
    },
    {
        "class": "non-positive-savings",
        "priority": 55,
        "aliases": ("non-positive-savings", "negative-savings", "zero-savings", "savings-not-positive"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Keep the candidate out of activation until observed or projected savings per 1000 calls becomes positive.",
        "acceptance_check": "The candidate reports positive savings per 1000 calls or remains explicitly omitted as non-actionable.",
    },
    {
        "class": "missing-dependency-evidence",
        "priority": 60,
        "aliases": ("missing-dependency-evidence", "missing-dependency", "need-more-samples", "insufficient-samples"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Collect the missing dependency, sample, holdout, or invalidation evidence before activation.",
        "acceptance_check": "The next report includes the missing evidence or a narrower blocked reason with the remaining gap.",
    },
    {
        "class": "privacy-blocked",
        "priority": 65,
        "aliases": ("privacy-blocked", "privacy-blocker", "privacy"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Replace the blocked evidence with metadata-only fields or keep the optimization out of unattended issue generation.",
        "acceptance_check": "Generated evidence remains metadata-only and excludes prompts, provider bodies, file paths, request IDs, session IDs, and raw identifiers.",
    },
    {
        "class": "unclassified-skip-or-blocker",
        "priority": 90,
        "aliases": ("unclassified-skip-or-blocker",),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Trace the unclassified skip/blocker to a bounded diagnostic reason before creating activation work.",
        "acceptance_check": "The next research plan reports a classified reason or a narrow issue for the emitting report.",
    },
)

_CRUNCH_OPPORTUNITY_REPORT_KEYS = (
    "old_context_summary_opportunity",
    "terminal_output_compaction_opportunity",
    "instruction_dedup_opportunity",
    "anthropic_thinking_compaction_opportunity",
    "codex_terminal_transcript_opportunity",
    "repeated_scaffold_opportunity",
)

_LOCAL_POLICY_REPRESENTATIONS = {
    "routing": ("routing", "routing_rules.yaml"),
    "crunch": ("crunch", "crunch_rules.yaml"),
    "cache": ("cache", "cache_rules.yaml"),
    "routing-experiment": ("routing_experiments", "routing_experiments.yaml"),
    "codex-app": ("codex_app", "codex_app_rules.yaml"),
}

_UNSUPPORTED_LOCAL_ACTION_FAMILIES = {
    "server-content-processing",
    "prompt-replacement",
    "provider-body-rewrite",
}


def redact_text(value: str) -> str:
    text = _SECRET_RE.sub(SENSITIVE_SECRET, value)
    text = _PATH_RE.sub(SENSITIVE_PATH, text)
    text = _ID_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}={SENSITIVE_ID}", text)
    return text


def sanitize_value(value: Any, *, key: str | None = None) -> Any:
    key_l = (key or "").lower()
    if key_l in _RAW_FIELD_NAMES:
        return SENSITIVE_VALUE
    if _ID_FIELD_RE.search(key_l):
        return SENSITIVE_ID
    if isinstance(value, dict):
        return {str(k): sanitize_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _label_names(issue: dict[str, Any]) -> set[str]:
    labels = issue.get("labels") or []
    names: set[str] = set()
    for label in labels:
        if isinstance(label, str):
            names.add(label)
        elif isinstance(label, dict) and label.get("name"):
            names.add(str(label["name"]))
    return names


def _author_login(issue: dict[str, Any]) -> str:
    author = issue.get("author")
    if isinstance(author, dict):
        return str(author.get("login") or "")
    if isinstance(author, str):
        return author
    return ""


def _issue_number(issue: dict[str, Any]) -> int | None:
    number = issue.get("number")
    try:
        return int(number)
    except (TypeError, ValueError):
        return None


def _is_open(issue: dict[str, Any]) -> bool:
    return str(issue.get("state") or "OPEN").upper() == "OPEN"


def _is_trusted(issue: dict[str, Any], trusted_author: str) -> bool:
    return not trusted_author or _author_login(issue) == trusted_author


def _is_actionable_ready(issue: dict[str, Any], trusted_author: str) -> bool:
    labels = _label_names(issue)
    return _is_open(issue) and _is_trusted(issue, trusted_author) and "status:ready" in labels and "status:blocked" not in labels


def _is_blocked(issue: dict[str, Any], trusted_author: str) -> bool:
    labels = _label_names(issue)
    return _is_open(issue) and _is_trusted(issue, trusted_author) and "status:blocked" in labels


def _issue_ref(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": issue.get("repo") or issue.get("repository") or "unknown",
        "number": _issue_number(issue),
        "title": sanitize_value(str(issue.get("title") or "")),
        "url": sanitize_value(issue.get("url") or issue.get("html_url") or ""),
        "labels": sorted(_label_names(issue)),
    }


def _stale_blocked_issues(
    issues: Iterable[dict[str, Any]],
    *,
    trusted_author: str,
    now: datetime,
    stale_days: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for issue in issues:
        if not _is_blocked(issue, trusted_author):
            continue
        updated = _parse_time(issue.get("updatedAt") or issue.get("updated_at") or issue.get("createdAt") or issue.get("created_at"))
        age_days = (now - updated).days if updated is not None else None
        if age_days is None or age_days >= stale_days:
            item = _issue_ref(issue)
            item["age_days"] = age_days
            stale.append(item)
    return stale


def _load_log_text(log_sources: Iterable[str | Path]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for source in log_sources:
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = str(source)
            source_name = "inline"
        else:
            source_name = path.name
        loaded.append({"source": sanitize_value(source_name), "text": text})
    return loaded


def _diagnostics_from_logs(log_sources: Iterable[str | Path], *, limit: int = 10) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for loaded in _load_log_text(log_sources):
        for raw_line in loaded["text"].splitlines():
            line = redact_text(raw_line.strip())
            if not line:
                continue
            lowered = line.lower()
            matched = False
            for match in _DIAGNOSTIC_RE.finditer(line):
                reason = match.group(1).strip(" .'\",").lower().replace(" ", "-")
                if reason:
                    counter[reason] += 1
                    examples.setdefault(reason, line[:240])
                    matched = True
            for term in _KNOWN_DIAGNOSTIC_TERMS:
                if term in lowered:
                    reason = term.lower().replace(" ", "-")
                    counter[reason] += 1
                    examples.setdefault(reason, line[:240])
                    matched = True
            if not matched and ("skip" in lowered or "blocked" in lowered or "omitted" in lowered):
                reason = "unclassified-skip-or-blocker"
                counter[reason] += 1
                examples.setdefault(reason, line[:240])
    return [
        {"reason": reason, "count": count, "example": examples.get(reason, "")}
        for reason, count in counter.most_common(limit)
    ]


def _diagnostic_taxonomy(reason: Any) -> dict[str, Any] | None:
    text = str(reason or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text or text in _PASS_DIAGNOSTIC_REASONS:
        return None
    if text.startswith("pass-") or text.endswith("-passed"):
        return None
    for entry in _DIAGNOSTIC_TAXONOMY:
        aliases = tuple(str(alias).lower().replace("_", "-").replace(" ", "-") for alias in entry["aliases"])
        if any(text == alias or alias in text for alias in aliases):
            return entry
    return None


def _diagnostic_source_lever(reason: str, diagnostic_class: str) -> str:
    text = f"{reason} {diagnostic_class}".lower()
    if "cache" in text or "replay" in text:
        return "cache"
    if "routing" in text or "canary" in text or "model" in text:
        return "routing"
    if "crunch" in text or "compaction" in text or "summary" in text or "dedup" in text:
        return "crunch"
    if "managed" in text or "provider-capability" in text or "unsupported" in text:
        return "managed-recommendation"
    return "activation-feedback"


def _actionable_diagnostics(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actionable: list[dict[str, Any]] = []
    for index, item in enumerate(diagnostics):
        reason = str(item.get("reason") or "")
        taxonomy = _diagnostic_taxonomy(reason)
        if taxonomy is None:
            continue
        diagnostic_class = str(taxonomy["class"])
        enriched = dict(item)
        enriched.update(
            {
                "diagnostic_class": diagnostic_class,
                "source_lever": _diagnostic_source_lever(reason, diagnostic_class),
                "backlog_action": taxonomy["backlog_action"],
                "expected_unblock_path": taxonomy["unblock_path"],
                "acceptance_check": taxonomy["acceptance_check"],
                "_priority": _to_int(taxonomy["priority"], 100),
                "_index": index,
            }
        )
        actionable.append(enriched)
    actionable.sort(key=lambda row: (_to_int(row.get("_priority"), 100), -_to_int(row.get("count")), _to_int(row.get("_index"))))
    for row in actionable:
        row.pop("_priority", None)
        row.pop("_index", None)
    return actionable


def _stats_summary(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = sanitize_value(stats or {})
    if not isinstance(stats, dict):
        return {}
    keys = (
        "calls",
        "today_calls",
        "cache_hits",
        "cache_hit_rate",
        "today_cost_usd",
        "cost_est_usd",
        "today_errors",
        "error_rate",
        "today_prompt_cache_savings_usd",
        "today_routing_savings_usd",
        "crunched_count",
        "crunch_chars_saved",
        "crunch_tokens_saved",
        "crunch_savings_usd",
        "today_crunch_savings_usd",
        "avg_crunch_ratio",
    )
    summary = {key: stats[key] for key in keys if key in stats}
    routing = stats.get("routing")
    if isinstance(routing, list):
        summary["routing_top"] = routing[:5]
        pass_through_report = _pass_through_routing_report(routing)
        if pass_through_report is not None:
            summary["pass_through_routing_report"] = pass_through_report
    cache = stats.get("cache_decision_breakdown")
    if isinstance(cache, list):
        summary["cache_decision_breakdown_top"] = cache[:5]
    ladder = stats.get("cache_zero_hit_blocker_ladder")
    if isinstance(ladder, dict):
        ladder_summary = ladder.get("summary") if isinstance(ladder.get("summary"), dict) else {}
        ladder_rows = ladder.get("ladder") if isinstance(ladder.get("ladder"), list) else []
        summary["cache_zero_hit_blocker_ladder"] = {
            "schema": ladder.get("schema"),
            "summary": ladder_summary,
            "ladder": ladder_rows[:5],
            "privacy": ladder.get("privacy") if isinstance(ladder.get("privacy"), dict) else {},
        }
    hit_recovery = stats.get("streaming_cache_hit_recovery") or stats.get("cache_replay_hit_recovery")
    if isinstance(hit_recovery, dict):
        recovery_rows = hit_recovery.get("cohorts") if isinstance(hit_recovery.get("cohorts"), list) else []
        summary["streaming_cache_hit_recovery"] = {
            "schema": hit_recovery.get("schema"),
            "summary": hit_recovery.get("summary") if isinstance(hit_recovery.get("summary"), dict) else {},
            "verdict_breakdown": hit_recovery.get("verdict_breakdown")
            if isinstance(hit_recovery.get("verdict_breakdown"), list)
            else [],
            "cohorts": recovery_rows[:5],
            "privacy": hit_recovery.get("privacy") if isinstance(hit_recovery.get("privacy"), dict) else {},
        }
    cohorts = (
        stats.get("cache_replay_cohort_ranking")
        or stats.get("cache_replay_cohorts")
        or stats.get("cache_replay_plateau_cohort_ranking")
    )
    if isinstance(cohorts, dict):
        cohort_rows = cohorts.get("cohorts") if isinstance(cohorts.get("cohorts"), list) else []
        cohort_summary = cohorts.get("summary") if isinstance(cohorts.get("summary"), dict) else {}
        if cohort_rows or _to_int(cohort_summary.get("candidate_rows")) > 0:
            summary["cache_replay_cohort_ranking"] = {
                "schema": cohorts.get("schema"),
                "summary": cohort_summary,
                "cohorts": cohort_rows[:5],
                "privacy": cohorts.get("privacy") if isinstance(cohorts.get("privacy"), dict) else {},
            }
    openai_canary = stats.get("openai_canary_impact") or stats.get("openai_routing_canary_impact")
    if isinstance(openai_canary, dict):
        candidate_rows = openai_canary.get("candidates") if isinstance(openai_canary.get("candidates"), list) else []
        lifecycle_feedback = (
            openai_canary.get("activation_lifecycle_feedback")
            if isinstance(openai_canary.get("activation_lifecycle_feedback"), dict)
            else {}
        )
        summary["openai_canary_impact"] = {
            "schema": openai_canary.get("schema"),
            "status": openai_canary.get("status"),
            "summary": openai_canary.get("summary") if isinstance(openai_canary.get("summary"), dict) else {},
            "candidates": candidate_rows[:5],
            "activation_lifecycle_feedback": lifecycle_feedback,
            "privacy": openai_canary.get("privacy") if isinstance(openai_canary.get("privacy"), dict) else {},
        }
    crunch_signal = _crunch_savings_signal(stats)
    if crunch_signal is not None:
        summary["crunch_savings_signal"] = crunch_signal
    managed_health = _managed_recommendation_health_signal(stats)
    if managed_health is not None:
        summary["managed_recommendation_health"] = managed_health
    shape_signal = _request_shape_rollup_signal(stats)
    if shape_signal is not None:
        summary["request_shape_rollup_candidates"] = shape_signal
    activation_loop = _evidence_to_activation_loop(summary)
    if activation_loop is not None:
        summary["evidence_to_activation_loop"] = activation_loop
    return summary


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_numeric(summary: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return _to_float(value)
    return 0.0


def _first_int(summary: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return _to_int(value)
    return 0


def _top_breakdown_item(report: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        rows = report.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                return row
    return None


def _crunch_report_rollup(report_key: str, report: dict[str, Any]) -> dict[str, Any] | None:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary and not report.get("schema"):
        return None
    top_blocker = _top_breakdown_item(report, "blocker_reason_breakdown", "reason_breakdown")
    blocker_value = top_blocker.get("value") if isinstance(top_blocker, dict) else None
    blocker_count = _to_int(top_blocker.get("count")) if isinstance(top_blocker, dict) else 0
    candidate_count = _first_int(
        summary,
        (
            "candidate_count",
            "metadata_candidate_count",
            "recommended_count",
            "planned_count",
            "summary_model_hint_rows",
        ),
    )
    matched_count = _first_int(
        summary,
        (
            "matched_count",
            "scanned_call_count",
            "sampled_call_count",
            "turn_start_rows",
            "completed_rows",
        ),
    )
    projected_usd = _first_numeric(
        summary,
        (
            "projected_saved_usd",
            "estimated_opportunity_usd",
            "estimated_savings_usd",
            "summary_model_hint_estimated_savings_usd",
        ),
    )
    projected_tokens = _first_int(
        summary,
        (
            "projected_saved_tokens",
            "estimated_opportunity_tokens",
            "total_saved_tokens_est",
            "tokens_saved_est",
        ),
    )
    projected_chars = _first_int(
        summary,
        (
            "projected_saved_chars",
            "estimated_opportunity_saved_chars",
            "total_saved_chars",
            "saved_chars",
        ),
    )
    status = "projected-savings-ranked" if projected_usd > 0 or projected_tokens > 0 or projected_chars > 0 else "no-positive-projection"
    return {
        "report_key": report_key,
        "schema": sanitize_value(report.get("schema")),
        "status": status,
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "projected_saved_usd": round(projected_usd, 6),
        "projected_saved_tokens": projected_tokens,
        "projected_saved_chars": projected_chars,
        "top_blocker": sanitize_value(blocker_value),
        "top_blocker_count": blocker_count,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "absolute_paths_included": False,
        },
    }


def _crunch_savings_signal(stats: dict[str, Any]) -> dict[str, Any] | None:
    calls = _to_int(stats.get("today_calls") or stats.get("calls"))
    today_savings = _to_float(stats.get("today_crunch_savings_usd"))
    total_savings = _to_float(stats.get("crunch_savings_usd"))
    tokens_saved = _to_int(stats.get("crunch_tokens_saved"))
    chars_saved = _to_int(stats.get("crunch_chars_saved"))
    crunched_count = _to_int(stats.get("crunched_count"))

    reports: list[dict[str, Any]] = []
    for key in _CRUNCH_OPPORTUNITY_REPORT_KEYS:
        report = stats.get(key)
        if not isinstance(report, dict):
            continue
        rollup = _crunch_report_rollup(key, report)
        if rollup is not None:
            reports.append(rollup)
    reports.sort(
        key=lambda item: (
            _to_float(item.get("projected_saved_usd")),
            _to_int(item.get("projected_saved_tokens")),
            _to_int(item.get("projected_saved_chars")),
            _to_int(item.get("candidate_count")),
        ),
        reverse=True,
    )
    top_report = reports[0] if reports else None
    positive_projection = top_report is not None and (
        _to_float(top_report.get("projected_saved_usd")) > 0
        or _to_int(top_report.get("projected_saved_tokens")) > 0
        or _to_int(top_report.get("projected_saved_chars")) > 0
    )
    observed_positive = today_savings > 0 or total_savings > 0 or tokens_saved > 0 or chars_saved > 0

    if positive_projection:
        status = "projected-savings-ranked"
        missing: list[str] = []
    elif observed_positive:
        status = "observed-savings-ranked"
        missing = []
    elif reports:
        status = "non-positive-projection"
        missing = ["positive-projected-savings"]
    elif calls > 0:
        status = "missing-crunch-measurement"
        missing = ["crunch-opportunity-report", "positive-observed-or-projected-savings"]
    else:
        return None

    return {
        "schema": "agentflow.crunch_savings_signal.v1",
        "status": status,
        "calls": calls,
        "observed": {
            "crunched_count": crunched_count,
            "crunch_chars_saved": chars_saved,
            "crunch_tokens_saved": tokens_saved,
            "crunch_savings_usd": round(total_savings, 6),
            "today_crunch_savings_usd": round(today_savings, 6),
            "avg_crunch_ratio": round(_to_float(stats.get("avg_crunch_ratio")), 6),
        },
        "top_report": top_report,
        "report_count": len(reports),
        "reports": reports[:5],
        "missing_measurements": missing,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "absolute_paths_included": False,
        },
    }


def _breakdown_value(row: dict[str, Any]) -> str:
    for key in ("value", "reason", "code", "status", "state", "kind"):
        text = str(row.get(key) or "").strip()
        if text:
            return sanitize_value(text)
    return ""


def _breakdown_count(row: dict[str, Any]) -> int:
    for key in ("count", "rows", "candidate_count", "sample_count", "warning_count"):
        if row.get(key) is not None:
            return _to_int(row.get(key), 1)
    return 1


def _managed_health_report(stats: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "managed_recommendations",
        "managed_recommendation_report",
        "managed_recommendation_status",
    ):
        report = stats.get(key)
        if isinstance(report, dict):
            return report
    health = stats.get("managed_recommendation_health")
    if isinstance(health, dict):
        return {"recommendation_health": health}
    phase = stats.get("phase_routing")
    if isinstance(phase, dict) and isinstance(phase.get("managed_recommendation_health"), dict):
        return {"recommendation_health": phase["managed_recommendation_health"]}
    return None


def _managed_health_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    health = report.get("recommendation_health") if isinstance(report.get("recommendation_health"), dict) else {}
    latest = health.get("latest_fetch_review") if isinstance(health.get("latest_fetch_review"), dict) else {}
    rows = latest.get("rows") if isinstance(latest.get("rows"), list) else health.get("rows")
    return [row for row in (rows or []) if isinstance(row, dict)]


def _managed_omission_reason(row: dict[str, Any], *, fallback: str = "unknown") -> str:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    for source in (row, details):
        for key in ("omitted_reason", "omission_reason", "reason", "code", "value", "status", "kind"):
            text = str(source.get(key) or "").strip()
            if text:
                return sanitize_value(text)
    return fallback


def _managed_action_family(row: dict[str, Any]) -> str | None:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    texts: list[str] = []
    for source in (row, details):
        for key in (
            "local_action",
            "local_action_family",
            "action_family",
            "required_local_action",
            "policy_section",
            "expected_policy_section",
            "section",
            "type",
            "kind",
            "code",
            "reason",
            "value",
        ):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip().lower())
        families = source.get("local_action_families") or source.get("action_families")
        if isinstance(families, list):
            texts.extend(str(item).strip().lower() for item in families if str(item).strip())
    haystack = " ".join(texts)
    if not haystack:
        return None
    if any(term in haystack for term in ("prompt-replacement", "replacement_prompt", "prompt replacement")):
        return "prompt-replacement"
    if any(term in haystack for term in ("provider-body", "body-rewrite", "request-rewrite", "content-processing")):
        return "provider-body-rewrite"
    if any(term in haystack for term in ("routing_experiment", "canary", "route-canary", "routing-experiment")):
        return "routing-experiment"
    if "codex" in haystack:
        return "codex-app"
    if any(term in haystack for term in ("cache", "replay", "exact-match")):
        return "cache"
    if any(term in haystack for term in ("crunch", "compaction", "summarization", "dedup", "scaffold")):
        return "crunch"
    if any(term in haystack for term in ("routing", "route", "model")):
        return "routing"
    return None


def _local_file_backed_representation(action_family: str | None) -> dict[str, Any]:
    if action_family in _LOCAL_POLICY_REPRESENTATIONS:
        section, filename = _LOCAL_POLICY_REPRESENTATIONS[action_family]
        bundled = Path(__file__).with_name(filename)
        return {
            "exists": bundled.exists(),
            "policy_section": section,
            "rule_file": filename,
            "policy_source": "local-file-backed",
            "reason": "file-backed-local-policy",
        }
    if action_family in _UNSUPPORTED_LOCAL_ACTION_FAMILIES:
        return {
            "exists": False,
            "policy_section": None,
            "rule_file": None,
            "policy_source": None,
            "reason": "server-content-processing-not-local-policy",
        }
    return {
        "exists": False,
        "policy_section": None,
        "rule_file": None,
        "policy_source": None,
        "reason": "unknown-local-action-family",
    }


def _managed_omission_priority(reason: str, representation: dict[str, Any]) -> int:
    reason_l = reason.lower()
    if not representation.get("exists"):
        return 10
    if any(term in reason_l for term in ("safety", "privacy", "raw", "unsupported", "no-local", "omitted")):
        return 20
    if any(term in reason_l for term in ("server-error", "invalid", "threshold", "stale", "insufficient")):
        return 30
    if any(term in reason_l for term in ("disabled", "missing", "historical-null")):
        return 40
    return 50


def _managed_omission_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    reason = _managed_omission_reason(row)
    action_family = _managed_action_family(row)
    representation = _local_file_backed_representation(action_family)
    return {
        "source": source,
        "omitted_reason": reason,
        "count": _breakdown_count(row),
        "local_action_family": action_family or "unknown",
        "local_file_backed_representation": representation,
        "follow_up_owner": "local-policy" if representation.get("exists") else "blocked-boundary-review",
        "next_action": "review-local-policy-representation" if representation.get("exists") else "define-or-keep-omitted-local-action",
        "_priority": _managed_omission_priority(reason, representation),
    }


def _managed_recommendation_health_signal(stats: dict[str, Any]) -> dict[str, Any] | None:
    calls = _to_int(stats.get("today_calls") or stats.get("calls"))
    report = _managed_health_report(stats)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    config: dict[str, Any] = {}
    report_schema = None

    if report is not None:
        report_schema = sanitize_value(report.get("schema"))
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        config = report.get("current_config") if isinstance(report.get("current_config"), dict) else {}
        for row in report.get("reason_breakdown") or []:
            if isinstance(row, dict):
                reason = _breakdown_value(row)
                if reason and reason not in {"received", "applied", "ok", "success"}:
                    rows.append(_managed_omission_row(row, source="reason_breakdown"))
        for row in report.get("status_breakdown") or []:
            if isinstance(row, dict):
                status = _breakdown_value(row)
                if status and status not in {"received", "applied", "available"}:
                    rows.append(_managed_omission_row(row, source="status_breakdown"))
        for row in _managed_health_rows(report):
            rows.append(_managed_omission_row(row, source="recommendation_health"))

        if not rows and config.get("enabled") is False:
            disabled_count = _to_int(summary.get("disabled_count") or summary.get("window_calls") or calls)
            rows.append(
                _managed_omission_row(
                    {"reason": "managed-recommendations-disabled", "count": disabled_count},
                    source="current_config",
                )
            )

    if rows:
        rows.sort(key=lambda item: (_to_int(item.get("_priority"), 50), -_to_int(item.get("count"))))
        ranked = []
        for rank, row in enumerate(rows[:10], start=1):
            clean = dict(row)
            clean.pop("_priority", None)
            clean["rank"] = rank
            ranked.append(clean)
        top = ranked[0]
        status = "omission-reasons-ranked"
        missing: list[str] = []
    elif report is not None:
        ranked = []
        top = None
        status = "no-omission-reasons-reported"
        missing = []
    elif calls > 0:
        ranked = []
        top = {
            "rank": 1,
            "source": "missing_report",
            "omitted_reason": "managed-recommendation-health-report-missing",
            "count": calls,
            "local_action_family": "unknown",
            "local_file_backed_representation": _local_file_backed_representation(None),
            "follow_up_owner": "local-policy",
            "next_action": "emit-managed-recommendation-health-rollup",
        }
        status = "missing-managed-recommendation-health-report"
        missing = ["managed_recommendations_report", "omitted_local_action_reason"]
    else:
        return None

    represented = sum(1 for row in ranked if row["local_file_backed_representation"].get("exists"))
    unrepresented = len(ranked) - represented
    return {
        "schema": "agentflow.managed_recommendation_handoff_health.v1",
        "status": status,
        "source_schema": report_schema,
        "calls": calls,
        "summary": {
            "window_calls": _to_int(summary.get("window_calls") or calls),
            "metadata_rows": _to_int(summary.get("metadata_rows")),
            "received_count": _to_int(summary.get("received_count")),
            "applied_count": _to_int(summary.get("applied_count")),
            "observed_savings_usd": round(_to_float(summary.get("observed_savings_usd")), 8),
            "ranked_omission_count": len(ranked),
            "local_file_backed_count": represented,
            "no_local_representation_count": unrepresented,
            "managed_dependency": "optional",
        },
        "top_omission": top,
        "omissions": ranked,
        "missing_measurements": missing,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "absolute_paths_included": False,
        },
    }


def _request_shape_report(stats: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("request_shape_rollups", "request_shape_rollup_report", "request_shape_rollup_candidates_report"):
        report = stats.get(key)
        if isinstance(report, dict):
            return report
    return None


def _shape_row_classes(row: dict[str, Any]) -> list[str]:
    classes = row.get("candidate_work_classes")
    if isinstance(classes, list):
        cleaned = [sanitize_value(str(item)) for item in classes if str(item or "").strip()]
        if cleaned:
            return sorted(set(cleaned))
    families = [str(item) for item in row.get("candidate_families") or []]
    blockers = [str(item) for item in row.get("blocker_codes") or []]
    text_bucket = str(row.get("text_bucket") or "")
    row_count = _to_int(row.get("row_count") or row.get("count"))
    derived: set[str] = set()
    if row_count >= 2 and text_bucket in {"8k_32k_chars", "32k_128k_chars", "gte_128k_chars"}:
        derived.update({"repeated_context", "crunch"})
    if any(family in {"cache_replay", "cache_blocker"} for family in families) or any(
        blocker
        in {
            "unsupported-streaming-shape",
            "tool-call-cache-disabled",
            "semantic-cache-disabled",
            "exact-cache-miss",
            "cache-skipped",
        }
        for blocker in blockers
    ):
        derived.add("replayability")
    if "routing_candidate" in families or row.get("routing_status") == "passthrough":
        derived.add("routing")
    if "routing_evidence" in families or _to_float(row.get("observed_savings_usd")) > 0:
        derived.add("routing_evidence")
    return sorted(derived or {"observability"})


def _request_shape_next_action(classes: list[str], blockers: list[str]) -> str:
    class_set = set(classes)
    blocker_set = set(blockers)
    if "repeated_context" in class_set and "replayability" in class_set:
        return "rank-repeated-context-replayability-cohort"
    if "repeated_context" in class_set and "crunch" in class_set:
        return "rank-repeated-context-crunch-dry-run"
    if "routing" in class_set:
        return "stage-routing-lifecycle-evidence"
    if blocker_set:
        return "classify-request-shape-blocker"
    return "keep-observability-only"


def _request_shape_candidate_row(row: dict[str, Any], *, source_schema: Any, rank: int) -> dict[str, Any]:
    classes = _shape_row_classes(row)
    blockers = [sanitize_value(str(item)) for item in row.get("blocker_codes") or [] if str(item or "").strip()]
    families = [sanitize_value(str(item)) for item in row.get("candidate_families") or [] if str(item or "").strip()]
    provider = sanitize_value(row.get("provider_family") or row.get("provider") or "unknown")
    source_surface = sanitize_value(row.get("source_surface") or "unknown")
    endpoint = sanitize_value(row.get("endpoint") or "unknown")
    count = _to_int(row.get("row_count") or row.get("count"))
    cost = _to_float(row.get("cost_est_usd"))
    savings = _to_float(row.get("observed_savings_usd"))
    error_count = _to_int(row.get("error_count"))
    retry_count = _to_int(row.get("retry_count"))
    return {
        "rank": rank,
        "source_schema": sanitize_value(source_schema),
        "provider_surface_bucket": "/".join(part for part in (provider, source_surface, endpoint) if part) or "mixed",
        "provider_family": provider,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "requested_model_family": sanitize_value(row.get("requested_model_family") or "unknown"),
        "routed_model_family": sanitize_value(row.get("routed_model_family") or "unknown"),
        "category": sanitize_value(row.get("category") or "unknown"),
        "workflow_phase": sanitize_value(row.get("workflow_phase") or "unknown"),
        "stream": bool(row.get("stream")),
        "has_tools": bool(row.get("has_tools")),
        "text_bucket": sanitize_value(row.get("text_bucket") or "unknown"),
        "token_bucket": sanitize_value(row.get("token_bucket") or "unknown"),
        "cache_status": sanitize_value(row.get("cache_status") or "unknown"),
        "routing_status": sanitize_value(row.get("routing_status") or "unknown"),
        "row_count": count,
        "error_count": error_count,
        "retry_count": retry_count,
        "cost_est_usd": round(cost, 6),
        "observed_savings_usd": round(savings, 6),
        "candidate_work_classes": classes,
        "candidate_families": sorted(set(families)),
        "blocker_codes": sorted(set(blockers)),
        "next_action": _request_shape_next_action(classes, blockers),
        "_score": (
            count
            + cost * 1000.0
            + savings * 2000.0
            + (350.0 if "repeated_context" in classes else 0.0)
            + (250.0 if "replayability" in classes else 0.0)
            + (150.0 if "routing" in classes else 0.0)
            + (125.0 if "crunch" in classes else 0.0)
            - error_count * 5.0
            - retry_count * 0.5
        ),
    }


def _request_shape_rollup_signal(stats: dict[str, Any]) -> dict[str, Any] | None:
    calls = _to_int(stats.get("today_calls") or stats.get("calls"))
    report = _request_shape_report(stats)
    if report is None:
        if calls <= 0:
            return None
        return {
            "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
            "status": "missing-request-shape-rollups",
            "source_schema": None,
            "summary": {
                "calls": calls,
                "rows_considered": 0,
                "rollup_count": 0,
                "ranked_candidate_count": 0,
                "top_next_action": "emit-request-shape-rollups",
            },
            "top_candidate": None,
            "candidates": [],
            "missing_measurements": ["request_shape_rollups"],
            "privacy": _candidate_privacy(),
        }

    rollups = [row for row in report.get("rollups") or report.get("candidates") or [] if isinstance(row, dict)]
    source_schema = report.get("schema")
    ranked = [
        _request_shape_candidate_row(row, source_schema=source_schema, rank=index)
        for index, row in enumerate(rollups, start=1)
    ]
    ranked.sort(key=lambda item: (_to_float(item.get("_score")), _to_int(item.get("row_count"))), reverse=True)
    class_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for rank, row in enumerate(ranked[:10], start=1):
        row["rank"] = rank
        for value in row.get("candidate_work_classes") or []:
            class_counts[str(value)] += _to_int(row.get("row_count"))
        for value in row.get("blocker_codes") or []:
            blocker_counts[str(value)] += _to_int(row.get("row_count"))
    clean_ranked = []
    for row in ranked[:10]:
        clean = dict(row)
        clean.pop("_score", None)
        clean_ranked.append(clean)
    report_summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    replay_dry_run = report.get("cache_replayability_dry_run") if isinstance(report.get("cache_replayability_dry_run"), dict) else None
    replay_summary = replay_dry_run.get("summary") if isinstance(replay_dry_run, dict) and isinstance(replay_dry_run.get("summary"), dict) else {}
    replay_cohorts = replay_dry_run.get("cohorts") if isinstance(replay_dry_run, dict) and isinstance(replay_dry_run.get("cohorts"), list) else []
    status = "candidates-ranked" if clean_ranked else "no-request-shape-candidates"
    result = {
        "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
        "status": status,
        "source_schema": sanitize_value(source_schema),
        "summary": {
            "calls": calls,
            "rows_considered": _to_int(report_summary.get("rows_considered") or report_summary.get("scanned_rows")),
            "rollup_count": _to_int(report_summary.get("rollup_count") or len(rollups)),
            "ranked_candidate_count": len(clean_ranked),
            "top_next_action": clean_ranked[0]["next_action"] if clean_ranked else None,
            "cache_replayability_top_blocker": sanitize_value(replay_summary.get("top_blocker_code"))
            if replay_summary
            else None,
            "cache_replayability_replay_ready_cohort_count": _to_int(replay_summary.get("replay_ready_cohort_count"))
            if replay_summary
            else 0,
            "cache_replayability_skipped_cohort_count": _to_int(replay_summary.get("skipped_cohort_count"))
            if replay_summary
            else 0,
            "class_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "blocker_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
        },
        "top_candidate": clean_ranked[0] if clean_ranked else None,
        "candidates": clean_ranked,
        "missing_measurements": [] if clean_ranked else ["ranked_request_shape_rollup"],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "absolute_paths_included": False,
        },
    }
    if replay_dry_run is not None:
        result["cache_replayability_dry_run"] = {
            "schema": sanitize_value(replay_dry_run.get("schema")),
            "status": sanitize_value(replay_dry_run.get("status")),
            "summary": sanitize_value(replay_summary),
            "readiness_breakdown": sanitize_value(replay_dry_run.get("readiness_breakdown") or []),
            "skipped_reason_breakdown": sanitize_value(replay_dry_run.get("skipped_reason_breakdown") or []),
            "blocker_breakdown": sanitize_value(replay_dry_run.get("blocker_breakdown") or []),
            "cohorts": sanitize_value(replay_cohorts[:5]),
            "privacy": sanitize_value(replay_dry_run.get("privacy") if isinstance(replay_dry_run.get("privacy"), dict) else {}),
        }
    return result


def _loop_privacy() -> dict[str, Any]:
    privacy = _candidate_privacy()
    privacy["cache_keys_included"] = False
    return privacy


def _loop_state_rank(state: str) -> int:
    return {
        "activation-ready": 0,
        "replay-ready": 1,
        "measured-savings": 2,
        "projected-savings": 3,
        "ranked-evidence": 4,
        "missing-evidence": 5,
        "blocked": 6,
        "no-op": 7,
    }.get(state, 9)


def _loop_missing_state(state: str) -> bool:
    return state in {"missing-evidence", "blocked"}


def _loop_progress_state(state: str) -> bool:
    return state in {"activation-ready", "replay-ready", "measured-savings", "projected-savings", "ranked-evidence"}


def _routing_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    canary_row, source = _top_openai_routing_canary_row(stats_summary)
    if canary_row is not None:
        action = _openai_routing_candidate_action(canary_row, stats_summary)
        counts = canary_row.get("cohort_counts") if isinstance(canary_row.get("cohort_counts"), dict) else {}
        state = "activation-ready" if action["activation_ready"] else "blocked"
        next_action = str(canary_row.get("next_action") or "inspect-openai-routing-canary")
        return {
            "lever": "routing",
            "state": state,
            "evidence_source": source,
            "local_action_family": "routing",
            "next_action": next_action if state == "activation-ready" else "resolve-openai-routing-canary-blocker",
            "blocker_codes": [] if state == "activation-ready" else [action["omission_reason"]],
            "sample_count": _to_int(canary_row.get("sample_count")) or sum(_to_int(value) for value in counts.values()),
            "applied_count": _to_int(counts.get("canary_applied")),
            "holdout_count": _to_int(counts.get("canary_holdout")),
            "savings_per_1000_calls_usd": action["savings_per_1000_calls_usd"],
        }

    report = stats_summary.get("pass_through_routing_report")
    if not isinstance(report, dict):
        return None
    buckets = [row for row in report.get("buckets") or [] if isinstance(row, dict)]
    actionable = [row for row in buckets if row.get("actionability") == "actionable"]
    top = (actionable or buckets or [None])[0]
    if top is None:
        return None
    lifecycle = top.get("openai_canary_lifecycle_evidence") if isinstance(top.get("openai_canary_lifecycle_evidence"), dict) else {}
    blockers = [str(item) for item in lifecycle.get("blocker_codes") or [] if str(item or "").strip()]
    actionability = str(top.get("actionability") or "unknown")
    if actionability == "actionable" and blockers:
        state = "missing-evidence"
        next_action = "activate-openai-routing-canary-cohorts"
    elif actionability == "actionable":
        state = "ranked-evidence"
        next_action = "stage-openai-routing-canary"
    else:
        state = "no-op" if actionability == "already-cheapest" else "blocked"
        next_action = "keep-routing-no-op-reason"
    return {
        "lever": "routing",
        "state": state,
        "evidence_source": report.get("schema"),
        "local_action_family": "routing",
        "next_action": next_action,
        "actionability": actionability,
        "blocker_codes": blockers or ([str(top.get("no_op_reason"))] if top.get("no_op_reason") else []),
        "sample_count": _to_int(top.get("sample_count")),
        "requested_model": top.get("requested_model"),
        "candidate_target_model": top.get("candidate_target_model"),
        "savings_per_1000_calls_usd": _to_float(top.get("estimated_savings_per_1000_calls_usd")),
    }


def _cache_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    cohorts = stats_summary.get("cache_replay_cohort_ranking")
    if isinstance(cohorts, dict):
        summary = cohorts.get("summary") if isinstance(cohorts.get("summary"), dict) else {}
        rows = [row for row in cohorts.get("cohorts") or [] if isinstance(row, dict)]
        top = rows[0] if rows else {}
        readiness = str(top.get("readiness") or "")
        activation_ready = _to_int(summary.get("activation_ready_count")) > 0 or readiness == "activation-ready"
        state = "replay-ready" if activation_ready else "missing-evidence"
        blockers = [str(item) for item in top.get("blocker_reasons") or [] if str(item or "").strip()]
        return {
            "lever": "cache",
            "state": state,
            "evidence_source": cohorts.get("schema"),
            "local_action_family": "cache",
            "next_action": "stage-cache-replay-canary" if activation_ready else "dry-run-cache-replayability",
            "blocker_codes": [] if activation_ready else blockers or [str(top.get("readiness") or "cache-replay-evidence-missing")],
            "sample_count": _to_int(top.get("count") or summary.get("candidate_rows")),
            "projected_hits": _to_int(top.get("projected_hits") or summary.get("projected_ready_hits")),
            "projected_saved_cost_usd": round(_to_float(top.get("projected_saved_cost_usd")), 8),
        }

    shape_signal = stats_summary.get("request_shape_rollup_candidates")
    shape_replay = shape_signal.get("cache_replayability_dry_run") if isinstance(shape_signal, dict) else None
    if isinstance(shape_replay, dict):
        summary = shape_replay.get("summary") if isinstance(shape_replay.get("summary"), dict) else {}
        rows = [row for row in shape_replay.get("cohorts") or [] if isinstance(row, dict)]
        top = rows[0] if rows else {}
        ready = _to_int(summary.get("replay_ready_cohort_count")) > 0 or top.get("readiness") == "replay-ready"
        blockers = [str(item) for item in top.get("blockers") or [] if str(item or "").strip()]
        top_blocker = str(summary.get("top_blocker_code") or (blockers[0] if blockers else "cache-replayability-evidence-missing"))
        return {
            "lever": "cache",
            "state": "replay-ready" if ready else "missing-evidence",
            "evidence_source": shape_replay.get("schema"),
            "local_action_family": "cache",
            "next_action": "stage-cache-replay-canary" if ready else "resolve-cache-replayability-blocker",
            "blocker_codes": [] if ready else blockers or [top_blocker],
            "sample_count": _to_int(top.get("row_count") or summary.get("rows_considered")),
            "projected_hits": _to_int(top.get("projected_hits") or summary.get("projected_hits")),
            "projected_saved_cost_usd": round(_to_float(top.get("projected_savings_usd") or summary.get("projected_savings_usd")), 8),
        }

    ladder = stats_summary.get("cache_zero_hit_blocker_ladder")
    if isinstance(ladder, dict):
        summary = ladder.get("summary") if isinstance(ladder.get("summary"), dict) else {}
        rows = [row for row in ladder.get("ladder") or [] if isinstance(row, dict)]
        top = rows[0] if rows else {}
        blocker = str(top.get("blocker_code") or summary.get("top_blocker_code") or "zero-cache-hits")
        return {
            "lever": "cache",
            "state": "missing-evidence",
            "evidence_source": ladder.get("schema"),
            "local_action_family": "cache",
            "next_action": str(top.get("next_action_family") or summary.get("top_next_action_family") or "dry-run-cache-replayability"),
            "blocker_codes": [blocker],
            "sample_count": _to_int(top.get("count") or summary.get("scanned_rows")),
            "cache_hits": _to_int(summary.get("cache_hits")),
        }

    calls = _to_int(stats_summary.get("calls") or stats_summary.get("today_calls"))
    if calls <= 0:
        return None
    cache_hits = _to_int(stats_summary.get("cache_hits"))
    return {
        "lever": "cache",
        "state": "ranked-evidence" if cache_hits > 0 else "missing-evidence",
        "evidence_source": "stats_summary",
        "local_action_family": "cache",
        "next_action": "inspect-cache-hit-cohorts" if cache_hits > 0 else "dry-run-cache-replayability",
        "blocker_codes": [] if cache_hits > 0 else ["zero-cache-hits"],
        "sample_count": calls,
        "cache_hits": cache_hits,
    }


def _crunch_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = stats_summary.get("crunch_savings_signal")
    if not isinstance(signal, dict):
        return None
    status = str(signal.get("status") or "")
    observed = signal.get("observed") if isinstance(signal.get("observed"), dict) else {}
    top_report = signal.get("top_report") if isinstance(signal.get("top_report"), dict) else {}
    if status == "observed-savings-ranked":
        state = "measured-savings"
        next_action = "produce-crunch-activation-follow-up"
    elif status == "projected-savings-ranked":
        state = "projected-savings"
        next_action = "produce-crunch-opportunity-measurements"
    else:
        state = "missing-evidence"
        next_action = "emit-crunch-opportunity-report"
    return {
        "lever": "crunch",
        "state": state,
        "evidence_source": signal.get("schema"),
        "local_action_family": "crunch",
        "next_action": next_action,
        "blocker_codes": [str(item) for item in signal.get("missing_measurements") or [] if str(item or "").strip()],
        "sample_count": _to_int(signal.get("calls")),
        "crunch_savings_usd": round(_to_float(observed.get("crunch_savings_usd")), 8),
        "today_crunch_savings_usd": round(_to_float(observed.get("today_crunch_savings_usd")), 8),
        "projected_saved_usd": round(_to_float(top_report.get("projected_saved_usd")), 8),
    }


def _request_shape_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = stats_summary.get("request_shape_rollup_candidates")
    if not isinstance(signal, dict):
        return None
    summary = signal.get("summary") if isinstance(signal.get("summary"), dict) else {}
    top = signal.get("top_candidate") if isinstance(signal.get("top_candidate"), dict) else {}
    state = "ranked-evidence" if str(signal.get("status") or "") == "candidates-ranked" and top else "missing-evidence"
    return {
        "lever": "request-shape-rollups",
        "state": state,
        "evidence_source": signal.get("source_schema") or signal.get("schema"),
        "local_action_family": "cohort-ranking",
        "next_action": str(summary.get("top_next_action") or "emit-request-shape-rollups"),
        "blocker_codes": [str(item) for item in signal.get("missing_measurements") or [] if str(item or "").strip()]
        or [str(item) for item in top.get("blocker_codes") or [] if str(item or "").strip()],
        "sample_count": _to_int(top.get("row_count") or summary.get("rows_considered")),
        "ranked_candidate_count": _to_int(summary.get("ranked_candidate_count")),
    }


def _evidence_to_activation_loop(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    stages = [
        stage
        for stage in (
            _routing_loop_stage(stats_summary),
            _cache_loop_stage(stats_summary),
            _crunch_loop_stage(stats_summary),
            _request_shape_loop_stage(stats_summary),
        )
        if stage is not None
    ]
    if not stages:
        return None

    stages.sort(
        key=lambda item: (
            _loop_state_rank(str(item.get("state") or "")),
            -_to_float(item.get("savings_per_1000_calls_usd") or item.get("projected_saved_cost_usd") or item.get("projected_saved_usd")),
            -_to_int(item.get("sample_count")),
            str(item.get("lever") or ""),
        )
    )
    missing_count = sum(1 for stage in stages if _loop_missing_state(str(stage.get("state") or "")))
    progressed_count = sum(1 for stage in stages if _loop_progress_state(str(stage.get("state") or "")))
    activation_ready_count = sum(1 for stage in stages if str(stage.get("state")) in {"activation-ready", "replay-ready"})
    if activation_ready_count:
        status = "activation-ready"
    elif progressed_count:
        status = "evidence-progress"
    else:
        status = "missing-evidence"

    return {
        "schema": "agentflow.evidence_to_activation_savings_loop.v1",
        "status": status,
        "summary": {
            "tracked_lever_count": len(stages),
            "missing_evidence_count": missing_count,
            "progressed_count": progressed_count,
            "activation_ready_count": activation_ready_count,
            "top_lever": stages[0].get("lever"),
            "top_state": stages[0].get("state"),
            "top_next_action": stages[0].get("next_action"),
        },
        "levers": stages,
        "privacy": _loop_privacy(),
    }


def _candidate_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "absolute_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "individual_candidate_ids_included": False,
    }


def _surface_bucket(row: dict[str, Any], *, fallback: str = "mixed") -> str:
    provider = str(row.get("provider") or "").strip().lower()
    source_surface = str(row.get("source_surface") or row.get("surface") or "").strip().lower()
    endpoint = str(row.get("endpoint") or "").strip().lower()
    parts = [part for part in (provider, source_surface, endpoint) if part]
    return "/".join(parts) if parts else fallback


def _routing_text(value: Any) -> str:
    return str(value or "").strip()


def _routing_lower(value: Any) -> str:
    return _routing_text(value).lower()


def _routing_skip_reason(row: dict[str, Any]) -> str:
    for key in ("skip_reason", "routing_skip_reason", "reason", "blocker", "omission_reason", "omitted_reason"):
        text = _routing_text(row.get(key))
        if text:
            return sanitize_value(text)
    return ""


def _routing_candidate_target(provider: str, requested_model: str, category: str, phase: str) -> tuple[str | None, str | None, str]:
    requested = requested_model.lower()
    category = category.lower()
    phase = phase.lower()
    if provider == "openai":
        if requested == "gpt-5.4":
            return "gpt-5.4-mini", "openai-routing-canary", "gpt-5.4 canary can evaluate gpt-5.4-mini on local metadata cohorts"
        return None, None, "no lower safe OpenAI broad-routing target is available from aggregate metadata"

    if provider == "anthropic":
        if "opus" in requested:
            return "claude-sonnet-4-6", "anthropic-routing-rules", "Opus pass-through may have a Sonnet downgrade path after quality evidence"
        if "sonnet" in requested and (category in {"tool-result", "tool-light", "short-completion", "summary"} or phase in {"tool-execution", "summary"}):
            return "claude-haiku-4-5-20251001", "anthropic-routing-rules", "phase/category metadata matches existing local Haiku executor shapes"
        return None, None, "Anthropic aggregate bucket needs phase or thinking/tool safety evidence before downgrade"

    return None, None, "provider/action pair has no local routing executor"


def _already_cheapest_reason(provider: str, requested_model: str) -> str | None:
    requested = requested_model.lower()
    if provider == "openai" and requested in {"gpt-5.4-mini", "gpt-5-mini", "gpt-5-nano"}:
        return "already at the lowest broad safe OpenAI tier represented by aggregate routing metadata"
    if provider == "anthropic" and "haiku" in requested:
        return "already at the lowest broad Anthropic tier represented by aggregate routing metadata"
    return None


def _estimated_savings_per_1000(provider: str, requested_model: str, target_model: str | None) -> float:
    if not target_model:
        return 0.0
    requested = estimate_cost(requested_model, 1000, 250, provider=provider)
    target = estimate_cost(target_model, 1000, 250, provider=provider)
    if requested is None or target is None:
        return 0.0
    return round(max(0.0, requested - target) * 1000.0, 6)


def _aggregate_openai_canary_lifecycle_evidence(row: dict[str, Any], *, sample_count: int) -> dict[str, Any]:
    applied = _to_int(row.get("openai_canary_applied_count") or row.get("canary_applied_count"))
    holdout = _to_int(row.get("openai_canary_holdout_count") or row.get("canary_holdout_count"))
    safety_stopped = _to_int(row.get("openai_canary_safety_stopped_count") or row.get("safety_stopped_count"))
    skipped = _to_int(row.get("openai_canary_skipped_count") or row.get("canary_skipped_count"))
    bypassed = _to_int(row.get("openai_canary_bypassed_count") or row.get("canary_bypassed_count"))
    unknown = _to_int(row.get("openai_canary_unknown_count") or row.get("canary_unknown_count"))
    error_count = _to_int(row.get("openai_canary_error_count") or row.get("canary_error_count"))
    retry_count = _to_int(row.get("openai_canary_retry_count") or row.get("canary_retry_count"))
    fallback_count = _to_int(row.get("openai_canary_fallback_count") or row.get("canary_fallback_count"))
    observed = applied + holdout + safety_stopped + skipped + bypassed + unknown
    stale = bool(row.get("openai_canary_stale_evidence") or row.get("stale_evidence"))

    blockers: Counter[str] = Counter()
    if observed == 0:
        blockers["missing-canary-lifecycle-evidence"] = sample_count
    if applied == 0:
        blockers["missing-applied-coverage"] = sample_count
    if holdout == 0:
        blockers["missing-holdout-coverage"] = sample_count
    if error_count:
        blockers["error-observed"] = error_count
    if retry_count:
        blockers["retry-observed"] = retry_count
    if fallback_count:
        blockers["fallback-observed"] = fallback_count
    if safety_stopped:
        blockers["safety-stop-observed"] = safety_stopped
    if stale:
        blockers["stale-evidence"] = observed or sample_count

    return {
        "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
        "status": "matched" if observed else "no-openai-canary-metadata",
        "observed_count": observed,
        "cohort_counts": {
            "canary_applied": applied,
            "canary_holdout": holdout,
            "safety_stopped": safety_stopped,
            "skipped": skipped,
            "bypassed_or_disabled": bypassed,
            "unknown": unknown,
        },
        "coverage": {
            "matched_count": sample_count,
            "observed_rate": round(observed / sample_count, 6) if sample_count else 0.0,
            "applied_rate": round(applied / sample_count, 6) if sample_count else 0.0,
            "holdout_rate": round(holdout / sample_count, 6) if sample_count else 0.0,
        },
        "error_count": error_count,
        "retry_count": retry_count,
        "fallback_count": fallback_count,
        "latest_observed_at": sanitize_value(row.get("openai_canary_latest_observed_at") or row.get("canary_latest_observed_at")),
        "stale_evidence": {
            "stale": stale,
            "max_age_hours": 72.0,
        },
        "blocker_codes": sorted(blockers),
        "blocker_reason_breakdown": [
            {"value": key, "count": value}
            for key, value in sorted(blockers.items(), key=lambda item: (-item[1], item[0]))
        ],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "absolute_paths_included": False,
        },
    }


def _classify_pass_through_bucket(row: dict[str, Any]) -> dict[str, Any]:
    provider = _routing_lower(row.get("provider")) or "unknown"
    requested = _routing_text(row.get("requested_model")) or "unknown"
    routed = row.get("routed_model")
    routed_text = _routing_text(routed)
    category = _routing_text(row.get("category")) or "unknown"
    phase = _routing_text(row.get("phase") or row.get("workflow_phase")) or "unknown"
    skip_reason = _routing_skip_reason(row)
    count = _to_int(row.get("c") or row.get("count"))
    source_surface = _routing_text(row.get("source_surface") or row.get("surface")) or "unknown"
    endpoint = _routing_text(row.get("endpoint")) or "unknown"
    target_model, executor, target_reason = _routing_candidate_target(provider, requested, category, phase)
    cheapest_reason = _already_cheapest_reason(provider, requested)
    savings_per_1000 = _estimated_savings_per_1000(provider, requested, target_model)
    cost_known = bool(pricing_basis(requested, provider=provider).get("cost_known")) if provider in {"openai", "anthropic"} else False
    skip_l = skip_reason.lower()

    actionability = "needs-lifecycle-evidence"
    no_op_reason = ""
    if provider not in {"openai", "anthropic"} or not cost_known:
        actionability = "unsupported-provider-action"
        no_op_reason = "missing supported provider pricing or local routing executor"
        target_model = None
        executor = None
        savings_per_1000 = 0.0
    elif routed is None:
        actionability = "unsupported-provider-action"
        no_op_reason = "routed model metadata is missing for this bucket"
        target_model = None
        executor = None
        savings_per_1000 = 0.0
    elif cheapest_reason:
        actionability = "already-cheapest"
        no_op_reason = cheapest_reason
        target_model = None
        executor = None
        savings_per_1000 = 0.0
    elif any(term in skip_l for term in ("thinking", "tool-protocol", "safety", "unsafe")):
        actionability = "safety-blocked"
        no_op_reason = skip_reason or "safety metadata blocks automatic routing activation"
    elif target_model and savings_per_1000 > 0:
        actionability = "actionable"
    elif target_model:
        actionability = "needs-lifecycle-evidence"
        no_op_reason = "candidate target has no positive aggregate savings estimate yet"
    else:
        actionability = "needs-lifecycle-evidence" if provider in {"openai", "anthropic"} else "unsupported-provider-action"
        no_op_reason = target_reason

    return {
        "provider": provider,
        "source_surface": sanitize_value(source_surface),
        "endpoint": sanitize_value(endpoint),
        "requested_model": sanitize_value(requested),
        "routed_model": sanitize_value(routed_text or None),
        "category": sanitize_value(category),
        "workflow_phase": sanitize_value(phase),
        "skip_reason": sanitize_value(skip_reason or None),
        "sample_count": count,
        "actionability": actionability,
        "candidate_target_model": sanitize_value(target_model),
        "required_local_executor": sanitize_value(executor),
        "candidate_reason": sanitize_value(target_reason),
        "no_op_reason": sanitize_value(no_op_reason or None),
        "estimated_savings_per_1000_calls_usd": savings_per_1000,
        "estimate_basis": "1000 input tokens and 250 output tokens per reference call; aggregate bucket ranking only",
        "openai_canary_lifecycle_evidence": _aggregate_openai_canary_lifecycle_evidence(row, sample_count=count)
        if provider == "openai" and target_model
        else None,
    }


def _bucket_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    action_rank = {
        "actionable": 0,
        "needs-lifecycle-evidence": 1,
        "safety-blocked": 2,
        "unsupported-provider-action": 3,
        "already-cheapest": 4,
    }.get(str(row.get("actionability")), 9)
    return (
        action_rank,
        -_to_float(row.get("estimated_savings_per_1000_calls_usd")),
        -_to_int(row.get("sample_count")),
        str(row.get("requested_model") or ""),
    )


def _pass_through_routing_report(routing_rows: Any, *, limit: int = 10) -> dict[str, Any] | None:
    if not isinstance(routing_rows, list):
        return None
    buckets: list[dict[str, Any]] = []
    total_rows = 0
    pass_through_rows = 0
    routed_down_rows = 0
    for raw in routing_rows:
        if not isinstance(raw, dict):
            continue
        count = _to_int(raw.get("c") or raw.get("count"))
        if count <= 0:
            continue
        total_rows += count
        requested = raw.get("requested_model")
        routed = raw.get("routed_model")
        if routed is not None and requested != routed:
            routed_down_rows += count
            continue
        pass_through_rows += count
        buckets.append(_classify_pass_through_bucket(raw))

    if pass_through_rows <= 0:
        return None

    buckets.sort(key=_bucket_sort_key)
    actionability_counts: Counter[str] = Counter()
    lifecycle_totals: Counter[str] = Counter()
    for bucket in buckets:
        actionability_counts[str(bucket.get("actionability") or "unknown")] += _to_int(bucket.get("sample_count"))
        lifecycle = bucket.get("openai_canary_lifecycle_evidence")
        if isinstance(lifecycle, dict):
            cohorts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
            lifecycle_totals["canary_applied"] += _to_int(cohorts.get("canary_applied"))
            lifecycle_totals["canary_holdout"] += _to_int(cohorts.get("canary_holdout"))
            lifecycle_totals["safety_stopped"] += _to_int(cohorts.get("safety_stopped"))
            lifecycle_totals["error"] += _to_int(lifecycle.get("error_count"))
            lifecycle_totals["retry"] += _to_int(lifecycle.get("retry_count"))
            lifecycle_totals["fallback"] += _to_int(lifecycle.get("fallback_count"))
    ranked = []
    for rank, bucket in enumerate(buckets[: max(1, limit)], start=1):
        item = dict(bucket)
        item["rank"] = rank
        ranked.append(item)
    top = ranked[0] if ranked else {}
    return {
        "schema": "agentflow.pass_through_routing_activation_candidates.v1",
        "summary": {
            "routing_rows_scanned": total_rows,
            "pass_through_rows": pass_through_rows,
            "routed_down_rows": routed_down_rows,
            "candidate_bucket_count": len(buckets),
            "top_actionability": top.get("actionability"),
            "top_requested_model": top.get("requested_model"),
            "top_candidate_target_model": top.get("candidate_target_model"),
            "openai_canary_applied_count": lifecycle_totals["canary_applied"],
            "openai_canary_holdout_count": lifecycle_totals["canary_holdout"],
            "openai_canary_safety_stopped_count": lifecycle_totals["safety_stopped"],
            "openai_canary_error_count": lifecycle_totals["error"],
            "openai_canary_retry_count": lifecycle_totals["retry"],
            "openai_canary_fallback_count": lifecycle_totals["fallback"],
        },
        "actionability_breakdown": [
            {"class": key, "count": value}
            for key, value in sorted(actionability_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "buckets": ranked,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "absolute_paths_included": False,
        },
    }


def _candidate(
    *,
    lever: str,
    provider_surface_bucket: str,
    blocker: str,
    estimated_savings_path: str,
    projected_savings_signal: dict[str, Any],
    confidence: str,
    sequencing: str,
    repo: str = "lutzkuen/agentflow",
    safety_status: str = "review-required",
    score: float = 0.0,
) -> dict[str, Any]:
    return {
        "lever": lever,
        "provider_surface_bucket": sanitize_value(provider_surface_bucket),
        "blocker": sanitize_value(blocker),
        "estimated_savings_path": sanitize_value(estimated_savings_path),
        "projected_savings_signal": sanitize_value(projected_savings_signal),
        "confidence": confidence,
        "safety_status": safety_status,
        "privacy": _candidate_privacy(),
        "repo": repo,
        "sequencing": sanitize_value(sequencing),
        "_score": score,
    }


def _routing_candidate(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    canary_row, _ = _top_openai_routing_canary_row(stats_summary)
    if canary_row is not None:
        action = _openai_routing_candidate_action(canary_row, stats_summary)
        impact = stats_summary.get("openai_canary_impact") if isinstance(stats_summary.get("openai_canary_impact"), dict) else {}
        feedback = impact.get("activation_lifecycle_feedback") if isinstance(impact.get("activation_lifecycle_feedback"), dict) else {}
        counts = canary_row.get("cohort_counts") if isinstance(canary_row.get("cohort_counts"), dict) else {}
        reason_codes = [str(reason) for reason in canary_row.get("reason_codes") or []]
        blocker = (
            "openai-routing-canary-ready"
            if action["activation_ready"]
            else f"openai-routing-canary-{action['omission_reason']}"
        )
        return _candidate(
            lever="routing",
            provider_surface_bucket=_openai_routing_canary_bucket(canary_row),
            blocker=blocker,
            estimated_savings_path=(
                "Convert OpenAI routing canary lifecycle evidence into a reviewed local canary widening or policy-bundle action."
                if action["activation_ready"]
                else "Resolve the OpenAI routing canary blocker before local routing activation or policy-bundle review."
            ),
            projected_savings_signal={
                "verdict": canary_row.get("verdict"),
                "next_action": canary_row.get("next_action"),
                "reason_codes": reason_codes,
                "applied_count": _to_int(counts.get("canary_applied")),
                "holdout_count": _to_int(counts.get("canary_holdout")),
                "safety_stopped_count": _to_int(counts.get("safety_stopped")),
                "savings_per_1000_calls_usd": action["savings_per_1000_calls_usd"],
                "omission_reason": action["omission_reason"],
                "cohort_lifecycle_metadata": _cohort_lifecycle_metadata(feedback, limit=5),
            },
            confidence="high" if action["activation_ready"] else "medium",
            sequencing="Use canary lifecycle feedback before generic pass-through routing issues so activation work cites applied, holdout, and safety evidence.",
            safety_status="review-required" if action["activation_ready"] else "blocked",
            score=10_000.0 + action["savings_per_1000_calls_usd"],
        )

    pass_through_report = stats_summary.get("pass_through_routing_report")
    if isinstance(pass_through_report, dict):
        buckets = [row for row in pass_through_report.get("buckets") or [] if isinstance(row, dict)]
        actionable = [row for row in buckets if row.get("actionability") == "actionable"]
        top = (actionable or buckets or [None])[0]
        if top is not None:
            actionability = str(top.get("actionability") or "unknown")
            count = _to_int(top.get("sample_count"))
            target = str(top.get("candidate_target_model") or "")
            requested = str(top.get("requested_model") or "unknown")
            blocker = (
                f"pass-through-routing-{actionability}"
                if actionability != "actionable"
                else "pass-through-routing-activation-candidate"
            )
            return _candidate(
                lever="routing",
                provider_surface_bucket=_surface_bucket(top, fallback="mixed"),
                blocker=blocker,
                estimated_savings_path=(
                    f"Stage a local routing canary from {requested} to {target} for the ranked pass-through bucket."
                    if target
                    else f"Keep {requested} pass-through explicit with a no-op reason before creating activation work."
                ),
                projected_savings_signal={
                    "report_schema": pass_through_report.get("schema"),
                    "actionability": actionability,
                    "sample_count": count,
                    "requested_model": requested,
                    "candidate_target_model": top.get("candidate_target_model"),
                    "required_local_executor": top.get("required_local_executor"),
                    "estimated_savings_per_1000_calls_usd": top.get("estimated_savings_per_1000_calls_usd"),
                    "no_op_reason": top.get("no_op_reason"),
                    "actionability_breakdown": pass_through_report.get("actionability_breakdown"),
                },
                confidence="medium" if actionability == "actionable" else "low",
                sequencing="Use the ranked pass-through bucket before generic routing activation issues; already-cheapest and safety-blocked buckets should remain explicit no-ops.",
                safety_status="review-required" if actionability == "actionable" else "blocked",
                score=float(count) + (1000.0 if actionability == "actionable" else 0.0),
            )

    routing_rows = stats_summary.get("routing_top")
    if not isinstance(routing_rows, list) or not routing_rows:
        calls = _to_int(stats_summary.get("calls"))
        if calls <= 0:
            return None
        return _candidate(
            lever="routing",
            provider_surface_bucket="mixed",
            blocker="missing-routing-breakdown",
            estimated_savings_path="Collect provider/model routing breakdowns before choosing a downgrade or canary activation issue.",
            projected_savings_signal={"calls": calls, "routed_down_calls": 0, "pass_through_calls": calls},
            confidence="low",
            sequencing="Add routing candidate evidence after cache and repeated-diagnostic blockers if no model bucket dominates.",
            score=float(calls) * 0.2,
        )

    pass_through = 0
    routed_down = 0
    top_bucket = "mixed"
    top_count = 0
    for row in routing_rows:
        if not isinstance(row, dict):
            continue
        count = _to_int(row.get("c") or row.get("count"))
        requested = row.get("requested_model")
        routed = row.get("routed_model")
        if count > top_count:
            top_count = count
            top_bucket = _surface_bucket(row, fallback="mixed")
        if routed is None or requested == routed:
            pass_through += count
        else:
            routed_down += count

    if pass_through <= 0:
        return None
    total = pass_through + routed_down
    confidence = "medium" if total and pass_through / total >= 0.8 else "low"
    return _candidate(
        lever="routing",
        provider_surface_bucket=top_bucket,
        blocker="high-pass-through-routing",
        estimated_savings_path="Prioritize canary lifecycle or rule activation work for high-volume model buckets that still forward unchanged.",
        projected_savings_signal={
            "top_bucket_calls": top_count,
            "pass_through_calls": pass_through,
            "routed_down_calls": routed_down,
            "pass_through_share": round(pass_through / total, 4) if total else 0.0,
        },
        confidence=confidence,
        sequencing="Sequence after candidate schema support and before issue-body generation so routing issues cite the dominant pass-through bucket.",
        score=float(pass_through),
    )


def _cache_candidate(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    cache_hits = _to_int(stats_summary.get("cache_hits"))
    cache_hit_rate = _to_float(stats_summary.get("cache_hit_rate"))
    calls = _to_int(stats_summary.get("calls") or stats_summary.get("today_calls"))
    breakdown = stats_summary.get("cache_decision_breakdown_top")
    top_blocker = "zero-cache-hits" if cache_hits == 0 and calls > 0 else "cache-hit-rate-low"
    top_bucket = "mixed"
    top_count = 0
    if isinstance(breakdown, list) and breakdown:
        for row in breakdown:
            if not isinstance(row, dict):
                continue
            count = _to_int(row.get("count"))
            if count > top_count:
                top_count = count
                top_bucket = _surface_bucket(row, fallback=str(row.get("source_surface") or "mixed"))
                status = str(row.get("status") or "unknown")
                reason = str(row.get("reason") or "unknown")
                top_blocker = f"cache-{status}-{reason}"

    if calls <= 0 and top_count <= 0:
        return None
    if cache_hits > 0 and cache_hit_rate >= 0.05 and not breakdown:
        return None
    return _candidate(
        lever="cache",
        provider_surface_bucket=top_bucket,
        blocker=top_blocker,
        estimated_savings_path="Turn the largest cache skip or miss cohort into replayability, invalidation, or exact-cache activation work.",
        projected_savings_signal={
            "calls": calls,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "top_blocker_count": top_count,
        },
        confidence="medium" if cache_hits == 0 and (calls or top_count) else "low",
        sequencing="File after the candidate schema exists, before cache replay issue generation, so activation issues target the largest blocker cohort.",
        score=float(top_count or calls) * (1.0 - min(max(cache_hit_rate, 0.0), 1.0)),
    )


def _crunch_candidate(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = stats_summary.get("crunch_savings_signal") if isinstance(stats_summary.get("crunch_savings_signal"), dict) else {}
    observed = signal.get("observed") if isinstance(signal.get("observed"), dict) else {}
    top_report = signal.get("top_report") if isinstance(signal.get("top_report"), dict) else {}
    today_savings = _to_float(observed.get("today_crunch_savings_usd") if observed else stats_summary.get("today_crunch_savings_usd"))
    total_savings = _to_float(observed.get("crunch_savings_usd") if observed else stats_summary.get("crunch_savings_usd"))
    tokens_saved = _to_int(observed.get("crunch_tokens_saved") if observed else stats_summary.get("crunch_tokens_saved"))
    chars_saved = _to_int(observed.get("crunch_chars_saved") if observed else stats_summary.get("crunch_chars_saved"))
    projected_usd = _to_float(top_report.get("projected_saved_usd"))
    projected_tokens = _to_int(top_report.get("projected_saved_tokens"))
    projected_chars = _to_int(top_report.get("projected_saved_chars"))
    calls = _to_int(signal.get("calls") or stats_summary.get("today_calls") or stats_summary.get("calls"))
    if calls <= 0 and today_savings <= 0 and total_savings <= 0 and projected_usd <= 0 and projected_tokens <= 0 and projected_chars <= 0:
        return None
    status = str(signal.get("status") or "")
    signal_payload = dict(signal) if signal else {
        "schema": "agentflow.crunch_savings_signal.v1",
        "status": "observed-savings-ranked" if today_savings > 0 else "missing-crunch-measurement",
        "calls": calls,
        "observed": {
            "today_crunch_savings_usd": today_savings,
            "crunch_savings_usd": total_savings,
            "crunch_tokens_saved": tokens_saved,
            "crunch_chars_saved": chars_saved,
        },
        "missing_measurements": [] if today_savings > 0 else ["crunch-opportunity-report", "positive-observed-or-projected-savings"],
        "privacy": _candidate_privacy(),
    }
    if projected_usd > 0 or projected_tokens > 0 or projected_chars > 0:
        blocker = "crunch-projected-savings-ranked"
        path = "Convert the highest projected crunch opportunity report into a dry-run, activation, or rollout-safety follow-up."
        confidence = "medium"
        score = max(projected_usd * 2000.0, projected_tokens / 10.0, projected_chars / 100.0, 1.0) + 500.0
        bucket = str(top_report.get("report_key") or "mixed")
    elif today_savings > 0 or total_savings > 0 or tokens_saved > 0 or chars_saved > 0:
        blocker = "crunch-observed-savings-ranked"
        path = "Convert observed crunch savings into the next compaction activation or rollout-safety issue."
        confidence = "medium"
        score = max(today_savings * 1000.0, total_savings * 250.0, tokens_saved / 10.0, chars_saved / 100.0, 1.0)
        bucket = "observed-crunch"
    else:
        blocker = "missing-crunch-savings-signal" if status != "non-positive-projection" else "crunch-non-positive-projection"
        path = "Add or inspect crunch opportunity rollups before selecting more aggressive compaction work."
        confidence = "low"
        score = float(calls) * 0.1
        bucket = "missing-measurement"
    return _candidate(
        lever="crunch",
        provider_surface_bucket=bucket,
        blocker=blocker,
        estimated_savings_path=path,
        projected_savings_signal=signal_payload,
        confidence=confidence,
        sequencing="Sequence behind routing/cache blockers unless crunch savings is already the strongest positive dollar signal.",
        score=score,
    )


def _diagnostic_candidate(diagnostics: list[dict[str, Any]]) -> dict[str, Any] | None:
    actionable = _actionable_diagnostics(diagnostics)
    if not actionable:
        return None
    top = actionable[0]
    reason = str(top.get("reason") or "unknown-diagnostic")
    count = _to_int(top.get("count"))
    return _candidate(
        lever=str(top.get("source_lever") or "activation-feedback"),
        provider_surface_bucket="mixed",
        blocker=f"repeated-{top.get('diagnostic_class') or reason}",
        estimated_savings_path=str(
            top.get("expected_unblock_path")
            or "Promote repeated blocker diagnostics into a narrow issue that unlocks the affected routing, crunching, cache, or managed recommendation path."
        ),
        projected_savings_signal={
            "diagnostic_reason": reason,
            "diagnostic_class": top.get("diagnostic_class"),
            "observations": count,
            "backlog_action": top.get("backlog_action"),
            "acceptance_check": top.get("acceptance_check"),
        },
        confidence="medium" if count > 1 else "low",
        sequencing="File after direct cache/routing/crunch candidates unless the diagnostic is a safety stop or privacy blocker.",
        score=float(count) * 10.0,
    )


def _managed_recommendation_candidate(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = (
        stats_summary.get("managed_recommendation_health")
        if isinstance(stats_summary.get("managed_recommendation_health"), dict)
        else {}
    )
    if not signal:
        return None
    calls = _to_int(signal.get("calls") or stats_summary.get("calls") or stats_summary.get("today_calls"))
    top = signal.get("top_omission") if isinstance(signal.get("top_omission"), dict) else {}
    status = str(signal.get("status") or "unknown")
    omitted_reason = str(top.get("omitted_reason") or status or "unknown")
    representation = (
        top.get("local_file_backed_representation")
        if isinstance(top.get("local_file_backed_representation"), dict)
        else {}
    )
    represented = bool(representation.get("exists"))
    if status == "missing-managed-recommendation-health-report":
        blocker = "managed-recommendation-health-report-missing"
        path = "Emit a bounded managed recommendation health rollup before choosing local policy or managed optimizer follow-up."
        confidence = "low"
        safety_status = "blocked"
        score = float(calls) * 0.05
    elif status == "no-omission-reasons-reported":
        blocker = "managed-recommendation-no-omission-reasons-reported"
        path = "Keep managed recommendations as local-only/no-op until omitted local-action reasons appear in health metadata."
        confidence = "low"
        safety_status = "blocked"
        score = float(calls) * 0.05
    elif represented:
        blocker = f"managed-recommendation-{omitted_reason}"
        path = "Hand the top omitted managed recommendation reason to the matching local file-backed policy representation."
        confidence = "medium"
        safety_status = "review-required"
        score = float(_to_int(top.get("count")) or calls) + 250.0
    else:
        blocker = "managed-recommendation-no-local-representation"
        path = "Keep the recommendation omitted or define a safe local file-backed representation before any local handoff."
        confidence = "medium"
        safety_status = "blocked"
        score = float(_to_int(top.get("count")) or calls) + 150.0

    return _candidate(
        lever="managed-recommendation",
        provider_surface_bucket=str(top.get("local_action_family") or "mixed"),
        blocker=blocker,
        estimated_savings_path=path,
        projected_savings_signal=signal,
        confidence=confidence,
        sequencing="Use after routing/cache/crunch evidence and before creating managed-server work so the local handoff boundary is explicit.",
        safety_status=safety_status,
        score=score,
    )


def _request_shape_candidate(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = (
        stats_summary.get("request_shape_rollup_candidates")
        if isinstance(stats_summary.get("request_shape_rollup_candidates"), dict)
        else {}
    )
    if not signal:
        return None
    status = str(signal.get("status") or "unknown")
    calls = _to_int(signal.get("summary", {}).get("calls")) if isinstance(signal.get("summary"), dict) else 0
    top = signal.get("top_candidate") if isinstance(signal.get("top_candidate"), dict) else {}
    if top:
        next_action = str(top.get("next_action") or "rank-request-shape-cohort")
        blocker = f"request-shape-{next_action}"
        path = "Use the top aggregate request-shape cohort to choose the next repeated-context, replayability, routing, or crunch follow-up."
        confidence = "medium" if "repeated_context" in (top.get("candidate_work_classes") or []) else "low"
        safety_status = "review-required"
        score = float(_to_int(top.get("row_count"))) + _to_float(top.get("cost_est_usd")) * 1000.0 + 300.0
        bucket = str(top.get("provider_surface_bucket") or "mixed")
    elif status == "missing-request-shape-rollups":
        blocker = "request-shape-rollup-report-missing"
        path = "Emit a bounded request-shape rollup report before selecting repeated-context or replayability activation work."
        confidence = "low"
        safety_status = "blocked"
        score = float(calls) * 0.04
        bucket = "mixed"
    else:
        blocker = "request-shape-rollup-candidates-empty"
        path = "Keep request-shape follow-up blocked until aggregate rollups contain at least one repeated-context, replayability, routing, or crunch cohort."
        confidence = "low"
        safety_status = "blocked"
        score = float(calls) * 0.02
        bucket = "mixed"

    return _candidate(
        lever="request-shape-rollups",
        provider_surface_bucket=bucket,
        blocker=blocker,
        estimated_savings_path=path,
        projected_savings_signal=signal,
        confidence=confidence,
        sequencing="Use after direct routing/cache/crunch blockers; the top shape row should decide which follow-up lever owns the next implementation issue.",
        safety_status=safety_status,
        score=score,
    )


def _fallback_candidates(existing_levers: set[str], stats_summary: dict[str, Any]) -> list[dict[str, Any]]:
    calls = _to_int(stats_summary.get("calls") or stats_summary.get("today_calls"))
    fallbacks: list[dict[str, Any]] = []
    if "managed-recommendation" not in existing_levers:
        fallbacks.append(
            _candidate(
                lever="managed-recommendation",
                provider_surface_bucket="mixed",
                blocker="managed-recommendation-health-not-ranked",
                estimated_savings_path="Rank omitted managed recommendation health reasons before deciding whether local or managed follow-up should own the next savings issue.",
                projected_savings_signal={"calls": calls, "managed_dependency": "optional"},
                confidence="low",
                sequencing="Keep local runtime independent; use this only as feature-only input for future managed optimizer issues.",
                score=float(calls) * 0.05,
            )
        )
    if "request-shape-rollups" not in existing_levers:
        fallbacks.append(
            _candidate(
                lever="request-shape-rollups",
                provider_surface_bucket="mixed",
                blocker="shape-rollup-candidates-not-ranked",
                estimated_savings_path="Use aggregate request-shape rollups to identify the next repeated context, replayability, or routing bucket.",
                projected_savings_signal={"calls": calls},
                confidence="low",
                sequencing="Use when direct cache/routing/crunch signals are too thin to create implementation-ready issues.",
                score=float(calls) * 0.04,
            )
        )
    return fallbacks


def _optimization_candidates(
    *,
    stats_summary: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    minimum: int = 6,
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in (
            _cache_candidate(stats_summary),
            _routing_candidate(stats_summary),
            _crunch_candidate(stats_summary),
            _managed_recommendation_candidate(stats_summary),
            _request_shape_candidate(stats_summary),
            _diagnostic_candidate(diagnostics),
        )
        if item is not None
    ]
    existing_levers = {str(item.get("lever")) for item in candidates}
    if len(candidates) < minimum:
        candidates.extend(_fallback_candidates(existing_levers, stats_summary))
    candidates.sort(key=lambda item: _to_float(item.get("_score")), reverse=True)
    ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(candidates, start=1):
        clean = dict(item)
        clean.pop("_score", None)
        clean["rank"] = rank
        ranked.append(clean)
    return ranked


def _issue_body(
    *,
    title: str,
    rationale: str,
    evidence: list[str],
    implementation: list[str],
    acceptance: list[str],
    sequencing: str,
    savings_path: str | None = None,
) -> str:
    evidence_lines = "\n".join(f"- {redact_text(item)}" for item in evidence) or "- No local evidence was available."
    implementation_lines = "\n".join(f"- {redact_text(item)}" for item in implementation)
    acceptance_lines = "\n".join(f"- {redact_text(item)}" for item in acceptance)
    savings_text = savings_path or "This removes a planning or activation bottleneck using metadata-only local evidence."
    savings_section = "## Expected Savings Path Or Bottleneck Removed\n\n" f"{redact_text(savings_text)}\n\n"
    return (
        "## Rationale\n\n"
        f"{redact_text(rationale)}\n\n"
        "## Evidence\n\n"
        f"{evidence_lines}\n\n"
        "## Implementation Approach\n\n"
        f"{implementation_lines}\n\n"
        "## Acceptance Criteria\n\n"
        f"{acceptance_lines}\n\n"
        f"{savings_section}"
        "## Sequencing Notes\n\n"
        f"{redact_text(sequencing)}\n"
    )


def _finalize_create_issue_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(proposal)
    labels = [redact_text(str(label)) for label in finalized.get("labels") or []]
    finalized["labels"] = list(dict.fromkeys(label for label in labels if label))
    body = redact_text(str(finalized.get("body") or ""))
    if "## Labels" not in body:
        label_lines = "\n".join(f"- {label}" for label in finalized["labels"]) or "- none"
        label_section = f"## Labels\n\n{label_lines}\n\n"
        marker = "## Sequencing Notes\n\n"
        if marker in body:
            body = body.replace(marker, label_section + marker, 1)
        else:
            body = f"{body.rstrip()}\n\n{label_section.rstrip()}\n"
    finalized["body"] = body
    finalized["title"] = redact_text(str(finalized.get("title") or ""))
    finalized["repo"] = redact_text(str(finalized.get("repo") or "lutzkuen/agentflow"))
    return finalized


def _default_issue_labels(priority: str = "priority:p1") -> list[str]:
    return ["backlog", "status:ready", priority, "core-feature", "correctness"]


def _issue_title_key(title: Any) -> str:
    text = redact_text(str(title or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stop_words = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "into",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)


def _dedupe_create_issue_proposals(
    proposals: list[dict[str, Any]],
    *,
    existing_issues: Iterable[dict[str, Any]],
    max_count: int = 10,
) -> list[dict[str, Any]]:
    existing_keys = {_issue_title_key(issue.get("title")) for issue in existing_issues}
    existing_keys.discard("")
    seen = set(existing_keys)
    deduped: list[dict[str, Any]] = []
    for proposal in proposals:
        key = _issue_title_key(proposal.get("title"))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(proposal)
        if len(deduped) >= max_count:
            break
    return deduped


def _candidate_title(candidate: dict[str, Any]) -> str:
    lever = str(candidate.get("lever") or "optimization")
    blocker = str(candidate.get("blocker") or "candidate").replace("_", "-")
    signal = candidate.get("projected_savings_signal") if isinstance(candidate.get("projected_savings_signal"), dict) else {}
    if lever == "cache":
        return f"Turn {blocker} cache candidate into local replay evidence"
    if lever == "routing":
        requested = str(signal.get("requested_model") or "").strip()
        target = str(signal.get("candidate_target_model") or "").strip()
        if requested and target:
            return f"Stage routing evidence for {requested} to {target}"
        return f"Collect routing lifecycle evidence for {blocker}"
    if lever == "crunch":
        return f"Rank crunch savings follow-up for {blocker}"
    if lever == "activation-feedback":
        return f"Resolve {blocker} activation feedback blocker"
    if lever == "managed-recommendation":
        return "Rank managed recommendation omission reasons for local policy handoff"
    if lever == "request-shape-rollups":
        return "Generate request-shape rollup candidates for repeated context work"
    return f"Convert {lever} candidate into implementation-ready savings issue"


def _candidate_labels(candidate: dict[str, Any]) -> list[str]:
    labels = _default_issue_labels("priority:p1" if candidate.get("confidence") in {"high", "medium"} else "priority:p2")
    lever = str(candidate.get("lever") or "")
    if lever in {"cache", "routing", "crunch"}:
        labels.append(lever)
    if lever in {"activation-feedback", "managed-recommendation"}:
        labels.append("privacy")
    elif "privacy" not in labels:
        labels.append("privacy")
    return list(dict.fromkeys(labels))


def _candidate_issue_acceptance(candidate: dict[str, Any]) -> list[str]:
    lever = str(candidate.get("lever") or "")
    blocker = str(candidate.get("blocker") or "candidate")
    common = [
        "The issue includes a measurable item-specific verification check tied to the ranked candidate evidence.",
        "Generated and follow-up evidence remains metadata-only and excludes prompts, provider bodies, file paths, request IDs, session IDs, cache keys, and individual candidate IDs.",
    ]
    if lever == "cache":
        first = "The next research plan reports hit recovery, replay readiness, invalidation evidence, or a reduced cache blocker count for this cohort."
    elif lever == "routing":
        first = "The next routing report names applied/holdout coverage, savings per 1000 calls, and any error, retry, fallback, stale-evidence, or safety-stop blocker."
    elif lever == "crunch":
        first = "The next crunch report ranks observed or projected savings for this blocker, or records the missing measurement that prevents activation."
    elif lever == "activation-feedback":
        signal = candidate.get("projected_savings_signal") if isinstance(candidate.get("projected_savings_signal"), dict) else {}
        first = str(signal.get("acceptance_check") or "The feedback report names the privacy-safe cohort lifecycle field needed to unblock the diagnostic.")
    elif lever == "managed-recommendation":
        first = "Managed recommendation health reports the omitted local-action reason and whether a local file-backed representation exists."
    elif lever == "request-shape-rollups":
        first = "Request-shape rollups produce at least one repeated-context, replayability, routing, or crunch cohort with bounded aggregate evidence."
    else:
        first = f"The {blocker} candidate is either converted into a safe local action or kept blocked with a machine-readable reason."
    return [first, *common]


def _proposal_from_optimization_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    title = _candidate_title(candidate)
    signal = candidate.get("projected_savings_signal") if isinstance(candidate.get("projected_savings_signal"), dict) else {}
    evidence = [
        f"Ranked candidate: {candidate.get('rank')}",
        f"Lever: {candidate.get('lever')}",
        f"Provider/surface bucket: {candidate.get('provider_surface_bucket')}",
        f"Current blocker: {candidate.get('blocker')}",
        f"Confidence: {candidate.get('confidence')}",
        f"Safety status: {candidate.get('safety_status')}",
        f"Projected savings signal: {json.dumps(signal, sort_keys=True)}",
    ]
    return {
        "repo": candidate.get("repo") or "lutzkuen/agentflow",
        "title": title,
        "labels": _candidate_labels(candidate),
        "body": _issue_body(
            title=title,
            rationale=(
                "Research mode ranked this metadata-only optimization candidate as part of the telemetry-to-activation milestone. "
                "The follow-up should be narrow enough for unattended implementation and should avoid re-scanning raw dashboard or provider history."
            ),
            evidence=evidence,
            implementation=[
                "Start from the ranked optimization candidate in the research plan rather than rediscovering raw traffic.",
                "Use only bounded local metadata reports and file-backed local policy interfaces for routing, crunching, cache, or policy-bundle work.",
                "Prefer a dry-run, review, canary, or instrumentation step when lifecycle, invalidation, or safety evidence is incomplete.",
                "Record the outcome in machine-readable metadata so the next research run can rank the candidate as improved, blocked, or superseded.",
            ],
            acceptance=_candidate_issue_acceptance(candidate),
            savings_path=str(
                candidate.get("estimated_savings_path")
                or "This removes a planning bottleneck by turning aggregate metadata into a specific local optimization follow-up."
            ),
            sequencing=str(candidate.get("sequencing") or "Sequence after higher-ranked ready candidates in the same milestone."),
        ),
    }


def _proposals_from_optimization_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_proposal_from_optimization_candidate(candidate) for candidate in candidates]


def _proposal_from_low_backlog(
    *,
    ready_count: int,
    threshold: int,
    stats_summary: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = [
        f"Actionable status:ready issue count is {ready_count}, below threshold {threshold}.",
    ]
    if stats_summary:
        evidence.append(f"Recent metadata summary: {json.dumps(stats_summary, sort_keys=True)}")
    actionable_diagnostics = _actionable_diagnostics(diagnostics)
    if actionable_diagnostics:
        top = actionable_diagnostics[0]
        evidence.append(
            f"Top actionable diagnostic: {top['diagnostic_class']} from {top['source_lever']} "
            f"({top['count']} observations; backlog action: {top['backlog_action']})."
        )
    elif diagnostics:
        top = diagnostics[0]
        evidence.append(f"Top repeated diagnostic is non-actionable evidence: {top['reason']} ({top['count']} observations).")
    return {
        "repo": "lutzkuen/agentflow",
        "title": "Generate next backlog milestone from local telemetry evidence",
        "labels": _default_issue_labels("priority:p1"),
        "body": _issue_body(
            title="Generate next backlog milestone from local telemetry evidence",
            rationale=(
                "Research mode needs to create implementation-ready issues from local metadata "
                "when the ready backlog falls below the configured threshold."
            ),
            evidence=evidence,
            implementation=[
                "Read only metadata summaries from local stats, recent run logs, and GitHub issue state.",
                "Rank the next milestone around routing, crunching, caching, replay, or managed optimization.",
                "Emit issue-ready proposals with rationale, scope, acceptance criteria, labels, and sequencing.",
                "Keep raw prompts, request/response bodies, absolute file paths, request IDs, and session IDs out of generated issue bodies.",
            ],
            acceptance=[
                "A research run with a low ready backlog emits at least one targeted create_issue proposal.",
                "The emitted issue body includes evidence, implementation approach, acceptance criteria, labels, and sequencing notes.",
                "Privacy tests prove raw prompts, request bodies, file paths, request IDs, and session IDs are redacted.",
            ],
            sequencing="Use this as the first fallback issue when no more specific blocker dominates the local evidence.",
        ),
    }


def _proposal_from_repeated_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    reason = str(diagnostic.get("reason") or "unknown-diagnostic")
    diagnostic_class = str(diagnostic.get("diagnostic_class") or reason)
    source_lever = str(diagnostic.get("source_lever") or _diagnostic_source_lever(reason, diagnostic_class))
    backlog_action = str(diagnostic.get("backlog_action") or "create-ready-issue")
    expected_unblock_path = str(
        diagnostic.get("expected_unblock_path")
        or "Promote repeated blocker diagnostics into a narrow issue that unlocks the affected routing, crunching, cache, or managed recommendation path."
    )
    acceptance_check = str(
        diagnostic.get("acceptance_check")
        or "The repeated diagnostic is represented by a concrete GitHub issue or an updated blocked issue comment."
    )
    title_reason = diagnostic_class.replace("-", " ")
    return {
        "repo": "lutzkuen/agentflow",
        "title": f"Turn repeated {title_reason} diagnostics into an actionable optimization issue",
        "labels": _default_issue_labels("priority:p2"),
        "body": _issue_body(
            title=f"Turn repeated {title_reason} diagnostics into an actionable optimization issue",
            rationale=(
                "Research mode found the same skip or blocker diagnostic repeatedly. That should become a "
                "narrow follow-up issue instead of disappearing into run prose."
            ),
            evidence=[
                f"Diagnostic reason: {reason}",
                f"Diagnostic class: {diagnostic_class}",
                f"Source lever: {source_lever}",
                f"Backlog action: {backlog_action}",
                f"Observed count: {diagnostic.get('count', 0)}",
                f"Sanitized example: {diagnostic.get('example', '')}",
                f"Expected unblock path: {expected_unblock_path}",
            ],
            implementation=[
                "Trace the diagnostic to the local report, rollout, or canary gate that emits it.",
                f"Focus the follow-up on the {source_lever} path and the taxonomy action `{backlog_action}`.",
                "Decide whether the blocker needs more samples, safer policy metadata, a rollback, or a narrower feature slice.",
                "Create or update the smallest issue that directly unlocks the affected routing, crunching, caching, or replay milestone.",
            ],
            acceptance=[
                acceptance_check,
                "The issue includes an implementation path and a measurable acceptance check tied to the diagnostic.",
                "Generated text remains metadata-only and contains no raw prompts, provider bodies, file paths, or request/session IDs.",
            ],
            sequencing="File after the low-backlog milestone proposal only when this diagnostic appears more than once.",
        ),
    }


_CACHE_ACTION_ISSUE_TITLES = {
    "stage-replay-policy": "Stage cache replay canary for {cohort}",
    "collect-dependency-evidence": "Collect cache replay dependency evidence for {cohort}",
    "reload-cache-policy": "Reload or enable cache policy for {cohort}",
    "promote-or-rebalance-canary": "Review cache replay holdout evidence for {cohort}",
    "review-safety-stop": "Review cache replay safety stop for {cohort}",
    "instrument-cache-decision": "Instrument cache decisions for {cohort}",
    "accept-non-repeatable-traffic": "Confirm non-repeatable cache cohort for {cohort}",
}

_CACHE_ACTION_LABELS = {
    "stage-replay-policy": "stage a local cache replay canary",
    "collect-dependency-evidence": "collect local invalidation and dependency evidence",
    "reload-cache-policy": "enable or reload the local cache policy",
    "promote-or-rebalance-canary": "review holdout evidence before promotion",
    "review-safety-stop": "review the safety stop before activation",
    "instrument-cache-decision": "add explicit cache decision instrumentation",
    "accept-non-repeatable-traffic": "prove this traffic is non-repeatable or find a narrower repeated shape",
}

_CACHE_ACTION_ACCEPTANCE = {
    "stage-replay-policy": "A dry-run or canary check shows projected hit recovery for the cohort, with holdout traffic still bypassing unchanged.",
    "collect-dependency-evidence": "The cohort reports stable dependency evidence or a smaller stale-risk blocker count before replay activation is considered.",
    "reload-cache-policy": "The cache policy state is loaded and the cohort's disabled/reload blocker count falls in the next bounded metadata window.",
    "promote-or-rebalance-canary": "Holdout evidence is summarized with projected hits and a safe promote, rebalance, or keep-holdout decision.",
    "review-safety-stop": "The safety-stop reason is either resolved with an explicit safe bypass reduction check or kept blocked with a narrow blocker.",
    "instrument-cache-decision": "New rows for the surface include explicit cache status and reason metadata instead of unknown-cache-decision.",
    "accept-non-repeatable-traffic": "The cohort is marked research-only or blocked unless repeated request-shape evidence appears.",
}

_CACHE_ACTION_CONCRETE = {
    "stage-replay-policy",
    "collect-dependency-evidence",
    "reload-cache-policy",
    "promote-or-rebalance-canary",
    "review-safety-stop",
    "instrument-cache-decision",
}


_OPENAI_ROUTING_ACTION_LABELS = {
    "widen_local_openai_canary": "widen the reviewed local OpenAI routing canary",
    "keep_current_openai_canary_fraction": "keep the current canary fraction and collect more lifecycle evidence",
    "rollback_or_disable_openai_canary": "rollback or disable the unsafe local OpenAI routing canary",
    "collect_openai_canary_holdout_evidence_or_run_eval": "collect OpenAI canary holdout evidence or run a local eval",
}

_OPENAI_ROUTING_BLOCKED_ACCEPTANCE = {
    "missing-openai-canary-impact": "The next research run includes bounded OpenAI routing canary impact metadata before activation is reconsidered.",
    "missing-canary-candidate": "The OpenAI canary impact report includes at least one candidate row with applied and holdout cohort counts.",
    "missing-lifecycle-feedback": "Activation lifecycle feedback reports a healthy canary state before a widening or policy-bundle issue is generated.",
    "aggregate-only-feedback": "The blocker update remains review-only until lifecycle feedback includes candidate-level canary state without raw prompts, provider bodies, request IDs, or session IDs.",
    "stale-evidence": "Fresh canary lifecycle evidence is collected inside the configured evidence window before activation is reconsidered.",
    "non-positive-savings": "The candidate reports positive observed or projected savings per 1000 calls before activation is reconsidered.",
    "insufficient-cohort-coverage": "The candidate reports both applied and holdout cohort samples before activation is reconsidered.",
    "safety-stop-observed": "The safety stop is resolved or the canary is rolled back with a machine-readable reason before widening is reconsidered.",
    "canary-verdict-hold": "The hold reason is resolved and a later canary impact report returns a widen verdict before activation is reconsidered.",
    "canary-verdict-rollback": "The canary rollback or disable action is recorded before any new OpenAI routing activation issue is generated.",
    "canary-verdict-needs_eval": "Additional holdout samples or eval evidence move the verdict out of needs_eval before activation is reconsidered.",
}


def _breakdown_counts(rows: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("value") or row.get("state") or "").strip()
        if value:
            counts[value] = counts.get(value, 0) + _to_int(row.get("count"), 1)
    return counts


def _cohort_lifecycle_metadata(feedback: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    rows = feedback.get("cohort_lifecycle_metadata") if isinstance(feedback.get("cohort_lifecycle_metadata"), list) else []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata.append(
            {
                "policy_ref": sanitize_value(row.get("policy_ref") or row.get("policy_public_id") or "unknown"),
                "cohort_label": sanitize_value(row.get("cohort_label") or row.get("cohort") or "unknown"),
                "action_family": sanitize_value(row.get("action_family") or "unknown"),
                "event_count": _to_int(row.get("event_count") or row.get("count")),
                "applied_count": _to_int(row.get("applied_count")),
                "holdout_count": _to_int(row.get("holdout_count")),
                "fallback_count": _to_int(row.get("fallback_count")),
                "error_rate": round(_to_float(row.get("error_rate")), 6),
                "savings_estimate_usd": round(_to_float(row.get("savings_estimate_usd")), 8),
            }
        )
        if len(metadata) >= limit:
            break
    return metadata


def _has_policy_cohort_lifecycle_metadata(feedback: dict[str, Any]) -> bool:
    for row in _cohort_lifecycle_metadata(feedback, limit=50):
        if row["policy_ref"] == "unknown" or row["cohort_label"] == "unknown":
            continue
        if any(
            _to_int(row.get(key)) > 0
            for key in ("event_count", "applied_count", "holdout_count", "fallback_count")
        ):
            return True
    return False


def _openai_routing_canary_bucket(row: dict[str, Any]) -> str:
    provider = "openai"
    source_surface = str(row.get("source_surface") or "openai_provider_request").strip()
    original = str(row.get("original_model") or row.get("requested_model") or "").strip()
    target = str(row.get("candidate_target_model") or row.get("target_model") or "").strip()
    model_pair = f"{original}->{target}" if original and target else "routing-canary"
    return "/".join(part for part in (provider, source_surface, model_pair) if part)


def _openai_routing_canary_title_suffix(row: dict[str, Any]) -> str:
    original = str(row.get("original_model") or row.get("requested_model") or "source model").strip()
    target = str(row.get("candidate_target_model") or row.get("target_model") or "target model").strip()
    source_surface = str(row.get("source_surface") or "openai_provider_request").strip()
    return f"{original} to {target} on {source_surface}"


def _openai_savings_per_1000_calls(row: dict[str, Any]) -> float:
    counts = row.get("cohort_counts") if isinstance(row.get("cohort_counts"), dict) else {}
    applied_count = _to_int(counts.get("canary_applied"))
    sample_count = _to_int(row.get("sample_count"))
    observed = _to_float(row.get("observed_savings_usd"))
    projected = _to_float(row.get("projected_savings_usd"))
    if observed > 0 and applied_count > 0:
        return round((observed / applied_count) * 1000.0, 6)
    if projected > 0 and sample_count > 0:
        return round((projected / sample_count) * 1000.0, 6)
    return 0.0


def _openai_lifecycle_omission_reason(feedback: dict[str, Any]) -> str | None:
    if not isinstance(feedback, dict) or _to_int(feedback.get("queue_rows")) <= 0:
        return "missing-lifecycle-feedback"
    privacy = feedback.get("privacy") if isinstance(feedback.get("privacy"), dict) else {}
    has_cohort_lifecycle = _has_policy_cohort_lifecycle_metadata(feedback)
    if privacy.get("aggregate_only") is True and not has_cohort_lifecycle:
        return "aggregate-only-feedback"
    state_counts = _breakdown_counts(feedback.get("state_breakdown"))
    if state_counts.get("rollback_required", 0) > 0:
        return "safety-stop-observed"
    if state_counts.get("suppressed", 0) > 0:
        return "safety-stop-observed"
    if state_counts.get("missing_feedback", 0) > 0 and state_counts.get("healthy_canary", 0) <= 0:
        return "missing-lifecycle-feedback"
    if state_counts.get("holdout_only", 0) > 0 and state_counts.get("healthy_canary", 0) <= 0:
        return "insufficient-cohort-coverage"
    if state_counts.get("healthy_canary", 0) <= 0:
        return "missing-lifecycle-feedback"
    return None


def _top_openai_routing_canary_row(stats_summary: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    impact = stats_summary.get("openai_canary_impact")
    if not isinstance(impact, dict):
        return None, ""
    candidates = [row for row in impact.get("candidates") or [] if isinstance(row, dict)]
    if not candidates:
        return None, "missing-canary-candidate"

    verdict_rank = {"widen": 0, "hold": 1, "needs_eval": 2, "rollback": 3}

    def sort_key(row: dict[str, Any]) -> tuple[int, float, int]:
        counts = row.get("cohort_counts") if isinstance(row.get("cohort_counts"), dict) else {}
        sample_count = _to_int(row.get("sample_count")) or sum(_to_int(value) for value in counts.values())
        return (
            verdict_rank.get(str(row.get("verdict") or ""), 4),
            -_openai_savings_per_1000_calls(row),
            -sample_count,
        )

    return sorted(candidates, key=sort_key)[0], "openai_canary_impact"


def _openai_routing_candidate_action(row: dict[str, Any], stats_summary: dict[str, Any]) -> dict[str, Any]:
    impact = stats_summary.get("openai_canary_impact") if isinstance(stats_summary.get("openai_canary_impact"), dict) else {}
    feedback = impact.get("activation_lifecycle_feedback") if isinstance(impact.get("activation_lifecycle_feedback"), dict) else {}
    counts = row.get("cohort_counts") if isinstance(row.get("cohort_counts"), dict) else {}
    applied_count = _to_int(counts.get("canary_applied"))
    holdout_count = _to_int(counts.get("canary_holdout"))
    safety_count = _to_int(counts.get("safety_stopped"))
    savings_per_1000 = _openai_savings_per_1000_calls(row)
    reason_codes = [str(reason) for reason in row.get("reason_codes") or []]
    stale = row.get("stale_evidence") if isinstance(row.get("stale_evidence"), dict) else {}
    verdict = str(row.get("verdict") or "unknown")
    omission_reason = None
    has_cohort_lifecycle = _has_policy_cohort_lifecycle_metadata(feedback)

    if row.get("aggregate_only_feedback") is True and not has_cohort_lifecycle:
        omission_reason = "aggregate-only-feedback"
    elif stale.get("stale"):
        omission_reason = "stale-evidence"
    elif safety_count > 0 or "safety-stop-observed" in reason_codes:
        omission_reason = "safety-stop-observed"
    elif applied_count <= 0 or holdout_count <= 0:
        omission_reason = "insufficient-cohort-coverage"
    elif savings_per_1000 <= 0:
        omission_reason = "non-positive-savings"
    else:
        omission_reason = _openai_lifecycle_omission_reason(feedback)

    if omission_reason is None and verdict != "widen":
        omission_reason = f"canary-verdict-{verdict}"
    activation_ready = omission_reason is None and verdict == "widen"
    return {
        "activation_ready": activation_ready,
        "omission_reason": omission_reason or "none",
        "savings_per_1000_calls_usd": savings_per_1000,
        "action_label": _OPENAI_ROUTING_ACTION_LABELS.get(
            str(row.get("next_action") or ""),
            "inspect the OpenAI routing canary lifecycle evidence",
        ),
    }


def _cache_cohort_name(row: dict[str, Any]) -> str:
    blocker = str(row.get("blocker_code") or row.get("readiness") or "cache-replay").strip() or "cache-replay"
    provider = str(row.get("provider") or "").strip()
    surface = str(row.get("source_surface") or "").strip()
    endpoint = str(row.get("endpoint") or "").strip()
    parts = [part for part in (provider, surface, endpoint) if part]
    if parts:
        return f"{blocker} on {'/'.join(parts)}"
    return blocker


def _top_cache_replay_issue_row(stats_summary: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    cohorts = stats_summary.get("cache_replay_cohort_ranking")
    if isinstance(cohorts, dict):
        for row in cohorts.get("cohorts") or []:
            if not isinstance(row, dict):
                continue
            readiness = str(row.get("readiness") or "")
            if readiness not in {"activation-ready", "needs-more-evidence", "blocked"}:
                continue
            action = "stage-replay-policy" if readiness == "activation-ready" else "collect-dependency-evidence"
            issue_row = dict(row)
            issue_row["next_action_family"] = action
            issue_row["next_action_label"] = _CACHE_ACTION_LABELS[action]
            if not issue_row.get("blocker_code"):
                blockers = issue_row.get("blocker_reasons") if isinstance(issue_row.get("blocker_reasons"), list) else []
                issue_row["blocker_code"] = blockers[0] if blockers else readiness
            return issue_row, "cache_replay_cohort_ranking"

    ladder = stats_summary.get("cache_zero_hit_blocker_ladder")
    if isinstance(ladder, dict):
        ladder_summary = ladder.get("summary") if isinstance(ladder.get("summary"), dict) else {}
        if not bool(ladder_summary.get("zero_hit_window")):
            return None, ""
        for row in ladder.get("ladder") or []:
            if not isinstance(row, dict):
                continue
            blocker = str(row.get("blocker_code") or "")
            if blocker == "cache-hit-observed":
                continue
            action = str(row.get("next_action_family") or "")
            if action == "none":
                continue
            return row, "cache_zero_hit_blocker_ladder"
    return None, ""


def _proposal_from_cache_replay_blocker(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    row, source = _top_cache_replay_issue_row(stats_summary)
    if row is None:
        return None
    action_family = str(row.get("next_action_family") or "instrument-cache-decision")
    cohort = _cache_cohort_name(row)
    action_label = str(row.get("next_action_label") or _CACHE_ACTION_LABELS.get(action_family, "inspect the cache replay cohort"))
    replayability = str(row.get("replayability_level") or "")
    aggregate_only_ladder = source == "cache_zero_hit_blocker_ladder" and replayability in {"features_only", "aggregate-only"}
    is_activation_action = action_family in _CACHE_ACTION_CONCRETE and not aggregate_only_ladder
    activation_mode = "activation-candidate" if is_activation_action else "research-only"
    if aggregate_only_ladder:
        title = f"Collect cache replay evidence for {cohort}"
    else:
        title_template = _CACHE_ACTION_ISSUE_TITLES.get(action_family, _CACHE_ACTION_ISSUE_TITLES["instrument-cache-decision"])
        title = title_template.format(cohort=cohort)
    evidence = [
        f"Source metadata: {source}",
        f"Top blocker cohort: {cohort}",
        f"Local action needed: {action_label}",
        f"Activation mode: {activation_mode}",
    ]
    for key in (
        "count",
        "projected_hits",
        "projected_saved_cost_usd",
        "readiness",
        "dependency_state",
        "provider_adoption_state",
        "replayability_level",
        "stream_mode",
        "tool_presence",
        "cache_status",
        "cache_reason",
    ):
        if row.get(key) is not None:
            evidence.append(f"{key}: {row.get(key)}")
    implementation = [
        "Use the bounded cache zero-hit blocker ladder and replay cohort ranking; do not inspect prompts, responses, cache keys, file paths, request IDs, or session IDs.",
        f"Focus on the named cohort and {action_label}.",
        "If the cohort remains aggregate-only or stale-risk blocked, keep the follow-up as evidence collection instead of enabling replay.",
        "Record the resulting local cache policy, canary, reload, or evidence decision in machine-readable cache metadata.",
    ]
    acceptance = [
        _CACHE_ACTION_ACCEPTANCE.get(action_family, _CACHE_ACTION_ACCEPTANCE["instrument-cache-decision"]),
        "The follow-up reports either hit recovery for the cohort or a measurable safe bypass/blocker reduction in the next bounded metadata window.",
        "Generated and follow-up evidence remains aggregate-only and excludes prompts, provider bodies, file paths, cache keys, request IDs, and session IDs.",
    ]
    return {
        "repo": "lutzkuen/agentflow",
        "title": title,
        "labels": ["backlog", "status:ready", "priority:p1", "core-feature", "correctness", "cache", "privacy"],
        "body": _issue_body(
            title=title,
            rationale=(
                "Research mode found a zero-hit cache window with a named blocker cohort. "
                "The next cache issue should target that cohort's local replay, invalidation, policy reload, or instrumentation action instead of treating zero hits as a generic observation."
            ),
            evidence=evidence,
            implementation=implementation,
            acceptance=acceptance,
            sequencing="Sequence before broad cache canaries so replay work targets the highest-value zero-hit blocker cohort first.",
        ),
    }


def _proposal_from_openai_routing_canary_feedback(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    row, source = _top_openai_routing_canary_row(stats_summary)
    if row is None:
        return None
    action = _openai_routing_candidate_action(row, stats_summary)
    counts = row.get("cohort_counts") if isinstance(row.get("cohort_counts"), dict) else {}
    deltas = row.get("applied_vs_holdout_deltas") if isinstance(row.get("applied_vs_holdout_deltas"), dict) else {}
    impact = stats_summary.get("openai_canary_impact") if isinstance(stats_summary.get("openai_canary_impact"), dict) else {}
    feedback = impact.get("activation_lifecycle_feedback") if isinstance(impact.get("activation_lifecycle_feedback"), dict) else {}
    state_counts = _breakdown_counts(feedback.get("state_breakdown"))
    lifecycle_metadata = _cohort_lifecycle_metadata(feedback, limit=5)
    next_action = str(row.get("next_action") or "inspect_openai_canary_lifecycle_evidence")
    reason_codes = [str(reason) for reason in row.get("reason_codes") or []]
    warning_codes = [str(reason) for reason in row.get("warning_codes") or []]
    suffix = _openai_routing_canary_title_suffix(row)
    local_action = action["action_label"]
    activation_mode = "activation-candidate" if action["activation_ready"] else "blocked"
    if action["activation_ready"]:
        title = f"Widen OpenAI routing canary for {suffix}"
        labels = ["backlog", "status:ready", "priority:p1", "core-feature", "correctness", "routing", "privacy"]
        acceptance_first = (
            "A local canary review or policy-bundle dry run widens the OpenAI routing canary while preserving explicit holdout traffic and rollback metadata."
        )
        rationale = (
            "Research mode found OpenAI routing canary lifecycle feedback with applied and holdout coverage, positive savings, and no regression or safety-stop verdict. "
            "The next local issue can move from observation to reviewed canary widening without relying on raw provider content."
        )
    else:
        omission = action["omission_reason"]
        title = f"Blocked: Resolve OpenAI routing canary evidence for {suffix}"
        labels = ["backlog", "status:blocked", "priority:p1", "core-feature", "correctness", "routing", "privacy"]
        acceptance_first = _OPENAI_ROUTING_BLOCKED_ACCEPTANCE.get(
            omission,
            "The explicit OpenAI routing canary blocker is resolved before activation is reconsidered.",
        )
        rationale = (
            "Research mode found OpenAI routing canary metadata, but the lifecycle feedback is not activation-ready. "
            "The next update should keep the canary blocked and name the exact evidence gate instead of producing a widening issue."
        )
    evidence = [
        f"Source metadata: {source}",
        f"Ranked routing canary: {suffix}",
        f"Activation mode: {activation_mode}",
        f"Local action needed: {local_action}",
        f"Next action: {next_action}",
        f"Omission reason: {action['omission_reason']}",
        f"Savings per 1000 calls estimate: {action['savings_per_1000_calls_usd']}",
        f"Verdict: {row.get('verdict') or 'unknown'}",
        f"Reason codes: {', '.join(reason_codes) if reason_codes else 'none'}",
        f"Lifecycle states: {json.dumps(state_counts, sort_keys=True)}",
        f"Applied samples: {_to_int(counts.get('canary_applied'))}",
        f"Holdout samples: {_to_int(counts.get('canary_holdout'))}",
        f"Safety-stopped samples: {_to_int(counts.get('safety_stopped'))}",
    ]
    if lifecycle_metadata:
        evidence.append(f"Cohort lifecycle metadata: {json.dumps(lifecycle_metadata, sort_keys=True)}")
    for key in (
        "observed_savings_usd",
        "projected_savings_usd",
        "latest_observed_at",
    ):
        if row.get(key) is not None:
            evidence.append(f"{key}: {row.get(key)}")
    for key in (
        "applied_minus_holdout_error_rate",
        "applied_minus_holdout_retry_rate",
        "applied_minus_holdout_fallback_rate",
        "applied_minus_holdout_latency_avg_ms",
    ):
        if deltas.get(key) is not None:
            evidence.append(f"{key}: {deltas.get(key)}")
    if warning_codes:
        evidence.append(f"Warning codes: {', '.join(warning_codes)}")

    return {
        "repo": "lutzkuen/agentflow",
        "title": title,
        "labels": labels,
        "body": _issue_body(
            title=title,
            rationale=rationale,
            evidence=evidence,
            implementation=[
                "Use the bounded OpenAI canary impact report and activation lifecycle feedback; do not inspect prompts, provider bodies, request IDs, session IDs, candidate identifiers, or individual comparison records.",
                f"Focus on the ranked canary and {local_action}.",
                "If evidence is stale, aggregate-only, missing holdout coverage, regressed, or safety-stopped, keep the work blocked and collect the named evidence instead of widening.",
                "Record the outcome in machine-readable routing canary, policy bundle, or lifecycle feedback metadata so the next research run can verify the gate.",
            ],
            acceptance=[
                acceptance_first,
                "The follow-up reports the item-specific verification check: applied/holdout coverage, error/retry/fallback deltas, lifecycle state, and savings per 1000 calls or the explicit omission reason.",
                "Generated and follow-up evidence remains metadata-only and excludes prompts, provider bodies, file paths, request IDs, session IDs, candidate identifiers, and individual comparison records.",
            ],
            sequencing="Sequence before generic OpenAI pass-through routing work so local activation follows canary lifecycle evidence rather than raw traffic volume alone.",
        ),
    }


def _blocked_comment(issue: dict[str, Any], diagnostics: list[dict[str, Any]], stats_summary: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        f"Blocked issue has been stale for {issue.get('age_days', 'unknown')} days.",
    ]
    actionable_diagnostics = _actionable_diagnostics(diagnostics)
    if actionable_diagnostics:
        top = actionable_diagnostics[0]
        evidence.append(
            f"Top current actionable diagnostic: {top['diagnostic_class']} from {top['source_lever']} "
            f"({top['count']} observations; expected unblock: {top['expected_unblock_path']})."
        )
    elif diagnostics:
        evidence.append(f"Top current diagnostic is non-actionable evidence: {diagnostics[0]['reason']} ({diagnostics[0]['count']} observations).")
    if stats_summary:
        evidence.append(f"Current sanitized stats summary: {json.dumps(stats_summary, sort_keys=True)}")
    body = _issue_body(
        title=str(issue.get("title") or ""),
        rationale="Research mode found this blocked issue stale and is attaching current metadata evidence for the next human or unattended pass.",
        evidence=evidence,
        implementation=[
            "Check whether the original blocker is still present in the current reports or stats.",
            "If still blocked, keep status:blocked and add the missing evidence or dependency as a narrower issue.",
            "If resolved, remove status:blocked and add status:ready with the concrete next acceptance metric.",
        ],
        acceptance=[
            "The issue has a current blocker comment with sanitized local evidence.",
            "The next action is either a narrower ready issue, a clear unblock condition, or a justified close.",
        ],
        sequencing="Use before creating duplicate replacement issues for the same milestone.",
    )
    return {
        "repo": issue.get("repo") or "unknown",
        "number": issue.get("number"),
        "action": "comment",
        "body": body,
    }


def build_research_plan(
    *,
    issues: Iterable[dict[str, Any]],
    stats: dict[str, Any] | None = None,
    log_sources: Iterable[str | Path] = (),
    threshold: int = 3,
    trusted_author: str = "lutzkuen",
    stale_days: int = 14,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issue_list = [sanitize_value(issue) for issue in issues]
    ready_issues = [issue for issue in issue_list if _is_actionable_ready(issue, trusted_author)]
    blocked_stale = _stale_blocked_issues(issue_list, trusted_author=trusted_author, now=now, stale_days=stale_days)
    diagnostics = _diagnostics_from_logs(log_sources)
    summary = _stats_summary(stats)
    optimization_candidates = _optimization_candidates(stats_summary=summary, diagnostics=diagnostics)

    ready_count = len(ready_issues)
    should_run = threshold > 0 and ready_count < threshold
    trigger_reason = "ready-actionable-count-below-threshold" if should_run else "enough-ready-actionable-issues"

    create_issues: list[dict[str, Any]] = []
    comment_issues: list[dict[str, Any]] = []
    close_issues: list[dict[str, Any]] = []

    if should_run:
        create_issues.append(
            _proposal_from_low_backlog(
                ready_count=ready_count,
                threshold=threshold,
                stats_summary=summary,
                diagnostics=diagnostics,
            )
        )
        cache_replay_proposal = _proposal_from_cache_replay_blocker(summary)
        if cache_replay_proposal is not None:
            create_issues.append(cache_replay_proposal)
        openai_routing_proposal = _proposal_from_openai_routing_canary_feedback(summary)
        if openai_routing_proposal is not None:
            create_issues.append(openai_routing_proposal)
        create_issues.extend(_proposals_from_optimization_candidates(optimization_candidates))
        repeated_actionable_diagnostics = [item for item in _actionable_diagnostics(diagnostics) if _to_int(item.get("count")) > 1]
        if repeated_actionable_diagnostics:
            create_issues.append(_proposal_from_repeated_diagnostic(repeated_actionable_diagnostics[0]))
        for issue in blocked_stale[:3]:
            comment_issues.append(_blocked_comment(issue, diagnostics, summary))
        create_issues = _dedupe_create_issue_proposals(create_issues, existing_issues=issue_list, max_count=10)
        create_issues = [_finalize_create_issue_proposal(proposal) for proposal in create_issues]

    inspected_sources = ["github_issues"]
    if summary:
        inspected_sources.append("local_stats")
    if diagnostics:
        inspected_sources.append("orchestrator_logs")

    result = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "research_trigger": {
            "should_run": should_run,
            "reason": trigger_reason,
            "actionable_ready_count": ready_count,
            "threshold": threshold,
            "trusted_author": trusted_author,
        },
        "evidence": {
            "inspected_sources": inspected_sources,
            "stats_summary": summary,
            "ready_issues": [_issue_ref(issue) for issue in ready_issues],
            "stale_blocked_issues": blocked_stale,
            "repeated_diagnostics": diagnostics,
            "optimization_candidates": optimization_candidates if should_run else [],
        },
        "backlog_changes": {
            "create_issues": create_issues,
            "comment_issues": comment_issues,
            "close_issues": close_issues,
        },
        "run_log_summary": (
            f"Research mode {'should run' if should_run else 'should not run'}: "
            f"{ready_count} status:ready actionable issues, threshold {threshold}; "
            f"{len(create_issues)} issue proposals, {len(comment_issues)} blocked issue comments."
        ),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }
    return sanitize_value(result)


def load_json_file(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def write_json(stream: Any, payload: dict[str, Any], *, pretty: bool = False) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
