from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable


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
    "stale quality evidence",
    "stale-quality-evidence",
    "holdout regression",
    "holdout-regression",
    "privacy-blocked",
    "aggregate-only",
    "safety-stop",
    "unsupported local executor",
    "unsupported-local-executor",
)


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
        "today_crunch_savings_usd",
    )
    summary = {key: stats[key] for key in keys if key in stats}
    routing = stats.get("routing")
    if isinstance(routing, list):
        summary["routing_top"] = routing[:5]
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
    cohorts = (
        stats.get("cache_replay_cohort_ranking")
        or stats.get("cache_replay_cohorts")
        or stats.get("cache_replay_plateau_cohort_ranking")
    )
    if isinstance(cohorts, dict):
        cohort_rows = cohorts.get("cohorts") if isinstance(cohorts.get("cohorts"), list) else []
        summary["cache_replay_cohort_ranking"] = {
            "schema": cohorts.get("schema"),
            "summary": cohorts.get("summary") if isinstance(cohorts.get("summary"), dict) else {},
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
            },
            confidence="high" if action["activation_ready"] else "medium",
            sequencing="Use canary lifecycle feedback before generic pass-through routing issues so activation work cites applied, holdout, and safety evidence.",
            safety_status="review-required" if action["activation_ready"] else "blocked",
            score=10_000.0 + action["savings_per_1000_calls_usd"],
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
    today_savings = _to_float(stats_summary.get("today_crunch_savings_usd"))
    calls = _to_int(stats_summary.get("today_calls") or stats_summary.get("calls"))
    if calls <= 0 and today_savings <= 0:
        return None
    if today_savings > 0:
        blocker = "crunch-savings-not-ranked-into-backlog"
        path = "Convert observed crunch savings into the next compaction activation or rollout-safety issue."
        confidence = "medium"
        score = max(today_savings * 1000.0, 1.0)
    else:
        blocker = "missing-crunch-savings-signal"
        path = "Add or inspect crunch opportunity rollups before selecting more aggressive compaction work."
        confidence = "low"
        score = float(calls) * 0.1
    return _candidate(
        lever="crunch",
        provider_surface_bucket="mixed",
        blocker=blocker,
        estimated_savings_path=path,
        projected_savings_signal={"today_crunch_savings_usd": today_savings, "calls": calls},
        confidence=confidence,
        sequencing="Sequence behind routing/cache blockers unless crunch savings is already the strongest positive dollar signal.",
        score=score,
    )


def _diagnostic_candidate(diagnostics: list[dict[str, Any]]) -> dict[str, Any] | None:
    actionable = [item for item in diagnostics if str(item.get("reason") or "") != "pass"]
    if not actionable:
        return None
    top = actionable[0]
    reason = str(top.get("reason") or "unknown-diagnostic")
    count = _to_int(top.get("count"))
    return _candidate(
        lever="activation-feedback",
        provider_surface_bucket="mixed",
        blocker=f"repeated-{reason}",
        estimated_savings_path="Promote repeated blocker diagnostics into a narrow issue that unlocks the affected routing, crunching, cache, or managed recommendation path.",
        projected_savings_signal={"diagnostic_reason": reason, "observations": count},
        confidence="medium" if count > 1 else "low",
        sequencing="File after direct cache/routing/crunch candidates unless the diagnostic is a safety stop or privacy blocker.",
        score=float(count) * 10.0,
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
    minimum: int = 3,
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in (
            _cache_candidate(stats_summary),
            _routing_candidate(stats_summary),
            _crunch_candidate(stats_summary),
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
) -> str:
    evidence_lines = "\n".join(f"- {redact_text(item)}" for item in evidence) or "- No local evidence was available."
    implementation_lines = "\n".join(f"- {item}" for item in implementation)
    acceptance_lines = "\n".join(f"- {item}" for item in acceptance)
    return (
        "## Rationale\n\n"
        f"{redact_text(rationale)}\n\n"
        "## Evidence\n\n"
        f"{evidence_lines}\n\n"
        "## Implementation Approach\n\n"
        f"{implementation_lines}\n\n"
        "## Acceptance Criteria\n\n"
        f"{acceptance_lines}\n\n"
        "## Sequencing Notes\n\n"
        f"{redact_text(sequencing)}\n"
    )


def _default_issue_labels(priority: str = "priority:p1") -> list[str]:
    return ["backlog", "status:ready", priority, "core-feature", "correctness"]


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
    if diagnostics:
        top = diagnostics[0]
        evidence.append(f"Top repeated diagnostic: {top['reason']} ({top['count']} observations).")
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
    title_reason = reason.replace("-", " ")
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
                f"Observed count: {diagnostic.get('count', 0)}",
                f"Sanitized example: {diagnostic.get('example', '')}",
            ],
            implementation=[
                "Trace the diagnostic to the local report, rollout, or canary gate that emits it.",
                "Decide whether the blocker needs more samples, safer policy metadata, a rollback, or a narrower feature slice.",
                "Create or update the smallest issue that directly unlocks the affected routing, crunching, caching, or replay milestone.",
            ],
            acceptance=[
                "The repeated diagnostic is represented by a concrete GitHub issue or an updated blocked issue comment.",
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
    if privacy.get("aggregate_only") is True:
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

    if row.get("aggregate_only_feedback") is True:
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
    if diagnostics:
        evidence.append(f"Top current diagnostic: {diagnostics[0]['reason']} ({diagnostics[0]['count']} observations).")
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
        if diagnostics and diagnostics[0].get("count", 0) > 1:
            create_issues.append(_proposal_from_repeated_diagnostic(diagnostics[0]))
        for issue in blocked_stale[:3]:
            comment_issues.append(_blocked_comment(issue, diagnostics, summary))

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
