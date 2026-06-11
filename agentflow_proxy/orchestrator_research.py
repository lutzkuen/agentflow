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
    r"\b((?:request|session|thread|tenant|cache|run|trace)[_-]?id)\s*[:=]\s*[\"']?([A-Za-z0-9_.:/@-]{6,})",
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
_ID_FIELD_RE = re.compile(r"(?:^|_)(?:request|session|thread|tenant|trace|cache)_?id$")
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
    return summary


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
