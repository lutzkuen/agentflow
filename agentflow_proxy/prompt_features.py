from __future__ import annotations

import re
from typing import Any


PROMPT_DIFFICULTY_FEATURE_SCHEMA = "agentflow.prompt_difficulty_features.v1"

_WORD_RE = re.compile(r"[a-z0-9_]+(?:[-'][a-z0-9_]+)?", re.IGNORECASE)
_QUESTION_RE = re.compile(r"\?")
_STEP_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*]\s+|\b(?:then|after|next|finally)\b)", re.IGNORECASE | re.MULTILINE)
_CODE_OR_ERROR_RE = re.compile(r"```|\b(?:traceback|exception|stack trace|assertionerror|typeerror|failed|failure|error)\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)", re.IGNORECASE)

_ACK_PHRASES = (
    "thank you",
    "thanks",
    "thx",
    "got it",
    "sounds good",
    "ok",
    "okay",
    "understood",
    "appreciate it",
)
_CURRENT_STATE_TERMS = {
    "current",
    "currently",
    "latest",
    "live",
    "now",
    "today",
    "outstanding",
    "open",
    "pending",
    "active",
    "unresolved",
    "recent",
    "fresh",
}
_LOOKUP_TERMS = {
    "find",
    "lookup",
    "look",
    "search",
    "check",
    "inspect",
    "retrieve",
    "list",
    "query",
    "investigate",
    "identify",
    "compare",
    "audit",
}
_EXECUTABLE_TERMS = _LOOKUP_TERMS | {
    "run",
    "read",
    "write",
    "edit",
    "update",
    "change",
    "fix",
    "debug",
    "implement",
    "create",
    "delete",
    "deploy",
}
_DEBUG_TERMS = {"debug", "failing", "failure", "failed", "error", "exception", "traceback", "bug", "regression"}
_PLANNING_TERMS = {"plan", "design", "strategy", "approach", "roadmap", "sequence"}
_SUMMARY_TERMS = {"summarize", "summarise", "summary", "recap", "brief", "tl;dr"}
_EDIT_TERMS = {"edit", "rewrite", "reword", "polish", "format", "translate"}
_STATUS_TERMS = {"status", "progress", "done", "complete", "blocked", "remaining"}
_VERIFY_TERMS = {"verify", "confirm", "validate", "check", "test", "prove", "reconcile", "audit"}

_SOURCE_GROUPS = (
    ("web", {"web", "internet", "online", "site", "url", "browser", "website"}),
    ("docs", {"docs", "documentation", "manual", "reference", "spec", "release", "changelog"}),
    ("database", {"database", "db", "sql", "table", "tables", "record", "records", "row", "rows", "query"}),
    ("repository", {"repo", "repository", "codebase", "git", "branch", "commit", "diff", "pull", "pr"}),
    ("filesystem", {"file", "files", "directory", "folder", "path", "grep", "ripgrep", "read"}),
    ("logs", {"log", "logs", "traceback", "stack", "stderr", "stdout", "failure", "error"}),
    ("tickets", {"ticket", "tickets", "issue", "issues", "jira", "linear", "backlog"}),
)
_BUSINESS_HIGH_TERMS = {
    "production",
    "prod",
    "incident",
    "security",
    "secret",
    "payment",
    "billing",
    "invoice",
    "customer",
    "legal",
    "delete",
    "deploy",
}
_BUSINESS_MEDIUM_TERMS = {
    "account",
    "user",
    "voucher",
    "vouchers",
    "order",
    "orders",
    "revenue",
    "subscription",
    "access",
}


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _contains_phrase(text_l: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        if " " in phrase:
            if phrase in text_l:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])", text_l):
            return True
    return False


def _bucket_score(score: int) -> str:
    if score <= 0:
        return "none"
    if score == 1:
        return "low"
    if score <= 3:
        return "medium"
    return "high"


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "two_three"
    if value <= 10:
        return "four_ten"
    return "gte_11"


def _source_dependency(words: set[str], *, requires_current_state: bool, lookup_like: bool, url_like: bool) -> str:
    if url_like:
        return "web"
    for name, terms in _SOURCE_GROUPS:
        if words & terms:
            return name
    if requires_current_state or lookup_like:
        return "unknown"
    return "none"


def prompt_difficulty_features_from_text(text: str | None) -> dict[str, Any]:
    """Return metadata-only task difficulty and downgrade-risk features."""
    safe_text = text if isinstance(text, str) else ""
    text_l = safe_text.lower()
    words = _tokens(safe_text)
    word_count = len(_WORD_RE.findall(safe_text))
    sentence_count = len([part for part in re.split(r"[.!?\n]+", safe_text) if part.strip()])
    question_present = bool(_QUESTION_RE.search(safe_text)) or words.intersection({"what", "why", "how", "when", "where", "who"})
    step_markers = len(_STEP_MARKER_RE.findall(safe_text))
    acknowledgement = _contains_phrase(text_l, _ACK_PHRASES) and word_count <= 18
    correction = bool(words & {"actually", "correction", "meant", "instead"})
    requires_current_state = bool(words & _CURRENT_STATE_TERMS)
    lookup_like = bool(words & _LOOKUP_TERMS)
    executable_like = bool(words & _EXECUTABLE_TERMS)
    debugging_like = bool(words & _DEBUG_TERMS) or bool(_CODE_OR_ERROR_RE.search(safe_text))
    planning_like = bool(words & _PLANNING_TERMS)
    summary_like = bool(words & _SUMMARY_TERMS)
    edit_like = bool(words & _EDIT_TERMS)
    status_like = bool(words & _STATUS_TERMS)
    verification_required = bool(words & _VERIFY_TERMS) or requires_current_state or debugging_like
    source_dependency = _source_dependency(
        words,
        requires_current_state=requires_current_state,
        lookup_like=lookup_like,
        url_like=bool(_URL_RE.search(safe_text)),
    )
    explicit_source = source_dependency not in {"none", "unknown"}

    multi_step_score = 0
    multi_step_score += 1 if word_count >= 30 or sentence_count >= 3 else 0
    multi_step_score += 1 if step_markers else 0
    multi_step_score += 1 if lookup_like and requires_current_state else 0
    multi_step_score += 1 if explicit_source else 0
    multi_step_score += 1 if debugging_like else 0
    multi_step_score += 1 if len(words & {"and", "then", "compare", "across", "after"}) >= 2 else 0

    dependency_score = 0
    dependency_score += 2 if explicit_source else 0
    dependency_score += 2 if requires_current_state else 0
    dependency_score += 1 if lookup_like else 0
    dependency_score += 1 if debugging_like else 0
    dependency_score += 1 if source_dependency == "unknown" and requires_current_state and lookup_like else 0

    if acknowledgement:
        prompt_role = "acknowledgement"
    elif correction:
        prompt_role = "correction"
    elif any(term in text_l for term in ("respond as", "use this format", "follow these instructions")):
        prompt_role = "meta-instruction"
    elif sentence_count > 1 or question_present:
        prompt_role = "follow-up" if correction else "request"
    else:
        prompt_role = "request"

    if acknowledgement:
        task_intent = "acknowledgement"
    elif debugging_like:
        task_intent = "debugging"
    elif lookup_like and (requires_current_state or explicit_source):
        task_intent = "data_lookup"
    elif lookup_like:
        task_intent = "investigation"
    elif planning_like:
        task_intent = "planning"
    elif edit_like:
        task_intent = "edit/rewrite"
    elif summary_like:
        task_intent = "summary"
    elif status_like:
        task_intent = "status_check"
    elif question_present:
        task_intent = "question"
    else:
        task_intent = "instruction" if executable_like else "question"

    if acknowledgement:
        actionability = "passive"
    elif any(words & terms for _, terms in _SOURCE_GROUPS) or executable_like:
        actionability = "executable"
    elif question_present or task_intent in {"question", "summary", "status_check"}:
        actionability = "informational"
    else:
        actionability = "informational"
    if words & {"delete", "deploy", "write", "update", "change"}:
        actionability = "state-changing"

    if acknowledgement or (question_present and not requires_current_state and source_dependency == "none" and not debugging_like):
        answerability = "likely"
    elif requires_current_state or source_dependency != "none" or debugging_like:
        answerability = "unlikely"
    else:
        answerability = "unknown"

    if words & _BUSINESS_HIGH_TERMS:
        impact = "high"
    elif words & _BUSINESS_MEDIUM_TERMS or verification_required:
        impact = "medium"
    else:
        impact = "low"

    if acknowledgement:
        downgrade_risk = "safe"
    elif requires_current_state and (lookup_like or source_dependency != "none"):
        downgrade_risk = "block"
    elif debugging_like or source_dependency in {"database", "repository", "filesystem", "logs", "tickets", "web"}:
        downgrade_risk = "block"
    elif verification_required or dependency_score >= 2 or impact == "high":
        downgrade_risk = "caution"
    else:
        downgrade_risk = "safe"

    return {
        "schema": PROMPT_DIFFICULTY_FEATURE_SCHEMA,
        "detector_version": "2026-06-09.1",
        "task_intent": task_intent,
        "requires_current_state": requires_current_state,
        "external_source_dependency": source_dependency,
        "multi_step_likelihood_bucket": _bucket_score(multi_step_score),
        "tool_or_data_dependency_likelihood": _bucket_score(dependency_score),
        "verification_required": verification_required,
        "answerability_from_prompt_only": answerability,
        "business_or_user_impact": impact,
        "downgrade_risk": downgrade_risk,
        "prompt_role": prompt_role,
        "actionability": actionability,
        "signal_buckets": {
            "word_count": _count_bucket(word_count),
            "sentence_count": _count_bucket(sentence_count),
            "step_marker_count": _count_bucket(step_markers),
            "source_signal_count": _count_bucket(sum(1 for _, terms in _SOURCE_GROUPS if words & terms)),
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompt_text_included": False,
            "raw_entities_included": False,
            "raw_values_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "source_text_included": False,
        },
    }
