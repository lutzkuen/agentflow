from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable

from tokenclaw.public_metadata import public_id
from tokenclaw.pricing import estimate_cost, pricing_basis


SCHEMA = "tokenclaw.orchestrator_research_plan.v1"
ACTIVATION_BURNDOWN_SCHEMA = "tokenclaw.activation_burndown.v1"
ACTIVATION_BURNDOWN_ROW_SCHEMA = "tokenclaw.activation_burndown_row.v1"
EVIDENCE_TO_ACTIVATION_BURNDOWN_SCHEMA = "tokenclaw.evidence_to_activation_burndown.v1"
EVIDENCE_TO_ACTIVATION_LEDGER_SCHEMA = "tokenclaw.evidence_to_activation_next_action_ledger.v1"
EVIDENCE_TO_ACTIVATION_LEDGER_ENTRY_SCHEMA = "tokenclaw.evidence_to_activation_next_action_ledger_entry.v1"
LOCAL_ACTIVATION_NEXT_ACTION_QUEUE_SCHEMA = "tokenclaw.local_activation_next_action_queue.v1"
LOCAL_ACTIVATION_NEXT_ACTION_QUEUE_ENTRY_SCHEMA = "tokenclaw.local_activation_next_action_queue_entry.v1"
LOCAL_ACTIVATION_SUCCESSOR_ACTION_SCHEMA = "tokenclaw.local_activation_successor_action.v1"
LOCAL_ACTIVATION_SUCCESSOR_DECISION_SCHEMA = "tokenclaw.local_activation_successor_decision.v1"
PREVIEW_VERIFIED_SUCCESSOR_GATE_SCHEMA = "tokenclaw.preview_verified_activation_successor_gate.v1"
MANAGED_PREVIEW_HEALTH_GATE_SCHEMA = "tokenclaw.managed_activation_preview_health_gate.v1"
ACTIVATION_FEEDBACK_FRESHNESS_GATE_SCHEMA = "tokenclaw.activation_feedback_evidence_freshness_gate.v1"
OPENAI_ACTIVE_LOCAL_POLICY_OUTCOME_GATE_SCHEMA = "tokenclaw.openai_routing_active_local_policy_outcome_gate.v1"
FULL_ROLLOUT_CRUNCH_ACTIVATION_MEASUREMENT_SCHEMA = "tokenclaw.full_rollout_crunch_activation_measurement.v1"
FULL_ROLLOUT_CRUNCH_KEEP_ACTIVE_GATE_SCHEMA = "tokenclaw.full_rollout_crunch_keep_active_regression_gate.v1"
FULL_ROLLOUT_CRUNCH_ACTIVATION_OUTCOME_SCHEMA = "tokenclaw.full_rollout_crunch_activation_outcome.v1"
FULL_ROLLOUT_CRUNCH_POST_ROLLOUT_RANKING_SCHEMA = "tokenclaw.full_rollout_crunch_post_rollout_cohort_ranking.v1"
FULL_ROLLOUT_CRUNCH_POST_ROLLOUT_RANKING_ENTRY_SCHEMA = "tokenclaw.full_rollout_crunch_post_rollout_cohort_ranking_entry.v1"
LOW_BACKLOG_MILESTONE_TITLE = "Rank next savings milestone from local telemetry evidence gaps"
OPENAI_MIN_HOLDOUT_VOLUME = 10

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
    r"\b((?:(?:request|session|thread|tenant|candidate|run|trace)[_-]?id)|(?:cache[_-]?(?:id|key)))\s*[:=]\s*[\"']?([A-Za-z0-9_.:/@-]{6,})",
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
_SUCCESS_LINE_RE = re.compile(
    r"(?:verdict[:\s]+pass\b|quality[-_]gate[-_]passed|eval[-_]pass\b|"
    r"canary[-_]holdout[-_]thresholds[-_]met|promotion[-_]thresholds[-_]met|"
    r"test[-_]verdict[-_]pass|pass[-_]threshold[-_]met|offline[-_]fixture[-_]passed)",
    re.IGNORECASE,
)
_MANAGED_OMISSION_LINE_RE = re.compile(
    r"(?:server[-_]content[-_]processing|provider[-_]body[-_]rewrite|"
    r"prompt[-_]replacement|no[-_]local[-_]representation|local[-_]representation[-_]missing|"
    r"unsupported[-_]local[-_]executor|capability[-_]mismatch)",
    re.IGNORECASE,
)
_MISSING_MEASUREMENT_LINE_RE = re.compile(
    r"(?:missing[-_]crunch[-_]measurement|missing[-_]managed[-_]recommendation[-_]health|"
    r"missing[-_]request[-_]shape[-_]rollup|emit[-_][a-z]+[-_](?:report|measurement)|"
    r"need[-_]more[-_]samples\b|insufficient[-_]samples\b|missing[-_]lifecycle[-_]measurement)",
    re.IGNORECASE,
)
_SAFETY_STOP_UNCLASSIFIED_RE = re.compile(
    r"(?:safety[-_]stop(?:ped)?|stopped[-_]by[-_]safety|canary[-_]safety[-_]stop|safety[-_]gate[-_]block(?:ed)?)",
    re.IGNORECASE,
)
_SAFETY_STOP_SIGNAL_RE = re.compile(
    r"(?:"
    r"\b(?:blocker|blocked|skip(?:ped)?|omitted|reason|status|cohort|verdict)\s*[:= -]*[\"']?"
    r"(?:safety[-_]stop(?:ped)?|safety[-_]stopped)"
    r"|(?:safety[-_]stop(?:ped)?|safety[-_]stopped)\b.*\b(?:by|blocker|blocked|skip(?:ped)?|tripped|regression|failed|gate|canary)\b"
    r")",
    re.IGNORECASE,
)
_CANARY_COHORT_SKIP_RE = re.compile(
    r"(?:not[-_]in[-_](?:canary|applied|holdout)[-_]cohort|"
    r"canary[-_]cohort[-_](?:not[-_]selected|skipped|bypassed)|"
    r"(?:canary|routing|crunch|cache)[-_]canary[-_](?:not[-_]applied|not[-_]selected|skipped)|"
    r"no[-_]canary[-_](?:assignment|selection|cohort)|"
    r"holdout[-_]not[-_]selected|"
    r"outside[-_]canary[-_]window|"
    r"activation[-_]canary[-_](?:skipped|not[-_]applied))",
    re.IGNORECASE,
)
_BELOW_THRESHOLD_SKIP_RE = re.compile(
    r"(?:(?:context|token)[-_](?:count[-_])?below[-_](?:minimum|threshold)|"
    r"too[-_]short[-_]to[-_](?:crunch|compress|summarize)|"
    r"compression[-_]ratio[-_](?:insufficient|below[-_]threshold)|"
    r"context[-_]length[-_]too[-_]short|"
    r"below[-_]crunch[-_](?:minimum|threshold)|"
    r"below[-_](?:summary|cache)[-_]threshold)",
    re.IGNORECASE,
)
_STALE_LIFECYCLE_SKIP_RE = re.compile(
    r"(?:stale[-_](?:lifecycle|canary|activation)[-_](?:evidence|feedback|data)|"
    r"(?:lifecycle|activation|canary)[-_]evidence[-_](?:stale|outdated|expired)|"
    r"evidence[-_](?:stale|outdated)[-_]retry|"
    r"activation[-_]data[-_](?:not[-_]fresh|stale|outdated))",
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

_ACTIVATION_FEEDBACK_NEW_SANITIZED_EVIDENCE_RE = re.compile(
    r"(?:new[-_ ]sanitized[-_ ]evidence|sanitized[-_ ]activation[-_ ]feedback[-_ ]evidence|"
    r"bounded[-_ ]successor[-_ ]input|metadata[-_ ]only[-_ ]activation[-_ ]feedback)",
    re.IGNORECASE,
)
_ACTIVATION_FEEDBACK_HUMAN_REVIEW_REQUIRED_RE = re.compile(
    r"(?:human[-_ ]review[-_ ]required|needs[-_ ]human[-_ ]review|manual[-_ ]review[-_ ]required)",
    re.IGNORECASE,
)
_METADATA_ONLY_TRUE_RE = re.compile(r"metadata[-_ ]only\s*[:=]\s*(?:true|1|yes)", re.IGNORECASE)
_AGGREGATE_ONLY_TRUE_RE = re.compile(r"aggregate[-_ ]only\s*[:=]\s*(?:true|1|yes)", re.IGNORECASE)
_NEXT_ACTION_RE = re.compile(r"next[-_ ]action\s*[:=]\s*[\"']?([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_PASS_VERIFIED_TOKENCLAW_PORT_RE = re.compile(
    r"\bpass\b.*\bverified\b.*\b(?:tokenclaw|tokenclaw)[-_ ]?port\b",
    re.IGNORECASE,
)


def _is_safety_stop_signal_line(line: str) -> bool:
    lowered = line.lower()
    if "safety-stop" not in lowered and "safety_st" not in lowered:
        return False
    if _SUCCESS_LINE_RE.search(line):
        return False
    return bool(_SAFETY_STOP_SIGNAL_RE.search(line))


def _activation_feedback_diagnostic_metadata_from_line(line: str, reason: str) -> dict[str, Any] | None:
    normalized_reason = _normal_diagnostic_token(reason)
    if normalized_reason != "activation-feedback-blocker-review":
        return None
    if _ACTIVATION_FEEDBACK_HUMAN_REVIEW_REQUIRED_RE.search(line):
        return {
            "activation_feedback_diagnostic_classification": {
                "schema": "tokenclaw.activation_feedback_diagnostic_classification.v1",
                "status": "human-review-required",
                "decision": "keep-blocked",
                "reason": "human-review-required",
                "privacy": _candidate_privacy(),
            },
            "review_status": "human-review-required",
        }
    if not _ACTIVATION_FEEDBACK_NEW_SANITIZED_EVIDENCE_RE.search(line):
        return None

    next_action_match = _NEXT_ACTION_RE.search(line)
    next_action = (
        _normal_diagnostic_token(next_action_match.group(1))
        if next_action_match
        else "review-new-sanitized-activation-feedback-evidence"
    )
    metadata_only = bool(_METADATA_ONLY_TRUE_RE.search(line))
    aggregate_only = bool(_AGGREGATE_ONLY_TRUE_RE.search(line))
    privacy = _candidate_privacy()
    privacy["metadata_only"] = metadata_only or privacy["metadata_only"]
    privacy["aggregate_only"] = aggregate_only or privacy["aggregate_only"]
    classification = {
        "schema": "tokenclaw.activation_feedback_diagnostic_classification.v1",
        "status": "new-sanitized-evidence",
        "decision": "emit-bounded-successor-input",
        "reason": "new-sanitized-evidence",
        "next_action": sanitize_value(next_action),
        "privacy": privacy,
    }
    return {
        "activation_feedback_diagnostic_classification": classification,
        "diagnostic_evidence_status": "new-sanitized-evidence",
        "review_status": "new-sanitized-evidence",
        "next_action_override": next_action,
        "example_override": "metadata-only activation-feedback diagnostic evidence",
    }

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

_RESOLVED_ACTIVATION_FEEDBACK_PASS_DIAGNOSTIC_CLASS = "resolved-activation-feedback-pass-diagnostic"
_RESOLVED_ACTIVATION_FEEDBACK_PASS_KEEP_BLOCKED_REASON = (
    "resolved-activation-feedback-pass-diagnostic-suppressed"
)
_RESOLVED_ACTIVATION_FEEDBACK_PASS_NEXT_ACTION = (
    "suppress-resolved-activation-feedback-diagnostic"
)
_TERMINAL_ACTIVATION_SUCCESSOR_STATES = {
    "resolved-no-action",
}


def _is_resolved_pass_diagnostic(reason: Any, diagnostic_class: Any = None) -> bool:
    reason_text = _normal_diagnostic_token(reason)
    class_text = _normal_diagnostic_token(diagnostic_class)
    for text in {reason_text, class_text}:
        if not text:
            continue
        if text in _PASS_DIAGNOSTIC_REASONS:
            return True
        if text.startswith("pass-") and "verified-tokenclaw-port" not in text and "verified-tokenclaw-port" not in text:
            return True
        if text.endswith("-passed"):
            return True
    return False

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
            "missing-canary-lifecycle-evidence",
            "missing-applied-coverage",
            "missing-holdout-coverage",
            "insufficient-cohort-coverage",
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
        "class": "activation-feedback-blocker-review",
        "priority": 80,
        "aliases": ("activation-feedback-blocker-review", "bounded-activation-feedback-review"),
        "backlog_action": "create-ready-issue",
        "unblock_path": "Cut a bounded activation-feedback issue from sanitized repeated diagnostics, then record the narrower blocker in the local action ledger.",
        "acceptance_check": "The next research plan emits a durable activation-feedback ledger entry with a concrete next action, stable fingerprint, and metadata-only privacy flags.",
    },
    {
        "class": "live-service-deploy-failed-after-development",
        "priority": 82,
        "aliases": (
            "live-service-deploy-failed-after-development",
            "live-service-deploy-failed-after-development-landing",
            "deploy-failed-after-development",
            "development-landing-deploy-failed",
        ),
        "backlog_action": "needs-human-review",
        "unblock_path": "Keep prod restart ownership with the shell guard and record a bounded deployment-health follow-up instead of creating generic activation work.",
        "acceptance_check": "The next research plan reports the deploy diagnostic with a stable fingerprint, concrete next action, activation-feedback local action family, and metadata-only privacy flags.",
    },
    {
        "class": "pass-verified-tokenclaw-port",
        "priority": 84,
        "aliases": (
            "pass-verified-tokenclaw-port",
            "pass-verified-tokenclaw-port-4001",
            "pass-verified-tokenclaw-port",
            "verified-tokenclaw-port",
            "verified-tokenclaw-port-4001",
        ),
        "backlog_action": "needs-human-review",
        "unblock_path": "Record the verified dev-port smoke result as bounded activation-feedback context and keep any live deploy follow-up narrow.",
        "acceptance_check": "The next research plan reports the verified-port diagnostic with a stable fingerprint, concrete next action, activation-feedback local action family, and metadata-only privacy flags.",
    },
    {
        "class": "unclassified-skip-or-blocker",
        "priority": 90,
        "aliases": ("unclassified-skip-or-blocker",),
        "backlog_action": "needs-human-review",
        "unblock_path": "Trace the unclassified skip/blocker to a bounded diagnostic reason before creating activation work.",
        "acceptance_check": "The next research plan reports a classified reason or a narrow issue for the emitting report.",
    },
)

_CRUNCH_OPPORTUNITY_REPORT_KEYS = (
    "request_shape_crunch_opportunity",
    "request_shape_crunch_opportunity_dry_run",
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

_ACTIVATION_FEEDBACK_BLOCKER_KEEP_BLOCKED_REASON = (
    "activation-feedback-blocker-review-already-resolved-to-bounded-local-action-ledger"
)
_ACTIVATION_FEEDBACK_BLOCKER_NEXT_ACTION = (
    "keep-activation-feedback-blocker-review-blocked-until-new-sanitized-local-evidence"
)
_ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_KEEP_BLOCKED_REASON = (
    "activation-feedback-missing-dependency-evidence-needs-sanitized-source-report"
)
_ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_NEXT_ACTION = (
    "keep-missing-dependency-evidence-blocked-until-sanitized-source-report"
)
_ACTIVATION_FEEDBACK_NO_LOCAL_REPRESENTATION_KEEP_BLOCKED_REASON = (
    "activation-feedback-no-local-representation-resolved-to-review-only-local-artifact"
)
_ACTIVATION_FEEDBACK_NO_LOCAL_REPRESENTATION_NEXT_ACTION = (
    "record-review-only-local-representation-and-wait-for-supported-local-action"
)
_ACTIVATION_FEEDBACK_STALE_EVIDENCE_KEEP_BLOCKED_REASON = (
    "activation-feedback-stale-evidence-blocked-until-fresh-local-evidence"
)
_ACTIVATION_FEEDBACK_STALE_EVIDENCE_NEXT_ACTION = (
    "collect-fresh-activation-feedback-evidence-before-activation"
)
_ACTIVATION_FEEDBACK_DEPLOY_FAILURE_KEEP_BLOCKED_REASON = (
    "live-service-deploy-failed-after-development-owned-by-shell-guard"
)
_ACTIVATION_FEEDBACK_DEPLOY_FAILURE_NEXT_ACTION = (
    "record-live-service-deploy-failure-and-wait-for-shell-guard-health"
)
_ACTIVATION_FEEDBACK_PORT_VERIFIED_KEEP_BLOCKED_REASON = (
    "pass-verified-tokenclaw-port-recorded-as-dev-smoke-context"
)
_ACTIVATION_FEEDBACK_PORT_VERIFIED_NEXT_ACTION = (
    "record-tokenclaw-dev-port-verification-and-keep-prod-deploy-follow-up-narrow"
)
_ACTIVATION_FEEDBACK_FRESHNESS_MAX_AGE_HOURS = 72.0

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


def _issue_closed_time(issue: dict[str, Any]) -> datetime | None:
    return _parse_time(
        issue.get("closedAt")
        or issue.get("closed_at")
        or issue.get("updatedAt")
        or issue.get("updated_at")
    )


def _is_recent_closed_issue(issue: dict[str, Any], *, now: datetime, recent_days: int) -> bool:
    if _is_open(issue):
        return False
    closed_at = _issue_closed_time(issue)
    if closed_at is None:
        return True
    age_seconds = (now - closed_at).total_seconds()
    return age_seconds >= 0 and age_seconds <= max(0, recent_days) * 86400


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
    metadata: dict[str, dict[str, Any]] = {}
    for loaded in _load_log_text(log_sources):
        for raw_line in loaded["text"].splitlines():
            line = redact_text(raw_line.strip())
            if not line:
                continue
            lowered = line.lower()
            matched = False
            for match in _DIAGNOSTIC_RE.finditer(line):
                reason = match.group(1).strip(" .'\",").lower().replace(" ", "-")
                if reason.startswith("activation-feedback-blocker-review"):
                    reason = "activation-feedback-blocker-review"
                if reason:
                    if "safety-stop" in reason and not _is_safety_stop_signal_line(line):
                        continue
                    counter[reason] += 1
                    line_metadata = _activation_feedback_diagnostic_metadata_from_line(line, reason)
                    if line_metadata:
                        metadata[reason] = {**metadata.get(reason, {}), **line_metadata}
                        examples[reason] = str(line_metadata.get("example_override") or examples.get(reason) or line[:240])
                    else:
                        examples.setdefault(reason, line[:240])
                    matched = True
            for term in _KNOWN_DIAGNOSTIC_TERMS:
                if term in lowered:
                    reason = term.lower().replace(" ", "-")
                    if reason == "safety-stop" and not _is_safety_stop_signal_line(line):
                        continue
                    counter[reason] += 1
                    line_metadata = _activation_feedback_diagnostic_metadata_from_line(line, reason)
                    if line_metadata:
                        metadata[reason] = {**metadata.get(reason, {}), **line_metadata}
                        examples[reason] = str(line_metadata.get("example_override") or examples.get(reason) or line[:240])
                    else:
                        examples.setdefault(reason, line[:240])
                    matched = True
            if not matched and ("skip" in lowered or "blocked" in lowered or "omitted" in lowered):
                reason = "unclassified-skip-or-blocker"
                counter[reason] += 1
                examples.setdefault(reason, line[:240])
            if _PASS_VERIFIED_TOKENCLAW_PORT_RE.search(line):
                reason = "pass-verified-tokenclaw-port"
                counter[reason] += 1
                examples.setdefault(reason, "metadata-only tokenclaw dev-port verification")
    diagnostics: list[dict[str, Any]] = []
    for reason, count in counter.most_common(limit):
        item = {"reason": reason, "count": count, "example": examples.get(reason, "")}
        if reason in metadata:
            item.update(sanitize_value({key: value for key, value in metadata[reason].items() if key != "example_override"}))
        diagnostics.append(item)
    return diagnostics


def _diagnostic_taxonomy(reason: Any) -> dict[str, Any] | None:
    text = str(reason or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text or text in _PASS_DIAGNOSTIC_REASONS:
        return None
    if (
        (text.startswith("pass-") or text.endswith("-passed"))
        and "verified-tokenclaw-port" not in text
        and "verified-tokenclaw-port" not in text
    ):
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
                "backlog_action": enriched.get("backlog_action_override") or taxonomy["backlog_action"],
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


def _diagnostic_fingerprint(diagnostic_class: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", diagnostic_class.lower().strip()).strip("-")
    return f"tokenclaw.repeated-diagnostic.{normalized}.v1"


def _sample_count_bucket(value: Any) -> str:
    count = _to_int(value)
    if count <= 0:
        return "none"
    if count < 10:
        return "lt_10"
    if count < 100:
        return "10_99"
    if count < 1000:
        return "100_999"
    return "gte_1000"


def _activation_loop_lifecycle_context(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    loop = stats_summary.get("evidence_to_activation_loop")
    if not isinstance(loop, dict):
        return None
    levers = [row for row in loop.get("levers") or [] if isinstance(row, dict)]
    if not levers:
        return None
    blocked = [
        row
        for row in levers
        if str(row.get("state") or "") in {"missing-evidence", "blocked"}
        and any(str(item or "").strip() for item in row.get("blocker_codes") or [])
    ]
    row = (blocked or levers)[0]
    blockers = [str(item) for item in row.get("blocker_codes") or [] if str(item or "").strip()]
    blocker = blockers[0] if blockers else ""
    state = str(row.get("state") or "unknown")
    status = "specific-blocker" if blocker else "lifecycle-state-known"
    return {
        "status": status,
        "source": "evidence_to_activation_loop",
        "report_schema": sanitize_value(row.get("evidence_source") or loop.get("schema")),
        "action_family": sanitize_value(row.get("local_action_family") or row.get("lever") or "unknown"),
        "lifecycle_state": sanitize_value(state),
        "blocker_code": sanitize_value(blocker or state),
        "sample_count_bucket": _sample_count_bucket(row.get("sample_count")),
        "next_action": sanitize_value(row.get("next_action") or loop.get("summary", {}).get("top_next_action")),
        "aggregate_only_replaced": bool(blocker),
        "aggregate_only_cleared": status == "lifecycle-state-known" and state not in {"missing-evidence", "blocked"},
        "privacy": _candidate_privacy(),
    }


def _pass_through_lifecycle_context(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    report = stats_summary.get("pass_through_routing_report")
    if not isinstance(report, dict):
        return None
    for bucket in report.get("buckets") or []:
        if not isinstance(bucket, dict):
            continue
        lifecycle = _routing_lifecycle_evidence(bucket)
        if not isinstance(lifecycle, dict):
            continue
        blockers = [str(item) for item in lifecycle.get("blocker_codes") or [] if str(item or "").strip()]
        if not blockers:
            continue
        return {
            "status": "specific-blocker",
            "source": "pass_through_routing_report",
            "report_schema": sanitize_value(lifecycle.get("schema") or report.get("schema")),
            "action_family": "routing",
            "lifecycle_state": sanitize_value(lifecycle.get("status") or "missing-evidence"),
            "blocker_code": sanitize_value(blockers[0]),
            "sample_count_bucket": _sample_count_bucket(bucket.get("sample_count")),
            "next_action": (
                "activate-anthropic-routing-canary-cohorts"
                if str(bucket.get("provider") or "") == "anthropic"
                else "activate-openai-routing-canary-cohorts"
            ),
            "aggregate_only_replaced": True,
            "aggregate_only_cleared": False,
            "privacy": _candidate_privacy(),
        }
    return None


def _openai_feedback_lifecycle_context(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    impact = stats_summary.get("openai_canary_impact")
    if not isinstance(impact, dict):
        return None
    feedback = impact.get("activation_lifecycle_feedback")
    if not isinstance(feedback, dict):
        return None
    state_counts = _breakdown_counts(feedback.get("state_breakdown"))
    top_state = next(iter(state_counts), "unknown")
    blocker = _openai_lifecycle_omission_reason(feedback)
    row, _ = _top_openai_routing_canary_row(stats_summary)
    action = _openai_routing_candidate_action(row, stats_summary) if row is not None else {"omission_reason": blocker or "none"}
    blocker = blocker or (None if action.get("omission_reason") == "none" else str(action.get("omission_reason") or ""))
    return {
        "status": "specific-blocker" if blocker else "lifecycle-state-known",
        "source": "openai_canary_impact.activation_lifecycle_feedback",
        "report_schema": sanitize_value(feedback.get("schema") or impact.get("schema")),
        "action_family": "routing",
        "lifecycle_state": sanitize_value(top_state),
        "blocker_code": sanitize_value(blocker or top_state),
        "sample_count_bucket": _sample_count_bucket(row.get("sample_count") if isinstance(row, dict) else feedback.get("family_event_count")),
        "next_action": sanitize_value(row.get("next_action") if isinstance(row, dict) else "inspect_openai_canary_lifecycle_evidence"),
        "cohort_lifecycle_metadata": _cohort_lifecycle_metadata(feedback, limit=5),
        "aggregate_only_replaced": bool(blocker),
        "aggregate_only_cleared": not blocker,
        "privacy": _candidate_privacy(),
    }


def _aggregate_only_lifecycle_context(stats_summary: dict[str, Any]) -> dict[str, Any]:
    for context in (
        _activation_loop_lifecycle_context(stats_summary),
        _openai_feedback_lifecycle_context(stats_summary),
        _pass_through_lifecycle_context(stats_summary),
    ):
        if context is not None and context.get("status") == "specific-blocker":
            return context
    for context in (
        _activation_loop_lifecycle_context(stats_summary),
        _openai_feedback_lifecycle_context(stats_summary),
        _pass_through_lifecycle_context(stats_summary),
    ):
        if context is not None:
            return context
    return {
        "status": "specific-blocker",
        "source": "diagnostic-log",
        "report_schema": None,
        "action_family": "activation-feedback",
        "lifecycle_state": "missing_feedback",
        "blocker_code": "missing-lifecycle-feedback",
        "sample_count_bucket": "unknown",
        "next_action": "emit-activation-lifecycle-feedback",
        "aggregate_only_replaced": True,
        "aggregate_only_cleared": False,
        "privacy": _candidate_privacy(),
    }


def _resolve_aggregate_only_diagnostics(
    diagnostics: list[dict[str, Any]],
    stats_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not diagnostics:
        return diagnostics
    context = _aggregate_only_lifecycle_context(stats_summary)
    resolved: list[dict[str, Any]] = []
    for item in diagnostics:
        reason = str(item.get("reason") or "")
        taxonomy = _diagnostic_taxonomy(reason)
        if taxonomy is None or taxonomy.get("class") != "aggregate-only":
            resolved.append(item)
            continue
        if context.get("aggregate_only_cleared"):
            continue
        replacement_reason = str(context.get("blocker_code") or "missing-lifecycle-feedback")
        replacement = dict(item)
        replacement["reason"] = replacement_reason
        replacement["diagnostic_class"] = "missing-lifecycle-feedback"
        replacement["lifecycle_context"] = context
        replacement["example"] = (
            f"aggregate-only diagnostic replaced by {replacement_reason}; "
            f"source={context.get('source')}; state={context.get('lifecycle_state')}; "
            f"next_action={context.get('next_action')}"
        )
        resolved.append(replacement)
    return resolved


def _classify_unclassified_diagnostic(example: str) -> dict[str, Any] | None:
    """
    Map an unclassified-skip-or-blocker example line to a specific class.
    Returns None to signal the item should be dropped (ignore-success).
    """
    if _SUCCESS_LINE_RE.search(example):
        return None
    if _MANAGED_OMISSION_LINE_RE.search(example):
        return {
            "reason": "unsupported-provider-action",
            "diagnostic_class": "unsupported-provider-action",
            "backlog_action": "create-ready-issue",
            "reclassification_source": "managed-omission-pattern",
        }
    if _MISSING_MEASUREMENT_LINE_RE.search(example):
        return {
            "reason": "missing-dependency-evidence",
            "diagnostic_class": "missing-dependency-evidence",
            "backlog_action": "create-ready-issue",
            "reclassification_source": "missing-measurement-pattern",
        }
    if _SAFETY_STOP_UNCLASSIFIED_RE.search(example):
        return {
            "reason": "safety-stop",
            "diagnostic_class": "safety-stop",
            "backlog_action": "create-ready-issue",
            "reclassification_source": "safety-stop-unclassified-pattern",
        }
    if _CANARY_COHORT_SKIP_RE.search(example):
        return {
            "reason": "missing-lifecycle-feedback",
            "diagnostic_class": "missing-lifecycle-feedback",
            "backlog_action": "create-ready-issue",
            "reclassification_source": "canary-cohort-skip-pattern",
        }
    if _BELOW_THRESHOLD_SKIP_RE.search(example):
        return {
            "reason": "missing-dependency-evidence",
            "diagnostic_class": "missing-dependency-evidence",
            "backlog_action": "create-ready-issue",
            "reclassification_source": "below-threshold-pattern",
        }
    if _STALE_LIFECYCLE_SKIP_RE.search(example):
        return {
            "reason": "stale-quality-evidence",
            "diagnostic_class": "stale-evidence",
            "backlog_action": "create-ready-issue",
            "reclassification_source": "stale-lifecycle-pattern",
        }
    return {
        "reason": "activation-feedback-blocker-review",
        "diagnostic_class": "activation-feedback-blocker-review",
        "backlog_action": "create-ready-issue",
        "reclassification_source": "no-match-bounded-human-review",
    }


def _resolve_unclassified_diagnostics(
    diagnostics: list[dict[str, Any]],
    stats_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not diagnostics:
        return diagnostics
    resolved: list[dict[str, Any]] = []
    for item in diagnostics:
        reason = str(item.get("reason") or "")
        if reason != "unclassified-skip-or-blocker":
            resolved.append(item)
            continue
        example = str(item.get("example") or "")
        classification = _classify_unclassified_diagnostic(example)
        if classification is None:
            continue
        replacement = dict(item)
        replacement["reason"] = classification["reason"]
        replacement["backlog_action_override"] = classification["backlog_action"]
        replacement["reclassification_source"] = classification["reclassification_source"]
        if classification["reason"] != "unclassified-skip-or-blocker":
            replacement["diagnostic_class"] = classification["diagnostic_class"]
        resolved.append(replacement)
    return resolved


def _normal_diagnostic_token(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return re.sub(
        r"-(?:(?:request|session|thread|tenant|candidate|run|trace)-?id|cache-?(?:id|key))$",
        "",
        text,
    )


_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b")


def _parse_public_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _diagnostic_evidence_timestamp(diagnostic: dict[str, Any], lifecycle_context: dict[str, Any]) -> tuple[str | None, str]:
    for key in (
        "latest_observed_at",
        "evidence_timestamp",
        "observed_at",
        "generated_at",
        "created_at",
    ):
        value = diagnostic.get(key)
        if value:
            return str(value), f"diagnostic.{key}"
    for key in (
        "latest_observed_at",
        "evidence_timestamp",
        "observed_at",
        "generated_at",
    ):
        value = lifecycle_context.get(key)
        if value:
            return str(value), f"lifecycle_context.{key}"
    example = str(diagnostic.get("example") or "")
    match = _ISO_TIMESTAMP_RE.search(example)
    if match:
        return match.group(0), "diagnostic.example"
    return None, "missing"


def _activation_feedback_freshness_gate(
    diagnostic: dict[str, Any],
    lifecycle_context: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    max_age_hours = _to_float(
        diagnostic.get("max_age_hours")
        or lifecycle_context.get("max_age_hours")
        or _ACTIVATION_FEEDBACK_FRESHNESS_MAX_AGE_HOURS,
        _ACTIVATION_FEEDBACK_FRESHNESS_MAX_AGE_HOURS,
    )
    timestamp, timestamp_source = _diagnostic_evidence_timestamp(diagnostic, lifecycle_context)
    parsed = _parse_public_timestamp(timestamp)
    age_hours = None
    if parsed is not None:
        age_hours = round((datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0, 3)
    reason_text = " ".join(
        str(part or "")
        for part in (
            reason,
            diagnostic.get("diagnostic_class"),
            diagnostic.get("example"),
            lifecycle_context.get("blocker_code"),
        )
    ).lower()
    superseded = any(
        token in reason_text
        for token in (
            "full-rollout-local-policy-active",
            "repeated-context-crunch-full-rollout-active",
            "already-resolved",
            "superseded",
        )
    )
    if superseded:
        status = "superseded-by-local-policy"
    elif parsed is None:
        status = "missing"
    elif age_hours is not None and age_hours > max_age_hours:
        status = "stale-blocked"
    else:
        status = "fresh"
    next_action = (
        "use-fresh-activation-feedback-evidence"
        if status == "fresh"
        else "keep-current-local-policy-and-suppress-stale-activation-issue"
        if status == "superseded-by-local-policy"
        else _ACTIVATION_FEEDBACK_STALE_EVIDENCE_NEXT_ACTION
    )
    return {
        "schema": ACTIVATION_FEEDBACK_FRESHNESS_GATE_SCHEMA,
        "status": status,
        "deterministic_decision": status,
        "next_action": next_action,
        "evidence_timestamp": sanitize_value(timestamp),
        "evidence_timestamp_source": sanitize_value(timestamp_source),
        "timestamp_present": parsed is not None,
        "age_hours": age_hours,
        "max_age_hours": max_age_hours,
        "lifecycle_source": sanitize_value(lifecycle_context.get("source") or "orchestrator-log-diagnostic"),
        "source_schema": sanitize_value(lifecycle_context.get("report_schema") or "tokenclaw.orchestrator_research_log_diagnostics.v1"),
        "reason": sanitize_value(reason or "stale-evidence"),
        "privacy": _candidate_privacy(),
    }


def _diagnostic_ledger_stage(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    reason = _normal_diagnostic_token(diagnostic.get("reason"))
    diagnostic_class = _normal_diagnostic_token(diagnostic.get("diagnostic_class") or reason)
    if reason == "unclassified-skip-or-blocker" or diagnostic_class == "unclassified-skip-or-blocker":
        reason = "activation-feedback-blocker-review"
        diagnostic_class = "activation-feedback-blocker-review"
    taxonomy = _diagnostic_taxonomy(diagnostic_class) or _diagnostic_taxonomy(reason)
    if taxonomy is not None:
        diagnostic_class = str(taxonomy["class"])
    if not reason:
        return None

    lifecycle_context = diagnostic.get("lifecycle_context") if isinstance(diagnostic.get("lifecycle_context"), dict) else {}
    text = " ".join(
        str(part or "")
        for part in (
            reason,
            diagnostic_class,
            diagnostic.get("example"),
            lifecycle_context.get("blocker_code"),
            lifecycle_context.get("next_action"),
            lifecycle_context.get("action_family"),
        )
    ).lower()
    count = _to_int(diagnostic.get("count") or diagnostic.get("observations"))
    source_lever = str(diagnostic.get("source_lever") or _diagnostic_source_lever(reason, diagnostic_class))

    stage: dict[str, Any] = {
        "lever": source_lever,
        "state": "blocked",
        "evidence_source": "tokenclaw.orchestrator_research_log_diagnostics.v1",
        "local_action_family": source_lever if source_lever in {"routing", "cache", "crunch"} else "activation-feedback",
        "next_action": "classify-activation-feedback-blocker-for-local-action-ledger",
        "blocker_codes": [reason],
        "sample_count": count,
        "projected_saved_usd": 0.0,
        "diagnostic_class": diagnostic_class,
        "diagnostic_reason": reason,
        "diagnostic_fingerprint": _diagnostic_fingerprint(diagnostic_class),
        "issue_worthy_status": (
            "ready" if count > 1 and str(diagnostic.get("backlog_action") or "") != "needs-human-review" else "review"
        ),
        "source": "repeated-diagnostic",
    }
    acceptance_check = str(
        diagnostic.get("acceptance_check")
        or (taxonomy or {}).get("acceptance_check")
        or ""
    ).strip()
    if acceptance_check:
        stage["verification_check"] = acceptance_check
    if diagnostic_class:
        stage["privacy"] = _candidate_privacy()

    if _is_resolved_pass_diagnostic(reason, diagnostic_class):
        pass_reason = reason or diagnostic_class or "pass"
        stage.update(
            {
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "state": "resolved-no-action",
                "cohort_bucket": f"activation-feedback:{pass_reason}",
                "fingerprint_cohort_bucket": f"activation-feedback:{pass_reason}",
                "next_action": _RESOLVED_ACTIVATION_FEEDBACK_PASS_NEXT_ACTION,
                "review_status": "resolved-to-terminal-pass-diagnostic",
                "issue_worthy_status": "suppressed",
                "keep_blocked_reason": _RESOLVED_ACTIVATION_FEEDBACK_PASS_KEEP_BLOCKED_REASON,
                "next_state": "resolved-no-action",
                "next_state_reason": _RESOLVED_ACTIVATION_FEEDBACK_PASS_KEEP_BLOCKED_REASON,
                "needed_resolution": ["new_sanitized_evidence"],
                "durable_action_ledger_entry": True,
                "terminal_successor_state": True,
                "managed_preview_required": False,
                "diagnostic_class": _RESOLVED_ACTIVATION_FEEDBACK_PASS_DIAGNOSTIC_CLASS,
                "diagnostic_reason": pass_reason,
                "diagnostic_fingerprint": _diagnostic_fingerprint(pass_reason),
                "activation_feedback_diagnostic_classification": {
                    "schema": "tokenclaw.activation_feedback_diagnostic_classification.v1",
                    "status": "resolved-pass-diagnostic-terminal",
                    "decision": "resolved-no-action",
                    "reason": _RESOLVED_ACTIVATION_FEEDBACK_PASS_KEEP_BLOCKED_REASON,
                    "next_action": _RESOLVED_ACTIVATION_FEEDBACK_PASS_NEXT_ACTION,
                    "privacy": _candidate_privacy(),
                },
            }
        )
    elif "missing-anthropic-canary-lifecycle-evidence" in text:
        stage.update(
            {
                "lever": "routing",
                "local_action_family": "routing",
                "state": "missing-evidence",
                "next_action": "activate-anthropic-routing-canary-cohorts",
            }
        )
    elif "missing-applied-coverage" in text or "missing-holdout-coverage" in text or "missing-lifecycle-feedback" in text:
        stage.update(
            {
                "lever": "routing",
                "local_action_family": "routing",
                "state": "missing-evidence",
                "next_action": "collect-routing-applied-and-holdout-lifecycle-evidence",
            }
        )
    elif "repeated-context-crunch-opportunity" in text:
        stage.update(
            {
                "lever": "crunch",
                "local_action_family": "crunch",
                "state": "activation-ready",
                "next_action": "stage-repeated-context-crunch-canary",
            }
        )
    elif "invalidation-evidence-missing" in text:
        stage.update(
            {
                "lever": "cache",
                "local_action_family": "cache",
                "state": "missing-evidence",
                "next_action": "collect-cache-invalidation-evidence",
            }
        )
    elif "unsupported-streaming-shape" in text or "streaming-replay-not-supported" in text:
        stage.update(
            {
                "lever": "cache",
                "local_action_family": "cache",
                "state": "blocked",
                "next_action": "add-streaming-cache-replay-support-or-route-to-crunch-canary",
            }
        )
    elif "tool-call-cache-disabled" in text or "unsafe-tool-calls-without-invalidation" in text:
        stage.update(
            {
                "lever": "cache",
                "local_action_family": "cache",
                "state": "missing-evidence",
                "next_action": "collect-tool-cache-invalidation-evidence",
            }
        )
    elif "thinking-routing-guard" in text:
        stage.update(
            {
                "lever": "routing",
                "local_action_family": "routing",
                "state": "missing-evidence",
                "next_action": "collect-thinking-routing-lifecycle-evidence",
            }
        )
    elif diagnostic_class == "unsupported-provider-action":
        stage.update(
            {
                "lever": "managed-recommendation",
                "local_action_family": "unknown",
                "state": "missing-evidence",
                "next_action": "map-recommendation-to-supported-local-executor",
            }
        )
    elif diagnostic_class == "missing-dependency-evidence":
        stage.update(
            {
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "state": "keep-blocked",
                "next_action": _ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_NEXT_ACTION,
                "review_status": "resolved-to-narrower-blocker",
                "issue_worthy_status": "blocked",
                "keep_blocked_reason": _ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_KEEP_BLOCKED_REASON,
                "next_state": "keep-blocked",
                "next_state_reason": _ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_KEEP_BLOCKED_REASON,
                "needed_resolution": [
                    "sanitized_source_report",
                    "dependency_evidence_summary",
                    "bounded_local_action_issue",
                ],
                "durable_action_ledger_entry": True,
                "dependency_evidence_status": "missing-sanitized-source-report",
                "missing_dependency_evidence_review": {
                    "schema": "tokenclaw.activation_feedback_missing_dependency_evidence_review.v1",
                    "status": "blocked",
                    "reason": _ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_KEEP_BLOCKED_REASON,
                    "next_action": _ACTIVATION_FEEDBACK_MISSING_DEPENDENCY_NEXT_ACTION,
                    "needed_resolution": [
                        "sanitized_source_report",
                        "dependency_evidence_summary",
                        "bounded_local_action_issue",
                    ],
                    "privacy": _candidate_privacy(),
                },
            }
        )
    elif diagnostic_class == "no-local-representation":
        stage.update(
            {
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "state": "keep-blocked",
                "next_action": _ACTIVATION_FEEDBACK_NO_LOCAL_REPRESENTATION_NEXT_ACTION,
                "review_status": "resolved-to-review-only-no-op",
                "issue_worthy_status": "blocked",
                "keep_blocked_reason": _ACTIVATION_FEEDBACK_NO_LOCAL_REPRESENTATION_KEEP_BLOCKED_REASON,
                "next_state": "keep-blocked",
                "next_state_reason": _ACTIVATION_FEEDBACK_NO_LOCAL_REPRESENTATION_KEEP_BLOCKED_REASON,
                "needed_resolution": [
                    "supported_file_backed_local_action",
                    "dry_run_or_canary_evidence",
                    "new_sanitized_evidence",
                ],
                "durable_action_ledger_entry": True,
                "policy_files_written": False,
                "managed_dependency": "optional",
                "local_action_representation": {
                    "schema": "tokenclaw.activation_feedback_local_action_representation.v1",
                    "status": "represented",
                    "representation_kind": "review-only-no-op",
                    "review_artifact": "evidence-to-activation-next-action-ledger",
                    "local_action_family": "activation-feedback",
                    "local_rule_available": False,
                    "file_backed_policy_available": False,
                    "dry_run_evidence_available": False,
                    "canary_evidence_available": False,
                    "managed_enforcement_required": False,
                    "managed_enforced": False,
                    "provider_body_rewrite_required": False,
                    "provider_body_rewrite": False,
                    "policy_files_written": False,
                    "reason": "no-supported-local-rule-dry-run-or-canary-representation",
                    "next_action": _ACTIVATION_FEEDBACK_NO_LOCAL_REPRESENTATION_NEXT_ACTION,
                    "privacy": _candidate_privacy(),
                },
            }
        )
    elif diagnostic_class == "stale-evidence":
        freshness_gate = _activation_feedback_freshness_gate(
            diagnostic,
            lifecycle_context,
            reason=reason or "stale-evidence",
        )
        freshness_status = str(freshness_gate.get("status") or "missing")
        fresh_enough = freshness_status == "fresh"
        stage.update(
            {
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "state": "missing-evidence" if fresh_enough else "keep-blocked",
                "next_action": (
                    "review-fresh-activation-feedback-evidence"
                    if fresh_enough
                    else _ACTIVATION_FEEDBACK_STALE_EVIDENCE_NEXT_ACTION
                ),
                "review_status": "freshness-gate-passed" if fresh_enough else "blocked-by-evidence-freshness-gate",
                "issue_worthy_status": "ready" if fresh_enough else "blocked",
                "keep_blocked_reason": None if fresh_enough else _ACTIVATION_FEEDBACK_STALE_EVIDENCE_KEEP_BLOCKED_REASON,
                "next_state": "missing-evidence" if fresh_enough else "keep-blocked",
                "next_state_reason": "fresh-activation-feedback-evidence-available"
                if fresh_enough
                else _ACTIVATION_FEEDBACK_STALE_EVIDENCE_KEEP_BLOCKED_REASON,
                "needed_resolution": []
                if fresh_enough
                else [
                    "fresh_sanitized_evidence_timestamp",
                    "activation_feedback_source_report",
                    "bounded_local_action_issue",
                ],
                "durable_action_ledger_entry": True,
                "activation_feedback_freshness_gate": freshness_gate,
                "evidence_freshness_status": freshness_status,
                "max_evidence_age_hours": freshness_gate.get("max_age_hours"),
                "evidence_age_hours": freshness_gate.get("age_hours"),
            }
        )
    elif diagnostic_class == "live-service-deploy-failed-after-development":
        stage.update(
            {
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "state": "resolved-no-action",
                "next_action": _ACTIVATION_FEEDBACK_DEPLOY_FAILURE_NEXT_ACTION,
                "review_status": "resolved-to-shell-guard-owned-deploy-health",
                "issue_worthy_status": "suppressed",
                "keep_blocked_reason": _ACTIVATION_FEEDBACK_DEPLOY_FAILURE_KEEP_BLOCKED_REASON,
                "next_state": "resolved-no-action",
                "next_state_reason": _ACTIVATION_FEEDBACK_DEPLOY_FAILURE_KEEP_BLOCKED_REASON,
                "needed_resolution": [
                    "shell_guard_deploy_health",
                    "bounded_local_action_issue",
                    "new_sanitized_evidence",
                ],
                "durable_action_ledger_entry": True,
                "terminal_successor_state": True,
                "managed_preview_required": False,
                "policy_files_written": False,
                "activation_feedback_diagnostic_classification": {
                    "schema": "tokenclaw.activation_feedback_diagnostic_classification.v1",
                    "status": "deploy-health-terminal-shell-guard-owned",
                    "decision": "resolved-no-action",
                    "reason": _ACTIVATION_FEEDBACK_DEPLOY_FAILURE_KEEP_BLOCKED_REASON,
                    "next_action": _ACTIVATION_FEEDBACK_DEPLOY_FAILURE_NEXT_ACTION,
                    "privacy": _candidate_privacy(),
                },
            }
        )
    elif diagnostic_class == "pass-verified-tokenclaw-port":
        stage.update(
            {
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "state": "resolved-no-action",
                "next_action": _ACTIVATION_FEEDBACK_PORT_VERIFIED_NEXT_ACTION,
                "review_status": "resolved-to-dev-port-smoke-context",
                "issue_worthy_status": "suppressed",
                "keep_blocked_reason": _ACTIVATION_FEEDBACK_PORT_VERIFIED_KEEP_BLOCKED_REASON,
                "next_state": "resolved-no-action",
                "next_state_reason": _ACTIVATION_FEEDBACK_PORT_VERIFIED_KEEP_BLOCKED_REASON,
                "needed_resolution": [
                    "bounded_deploy_health_follow_up",
                    "new_sanitized_evidence",
                ],
                "durable_action_ledger_entry": True,
                "terminal_successor_state": True,
                "managed_preview_required": False,
                "policy_files_written": False,
                "activation_feedback_diagnostic_classification": {
                    "schema": "tokenclaw.activation_feedback_diagnostic_classification.v1",
                    "status": "dev-port-smoke-terminal",
                    "decision": "resolved-no-action",
                    "reason": _ACTIVATION_FEEDBACK_PORT_VERIFIED_KEEP_BLOCKED_REASON,
                    "next_action": _ACTIVATION_FEEDBACK_PORT_VERIFIED_NEXT_ACTION,
                    "privacy": _candidate_privacy(),
                },
            }
        )
    elif diagnostic_class == "activation-feedback-blocker-review":
        classification = (
            diagnostic.get("activation_feedback_diagnostic_classification")
            if isinstance(diagnostic.get("activation_feedback_diagnostic_classification"), dict)
            else {}
        )
        classification_status = str(classification.get("status") or "").strip()
        if classification_status == "new-sanitized-evidence":
            next_action = str(
                diagnostic.get("next_action_override")
                or classification.get("next_action")
                or "review-new-sanitized-activation-feedback-evidence"
            ).strip()
            stage.update(
                {
                    "lever": "activation-feedback",
                    "local_action_family": "activation-feedback",
                    "state": "ranked-evidence",
                    "next_action": sanitize_value(next_action),
                    "review_status": "new-sanitized-evidence",
                    "issue_worthy_status": "ready" if count > 1 else "review",
                    "next_state": "ranked-evidence",
                    "next_state_reason": "new-sanitized-activation-feedback-evidence",
                    "needed_resolution": ["bounded_local_action_issue"],
                    "durable_action_ledger_entry": True,
                    "policy_files_written": False,
                    "managed_preview_required": False,
                    "activation_feedback_diagnostic_classification": classification,
                    "diagnostic_evidence_status": "new-sanitized-evidence",
                }
            )
        elif classification_status == "human-review-required":
            stage.update(
                {
                    "lever": "activation-feedback",
                    "local_action_family": "activation-feedback",
                    "state": "keep-blocked",
                    "next_action": "review-activation-feedback-diagnostic-before-successor-issue",
                    "review_status": "human-review-required",
                    "issue_worthy_status": "blocked",
                    "keep_blocked_reason": "activation-feedback-blocker-review-needs-human-review",
                    "next_state": "keep-blocked",
                    "next_state_reason": "activation-feedback-blocker-review-needs-human-review",
                    "needed_resolution": ["human_review", "new_sanitized_evidence", "bounded_local_action_issue"],
                    "durable_action_ledger_entry": True,
                    "activation_feedback_diagnostic_classification": classification,
                }
            )
        else:
            stage.update(
                {
                    "lever": "activation-feedback",
                    "local_action_family": "activation-feedback",
                    "state": "keep-blocked",
                    "next_action": _ACTIVATION_FEEDBACK_BLOCKER_NEXT_ACTION,
                    "review_status": "resolved-to-keep-blocked",
                    "issue_worthy_status": "blocked",
                    "keep_blocked_reason": _ACTIVATION_FEEDBACK_BLOCKER_KEEP_BLOCKED_REASON,
                    "next_state": "keep-blocked",
                    "next_state_reason": _ACTIVATION_FEEDBACK_BLOCKER_KEEP_BLOCKED_REASON,
                    "needed_resolution": ["new_sanitized_evidence", "bounded_local_action_issue"],
                    "durable_action_ledger_entry": True,
                    "activation_feedback_diagnostic_classification": {
                        "schema": "tokenclaw.activation_feedback_diagnostic_classification.v1",
                        "status": "already-resolved-keep-blocked",
                        "decision": "keep-blocked",
                        "reason": _ACTIVATION_FEEDBACK_BLOCKER_KEEP_BLOCKED_REASON,
                        "privacy": _candidate_privacy(),
                    },
                }
            )

    if lifecycle_context:
        action_family = str(lifecycle_context.get("action_family") or "").strip()
        if action_family in {"routing", "cache", "crunch"}:
            stage["local_action_family"] = action_family
            if stage.get("lever") == "activation-feedback":
                stage["lever"] = action_family
        if lifecycle_context.get("report_schema"):
            stage["evidence_source"] = lifecycle_context.get("report_schema")
        if lifecycle_context.get("next_action"):
            stage["next_action"] = lifecycle_context.get("next_action")
        blocker = str(lifecycle_context.get("blocker_code") or "").strip()
        if blocker and reason in {"missing-lifecycle-feedback", "aggregate-only"}:
            stage["blocker_codes"] = [blocker]
        sample_bucket = str(lifecycle_context.get("sample_count_bucket") or "").strip()
        if sample_bucket and sample_bucket != "unknown":
            stage["cohort_bucket"] = f"{stage.get('lever')}:{sample_bucket}"
        lifecycle_state = _normal_diagnostic_token(lifecycle_context.get("lifecycle_state"))
        if lifecycle_state in {"missing-evidence", "blocked", "activation-ready", "replay-ready", "measured-savings", "projected-savings", "ranked-evidence"}:
            stage["state"] = lifecycle_state

    return stage


def _diagnostic_ledger_stages(diagnostics: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        stage = _diagnostic_ledger_stage(diagnostic)
        if stage is None:
            continue
        if stage.get("durable_action_ledger_entry") and stage.get("diagnostic_fingerprint"):
            key = (
                str(stage.get("lever") or ""),
                str(stage.get("local_action_family") or ""),
                str(stage.get("next_action") or ""),
                str(stage.get("diagnostic_fingerprint") or ""),
            )
        else:
            key = (
                str(stage.get("lever") or ""),
                str(stage.get("local_action_family") or ""),
                str(stage.get("next_action") or ""),
                ",".join(str(item) for item in stage.get("blocker_codes") or []),
            )
        if key in seen:
            continue
        seen.add(key)
        stages.append(stage)
    return stages


def _activation_feedback_blocker_review_suppression(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    stage = _diagnostic_ledger_stage(diagnostic)
    if not stage:
        return None
    diagnostic_class = _normal_diagnostic_token(stage.get("diagnostic_class") or diagnostic.get("diagnostic_class"))
    if diagnostic_class not in {
        "activation-feedback-blocker-review",
        "live-service-deploy-failed-after-development",
        "missing-dependency-evidence",
        "no-local-representation",
        "pass-verified-tokenclaw-port",
        _RESOLVED_ACTIVATION_FEEDBACK_PASS_DIAGNOSTIC_CLASS,
        "stale-evidence",
    }:
        return None
    keep_blocked_reason = str(stage.get("keep_blocked_reason") or "").strip()
    if not keep_blocked_reason:
        return None
    return {
        "diagnostic_class": diagnostic_class,
        "reason": sanitize_value(stage.get("diagnostic_reason") or diagnostic.get("reason") or diagnostic_class),
        "fingerprint": sanitize_value(stage.get("diagnostic_fingerprint") or _diagnostic_fingerprint(diagnostic_class)),
        "suppression_kind": "durable-keep-blocked-ledger-record",
        "keep_blocked_reason": sanitize_value(keep_blocked_reason),
        "next_action": sanitize_value(stage.get("next_action") or _ACTIVATION_FEEDBACK_BLOCKER_NEXT_ACTION),
        "needed_resolution": sanitize_value(stage.get("needed_resolution") or []),
    }


def _without_suppressed_activation_feedback_blocker_review_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    suppressed_fingerprints: set[str] = set()
    for diagnostic in diagnostics:
        suppression = _activation_feedback_blocker_review_suppression(diagnostic)
        if suppression is None:
            filtered.append(diagnostic)
            continue
        fingerprint = str(suppression.get("fingerprint") or "")
        if fingerprint not in suppressed_fingerprints:
            suppressed_fingerprints.add(fingerprint)
            suppressed.append(suppression)
    return filtered, suppressed


def _safety_stop_ledger_stage(group: dict[str, Any]) -> dict[str, Any] | None:
    reason = str(group.get("keep_blocked_reason") or group.get("blocker_code") or group.get("safety_stop_reason") or "").strip()
    safety_stop_count = _to_int(group.get("safety_stop_count"))
    count = safety_stop_count or _to_int(group.get("event_count") or group.get("sample_count") or group.get("observed_count"))
    if not reason or count <= 0:
        return None
    next_state = str(group.get("next_state") or "keep-blocked").strip().replace("_", "-")
    if next_state not in {"keep-blocked", "retry-later", "superseded", "unblock-ready", "recovery-ready"}:
        next_state = "keep-blocked"
    needed = [str(item) for item in group.get("needed_resolution") or [] if str(item or "").strip()]
    action_family = str(group.get("action_family") or "activation-feedback").strip() or "activation-feedback"
    stage = {
        "lever": "activation-feedback",
        "local_action_family": action_family,
        "state": next_state,
        "status": group.get("status") or ("blocked" if next_state == "keep-blocked" else next_state),
        "evidence_source": "tokenclaw.activation_safety_stop_burndown.v1",
        "durable_action_ledger_entry": True,
        "next_action": group.get("next_action") or "review-activation-feedback-safety-stop-and-record-keep-blocked-reason",
        "blocker_codes": [reason],
        "sample_count": count,
        "projected_saved_usd": round(_to_float(group.get("savings_estimate_usd")), 8),
        "issue_worthy_status": "blocked" if next_state == "keep-blocked" else "review",
        "source": "activation-safety-stop-burndown",
        "keep_blocked_reason": reason,
        "needed_resolution": needed,
        "next_state": next_state,
        "next_state_reason": group.get("next_state_reason") or reason,
        "safety_stop_count": safety_stop_count,
        "safety_stopped_count": safety_stop_count,
        "applied_count": _to_int(group.get("applied_count")),
        "holdout_count": _to_int(group.get("holdout_count")),
        "fallback_count": _to_int(group.get("fallback_count")),
        "rollback_count": _to_int(group.get("rollback_count")),
    }
    for source_key in (
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "requested_model",
        "required_local_executor",
    ):
        value = group.get(source_key)
        if value not in (None, "", [], {}):
            stage[source_key] = value
    target_model = group.get("target_model") or group.get("candidate_target_model")
    if target_model not in (None, "", [], {}):
        stage["candidate_target_model"] = target_model
    if group.get("safety_stop_breakdown"):
        stage["safety_stop_breakdown"] = group.get("safety_stop_breakdown")
    cohort_parts = [
        str(group.get("source_surface") or "").strip(),
        str(group.get("endpoint") or "").strip(),
        str(group.get("category") or "").strip(),
        str(group.get("workflow_phase") or "").strip(),
        str(group.get("requested_model") or "").strip(),
        str(target_model or "").strip(),
    ]
    if any(cohort_parts):
        cohort_bucket = "|".join(part or "unknown" for part in cohort_parts)
        stage["cohort_bucket"] = cohort_bucket
        stage["fingerprint_cohort_bucket"] = cohort_bucket
    if isinstance(group.get("duplicate_suppression"), dict):
        stage["duplicate_suppression"] = group.get("duplicate_suppression")
    if isinstance(group.get("unblock_criteria"), dict):
        stage["unblock_criteria"] = group.get("unblock_criteria")
    if isinstance(group.get("rollback_metadata"), dict):
        stage["rollback_metadata"] = group.get("rollback_metadata")
    for review_key in (
        "safety_stop_reason_review",
        "safer_threshold_or_executor_guard",
        "rollback_proof",
        "applied_coverage",
        "holdout_coverage",
    ):
        if isinstance(group.get(review_key), dict):
            stage[review_key] = group.get(review_key)
    if isinstance(group.get("local_file_backed_representation"), dict):
        stage["local_file_backed_representation"] = group.get("local_file_backed_representation")
    if group.get("target_local_rule_file"):
        stage["target_local_rule_file"] = group.get("target_local_rule_file")
    if group.get("target_local_policy_section"):
        stage["target_local_policy_section"] = group.get("target_local_policy_section")
    if group.get("executor_compatible") is not None:
        stage["executor_compatible"] = bool(group.get("executor_compatible"))
    for gate_key in ("promotion_allowed", "stage_allowed", "active_policy_changed", "wrote_active_policy_files"):
        if group.get(gate_key) is not None:
            stage[gate_key] = bool(group.get(gate_key))
    if group.get("missing_applied_coverage") is not None:
        stage["missing_applied_coverage"] = bool(group.get("missing_applied_coverage"))
    if group.get("missing_holdout_coverage") is not None:
        stage["missing_holdout_coverage"] = bool(group.get("missing_holdout_coverage"))
    if group.get("burndown_status"):
        stage["burndown_status"] = group.get("burndown_status")
    for freshness_key in ("evidence_freshness_status", "evidence_age_hours", "max_evidence_age_hours"):
        if group.get(freshness_key) is not None:
            stage[freshness_key] = group.get(freshness_key)
    if isinstance(group.get("evidence_freshness"), dict):
        stage["evidence_freshness"] = group.get("evidence_freshness")
    policy_ref = str(group.get("policy_ref") or "").strip()
    if policy_ref and policy_ref != "unknown":
        stage["policy_ref"] = policy_ref
    return stage


def _safety_stop_ledger_stages(safety_stop_burndown: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(safety_stop_burndown, dict):
        return []
    stages: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for group in safety_stop_burndown.get("groups") or []:
        if not isinstance(group, dict):
            continue
        stage = _safety_stop_ledger_stage(group)
        if stage is None:
            continue
        key = (
            str(stage.get("local_action_family") or ""),
            str(stage.get("keep_blocked_reason") or ""),
            str(stage.get("next_state") or ""),
            str(stage.get("fingerprint_cohort_bucket") or stage.get("cohort_bucket") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        stages.append(stage)
    return stages


def _current_safety_stop_keep_blocked_group(safety_stop_burndown: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(safety_stop_burndown, dict):
        return None
    for group in safety_stop_burndown.get("groups") or []:
        if not isinstance(group, dict):
            continue
        if str(group.get("next_state") or "").replace("_", "-") != "keep-blocked":
            continue
        reason = str(group.get("keep_blocked_reason") or "").strip()
        needed = {str(item) for item in group.get("needed_resolution") or []}
        if reason and (
            needed
            & {
                "human_review",
                "safer_threshold",
                "safer_threshold_or_executor_guard",
                "rollback_proof",
                "applied_coverage",
                "holdout_coverage",
                "lifecycle_evidence",
            }
        ):
            return group
    return None


def _without_suppressed_safety_stop_diagnostics(
    diagnostics: list[dict[str, Any]],
    safety_stop_burndown: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keep_blocked_group = _current_safety_stop_keep_blocked_group(safety_stop_burndown)
    if keep_blocked_group is None:
        return diagnostics, []
    filtered: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    suppressed_fingerprints: set[str] = set()
    for diagnostic in diagnostics:
        diagnostic_class = _normal_diagnostic_token(diagnostic.get("diagnostic_class") or diagnostic.get("reason"))
        reason = _normal_diagnostic_token(diagnostic.get("reason"))
        if diagnostic_class == "safety-stop" or reason in {"safety-stop", "safety-stopped", "safety-stop-tripped"}:
            fingerprint = _diagnostic_fingerprint(diagnostic_class or reason or "safety-stop")
            if fingerprint not in suppressed_fingerprints:
                suppressed_fingerprints.add(fingerprint)
                suppression = {
                    "diagnostic_class": diagnostic_class or reason,
                    "reason": reason or diagnostic_class,
                    "fingerprint": fingerprint,
                    "suppression_kind": "current-keep-blocked-ledger-record",
                    "keep_blocked_reason": sanitize_value(
                        keep_blocked_group.get("keep_blocked_reason")
                        or (
                            (safety_stop_burndown.get("summary") or {}).get("top_keep_blocked_reason")
                            if isinstance(safety_stop_burndown.get("summary"), dict)
                            else "safety-stop-keep-blocked"
                        )
                    ),
                    "next_action": sanitize_value(
                        keep_blocked_group.get("next_action")
                        or "review-activation-feedback-safety-stop-and-record-keep-blocked-reason"
                    ),
                    "needed_resolution": sanitize_value(keep_blocked_group.get("needed_resolution") or []),
                    "safety_stop_count": _to_int(keep_blocked_group.get("safety_stop_count")),
                    "applied_count": _to_int(keep_blocked_group.get("applied_count")),
                    "holdout_count": _to_int(keep_blocked_group.get("holdout_count")),
                }
                if isinstance(keep_blocked_group.get("unblock_criteria"), dict):
                    suppression["unblock_criteria"] = sanitize_value(keep_blocked_group.get("unblock_criteria"))
                if keep_blocked_group.get("target_local_rule_file"):
                    suppression["target_local_rule_file"] = sanitize_value(keep_blocked_group.get("target_local_rule_file"))
                if keep_blocked_group.get("target_local_policy_section"):
                    suppression["target_local_policy_section"] = sanitize_value(
                        keep_blocked_group.get("target_local_policy_section")
                    )
                suppressed.append(suppression)
            continue
        filtered.append(diagnostic)
    return filtered, suppressed


def _promotion_blocker_source_report(stats: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "promotion_blocker_next_action_status",
        "promotion_blocker_next_actions",
        "promotion_blocker_next_actions_dashboard",
        "promotion_blocker_recommendation_review",
    ):
        report = stats.get(key)
        if isinstance(report, dict):
            return report
    return None


def _promotion_blocker_privacy() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "individual_candidate_ids_included": False,
        "absolute_paths_included": False,
    }


def _promotion_blocker_action_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("next_actions") or []:
        if isinstance(row, dict):
            action = str(row.get("value") or row.get("next_action") or "").strip()
            if action:
                rows.append({"next_action": sanitize_value(action), "count": _to_int(row.get("count"), 1)})

    for group in report.get("groups") or []:
        if not isinstance(group, dict):
            continue
        action = str(group.get("top_next_action") or "").strip()
        if not action:
            continue
        rows.append(
            {
                "next_action": sanitize_value(action),
                "count": _to_int(group.get("candidate_count"), 1),
                "local_action_family": sanitize_value(group.get("local_action_family") or "unknown"),
                "projected_savings_usd": round(_to_float(group.get("projected_savings_usd")), 8),
                "top_blocker_reason": sanitize_value(group.get("top_blocker_reason")),
                "top_safety_stop_reason": sanitize_value(group.get("top_safety_stop_reason")),
            }
        )

    for candidate in report.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        action = str(candidate.get("next_action") or "").strip()
        if not action:
            continue
        rows.append(
            {
                "next_action": sanitize_value(action),
                "count": _to_int(candidate.get("blocker_count"), 1),
                "local_action_family": sanitize_value(candidate.get("local_action_family") or "unknown"),
                "projected_savings_usd": round(_to_float(candidate.get("projected_savings_usd")), 8),
                "expected_local_executor": sanitize_value(candidate.get("expected_local_executor")),
                "blocker_family": sanitize_value(candidate.get("blocker_family")),
            }
        )

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        action = str(row.get("next_action") or "")
        family = str(row.get("local_action_family") or "unknown")
        key = (action, family)
        existing = merged.setdefault(
            key,
            {
                "next_action": action,
                "local_action_family": family,
                "count": 0,
                "projected_savings_usd": 0.0,
            },
        )
        existing["count"] = _to_int(existing.get("count")) + max(1, _to_int(row.get("count"), 1))
        existing["projected_savings_usd"] = round(
            _to_float(existing.get("projected_savings_usd")) + _to_float(row.get("projected_savings_usd")),
            8,
        )
        for field in ("top_blocker_reason", "top_safety_stop_reason", "expected_local_executor", "blocker_family"):
            if row.get(field) and not existing.get(field):
                existing[field] = row[field]

    result = list(merged.values())
    result.sort(key=lambda item: (-_to_float(item.get("projected_savings_usd")), -_to_int(item.get("count")), str(item.get("next_action"))))
    return result[:10]


def _promotion_blocker_next_action_status(stats: dict[str, Any]) -> dict[str, Any] | None:
    report = _promotion_blocker_source_report(stats)
    if report is None:
        return None
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    action_rows = _promotion_blocker_action_rows(report)
    top_action = str(summary.get("top_next_action") or (action_rows[0].get("next_action") if action_rows else "") or "").strip()
    candidate_count = _to_int(
        summary.get("review_candidate_count")
        or summary.get("source_recommendation_count")
        or summary.get("candidate_count")
        or len(report.get("candidates") if isinstance(report.get("candidates"), list) else [])
    )
    if not top_action and candidate_count <= 0 and str(report.get("status") or "") in {"", "no-data", "unavailable"}:
        return None
    return {
        "schema": "tokenclaw.promotion_blocker_next_action_research_status.v1",
        "source_schema": sanitize_value(report.get("schema")),
        "source_status": sanitize_value(report.get("status") or "available"),
        "summary": {
            "review_candidate_count": candidate_count,
            "recommended_count": _to_int(summary.get("recommended_count")),
            "noop_count": _to_int(summary.get("noop_count")),
            "projected_savings_usd": round(_to_float(summary.get("projected_savings_usd")), 8),
            "top_local_action_family": sanitize_value(summary.get("top_local_action_family")),
            "top_blocker_reason": sanitize_value(summary.get("top_blocker_reason")),
            "top_safety_stop_reason": sanitize_value(summary.get("top_safety_stop_reason")),
            "top_next_action": sanitize_value(top_action),
            "top_expected_local_executor": sanitize_value(summary.get("top_expected_local_executor")),
        },
        "next_actions": action_rows,
        "privacy": _promotion_blocker_privacy(),
    }


def _post_promotion_priority_source_report(stats: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "post_promotion_priority_handoff",
        "post_promotion_priority_handoff_dashboard",
        "post_promotion_priority_review",
        "post_promotion_priority_delta_review",
    ):
        report = stats.get(key)
        if isinstance(report, dict):
            return report
    return None


def _post_promotion_priority_privacy() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "individual_candidate_ids_included": False,
        "absolute_paths_included": False,
    }


def _post_promotion_priority_action_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    for row in report.get("next_action_counts") or []:
        if not isinstance(row, dict):
            continue
        action = str(row.get("value") or row.get("next_action") or "").strip()
        if action:
            rows.append(
                {
                    "next_action": sanitize_value(action),
                    "count": _to_int(row.get("count"), 1),
                    "local_action_family": sanitize_value(summary.get("top_local_action_family") or "unknown"),
                }
            )

    for group in report.get("groups") or []:
        if not isinstance(group, dict):
            continue
        action = str(group.get("top_next_action") or group.get("next_action") or "").strip()
        if not action:
            continue
        rows.append(
            {
                "next_action": sanitize_value(action),
                "count": _to_int(group.get("candidate_count"), 1),
                "local_action_family": sanitize_value(group.get("action_family") or group.get("local_action_family") or "unknown"),
                "savings_delta_usd": round(_to_float(group.get("savings_delta_usd") or group.get("projected_savings_usd")), 8),
                "status": sanitize_value(group.get("status")),
                "top_no_op_reason": sanitize_value(group.get("top_no_op_reason")),
            }
        )

    for candidate in report.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        action = str(candidate.get("next_action") or "").strip()
        if not action:
            continue
        rows.append(
            {
                "next_action": sanitize_value(action),
                "count": 1,
                "local_action_family": sanitize_value(
                    candidate.get("action_family")
                    or candidate.get("local_action_family")
                    or candidate.get("policy_section")
                    or "unknown"
                ),
                "savings_delta_usd": round(_to_float(candidate.get("savings_delta_usd") or candidate.get("projected_savings_usd")), 8),
                "status": sanitize_value(candidate.get("status")),
                "confidence": round(_to_float(candidate.get("confidence")), 6),
                "recommendation_type": sanitize_value(candidate.get("recommendation_type")),
                "policy_section": sanitize_value(candidate.get("policy_section")),
                "no_op_reasons": [sanitize_value(reason) for reason in candidate.get("no_op_reasons") or []],
            }
        )

    top_action = str(summary.get("top_next_action") or "").strip()
    if top_action and not any(row.get("next_action") == top_action for row in rows):
        rows.append(
            {
                "next_action": sanitize_value(top_action),
                "count": _to_int(summary.get("priority_review_candidate_count") or summary.get("review_candidate_count"), 1),
                "local_action_family": sanitize_value(summary.get("top_local_action_family") or "unknown"),
            }
        )

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        action = str(row.get("next_action") or "")
        family = str(row.get("local_action_family") or "unknown")
        key = (action, family)
        existing = merged.setdefault(
            key,
            {
                "next_action": action,
                "local_action_family": family,
                "count": 0,
                "savings_delta_usd": 0.0,
            },
        )
        existing["count"] = _to_int(existing.get("count")) + max(1, _to_int(row.get("count"), 1))
        existing["savings_delta_usd"] = round(
            _to_float(existing.get("savings_delta_usd")) + _to_float(row.get("savings_delta_usd")),
            8,
        )
        for field in ("status", "confidence", "recommendation_type", "policy_section", "top_no_op_reason", "no_op_reasons"):
            if row.get(field) and not existing.get(field):
                existing[field] = row[field]

    result = list(merged.values())
    preferred_action = str(summary.get("top_next_action") or "").strip()

    def sort_key(item: dict[str, Any]) -> tuple[int, float, int, str]:
        preferred = 0 if preferred_action and item.get("next_action") == preferred_action else 1
        return (
            preferred,
            -abs(_to_float(item.get("savings_delta_usd"))),
            -_to_int(item.get("count")),
            str(item.get("next_action")),
        )

    result.sort(key=sort_key)
    return result[:10]


def _post_promotion_priority_status(stats: dict[str, Any]) -> dict[str, Any] | None:
    report = _post_promotion_priority_source_report(stats)
    if report is None:
        return None
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    action_rows = _post_promotion_priority_action_rows(report)
    top = action_rows[0] if action_rows else {}
    top_action = str(summary.get("top_next_action") or top.get("next_action") or "").strip()
    candidate_count = _to_int(
        summary.get("priority_review_candidate_count")
        or summary.get("review_candidate_count")
        or summary.get("source_delta_count")
        or len(report.get("candidates") if isinstance(report.get("candidates"), list) else [])
    )
    if not top_action and candidate_count <= 0 and str(report.get("status") or "") in {"", "no-data", "unavailable"}:
        return None
    return {
        "schema": "tokenclaw.post_promotion_priority_delta_research_status.v1",
        "source_schema": sanitize_value(report.get("schema")),
        "source_status": sanitize_value(report.get("status") or "available"),
        "summary": {
            "priority_review_candidate_count": candidate_count,
            "recommended_count": _to_int(summary.get("recommended_count")),
            "noop_count": _to_int(summary.get("noop_count")),
            "widen_count": _to_int(summary.get("widen_count")),
            "collect_holdout_evidence_count": _to_int(summary.get("collect_holdout_evidence_count")),
            "rollback_count": _to_int(summary.get("rollback_count")),
            "keep_blocked_count": _to_int(summary.get("keep_blocked_count")),
            "policy_draft_status": sanitize_value(summary.get("policy_draft_status")),
            "impact_gate_status": sanitize_value(summary.get("impact_gate_status")),
            "outcome_flush_status": sanitize_value(summary.get("outcome_flush_status")),
            "outcome_rollup_count": _to_int(summary.get("outcome_rollup_count")),
            "freshness_state": sanitize_value(summary.get("freshness_state")),
            "top_next_action": sanitize_value(top_action),
            "top_local_action_family": sanitize_value(summary.get("top_local_action_family") or top.get("local_action_family")),
        },
        "next_actions": action_rows,
        "privacy": _post_promotion_priority_privacy(),
    }


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
    for key in (
        "pass_through_routing_report",
        "openai_routing_promotion_decision",
        "request_shape_cache_replay_evidence",
        "request_shape_cache_replay_policy_decision",
        "crunch_savings_signal",
        "request_shape_rollup_candidates",
        "managed_recommendation_health",
        "managed_activation_preview_outcomes",
        "managed_preview_outcomes",
        "local_activation_outcome_summary",
        "promotion_blocker_next_action_status",
        "post_promotion_priority_delta_status",
        "promotion_outcome_feedback",
        "evidence_to_activation_loop",
        "evidence_to_activation_next_action_ledger",
    ):
        value = stats.get(key)
        if isinstance(value, dict):
            summary[key] = value
    precomputed_queue = stats.get("local_activation_next_action_queue")
    if (
        isinstance(precomputed_queue, dict)
        and precomputed_queue.get("entries")
    ):
        summary["local_activation_next_action_queue"] = precomputed_queue
    routing = stats.get("routing")
    if isinstance(routing, list):
        summary["routing_top"] = routing[:5]
        pass_through_report = _pass_through_routing_report(routing)
        if pass_through_report is not None:
            summary["pass_through_routing_report"] = pass_through_report
            if "openai_routing_promotion_decision" not in summary:
                promotion_decision = _openai_routing_promotion_decision_from_pass_through(pass_through_report)
                if promotion_decision is not None:
                    summary["openai_routing_promotion_decision"] = promotion_decision
    cache = stats.get("cache_decision_breakdown")
    if isinstance(cache, list):
        summary["cache_decision_breakdown_top"] = cache[:5]
    active_crunch = stats.get("active_crunch_rule_coverage")
    if not isinstance(active_crunch, dict) and isinstance(stats.get("summary"), dict):
        active_crunch = stats["summary"].get("active_crunch_rule_coverage")
    if isinstance(active_crunch, dict):
        summary["active_crunch_rule_coverage"] = {
            "schema": sanitize_value(active_crunch.get("schema")),
            "status": sanitize_value(active_crunch.get("status")),
            "rule_file": sanitize_value(active_crunch.get("rule_file") or "crunch_rules.yaml"),
            "target_local_policy": sanitize_value(active_crunch.get("target_local_policy") or "crunch_rules"),
            "target_local_policy_section": sanitize_value(
                active_crunch.get("target_local_policy_section") or "crunch.rules"
            ),
            "summary": sanitize_value(active_crunch.get("summary") if isinstance(active_crunch.get("summary"), dict) else {}),
            "rules": sanitize_value((active_crunch.get("rules") or [])[:10]),
            "missing_measurements": sanitize_value(active_crunch.get("missing_measurements") or []),
            "privacy": sanitize_value(active_crunch.get("privacy") if isinstance(active_crunch.get("privacy"), dict) else {}),
        }
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
    openai_cache_impact = stats.get("openai_cache_replay_impact") or stats.get("cache_replay_impact")
    if isinstance(openai_cache_impact, dict):
        candidate_rows = (
            openai_cache_impact.get("candidates")
            if isinstance(openai_cache_impact.get("candidates"), list)
            else []
        )
        summary["openai_cache_replay_impact"] = {
            "schema": openai_cache_impact.get("schema"),
            "status": openai_cache_impact.get("status"),
            "summary": openai_cache_impact.get("summary") if isinstance(openai_cache_impact.get("summary"), dict) else {},
            "cohort_breakdown": openai_cache_impact.get("cohort_breakdown")
            if isinstance(openai_cache_impact.get("cohort_breakdown"), list)
            else [],
            "remaining_blocker_breakdown": openai_cache_impact.get("remaining_blocker_breakdown")
            if isinstance(openai_cache_impact.get("remaining_blocker_breakdown"), list)
            else [],
            "candidates": candidate_rows[:5],
            "quality_gate": openai_cache_impact.get("quality_gate")
            if isinstance(openai_cache_impact.get("quality_gate"), dict)
            else {},
            "privacy": openai_cache_impact.get("privacy") if isinstance(openai_cache_impact.get("privacy"), dict) else {},
        }
    request_shape_cache_evidence = stats.get("request_shape_cache_replay_evidence")
    if isinstance(request_shape_cache_evidence, dict):
        staged_canaries = (
            request_shape_cache_evidence.get("staged_canaries")
            if isinstance(request_shape_cache_evidence.get("staged_canaries"), list)
            else []
        )
        summary["request_shape_cache_replay_evidence"] = {
            "schema": request_shape_cache_evidence.get("schema"),
            "status": request_shape_cache_evidence.get("status"),
            "reason": request_shape_cache_evidence.get("reason"),
            "next_action": request_shape_cache_evidence.get("next_action"),
            "staged_canary_count": _to_int(request_shape_cache_evidence.get("staged_canary_count")),
            "summary": request_shape_cache_evidence.get("summary")
            if isinstance(request_shape_cache_evidence.get("summary"), dict)
            else {},
            "lifecycle_counts": request_shape_cache_evidence.get("lifecycle_counts")
            if isinstance(request_shape_cache_evidence.get("lifecycle_counts"), dict)
            else {},
            "blocker_breakdown": request_shape_cache_evidence.get("blocker_breakdown")
            if isinstance(request_shape_cache_evidence.get("blocker_breakdown"), list)
            else [],
            "stale_evidence": request_shape_cache_evidence.get("stale_evidence")
            if isinstance(request_shape_cache_evidence.get("stale_evidence"), dict)
            else {},
            "staged_canaries": staged_canaries[:5],
            "privacy": request_shape_cache_evidence.get("privacy")
            if isinstance(request_shape_cache_evidence.get("privacy"), dict)
            else {},
        }
    request_shape_cache_policy_decision = stats.get("request_shape_cache_replay_policy_decision")
    if isinstance(request_shape_cache_policy_decision, dict):
        decisions = (
            request_shape_cache_policy_decision.get("decisions")
            if isinstance(request_shape_cache_policy_decision.get("decisions"), list)
            else []
        )
        source_evidence = (
            request_shape_cache_policy_decision.get("source_evidence")
            if isinstance(request_shape_cache_policy_decision.get("source_evidence"), dict)
            else {}
        )
        summary["request_shape_cache_replay_policy_decision"] = {
            "schema": request_shape_cache_policy_decision.get("schema"),
            "status": request_shape_cache_policy_decision.get("status"),
            "decision": request_shape_cache_policy_decision.get("decision"),
            "promotion_decision": request_shape_cache_policy_decision.get("promotion_decision"),
            "promotion_readiness": request_shape_cache_policy_decision.get("promotion_readiness"),
            "reason": request_shape_cache_policy_decision.get("reason"),
            "reason_codes": request_shape_cache_policy_decision.get("reason_codes")
            if isinstance(request_shape_cache_policy_decision.get("reason_codes"), list)
            else [],
            "next_action": request_shape_cache_policy_decision.get("next_action"),
            "summary": request_shape_cache_policy_decision.get("summary")
            if isinstance(request_shape_cache_policy_decision.get("summary"), dict)
            else {},
            "top_decision": request_shape_cache_policy_decision.get("top_decision")
            if isinstance(request_shape_cache_policy_decision.get("top_decision"), dict)
            else {},
            "decisions": decisions[:5],
            "source_evidence": {
                "schema": source_evidence.get("schema"),
                "status": source_evidence.get("status"),
                "summary": source_evidence.get("summary") if isinstance(source_evidence.get("summary"), dict) else {},
                "applied_miss_blocker_breakdown": source_evidence.get("applied_miss_blocker_breakdown")
                if isinstance(source_evidence.get("applied_miss_blocker_breakdown"), list)
                else [],
                "stale_evidence": source_evidence.get("stale_evidence")
                if isinstance(source_evidence.get("stale_evidence"), dict)
                else {},
                "privacy": source_evidence.get("privacy") if isinstance(source_evidence.get("privacy"), dict) else {},
            },
            "acceptance": request_shape_cache_policy_decision.get("acceptance")
            if isinstance(request_shape_cache_policy_decision.get("acceptance"), dict)
            else {},
            "privacy": request_shape_cache_policy_decision.get("privacy")
            if isinstance(request_shape_cache_policy_decision.get("privacy"), dict)
            else {},
        }
    openai_cache_readiness = stats.get("openai_cache_replay_readiness")
    if isinstance(openai_cache_readiness, dict):
        candidate_rows = (
            openai_cache_readiness.get("candidates")
            if isinstance(openai_cache_readiness.get("candidates"), list)
            else []
        )
        summary["openai_cache_replay_readiness"] = {
            "schema": openai_cache_readiness.get("schema"),
            "state": openai_cache_readiness.get("state"),
            "state_reason": openai_cache_readiness.get("state_reason"),
            "summary": openai_cache_readiness.get("summary")
            if isinstance(openai_cache_readiness.get("summary"), dict)
            else {},
            "lifecycle_diagnostics": openai_cache_readiness.get("lifecycle_diagnostics")
            if isinstance(openai_cache_readiness.get("lifecycle_diagnostics"), dict)
            else {},
            "candidates": candidate_rows[:5],
            "privacy": openai_cache_readiness.get("privacy")
            if isinstance(openai_cache_readiness.get("privacy"), dict)
            else {},
        }
    crunch_signal = None if isinstance(stats.get("crunch_savings_signal"), dict) else _crunch_savings_signal(stats)
    if crunch_signal is not None:
        summary["crunch_savings_signal"] = crunch_signal
    shape_signal = (
        None
        if isinstance(stats.get("request_shape_rollup_candidates"), dict)
        else _request_shape_rollup_signal(stats)
    )
    if shape_signal is not None:
        summary["request_shape_rollup_candidates"] = shape_signal
    managed_health = (
        None
        if isinstance(stats.get("managed_recommendation_health"), dict)
        else _managed_recommendation_health_signal(stats, local_summary=summary)
    )
    if managed_health is not None:
        summary["managed_recommendation_health"] = managed_health
    if not isinstance(summary.get("local_activation_outcome_summary"), dict):
        local_activation = _local_activation_keep_active_outcome_summary(summary)
        if local_activation is not None:
            summary["local_activation_outcome_summary"] = local_activation
    promotion_blocker = (
        None
        if isinstance(stats.get("promotion_blocker_next_action_status"), dict)
        else _promotion_blocker_next_action_status(stats)
    )
    if promotion_blocker is not None:
        summary["promotion_blocker_next_action_status"] = promotion_blocker
    post_promotion_priority = (
        None
        if isinstance(stats.get("post_promotion_priority_delta_status"), dict)
        else _post_promotion_priority_status(stats)
    )
    if post_promotion_priority is not None:
        summary["post_promotion_priority_delta_status"] = post_promotion_priority
    promotion_feedback = None if isinstance(stats.get("promotion_outcome_feedback"), dict) else stats.get("promotion_outcome_feedback")
    if not isinstance(promotion_feedback, dict):
        promotion_report = stats.get("promotion_report")
        if isinstance(promotion_report, dict):
            promotion_feedback = promotion_report.get("promotion_outcome_feedback")
    if isinstance(promotion_feedback, dict):
        entries = promotion_feedback.get("entries") if isinstance(promotion_feedback.get("entries"), list) else []
        feedback_summary = promotion_feedback.get("summary") if isinstance(promotion_feedback.get("summary"), dict) else {}
        summary["promotion_outcome_feedback"] = {
            "schema": promotion_feedback.get("schema"),
            "entry_count": _to_int(promotion_feedback.get("entry_count") or feedback_summary.get("entry_count")),
            "entries": entries[:50],
            "summary": feedback_summary,
            "privacy": promotion_feedback.get("privacy") if isinstance(promotion_feedback.get("privacy"), dict) else {},
        }
    activation_loop = None if isinstance(stats.get("evidence_to_activation_loop"), dict) else _evidence_to_activation_loop(summary)
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


def _crunch_source_traffic_drill(
    *,
    report_key: str,
    report: dict[str, Any],
    summary: dict[str, Any],
    rows_considered: int,
    candidate_count: int,
    matched_count: int,
    projected_usd: float,
    projected_tokens: int,
    projected_chars: int,
    report_missing: list[str],
    current_next_action: str,
) -> dict[str, Any] | None:
    if report_key != "request_shape_crunch_opportunity":
        return None
    if candidate_count > 0 or projected_usd > 0 or projected_tokens > 0 or projected_chars > 0:
        return None

    source_schema = sanitize_value(report.get("schema") or "tokenclaw.request_shape_crunch_opportunity_dry_run.v1")
    missing = [item for item in report_missing if item]
    if not missing:
        if rows_considered <= 0:
            missing = ["no-source-traffic-for-request-shape-rollups"]
        elif matched_count <= 0:
            missing = ["repeated-context-crunch-cohorts"]
        else:
            missing = ["positive-observed-or-projected-savings"]
    top_missing = sanitize_value(missing[0])
    missing_text = " ".join(missing).lower()
    if "legacy" in missing_text:
        next_action = "adopt-legacy-evidence"
    elif "rollup" in missing_text and rows_considered <= 0:
        next_action = "refresh-rollups"
    elif "no-source-traffic" in missing_text or "repeated-context-crunch-cohorts" in missing_text:
        next_action = "collect-source-traffic"
    else:
        next_action = sanitize_value(current_next_action or "rank-repeated-context-cohorts")

    window = report.get("window") if isinstance(report.get("window"), dict) else {}
    source_window = {
        "schema": "tokenclaw.request_shape_crunch_source_window.v1",
        "status": "missing" if rows_considered <= 0 else "present",
        "rows_considered": rows_considered,
        "matched_count": matched_count,
        "source": sanitize_value(window.get("source") or summary.get("source_window") or "request-shape-rollups"),
    }
    if window.get("start") or summary.get("window_start"):
        source_window["start"] = sanitize_value(window.get("start") or summary.get("window_start"))
    if window.get("end") or summary.get("window_end"):
        source_window["end"] = sanitize_value(window.get("end") or summary.get("window_end"))

    fingerprint = public_id(
        json.dumps(
            {
                "source_schema": source_schema,
                "top_missing_measurement": top_missing,
                "next_action": next_action,
                "rows_considered": rows_considered,
                "candidate_count": candidate_count,
                "matched_count": matched_count,
            },
            sort_keys=True,
        ),
        prefix="crunch-source-drill",
    )
    duplicate_suppression = {
        "schema": "tokenclaw.request_shape_crunch_source_traffic_drill_duplicate_suppression.v1",
        "fingerprint": fingerprint,
        "reason": top_missing,
        "metadata_only": True,
        "aggregate_only": True,
        "suppresses_duplicate_source_traffic_drill": True,
        "suppresses_policy_write_candidate": True,
    }
    return {
        "schema": "tokenclaw.request_shape_crunch_source_traffic_drill.v1",
        "status": "source-traffic-drill",
        "source_schema": source_schema,
        "top_missing_measurement": top_missing,
        "missing_measurements": missing[:10],
        "source_rollup_table": "request_shape_rollups",
        "source_window": source_window,
        "activation_snapshot": {
            "schema": "tokenclaw.request_shape_crunch_activation_snapshot.v1",
            "status": sanitize_value(summary.get("activation_state") or "missing-evidence"),
            "candidate_count": candidate_count,
            "projected_saved_tokens": projected_tokens,
            "projected_saved_usd": round(projected_usd, 6),
        },
        "next_action": next_action,
        "duplicate_suppression": duplicate_suppression,
        "privacy": _candidate_privacy(),
    }


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
            "decision_count",
            "metadata_candidate_count",
            "recommended_count",
            "planned_count",
            "ranked_candidate_count",
            "cohort_count",
            "summary_model_hint_rows",
            "matched_candidates",
        ),
    )
    rows_considered = _first_int(
        summary,
        (
            "rows_considered",
            "total_rows_considered",
            "scanned_rows",
            "scanned_call_count",
            "sampled_call_count",
            "observed_canary_metadata_row_count",
            "calls",
            "window_calls",
        ),
    )
    matched_count = _first_int(
        summary,
        (
            "matched_count",
            "matched_candidates",
            "scanned_call_count",
            "sampled_call_count",
            "observed_canary_metadata_row_count",
            "turn_start_rows",
            "completed_rows",
            "applied_count",
            "crunched_count",
        ),
    )
    applied_count = _first_int(summary, ("applied_count", "crunched_count", "summary_applied_count"))
    holdout_count = _first_int(summary, ("holdout_count", "summary_holdout_count"))
    skipped_count = _first_int(summary, ("skipped_count", "no_op_count", "ineligible_count", "suppressed_count"))
    blocked_count = _first_int(summary, ("blocked_count", "safety_stop_count", "rollback_count"))
    recommended_action_count = _first_int(summary, ("recommended_action_count", "recommended_count", "planned_count"))
    projected_usd = _first_numeric(
        summary,
        (
            "observed_saved_usd",
            "projected_saved_usd",
            "estimated_opportunity_usd",
            "estimated_savings_usd",
            "summary_model_hint_estimated_savings_usd",
            "saved_usd",
        ),
    )
    projected_tokens = _first_int(
        summary,
        (
            "observed_saved_tokens",
            "projected_saved_tokens",
            "estimated_opportunity_tokens",
            "total_saved_tokens_est",
            "tokens_saved_est",
            "saved_tokens",
        ),
    )
    projected_chars = _first_int(
        summary,
        (
            "projected_saved_chars",
            "observed_saved_chars",
            "estimated_opportunity_saved_chars",
            "total_saved_chars",
            "saved_chars",
        ),
    )
    is_policy_decision = report_key == "request_shape_crunch_policy_decision"
    is_active_coverage = report_key == "active_crunch_rule_coverage"
    is_activation_evidence = report_key == "request_shape_crunch_activation_evidence"
    if is_policy_decision:
        status = "policy-decision-emitted"
    elif is_activation_evidence:
        status = "active-rule-evidence-observed" if applied_count > 0 or projected_usd > 0 or projected_tokens > 0 else "missing-crunch-activation-evidence"
    elif is_active_coverage:
        status = "active-rule-coverage-observed" if applied_count > 0 or projected_usd > 0 or projected_tokens > 0 or projected_chars > 0 else "no-applied-coverage"
    else:
        status = "projected-savings-ranked" if projected_usd > 0 or projected_tokens > 0 or projected_chars > 0 else "no-positive-projection"
    activation_follow_up = report.get("activation_follow_up") if isinstance(report.get("activation_follow_up"), dict) else {}
    no_op_reason = sanitize_value(
        activation_follow_up.get("no_op_reason")
        or summary.get("no_op_reason")
        or summary.get("top_no_op_reason")
        or ("no-applied-coverage" if status == "no-applied-coverage" else None)
        or summary.get("top_blocker_reason")
        or summary.get("top_blocker")
        or blocker_value
        or ("no-positive-projected-savings" if status == "no-positive-projection" else None)
    )
    next_action = sanitize_value(
        activation_follow_up.get("next_action")
        or summary.get("top_next_action")
        or summary.get("next_action")
        or summary.get("graduation_decision")
        or summary.get("decision")
        or ("inspect-active-crunch-rule-coverage" if status == "no-applied-coverage" else None)
        or (
            "rank-crunch-opportunity-follow-up"
            if status == "projected-savings-ranked"
            else "inspect-crunch-coverage-and-projection"
        )
    )
    activation_state = sanitize_value(summary.get("activation_state") or activation_follow_up.get("activation_state"))
    report_missing = [
        sanitize_value(item)
        for item in (report.get("missing_measurements") or activation_follow_up.get("missing_measurements") or [])
        if sanitize_value(item)
    ][:10]
    rules = report.get("rules") if isinstance(report.get("rules"), list) else []
    top_rule = next((item for item in rules if isinstance(item, dict)), {})
    source_traffic_drill = _crunch_source_traffic_drill(
        report_key=report_key,
        report=report,
        summary=summary,
        rows_considered=rows_considered,
        candidate_count=candidate_count,
        matched_count=matched_count,
        projected_usd=projected_usd,
        projected_tokens=projected_tokens,
        projected_chars=projected_chars,
        report_missing=report_missing,
        current_next_action=next_action,
    )
    duplicate_suppression = (
        sanitize_value(
            activation_follow_up.get("duplicate_suppression")
            if isinstance(activation_follow_up.get("duplicate_suppression"), dict)
            else report.get("duplicate_suppression")
        )
        if (
            isinstance(activation_follow_up.get("duplicate_suppression"), dict)
            or isinstance(report.get("duplicate_suppression"), dict)
        )
        else {}
    )
    if source_traffic_drill is not None:
        duplicate_suppression = source_traffic_drill["duplicate_suppression"]
        no_op_reason = source_traffic_drill["top_missing_measurement"]
        next_action = source_traffic_drill["next_action"]
        report_missing = source_traffic_drill["missing_measurements"]

    result = {
        "report_key": report_key,
        "schema": sanitize_value(report.get("schema")),
        "status": status,
        "rows_considered": rows_considered,
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "blocked_count": blocked_count,
        "recommended_action_count": recommended_action_count,
        "projected_saved_usd": round(projected_usd, 6),
        "projected_saved_tokens": projected_tokens,
        "projected_saved_chars": projected_chars,
        "error_rate_delta": round(_to_float(summary.get("error_rate_delta")), 6),
        "retry_rate_delta": round(_to_float(summary.get("retry_rate_delta")), 6),
        "fallback_rate_delta": round(_to_float(summary.get("fallback_rate_delta")), 6),
        "safety_stop_count": _to_int(summary.get("safety_stop_count")),
        "fallback_count": _to_int(summary.get("fallback_count")),
        "rollback_count": _to_int(summary.get("rollback_count")),
        "safety_stop_state": sanitize_value(summary.get("safety_stop_state")),
        "post_widening_status": sanitize_value(summary.get("post_widening_status")),
        "post_widening_next_action": sanitize_value(summary.get("post_widening_next_action")),
        "post_widening_reason_codes": sanitize_value(
            [str(item) for item in summary.get("post_widening_reason_codes") or [] if str(item or "").strip()]
        ),
        "post_max_rollout_status": sanitize_value(summary.get("post_max_rollout_status")),
        "post_max_rollout_decision": sanitize_value(summary.get("post_max_rollout_decision")),
        "post_max_rollout_next_action": sanitize_value(summary.get("post_max_rollout_next_action")),
        "post_max_rollout_reason_codes": sanitize_value(
            [str(item) for item in summary.get("post_max_rollout_reason_codes") or [] if str(item or "").strip()]
        ),
        "post_max_rollout_promotion_allowed": bool(summary.get("post_max_rollout_promotion_allowed")),
        "post_max_rollout_cap_reason": sanitize_value(summary.get("post_max_rollout_cap_reason")),
        "canary_fraction": round(_to_float(summary.get("canary_fraction")), 6),
        "max_rollout_fraction": round(_to_float(summary.get("max_rollout_fraction")), 6),
        "activation_state": activation_state,
        "activation_mode": sanitize_value(activation_follow_up.get("activation_mode")),
        "follow_up_status": sanitize_value(activation_follow_up.get("status")),
        "savings_status": sanitize_value(activation_follow_up.get("savings_status") or status),
        "top_blocker": sanitize_value(blocker_value),
        "top_blocker_count": blocker_count,
        "no_op_reason": no_op_reason,
        "next_action": next_action,
        "missing_measurements": report_missing,
        "canary_already_staged": bool(activation_follow_up.get("canary_already_staged")),
        "canary_already_applied": bool(activation_follow_up.get("canary_already_applied")),
        "duplicate_suppression": duplicate_suppression,
        "decision": sanitize_value(report.get("decision") or summary.get("decision")) if is_policy_decision or is_activation_evidence else None,
        "graduation_decision": sanitize_value(report.get("graduation_decision") or summary.get("graduation_decision"))
        if is_policy_decision or is_activation_evidence
        else None,
        "decision_id": sanitize_value(report.get("decision_id") or summary.get("decision_id")),
        "active_rule_count": _to_int(summary.get("active_rule_count")),
        "widened_rule_count": _to_int(summary.get("widened_rule_count")),
        "active_rule_ref": sanitize_value(top_rule.get("rule_ref") or top_rule.get("rule_id")) if top_rule else None,
        "active_rule_source": sanitize_value(top_rule.get("policy_source") or summary.get("policy_source")),
        "active_rule_decision_id": sanitize_value(top_rule.get("decision_id")) if top_rule else None,
        "active_rule_source_evidence_schema": sanitize_value(top_rule.get("source_evidence_schema")) if top_rule else None,
        "target_local_rule_file": sanitize_value(summary.get("target_local_rule_file") or report.get("rule_file")),
        "target_local_policy_section": sanitize_value(summary.get("target_local_policy_section") or report.get("target_local_policy_section")),
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
    if source_traffic_drill is not None:
        result["source_traffic_drill"] = source_traffic_drill
    return result


def _crunch_policy_decision_has_measured_evidence(report: dict[str, Any]) -> bool:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if _to_int(summary.get("applied_count")) > 0 or _to_int(summary.get("holdout_count")) > 0:
        return True
    if bool(summary.get("rollback_required")):
        return True
    if str(summary.get("safety_stop_state") or "") == "observed":
        return True
    top = report.get("top_decision") if isinstance(report.get("top_decision"), dict) else {}
    metrics = top.get("metrics") if isinstance(top.get("metrics"), dict) else {}
    return any(
        _to_int(metrics.get(key)) > 0
        for key in (
            "applied_count",
            "holdout_count",
            "fallback_count",
            "safety_stop_count",
            "rollback_count",
        )
    )


def _aggregate_crunch_report_rollup(
    stats: dict[str, Any],
    *,
    calls: int,
    today_savings: float,
    total_savings: float,
    tokens_saved: int,
    chars_saved: int,
    crunched_count: int,
    missing_measurements: list[str] | None = None,
) -> dict[str, Any] | None:
    missing_measurements = missing_measurements or []
    if (
        calls <= 0
        and crunched_count <= 0
        and tokens_saved <= 0
        and chars_saved <= 0
        and today_savings <= 0
        and total_savings <= 0
        and not missing_measurements
    ):
        return None
    projected_tokens = max(0, tokens_saved)
    projected_chars = max(0, chars_saved)
    projected_usd = max(0.0, today_savings or total_savings)
    skipped_count = max(0, calls - crunched_count)
    if missing_measurements:
        no_op_reason = "missing-crunch-aggregate-measurement"
        next_action = "emit-crunch-aggregate-measurement"
        status = "missing-measurement"
    elif crunched_count > 0 or tokens_saved > 0 or chars_saved > 0 or projected_usd > 0:
        no_op_reason = None
        next_action = "rank-observed-crunch-family-follow-up"
        status = "projected-savings-ranked"
    else:
        no_op_reason = "no-observed-or-projected-crunch-savings"
        next_action = "inspect-crunch-coverage-and-projection"
        status = "no-positive-projection"
    return {
        "report_key": "aggregate_crunch_measurement",
        "schema": "tokenclaw.aggregate_crunch_measurement.v1",
        "status": status,
        "rows_considered": calls,
        "candidate_count": crunched_count,
        "matched_count": crunched_count,
        "applied_count": crunched_count,
        "skipped_count": skipped_count,
        "projected_saved_usd": round(projected_usd, 6),
        "projected_saved_tokens": projected_tokens,
        "projected_saved_chars": projected_chars,
        "observed_savings_usd": round(total_savings, 6),
        "today_observed_savings_usd": round(today_savings, 6),
        "avg_crunch_ratio": round(_to_float(stats.get("avg_crunch_ratio")), 6),
        "top_blocker": no_op_reason,
        "top_blocker_count": skipped_count if no_op_reason else 0,
        "no_op_reason": no_op_reason,
        "next_action": next_action,
        "missing_measurements": missing_measurements,
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


def _missing_aggregate_crunch_measurements(stats: dict[str, Any]) -> list[str]:
    aggregate_keys = (
        "crunched_count",
        "crunch_chars_saved",
        "crunch_tokens_saved",
        "crunch_savings_usd",
        "today_crunch_savings_usd",
        "avg_crunch_ratio",
    )
    if any(key in stats for key in aggregate_keys):
        return []
    return [
        "crunched-count",
        "crunch-token-or-char-savings",
        "crunch-savings-usd",
        "avg-crunch-ratio",
    ]


def _crunch_savings_signal(stats: dict[str, Any]) -> dict[str, Any] | None:
    calls = _to_int(stats.get("today_calls") or stats.get("calls"))
    today_savings = _to_float(stats.get("today_crunch_savings_usd"))
    total_savings = _to_float(stats.get("crunch_savings_usd"))
    tokens_saved = _to_int(stats.get("crunch_tokens_saved"))
    chars_saved = _to_int(stats.get("crunch_chars_saved"))
    crunched_count = _to_int(stats.get("crunched_count"))

    reports: list[dict[str, Any]] = []
    active_coverage = stats.get("active_crunch_rule_coverage")
    if isinstance(active_coverage, dict):
        rollup = _crunch_report_rollup("active_crunch_rule_coverage", active_coverage)
        if rollup is not None:
            reports.append(rollup)
    for key in _CRUNCH_OPPORTUNITY_REPORT_KEYS:
        report = stats.get(key)
        if not isinstance(report, dict):
            continue
        rollup = _crunch_report_rollup(key, report)
        if rollup is not None:
            reports.append(rollup)
    shape_report = _request_shape_report(stats)
    if isinstance(shape_report, dict):
        shape_activation_evidence = shape_report.get("crunch_activation_evidence")
        if isinstance(shape_activation_evidence, dict):
            rollup = _crunch_report_rollup("request_shape_crunch_activation_evidence", shape_activation_evidence)
            if rollup is not None and shape_activation_evidence.get("status") == "active-rule-evidence-observed":
                reports.append(rollup)
        shape_policy_decision = shape_report.get("crunch_policy_decision")
        if isinstance(shape_policy_decision, dict) and _crunch_policy_decision_has_measured_evidence(shape_policy_decision):
            rollup = _crunch_report_rollup("request_shape_crunch_policy_decision", shape_policy_decision)
            if rollup is not None:
                reports.append(rollup)
        shape_impact = shape_report.get("crunch_canary_impact")
        if isinstance(shape_impact, dict) and _to_int((shape_impact.get("summary") or {}).get("candidate_count")) > 0:
            rollup = _crunch_report_rollup("request_shape_crunch_canary_impact", shape_impact)
            if rollup is not None:
                reports.append(rollup)
        shape_crunch = shape_report.get("crunch_opportunity_dry_run")
        if isinstance(shape_crunch, dict):
            rollup = _crunch_report_rollup("request_shape_crunch_opportunity", shape_crunch)
            if rollup is not None:
                reports.append(rollup)
    if not reports:
        aggregate_rollup = _aggregate_crunch_report_rollup(
            stats,
            calls=calls,
            today_savings=today_savings,
            total_savings=total_savings,
            tokens_saved=tokens_saved,
            chars_saved=chars_saved,
            crunched_count=crunched_count,
            missing_measurements=_missing_aggregate_crunch_measurements(stats),
        )
        if aggregate_rollup is not None:
            reports.append(aggregate_rollup)
    reports.sort(
        key=lambda item: (
            item.get("report_key") == "active_crunch_rule_coverage"
            and (
                _to_int(item.get("applied_count")) > 0
                or _to_float(item.get("projected_saved_usd")) > 0
                or _to_int(item.get("projected_saved_tokens")) > 0
                or _to_int(item.get("projected_saved_chars")) > 0
            ),
            item.get("report_key") == "request_shape_crunch_activation_evidence",
            item.get("report_key") == "request_shape_crunch_policy_decision",
            item.get("report_key") == "request_shape_crunch_canary_impact",
            _to_float(item.get("projected_saved_usd")),
            _to_int(item.get("projected_saved_tokens")),
            _to_int(item.get("projected_saved_chars")),
            _to_int(item.get("candidate_count")),
        ),
        reverse=True,
    )
    top_report = reports[0] if reports else None
    observed_source = "aggregate-crunch-measurement"
    observed_reports = [
        report
        for report in reports
        if report.get("report_key") in {
            "active_crunch_rule_coverage",
            "request_shape_crunch_activation_evidence",
            "request_shape_crunch_policy_decision",
            "request_shape_crunch_canary_impact",
        }
    ]
    observed_reports.sort(
        key=lambda item: (
            _to_int(item.get("applied_count")),
            _to_float(item.get("projected_saved_usd")),
            _to_int(item.get("projected_saved_tokens")),
            _to_int(item.get("projected_saved_chars")),
        ),
        reverse=True,
    )
    observed_report = observed_reports[0] if observed_reports else {}
    if observed_report:
        if crunched_count <= 0:
            crunched_count = _to_int(observed_report.get("applied_count"))
        if tokens_saved <= 0:
            tokens_saved = _to_int(observed_report.get("projected_saved_tokens"))
        if chars_saved <= 0:
            chars_saved = _to_int(observed_report.get("projected_saved_chars"))
        if total_savings <= 0:
            total_savings = _to_float(observed_report.get("projected_saved_usd"))
        if crunched_count > 0 or tokens_saved > 0 or chars_saved > 0 or total_savings > 0:
            observed_source = str(observed_report.get("report_key") or "aggregate-crunch-measurement")
    positive_projection = top_report is not None and (
        _to_float(top_report.get("projected_saved_usd")) > 0
        or _to_int(top_report.get("projected_saved_tokens")) > 0
        or _to_int(top_report.get("projected_saved_chars")) > 0
    )
    observed_positive = today_savings > 0 or total_savings > 0 or tokens_saved > 0 or chars_saved > 0
    top_report_missing = [
        str(item)
        for item in (top_report or {}).get("missing_measurements", [])
        if str(item or "").strip()
    ]
    activation_state = str((top_report or {}).get("activation_state") or "")

    top_report_status = str((top_report or {}).get("status") or "")

    if top_report and top_report.get("report_key") == "request_shape_crunch_activation_evidence":
        status = "observed-savings-ranked"
        missing = top_report_missing
    elif top_report and top_report.get("report_key") == "request_shape_crunch_policy_decision":
        status = "policy-decision-emitted"
        missing = top_report_missing
    elif top_report and top_report.get("report_key") == "active_crunch_rule_coverage" and observed_positive:
        status = "observed-savings-ranked"
        missing = []
    elif positive_projection:
        status = "projected-savings-ranked"
        if activation_state in {"blocked", "measurement-required", "missing-measurement", "missing-evidence"}:
            missing = top_report_missing
        else:
            missing = []
    elif observed_positive:
        status = "observed-savings-ranked"
        missing = []
    elif top_report_status == "missing-measurement":
        status = "missing-crunch-measurement"
        missing = top_report_missing or ["crunch-aggregate-measurement"]
    elif reports:
        status = "non-positive-projection"
        missing = top_report_missing or ["positive-observed-or-projected-savings"]
    elif calls > 0:
        status = "missing-crunch-measurement"
        missing = ["crunch-opportunity-report", "positive-observed-or-projected-savings"]
    else:
        return None

    return {
        "schema": "tokenclaw.crunch_savings_signal.v1",
        "status": status,
        "calls": calls,
        "observed": {
            "crunched_count": crunched_count,
            "crunch_chars_saved": chars_saved,
            "crunch_tokens_saved": tokens_saved,
            "crunch_savings_usd": round(total_savings, 6),
            "today_crunch_savings_usd": round(today_savings, 6),
            "avg_crunch_ratio": round(_to_float(stats.get("avg_crunch_ratio")), 6),
            "source": observed_source,
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
        if representation.get("reason") == "unknown-local-action-family":
            return 70
        return 10
    if any(term in reason_l for term in ("safety", "privacy", "raw", "unsupported", "no-local", "omitted")):
        return 20
    if any(term in reason_l for term in ("server-error", "invalid", "threshold", "stale", "insufficient")):
        return 30
    if any(term in reason_l for term in ("disabled", "missing", "historical-null")):
        return 40
    return 50


def _managed_local_handoff_reason(
    representation: dict[str, Any],
    *,
    next_action: str,
) -> str:
    if representation.get("exists"):
        section = str(representation.get("policy_section") or "local-policy").strip() or "local-policy"
        rule_file = str(representation.get("rule_file") or "local-rule-file").strip() or "local-rule-file"
        action = str(next_action or "review-local-policy-representation").strip() or "review-local-policy-representation"
        return sanitize_value(f"local-file-backed-policy-handoff:{section}:{rule_file}:{action}")
    reason = str(representation.get("reason") or "no-local-policy-representation").strip()
    return sanitize_value(f"local-handoff-blocked:{reason}")


def _managed_omission_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    reason = _managed_omission_reason(row)
    action_family = _managed_action_family(row)
    representation = _local_file_backed_representation(action_family)
    follow_up_owner = "local-policy" if representation.get("exists") else "blocked-boundary-review"
    next_action = "review-local-policy-representation" if representation.get("exists") else "define-or-keep-omitted-local-action"
    return {
        "source": source,
        "omitted_reason": reason,
        "count": _breakdown_count(row),
        "local_action_family": action_family or "unknown",
        "local_file_backed_representation": representation,
        "follow_up_owner": follow_up_owner,
        "next_action": next_action,
        "local_handoff_reason": _managed_local_handoff_reason(representation, next_action=next_action),
        "managed_dependency": "optional",
        "_priority": _managed_omission_priority(reason, representation),
    }


def _managed_local_handoff_stage_rows(stats_summary: dict[str, Any]) -> list[dict[str, Any]]:
    stages = [
        stage
        for stage in (
            _routing_loop_stage(stats_summary),
            _cache_loop_stage(stats_summary),
            _request_shape_loop_stage(stats_summary),
            _crunch_loop_stage(stats_summary),
        )
        if stage is not None
    ]
    if not stages:
        return []

    def score(stage: dict[str, Any]) -> tuple[int, float, int]:
        savings = max(
            _to_float(stage.get("savings_per_1000_calls_usd")),
            _to_float(stage.get("projected_saved_usd")),
            _to_float(stage.get("projected_saved_cost_usd")),
            _to_float(stage.get("crunch_savings_usd")),
            _to_float(stage.get("today_crunch_savings_usd")),
        )
        return (
            _loop_state_rank(str(stage.get("state") or "")),
            -savings,
            -_to_int(stage.get("sample_count")),
        )

    rows: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for stage in sorted(stages, key=score):
        family = str(stage.get("local_action_family") or stage.get("lever") or "unknown")
        representation = _local_file_backed_representation(family)
        if stage.get("lever") == "request-shape-rollups" and not representation.get("exists"):
            continue
        if family in seen_families:
            continue
        seen_families.add(family)
        blockers = [str(item) for item in stage.get("blocker_codes") or [] if str(item or "").strip()]
        reason = "managed-recommendation-health-report-missing"
        if blockers:
            reason = f"managed-recommendation-health-report-missing:{sanitize_value(blockers[0])}"
        follow_up_owner = "local-policy" if representation.get("exists") else "blocked-boundary-review"
        next_action = sanitize_value(stage.get("next_action") or "emit-managed-recommendation-health-rollup")
        rows.append(
            {
                "source": "local_policy_evidence",
                "omitted_reason": reason,
                "count": _to_int(stage.get("sample_count") or stage.get("cache_hits") or 1, 1),
                "local_action_family": family,
                "local_file_backed_representation": representation,
                "follow_up_owner": follow_up_owner,
                "next_action": next_action,
                "local_handoff_reason": _managed_local_handoff_reason(representation, next_action=next_action),
                "managed_dependency": "optional",
                "local_evidence_state": sanitize_value(stage.get("state") or "unknown"),
                "local_evidence_source": sanitize_value(stage.get("evidence_source") or "stats_summary"),
                "blocker_codes": sanitize_value(blockers),
                "_priority": _managed_omission_priority(reason, representation),
            }
        )
    return rows


def _managed_local_file_backed_handoff_outcomes(
    ranked_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    generic_missing = False
    for row in ranked_rows:
        representation = (
            row.get("local_file_backed_representation")
            if isinstance(row.get("local_file_backed_representation"), dict)
            else {}
        )
        if not representation.get("exists") or row.get("follow_up_owner") != "local-policy":
            continue
        omitted_reason = sanitize_value(row.get("omitted_reason") or "unknown")
        if str(omitted_reason).startswith("managed-recommendation-health-report-missing"):
            generic_missing = True
        outcomes.append(
            {
                "schema": "tokenclaw.managed_recommendation_local_file_backed_handoff_outcome.v1",
                "rank": _to_int(row.get("rank")) or len(outcomes) + 1,
                "outcome": "local-file-backed-handoff-recorded",
                "source": sanitize_value(row.get("source") or "unknown"),
                "omitted_reason": omitted_reason,
                "count": _to_int(row.get("count"), 1),
                "local_action_family": sanitize_value(row.get("local_action_family") or "unknown"),
                "follow_up_owner": "local-policy",
                "managed_dependency": "optional",
                "policy_source": sanitize_value(representation.get("policy_source") or "local-file-backed"),
                "policy_section": sanitize_value(representation.get("policy_section") or "local-policy"),
                "rule_file": sanitize_value(representation.get("rule_file") or "local-rule-file"),
                "next_action": sanitize_value(row.get("next_action") or "review-local-policy-representation"),
                "local_handoff_reason": sanitize_value(row.get("local_handoff_reason") or ""),
                "local_evidence_state": sanitize_value(row.get("local_evidence_state") or ""),
                "local_evidence_source": sanitize_value(row.get("local_evidence_source") or ""),
                "blocker_codes": sanitize_value(row.get("blocker_codes") or []),
            }
        )
    families = sorted(
        {
            str(row.get("local_action_family"))
            for row in outcomes
            if str(row.get("local_action_family") or "").strip()
        }
    )
    duplicate_suppression = {
        "schema": "tokenclaw.managed_recommendation_handoff_duplicate_suppression.v1",
        "suppresses_generic_missing_health_issue": bool(outcomes and generic_missing),
        "reason": (
            "local-file-backed-handoff-outcome-recorded"
            if outcomes and generic_missing
            else "no-generic-managed-health-omission-covered"
        ),
        "covered_local_action_families": families,
        "local_file_backed_handoff_outcome_count": len(outcomes),
        "managed_dependency": "optional",
        "metadata_only": True,
    }
    return outcomes, duplicate_suppression


def _managed_recommendation_health_signal(
    stats: dict[str, Any],
    *,
    local_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
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

        fallback_rows = _managed_local_handoff_stage_rows(local_summary or stats)
        represented_families = {
            str(row.get("local_action_family") or "")
            for row in rows
            if isinstance(row.get("local_file_backed_representation"), dict)
            and row["local_file_backed_representation"].get("exists")
        }
        has_generic_managed_omission = any(
            row.get("local_action_family") == "unknown"
            and isinstance(row.get("local_file_backed_representation"), dict)
            and row["local_file_backed_representation"].get("reason") == "unknown-local-action-family"
            for row in rows
        )
        if fallback_rows and (not represented_families or has_generic_managed_omission):
            existing = {
                (
                    str(row.get("omitted_reason") or ""),
                    str(row.get("local_action_family") or ""),
                    str(row.get("next_action") or ""),
                )
                for row in rows
            }
            for row in fallback_rows:
                family = str(row.get("local_action_family") or "")
                if family in represented_families:
                    for existing_row in rows:
                        if str(existing_row.get("local_action_family") or "") != family:
                            continue
                        if str(existing_row.get("next_action") or "") == "review-local-policy-representation":
                            existing_row["next_action"] = row.get("next_action")
                            existing_row["local_handoff_reason"] = row.get("local_handoff_reason")
                            existing_row["local_evidence_state"] = row.get("local_evidence_state")
                            existing_row["local_evidence_source"] = row.get("local_evidence_source")
                            if row.get("blocker_codes"):
                                existing_row["blocker_codes"] = row.get("blocker_codes")
                        break
                    continue
                key = (
                    str(row.get("omitted_reason") or ""),
                    family,
                    str(row.get("next_action") or ""),
                )
                if key not in existing:
                    rows.append(row)
                    existing.add(key)
                    represented_families.add(family)

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
        fallback_rows = _managed_local_handoff_stage_rows(local_summary or stats)
        ranked = []
        for rank, row in enumerate(fallback_rows[:10], start=1):
            clean = dict(row)
            clean.pop("_priority", None)
            clean["rank"] = rank
            ranked.append(clean)
        top = ranked[0] if ranked else {
            "rank": 1,
            "source": "missing_report",
            "omitted_reason": "managed-recommendation-health-report-missing",
            "count": calls,
            "local_action_family": "unknown",
            "local_file_backed_representation": _local_file_backed_representation(None),
            "follow_up_owner": "local-policy",
            "next_action": "emit-managed-recommendation-health-rollup",
            "local_handoff_reason": "local-handoff-blocked:unknown-local-action-family",
            "managed_dependency": "optional",
        }
        status = "missing-managed-recommendation-health-report"
        missing = ["managed_recommendations_report", "omitted_local_action_reason"]
    else:
        return None

    represented = sum(1 for row in ranked if row["local_file_backed_representation"].get("exists"))
    unrepresented = len(ranked) - represented
    handoff_outcomes, duplicate_suppression = _managed_local_file_backed_handoff_outcomes(ranked)
    top_reason = str(top.get("omitted_reason") or "") if isinstance(top, dict) else ""
    top_repr = top.get("local_file_backed_representation") if isinstance(top, dict) else None
    omitted_local_action_reason = top_reason or None
    top_local_file_backed_exists = bool(top_repr.get("exists")) if isinstance(top_repr, dict) else None
    return {
        "schema": "tokenclaw.managed_recommendation_handoff_health.v1",
        "status": status,
        "source_schema": report_schema,
        "calls": calls,
        "managed_dependency": "optional",
        "omitted_local_action_reason": omitted_local_action_reason,
        "top_local_file_backed_exists": top_local_file_backed_exists,
        "summary": {
            "window_calls": _to_int(summary.get("window_calls") or calls),
            "metadata_rows": _to_int(summary.get("metadata_rows")),
            "received_count": _to_int(summary.get("received_count")),
            "applied_count": _to_int(summary.get("applied_count")),
            "observed_savings_usd": round(_to_float(summary.get("observed_savings_usd")), 8),
            "omitted_count": sum(_to_int(row.get("count"), 1) for row in ranked),
            "ranked_omission_count": len(ranked),
            "local_file_backed_count": represented,
            "local_file_backed_handoff_outcome_count": len(handoff_outcomes),
            "no_local_representation_count": unrepresented,
            "local_policy_followup_count": len(
                [row for row in ranked if row.get("follow_up_owner") == "local-policy"]
            ),
            "managed_dependency": "optional",
        },
        "top_omission": top,
        "omissions": ranked,
        "top_local_file_backed_handoff_outcome": handoff_outcomes[0] if handoff_outcomes else None,
        "local_file_backed_handoff_outcomes": handoff_outcomes,
        "duplicate_suppression": duplicate_suppression,
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


def build_managed_recommendation_handoff_report(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Build the local, metadata-only managed recommendation handoff report.

    The managed optimizer is optional for this report. When managed recommendation
    health is absent, disabled, or has no omission rows, local aggregate evidence
    is ranked into file-backed routing/crunch/cache policy handoffs instead.
    """
    summary = _stats_summary(stats)
    signal = summary.get("managed_recommendation_health")
    if not isinstance(signal, dict):
        signal = {
            "schema": "tokenclaw.managed_recommendation_handoff_health.v1",
            "status": "no-local-traffic",
            "source_schema": None,
            "calls": 0,
            "managed_dependency": "optional",
            "omitted_local_action_reason": None,
            "top_local_file_backed_exists": None,
            "summary": {
                "window_calls": 0,
                "metadata_rows": 0,
                "received_count": 0,
                "applied_count": 0,
                "observed_savings_usd": 0.0,
                "omitted_count": 0,
                "ranked_omission_count": 0,
                "local_file_backed_count": 0,
                "no_local_representation_count": 0,
                "local_policy_followup_count": 0,
                "managed_dependency": "optional",
            },
            "top_omission": None,
            "omissions": [],
            "top_local_file_backed_handoff_outcome": None,
            "local_file_backed_handoff_outcomes": [],
            "duplicate_suppression": {
                "schema": "tokenclaw.managed_recommendation_handoff_duplicate_suppression.v1",
                "suppresses_generic_missing_health_issue": False,
                "reason": "no-local-traffic",
                "covered_local_action_families": [],
                "local_file_backed_handoff_outcome_count": 0,
                "managed_dependency": "optional",
                "metadata_only": True,
            },
            "missing_measurements": ["local_metadata_calls"],
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
    result = dict(signal)
    result["read_only"] = True
    result["provider_calls_made"] = False
    result["managed_server_calls_made"] = False
    result["local_policy_handoff"] = {
        "source": "local-file-backed-policy",
        "managed_dependency": "optional",
        "supported_local_action_families": ["routing", "crunch", "cache"],
    }
    return sanitize_value(result)


def _request_shape_report(stats: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("request_shape_rollups", "request_shape_rollup_report", "request_shape_rollup_candidates_report"):
        report = stats.get(key)
        if isinstance(report, dict):
            return report
    return None


def _local_activation_outcome_privacy() -> dict[str, Any]:
    return {
        "schema": "tokenclaw.local_activation_outcome_summary_privacy.v1",
        "feature_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "provider_bodies_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "individual_candidate_ids_included": False,
        "absolute_paths_included": False,
        "policy_file_contents_included": False,
    }


def _local_activation_keep_active_outcome_summary(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = stats_summary.get("crunch_savings_signal")
    if not isinstance(signal, dict):
        return None
    reports: list[dict[str, Any]] = []
    top_report = signal.get("top_report") if isinstance(signal.get("top_report"), dict) else None
    if top_report is not None:
        reports.append(top_report)
    reports.extend(item for item in signal.get("reports") or [] if isinstance(item, dict))

    report: dict[str, Any] | None = None
    for item in reports:
        if item.get("report_key") != "request_shape_crunch_activation_evidence":
            continue
        if str(item.get("post_widening_status") or "") != "post-widening-active-at-max-rollout":
            continue
        if str(item.get("post_widening_next_action") or "") != "keep-active" and str(item.get("next_action") or "") != "keep-active":
            continue
        if _to_int(item.get("active_rule_count")) <= 0:
            continue
        report = item
        break
    if report is None:
        return None

    target_rule_file = sanitize_value(report.get("target_local_rule_file") or "crunch_rules.yaml")
    target_policy_section = sanitize_value(report.get("target_local_policy_section") or "crunch.rules")
    applied_count = _to_int(report.get("applied_count"))
    holdout_count = _to_int(report.get("holdout_count"))
    skipped_count = _to_int(report.get("skipped_count"))
    safety_stop_count = _to_int(report.get("safety_stop_count"))
    fallback_count = _to_int(report.get("fallback_count"))
    rollback_count = _to_int(report.get("rollback_count"))
    observed_savings = round(_to_float(report.get("projected_saved_usd")), 8)
    observed_tokens = _to_int(report.get("projected_saved_tokens"))
    duplicate_suppression = (
        sanitize_value(report.get("duplicate_suppression"))
        if isinstance(report.get("duplicate_suppression"), dict)
        else {
            "schema": "tokenclaw.local_activation_keep_active_duplicate_suppression.v1",
            "suppresses_new_activation_issue": True,
            "suppresses_generic_crunch_activation_issue": True,
            "reason": "repeated-context-crunch-active-at-max-rollout",
            "activation_ref": public_id(
                json.dumps(
                    {
                        "decision_id": report.get("decision_id"),
                        "active_rule_ref": report.get("active_rule_ref"),
                        "target_local_rule_file": target_rule_file,
                        "target_local_policy_section": target_policy_section,
                    },
                    sort_keys=True,
                ),
                prefix="activation",
            ),
            "matching_local_policy": "crunch_rules",
            "target_local_rule_file": target_rule_file,
            "target_local_policy_section": target_policy_section,
            "metadata_only": True,
            "aggregate_only": True,
        }
    )
    row_count = applied_count + holdout_count + skipped_count
    coverage = {
        "schema": "tokenclaw.local_activation_outcome_decision_coverage.v1",
        "source_schema": sanitize_value(report.get("schema")),
        "metadata_only": True,
        "aggregate_only": True,
        "observed_count": row_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "safety_stop_count": safety_stop_count,
        "rollback_count": rollback_count,
        "fallback_count": fallback_count,
        "error_rate_delta": round(_to_float(report.get("error_rate_delta")), 6),
        "retry_rate_delta": round(_to_float(report.get("retry_rate_delta")), 6),
        "fallback_rate_delta": round(_to_float(report.get("fallback_rate_delta")), 6),
        "canary_fraction": round(_to_float(report.get("canary_fraction")), 6),
        "max_rollout_fraction": round(_to_float(report.get("max_rollout_fraction")), 6),
    }
    stale_source = report.get("stale_evidence") if isinstance(report.get("stale_evidence"), dict) else {}
    stale_evidence = {
        "metadata_only": True,
        "aggregate_only": True,
        "stale": bool(stale_source.get("stale", False)),
        "status": sanitize_value(stale_source.get("status") or ("stale" if stale_source.get("stale") else "fresh-or-active")),
        "reason": sanitize_value(stale_source.get("reason") or "full-rollout-local-policy-active"),
    }
    keep_active_gate = _full_rollout_crunch_keep_active_gate(
        applied_count=applied_count,
        holdout_count=holdout_count,
        skipped_count=skipped_count,
        fallback_count=fallback_count,
        retry_count=_to_int(report.get("retry_count")),
        rollback_count=rollback_count,
        safety_stop_count=safety_stop_count,
        error_rate_delta=coverage["error_rate_delta"],
        retry_rate_delta=coverage["retry_rate_delta"],
        fallback_rate_delta=coverage["fallback_rate_delta"],
        stale_evidence=stale_evidence,
        decision_age_hours=_to_float(stale_source.get("age_hours") or stale_source.get("decision_age_hours")),
        full_rollout_active=True,
        target_local_policy_section=target_policy_section,
        target_local_rule_file=target_rule_file,
    )
    post_max_status = sanitize_value(report.get("post_max_rollout_status"))
    post_max_decision = sanitize_value(report.get("post_max_rollout_decision"))
    post_max_next_action = sanitize_value(report.get("post_max_rollout_next_action"))
    post_max_reason_codes = sanitize_value(report.get("post_max_rollout_reason_codes"))
    post_max_promotion_allowed = bool(report.get("post_max_rollout_promotion_allowed"))
    outcome_value = post_max_decision if post_max_decision in {"promote-full", "keep-capped", "rollback"} else "keep-active"
    next_action_value = post_max_next_action or "keep-active"
    outcome = {
        "schema": "tokenclaw.local_activation_outcome_summary_row.v1",
        "policy_section": "crunch",
        "local_action_family": "crunch",
        "local_file_backed_representation": {
            "policy_section": "crunch",
            "rule_file": target_rule_file,
            "exists": True,
            "policy_source": "local-file-backed",
            "reason": "file-backed-local-policy",
            "path_included": False,
            "policy_file_contents_included": False,
        },
        "row_count": row_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "safety_stopped_count": safety_stop_count,
        "fallback_count": fallback_count,
        "rollback_count": rollback_count,
        "observed_savings_usd": observed_savings,
        "projected_savings_usd": observed_savings,
        "observed_saved_tokens": observed_tokens,
        "projected_saved_tokens": observed_tokens,
        "source_evidence_schema": sanitize_value(report.get("schema")),
        "source_decision_id": sanitize_value(report.get("decision_id")),
        "source_decision": sanitize_value(report.get("decision")),
        "graduation_decision": sanitize_value(report.get("graduation_decision")),
        "safety_stop_state": sanitize_value(report.get("safety_stop_state")),
        "target_local_rule_file": target_rule_file,
        "target_local_policy_section": target_policy_section,
        "active_rule_count": _to_int(report.get("active_rule_count")),
        "widened_rule_count": _to_int(report.get("widened_rule_count")),
        "active_rule_ref": sanitize_value(report.get("active_rule_ref")),
        "active_rule_source": sanitize_value(report.get("active_rule_source")),
        "active_rule_decision_id": sanitize_value(report.get("active_rule_decision_id")),
        "active_rule_source_evidence_schema": sanitize_value(report.get("active_rule_source_evidence_schema")),
        "post_widening_status": sanitize_value(report.get("post_widening_status")),
        "post_widening_next_action": "keep-active",
        "post_widening_reason_codes": sanitize_value(report.get("post_widening_reason_codes")),
        "post_max_rollout_status": post_max_status,
        "post_max_rollout_decision": post_max_decision,
        "post_max_rollout_next_action": post_max_next_action,
        "post_max_rollout_reason_codes": post_max_reason_codes,
        "post_max_rollout_promotion_allowed": post_max_promotion_allowed,
        "post_max_rollout_cap_reason": sanitize_value(report.get("post_max_rollout_cap_reason")),
        "outcome": outcome_value,
        "next_action": next_action_value,
        "error_rate_delta": round(_to_float(report.get("error_rate_delta")), 6),
        "retry_rate_delta": round(_to_float(report.get("retry_rate_delta")), 6),
        "fallback_rate_delta": round(_to_float(report.get("fallback_rate_delta")), 6),
        "coverage": coverage,
        "keep_active_regression_gate": keep_active_gate,
        "duplicate_suppression": duplicate_suppression,
        "source_report": {
            "schema": sanitize_value(report.get("schema")),
            "status": sanitize_value(report.get("status")),
            "decision": sanitize_value(report.get("decision")),
            "post_widening_status": sanitize_value(report.get("post_widening_status")),
            "post_max_rollout_status": post_max_status,
            "post_max_rollout_decision": post_max_decision,
            "metadata_only": True,
            "aggregate_only": True,
        },
    }
    privacy = _local_activation_outcome_privacy()
    return {
        "schema": "tokenclaw.local_activation_outcome_summary.v1",
        "status": "tracked",
        "read_only": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_dependency": "optional",
        "summary": {
            "rows_considered": 0,
            "local_action_family_count": 1,
            "policy_decision_report_count": 1,
            "policy_decision_families": ["crunch"],
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "error_count": 0,
            "retry_count": 0,
            "fallback_count": fallback_count,
            "observed_savings_usd": observed_savings,
            "projected_savings_usd": observed_savings,
        },
        "outcome_summaries": [outcome],
        "local_policy_handoff": {
            "source": "local-activation-outcome-summary",
            "supported_local_action_families": ["crunch"],
            "source_policy_decision_schemas": [sanitize_value(report.get("schema"))],
            "managed_dependency": "optional",
            "server_ingestion_required": False,
        },
        "privacy": privacy,
        "egress_guard": {
            "schema": "tokenclaw.managed_egress_guard.v1",
            "status": "passed",
            "blocked": False,
            "violation_count": 0,
            "raw_values_logged": False,
        },
    }


def _full_rollout_crunch_activation_measurement_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "individual_candidate_ids_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _full_rollout_crunch_keep_active_gate(
    *,
    applied_count: int,
    holdout_count: int,
    skipped_count: int,
    fallback_count: int,
    retry_count: int,
    rollback_count: int,
    safety_stop_count: int,
    error_rate_delta: float,
    retry_rate_delta: float,
    fallback_rate_delta: float,
    stale_evidence: dict[str, Any],
    decision_age_hours: float,
    full_rollout_active: bool,
    target_local_policy_section: Any,
    target_local_rule_file: Any,
) -> dict[str, Any]:
    stale = bool(stale_evidence.get("stale"))
    reason_codes: list[str] = []
    if not full_rollout_active:
        reason_codes.append("full-rollout-policy-not-active")
    if applied_count <= 0:
        reason_codes.append("missing-applied-coverage")
    if rollback_count > 0:
        reason_codes.append("rollback-observed")
    if safety_stop_count > 0:
        reason_codes.append("safety-stop-observed")
    if fallback_count > 0:
        reason_codes.append("fallback-observed")
    if error_rate_delta > 0:
        reason_codes.append("error-rate-regression")
    if retry_rate_delta > 0:
        reason_codes.append("retry-rate-regression")
    if fallback_rate_delta > 0:
        reason_codes.append("fallback-rate-regression")
    if stale:
        reason_codes.append("stale-evidence")

    rollback_reasons = {
        "rollback-observed",
        "safety-stop-observed",
        "fallback-observed",
        "error-rate-regression",
        "retry-rate-regression",
        "fallback-rate-regression",
    }
    if any(reason in rollback_reasons for reason in reason_codes):
        state = "rollback-required"
        next_action = "rollback-full-rollout-repeated-context-crunch-rule"
        gate_passed = False
    elif "stale-evidence" in reason_codes:
        state = "review-stale-evidence"
        next_action = "refresh-full-rollout-repeated-context-crunch-evidence"
        gate_passed = False
    elif {"full-rollout-policy-not-active", "missing-applied-coverage"} & set(reason_codes):
        state = "keep-blocked"
        next_action = "keep-crunch-rollout-blocked-until-applied-coverage"
        gate_passed = False
    else:
        state = "keep-active"
        next_action = "keep-active"
        gate_passed = True

    return {
        "schema": FULL_ROLLOUT_CRUNCH_KEEP_ACTIVE_GATE_SCHEMA,
        "state": state,
        "gate_passed": gate_passed,
        "deterministic_next_action": next_action,
        "next_action": next_action,
        "reason_codes": reason_codes,
        "target_local_policy_section": sanitize_value(target_local_policy_section or "crunch.rules"),
        "target_local_rule_file": sanitize_value(target_local_rule_file or "crunch_rules.yaml"),
        "regression_counters": {
            "schema": "tokenclaw.full_rollout_crunch_keep_active_regression_counters.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "rollback_count": rollback_count,
            "safety_stop_count": safety_stop_count,
            "error_rate_delta": round(error_rate_delta, 6),
            "retry_rate_delta": round(retry_rate_delta, 6),
            "fallback_rate_delta": round(fallback_rate_delta, 6),
            "decision_age_hours": round(max(0.0, decision_age_hours), 3),
            "stale_evidence": {
                "metadata_only": True,
                "aggregate_only": True,
                "stale": stale,
                "status": sanitize_value(stale_evidence.get("status") or ("stale" if stale else "fresh-or-active")),
                "reason": sanitize_value(stale_evidence.get("reason") or "full-rollout-local-policy-active"),
            },
        },
        "decision_options": ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"],
        "privacy": _full_rollout_crunch_activation_measurement_privacy(),
    }


def _openai_routing_active_policy_outcome_gate(
    *,
    applied_count: int,
    holdout_count: int,
    skipped_count: int,
    unknown_count: int,
    error_count: int,
    fallback_count: int,
    retry_count: int,
    safety_stop_count: int,
    stale_evidence: dict[str, Any],
    savings_per_1000_calls_usd: float,
    target_local_policy_section: str = "routing.rules",
    target_local_rule_file: str = "routing_rules.yaml",
) -> dict[str, Any]:
    stale = bool(stale_evidence.get("stale"))
    reason_codes: list[str] = []
    if applied_count <= 0:
        reason_codes.append("missing-applied-coverage")
    if holdout_count <= 0:
        reason_codes.append("missing-holdout-coverage")
    if error_count > 0:
        reason_codes.append("error-observed")
    if fallback_count > 0:
        reason_codes.append("fallback-observed")
    if retry_count > 0:
        reason_codes.append("retry-observed")
    if safety_stop_count > 0:
        reason_codes.append("safety-stop-observed")
    if stale:
        reason_codes.append("stale-evidence")
    if unknown_count > 0:
        reason_codes.append("unknown-coverage-observed")
    if savings_per_1000_calls_usd <= 0:
        reason_codes.append("non-positive-routing-savings")

    rollback_reasons = {"error-observed", "fallback-observed", "retry-observed", "safety-stop-observed"}
    if any(reason in rollback_reasons for reason in reason_codes):
        state = "rollback-required"
        next_action = "rollback-required"
        gate_passed = False
    elif "stale-evidence" in reason_codes:
        state = "review-stale-evidence"
        next_action = "review-stale-evidence"
        gate_passed = False
    elif {"missing-applied-coverage", "missing-holdout-coverage", "non-positive-routing-savings"} & set(reason_codes):
        state = "keep-blocked"
        next_action = "keep-blocked"
        gate_passed = False
    elif "unknown-coverage-observed" in reason_codes:
        state = "keep-blocked"
        next_action = "keep-blocked"
        gate_passed = False
    else:
        state = "keep-active"
        next_action = "keep-active"
        gate_passed = True

    return {
        "schema": OPENAI_ACTIVE_LOCAL_POLICY_OUTCOME_GATE_SCHEMA,
        "state": state,
        "gate_passed": gate_passed,
        "deterministic_next_action": next_action,
        "next_action": next_action,
        "reason_codes": reason_codes,
        "target_local_policy_section": sanitize_value(target_local_policy_section),
        "target_local_rule_file": sanitize_value(target_local_rule_file),
        "savings_per_1000_calls_usd": round(float(savings_per_1000_calls_usd), 8),
        "regression_counters": {
            "schema": "tokenclaw.openai_routing_active_local_policy_regression_counters.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "unknown_count": unknown_count,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "safety_stop_count": safety_stop_count,
            "stale_evidence": {
                "metadata_only": True,
                "aggregate_only": True,
                "stale": stale,
                "age_hours": stale_evidence.get("age_hours"),
                "max_age_hours": stale_evidence.get("max_age_hours", 72.0),
                "status": sanitize_value(stale_evidence.get("status") or ("stale" if stale else "fresh")),
            },
        },
        "decision_options": ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_requests_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_written": False,
        },
    }


def _is_full_rollout_crunch_entry(entry: dict[str, Any]) -> bool:
    return bool(
        entry.get("current_status") == "full-rollout"
        or entry.get("state") == "full-rollout-active"
        or entry.get("post_max_rollout_decision") == "full-rollout-applied"
    )


def _post_rollout_crunch_decision(
    entry: dict[str, Any],
    *,
    is_current_rule: bool,
    duplicate_suppression_status: str = "none",
    duplicate_suppression_reason: str | None = None,
) -> tuple[str, str, str | None]:
    next_action = str(entry.get("next_action") or "").strip()
    current_status = str(entry.get("current_status") or "").strip()
    state = str(entry.get("state") or "").strip()
    regression_count = (
        _to_int(entry.get("fallback_count"))
        + _to_int(entry.get("retry_count"))
        + _to_int(entry.get("rollback_count"))
        + _to_int(entry.get("safety_stop_count"))
    )
    if _to_float(entry.get("error_rate_delta")) > 0:
        regression_count += 1
    if _to_float(entry.get("retry_rate_delta")) > 0:
        regression_count += 1
    if _to_float(entry.get("fallback_rate_delta")) > 0:
        regression_count += 1

    if is_current_rule:
        return "keep-current-rule-only", "keep-active", None
    if duplicate_suppression_status == "suppressed":
        return (
            "no-op",
            "keep-current-rule-only",
            duplicate_suppression_reason or "duplicate-suppressed-post-full-rollout-crunch-cohort",
        )
    if regression_count > 0:
        return "no-op", "keep-crunch-cohort-blocked-until-regression-clears", "regression-counters-present"
    if "widen" in next_action or current_status in {"applied", "holdout", "measured"}:
        return "widen-staged-cohort", next_action or "widen-repeated-context-crunch-cohort", None
    if "stage-repeated-context-crunch-canary" in next_action or state in {"activation-ready", "ready"}:
        return "stage-new-cohort", next_action or "stage-repeated-context-crunch-canary", None
    if state in {"retired-no-repeat", "superseded", "keep-blocked"} or current_status in {"blocked", "superseded"}:
        return "no-op", next_action or "keep-crunch-cohort-blocked", _queue_unblock_reason(entry)
    return "no-op", next_action or "keep-current-rule-only", _queue_unblock_reason(entry)


def _full_rollout_crunch_post_rollout_cohort_ranking(
    *,
    ledger_entries: list[dict[str, Any]],
    current_fingerprint: Any,
    keep_active_gate: dict[str, Any],
    observed_savings: float,
) -> dict[str, Any] | None:
    current_fingerprint_text = str(current_fingerprint or "")
    rows: list[dict[str, Any]] = []
    for entry in ledger_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("local_action_family") or entry.get("lever") or "") != "crunch":
            continue
        if str(entry.get("lever") or "") not in {"crunch", "request-shape-rollups", "managed-recommendation"}:
            continue

        projected = round(_queue_projected_savings(entry), 8)
        realized = round(_queue_realized_savings(entry, projected), 8)
        is_current_rule = bool(
            (current_fingerprint_text and str(entry.get("fingerprint") or "") == current_fingerprint_text)
            or _is_full_rollout_crunch_entry(entry)
        )
        duplicate_suppression = entry.get("duplicate_suppression") if isinstance(entry.get("duplicate_suppression"), dict) else {}
        duplicate_suppression_status = _queue_duplicate_suppression_status(entry)
        duplicate_suppression_reason = (
            str(duplicate_suppression.get("reason") or "").strip() if duplicate_suppression else ""
        )
        decision, recommended_next_action, no_op_reason = _post_rollout_crunch_decision(
            entry,
            is_current_rule=is_current_rule,
            duplicate_suppression_status=duplicate_suppression_status,
            duplicate_suppression_reason=duplicate_suppression_reason or None,
        )
        is_successor_candidate = bool(
            decision in {"widen-staged-cohort", "stage-new-cohort"}
            and duplicate_suppression_status != "suppressed"
        )
        applied_count = _to_int(entry.get("applied_count"))
        holdout_count = _to_int(entry.get("holdout_count"))
        skipped_count = _to_int(entry.get("skipped_count"))
        observed_count = applied_count + holdout_count + skipped_count
        regression_counters = {
            "schema": "tokenclaw.full_rollout_crunch_post_rollout_regression_counters.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "fallback_count": _to_int(entry.get("fallback_count")),
            "retry_count": _to_int(entry.get("retry_count")),
            "rollback_count": _to_int(entry.get("rollback_count")),
            "safety_stop_count": _to_int(entry.get("safety_stop_count")),
            "error_rate_delta": round(_to_float(entry.get("error_rate_delta")), 6),
            "retry_rate_delta": round(_to_float(entry.get("retry_rate_delta")), 6),
            "fallback_rate_delta": round(_to_float(entry.get("fallback_rate_delta")), 6),
        }
        row = {
            "schema": FULL_ROLLOUT_CRUNCH_POST_ROLLOUT_RANKING_ENTRY_SCHEMA,
            "rank": 0,
            "ledger_rank": _to_int(entry.get("rank")),
            "fingerprint": sanitize_value(entry.get("fingerprint")),
            "lever": sanitize_value(entry.get("lever") or "crunch"),
            "local_action_family": "crunch",
            "state": sanitize_value(entry.get("state") or "unknown"),
            "current_status": sanitize_value(entry.get("current_status") or "unknown"),
            "is_current_full_rollout_rule": is_current_rule,
            "cohort_decision": decision,
            "recommended_next_action": sanitize_value(recommended_next_action),
            "no_op_reason": sanitize_value(no_op_reason) if no_op_reason else None,
            "unblock_reason": _queue_unblock_reason(entry),
            "sample_count": _to_int(entry.get("sample_count")),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "skipped_coverage_rate": round(skipped_count / observed_count, 6) if observed_count > 0 else 0.0,
            "realized_savings_usd": realized,
            "projected_savings_usd": projected,
            "target_local_policy_section": sanitize_value(entry.get("target_local_policy_section") or "crunch.rules"),
            "target_local_rule_file": sanitize_value(entry.get("target_local_rule_file") or "crunch_rules.yaml"),
            "duplicate_suppression_status": _queue_duplicate_suppression_status(entry),
            "duplicate_suppression_reason": sanitize_value(duplicate_suppression_reason) if duplicate_suppression_reason else None,
            "review_only": True if is_successor_candidate else False,
            "policy_files_written": False,
            "regression_counters": regression_counters,
            "privacy": _full_rollout_crunch_activation_measurement_privacy(),
        }
        preserved_empty_keys = {
            "rank",
            "ledger_rank",
            "sample_count",
            "applied_count",
            "holdout_count",
            "skipped_count",
            "skipped_coverage_rate",
            "realized_savings_usd",
            "projected_savings_usd",
            "review_only",
            "policy_files_written",
        }
        rows.append({key: value for key, value in row.items() if value not in (None, "", [], 0) or key in preserved_empty_keys})

    if not rows:
        return None

    rows.sort(
        key=lambda item: (
            0 if item.get("is_current_full_rollout_rule") else 1,
            0 if item.get("cohort_decision") in {"widen-staged-cohort", "stage-new-cohort"} else 1,
            1 if item.get("duplicate_suppression_status") == "suppressed" else 0,
            -_to_float(item.get("realized_savings_usd")),
            -_to_int(item.get("skipped_count")),
            _to_int(item.get("regression_counters", {}).get("fallback_count"))
            + _to_int(item.get("regression_counters", {}).get("retry_count"))
            + _to_int(item.get("regression_counters", {}).get("rollback_count"))
            + _to_int(item.get("regression_counters", {}).get("safety_stop_count")),
            -_to_float(item.get("projected_savings_usd")),
            _to_int(item.get("ledger_rank")),
        )
    )
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank

    next_candidate = next(
        (
            item
            for item in rows
            if item.get("cohort_decision") in {"widen-staged-cohort", "stage-new-cohort"}
            and item.get("duplicate_suppression_status") != "suppressed"
        ),
        None,
    )
    if next_candidate is None:
        next_candidate = {
            "cohort_decision": "no-op",
            "recommended_next_action": "keep-current-rule-only",
            "no_op_reason": "no-unsuppressed-post-full-rollout-crunch-cohort",
            "rank": 0,
            "review_only": False,
            "policy_files_written": False,
        }

    top = rows[0]
    decision_counts = Counter(str(item.get("cohort_decision") or "unknown") for item in rows)
    suppressed_count = sum(1 for item in rows if item.get("duplicate_suppression_status") == "suppressed")
    unsuppressed_candidate_count = sum(
        1
        for item in rows
        if item.get("cohort_decision") in {"widen-staged-cohort", "stage-new-cohort"}
        and item.get("duplicate_suppression_status") != "suppressed"
    )
    result = {
        "schema": FULL_ROLLOUT_CRUNCH_POST_ROLLOUT_RANKING_SCHEMA,
        "status": "ranked",
        "current_rule_decision": sanitize_value(keep_active_gate.get("state") or "unknown"),
        "current_rule_next_action": sanitize_value(keep_active_gate.get("deterministic_next_action") or keep_active_gate.get("next_action")),
        "current_rule_realized_savings_usd": round(observed_savings, 8),
        "next_cohort_recommendation": sanitize_value(
            {
                "schema": "tokenclaw.full_rollout_crunch_next_cohort_recommendation.v1",
                "metadata_only": True,
                "aggregate_only": True,
                "cohort_decision": next_candidate.get("cohort_decision"),
                "recommended_next_action": next_candidate.get("recommended_next_action"),
                "no_op_reason": next_candidate.get("no_op_reason"),
                "rank": next_candidate.get("rank"),
                "review_only": next_candidate.get("review_only"),
                "policy_files_written": next_candidate.get("policy_files_written"),
                "realized_savings_usd": next_candidate.get("realized_savings_usd"),
                "projected_savings_usd": next_candidate.get("projected_savings_usd"),
                "unblock_reason": next_candidate.get("unblock_reason"),
                "target_local_policy_section": next_candidate.get("target_local_policy_section"),
                "target_local_rule_file": next_candidate.get("target_local_rule_file"),
            }
        ),
        "summary": {
            "ranked_cohort_count": len(rows),
            "top_cohort_decision": top.get("cohort_decision"),
            "top_realized_savings_usd": top.get("realized_savings_usd"),
            "top_projected_savings_usd": top.get("projected_savings_usd"),
            "top_skipped_count": top.get("skipped_count"),
            "next_cohort_decision": next_candidate.get("cohort_decision"),
            "next_cohort_next_action": next_candidate.get("recommended_next_action"),
            "next_cohort_no_op_reason": next_candidate.get("no_op_reason"),
            "unsuppressed_successor_candidate_count": unsuppressed_candidate_count,
            "suppressed_cohort_count": suppressed_count,
            "decision_counts": [{"value": key, "count": count} for key, count in sorted(decision_counts.items())],
        },
        "entries": rows[:10],
        "privacy": _full_rollout_crunch_activation_measurement_privacy(),
    }
    return sanitize_value(result)


def _full_rollout_crunch_activation_outcome(
    *,
    entry: dict[str, Any],
    keep_active_gate: dict[str, Any],
    post_rollout_ranking: dict[str, Any] | None,
    applied_count: int,
    holdout_count: int,
    skipped_count: int,
    fallback_count: int,
    retry_count: int,
    rollback_count: int,
    safety_stop_count: int,
    error_rate_delta: float,
    retry_rate_delta: float,
    fallback_rate_delta: float,
    observed_saved_tokens: int,
    observed_savings: float,
    projected_saved_tokens: int,
    projected_savings: float,
    target_local_policy_section: Any,
    target_local_rule_file: Any,
) -> dict[str, Any]:
    next_recommendation = (
        post_rollout_ranking.get("next_cohort_recommendation")
        if isinstance(post_rollout_ranking, dict)
        and isinstance(post_rollout_ranking.get("next_cohort_recommendation"), dict)
        else {}
    )
    state = str(keep_active_gate.get("state") or "keep-blocked")
    outcome_next_action = str(
        keep_active_gate.get("deterministic_next_action")
        or keep_active_gate.get("next_action")
        or "keep-crunch-rollout-blocked-until-applied-coverage"
    )
    no_op_reason = (
        next_recommendation.get("no_op_reason")
        if isinstance(next_recommendation, dict) and next_recommendation.get("no_op_reason")
        else "current-full-rollout-crunch-rule-measured"
    )
    successor_next_action = (
        next_recommendation.get("recommended_next_action")
        if isinstance(next_recommendation, dict) and next_recommendation.get("recommended_next_action")
        else "keep-current-rule-only"
    )
    successor_decision = (
        next_recommendation.get("cohort_decision")
        if isinstance(next_recommendation, dict) and next_recommendation.get("cohort_decision")
        else "no-op"
    )
    return sanitize_value(
        {
            "schema": FULL_ROLLOUT_CRUNCH_ACTIVATION_OUTCOME_SCHEMA,
            "durable_outcome_ledger_entry": True,
            "source_schema": FULL_ROLLOUT_CRUNCH_ACTIVATION_MEASUREMENT_SCHEMA,
            "source_ledger_schema": entry.get("schema"),
            "ledger_fingerprint": entry.get("fingerprint"),
            "ledger_rank": _to_int(entry.get("rank")),
            "lever": "crunch",
            "local_action_family": "crunch",
            "state": state,
            "outcome": state,
            "outcome_options": ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"],
            "next_action": outcome_next_action,
            "current_status": entry.get("current_status") or "full-rollout",
            "evidence_schema": entry.get("evidence_schema"),
            "activation_follow_up_evidence_schema": entry.get("activation_follow_up_evidence_schema"),
            "target_local_policy_section": target_local_policy_section or "crunch.rules",
            "target_local_rule_file": target_local_rule_file or "crunch_rules.yaml",
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "rollback_count": rollback_count,
            "safety_stop_count": safety_stop_count,
            "error_rate_delta": round(error_rate_delta, 6),
            "retry_rate_delta": round(retry_rate_delta, 6),
            "fallback_rate_delta": round(fallback_rate_delta, 6),
            "observed_saved_tokens": observed_saved_tokens,
            "observed_savings_usd": round(observed_savings, 8),
            "projected_saved_tokens": projected_saved_tokens,
            "projected_savings_usd": round(projected_savings, 8),
            "keep_active_regression_gate": keep_active_gate,
            "successor_decision": successor_decision,
            "successor_next_action": successor_next_action,
            "successor_no_op_reason": no_op_reason,
            "privacy": _full_rollout_crunch_activation_measurement_privacy(),
        }
    )


def _full_rollout_crunch_activation_measurement(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    ledger = stats_summary.get("evidence_to_activation_next_action_ledger")
    if not isinstance(ledger, dict):
        return None
    entries = [entry for entry in ledger.get("entries") or [] if isinstance(entry, dict)]
    entry = next(
        (
            item
            for item in entries
            if item.get("lever") == "crunch"
            and item.get("local_action_family") == "crunch"
            and item.get("next_action") == "measure-full-rollout-repeated-context-crunch-outcomes"
            and (
                item.get("current_status") == "full-rollout"
                or item.get("state") == "full-rollout-active"
                or item.get("post_max_rollout_decision") == "full-rollout-applied"
            )
        ),
        None,
    )
    if entry is None:
        return None

    local_summary = (
        stats_summary.get("local_activation_outcome_summary")
        if isinstance(stats_summary.get("local_activation_outcome_summary"), dict)
        else {}
    )
    local_rows = [
        row
        for row in local_summary.get("outcome_summaries") or []
        if isinstance(row, dict) and row.get("local_action_family") == "crunch"
    ]
    local_row = next(
        (
            row
            for row in local_rows
            if row.get("next_action") == "measure-full-rollout-repeated-context-crunch-outcomes"
            or row.get("post_max_rollout_decision") == "full-rollout-applied"
            or row.get("full_rollout_active") is True
        ),
        local_rows[0] if local_rows else {},
    )

    coverage = local_row.get("coverage") if isinstance(local_row.get("coverage"), dict) else {}
    duplicate_suppression = (
        local_row.get("duplicate_suppression")
        if isinstance(local_row.get("duplicate_suppression"), dict)
        else entry.get("duplicate_suppression")
        if isinstance(entry.get("duplicate_suppression"), dict)
        else {}
    )
    prior_issue = entry.get("prior_issue") if isinstance(entry.get("prior_issue"), dict) else {}
    closed_prior_seen = entry.get("issue_status") == "closed-issue-seen"
    observed_savings = _to_float(
        local_row.get("observed_savings_usd")
        or entry.get("observed_savings_usd")
        or entry.get("crunch_savings_usd")
        or entry.get("projected_saved_usd")
    )
    projected_savings = _to_float(
        local_row.get("projected_savings_usd")
        or entry.get("projected_savings_usd")
        or entry.get("projected_saved_usd")
    )
    observed_saved_tokens = _to_int(local_row.get("observed_saved_tokens") or entry.get("observed_saved_tokens") or entry.get("projected_saved_tokens"))
    projected_saved_tokens = _to_int(local_row.get("projected_saved_tokens") or entry.get("projected_saved_tokens"))
    stale_source = entry.get("stale_evidence") if isinstance(entry.get("stale_evidence"), dict) else {}
    stale_evidence = {
        "schema": "tokenclaw.full_rollout_crunch_activation_stale_evidence.v1",
        "metadata_only": True,
        "aggregate_only": True,
        "stale": bool(stale_source.get("stale", False)),
        "status": sanitize_value(stale_source.get("status") or ("stale" if stale_source.get("stale") else "fresh-or-active")),
        "reason": sanitize_value(stale_source.get("reason") or "full-rollout-local-policy-active"),
    }
    decision_age_hours = _to_float(
        stale_source.get("age_hours")
        or stale_source.get("decision_age_hours")
        or local_row.get("evidence_age_hours")
        or local_row.get("decision_age_hours")
    )
    applied_count = _to_int(local_row.get("applied_count") or entry.get("applied_count"))
    holdout_count = _to_int(local_row.get("holdout_count") or entry.get("holdout_count"))
    skipped_count = _to_int(local_row.get("skipped_count") or entry.get("skipped_count"))
    fallback_count = _to_int(local_row.get("fallback_count") or entry.get("fallback_count"))
    retry_count = _to_int(local_row.get("retry_count") or entry.get("retry_count"))
    rollback_count = _to_int(local_row.get("rollback_count") or entry.get("rollback_count"))
    safety_stop_count = _to_int(local_row.get("safety_stopped_count") or entry.get("safety_stop_count"))
    error_rate_delta = round(_to_float(local_row.get("error_rate_delta") or entry.get("error_rate_delta")), 6)
    retry_rate_delta = round(_to_float(local_row.get("retry_rate_delta") or entry.get("retry_rate_delta")), 6)
    fallback_rate_delta = round(_to_float(local_row.get("fallback_rate_delta") or entry.get("fallback_rate_delta")), 6)
    target_local_policy_section = local_row.get("target_local_policy_section") or entry.get("target_local_policy_section") or "crunch.rules"
    target_local_rule_file = local_row.get("target_local_rule_file") or entry.get("target_local_rule_file") or "crunch_rules.yaml"
    active_rule_count = _to_int(local_row.get("active_rule_count") or entry.get("active_rule_count"))
    full_rollout_active = bool(
        local_row.get("full_rollout_active") is True
        or entry.get("state") == "full-rollout-active"
        or entry.get("post_max_rollout_decision") == "full-rollout-applied"
    )
    keep_active_gate = _full_rollout_crunch_keep_active_gate(
        applied_count=applied_count,
        holdout_count=holdout_count,
        skipped_count=skipped_count,
        fallback_count=fallback_count,
        retry_count=retry_count,
        rollback_count=rollback_count,
        safety_stop_count=safety_stop_count,
        error_rate_delta=error_rate_delta,
        retry_rate_delta=retry_rate_delta,
        fallback_rate_delta=fallback_rate_delta,
        stale_evidence=stale_evidence,
        decision_age_hours=decision_age_hours,
        full_rollout_active=full_rollout_active,
        target_local_policy_section=target_local_policy_section,
        target_local_rule_file=target_local_rule_file,
    )
    post_rollout_ranking = _full_rollout_crunch_post_rollout_cohort_ranking(
        ledger_entries=entries,
        current_fingerprint=entry.get("fingerprint"),
        keep_active_gate=keep_active_gate,
        observed_savings=observed_savings,
    )
    durable_outcome = _full_rollout_crunch_activation_outcome(
        entry=entry,
        keep_active_gate=keep_active_gate,
        post_rollout_ranking=post_rollout_ranking,
        applied_count=applied_count,
        holdout_count=holdout_count,
        skipped_count=skipped_count,
        fallback_count=fallback_count,
        retry_count=retry_count,
        rollback_count=rollback_count,
        safety_stop_count=safety_stop_count,
        error_rate_delta=error_rate_delta,
        retry_rate_delta=retry_rate_delta,
        fallback_rate_delta=fallback_rate_delta,
        observed_saved_tokens=observed_saved_tokens,
        observed_savings=observed_savings,
        projected_saved_tokens=projected_saved_tokens,
        projected_savings=projected_savings,
        target_local_policy_section=target_local_policy_section,
        target_local_rule_file=target_local_rule_file,
    )
    measurement = {
        "schema": FULL_ROLLOUT_CRUNCH_ACTIVATION_MEASUREMENT_SCHEMA,
        "status": "progress-recorded",
        "measurement_action": "measure-full-rollout-repeated-context-crunch-outcomes",
        "ledger_fingerprint": sanitize_value(entry.get("fingerprint")),
        "ledger_rank": _to_int(entry.get("rank")),
        "lever": "crunch",
        "local_action_family": "crunch",
        "current_status": sanitize_value(entry.get("current_status") or "full-rollout"),
        "state": sanitize_value(entry.get("state") or "full-rollout-active"),
        "next_action": sanitize_value(entry.get("next_action")),
        "evidence_schema": sanitize_value(entry.get("evidence_schema")),
        "activation_follow_up_evidence_schema": sanitize_value(entry.get("activation_follow_up_evidence_schema")),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "fallback_count": fallback_count,
        "retry_count": retry_count,
        "rollback_count": rollback_count,
        "safety_stop_count": safety_stop_count,
        "error_rate_delta": error_rate_delta,
        "retry_rate_delta": retry_rate_delta,
        "fallback_rate_delta": fallback_rate_delta,
        "observed_saved_tokens": observed_saved_tokens,
        "observed_savings_usd": round(observed_savings, 8),
        "projected_saved_tokens": projected_saved_tokens,
        "projected_savings_usd": round(projected_savings, 8),
        "today_crunch_savings_usd": round(_to_float(entry.get("today_crunch_savings_usd")), 8),
        "target_local_policy_section": sanitize_value(target_local_policy_section),
        "target_local_rule_file": sanitize_value(target_local_rule_file),
        "active_rule_count": active_rule_count,
        "active_rule_ref": sanitize_value(local_row.get("active_rule_ref") or entry.get("active_rule_ref")),
        "active_rule_source": sanitize_value(local_row.get("active_rule_source") or entry.get("active_rule_source")),
        "active_rule_decision_id": sanitize_value(local_row.get("active_rule_decision_id") or entry.get("active_rule_decision_id")),
        "post_max_rollout_status": sanitize_value(local_row.get("post_max_rollout_status") or entry.get("post_max_rollout_status")),
        "post_max_rollout_decision": sanitize_value(local_row.get("post_max_rollout_decision") or entry.get("post_max_rollout_decision")),
        "post_max_rollout_next_action": sanitize_value(local_row.get("post_max_rollout_next_action") or entry.get("post_max_rollout_next_action")),
        "post_max_rollout_reason_codes": sanitize_value(local_row.get("post_max_rollout_reason_codes") or entry.get("post_max_rollout_reason_codes")),
        "coverage": {
            "schema": "tokenclaw.full_rollout_crunch_activation_measurement_coverage.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "applied_count": _to_int(coverage.get("applied_count") or local_row.get("applied_count") or entry.get("applied_count")),
            "holdout_count": _to_int(coverage.get("holdout_count") or local_row.get("holdout_count") or entry.get("holdout_count")),
            "skipped_count": _to_int(coverage.get("skipped_count") or local_row.get("skipped_count")),
            "fallback_count": _to_int(coverage.get("fallback_count") or local_row.get("fallback_count") or entry.get("fallback_count")),
            "rollback_count": _to_int(coverage.get("rollback_count") or local_row.get("rollback_count") or entry.get("rollback_count")),
            "safety_stop_count": _to_int(coverage.get("safety_stop_count") or local_row.get("safety_stopped_count") or entry.get("safety_stop_count")),
            "canary_fraction": round(_to_float(coverage.get("canary_fraction") or entry.get("canary_fraction")), 6),
            "max_rollout_fraction": round(_to_float(coverage.get("max_rollout_fraction") or entry.get("max_rollout_fraction")), 6),
            "full_rollout_active": True,
        },
        "stale_evidence": stale_evidence,
        "keep_active_regression_gate": keep_active_gate,
        "durable_full_rollout_outcome": durable_outcome,
        "post_full_rollout_cohort_ranking": post_rollout_ranking,
        "post_full_rollout_next_cohort_recommendation": (
            post_rollout_ranking.get("next_cohort_recommendation")
            if isinstance(post_rollout_ranking, dict)
            else None
        ),
        "duplicate_suppression": sanitize_value(duplicate_suppression),
        "closed_predecessor_suppression": {
            "schema": "tokenclaw.full_rollout_crunch_activation_predecessor_suppression.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "closed_prior_issue_seen": bool(closed_prior_seen),
            "suppresses_closed_title_recreation": True,
            "prior_issue_number": _to_int(prior_issue.get("number")) if prior_issue else 0,
            "prior_issue_title": sanitize_value(prior_issue.get("title")) if prior_issue else None,
        },
        "privacy": _full_rollout_crunch_activation_measurement_privacy(),
    }
    preserved_empty_keys = {
        "ledger_rank",
        "applied_count",
        "holdout_count",
        "skipped_count",
        "fallback_count",
        "retry_count",
        "rollback_count",
        "safety_stop_count",
        "error_rate_delta",
        "retry_rate_delta",
        "fallback_rate_delta",
        "observed_saved_tokens",
        "observed_savings_usd",
        "projected_saved_tokens",
        "projected_savings_usd",
        "today_crunch_savings_usd",
        "active_rule_count",
    }
    return sanitize_value({key: value for key, value in measurement.items() if value not in (None, "", [], 0) or key in preserved_empty_keys})


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
    if "repeated_context" in class_set and "crunch" in class_set:
        return "stage-repeated-context-crunch-canary"
    if "tool-call-cache-disabled" in blocker_set:
        return "collect-tool-call-cache-invalidation-evidence"
    if "unsupported-streaming-shape" in blocker_set and "replayability" in class_set:
        return "add-streaming-cache-replay-support"
    if "thinking-routing-guard" in blocker_set:
        return "collect-thinking-routing-lifecycle-evidence"
    if "repeated_context" in class_set and "replayability" in class_set:
        return "stage-cache-replay-canary"
    if "routing" in class_set:
        return "stage-routing-lifecycle-evidence"
    if blocker_set:
        return "classify-request-shape-blocker"
    return "keep-observability-only"


def _request_shape_local_action_family(
    classes: list[str],
    blockers: list[str],
    next_action: str,
    explicit: Any = None,
) -> str:
    value = str(explicit or "").strip()
    if value:
        return sanitize_value(value)
    action = next_action.lower().replace("_", "-")
    class_set = set(classes)
    blocker_set = set(blockers)
    if "crunch" in action or {"repeated_context", "crunch"}.issubset(class_set):
        return "crunch"
    if (
        "cache" in action
        or "replay" in action
        or "invalidation" in action
        or "tool-call-cache-disabled" in blocker_set
        or "replayability" in class_set
    ):
        return "cache"
    if "routing" in action or "thinking-routing-guard" in blocker_set or "routing" in class_set:
        return "routing"
    return "observability"


def _request_shape_readiness_state(row: dict[str, Any], next_action: str) -> str:
    explicit = str(row.get("readiness_state") or row.get("readiness") or "").strip()
    if explicit:
        return sanitize_value(explicit)
    action = next_action.lower().replace("_", "-")
    blockers = {str(item) for item in row.get("blocker_codes") or [] if str(item or "").strip()}
    if "measure" in action:
        return "measurement-required"
    if "blocked" in action or "safety" in action:
        return "blocked"
    if action.startswith(("stage-", "widen", "apply-", "add-")):
        return "activation-ready"
    if action.startswith("collect-") or "thinking-routing-guard" in blockers:
        return "needs-lifecycle-evidence"
    if blockers:
        return "ranked-evidence"
    return "observability-only"


def _request_shape_blocker_reason(row: dict[str, Any], blockers: list[str], next_action: str) -> str:
    for key in ("actionability_reason", "reason", "no_op_reason"):
        value = str(row.get(key) or "").strip()
        if value:
            return sanitize_value(value)
    if blockers:
        return sanitize_value(blockers[0])
    return sanitize_value(next_action or "request-shape-ranked")


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
    projected_savings = _to_float(row.get("projected_savings_usd") or row.get("projected_crunch_savings_usd"))
    projected_tokens = _to_int(row.get("projected_saved_tokens") or row.get("projected_crunch_tokens_saved"))
    projected_hits = _to_int(row.get("projected_hits"))
    error_count = _to_int(row.get("error_count"))
    retry_count = _to_int(row.get("retry_count"))
    next_action = sanitize_value(row.get("next_action") or _request_shape_next_action(classes, blockers))
    readiness = _request_shape_readiness_state(row, next_action)
    local_action_family = _request_shape_local_action_family(classes, blockers, next_action, row.get("local_action_family"))
    blocker_reason = _request_shape_blocker_reason(row, blockers, next_action)
    result = {
        "schema": sanitize_value(row.get("schema")),
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
        "sample_count": _to_int(row.get("sample_count") or count),
        "error_count": error_count,
        "retry_count": retry_count,
        "cost_est_usd": round(cost, 6),
        "observed_savings_usd": round(savings, 6),
        "projected_hits": projected_hits,
        "projected_saved_tokens": projected_tokens,
        "projected_savings_usd": round(projected_savings, 6),
        "candidate_work_classes": classes,
        "candidate_families": sorted(set(families)),
        "blocker_codes": sorted(set(blockers)),
        "readiness_state": readiness,
        "local_action_family": local_action_family,
        "blocker_reason": blocker_reason,
        "actionability_reason": sanitize_value(row.get("actionability_reason")),
        "next_action": next_action,
        "_score": (
            count
            + cost * 1000.0
            + savings * 2000.0
            + projected_savings * 2500.0
            + projected_tokens / 1000.0
            + projected_hits * 25.0
            + (350.0 if "repeated_context" in classes else 0.0)
            + (250.0 if "replayability" in classes else 0.0)
            + (150.0 if "routing" in classes else 0.0)
            + (125.0 if "crunch" in classes else 0.0)
            + (300.0 if readiness == "activation-ready" else 0.0)
            + (150.0 if readiness == "measurement-required" else 0.0)
            - error_count * 5.0
            - retry_count * 0.5
        ),
    }
    for key in ("fingerprint", "source_fingerprint", "source_queue_rank", "source_ledger_rank", "no_source_traffic_reason"):
        if row.get(key):
            result[key] = sanitize_value(row.get(key))
    return result


def _request_shape_successor_gap_rows(stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Return bounded metadata-only rows for request-shape successor evidence gaps."""
    sources: list[dict[str, Any]] = []
    queue = stats.get("local_activation_next_action_queue")
    if isinstance(queue, dict):
        for key in ("successor_actions", "entries"):
            values = queue.get(key)
            if isinstance(values, list):
                sources.extend(row for row in values if isinstance(row, dict))
    ledger = stats.get("evidence_to_activation_next_action_ledger")
    if isinstance(ledger, dict) and isinstance(ledger.get("entries"), list):
        sources.extend(row for row in ledger.get("entries") if isinstance(row, dict))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        family = str(source.get("local_action_family") or "").strip()
        next_action = str(source.get("next_action") or source.get("recommended_next_action") or "").strip()
        lever = str(source.get("lever") or "").strip()
        blockers = {str(item) for item in source.get("blocker_codes") or [] if str(item or "").strip()}
        source_fingerprint = str(source.get("source_fingerprint") or source.get("fingerprint") or "").strip()
        if not source_fingerprint or source_fingerprint in seen:
            continue
        if lever != "request-shape-rollups" and family != "cohort-ranking":
            continue
        if next_action != "emit-request-shape-rollups":
            continue
        if "ranked_request_shape_rollup" not in blockers:
            continue
        seen.add(source_fingerprint)
        fingerprint = public_id(
            json.dumps(
                {
                    "source_fingerprint": source_fingerprint,
                    "next_action": "emit-request-shape-rollups",
                    "reason": "no-source-traffic-for-request-shape-rollups",
                },
                sort_keys=True,
            ),
            prefix="request-shape-gap",
        )
        rows.append(
            {
                "schema": "tokenclaw.request_shape_blocker_cohort.v1",
                "fingerprint": fingerprint,
                "source_fingerprint": sanitize_value(source_fingerprint),
                "source_queue_rank": _to_int(source.get("rank")),
                "source_ledger_rank": _to_int(source.get("ledger_rank") or source.get("rank")),
                "provider_family": "unknown",
                "source_surface": "activation-successor",
                "endpoint": "request-shape-rollups",
                "requested_model_family": "unknown",
                "routed_model_family": "unknown",
                "category": "activation-successor-gap",
                "workflow_phase": "evidence-gap",
                "stream": False,
                "has_tools": False,
                "text_bucket": "unknown",
                "token_bucket": "unknown",
                "cache_status": "unknown",
                "routing_status": "unknown",
                "row_count": 0,
                "sample_count": _to_int(source.get("sample_count")),
                "projected_hits": _to_int(source.get("projected_hits")),
                "projected_saved_tokens": _to_int(source.get("projected_saved_tokens")),
                "projected_savings_usd": round(_to_float(source.get("projected_savings_usd") or source.get("projected_saved_usd")), 8),
                "observed_savings_usd": 0.0,
                "candidate_work_classes": ["request_shape_rollup_evidence_gap"],
                "candidate_families": ["activation_successor_gap"],
                "blocker_codes": ["no-source-traffic-for-request-shape-rollups"],
                "readiness_state": "blocked",
                "local_action_family": "cohort-ranking",
                "actionability_reason": "no-source-traffic-for-request-shape-rollups",
                "no_source_traffic_reason": "no-source-traffic-for-request-shape-rollups",
                "next_action": "emit-request-shape-rollups",
                "privacy": _candidate_privacy(),
            }
        )
    return rows


def _request_shape_rollup_signal(stats: dict[str, Any]) -> dict[str, Any] | None:
    calls = _to_int(stats.get("today_calls") or stats.get("calls"))
    report = _request_shape_report(stats)
    if report is None:
        gap_rows = _request_shape_successor_gap_rows(stats)
        if gap_rows:
            ranked = [
                _request_shape_candidate_row(row, source_schema="tokenclaw.request_shape_follow_up_candidates.v1", rank=index)
                for index, row in enumerate(gap_rows, start=1)
            ]
            top = ranked[0]
            return {
                "schema": "tokenclaw.request_shape_rollup_candidate_signal.v1",
                "status": "evidence-gap-ranked",
                "source_schema": "tokenclaw.request_shape_follow_up_candidates.v1",
                "summary": {
                    "calls": calls,
                    "rows_considered": 0,
                    "rollup_count": len(ranked),
                    "ranked_candidate_count": len(ranked),
                    "top_next_action": "emit-request-shape-rollups",
                    "top_local_action_family": "cohort-ranking",
                    "top_readiness_state": "blocked",
                    "no_source_traffic_reason": "no-source-traffic-for-request-shape-rollups",
                    "class_breakdown": [{"value": "request_shape_rollup_evidence_gap", "count": len(ranked)}],
                    "blocker_breakdown": [{"value": "no-source-traffic-for-request-shape-rollups", "count": len(ranked)}],
                    "local_action_family_breakdown": [{"value": "cohort-ranking", "count": len(ranked)}],
                    "readiness_breakdown": [{"value": "blocked", "count": len(ranked)}],
                    "next_action_breakdown": [{"value": "emit-request-shape-rollups", "count": len(ranked)}],
                },
                "top_candidate": top,
                "candidates": ranked,
                "local_action_cohorts": ranked,
                "missing_measurements": [],
                "privacy": _candidate_privacy(),
            }
        if calls <= 0:
            return None
        return {
            "schema": "tokenclaw.request_shape_rollup_candidate_signal.v1",
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

    follow_up_report = report.get("follow_up_candidates") if isinstance(report.get("follow_up_candidates"), dict) else None
    follow_up_rows = (
        [row for row in follow_up_report.get("blocker_cohorts") or follow_up_report.get("candidates") or [] if isinstance(row, dict)]
        if isinstance(follow_up_report, dict)
        else []
    )
    rollups = [
        row
        for row in (follow_up_rows or report.get("rollups") or report.get("candidates") or [])
        if isinstance(row, dict)
    ]
    source_schema = follow_up_report.get("schema") if isinstance(follow_up_report, dict) else report.get("schema")
    ranked = [
        _request_shape_candidate_row(row, source_schema=source_schema, rank=index)
        for index, row in enumerate(rollups, start=1)
    ]
    if not ranked:
        ranked = [
            _request_shape_candidate_row(row, source_schema=source_schema, rank=index)
            for index, row in enumerate(_request_shape_successor_gap_rows(stats), start=1)
        ]
    ranked.sort(key=lambda item: (_to_float(item.get("_score")), _to_int(item.get("row_count"))), reverse=True)
    class_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    readiness_counts: Counter[str] = Counter()
    next_action_counts: Counter[str] = Counter()
    for rank, row in enumerate(ranked[:10], start=1):
        row["rank"] = rank
        for value in row.get("candidate_work_classes") or []:
            class_counts[str(value)] += _to_int(row.get("row_count"))
        for value in row.get("blocker_codes") or []:
            blocker_counts[str(value)] += _to_int(row.get("row_count"))
        family_counts[str(row.get("local_action_family") or "unknown")] += _to_int(row.get("row_count"))
        readiness_counts[str(row.get("readiness_state") or "unknown")] += _to_int(row.get("row_count"))
        next_action_counts[str(row.get("next_action") or "unknown")] += _to_int(row.get("row_count"))
    clean_ranked = []
    for row in ranked[:10]:
        clean = dict(row)
        clean.pop("_score", None)
        if not clean.get("schema"):
            clean.pop("schema", None)
        for optional in ("actionability_reason",):
            if not clean.get(optional):
                clean.pop(optional, None)
        clean_ranked.append(clean)
    report_summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    follow_up_summary = (
        follow_up_report.get("summary")
        if isinstance(follow_up_report, dict) and isinstance(follow_up_report.get("summary"), dict)
        else {}
    )
    replay_dry_run = report.get("cache_replayability_dry_run") if isinstance(report.get("cache_replayability_dry_run"), dict) else None
    replay_summary = replay_dry_run.get("summary") if isinstance(replay_dry_run, dict) and isinstance(replay_dry_run.get("summary"), dict) else {}
    replay_cohorts = replay_dry_run.get("cohorts") if isinstance(replay_dry_run, dict) and isinstance(replay_dry_run.get("cohorts"), list) else []
    crunch_policy_decision = report.get("crunch_policy_decision") if isinstance(report.get("crunch_policy_decision"), dict) else None
    snapshot = report.get("rollup_snapshot") if isinstance(report.get("rollup_snapshot"), dict) else None
    snapshot_freshness = report.get("snapshot_freshness") if isinstance(report.get("snapshot_freshness"), dict) else {}
    if not clean_ranked and snapshot is not None:
        stale = bool(snapshot_freshness.get("stale") or report.get("snapshot_stale"))
        snapshot_summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
        signal_summary = {
            "calls": calls,
            "rows_considered": _to_int(snapshot_summary.get("rows_considered") or report_summary.get("rows_considered")),
            "rollup_count": _to_int(snapshot_summary.get("rollup_count") or report_summary.get("rollup_count")),
            "ranked_candidate_count": _to_int(snapshot_summary.get("ranked_candidate_count")),
            "top_next_action": sanitize_value(snapshot_summary.get("top_next_action") or follow_up_summary.get("top_next_action")),
            "top_local_action_family": sanitize_value(
                snapshot_summary.get("top_local_action_family") or follow_up_summary.get("top_local_action_family")
            ),
            "top_readiness_state": sanitize_value(snapshot_summary.get("top_readiness_state")),
            "class_breakdown": sanitize_value(snapshot_summary.get("class_breakdown") or []),
            "blocker_breakdown": sanitize_value(snapshot_summary.get("blocker_breakdown") or []),
            "local_action_family_breakdown": sanitize_value(snapshot_summary.get("local_action_family_breakdown") or []),
            "readiness_breakdown": sanitize_value(snapshot_summary.get("readiness_breakdown") or []),
            "next_action_breakdown": sanitize_value(snapshot_summary.get("next_action_breakdown") or []),
            "cache_replayability_replay_ready_cohort_count": _to_int(
                snapshot_summary.get("cache_replayability_replay_ready_cohort_count")
            ),
            "cache_replayability_skipped_cohort_count": _to_int(
                snapshot_summary.get("cache_replayability_skipped_cohort_count")
            ),
            "cache_replayability_projected_hits": _to_int(snapshot_summary.get("cache_replayability_projected_hits")),
            "cache_replayability_projected_savings_usd": round(
                _to_float(snapshot_summary.get("cache_replayability_projected_savings_usd")),
                8,
            ),
            "projected_crunch_tokens_saved": _to_int(snapshot_summary.get("projected_crunch_tokens_saved")),
            "projected_crunch_savings_usd": round(_to_float(snapshot_summary.get("projected_crunch_savings_usd")), 8),
            "total_projected_savings_usd": round(_to_float(snapshot_summary.get("total_projected_savings_usd")), 8),
            "snapshot_status": "snapshot-stale" if stale else "snapshot-reused",
            "snapshot_age_hours": snapshot_freshness.get("age_hours"),
            "snapshot_max_age_hours": snapshot_freshness.get("max_age_hours"),
        }
        return {
            "schema": "tokenclaw.request_shape_rollup_candidate_signal.v1",
            "status": "snapshot-stale" if stale else "snapshot-reused",
            "source_schema": sanitize_value(snapshot.get("source_schema") or report.get("schema")),
            "source_snapshot_schema": sanitize_value(snapshot.get("schema")),
            "summary": signal_summary,
            "top_candidate": None,
            "candidates": [],
            "local_action_cohorts": [],
            "missing_measurements": ["snapshot-stale"] if stale else [],
            "snapshot_freshness": sanitize_value(snapshot_freshness),
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
    evidence_gap_ranked = bool(clean_ranked) and all(
        str(row.get("no_source_traffic_reason") or "") == "no-source-traffic-for-request-shape-rollups"
        for row in clean_ranked
    )
    status = "evidence-gap-ranked" if evidence_gap_ranked else "candidates-ranked" if clean_ranked else "no-request-shape-candidates"
    result = {
        "schema": "tokenclaw.request_shape_rollup_candidate_signal.v1",
        "status": status,
        "source_schema": sanitize_value(source_schema),
        "summary": {
            "calls": calls,
            "rows_considered": _to_int(report_summary.get("rows_considered") or report_summary.get("scanned_rows")),
            "rollup_count": max(_to_int(report_summary.get("rollup_count") or len(rollups)), len(clean_ranked))
            if evidence_gap_ranked
            else _to_int(report_summary.get("rollup_count") or len(rollups)),
            "ranked_candidate_count": len(clean_ranked),
            "top_next_action": sanitize_value(follow_up_summary.get("top_next_action"))
            if follow_up_summary
            else (clean_ranked[0]["next_action"] if clean_ranked else None),
            "top_local_action_family": sanitize_value(follow_up_summary.get("top_local_action_family"))
            if follow_up_summary
            else None,
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
            "local_action_family_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "readiness_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(readiness_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "next_action_breakdown": [
                {"value": key, "count": value}
                for key, value in sorted(next_action_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "no_source_traffic_reason": sanitize_value(
                follow_up_summary.get("no_source_traffic_reason")
                if follow_up_summary
                else clean_ranked[0].get("no_source_traffic_reason")
                if evidence_gap_ranked
                else None
            ),
        },
        "top_candidate": clean_ranked[0] if clean_ranked else None,
        "candidates": clean_ranked,
        "local_action_cohorts": clean_ranked,
        "missing_measurements": []
        if clean_ranked
        else [
            str(item)
            for item in (
                follow_up_report.get("missing_measurements")
                if isinstance(follow_up_report, dict)
                else []
            )
            if str(item or "").strip()
        ]
        or ["ranked_request_shape_rollup"],
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
        skipped_openai_blockers = (
            replay_dry_run.get("skipped_openai_blockers")
            if isinstance(replay_dry_run.get("skipped_openai_blockers"), dict)
            else None
        )
        tool_replay_evidence = (
            replay_dry_run.get("tool_replay_evidence")
            if isinstance(replay_dry_run.get("tool_replay_evidence"), dict)
            else None
        )
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
        if skipped_openai_blockers is not None:
            result["cache_replayability_dry_run"]["skipped_openai_blockers"] = sanitize_value(
                {
                    "schema": skipped_openai_blockers.get("schema"),
                    "status": skipped_openai_blockers.get("status"),
                    "next_action": skipped_openai_blockers.get("next_action"),
                    "summary": skipped_openai_blockers.get("summary")
                    if isinstance(skipped_openai_blockers.get("summary"), dict)
                    else {},
                    "blocker_breakdown": skipped_openai_blockers.get("blocker_breakdown") or [],
                    "next_action_breakdown": skipped_openai_blockers.get("next_action_breakdown") or [],
                    "cohorts": (skipped_openai_blockers.get("cohorts") or [])[:5],
                    "acceptance": skipped_openai_blockers.get("acceptance")
                    if isinstance(skipped_openai_blockers.get("acceptance"), dict)
                    else {},
                    "privacy": skipped_openai_blockers.get("privacy")
                    if isinstance(skipped_openai_blockers.get("privacy"), dict)
                    else {},
                }
            )
        if tool_replay_evidence is not None:
            result["cache_replayability_dry_run"]["tool_replay_evidence"] = sanitize_value(
                {
                    "schema": tool_replay_evidence.get("schema"),
                    "status": tool_replay_evidence.get("status"),
                    "next_action": tool_replay_evidence.get("next_action"),
                    "summary": tool_replay_evidence.get("summary")
                    if isinstance(tool_replay_evidence.get("summary"), dict)
                    else {},
                    "evidence_state_breakdown": tool_replay_evidence.get("evidence_state_breakdown") or [],
                    "dependency_evidence_decision_breakdown": tool_replay_evidence.get("dependency_evidence_decision_breakdown") or [],
                    "next_action_breakdown": tool_replay_evidence.get("next_action_breakdown") or [],
                    "blocker_breakdown": tool_replay_evidence.get("blocker_breakdown") or [],
                    "cohorts": (tool_replay_evidence.get("cohorts") or [])[:5],
                    "acceptance": tool_replay_evidence.get("acceptance")
                    if isinstance(tool_replay_evidence.get("acceptance"), dict)
                    else {},
                    "privacy": tool_replay_evidence.get("privacy")
                    if isinstance(tool_replay_evidence.get("privacy"), dict)
                    else {},
                }
            )
    if crunch_policy_decision is not None:
        decision_summary = (
            crunch_policy_decision.get("summary")
            if isinstance(crunch_policy_decision.get("summary"), dict)
            else {}
        )
        result["crunch_policy_decision"] = {
            "schema": sanitize_value(crunch_policy_decision.get("schema")),
            "status": sanitize_value(crunch_policy_decision.get("status")),
            "decision": sanitize_value(crunch_policy_decision.get("decision")),
            "graduation_decision": sanitize_value(crunch_policy_decision.get("graduation_decision")),
            "decision_id": sanitize_value(crunch_policy_decision.get("decision_id")),
            "summary": sanitize_value(decision_summary),
            "privacy": sanitize_value(
                crunch_policy_decision.get("privacy")
                if isinstance(crunch_policy_decision.get("privacy"), dict)
                else {}
            ),
        }
    if follow_up_report is not None:
        result["follow_up_candidates"] = {
            "schema": sanitize_value(follow_up_report.get("schema")),
            "status": sanitize_value(follow_up_report.get("status")),
            "summary": sanitize_value(follow_up_summary),
            "top_candidate": sanitize_value(follow_up_report.get("top_candidate") if isinstance(follow_up_report.get("top_candidate"), dict) else None),
            "candidates": sanitize_value(follow_up_rows[:5]),
            "blocker_cohorts": sanitize_value((follow_up_report.get("blocker_cohorts") or [])[:5]),
            "missing_measurements": sanitize_value(follow_up_report.get("missing_measurements") or []),
            "privacy": sanitize_value(follow_up_report.get("privacy") if isinstance(follow_up_report.get("privacy"), dict) else {}),
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
        "canary-staged": 1,
        "measured-savings": 2,
        "measured-active": 2,
        "full-rollout-active": 2,
        "active-local-policy": 2,
        "projected-savings": 3,
        "ranked-evidence": 4,
        "missing-evidence": 5,
        "blocked": 6,
        "keep-blocked": 6,
        "no-op": 7,
        "retired-stale-no-traffic": 8,
    }.get(state, 9)


def _loop_missing_state(state: str) -> bool:
    return state in {"missing-evidence", "blocked", "keep-blocked", "retry-later"}


def _loop_progress_state(state: str) -> bool:
    return state in {
        "activation-ready",
        "replay-ready",
        "recovery-ready",
        "canary-staged",
        "measured-savings",
        "measured-active",
        "full-rollout-active",
        "active-local-policy",
        "projected-savings",
        "ranked-evidence",
        "superseded",
    }


def _ledger_status_from_stage(stage: dict[str, Any]) -> str:
    state = str(stage.get("state") or "").strip().lower().replace("_", "-")
    blockers = [str(item).lower().replace("_", "-") for item in stage.get("blocker_codes") or []]
    if state in _TERMINAL_ACTIVATION_SUCCESSOR_STATES:
        return state
    if state in {"unblock-ready", "recovery-ready"}:
        return "staged"
    if state in {"retired-no-repeat", "retired-stale-no-traffic"}:
        return "superseded"
    if state == "suppressed":
        return "suppressed"
    if state in {"keep-blocked", "retry-later", "superseded"}:
        return state
    if _to_int(stage.get("safety_stopped_count")) > 0 or any("safety" in blocker for blocker in blockers):
        return "safety-stopped"
    if state in {"blocked", "missing-evidence"}:
        return "blocked"
    if state in {"full-rollout", "full-rollout-active"}:
        return "full-rollout"
    if state in {"applied", "active", "measured-active", "active-local-policy"}:
        return "applied"
    if _to_int(stage.get("applied_count")) > 0 and _to_int(stage.get("holdout_count")) > 0:
        return "holdout"
    if _to_int(stage.get("applied_count")) > 0:
        return "applied"
    if state in {"activation-ready", "replay-ready"}:
        return "staged"
    if state in {"measurement-required", "canary-staged"}:
        return "staged"
    if state == "measured-savings":
        return "measured"
    if state in {"projected-savings", "ranked-evidence"}:
        return "projected"
    if state == "no-op":
        return "superseded"
    return "projected" if stage else "unknown"


def _ledger_cohort_bucket(stage: dict[str, Any]) -> str:
    requested = str(stage.get("requested_model") or "").strip()
    target = str(stage.get("candidate_target_model") or "").strip()
    if requested and target:
        return sanitize_value(f"{requested}->{target}")
    for key in ("cohort_bucket", "provider_surface_bucket", "source_surface"):
        value = str(stage.get(key) or "").strip()
        if value:
            return sanitize_value(value)
    sample_bucket = _sample_count_bucket(stage.get("sample_count"))
    return sanitize_value(f"{stage.get('lever') or 'unknown'}:{sample_bucket}")


def _ledger_expected_savings_path(stage: dict[str, Any]) -> str:
    lever = str(stage.get("lever") or "optimization")
    next_action = str(stage.get("next_action") or "inspect-local-evidence")
    paths = {
        "routing": "Move routing from local lifecycle evidence into the next canary, widening, or blocked-review step.",
        "cache": "Move cache replay evidence toward a staged local replay canary or a narrower invalidation blocker.",
        "crunch": "Move crunch opportunity evidence from projected savings into measurement, canary, or activation follow-up.",
        "request-shape-rollups": "Move request-shape rollups into the next repeated-context replay, routing, or crunch cohort issue.",
        "managed-recommendation": "Move managed omission evidence into a local file-backed policy handoff or explicit no-op reason.",
        "activation-feedback": "Convert repeated activation-feedback diagnostics into a durable local action issue without rediscovering the same blocker.",
    }
    return sanitize_value(paths.get(lever, f"Advance the local {lever} evidence path through `{next_action}`."))


def _ledger_issue_match(entry: dict[str, Any], issues: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    lever = str(entry.get("lever") or "")
    action_family = str(entry.get("local_action_family") or "")
    title_key_words = {word for word in _issue_title_key(entry.get("legacy_issue_title")).split() if word}
    for issue in issues:
        title_key = _issue_title_key(issue.get("title"))
        if not title_key:
            continue
        if title_key_words and title_key_words.issubset(set(title_key.split())):
            return _issue_ref(issue)
        if lever and lever in title_key:
            if action_family and action_family not in title_key:
                continue
            if not _is_open(issue):
                return _issue_ref(issue)
    return None


def _legacy_issue_title_for_ledger_entry(entry: dict[str, Any]) -> str:
    lever = str(entry.get("lever") or "")
    blockers = [str(item) for item in entry.get("blocker_codes") or [] if str(item or "").strip()]
    blocker = blockers[0] if blockers else str(entry.get("state") or "candidate")
    if lever == "cache":
        if blocker == "remaining-replay-ready":
            return "Advance remaining replay-ready cache cohort into local replay evidence"
        return f"Turn {blocker} cache candidate into local replay evidence"
    if lever == "routing":
        requested = str(entry.get("requested_model") or "").strip()
        target = str(entry.get("candidate_target_model") or "").strip()
        if requested and target:
            return f"Stage routing evidence for {requested} to {target}"
        return f"Collect routing lifecycle evidence for {blocker}"
    if lever == "crunch":
        return f"Rank crunch savings follow-up for {blocker}"
    if lever == "request-shape-rollups":
        next_action = str(entry.get("next_action") or "")
        action_family = str(entry.get("local_action_family") or "")
        if "widen" in next_action and action_family == "crunch":
            return "Apply measured request-shape crunch widening to local rules"
        if "measure" in next_action and action_family == "crunch":
            return "Measure request-shape repeated-context crunch canary impact"
        if "crunch" in next_action or action_family == "crunch":
            return "Stage request-shape repeated-context crunch canary"
        if "cache" in next_action or "replay" in next_action:
            return "Stage request-shape cache replay cohort"
        if "routing" in next_action:
            return "Collect request-shape routing lifecycle evidence"
        return "Rank request-shape blockers into local action cohorts"
    if lever == "managed-recommendation":
        return "Rank managed recommendation omission reasons for local policy handoff"
    return f"Convert {lever} candidate into implementation-ready savings issue"


def build_evidence_to_activation_next_action_ledger(
    stats_summary: dict[str, Any],
    *,
    existing_issues: Iterable[dict[str, Any]] = (),
    diagnostics: Iterable[dict[str, Any]] = (),
    safety_stop_burndown: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    loop = stats_summary.get("evidence_to_activation_loop") if isinstance(stats_summary.get("evidence_to_activation_loop"), dict) else {}
    stages = [stage for stage in loop.get("levers") or [] if isinstance(stage, dict)]
    cache_policy_stage = _request_shape_cache_replay_policy_decision_loop_stage(stats_summary)
    if isinstance(cache_policy_stage, dict):
        stages = [
            stage for stage in stages
            if not (
                str(stage.get("lever") or "") == "cache"
                and str(stage.get("evidence_source") or "") == str(cache_policy_stage.get("evidence_source") or "")
            )
        ]
        stages.insert(0, cache_policy_stage)
    stages.extend(_request_shape_tool_cache_dependency_stages(stats_summary))
    stages.extend(_safety_stop_ledger_stages(safety_stop_burndown))
    stages.extend(_diagnostic_ledger_stages(diagnostics))
    if not stages:
        return None

    entries: list[dict[str, Any]] = []
    for stage in stages:
        lever = sanitize_value(stage.get("lever") or "unknown")
        action_family = sanitize_value(stage.get("local_action_family") or lever)
        evidence_schema = sanitize_value(stage.get("evidence_source") or loop.get("schema"))
        cohort_bucket = _ledger_cohort_bucket(stage)
        fingerprint_evidence_schema = sanitize_value(stage.get("fingerprint_evidence_source") or evidence_schema)
        fingerprint_cohort_bucket = sanitize_value(stage.get("fingerprint_cohort_bucket") or cohort_bucket)
        raw_policy_ref = str(stage.get("policy_id") or stage.get("policy_ref") or "").strip()
        policy_ref = public_id(f"raw:{raw_policy_ref}", prefix="policy") if raw_policy_ref else None
        fingerprint_material = {
            "lever": lever,
            "local_action_family": action_family,
            "evidence_schema": fingerprint_evidence_schema,
            "cohort_bucket": fingerprint_cohort_bucket,
            "policy_ref": policy_ref,
            "next_action": sanitize_value(
                stage.get("fingerprint_next_action")
                or stage.get("next_action")
                or "inspect-local-evidence"
            ),
        }
        entry = {
            "schema": EVIDENCE_TO_ACTIVATION_LEDGER_ENTRY_SCHEMA,
            "fingerprint": public_id(json.dumps(fingerprint_material, sort_keys=True), prefix="activation"),
            "lever": lever,
            "local_action_family": action_family,
            "evidence_schema": evidence_schema,
            "cohort_bucket": cohort_bucket,
            "policy_ref": policy_ref,
            "current_status": _ledger_status_from_stage(stage),
            "state": sanitize_value(stage.get("state") or "unknown"),
            "next_action": sanitize_value(stage.get("next_action") or "inspect-local-evidence"),
            "blocker_codes": sanitize_value([str(item) for item in stage.get("blocker_codes") or [] if str(item or "").strip()]),
            "sample_count": _to_int(stage.get("sample_count")),
            "applied_count": _to_int(stage.get("applied_count")),
            "holdout_count": _to_int(stage.get("holdout_count")),
            "fallback_count": _to_int(stage.get("fallback_count")),
            "safety_stop_count": _to_int(stage.get("safety_stop_count")),
            "rollback_count": _to_int(stage.get("rollback_count")),
            "error_rate_delta": round(_to_float(stage.get("error_rate_delta")), 6),
            "retry_rate_delta": round(_to_float(stage.get("retry_rate_delta")), 6),
            "fallback_rate_delta": round(_to_float(stage.get("fallback_rate_delta")), 6),
            "post_widening_status": sanitize_value(stage.get("post_widening_status")),
            "post_widening_next_action": sanitize_value(stage.get("post_widening_next_action")),
            "post_widening_reason_codes": sanitize_value(stage.get("post_widening_reason_codes")),
            "post_max_rollout_status": sanitize_value(stage.get("post_max_rollout_status")),
            "post_max_rollout_decision": sanitize_value(stage.get("post_max_rollout_decision")),
            "post_max_rollout_next_action": sanitize_value(stage.get("post_max_rollout_next_action")),
            "post_max_rollout_reason_codes": sanitize_value(stage.get("post_max_rollout_reason_codes")),
            "post_max_rollout_promotion_allowed": bool(stage.get("post_max_rollout_promotion_allowed")),
            "post_max_rollout_cap_reason": sanitize_value(stage.get("post_max_rollout_cap_reason")),
            "projected_hits": _to_int(stage.get("projected_hits")),
            "actual_hits": _to_int(stage.get("actual_hits")),
            "actual_saved_cost_usd": round(_to_float(stage.get("actual_saved_cost_usd")), 8),
            "miss_count": _to_int(stage.get("miss_count")),
            "bypass_skipped_count": _to_int(stage.get("bypass_skipped_count")),
            "savings_per_1000_calls_usd": round(_to_float(stage.get("savings_per_1000_calls_usd")), 8),
            "projected_saved_usd": round(_to_float(stage.get("projected_saved_usd") or stage.get("projected_saved_cost_usd")), 8),
            "crunch_savings_usd": round(_to_float(stage.get("crunch_savings_usd")), 8),
            "today_crunch_savings_usd": round(_to_float(stage.get("today_crunch_savings_usd")), 8),
            "expected_savings_path": _ledger_expected_savings_path(stage),
            "legacy_issue_title": _legacy_issue_title_for_ledger_entry(stage),
        }
        if stage.get("fingerprint_next_action") and stage.get("fingerprint_next_action") != stage.get("next_action"):
            entry["fingerprint_next_action"] = sanitize_value(stage.get("fingerprint_next_action"))
            entry["lifecycle_progressed_from_next_action"] = sanitize_value(stage.get("fingerprint_next_action"))
        if fingerprint_evidence_schema != evidence_schema:
            entry["fingerprint_evidence_schema"] = fingerprint_evidence_schema
            entry["lifecycle_progressed_from_evidence_schema"] = fingerprint_evidence_schema
        if fingerprint_cohort_bucket != cohort_bucket:
            entry["fingerprint_cohort_bucket"] = fingerprint_cohort_bucket
            entry["lifecycle_progressed_from_cohort_bucket"] = fingerprint_cohort_bucket
        current_status = str(entry.get("current_status") or "")
        entry["issue_worthy_status"] = sanitize_value(
            stage.get("issue_worthy_status")
            or (
                "blocked"
                if current_status in {"blocked", "safety-stopped", "keep-blocked", "retry-later"}
                else "ready"
                if current_status in {"staged", "projected", "measured", "closed-issue-seen"}
                else "evidence-ready"
                if current_status in {"applied", "holdout"}
                else "review"
            )
        )
        if stage.get("diagnostic_class"):
            entry["diagnostic_class"] = sanitize_value(stage.get("diagnostic_class"))
        if stage.get("diagnostic_reason"):
            entry["diagnostic_reason"] = sanitize_value(stage.get("diagnostic_reason"))
        if stage.get("diagnostic_fingerprint"):
            entry["diagnostic_fingerprint"] = sanitize_value(stage.get("diagnostic_fingerprint"))
        if stage.get("verification_check"):
            entry["verification_check"] = sanitize_value(stage.get("verification_check"))
        if stage.get("review_status"):
            entry["review_status"] = sanitize_value(stage.get("review_status"))
        if stage.get("diagnostic_evidence_status"):
            entry["diagnostic_evidence_status"] = sanitize_value(stage.get("diagnostic_evidence_status"))
        if stage.get("evidence_freshness_status"):
            entry["evidence_freshness_status"] = sanitize_value(stage.get("evidence_freshness_status"))
        if stage.get("evidence_age_hours") is not None:
            entry["evidence_age_hours"] = round(_to_float(stage.get("evidence_age_hours")), 3)
        if stage.get("max_evidence_age_hours") is not None:
            entry["max_evidence_age_hours"] = round(_to_float(stage.get("max_evidence_age_hours")), 3)
        if stage.get("durable_action_ledger_entry"):
            entry["durable_action_ledger_entry"] = bool(stage.get("durable_action_ledger_entry"))
        if isinstance(stage.get("privacy"), dict):
            entry["privacy"] = sanitize_value(stage.get("privacy"))
        if stage.get("keep_blocked_reason"):
            entry["keep_blocked_reason"] = sanitize_value(stage.get("keep_blocked_reason"))
        if stage.get("needed_resolution"):
            entry["needed_resolution"] = sanitize_value(stage.get("needed_resolution"))
        if stage.get("next_state"):
            entry["next_state"] = sanitize_value(stage.get("next_state"))
        if stage.get("next_state_reason"):
            entry["next_state_reason"] = sanitize_value(stage.get("next_state_reason"))
        if stage.get("status"):
            entry["status"] = sanitize_value(stage.get("status"))
        if stage.get("safety_stop_count"):
            entry["safety_stop_count"] = _to_int(stage.get("safety_stop_count"))
        if stage.get("safety_stop_breakdown"):
            entry["safety_stop_breakdown"] = sanitize_value(stage.get("safety_stop_breakdown"))
        if isinstance(stage.get("duplicate_suppression"), dict):
            entry["duplicate_suppression"] = sanitize_value(stage.get("duplicate_suppression"))
        for regression_gate_key in ("active_rule_regression_gate", "outcome_gate"):
            if isinstance(stage.get(regression_gate_key), dict):
                entry[regression_gate_key] = sanitize_value(stage.get(regression_gate_key))
        if isinstance(stage.get("miss_reason_breakdown"), list):
            entry["miss_reason_breakdown"] = sanitize_value(stage.get("miss_reason_breakdown"))
        if stage.get("top_miss_reason"):
            entry["top_miss_reason"] = sanitize_value(stage.get("top_miss_reason"))
        if stage.get("observed_hit_blocker"):
            entry["observed_hit_blocker"] = sanitize_value(stage.get("observed_hit_blocker"))
        if stage.get("promotion_blocker"):
            entry["promotion_blocker"] = sanitize_value(stage.get("promotion_blocker"))
        if isinstance(stage.get("unblock_criteria"), dict):
            entry["unblock_criteria"] = sanitize_value(stage.get("unblock_criteria"))
        for review_key in (
            "safety_stop_reason_review",
            "safer_threshold_or_executor_guard",
            "rollback_proof",
            "rollback_metadata",
            "applied_coverage",
            "holdout_coverage",
            "missing_dependency_evidence_review",
            "activation_feedback_freshness_gate",
            "activation_feedback_diagnostic_classification",
        ):
            if isinstance(stage.get(review_key), dict):
                entry[review_key] = sanitize_value(stage.get(review_key))
        if stage.get("dependency_evidence_status"):
            entry["dependency_evidence_status"] = sanitize_value(stage.get("dependency_evidence_status"))
        for dependency_key in (
            "dependency_evidence_class",
            "dependency_evidence_decision",
            "dependency_evidence_reason",
            "evidence_state",
        ):
            if stage.get(dependency_key):
                entry[dependency_key] = sanitize_value(stage.get(dependency_key))
        for count_key in (
            "affected_rows",
            "tools_present_rows",
            "tools_present_replay_evidence_rows",
            "generic_tools_present_blocker_reduced_rows",
            "unsafe_tool_call_blocker_rows",
            "missing_dependency_evidence_rows",
            "stable_dependency_evidence_rows",
            "stale_dependency_evidence_rows",
            "unsafe_dependency_evidence_rows",
            "unknown_dependency_evidence_rows",
            "cache_apply_action_count",
            "cache_entries_written",
            "warmup_miss_count",
            "exact_hit_count",
        ):
            if stage.get(count_key) is not None:
                entry[count_key] = _to_int(stage.get(count_key))
        for gate_key in (
            "tool_cache_replay_enabled",
            "streaming_replay_enabled",
            "emits_cache_apply_action",
            "tools_present_replay_evidence",
            "generic_tools_present_blocker_reduced",
        ):
            if stage.get(gate_key) is not None:
                entry[gate_key] = bool(stage.get(gate_key))
        for breakdown_key in (
            "blocker_breakdown",
            "dependency_evidence_decision_breakdown",
            "evidence_state_breakdown",
            "next_action_breakdown",
        ):
            if isinstance(stage.get(breakdown_key), list):
                entry[breakdown_key] = sanitize_value(stage.get(breakdown_key))
        if isinstance(stage.get("dependency_evidence_review"), dict):
            entry["dependency_evidence_review"] = sanitize_value(stage.get("dependency_evidence_review"))
        if isinstance(stage.get("local_action_representation"), dict):
            entry["local_action_representation"] = sanitize_value(stage.get("local_action_representation"))
        if stage.get("source"):
            entry["source"] = sanitize_value(stage.get("source"))
        for decision_key in (
            "policy_decision",
            "policy_decision_id",
            "promotion_decision",
            "promotion_readiness",
            "reason",
            "reason_codes",
            "source_evidence_schema",
        ):
            if stage.get(decision_key):
                entry[decision_key] = sanitize_value(stage.get(decision_key))
        for source_key in ("source_surface", "endpoint", "category", "workflow_phase", "required_local_executor"):
            if stage.get(source_key):
                entry[source_key] = sanitize_value(stage.get(source_key))
        if stage.get("executor_compatible") is not None:
            entry["executor_compatible"] = bool(stage.get("executor_compatible"))
        if stage.get("requested_model"):
            entry["requested_model"] = sanitize_value(stage.get("requested_model"))
        if stage.get("candidate_target_model"):
            entry["candidate_target_model"] = sanitize_value(stage.get("candidate_target_model"))
        if stage.get("missing_applied_coverage") is not None:
            entry["missing_applied_coverage"] = bool(stage.get("missing_applied_coverage"))
        if stage.get("missing_holdout_coverage") is not None:
            entry["missing_holdout_coverage"] = bool(stage.get("missing_holdout_coverage"))
        if stage.get("burndown_status"):
            entry["burndown_status"] = sanitize_value(stage.get("burndown_status"))
        for gate_key in ("promotion_allowed", "stage_allowed", "active_policy_changed", "wrote_active_policy_files"):
            if stage.get(gate_key) is not None:
                entry[gate_key] = bool(stage.get(gate_key))
        if stage.get("policy_files_written") is not None:
            entry["policy_files_written"] = bool(stage.get("policy_files_written"))
        if stage.get("rollback_required") is not None:
            entry["rollback_required"] = bool(stage.get("rollback_required"))
        if stage.get("rollback_applied") is not None:
            entry["rollback_applied"] = bool(stage.get("rollback_applied"))
        if stage.get("rollback_applied_rule_count") is not None:
            entry["rollback_applied_rule_count"] = _to_int(stage.get("rollback_applied_rule_count"))
        if isinstance(stage.get("rollback_applied_rules"), list):
            entry["rollback_applied_rules"] = sanitize_value(stage.get("rollback_applied_rules"))
        if stage.get("stale_no_traffic_retirement") is not None:
            entry["stale_no_traffic_retirement"] = bool(stage.get("stale_no_traffic_retirement"))
        if stage.get("managed_preview_required") is not None:
            entry["managed_preview_required"] = bool(stage.get("managed_preview_required"))
        if stage.get("terminal_successor_state") is not None:
            entry["terminal_successor_state"] = bool(stage.get("terminal_successor_state"))
        if stage.get("omitted_reason"):
            entry["omitted_reason"] = sanitize_value(stage.get("omitted_reason"))
        if stage.get("follow_up_owner"):
            entry["follow_up_owner"] = sanitize_value(stage.get("follow_up_owner"))
        if stage.get("managed_dependency"):
            entry["managed_dependency"] = sanitize_value(stage.get("managed_dependency"))
        if stage.get("local_handoff_reason"):
            entry["local_handoff_reason"] = sanitize_value(stage.get("local_handoff_reason"))
        if stage.get("activation_follow_up_evidence_schema"):
            entry["activation_follow_up_evidence_schema"] = sanitize_value(stage.get("activation_follow_up_evidence_schema"))
        if stage.get("active_rule_count") is not None:
            entry["active_rule_count"] = _to_int(stage.get("active_rule_count"))
        if stage.get("widened_rule_count") is not None:
            entry["widened_rule_count"] = _to_int(stage.get("widened_rule_count"))
        if stage.get("active_rule_ref"):
            entry["active_rule_ref"] = sanitize_value(stage.get("active_rule_ref"))
        if stage.get("active_rule_source"):
            entry["active_rule_source"] = sanitize_value(stage.get("active_rule_source"))
        if stage.get("active_rule_decision_id"):
            entry["active_rule_decision_id"] = sanitize_value(stage.get("active_rule_decision_id"))
        if stage.get("active_rule_source_evidence_schema"):
            entry["active_rule_source_evidence_schema"] = sanitize_value(stage.get("active_rule_source_evidence_schema"))
        if stage.get("target_local_rule_file"):
            entry["target_local_rule_file"] = sanitize_value(stage.get("target_local_rule_file"))
        if stage.get("target_local_policy_section"):
            entry["target_local_policy_section"] = sanitize_value(stage.get("target_local_policy_section"))
        if stage.get("activation_state"):
            entry["activation_state"] = sanitize_value(stage.get("activation_state"))
        if stage.get("activation_mode"):
            entry["activation_mode"] = sanitize_value(stage.get("activation_mode"))
        if stage.get("follow_up_status"):
            entry["follow_up_status"] = sanitize_value(stage.get("follow_up_status"))
        if stage.get("canary_already_staged") is not None:
            entry["canary_already_staged"] = bool(stage.get("canary_already_staged"))
        if stage.get("canary_already_applied") is not None:
            entry["canary_already_applied"] = bool(stage.get("canary_already_applied"))
        representation = (
            stage.get("local_file_backed_representation")
            if isinstance(stage.get("local_file_backed_representation"), dict)
            else {}
        )
        if representation:
            entry["local_file_backed_representation"] = sanitize_value(representation)
        matched = _ledger_issue_match(entry, existing_issues)
        if matched is not None and str(matched.get("number") or ""):
            entry["prior_issue"] = matched
            if not any(_issue_number(issue) == matched.get("number") and _is_open(issue) for issue in existing_issues):
                entry["issue_status"] = "closed-issue-seen"
        preserved_empty_keys = {
            "sample_count",
            "applied_count",
            "holdout_count",
            "fallback_count",
            "safety_stop_count",
            "rollback_count",
            "error_rate_delta",
            "retry_rate_delta",
            "fallback_rate_delta",
            "projected_saved_usd",
            "crunch_savings_usd",
            "today_crunch_savings_usd",
            "affected_rows",
            "tools_present_rows",
            "tools_present_replay_evidence_rows",
            "generic_tools_present_blocker_reduced_rows",
            "unsafe_tool_call_blocker_rows",
            "missing_dependency_evidence_rows",
            "stable_dependency_evidence_rows",
            "stale_dependency_evidence_rows",
            "unsafe_dependency_evidence_rows",
            "unknown_dependency_evidence_rows",
            "cache_apply_action_count",
            "cache_entries_written",
            "tool_cache_replay_enabled",
            "streaming_replay_enabled",
            "emits_cache_apply_action",
            "tools_present_replay_evidence",
            "generic_tools_present_blocker_reduced",
            "promotion_allowed",
            "stage_allowed",
            "active_policy_changed",
            "wrote_active_policy_files",
            "policy_files_written",
            "managed_preview_required",
            "executor_compatible",
            "evidence_age_hours",
            "max_evidence_age_hours",
        }
        entries.append({key: value for key, value in entry.items() if value not in (None, "", [], 0) or key in preserved_empty_keys})

    entries.sort(
        key=lambda item: (
            _loop_state_rank(str(item.get("state") or "")),
            -_to_float(item.get("savings_per_1000_calls_usd") or item.get("projected_saved_usd")),
            -_to_int(item.get("sample_count")),
            str(item.get("lever") or ""),
        )
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    top = entries[0] if entries else {}
    status_counts = Counter(str(entry.get("current_status") or "unknown") for entry in entries)
    issue_status_counts = Counter(str(entry.get("issue_status") or "none") for entry in entries)
    result = {
        "schema": EVIDENCE_TO_ACTIVATION_LEDGER_SCHEMA,
        "status": "tracked" if entries else "empty",
        "summary": {
            "tracked_entry_count": len(entries),
            "closed_issue_seen_count": sum(1 for entry in entries if entry.get("issue_status") == "closed-issue-seen"),
            "top_lever": top.get("lever"),
            "top_current_status": top.get("current_status"),
            "top_next_action": top.get("next_action"),
            "top_blocker_codes": top.get("blocker_codes") or [],
            "top_expected_savings_path": top.get("expected_savings_path"),
            "status_counts": [{"status": status, "count": count} for status, count in sorted(status_counts.items())],
            "issue_status_counts": [
                {"status": status, "count": count}
                for status, count in sorted(issue_status_counts.items())
                if status != "none"
            ],
        },
        "entries": entries[:20],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
        },
    }
    return sanitize_value(result)


def _refresh_ledger_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for entry in ledger.get("entries") or [] if isinstance(entry, dict)]
    top = entries[0] if entries else {}
    status_counts = Counter(str(entry.get("current_status") or "unknown") for entry in entries)
    issue_status_counts = Counter(str(entry.get("issue_status") or "none") for entry in entries)
    summary = dict(ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {})
    summary.update(
        {
            "tracked_entry_count": len(entries),
            "closed_issue_seen_count": sum(1 for entry in entries if entry.get("issue_status") == "closed-issue-seen"),
            "top_lever": top.get("lever"),
            "top_current_status": top.get("current_status"),
            "top_next_action": top.get("next_action"),
            "top_blocker_codes": top.get("blocker_codes") or [],
            "top_expected_savings_path": top.get("expected_savings_path"),
            "status_counts": [{"status": status, "count": count} for status, count in sorted(status_counts.items())],
            "issue_status_counts": [
                {"status": status, "count": count}
                for status, count in sorted(issue_status_counts.items())
                if status != "none"
            ],
        }
    )
    refreshed = dict(ledger)
    refreshed["summary"] = summary
    return refreshed


def _merge_full_rollout_crunch_measurement_into_ledger(
    ledger: dict[str, Any],
    measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(ledger, dict) or not isinstance(measurement, dict):
        return ledger
    fingerprint = str(measurement.get("ledger_fingerprint") or "").strip()
    durable_outcome = (
        measurement.get("durable_full_rollout_outcome")
        if isinstance(measurement.get("durable_full_rollout_outcome"), dict)
        else {}
    )
    if not fingerprint or not durable_outcome:
        return ledger

    merged_entries: list[dict[str, Any]] = []
    changed = False
    for entry in ledger.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        merged = dict(entry)
        if str(entry.get("fingerprint") or "") == fingerprint:
            changed = True
            merged.update(
                {
                    "durable_outcome_ledger_entry": True,
                    "full_rollout_activation_outcome": sanitize_value(durable_outcome),
                    "full_rollout_outcome": sanitize_value(durable_outcome.get("outcome")),
                    "full_rollout_outcome_next_action": sanitize_value(durable_outcome.get("next_action")),
                    "full_rollout_successor_decision": sanitize_value(durable_outcome.get("successor_decision")),
                    "full_rollout_successor_next_action": sanitize_value(durable_outcome.get("successor_next_action")),
                    "full_rollout_successor_no_op_reason": sanitize_value(durable_outcome.get("successor_no_op_reason")),
                    "keep_active_regression_gate": sanitize_value(measurement.get("keep_active_regression_gate")),
                    "measured_full_rollout_activation": True,
                }
            )
        merged_entries.append(merged)
    if not changed:
        return ledger
    merged_ledger = dict(ledger)
    merged_ledger["entries"] = merged_entries
    return _refresh_ledger_summary(merged_ledger)


def _queue_duplicate_suppression_status(entry: dict[str, Any]) -> str:
    suppression = entry.get("duplicate_suppression")
    if not isinstance(suppression, dict):
        return "none"
    for key, value in suppression.items():
        if str(key).startswith("suppresses_") and value is True:
            return "suppressed"
    return "present"


def _queue_unblock_reason(entry: dict[str, Any]) -> str:
    blockers = [str(item) for item in entry.get("blocker_codes") or [] if str(item or "").strip()]
    if blockers:
        return sanitize_value(blockers[0])
    for key in (
        "promotion_blocker",
        "observed_hit_blocker",
        "top_miss_reason",
        "reason",
        "omitted_reason",
        "keep_blocked_reason",
    ):
        value = str(entry.get(key) or "").strip()
        if value:
            return sanitize_value(value)
    suppression = entry.get("duplicate_suppression")
    if isinstance(suppression, dict) and suppression.get("reason"):
        return sanitize_value(suppression.get("reason"))
    status = str(entry.get("current_status") or entry.get("state") or "inspect-local-evidence").strip()
    return sanitize_value(status)


def _queue_projected_savings(entry: dict[str, Any]) -> float:
    projected = _to_float(entry.get("projected_saved_usd") or entry.get("projected_savings_usd"))
    if projected > 0:
        return projected
    savings_per_1000 = _to_float(entry.get("savings_per_1000_calls_usd"))
    sample_count = _to_int(entry.get("sample_count"))
    if savings_per_1000 > 0 and sample_count > 0:
        return savings_per_1000 * sample_count / 1000.0
    return 0.0


def _queue_realized_savings(entry: dict[str, Any], projected: float) -> float:
    for key in (
        "actual_saved_cost_usd",
        "observed_savings_usd",
        "observed_saved_usd",
        "crunch_savings_usd",
        "today_crunch_savings_usd",
    ):
        value = _to_float(entry.get(key))
        if value > 0:
            return value
    if str(entry.get("current_status") or "") in {"applied", "measured", "holdout", "full-rollout"} and projected > 0:
        return projected
    return 0.0


def _cache_replay_rollback_applied(entry: dict[str, Any]) -> bool:
    if bool(entry.get("rollback_applied")):
        return True
    for key in ("rollback_applied_rules", "applied_rollback_rules"):
        value = entry.get(key)
        if isinstance(value, list) and value:
            return True
    for key in ("rollback_applied_rule_count", "applied_rollback_rule_count"):
        if _to_int(entry.get(key)) > 0:
            return True
    return False


def _cache_replay_post_rollback_classification(entry: dict[str, Any]) -> dict[str, Any] | None:
    family = str(entry.get("local_action_family") or entry.get("lever") or "").strip()
    if family != "cache":
        return None
    target_file = str(entry.get("target_local_rule_file") or "").strip()
    target_section = str(entry.get("target_local_policy_section") or "").strip()
    evidence_schema = str(entry.get("evidence_schema") or entry.get("source_evidence_schema") or "")
    text = " ".join(
        [
            str(entry.get("state") or ""),
            str(entry.get("current_status") or ""),
            str(entry.get("next_action") or ""),
            str(entry.get("promotion_readiness") or ""),
            str(entry.get("promotion_recommendation") or ""),
            str(entry.get("policy_decision") or ""),
            str(entry.get("reason") or ""),
            " ".join(str(item) for item in entry.get("blocker_codes") or []),
        ]
    ).lower()
    cache_replay_shape = bool(
        target_file == "cache_rules.yaml"
        or target_section == "cache.pattern_rules"
        or "cache_replay" in evidence_schema
        or "cache-replay" in evidence_schema
        or "cache-replay" in text
        or "cache replay" in text
    )
    if not cache_replay_shape:
        return None

    no_repeat = bool(
        entry.get("stale_no_traffic_retirement")
        or "retire-staged-no-repeat" in text
        or "retired-no-repeat" in text
        or "retired-stale-no-traffic" in text
    )
    if no_repeat:
        return None

    rollback_required = bool(
        entry.get("rollback_required")
        or "rollback-required" in text
        or "rollback-cache-replay-rule" in text
        or str(entry.get("policy_decision") or "") == "rollback"
    )
    if not rollback_required:
        return None

    evidence_age_hours = _queue_evidence_age_hours(entry)
    max_age_hours = _queue_max_evidence_age_hours(entry)
    if max_age_hours is None:
        max_age_hours = 72.0
    applied_count = _to_int(entry.get("applied_count"))
    holdout_count = _to_int(entry.get("holdout_count"))
    miss_count = _to_int(entry.get("miss_count"))
    warmup_miss_count = _to_int(entry.get("warmup_miss_count"))
    exact_hit_count = _to_int(entry.get("exact_hit_count") or entry.get("actual_hits"))
    no_reobserve_traffic = bool(
        applied_count <= 0
        and holdout_count <= 0
        and miss_count <= 0
        and warmup_miss_count <= 0
        and exact_hit_count <= 0
    )
    observation_window_elapsed = bool(
        evidence_age_hours is not None
        and max_age_hours is not None
        and evidence_age_hours >= max_age_hours
    )

    if _cache_replay_rollback_applied(entry):
        if no_reobserve_traffic and observation_window_elapsed:
            return {
                "decision": "retired-stale-no-traffic",
                "next_action": "retire-stale-cache-replay-successor-no-traffic",
                "reason": "post-rollback-observation-window-elapsed-no-traffic",
                "issue_worthy_status": "suppressed",
                "current_status": "superseded",
                "state": "retired-stale-no-traffic",
                "blocker_codes": ["post-rollback-observation-window-elapsed-no-traffic"],
                "stale_no_traffic_retirement": True,
                "durable_action_ledger_entry": True,
                "terminal_successor_state": True,
                "observation_window_elapsed": observation_window_elapsed,
                "no_reobserve_traffic": no_reobserve_traffic,
                "evidence_age_hours": evidence_age_hours,
                "max_evidence_age_hours": max_age_hours,
            }
        return {
            "decision": "reobserve-after-rollback",
            "next_action": "reobserve-cache-replay-after-rollback",
            "reason": "cache-replay-rollback-applied",
            "issue_worthy_status": "review",
            "current_status": "review",
            "state": "reobserve-after-rollback",
            "blocker_codes": ["cache-replay-rollback-applied"],
            "observation_window_elapsed": observation_window_elapsed,
            "no_reobserve_traffic": no_reobserve_traffic,
            "evidence_age_hours": evidence_age_hours,
            "max_evidence_age_hours": max_age_hours,
        }

    return {
        "decision": "keep-blocked-narrow",
        "next_action": "apply-cache-replay-rollback-before-reobserve",
        "reason": "cache-replay-rollback-not-applied",
        "issue_worthy_status": "blocked",
        "current_status": "blocked",
        "state": "keep-blocked-narrow",
        "blocker_codes": ["cache-replay-rollback-not-applied"],
        "observation_window_elapsed": observation_window_elapsed,
        "no_reobserve_traffic": no_reobserve_traffic,
        "evidence_age_hours": evidence_age_hours,
        "max_evidence_age_hours": max_age_hours,
    }


def _apply_cache_replay_post_rollback_classification(entry: dict[str, Any]) -> None:
    classification = _cache_replay_post_rollback_classification(entry)
    if not classification:
        return
    entry["post_rollback_successor_decision"] = classification["decision"]
    entry["post_rollback_next_action"] = classification["next_action"]
    entry["post_rollback_reason"] = classification["reason"]
    entry["next_action"] = classification["next_action"]
    entry["unblock_reason"] = classification["reason"]
    entry["blocking_reason"] = classification["reason"]
    entry["blocker_codes"] = classification["blocker_codes"]
    entry["issue_worthy_status"] = classification["issue_worthy_status"]
    entry["current_status"] = classification["current_status"]
    entry["state"] = classification["state"]
    entry["cache_apply_action_count"] = 0
    entry["cache_entries_written"] = 0
    entry["emits_cache_apply_action"] = False
    entry["policy_files_written"] = False
    if classification["decision"] in {"reobserve-after-rollback", "keep-blocked-narrow", "retired-stale-no-traffic"}:
        entry["rollback_required"] = True
    if classification.get("stale_no_traffic_retirement") is not None:
        entry["stale_no_traffic_retirement"] = bool(classification.get("stale_no_traffic_retirement"))
    if classification.get("durable_action_ledger_entry") is not None:
        entry["durable_action_ledger_entry"] = bool(classification.get("durable_action_ledger_entry"))
    if classification.get("terminal_successor_state") is not None:
        entry["terminal_successor_state"] = bool(classification.get("terminal_successor_state"))
    if classification["decision"] == "retired-stale-no-traffic":
        entry["duplicate_suppression"] = _cache_replay_stale_no_traffic_duplicate_suppression(
            decision_report={},
            top_decision={},
            reason=classification["reason"],
            reason_codes=classification["blocker_codes"],
            target_local_rule_file=entry.get("target_local_rule_file"),
            target_local_policy_section=entry.get("target_local_policy_section"),
        )
    entry["post_rollback_observation"] = {
        "schema": "tokenclaw.cache_replay_post_rollback_observation.v1",
        "decision": classification["decision"],
        "next_action": classification["next_action"],
        "reason": classification["reason"],
        "applied_count": _to_int(entry.get("applied_count")),
        "holdout_count": _to_int(entry.get("holdout_count")),
        "miss_count": _to_int(entry.get("miss_count")),
        "warmup_miss_count": _to_int(entry.get("warmup_miss_count")),
        "exact_hit_count": _to_int(entry.get("exact_hit_count") or entry.get("actual_hits")),
        "observation_age_hours": round(_to_float(classification.get("evidence_age_hours")), 3),
        "max_observation_age_hours": round(_to_float(classification.get("max_evidence_age_hours")), 3),
        "observation_window_elapsed": bool(classification.get("observation_window_elapsed")),
        "no_reobserve_traffic": bool(classification.get("no_reobserve_traffic")),
        "stale_age_hours": round(_to_float(classification.get("evidence_age_hours")), 3),
        "blocker_codes": classification["blocker_codes"],
        "terminal_successor_state": bool(classification.get("terminal_successor_state")),
        "metadata_only": True,
        "aggregate_only": True,
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "emits_cache_apply_action": False,
        "policy_files_written": False,
    }


def _successor_action_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _successor_action_acceptance_metric(entry: dict[str, Any]) -> str:
    family = str(entry.get("local_action_family") or entry.get("lever") or "").strip()
    blockers = {str(item) for item in entry.get("blocker_codes") or []}
    next_action = str(entry.get("next_action") or "")
    state = str(entry.get("state") or "")
    if family == "routing" and "semantic-quality-regression-observed" in blockers:
        return (
            "A narrower routing canary or rollback review records applied/holdout coverage, "
            "semantic regression blocker status, and no local policy write until the blocker clears."
        )
    if family == "routing" and (
        entry.get("safety_stop_count")
        or "safety-stop" in " ".join(sorted(blockers))
        or "anthropic-routing" in str(entry.get("unblock_reason") or "")
    ):
        return (
            "Anthropic routing remains blocked until safety_stop_count is zero and safety-stop reason "
            "review, safer executor guard, rollback proof, applied coverage, and holdout coverage pass."
        )
    if family == "routing":
        return (
            "Routing successors record applied/holdout coverage, blocker status, regression status, "
            "and the narrower local canary or rollback action before any local policy write."
        )
    if family == "cache":
        return (
            "Tool-cache replay candidates classify dependency evidence as stable, stale, unsafe, "
            "unknown, or missing, while emitting no cache apply actions or cache entries for unsafe shapes."
        )
    if family == "crunch" or "crunch" in next_action or "crunch" in state:
        return (
            "Full-rollout repeated-context crunch stays keep-active with applied coverage, holdout coverage, "
            "zero safety stops, zero rollbacks, and duplicate activation issue suppression."
        )
    if family == "managed-recommendation":
        return (
            "Managed recommendation omissions resolve to local file-backed handoff or explicit no-op rows "
            "without requiring managed-server execution."
        )
    return (
        "The successor action has a stable fingerprint, concrete next action, local action family, "
        "and metadata-only privacy flags."
    )


def _successor_action_status(entry: dict[str, Any]) -> str:
    post_rollback_decision = str(entry.get("post_rollback_successor_decision") or "").strip()
    if post_rollback_decision in {
        "reobserve-after-rollback",
        "keep-blocked-narrow",
        "retire-staged-no-repeat",
        "suppressed-closed-successor",
    }:
        return post_rollback_decision
    duplicate_status = str(entry.get("duplicate_suppression_status") or "")
    current_status = str(entry.get("current_status") or "")
    state = str(entry.get("state") or "")
    issue_status = str(entry.get("issue_worthy_status") or "")
    if state == "retired-stale-no-traffic":
        return "retired-stale-no-traffic"
    if (
        current_status in _TERMINAL_ACTIVATION_SUCCESSOR_STATES
        or state in _TERMINAL_ACTIVATION_SUCCESSOR_STATES
        or bool(entry.get("terminal_successor_state"))
    ):
        return "resolved-no-action"
    if current_status == "suppressed" or state == "suppressed" or issue_status == "suppressed":
        return "suppress-duplicate"
    if duplicate_status == "suppressed":
        if current_status == "full-rollout" or entry.get("measured_full_rollout_activation"):
            return "keep-current-rule"
        return "suppress-duplicate"
    if issue_status == "blocked" or current_status in {"blocked", "keep-blocked"} or state == "keep-blocked":
        return "keep-blocked"
    if entry.get("emits_cache_apply_action") is False or entry.get("policy_files_written") is False:
        return "review-only"
    if issue_status == "ready":
        return "ready"
    return "review"


def _managed_preview_outcomes_report(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "managed_activation_preview_outcomes",
        "managed_preview_outcomes",
        "managed_activation_preview_outcome_summary",
    ):
        value = stats_summary.get(key)
        if isinstance(value, dict):
            return value
    return None


def _managed_preview_health_report(
    stats_summary: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any] | None:
    for key in (
        "managed_activation_preview_health",
        "local_activation_executor_handoff_preview_health",
        "managed_activation_handoff_preview_health",
        "managed_preview_health",
    ):
        value = stats_summary.get(key)
        if isinstance(value, dict):
            return value
    if isinstance(stats_summary.get("managed_activation_preview_result"), dict):
        return stats_summary["managed_activation_preview_result"]
    if isinstance(managed_preview_outcomes, dict):
        for key in (
            "managed_activation_preview_health",
            "local_activation_executor_handoff_preview_health",
            "health",
            "source_health",
        ):
            value = managed_preview_outcomes.get(key)
            if isinstance(value, dict):
                return value
    return managed_preview_outcomes if isinstance(managed_preview_outcomes, dict) else None


def _managed_preview_successor_privacy(*, managed_server_calls_made: bool = False) -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "feature_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": bool(managed_server_calls_made),
        "current_run_managed_server_calls_made": False,
    }


def _first_preview_reason(rows: Any) -> str | None:
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("reason_code", "reason", "value", "code", "status"):
                value = str(row.get(key) or "").strip()
                if value:
                    return sanitize_value(value)
    return None


def _managed_preview_family_count(report: dict[str, Any], keys: tuple[str, ...], family: str) -> int:
    for key in keys:
        value = report.get(key)
        if isinstance(value, dict):
            return _to_int(value.get(family))
    return 0


def _managed_preview_health_age_hours(report: dict[str, Any]) -> float | None:
    for key in ("latest_preview_age_hours", "preview_age_hours", "age_hours"):
        if report.get(key) is not None:
            return _to_float(report.get(key))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    for key in ("latest_preview_age_hours", "preview_age_hours", "age_hours"):
        if summary.get(key) is not None:
            return _to_float(summary.get(key))
    latest = report.get("newest_preview_at") or report.get("latest_preview_at") or report.get("generated_at")
    parsed = _parse_time(latest)
    if parsed is None:
        return None
    return max(0.0, round((datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0, 3))


def _managed_preview_health_gate(
    health_report: dict[str, Any] | None,
    *,
    family: str,
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any]:
    report = health_report if isinstance(health_report, dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    fetch = report.get("fetch") if isinstance(report.get("fetch"), dict) else {}
    outcome_summary = (
        managed_preview_outcomes.get("summary")
        if isinstance(managed_preview_outcomes, dict) and isinstance(managed_preview_outcomes.get("summary"), dict)
        else {}
    )
    outcomes = [row for row in (managed_preview_outcomes or {}).get("outcomes", []) if isinstance(row, dict)]
    family_outcomes = [
        row
        for row in outcomes
        if str(row.get("local_action_family") or "") == family
    ]
    status_text = str(report.get("status") or summary.get("status") or "").strip()
    fetch_status = str(fetch.get("status") or "").strip()
    stale_after_hours = _to_float(
        report.get("stale_after_hours")
        or summary.get("stale_after_hours")
        or (managed_preview_outcomes or {}).get("stale_after_hours")
        or outcome_summary.get("stale_after_hours")
        or 72.0,
        72.0,
    )
    age_hours = _managed_preview_health_age_hours(report)
    stale = bool(
        report.get("stale")
        or summary.get("stale")
        or (age_hours is not None and age_hours > max(0.0, stale_after_hours))
    )
    accepted_batch_count = _to_int(
        report.get("accepted_batch_count")
        or summary.get("accepted_batch_count")
        or (1 if fetch_status == "ok" or status_text in {"previewed", "tracked"} else 0)
    )
    rejected_batch_count = _to_int(report.get("rejected_batch_count") or summary.get("rejected_batch_count"))
    submitted_row_count = _to_int(
        report.get("submitted_row_count")
        or summary.get("submitted_row_count")
        or coverage.get("handoff_row_count")
        or summary.get("handoff_row_count")
    )
    previewed_row_count = _to_int(
        report.get("previewed_row_count")
        or summary.get("previewed_row_count")
        or coverage.get("preview_decision_count")
        or summary.get("preview_decision_count")
        or summary.get("preview_row_count")
        or len(outcomes)
    )
    omitted_row_count = _to_int(
        report.get("omitted_row_count")
        or summary.get("omitted_row_count")
        or coverage.get("omitted_count")
        or summary.get("omitted_count")
        or summary.get("omission_count")
        or sum(1 for row in outcomes if row.get("omitted_reason"))
    )
    rejected_row_count = _to_int(report.get("rejected_row_count") or summary.get("rejected_row_count"))
    privacy_rejection_count = _to_int(report.get("privacy_rejection_count") or summary.get("privacy_rejection_count"))
    validation_error_count = _to_int(report.get("validation_error_count") or summary.get("validation_error_count"))
    stored_count = _to_int(
        summary.get("stored_preview_outcome_count")
        or report.get("stored_preview_outcome_count")
        or len(outcomes)
    )
    fresh_outcome_count = sum(
        1
        for row in family_outcomes
        if not bool(row.get("stale"))
        and not bool(row.get("missing_preview_decision"))
        and not bool(row.get("no_data_preview_health"))
        and str(row.get("classification") or "") != "no-data-preview-health"
        and not bool(row.get("failed_closed"))
        and not bool(row.get("disagrees_with_local_evidence"))
    )
    no_data_outcome_count = sum(
        1
        for row in family_outcomes
        if bool(row.get("no_data_preview_health")) or str(row.get("classification") or "") == "no-data-preview-health"
    )
    family_previewed_count = _managed_preview_family_count(
        report,
        ("previewed_counts_by_local_action_family", "local_action_family_counts"),
        family,
    ) or len(family_outcomes)
    family_omitted_count = _managed_preview_family_count(
        report,
        ("omitted_counts_by_local_action_family",),
        family,
    )
    family_rejected_count = _managed_preview_family_count(
        report,
        ("rejected_counts_by_local_action_family",),
        family,
    )
    top_omission_reason = _first_preview_reason(report.get("top_omission_reasons"))
    top_rejection_reason = _first_preview_reason(report.get("top_rejection_reasons"))
    fetch_reason = str(fetch.get("reason") or report.get("reason") or "").strip()
    if top_rejection_reason is None and fetch_reason:
        top_rejection_reason = sanitize_value(fetch_reason)
    has_report = bool(report)
    if privacy_rejection_count > 0 or str(top_rejection_reason or "").lower() == "privacy-rejection":
        status = "privacy-rejected-preview-health"
        reason = "privacy-rejection"
        next_action = "review-managed-activation-preview-privacy-rejection"
    elif rejected_batch_count > 0 or rejected_row_count > 0 or validation_error_count > 0 or fetch_status in {"blocked", "error"} or status_text in {"blocked", "error", "unavailable"}:
        status = "rejected-preview-health"
        reason = top_rejection_reason or fetch_reason or "managed-preview-health-rejected"
        next_action = "review-managed-activation-preview-rejection"
    elif (
        no_data_outcome_count > 0
        or not has_report
        or status_text in {"", "no-data", "skipped"}
        or (accepted_batch_count == 0 and previewed_row_count == 0 and stored_count == 0)
    ):
        status = "no-data-preview-health"
        reason = fetch_reason or "managed-preview-health-no-data"
        next_action = "refresh-managed-activation-preview"
    elif stale:
        status = "stale-preview-health"
        reason = "managed-preview-health-stale"
        next_action = "refresh-managed-activation-preview"
    elif accepted_batch_count <= 0 or previewed_row_count <= 0 or (family_previewed_count <= 0 and family_outcomes):
        status = "incomplete-preview-health"
        reason = "managed-preview-health-incomplete"
        next_action = "refresh-managed-activation-preview"
    else:
        status = "fresh-preview-health"
        reason = "managed-preview-health-fresh"
        next_action = "use-managed-preview-gate"
    passed = status == "fresh-preview-health"
    gate = {
        "schema": MANAGED_PREVIEW_HEALTH_GATE_SCHEMA,
        "status": status,
        "passed": passed,
        "reason": reason,
        "next_action": next_action,
        "managed_dependency": "optional",
        "managed_server_calls_made": bool(report.get("managed_server_calls_made") or fetch.get("managed_server_calls_made")),
        "policy_files_written": False,
        "provider_calls_made": False,
        "accepted_batch_count": accepted_batch_count,
        "rejected_batch_count": rejected_batch_count,
        "submitted_row_count": submitted_row_count,
        "previewed_row_count": previewed_row_count,
        "omitted_row_count": omitted_row_count,
        "rejected_row_count": rejected_row_count,
        "privacy_rejection_count": privacy_rejection_count,
        "validation_error_count": validation_error_count,
        "stored_preview_outcome_count": stored_count,
        "family_previewed_count": family_previewed_count,
        "family_omitted_count": family_omitted_count,
        "family_rejected_count": family_rejected_count,
        "latest_preview_age_hours": age_hours,
        "stale_after_hours": stale_after_hours,
        "top_omission_reason": top_omission_reason,
        "top_rejection_reason": top_rejection_reason,
        "privacy": _managed_preview_successor_privacy(
            managed_server_calls_made=bool(report.get("managed_server_calls_made") or fetch.get("managed_server_calls_made"))
        ),
    }
    return sanitize_value(gate)


def _managed_preview_required_for_successor(entry: dict[str, Any]) -> bool:
    if entry.get("managed_preview_required") is not None:
        return bool(entry.get("managed_preview_required"))
    family = str(entry.get("local_action_family") or entry.get("lever") or "").strip()
    current_status = str(entry.get("current_status") or "").strip()
    state = str(entry.get("state") or "").strip()
    duplicate_status = str(entry.get("duplicate_suppression_status") or "").strip()
    if current_status in _TERMINAL_ACTIVATION_SUCCESSOR_STATES or state in _TERMINAL_ACTIVATION_SUCCESSOR_STATES:
        return False
    if duplicate_status == "suppressed" or current_status in {"full-rollout", "superseded"}:
        return False
    return family in {"routing", "cache", "managed-recommendation", "activation-feedback"}


def _is_request_shape_rollup_successor(entry: dict[str, Any], outcome: dict[str, Any] | None = None) -> bool:
    outcome = outcome if isinstance(outcome, dict) else {}
    family = str(entry.get("local_action_family") or entry.get("lever") or outcome.get("local_action_family") or "").strip()
    lever = str(entry.get("lever") or outcome.get("lever") or "").strip()
    evidence_schema = str(entry.get("evidence_schema") or outcome.get("evidence_schema") or "").strip()
    next_action = str(entry.get("next_action") or outcome.get("next_action") or "").strip()
    blockers = {str(item) for item in entry.get("blocker_codes") or [] if str(item or "").strip()}
    return bool(
        family == "cohort-ranking"
        or lever == "request-shape-rollups"
        or evidence_schema == "tokenclaw.request_shape_follow_up_candidates.v1"
        or next_action == "emit-request-shape-rollups"
        or "ranked_request_shape_rollup" in blockers
    )


def _request_shape_rollup_outcome_class(outcome: dict[str, Any]) -> str:
    for key in ("cohort_class", "rollup_outcome_status", "request_shape_rollup_outcome_status"):
        text = str(outcome.get(key) or "").strip().lower().replace("_", "-")
        if text in {"no-data", "stale", "too-small", "unsafe", "review-ready"}:
            return text
    return ""


def _request_shape_rollup_preview_status(outcome_class: str) -> str:
    return {
        "no-data": "no-data-preview-health",
        "stale": "stale-preview",
        "too-small": "request-shape-rollup-too-small",
        "unsafe": "unsafe-request-shape-rollup",
        "review-ready": "preview-verified",
    }.get(outcome_class, "")


def _request_shape_rollup_preview_decision(outcome_class: str) -> str:
    if outcome_class == "review-ready":
        return "ready"
    if outcome_class == "stale":
        return "review-stale-preview"
    return "keep-blocked"


def _request_shape_rollup_next_action(outcome_class: str, outcome: dict[str, Any], entry: dict[str, Any]) -> str:
    explicit = str(outcome.get("next_action") or outcome.get("recommended_next_action") or "").strip()
    if outcome_class == "review-ready":
        return sanitize_value(explicit or "review-request-shape-rollup-local-action")
    if outcome_class == "stale":
        return sanitize_value(explicit or "refresh-managed-activation-preview")
    if outcome_class == "too-small":
        return sanitize_value(explicit or "keep-request-shape-rollups-observing")
    if outcome_class == "unsafe":
        return sanitize_value(explicit or "keep-request-shape-rollup-blocked-for-safety")
    if outcome_class == "no-data":
        return sanitize_value(explicit or entry.get("next_action") or "emit-request-shape-rollups")
    return sanitize_value(explicit or entry.get("next_action") or "inspect-local-evidence")


def _is_repeated_context_crunch_successor(entry: dict[str, Any], outcome: dict[str, Any] | None = None) -> bool:
    outcome = outcome if isinstance(outcome, dict) else {}
    family = str(entry.get("local_action_family") or entry.get("lever") or outcome.get("local_action_family") or "").strip()
    evidence_schema = str(entry.get("evidence_schema") or outcome.get("evidence_schema") or "").strip()
    next_action = str(entry.get("next_action") or outcome.get("next_action") or "").strip()
    cohort_class = str(entry.get("cohort_class") or outcome.get("cohort_class") or "").strip()
    blockers = {str(item) for item in entry.get("blocker_codes") or [] if str(item or "").strip()}
    return bool(
        family == "crunch"
        and (
            evidence_schema
            in {
                "tokenclaw.crunch_savings_signal.v1",
                "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
                "tokenclaw.request_shape_crunch_canary_impact.v1",
                "tokenclaw.request_shape_crunch_policy_decision.v1",
                "tokenclaw.request_shape_crunch_activation_evidence.v1",
            }
            or "repeated-context-crunch" in next_action
            or "repeated-context-crunch" in cohort_class
            or any("repeated-context-crunch" in blocker or "crunch-preview" in blocker for blocker in blockers)
        )
    )


def _crunch_preview_decision(outcome: dict[str, Any]) -> str:
    for key in ("crunch_preview_decision", "decision", "status"):
        text = str(outcome.get(key) or "").strip().lower().replace("_", "-")
        if text in {"review-ready", "keep-staged", "keep-blocked", "too-small", "quality-risk"}:
            return text
    return ""


def _crunch_preview_status(decision: str) -> str:
    return {
        "review-ready": "preview-verified",
        "keep-staged": "crunch-preview-keep-staged",
        "keep-blocked": "crunch-preview-keep-blocked",
        "too-small": "crunch-preview-too-small",
        "quality-risk": "crunch-preview-quality-risk",
    }.get(decision, "")


def _crunch_preview_successor_decision(decision: str) -> str:
    if decision == "review-ready":
        return "review-only"
    if decision == "keep-staged":
        return "review-staged"
    return "keep-blocked"


def _crunch_preview_next_action(decision: str, outcome: dict[str, Any], entry: dict[str, Any]) -> str:
    explicit = str(outcome.get("next_action") or outcome.get("recommended_next_action") or "").strip()
    if decision == "review-ready":
        return sanitize_value(explicit or "review-repeated-context-crunch-preview")
    if decision == "keep-staged":
        return sanitize_value(explicit or "keep-repeated-context-crunch-staged")
    if decision == "too-small":
        return sanitize_value(explicit or "keep-repeated-context-crunch-observing")
    if decision == "quality-risk":
        return sanitize_value(explicit or "keep-repeated-context-crunch-blocked")
    if decision == "keep-blocked":
        return sanitize_value(explicit or "collect-repeated-context-crunch-evidence")
    return sanitize_value(explicit or entry.get("next_action") or "rank-repeated-context-crunch-dry-run")


def _managed_preview_local_executor_gate(entry: dict[str, Any]) -> dict[str, Any]:
    action_status = _successor_action_status(entry)
    policy_write_candidate = bool(
        entry.get("promotion_allowed")
        or entry.get("stage_allowed")
        or action_status in {"ready", "review-only", "review"}
    )
    passed = bool(
        entry.get("promotion_allowed")
        or entry.get("stage_allowed")
        or action_status in {"keep-current-rule", "suppress-duplicate", "keep-blocked", "resolved-no-action"}
    )
    return {
        "schema": "tokenclaw.preview_verified_successor_local_executor_gate.v1",
        "passed": passed,
        "policy_write_candidate": policy_write_candidate,
        "policy_files_written": False,
        "provider_calls_made": False,
        "reason": "local-executor-gate-passed" if passed else "local-executor-gate-not-passed",
    }


def _managed_preview_public_ref(value: Any, *, prefix: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return public_id(text, prefix=prefix, fallback=None)


def _managed_preview_entry_successor_fingerprint(entry: dict[str, Any]) -> str | None:
    source_fingerprint = sanitize_value(entry.get("source_fingerprint") or entry.get("fingerprint") or "")
    if not source_fingerprint:
        return None
    material = {
        "fingerprint": source_fingerprint,
        "next_action": sanitize_value(entry.get("next_action") or "inspect-local-evidence"),
        "local_action_family": sanitize_value(entry.get("local_action_family") or entry.get("lever") or "unknown"),
    }
    return public_id(json.dumps(material, sort_keys=True), prefix="successor", fallback=None)


def _managed_preview_outcome_match_score(entry: dict[str, Any], outcome: dict[str, Any]) -> int:
    score = 0
    entry_source = str(entry.get("source_fingerprint") or entry.get("fingerprint") or "").strip()
    outcome_source = str(
        outcome.get("source_fingerprint")
        or outcome.get("source_activation_fingerprint")
        or outcome.get("activation_fingerprint")
        or outcome.get("local_activation_fingerprint")
        or ""
    ).strip()
    if entry_source and outcome_source:
        if entry_source == outcome_source:
            score += 25
        else:
            return -1
    entry_source_ref = _managed_preview_public_ref(entry_source, prefix="activation-ref")
    outcome_source_ref = str(outcome.get("source_activation_ref") or "").strip()
    if entry_source_ref and outcome_source_ref:
        if entry_source_ref == outcome_source_ref:
            score += 24
        else:
            return -1
    explicit_entry_successor = str(entry.get("source_successor_fingerprint") or "").strip()
    entry_successor = explicit_entry_successor or _managed_preview_entry_successor_fingerprint(entry) or ""
    outcome_successor = str(outcome.get("source_successor_fingerprint") or outcome.get("successor_fingerprint") or "").strip()
    if entry_successor and outcome_successor:
        if entry_successor == outcome_successor:
            score += 12
        elif explicit_entry_successor:
            return -1
    entry_successor_ref = _managed_preview_public_ref(entry_successor, prefix="successor-ref")
    outcome_successor_ref = str(outcome.get("source_successor_ref") or "").strip()
    if entry_successor_ref and outcome_successor_ref:
        if entry_successor_ref == outcome_successor_ref:
            score += 11
        else:
            return -1
    entry_family = str(entry.get("local_action_family") or entry.get("lever") or "").strip()
    outcome_family = str(outcome.get("local_action_family") or "").strip()
    if entry_family and outcome_family and entry_family == outcome_family:
        score += 10
    else:
        return -1
    entry_schema = str(entry.get("evidence_schema") or "").strip()
    outcome_schema = str(outcome.get("evidence_schema") or "").strip()
    if entry_schema and outcome_schema:
        if entry_schema == outcome_schema:
            score += 8
        else:
            return -1
    current_status = str(entry.get("current_status") or "").strip()
    outcome_status = str(outcome.get("current_status") or "").strip()
    if current_status and outcome_status and current_status == outcome_status:
        score += 2
    entry_next = str(entry.get("next_action") or "").strip()
    outcome_next = str(outcome.get("next_action") or "").strip()
    if entry_next and outcome_next and entry_next == outcome_next:
        score += 1
    return score


def _managed_preview_outcome_for_entry(
    entry: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(managed_preview_outcomes, dict):
        return None
    outcomes = [row for row in managed_preview_outcomes.get("outcomes") or [] if isinstance(row, dict)]
    best: tuple[int, float, dict[str, Any]] | None = None
    for outcome in outcomes:
        score = _managed_preview_outcome_match_score(entry, outcome)
        if score < 0:
            continue
        age = _to_float(outcome.get("preview_age_hours"))
        rank_key = (score, -age, outcome)
        if best is None or rank_key[:2] > best[:2]:
            best = rank_key
    return best[2] if best else None


def _managed_preview_classification_agrees(classification: str) -> bool:
    return classification in {"review-only", "accepted", "preview-agreed"}


def _managed_preview_public_outcome_status(gate: dict[str, Any]) -> str:
    if gate.get("verified"):
        return "preview-agreed"
    status = str(gate.get("status") or "not-previewed").strip()
    if status in {
        "managed-local-disagreement",
        "failed-closed",
        "unsafe-preview-side-effect",
        "unsafe-request-shape-rollup",
        "crunch-preview-quality-risk",
        "rejected",
    }:
        return "preview-disagreed"
    if status in {"stale-preview", "stale-preview-health"}:
        return "preview-stale"
    if status in {
        "preview-omitted",
        "omitted",
        "needs-local-evidence",
        "request-shape-rollup-too-small",
        "crunch-preview-too-small",
        "crunch-preview-keep-blocked",
    }:
        return "preview-omitted"
    if status in {
        "missing-preview",
        "missing-preview-decision",
        "no-data-preview-health",
        "incomplete-preview-health",
        "privacy-rejected-preview-health",
        "rejected-preview-health",
    }:
        return "preview-missing"
    return "preview-missing"


def _managed_preview_rank_is_usable(gate: dict[str, Any]) -> bool:
    rank = _to_int(gate.get("managed_rank"))
    if rank <= 0:
        return False
    status = str(gate.get("status") or "").strip()
    if status in {
        "missing-preview",
        "missing-preview-decision",
        "no-data-preview-health",
        "stale-preview",
        "stale-preview-health",
        "failed-closed",
        "unsafe-preview-side-effect",
        "privacy-rejected-preview-health",
        "rejected-preview-health",
    }:
        return False
    return not bool(
        gate.get("stale")
        or gate.get("missing_preview_decision")
        or gate.get("failed_closed")
    )


def _queue_managed_priority_rank(entry: dict[str, Any]) -> int:
    gate = entry.get("managed_preview_gate") if isinstance(entry.get("managed_preview_gate"), dict) else {}
    return _to_int(gate.get("managed_rank")) if _managed_preview_rank_is_usable(gate) else 0


def _cache_rollback_guidance_status(guidance: dict[str, Any]) -> str:
    target_file = str(guidance.get("target_local_rule_file") or "").strip()
    target_section = str(guidance.get("target_local_policy_section") or "").strip()
    action_type = str(guidance.get("rollback_action_type") or "").strip()
    disabled_reason = str(guidance.get("disabled_reason") or "").strip()
    if target_file != "cache_rules.yaml":
        return "cache-rollback-guidance-missing-target-rule-file"
    if target_section != "cache.pattern_rules":
        return "cache-rollback-guidance-missing-target-policy-section"
    if not action_type or "cache_replay" not in action_type:
        return "cache-rollback-guidance-missing-action-type"
    if not disabled_reason:
        return "cache-rollback-guidance-missing-disabled-reason"
    return "accepted"


def _cache_rollback_guidance_from_outcome(outcome: dict[str, Any]) -> dict[str, Any] | None:
    if str(outcome.get("local_action_family") or "").strip() != "cache":
        return None
    guidance = outcome.get("cache_rollback_guidance") if isinstance(outcome.get("cache_rollback_guidance"), dict) else {}
    rollback_required = bool(
        outcome.get("rollback_required")
        or guidance.get("rollback_required")
        or str(outcome.get("promotion_readiness") or guidance.get("promotion_readiness") or "").strip()
        == "rollback-required"
        or str(outcome.get("next_action") or guidance.get("recommended_next_action") or "").strip()
        == "rollback-cache-replay-rule"
    )
    if not rollback_required and not guidance:
        return None
    clean = {
        "schema": "tokenclaw.local_activation_cache_rollback_guidance.v1",
        "rollback_required": True,
        "promotion_readiness": sanitize_value(
            outcome.get("promotion_readiness")
            or guidance.get("promotion_readiness")
            or "rollback-required"
        ),
        "rollback_action_type": sanitize_value(
            outcome.get("rollback_action_type") or guidance.get("rollback_action_type")
        ),
        "disabled_reason": sanitize_value(outcome.get("disabled_reason") or guidance.get("disabled_reason")),
        "target_local_rule_file": sanitize_value(
            outcome.get("target_local_rule_file") or guidance.get("target_local_rule_file")
        ),
        "target_local_policy_section": sanitize_value(
            outcome.get("target_local_policy_section") or guidance.get("target_local_policy_section")
        ),
        "recommended_next_action": "rollback-cache-replay-rule",
        "policy_files_written": False,
        "provider_calls_made": False,
        "cache_apply_action_count": 0,
        "cache_entries_written": 0,
        "emits_cache_apply_action": False,
        "privacy": _managed_preview_successor_privacy(
            managed_server_calls_made=bool(outcome.get("managed_server_calls_made"))
        ),
    }
    clean["status"] = _cache_rollback_guidance_status(clean)
    return {
        key: value
        for key, value in clean.items()
        if value not in (None, "", [], 0)
        or key in {"cache_apply_action_count", "cache_entries_written", "emits_cache_apply_action"}
    }


def _managed_preview_successor_gate(
    entry: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
    managed_preview_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = _managed_preview_required_for_successor(entry)
    local_gate = _managed_preview_local_executor_gate(entry)
    family = str(entry.get("local_action_family") or entry.get("lever") or "").strip()
    health_gate = _managed_preview_health_gate(
        managed_preview_health,
        family=family,
        managed_preview_outcomes=managed_preview_outcomes,
    )
    health_blocks_required = bool(required and not health_gate.get("passed"))
    report_summary = (
        managed_preview_outcomes.get("summary")
        if isinstance(managed_preview_outcomes, dict) and isinstance(managed_preview_outcomes.get("summary"), dict)
        else {}
    )
    outcome = _managed_preview_outcome_for_entry(entry, managed_preview_outcomes)
    if outcome is None or health_blocks_required:
        status = str(health_gate.get("status") or "missing-preview") if health_blocks_required else "missing-preview"
        decision = "keep-blocked" if required else "preview-optional"
        if status == "stale-preview-health":
            decision = "review-stale-preview"
        next_action = (
            health_gate.get("next_action")
            if health_blocks_required
            else "refresh-managed-activation-preview"
        )
        reason = (
            health_gate.get("reason")
            if health_blocks_required
            else ("managed-preview-outcome-missing" if required else "managed-preview-optional")
        )
        return {
            "schema": PREVIEW_VERIFIED_SUCCESSOR_GATE_SCHEMA,
            "required": required,
            "status": status,
            "verified": False,
            "decision": decision,
            "next_action": sanitize_value(next_action if required else entry.get("next_action")),
            "local_executor_gate": local_gate,
            "health_gate": health_gate,
            "managed_dependency": "optional",
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": bool(health_gate.get("managed_server_calls_made")),
            "stored_preview_outcome_count": _to_int(report_summary.get("stored_preview_outcome_count")),
            "reason": reason,
            "managed_priority_source": "local-successor-rank",
            "managed_rank_fallback_reason": "managed-preview-health-blocked" if health_blocks_required else "managed-preview-outcome-missing",
            "privacy": _managed_preview_successor_privacy(
                managed_server_calls_made=bool(health_gate.get("managed_server_calls_made"))
            ),
        }

    classification = str(outcome.get("classification") or "unknown").strip() or "unknown"
    request_shape_outcome_class = (
        _request_shape_rollup_outcome_class(outcome)
        if _is_request_shape_rollup_successor(entry, outcome)
        else ""
    )
    crunch_preview_decision = (
        _crunch_preview_decision(outcome)
        if _is_repeated_context_crunch_successor(entry, outcome)
        else ""
    )
    stale = bool(outcome.get("stale"))
    missing = bool(outcome.get("missing_preview_decision"))
    failed_closed = bool(outcome.get("failed_closed"))
    disagreement = bool(outcome.get("disagrees_with_local_evidence"))
    preview_policy_write = bool(
        outcome.get("managed_preview_policy_files_written")
        or outcome.get("policy_files_written")
    )
    preview_provider_call = bool(
        outcome.get("managed_preview_provider_calls_made")
        or outcome.get("provider_calls_made")
    )
    cache_rollback_guidance = _cache_rollback_guidance_from_outcome(outcome)
    cache_rollback_status = (
        str(cache_rollback_guidance.get("status") or "")
        if isinstance(cache_rollback_guidance, dict)
        else ""
    )
    cache_rollback_accepted = cache_rollback_status == "accepted"
    cache_rollback_invalid = bool(cache_rollback_guidance) and not cache_rollback_accepted
    verified = bool(
        _managed_preview_classification_agrees(classification)
        and not stale
        and not missing
        and not failed_closed
        and not disagreement
        and not preview_policy_write
        and not preview_provider_call
        and not cache_rollback_invalid
        and (health_gate.get("passed") or not required)
    )
    if request_shape_outcome_class:
        verified = bool(
            request_shape_outcome_class == "review-ready"
            and not preview_policy_write
            and not preview_provider_call
            and not failed_closed
            and not disagreement
        )
    if crunch_preview_decision:
        verified = bool(
            crunch_preview_decision in {"review-ready", "keep-staged"}
            and not stale
            and not missing
            and not preview_policy_write
            and not preview_provider_call
            and not failed_closed
            and not disagreement
        )
    action_status = _successor_action_status(entry)
    if request_shape_outcome_class:
        status = _request_shape_rollup_preview_status(request_shape_outcome_class)
        decision = _request_shape_rollup_preview_decision(request_shape_outcome_class)
        next_action = _request_shape_rollup_next_action(request_shape_outcome_class, outcome, entry)
        reason = f"request-shape-rollup-{request_shape_outcome_class}"
    elif crunch_preview_decision:
        status = _crunch_preview_status(crunch_preview_decision)
        decision = _crunch_preview_successor_decision(crunch_preview_decision)
        next_action = _crunch_preview_next_action(crunch_preview_decision, outcome, entry)
        reason = f"crunch-preview-{crunch_preview_decision}"
    elif verified:
        if cache_rollback_accepted:
            decision = "rollback-required"
        elif action_status in {"keep-current-rule", "suppress-duplicate"}:
            decision = action_status
        elif local_gate["passed"] and str(entry.get("issue_worthy_status") or "") == "ready":
            decision = "ready"
        elif action_status == "keep-blocked":
            decision = "keep-blocked"
        elif local_gate["passed"]:
            decision = "ready"
        else:
            decision = "review-only"
        status = "preview-verified"
        reason = "local-managed-cache-rollback-guidance" if cache_rollback_accepted else "local-managed-preview-agree"
        next_action = (
            "rollback-cache-replay-rule"
            if cache_rollback_accepted
            else outcome.get("next_action") or entry.get("next_action")
        )
    else:
        decision = "keep-blocked" if required else "review"
        if failed_closed:
            status = "failed-closed"
        elif stale:
            status = "stale-preview"
        elif missing:
            status = "missing-preview-decision"
        elif disagreement:
            status = "managed-local-disagreement"
        elif preview_policy_write or preview_provider_call:
            status = "unsafe-preview-side-effect"
        elif cache_rollback_invalid:
            status = cache_rollback_status
        elif classification in {"preview-omitted", "omitted", "needs-local-evidence"}:
            status = "preview-omitted"
        else:
            status = classification
        decision = "review-stale-preview" if stale else decision
        reason = status
        next_action = "review-managed-cache-rollback-guidance" if cache_rollback_invalid else outcome.get("next_action") or "refresh-managed-activation-preview"
    gate = {
        "schema": PREVIEW_VERIFIED_SUCCESSOR_GATE_SCHEMA,
        "required": required,
        "status": status,
        "verified": verified,
        "decision": decision,
        "next_action": sanitize_value(next_action),
        "local_executor_gate": local_gate,
        "health_gate": health_gate,
        "managed_dependency": "optional",
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": bool(outcome.get("managed_server_calls_made")),
        "stored_preview_outcome_count": _to_int(report_summary.get("stored_preview_outcome_count")),
        "outcome_fingerprint": sanitize_value(outcome.get("outcome_fingerprint")),
        "handoff_ref": sanitize_value(outcome.get("handoff_ref")),
        "preview_ref": sanitize_value(outcome.get("preview_ref")),
        "source_executor_ref": sanitize_value(outcome.get("source_executor_ref")),
        "source_activation_ref": sanitize_value(outcome.get("source_activation_ref")),
        "source_successor_ref": sanitize_value(outcome.get("source_successor_ref")),
        "preview_age_hours": outcome.get("preview_age_hours"),
        "stale_after_hours": outcome.get("stale_after_hours"),
        "classification": classification,
        "managed_preview_classification": sanitize_value(outcome.get("managed_preview_classification")),
        "cohort_class": sanitize_value(outcome.get("cohort_class")),
        "rollup_outcome_status": sanitize_value(outcome.get("rollup_outcome_status")),
        "request_shape_rollup_outcome_class": sanitize_value(request_shape_outcome_class),
        "crunch_preview_decision": sanitize_value(crunch_preview_decision),
        "crunch_preview_confidence": sanitize_value(outcome.get("crunch_preview_confidence")),
        "quality_risk_reason_codes": sanitize_value(outcome.get("quality_risk_reason_codes") or []),
        "projected_saved_tokens": sanitize_value(outcome.get("projected_saved_tokens")),
        "projected_saved_usd": sanitize_value(
            outcome.get("projected_saved_usd") or outcome.get("projected_savings_usd")
        ),
        "projected_savings_usd": sanitize_value(
            outcome.get("projected_savings_usd") or outcome.get("projected_saved_usd")
        ),
        "observed_saved_tokens": sanitize_value(outcome.get("observed_saved_tokens")),
        "observed_saved_usd": sanitize_value(outcome.get("observed_saved_usd")),
        "observed_crunch_ratio": sanitize_value(outcome.get("observed_crunch_ratio")),
        "successor_action_fingerprint": sanitize_value(outcome.get("successor_action_fingerprint")),
        "successor_decision_fingerprint": sanitize_value(outcome.get("successor_decision_fingerprint")),
        "managed_rank": _to_int(outcome.get("managed_rank")),
        "managed_recommended_next_action": sanitize_value(outcome.get("managed_recommended_next_action")),
        "managed_expected_savings_path": sanitize_value(outcome.get("managed_expected_savings_path")),
        "managed_preview_action_ref": sanitize_value(outcome.get("managed_preview_action_ref")),
        "managed_priority_source": sanitize_value(outcome.get("managed_priority_source") or "local-successor-rank"),
        "managed_rank_fallback_reason": sanitize_value(outcome.get("managed_rank_fallback_reason")),
        "preview_outcome_status": _managed_preview_public_outcome_status(
            {"verified": verified, "status": status}
        ),
        "reason": reason,
        "reason_codes": sanitize_value([str(item) for item in outcome.get("reason_codes") or [] if str(item or "").strip()]),
        "omitted_reason": sanitize_value(outcome.get("omitted_reason")),
        "no_op_reason": sanitize_value(outcome.get("no_op_reason")),
        "stale": stale,
        "missing_preview_decision": missing,
        "failed_closed": failed_closed,
        "disagrees_with_local_evidence": disagreement,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "privacy": _managed_preview_successor_privacy(
            managed_server_calls_made=bool(outcome.get("managed_server_calls_made"))
        ),
    }
    if cache_rollback_guidance is not None:
        gate["cache_rollback_guidance"] = sanitize_value(cache_rollback_guidance)
        gate["cache_rollback_guidance_status"] = sanitize_value(cache_rollback_status)
    return gate


def _local_activation_successor_action(entry: dict[str, Any]) -> dict[str, Any]:
    fingerprint = sanitize_value(entry.get("fingerprint") or "")
    material = {
        "fingerprint": fingerprint,
        "next_action": sanitize_value(entry.get("next_action") or "inspect-local-evidence"),
        "local_action_family": sanitize_value(entry.get("local_action_family") or entry.get("lever") or "unknown"),
    }
    action_fingerprint = public_id(json.dumps(material, sort_keys=True), prefix="successor")
    action = {
        "schema": LOCAL_ACTIVATION_SUCCESSOR_ACTION_SCHEMA,
        "rank": 0,
        "source_queue_rank": _to_int(entry.get("rank")),
        "source_ledger_rank": _to_int(entry.get("ledger_rank")),
        "fingerprint": action_fingerprint,
        "source_fingerprint": fingerprint,
        "lever": sanitize_value(entry.get("lever") or "unknown"),
        "local_action_family": sanitize_value(entry.get("local_action_family") or entry.get("lever") or "unknown"),
        "successor_status": _successor_action_status(entry),
        "current_status": sanitize_value(entry.get("current_status") or "unknown"),
        "state": sanitize_value(entry.get("state") or "unknown"),
        "recommended_next_action": sanitize_value(entry.get("next_action") or "inspect-local-evidence"),
        "unblock_reason": sanitize_value(entry.get("unblock_reason")),
        "blocker_codes": sanitize_value([str(item) for item in entry.get("blocker_codes") or [] if str(item or "").strip()]),
        "target_local_rule_file": sanitize_value(entry.get("target_local_rule_file")),
        "target_local_policy_section": sanitize_value(entry.get("target_local_policy_section")),
        "acceptance_metric": _successor_action_acceptance_metric(entry),
        "expected_savings_path": sanitize_value(entry.get("expected_savings_path")),
        "sample_count": _to_int(entry.get("sample_count")),
        "realized_savings_usd": round(_to_float(entry.get("realized_savings_usd")), 8),
        "projected_savings_usd": round(_to_float(entry.get("projected_savings_usd")), 8),
        "savings_per_1000_calls_usd": round(_to_float(entry.get("savings_per_1000_calls_usd")), 8),
        "freshness_adjusted_savings_per_1000_calls_usd": round(
            _to_float(entry.get("freshness_adjusted_savings_per_1000_calls_usd")),
            8,
        ),
        "freshness_state": sanitize_value(entry.get("freshness_state")),
        "blocking_reason": sanitize_value(entry.get("blocking_reason")),
        "rank_basis": sanitize_value(entry.get("rank_basis")) if isinstance(entry.get("rank_basis"), dict) else None,
        "issue_worthy_status": sanitize_value(entry.get("issue_worthy_status") or "review"),
        "duplicate_suppression_status": sanitize_value(entry.get("duplicate_suppression_status")),
        "duplicate_suppression_reason": sanitize_value(entry.get("duplicate_suppression_reason")),
        "privacy": _successor_action_privacy(),
    }
    preview_gate = entry.get("managed_preview_gate") if isinstance(entry.get("managed_preview_gate"), dict) else None
    if preview_gate is not None:
        action["managed_preview_gate"] = sanitize_value(preview_gate)
        action["preview_verified"] = bool(preview_gate.get("verified"))
        action["preview_verification_status"] = sanitize_value(preview_gate.get("status"))
        action["preview_verification_decision"] = sanitize_value(preview_gate.get("decision"))
        if preview_gate.get("request_shape_rollup_outcome_class"):
            action["successor_status"] = sanitize_value(preview_gate.get("decision") or action["successor_status"])
            action["recommended_next_action"] = sanitize_value(preview_gate.get("next_action") or action["recommended_next_action"])
        elif preview_gate.get("crunch_preview_decision"):
            action["successor_status"] = sanitize_value(preview_gate.get("decision") or action["successor_status"])
            action["recommended_next_action"] = sanitize_value(preview_gate.get("next_action") or action["recommended_next_action"])
        elif preview_gate.get("decision") == "review-stale-preview":
            action["successor_status"] = "review-stale-preview"
            action["recommended_next_action"] = sanitize_value(
                preview_gate.get("next_action") or "refresh-managed-activation-preview"
            )
        if preview_gate.get("required") and not preview_gate.get("verified"):
            if preview_gate.get("decision") != "review-stale-preview":
                action["successor_status"] = "keep-blocked"
                action["recommended_next_action"] = sanitize_value(
                    preview_gate.get("next_action") or "refresh-managed-activation-preview"
                )
        elif preview_gate.get("verified") and preview_gate.get("decision") == "ready":
            action["successor_status"] = "ready"
            action["recommended_next_action"] = sanitize_value(preview_gate.get("next_action") or action["recommended_next_action"])
        elif preview_gate.get("verified") and preview_gate.get("decision") == "rollback-required":
            rollback_guidance = (
                preview_gate.get("cache_rollback_guidance")
                if isinstance(preview_gate.get("cache_rollback_guidance"), dict)
                else {}
            )
            action["successor_status"] = "rollback-required"
            action["recommended_next_action"] = "rollback-cache-replay-rule"
            action["rollback_required"] = True
            action["promotion_readiness"] = "rollback-required"
            action["promotion_recommendation"] = "rollback-required"
            action["rollback_action_type"] = sanitize_value(rollback_guidance.get("rollback_action_type"))
            action["disabled_reason"] = sanitize_value(rollback_guidance.get("disabled_reason"))
            action["target_local_rule_file"] = sanitize_value(
                rollback_guidance.get("target_local_rule_file") or action.get("target_local_rule_file")
            )
            action["target_local_policy_section"] = sanitize_value(
                rollback_guidance.get("target_local_policy_section") or action.get("target_local_policy_section")
            )
            action["cache_rollback_guidance"] = sanitize_value(rollback_guidance)
            action["cache_apply_action_count"] = 0
            action["cache_entries_written"] = 0
            action["emits_cache_apply_action"] = False
            action["policy_files_written"] = False
        elif preview_gate.get("verified") and preview_gate.get("decision") == "keep-blocked":
            action["successor_status"] = "keep-blocked"
        elif preview_gate.get("verified") and preview_gate.get("decision") in {"keep-current-rule", "suppress-duplicate"}:
            action["successor_status"] = sanitize_value(preview_gate.get("decision"))
            action["recommended_next_action"] = sanitize_value(preview_gate.get("next_action") or action["recommended_next_action"])
        if preview_gate.get("request_shape_rollup_outcome_class"):
            action["request_shape_rollup_outcome_class"] = sanitize_value(preview_gate.get("request_shape_rollup_outcome_class"))
            action["cohort_class"] = sanitize_value(preview_gate.get("cohort_class"))
            action["rollup_outcome_status"] = sanitize_value(preview_gate.get("rollup_outcome_status"))
        if preview_gate.get("crunch_preview_decision"):
            action["crunch_preview_decision"] = sanitize_value(preview_gate.get("crunch_preview_decision"))
            action["quality_risk_reason_codes"] = sanitize_value(preview_gate.get("quality_risk_reason_codes") or [])
            action["cohort_class"] = sanitize_value(preview_gate.get("cohort_class"))
            action["successor_action_fingerprint"] = sanitize_value(preview_gate.get("successor_action_fingerprint"))
            action["successor_decision_fingerprint"] = sanitize_value(preview_gate.get("successor_decision_fingerprint"))
            if preview_gate.get("crunch_preview_confidence") is not None:
                action["crunch_preview_confidence"] = round(_to_float(preview_gate.get("crunch_preview_confidence")), 8)
            if preview_gate.get("projected_saved_tokens") is not None:
                action["projected_saved_tokens"] = _to_int(preview_gate.get("projected_saved_tokens"))
            if preview_gate.get("projected_saved_usd") is not None:
                action["projected_saved_usd"] = round(_to_float(preview_gate.get("projected_saved_usd")), 8)
            if preview_gate.get("projected_savings_usd") is not None:
                action["projected_savings_usd"] = round(_to_float(preview_gate.get("projected_savings_usd")), 8)
            if preview_gate.get("observed_saved_tokens") is not None:
                action["observed_saved_tokens"] = _to_int(preview_gate.get("observed_saved_tokens"))
            if preview_gate.get("observed_saved_usd") is not None:
                action["observed_saved_usd"] = round(_to_float(preview_gate.get("observed_saved_usd")), 8)
            if preview_gate.get("observed_crunch_ratio") is not None:
                action["observed_crunch_ratio"] = round(_to_float(preview_gate.get("observed_crunch_ratio")), 8)
        if preview_gate.get("managed_rank") is not None:
            action["managed_rank"] = _to_int(preview_gate.get("managed_rank"))
            action["managed_priority_source"] = (
                "ranked-managed-preview"
                if _managed_preview_rank_is_usable(preview_gate)
                else "local-successor-rank"
            )
        if preview_gate.get("managed_recommended_next_action"):
            action["managed_recommended_next_action"] = sanitize_value(preview_gate.get("managed_recommended_next_action"))
        if preview_gate.get("managed_expected_savings_path"):
            action["managed_expected_savings_path"] = sanitize_value(preview_gate.get("managed_expected_savings_path"))
        if preview_gate.get("managed_preview_action_ref"):
            action["managed_preview_action_ref"] = sanitize_value(preview_gate.get("managed_preview_action_ref"))
        if preview_gate.get("managed_rank_fallback_reason"):
            action["managed_rank_fallback_reason"] = sanitize_value(preview_gate.get("managed_rank_fallback_reason"))
    for key in (
        "evidence_schema",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "provider_family",
        "requested_model",
        "candidate_target_model",
        "required_local_executor",
        "diagnostic_class",
        "diagnostic_reason",
        "diagnostic_fingerprint",
        "diagnostic_evidence_status",
        "review_status",
        "keep_blocked_reason",
        "next_state",
        "next_state_reason",
        "promotion_readiness",
        "promotion_recommendation",
        "rollback_action_type",
        "disabled_reason",
        "request_shape_rollup_outcome_class",
        "crunch_preview_decision",
        "cohort_class",
        "rollup_outcome_status",
        "successor_action_fingerprint",
        "successor_decision_fingerprint",
        "post_rollback_successor_decision",
        "post_rollback_next_action",
        "post_rollback_reason",
        "managed_priority_source",
        "managed_rank_fallback_reason",
        "managed_recommended_next_action",
        "managed_expected_savings_path",
        "managed_preview_action_ref",
    ):
        if entry.get(key) is not None:
            action[key] = sanitize_value(entry.get(key))
    for key in (
        "emits_cache_apply_action",
        "policy_files_written",
        "tool_cache_replay_enabled",
        "streaming_replay_enabled",
        "live_repeat_confirmed",
        "observed_hit_proof",
        "review_only",
        "promotion_allowed",
        "stage_allowed",
        "missing_applied_coverage",
        "missing_holdout_coverage",
        "measured_full_rollout_activation",
        "durable_action_ledger_entry",
        "durable_outcome_ledger_entry",
        "terminal_successor_state",
        "rollback_required",
        "rollback_applied",
        "stale_no_traffic_retirement",
    ):
        if entry.get(key) is not None:
            action[key] = bool(entry.get(key))
    for key in (
        "applied_count",
        "matched_count",
        "holdout_count",
        "skipped_count",
        "fallback_count",
        "retry_count",
        "rollback_count",
        "safety_stop_count",
        "affected_rows",
        "cache_apply_action_count",
        "cache_entries_written",
        "warmup_miss_count",
        "observed_hits",
        "exact_hit_count",
        "tools_present_rows",
        "tools_present_replay_evidence_rows",
        "generic_tools_present_blocker_reduced_rows",
        "unsafe_tool_call_blocker_rows",
        "missing_dependency_evidence_rows",
        "stable_dependency_evidence_rows",
        "stale_dependency_evidence_rows",
        "unsafe_dependency_evidence_rows",
        "unknown_dependency_evidence_rows",
        "observed_saved_tokens",
        "projected_saved_tokens",
        "managed_rank",
    ):
        if entry.get(key) is not None:
            action[key] = _to_int(entry.get(key))
    for key in (
        "error_rate_delta",
        "retry_rate_delta",
        "fallback_rate_delta",
        "canary_fraction",
        "holdout_fraction",
        "savings_per_1000_calls_usd",
        "freshness_adjusted_savings_per_1000_calls_usd",
        "freshness_multiplier",
        "crunch_preview_confidence",
        "observed_crunch_ratio",
        "projected_saved_usd",
        "observed_saved_usd",
    ):
        if entry.get(key) is not None:
            action[key] = round(_to_float(entry.get(key)), 8)
    for key in (
        "coverage",
        "recovery_plan",
        "activation_feedback_diagnostic_classification",
        "post_rollback_observation",
    ):
        if isinstance(entry.get(key), dict):
            action[key] = sanitize_value(entry.get(key))
    for key in (
        "full_rollout_outcome",
        "full_rollout_outcome_next_action",
        "full_rollout_successor_decision",
        "full_rollout_successor_next_action",
        "full_rollout_successor_no_op_reason",
        "dependency_evidence_class",
        "dependency_evidence_decision",
        "dependency_evidence_reason",
        "dependency_evidence_status",
        "evidence_state",
    ):
        if entry.get(key) is not None:
            action[key] = sanitize_value(entry.get(key))
    for key in (
        "full_rollout_activation_outcome",
        "keep_active_regression_gate",
        "dependency_evidence_review",
        "cache_rollback_guidance",
    ):
        if isinstance(entry.get(key), dict):
            action[key] = sanitize_value(entry.get(key))
    for key in (
        "blocker_breakdown",
        "dependency_evidence_decision_breakdown",
        "evidence_state_breakdown",
        "next_action_breakdown",
        "needed_resolution",
        "quality_risk_reason_codes",
    ):
        if isinstance(entry.get(key), list):
            action[key] = sanitize_value(entry.get(key))
    return {
        key: value
        for key, value in action.items()
        if value not in (None, "", [], 0) or key in {
            "rank",
            "source_queue_rank",
            "source_ledger_rank",
            "sample_count",
            "realized_savings_usd",
            "projected_savings_usd",
            "preview_verified",
            "preview_verification_status",
            "preview_verification_decision",
            "savings_per_1000_calls_usd",
            "freshness_adjusted_savings_per_1000_calls_usd",
            "cache_apply_action_count",
            "cache_entries_written",
            "tool_cache_replay_enabled",
            "streaming_replay_enabled",
            "emits_cache_apply_action",
            "live_repeat_confirmed",
            "observed_hit_proof",
            "observed_hits",
            "exact_hit_count",
            "missing_dependency_evidence_rows",
            "stable_dependency_evidence_rows",
            "stale_dependency_evidence_rows",
            "unsafe_dependency_evidence_rows",
            "unknown_dependency_evidence_rows",
            "rollback_required",
            "policy_files_written",
            "crunch_preview_confidence",
            "projected_saved_tokens",
            "projected_saved_usd",
            "observed_saved_tokens",
            "observed_saved_usd",
            "observed_crunch_ratio",
            "managed_rank",
        }
    }


def _successor_decision_issue_status(action: dict[str, Any]) -> str:
    decision = str(action.get("successor_status") or "").strip()
    if decision == "ready":
        return "ready"
    if decision == "rollback-required":
        return "blocked"
    if decision == "reobserve-after-rollback":
        return "review"
    if decision == "keep-blocked-narrow":
        return "blocked"
    if decision in _TERMINAL_ACTIVATION_SUCCESSOR_STATES:
        return "suppressed"
    if decision in {"retire-staged-no-repeat", "suppressed-closed-successor"}:
        return "suppressed"
    if decision == "retired-stale-no-traffic":
        return "suppressed"
    if decision in {"keep-current-rule", "suppress-duplicate"}:
        return "suppressed"
    if decision in {"keep-blocked", "review-stale-preview"}:
        return "blocked"
    return "review"


def _local_activation_successor_decision(action: dict[str, Any]) -> dict[str, Any]:
    decision = str(action.get("successor_status") or "review").strip() or "review"
    source_fingerprint = sanitize_value(action.get("source_fingerprint") or "")
    material = {
        "source_fingerprint": source_fingerprint,
        "local_action_family": sanitize_value(action.get("local_action_family") or "unknown"),
        "decision": decision,
        "recommended_next_action": sanitize_value(action.get("recommended_next_action") or "inspect-local-evidence"),
    }
    gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
    row = {
        "schema": LOCAL_ACTIVATION_SUCCESSOR_DECISION_SCHEMA,
        "fingerprint": public_id(json.dumps(material, sort_keys=True), prefix="successor-decision"),
        "source_fingerprint": source_fingerprint,
        "successor_action_fingerprint": sanitize_value(action.get("fingerprint")),
        "local_action_family": sanitize_value(action.get("local_action_family") or "unknown"),
        "decision": sanitize_value(decision),
        "recommended_next_action": sanitize_value(action.get("recommended_next_action") or "inspect-local-evidence"),
        "issue_worthy_status": _successor_decision_issue_status(action),
        "preview_verified": bool(action.get("preview_verified")),
        "preview_verification_status": sanitize_value(action.get("preview_verification_status")),
        "preview_verification_decision": sanitize_value(action.get("preview_verification_decision")),
        "preview_agreement_status": "agreed" if gate.get("verified") else sanitize_value(gate.get("status") or "not-previewed"),
        "preview_outcome_status": _managed_preview_public_outcome_status(gate),
        "preview_omitted_reason": sanitize_value(gate.get("omitted_reason")) if gate else None,
        "preview_no_op_reason": sanitize_value(gate.get("no_op_reason")) if gate else None,
        "request_shape_rollup_outcome_class": sanitize_value(gate.get("request_shape_rollup_outcome_class")) if gate else None,
        "crunch_preview_decision": sanitize_value(gate.get("crunch_preview_decision")) if gate else None,
        "crunch_preview_confidence": round(_to_float(gate.get("crunch_preview_confidence")), 8)
        if gate and gate.get("crunch_preview_confidence") is not None
        else None,
        "quality_risk_reason_codes": sanitize_value(gate.get("quality_risk_reason_codes") or []) if gate else None,
        "cohort_class": sanitize_value(gate.get("cohort_class")) if gate else None,
        "rollup_outcome_status": sanitize_value(gate.get("rollup_outcome_status")) if gate else None,
        "projected_saved_tokens": _to_int(gate.get("projected_saved_tokens"))
        if gate and gate.get("projected_saved_tokens") is not None
        else None,
        "projected_saved_usd": round(_to_float(gate.get("projected_saved_usd")), 8)
        if gate and gate.get("projected_saved_usd") is not None
        else None,
        "projected_savings_usd": round(_to_float(gate.get("projected_savings_usd")), 8)
        if gate and gate.get("projected_savings_usd") is not None
        else None,
        "observed_saved_tokens": _to_int(gate.get("observed_saved_tokens"))
        if gate and gate.get("observed_saved_tokens") is not None
        else None,
        "observed_saved_usd": round(_to_float(gate.get("observed_saved_usd")), 8)
        if gate and gate.get("observed_saved_usd") is not None
        else None,
        "observed_crunch_ratio": round(_to_float(gate.get("observed_crunch_ratio")), 8)
        if gate and gate.get("observed_crunch_ratio") is not None
        else None,
        "successor_decision_fingerprint": sanitize_value(gate.get("successor_decision_fingerprint")) if gate else None,
        "managed_rank": _to_int(gate.get("managed_rank")) if gate and gate.get("managed_rank") is not None else None,
        "managed_recommended_next_action": sanitize_value(gate.get("managed_recommended_next_action")) if gate else None,
        "managed_expected_savings_path": sanitize_value(gate.get("managed_expected_savings_path")) if gate else None,
        "managed_preview_action_ref": sanitize_value(gate.get("managed_preview_action_ref")) if gate else None,
        "managed_priority_source": (
            "ranked-managed-preview" if gate and _managed_preview_rank_is_usable(gate) else "local-successor-rank"
        ),
        "managed_rank_fallback_reason": sanitize_value(gate.get("managed_rank_fallback_reason")) if gate else None,
        "top_preview_omission_reason": sanitize_value((gate.get("health_gate") or {}).get("top_omission_reason")) if isinstance(gate.get("health_gate"), dict) else None,
        "target_local_rule_file": sanitize_value(action.get("target_local_rule_file")),
        "target_local_policy_section": sanitize_value(action.get("target_local_policy_section")),
        "promotion_readiness": sanitize_value(action.get("promotion_readiness")),
        "promotion_recommendation": sanitize_value(action.get("promotion_recommendation")),
        "rollback_required": bool(action.get("rollback_required")),
        "rollback_action_type": sanitize_value(action.get("rollback_action_type")),
        "disabled_reason": sanitize_value(action.get("disabled_reason")),
        "post_rollback_successor_decision": sanitize_value(action.get("post_rollback_successor_decision")),
        "post_rollback_next_action": sanitize_value(action.get("post_rollback_next_action")),
        "post_rollback_reason": sanitize_value(action.get("post_rollback_reason")),
        "post_rollback_observation": sanitize_value(action.get("post_rollback_observation"))
        if isinstance(action.get("post_rollback_observation"), dict)
        else None,
        "cache_apply_action_count": _to_int(action.get("cache_apply_action_count")),
        "cache_entries_written": _to_int(action.get("cache_entries_written")),
        "emits_cache_apply_action": bool(action.get("emits_cache_apply_action")),
        "policy_files_written": bool(action.get("policy_files_written")),
        "privacy": _successor_action_privacy(),
    }
    return {
        key: value
        for key, value in sanitize_value(row).items()
        if value not in (None, "", [], 0)
        or key in {
            "preview_verified",
            "rollback_required",
            "cache_apply_action_count",
            "cache_entries_written",
            "emits_cache_apply_action",
            "policy_files_written",
            "crunch_preview_confidence",
            "projected_saved_tokens",
            "projected_saved_usd",
            "projected_savings_usd",
            "observed_saved_tokens",
            "observed_saved_usd",
            "observed_crunch_ratio",
            "managed_rank",
        }
    }


def build_local_activation_successor_decisions(queue: dict[str, Any]) -> list[dict[str, Any]]:
    actions = queue.get("successor_actions")
    if not isinstance(actions, list):
        actions = build_local_activation_successor_actions(queue)
    rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for action in actions:
        if not isinstance(action, dict):
            continue
        row = _local_activation_successor_decision(action)
        source = str(row.get("source_fingerprint") or row.get("fingerprint") or "")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        rows.append(row)
    return rows


def _activation_successor_issue_labels(action: dict[str, Any], status_label: str) -> list[str]:
    priority = "priority:p1" if status_label == "status:ready" else "priority:p2"
    labels = _default_issue_labels(priority)
    labels = [label for label in labels if not label.startswith("status:")]
    labels.append(status_label)
    family = str(action.get("local_action_family") or "").strip().replace("_", "-")
    if family in {"routing", "cache", "crunch"}:
        labels.append(family)
    labels.append("privacy")
    return list(dict.fromkeys(labels))


def _activation_successor_title_token(action: dict[str, Any], decision: dict[str, Any]) -> str:
    for value in (action.get("source_fingerprint"), decision.get("source_fingerprint")):
        text = str(value or "").strip()
        if not text:
            continue
        suffix = re.sub(r"[^a-zA-Z0-9]+", "", text.rsplit(":", 1)[-1])
        if suffix:
            return f"evidence {suffix[:12]}"
    return "current successor"


def _activation_successor_blocker(action: dict[str, Any], decision: dict[str, Any]) -> str:
    for value in (
        decision.get("preview_omitted_reason"),
        decision.get("top_preview_omission_reason"),
        decision.get("preview_no_op_reason"),
        action.get("unblock_reason"),
    ):
        text = str(value or "").strip()
        if text:
            return sanitize_value(text)
    blockers = [str(item) for item in action.get("blocker_codes") or [] if str(item or "").strip()]
    if blockers:
        return sanitize_value(blockers[0])
    status = str(decision.get("preview_agreement_status") or action.get("successor_status") or "").strip()
    return sanitize_value(status or "successor-blocker")


def _activation_successor_issue_title(
    action: dict[str, Any],
    decision: dict[str, Any],
    *,
    status_label: str,
) -> str:
    family = str(action.get("local_action_family") or decision.get("local_action_family") or "optimization").strip()
    family = family.replace("_", "-") or "optimization"
    next_action = str(decision.get("recommended_next_action") or action.get("recommended_next_action") or "").strip()
    next_action = next_action.replace("_", "-") or "inspect-local-evidence"
    token = _activation_successor_title_token(action, decision)
    if status_label == "status:blocked":
        blocker = _activation_successor_blocker(action, decision).replace("_", "-")
        return f"Keep {family} activation successor blocked on {blocker} ({token})"
    return f"Advance preview-verified {family} activation successor for {next_action} ({token})"


def _activation_successor_privacy_evidence(action: dict[str, Any], decision: dict[str, Any]) -> str:
    privacy = action.get("privacy") if isinstance(action.get("privacy"), dict) else {}
    if not privacy and isinstance(decision.get("privacy"), dict):
        privacy = decision["privacy"]
    keys = (
        "metadata_only",
        "aggregate_only",
        "raw_prompts_included",
        "provider_bodies_included",
        "absolute_paths_included",
        "request_ids_included",
        "session_ids_included",
        "cache_keys_included",
        "individual_candidate_ids_included",
        "managed_server_calls_made",
        "provider_calls_made",
        "policy_file_contents_included",
    )
    parts = [f"{key}={bool(privacy.get(key))}" for key in keys if key in privacy]
    return "Privacy flags: " + (", ".join(parts) if parts else "metadata_only=True, aggregate_only=True")


def _activation_successor_has_concrete_blocker(action: dict[str, Any], decision: dict[str, Any]) -> bool:
    if _activation_successor_blocker(action, decision) not in {"", "successor-blocker"}:
        return True
    gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
    if str(gate.get("reason") or "").strip():
        return True
    if isinstance(action.get("unblock_criteria"), dict):
        return True
    return bool(action.get("needed_resolution") or action.get("blocker_codes"))


def _is_openai_semantic_regression_routing_successor(action: dict[str, Any]) -> bool:
    family = str(action.get("local_action_family") or action.get("lever") or "").strip()
    blockers = {str(item) for item in action.get("blocker_codes") or [] if str(item or "").strip()}
    evidence_schema = str(action.get("evidence_schema") or "")
    source_surface = str(action.get("source_surface") or "")
    provider_family = str(action.get("provider_family") or "")
    return bool(
        family == "routing"
        and "semantic-quality-regression-observed" in blockers
        and (
            "openai_routing" in evidence_schema
            or source_surface.startswith("openai")
            or provider_family == "openai"
            or str(action.get("target_local_rule_file") or "") == "routing_rules.yaml"
        )
    )


def _activation_successor_preview_reason(action: dict[str, Any], decision: dict[str, Any]) -> str:
    gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
    for value in (
        decision.get("preview_omitted_reason"),
        decision.get("preview_no_op_reason"),
        decision.get("top_preview_omission_reason"),
        gate.get("omitted_reason"),
        gate.get("no_op_reason"),
        gate.get("reason"),
        decision.get("preview_agreement_status"),
        decision.get("preview_verification_status"),
    ):
        text = str(value or "").strip()
        if text and text != "agreed":
            return sanitize_value(text)
    return "local-managed-preview-agree" if bool(decision.get("preview_verified")) else "preview-not-verified"


def _openai_routing_recovery_issue_title(
    action: dict[str, Any],
    decision: dict[str, Any],
    *,
    status_label: str,
) -> str:
    token = _activation_successor_title_token(action, decision)
    if status_label == "status:ready":
        return f"Review OpenAI routing recovery canary for semantic regression ({token})"
    blocker = _activation_successor_preview_reason(action, decision).replace("_", "-")
    return f"Keep OpenAI routing recovery blocked on {blocker} ({token})"


def _count_text(value: Any) -> str:
    count = _to_int(value)
    return str(count) if count else "0"


def _openai_routing_recovery_evidence(
    action: dict[str, Any],
    decision: dict[str, Any],
    *,
    status_label: str,
) -> list[str]:
    recovery_plan = action.get("recovery_plan") if isinstance(action.get("recovery_plan"), dict) else {}
    coverage = (
        recovery_plan.get("coverage")
        if isinstance(recovery_plan.get("coverage"), dict)
        else action.get("coverage")
        if isinstance(action.get("coverage"), dict)
        else {}
    )
    gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
    health_gate = gate.get("health_gate") if isinstance(gate.get("health_gate"), dict) else {}
    preview_reason = _activation_successor_preview_reason(action, decision)
    canary_fraction = action.get("canary_fraction") or recovery_plan.get("canary_fraction")
    holdout_fraction = action.get("holdout_fraction") or recovery_plan.get("holdout_fraction")
    blocker_status = recovery_plan.get("blocker_status") or action.get("current_status") or action.get("state")
    source_fingerprint = str(decision.get("source_fingerprint") or action.get("source_fingerprint") or "").strip()
    action_fingerprint = str(decision.get("successor_action_fingerprint") or action.get("fingerprint") or "").strip()
    return [
        f"Source metadata: {LOCAL_ACTIVATION_SUCCESSOR_DECISION_SCHEMA}",
        f"Fingerprint: {source_fingerprint}",
        f"Successor action fingerprint: {action_fingerprint}",
        "Provider scope: openai",
        "Local action family: routing",
        "Semantic regression blocker: semantic-quality-regression-observed",
        f"Blocker status: {blocker_status}",
        f"Applied coverage count: {_count_text(coverage.get('applied_count') or action.get('applied_count'))}",
        f"Holdout coverage count: {_count_text(coverage.get('holdout_count') or action.get('holdout_count'))}",
        f"Matched/sample count: {_count_text(action.get('matched_count') or action.get('sample_count'))}",
        f"Safety stop count: {_count_text(action.get('safety_stop_count'))}",
        f"Rollback count: {_count_text(action.get('rollback_count'))}",
        f"Preview verified: {decision.get('preview_verified')}",
        f"Preview agreement status: {decision.get('preview_agreement_status')}",
        f"Preview verification status: {decision.get('preview_verification_status')}",
        f"Preview blocker reason: {preview_reason}",
        f"Preview omitted reason: {decision.get('preview_omitted_reason')}",
        f"Preview no-op reason: {decision.get('preview_no_op_reason')}",
        f"Preview health status: {health_gate.get('status')}",
        f"Preview health reason: {health_gate.get('reason')}",
        f"Recovery selected option: {recovery_plan.get('selected_option')}",
        f"Recovery canary fraction: {canary_fraction}",
        f"Recovery holdout fraction: {holdout_fraction}",
        f"Savings per 1000 calls USD: {action.get('savings_per_1000_calls_usd')}",
        f"Projected savings USD: {action.get('projected_savings_usd')}",
        f"Expected savings path: {action.get('expected_savings_path')}",
        f"Acceptance metric: {action.get('acceptance_metric')}",
        "No-policy-write gate: do not write routing_rules.yaml or routing.rules until semantic-quality-regression-observed clears and preview/local gates pass.",
        f"Status label: {status_label}",
        f"Recommended next action: {decision.get('recommended_next_action') or action.get('recommended_next_action')}",
        _activation_successor_privacy_evidence(action, decision),
    ]


def _openai_routing_recovery_proposal_from_rows(
    action: dict[str, Any],
    decision: dict[str, Any],
    *,
    status_label: str,
) -> dict[str, Any]:
    title = _openai_routing_recovery_issue_title(action, decision, status_label=status_label)
    source_fingerprint = str(decision.get("source_fingerprint") or action.get("source_fingerprint") or "").strip()
    action_fingerprint = str(decision.get("successor_action_fingerprint") or action.get("fingerprint") or "").strip()
    preview_reason = _activation_successor_preview_reason(action, decision)
    status_text = "ready" if status_label == "status:ready" else "blocked"
    next_action = str(decision.get("recommended_next_action") or action.get("recommended_next_action") or "review-openai-routing-canary-blockers")
    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": _activation_successor_issue_labels(action, status_label),
        "proposal_source": "preview-verified-activation-successor",
        "fingerprint": sanitize_value(source_fingerprint),
        "successor_action_fingerprint": sanitize_value(action_fingerprint),
        "expected_savings_path": sanitize_value(
            action.get("expected_savings_path")
            or "Moves OpenAI routing semantic-regression evidence into a narrow recovery-review path without unsafe local policy writes."
        ),
        "body": _issue_body(
            title=title,
            rationale=(
                "The local activation queue has an OpenAI routing successor for "
                "`semantic-quality-regression-observed`. This proposal turns it into a narrow "
                f"{status_text} recovery review issue while preserving applied/holdout coverage, "
                "managed preview status, and the no-policy-write gate."
            ),
            evidence=_openai_routing_recovery_evidence(action, decision, status_label=status_label),
            implementation=[
                "Start from the matching local_activation_successor_decision and successor action in the research plan.",
                f"Use `{next_action}` and the OpenAI routing narrow-canary review path to inspect only the semantic-regression successor.",
                "Preserve applied_count, holdout_count, semantic-quality-regression-observed status, preview reason, and recovery canary sizing in the resulting evidence.",
                "Do not write routing_rules.yaml or change routing.rules until semantic-quality-regression-observed clears and the managed preview is fresh and agreed.",
                "For stale, disagreed, rejected, or no-op previews, keep the issue blocked and include the exact preview blocker reason.",
            ],
            acceptance=[
                "The emitted proposal has a stable fingerprint, labels include `routing`, and the status label is `status:ready` only when the managed preview is fresh and agrees.",
                f"Blocked proposals include the preview blocker reason `{preview_reason}` and remain `status:blocked` until the preview/local gates pass.",
                "The body states that no routing_rules.yaml write may happen until semantic-quality-regression-observed clears.",
                str(action.get("acceptance_metric") or "A narrower routing canary or rollback review records coverage and no unsafe policy write."),
                "Generated and follow-up evidence excludes raw prompts, provider bodies, absolute paths, request IDs, session IDs, cache keys, and individual candidate IDs.",
            ],
            savings_path=str(
                action.get("expected_savings_path")
                or "Moves the OpenAI routing opportunity from repeated blocked evidence into a precise recovery review path."
            ),
            sequencing=(
                "Sequence after managed preview agreement is attached to successor decisions and before any local routing policy write."
            ),
        ),
    }


def _is_tool_cache_dependency_successor(action: dict[str, Any]) -> bool:
    family = str(action.get("local_action_family") or action.get("lever") or "").strip()
    evidence_schema = str(action.get("evidence_schema") or "")
    blockers = {str(item) for item in action.get("blocker_codes") or [] if str(item or "").strip()}
    return bool(
        family == "cache"
        and (
            evidence_schema == "tokenclaw.request_shape_tool_cache_replay_evidence.v1"
            or str(action.get("dependency_evidence_class") or "")
            or str(action.get("dependency_evidence_decision") or "")
            or "invalidation-evidence-missing" in blockers
            or "unsafe-tool-calls-without-invalidation" in blockers
        )
    )


def _is_missing_tool_cache_dependency_drill(action: dict[str, Any], decision: dict[str, Any]) -> bool:
    if not _is_tool_cache_dependency_successor(action):
        return False
    evidence_class = str(action.get("dependency_evidence_class") or "").strip()
    evidence_decision = str(action.get("dependency_evidence_decision") or "").strip()
    evidence_status = str(action.get("dependency_evidence_status") or "").strip()
    reason = str(action.get("dependency_evidence_reason") or action.get("unblock_reason") or "").strip()
    next_action = str(decision.get("recommended_next_action") or action.get("recommended_next_action") or "").strip()
    blockers = {str(item) for item in action.get("blocker_codes") or [] if str(item or "").strip()}
    preview_status = str(decision.get("preview_agreement_status") or "").strip()
    missing = bool(
        "missing" in evidence_class
        or "missing" in evidence_decision
        or evidence_status == "missing"
        or reason == "invalidation-evidence-missing"
        or "invalidation-evidence-missing" in blockers
    )
    safe_collection_action = next_action in {
        "collect-file-invalidation-evidence",
        "collect-tool-cache-invalidation-evidence",
        "collect-tool-call-cache-invalidation-evidence",
        "add-invalidation-evidence",
    }
    return bool(missing and safe_collection_action and decision.get("preview_verified") and preview_status == "agreed")


def _tool_cache_dependency_issue_title(
    action: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    token = _activation_successor_title_token(action, decision)
    category = str(action.get("category") or "tool-cache").replace("_", "-")
    return f"Collect tool-cache invalidation drill for missing dependency evidence ({category}, {token})"


def _breakdown_evidence_lines(prefix: str, rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return [f"{prefix}: []"]
    lines: list[str] = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        value = sanitize_value(row.get("value") or row.get("reason_code") or row.get("reason") or row.get("status") or "unknown")
        count = _to_int(row.get("count"))
        lines.append(f"{prefix}: {value} count={count}")
    return lines or [f"{prefix}: []"]


def _tool_cache_dependency_evidence(
    action: dict[str, Any],
    decision: dict[str, Any],
) -> list[str]:
    source_fingerprint = str(decision.get("source_fingerprint") or action.get("source_fingerprint") or "").strip()
    action_fingerprint = str(decision.get("successor_action_fingerprint") or action.get("fingerprint") or "").strip()
    gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
    health_gate = gate.get("health_gate") if isinstance(gate.get("health_gate"), dict) else {}
    evidence = [
        f"Source metadata: {LOCAL_ACTIVATION_SUCCESSOR_DECISION_SCHEMA}",
        f"Fingerprint: {source_fingerprint}",
        f"Successor action fingerprint: {action_fingerprint}",
        "Provider scope: openai",
        "Local action family: cache",
        f"Dependency evidence class: {action.get('dependency_evidence_class')}",
        f"Dependency evidence decision: {action.get('dependency_evidence_decision')}",
        f"Dependency evidence status: {action.get('dependency_evidence_status')}",
        f"Dependency evidence reason: {action.get('dependency_evidence_reason')}",
        f"Evidence state: {action.get('evidence_state')}",
        f"Affected rows: {_count_text(action.get('affected_rows') or action.get('sample_count'))}",
        f"Sample count: {_count_text(action.get('sample_count'))}",
        f"Missing dependency evidence rows: {_count_text(action.get('missing_dependency_evidence_rows'))}",
        f"Stable dependency evidence rows: {_count_text(action.get('stable_dependency_evidence_rows'))}",
        f"Stale dependency evidence rows: {_count_text(action.get('stale_dependency_evidence_rows'))}",
        f"Unsafe dependency evidence rows: {_count_text(action.get('unsafe_dependency_evidence_rows'))}",
        f"Unknown dependency evidence rows: {_count_text(action.get('unknown_dependency_evidence_rows'))}",
        f"Unsafe tool-call blocker rows: {_count_text(action.get('unsafe_tool_call_blocker_rows'))}",
        f"Tools-present replay evidence rows: {_count_text(action.get('tools_present_replay_evidence_rows'))}",
        f"Live repeat confirmed: {bool(action.get('live_repeat_confirmed'))}",
        f"Observed hit proof: {bool(action.get('observed_hit_proof'))}",
        f"Observed hits: {_count_text(action.get('observed_hits'))}",
        f"Exact hit count: {_count_text(action.get('exact_hit_count'))}",
        f"Tool-cache replay enabled: {bool(action.get('tool_cache_replay_enabled'))}",
        f"Streaming replay enabled: {bool(action.get('streaming_replay_enabled'))}",
        f"Emits cache apply action: {bool(action.get('emits_cache_apply_action'))}",
        f"Cache apply action count: {_count_text(action.get('cache_apply_action_count'))}",
        f"Cache entries written: {_count_text(action.get('cache_entries_written'))}",
        f"Policy files written: {bool(action.get('policy_files_written'))}",
        f"Preview verified: {decision.get('preview_verified')}",
        f"Preview agreement status: {decision.get('preview_agreement_status')}",
        f"Preview verification status: {decision.get('preview_verification_status')}",
        f"Preview omitted reason: {decision.get('preview_omitted_reason')}",
        f"Preview no-op reason: {decision.get('preview_no_op_reason')}",
        f"Preview health status: {health_gate.get('status')}",
        f"Preview health reason: {health_gate.get('reason')}",
        f"Target local rule file: {action.get('target_local_rule_file')}",
        f"Target local policy section: {action.get('target_local_policy_section')}",
        f"Expected savings path: {action.get('expected_savings_path')}",
        f"Acceptance metric: {action.get('acceptance_metric')}",
        "Replay-disabled acceptance gate: keep tool-cache replay and streaming replay disabled unless stable dependency evidence plus live-repeat or observed-hit proof is present.",
        _activation_successor_privacy_evidence(action, decision),
    ]
    evidence.extend(_breakdown_evidence_lines("Dependency decision breakdown", action.get("dependency_evidence_decision_breakdown")))
    evidence.extend(_breakdown_evidence_lines("Evidence state breakdown", action.get("evidence_state_breakdown")))
    evidence.extend(_breakdown_evidence_lines("Blocker breakdown", action.get("blocker_breakdown")))
    return evidence


def _tool_cache_dependency_proposal_from_rows(
    action: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    title = _tool_cache_dependency_issue_title(action, decision)
    source_fingerprint = str(decision.get("source_fingerprint") or action.get("source_fingerprint") or "").strip()
    action_fingerprint = str(decision.get("successor_action_fingerprint") or action.get("fingerprint") or "").strip()
    next_action = str(decision.get("recommended_next_action") or action.get("recommended_next_action") or "collect-file-invalidation-evidence")
    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": _activation_successor_issue_labels(action, "status:ready"),
        "proposal_source": "preview-verified-activation-successor",
        "fingerprint": sanitize_value(source_fingerprint),
        "successor_action_fingerprint": sanitize_value(action_fingerprint),
        "expected_savings_path": sanitize_value(
            action.get("expected_savings_path")
            or "Removes the missing invalidation-evidence blocker before any tool-cache replay activation is considered."
        ),
        "body": _issue_body(
            title=title,
            rationale=(
                "The local activation successor queue has a preview-agreed tool-cache dependency blocker. "
                "This proposal turns missing dependency evidence into a safe invalidation drill while "
                "leaving tool-call replay disabled."
            ),
            evidence=_tool_cache_dependency_evidence(action, decision),
            implementation=[
                "Start from the matching local_activation_successor_decision and cache successor action in the research plan.",
                f"Run `{next_action}` as an invalidation/dependency evidence drill only; do not stage, widen, or enable tool-cache replay.",
                "Classify dependency evidence as stable, stale, unsafe, unknown, or missing using metadata-only file dependency snapshots.",
                "Keep unsafe, stale, unknown, and stable-without-proof cohorts blocked; do not write cache_rules.yaml for replay activation.",
                "Record dependency class counts, live-repeat or observed-hit proof status, and zero cache apply actions in the next report.",
            ],
            acceptance=[
                "A fixture with missing, unsafe, stale, unknown, and stable dependency evidence emits one ready invalidation drill issue for missing evidence.",
                "Unsafe, stale, unknown, and stable-without-proof shapes remain blocked from replay activation.",
                "The report emits no cache apply actions, no cache entries, no provider calls, and no cache policy writes for unsafe or missing dependency evidence.",
                "The issue body includes dependency class counts and the explicit replay-disabled acceptance gate.",
                "Generated and follow-up evidence excludes raw prompts, provider bodies, absolute paths, request IDs, session IDs, cache keys, file paths, and individual candidate IDs.",
            ],
            savings_path=str(
                action.get("expected_savings_path")
                or "Removes the invalidation-evidence bottleneck that prevents safe tool-call cache replay evaluation."
            ),
            sequencing=(
                "Sequence after preview-gated successor decisions and before any tool-cache replay canary, cache_rules.yaml write, or cache entry creation."
            ),
        ),
    }


def _activation_successor_proposal_from_rows(
    action: dict[str, Any],
    decision: dict[str, Any],
    *,
    status_label: str,
) -> dict[str, Any]:
    if _is_openai_semantic_regression_routing_successor(action):
        return _openai_routing_recovery_proposal_from_rows(action, decision, status_label=status_label)
    if _is_missing_tool_cache_dependency_drill(action, decision):
        return _tool_cache_dependency_proposal_from_rows(action, decision)
    title = _activation_successor_issue_title(action, decision, status_label=status_label)
    family = str(action.get("local_action_family") or decision.get("local_action_family") or "optimization")
    next_action = str(decision.get("recommended_next_action") or action.get("recommended_next_action") or "inspect-local-evidence")
    blocker_codes = [str(item) for item in action.get("blocker_codes") or [] if str(item or "").strip()]
    gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
    source_fingerprint = str(decision.get("source_fingerprint") or action.get("source_fingerprint") or "").strip()
    action_fingerprint = str(decision.get("successor_action_fingerprint") or action.get("fingerprint") or "").strip()
    status_text = "ready" if status_label == "status:ready" else "blocked"
    evidence = [
        f"Source metadata: {LOCAL_ACTIVATION_SUCCESSOR_DECISION_SCHEMA}",
        f"Fingerprint: {source_fingerprint}",
        f"Successor action fingerprint: {action_fingerprint}",
        f"Local action family: {family}",
        f"Decision: {decision.get('decision')}",
        f"Issue worthy status: {decision.get('issue_worthy_status')}",
        f"Preview verified: {decision.get('preview_verified')}",
        f"Preview agreement status: {decision.get('preview_agreement_status')}",
        f"Preview verification status: {decision.get('preview_verification_status')}",
        f"Preview verification decision: {decision.get('preview_verification_decision')}",
        f"Recommended next action: {next_action}",
        f"Blocker codes: {json.dumps(blocker_codes)}",
        f"Unblock reason: {action.get('unblock_reason')}",
        f"Preview omitted reason: {decision.get('preview_omitted_reason')}",
        f"Preview no-op reason: {decision.get('preview_no_op_reason')}",
        f"Expected savings path: {action.get('expected_savings_path')}",
        f"Acceptance metric: {action.get('acceptance_metric')}",
        _activation_successor_privacy_evidence(action, decision),
    ]
    if isinstance(gate.get("health_gate"), dict):
        health_gate = gate["health_gate"]
        evidence.extend(
            [
                f"Preview health status: {health_gate.get('status')}",
                f"Preview health reason: {health_gate.get('reason')}",
                f"Preview health next action: {health_gate.get('next_action')}",
            ]
        )
    if action.get("target_local_rule_file"):
        evidence.append(f"Target local rule file: {action.get('target_local_rule_file')}")
    if action.get("target_local_policy_section"):
        evidence.append(f"Target local policy section: {action.get('target_local_policy_section')}")

    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": _activation_successor_issue_labels(action, status_label),
        "proposal_source": "preview-verified-activation-successor",
        "fingerprint": sanitize_value(source_fingerprint),
        "successor_action_fingerprint": sanitize_value(action_fingerprint),
        "expected_savings_path": sanitize_value(
            action.get("expected_savings_path")
            or "This converts activation successor evidence into a concrete local follow-up."
        ),
        "body": _issue_body(
            title=title,
            rationale=(
                "The local activation successor queue contains a managed-preview-gated successor decision. "
                f"This proposal turns the {status_text} successor into a GitHub-ready local follow-up "
                "using only sanitized metadata."
            ),
            evidence=evidence,
            implementation=[
                "Start from the local_activation_successor_decision and matching successor action in the research plan.",
                f"Implement or review the `{next_action}` follow-up for the `{family}` local action family.",
                "Use local file-backed policy, dry-run, canary, rollback, or evidence modules only; do not call provider APIs or managed server APIs while generating the issue.",
                "Keep policy writes disabled until the item-specific acceptance metric and preview/local gates allow it.",
                "Record the outcome back into activation successor, preview, or burndown metadata so later research suppresses completed predecessors.",
            ],
            acceptance=[
                str(action.get("acceptance_metric") or "The successor action is resolved with a measurable local metadata outcome."),
                f"The next research plan reports source fingerprint {source_fingerprint} as progressed, suppressed, or blocked with a narrower reason.",
                "The emitted issue includes labels, acceptance metric, expected savings path, sequencing notes, and metadata-only privacy flags.",
                "Generated and follow-up evidence excludes raw prompts, provider bodies, absolute paths, request IDs, session IDs, cache keys, and individual candidate IDs.",
            ],
            savings_path=str(
                action.get("expected_savings_path")
                or "This converts the preview-verified activation loop into a self-refilling actionable backlog."
            ),
            sequencing=(
                "Sequence after managed preview agreement is attached to successor decisions and before generic low-backlog research proposals for the same local action family."
            ),
        ),
    }


def _is_new_sanitized_activation_feedback_successor(action: dict[str, Any], decision: dict[str, Any]) -> bool:
    family = str(action.get("local_action_family") or decision.get("local_action_family") or "").strip()
    if family != "activation-feedback":
        return False
    classification = (
        action.get("activation_feedback_diagnostic_classification")
        if isinstance(action.get("activation_feedback_diagnostic_classification"), dict)
        else {}
    )
    return str(classification.get("status") or action.get("diagnostic_evidence_status") or "").strip() == "new-sanitized-evidence"


def _activation_feedback_diagnostic_proposal_from_rows(
    action: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    next_action = str(decision.get("recommended_next_action") or action.get("recommended_next_action") or "review-new-sanitized-activation-feedback-evidence")
    source_fingerprint = str(decision.get("source_fingerprint") or action.get("source_fingerprint") or "").strip()
    action_fingerprint = str(decision.get("successor_action_fingerprint") or action.get("fingerprint") or "").strip()
    token = _activation_successor_title_token(action, decision)
    title = f"Record bounded activation-feedback successor input for {next_action} ({token})"
    classification = (
        action.get("activation_feedback_diagnostic_classification")
        if isinstance(action.get("activation_feedback_diagnostic_classification"), dict)
        else {}
    )
    evidence = [
        f"Source metadata: {LOCAL_ACTIVATION_SUCCESSOR_DECISION_SCHEMA}",
        f"Fingerprint: {source_fingerprint}",
        f"Successor action fingerprint: {action_fingerprint}",
        f"Diagnostic class: {action.get('diagnostic_class')}",
        f"Diagnostic fingerprint: {action.get('diagnostic_fingerprint')}",
        f"Diagnostic classification: {classification.get('status')}",
        f"Classification decision: {classification.get('decision')}",
        f"Local action family: {action.get('local_action_family')}",
        f"Decision: {decision.get('decision')}",
        f"Issue worthy status: {decision.get('issue_worthy_status')}",
        f"Recommended next action: {next_action}",
        f"Needed resolution: {json.dumps(action.get('needed_resolution') or [])}",
        f"Expected savings path: {action.get('expected_savings_path')}",
        f"Acceptance metric: {action.get('acceptance_metric')}",
        _activation_successor_privacy_evidence(action, decision),
    ]
    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": _activation_successor_issue_labels(action, "status:ready"),
        "proposal_source": "bounded-activation-feedback-successor",
        "fingerprint": sanitize_value(source_fingerprint),
        "successor_action_fingerprint": sanitize_value(action_fingerprint),
        "expected_savings_path": sanitize_value(
            action.get("expected_savings_path")
            or "This converts sanitized activation-feedback diagnostics into a bounded local follow-up."
        ),
        "body": _issue_body(
            title=title,
            rationale=(
                "A repeated activation-feedback diagnostic now carries new sanitized evidence. "
                "This proposal records it as a bounded local successor input without copying raw log snippets."
            ),
            evidence=evidence,
            implementation=[
                "Start from the activation-feedback successor action in the research plan.",
                f"Review the `{next_action}` follow-up using only metadata-only evidence.",
                "Do not include raw prompts, provider payloads, request IDs, cache keys, session IDs, tenant IDs, file paths, or raw log excerpts.",
                "Record the outcome back into the activation-feedback ledger as progressed, narrowed, or keep-blocked with a durable reason.",
            ],
            acceptance=[
                str(action.get("acceptance_metric") or "The activation-feedback successor input is resolved with a durable metadata-only outcome."),
                f"The next research plan reports source fingerprint {source_fingerprint} as progressed, suppressed, or blocked with a narrower reason.",
                "The successor evidence keeps metadata-only privacy flags and does not include raw log snippets or raw provider/client identifiers.",
            ],
            savings_path=str(
                action.get("expected_savings_path")
                or "This removes repeated activation-feedback rediscovery from the backlog loop."
            ),
            sequencing=(
                "Sequence before generic repeated-diagnostic proposals for activation-feedback so sanitized successor evidence is handled once."
            ),
        ),
    }


def _proposals_from_activation_successor_decisions(stats_summary: dict[str, Any]) -> list[dict[str, Any]]:
    queue = stats_summary.get("local_activation_next_action_queue")
    if not isinstance(queue, dict):
        return []
    actions = queue.get("successor_actions")
    if not isinstance(actions, list):
        actions = build_local_activation_successor_actions(queue)
    decisions = queue.get("successor_decisions")
    if not isinstance(decisions, list):
        decisions = build_local_activation_successor_decisions({"successor_actions": actions})
    action_by_fingerprint = {
        str(action.get("fingerprint")): action
        for action in actions
        if isinstance(action, dict) and str(action.get("fingerprint") or "").strip()
    }
    action_by_source = {
        str(action.get("source_fingerprint")): action
        for action in actions
        if isinstance(action, dict) and str(action.get("source_fingerprint") or "").strip()
    }
    proposals: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        source = str(decision.get("source_fingerprint") or "").strip()
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        action = action_by_fingerprint.get(str(decision.get("successor_action_fingerprint") or "")) or action_by_source.get(source)
        if not isinstance(action, dict):
            continue
        decision_text = str(decision.get("decision") or "").strip()
        issue_status = str(decision.get("issue_worthy_status") or "").strip()
        if issue_status == "suppressed" or decision_text in {"keep-current-rule", "suppress-duplicate"}:
            continue
        if _is_new_sanitized_activation_feedback_successor(action, decision):
            if issue_status in {"ready", "review"} and decision_text in {"ready", "review", "review-only"}:
                proposals.append(_activation_feedback_diagnostic_proposal_from_rows(action, decision))
            continue
        if _is_tool_cache_dependency_successor(action):
            if _is_missing_tool_cache_dependency_drill(action, decision):
                proposals.append(_tool_cache_dependency_proposal_from_rows(action, decision))
            continue
        preview_verified = bool(decision.get("preview_verified") or action.get("preview_verified"))
        if preview_verified and issue_status in {"ready", "review"} and decision_text in {"ready", "review", "review-only"}:
            proposals.append(
                _activation_successor_proposal_from_rows(action, decision, status_label="status:ready")
            )
            continue
        preview_status = str(decision.get("preview_agreement_status") or "").strip()
        if (
            preview_status
            and preview_status != "not-previewed"
            and decision_text in {"keep-blocked", "review-stale-preview"}
            and _activation_successor_has_concrete_blocker(action, decision)
        ):
            proposals.append(
                _activation_successor_proposal_from_rows(action, decision, status_label="status:blocked")
            )
    proposals.sort(
        key=lambda proposal: (
            _proposal_priority_score(_proposal_priority([str(label) for label in proposal.get("labels") or []])),
            str(proposal.get("title") or ""),
        )
    )
    return proposals


def _preview_successor_summary(successor_actions: list[dict[str, Any]]) -> dict[str, Any]:
    family: dict[str, Counter[str]] = {}
    omitted_reasons: Counter[str] = Counter()
    no_op_reasons: Counter[str] = Counter()
    request_shape_classes: Counter[str] = Counter()
    crunch_preview_decisions: Counter[str] = Counter()
    for action in successor_actions:
        if not isinstance(action, dict):
            continue
        family_name = str(action.get("local_action_family") or "unknown")
        counts = family.setdefault(family_name, Counter())
        gate = action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {}
        request_shape_class = str(
            action.get("request_shape_rollup_outcome_class")
            or gate.get("request_shape_rollup_outcome_class")
            or ""
        ).strip()
        if request_shape_class:
            request_shape_classes[request_shape_class] += 1
            counts[f"request-shape:{request_shape_class}"] += 1
        crunch_preview_decision = str(
            action.get("crunch_preview_decision")
            or gate.get("crunch_preview_decision")
            or ""
        ).strip()
        if crunch_preview_decision:
            crunch_preview_decisions[crunch_preview_decision] += 1
            counts[f"crunch-preview:{crunch_preview_decision}"] += 1
        if gate.get("verified"):
            counts["agreed"] += 1
        elif gate:
            status = str(gate.get("status") or "unknown")
            outcome_status = _managed_preview_public_outcome_status(gate)
            if outcome_status == "preview-omitted":
                counts["omitted"] += 1
            elif status in {"managed-local-disagreement", "failed-closed", "unsafe-preview-side-effect"}:
                counts["disagreed"] += 1
            elif status in {"stale-preview", "stale-preview-health"}:
                counts["stale"] += 1
            elif status in {"missing-preview", "missing-preview-decision", "no-data-preview-health"}:
                counts["missing"] += 1
            else:
                counts["blocked"] += 1
        else:
            counts["not_previewed"] += 1
        omitted = str(gate.get("omitted_reason") or "").strip()
        no_op = str(gate.get("no_op_reason") or "").strip()
        if omitted:
            omitted_reasons[omitted] += 1
            counts[f"omitted:{omitted}"] += 1
        if no_op:
            no_op_reasons[no_op] += 1
            counts[f"no-op:{no_op}"] += 1
    family_rows = []
    for family_name, counts in sorted(family.items()):
        family_rows.append(
            {
                "local_action_family": family_name,
                "agreed_count": counts["agreed"],
                "disagreed_count": counts["disagreed"],
                "omitted_count": counts["omitted"],
                "stale_count": counts["stale"],
                "missing_count": counts["missing"],
                "blocked_count": counts["blocked"],
                "not_previewed_count": counts["not_previewed"],
                "request_shape_no_data_count": counts["request-shape:no-data"],
                "request_shape_stale_count": counts["request-shape:stale"],
                "request_shape_too_small_count": counts["request-shape:too-small"],
                "request_shape_unsafe_count": counts["request-shape:unsafe"],
                "request_shape_review_ready_count": counts["request-shape:review-ready"],
                "crunch_preview_review_ready_count": counts["crunch-preview:review-ready"],
                "crunch_preview_keep_staged_count": counts["crunch-preview:keep-staged"],
                "crunch_preview_keep_blocked_count": counts["crunch-preview:keep-blocked"],
                "crunch_preview_too_small_count": counts["crunch-preview:too-small"],
                "crunch_preview_quality_risk_count": counts["crunch-preview:quality-risk"],
            }
        )
    return {
        "preview_agreement_by_local_action_family": family_rows,
        "request_shape_rollup_outcome_class_counts": [
            {"value": key, "count": count}
            for key, count in sorted(request_shape_classes.items())
        ],
        "crunch_preview_decision_counts": [
            {"value": key, "count": count}
            for key, count in sorted(crunch_preview_decisions.items())
        ],
        "preview_top_omitted_reasons": [
            {"value": key, "count": count}
            for key, count in sorted(omitted_reasons.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
        "preview_top_no_op_reasons": [
            {"value": key, "count": count}
            for key, count in sorted(no_op_reasons.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def build_local_activation_successor_actions(queue: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in queue.get("entries") or [] if isinstance(entry, dict)]
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        action = _local_activation_successor_action(entry)
        fingerprint = str(action.get("fingerprint") or "")
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        action["rank"] = len(actions) + 1
        actions.append(action)
    return actions


def _queue_rank_bucket(entry: dict[str, Any], realized: float, projected: float) -> int:
    status = str(entry.get("current_status") or "").strip()
    state = str(entry.get("state") or "").strip()
    if status in {"superseded"} or state in {"no-op", "retired-no-repeat", "superseded"}:
        return 3
    if realized > 0:
        return 0
    if projected > 0 or _to_float(entry.get("savings_per_1000_calls_usd")) > 0:
        return 1
    return 2


def _queue_savings_per_1000(entry: dict[str, Any], projected: float, realized: float) -> float:
    explicit = _to_float(entry.get("savings_per_1000_calls_usd"))
    if explicit > 0:
        return explicit
    sample_count = _to_int(entry.get("sample_count"))
    if sample_count <= 0:
        return 0.0
    value = realized if realized > 0 else projected
    return (value / sample_count) * 1000.0 if value > 0 else 0.0


def _queue_evidence_age_hours(entry: dict[str, Any]) -> float | None:
    for key in ("evidence_age_hours", "preview_age_hours"):
        if entry.get(key) is not None:
            return _to_float(entry.get(key))
    for key in (
        "evidence_freshness",
        "activation_feedback_freshness_gate",
        "stale_evidence",
    ):
        value = entry.get(key)
        if isinstance(value, dict):
            for age_key in ("age_hours", "preview_age_hours", "evidence_age_hours"):
                if value.get(age_key) is not None:
                    return _to_float(value.get(age_key))
    gate = entry.get("managed_preview_gate") if isinstance(entry.get("managed_preview_gate"), dict) else {}
    if gate.get("preview_age_hours") is not None:
        return _to_float(gate.get("preview_age_hours"))
    health_gate = gate.get("health_gate") if isinstance(gate.get("health_gate"), dict) else {}
    if health_gate.get("latest_preview_age_hours") is not None:
        return _to_float(health_gate.get("latest_preview_age_hours"))
    return None


def _queue_max_evidence_age_hours(entry: dict[str, Any]) -> float | None:
    for key in ("max_evidence_age_hours", "stale_after_hours"):
        if entry.get(key) is not None:
            return _to_float(entry.get(key))
    for key in (
        "evidence_freshness",
        "activation_feedback_freshness_gate",
        "stale_evidence",
    ):
        value = entry.get(key)
        if isinstance(value, dict):
            for max_key in ("max_age_hours", "stale_after_hours", "max_evidence_age_hours"):
                if value.get(max_key) is not None:
                    return _to_float(value.get(max_key))
    gate = entry.get("managed_preview_gate") if isinstance(entry.get("managed_preview_gate"), dict) else {}
    if gate.get("stale_after_hours") is not None:
        return _to_float(gate.get("stale_after_hours"))
    health_gate = gate.get("health_gate") if isinstance(gate.get("health_gate"), dict) else {}
    if health_gate.get("stale_after_hours") is not None:
        return _to_float(health_gate.get("stale_after_hours"))
    return None


def _queue_has_rollback_required_action(entry: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(entry.get(key) or "")
        for key in (
            "next_action",
            "recommended_next_action",
            "promotion_readiness",
            "promotion_recommendation",
            "impact_recommendation",
            "observed_hit_blocker",
            "unblock_reason",
            "reason",
        )
    ).lower()
    for code in entry.get("blocker_codes") or []:
        haystack += f" {code}".lower()
    return bool(entry.get("rollback_required")) or "rollback-required" in haystack or "rollback-" in haystack


def _queue_freshness_state(entry: dict[str, Any]) -> str:
    gate = entry.get("managed_preview_gate") if isinstance(entry.get("managed_preview_gate"), dict) else {}
    gate_status = str(gate.get("status") or "").strip()
    raw_statuses = [
        str(entry.get("evidence_freshness_status") or "").strip(),
        str(entry.get("freshness_state") or "").strip(),
        gate_status,
    ]
    for key in ("evidence_freshness", "activation_feedback_freshness_gate", "stale_evidence"):
        value = entry.get(key)
        if isinstance(value, dict):
            raw_statuses.append(str(value.get("status") or "").strip())
            raw_statuses.append(str(value.get("reason") or "").strip())
            if value.get("stale") is True:
                raw_statuses.append("stale")
    lowered = " ".join(status.lower() for status in raw_statuses if status)
    blockers = " ".join(str(item).lower() for item in entry.get("blocker_codes") or [])
    text = f"{lowered} {blockers} {entry.get('unblock_reason') or ''} {entry.get('next_action') or ''}".lower()
    age = _queue_evidence_age_hours(entry)
    max_age = _queue_max_evidence_age_hours(entry)
    stale = (
        "stale" in text
        or "evidence-older-than-max-age" in text
        or (age is not None and max_age is not None and max_age > 0 and age > max_age)
    )
    if stale and _queue_has_rollback_required_action(entry):
        return "stale-rollback-required"
    if stale:
        return "stale"
    if "no-data" in text or "missing-preview" in text or "missing-evidence" in text:
        return "no-data"
    if "fresh" in text or (age is not None and max_age is not None and max_age > 0 and age <= max_age):
        return "fresh"
    return "unknown"


def _queue_freshness_multiplier(freshness_state: str) -> float:
    if freshness_state == "fresh":
        return 1.0
    if freshness_state == "stale-rollback-required":
        return 0.9
    if freshness_state == "unknown":
        return 0.7
    if freshness_state == "no-data":
        return 0.35
    return 0.25


def _queue_adjusted_rank_bucket(entry: dict[str, Any]) -> int:
    status = str(entry.get("current_status") or "").strip()
    state = str(entry.get("state") or "").strip()
    successor_status = str(entry.get("successor_status") or "").strip()
    issue_status = str(entry.get("issue_worthy_status") or "").strip()
    freshness_state = str(entry.get("freshness_state") or _queue_freshness_state(entry))
    if (
        status in {"superseded", "suppressed"} | _TERMINAL_ACTIVATION_SUCCESSOR_STATES
        or state in {"no-op", "retired-no-repeat", "superseded", "suppressed"} | _TERMINAL_ACTIVATION_SUCCESSOR_STATES
    ):
        return 5
    if _to_float(entry.get("realized_savings_usd")) > 0:
        return 0
    if freshness_state == "stale-rollback-required":
        return 1
    if issue_status == "ready" or successor_status == "ready":
        return 1 if freshness_state in {"fresh", "unknown"} else 3
    if _to_float(entry.get("projected_savings_usd")) > 0 or _to_float(entry.get("savings_per_1000_calls_usd")) > 0:
        return 2 if freshness_state in {"fresh", "unknown"} else 3
    if str(entry.get("duplicate_suppression_status") or "") == "suppressed":
        return 4
    return 3


def _queue_rank_basis(entry: dict[str, Any]) -> dict[str, Any]:
    projected = _to_float(entry.get("projected_savings_usd"))
    realized = _to_float(entry.get("realized_savings_usd"))
    savings_per_1000 = _queue_savings_per_1000(entry, projected, realized)
    freshness_state = _queue_freshness_state(entry)
    multiplier = _queue_freshness_multiplier(freshness_state)
    age = _queue_evidence_age_hours(entry)
    max_age = _queue_max_evidence_age_hours(entry)
    basis = {
        "schema": "tokenclaw.local_activation_successor_rank_basis.v1",
        "rank_bucket": _queue_adjusted_rank_bucket({**entry, "freshness_state": freshness_state}),
        "freshness_state": freshness_state,
        "freshness_multiplier": multiplier,
        "freshness_adjusted_savings_per_1000_calls_usd": round(savings_per_1000 * multiplier, 8),
        "savings_per_1000_calls_usd": round(savings_per_1000, 8),
        "sample_count": _to_int(entry.get("sample_count")),
        "projected_savings_usd": round(projected, 8),
        "realized_savings_usd": round(realized, 8),
        "rollback_required": _queue_has_rollback_required_action(entry),
        "blocking_reason": sanitize_value(entry.get("blocking_reason") or entry.get("unblock_reason")),
    }
    if age is not None:
        basis["evidence_age_hours"] = round(age, 3)
    if max_age is not None:
        basis["max_evidence_age_hours"] = round(max_age, 3)
    return {key: value for key, value in basis.items() if value not in (None, "", [])}


def _apply_queue_rank_metadata(entry: dict[str, Any]) -> None:
    basis = _queue_rank_basis(entry)
    entry["freshness_state"] = basis["freshness_state"]
    entry["blocking_reason"] = sanitize_value(entry.get("unblock_reason") or basis.get("blocking_reason"))
    entry["rank_bucket"] = basis["rank_bucket"]
    entry["freshness_multiplier"] = basis["freshness_multiplier"]
    entry["freshness_adjusted_savings_per_1000_calls_usd"] = basis[
        "freshness_adjusted_savings_per_1000_calls_usd"
    ]
    entry["savings_per_1000_calls_usd"] = basis["savings_per_1000_calls_usd"]
    entry["rank_basis"] = basis


def _apply_queue_duplicate_fingerprint_suppression(entries: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for entry in entries:
        fingerprint = str(entry.get("fingerprint") or "").strip()
        if not fingerprint:
            continue
        if fingerprint not in seen:
            seen.add(fingerprint)
            continue
        entry["duplicate_suppression_status"] = "suppressed"
        entry["duplicate_suppression_reason"] = "duplicate-successor-fingerprint"
        entry["issue_worthy_status"] = "suppressed"
        entry["duplicate_suppression"] = {
            "schema": "tokenclaw.local_activation_successor_duplicate_suppression.v1",
            "reason": "duplicate-successor-fingerprint",
            "suppresses_duplicate_successor_issue": True,
            "metadata_only": True,
            "aggregate_only": True,
        }


def _local_activation_next_action_queue_entry(entry: dict[str, Any]) -> dict[str, Any]:
    projected = round(_queue_projected_savings(entry), 8)
    realized = round(_queue_realized_savings(entry, projected), 8)
    savings_per_1000 = round(_queue_savings_per_1000(entry, projected, realized), 8)
    blocker_codes = sanitize_value([str(item) for item in entry.get("blocker_codes") or [] if str(item or "").strip()])
    duplicate_status = _queue_duplicate_suppression_status(entry)
    suppression = entry.get("duplicate_suppression") if isinstance(entry.get("duplicate_suppression"), dict) else {}
    clean = {
        "schema": LOCAL_ACTIVATION_NEXT_ACTION_QUEUE_ENTRY_SCHEMA,
        "rank": 0,
        "ledger_rank": _to_int(entry.get("rank")),
        "fingerprint": sanitize_value(entry.get("fingerprint")),
        "lever": sanitize_value(entry.get("lever") or "unknown"),
        "local_action_family": sanitize_value(entry.get("local_action_family") or entry.get("lever") or "unknown"),
        "state": sanitize_value(entry.get("state") or "unknown"),
        "current_status": sanitize_value(entry.get("current_status") or "unknown"),
        "issue_worthy_status": sanitize_value(entry.get("issue_worthy_status") or "review"),
        "next_action": sanitize_value(entry.get("next_action") or "inspect-local-evidence"),
        "unblock_reason": _queue_unblock_reason(entry),
        "blocker_codes": blocker_codes,
        "sample_count": _to_int(entry.get("sample_count")),
        "matched_count": _to_int(entry.get("matched_count")),
        "applied_count": _to_int(entry.get("applied_count")),
        "holdout_count": _to_int(entry.get("holdout_count")),
        "fallback_count": _to_int(entry.get("fallback_count")),
        "safety_stop_count": _to_int(entry.get("safety_stop_count")),
        "rollback_count": _to_int(entry.get("rollback_count")),
        "realized_savings_usd": realized,
        "projected_savings_usd": projected,
        "savings_per_1000_calls_usd": savings_per_1000,
        "target_local_rule_file": sanitize_value(entry.get("target_local_rule_file")),
        "target_local_policy_section": sanitize_value(entry.get("target_local_policy_section")),
        "duplicate_suppression_status": duplicate_status,
        "duplicate_suppression_reason": sanitize_value(suppression.get("reason")) if suppression else None,
        "duplicate_suppression": sanitize_value(suppression) if suppression else None,
        "evidence_schema": sanitize_value(entry.get("evidence_schema")),
        "expected_savings_path": sanitize_value(entry.get("expected_savings_path")),
        "diagnostic_class": sanitize_value(entry.get("diagnostic_class")),
        "diagnostic_reason": sanitize_value(entry.get("diagnostic_reason")),
        "diagnostic_fingerprint": sanitize_value(entry.get("diagnostic_fingerprint")),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "provider_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "individual_candidate_ids_included": False,
        },
    }
    passthrough_keys = (
        "affected_rows",
        "cache_apply_action_count",
        "cache_entries_written",
        "dependency_evidence_class",
        "dependency_evidence_decision",
        "dependency_evidence_reason",
        "dependency_evidence_status",
        "emits_cache_apply_action",
        "evidence_state",
        "diagnostic_evidence_status",
        "review_status",
        "keep_blocked_reason",
        "needed_resolution",
        "next_state",
        "next_state_reason",
        "status",
        "source",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "provider_family",
        "has_tools",
        "requested_model",
        "candidate_target_model",
        "required_local_executor",
        "canary_fraction",
        "holdout_fraction",
        "executor_compatible",
        "tool_cache_replay_enabled",
        "streaming_replay_enabled",
        "tools_present_replay_evidence",
        "generic_tools_present_blocker_reduced",
        "tools_present_rows",
        "tools_present_replay_evidence_rows",
        "generic_tools_present_blocker_reduced_rows",
        "unsafe_tool_call_blocker_rows",
        "missing_dependency_evidence_rows",
        "stable_dependency_evidence_rows",
        "stale_dependency_evidence_rows",
        "unsafe_dependency_evidence_rows",
        "unknown_dependency_evidence_rows",
        "missing_applied_coverage",
        "missing_holdout_coverage",
        "burndown_status",
        "promotion_allowed",
        "stage_allowed",
        "active_policy_changed",
        "wrote_active_policy_files",
        "policy_files_written",
        "rollback_required",
        "stale_no_traffic_retirement",
        "durable_action_ledger_entry",
        "durable_outcome_ledger_entry",
        "terminal_successor_state",
        "managed_preview_required",
        "full_rollout_outcome",
        "full_rollout_outcome_next_action",
        "full_rollout_successor_decision",
        "full_rollout_successor_next_action",
        "full_rollout_successor_no_op_reason",
        "measured_full_rollout_activation",
        "evidence_freshness_status",
        "evidence_age_hours",
        "max_evidence_age_hours",
        "stale_after_hours",
        "promotion_readiness",
        "promotion_recommendation",
        "impact_recommendation",
        "observed_hit_blocker",
        "rollback_applied",
        "post_rollback_successor_decision",
        "post_rollback_next_action",
        "post_rollback_reason",
        "warmup_miss_count",
        "exact_hit_count",
    )
    for key in passthrough_keys:
        if entry.get(key) is not None:
            value = entry.get(key)
            clean[key] = bool(value) if isinstance(value, bool) else sanitize_value(value)
    for review_key in (
        "unblock_criteria",
        "safety_stop_reason_review",
        "safer_threshold_or_executor_guard",
        "rollback_proof",
        "rollback_metadata",
        "applied_coverage",
        "holdout_coverage",
        "local_file_backed_representation",
        "dependency_evidence_review",
        "active_rule_regression_gate",
        "outcome_gate",
        "keep_active_regression_gate",
        "full_rollout_activation_outcome",
        "activation_feedback_freshness_gate",
        "activation_feedback_diagnostic_classification",
        "managed_preview_gate",
        "evidence_freshness",
        "coverage",
        "recovery_plan",
        "post_rollback_observation",
        "local_policy_patch",
    ):
        if isinstance(entry.get(review_key), dict):
            clean[review_key] = sanitize_value(entry.get(review_key))
    for list_key in ("rollback_applied_rules", "applied_rollback_rules"):
        if isinstance(entry.get(list_key), list):
            clean[list_key] = sanitize_value(entry.get(list_key))
    for breakdown_key in (
        "blocker_breakdown",
        "dependency_evidence_decision_breakdown",
        "evidence_state_breakdown",
        "next_action_breakdown",
    ):
        if isinstance(entry.get(breakdown_key), list):
            clean[breakdown_key] = sanitize_value(entry.get(breakdown_key))
    preserved_empty_keys = {
        "rank",
        "rank_bucket",
        "ledger_rank",
        "sample_count",
        "matched_count",
        "applied_count",
        "holdout_count",
        "fallback_count",
        "safety_stop_count",
        "rollback_count",
        "realized_savings_usd",
        "projected_savings_usd",
        "savings_per_1000_calls_usd",
        "freshness_adjusted_savings_per_1000_calls_usd",
        "freshness_multiplier",
        "canary_fraction",
        "holdout_fraction",
        "affected_rows",
        "cache_apply_action_count",
        "cache_entries_written",
        "emits_cache_apply_action",
        "tool_cache_replay_enabled",
        "streaming_replay_enabled",
        "tools_present_replay_evidence",
        "generic_tools_present_blocker_reduced",
        "tools_present_rows",
        "tools_present_replay_evidence_rows",
        "generic_tools_present_blocker_reduced_rows",
        "unsafe_tool_call_blocker_rows",
        "missing_dependency_evidence_rows",
        "stable_dependency_evidence_rows",
        "stale_dependency_evidence_rows",
        "unsafe_dependency_evidence_rows",
        "unknown_dependency_evidence_rows",
        "promotion_allowed",
        "stage_allowed",
        "active_policy_changed",
        "wrote_active_policy_files",
        "policy_files_written",
        "rollback_required",
        "executor_compatible",
        "missing_applied_coverage",
        "missing_holdout_coverage",
        "durable_action_ledger_entry",
        "durable_outcome_ledger_entry",
        "measured_full_rollout_activation",
        "evidence_age_hours",
        "max_evidence_age_hours",
        "warmup_miss_count",
        "exact_hit_count",
    }
    _apply_cache_replay_post_rollback_classification(clean)
    return {key: value for key, value in clean.items() if value not in (None, "", [], 0) or key in preserved_empty_keys}


def build_local_activation_next_action_queue(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    """Rank local activation next actions by realized savings and unblock reason."""
    ledger = (
        stats_summary
        if isinstance(stats_summary.get("entries"), list)
        and stats_summary.get("schema") == EVIDENCE_TO_ACTIVATION_LEDGER_SCHEMA
        else stats_summary.get("evidence_to_activation_next_action_ledger")
        if isinstance(stats_summary.get("evidence_to_activation_next_action_ledger"), dict)
        else None
    )
    if not isinstance(ledger, dict):
        safety_stop_burndown = (
            stats_summary
            if stats_summary.get("schema") == "tokenclaw.activation_safety_stop_burndown.v1"
            else stats_summary.get("activation_safety_stop_burndown")
            if isinstance(stats_summary.get("activation_safety_stop_burndown"), dict)
            else None
        )
        if isinstance(safety_stop_burndown, dict):
            ledger = build_evidence_to_activation_next_action_ledger(
                {},
                safety_stop_burndown=safety_stop_burndown,
            )
    if not isinstance(ledger, dict):
        return None
    entries = [
        _local_activation_next_action_queue_entry(entry)
        for entry in ledger.get("entries") or []
        if isinstance(entry, dict)
    ]
    if not entries:
        return None
    managed_preview_outcomes = _managed_preview_outcomes_report(stats_summary)
    managed_preview_health = _managed_preview_health_report(stats_summary, managed_preview_outcomes)
    if isinstance(managed_preview_outcomes, dict) or isinstance(managed_preview_health, dict):
        for entry in entries:
            entry["managed_preview_gate"] = _managed_preview_successor_gate(
                entry,
                managed_preview_outcomes,
                managed_preview_health,
            )
    _apply_queue_duplicate_fingerprint_suppression(entries)
    for entry in entries:
        _apply_queue_rank_metadata(entry)
        managed_rank = _queue_managed_priority_rank(entry)
        gate = entry.get("managed_preview_gate") if isinstance(entry.get("managed_preview_gate"), dict) else {}
        entry["managed_priority_rank"] = managed_rank
        entry["managed_priority_source"] = (
            "ranked-managed-preview" if managed_rank > 0 else "local-successor-rank"
        )
        if managed_rank > 0:
            entry["managed_rank"] = managed_rank
            entry["managed_recommended_next_action"] = sanitize_value(gate.get("managed_recommended_next_action"))
            entry["managed_expected_savings_path"] = sanitize_value(gate.get("managed_expected_savings_path"))
            entry["managed_preview_action_ref"] = sanitize_value(gate.get("managed_preview_action_ref"))
        else:
            entry["managed_rank_fallback_reason"] = sanitize_value(
                gate.get("managed_rank_fallback_reason") or "managed-ranked-next-action-missing"
            )
    entries.sort(
        key=lambda item: (
            0 if _to_int(item.get("managed_priority_rank")) > 0 else 1,
            _to_int(item.get("managed_priority_rank")) if _to_int(item.get("managed_priority_rank")) > 0 else 999_999,
            _to_int(item.get("rank_bucket")),
            -_to_float(item.get("freshness_adjusted_savings_per_1000_calls_usd")),
            -_to_float(item.get("realized_savings_usd")),
            -_to_float(item.get("projected_savings_usd")),
            -_to_float(item.get("savings_per_1000_calls_usd")),
            -_to_int(item.get("sample_count")),
            str(item.get("lever") or ""),
            str(item.get("next_action") or ""),
        )
    )
    for rank, item in enumerate(entries, start=1):
        item["rank"] = rank

    top = entries[0]
    blocker_counts: Counter[str] = Counter()
    lever_counts: Counter[str] = Counter(str(item.get("lever") or "unknown") for item in entries)
    status_counts: Counter[str] = Counter(str(item.get("current_status") or "unknown") for item in entries)
    for item in entries:
        reason = str(item.get("unblock_reason") or "").strip()
        if reason:
            blocker_counts[reason] += 1
        for code in item.get("blocker_codes") or []:
            blocker_counts[str(code)] += 1
    successor_actions = build_local_activation_successor_actions({"entries": entries})
    successor_decisions = build_local_activation_successor_decisions({"successor_actions": successor_actions})
    preview_successor_summary = _preview_successor_summary(successor_actions)
    preview_gates = [
        entry.get("managed_preview_gate")
        for entry in entries
        if isinstance(entry.get("managed_preview_gate"), dict)
    ]
    preview_verified_count = sum(1 for gate in preview_gates if gate.get("verified"))
    preview_required_count = sum(1 for gate in preview_gates if gate.get("required"))
    preview_blocked_count = sum(1 for gate in preview_gates if gate.get("required") and not gate.get("verified"))
    managed_ranked_count = sum(1 for item in entries if _to_int(item.get("managed_priority_rank")) > 0)
    managed_rank_fallback_counts: Counter[str] = Counter(
        str(item.get("managed_rank_fallback_reason") or "none")
        for item in entries
        if _to_int(item.get("managed_priority_rank")) <= 0
    )
    preview_status_counts: Counter[str] = Counter(str(gate.get("status") or "unknown") for gate in preview_gates)
    preview_decision_counts: Counter[str] = Counter(str(gate.get("decision") or "unknown") for gate in preview_gates)

    result = {
        "schema": LOCAL_ACTIVATION_NEXT_ACTION_QUEUE_SCHEMA,
        "status": "ranked",
        "source_schema": ledger.get("schema"),
        "summary": {
            "queued_action_count": len(entries),
            "successor_action_count": len(successor_actions),
            "non_duplicate_successor_action_count": len({str(action.get("fingerprint")) for action in successor_actions if action.get("fingerprint")}),
            "successor_decision_count": len(successor_decisions),
            "non_duplicate_successor_decision_count": len({str(row.get("source_fingerprint")) for row in successor_decisions if row.get("source_fingerprint")}),
            "preview_verified_successor_count": preview_verified_count,
            "preview_required_successor_count": preview_required_count,
            "preview_blocked_successor_count": preview_blocked_count,
            "managed_ranked_successor_count": managed_ranked_count,
            "managed_priority_overlay_status": "ranked-managed-preview" if managed_ranked_count else "local-successor-rank",
            "managed_rank_fallback_reason_counts": [
                {"value": key, "count": count} for key, count in sorted(managed_rank_fallback_counts.items())
            ],
            "preview_gate_status_counts": [{"value": key, "count": count} for key, count in sorted(preview_status_counts.items())],
            "preview_gate_decision_counts": [{"value": key, "count": count} for key, count in sorted(preview_decision_counts.items())],
            "preview_agreement_by_local_action_family": preview_successor_summary["preview_agreement_by_local_action_family"],
            "request_shape_rollup_outcome_class_counts": preview_successor_summary[
                "request_shape_rollup_outcome_class_counts"
            ],
            "crunch_preview_decision_counts": preview_successor_summary[
                "crunch_preview_decision_counts"
            ],
            "preview_top_omitted_reasons": preview_successor_summary["preview_top_omitted_reasons"],
            "preview_top_no_op_reasons": preview_successor_summary["preview_top_no_op_reasons"],
            "top_lever": top.get("lever"),
            "top_state": top.get("state"),
            "top_current_status": top.get("current_status"),
            "top_next_action": top.get("next_action"),
            "top_unblock_reason": top.get("unblock_reason"),
            "top_blocking_reason": top.get("blocking_reason"),
            "top_freshness_state": top.get("freshness_state"),
            "top_savings_per_1000_calls_usd": top.get("savings_per_1000_calls_usd"),
            "top_freshness_adjusted_savings_per_1000_calls_usd": top.get("freshness_adjusted_savings_per_1000_calls_usd"),
            "top_rank_basis": top.get("rank_basis"),
            "top_realized_savings_usd": top.get("realized_savings_usd"),
            "top_projected_savings_usd": top.get("projected_savings_usd"),
            "total_realized_savings_usd": round(sum(_to_float(item.get("realized_savings_usd")) for item in entries), 8),
            "total_projected_savings_usd": round(sum(_to_float(item.get("projected_savings_usd")) for item in entries), 8),
            "lever_counts": [{"value": key, "count": count} for key, count in sorted(lever_counts.items())],
            "status_counts": [{"value": key, "count": count} for key, count in sorted(status_counts.items())],
            "unblock_reason_counts": [
                {"value": key, "count": count}
                for key, count in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
            ],
        },
        "entries": entries[:20],
        "successor_actions": successor_actions[:20],
        "successor_decisions": successor_decisions[:20],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "provider_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tenant_ids_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_written": False,
            "read_only": True,
        },
    }
    return sanitize_value(result)


def _activation_burndown_provider_scope(row: dict[str, Any]) -> str | None:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "provider_family",
            "source_surface",
            "endpoint",
            "requested_model",
            "candidate_target_model",
            "required_local_executor",
            "next_action",
            "recommended_next_action",
            "unblock_reason",
            "evidence_schema",
        )
    ).lower()
    for code in row.get("blocker_codes") or []:
        text += f" {code}".lower()
    if "openai" in text or "gpt-" in text:
        return "openai"
    if "anthropic" in text or "claude-" in text:
        return "anthropic"
    return None


def _activation_burndown_action_bucket(row: dict[str, Any]) -> int:
    family = str(row.get("local_action_family") or row.get("lever") or "").strip()
    provider_scope = _activation_burndown_provider_scope(row)
    status = str(row.get("successor_status") or "").strip()
    current_status = str(row.get("current_status") or "").strip()
    state = str(row.get("state") or "").strip()
    if family in {"routing", "cache"} and provider_scope == "openai" and status in {"ready", "review", "review-only", "keep-blocked"}:
        return 0
    if status in {"ready", "review", "review-only", "keep-blocked"} and current_status not in {"full-rollout", "superseded"}:
        return 1
    if status in {"keep-current-rule", "suppress-duplicate"} or current_status == "full-rollout":
        return 3
    if current_status == "superseded" or state in {"superseded", "retired-no-repeat"}:
        return 4
    return 2


def _activation_burndown_row_from_successor(action: dict[str, Any]) -> dict[str, Any]:
    row = {
        "schema": ACTIVATION_BURNDOWN_ROW_SCHEMA,
        "rank": 0,
        "source_schema": sanitize_value(action.get("schema")),
        "source_rank": _to_int(action.get("rank")),
        "source_queue_rank": _to_int(action.get("source_queue_rank")),
        "source_ledger_rank": _to_int(action.get("source_ledger_rank")),
        "fingerprint": sanitize_value(action.get("fingerprint")),
        "source_fingerprint": sanitize_value(action.get("source_fingerprint")),
        "lever": sanitize_value(action.get("lever") or "unknown"),
        "local_action_family": sanitize_value(action.get("local_action_family") or action.get("lever") or "unknown"),
        "provider_scope": _activation_burndown_provider_scope(action),
        "current_state": sanitize_value(action.get("state") or "unknown"),
        "current_status": sanitize_value(action.get("current_status") or "unknown"),
        "successor_status": sanitize_value(action.get("successor_status") or "review"),
        "next_action": sanitize_value(action.get("recommended_next_action") or "inspect-local-evidence"),
        "blocker_codes": sanitize_value([str(item) for item in action.get("blocker_codes") or [] if str(item or "").strip()]),
        "unblock_reason": sanitize_value(action.get("unblock_reason")),
        "target_local_rule_file": sanitize_value(action.get("target_local_rule_file")),
        "target_local_policy_section": sanitize_value(action.get("target_local_policy_section")),
        "duplicate_suppression_status": sanitize_value(action.get("duplicate_suppression_status")),
        "duplicate_suppression_reason": sanitize_value(action.get("duplicate_suppression_reason")),
        "projected_savings_usd": round(_to_float(action.get("projected_savings_usd")), 8),
        "realized_savings_usd": round(_to_float(action.get("realized_savings_usd")), 8),
        "sample_count": _to_int(action.get("sample_count")),
        "acceptance_metric": sanitize_value(action.get("acceptance_metric")),
        "expected_savings_path": sanitize_value(action.get("expected_savings_path")),
        "privacy": _successor_action_privacy(),
    }
    if isinstance(action.get("managed_preview_gate"), dict):
        row["managed_preview_gate"] = sanitize_value(action.get("managed_preview_gate"))
        row["preview_verified"] = bool(action.get("preview_verified"))
        row["preview_verification_status"] = sanitize_value(action.get("preview_verification_status"))
        row["preview_verification_decision"] = sanitize_value(action.get("preview_verification_decision"))
    for key in (
        "evidence_schema",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "provider_family",
        "requested_model",
        "candidate_target_model",
        "required_local_executor",
    ):
        if action.get(key) is not None:
            row[key] = sanitize_value(action.get(key))
    for key in (
        "emits_cache_apply_action",
        "policy_files_written",
        "tool_cache_replay_enabled",
        "streaming_replay_enabled",
        "promotion_allowed",
        "stage_allowed",
        "missing_applied_coverage",
        "missing_holdout_coverage",
        "measured_full_rollout_activation",
        "durable_action_ledger_entry",
        "durable_outcome_ledger_entry",
    ):
        if action.get(key) is not None:
            row[key] = bool(action.get(key))
    for key in (
        "applied_count",
        "holdout_count",
        "skipped_count",
        "fallback_count",
        "retry_count",
        "rollback_count",
        "safety_stop_count",
        "observed_saved_tokens",
        "projected_saved_tokens",
    ):
        if action.get(key) is not None:
            row[key] = _to_int(action.get(key))
    for key in (
        "error_rate_delta",
        "retry_rate_delta",
        "fallback_rate_delta",
    ):
        if action.get(key) is not None:
            row[key] = round(_to_float(action.get(key)), 8)
    for key in (
        "full_rollout_outcome",
        "full_rollout_outcome_next_action",
        "full_rollout_successor_decision",
        "full_rollout_successor_next_action",
        "full_rollout_successor_no_op_reason",
    ):
        if action.get(key) is not None:
            row[key] = sanitize_value(action.get(key))
    for key in (
        "full_rollout_activation_outcome",
        "keep_active_regression_gate",
    ):
        if isinstance(action.get(key), dict):
            row[key] = sanitize_value(action.get(key))
    preserved = {
        "rank",
        "source_rank",
        "source_queue_rank",
        "source_ledger_rank",
        "projected_savings_usd",
        "realized_savings_usd",
        "sample_count",
        "emits_cache_apply_action",
        "policy_files_written",
        "tool_cache_replay_enabled",
        "streaming_replay_enabled",
        "promotion_allowed",
        "stage_allowed",
        "missing_applied_coverage",
        "missing_holdout_coverage",
        "measured_full_rollout_activation",
        "durable_action_ledger_entry",
        "durable_outcome_ledger_entry",
        "applied_count",
        "holdout_count",
        "skipped_count",
        "fallback_count",
        "retry_count",
        "rollback_count",
        "safety_stop_count",
        "observed_saved_tokens",
        "projected_saved_tokens",
        "error_rate_delta",
        "retry_rate_delta",
        "fallback_rate_delta",
    }
    return {key: value for key, value in row.items() if value not in (None, "", [], 0) or key in preserved}


def _activation_burndown_selected(row: dict[str, Any]) -> bool:
    if str(row.get("current_status") or "") in {"full-rollout", "superseded"}:
        return False
    if str(row.get("current_state") or "") in {"superseded", "retired-no-repeat"}:
        return False
    return str(row.get("successor_status") or "") in {"ready", "review", "review-only", "keep-blocked", "review-stale-preview"}


def build_activation_burndown_report(
    plan: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the unified local activation burndown report for successor work."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    evidence_report = build_evidence_to_activation_burndown(plan, now=now)
    queue = evidence_report.get("next_action_queue") if isinstance(evidence_report.get("next_action_queue"), dict) else None
    successor_actions = (
        build_local_activation_successor_actions(queue)
        if isinstance(queue, dict) and isinstance(queue.get("entries"), list)
        else evidence_report.get("successor_actions")
        if isinstance(evidence_report.get("successor_actions"), list)
        else []
    )
    rows = [
        _activation_burndown_row_from_successor(action)
        for action in successor_actions
        if isinstance(action, dict)
    ]
    rows.sort(
        key=lambda row: (
            _activation_burndown_action_bucket(row),
            -_to_float(row.get("projected_savings_usd")),
            -_to_float(row.get("realized_savings_usd")),
            -_to_int(row.get("sample_count")),
            str(row.get("local_action_family") or ""),
            str(row.get("next_action") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    selected_rows = [row for row in rows if _activation_burndown_selected(row)]
    family_counts: Counter[str] = Counter(str(row.get("local_action_family") or "unknown") for row in rows)
    status_counts: Counter[str] = Counter(str(row.get("successor_status") or "unknown") for row in rows)
    current_status_counts: Counter[str] = Counter(str(row.get("current_status") or "unknown") for row in rows)
    all_blockers = [code for row in rows for code in row.get("blocker_codes") or []]
    top = rows[0] if rows else {}
    top_selected = selected_rows[0] if selected_rows else {}
    result = {
        "schema": ACTIVATION_BURNDOWN_SCHEMA,
        "generated_at": now.isoformat(),
        "source_schema": sanitize_value(plan.get("schema")),
        "source_generated_at": sanitize_value(plan.get("generated_at")),
        "source_report_schema": evidence_report.get("schema"),
        "status": "empty" if not rows else "ranked",
        "summary": {
            "ranked_row_count": len(rows),
            "selected_successor_count": len(selected_rows),
            "successor_action_count": len(successor_actions),
            "non_duplicate_successor_action_count": len({
                str(action.get("fingerprint"))
                for action in successor_actions
                if isinstance(action, dict) and action.get("fingerprint")
            }),
            "represented_local_action_families": sorted(family_counts),
            "local_action_family_counts": [{"value": key, "count": count} for key, count in sorted(family_counts.items())],
            "successor_status_counts": [{"value": key, "count": count} for key, count in sorted(status_counts.items())],
            "current_status_counts": [{"value": key, "count": count} for key, count in sorted(current_status_counts.items())],
            "top_local_action_family": top.get("local_action_family"),
            "top_provider_scope": top.get("provider_scope"),
            "top_current_status": top.get("current_status"),
            "top_successor_status": top.get("successor_status"),
            "top_next_action": top.get("next_action"),
            "top_selected_local_action_family": top_selected.get("local_action_family"),
            "top_selected_provider_scope": top_selected.get("provider_scope"),
            "top_selected_next_action": top_selected.get("next_action"),
            "top_selected_blocker_code": (top_selected.get("blocker_codes") or [None])[0],
            "unique_blocker_codes": sorted({str(code) for code in all_blockers if str(code or "").strip()}),
            "total_projected_savings_usd": round(sum(_to_float(row.get("projected_savings_usd")) for row in rows), 8),
            "total_realized_savings_usd": round(sum(_to_float(row.get("realized_savings_usd")) for row in rows), 8),
        },
        "rows": rows,
        "selected_successor_rows": selected_rows[:20],
        "source_evidence_to_activation_summary": sanitize_value(evidence_report.get("summary") or {}),
        "privacy": _successor_action_privacy() | {
            "read_only": True,
            "policy_files_written": False,
        },
    }
    if isinstance(queue, dict):
        result["next_action_queue_summary"] = sanitize_value(queue.get("summary") or {})
    return sanitize_value(result)


def _merge_precomputed_ledger_context(
    ledger: dict[str, Any],
    precomputed_ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(precomputed_ledger, dict):
        return ledger
    prior_by_fingerprint = {
        str(entry.get("fingerprint") or ""): entry
        for entry in precomputed_ledger.get("entries") or []
        if isinstance(entry, dict) and str(entry.get("fingerprint") or "")
    }
    if not prior_by_fingerprint:
        return ledger
    merged_entries: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for entry in ledger.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        merged = dict(entry)
        fingerprint = str(entry.get("fingerprint") or "")
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        prior = prior_by_fingerprint.get(fingerprint)
        if isinstance(prior, dict):
            for key in (
                "next_action",
                "current_status",
                "state",
                "evidence_schema",
                "cohort_bucket",
                "expected_savings_path",
            ):
                if prior.get(key):
                    merged[key] = sanitize_value(prior.get(key))
            for key in (
                "lifecycle_progressed_from_next_action",
                "lifecycle_progressed_from_evidence_schema",
                "lifecycle_progressed_from_cohort_bucket",
            ):
                if prior.get(key) and not merged.get(key):
                    merged[key] = sanitize_value(prior.get(key))
            for key in ("prior_issue", "issue_status"):
                if prior.get(key):
                    merged[key] = sanitize_value(prior.get(key))
            if prior.get("issue_status") == "closed-issue-seen" and prior.get("prior_issue"):
                merged["issue_status"] = "closed-issue-seen"
        merged_entries.append(merged)
    for fingerprint, prior in prior_by_fingerprint.items():
        if fingerprint not in seen_fingerprints:
            merged_entries.append(sanitize_value(prior))
    merged_ledger = dict(ledger)
    merged_ledger["entries"] = merged_entries
    return _refresh_ledger_summary(merged_ledger)


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

    promotion_report = stats_summary.get("openai_routing_promotion_decision")
    promotion_decision = (
        promotion_report.get("promotion_decision")
        if isinstance(promotion_report, dict) and isinstance(promotion_report.get("promotion_decision"), dict)
        else None
    )
    if promotion_decision is not None:
        decision = str(promotion_decision.get("decision") or "unknown")
        lifecycle = promotion_decision.get("lifecycle") if isinstance(promotion_decision.get("lifecycle"), dict) else {}
        active_outcome = (
            promotion_decision.get("active_local_policy_outcome")
            if isinstance(promotion_decision.get("active_local_policy_outcome"), dict)
            else {}
        )
        outcome_gate = (
            active_outcome.get("outcome_gate")
            if isinstance(active_outcome.get("outcome_gate"), dict)
            else active_outcome.get("active_rule_regression_gate")
            if isinstance(active_outcome.get("active_rule_regression_gate"), dict)
            else {}
        )
        state = (
            "activation-ready"
            if decision == "promote"
            else decision
            if decision in {"active-local-policy", "keep-staged", "keep-blocked"}
            else "blocked"
        )
        return {
            "lever": "routing",
            "state": state,
            "evidence_source": promotion_report.get("schema") or promotion_decision.get("schema"),
            "local_action_family": "routing",
            "next_action": str(promotion_decision.get("next_action") or "review-openai-routing-promotion-decision"),
            "blocker_codes": [str(item) for item in promotion_decision.get("reason_codes") or []],
            "sample_count": _to_int(promotion_decision.get("matched_count")),
            "applied_count": _to_int(lifecycle.get("applied_count")),
            "holdout_count": _to_int(lifecycle.get("holdout_count")),
            "safety_stop_count": _to_int(lifecycle.get("safety_stop_count")),
            "error_count": _to_int(lifecycle.get("error_count")),
            "fallback_count": _to_int(lifecycle.get("fallback_count")),
            "retry_count": _to_int(lifecycle.get("retry_count")),
            "savings_per_1000_calls_usd": _to_float(promotion_decision.get("savings_per_1000_calls_usd")),
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
            "active_rule_regression_gate": sanitize_value(outcome_gate) if outcome_gate else None,
        }

    report = stats_summary.get("pass_through_routing_report")
    if not isinstance(report, dict):
        return None
    buckets = [row for row in report.get("buckets") or [] if isinstance(row, dict)]
    actionable = [row for row in buckets if row.get("actionability") == "actionable"]
    top = (actionable or buckets or [None])[0]
    if top is None:
        return None
    lifecycle = _routing_lifecycle_evidence(top) or {}
    blockers = [str(item) for item in lifecycle.get("blocker_codes") or [] if str(item or "").strip()]
    safety_breakdown = [
        row for row in lifecycle.get("safety_stop_breakdown") or []
        if isinstance(row, dict) and _to_int(row.get("count")) > 0
    ]
    actionability = str(top.get("actionability") or "unknown")
    provider = str(top.get("provider") or "unknown")
    if actionability == "actionable" and safety_breakdown:
        state = "keep-blocked"
        next_action = str(
            lifecycle.get("next_action")
            or safety_breakdown[0].get("next_action")
            or "keep-anthropic-routing-blocked-until-safety-stop-burndown"
        )
    elif actionability == "actionable" and blockers:
        state = "missing-evidence"
        next_action = (
            "activate-anthropic-routing-canary-cohorts"
            if provider == "anthropic"
            else "activate-openai-routing-canary-cohorts"
        )
    elif actionability == "actionable":
        state = "ranked-evidence"
        next_action = "stage-anthropic-routing-canary" if provider == "anthropic" else "stage-openai-routing-canary"
    else:
        state = "no-op" if actionability == "already-cheapest" else "blocked"
        next_action = "keep-routing-no-op-reason"
    safety_stop_count = sum(_to_int(row.get("count")) for row in safety_breakdown)
    applied_missing = _to_int((lifecycle.get("cohort_counts") or {}).get("canary_applied")) <= 0
    holdout_missing = _to_int((lifecycle.get("cohort_counts") or {}).get("canary_holdout")) <= 0
    duplicate_suppression = None
    if provider == "anthropic" and state == "keep-blocked":
        suppression_material = {
            "schema": report.get("schema"),
            "provider": provider,
            "requested_model": top.get("requested_model"),
            "candidate_target_model": top.get("candidate_target_model"),
            "activation_gate": "anthropic-routing-safety-stop-burndown",
        }
        duplicate_suppression = {
            "schema": "tokenclaw.anthropic_routing_activation_issue_duplicate_suppression.v1",
            "reason": "anthropic-routing-safety-stop-burndown-not-cleared",
            "fingerprint": public_id(json.dumps(suppression_material, sort_keys=True), prefix="activation"),
            "suppresses_new_activation_issue": True,
            "suppresses_ready_issue_until": "safety_stop_count_zero_and_applied_holdout_coverage_present",
            "safety_stop_count": safety_stop_count,
            "missing_applied_coverage": applied_missing,
            "missing_holdout_coverage": holdout_missing,
            "metadata_only": True,
            "aggregate_only": True,
        }
    stage = {
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
        "safety_stop_count": safety_stop_count,
        "safety_stop_breakdown": sanitize_value(safety_breakdown),
        "duplicate_suppression": duplicate_suppression,
        "issue_worthy_status": "blocked" if state == "keep-blocked" else None,
        "keep_blocked_reason": sanitize_value(
            lifecycle.get("durable_blocked_reason")
            or (safety_breakdown[0].get("durable_blocked_reason") if safety_breakdown else None)
        ),
        "needed_resolution": sanitize_value(
            [
                "safety_stop_reason_review",
                "safer_threshold_or_executor_guard",
                "rollback_proof",
                *(
                    ["applied_coverage"]
                    if any(row.get("missing_applied_coverage") for row in safety_breakdown)
                    else []
                ),
                *(
                    ["holdout_coverage"]
                    if any(row.get("missing_holdout_coverage") for row in safety_breakdown)
                    else []
                ),
            ]
            if safety_breakdown
            else []
        ),
    }
    return {key: value for key, value in stage.items() if value not in (None, [], "")}


def _cache_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    request_shape_policy_stage = _request_shape_cache_replay_policy_decision_loop_stage(stats_summary)
    if request_shape_policy_stage is not None:
        return request_shape_policy_stage
    impact_stage = _openai_cache_replay_impact_loop_stage(stats_summary)
    if impact_stage is not None:
        return impact_stage
    request_shape_evidence_stage = _request_shape_cache_replay_evidence_loop_stage(stats_summary)
    if request_shape_evidence_stage is not None:
        return request_shape_evidence_stage
    readiness_stage = _openai_cache_replay_readiness_loop_stage(stats_summary)
    if readiness_stage is not None:
        return readiness_stage

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
        remaining_rows = [
            row
            for row in shape_replay.get("remaining_replay_ready_cohorts") or []
            if isinstance(row, dict)
        ]
        skipped_openai = (
            shape_replay.get("skipped_openai_blockers")
            if isinstance(shape_replay.get("skipped_openai_blockers"), dict)
            else {}
        )
        skipped_rows = [
            row
            for row in skipped_openai.get("cohorts") or []
            if isinstance(row, dict)
        ] if isinstance(skipped_openai, dict) else []
        rows = remaining_rows or [row for row in shape_replay.get("cohorts") or [] if isinstance(row, dict)]
        top = rows[0] if rows else {}
        remaining_ready_count = _to_int(summary.get("remaining_replay_ready_cohort_count"))
        remaining_ready_rows = _to_int(summary.get("remaining_replay_ready_rows"))
        remaining_fields_present = (
            "remaining_replay_ready_cohort_count" in summary
            or "remaining_replay_ready_rows" in summary
        )
        if remaining_fields_present:
            ready = remaining_ready_count > 0 or remaining_ready_rows > 0 or bool(top.get("remaining_replay_ready"))
        else:
            ready = _to_int(summary.get("replay_ready_cohort_count")) > 0 or top.get("readiness") == "replay-ready"
        if not ready and skipped_rows:
            top = skipped_rows[0]
        blockers = [str(item) for item in top.get("blockers") or [] if str(item or "").strip()]
        for code in top.get("blocker_codes") or []:
            if str(code or "").strip() and str(code) not in blockers:
                blockers.append(str(code))
        top_blocker = str(summary.get("top_blocker_code") or (blockers[0] if blockers else "cache-replayability-evidence-missing"))
        skipped_summary = (
            skipped_openai.get("summary")
            if isinstance(skipped_openai, dict) and isinstance(skipped_openai.get("summary"), dict)
            else {}
        )
        return {
            "lever": "cache",
            "state": "replay-ready" if ready else "missing-evidence",
            "evidence_source": shape_replay.get("schema"),
            "local_action_family": "cache",
            "next_action": "stage-cache-replay-canary" if ready else "resolve-cache-replayability-blocker",
            "blocker_codes": [] if ready else blockers or [top_blocker],
            "sample_count": _to_int(top.get("row_count") or top.get("sample_count") or summary.get("rows_considered")),
            "projected_hits": _to_int(
                top.get("projected_hits")
                or (summary.get("remaining_projected_hits") if ready else skipped_summary.get("projected_hits"))
                or 0
            ),
            "projected_saved_cost_usd": round(
                _to_float(
                    top.get("projected_savings_usd")
                    or (summary.get("remaining_projected_savings_usd") if ready else skipped_summary.get("projected_savings_usd"))
                    or 0.0
                ),
                8,
            ),
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


def _first_breakdown_value(rows: Any) -> str | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("value") or "").strip():
            return str(row.get("value"))
    return None


def _request_shape_tool_cache_dependency_stages(stats_summary: dict[str, Any]) -> list[dict[str, Any]]:
    shape_signal = stats_summary.get("request_shape_rollup_candidates")
    shape_replay = shape_signal.get("cache_replayability_dry_run") if isinstance(shape_signal, dict) else None
    if not isinstance(shape_replay, dict):
        return []
    report = shape_replay.get("tool_replay_evidence")
    if not isinstance(report, dict):
        return []
    rows = [row for row in report.get("cohorts") or [] if isinstance(row, dict)]
    if not rows:
        return []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    stages: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows[:10]:
        provider_family = str(row.get("provider_family") or "openai").strip() or "openai"
        source_surface = str(row.get("source_surface") or "openai_responses").strip() or "openai_responses"
        endpoint = str(row.get("endpoint") or "responses").strip() or "responses"
        category = str(row.get("category") or "tool-cache").strip() or "tool-cache"
        workflow_phase = str(row.get("workflow_phase") or "tool-cache").strip() or "tool-cache"
        next_action = str(row.get("next_action") or report.get("next_action") or summary.get("top_next_action") or "collect-file-invalidation-evidence").strip()
        blocker_codes = [
            str(item)
            for item in row.get("blocker_codes") or []
            if str(item or "").strip()
        ]
        if not blocker_codes:
            for breakdown in report.get("blocker_breakdown") or []:
                if isinstance(breakdown, dict) and breakdown.get("value"):
                    blocker_codes.append(str(breakdown.get("value")))
        reason = str(row.get("evidence_reason") or row.get("reason") or summary.get("top_blocker_code") or (blocker_codes[0] if blocker_codes else "invalidation-evidence-missing")).strip()
        if reason and reason not in blocker_codes:
            blocker_codes.insert(0, reason)
        decision = row.get("dependency_evidence_decision") if isinstance(row.get("dependency_evidence_decision"), dict) else {}
        decision_value = str(
            decision.get("decision")
            or _first_breakdown_value(report.get("dependency_evidence_decision_breakdown"))
            or "missing-dependency-evidence"
        ).strip()
        evidence_class = str(decision.get("evidence_class") or decision_value or "missing-dependency-evidence").strip()
        evidence_state = str(row.get("evidence_state") or summary.get("top_evidence_state") or "blocked-missing-dependency-evidence").strip()
        dependency_status = str(row.get("dependency_evidence_status") or decision.get("status") or row.get("file_dependency_status") or "missing").strip()
        sample_count = _to_int(row.get("sample_count") or row.get("row_count") or summary.get("affected_rows") or summary.get("sample_count"))
        cohort_bucket = "/".join(
            part for part in (provider_family, source_surface, endpoint, category, workflow_phase) if part
        )
        key = (source_surface, endpoint, category, workflow_phase)
        if key in seen:
            continue
        seen.add(key)
        stages.append(
            {
                "lever": "cache",
                "state": "missing-evidence",
                "evidence_source": report.get("schema") or "tokenclaw.request_shape_tool_cache_replay_evidence.v1",
                "local_action_family": "cache",
                "next_action": next_action,
                "fingerprint_next_action": next_action,
                "fingerprint_evidence_source": report.get("schema") or "tokenclaw.request_shape_tool_cache_replay_evidence.v1",
                "fingerprint_cohort_bucket": sanitize_value(cohort_bucket),
                "blocker_codes": sanitize_value(blocker_codes),
                "sample_count": sample_count,
                "affected_rows": _to_int(summary.get("affected_rows") or sample_count),
                "projected_hits": _to_int(row.get("projected_hits") or summary.get("projected_hits")),
                "projected_saved_usd": round(_to_float(row.get("projected_savings_usd") or summary.get("projected_savings_usd")), 8),
                "cohort_bucket": sanitize_value(cohort_bucket),
                "provider_family": sanitize_value(provider_family),
                "source_surface": sanitize_value(source_surface),
                "endpoint": sanitize_value(endpoint),
                "category": sanitize_value(category),
                "workflow_phase": sanitize_value(workflow_phase),
                "has_tools": bool(row.get("has_tools", True)),
                "dependency_evidence_class": sanitize_value(evidence_class),
                "dependency_evidence_decision": sanitize_value(decision_value),
                "dependency_evidence_status": sanitize_value(dependency_status),
                "dependency_evidence_reason": sanitize_value(reason),
                "evidence_state": sanitize_value(evidence_state),
                "dependency_evidence_review": {
                    "schema": "tokenclaw.openai_tool_cache_dependency_next_action.v1",
                    "status": sanitize_value(dependency_status),
                    "evidence_class": sanitize_value(evidence_class),
                    "decision": sanitize_value(decision_value),
                    "reason": sanitize_value(reason),
                    "next_action": sanitize_value(next_action),
                    "requires_explicit_invalidation_safety_evidence": True,
                    "tool_cache_replay_enabled": False,
                    "streaming_replay_enabled": False,
                    "emits_cache_apply_action": False,
                    "privacy": _candidate_privacy(),
                },
                "blocker_breakdown": sanitize_value(report.get("blocker_breakdown") or []),
                "dependency_evidence_decision_breakdown": sanitize_value(report.get("dependency_evidence_decision_breakdown") or []),
                "evidence_state_breakdown": sanitize_value(report.get("evidence_state_breakdown") or []),
                "next_action_breakdown": sanitize_value(report.get("next_action_breakdown") or []),
                "tool_cache_replay_enabled": bool(row.get("tool_cache_replay_enabled")) if row.get("tool_cache_replay_enabled") is not None else False,
                "streaming_replay_enabled": bool(row.get("streaming_replay_enabled")) if row.get("streaming_replay_enabled") is not None else False,
                "emits_cache_apply_action": bool(row.get("emits_cache_apply_action")) if row.get("emits_cache_apply_action") is not None else False,
                "live_repeat_confirmed": bool(row.get("live_repeat_confirmed")),
                "observed_hit_proof": bool(row.get("observed_hit_proof")),
                "observed_hits": _to_int(row.get("observed_hits") or summary.get("observed_hits")),
                "exact_hit_count": _to_int(row.get("exact_hit_count") or summary.get("exact_hit_count")),
                "tools_present_replay_evidence": bool(row.get("tools_present_replay_evidence", True)),
                "generic_tools_present_blocker_reduced": bool(row.get("generic_tools_present_blocker_reduced", True)),
                "tools_present_rows": _to_int(summary.get("tools_present_rows")),
                "tools_present_replay_evidence_rows": _to_int(summary.get("tools_present_replay_evidence_rows")),
                "generic_tools_present_blocker_reduced_rows": _to_int(summary.get("generic_tools_present_blocker_reduced_rows")),
                "unsafe_tool_call_blocker_rows": _to_int(summary.get("unsafe_tool_call_blocker_rows")),
                "missing_dependency_evidence_rows": _to_int(summary.get("missing_dependency_evidence_rows")),
                "stable_dependency_evidence_rows": _to_int(summary.get("stable_dependency_evidence_rows")),
                "stale_dependency_evidence_rows": _to_int(summary.get("stale_dependency_evidence_rows")),
                "unsafe_dependency_evidence_rows": _to_int(summary.get("unsafe_dependency_evidence_rows")),
                "unknown_dependency_evidence_rows": _to_int(summary.get("unknown_dependency_evidence_rows")),
                "cache_apply_action_count": _to_int(summary.get("cache_apply_action_count")),
                "cache_entries_written": _to_int(summary.get("cache_entries_written")),
                "policy_files_written": False,
                "target_local_rule_file": "cache_rules.yaml",
                "target_local_policy_section": "cache.pattern_rules",
                "privacy": _candidate_privacy(),
            }
        )
    return stages


def _first_cache_replay_candidate(report: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in report.get("candidates") or [] if isinstance(row, dict)]
    if not rows:
        return {}
    def key(row: dict[str, Any]) -> tuple[int, int, int, float]:
        readiness = str(row.get("readiness") or "")
        verdict = str(row.get("verdict") or "")
        return (
            0 if readiness == "replay-ready" else 1,
            0 if verdict in {"widen", "promote"} else 1,
            -_to_int(row.get("actual_hits") or row.get("actual_hit_count")),
            -_to_float(row.get("actual_saved_cost_usd") or row.get("observed_savings_usd")),
        )
    rows.sort(key=key)
    return rows[0]


def _cache_replay_blockers_from_candidate(candidate: dict[str, Any], report: dict[str, Any]) -> list[str]:
    blockers = [str(item) for item in candidate.get("reason_codes") or [] if str(item or "").strip()]
    top_remaining = str(candidate.get("top_remaining_blocker") or "").strip()
    if top_remaining and top_remaining not in blockers:
        blockers.append(top_remaining)
    if blockers:
        return blockers
    for row in report.get("remaining_blocker_breakdown") or []:
        if isinstance(row, dict) and row.get("value"):
            return [str(row.get("value"))]
    return []


def _cache_replay_observed_state(*, actual_hits: int, observed_savings: float, applied: int, holdout: int, blockers: list[str]) -> str:
    if any("safety" in blocker for blocker in blockers):
        return "blocked"
    if actual_hits > 0 or observed_savings > 0:
        return "measured-savings"
    if applied > 0 or holdout > 0:
        return "replay-ready"
    return "missing-evidence"


def _request_shape_cache_replay_evidence_blockers(evidence: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    stale = evidence.get("stale_evidence") if isinstance(evidence.get("stale_evidence"), dict) else {}
    if stale.get("stale"):
        blockers.append(str(stale.get("reason") or "stale-cache-replay-evidence"))
    reason = str(evidence.get("reason") or "").strip()
    status = str(evidence.get("status") or "").strip()
    if status in {"no-canary-policy", "no-request-shape-cache-replay-canary"}:
        blockers.append(reason or status)
    for row in evidence.get("blocker_breakdown") or []:
        if isinstance(row, dict) and row.get("value"):
            blockers.append(str(row.get("value")))
    return list(dict.fromkeys(blocker for blocker in blockers if blocker))


def _request_shape_cache_replay_policy_decision_state(decision: str, promotion_readiness: str) -> str:
    if decision == "widen" or promotion_readiness == "promotion-ready":
        return "measured-savings"
    if decision == "retire-staged-no-repeat" or promotion_readiness == "retire-staged-no-repeat":
        return "retired-no-repeat"
    if decision in {"rollback", "keep-blocked"} or promotion_readiness == "rollback-required":
        return "keep-blocked"
    if decision == "keep-staged":
        return "replay-ready"
    return "replay-ready"


def _cache_replay_warmup_duplicate_suppression(
    *,
    decision_report: dict[str, Any],
    top_decision: dict[str, Any],
    reason: str,
    reason_codes: list[str],
    target_local_rule_file: Any,
    target_local_policy_section: Any,
) -> dict[str, Any]:
    existing = (
        decision_report.get("duplicate_suppression")
        if isinstance(decision_report.get("duplicate_suppression"), dict)
        else top_decision.get("duplicate_suppression")
        if isinstance(top_decision.get("duplicate_suppression"), dict)
        else {}
    )
    suppression = dict(existing)
    suppression.setdefault(
        "schema",
        "tokenclaw.request_shape_cache_replay_warmup_carry_forward_duplicate_suppression.v1",
    )
    suppression.setdefault("reason", reason or "cache-replay-canary-warmup-carry-forward")
    suppression.setdefault("metadata_only", True)
    suppression.setdefault("aggregate_only", True)
    suppression.setdefault("suppresses_generic_cache_replay_activation_issue", True)
    suppression.setdefault("suppresses_generic_replay_ready_issue", True)
    suppression.setdefault("suppresses_new_cache_replay_stage_issue", True)
    suppression.setdefault("suppresses_closed_stage_replay_predecessor_titles", True)
    suppression.setdefault(
        "suppressed_predecessor_next_actions",
        ["stage-cache-replay-canary", "turn-cache-candidate-into-local-replay-evidence"],
    )
    suppression.setdefault(
        "suppressed_predecessor_title_families",
        [
            "Stage cache replay canary from evidence-to-activation ledger",
            "Stage request-shape cache replay cohort",
            "Turn tools-present cache candidate into local replay evidence",
            "Turn missing-observed-cache-hits cache candidate into local replay evidence",
        ],
    )
    if reason_codes:
        suppression.setdefault("reason_codes", reason_codes)
    if target_local_rule_file:
        suppression.setdefault("target_local_rule_file", target_local_rule_file)
    if target_local_policy_section:
        suppression.setdefault("target_local_policy_section", target_local_policy_section)
    return suppression


def _cache_replay_stale_no_traffic_duplicate_suppression(
    *,
    decision_report: dict[str, Any],
    top_decision: dict[str, Any],
    reason: str,
    reason_codes: list[str],
    target_local_rule_file: Any,
    target_local_policy_section: Any,
) -> dict[str, Any]:
    existing = (
        decision_report.get("duplicate_suppression")
        if isinstance(decision_report.get("duplicate_suppression"), dict)
        else top_decision.get("duplicate_suppression")
        if isinstance(top_decision.get("duplicate_suppression"), dict)
        else {}
    )
    suppression = dict(existing)
    suppression["schema"] = "tokenclaw.request_shape_cache_replay_stale_no_traffic_retirement_duplicate_suppression.v1"
    suppression["reason"] = "rollback-stale-no-traffic-retired"
    suppression["metadata_only"] = True
    suppression["aggregate_only"] = True
    suppression["suppresses_generic_cache_replay_activation_issue"] = True
    suppression["suppresses_generic_replay_ready_issue"] = True
    suppression["suppresses_new_cache_replay_stage_issue"] = True
    suppression["suppresses_duplicate_successor_issue"] = True
    suppression["suppresses_closed_stage_replay_predecessor_titles"] = True
    suppression.setdefault(
        "suppressed_predecessor_next_actions",
        [
            "stage-cache-replay-canary",
            "turn-cache-candidate-into-local-replay-evidence",
            "rollback-cache-replay-rule",
        ],
    )
    suppression.setdefault(
        "suppressed_predecessor_title_families",
        [
            "Stage cache replay canary from evidence-to-activation ledger",
            "Stage request-shape cache replay cohort",
            "Turn evidence-older-than-max-age cache candidate into local replay evidence",
            "Keep cache activation successor blocked on evidence-older-than-max-age",
        ],
    )
    if reason:
        suppression["retirement_reason"] = sanitize_value(reason)
    if reason_codes:
        suppression["reason_codes"] = sanitize_value(reason_codes)
    if target_local_rule_file:
        suppression["target_local_rule_file"] = sanitize_value(target_local_rule_file)
    if target_local_policy_section:
        suppression["target_local_policy_section"] = sanitize_value(target_local_policy_section)
    return suppression


def _request_shape_cache_replay_policy_decision_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    decision_report = stats_summary.get("request_shape_cache_replay_policy_decision")
    if not isinstance(decision_report, dict):
        return None
    summary = decision_report.get("summary") if isinstance(decision_report.get("summary"), dict) else {}
    decision = str(decision_report.get("decision") or summary.get("decision") or "").strip()
    if not decision:
        return None

    evidence = stats_summary.get("request_shape_cache_replay_evidence")
    if not isinstance(evidence, dict):
        source_evidence = decision_report.get("source_evidence") if isinstance(decision_report.get("source_evidence"), dict) else {}
        evidence = source_evidence
    evidence_summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
    stale_evidence = evidence.get("stale_evidence") if isinstance(evidence.get("stale_evidence"), dict) else {}
    observed_row_count = _to_int(summary.get("observed_row_count") or evidence_summary.get("observed_row_count"))
    applied_count = _to_int(summary.get("applied_count") or evidence_summary.get("applied_count"))
    holdout_count = _to_int(summary.get("holdout_count") or evidence_summary.get("holdout_count"))
    observed_hits = _to_int(summary.get("observed_hits") or summary.get("exact_hit_count") or evidence_summary.get("observed_hits"))
    observed_savings = _to_float(summary.get("observed_savings_usd") or evidence_summary.get("observed_savings_usd"))
    rollback_required = bool(
        decision == "rollback"
        or summary.get("rollback_required")
        or str(decision_report.get("promotion_readiness") or summary.get("promotion_readiness") or "").strip()
        == "rollback-required"
    )
    rollback_applied_rules = (
        decision_report.get("rollback_applied_rules")
        if isinstance(decision_report.get("rollback_applied_rules"), list)
        else summary.get("rollback_applied_rules")
        if isinstance(summary.get("rollback_applied_rules"), list)
        else []
    )
    rollback_applied = bool(
        decision_report.get("rollback_applied")
        or summary.get("rollback_applied")
        or rollback_applied_rules
        or _to_int(decision_report.get("rollback_applied_rule_count") or summary.get("rollback_applied_rule_count")) > 0
    )
    stale_no_traffic_retirement = bool(
        rollback_required
        and stale_evidence.get("stale")
        and observed_row_count <= 0
        and applied_count <= 0
        and holdout_count <= 0
        and observed_hits <= 0
        and observed_savings <= 0.0
    )
    if (
        observed_row_count <= 0
        and applied_count <= 0
        and holdout_count <= 0
        and not stale_no_traffic_retirement
    ):
        return None
    promotion_readiness = str(
        decision_report.get("promotion_readiness")
        or summary.get("promotion_readiness")
        or decision_report.get("promotion_recommendation")
        or ""
    ).strip()
    reason = str(decision_report.get("reason") or "").strip()
    reason_codes = [
        str(item)
        for item in decision_report.get("reason_codes") or []
        if str(item or "").strip()
    ]
    if reason and reason not in reason_codes:
        reason_codes.insert(0, reason)
    state = (
        "retired-stale-no-traffic"
        if stale_no_traffic_retirement
        else _request_shape_cache_replay_policy_decision_state(decision, promotion_readiness)
    )
    blockers = [] if state == "measured-savings" else reason_codes

    staged = evidence.get("staged_canaries") if isinstance(evidence.get("staged_canaries"), list) else []
    top_canary = staged[0] if staged and isinstance(staged[0], dict) else {}
    shape = top_canary.get("shape") if isinstance(top_canary.get("shape"), dict) else {}
    cohort_bucket = "/".join(
        str(part)
        for part in (shape.get("source_surface"), shape.get("endpoint"), shape.get("category"))
        if part
    ) or "openai-cache-replay"
    observed_sample_count = _to_int(summary.get("observed_row_count") or evidence_summary.get("observed_row_count"))
    projected_sample_count = _to_int(top_canary.get("sample_count"))
    sample_count = observed_sample_count or projected_sample_count or (
        _to_int(summary.get("applied_count")) + _to_int(summary.get("holdout_count"))
    )
    fingerprint_sample_count = projected_sample_count or sample_count
    top_decision = decision_report.get("top_decision") if isinstance(decision_report.get("top_decision"), dict) else {}
    policy_decision_id = str(top_decision.get("decision_id") or "").strip()
    miss_breakdown = (
        decision_report.get("applied_miss_blocker_breakdown")
        if isinstance(decision_report.get("applied_miss_blocker_breakdown"), list)
        else top_decision.get("applied_miss_blocker_breakdown")
        if isinstance(top_decision.get("applied_miss_blocker_breakdown"), list)
        else evidence.get("applied_miss_blocker_breakdown")
        if isinstance(evidence.get("applied_miss_blocker_breakdown"), list)
        else []
    )
    top_miss_reason = (
        summary.get("top_applied_miss_blocker")
        or summary.get("top_blocking_applied_miss_blocker")
        or reason
    )
    observed_hit_blocker = (
        summary.get("observed_hit_blocker")
        or decision_report.get("observed_hit_blocker")
        or summary.get("promotion_blocker")
        or decision_report.get("promotion_blocker")
        or (top_miss_reason if _to_int(summary.get("observed_hits") or evidence_summary.get("observed_hits")) <= 0 else None)
    )
    promotion_blocker = (
        summary.get("promotion_blocker")
        or decision_report.get("promotion_blocker")
        or observed_hit_blocker
        or reason
    )
    target_local_rule_file = summary.get("target_local_rule_file") or top_decision.get("target_local_rule_file")
    target_local_policy_section = summary.get("target_local_policy_section") or top_decision.get("target_local_policy_section")
    duplicate_suppression = (
        _cache_replay_stale_no_traffic_duplicate_suppression(
            decision_report=decision_report,
            top_decision=top_decision,
            reason=reason,
            reason_codes=reason_codes,
            target_local_rule_file=target_local_rule_file,
            target_local_policy_section=target_local_policy_section,
        )
        if stale_no_traffic_retirement
        else
        _cache_replay_warmup_duplicate_suppression(
            decision_report=decision_report,
            top_decision=top_decision,
            reason=reason,
            reason_codes=reason_codes,
            target_local_rule_file=target_local_rule_file,
            target_local_policy_section=target_local_policy_section,
        )
        if decision == "keep-staged" or promotion_readiness == "keep-staged-warmup"
        else decision_report.get("duplicate_suppression")
        if isinstance(decision_report.get("duplicate_suppression"), dict)
        else top_decision.get("duplicate_suppression")
        if isinstance(top_decision.get("duplicate_suppression"), dict)
        else {}
    )
    next_action = str(
        decision_report.get("next_action") or summary.get("next_action") or "review-cache-replay-canary-promotion-readiness"
    )
    if stale_no_traffic_retirement:
        next_action = "retire-stale-cache-replay-successor-no-traffic"

    return {
        "lever": "cache",
        "state": state,
        "evidence_source": decision_report.get("schema"),
        "source_evidence_schema": evidence.get("schema"),
        "local_action_family": "cache",
        "next_action": next_action,
        "fingerprint_next_action": "stage-cache-replay-canary",
        "fingerprint_evidence_source": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
        "fingerprint_cohort_bucket": sanitize_value(f"cache:{_sample_count_bucket(fingerprint_sample_count)}"),
        "blocker_codes": sanitize_value(blockers),
        "sample_count": sample_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "actual_hits": observed_hits,
        "actual_saved_cost_usd": round(observed_savings, 8),
        "miss_count": _to_int(summary.get("miss_count") or evidence_summary.get("miss_count")),
        "warmup_miss_count": _to_int(summary.get("warmup_miss_count") or evidence_summary.get("warmup_miss_count")),
        "exact_hit_count": _to_int(summary.get("exact_hit_count") or evidence_summary.get("exact_hit_count")),
        "miss_reason_breakdown": sanitize_value(miss_breakdown),
        "top_miss_reason": sanitize_value(top_miss_reason),
        "observed_hit_blocker": sanitize_value(observed_hit_blocker),
        "promotion_blocker": sanitize_value(promotion_blocker),
        "bypass_skipped_count": _to_int(summary.get("bypass_count") or evidence_summary.get("bypass_count") or evidence_summary.get("unsupported_shape_count")),
        "projected_hits": _to_int(summary.get("projected_hits") or evidence_summary.get("projected_hits") or top_canary.get("projected_hits")),
        "projected_saved_usd": round(_to_float(summary.get("projected_savings_usd") or evidence_summary.get("projected_savings_usd") or top_canary.get("projected_savings_usd")), 8),
        "cohort_bucket": sanitize_value(cohort_bucket),
        "staged_canary_count": _to_int(summary.get("staged_canary_count") or evidence.get("staged_canary_count")),
        "policy_decision": sanitize_value(decision),
        "policy_decision_id": sanitize_value(policy_decision_id),
        "promotion_decision": sanitize_value(decision_report.get("promotion_decision") or summary.get("promotion_decision")),
        "promotion_readiness": sanitize_value(promotion_readiness),
        "reason": sanitize_value(reason),
        "reason_codes": sanitize_value(reason_codes),
        "promotion_allowed": bool(summary.get("promotion_allowed")),
        "rollback_count": 1 if rollback_required else 0,
        "rollback_required": rollback_required,
        "rollback_applied": rollback_applied,
        "rollback_applied_rule_count": _to_int(
            decision_report.get("rollback_applied_rule_count") or summary.get("rollback_applied_rule_count") or len(rollback_applied_rules)
        ),
        "rollback_applied_rules": sanitize_value(rollback_applied_rules[:5]),
        "stale_no_traffic_retirement": stale_no_traffic_retirement,
        "durable_action_ledger_entry": stale_no_traffic_retirement,
        "issue_worthy_status": "suppressed" if stale_no_traffic_retirement else None,
        "cache_apply_action_count": 0 if stale_no_traffic_retirement else None,
        "cache_entries_written": 0 if stale_no_traffic_retirement else None,
        "emits_cache_apply_action": False if stale_no_traffic_retirement else None,
        "policy_files_written": False if stale_no_traffic_retirement else None,
        "target_local_rule_file": sanitize_value(target_local_rule_file),
        "target_local_policy_section": sanitize_value(target_local_policy_section),
        "duplicate_suppression": sanitize_value(duplicate_suppression),
    }


def _request_shape_cache_replay_evidence_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    evidence = stats_summary.get("request_shape_cache_replay_evidence")
    if not isinstance(evidence, dict):
        return None
    staged_count = _to_int(evidence.get("staged_canary_count"))
    summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
    if staged_count <= 0:
        return None

    applied = _to_int(summary.get("applied_count"))
    holdout = _to_int(summary.get("holdout_count"))
    actual_hits = _to_int(summary.get("observed_hits") or summary.get("exact_hit_count"))
    actual_saved = _to_float(summary.get("observed_savings_usd"))
    blockers = _request_shape_cache_replay_evidence_blockers(evidence)
    stale = evidence.get("stale_evidence") if isinstance(evidence.get("stale_evidence"), dict) else {}
    status = str(evidence.get("status") or "").strip()
    next_action = str(evidence.get("next_action") or "").strip()

    if stale.get("stale"):
        state = "blocked"
        if not next_action:
            next_action = "refresh-cache-replay-canary-evidence"
    elif actual_hits > 0 or actual_saved > 0:
        state = "measured-savings"
        if not next_action:
            next_action = "review-cache-replay-canary-promotion-readiness"
    elif status == "staged-no-traffic" or (applied <= 0 and holdout <= 0):
        state = "canary-staged"
        if not next_action:
            next_action = "collect-cache-replay-canary-traffic"
    else:
        state = _cache_replay_observed_state(
            actual_hits=actual_hits,
            observed_savings=actual_saved,
            applied=applied,
            holdout=holdout,
            blockers=blockers,
        )
        if not next_action:
            next_action = "collect-more-cache-replay-evidence" if state == "replay-ready" else "resolve-cache-replay-impact-blocker"

    top_canary = {}
    staged = evidence.get("staged_canaries") if isinstance(evidence.get("staged_canaries"), list) else []
    if staged and isinstance(staged[0], dict):
        top_canary = staged[0]
    shape = top_canary.get("shape") if isinstance(top_canary.get("shape"), dict) else {}
    cohort_bucket = "/".join(
        str(part)
        for part in (shape.get("source_surface"), shape.get("endpoint"), shape.get("category"))
        if part
    ) or "openai-cache-replay"
    observed_sample_count = _to_int(summary.get("observed_row_count"))
    projected_sample_count = _to_int(top_canary.get("sample_count"))
    sample_count = observed_sample_count or projected_sample_count
    fingerprint_sample_count = projected_sample_count or sample_count

    return {
        "lever": "cache",
        "state": state,
        "evidence_source": evidence.get("schema"),
        "local_action_family": "cache",
        "next_action": next_action,
        "fingerprint_next_action": "stage-cache-replay-canary",
        "fingerprint_evidence_source": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
        "fingerprint_cohort_bucket": sanitize_value(f"cache:{_sample_count_bucket(fingerprint_sample_count)}"),
        "blocker_codes": [] if state in {"measured-savings", "canary-staged", "replay-ready"} else blockers,
        "sample_count": sample_count,
        "applied_count": applied,
        "holdout_count": holdout,
        "actual_hits": actual_hits,
        "actual_saved_cost_usd": round(actual_saved, 8),
        "miss_count": _to_int(summary.get("miss_count")),
        "miss_reason_breakdown": sanitize_value(evidence.get("applied_miss_blocker_breakdown") or evidence.get("miss_reason_breakdown") or []),
        "top_miss_reason": sanitize_value(summary.get("top_applied_miss_blocker") or summary.get("top_miss_reason")),
        "bypass_skipped_count": _to_int(summary.get("bypass_count") or summary.get("unsupported_shape_count")),
        "projected_hits": _to_int(summary.get("projected_hits") or top_canary.get("projected_hits")),
        "projected_saved_usd": round(_to_float(summary.get("projected_savings_usd") or top_canary.get("projected_savings_usd")), 8),
        "cohort_bucket": sanitize_value(cohort_bucket),
        "staged_canary_count": staged_count,
    }


def _openai_cache_replay_impact_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    impact = stats_summary.get("openai_cache_replay_impact")
    if not isinstance(impact, dict):
        return None
    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    observed_rows = _to_int(summary.get("observed_openai_cache_replay_metadata_row_count"))
    if observed_rows <= 0:
        return None

    candidate = _first_cache_replay_candidate(impact)
    applied = _to_int(candidate.get("applied_count") or summary.get("applied_count"))
    holdout = _to_int(candidate.get("holdout_count") or summary.get("holdout_count"))
    actual_hits = _to_int(candidate.get("actual_hits") or candidate.get("actual_hit_count") or summary.get("actual_hits"))
    actual_saved = _to_float(candidate.get("actual_saved_cost_usd") or summary.get("actual_saved_cost_usd"))
    blockers = _cache_replay_blockers_from_candidate(candidate, impact)
    miss_reason_breakdown = (
        candidate.get("miss_reason_breakdown")
        if isinstance(candidate.get("miss_reason_breakdown"), list)
        else impact.get("miss_reason_breakdown")
        if isinstance(impact.get("miss_reason_breakdown"), list)
        else []
    )
    state = _cache_replay_observed_state(
        actual_hits=actual_hits,
        observed_savings=actual_saved,
        applied=applied,
        holdout=holdout,
        blockers=blockers,
    )
    next_action = str(candidate.get("next_action") or "").strip()
    if not next_action:
        if state == "measured-savings":
            next_action = "promote-or-widen-cache-replay-canary"
        elif state == "replay-ready":
            next_action = "collect-more-cache-replay-evidence"
        elif state == "blocked":
            next_action = "resolve-cache-replay-impact-blocker"
        else:
            next_action = "stage-cache-replay-canary"

    return {
        "lever": "cache",
        "state": state,
        "evidence_source": impact.get("schema"),
        "local_action_family": "cache",
        "next_action": next_action,
        "blocker_codes": [] if state == "measured-savings" else blockers,
        "sample_count": _to_int(candidate.get("sample_count") or observed_rows),
        "applied_count": applied,
        "holdout_count": holdout,
        "actual_hits": actual_hits,
        "actual_saved_cost_usd": round(actual_saved, 8),
        "miss_count": _to_int(candidate.get("miss_count") or summary.get("miss_count")),
        "miss_reason_breakdown": sanitize_value(miss_reason_breakdown),
        "top_miss_reason": sanitize_value(candidate.get("top_miss_reason") or summary.get("top_miss_reason")),
        "bypass_skipped_count": _to_int(candidate.get("bypass_skipped_count") or summary.get("bypass_skipped_count")),
        "projected_hits": _to_int(candidate.get("projected_hits") or summary.get("projected_hits")),
        "projected_saved_usd": round(_to_float(candidate.get("projected_saved_usd") or summary.get("projected_saved_usd")), 8),
        "cohort_bucket": sanitize_value(
            "/".join(
                str(part)
                for part in (candidate.get("source_surface"), candidate.get("endpoint"), candidate.get("category"))
                if part
            )
            or "openai-cache-replay"
        ),
    }


def _openai_cache_replay_readiness_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    readiness = stats_summary.get("openai_cache_replay_readiness")
    if not isinstance(readiness, dict):
        return None
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), dict) else {}
    observed_rows = _to_int(summary.get("observed_replay_metadata_rows"))
    if observed_rows <= 0:
        return None
    candidates = [row for row in readiness.get("candidates") or [] if isinstance(row, dict)]
    top = candidates[0] if candidates else {}
    applied = _to_int(top.get("applied_count") or summary.get("applied_count"))
    holdout = _to_int(top.get("holdout_count") or summary.get("holdout_count"))
    actual_saved = _to_float(top.get("observed_savings_usd") or summary.get("observed_savings_usd"))
    blockers = [str(item) for item in top.get("reason_codes") or [] if str(item or "").strip()]
    if not blockers and readiness.get("state_reason"):
        blockers = [str(readiness.get("state_reason"))]
    decision = readiness.get("promotion_decision") if isinstance(readiness.get("promotion_decision"), dict) else {}
    miss_reason_breakdown = (
        decision.get("miss_reason_breakdown")
        if isinstance(decision.get("miss_reason_breakdown"), list)
        else decision.get("applied_miss_blocker_breakdown")
        if isinstance(decision.get("applied_miss_blocker_breakdown"), list)
        else []
    )
    state = _cache_replay_observed_state(
        actual_hits=0,
        observed_savings=actual_saved,
        applied=applied,
        holdout=holdout,
        blockers=blockers,
    )
    return {
        "lever": "cache",
        "state": state,
        "evidence_source": readiness.get("schema"),
        "local_action_family": "cache",
        "next_action": str(top.get("next_action") or "inspect-openai-cache-replay-readiness"),
        "blocker_codes": [] if state == "measured-savings" else blockers,
        "sample_count": _to_int(top.get("sample_count") or observed_rows),
        "applied_count": applied,
        "holdout_count": holdout,
        "actual_saved_cost_usd": round(actual_saved, 8),
        "miss_reason_breakdown": sanitize_value(miss_reason_breakdown),
        "top_miss_reason": sanitize_value((decision.get("summary") or {}).get("top_miss_reason") if isinstance(decision.get("summary"), dict) else None),
        "projected_saved_usd": round(_to_float(top.get("projected_savings_usd") or summary.get("projected_savings_usd")), 8),
        "cohort_bucket": sanitize_value(
            "/".join(str(part) for part in (top.get("endpoint"), top.get("category")) if part)
            or "openai-cache-replay-readiness"
        ),
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
        if top_report.get("report_key") == "request_shape_crunch_activation_evidence":
            progress_state = _request_shape_crunch_progress_state(top_report)
            if progress_state:
                state = progress_state
            next_action = str(
                top_report.get("post_max_rollout_next_action")
                or top_report.get("post_widening_next_action")
                or top_report.get("next_action")
                or "monitor-post-widening-crunch-activation"
            )
        shape_signal = stats_summary.get("request_shape_rollup_candidates")
        shape_summary = shape_signal.get("summary") if isinstance(shape_signal, dict) and isinstance(shape_signal.get("summary"), dict) else {}
        shape_top = shape_signal.get("top_candidate") if isinstance(shape_signal, dict) and isinstance(shape_signal.get("top_candidate"), dict) else {}
        shape_action = str(shape_summary.get("top_next_action") or shape_top.get("next_action") or "").strip()
        shape_family = str(shape_summary.get("top_local_action_family") or shape_top.get("local_action_family") or "").strip()
        if (
            top_report.get("report_key") != "request_shape_crunch_activation_evidence"
            and observed.get("source") == "active_crunch_rule_coverage"
            and shape_family == "crunch"
            and shape_action
        ):
            next_action = shape_action
            readiness = str(shape_top.get("readiness_state") or "").strip()
            if readiness:
                state = readiness
    elif status == "projected-savings-ranked":
        state = "projected-savings"
        next_action = str(top_report.get("next_action") or "produce-crunch-opportunity-measurements")
    elif status == "non-positive-projection":
        state = "no-op"
        next_action = str(top_report.get("next_action") or "inspect-crunch-coverage-and-projection")
    elif status == "missing-crunch-measurement":
        state = "missing-evidence"
        next_action = str(top_report.get("next_action") or "emit-crunch-aggregate-measurement")
    else:
        state = "missing-evidence"
        next_action = "emit-crunch-opportunity-report"
    blockers = [str(item) for item in signal.get("missing_measurements") or [] if str(item or "").strip()]
    if not blockers and top_report.get("no_op_reason"):
        blockers = [str(top_report.get("no_op_reason"))]
    if not blockers and top_report.get("top_blocker"):
        blockers = [str(top_report.get("top_blocker"))]
    stage = {
        "lever": "crunch",
        "state": state,
        "evidence_source": signal.get("schema"),
        "local_action_family": "crunch",
        "next_action": next_action,
        "blocker_codes": blockers,
        "sample_count": _to_int(signal.get("calls")),
        "crunch_savings_usd": round(_to_float(observed.get("crunch_savings_usd")), 8),
        "today_crunch_savings_usd": round(_to_float(observed.get("today_crunch_savings_usd")), 8),
        "projected_saved_usd": round(_to_float(top_report.get("projected_saved_usd")), 8),
    }
    if top_report.get("report_key") == "request_shape_crunch_activation_evidence":
        stage.update(
            {
                "activation_follow_up_evidence_schema": sanitize_value(top_report.get("schema")),
                "applied_count": _to_int(top_report.get("applied_count")),
                "holdout_count": _to_int(top_report.get("holdout_count")),
                "fallback_count": _to_int(top_report.get("fallback_count")),
                "safety_stop_count": _to_int(top_report.get("safety_stop_count")),
                "rollback_count": _to_int(top_report.get("rollback_count")),
                "error_rate_delta": round(_to_float(top_report.get("error_rate_delta")), 6),
                "retry_rate_delta": round(_to_float(top_report.get("retry_rate_delta")), 6),
                "fallback_rate_delta": round(_to_float(top_report.get("fallback_rate_delta")), 6),
                "post_widening_status": sanitize_value(top_report.get("post_widening_status")),
                "post_widening_next_action": sanitize_value(top_report.get("post_widening_next_action")),
                "post_widening_reason_codes": sanitize_value(top_report.get("post_widening_reason_codes")),
                "post_max_rollout_status": sanitize_value(top_report.get("post_max_rollout_status")),
                "post_max_rollout_decision": sanitize_value(top_report.get("post_max_rollout_decision")),
                "post_max_rollout_next_action": sanitize_value(top_report.get("post_max_rollout_next_action")),
                "post_max_rollout_reason_codes": sanitize_value(top_report.get("post_max_rollout_reason_codes")),
                "post_max_rollout_promotion_allowed": bool(top_report.get("post_max_rollout_promotion_allowed")),
                "post_max_rollout_cap_reason": sanitize_value(top_report.get("post_max_rollout_cap_reason")),
                "canary_fraction": round(_to_float(top_report.get("canary_fraction")), 6),
                "max_rollout_fraction": round(_to_float(top_report.get("max_rollout_fraction")), 6),
                "active_rule_count": _to_int(top_report.get("active_rule_count")),
                "widened_rule_count": _to_int(top_report.get("widened_rule_count")),
                "active_rule_ref": sanitize_value(top_report.get("active_rule_ref")),
                "active_rule_source": sanitize_value(top_report.get("active_rule_source")),
                "active_rule_decision_id": sanitize_value(top_report.get("active_rule_decision_id")),
                "active_rule_source_evidence_schema": sanitize_value(top_report.get("active_rule_source_evidence_schema")),
                "target_local_rule_file": sanitize_value(top_report.get("target_local_rule_file")),
                "target_local_policy_section": sanitize_value(top_report.get("target_local_policy_section")),
                "projected_saved_tokens": _to_int(top_report.get("projected_saved_tokens")),
            }
        )
        if isinstance(top_report.get("duplicate_suppression"), dict):
            stage["duplicate_suppression"] = sanitize_value(top_report.get("duplicate_suppression"))
    return stage


def _request_shape_crunch_progress_report(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = stats_summary.get("crunch_savings_signal")
    if not isinstance(signal, dict):
        return None
    reports = []
    top_report = signal.get("top_report") if isinstance(signal.get("top_report"), dict) else None
    if top_report is not None:
        reports.append(top_report)
    reports.extend(row for row in signal.get("reports") or [] if isinstance(row, dict))
    for report in reports:
        if report.get("report_key") != "request_shape_crunch_activation_evidence":
            continue
        if _to_int(report.get("active_rule_count")) <= 0:
            continue
        if _to_int(report.get("applied_count")) > 0 or _to_int(report.get("projected_saved_tokens")) > 0:
            return report
    for report in reports:
        if report.get("report_key") != "active_crunch_rule_coverage":
            continue
        if _to_int(report.get("widened_rule_count")) <= 0:
            continue
        if (
            _to_int(report.get("applied_count")) > 0
            or _to_int(report.get("projected_saved_tokens")) > 0
            or _to_float(report.get("projected_saved_usd")) > 0
        ):
            return report
    for report in reports:
        if report.get("report_key") == "request_shape_crunch_policy_decision":
            return report
    for report in reports:
        if report.get("report_key") != "request_shape_crunch_canary_impact":
            continue
        if _to_int(report.get("candidate_count")) <= 0 and _to_int(report.get("matched_count")) <= 0:
            continue
        return report
    for report in reports:
        if report.get("report_key") != "request_shape_crunch_opportunity":
            continue
        next_action = str(report.get("next_action") or "").strip()
        activation_state = str(report.get("activation_state") or "").strip()
        follow_up_status = str(report.get("follow_up_status") or "").strip()
        if (
            report.get("canary_already_staged")
            or activation_state in {"measurement-required", "canary-staged"}
            or follow_up_status == "canary-staged"
            or next_action == "measure-repeated-context-crunch-canary-impact"
        ):
            return report
    return None


def _request_shape_crunch_progress_state(progress: dict[str, Any]) -> str:
    activation_state = str(progress.get("activation_state") or "").strip()
    if activation_state:
        return sanitize_value(activation_state)
    if progress.get("report_key") == "request_shape_crunch_canary_impact":
        next_action = str(progress.get("next_action") or "").strip()
        if next_action == "widen":
            return "measured-savings"
        if next_action == "rollback":
            return "blocked"
        if progress.get("missing_measurements"):
            return "measurement-required"
        return "measured-savings"
    if progress.get("report_key") == "request_shape_crunch_policy_decision":
        decision = str(progress.get("decision") or progress.get("graduation_decision") or progress.get("next_action") or "").strip()
        if decision == "widen":
            return "measured-savings"
        if decision == "rollback":
            return "blocked"
        if decision == "keep-staged":
            return "canary-staged"
        if decision == "blocked":
            return "blocked"
        return "measured-savings"
    if progress.get("report_key") == "request_shape_crunch_activation_evidence":
        post_max_decision = str(progress.get("post_max_rollout_decision") or "").strip()
        post_max_status = str(progress.get("post_max_rollout_status") or "").strip()
        if post_max_decision == "full-rollout-applied" or post_max_status == "post-max-rollout-full-rollout-applied":
            return "full-rollout-active"
        if _to_int(progress.get("active_rule_count")) > 0 and (
            _to_int(progress.get("applied_count")) > 0 or _to_int(progress.get("projected_saved_tokens")) > 0
        ):
            return "measured-active"
        if progress.get("missing_measurements"):
            return "missing-evidence"
        return "measured-savings"
    if progress.get("report_key") == "active_crunch_rule_coverage":
        if _to_int(progress.get("applied_count")) > 0 or _to_int(progress.get("projected_saved_tokens")) > 0:
            return "measured-active"
        if progress.get("missing_measurements"):
            return "missing-evidence"
        return "projected-savings"
    return "measurement-required"


def _request_shape_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    signal = stats_summary.get("request_shape_rollup_candidates")
    if not isinstance(signal, dict):
        return None
    summary = signal.get("summary") if isinstance(signal.get("summary"), dict) else {}
    top = signal.get("top_candidate") if isinstance(signal.get("top_candidate"), dict) else {}
    local_action_family = str(top.get("local_action_family") or summary.get("top_local_action_family") or "cohort-ranking")
    next_action = str(summary.get("top_next_action") or "emit-request-shape-rollups")
    readiness = str(top.get("readiness_state") or "").strip()
    if readiness in {"activation-ready", "measurement-required", "needs-lifecycle-evidence", "blocked"}:
        state = readiness
    else:
        state = "ranked-evidence" if str(signal.get("status") or "") == "candidates-ranked" and top else "missing-evidence"

    stage: dict[str, Any] = {
        "lever": "request-shape-rollups",
        "state": state,
        "evidence_source": signal.get("source_schema") or signal.get("schema"),
        "local_action_family": local_action_family,
        "next_action": next_action,
        "blocker_codes": [str(item) for item in signal.get("missing_measurements") or [] if str(item or "").strip()]
        or [str(item) for item in top.get("blocker_codes") or [] if str(item or "").strip()],
        "sample_count": _to_int(top.get("sample_count") or top.get("row_count") or summary.get("rows_considered")),
        "projected_saved_usd": round(_to_float(top.get("projected_savings_usd")), 8),
        "projected_saved_tokens": _to_int(top.get("projected_saved_tokens")),
        "ranked_candidate_count": _to_int(summary.get("ranked_candidate_count")),
    }
    progress = _request_shape_crunch_progress_report(stats_summary)
    if (
        local_action_family == "crunch"
        and next_action in {"stage-repeated-context-crunch-canary", "measure-repeated-context-crunch-canary-impact"}
        and progress is not None
    ):
        progress_next_action = str(progress.get("next_action") or "").strip()
        if progress_next_action:
            stage["next_action"] = progress_next_action
        stage["fingerprint_next_action"] = next_action
        progress_state = _request_shape_crunch_progress_state(progress)
        stage["state"] = progress_state
        stage["activation_state"] = progress_state
        stage["activation_mode"] = sanitize_value(progress.get("activation_mode"))
        stage["follow_up_status"] = sanitize_value(progress.get("follow_up_status"))
        stage["canary_already_staged"] = bool(progress.get("canary_already_staged"))
        stage["canary_already_applied"] = bool(progress.get("canary_already_applied"))
        missing = [str(item) for item in progress.get("missing_measurements") or [] if str(item or "").strip()]
        if missing:
            stage["blocker_codes"] = missing
        elif progress.get("no_op_reason"):
            stage["blocker_codes"] = [str(progress.get("no_op_reason"))]
        else:
            stage["blocker_codes"] = []
        stage["activation_follow_up_evidence_schema"] = sanitize_value(progress.get("schema"))
        stage["applied_count"] = _to_int(progress.get("applied_count"))
        stage["holdout_count"] = _to_int(progress.get("holdout_count"))
        stage["fallback_count"] = _to_int(progress.get("fallback_count"))
        stage["safety_stop_count"] = _to_int(progress.get("safety_stop_count"))
        stage["rollback_count"] = _to_int(progress.get("rollback_count"))
        stage["error_rate_delta"] = round(_to_float(progress.get("error_rate_delta")), 6)
        stage["retry_rate_delta"] = round(_to_float(progress.get("retry_rate_delta")), 6)
        stage["fallback_rate_delta"] = round(_to_float(progress.get("fallback_rate_delta")), 6)
        stage["post_widening_status"] = sanitize_value(progress.get("post_widening_status"))
        stage["post_widening_next_action"] = sanitize_value(progress.get("post_widening_next_action"))
        stage["post_widening_reason_codes"] = sanitize_value(progress.get("post_widening_reason_codes"))
        stage["post_max_rollout_status"] = sanitize_value(progress.get("post_max_rollout_status"))
        stage["post_max_rollout_decision"] = sanitize_value(progress.get("post_max_rollout_decision"))
        stage["post_max_rollout_next_action"] = sanitize_value(progress.get("post_max_rollout_next_action"))
        stage["post_max_rollout_reason_codes"] = sanitize_value(progress.get("post_max_rollout_reason_codes"))
        stage["post_max_rollout_promotion_allowed"] = bool(progress.get("post_max_rollout_promotion_allowed"))
        stage["post_max_rollout_cap_reason"] = sanitize_value(progress.get("post_max_rollout_cap_reason"))
        stage["canary_fraction"] = round(_to_float(progress.get("canary_fraction")), 6)
        stage["max_rollout_fraction"] = round(_to_float(progress.get("max_rollout_fraction")), 6)
        if progress.get("report_key") in {"active_crunch_rule_coverage", "request_shape_crunch_activation_evidence"}:
            stage["active_rule_count"] = _to_int(progress.get("active_rule_count"))
            stage["widened_rule_count"] = _to_int(progress.get("widened_rule_count"))
            stage["active_rule_ref"] = sanitize_value(progress.get("active_rule_ref"))
            stage["active_rule_source"] = sanitize_value(progress.get("active_rule_source"))
            stage["active_rule_decision_id"] = sanitize_value(progress.get("active_rule_decision_id"))
            stage["active_rule_source_evidence_schema"] = sanitize_value(progress.get("active_rule_source_evidence_schema"))
            stage["target_local_rule_file"] = sanitize_value(progress.get("target_local_rule_file"))
            stage["target_local_policy_section"] = sanitize_value(progress.get("target_local_policy_section"))
        if isinstance(progress.get("duplicate_suppression"), dict):
            stage["duplicate_suppression"] = sanitize_value(progress.get("duplicate_suppression"))
        stage["projected_saved_usd"] = round(
            _to_float(progress.get("projected_saved_usd") or stage.get("projected_saved_usd")),
            8,
        )
        stage["projected_saved_tokens"] = _to_int(progress.get("projected_saved_tokens") or stage.get("projected_saved_tokens"))
    return stage


def _managed_recommendation_loop_stage(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    health = (
        stats_summary.get("managed_recommendation_health")
        if isinstance(stats_summary.get("managed_recommendation_health"), dict)
        else {}
    )
    if not health:
        return None
    top = health.get("top_omission") if isinstance(health.get("top_omission"), dict) else {}
    if not top:
        return None
    representation = (
        top.get("local_file_backed_representation")
        if isinstance(top.get("local_file_backed_representation"), dict)
        else {}
    )
    represented = bool(representation.get("exists"))
    follow_up_owner = str(top.get("follow_up_owner") or ("local-policy" if represented else "blocked-boundary-review"))
    omitted_reason = str(health.get("omitted_local_action_reason") or top.get("omitted_reason") or "").strip()
    missing = [str(item) for item in health.get("missing_measurements") or [] if str(item or "").strip()]
    blocker_codes = [str(item) for item in top.get("blocker_codes") or [] if str(item or "").strip()]
    if not blocker_codes and omitted_reason:
        blocker_codes = [omitted_reason]
    local_state = str(top.get("local_evidence_state") or "").strip()
    if (
        str(health.get("status") or "") == "missing-managed-recommendation-health-report"
        and local_state in {"missing-evidence", "blocked"}
    ):
        return None
    if represented and follow_up_owner == "local-policy":
        state = local_state if local_state in {"missing-evidence", "blocked"} else "ranked-evidence"
    elif missing or omitted_reason:
        state = "missing-evidence"
    else:
        state = "no-op"
    summary = health.get("summary") if isinstance(health.get("summary"), dict) else {}
    return {
        "lever": "managed-recommendation",
        "state": state,
        "evidence_source": health.get("schema"),
        "local_action_family": sanitize_value(top.get("local_action_family") or "unknown"),
        "next_action": sanitize_value(top.get("next_action") or "emit-managed-recommendation-health-rollup"),
        "blocker_codes": sanitize_value(blocker_codes or missing),
        "sample_count": _to_int(top.get("count") or health.get("calls")),
        "projected_saved_usd": round(_to_float(summary.get("observed_savings_usd")), 8),
        "omitted_reason": sanitize_value(omitted_reason or "unknown"),
        "follow_up_owner": sanitize_value(follow_up_owner),
        "managed_dependency": sanitize_value(summary.get("managed_dependency") or "optional"),
        "local_handoff_reason": sanitize_value(top.get("local_handoff_reason") or ""),
        "local_file_backed_representation": representation,
    }


def _evidence_to_activation_loop(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    stages = [
        stage
        for stage in (
            _routing_loop_stage(stats_summary),
            _cache_loop_stage(stats_summary),
            _crunch_loop_stage(stats_summary),
            _request_shape_loop_stage(stats_summary),
            _managed_recommendation_loop_stage(stats_summary),
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
        "schema": "tokenclaw.evidence_to_activation_savings_loop.v1",
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


def _burndown_state_rank(value: str) -> int:
    if value in {"activation-ready", "replay-ready"}:
        return 0
    if value in {"ranked-evidence", "measured-savings", "projected-savings", "evidence-progress"}:
        return 10
    if value in {"missing-evidence", "blocked", "keep-blocked", "retry-later"}:
        return 20
    if value == "superseded":
        return 25
    return 30


def _burndown_row_from_loop_stage(stage: dict[str, Any]) -> dict[str, Any]:
    blocker_codes = [str(item) for item in stage.get("blocker_codes") or [] if str(item or "").strip()]
    return {
        "lever": sanitize_value(stage.get("lever") or "unknown"),
        "local_action_family": sanitize_value(stage.get("local_action_family") or stage.get("lever") or "unknown"),
        "state": sanitize_value(stage.get("state") or "unknown"),
        "next_action": sanitize_value(stage.get("next_action") or "inspect-local-evidence"),
        "blocker_codes": sanitize_value(blocker_codes),
        "evidence_source": sanitize_value(stage.get("evidence_source")),
        "sample_count": _to_int(stage.get("sample_count")),
        "savings_per_1000_calls_usd": round(_to_float(stage.get("savings_per_1000_calls_usd")), 8),
        "projected_saved_usd": round(
            _to_float(stage.get("projected_saved_usd") or stage.get("projected_saved_cost_usd")),
            8,
        ),
        "owner": "local-policy",
        "_score": (
            -_burndown_state_rank(str(stage.get("state") or ""))
            + _to_float(stage.get("savings_per_1000_calls_usd") or stage.get("projected_saved_cost_usd") or stage.get("projected_saved_usd")) * 1000.0
            + min(_to_int(stage.get("sample_count")), 10000) / 100.0
        ),
    }


def _burndown_row_from_managed_health(health: dict[str, Any]) -> dict[str, Any] | None:
    top = health.get("top_omission") if isinstance(health.get("top_omission"), dict) else {}
    missing = [str(item) for item in health.get("missing_measurements") or [] if str(item or "").strip()]
    omitted_reason = str(health.get("omitted_local_action_reason") or top.get("omitted_reason") or "").strip()
    blocker_codes = missing or ([omitted_reason] if omitted_reason else [])
    if not blocker_codes and _to_int(health.get("calls")) <= 0:
        return None
    return {
        "lever": "managed-recommendation",
        "local_action_family": sanitize_value(top.get("local_action_family") or "unknown"),
        "state": "missing-evidence" if blocker_codes else "ranked-evidence",
        "next_action": sanitize_value(top.get("next_action") or "emit-managed-recommendation-health-rollup"),
        "blocker_codes": sanitize_value(blocker_codes),
        "evidence_source": sanitize_value(health.get("schema")),
        "sample_count": _to_int(top.get("count") or health.get("calls")),
        "savings_per_1000_calls_usd": 0.0,
        "projected_saved_usd": round(_to_float((health.get("summary") or {}).get("observed_savings_usd")) if isinstance(health.get("summary"), dict) else 0.0, 8),
        "owner": sanitize_value(top.get("follow_up_owner") or "local-policy"),
        "_score": -20.0 + min(_to_int(top.get("count") or health.get("calls")), 10000) / 150.0,
    }


def _burndown_row_from_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    reason = str(diagnostic.get("reason") or diagnostic.get("diagnostic_class") or "").strip()
    if not reason:
        return None
    return {
        "lever": sanitize_value(diagnostic.get("source_lever") or "activation-feedback"),
        "local_action_family": "activation-feedback",
        "state": "blocked",
        "next_action": sanitize_value(diagnostic.get("unblock_path") or "resolve-activation-feedback-blocker"),
        "blocker_codes": sanitize_value([reason]),
        "evidence_source": "orchestrator_logs",
        "sample_count": _to_int(diagnostic.get("count") or diagnostic.get("observations")),
        "savings_per_1000_calls_usd": 0.0,
        "projected_saved_usd": 0.0,
        "owner": "local-policy",
        "_score": -20.0 + min(_to_int(diagnostic.get("count") or diagnostic.get("observations")), 10000) / 200.0,
    }


def _burndown_row_from_safety_stop_group(group: dict[str, Any]) -> dict[str, Any] | None:
    blocker = str(group.get("keep_blocked_reason") or group.get("blocker_code") or group.get("safety_stop_reason") or "").strip()
    count = _to_int(group.get("safety_stop_count") or group.get("event_count"))
    if not blocker or count <= 0:
        return None
    needed = [str(item) for item in group.get("needed_resolution") or [] if str(item or "").strip()]
    next_state = str(group.get("next_state") or "keep-blocked").strip()
    if next_state not in {"keep-blocked", "retry-later", "superseded"}:
        next_state = "keep-blocked"
    row = {
        "lever": "activation-feedback",
        "local_action_family": sanitize_value(group.get("action_family") or "activation-feedback"),
        "state": sanitize_value(next_state),
        "next_action": sanitize_value(group.get("next_action") or "review-activation-feedback-safety-stop-and-record-keep-blocked-reason"),
        "blocker_codes": sanitize_value([blocker]),
        "next_state": sanitize_value(next_state),
        "next_state_reason": sanitize_value(group.get("next_state_reason") or blocker),
        "keep_blocked_reason": sanitize_value(group.get("keep_blocked_reason") or blocker),
        "needed_resolution": sanitize_value(needed),
        "evidence_source": "tokenclaw.activation_safety_stop_burndown.v1",
        "sample_count": count,
        "savings_per_1000_calls_usd": 0.0,
        "projected_saved_usd": round(_to_float(group.get("savings_estimate_usd")), 8),
        "owner": "local-policy",
        "_score": -10.0 + min(count, 10000) / 100.0,
    }
    if isinstance(group.get("unblock_criteria"), dict):
        row["unblock_criteria"] = sanitize_value(group.get("unblock_criteria"))
    return row


def _promotion_feedback_family(value: Any) -> str:
    family = str(value or "unknown").strip().lower().replace("_", "-")
    if family in {"cache-replay", "cache"}:
        return "cache"
    if family in {"old-context-summary", "old-context-summarization", "crunch"}:
        return "crunch"
    if family in {"phase-routing", "provider-routing", "routing"}:
        return "routing"
    return sanitize_value(family or "unknown")


def _promotion_feedback_state(status: Any, recommendation: Any = None, rollback_needed: Any = None) -> str:
    status_text = str(status or "").strip().lower().replace("_", "-")
    recommendation_text = str(recommendation or "").strip().lower().replace("_", "-")
    if rollback_needed or status_text in {"rollback-needed", "rollback-required"} or recommendation_text == "rollback":
        return "keep-blocked"
    if status_text in {"safety-stopped", "safety-stop", "keep-blocked", "regression-flagged"}:
        return "keep-blocked"
    if status_text in {"superseded", "closed-issue-seen"}:
        return "superseded"
    if status_text in {"needs-more-samples", "needs-more-evidence", "holdout", "applied", "needs-review"}:
        return "ranked-evidence"
    if status_text in {"measured", "positive", "promoted", "widened"} or recommendation_text in {"promote", "widen"}:
        return "measured-savings"
    return "ranked-evidence"


def _promotion_feedback_next_action(state: str, status: Any, recommendation: Any = None) -> str:
    status_text = str(status or "").strip().lower().replace("_", "-")
    recommendation_text = str(recommendation or "").strip().lower().replace("_", "-")
    if state == "keep-blocked":
        if "safety" in status_text:
            return "review-promotion-safety-stop-and-record-keep-blocked-reason"
        if recommendation_text == "rollback" or "rollback" in status_text:
            return "rollback-local-promotion-or-keep-blocked"
        return "review-promotion-regression-and-keep-blocked"
    if state == "superseded":
        return "suppress-closed-promotion-predecessor"
    if status_text in {"needs-more-samples", "needs-more-evidence", "holdout", "applied"}:
        return "collect-promotion-outcome-holdout-evidence"
    if recommendation_text in {"promote", "widen"}:
        return "widen-local-promotion-from-outcome-feedback"
    return "review-promotion-outcome-feedback"


def _promotion_feedback_summary_count(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return [{"value": value, "count": count} for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def _burndown_rows_from_promotion_feedback(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in feedback.get("entries") or [] if isinstance(entry, dict)]
    if not entries:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        family = _promotion_feedback_family(entry.get("action_family") or entry.get("policy_section"))
        grouped.setdefault(family, []).append(entry)

    rows: list[dict[str, Any]] = []
    for family, family_entries in grouped.items():
        state_counts = Counter(
            _promotion_feedback_state(
                entry.get("status"),
                entry.get("recommendation"),
                entry.get("rollback_needed"),
            )
            for entry in family_entries
        )
        state = sorted(
            state_counts,
            key=lambda value: (state_counts[value], -_burndown_state_rank(value), value),
            reverse=True,
        )[0]
        representative = sorted(
            family_entries,
            key=lambda entry: (
                _promotion_feedback_state(entry.get("status"), entry.get("recommendation"), entry.get("rollback_needed")) == state,
                str(entry.get("created_at") or entry.get("impact_generated_at") or ""),
            ),
            reverse=True,
        )[0]
        applied = sum(_to_int(entry.get("applied_count")) for entry in family_entries)
        holdout = sum(_to_int(entry.get("holdout_count")) for entry in family_entries)
        skipped = sum(_to_int(entry.get("skipped_count")) for entry in family_entries)
        bypassed = sum(_to_int(entry.get("bypassed_count")) for entry in family_entries)
        safety_stops = sum(_to_int(entry.get("safety_stop_count")) for entry in family_entries)
        observed = sum(_to_float(entry.get("observed_savings_usd")) for entry in family_entries)
        projected = sum(_to_float(entry.get("projected_savings_usd")) for entry in family_entries)
        blocker_codes = sorted({
            str(code)
            for entry in family_entries
            for code in [*(entry.get("reason_codes") or []), *(entry.get("warning_codes") or [])]
            if str(code or "").strip()
        })
        if state == "keep-blocked" and not blocker_codes:
            blocker_codes = ["promotion-outcome-review-required"]
        rows.append({
            "lever": family,
            "local_action_family": family,
            "state": state,
            "next_action": _promotion_feedback_next_action(state, representative.get("status"), representative.get("recommendation")),
            "blocker_codes": sanitize_value(blocker_codes),
            "evidence_source": sanitize_value(feedback.get("schema") or "tokenclaw.promotion_outcome_feedback_summary.v1"),
            "source": "promotion-outcome-feedback",
            "entry_count": len(family_entries),
            "sample_count": applied + holdout + skipped + bypassed + safety_stops,
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": skipped,
            "bypassed_count": bypassed,
            "safety_stop_count": safety_stops,
            "observed_savings_usd": round(observed, 8),
            "projected_saved_usd": round(projected, 8),
            "status_counts": _promotion_feedback_summary_count(family_entries, "status"),
            "recommendation_counts": _promotion_feedback_summary_count(family_entries, "recommendation"),
            "owner": "local-policy",
            "_score": (
                -_burndown_state_rank(state)
                + observed * 1000.0
                + projected * 100.0
                + min(applied + holdout + skipped + bypassed + safety_stops, 10000) / 100.0
                + len(family_entries)
            ),
        })
    return rows


def _promotion_feedback_supersedes_loop_row(row: dict[str, Any], promotion_families: set[str]) -> bool:
    if row.get("source") == "promotion-outcome-feedback":
        return False
    family = _promotion_feedback_family(row.get("lever") or row.get("local_action_family"))
    if family not in promotion_families:
        return False
    return str(row.get("state") or "") in {
        "activation-ready",
        "replay-ready",
        "projected-savings",
        "ranked-evidence",
        "missing-evidence",
    }


def build_evidence_to_activation_burndown(
    plan: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a metadata-only activation burn-down report from a research plan."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    clean_plan = sanitize_value(plan)
    evidence = clean_plan.get("evidence") if isinstance(clean_plan.get("evidence"), dict) else {}
    stats_summary = evidence.get("stats_summary") if isinstance(evidence.get("stats_summary"), dict) else {}
    loop = stats_summary.get("evidence_to_activation_loop") if isinstance(stats_summary.get("evidence_to_activation_loop"), dict) else {}

    rows: list[dict[str, Any]] = []
    for stage in loop.get("levers") or []:
        if isinstance(stage, dict):
            rows.append(_burndown_row_from_loop_stage(stage))

    promotion_feedback = stats_summary.get("promotion_outcome_feedback")
    promotion_rows: list[dict[str, Any]] = []
    if isinstance(promotion_feedback, dict):
        promotion_rows = _burndown_rows_from_promotion_feedback(promotion_feedback)
        if promotion_rows:
            promotion_families = {str(row.get("lever")) for row in promotion_rows if row.get("lever")}
            rows = [row for row in rows if not _promotion_feedback_supersedes_loop_row(row, promotion_families)]
            rows.extend(promotion_rows)

    managed = stats_summary.get("managed_recommendation_health")
    if isinstance(managed, dict):
        row = _burndown_row_from_managed_health(managed)
        if row is not None:
            rows.append(row)

    safety_stop_burndown = evidence.get("activation_safety_stop_burndown")
    if isinstance(safety_stop_burndown, dict):
        for group in safety_stop_burndown.get("groups") or []:
            if isinstance(group, dict):
                row = _burndown_row_from_safety_stop_group(group)
                if row is not None:
                    rows.append(row)

    diagnostics = evidence.get("repeated_diagnostics") if isinstance(evidence.get("repeated_diagnostics"), list) else []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        diagnostic_class = str(diagnostic.get("diagnostic_class") or diagnostic.get("reason") or "")
        if diagnostic_class in {"safety-stop", "privacy-blocked", "regression", "high-retry-error-rate"}:
            row = _burndown_row_from_diagnostic(diagnostic)
            if row is not None:
                rows.append(row)

    rows.sort(
        key=lambda item: (
            _to_float(item.get("_score")),
            -_burndown_state_rank(str(item.get("state") or "")),
            _to_int(item.get("sample_count")),
            str(item.get("lever") or ""),
        ),
        reverse=True,
    )
    clean_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        clean = dict(row)
        clean.pop("_score", None)
        clean["rank"] = rank
        clean_rows.append(clean)

    top = clean_rows[0] if clean_rows else {}
    missing_count = sum(1 for row in clean_rows if _loop_missing_state(str(row.get("state") or "")) or str(row.get("state")) == "blocked")
    progressed_count = sum(1 for row in clean_rows if _loop_progress_state(str(row.get("state") or "")))
    activation_ready_count = sum(1 for row in clean_rows if str(row.get("state")) in {"activation-ready", "replay-ready"})
    represented = sorted({str(row.get("lever")) for row in clean_rows if row.get("lever")})
    all_blockers = [code for row in clean_rows for code in row.get("blocker_codes") or []]

    result = {
        "schema": EVIDENCE_TO_ACTIVATION_BURNDOWN_SCHEMA,
        "generated_at": now.isoformat(),
        "source_schema": sanitize_value(clean_plan.get("schema")),
        "source_generated_at": sanitize_value(clean_plan.get("generated_at")),
        "status": loop.get("status") or ("empty" if not clean_rows else "ranked"),
        "summary": {
            "blocker_family_count": len(represented),
            "represented_blocker_families": represented,
            "ranked_blocker_count": len(clean_rows),
            "missing_evidence_count": missing_count,
            "progressed_count": progressed_count,
            "activation_ready_count": activation_ready_count,
            "total_sample_count": sum(_to_int(row.get("sample_count")) for row in clean_rows),
            "top_lever": top.get("lever"),
            "top_state": top.get("state"),
            "top_next_action": top.get("next_action"),
            "top_blocker_code": (top.get("blocker_codes") or [None])[0],
            "unique_blocker_codes": sorted({str(code) for code in all_blockers if str(code or "").strip()}),
        },
        "blockers": clean_rows,
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
    next_action_queue = (
        stats_summary.get("local_activation_next_action_queue")
        if isinstance(stats_summary.get("local_activation_next_action_queue"), dict)
        else build_local_activation_next_action_queue(stats_summary)
    )
    if isinstance(next_action_queue, dict):
        result["next_action_queue"] = next_action_queue
        successor_actions = next_action_queue.get("successor_actions")
        if not isinstance(successor_actions, list):
            successor_actions = build_local_activation_successor_actions(next_action_queue)
        if isinstance(successor_actions, list):
            result["successor_actions"] = successor_actions
            successor_decisions = next_action_queue.get("successor_decisions")
            if not isinstance(successor_decisions, list):
                successor_decisions = build_local_activation_successor_decisions(
                    {"successor_actions": successor_actions}
                )
            result["successor_decisions"] = successor_decisions
            result["summary"]["successor_action_count"] = len(successor_actions)
            result["summary"]["non_duplicate_successor_action_count"] = len({
                str(action.get("fingerprint"))
                for action in successor_actions
                if isinstance(action, dict) and action.get("fingerprint")
            })
            result["summary"]["successor_decision_count"] = len(successor_decisions)
            result["summary"]["non_duplicate_successor_decision_count"] = len({
                str(row.get("source_fingerprint"))
                for row in successor_decisions
                if isinstance(row, dict) and row.get("source_fingerprint")
            })
            queue_summary = (
                next_action_queue.get("summary")
                if isinstance(next_action_queue.get("summary"), dict)
                else {}
            )
            for key in (
                "successor_decision_count",
                "non_duplicate_successor_decision_count",
                "preview_verified_successor_count",
                "preview_required_successor_count",
                "preview_blocked_successor_count",
                "preview_gate_status_counts",
                "preview_gate_decision_counts",
                "preview_agreement_by_local_action_family",
                "preview_top_omitted_reasons",
                "preview_top_no_op_reasons",
            ):
                if key in queue_summary:
                    result["summary"][key] = sanitize_value(queue_summary.get(key))
    return sanitize_value(result)


def _candidate_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "absolute_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
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
    bypassed = _to_int(
        row.get("openai_canary_bypassed_or_disabled_count")
        or row.get("openai_canary_bypassed_count")
        or row.get("canary_bypassed_or_disabled_count")
        or row.get("canary_bypassed_count")
    )
    unknown = _to_int(row.get("openai_canary_unknown_count") or row.get("canary_unknown_count"))
    error_count = _to_int(row.get("openai_canary_error_count") or row.get("canary_error_count"))
    retry_count = _to_int(row.get("openai_canary_retry_count") or row.get("canary_retry_count"))
    fallback_count = _to_int(row.get("openai_canary_fallback_count") or row.get("canary_fallback_count"))
    observed = applied + holdout + safety_stopped + skipped + bypassed + unknown
    stale_raw = row.get("openai_canary_stale_evidence") if row.get("openai_canary_stale_evidence") is not None else row.get("stale_evidence")
    if isinstance(stale_raw, str):
        stale = stale_raw.strip().lower() in {"1", "true", "yes", "stale"}
    else:
        stale = bool(stale_raw)
    latest_observed = row.get("openai_canary_latest_observed_at") or row.get("canary_latest_observed_at")
    latest_dt = _parse_time(latest_observed)
    age_hours = None
    if latest_dt is not None:
        age_hours = round((datetime.now(timezone.utc) - latest_dt).total_seconds() / 3600.0, 3)
        stale = stale or age_hours > 72.0

    blockers: Counter[str] = Counter()
    integrity_warnings: Counter[str] = Counter()
    if observed == 0:
        blockers["missing-canary-lifecycle-evidence"] = sample_count
    if applied == 0:
        blockers["missing-applied-coverage"] = sample_count
    if holdout == 0:
        blockers["missing-holdout-coverage"] = sample_count
        if 0 < observed < OPENAI_MIN_HOLDOUT_VOLUME:
            blockers["insufficient-volume-for-holdout"] = max(observed, sample_count)
    if sample_count > 0 and observed > sample_count:
        integrity_warnings["lifecycle-observed-count-exceeds-matched-count"] = observed
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
        "schema": "tokenclaw.openai_routing_canary_lifecycle_evidence.v1",
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
            "observed_rate": round(min(observed, sample_count) / sample_count, 6) if sample_count else 0.0,
            "applied_rate": round(min(applied, sample_count) / sample_count, 6) if sample_count else 0.0,
            "holdout_rate": round(min(holdout, sample_count) / sample_count, 6) if sample_count else 0.0,
        },
        "error_count": error_count,
        "retry_count": retry_count,
        "fallback_count": fallback_count,
        "latest_observed_at": sanitize_value(latest_observed),
        "stale_evidence": {
            "stale": stale,
            "age_hours": age_hours,
            "max_age_hours": 72.0,
        },
        "blocker_codes": sorted(blockers),
        "blocker_reason_breakdown": [
            {"value": key, "count": value}
            for key, value in sorted(blockers.items(), key=lambda item: (-item[1], item[0]))
        ],
        "integrity_warning_codes": sorted(integrity_warnings),
        "integrity_warning_breakdown": [
            {"value": key, "count": value}
            for key, value in sorted(integrity_warnings.items(), key=lambda item: (-item[1], item[0]))
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


def _aggregate_anthropic_canary_lifecycle_evidence(row: dict[str, Any], *, sample_count: int) -> dict[str, Any]:
    applied = _to_int(
        row.get("anthropic_canary_applied_count")
        or row.get("claude_canary_applied_count")
        or row.get("canary_applied_count")
    )
    holdout = _to_int(
        row.get("anthropic_canary_holdout_count")
        or row.get("claude_canary_holdout_count")
        or row.get("canary_holdout_count")
    )
    safety_stopped = _to_int(
        row.get("anthropic_canary_safety_stopped_count")
        or row.get("claude_canary_safety_stopped_count")
        or row.get("safety_stopped_count")
    )
    skipped = _to_int(
        row.get("anthropic_canary_skipped_count")
        or row.get("claude_canary_skipped_count")
        or row.get("canary_skipped_count")
    )
    bypassed = _to_int(
        row.get("anthropic_canary_bypassed_or_disabled_count")
        or row.get("anthropic_canary_bypassed_count")
        or row.get("claude_canary_bypassed_or_disabled_count")
        or row.get("claude_canary_bypassed_count")
        or row.get("canary_bypassed_or_disabled_count")
        or row.get("canary_bypassed_count")
    )
    unknown = _to_int(
        row.get("anthropic_canary_unknown_count")
        or row.get("claude_canary_unknown_count")
        or row.get("canary_unknown_count")
    )
    error_count = _to_int(
        row.get("anthropic_canary_error_count")
        or row.get("claude_canary_error_count")
        or row.get("canary_error_count")
    )
    retry_count = _to_int(
        row.get("anthropic_canary_retry_count")
        or row.get("claude_canary_retry_count")
        or row.get("canary_retry_count")
    )
    fallback_count = _to_int(
        row.get("anthropic_canary_fallback_count")
        or row.get("claude_canary_fallback_count")
        or row.get("canary_fallback_count")
    )
    observed = applied + holdout + safety_stopped + skipped + bypassed + unknown
    stale_raw = (
        row.get("anthropic_canary_stale_evidence")
        if row.get("anthropic_canary_stale_evidence") is not None
        else row.get("claude_canary_stale_evidence")
        if row.get("claude_canary_stale_evidence") is not None
        else row.get("stale_evidence")
    )
    if isinstance(stale_raw, str):
        stale = stale_raw.strip().lower() in {"1", "true", "yes", "stale"}
    else:
        stale = bool(stale_raw)
    latest_observed = (
        row.get("anthropic_canary_latest_observed_at")
        or row.get("claude_canary_latest_observed_at")
        or row.get("canary_latest_observed_at")
    )
    latest_dt = _parse_time(latest_observed)
    age_hours = None
    if latest_dt is not None:
        age_hours = round((datetime.now(timezone.utc) - latest_dt).total_seconds() / 3600.0, 3)
        stale = stale or age_hours > 72.0

    blockers: Counter[str] = Counter()
    if observed == 0:
        blockers["missing-anthropic-canary-lifecycle-evidence"] = sample_count
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
    safety_stop_breakdown = (
        [item for item in row.get("anthropic_canary_safety_stop_breakdown") or [] if isinstance(item, dict)]
        or _anthropic_canary_safety_stop_breakdown(row)
    )
    durable_blocked_reason = None
    safety_next_action = None
    if safety_stop_breakdown:
        durable_blocked_reason = str(safety_stop_breakdown[0].get("durable_blocked_reason") or "").strip() or None
        safety_next_action = str(safety_stop_breakdown[0].get("next_action") or "").strip() or None

    return {
        "schema": "tokenclaw.anthropic_routing_canary_lifecycle_evidence.v1",
        "status": "matched" if observed else "no-anthropic-canary-metadata",
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
        "latest_observed_at": sanitize_value(latest_observed),
        "stale_evidence": {
            "stale": stale,
            "age_hours": age_hours,
            "max_age_hours": 72.0,
        },
        "blocker_codes": sorted(blockers),
        "blocker_reason_breakdown": [
            {"value": key, "count": value}
            for key, value in sorted(blockers.items(), key=lambda item: (-item[1], item[0]))
        ],
        "safety_stop_breakdown": sanitize_value(safety_stop_breakdown),
        "durable_blocked_reason": sanitize_value(durable_blocked_reason),
        "next_action": sanitize_value(safety_next_action),
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


_OPENAI_CANARY_COUNTER_FIELDS = (
    "openai_canary_applied_count",
    "openai_canary_holdout_count",
    "openai_canary_safety_stopped_count",
    "openai_canary_skipped_count",
    "openai_canary_bypassed_or_disabled_count",
    "openai_canary_unknown_count",
    "openai_canary_error_count",
    "openai_canary_retry_count",
    "openai_canary_fallback_count",
)


_ANTHROPIC_CANARY_COUNTER_FIELDS = (
    "anthropic_canary_applied_count",
    "anthropic_canary_holdout_count",
    "anthropic_canary_safety_stopped_count",
    "anthropic_canary_skipped_count",
    "anthropic_canary_bypassed_or_disabled_count",
    "anthropic_canary_unknown_count",
    "anthropic_canary_error_count",
    "anthropic_canary_retry_count",
    "anthropic_canary_fallback_count",
)


def _pass_through_lifecycle_key(row: dict[str, Any], target_model: str | None = None) -> tuple[str, ...]:
    provider = _routing_lower(row.get("provider")) or "unknown"
    requested = _routing_text(row.get("requested_model")) or "unknown"
    target = _routing_text(target_model or row.get("candidate_target_model") or row.get("target_model") or row.get("routed_model")) or "unknown"
    if provider != "openai":
        return provider, requested, target
    source_surface = _routing_lower(row.get("source_surface") or row.get("surface")) or "unknown"
    endpoint = _routing_lower(row.get("endpoint")) or "unknown"
    category = _routing_lower(row.get("category")) or "unknown"
    phase = _routing_lower(row.get("phase") or row.get("workflow_phase")) or "unknown"
    return provider, requested, target, source_surface, endpoint, category, phase


def _merge_openai_canary_lifecycle_counts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in _OPENAI_CANARY_COUNTER_FIELDS:
        merged[key] = _to_int(base.get(key)) + _to_int(extra.get(key))
    base_latest = _routing_text(base.get("openai_canary_latest_observed_at") or base.get("canary_latest_observed_at"))
    extra_latest = _routing_text(extra.get("openai_canary_latest_observed_at") or extra.get("canary_latest_observed_at"))
    if extra_latest and (not base_latest or extra_latest > base_latest):
        merged["openai_canary_latest_observed_at"] = extra_latest
    elif base_latest:
        merged["openai_canary_latest_observed_at"] = base_latest
    stale = base.get("openai_canary_stale_evidence") if base.get("openai_canary_stale_evidence") is not None else base.get("stale_evidence")
    extra_stale = extra.get("openai_canary_stale_evidence") if extra.get("openai_canary_stale_evidence") is not None else extra.get("stale_evidence")
    if stale or extra_stale:
        merged["openai_canary_stale_evidence"] = bool(stale or extra_stale)
    return merged


def _merge_anthropic_canary_lifecycle_counts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in _ANTHROPIC_CANARY_COUNTER_FIELDS:
        merged[key] = _to_int(base.get(key)) + _to_int(extra.get(key))
    breakdown: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in (
        *(
            item
            for item in base.get("anthropic_canary_safety_stop_breakdown") or []
            if isinstance(item, dict)
        ),
        *(
            item
            for item in extra.get("anthropic_canary_safety_stop_breakdown") or []
            if isinstance(item, dict)
        ),
        *(
            []
            if isinstance(extra.get("anthropic_canary_safety_stop_breakdown"), list)
            else _anthropic_canary_safety_stop_breakdown(extra)
        ),
    ):
        key = (
            str(row.get("reason_code") or "unknown-safety-stop"),
            str(row.get("source_surface") or "unknown"),
            str(row.get("endpoint") or "unknown"),
            str(row.get("category") or "unknown"),
            str(row.get("workflow_phase") or "unknown"),
            str(row.get("expected_local_executor") or "anthropic-routing-rules"),
        )
        if key not in breakdown:
            breakdown[key] = dict(row)
        else:
            current = breakdown[key]
            current["count"] = _to_int(current.get("count")) + _to_int(row.get("count"))
            current["missing_applied_coverage"] = bool(current.get("missing_applied_coverage")) or bool(row.get("missing_applied_coverage"))
            current["missing_holdout_coverage"] = bool(current.get("missing_holdout_coverage")) or bool(row.get("missing_holdout_coverage"))
            current["executor_compatible"] = bool(current.get("executor_compatible")) and bool(row.get("executor_compatible"))
    if breakdown:
        merged["anthropic_canary_safety_stop_breakdown"] = sorted(
            breakdown.values(),
            key=lambda item: (-_to_int(item.get("count")), str(item.get("reason_code") or "")),
        )
    base_latest = _routing_text(
        base.get("anthropic_canary_latest_observed_at")
        or base.get("claude_canary_latest_observed_at")
        or base.get("canary_latest_observed_at")
    )
    extra_latest = _routing_text(
        extra.get("anthropic_canary_latest_observed_at")
        or extra.get("claude_canary_latest_observed_at")
        or extra.get("canary_latest_observed_at")
    )
    if extra_latest and (not base_latest or extra_latest > base_latest):
        merged["anthropic_canary_latest_observed_at"] = extra_latest
    elif base_latest:
        merged["anthropic_canary_latest_observed_at"] = base_latest
    stale = (
        base.get("anthropic_canary_stale_evidence")
        if base.get("anthropic_canary_stale_evidence") is not None
        else base.get("claude_canary_stale_evidence")
        if base.get("claude_canary_stale_evidence") is not None
        else base.get("stale_evidence")
    )
    extra_stale = (
        extra.get("anthropic_canary_stale_evidence")
        if extra.get("anthropic_canary_stale_evidence") is not None
        else extra.get("claude_canary_stale_evidence")
        if extra.get("claude_canary_stale_evidence") is not None
        else extra.get("stale_evidence")
    )
    if stale or extra_stale:
        merged["anthropic_canary_stale_evidence"] = bool(stale or extra_stale)
    return merged


def _has_anthropic_canary_lifecycle_counts(row: dict[str, Any]) -> bool:
    return any(_to_int(row.get(key)) > 0 for key in _ANTHROPIC_CANARY_COUNTER_FIELDS) or bool(
        row.get("anthropic_canary_latest_observed_at")
        or row.get("claude_canary_latest_observed_at")
        or row.get("canary_latest_observed_at")
    )


def _subtract_anthropic_canary_lifecycle_counts(total: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
    remaining = dict(total)
    for key in _ANTHROPIC_CANARY_COUNTER_FIELDS:
        remaining[key] = max(0, _to_int(total.get(key)) - _to_int(own.get(key)))
    own_safety = _to_int(
        own.get("anthropic_canary_safety_stopped_count")
        or own.get("claude_canary_safety_stopped_count")
        or own.get("safety_stopped_count")
    )
    if own_safety > 0 and _to_int(remaining.get("anthropic_canary_safety_stopped_count")) <= 0:
        remaining["anthropic_canary_safety_stop_breakdown"] = []
    return remaining


def _routing_lifecycle_evidence(bucket: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("openai_canary_lifecycle_evidence", "anthropic_canary_lifecycle_evidence"):
        lifecycle = bucket.get(key)
        if isinstance(lifecycle, dict):
            return lifecycle
    return None


def _anthropic_canary_safety_stop_reason(row: dict[str, Any]) -> str:
    skip_reason = _routing_skip_reason(row).lower()
    category = _routing_text(row.get("category")).lower()
    phase = _routing_text(row.get("phase") or row.get("workflow_phase")).lower()
    source_surface = _routing_text(row.get("source_surface") or row.get("surface")).lower()
    endpoint = _routing_text(row.get("endpoint")).lower()
    if "thinking" in skip_reason or phase == "thinking":
        return "thinking-routing-guard"
    if "stream" in skip_reason:
        return "unsupported-streaming-shape"
    if "tool-protocol" in skip_reason or "unsafe" in skip_reason:
        return "unsafe-tool-call-context"
    if source_surface in {"", "unknown"} or endpoint in {"", "unknown"}:
        return "aggregate-cohort-needs-narrower-surface"
    if category not in {"tool-result", "tool-light", "short-completion", "summary"}:
        return "category-not-enabled"
    return "local-canary-safety-stop"


def _anthropic_canary_safety_stop_breakdown(row: dict[str, Any]) -> list[dict[str, Any]]:
    safety_stopped = _to_int(
        row.get("anthropic_canary_safety_stopped_count")
        or row.get("claude_canary_safety_stopped_count")
        or row.get("safety_stopped_count")
    )
    if safety_stopped <= 0:
        return []
    applied = _to_int(
        row.get("anthropic_canary_applied_count")
        or row.get("claude_canary_applied_count")
        or row.get("canary_applied_count")
    )
    holdout = _to_int(
        row.get("anthropic_canary_holdout_count")
        or row.get("claude_canary_holdout_count")
        or row.get("canary_holdout_count")
    )
    reason = _anthropic_canary_safety_stop_reason(row)
    executor_compatible = reason in {"local-canary-safety-stop"}
    source_surface = _routing_text(row.get("source_surface") or row.get("surface")) or "unknown"
    endpoint = _routing_text(row.get("endpoint")) or "unknown"
    category = _routing_text(row.get("category")) or "unknown"
    phase = _routing_text(row.get("phase") or row.get("workflow_phase")) or "unknown"
    durable_reason = f"anthropic-routing-safety-stop-{reason}-keep-blocked"
    return [
        {
            "reason_code": sanitize_value(reason),
            "count": safety_stopped,
            "source_surface": sanitize_value(source_surface),
            "endpoint": sanitize_value(endpoint),
            "category": sanitize_value(category),
            "workflow_phase": sanitize_value(phase),
            "expected_local_executor": "anthropic-routing-rules",
            "executor_compatible": executor_compatible,
            "missing_applied_coverage": applied <= 0,
            "missing_holdout_coverage": holdout <= 0,
            "durable_blocked_reason": sanitize_value(durable_reason),
            "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
        }
    ]


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
        "anthropic_canary_lifecycle_evidence": _aggregate_anthropic_canary_lifecycle_evidence(row, sample_count=count)
        if provider == "anthropic" and target_model
        else None,
        "anthropic_canary_lifecycle_related_only": bool(row.get("_anthropic_canary_lifecycle_related_only")),
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
    routed_openai_lifecycle_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    routed_anthropic_lifecycle_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in routing_rows:
        if not isinstance(raw, dict):
            continue
        count = _to_int(raw.get("c") or raw.get("count"))
        if count <= 0:
            continue
        provider = _routing_lower(raw.get("provider")) or "unknown"
        requested = _routing_text(raw.get("requested_model")) or "unknown"
        routed = raw.get("routed_model")
        routed_text = _routing_text(routed)
        target_model, _, _ = _routing_candidate_target(
            provider,
            requested,
            _routing_text(raw.get("category")) or "unknown",
            _routing_text(raw.get("phase") or raw.get("workflow_phase")) or "unknown",
        )
        if provider == "anthropic" and target_model and _has_anthropic_canary_lifecycle_counts(raw):
            key = _pass_through_lifecycle_key(raw, target_model=target_model)
            routed_anthropic_lifecycle_by_key[key] = _merge_anthropic_canary_lifecycle_counts(
                routed_anthropic_lifecycle_by_key.get(key, {}),
                dict(raw),
            )
            continue
        if not routed_text or requested == routed_text:
            continue
        if target_model != routed_text:
            continue
        lifecycle_row = dict(raw)
        if _to_int(lifecycle_row.get("openai_canary_applied_count")) <= 0 and provider == "openai":
            lifecycle_row["openai_canary_applied_count"] = count
        if _to_int(lifecycle_row.get("anthropic_canary_applied_count")) <= 0 and provider == "anthropic":
            lifecycle_row["anthropic_canary_applied_count"] = count
        key = _pass_through_lifecycle_key(raw, target_model=target_model)
        if provider == "openai":
            routed_openai_lifecycle_by_key[key] = _merge_openai_canary_lifecycle_counts(
                routed_openai_lifecycle_by_key.get(key, {}),
                lifecycle_row,
            )
        elif provider == "anthropic":
            routed_anthropic_lifecycle_by_key[key] = _merge_anthropic_canary_lifecycle_counts(
                routed_anthropic_lifecycle_by_key.get(key, {}),
                lifecycle_row,
            )

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
        classified_input = dict(raw)
        target_model, _, _ = _routing_candidate_target(
            _routing_lower(raw.get("provider")) or "unknown",
            _routing_text(raw.get("requested_model")) or "unknown",
            _routing_text(raw.get("category")) or "unknown",
            _routing_text(raw.get("phase") or raw.get("workflow_phase")) or "unknown",
        )
        key = _pass_through_lifecycle_key(raw, target_model=target_model)
        openai_lifecycle_extra = routed_openai_lifecycle_by_key.get(key)
        if openai_lifecycle_extra:
            classified_input = _merge_openai_canary_lifecycle_counts(classified_input, openai_lifecycle_extra)
        anthropic_lifecycle_extra = routed_anthropic_lifecycle_by_key.get(key)
        if anthropic_lifecycle_extra:
            has_own_anthropic_lifecycle = _has_anthropic_canary_lifecycle_counts(raw)
            if has_own_anthropic_lifecycle:
                anthropic_lifecycle_extra = _subtract_anthropic_canary_lifecycle_counts(anthropic_lifecycle_extra, raw)
            else:
                classified_input["_anthropic_canary_lifecycle_related_only"] = True
            classified_input = _merge_anthropic_canary_lifecycle_counts(classified_input, anthropic_lifecycle_extra)
        buckets.append(_classify_pass_through_bucket(classified_input))

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
        lifecycle = bucket.get("anthropic_canary_lifecycle_evidence")
        if isinstance(lifecycle, dict):
            if bucket.get("anthropic_canary_lifecycle_related_only"):
                continue
            cohorts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
            lifecycle_totals["anthropic_canary_applied"] += _to_int(cohorts.get("canary_applied"))
            lifecycle_totals["anthropic_canary_holdout"] += _to_int(cohorts.get("canary_holdout"))
            lifecycle_totals["anthropic_safety_stopped"] += _to_int(cohorts.get("safety_stopped"))
            lifecycle_totals["anthropic_error"] += _to_int(lifecycle.get("error_count"))
            lifecycle_totals["anthropic_retry"] += _to_int(lifecycle.get("retry_count"))
            lifecycle_totals["anthropic_fallback"] += _to_int(lifecycle.get("fallback_count"))
    ranked = []
    for rank, bucket in enumerate(buckets[: max(1, limit)], start=1):
        item = dict(bucket)
        item["rank"] = rank
        ranked.append(item)
    top = ranked[0] if ranked else {}
    return {
        "schema": "tokenclaw.pass_through_routing_activation_candidates.v1",
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
            "anthropic_canary_applied_count": lifecycle_totals["anthropic_canary_applied"],
            "anthropic_canary_holdout_count": lifecycle_totals["anthropic_canary_holdout"],
            "anthropic_canary_safety_stopped_count": lifecycle_totals["anthropic_safety_stopped"],
            "anthropic_canary_error_count": lifecycle_totals["anthropic_error"],
            "anthropic_canary_retry_count": lifecycle_totals["anthropic_retry"],
            "anthropic_canary_fallback_count": lifecycle_totals["anthropic_fallback"],
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


def _openai_routing_promotion_decision_from_pass_through(report: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    target_buckets = [
        bucket
        for bucket in report.get("buckets") or []
        if isinstance(bucket, dict)
        and str(bucket.get("provider") or "").lower() == "openai"
        and str(bucket.get("requested_model") or "").lower() == "gpt-5.4"
        and str(bucket.get("candidate_target_model") or bucket.get("target_model") or "").lower() == "gpt-5.4-mini"
        and str(bucket.get("source_surface") or "").lower() == "openai_responses"
        and str(bucket.get("endpoint") or "").lower() == "responses"
        and str(bucket.get("category") or "").lower() == "tool-light"
    ]
    if not target_buckets:
        return None

    cohort_counts = {
        "canary_applied": 0,
        "canary_holdout": 0,
        "safety_stopped": 0,
        "skipped": 0,
        "bypassed_or_disabled": 0,
        "unknown": 0,
    }
    matched_count = 0
    observed_count = 0
    error_count = 0
    fallback_count = 0
    retry_count = 0
    latest_observed_at: str | None = None
    stale_evidence_count = 0
    skipped_reason_counts: Counter[str] = Counter()
    unknown_reason_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    projected_savings_usd = 0.0
    candidate_ids: list[str] = []

    for bucket in target_buckets:
        count = _to_int(bucket.get("sample_count") or bucket.get("matched_count"))
        matched_count += count
        candidate_id = str(bucket.get("candidate_id") or f"pass-through-openai-gpt54-tool-light-{len(candidate_ids) + 1}")
        candidate_ids.append(candidate_id)
        savings_per_1000 = _to_float(bucket.get("estimated_savings_per_1000_calls_usd"))
        projected_savings_usd += (savings_per_1000 / 1000.0) * count

        lifecycle = bucket.get("openai_canary_lifecycle_evidence")
        if not isinstance(lifecycle, dict):
            continue
        observed_count += _to_int(lifecycle.get("observed_count"))
        counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
        for key in cohort_counts:
            cohort_counts[key] += _to_int(counts.get(key))
        error_count += _to_int(lifecycle.get("error_count"))
        fallback_count += _to_int(lifecycle.get("fallback_count"))
        retry_count += _to_int(lifecycle.get("retry_count"))
        latest = lifecycle.get("latest_observed_at")
        if isinstance(latest, str) and (latest_observed_at is None or latest > latest_observed_at):
            latest_observed_at = latest
        stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
        if stale.get("stale"):
            stale_evidence_count += _to_int(lifecycle.get("observed_count"))
        for item in lifecycle.get("skipped_reason_breakdown") or []:
            if isinstance(item, dict):
                skipped_reason_counts[str(item.get("value") or "unknown")] += _to_int(item.get("count"), 1)
        for item in lifecycle.get("unknown_reason_breakdown") or []:
            if isinstance(item, dict):
                unknown_reason_counts[str(item.get("value") or "unknown")] += _to_int(item.get("count"), 1)
        for code in lifecycle.get("blocker_codes") or []:
            code_text = str(code or "").strip()
            if code_text:
                blocker_counts[code_text] += max(1, count)

    applied = cohort_counts["canary_applied"]
    holdout = cohort_counts["canary_holdout"]
    safety = cohort_counts["safety_stopped"]
    unknown = cohort_counts["unknown"]
    if applied <= 0 and holdout <= 0:
        return None
    savings_per_1000 = round((projected_savings_usd / matched_count) * 1000.0, 6) if matched_count else 0.0

    if matched_count <= 0:
        blocker_counts["insufficient-samples"] += 1
    if observed_count <= 0:
        blocker_counts["missing-canary-lifecycle-evidence"] += max(1, matched_count)
    if applied <= 0:
        blocker_counts["missing-applied-coverage"] += max(1, matched_count)
    if holdout <= 0:
        blocker_counts["missing-holdout-coverage"] += max(1, matched_count)
    if safety:
        blocker_counts["safety-stop-observed"] += safety
    if error_count:
        blocker_counts["error-observed"] += error_count
    if fallback_count:
        blocker_counts["fallback-observed"] += fallback_count
    if retry_count:
        blocker_counts["retry-observed"] += retry_count
    if stale_evidence_count:
        blocker_counts["stale-evidence"] += stale_evidence_count
    if savings_per_1000 <= 0:
        blocker_counts["non-positive-estimated-savings"] += max(1, matched_count)

    hard_blockers = [
        reason
        for reason in sorted(blocker_counts)
        if reason not in {
            "missing-canary-lifecycle-evidence",
            "missing-applied-coverage",
            "missing-holdout-coverage",
        }
    ]
    target = {
        "provider": "openai",
        "source_surface": "openai_responses",
        "endpoint": "responses",
        "category": "tool-light",
        "requested_model": "gpt-5.4",
        "target_model": "gpt-5.4-mini",
        "required_local_executor": "openai-routing-canary",
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
    }
    active_local_policy_rule = None
    try:
        from tokenclaw.openai_routing_report import _active_openai_local_policy_rule

        active_local_policy_rule = _active_openai_local_policy_rule(target)
    except Exception:
        active_local_policy_rule = None
    disabled_local_policy_rule = _disabled_openai_local_policy_rule(target)
    if disabled_local_policy_rule is not None:
        blocker_counts[str(disabled_local_policy_rule.get("reason") or "disabled-local-policy")] += max(1, matched_count)
    hard_blockers = [
        reason
        for reason in sorted(blocker_counts)
        if reason not in {
            "missing-canary-lifecycle-evidence",
            "missing-applied-coverage",
            "missing-holdout-coverage",
        }
    ]

    if applied > 0 and holdout > 0 and not blocker_counts and savings_per_1000 > 0 and active_local_policy_rule is not None:
        decision = "active-local-policy"
        next_action = "measure-openai-routing-rule-outcomes"
        reason = "matching-openai-routing-rule-active-in-local-policy"
    elif applied > 0 and holdout > 0 and not blocker_counts and savings_per_1000 > 0:
        decision = "promote"
        next_action = "draft-openai-routing-rule"
        reason = "promotion-ready"
    elif hard_blockers:
        decision = "keep-blocked"
        next_action = "review-openai-routing-canary-blockers"
        reason = hard_blockers[0]
    else:
        decision = "keep-staged"
        next_action = "collect-openai-routing-canary-evidence"
        reason = sorted(blocker_counts)[0] if blocker_counts else "insufficient-promotion-evidence"

    breakdown = [{"value": key, "count": value} for key, value in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))]
    skipped_breakdown = [{"value": key, "count": value} for key, value in sorted(skipped_reason_counts.items(), key=lambda item: (-item[1], item[0]))]
    unknown_breakdown = [{"value": key, "count": value} for key, value in sorted(unknown_reason_counts.items(), key=lambda item: (-item[1], item[0]))]
    privacy = {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }
    candidate_set = {
        "schema": "tokenclaw.openai_routing_candidate_set_metadata.v1",
        "candidate_count": len(candidate_ids),
        "candidate_fingerprint": public_id("|".join(sorted(candidate_ids)), prefix="openai-routing-candidates"),
        "candidate_ids_included": False,
        "individual_candidate_ids_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }
    active_outcome = None
    if decision == "active-local-policy" and active_local_policy_rule is not None:
        stale_age_hours = None
        latest_dt = _parse_time(latest_observed_at)
        if latest_dt is not None:
            stale_age_hours = round((datetime.now(timezone.utc) - latest_dt).total_seconds() / 3600.0, 3)
        stale_evidence = {
            "metadata_only": True,
            "aggregate_only": True,
            "stale": bool(stale_evidence_count),
            "age_hours": stale_age_hours,
            "max_age_hours": 72.0,
            "status": "stale" if stale_evidence_count else "fresh",
        }
        outcome_gate = _openai_routing_active_policy_outcome_gate(
            applied_count=applied,
            holdout_count=holdout,
            skipped_count=cohort_counts["skipped"],
            unknown_count=unknown,
            error_count=error_count,
            fallback_count=fallback_count,
            retry_count=retry_count,
            safety_stop_count=safety,
            stale_evidence=stale_evidence,
            savings_per_1000_calls_usd=savings_per_1000,
            target_local_policy_section="routing.rules",
            target_local_rule_file="routing_rules.yaml",
        )
        active_outcome = {
            "schema": "tokenclaw.openai_routing_active_local_policy_outcome.v1",
            "status": "active-local-policy",
            "state": "active-local-policy",
            "current_status": "applied",
            "outcome": outcome_gate["state"],
            "outcome_decision": outcome_gate["state"],
            "decision": decision,
            "measurement_next_action": next_action,
            "next_action": outcome_gate["next_action"],
            "deterministic_next_action": outcome_gate["deterministic_next_action"],
            "reason_codes": outcome_gate["reason_codes"],
            "gate_passed": outcome_gate["gate_passed"],
            "reason": reason,
            "local_action_family": "routing",
            "target": target,
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
            "matched_count": matched_count,
            "candidate_count": len(target_buckets),
            "candidate_set": candidate_set,
            "candidate_ids_included": False,
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": cohort_counts["skipped"],
            "unknown_count": unknown,
            "safety_stop_count": safety,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "regression_counters": {
                "schema": "tokenclaw.openai_routing_active_local_policy_regression_counters.v1",
                "error_count": error_count,
                "fallback_count": fallback_count,
                "retry_count": retry_count,
                "safety_stop_count": safety,
                "stale_evidence_count": stale_evidence_count,
                "rollback_count": 0,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "outcome_gate": outcome_gate,
            "active_rule_regression_gate": outcome_gate,
            "coverage": {
                "schema": "tokenclaw.openai_routing_active_local_policy_coverage.v1",
                "matched_count": matched_count,
                "observed_count": observed_count,
                "applied_count": applied,
                "holdout_count": holdout,
                "applied_rate": round(applied / matched_count, 6) if matched_count else 0.0,
                "holdout_rate": round(holdout / matched_count, 6) if matched_count else 0.0,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "latest_observed_at": latest_observed_at,
            "evidence_age_hours": stale_age_hours,
            "stale_evidence": stale_evidence,
            "savings_per_1000_calls_usd": savings_per_1000,
            "projected_savings_usd": round(projected_savings_usd, 6),
            "expected_savings_path": "Measure post-apply outcomes for the active local OpenAI routing rule.",
            "privacy": privacy,
        }
    decision_payload = {
        "schema": "tokenclaw.openai_routing_promotion_decision.v1",
        "decision": decision,
        "promotion_ready": decision == "promote",
        "next_action": next_action,
        "reason": reason,
        "reason_codes": sorted(blocker_counts),
        "blocker_reason_breakdown": breakdown,
        "target": target,
        "candidate_count": len(target_buckets),
        "candidate_set": candidate_set,
        "candidate_ids_included": False,
        "matched_count": matched_count,
        "blocked_count": matched_count if blocker_counts else 0,
        "projected_savings_usd": round(projected_savings_usd, 6),
        "savings_per_1000_calls_usd": savings_per_1000,
        "lifecycle": {
            "schema": "tokenclaw.openai_routing_canary_lifecycle_evidence.v1",
            "status": "matched" if observed_count else "no-openai-canary-metadata",
            "observed_count": observed_count,
            "cohort_counts": cohort_counts,
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": cohort_counts["skipped"],
            "unknown_count": unknown,
            "safety_stop_count": safety,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "latest_observed_at": latest_observed_at,
            "skipped_reason_breakdown": skipped_breakdown,
            "unknown_reason_breakdown": unknown_breakdown,
        },
        "active_local_policy_rule": active_local_policy_rule,
        "disabled_local_policy_rule": disabled_local_policy_rule,
        "active_local_policy_outcome": active_outcome,
        "privacy": privacy,
    }
    return {
        "schema": "tokenclaw.openai_routing_promotion_decision_report.v1",
        "source_report_schema": report.get("schema"),
        "target": decision_payload["target"],
        "decision": decision,
        "promotion_ready": decision_payload["promotion_ready"],
        "summary": {
            "decision_count": 1,
            "promote_count": 1 if decision == "promote" else 0,
            "active_local_policy_count": 1 if decision == "active-local-policy" else 0,
            "keep_staged_count": 1 if decision == "keep-staged" else 0,
            "keep_blocked_count": 1 if decision == "keep-blocked" else 0,
            "matched_count": matched_count,
            "blocked_count": decision_payload["blocked_count"],
            "candidate_count": len(target_buckets),
            "active_local_policy_outcome_count": 1 if active_outcome else 0,
            "active_local_policy_outcome_decision": active_outcome.get("outcome_decision") if active_outcome else None,
            "active_local_policy_next_action": active_outcome.get("deterministic_next_action") if active_outcome else None,
            "active_local_policy_gate_passed": active_outcome.get("gate_passed") if active_outcome else None,
            "active_local_policy_reason_codes": active_outcome.get("reason_codes") if active_outcome else [],
            "active_local_policy_evidence_age_hours": active_outcome.get("evidence_age_hours") if active_outcome else None,
            "candidate_ids_included": False,
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": cohort_counts["skipped"],
            "unknown_count": unknown,
            "safety_stop_count": safety,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "next_action": next_action,
            "reason": reason,
            "reason_codes": sorted(blocker_counts),
            "blocker_reason_breakdown": breakdown,
            "skipped_reason_breakdown": skipped_breakdown,
            "unknown_reason_breakdown": unknown_breakdown,
            "savings_per_1000_calls_usd": savings_per_1000,
            "projected_savings_usd": round(projected_savings_usd, 6),
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
        },
        "promotion_decision": decision_payload,
        "decisions": [decision_payload],
        "active_local_policy_outcomes": [active_outcome] if active_outcome else [],
        "privacy": privacy | {"basis": "sanitized pass-through routing lifecycle evidence only"},
    }


def _disabled_openai_local_policy_rule(target: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from tokenclaw import router as router_module
    except Exception:
        return None

    requested_model = str(target.get("requested_model") or "").lower()
    target_model = str(target.get("target_model") or "").lower()
    source_surface = str(target.get("source_surface") or "").lower()
    endpoint = str(target.get("endpoint") or "").lower()
    category = str(target.get("category") or "").lower()

    for rule in getattr(router_module, "ROUTING_RULES", []):
        if not isinstance(rule, dict) or rule.get("enabled") is not False:
            continue
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        if str(action.get("route_to") or "").lower() != target_model:
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        if str(conditions.get("provider") or "openai").lower() != "openai":
            continue
        model_pattern = str(conditions.get("model_pattern") or "").lower()
        if model_pattern and model_pattern not in requested_model:
            continue
        if str(conditions.get("source_surface") or source_surface).lower() != source_surface:
            continue
        if str(conditions.get("endpoint") or endpoint).lower() != endpoint:
            continue
        if str(conditions.get("category") or category).lower() != category:
            continue
        if "has_tools" in conditions and bool(conditions.get("has_tools")) != category.startswith("tool-"):
            continue
        if "stream" in conditions and bool(conditions.get("stream")):
            continue
        metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
        reason = str(rule.get("disabled_reason") or "disabled-local-policy")
        return {
            "schema": "tokenclaw.openai_routing_disabled_local_policy_rule.v1",
            "status": "disabled-local-policy",
            "reason": reason,
            "policy_source": str(rule.get("policy_source") or metadata.get("policy_source") or "local-promoted"),
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
            "rule_id_included": True,
            "target_rule_id": str(rule.get("id") or "promoted-openai-routing-rule"),
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
                "absolute_paths_included": False,
            },
        }
    return None


def _candidate(
    *,
    lever: str,
    provider_surface_bucket: str,
    blocker: str,
    estimated_savings_path: str,
    projected_savings_signal: dict[str, Any],
    confidence: str,
    sequencing: str,
    repo: str = "lutzkuen/tokenclaw",
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


_GOLDEN_PATH_READINESS_DIMENSIONS = {
    "openai_codex_local_capture": "OpenAI-compatible or Codex traffic can be captured locally",
    "safe_local_savings_action": "one local savings action can be safely applied",
    "metadata_only_outcome_evidence": "outcome evidence can be logged metadata-only",
    "tokenclaw_savings_reporting": "AgentFlow-generated savings can be reported separately from provider discounts",
    "rollback_safety_visibility": "rollback/safety state is visible",
    "user_explainability": "the dashboard or demo command can explain the state to a user",
}


def _golden_path_dimension_score(dimension: str | None) -> int:
    return {
        "safe_local_savings_action": 0,
        "rollback_safety_visibility": 1,
        "metadata_only_outcome_evidence": 2,
        "tokenclaw_savings_reporting": 3,
        "openai_codex_local_capture": 4,
        "user_explainability": 5,
    }.get(str(dimension or ""), 99)


def _candidate_golden_path_readiness_dimension(candidate: dict[str, Any]) -> str | None:
    lever = str(candidate.get("lever") or "").lower()
    blocker = str(candidate.get("blocker") or "").lower().replace("_", "-")
    bucket = str(candidate.get("provider_surface_bucket") or "").lower()
    safety = str(candidate.get("safety_status") or "").lower().replace("_", "-")
    path = str(candidate.get("estimated_savings_path") or "").lower()
    signal = candidate.get("projected_savings_signal") if isinstance(candidate.get("projected_savings_signal"), dict) else {}
    signal_text = json.dumps(signal, sort_keys=True).lower() if signal else ""
    combined = " ".join([lever, blocker, bucket, safety, path, signal_text])

    generic_churn_blockers = {
        "managed-recommendation-health-not-ranked",
        "shape-rollup-candidates-not-ranked",
        "request-shape-rollup-candidates-empty",
    }
    if blocker in generic_churn_blockers:
        return None

    if any(token in combined for token in ("rollback", "safety", "keep-blocked", "quality-regression", "regression")):
        return "rollback_safety_visibility"
    if any(token in combined for token in ("widen", "stage", "apply", "promotion-ready", "active-local-policy", "local-policy", "local action", "rule")):
        return "safe_local_savings_action"
    if any(token in combined for token in ("outcome", "metadata", "evidence", "applied", "holdout", "lifecycle", "measure")):
        return "metadata_only_outcome_evidence"
    if lever == "cache" and any(token in combined for token in ("zero-cache", "cache", "replay", "invalidation")):
        return "metadata_only_outcome_evidence"
    if lever == "crunch" and any(token in combined for token in ("savings", "saved", "projected")):
        return "tokenclaw_savings_reporting"
    if lever == "routing" and "missing-routing-breakdown" in blocker:
        return "openai_codex_local_capture"
    if lever == "request-shape-rollups" and "report-missing" in blocker:
        return "metadata_only_outcome_evidence"
    if any(token in combined for token in ("openai", "codex", "openai-responses", "openai_responses", "gpt-")):
        return "openai_codex_local_capture"
    return None


def _candidate_with_golden_path_readiness(candidate: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate)
    dimension = _candidate_golden_path_readiness_dimension(enriched)
    if dimension:
        enriched["golden_path_readiness_dimension"] = dimension
        enriched["golden_path_readiness_label"] = _GOLDEN_PATH_READINESS_DIMENSIONS[dimension]
        return enriched
    if not str(enriched.get("issue_generation_status") or "").startswith("suppressed-"):
        enriched["issue_generation_status"] = "suppressed-generic-telemetry-churn"
        enriched["issue_generation_suppression_reason"] = (
            "candidate-does-not-improve-openai-codex-golden-path-readiness"
        )
    return enriched


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

    promotion_report = stats_summary.get("openai_routing_promotion_decision")
    promotion_decision = (
        promotion_report.get("promotion_decision")
        if isinstance(promotion_report, dict) and isinstance(promotion_report.get("promotion_decision"), dict)
        else None
    )
    if promotion_decision is not None:
        lifecycle = promotion_decision.get("lifecycle") if isinstance(promotion_decision.get("lifecycle"), dict) else {}
        decision = str(promotion_decision.get("decision") or "unknown")
        ready = bool(promotion_decision.get("promotion_ready"))
        active = decision == "active-local-policy"
        reason = str(promotion_decision.get("reason") or decision)
        return _candidate(
            lever="routing",
            provider_surface_bucket="openai/openai_responses/responses/tool-light/gpt-5.4->gpt-5.4-mini",
            blocker=(
                "openai-routing-promotion-ready"
                if ready
                else "openai-routing-promotion-active-local-policy"
                if active
                else f"openai-routing-promotion-{reason}"
            ),
            estimated_savings_path=(
                "Convert the measured OpenAI routing promotion decision into a reviewed local file-backed routing rule."
                if ready
                else "Measure post-apply outcomes for the active local OpenAI routing rule."
                if active
                else "Resolve the OpenAI routing promotion blocker before local routing rule drafting."
            ),
            projected_savings_signal={
                "schema": promotion_report.get("schema"),
                "decision": decision,
                "next_action": promotion_decision.get("next_action"),
                "reason_codes": [str(item) for item in promotion_decision.get("reason_codes") or []],
                "applied_count": _to_int(lifecycle.get("applied_count")),
                "holdout_count": _to_int(lifecycle.get("holdout_count")),
                "safety_stop_count": _to_int(lifecycle.get("safety_stop_count")),
                "error_count": _to_int(lifecycle.get("error_count")),
                "fallback_count": _to_int(lifecycle.get("fallback_count")),
                "retry_count": _to_int(lifecycle.get("retry_count")),
                "savings_per_1000_calls_usd": _to_float(promotion_decision.get("savings_per_1000_calls_usd")),
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
            },
            confidence="high" if ready or active else "medium",
            sequencing="Use the OpenAI promotion decision before generic pass-through routing candidates so applied and holdout coverage remain durable.",
            safety_status="active-local-policy" if active else "review-required" if ready else "blocked",
            score=10_000.0 + _to_float(promotion_decision.get("savings_per_1000_calls_usd")),
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
            provider = str(top.get("provider") or "unknown")
            lifecycle = _routing_lifecycle_evidence(top) or {}
            lifecycle_counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
            lifecycle_blockers = [str(item) for item in lifecycle.get("blocker_codes") or [] if str(item or "").strip()]
            safety_stop_count = _to_int(lifecycle_counts.get("safety_stopped"))
            applied_count = _to_int(lifecycle_counts.get("canary_applied"))
            holdout_count = _to_int(lifecycle_counts.get("canary_holdout"))
            safety_stop_blocked = (
                provider == "anthropic"
                and actionability == "actionable"
                and bool(target)
                and (
                    safety_stop_count > 0
                    or "safety-stop-observed" in lifecycle_blockers
                )
            )
            blocker = (
                f"pass-through-routing-{actionability}"
                if actionability != "actionable"
                else "pass-through-routing-activation-blocked-until-safety-stop-burndown"
                if safety_stop_blocked
                else "pass-through-routing-activation-candidate"
            )
            projected_signal = {
                "report_schema": pass_through_report.get("schema"),
                "actionability": actionability,
                "sample_count": count,
                "requested_model": requested,
                "candidate_target_model": top.get("candidate_target_model"),
                "required_local_executor": top.get("required_local_executor"),
                "estimated_savings_per_1000_calls_usd": top.get("estimated_savings_per_1000_calls_usd"),
                "no_op_reason": top.get("no_op_reason"),
                "actionability_breakdown": pass_through_report.get("actionability_breakdown"),
            }
            if safety_stop_blocked:
                projected_signal["activation_gate"] = {
                    "schema": "tokenclaw.anthropic_routing_activation_issue_gate.v1",
                    "status": "suppressed-until-safety-stop-burndown-clears",
                    "safety_stop_count": safety_stop_count,
                    "applied_count": applied_count,
                    "holdout_count": holdout_count,
                    "blocker_codes": lifecycle_blockers,
                    "next_action": lifecycle.get("next_action")
                    or "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                    "privacy": _candidate_privacy(),
                }
            candidate = _candidate(
                lever="routing",
                provider_surface_bucket=_surface_bucket(top, fallback="mixed"),
                blocker=blocker,
                estimated_savings_path=(
                    "Suppress duplicate Anthropic activation issues until safety stops are zero and applied/holdout coverage exists."
                    if safety_stop_blocked
                    else
                    f"Stage a local routing canary from {requested} to {target} for the ranked pass-through bucket."
                    if target
                    else f"Keep {requested} pass-through explicit with a no-op reason before creating activation work."
                ),
                projected_savings_signal=projected_signal,
                confidence="medium" if actionability == "actionable" else "low",
                sequencing="Use the ranked pass-through bucket before generic routing activation issues; already-cheapest and safety-blocked buckets should remain explicit no-ops.",
                safety_status="blocked" if safety_stop_blocked else "review-required" if actionability == "actionable" else "blocked",
                score=float(count) + (1000.0 if actionability == "actionable" else 0.0),
            )
            if safety_stop_blocked:
                candidate["issue_generation_status"] = "suppressed-anthropic-routing-safety-stop-burndown"
                candidate["issue_generation_suppression_reason"] = (
                    "anthropic-routing-activation-suppressed-until-safety-stop-count-zero-and-applied-holdout-coverage-present"
                )
            return candidate

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
    shape_signal = (
        stats_summary.get("request_shape_rollup_candidates")
        if isinstance(stats_summary.get("request_shape_rollup_candidates"), dict)
        else {}
    )
    shape_replay = (
        shape_signal.get("cache_replayability_dry_run")
        if isinstance(shape_signal.get("cache_replayability_dry_run"), dict)
        else {}
    )
    if shape_replay and (
        _to_int((shape_replay.get("summary") if isinstance(shape_replay.get("summary"), dict) else {}).get("cohort_count")) > 0
        or shape_replay.get("cohorts")
    ):
        replay_summary = shape_replay.get("summary") if isinstance(shape_replay.get("summary"), dict) else {}
        remaining_rows = [
            row
            for row in shape_replay.get("remaining_replay_ready_cohorts") or []
            if isinstance(row, dict)
        ]
        skipped_openai = (
            shape_replay.get("skipped_openai_blockers")
            if isinstance(shape_replay.get("skipped_openai_blockers"), dict)
            else {}
        )
        skipped_rows = [
            row
            for row in skipped_openai.get("cohorts") or []
            if isinstance(row, dict)
        ] if isinstance(skipped_openai, dict) else []
        replay_rows = remaining_rows or [row for row in shape_replay.get("cohorts") or [] if isinstance(row, dict)]
        top_replay = replay_rows[0] if replay_rows else {}
        remaining_ready_count = _to_int(replay_summary.get("remaining_replay_ready_cohort_count"))
        remaining_ready_rows = _to_int(replay_summary.get("remaining_replay_ready_rows"))
        remaining_fields_present = (
            "remaining_replay_ready_cohort_count" in replay_summary
            or "remaining_replay_ready_rows" in replay_summary
        )
        if remaining_fields_present:
            ready = remaining_ready_count > 0 or remaining_ready_rows > 0 or bool(top_replay.get("remaining_replay_ready"))
        else:
            ready = _to_int(replay_summary.get("replay_ready_cohort_count")) > 0 or top_replay.get("readiness") == "replay-ready"
        if not ready and skipped_rows:
            top_replay = skipped_rows[0]
        remaining_ready = (
            bool(top_replay.get("remaining_replay_ready"))
            or remaining_ready_count > 0
            or remaining_ready_rows > 0
        )
        blockers = [str(item) for item in top_replay.get("blockers") or [] if str(item or "").strip()]
        for code in top_replay.get("blocker_codes") or []:
            if str(code or "").strip() and str(code) not in blockers:
                blockers.append(str(code))
        replay_blocker = "remaining-replay-ready" if remaining_ready else "replay-ready" if ready else str(
            replay_summary.get("top_blocker_code")
            or (blockers[0] if blockers else top_replay.get("reason"))
            or "cache-replayability-evidence-missing"
        )
        skipped_summary = (
            skipped_openai.get("summary")
            if isinstance(skipped_openai, dict) and isinstance(skipped_openai.get("summary"), dict)
            else {}
        )
        replay_hits = _to_int(
            top_replay.get("projected_hits")
            or (replay_summary.get("remaining_projected_hits") if ready else skipped_summary.get("projected_hits"))
            or 0
        )
        replay_rows_count = _to_int(top_replay.get("row_count") or top_replay.get("sample_count") or replay_summary.get("rows_considered"))
        replay_savings = _to_float(
            top_replay.get("projected_savings_usd")
            or (replay_summary.get("remaining_projected_savings_usd") if ready else skipped_summary.get("projected_savings_usd"))
            or 0.0
        )
        replay_bucket_row = dict(top_replay)
        if replay_bucket_row.get("provider") is None and replay_bucket_row.get("provider_family") is not None:
            replay_bucket_row["provider"] = replay_bucket_row.get("provider_family")
        return _candidate(
            lever="cache",
            provider_surface_bucket=_surface_bucket(replay_bucket_row, fallback=str(top_replay.get("source_surface") or "mixed")),
            blocker=replay_blocker,
            estimated_savings_path=(
                "Turn request-shape replayability evidence into a staged local exact-cache canary "
                "or a narrower invalidation/dependency blocker."
            ),
            projected_savings_signal={
                "source_schema": shape_replay.get("schema"),
                "status": shape_replay.get("status"),
                "readiness": "replay-ready" if ready else str(top_replay.get("readiness") or "skipped"),
                "reason": top_replay.get("reason"),
                "calls": calls,
                "cache_hits": cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "replay_ready_cohort_count": _to_int(replay_summary.get("replay_ready_cohort_count")),
                "replay_ready_rows": _to_int(replay_summary.get("replay_ready_rows")),
                "remaining_replay_ready_rows": remaining_ready_rows,
                "skipped_cohort_count": _to_int(replay_summary.get("skipped_cohort_count")),
                "skipped_openai_cohort_count": _to_int(
                    (skipped_openai.get("summary") if isinstance(skipped_openai.get("summary"), dict) else {}).get("skipped_openai_cohort_count")
                ) if isinstance(skipped_openai, dict) else 0,
                "projected_hits": replay_hits,
                "projected_savings_usd": round(replay_savings, 8),
                "top_blocker_code": replay_summary.get("top_blocker_code"),
            },
            confidence="medium" if ready or replay_rows_count > 0 else "low",
            sequencing="Use after request-shape rollups exist and before generic zero-cache-hit issue generation.",
            score=float(replay_hits or replay_rows_count or calls) + (500.0 if ready else 250.0),
        )
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
        "schema": "tokenclaw.crunch_savings_signal.v1",
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
    if status == "observed-savings-ranked" and (today_savings > 0 or total_savings > 0 or tokens_saved > 0 or chars_saved > 0):
        blocker = "crunch-observed-savings-ranked"
        path = "Convert observed crunch savings into the next compaction activation or rollout-safety issue."
        confidence = "medium"
        score = max(today_savings * 1000.0, total_savings * 250.0, tokens_saved / 10.0, chars_saved / 100.0, 1.0)
        bucket = "observed-crunch"
    elif projected_usd > 0 or projected_tokens > 0 or projected_chars > 0:
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
    candidate = _candidate(
        lever="crunch",
        provider_surface_bucket=bucket,
        blocker=blocker,
        estimated_savings_path=path,
        projected_savings_signal=signal_payload,
        confidence=confidence,
        sequencing="Sequence behind routing/cache blockers unless crunch savings is already the strongest positive dollar signal.",
        score=score,
    )
    duplicate_suppression = (
        top_report.get("duplicate_suppression")
        if isinstance(top_report.get("duplicate_suppression"), dict)
        else {}
    )
    if (
        blocker == "crunch-observed-savings-ranked"
        and top_report.get("report_key") == "request_shape_crunch_activation_evidence"
        and bool(duplicate_suppression.get("suppresses_generic_crunch_activation_issue"))
    ):
        candidate["issue_generation_status"] = "suppressed-active-crunch-keep-active"
        candidate["issue_generation_suppression_reason"] = sanitize_value(
            duplicate_suppression.get("reason") or "repeated-context-crunch-active-at-max-rollout"
        )
        candidate["duplicate_suppression"] = sanitize_value(duplicate_suppression)
    return candidate


def _diagnostic_candidate(diagnostics: list[dict[str, Any]]) -> dict[str, Any] | None:
    actionable = _actionable_diagnostics(diagnostics)
    if not actionable:
        return None
    top = actionable[0]
    reason = str(top.get("reason") or "unknown-diagnostic")
    count = _to_int(top.get("count"))
    lifecycle_context = top.get("lifecycle_context") if isinstance(top.get("lifecycle_context"), dict) else {}
    ledger_stage = _diagnostic_ledger_stage(top) or {}
    candidate = _candidate(
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
            "lifecycle_context": lifecycle_context,
            "ledger_fingerprint": ledger_stage.get("diagnostic_fingerprint"),
            "ledger_next_action": ledger_stage.get("next_action"),
            "ledger_current_status": _ledger_status_from_stage(ledger_stage) if ledger_stage else None,
            "verification_check": ledger_stage.get("verification_check"),
            "privacy": _candidate_privacy(),
        },
        confidence="medium" if count > 1 else "low",
        sequencing="File after direct cache/routing/crunch candidates unless the diagnostic is a safety stop or privacy blocker.",
        safety_status="blocked" if ledger_stage.get("keep_blocked_reason") else "review-required",
        score=float(count) * 10.0,
    )
    suppression = _activation_feedback_blocker_review_suppression(top)
    if suppression is not None:
        candidate["issue_generation_status"] = "suppressed-durable-keep-blocked-ledger-record"
        candidate["issue_generation_suppression_reason"] = suppression["keep_blocked_reason"]
    return candidate


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
    follow_up_owner = str(top.get("follow_up_owner") or "")
    suppress_generic_issue = bool(
        isinstance(signal.get("duplicate_suppression"), dict)
        and signal["duplicate_suppression"].get("suppresses_generic_missing_health_issue")
    )
    issue_generation_status = "active"
    if status == "missing-managed-recommendation-health-report" and represented and follow_up_owner == "local-policy":
        reason_fragment = omitted_reason if omitted_reason.startswith("managed-recommendation") else f"managed-recommendation-{omitted_reason}"
        blocker = reason_fragment
        path = "Continue with the ranked local file-backed policy follow-up while treating managed recommendation health as optional."
        confidence = "medium"
        safety_status = "review-required"
        score = float(_to_int(top.get("count")) or calls) + 250.0
        if suppress_generic_issue:
            issue_generation_status = "suppressed-local-file-backed-handoff"
    elif status == "missing-managed-recommendation-health-report":
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

    candidate = _candidate(
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
    if issue_generation_status != "active":
        candidate["issue_generation_status"] = issue_generation_status
        candidate["issue_generation_suppression_reason"] = "managed-health-missing-covered-by-local-file-backed-handoff"
    return candidate


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
    candidate_signal = signal
    request_shape_ledger = None
    ledger = stats_summary.get("evidence_to_activation_next_action_ledger")
    if isinstance(ledger, dict):
        for entry in ledger.get("entries") or []:
            if isinstance(entry, dict) and entry.get("lever") == "request-shape-rollups":
                request_shape_ledger = entry
                break
    if top:
        next_action = str(top.get("next_action") or "rank-request-shape-cohort")
        if isinstance(request_shape_ledger, dict):
            ledger_next_action = str(request_shape_ledger.get("next_action") or "").strip()
            if ledger_next_action:
                next_action = ledger_next_action
        ledger_family = str((request_shape_ledger or {}).get("local_action_family") or "").strip()
        family = ledger_family or str(top.get("local_action_family") or "")
        action_parts = ["request-shape"]
        if family:
            action_parts.append(family)
        action_parts.append(next_action)
        blocker = "-".join(action_parts)
        readiness = str((request_shape_ledger or {}).get("state") or top.get("readiness_state") or "")
        path = (
            f"Use the top aggregate request-shape cohort to advance `{next_action}` for the local `{family or 'policy'}` path."
        )
        confidence = "medium" if readiness in {"activation-ready", "measurement-required"} or "repeated_context" in (top.get("candidate_work_classes") or []) else "low"
        safety_status = "ready" if readiness == "activation-ready" else "review-required"
        if str((request_shape_ledger or {}).get("current_status") or "") in {"blocked", "safety-stopped", "keep-blocked"}:
            safety_status = "blocked"
        score = float(_to_int(top.get("row_count"))) + _to_float(top.get("cost_est_usd")) * 1000.0 + 300.0
        bucket = str(top.get("provider_surface_bucket") or "mixed")
        if isinstance(request_shape_ledger, dict):
            candidate_signal = dict(signal)
            candidate_signal["ledger_fingerprint"] = request_shape_ledger.get("fingerprint")
            candidate_signal["ledger_current_status"] = request_shape_ledger.get("current_status")
            candidate_signal["ledger_next_action"] = request_shape_ledger.get("next_action")
            candidate_signal["ledger_blocker_codes"] = request_shape_ledger.get("blocker_codes") or []
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
        projected_savings_signal=candidate_signal,
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
        ranked.append(_candidate_with_golden_path_readiness(clean))
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


def _proposal_golden_path_readiness_dimension(proposal: dict[str, Any]) -> str | None:
    explicit = str(proposal.get("golden_path_readiness_dimension") or "").strip()
    if explicit in _GOLDEN_PATH_READINESS_DIMENSIONS:
        return explicit
    labels = {str(label).lower() for label in proposal.get("labels") or []}
    text = " ".join(
        [
            str(proposal.get("title") or ""),
            str(proposal.get("body") or ""),
            " ".join(sorted(labels)),
        ]
    ).lower().replace("_", "-")

    if "rank next savings milestone" in text or "openai/codex" in text or "readiness" in text:
        return "user_explainability"
    if any(token in text for token in ("rollback", "safety", "keep-blocked", "quality-regression", "regression")):
        return "rollback_safety_visibility"
    if any(token in text for token in ("widen", "stage", "apply", "promote", "local rule", "local policy", "canary", "one local savings action")):
        return "safe_local_savings_action"
    if any(token in text for token in ("outcome evidence", "metadata-only", "applied/holdout", "applied", "holdout", "lifecycle", "measure")):
        return "metadata_only_outcome_evidence"
    if any(token in text for token in ("tokenclaw-generated savings", "provider prompt-cache", "savings separately", "savings demo", "crunch savings")):
        return "tokenclaw_savings_reporting"
    if any(token in text for token in ("openai-compatible", "codex traffic", "openai traffic", "openai api", "openai responses")):
        return "openai_codex_local_capture"
    return None


def _proposal_product_loop_impact(dimension: str, proposal: dict[str, Any]) -> str:
    label = _GOLDEN_PATH_READINESS_DIMENSIONS[dimension]
    title = str(proposal.get("title") or "this issue")
    if dimension == "safe_local_savings_action":
        action = "moves from evidence toward a reviewed local routing, crunching, cache, or rollback-safe policy action."
    elif dimension == "rollback_safety_visibility":
        action = "keeps the product loop safe by making rollback, blocker, safety-stop, or quality-regression state explicit before more traffic is widened."
    elif dimension == "metadata_only_outcome_evidence":
        action = "turns the next step into metadata-only applied, holdout, lifecycle, or outcome evidence that later runs can rank without raw content."
    elif dimension == "tokenclaw_savings_reporting":
        action = "helps separate AgentFlow-generated savings from provider discounts so a user can see whether AgentFlow itself created value."
    elif dimension == "openai_codex_local_capture":
        action = "improves local capture or classification of OpenAI-compatible or Codex traffic before savings are claimed."
    else:
        action = "makes the OpenAI/Codex savings state explainable in the dashboard, demo command, or research handoff."
    return f"{title} improves golden-path readiness because {label}; it {action}"


def _finalize_create_issue_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(proposal)
    labels = [redact_text(str(label)) for label in finalized.get("labels") or []]
    finalized["labels"] = list(dict.fromkeys(label for label in labels if label))
    body = redact_text(str(finalized.get("body") or ""))
    dimension = _proposal_golden_path_readiness_dimension(finalized)
    if dimension is not None:
        finalized["golden_path_readiness_dimension"] = dimension
        finalized["golden_path_readiness_label"] = _GOLDEN_PATH_READINESS_DIMENSIONS[dimension]
        impact = redact_text(str(finalized.get("openai_codex_product_loop_impact") or _proposal_product_loop_impact(dimension, finalized)))
        finalized["openai_codex_product_loop_impact"] = impact
        if "## OpenAI/Codex Product Loop Impact" not in body:
            impact_section = f"## OpenAI/Codex Product Loop Impact\n\n{impact}\n\n"
            marker = "## Implementation Approach\n\n"
            if marker in body:
                body = body.replace(marker, impact_section + marker, 1)
            else:
                body = f"{body.rstrip()}\n\n{impact_section.rstrip()}\n"
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
    finalized["repo"] = redact_text(str(finalized.get("repo") or "lutzkuen/tokenclaw"))
    return finalized


def _filter_golden_path_ready_proposals(
    proposals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for proposal in proposals:
        dimension = _proposal_golden_path_readiness_dimension(proposal)
        if dimension is None:
            suppressed.append(
                {
                    "title": sanitize_value(proposal.get("title")),
                    "repo": sanitize_value(proposal.get("repo") or "lutzkuen/tokenclaw"),
                    "reason": "candidate-does-not-improve-openai-codex-golden-path-readiness",
                    "suppression_kind": "generic-telemetry-churn",
                }
            )
            continue
        enriched = dict(proposal)
        enriched["golden_path_readiness_dimension"] = dimension
        enriched["golden_path_readiness_label"] = _GOLDEN_PATH_READINESS_DIMENSIONS[dimension]
        kept.append(enriched)
    return kept, suppressed


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


_PUBLIC_FINGERPRINT_RE = re.compile(
    r"\b(?:activation|diagnostic|policy|rule|cohort|candidate):[a-z0-9][a-z0-9_.:-]{6,}\b",
    re.IGNORECASE,
)
_LIFECYCLE_ACTION_RE = re.compile(
    r"(?:\b(?:top\s+)?next\s+action\b|next_action)\s*[:=]\s*`?([a-z0-9_.:-]+)`?",
    re.IGNORECASE,
)


def _proposal_fingerprints(item: dict[str, Any]) -> set[str]:
    fingerprints: set[str] = set()
    for key in (
        "fingerprint",
        "diagnostic_fingerprint",
        "proposal_fingerprint",
        "evidence_fingerprint",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            fingerprints.add(redact_text(value.strip().lower()))
    body = str(item.get("body") or "")
    for match in _PUBLIC_FINGERPRINT_RE.findall(body):
        fingerprints.add(redact_text(match.strip().lower()))
    return {fingerprint for fingerprint in fingerprints if fingerprint}


def _proposal_lifecycle_actions(item: dict[str, Any]) -> set[str]:
    actions: set[str] = set()
    for value in (
        item.get("next_action"),
        item.get("top_next_action"),
        item.get("lifecycle_next_action"),
    ):
        if isinstance(value, str) and value.strip():
            actions.add(value.strip().lower().replace("_", "-"))
    body = str(item.get("body") or "")
    for match in _LIFECYCLE_ACTION_RE.findall(body):
        action = match.strip().lower().replace("_", "-")
        if action:
            actions.add(action)
    return actions


def _closed_issue_is_lifecycle_predecessor(proposal: dict[str, Any], issue: dict[str, Any] | None) -> bool:
    if issue is None or _is_open(issue):
        return False
    proposal_actions = _proposal_lifecycle_actions(proposal)
    issue_actions = _proposal_lifecycle_actions(issue)
    return bool(proposal_actions and issue_actions and proposal_actions.isdisjoint(issue_actions))


def _dedupe_create_issue_proposals(
    proposals: list[dict[str, Any]],
    *,
    existing_issues: Iterable[dict[str, Any]],
    max_count: int = 10,
    trusted_author: str = "lutzkuen",
    now: datetime | None = None,
    recent_closed_days: int = 14,
) -> list[dict[str, Any]]:
    return _dedupe_create_issue_proposals_with_metadata(
        proposals,
        existing_issues=existing_issues,
        max_count=max_count,
        trusted_author=trusted_author,
        now=now,
        recent_closed_days=recent_closed_days,
    )[0]


def _dedupe_create_issue_proposals_with_metadata(
    proposals: list[dict[str, Any]],
    *,
    existing_issues: Iterable[dict[str, Any]],
    max_count: int = 10,
    trusted_author: str = "lutzkuen",
    now: datetime | None = None,
    recent_closed_days: int = 14,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    existing_by_key: dict[str, dict[str, Any]] = {}
    existing_by_fingerprint: dict[str, dict[str, Any]] = {}
    for issue in existing_issues:
        if not isinstance(issue, dict) or not _is_trusted(issue, trusted_author):
            continue
        if not _is_open(issue) and not _is_recent_closed_issue(issue, now=now, recent_days=recent_closed_days):
            continue
        key = _issue_title_key(issue.get("title"))
        if key and key not in existing_by_key:
            existing_by_key[key] = issue
        for fingerprint in _proposal_fingerprints(issue):
            existing_by_fingerprint.setdefault(fingerprint, issue)
    existing_keys = set(existing_by_key)
    existing_keys.discard("")
    seen = set(existing_keys)
    seen_fingerprints = set(existing_by_fingerprint)
    deduped: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for proposal in proposals:
        key = _issue_title_key(proposal.get("title"))
        if not key:
            continue
        proposal_fingerprints = _proposal_fingerprints(proposal)
        matched_fingerprint = next((item for item in proposal_fingerprints if item in seen_fingerprints), None)
        if key in seen or matched_fingerprint is not None:
            matched_issue = existing_by_key.get(key) or (
                existing_by_fingerprint.get(matched_fingerprint) if matched_fingerprint else None
            )
            if (
                key not in seen
                and matched_fingerprint is not None
                and _closed_issue_is_lifecycle_predecessor(proposal, matched_issue)
            ):
                progressed = dict(proposal)
                progressed["closed_lifecycle_predecessor"] = _issue_ref(matched_issue) if matched_issue is not None else None
                progressed["closed_lifecycle_predecessor_reason"] = "same-fingerprint-next-action-progressed"
                progressed["closed_lifecycle_predecessor_fingerprint"] = sanitize_value(matched_fingerprint)
                seen.add(key)
                seen_fingerprints.update(proposal_fingerprints)
                deduped.append(progressed)
                if len(deduped) >= max_count:
                    break
                continue
            reason = "exact-title-already-exists" if key in seen else "evidence-fingerprint-already-exists"
            row = {
                "title": sanitize_value(proposal.get("title")),
                "repo": sanitize_value(proposal.get("repo") or (matched_issue or {}).get("repo") or "unknown"),
                "proposal_key": key,
                "reason": reason,
            }
            if matched_fingerprint:
                row["evidence_fingerprint"] = sanitize_value(matched_fingerprint)
            if matched_issue is not None:
                matched_issue_is_closed = not _is_open(matched_issue)
                row.update(
                    {
                        "existing_issue": _issue_ref(matched_issue),
                        "existing_issue_state": sanitize_value(matched_issue.get("state") or "OPEN"),
                        "suppression_kind": "closed-prior-issue" if matched_issue_is_closed else "open-existing-issue",
                    }
                )
                if matched_issue_is_closed:
                    row["successor_required"] = True
                    row["successor_reason"] = "recent-closed-exact-title-match"
            else:
                row["suppression_kind"] = "duplicate-generated-proposal"
            suppressed.append(row)
            continue
        seen.add(key)
        seen_fingerprints.update(proposal_fingerprints)
        deduped.append(proposal)
        if len(deduped) >= max_count:
            break
    metadata = {
        "schema": "tokenclaw.research_issue_proposal_suppression.v1",
        "suppressed_count": len(suppressed),
        "closed_prior_issue_count": sum(1 for item in suppressed if item.get("suppression_kind") == "closed-prior-issue"),
        "suppressed_closed_predecessor_count": sum(
            1 for item in suppressed if item.get("suppression_kind") == "closed-prior-issue"
        ),
        "successor_required_count": sum(1 for item in suppressed if item.get("successor_required")),
        "open_existing_issue_count": sum(1 for item in suppressed if item.get("suppression_kind") == "open-existing-issue"),
        "fingerprint_match_count": sum(1 for item in suppressed if item.get("reason") == "evidence-fingerprint-already-exists"),
        "recent_closed_days": recent_closed_days,
        "suppressed": suppressed[:20],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
        },
    }
    return deduped, metadata


def _candidate_title(candidate: dict[str, Any]) -> str:
    lever = str(candidate.get("lever") or "optimization")
    blocker = str(candidate.get("blocker") or "candidate").replace("_", "-")
    signal = candidate.get("projected_savings_signal") if isinstance(candidate.get("projected_savings_signal"), dict) else {}
    if lever == "cache":
        if blocker == "remaining-replay-ready":
            return "Advance remaining replay-ready cache cohort into local replay evidence"
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
        blocker_text = blocker.replace("request-shape-", "")
        if "keep-active" in blocker_text and "crunch" in blocker_text:
            return "Record request-shape repeated-context crunch keep-active outcome"
        if "widen" in blocker_text and "crunch" in blocker_text:
            return "Apply measured request-shape crunch widening to local rules"
        if "measure" in blocker_text and "crunch" in blocker_text:
            return "Measure request-shape repeated-context crunch canary impact"
        if "safety" in blocker_text and "crunch" in blocker_text:
            return "Review request-shape repeated-context crunch canary safety stop"
        if "crunch" in blocker_text:
            return "Stage request-shape repeated-context crunch canary"
        if "tool-call-cache-invalidation" in blocker_text:
            return "Collect request-shape tool-cache invalidation evidence"
        if "streaming-cache-replay" in blocker_text:
            return "Add request-shape streaming cache replay support"
        if "cache-replay" in blocker_text:
            return "Stage request-shape cache replay cohort"
        if "routing" in blocker_text:
            return "Collect request-shape routing lifecycle evidence"
        return "Rank request-shape blockers into local action cohorts"
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
        first = (
            "Request-shape rollups report ranked blocker cohorts with rank, blocker codes, local action family, readiness state, "
            "next action, sample count, projected savings when known, and metadata-only privacy flags."
        )
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
        "repo": candidate.get("repo") or "lutzkuen/tokenclaw",
        "title": title,
        "labels": _candidate_labels(candidate),
        "golden_path_readiness_dimension": candidate.get("golden_path_readiness_dimension"),
        "golden_path_readiness_label": candidate.get("golden_path_readiness_label"),
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
    return [
        _proposal_from_optimization_candidate(candidate)
        for candidate in candidates
        if not str(candidate.get("issue_generation_status") or "").startswith("suppressed-")
    ]


def _bounded_low_backlog_evidence(
    stats_summary: dict[str, Any],
    optimization_candidates: list[dict[str, Any]],
) -> list[str]:
    evidence: list[str] = []
    calls = _to_int(stats_summary.get("today_calls") or stats_summary.get("calls"))
    if calls > 0:
        evidence.append(f"Local metadata window contains {calls} calls.")
    if "cache_hit_rate" in stats_summary:
        evidence.append(
            f"Cache hit signal: {_to_int(stats_summary.get('cache_hits'))} hits, "
            f"{_to_float(stats_summary.get('cache_hit_rate')):.4f} hit rate."
        )
    loop = stats_summary.get("evidence_to_activation_loop")
    if isinstance(loop, dict):
        summary = loop.get("summary") if isinstance(loop.get("summary"), dict) else {}
        top_lever = summary.get("top_lever")
        top_action = summary.get("top_next_action")
        top_state = summary.get("top_state")
        if top_lever or top_action or top_state:
            evidence.append(
                "Evidence-to-activation loop top item: "
                f"lever={sanitize_value(top_lever or 'unknown')}, "
                f"state={sanitize_value(top_state or 'unknown')}, "
                f"next_action={sanitize_value(top_action or 'unknown')}."
            )
    ledger = stats_summary.get("evidence_to_activation_next_action_ledger")
    if isinstance(ledger, dict):
        summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
        if summary:
            evidence.append(
                "Next-action ledger top item: "
                f"lever={sanitize_value(summary.get('top_lever') or 'unknown')}, "
                f"status={sanitize_value(summary.get('top_current_status') or 'unknown')}, "
                f"next_action={sanitize_value(summary.get('top_next_action') or 'unknown')}."
            )
    crunch = stats_summary.get("crunch_savings_signal")
    if isinstance(crunch, dict):
        top_report = crunch.get("top_report") if isinstance(crunch.get("top_report"), dict) else {}
        if top_report:
            evidence.append(
                "Crunch signal: "
                f"status={sanitize_value(crunch.get('status') or 'unknown')}, "
                f"matched={_to_int(top_report.get('matched_count'))}, "
                f"projected_saved_usd={_to_float(top_report.get('projected_saved_usd')):.6f}, "
                f"next_action={sanitize_value(top_report.get('next_action') or 'unknown')}."
            )
    routing = stats_summary.get("pass_through_routing_report")
    if isinstance(routing, dict):
        summary = routing.get("summary") if isinstance(routing.get("summary"), dict) else {}
        if summary:
            evidence.append(
                "Routing signal: "
                f"top_actionability={sanitize_value(summary.get('top_actionability') or 'unknown')}, "
                f"requested={sanitize_value(summary.get('top_requested_model') or 'unknown')}, "
                f"target={sanitize_value(summary.get('top_candidate_target_model') or 'unknown')}."
            )
    if optimization_candidates:
        top = optimization_candidates[0]
        evidence.append(
            "Top ranked optimization candidate: "
            f"rank={_to_int(top.get('rank'))}, "
            f"lever={sanitize_value(top.get('lever') or 'unknown')}, "
            f"blocker={sanitize_value(top.get('blocker') or 'unknown')}, "
            f"confidence={sanitize_value(top.get('confidence') or 'unknown')}."
        )
    return evidence[:8]


def _proposal_from_low_backlog(
    *,
    ready_count: int,
    threshold: int,
    stats_summary: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    optimization_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = [
        f"Actionable status:ready issue count is {ready_count}, below threshold {threshold}.",
    ]
    evidence.extend(_bounded_low_backlog_evidence(stats_summary, optimization_candidates))
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
        "repo": "lutzkuen/tokenclaw",
        "title": LOW_BACKLOG_MILESTONE_TITLE,
        "labels": _default_issue_labels("priority:p1"),
        "body": _issue_body(
            title=LOW_BACKLOG_MILESTONE_TITLE,
            rationale=(
                "Research mode needs to create implementation-ready issues from local metadata "
                "when the ready backlog falls below the configured threshold. This targeted milestone "
                "replaces the older generic backlog-generation title with a ranked savings gap review."
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
                "The research plan includes a compact next_backlog_milestone summary that ranks the generated issue proposals.",
                "Privacy tests prove raw prompts, request bodies, file paths, request IDs, and session IDs are redacted.",
            ],
            sequencing="Use this as the first fallback issue when no more specific blocker dominates the local evidence.",
        ),
    }


def _proposal_priority(labels: list[str]) -> str:
    for label in labels:
        if label.startswith("priority:"):
            return label
    return "priority:unknown"


def _proposal_priority_score(priority: str) -> int:
    match = re.fullmatch(r"priority:p(\d+)", priority)
    if match is None:
        return 99
    return _to_int(match.group(1), 99)


def _proposal_lever(proposal: dict[str, Any], candidate_by_title: dict[str, dict[str, Any]]) -> str:
    title = str(proposal.get("title") or "")
    candidate = candidate_by_title.get(title)
    if candidate is not None:
        return sanitize_value(candidate.get("lever") or "optimization")
    labels = {str(label) for label in proposal.get("labels") or []}
    for lever in ("cache", "routing", "crunch"):
        if lever in labels:
            return lever
    title_l = title.lower()
    if "managed recommendation" in title_l:
        return "managed-recommendation"
    if "request-shape" in title_l or "repeated context" in title_l:
        return "request-shape-rollups"
    if "safety" in title_l or "feedback" in title_l:
        return "activation-feedback"
    if "milestone" in title_l or "backlog" in title_l:
        return "milestone-planning"
    return "optimization"


def _proposal_summary_source(title: str, candidate: dict[str, Any] | None, proposal: dict[str, Any] | None = None) -> str:
    if isinstance(proposal, dict):
        explicit = str(proposal.get("proposal_source") or "").strip()
        if explicit:
            return sanitize_value(explicit)
    if candidate is not None:
        return "ranked-optimization-candidate"
    title_l = title.lower()
    if "activation successor" in title_l or "preview-verified successor" in title_l:
        return "preview-verified-activation-successor"
    if "evidence-to-activation ledger" in title_l:
        return "evidence-to-activation-ledger"
    if "replay-ready cache cohort" in title_l or "cache replay" in title_l:
        return "cache-replay-lifecycle"
    if "promotion" in title_l or "policy" in title_l:
        return "local-promotion-lifecycle"
    return "low-backlog-research"


def _proposal_source_score(source: str) -> int:
    return {
        "preview-verified-activation-successor": 0,
        "evidence-to-activation-ledger": 0,
        "cache-replay-lifecycle": 1,
        "local-promotion-lifecycle": 1,
        "ranked-optimization-candidate": 2,
        "low-backlog-research": 3,
    }.get(source, 4)


def _next_action_from_summary(stats_summary: dict[str, Any], top_candidate: dict[str, Any] | None) -> str | None:
    for key in ("evidence_to_activation_next_action_ledger", "evidence_to_activation_loop"):
        report = stats_summary.get(key)
        if not isinstance(report, dict):
            continue
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        action = summary.get("top_next_action")
        if action:
            return sanitize_value(action)
    if top_candidate is not None:
        signal = top_candidate.get("projected_savings_signal")
        if isinstance(signal, dict) and signal.get("next_action"):
            return sanitize_value(signal.get("next_action"))
    return None


def _next_backlog_milestone(
    *,
    create_issues: list[dict[str, Any]],
    optimization_candidates: list[dict[str, Any]],
    stats_summary: dict[str, Any],
) -> dict[str, Any]:
    candidate_by_title = {
        _candidate_title(candidate): candidate
        for candidate in optimization_candidates
    }
    top_candidate = optimization_candidates[0] if optimization_candidates else None
    issue_rows: list[dict[str, Any]] = []
    for rank, proposal in enumerate(create_issues, start=1):
        labels = [str(label) for label in proposal.get("labels") or []]
        title = str(proposal.get("title") or "")
        candidate = candidate_by_title.get(title)
        lever = _proposal_lever(proposal, candidate_by_title)
        priority = _proposal_priority(labels)
        source = _proposal_summary_source(title, candidate, proposal)
        dimension = _proposal_golden_path_readiness_dimension(proposal)
        issue_rows.append(
            {
                "rank": rank,
                "title": sanitize_value(title),
                "repo": sanitize_value(proposal.get("repo") or "lutzkuen/tokenclaw"),
                "lever": lever,
                "priority": priority,
                "labels": labels,
                "source": source,
                "golden_path_readiness_dimension": sanitize_value(dimension),
                "golden_path_readiness_label": sanitize_value(
                    _GOLDEN_PATH_READINESS_DIMENSIONS.get(str(dimension or ""), "")
                ),
                "candidate_rank": _to_int(candidate.get("rank")) if candidate is not None else None,
                "expected_savings_path": sanitize_value(
                    proposal.get("expected_savings_path")
                    or (candidate or {}).get("estimated_savings_path")
                    or "Turn metadata-only local evidence into an implementation-ready follow-up."
                ),
            }
        )
    top_issue = issue_rows[0] if issue_rows else None
    ranked_issue_rows = sorted(
        issue_rows,
        key=lambda row: (
            1 if row["lever"] == "milestone-planning" else 0,
            _proposal_source_score(str(row["source"])),
            _golden_path_dimension_score(str(row.get("golden_path_readiness_dimension") or "")),
            _proposal_priority_score(str(row["priority"])),
            _to_int(row.get("candidate_rank"), 10_000),
            _to_int(row["rank"], 10_000),
        ),
    )
    for implementation_rank, row in enumerate(ranked_issue_rows, start=1):
        row["implementation_rank"] = implementation_rank
    recommended_issue = ranked_issue_rows[0] if ranked_issue_rows else None
    top_lever = sanitize_value(top_candidate.get("lever")) if top_candidate is not None else None
    top_blocker = sanitize_value(top_candidate.get("blocker")) if top_candidate is not None else None
    return {
        "schema": "tokenclaw.next_backlog_milestone.v1",
        "status": "ready" if issue_rows else "empty",
        "summary": {
            "proposal_count": len(issue_rows),
            "ranked_candidate_count": len(optimization_candidates),
            "top_lever": top_lever,
            "top_blocker": top_blocker,
            "top_next_action": _next_action_from_summary(stats_summary, top_candidate),
            "top_issue": {
                "rank": top_issue["rank"],
                "title": top_issue["title"],
                "repo": top_issue["repo"],
                "lever": top_issue["lever"],
                "priority": top_issue["priority"],
                "golden_path_readiness_dimension": top_issue["golden_path_readiness_dimension"],
            } if top_issue is not None else None,
            "recommended_next_issue": {
                "rank": recommended_issue["rank"],
                "implementation_rank": recommended_issue["implementation_rank"],
                "title": recommended_issue["title"],
                "repo": recommended_issue["repo"],
                "lever": recommended_issue["lever"],
                "priority": recommended_issue["priority"],
                "source": recommended_issue["source"],
                "golden_path_readiness_dimension": recommended_issue["golden_path_readiness_dimension"],
            } if recommended_issue is not None else None,
        },
        "issues": issue_rows,
        "implementation_order": ranked_issue_rows,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
        },
    }


def _promotion_blocker_action_title(action: str, family: str) -> str:
    action_l = action.lower().replace("_", "-")
    family_l = family.lower().replace("_", "-") or "optimization"
    if "rollback" in action_l or "disable" in action_l:
        return f"Rollback unsafe promotion blocker {family_l} canary"
    if "widen" in action_l:
        return f"Widen promotion blocker {family_l} canary from next-action status"
    if "impact" in action_l or "measure" in action_l:
        return f"Measure promotion blocker {family_l} next-action impact"
    if "managed" in action_l or "feedback" in action_l:
        return f"Feed promotion blocker {family_l} outcomes into managed feedback"
    if "apply" in action_l or "activate" in action_l or "stage" in action_l:
        return f"Apply promotion blocker {family_l} next action"
    if "eval" in action_l or "holdout" in action_l or "backfill" in action_l or "collect" in action_l:
        return f"Backfill promotion blocker {family_l} evidence from next-action status"
    return f"Advance promotion blocker {family_l} next action"


def _proposal_from_promotion_blocker_next_action(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    status = stats_summary.get("promotion_blocker_next_action_status")
    if not isinstance(status, dict):
        return None
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    rows = [row for row in status.get("next_actions") or [] if isinstance(row, dict)]
    top = rows[0] if rows else {}
    next_action = str(summary.get("top_next_action") or top.get("next_action") or "").strip()
    if not next_action:
        return None
    family = str(summary.get("top_local_action_family") or top.get("local_action_family") or "optimization").strip() or "optimization"
    title = _promotion_blocker_action_title(next_action, family)
    labels = _default_issue_labels("priority:p1")
    family_label = family.replace("_", "-")
    if family_label in {"routing", "cache", "crunch"}:
        labels.append(family_label)
    labels.append("privacy")
    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": list(dict.fromkeys(labels)),
        "body": _issue_body(
            title=title,
            rationale=(
                "Research mode found a current local promotion blocker next-action status. "
                "The backlog should advance the named local action instead of recreating stale evidence-gathering issues that already landed."
            ),
            evidence=[
                f"Source metadata: {status.get('source_schema')}",
                f"Source status: {status.get('source_status')}",
                f"Top local action family: {family}",
                f"Top next action: {next_action}",
                f"Top blocker reason: {summary.get('top_blocker_reason')}",
                f"Top safety stop reason: {summary.get('top_safety_stop_reason')}",
                f"Expected local executor: {summary.get('top_expected_local_executor') or top.get('expected_local_executor')}",
                f"Projected savings USD: {summary.get('projected_savings_usd') or top.get('projected_savings_usd')}",
                f"Candidate count: {summary.get('review_candidate_count') or top.get('count')}",
            ],
            implementation=[
                "Start from the persisted research plan promotion_blocker_next_action_status section and the bounded local review report.",
                f"Implement the successor action `{next_action}` for the `{family}` family using existing local review, canary, impact, rollback, or feedback modules.",
                "Do not inspect prompts, provider bodies, request IDs, session IDs, cache keys, file paths, or individual candidate identifiers.",
                "Record the outcome in machine-readable promotion blocker, rollout, or lifecycle metadata so a later research plan can suppress this issue when it lands.",
            ],
            acceptance=[
                f"The next promotion blocker status reports progress for next_action={next_action} or a narrower keep-blocked reason.",
                "The research plan names the promotion blocker top next action and suppresses exact-title closed proposals instead of recreating them.",
                "Generated and follow-up evidence remains metadata-only and excludes prompts, provider bodies, absolute paths, request IDs, session IDs, cache keys, and individual candidate IDs.",
            ],
            savings_path=(
                "This removes stale backlog churn and moves the highest-ranked promotion blocker toward apply, impact, widen, rollback, or managed feedback work."
            ),
            sequencing="Sequence before generic telemetry-derived proposals when it is the freshest promotion blocker next-action status.",
        ),
    }


def _post_promotion_priority_action_title(action: str, family: str) -> str:
    action_l = action.lower().replace("_", "-")
    family_l = family.lower().replace("_", "-") or "optimization"
    if "rollback" in action_l or "disable" in action_l:
        return f"Rollback post-promotion {family_l} policy from priority deltas"
    if "widen" in action_l:
        return f"Widen post-promotion {family_l} policy from priority deltas"
    if "holdout" in action_l or "coverage" in action_l or "collect" in action_l:
        return f"Collect post-promotion {family_l} holdout evidence from priority deltas"
    if "flush" in action_l or "managed" in action_l or "outcome" in action_l or "feedback" in action_l:
        return f"Flush post-promotion {family_l} outcomes from priority deltas"
    if "keep-blocked" in action_l or "blocked" in action_l:
        return f"Keep post-promotion {family_l} policy blocked with priority-delta reason"
    return f"Advance post-promotion {family_l} priority-delta successor"


def _proposal_from_post_promotion_priority_deltas(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    status = stats_summary.get("post_promotion_priority_delta_status")
    if not isinstance(status, dict):
        return None
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    rows = [row for row in status.get("next_actions") or [] if isinstance(row, dict)]
    top = rows[0] if rows else {}
    next_action = str(summary.get("top_next_action") or top.get("next_action") or "").strip()
    if not next_action:
        return None
    family = str(summary.get("top_local_action_family") or top.get("local_action_family") or "optimization").strip() or "optimization"
    title = _post_promotion_priority_action_title(next_action, family)
    priority = "priority:p1" if next_action in {"widen-local-policy", "collect-holdout-evidence", "rollback-local-policy"} else "priority:p2"
    labels = _default_issue_labels(priority)
    family_label = family.replace("_", "-")
    if family_label in {"routing", "cache", "crunch"}:
        labels.append(family_label)
    labels.append("privacy")

    evidence = [
        f"Source metadata: {status.get('source_schema')}",
        f"Source status: {status.get('source_status')}",
        f"Top local action family: {family}",
        f"Top next action: {next_action}",
        f"Priority review candidate count: {summary.get('priority_review_candidate_count')}",
        f"Recommended count: {summary.get('recommended_count')}",
        f"No-op count: {summary.get('noop_count')}",
        f"Widen count: {summary.get('widen_count')}",
        f"Collect holdout evidence count: {summary.get('collect_holdout_evidence_count')}",
        f"Rollback count: {summary.get('rollback_count')}",
        f"Keep-blocked count: {summary.get('keep_blocked_count')}",
        f"Policy draft status: {summary.get('policy_draft_status')}",
        f"Impact gate status: {summary.get('impact_gate_status')}",
        f"Outcome flush status: {summary.get('outcome_flush_status')}",
        f"Outcome rollup count: {summary.get('outcome_rollup_count')}",
        f"Freshness state: {summary.get('freshness_state')}",
    ]
    if top:
        evidence.extend(
            [
                f"Top row count: {top.get('count')}",
                f"Top row savings delta USD: {top.get('savings_delta_usd')}",
                f"Top row status: {top.get('status')}",
                f"Top row policy section: {top.get('policy_section')}",
                f"Top row no-op reasons: {', '.join(str(reason) for reason in top.get('no_op_reasons') or []) or top.get('top_no_op_reason') or 'none'}",
            ]
        )

    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": list(dict.fromkeys(labels)),
        "body": _issue_body(
            title=title,
            rationale=(
                "Research mode found a post-promotion priority-delta artifact. "
                "The next backlog item should advance the priority review's successor action instead of recreating stale stage, replay, or rank issues that were already closed."
            ),
            evidence=evidence,
            implementation=[
                "Start from the persisted post_promotion_priority_delta_status section in the research plan and the local post-promotion priority review or handoff artifact.",
                f"Implement the successor action `{next_action}` for the `{family}` family using local policy draft, canary review, rollback, holdout, or managed-feedback flush modules.",
                "Suppress closed stage/replay/rank predecessor titles when their evidence is represented by the current priority-delta successor.",
                "Do not inspect prompts, provider bodies, request IDs, session IDs, cache keys, file paths, or individual candidate identifiers.",
                "Record the outcome in machine-readable post-promotion policy draft, impact, rollback, holdout, or feedback metadata so the next research plan can rank the following action.",
            ],
            acceptance=[
                f"The next research plan creates a successor issue for next_action={next_action} and does not recreate closed stage/replay/rank titles for the same milestone.",
                "The follow-up reports policy draft, impact gate, rollback, holdout, or outcome flush status for the post-promotion priority-delta action.",
                "Generated and follow-up evidence remains metadata-only and excludes prompts, provider bodies, absolute paths, request IDs, session IDs, cache keys, and individual candidate IDs.",
            ],
            savings_path=(
                "This keeps the unattended savings loop moving from post-promotion evidence into local widen, rollback, keep-blocked, holdout, or feedback work instead of cycling on completed staging issues."
            ),
            sequencing="Sequence before generic telemetry-derived proposals when a fresh post-promotion priority-delta artifact is present.",
        ),
    }


def _evidence_ledger_title_token(entry: dict[str, Any]) -> str:
    fingerprint = str(entry.get("fingerprint") or "").strip()
    if fingerprint:
        suffix = fingerprint.rsplit(":", 1)[-1]
        suffix = re.sub(r"[^a-zA-Z0-9]+", "", suffix)
        if suffix:
            return f"evidence {suffix[:12]}"
    cohort = str(entry.get("cohort_bucket") or "").strip()
    if cohort:
        words = re.sub(r"[^a-zA-Z0-9]+", "-", cohort).strip("-").lower()
        if words:
            return f"cohort {words[:32]}"
    return "current evidence"


def _evidence_ledger_action_title(entry: dict[str, Any]) -> str:
    lever = str(entry.get("lever") or "optimization").replace("_", "-")
    action = str(entry.get("next_action") or "advance-local-evidence").lower().replace("_", "-")
    promotion_readiness = str(entry.get("promotion_readiness") or "").lower().replace("_", "-")
    reason_codes = {str(item).lower().replace("_", "-") for item in entry.get("reason_codes") or []}
    if lever == "cache" and "retire-cache-replay-canary" in action:
        base = "Record cache replay canary retirement in evidence ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    if (
        lever == "cache"
        and (
            action == "keep-cache-replay-canary-staged"
            or promotion_readiness == "keep-staged-warmup"
            or "first-seen-cache-warmup" in reason_codes
        )
    ):
        base = "Record cache replay canary warmup carry-forward in evidence ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    if lever == "cache" and "stage" in action:
        base = "Stage cache replay canary from evidence-to-activation ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    if lever == "routing" and ("widen" in action or "activate" in action or "stage" in action):
        base = "Advance routing activation from evidence-to-activation ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    if lever == "crunch" and ("measure" in action or "impact" in action):
        base = "Measure crunch activation from evidence-to-activation ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    if lever == "request-shape-rollups":
        base = "Advance repeated-context cohort from evidence-to-activation ledger"
        family = str(entry.get("local_action_family") or "").strip()
        if ("widen" in action or "apply" in action) and family == "crunch":
            base = "Apply measured request-shape crunch widening from evidence-to-activation ledger"
        elif ("measure" in action or "impact" in action) and family == "crunch":
            base = "Measure request-shape crunch canary from evidence-to-activation ledger"
        elif "crunch" in action or family == "crunch":
            base = "Stage request-shape repeated-context crunch canary from evidence-to-activation ledger"
        elif "cache" in action or "replay" in action or family == "cache":
            base = "Advance request-shape cache replay cohort from evidence-to-activation ledger"
        elif "routing" in action or family == "routing":
            base = "Advance request-shape routing lifecycle evidence from evidence-to-activation ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    if lever == "managed-recommendation":
        base = "Advance managed recommendation handoff from evidence-to-activation ledger"
        return f"{base} ({_evidence_ledger_title_token(entry)})"
    return f"Advance {lever} next action from evidence-to-activation ledger ({_evidence_ledger_title_token(entry)})"


def _ledger_entry_has_progressed_next_action(entry: dict[str, Any]) -> bool:
    action = str(entry.get("next_action") or "").lower().replace("_", "-")
    previous = str(entry.get("fingerprint_next_action") or entry.get("lifecycle_progressed_from_next_action") or "").lower().replace("_", "-")
    if previous and action and action != previous:
        return True
    return any(token in action for token in ("review-", "measure-", "keep-active", "keep-cache-replay-canary-staged", "promotion-readiness", "widen", "apply", "retire"))


def _ledger_entry_successor_rank(entry: dict[str, Any]) -> int:
    action = str(entry.get("next_action") or "").lower().replace("_", "-")
    lever = str(entry.get("lever") or "").lower().replace("_", "-")
    promotion_readiness = str(entry.get("promotion_readiness") or "").lower().replace("_", "-")
    if lever == "cache" and (action == "keep-cache-replay-canary-staged" or promotion_readiness == "keep-staged-warmup"):
        return 0
    if lever == "cache" and "retire-cache-replay-canary" in action:
        return 0
    if lever == "cache" and "review-cache-replay-canary-promotion-readiness" in action:
        return 0
    if _ledger_entry_has_progressed_next_action(entry):
        return 1
    return 2


def _proposal_from_evidence_to_activation_ledger(stats_summary: dict[str, Any]) -> dict[str, Any] | None:
    ledger = stats_summary.get("evidence_to_activation_next_action_ledger")
    if not isinstance(ledger, dict):
        return None
    entries = [entry for entry in ledger.get("entries") or [] if isinstance(entry, dict)]
    if not entries:
        return None
    advanced_statuses = {"staged", "applied", "holdout", "measured", "full-rollout", "closed-issue-seen"}
    advanced = [
        entry for entry in entries
        if str(entry.get("issue_status") or "") == "closed-issue-seen"
        and str(entry.get("current_status") or "") in advanced_statuses
    ]
    if not advanced:
        return None
    advanced.sort(
        key=lambda item: (
            _ledger_entry_successor_rank(item),
            _to_int(item.get("rank"), 999),
        )
    )
    entry = advanced[0]
    next_action = str(entry.get("next_action") or "").strip()
    if not next_action:
        return None
    lever = str(entry.get("lever") or "optimization")
    title = _evidence_ledger_action_title(entry)
    labels = _default_issue_labels("priority:p2")
    lever_label = lever.replace("_", "-")
    if lever_label in {"routing", "cache", "crunch"}:
        labels.append(lever_label)
    action_family_label = str(entry.get("local_action_family") or "").strip().replace("_", "-")
    if action_family_label in {"routing", "cache", "crunch"}:
        labels.append(action_family_label)
    labels.append("privacy")
    closed_note = ""
    prior = entry.get("prior_issue") if isinstance(entry.get("prior_issue"), dict) else {}
    if entry.get("issue_status") == "closed-issue-seen" and prior:
        closed_note = (
            f"Continues closed predecessor: #{prior.get('number')} {prior.get('title')} "
            f"({prior.get('url') or 'no-url'})"
        )
    entry_evidence = [
        f"Ledger schema: {ledger.get('schema')}",
        f"Fingerprint: {entry.get('fingerprint')}",
        f"Lever: {lever}",
        f"Local action family: {entry.get('local_action_family')}",
        f"Evidence schema: {entry.get('evidence_schema')}",
        f"Cohort bucket: {entry.get('cohort_bucket')}",
        f"Current status: {entry.get('current_status')}",
        f"Issue status: {entry.get('issue_status')}",
        f"Top next action: {next_action}",
        f"Blocker codes: {json.dumps(entry.get('blocker_codes') or [])}",
        f"Expected savings path: {entry.get('expected_savings_path')}",
        closed_note,
    ]
    if entry.get("promotion_readiness"):
        entry_evidence.append(f"Promotion readiness: {entry.get('promotion_readiness')}")
    if entry.get("promotion_decision"):
        entry_evidence.append(f"Promotion decision: {entry.get('promotion_decision')}")
    if entry.get("policy_decision"):
        entry_evidence.append(f"Policy decision: {entry.get('policy_decision')}")
    if entry.get("observed_hit_blocker"):
        entry_evidence.append(f"Observed hit blocker: {entry.get('observed_hit_blocker')}")
    if entry.get("miss_reason_breakdown"):
        entry_evidence.append(f"Warmup miss blocker breakdown: {json.dumps(entry.get('miss_reason_breakdown') or [])}")
    if entry.get("duplicate_suppression"):
        entry_evidence.append(f"Duplicate suppression: {json.dumps(entry.get('duplicate_suppression') or {}, sort_keys=True)}")
    return {
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": list(dict.fromkeys(labels)),
        "body": _issue_body(
            title=title,
            rationale=(
                "The local evidence-to-activation ledger shows that a savings candidate has progressed past the earlier backlog title. "
                "Research mode should create the next lifecycle action instead of recreating stale projected-evidence work."
            ),
            evidence=entry_evidence,
            implementation=[
                "Start from the evidence_to_activation_next_action_ledger entry in the research plan.",
                f"Implement or narrow the lifecycle transition `{next_action}` for the `{lever}` lever using local file-backed policy and bounded metadata reports.",
                "Do not inspect prompts, provider bodies, request IDs, session IDs, cache keys, file paths, or individual candidate identifiers.",
                "Write the resulting applied, holdout, measured, blocked, safety-stopped, superseded, or closed-issue-seen status back into metadata so later research runs suppress stale titles.",
            ],
            acceptance=[
                f"The next research plan reports progress for ledger fingerprint {entry.get('fingerprint')} and next_action={next_action}, or records a narrower blocker.",
                "Closed earlier issue titles for the same fingerprint are not recreated when the ledger status has advanced.",
                "The ledger output names top next action, current status, blocker codes, expected savings path, and privacy flags for each tracked lever.",
                "Generated and follow-up evidence remains metadata-only and excludes prompts, provider bodies, file paths, request IDs, session IDs, cache keys, and individual candidate IDs.",
            ],
            savings_path=str(entry.get("expected_savings_path") or "This removes duplicate backlog churn before local activation work."),
            sequencing="Sequence before generic telemetry-derived proposals when the ledger names a later lifecycle transition for the same candidate.",
        ),
    }


def _proposal_from_repeated_diagnostic(
    diagnostic: dict[str, Any],
    *,
    fingerprint: str = "",
) -> dict[str, Any]:
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
    lifecycle_context = diagnostic.get("lifecycle_context") if isinstance(diagnostic.get("lifecycle_context"), dict) else {}
    fingerprint = fingerprint or _diagnostic_fingerprint(diagnostic_class)
    ledger_stage = _diagnostic_ledger_stage(diagnostic) or {}
    ledger_next_action = str(ledger_stage.get("next_action") or "").strip()
    ledger_current_status = _ledger_status_from_stage(ledger_stage) if ledger_stage else ""
    ledger_action_family = str(ledger_stage.get("local_action_family") or "").strip()
    ledger_verification_check = str(ledger_stage.get("verification_check") or "").strip()
    reclassification_source = str(diagnostic.get("reclassification_source") or "")
    is_bounded_human_review = reclassification_source == "no-match-bounded-human-review"
    if is_bounded_human_review:
        title_reason = "unclassified activation skip or blocker"
    else:
        title_reason = diagnostic_class.replace("-", " ")
    evidence_count = _to_int(diagnostic.get("count", 0))
    example_excerpt = str(diagnostic.get("example") or "")[:120]
    evidence = [
        f"Diagnostic reason: {reason}",
        f"Diagnostic class: {diagnostic_class}",
        f"Source lever: {source_lever}",
        f"Backlog action: {backlog_action}",
        f"Evidence count: {evidence_count}",
        f"Example excerpt: {example_excerpt}",
        f"Proposed owner: local-policy",
        f"Fingerprint: {fingerprint}",
        f"Action: create",
        f"Expected unblock path: {expected_unblock_path}",
    ]
    if ledger_stage:
        evidence.extend(
            [
                f"Ledger next action: {ledger_next_action}",
                f"Ledger current status: {ledger_current_status}",
                f"Ledger local action family: {ledger_action_family}",
            ]
        )
        if ledger_verification_check:
            evidence.append(f"Ledger verification check: {ledger_verification_check}")
    if is_bounded_human_review:
        evidence.extend([
            "Source schema: tokenclaw.orchestrator_research_log_diagnostics.v1",
            f"Report key: repeated_diagnostics.{diagnostic_class}",
            "Privacy: metadata-only, telemetry_profile=metadata-only, no raw prompts, no provider bodies, no request or session IDs",
        ])
    if lifecycle_context:
        evidence.extend(
            [
                f"Lifecycle source: {lifecycle_context.get('source')}",
                f"Lifecycle report schema: {lifecycle_context.get('report_schema')}",
                f"Lifecycle action family: {lifecycle_context.get('action_family')}",
                f"Lifecycle state: {lifecycle_context.get('lifecycle_state')}",
                f"Lifecycle blocker code: {lifecycle_context.get('blocker_code')}",
                f"Lifecycle sample count bucket: {lifecycle_context.get('sample_count_bucket')}",
                f"Lifecycle next action: {lifecycle_context.get('next_action')}",
            ]
        )
    return {
        "repo": "lutzkuen/tokenclaw",
        "title": f"Turn repeated {title_reason} diagnostics into an actionable optimization issue",
        "labels": _default_issue_labels("priority:p2"),
        "body": _issue_body(
            title=f"Turn repeated {title_reason} diagnostics into an actionable optimization issue",
            rationale=(
                "Research mode found the same skip or blocker diagnostic repeatedly. That should become a "
                "narrow follow-up issue instead of disappearing into run prose."
            ),
            evidence=[
                *evidence,
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


def _repeated_diagnostic_comment_for_issue(
    diagnostic: dict[str, Any],
    fingerprint: str,
    issue: dict[str, Any],
) -> dict[str, Any]:
    reason = str(diagnostic.get("reason") or "unknown-diagnostic")
    diagnostic_class = str(diagnostic.get("diagnostic_class") or reason)
    source_lever = str(diagnostic.get("source_lever") or _diagnostic_source_lever(reason, diagnostic_class))
    evidence_count = _to_int(diagnostic.get("count", 0))
    example_excerpt = str(diagnostic.get("example") or "")[:120]
    ledger_stage = _diagnostic_ledger_stage(diagnostic) or {}
    ledger_evidence = []
    if ledger_stage:
        ledger_evidence = [
            f"Ledger next action: {ledger_stage.get('next_action')}",
            f"Ledger current status: {_ledger_status_from_stage(ledger_stage)}",
            f"Ledger local action family: {ledger_stage.get('local_action_family')}",
        ]
        if ledger_stage.get("verification_check"):
            ledger_evidence.append(f"Ledger verification check: {ledger_stage.get('verification_check')}")
    body = _issue_body(
        title=str(issue.get("title") or ""),
        rationale=(
            "Research mode found the same repeated diagnostic again. "
            "The open issue already tracks this pattern; this comment adds current evidence."
        ),
        evidence=[
            f"Diagnostic class: {diagnostic_class}",
            f"Source lever: {source_lever}",
            f"Evidence count: {evidence_count}",
            f"Example excerpt: {example_excerpt}",
            f"Proposed owner: local-policy",
            f"Fingerprint: {fingerprint}",
            f"Action: update",
            f"Duplicate of open issue: #{_issue_number(issue)}",
            *ledger_evidence,
        ],
        implementation=[
            "Check whether the original blocker is still present in the current reports or stats.",
            "If still blocked, keep the issue open and add the missing evidence or dependency as a narrower issue.",
            "If resolved, remove status:blocked, add status:ready, and record the concrete next acceptance metric.",
        ],
        acceptance=[
            "The open issue has a current evidence comment with sanitized diagnostic metadata.",
            "Generated text remains metadata-only and contains no raw prompts, provider bodies, file paths, or request/session IDs.",
        ],
        sequencing="Use before creating duplicate replacement issues for the same diagnostic pattern.",
    )
    return {
        "repo": issue.get("repo") or "lutzkuen/tokenclaw",
        "number": _issue_number(issue),
        "action": "comment",
        "body": body,
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


def _cache_action_from_skipped_openai_next_action(next_action: Any) -> tuple[str, str]:
    action = str(next_action or "").strip()
    if action == "add-invalidation-evidence":
        return "collect-dependency-evidence", "collect local invalidation and dependency evidence"
    if action == "collect-more-repeat-evidence":
        return "accept-non-repeatable-traffic", "collect more repeat evidence before replay activation"
    if action == "wait-for-streaming-replay-support":
        return "accept-non-repeatable-traffic", "keep streaming replay blocked until replay support exists"
    if action == "keep-tool-cache-disabled":
        return "collect-dependency-evidence", "keep tool-call replay disabled while collecting invalidation evidence"
    if action == "unsupported-endpoint":
        return "accept-non-repeatable-traffic", "keep unsupported endpoint replay blocked"
    if action == "already-cache-hit":
        return "accept-non-repeatable-traffic", "keep already-hit traffic out of replay activation work"
    return "instrument-cache-decision", "rank or instrument the cache replay blocker"


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

    shape_signal = stats_summary.get("request_shape_rollup_candidates")
    shape_replay = shape_signal.get("cache_replayability_dry_run") if isinstance(shape_signal, dict) else None
    if isinstance(shape_replay, dict):
        replay_summary = shape_replay.get("summary") if isinstance(shape_replay.get("summary"), dict) else {}
        remaining_rows = (
            shape_replay.get("remaining_replay_ready_cohorts")
            if isinstance(shape_replay.get("remaining_replay_ready_cohorts"), list)
            else []
        )
        rows_to_scan = remaining_rows or shape_replay.get("cohorts") or []
        remaining_replay_ready_rows = _to_int(replay_summary.get("remaining_replay_ready_rows"))
        remaining_fields_present = (
            "remaining_replay_ready_cohort_count" in replay_summary
            or "remaining_replay_ready_rows" in replay_summary
        )
        for row in rows_to_scan:
            if not isinstance(row, dict):
                continue
            readiness = str(row.get("readiness") or "")
            if readiness != "replay-ready":
                continue
            if bool(row.get("handled_by_local_policy")) or row.get("remaining_replay_ready") is False:
                continue
            blockers = [str(item) for item in row.get("blockers") or [] if str(item or "").strip()]
            issue_row = dict(row)
            issue_row["next_action_family"] = "stage-replay-policy"
            issue_row["next_action_label"] = _CACHE_ACTION_LABELS["stage-replay-policy"]
            issue_row["provider"] = issue_row.get("provider") or issue_row.get("provider_family")
            issue_row["count"] = _to_int(issue_row.get("count") or issue_row.get("row_count"))
            issue_row["projected_saved_cost_usd"] = round(
                _to_float(issue_row.get("projected_saved_cost_usd") or issue_row.get("projected_savings_usd")),
                8,
            )
            issue_row["blocker_code"] = "replay-ready" if not blockers else blockers[0]
            return issue_row, "request_shape_cache_replayability_dry_run"
        skipped_openai = (
            shape_replay.get("skipped_openai_blockers")
            if isinstance(shape_replay.get("skipped_openai_blockers"), dict)
            else {}
        )
        if (not remaining_fields_present or remaining_replay_ready_rows <= 0) and isinstance(skipped_openai, dict):
            skipped_rows = [row for row in skipped_openai.get("cohorts") or [] if isinstance(row, dict)]
            if skipped_rows:
                row = dict(skipped_rows[0])
                action, label = _cache_action_from_skipped_openai_next_action(row.get("next_action"))
                blockers = [str(item) for item in row.get("blocker_codes") or [] if str(item or "").strip()]
                row["next_action_family"] = action
                row["next_action_label"] = label
                row["provider"] = row.get("provider") or row.get("provider_family")
                row["count"] = _to_int(row.get("count") or row.get("row_count") or row.get("sample_count"))
                row["projected_saved_cost_usd"] = round(
                    _to_float(row.get("projected_saved_cost_usd") or row.get("projected_savings_usd")),
                    8,
                )
                row["blocker_code"] = (
                    str(row.get("blocker_code") or "").strip()
                    or (blockers[0] if blockers else str(row.get("reason") or "cache-replayability-blocker"))
                )
                row["readiness"] = row.get("readiness") or "skipped"
                row["local_action_family"] = row.get("local_action_family") or "cache"
                row["remaining_replay_ready_rows"] = remaining_replay_ready_rows
                row["skipped_openai_summary"] = sanitize_value(
                    skipped_openai.get("summary") if isinstance(skipped_openai.get("summary"), dict) else {}
                )
                return row, "request_shape_skipped_openai_cache_replay_blockers"

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
    elif source == "request_shape_cache_replayability_dry_run" and bool(row.get("remaining_replay_ready")):
        title = f"Stage remaining replay-ready cache cohort for {cohort}"
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
        "Use the bounded cache zero-hit blocker ladder, replay cohort ranking, or request-shape replayability dry-run; do not inspect prompts, responses, cache keys, file paths, request IDs, or session IDs.",
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
        "repo": "lutzkuen/tokenclaw",
        "title": title,
        "labels": ["backlog", "status:ready", "priority:p1", "core-feature", "correctness", "cache", "privacy"],
        "body": _issue_body(
            title=title,
            rationale=(
                "Research mode found a zero-hit cache window with named replayability evidence or a blocker cohort. "
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
        "repo": "lutzkuen/tokenclaw",
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
    summary = _stats_summary(stats)
    precomputed_activation_ledger = (
        summary.get("evidence_to_activation_next_action_ledger")
        if isinstance(summary.get("evidence_to_activation_next_action_ledger"), dict)
        else None
    )
    diagnostics = _resolve_unclassified_diagnostics(
        _resolve_aggregate_only_diagnostics(_diagnostics_from_logs(log_sources), summary),
        summary,
    )
    activation_safety_stop_burndown = None
    if diagnostics or isinstance(summary.get("pass_through_routing_report"), dict):
        from tokenclaw.activation_lifecycle_feedback import build_activation_safety_stop_burndown

        stats_summary: dict[str, Any] = {}
        if isinstance(summary.get("pass_through_routing_report"), dict):
            stats_summary["pass_through_routing_report"] = summary["pass_through_routing_report"]
        activation_safety_stop_burndown = build_activation_safety_stop_burndown(
            research_plan={
                "schema": SCHEMA,
                "evidence": {
                    "repeated_diagnostics": diagnostics,
                    "stats_summary": stats_summary,
                },
            }
        )
    activation_ledger = build_evidence_to_activation_next_action_ledger(
        summary,
        existing_issues=issue_list,
        diagnostics=diagnostics,
        safety_stop_burndown=activation_safety_stop_burndown,
    )
    if activation_ledger is not None:
        activation_ledger = _merge_precomputed_ledger_context(activation_ledger, precomputed_activation_ledger)
        summary["evidence_to_activation_next_action_ledger"] = activation_ledger
    elif precomputed_activation_ledger is not None:
        summary["evidence_to_activation_next_action_ledger"] = precomputed_activation_ledger
    full_rollout_crunch_measurement = _full_rollout_crunch_activation_measurement(summary)
    if full_rollout_crunch_measurement is not None:
        summary["full_rollout_crunch_activation_measurement"] = full_rollout_crunch_measurement
        if isinstance(summary.get("evidence_to_activation_next_action_ledger"), dict):
            summary["evidence_to_activation_next_action_ledger"] = _merge_full_rollout_crunch_measurement_into_ledger(
                summary["evidence_to_activation_next_action_ledger"],
                full_rollout_crunch_measurement,
            )
    precomputed_activation_queue = summary.get("local_activation_next_action_queue")
    activation_queue = (
        precomputed_activation_queue
        if isinstance(precomputed_activation_queue, dict)
        and precomputed_activation_queue.get("entries")
        else build_local_activation_next_action_queue(summary)
    )
    if activation_queue is not None:
        summary["local_activation_next_action_queue"] = activation_queue
    candidate_diagnostics, safety_stop_suppressed_diagnostics = _without_suppressed_safety_stop_diagnostics(
        diagnostics,
        activation_safety_stop_burndown,
    )
    candidate_diagnostics, activation_feedback_suppressed_diagnostics = (
        _without_suppressed_activation_feedback_blocker_review_diagnostics(candidate_diagnostics)
    )
    optimization_candidates = _optimization_candidates(stats_summary=summary, diagnostics=candidate_diagnostics)

    ready_count = len(ready_issues)
    should_run = threshold > 0 and ready_count < threshold
    trigger_reason = "ready-actionable-count-below-threshold" if should_run else "enough-ready-actionable-issues"

    create_issues: list[dict[str, Any]] = []
    comment_issues: list[dict[str, Any]] = []
    close_issues: list[dict[str, Any]] = []
    proposal_suppression = {
        "schema": "tokenclaw.research_issue_proposal_suppression.v1",
        "suppressed_count": 0,
        "closed_prior_issue_count": 0,
        "open_existing_issue_count": 0,
        "suppressed": [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
        },
    }

    if should_run:
        create_issues.append(
            _proposal_from_low_backlog(
                ready_count=ready_count,
                threshold=threshold,
                stats_summary=summary,
                diagnostics=diagnostics,
                optimization_candidates=optimization_candidates,
            )
        )
        post_promotion_priority_proposal = _proposal_from_post_promotion_priority_deltas(summary)
        if post_promotion_priority_proposal is not None:
            create_issues.append(post_promotion_priority_proposal)
        promotion_blocker_proposal = _proposal_from_promotion_blocker_next_action(summary)
        if promotion_blocker_proposal is not None:
            create_issues.append(promotion_blocker_proposal)
        ledger_proposal = _proposal_from_evidence_to_activation_ledger(summary)
        if ledger_proposal is not None:
            create_issues.append(ledger_proposal)
        cache_replay_proposal = _proposal_from_cache_replay_blocker(summary)
        if cache_replay_proposal is not None:
            create_issues.append(cache_replay_proposal)
        openai_routing_proposal = _proposal_from_openai_routing_canary_feedback(summary)
        if openai_routing_proposal is not None:
            create_issues.append(openai_routing_proposal)
        create_issues.extend(_proposals_from_activation_successor_decisions(summary))
        create_issues.extend(_proposals_from_optimization_candidates(optimization_candidates))
        repeated_actionable_diagnostics = [
            item for item in _actionable_diagnostics(candidate_diagnostics)
            if _to_int(item.get("count")) > 1 and item.get("backlog_action") != "needs-human-review"
        ]
        if repeated_actionable_diagnostics:
            top_diag = repeated_actionable_diagnostics[0]
            diag_class = str(top_diag.get("diagnostic_class") or top_diag.get("reason") or "")
            fingerprint = _diagnostic_fingerprint(diag_class)
            proposal = _proposal_from_repeated_diagnostic(top_diag, fingerprint=fingerprint)
            proposal_key = _issue_title_key(proposal.get("title"))
            matching_open = next(
                (
                    iss for iss in issue_list
                    if _is_open(iss) and proposal_key and _issue_title_key(iss.get("title")) == proposal_key
                ),
                None,
            )
            if matching_open is not None:
                comment_issues.append(
                    _repeated_diagnostic_comment_for_issue(top_diag, fingerprint, matching_open)
                )
            else:
                create_issues.append(proposal)
        for issue in blocked_stale[:3]:
            comment_issues.append(_blocked_comment(issue, diagnostics, summary))
        create_issues, proposal_suppression = _dedupe_create_issue_proposals_with_metadata(
            create_issues,
            existing_issues=issue_list,
            max_count=10,
            trusted_author=trusted_author,
            now=now,
        )
        if safety_stop_suppressed_diagnostics:
            proposal_suppression["suppressed"].extend(safety_stop_suppressed_diagnostics[:20])
            proposal_suppression["suppressed_count"] = _to_int(proposal_suppression.get("suppressed_count")) + len(
                safety_stop_suppressed_diagnostics
            )
            proposal_suppression["keep_blocked_ledger_suppressed_count"] = len(safety_stop_suppressed_diagnostics)
        if activation_feedback_suppressed_diagnostics:
            proposal_suppression["suppressed"].extend(activation_feedback_suppressed_diagnostics[:20])
            proposal_suppression["suppressed_count"] = _to_int(proposal_suppression.get("suppressed_count")) + len(
                activation_feedback_suppressed_diagnostics
            )
            proposal_suppression["activation_feedback_keep_blocked_suppressed_count"] = len(
                activation_feedback_suppressed_diagnostics
            )
        create_issues, golden_path_suppressed = _filter_golden_path_ready_proposals(create_issues)
        if golden_path_suppressed:
            proposal_suppression["suppressed"].extend(golden_path_suppressed[:20])
            proposal_suppression["suppressed_count"] = _to_int(proposal_suppression.get("suppressed_count")) + len(
                golden_path_suppressed
            )
            proposal_suppression["golden_path_readiness_suppressed_count"] = len(golden_path_suppressed)
        create_issues = [_finalize_create_issue_proposal(proposal) for proposal in create_issues]
        next_backlog_milestone = _next_backlog_milestone(
            create_issues=create_issues,
            optimization_candidates=optimization_candidates,
            stats_summary=summary,
        )
    else:
        next_backlog_milestone = {
            "schema": "tokenclaw.next_backlog_milestone.v1",
            "status": "not-needed",
            "summary": {
                "proposal_count": 0,
                "ranked_candidate_count": 0,
                "top_lever": None,
                "top_blocker": None,
                "top_next_action": None,
            },
            "issues": [],
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "absolute_paths_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
                "individual_candidate_ids_included": False,
            },
        }

    inspected_sources = ["github_issues"]
    if summary:
        inspected_sources.append("local_stats")
    if "post_promotion_priority_delta_status" in summary:
        inspected_sources.append("post_promotion_priority_delta_status")
    if "promotion_blocker_next_action_status" in summary:
        inspected_sources.append("promotion_blocker_next_action_status")
    if "evidence_to_activation_next_action_ledger" in summary:
        inspected_sources.append("evidence_to_activation_next_action_ledger")
    if "full_rollout_crunch_activation_measurement" in summary:
        inspected_sources.append("full_rollout_crunch_activation_measurement")
    if "local_activation_next_action_queue" in summary:
        inspected_sources.append("local_activation_next_action_queue")
    if "promotion_outcome_feedback" in summary:
        inspected_sources.append("promotion_outcome_feedback")
    if diagnostics:
        inspected_sources.append("orchestrator_logs")
    if isinstance(activation_safety_stop_burndown, dict) and activation_safety_stop_burndown.get("status") == "ranked":
        inspected_sources.append("activation_safety_stop_burndown")

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
            "activation_safety_stop_burndown": activation_safety_stop_burndown,
            "optimization_candidates": optimization_candidates if should_run else [],
            "issue_proposal_suppression": proposal_suppression,
            "next_backlog_milestone": next_backlog_milestone,
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
