from __future__ import annotations

import re
from typing import Any


SAFETY_STOP_REASON_SCHEMA = "agentflow.promotion_family_safety_stop_reason.v1"

_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:@+-]{0,119}$")


def _safe_reason(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return None
    return text if _REASON_RE.match(text) else "unsanitized-reason-code"


def _safe_reasons(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({reason for value in values if (reason := _safe_reason(value))})


def _family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"routing", "model-routing", "phase-routing", "openai-local-routing"} or "routing" in text:
        return "routing"
    if text in {"cache", "cache-replay", "cache-replayability"} or "cache" in text:
        return "cache"
    if text in {"crunch", "pattern", "old-context-summarization", "old-context-summary"}:
        return "crunch"
    if "summary" in text or "summarization" in text or "crunch" in text:
        return "crunch"
    return text or "unknown"


def _contains(reasons: set[str], *needles: str) -> bool:
    return any(needle in reason for reason in reasons for needle in needles)


def _result(
    *,
    family: str,
    category: str,
    code: str,
    matched: set[str],
    next_action: str,
    blocked_state: str,
    local_unblock_action: str,
) -> dict[str, Any]:
    return {
        "schema": SAFETY_STOP_REASON_SCHEMA,
        "action_family": family,
        "code": code,
        "category": category,
        "matched_reason_codes": sorted(matched)[:10],
        "next_action": next_action,
        "blocked_state": blocked_state,
        "local_unblock_action": local_unblock_action,
        "machine_readable": True,
        "metadata_only": True,
    }


def classify_family_safety_stop_reason(
    *,
    action_family: Any,
    reason: Any = None,
    reason_codes: Any = None,
    no_op_reasons: Any = None,
    blocker_family: Any = None,
    next_action: Any = None,
    file_backed_policy_exists: bool | None = None,
    executor_status: Any = None,
) -> dict[str, Any] | None:
    """Return a privacy-safe family-specific safety/blocker reason.

    The classifier intentionally uses only normalized metadata labels. It should
    never receive or emit prompt text, request IDs, cache keys, or path values.
    """

    family = _family(action_family)
    reasons = set(_safe_reasons(reason_codes))
    reasons.update(_safe_reasons(no_op_reasons))
    for value in (reason, blocker_family, next_action, executor_status):
        if safe := _safe_reason(value):
            reasons.add(safe)
    if file_backed_policy_exists is False:
        reasons.add("missing-file-backed-local-policy")
    if not reasons:
        return None

    if _contains(reasons, "privacy", "raw-prompts", "provider-bodies", "cache-keys"):
        return _result(
            family=family,
            category="privacy-boundary",
            code=f"{family}-privacy-boundary-blocked",
            matched={item for item in reasons if _contains({item}, "privacy", "raw-prompts", "provider-bodies", "cache-keys")},
            next_action="keep-blocked",
            blocked_state="keep-blocked",
            local_unblock_action="remove-sensitive-fields-before-review",
        )

    if _contains(reasons, "unsupported-local-policy-section", "unsupported-local-action-family", "executor-incompatible", "unsupported-executor", "capability-unavailable"):
        return _result(
            family=family,
            category="executor-incompatible",
            code=f"{family}-executor-incompatible",
            matched={item for item in reasons if _contains({item}, "unsupported", "executor", "capability-unavailable")},
            next_action="keep-blocked",
            blocked_state="keep-blocked",
            local_unblock_action="wait-for-compatible-local-executor",
        )

    if (
        file_backed_policy_exists is False
        or _contains(reasons, "missing-file-backed", "no-local-representation", "unknown-local-action-family")
    ):
        action = "add-file-backed-local-policy"
        if family in {"routing", "cache", "crunch"}:
            action = f"add-file-backed-{family}-rule"
        return _result(
            family=family,
            category="missing-file-backed-representation",
            code=f"{family}-missing-file-backed-representation",
            matched={item for item in reasons if _contains({item}, "missing-file-backed", "no-local-representation", "unknown-local-action-family")},
            next_action=action,
            blocked_state="unblockable",
            local_unblock_action=action,
        )

    if family == "cache" and _contains(
        reasons,
        "stale-dependency",
        "dependency-changed",
        "dependency-created",
        "dependency-deleted",
        "dependency-invalidated",
        "dependency-freshness",
        "file-dependency-invalidated",
        "stale-risk",
    ):
        return _result(
            family=family,
            category="dependency-instability",
            code="cache-dependency-instability",
            matched={item for item in reasons if _contains({item}, "dependency", "stale-risk")},
            next_action="fix-dependency-freshness",
            blocked_state="unblockable",
            local_unblock_action="refresh-cache-dependency-evidence",
        )

    if family == "cache" and _contains(reasons, "missing-invalidation", "safe-invalidation", "file-dependency-missing", "dependency-missing", "file-watch-disabled", "tool-call-cache-disabled"):
        return _result(
            family=family,
            category="missing-invalidation-evidence",
            code="cache-missing-invalidation-evidence",
            matched={item for item in reasons if _contains({item}, "invalidation", "dependency-missing", "file-dependency-missing", "file-watch", "tool-call-cache")},
            next_action="fix-dependency-freshness",
            blocked_state="unblockable",
            local_unblock_action="collect-cache-invalidation-evidence",
        )

    if _contains(reasons, "insufficient-canary-holdout", "missing-holdout", "holdout-coverage"):
        return _result(
            family=family,
            category="missing-holdout-coverage",
            code=f"{family}-missing-holdout-coverage",
            matched={item for item in reasons if _contains({item}, "holdout")},
            next_action="collect-canary-holdout",
            blocked_state="unblockable",
            local_unblock_action="collect-canary-holdout",
        )

    if _contains(reasons, "insufficient-canary-applied", "missing-applied", "missing-canary-lifecycle", "applied-coverage"):
        return _result(
            family=family,
            category="missing-lifecycle-evidence",
            code=f"{family}-missing-lifecycle-evidence",
            matched={item for item in reasons if _contains({item}, "applied", "canary-lifecycle")},
            next_action="collect-canary-applied",
            blocked_state="unblockable",
            local_unblock_action="collect-canary-lifecycle-evidence",
        )

    if _contains(reasons, "stale-evidence", "stale-eval", "stale-lifecycle", "impact-stale"):
        return _result(
            family=family,
            category="stale-lifecycle-evidence",
            code=f"{family}-stale-lifecycle-evidence",
            matched={item for item in reasons if _contains({item}, "stale", "impact-stale")},
            next_action="refresh-lifecycle-evidence",
            blocked_state="unblockable",
            local_unblock_action="rerun-local-impact-report",
        )

    if _contains(reasons, "safety-stop", "rollback", "regression", "eval-failed", "quality-gate", "summary-failure", "error-rate", "retry-rate", "latency-regression"):
        return _result(
            family=family,
            category="quality-gate-failure",
            code=f"{family}-quality-gate-failed",
            matched={item for item in reasons if _contains({item}, "safety", "rollback", "regression", "eval-failed", "quality", "failure", "error-rate", "retry-rate", "latency")},
            next_action=f"review-{family}-quality-gate" if family in {"routing", "cache", "crunch"} else "review-safety-stop",
            blocked_state="keep-blocked",
            local_unblock_action="inspect-quality-regression-before-retry",
        )

    return None
