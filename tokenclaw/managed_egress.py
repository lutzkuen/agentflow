from __future__ import annotations

from typing import Any


RAW_FEATURE_KEYS = {
    "account_id",
    "arguments",
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "command",
    "command_text",
    "commands",
    "completion",
    "content",
    "developer",
    "file_content",
    "file_contents",
    "file_path",
    "file_paths",
    "filepath",
    "file_dependency_fingerprint",
    "file_dependency_fingerprint_sha256",
    "fingerprint",
    "fingerprints",
    "generated_summary",
    "generated_summaries",
    "input",
    "local_file",
    "local_path",
    "local_paths",
    "local_session_id",
    "local_session_ids",
    "message",
    "messages",
    "output",
    "params",
    "pattern_text",
    "prompt",
    "provider_body",
    "provider_request",
    "provider_response",
    "payload",
    "payloads",
    "raw_context",
    "raw_context_turns",
    "raw_messages",
    "raw_old_context",
    "raw_payload",
    "raw_payloads",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_body",
    "request_fingerprint",
    "request_fingerprint_sha256",
    "request_fingerprints",
    "request_id",
    "request_ids",
    "response",
    "response_body",
    "secret",
    "session_id",
    "session_ids",
    "summary_prompt",
    "summary_prompts",
    "summary_text",
    "system",
    "system_prompt",
    "tenant_id",
    "tenant_ids",
    "thread_id",
    "thread_ids",
    "token",
    "tool_input",
    "tool_output",
    "tool_payload",
    "tool_payloads",
    "transcript",
    "transcripts",
    "working_directory",
    "workspace_path",
}

LIFECYCLE_METADATA_COMMAND_SCHEMAS = {
    "tokenclaw.old_context_summary_lifecycle_metadata.v1",
    "tokenclaw.phase_routing_lifecycle_metadata.v1",
    "tokenclaw.optimization_promotion_lifecycle_feedback.v1",
    "tokenclaw.optimization_coordinator_lifecycle_feedback.v1",
    "tokenclaw.repeated_scaffold_lifecycle_feedback.v1",
    "tokenclaw.instruction_dedup_lifecycle_feedback.v1",
    "tokenclaw.codex_terminal_transcript_compaction_lifecycle_feedback.v1",
    "tokenclaw.rollout_action_lifecycle_metadata.v1",
    "tokenclaw.terminal_output_compaction_lifecycle_feedback.v1",
}


class ManagedEgressBlocked(ValueError):
    def __init__(self, violations: list[dict[str, str]]):
        super().__init__("managed egress payload contains raw-like fields")
        self.violations = violations


def managed_egress_violations(value: Any, *, max_results: int = 20) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []

    def visit(item: Any, path: str) -> None:
        if len(violations) >= max_results:
            return
        if isinstance(item, dict):
            allow_command = item.get("schema") in LIFECYCLE_METADATA_COMMAND_SCHEMAS
            for key, child in item.items():
                key_text = str(key)
                key_l = key_text.lower()
                child_path = f"{path}.{key_text}" if path else key_text
                if key_l in RAW_FEATURE_KEYS and not (key_l == "command" and allow_command):
                    violations.append({
                        "path": child_path,
                        "key": key_l,
                        "reason": "raw-like-key",
                    })
                    if len(violations) >= max_results:
                        return
                    continue
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if len(violations) >= max_results:
                    return
                visit(child, f"{path}[{index}]")

    visit(value, "$")
    return violations


def assert_managed_egress_safe(payload: Any) -> None:
    violations = managed_egress_violations(payload)
    if violations:
        raise ManagedEgressBlocked(violations)


def managed_egress_blocked_meta(
    *,
    endpoint: str | None,
    violations: list[dict[str, str]],
    optimization_unit_id: int | None = None,
    queue_id: str | None = None,
) -> dict[str, Any]:
    blocked_keys = sorted({item.get("key", "unknown") for item in violations})
    meta: dict[str, Any] = {
        "endpoint": endpoint,
        "status": "skipped",
        "reason": "unsafe-egress-payload",
        "fallback": "local-policy",
        "applied": False,
        "egress_guard": {
            "schema": "tokenclaw.managed_egress_guard.v1",
            "blocked": True,
            "violation_count": len(violations),
            "blocked_keys": blocked_keys,
            "first_violation_path": violations[0].get("path") if violations else None,
            "raw_values_logged": False,
        },
    }
    if optimization_unit_id is not None:
        meta["optimization_unit_id"] = optimization_unit_id
    if queue_id:
        meta["queue_id"] = queue_id
    return meta
