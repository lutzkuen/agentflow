from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.prompt_features import PROMPT_DIFFICULTY_FEATURE_SCHEMA, prompt_difficulty_features_from_text
from agentflow_proxy.store import stable_json
from agentflow_proxy.terminal_features import TERMINAL_LOG_FEATURE_SCHEMA, terminal_log_features_from_text


PATTERN_MODULES_SCHEMA = "agentflow.local_pattern_modules.v1"
PATTERN_MODULE_FEATURES_SCHEMA = "agentflow.local_pattern_module_features.v1"
TOOL_RESULT_FEATURE_SCHEMA = "agentflow.tool_result_features.v1"
TOKEN_CHARS = 4


_GREP_RESULT_RE = re.compile(r"^\s*[^:\s][^:\n]{0,240}:\d+(?::\d+)?:")
_SQL_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SQL_BORDER_RE = re.compile(r"^\s*[+\-|=\s:]{6,}\s*$")
_ISSUE_LIST_RE = re.compile(r"(?:^|\s)(?:#\d{1,8}|[A-Z][A-Z0-9]{1,10}-\d{1,8})(?:\s|$)|github\.com/.+/(?:issues|pull)/\d+", re.IGNORECASE)
_TEST_RESULT_RE = re.compile(
    r"\b(?:FAILED|FAILURES|ERROR|AssertionError|Traceback|passed|failed|skipped|Ran\s+\d+\s+tests?|"
    r"short test summary info|test session starts|exit code\s+[1-9]\d*)\b",
    re.IGNORECASE,
)
_COMMAND_STREAM_RE = re.compile(r"^\s*(?:stdout|stderr|exit\s+code|return\s+code)\s*[:=]", re.IGNORECASE)
_FILE_LISTING_RE = re.compile(
    r"^\s*(?:[-dl][rwxstST-]{9}\s+|\d{1,8}\s+[\w./-]+$|(?:\.{0,2}/)?[\w .@+=:,~-]+(?:/|\.(?:py|js|ts|tsx|json|yaml|yml|md|txt|toml|sql|go|rs|java|sh|css|html))\s*$)"
)
_ERROR_RE = re.compile(r"\b(?:ERROR|ERR|FATAL|CRITICAL|Exception|Traceback|AssertionError|failed|failure|panic|exit code\s+[1-9]\d*)\b", re.IGNORECASE)
_TOOL_FRAMING_RE = re.compile(
    r"^\s*(?:"
    r"</?tool[_ -]?results?>"
    r"|(?:begin|end)\s+(?:tool|function)\s+results?"
    r"|(?:tool|function)\s+results?\s*:?"
    r"|(?:[-=]{2,})\s*(?:tool|function)\s+results?\s*(?:[-=]{2,})"
    r")\s*$",
    re.IGNORECASE,
)


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


def _fraction_bucket(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator <= 0:
        return "none"
    ratio = numerator / denominator
    if ratio < 0.10:
        return "lt_10pct"
    if ratio < 0.25:
        return "10_25pct"
    if ratio < 0.50:
        return "25_50pct"
    if ratio < 0.75:
        return "50_75pct"
    return "gte_75pct"


def _extract_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, list):
        for item in value:
            text = _extract_text(item)
            if text:
                parts.append(text)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in {"text", "content", "input", "system", "instructions"}:
                text = _extract_text(child)
                if text:
                    parts.append(text)
            elif isinstance(child, (list, dict)):
                text = _extract_text(child)
                if text:
                    parts.append(text)
    return "\n".join(parts)


@dataclass(frozen=True)
class PatternModuleContext:
    body: dict[str, Any]
    text: str
    category: str | None
    policy_source: str
    rule_path: str | None = None


@dataclass(frozen=True)
class PatternDetection:
    detected: bool
    reason: str
    confidence: str = "unknown"


@dataclass(frozen=True)
class PatternCrunchResult:
    body: dict[str, Any]
    changed: bool
    saved_chars: int = 0
    reason: str = "no-local-crunch"


class LocalPatternModule:
    family = "unknown"
    version = "1"
    feature_schema = "agentflow.local_pattern_module.generic_features.v1"
    enabled_by_default = True
    supports_local_crunch = False

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        raise NotImplementedError

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict[str, Any]:
        raise NotImplementedError

    def apply_local_crunch(
        self,
        body: dict[str, Any],
        context: PatternModuleContext,
        detection: PatternDetection,
    ) -> PatternCrunchResult:
        return PatternCrunchResult(body=body, changed=False)

    def outcome_metadata(
        self,
        *,
        status: str,
        reason: str,
        detection: PatternDetection | None,
        features: dict[str, Any] | None,
        changed: bool,
        saved_chars: int,
    ) -> dict[str, Any]:
        return {
            "schema": "agentflow.local_pattern_module_outcome.v1",
            "family": self.family,
            "version": self.version,
            "status": status,
            "reason": reason,
            "detected": bool(detection.detected) if detection else False,
            "detection_confidence": detection.confidence if detection else "unknown",
            "features_emitted": isinstance(features, dict),
            "feature_schema": self.feature_schema if isinstance(features, dict) else None,
            "changed": bool(changed),
            "saved_chars": max(0, int(saved_chars)),
            "tokens_saved_est": max(0, int(saved_chars) // TOKEN_CHARS),
        }


class PatternModuleRegistry:
    def __init__(self, modules: list[LocalPatternModule] | None = None):
        self._modules: dict[str, LocalPatternModule] = {}
        for module in modules or []:
            self.register(module)

    def register(self, module: LocalPatternModule) -> None:
        family = str(module.family or "").strip()
        if not family:
            raise ValueError("pattern module family is required")
        if family in self._modules:
            raise ValueError(f"pattern module already registered: {family}")
        self._modules[family] = module

    def families(self) -> list[str]:
        return sorted(self._modules)

    def modules(self) -> list[LocalPatternModule]:
        return [self._modules[family] for family in self.families()]

    def get(self, family: str) -> LocalPatternModule | None:
        return self._modules.get(family)


class TerminalLogPatternModule(LocalPatternModule):
    family = "terminal_logs"
    version = "2026-06-09.1"
    feature_schema = TERMINAL_LOG_FEATURE_SCHEMA
    supports_local_crunch = False

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        features = terminal_log_features_from_text(context.text)
        detected = (
            features.get("terminal_output_char_fraction_bucket") != "none"
            or features.get("stack_trace_present") is True
            or features.get("test_output_present") is True
            or features.get("command_transcript_present") is True
        )
        return PatternDetection(
            detected=bool(detected),
            reason="terminal-log-signals-detected" if detected else "no-terminal-log-signals",
            confidence="high" if detected else "none",
        )

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict[str, Any]:
        result = terminal_log_features_from_text(context.text)
        result.update({
            "module_family": self.family,
            "module_version": self.version,
            "detected": detection.detected,
        })
        return result


class PromptRolePatternModule(LocalPatternModule):
    family = "prompt_role"
    version = "2026-06-09.1"
    feature_schema = PROMPT_DIFFICULTY_FEATURE_SCHEMA
    supports_local_crunch = False

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        detected = bool(context.text.strip())
        return PatternDetection(
            detected=detected,
            reason="prompt-role-signals-detected" if detected else "empty-prompt-text",
            confidence="medium" if detected else "none",
        )

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict[str, Any]:
        result = prompt_difficulty_features_from_text(context.text)
        result.update({
            "module_family": self.family,
            "module_version": self.version,
            "detected": detection.detected,
        })
        return result


def _iter_tool_result_blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return blocks
    for msg_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append({
                    "message_index": msg_index,
                    "block_index": block_index,
                    "block": block,
                })
    return blocks


def _tool_result_text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_tool_result_text_parts(item))
        return parts
    if isinstance(value, dict):
        parts = []
        for key in ("text", "content"):
            if key in value:
                parts.extend(_tool_result_text_parts(value[key]))
        return parts
    return []


def _tool_result_lines(texts: list[str]) -> list[str]:
    lines: list[str] = []
    for text in texts:
        lines.extend(line for line in text.splitlines() if line.strip())
    return lines


def _shape_counts(lines: list[str]) -> dict[str, int]:
    counts = {
        "file_listing": 0,
        "search_grep": 0,
        "sql_table": 0,
        "github_ticket_list": 0,
        "test_output": 0,
        "command_stream": 0,
        "unknown": 0,
    }
    for line in lines:
        matched = False
        if _GREP_RESULT_RE.search(line):
            counts["search_grep"] += 1
            matched = True
        if _SQL_TABLE_RE.search(line) or _SQL_BORDER_RE.search(line):
            counts["sql_table"] += 1
            matched = True
        if _ISSUE_LIST_RE.search(line):
            counts["github_ticket_list"] += 1
            matched = True
        if _TEST_RESULT_RE.search(line):
            counts["test_output"] += 1
            matched = True
        if _COMMAND_STREAM_RE.search(line):
            counts["command_stream"] += 1
            matched = True
        if _FILE_LISTING_RE.search(line):
            counts["file_listing"] += 1
            matched = True
        if not matched:
            counts["unknown"] += 1
    return counts


def _primary_shape(counts: dict[str, int]) -> str:
    ranked = [(shape, count) for shape, count in counts.items() if shape != "unknown" and count > 0]
    if not ranked:
        return "unknown_tool_result"
    ranked.sort(key=lambda item: (-item[1], item[0]))
    shape = ranked[0][0]
    if shape == "search_grep":
        return "search_grep_results"
    if shape == "sql_table":
        return "sql_table_rows"
    if shape == "github_ticket_list":
        return "github_ticket_list"
    if shape == "command_stream":
        return "command_stdout_stderr"
    return shape


def _normalized_line_fingerprint(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _framing_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 90:
        return False
    if _ERROR_RE.search(stripped):
        return False
    if _GREP_RESULT_RE.search(stripped) or _SQL_TABLE_RE.search(stripped) or _ISSUE_LIST_RE.search(stripped):
        return False
    if _COMMAND_STREAM_RE.search(stripped):
        return False
    return bool(_TOOL_FRAMING_RE.search(stripped))


def _compact_repeated_tool_framing_text(text: str) -> tuple[str, int, int]:
    lines = text.splitlines()
    seen: set[str] = set()
    output: list[str] = []
    removed = 0
    for line in lines:
        key = _normalized_line_fingerprint(line)
        if _framing_line(line) and key in seen:
            removed += 1
            continue
        if _framing_line(line):
            seen.add(key)
        output.append(line)
    if removed == 0:
        return text, 0, 0
    compacted = "\n".join(output)
    if text.endswith("\n"):
        compacted += "\n"
    return compacted, max(0, len(text) - len(compacted)), removed


def _compact_tool_result_value(value: Any) -> tuple[Any, int, int]:
    if isinstance(value, str):
        return _compact_repeated_tool_framing_text(value)
    if isinstance(value, list):
        changed_items: list[Any] = []
        saved = 0
        removed = 0
        for item in value:
            new_item, item_saved, item_removed = _compact_tool_result_value(item)
            changed_items.append(new_item)
            saved += item_saved
            removed += item_removed
        return changed_items, saved, removed
    if isinstance(value, dict):
        out = copy.deepcopy(value)
        saved = 0
        removed = 0
        for key in ("text", "content"):
            if key in out:
                out[key], item_saved, item_removed = _compact_tool_result_value(out[key])
                saved += item_saved
                removed += item_removed
        return out, saved, removed
    return value, 0, 0


class ToolResultPatternModule(LocalPatternModule):
    family = "tool_results"
    version = "2026-06-09.1"
    feature_schema = TOOL_RESULT_FEATURE_SCHEMA
    supports_local_crunch = True

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        blocks = _iter_tool_result_blocks(context.body)
        detected = bool(blocks)
        return PatternDetection(
            detected=detected,
            reason="tool-result-blocks-detected" if detected else "no-tool-result-blocks",
            confidence="high" if detected else "none",
        )

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict[str, Any]:
        blocks = _iter_tool_result_blocks(context.body)
        texts: list[str] = []
        for entry in blocks:
            block = entry["block"]
            if isinstance(block, dict):
                texts.extend(_tool_result_text_parts(block.get("content")))
        lines = _tool_result_lines(texts)
        normalized_lines = [_normalized_line_fingerprint(line) for line in lines if line.strip()]
        unique_lines = set(normalized_lines)
        duplicate_lines = max(0, len(normalized_lines) - len(unique_lines))
        counts = _shape_counts(lines)
        framing_lines = sum(1 for line in lines if _framing_line(line))
        mixed_prompt_text = any(
            isinstance(message, dict)
            and isinstance(message.get("content"), list)
            and any(isinstance(block, dict) and block.get("type") == "text" for block in message.get("content") or [])
            and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in message.get("content") or [])
            for message in context.body.get("messages") or []
        )
        error_present = any(_ERROR_RE.search(line) for line in lines)
        primary = _primary_shape(counts)
        exactness_required = error_present or primary in {
            "search_grep_results",
            "sql_table_rows",
            "github_ticket_list",
            "test_output",
            "command_stdout_stderr",
        }
        current_state_evidence = primary in {
            "file_listing",
            "search_grep_results",
            "sql_table_rows",
            "github_ticket_list",
            "test_output",
            "command_stdout_stderr",
        }
        return {
            "schema": self.feature_schema,
            "module_family": self.family,
            "module_version": self.version,
            "detected": detection.detected,
            "result_count_bucket": _count_bucket(len(blocks)),
            "tool_result_char_bucket": _text_bucket(sum(len(text) for text in texts)),
            "row_count_bucket": _count_bucket(len(lines)),
            "unique_item_count_bucket": _count_bucket(len(unique_lines)),
            "duplicate_result_fraction_bucket": _fraction_bucket(duplicate_lines, len(normalized_lines)),
            "repeated_framing_line_count_bucket": _count_bucket(framing_lines),
            "primary_result_shape": primary,
            "shape_count_buckets": {shape: _count_bucket(count) for shape, count in counts.items()},
            "error_presence": bool(error_present),
            "current_state_evidence_hint": bool(current_state_evidence),
            "exactness_required_hint": bool(exactness_required),
            "mixed_prompt_tool_result": bool(mixed_prompt_text),
            "safe_local_crunch_hint": bool(framing_lines >= 2),
            "privacy": {
                "metadata_only": True,
                "raw_tool_payload_included": False,
                "raw_tool_result_text_included": False,
                "file_paths_included": False,
                "issue_ids_included": False,
                "row_values_included": False,
                "diagnostic_lines_included": False,
            },
        }

    def apply_local_crunch(
        self,
        body: dict[str, Any],
        context: PatternModuleContext,
        detection: PatternDetection,
    ) -> PatternCrunchResult:
        new_body = copy.deepcopy(body)
        saved = 0
        removed_lines = 0
        for entry in _iter_tool_result_blocks(new_body):
            block = entry["block"]
            if not isinstance(block, dict) or "content" not in block:
                continue
            block["content"], item_saved, item_removed = _compact_tool_result_value(block["content"])
            saved += item_saved
            removed_lines += item_removed
        return PatternCrunchResult(
            body=new_body,
            changed=removed_lines > 0,
            saved_chars=saved,
            reason="safe-repeated-framing-compacted" if removed_lines else "no-safe-repeated-framing",
        )


DEFAULT_PATTERN_MODULE_REGISTRY = PatternModuleRegistry([
    PromptRolePatternModule(),
    TerminalLogPatternModule(),
    ToolResultPatternModule(),
])


def registered_pattern_modules(registry: PatternModuleRegistry | None = None) -> list[dict[str, Any]]:
    active = registry or DEFAULT_PATTERN_MODULE_REGISTRY
    return [
        {
            "family": module.family,
            "version": module.version,
            "feature_schema": module.feature_schema,
            "enabled_by_default": module.enabled_by_default,
            "supports_local_crunch": module.supports_local_crunch,
        }
        for module in active.modules()
    ]


def _module_setting(settings: dict[str, Any] | None, family: str) -> dict[str, Any]:
    if not isinstance(settings, dict):
        return {}
    raw = settings.get(family)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bool):
        return {"enabled": raw}
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return default


def _safe_feature_entry(module: LocalPatternModule, features: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": module.family,
        "version": module.version,
        "feature_schema": module.feature_schema,
        "features": features,
    }


def _privacy_guard_meta(violations: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema": "agentflow.local_pattern_module_privacy_guard.v1",
        "safe": not violations,
        "violation_count": len(violations),
        "blocked_keys": sorted({item.get("key", "unknown") for item in violations}),
        "first_violation_path": violations[0].get("path") if violations else None,
        "raw_values_logged": False,
    }


def evaluate_pattern_modules(
    body: dict[str, Any],
    *,
    module_settings: dict[str, Any] | None = None,
    registry: PatternModuleRegistry | None = None,
    apply_local_crunch: bool = True,
    policy_source: str = "local-default",
    rule_path: str | None = None,
    category: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active = registry or DEFAULT_PATTERN_MODULE_REGISTRY
    current_body = copy.deepcopy(body)
    server_features: list[dict[str, Any]] = []
    module_metas: list[dict[str, Any]] = []
    total_saved_chars = 0

    for module in active.modules():
        setting = _module_setting(module_settings, module.family)
        enabled = _as_bool(setting.get("enabled"), module.enabled_by_default)
        local_crunch_enabled = _as_bool(setting.get("local_crunch_enabled"), False)
        base_meta = {
            "family": module.family,
            "version": module.version,
            "enabled": enabled,
            "supports_local_crunch": module.supports_local_crunch,
            "local_crunch_enabled": local_crunch_enabled,
        }
        if not enabled:
            meta = module.outcome_metadata(
                status="skipped",
                reason="disabled",
                detection=None,
                features=None,
                changed=False,
                saved_chars=0,
            )
            meta.update(base_meta)
            module_metas.append(meta)
            continue

        context = PatternModuleContext(
            body=current_body,
            text=_extract_text(current_body),
            category=category,
            policy_source=policy_source,
            rule_path=rule_path,
        )
        detection = module.detect(context)
        if not detection.detected:
            meta = module.outcome_metadata(
                status="skipped",
                reason=detection.reason,
                detection=detection,
                features=None,
                changed=False,
                saved_chars=0,
            )
            meta.update(base_meta)
            module_metas.append(meta)
            continue

        features = module.features(context, detection)
        feature_entry = _safe_feature_entry(module, features)
        violations = managed_egress_violations(feature_entry)
        if violations:
            meta = module.outcome_metadata(
                status="bypass",
                reason="privacy-guard-rejected",
                detection=detection,
                features=None,
                changed=False,
                saved_chars=0,
            )
            meta.update(base_meta)
            meta["privacy_guard"] = _privacy_guard_meta(violations)
            module_metas.append(meta)
            continue

        changed = False
        saved_chars = 0
        reason = "feature-only-no-local-crunch"
        status = "skipped"
        if apply_local_crunch and local_crunch_enabled and module.supports_local_crunch:
            before_chars = len(stable_json(current_body))
            crunch_result = module.apply_local_crunch(current_body, context, detection)
            current_body = crunch_result.body
            after_chars = len(stable_json(current_body))
            changed = bool(crunch_result.changed or before_chars != after_chars)
            saved_chars = max(0, int(crunch_result.saved_chars or before_chars - after_chars))
            reason = crunch_result.reason
            status = "applied" if changed else "skipped"
            total_saved_chars += saved_chars

        server_features.append(feature_entry)
        meta = module.outcome_metadata(
            status=status,
            reason=reason,
            detection=detection,
            features=features,
            changed=changed,
            saved_chars=saved_chars,
        )
        meta.update(base_meta)
        meta["feature_schema"] = module.feature_schema
        meta["privacy_guard"] = _privacy_guard_meta([])
        meta["feature_summary"] = {
            "family": module.family,
            "version": module.version,
            "feature_schema": module.feature_schema,
            "text_bucket": _text_bucket(len(context.text)),
            "category": category or "unknown",
            "raw_content_included": False,
        }
        module_metas.append(meta)

    server_feature_bundle = {
        "schema": PATTERN_MODULE_FEATURES_SCHEMA,
        "module_feature_count": len(server_features),
        "features": server_features,
        "privacy": {
            "metadata_only": True,
            "raw_content_included": False,
            "raw_provider_body_included": False,
            "raw_tool_payload_included": False,
        },
    }
    server_violations = managed_egress_violations(server_feature_bundle)
    if server_violations:
        server_feature_bundle = {
            "schema": PATTERN_MODULE_FEATURES_SCHEMA,
            "module_feature_count": 0,
            "features": [],
            "privacy": {
                "metadata_only": True,
                "raw_content_included": False,
                "raw_provider_body_included": False,
                "raw_tool_payload_included": False,
            },
            "privacy_guard": _privacy_guard_meta(server_violations),
        }

    meta = {
        "schema": PATTERN_MODULES_SCHEMA,
        "registered_count": len(active.modules()),
        "enabled_count": sum(1 for item in module_metas if item.get("enabled")),
        "detected_count": sum(1 for item in module_metas if item.get("detected")),
        "features_emitted_count": len(server_features),
        "applied_count": sum(1 for item in module_metas if item.get("status") == "applied"),
        "bypass_count": sum(1 for item in module_metas if item.get("status") == "bypass"),
        "saved_chars": total_saved_chars,
        "tokens_saved_est": total_saved_chars // TOKEN_CHARS,
        "policy_source": policy_source,
        "rule_path": rule_path,
        "modules": module_metas,
        "server_features": server_feature_bundle,
        "raw_content_included": False,
    }
    return current_body, meta
