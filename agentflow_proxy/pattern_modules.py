from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.prompt_features import PROMPT_DIFFICULTY_FEATURE_SCHEMA, prompt_difficulty_features_from_text
from agentflow_proxy.store import stable_json
from agentflow_proxy.terminal_features import TERMINAL_LOG_FEATURE_SCHEMA, terminal_log_features_from_text


PATTERN_MODULES_SCHEMA = "agentflow.local_pattern_modules.v1"
PATTERN_MODULE_FEATURES_SCHEMA = "agentflow.local_pattern_module_features.v1"
TOKEN_CHARS = 4


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


DEFAULT_PATTERN_MODULE_REGISTRY = PatternModuleRegistry([
    PromptRolePatternModule(),
    TerminalLogPatternModule(),
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
