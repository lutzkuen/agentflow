from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy import __version__
from agentflow_proxy.store import utc_now

POLICY_BUNDLE_APPLY_SCHEMA = "agentflow.policy_bundle_apply.v1"
POLICY_BUNDLE_SCHEMA = "agentflow.policy_bundle.v1"
POLICY_BUNDLE_DIFF_SCHEMA = "agentflow.policy_bundle_diff.v1"
POLICY_BUNDLE_REVIEW_SCHEMA = "agentflow.policy_bundle_review.v1"
POLICY_BUNDLE_ROLLBACK_SCHEMA = "agentflow.policy_bundle_rollback.v1"
POLICY_BUNDLE_VALIDATION_SCHEMA = "agentflow.policy_bundle_validation.v1"
POLICY_STATE_SCHEMA = "agentflow.policy_state.v1"
POLICY_SOURCES = {
    "local-default",
    "local-manual",
    "managed-recommended",
    "managed-enforced",
}
REQUIRED_POLICY_SECTIONS = (
    "routing",
    "crunch",
    "cache",
    "routing_experiments",
)
_POLICY_SECTION_FILES = {
    "routing": "routing_rules.yaml",
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
    "routing_experiments": "routing_experiments.yaml",
}


async def build_policy_bundle() -> dict[str, Any]:
    from agentflow_proxy import stats

    policy_state = await stats.stats_policies()
    return {
        "schema": POLICY_BUNDLE_SCHEMA,
        "generated_at": utc_now(),
        "generator": {
            "name": "agentflow-proxy",
            "version": __version__,
            "mode": "local-offline",
        },
        "managed_optimizer": {
            "enabled": False,
            "note": "Export only. No managed optimizer communication is performed by this command.",
        },
        "policies": policy_state,
    }


def _is_iso_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _add_error(errors: list[dict[str, str]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


_BOOL_STRINGS = {"0", "1", "false", "true", "no", "yes", "off", "on"}


def _is_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _BOOL_STRINGS
    return isinstance(value, (int, float)) and value in (0, 1)


def _is_intish(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        try:
            int(value)
        except ValueError:
            return False
        return True
    return False


def _int_value(value: Any) -> int:
    return int(value)


def _is_floatish(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _float_value(value: Any) -> float:
    return float(value)


def _validate_boolish(errors: list[dict[str, str]], path: str, value: Any) -> None:
    if not _is_boolish(value):
        _add_error(errors, path, "expected boolean-like value")


def _validate_intish(
    errors: list[dict[str, str]],
    path: str,
    value: Any,
    *,
    min_value: int | None = 0,
) -> None:
    if not _is_intish(value):
        _add_error(errors, path, "expected integer-like value")
        return
    if min_value is not None and _int_value(value) < min_value:
        _add_error(errors, path, f"expected integer >= {min_value}")


def _validate_floatish(
    errors: list[dict[str, str]],
    path: str,
    value: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    if not _is_floatish(value):
        _add_error(errors, path, "expected numeric value")
        return
    numeric = _float_value(value)
    if min_value is not None and numeric < min_value:
        _add_error(errors, path, f"expected number >= {min_value}")
    if max_value is not None and numeric > max_value:
        _add_error(errors, path, f"expected number <= {max_value}")


def _validate_non_empty_string(errors: list[dict[str, str]], path: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        _add_error(errors, path, "expected non-empty string")


_ROUTING_CONDITION_KEYS = {
    "model_pattern",
    "text_chars_lt",
    "text_chars_gt",
    "text_chars_lte",
    "text_chars_gte",
    "has_tools",
    "max_tokens_lte",
    "env_flag",
    "category",
    "category_not_in",
}


def _validate_routing_policy(policy: dict[str, Any], errors: list[dict[str, str]]) -> None:
    if "enabled" in policy:
        _validate_boolish(errors, "$.policies.routing.enabled", policy["enabled"])
    rules = policy.get("rules", [])
    if not isinstance(rules, list):
        _add_error(errors, "$.policies.routing.rules", "expected list")
        return

    for index, rule in enumerate(rules):
        rule_path = f"$.policies.routing.rules[{index}]"
        if not isinstance(rule, dict):
            _add_error(errors, rule_path, "expected rule object")
            continue

        conditions = rule.get("conditions", {})
        if not isinstance(conditions, dict):
            _add_error(errors, f"{rule_path}.conditions", "expected object")
        else:
            for key in sorted(set(conditions) - _ROUTING_CONDITION_KEYS):
                _add_error(errors, f"{rule_path}.conditions.{key}", "unknown routing condition")
            for key in ("model_pattern", "env_flag", "category"):
                if key in conditions:
                    _validate_non_empty_string(errors, f"{rule_path}.conditions.{key}", conditions[key])
            for key in ("text_chars_lt", "text_chars_gt", "text_chars_lte", "text_chars_gte", "max_tokens_lte"):
                if key in conditions:
                    _validate_intish(errors, f"{rule_path}.conditions.{key}", conditions[key], min_value=0)
            if "has_tools" in conditions:
                _validate_boolish(errors, f"{rule_path}.conditions.has_tools", conditions["has_tools"])
            if "category_not_in" in conditions:
                value = conditions["category_not_in"]
                if isinstance(value, str):
                    _validate_non_empty_string(errors, f"{rule_path}.conditions.category_not_in", value)
                elif isinstance(value, list):
                    for item_index, item in enumerate(value):
                        _validate_non_empty_string(
                            errors,
                            f"{rule_path}.conditions.category_not_in[{item_index}]",
                            item,
                        )
                else:
                    _add_error(errors, f"{rule_path}.conditions.category_not_in", "expected string or list of strings")

        action = rule.get("action")
        if not isinstance(action, dict):
            _add_error(errors, f"{rule_path}.action", "expected object")
            continue
        _validate_non_empty_string(errors, f"{rule_path}.action.route_to", action.get("route_to"))
        if "reason" in action:
            _validate_non_empty_string(errors, f"{rule_path}.action.reason", action["reason"])


def _validate_object_field(policy: dict[str, Any], path: str, key: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    value = policy.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        _add_error(errors, f"{path}.{key}", "expected object")
        return {}
    return value


def _validate_crunch_policy(policy: dict[str, Any], errors: list[dict[str, str]]) -> None:
    base = "$.policies.crunch"
    if "enabled" in policy:
        _validate_boolish(errors, f"{base}.enabled", policy["enabled"])
    if "threshold_chars" in policy:
        _validate_intish(errors, f"{base}.threshold_chars", policy["threshold_chars"], min_value=0)

    prompt_cache = _validate_object_field(policy, base, "prompt_cache", errors)
    if "enabled" in prompt_cache:
        _validate_boolish(errors, f"{base}.prompt_cache.enabled", prompt_cache["enabled"])
    if "min_chars" in prompt_cache:
        _validate_intish(errors, f"{base}.prompt_cache.min_chars", prompt_cache["min_chars"], min_value=0)

    summary = _validate_object_field(policy, base, "old_context_summarization", errors)
    if "enabled" in summary:
        _validate_boolish(errors, f"{base}.old_context_summarization.enabled", summary["enabled"])
    if "model" in summary:
        _validate_non_empty_string(errors, f"{base}.old_context_summarization.model", summary["model"])
    if "placement" in summary:
        if summary["placement"] != "system":
            _add_error(errors, f"{base}.old_context_summarization.placement", "expected system")
    for key in (
        "min_request_chars",
        "min_summarized_chars",
        "max_turns",
        "keep_recent_turns",
        "max_summary_chars",
        "max_source_chars",
    ):
        if key in summary:
            _validate_intish(errors, f"{base}.old_context_summarization.{key}", summary[key], min_value=0)

    thinking_dedup = _validate_object_field(policy, base, "thinking_deduplication", errors)
    if "enabled" in thinking_dedup:
        _validate_boolish(errors, f"{base}.thinking_deduplication.enabled", thinking_dedup["enabled"])
    if "min_chars" in thinking_dedup:
        _validate_intish(errors, f"{base}.thinking_deduplication.min_chars", thinking_dedup["min_chars"], min_value=0)
    if "similarity_threshold" in thinking_dedup:
        _validate_floatish(
            errors,
            f"{base}.thinking_deduplication.similarity_threshold",
            thinking_dedup["similarity_threshold"],
            min_value=0.0,
            max_value=1.0,
        )
    if "skip_latest_assistant" in thinking_dedup:
        _validate_boolish(
            errors,
            f"{base}.thinking_deduplication.skip_latest_assistant",
            thinking_dedup["skip_latest_assistant"],
        )


def _validate_cache_policy(policy: dict[str, Any], errors: list[dict[str, str]]) -> None:
    base = "$.policies.cache"
    if "enabled" in policy:
        _validate_boolish(errors, f"{base}.enabled", policy["enabled"])

    exact = _validate_object_field(policy, base, "exact_cache", errors)
    if "enabled" in exact:
        _validate_boolish(errors, f"{base}.exact_cache.enabled", exact["enabled"])
    if "cache_tool_calls" in exact:
        _validate_boolish(errors, f"{base}.exact_cache.cache_tool_calls", exact["cache_tool_calls"])

    semantic = _validate_object_field(policy, base, "semantic_cache", errors)
    if "enabled" in semantic:
        _validate_boolish(errors, f"{base}.semantic_cache.enabled", semantic["enabled"])
    if "threshold" in semantic:
        _validate_floatish(
            errors,
            f"{base}.semantic_cache.threshold",
            semantic["threshold"],
            min_value=0.0,
            max_value=1.0,
        )

    file_watch = _validate_object_field(policy, base, "file_watch", errors)
    if "enabled" in file_watch:
        _validate_boolish(errors, f"{base}.file_watch.enabled", file_watch["enabled"])
    if "root" in file_watch:
        _validate_non_empty_string(errors, f"{base}.file_watch.root", file_watch["root"])
    if "max_paths" in file_watch:
        _validate_intish(errors, f"{base}.file_watch.max_paths", file_watch["max_paths"], min_value=0)


def _validate_routing_experiment_policy(policy: dict[str, Any], errors: list[dict[str, str]]) -> None:
    experiment = policy.get("policy", policy)
    if not isinstance(experiment, dict):
        _add_error(errors, "$.policies.routing_experiments.policy", "expected object")
        return

    base = "$.policies.routing_experiments"
    if "policy" in policy:
        base = f"{base}.policy"
    if "enabled" in experiment:
        _validate_boolish(errors, f"{base}.enabled", experiment["enabled"])
    for key in ("sample_rate", "similarity_threshold"):
        if key in experiment:
            _validate_floatish(errors, f"{base}.{key}", experiment[key], min_value=0.0, max_value=1.0)
    for key in ("min_text_chars", "max_text_chars", "min_samples_for_confidence"):
        if key in experiment:
            _validate_intish(errors, f"{base}.{key}", experiment[key], min_value=0)
    if "categories" in experiment:
        categories = experiment["categories"]
        if not isinstance(categories, list):
            _add_error(errors, f"{base}.categories", "expected list")
        else:
            for index, category in enumerate(categories):
                _validate_non_empty_string(errors, f"{base}.categories[{index}]", category)
    if "store_response_bodies" in experiment:
        _validate_boolish(errors, f"{base}.store_response_bodies", experiment["store_response_bodies"])


def _validate_policy_section_shape(section: str, value: dict[str, Any], errors: list[dict[str, str]]) -> None:
    if section == "routing":
        _validate_routing_policy(value, errors)
    elif section == "crunch":
        _validate_crunch_policy(value, errors)
    elif section == "cache":
        _validate_cache_policy(value, errors)
    elif section == "routing_experiments":
        _validate_routing_experiment_policy(value, errors)


def validate_policy_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "bundle must be a JSON object")
        return {
            "schema": POLICY_BUNDLE_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
        }

    bundle_schema = bundle.get("schema")
    if bundle_schema != POLICY_BUNDLE_SCHEMA:
        _add_error(errors, "$.schema", f"expected {POLICY_BUNDLE_SCHEMA}")

    if not _is_iso_datetime(bundle.get("generated_at")):
        _add_error(errors, "$.generated_at", "expected ISO-8601 timestamp string")

    generator = bundle.get("generator")
    if not isinstance(generator, dict):
        _add_error(errors, "$.generator", "expected object")
    else:
        if generator.get("name") != "agentflow-proxy":
            _add_error(errors, "$.generator.name", "expected agentflow-proxy")
        if not isinstance(generator.get("version"), str) or not generator.get("version"):
            _add_error(errors, "$.generator.version", "expected non-empty string")
        if generator.get("mode") != "local-offline":
            _add_error(errors, "$.generator.mode", "expected local-offline")

    managed_optimizer = bundle.get("managed_optimizer")
    if not isinstance(managed_optimizer, dict):
        _add_error(errors, "$.managed_optimizer", "expected object")
    elif managed_optimizer.get("enabled") is not False:
        _add_error(errors, "$.managed_optimizer.enabled", "expected false for local offline bundles")

    policies = bundle.get("policies")
    if not isinstance(policies, dict):
        _add_error(errors, "$.policies", "expected object")
    else:
        if policies.get("schema") != POLICY_STATE_SCHEMA:
            _add_error(errors, "$.policies.schema", f"expected {POLICY_STATE_SCHEMA}")
        for section in REQUIRED_POLICY_SECTIONS:
            value = policies.get(section)
            if not isinstance(value, dict):
                _add_error(errors, f"$.policies.{section}", "expected policy section object")
                continue
            source = value.get("policy_source")
            if source is not None and source not in POLICY_SOURCES:
                _add_error(errors, f"$.policies.{section}.policy_source", "unknown policy source")
            _validate_policy_section_shape(section, value, errors)

    return {
        "schema": POLICY_BUNDLE_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle_schema,
        "errors": errors,
        "warnings": warnings,
    }


_MISSING = object()


def _diff_values(path: str, before: Any, after: Any, changes: list[dict[str, Any]]) -> None:
    if before == after:
        return

    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}"
            _diff_values(child_path, before.get(key, _MISSING), after.get(key, _MISSING), changes)
        return

    if isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child_path = f"{path}[{index}]"
            before_value = before[index] if index < len(before) else _MISSING
            after_value = after[index] if index < len(after) else _MISSING
            _diff_values(child_path, before_value, after_value, changes)
        return

    if before is _MISSING:
        changes.append({"path": path, "change": "added", "old": None, "new": after})
    elif after is _MISSING:
        changes.append({"path": path, "change": "removed", "old": before, "new": None})
    else:
        changes.append({"path": path, "change": "changed", "old": before, "new": after})


def compare_policy_bundles(before: Any, after: Any) -> dict[str, Any]:
    before_validation = validate_policy_bundle(before)
    after_validation = validate_policy_bundle(after)
    ok = bool(before_validation["ok"] and after_validation["ok"])
    changes: list[dict[str, Any]] = []

    if ok:
        before_policies = before["policies"]
        after_policies = after["policies"]
        for section in REQUIRED_POLICY_SECTIONS:
            _diff_values(
                f"$.policies.{section}",
                before_policies.get(section, {}),
                after_policies.get(section, {}),
                changes,
            )

    changed_sections = sorted(
        {
            change["path"].removeprefix("$.policies.").split(".", 1)[0].split("[", 1)[0]
            for change in changes
        }
    )

    return {
        "schema": POLICY_BUNDLE_DIFF_SCHEMA,
        "ok": ok,
        "changed": bool(changes),
        "changed_sections": changed_sections,
        "change_count": len(changes),
        "changes": changes,
        "before_validation": before_validation,
        "after_validation": after_validation,
    }


def _add_warning(warnings: list[dict[str, str]], code: str, path: str, message: str) -> None:
    warnings.append({
        "code": code,
        "path": path,
        "severity": "warning",
        "message": message,
    })


def _section_policy(bundle: Any, section: str) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return {}
    policies = bundle.get("policies")
    if not isinstance(policies, dict):
        return {}
    value = policies.get(section)
    return value if isinstance(value, dict) else {}


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return False


def policy_bundle_safety_warnings(bundle: Any) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    if not isinstance(bundle, dict):
        return warnings

    managed_optimizer = bundle.get("managed_optimizer")
    if isinstance(managed_optimizer, dict) and managed_optimizer.get("enabled") is not False:
        _add_warning(
            warnings,
            "managed-optimizer-enabled",
            "$.managed_optimizer.enabled",
            "managed optimizer communication must remain opt-in and is not part of local offline review",
        )

    for section in REQUIRED_POLICY_SECTIONS:
        policy = _section_policy(bundle, section)
        if policy.get("policy_source") == "managed-enforced":
            _add_warning(
                warnings,
                "managed-enforced-policy-source",
                f"$.policies.{section}.policy_source",
                "managed-enforced policy source should not be accepted by the local module without an explicit future import/apply flow",
            )

    cache = _section_policy(bundle, "cache")
    exact_cache = cache.get("exact_cache") if isinstance(cache.get("exact_cache"), dict) else {}
    if _enabled(exact_cache.get("cache_tool_calls")):
        _add_warning(
            warnings,
            "tool-call-cache-enabled",
            "$.policies.cache.exact_cache.cache_tool_calls",
            "tool-call caching can return stale filesystem-dependent results unless invalidation is proven safe",
        )
    semantic_cache = cache.get("semantic_cache") if isinstance(cache.get("semantic_cache"), dict) else {}
    if _enabled(semantic_cache.get("enabled")):
        _add_warning(
            warnings,
            "semantic-cache-enabled",
            "$.policies.cache.semantic_cache.enabled",
            "semantic cache can produce false-positive response reuse and should stay opt-in with quality checks",
        )

    crunch = _section_policy(bundle, "crunch")
    old_context = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else {}
    if _enabled(old_context.get("enabled")):
        _add_warning(
            warnings,
            "old-context-summarization-enabled",
            "$.policies.crunch.old_context_summarization.enabled",
            "model-assisted summarization changes request context and should be reviewed against quality risk before enabling",
        )

    return warnings


def review_policy_bundle(current: Any, proposed: Any) -> dict[str, Any]:
    diff = compare_policy_bundles(current, proposed)
    warnings = policy_bundle_safety_warnings(proposed)
    return {
        "schema": POLICY_BUNDLE_REVIEW_SCHEMA,
        "ok": bool(diff["ok"]),
        "changed": bool(diff.get("changed", False)) if diff["ok"] else False,
        "changed_sections": diff.get("changed_sections", []) if diff["ok"] else [],
        "change_count": int(diff.get("change_count", 0)) if diff["ok"] else 0,
        "safety_warning_count": len(warnings),
        "safety_warnings": warnings,
        "current_validation": diff.get("before_validation"),
        "proposed_validation": diff.get("after_validation"),
        "diff": {
            "schema": diff.get("schema"),
            "ok": diff.get("ok"),
            "changed": diff.get("changed"),
            "changed_sections": diff.get("changed_sections", []),
            "change_count": diff.get("change_count", 0),
            "changes": diff.get("changes", []),
        },
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy_apply_yaml(section: str, policy: dict[str, Any]) -> dict[str, Any]:
    if section == "routing":
        return {"rules": policy.get("rules") if isinstance(policy.get("rules"), list) else []}
    if section == "crunch":
        payload: dict[str, Any] = {}
        if "enabled" in policy:
            payload["enabled"] = policy.get("enabled")
        if "threshold_chars" in policy:
            payload["threshold_chars"] = policy.get("threshold_chars")
        for key in ("prompt_cache", "old_context_summarization", "thinking_deduplication"):
            if isinstance(policy.get(key), dict):
                payload[key] = policy[key]
        return payload
    if section == "cache":
        return {key: policy[key] for key in ("exact_cache", "semantic_cache", "file_watch") if isinstance(policy.get(key), dict)}
    if section == "routing_experiments":
        experiment_policy = policy.get("policy")
        return experiment_policy if isinstance(experiment_policy, dict) else {}
    return {}


def _backup_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_policy_file(path: Path, text: str) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: str | None = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{_backup_suffix()}")
        backup.write_bytes(path.read_bytes())
        backup_path = str(backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return backup_path


def _latest_policy_backup(path: Path) -> Path | None:
    backups = sorted(path.parent.glob(f"{path.name}.bak-*"))
    return backups[-1] if backups else None


def apply_policy_bundle(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    allow_risky: bool = False,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    validation = validate_policy_bundle(bundle)
    warnings = policy_bundle_safety_warnings(bundle)
    requested_sections = list(sections or REQUIRED_POLICY_SECTIONS)
    invalid_sections = sorted(set(requested_sections) - set(REQUIRED_POLICY_SECTIONS))
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": POLICY_BUNDLE_APPLY_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "applied_sections": [],
        "skipped_sections": [],
        "files": [],
        "validation": validation,
        "safety_warning_count": len(warnings),
        "safety_warnings": warnings,
        "error": None,
    }

    if invalid_sections:
        result["error"] = {
            "type": "invalid_sections",
            "message": "unknown policy section requested",
            "sections": invalid_sections,
        }
        return result
    if not validation["ok"]:
        result["error"] = {"type": "validation_failed", "message": "policy bundle is invalid"}
        return result
    if warnings and not allow_risky:
        result["error"] = {
            "type": "risky_policy",
            "message": "policy bundle has safety warnings; rerun with --allow-risky to apply explicitly",
        }
        return result

    policies = bundle["policies"]
    for section in REQUIRED_POLICY_SECTIONS:
        if section not in requested_sections:
            result["skipped_sections"].append({"section": section, "reason": "not-requested"})
            continue

        policy = policies.get(section) if isinstance(policies.get(section), dict) else {}
        yaml_payload = _policy_apply_yaml(section, policy)
        text = yaml.safe_dump(yaml_payload, sort_keys=False)
        path = config_path / _POLICY_SECTION_FILES[section]
        old_text: str | None = None
        if path.exists():
            try:
                old_text = path.read_text(encoding="utf-8")
            except OSError:
                old_text = None
        changed = old_text != text
        backup_path = None
        if changed and not dry_run:
            backup_path = _write_policy_file(path, text)

        result["files"].append({
            "section": section,
            "path": str(path),
            "changed": bool(changed),
            "backup_path": backup_path,
            "sha256_before": _sha256_text(old_text) if old_text is not None else None,
            "sha256_after": _sha256_text(text),
            "bytes_after": len(text.encode("utf-8")),
        })
        result["applied_sections"].append(section)

    result["ok"] = True
    return result


def rollback_policy_files(
    *,
    config_dir: str | Path,
    dry_run: bool = False,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    requested_sections = list(sections or REQUIRED_POLICY_SECTIONS)
    invalid_sections = sorted(set(requested_sections) - set(REQUIRED_POLICY_SECTIONS))
    config_path = Path(config_dir).expanduser()
    result: dict[str, Any] = {
        "schema": POLICY_BUNDLE_ROLLBACK_SCHEMA,
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": str(config_path),
        "restored_sections": [],
        "skipped_sections": [],
        "files": [],
        "error": None,
    }

    if invalid_sections:
        result["error"] = {
            "type": "invalid_sections",
            "message": "unknown policy section requested",
            "sections": invalid_sections,
        }
        return result

    plans: list[dict[str, Any]] = []
    missing_sections: list[str] = []
    unreadable_backups: list[dict[str, str]] = []
    for section in REQUIRED_POLICY_SECTIONS:
        if section not in requested_sections:
            result["skipped_sections"].append({"section": section, "reason": "not-requested"})
            continue

        path = config_path / _POLICY_SECTION_FILES[section]
        backup = _latest_policy_backup(path)
        old_text: str | None = None
        if path.exists():
            try:
                old_text = path.read_text(encoding="utf-8")
            except OSError:
                old_text = None
        if backup is None:
            missing_sections.append(section)
            result["files"].append({
                "section": section,
                "path": str(path),
                "restored_from": None,
                "changed": False,
                "backup_path": None,
                "sha256_before": _sha256_text(old_text) if old_text is not None else None,
                "sha256_after": None,
                "bytes_after": None,
            })
            continue

        try:
            backup_text = backup.read_text(encoding="utf-8")
        except OSError as exc:
            unreadable_backups.append({"section": section, "path": str(backup), "message": str(exc)})
            continue

        plans.append({
            "section": section,
            "path": path,
            "backup": backup,
            "old_text": old_text,
            "backup_text": backup_text,
        })

    if missing_sections:
        result["error"] = {
            "type": "missing_backups",
            "message": "one or more requested policy sections have no backup file",
            "sections": missing_sections,
        }
        return result
    if unreadable_backups:
        result["error"] = {
            "type": "unreadable_backups",
            "message": "one or more requested policy backups could not be read",
            "backups": unreadable_backups,
        }
        return result

    for plan in plans:
        path = plan["path"]
        old_text = plan["old_text"]
        backup_text = plan["backup_text"]
        changed = old_text != backup_text
        backup_path = None
        if changed and not dry_run:
            backup_path = _write_policy_file(path, backup_text)

        result["files"].append({
            "section": plan["section"],
            "path": str(path),
            "restored_from": str(plan["backup"]),
            "changed": bool(changed),
            "backup_path": backup_path,
            "sha256_before": _sha256_text(old_text) if old_text is not None else None,
            "sha256_after": _sha256_text(backup_text),
            "bytes_after": len(backup_text.encode("utf-8")),
        })
        result["restored_sections"].append(plan["section"])

    result["ok"] = True
    return result
