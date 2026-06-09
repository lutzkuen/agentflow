from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentflow_proxy import __version__
from agentflow_proxy.codex_app_policy import CODEX_APP_SOURCE_SURFACE, canonical_source_surface
from agentflow_proxy.store import utc_now

POLICY_BUNDLE_APPLY_SCHEMA = "agentflow.policy_bundle_apply.v1"
POLICY_BUNDLE_SCHEMA = "agentflow.policy_bundle.v1"
POLICY_BUNDLE_DIFF_SCHEMA = "agentflow.policy_bundle_diff.v1"
POLICY_BUNDLE_REVIEW_SCHEMA = "agentflow.policy_bundle_review.v1"
POLICY_BUNDLE_ROLLBACK_SCHEMA = "agentflow.policy_bundle_rollback.v1"
POLICY_BUNDLE_VALIDATION_SCHEMA = "agentflow.policy_bundle_validation.v1"
POLICY_BUNDLE_PROVENANCE_SCHEMA = "agentflow.policy_bundle_provenance.v1"
POLICY_BUNDLE_PROVENANCE_VERIFICATION_SCHEMA = "agentflow.policy_bundle_provenance_verification.v1"
PATTERN_CANDIDATE_REVIEW_SCHEMA = "agentflow.pattern_candidate_review.v1"
POLICY_STATE_SCHEMA = "agentflow.policy_state.v1"
MANAGED_POLICY_VERIFICATION_SECRET_ENV = "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET"
MANAGED_POLICY_VERIFICATION_SECRETS_ENV = "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS"
MANAGED_POLICY_HMAC_SECRET_ENV = "AGENTFLOW_MANAGED_POLICY_HMAC_SECRET"
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
    "codex_app",
)
APPLY_POLICY_SECTIONS = (
    "routing",
    "crunch",
    "cache",
    "routing_experiments",
    "codex_app",
)
_POLICY_SECTION_FILES = {
    "routing": "routing_rules.yaml",
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
    "routing_experiments": "routing_experiments.yaml",
    "codex_app": "codex_app_rules.yaml",
}
POLICY_IMPACT_SCHEMA = "agentflow.policy_bundle_impact.v1"
_DEFAULT_IMPACT_LIMIT = 1000


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


def _add_validation_warning(warnings: list[dict[str, str]], path: str, message: str) -> None:
    warnings.append({"path": path, "message": message})


_BOOL_STRINGS = {"0", "1", "false", "true", "no", "yes", "off", "on"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _bundle_payload_for_hash(bundle: dict[str, Any]) -> dict[str, Any]:
    payload = _json_roundtrip(bundle)
    if isinstance(payload, dict):
        payload.pop("provenance", None)
    return payload if isinstance(payload, dict) else {}


def canonical_policy_bundle_hash(bundle: Any) -> str | None:
    if not isinstance(bundle, dict):
        return None
    payload = _bundle_payload_for_hash(bundle)
    return f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"


def _managed_source(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("managed-")


def policy_bundle_is_managed(bundle: Any) -> bool:
    if not isinstance(bundle, dict):
        return False
    recommendation = bundle.get("recommendation")
    if isinstance(recommendation, dict) and recommendation:
        return True
    managed_optimizer = bundle.get("managed_optimizer")
    if isinstance(managed_optimizer, dict) and _managed_source(managed_optimizer.get("policy_source")):
        return True
    policies = bundle.get("policies")
    if not isinstance(policies, dict):
        return False
    for section in REQUIRED_POLICY_SECTIONS:
        policy = policies.get(section)
        if isinstance(policy, dict) and _managed_source(policy.get("policy_source")):
            return True
    return False


def _verification_secrets() -> dict[str, str]:
    secrets: dict[str, str] = {}
    raw_mapping = os.getenv(MANAGED_POLICY_VERIFICATION_SECRETS_ENV)
    if raw_mapping:
        try:
            parsed = json.loads(raw_mapping)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if isinstance(key, str) and isinstance(value, str) and value:
                    secrets[key] = value
    single = os.getenv(MANAGED_POLICY_VERIFICATION_SECRET_ENV) or os.getenv(MANAGED_POLICY_HMAC_SECRET_ENV)
    if single:
        secrets.setdefault("", single)
    return secrets


def _secret_for_key_id(key_id: Any) -> tuple[str | None, bool]:
    secrets = _verification_secrets()
    if not secrets:
        return None, False
    if isinstance(key_id, str) and key_id in secrets:
        return secrets[key_id], True
    return secrets.get(""), True


def _signature_payload(provenance: dict[str, Any]) -> dict[str, Any]:
    payload = _json_roundtrip(provenance)
    if isinstance(payload, dict):
        payload.pop("signature", None)
    return payload if isinstance(payload, dict) else {}


def _hmac_signature(provenance: dict[str, Any], secret: str) -> str:
    payload = _canonical_json(_signature_payload(provenance)).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def _normalize_signature(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    if value.startswith("hmac-sha256:"):
        return value
    return f"hmac-sha256:{value}"


def attach_policy_bundle_provenance(
    bundle: dict[str, Any],
    *,
    secret: str,
    issuer: str,
    server_id: str,
    key_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    signed = _json_roundtrip(bundle)
    signed.pop("provenance", None)
    provenance = {
        "schema": POLICY_BUNDLE_PROVENANCE_SCHEMA,
        "algorithm": "hmac-sha256",
        "generated_at": generated_at or utc_now(),
        "issuer": issuer,
        "server_id": server_id,
        "key_id": key_id,
        "bundle_hash": canonical_policy_bundle_hash(signed),
    }
    provenance["signature"] = _hmac_signature(provenance, secret)
    signed["provenance"] = provenance
    return signed


def verify_policy_bundle_provenance(bundle: Any) -> dict[str, Any]:
    managed_bundle = policy_bundle_is_managed(bundle)
    provenance = bundle.get("provenance") if isinstance(bundle, dict) else None
    configured = bool(_verification_secrets())
    result: dict[str, Any] = {
        "schema": POLICY_BUNDLE_PROVENANCE_VERIFICATION_SCHEMA,
        "status": "missing",
        "ok": True,
        "managed_bundle": managed_bundle,
        "verification_configured": configured,
        "signature_required": bool(managed_bundle and configured),
        "algorithm": None,
        "issuer": None,
        "server_id": None,
        "key_id": None,
        "generated_at": None,
        "bundle_hash": None,
        "computed_bundle_hash": canonical_policy_bundle_hash(bundle),
        "signature_present": False,
        "errors": [],
        "warnings": [],
    }

    if not configured:
        result["status"] = "not-configured"
        if managed_bundle:
            result["warnings"].append({
                "path": "$.provenance",
                "message": f"managed policy bundle provenance was not verified because {MANAGED_POLICY_VERIFICATION_SECRET_ENV} is not configured",
            })
        return result

    if not isinstance(provenance, dict):
        if managed_bundle:
            result["status"] = "missing"
            result["ok"] = False
            result["errors"].append({
                "path": "$.provenance",
                "message": "managed policy bundle is missing provenance required by configured verification",
            })
        else:
            result["warnings"].append({
                "path": "$.provenance",
                "message": "local policy bundle is unsigned; provenance is not required for local-default/local-manual bundles",
            })
        return result

    for key in ("algorithm", "issuer", "server_id", "key_id", "generated_at", "bundle_hash"):
        result[key] = provenance.get(key)
    result["signature_present"] = bool(provenance.get("signature"))

    errors: list[dict[str, str]] = []
    if provenance.get("schema") != POLICY_BUNDLE_PROVENANCE_SCHEMA:
        _add_error(errors, "$.provenance.schema", f"expected {POLICY_BUNDLE_PROVENANCE_SCHEMA}")
    if provenance.get("algorithm") != "hmac-sha256":
        _add_error(errors, "$.provenance.algorithm", "expected hmac-sha256")
    for key in ("issuer", "server_id", "key_id"):
        if not isinstance(provenance.get(key), str) or not str(provenance.get(key)).strip():
            _add_error(errors, f"$.provenance.{key}", "expected non-empty string")
    if not _is_iso_datetime(provenance.get("generated_at")):
        _add_error(errors, "$.provenance.generated_at", "expected ISO-8601 timestamp string")

    expected_hash = canonical_policy_bundle_hash(bundle)
    if provenance.get("bundle_hash") != expected_hash:
        _add_error(errors, "$.provenance.bundle_hash", "bundle hash does not match canonical payload")

    signature = _normalize_signature(provenance.get("signature"))
    if not signature:
        _add_error(errors, "$.provenance.signature", "expected HMAC signature")
    secret, has_config = _secret_for_key_id(provenance.get("key_id"))
    if not has_config:
        result["status"] = "not-configured"
        result["warnings"].append({
            "path": "$.provenance.key_id",
            "message": "managed policy bundle provenance could not be verified because no verification secret is configured",
        })
        return result
    if secret is None:
        _add_error(errors, "$.provenance.key_id", "no configured verification secret for key_id")
    elif signature and not hmac.compare_digest(signature, _hmac_signature(provenance, secret)):
        _add_error(errors, "$.provenance.signature", "HMAC signature does not match provenance metadata")

    result["errors"] = errors
    if errors:
        result["status"] = "invalid"
        result["ok"] = False
    else:
        result["status"] = "verified"
        result["ok"] = True
    return result


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
    _validate_crunch_pattern_rules(policy.get("pattern_rules"), errors, base=base)
    _validate_pattern_recommendation(policy, errors, base=base, expected_section="crunch")


def _validate_crunch_pattern_rules(value: Any, errors: list[dict[str, str]], *, base: str) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        _add_error(errors, f"{base}.pattern_rules", "expected list")
        return
    for index, rule in enumerate(value):
        path = f"{base}.pattern_rules[{index}]"
        if not isinstance(rule, dict):
            _add_error(errors, path, "expected object")
            continue
        if "id" in rule:
            _validate_non_empty_string(errors, f"{path}.id", rule["id"])
        if "candidate_id" in rule:
            _validate_non_empty_string(errors, f"{path}.candidate_id", rule["candidate_id"])
        if "enabled" in rule:
            _validate_boolish(errors, f"{path}.enabled", rule["enabled"])
        if "policy_source" in rule and rule["policy_source"] not in POLICY_SOURCES:
            _add_error(errors, f"{path}.policy_source", "expected known policy source")
        conditions = rule.get("conditions")
        if not isinstance(conditions, dict):
            _add_error(errors, f"{path}.conditions", "expected object")
            continue
        hashes = conditions.get("pattern_hashes", conditions.get("pattern_hash"))
        if isinstance(hashes, str):
            _validate_non_empty_string(errors, f"{path}.conditions.pattern_hash", hashes)
        elif isinstance(hashes, list):
            if not hashes:
                _add_error(errors, f"{path}.conditions.pattern_hashes", "expected at least one hash")
            for hash_index, item in enumerate(hashes):
                _validate_non_empty_string(errors, f"{path}.conditions.pattern_hashes[{hash_index}]", item)
        else:
            _add_error(errors, f"{path}.conditions.pattern_hashes", "expected string or list")
        for key in ("model_pattern", "category", "workflow_phase"):
            if key in conditions:
                _validate_non_empty_string(errors, f"{path}.conditions.{key}", conditions[key])
        if "category_not_in" in conditions:
            categories = conditions["category_not_in"]
            if isinstance(categories, str):
                _validate_non_empty_string(errors, f"{path}.conditions.category_not_in", categories)
            elif isinstance(categories, list):
                for category_index, category in enumerate(categories):
                    _validate_non_empty_string(errors, f"{path}.conditions.category_not_in[{category_index}]", category)
            else:
                _add_error(errors, f"{path}.conditions.category_not_in", "expected string or list")
        for key in ("min_repeated_count", "keep_recent_matches", "min_text_chars", "max_text_chars", "max_applications"):
            if key in conditions:
                _validate_intish(errors, f"{path}.conditions.{key}", conditions[key], min_value=0)
        action = rule.get("action")
        if not isinstance(action, dict):
            _add_error(errors, f"{path}.action", "expected object")
            continue
        if "type" in action and action["type"] not in {"shorten", "omit"}:
            _add_error(errors, f"{path}.action.type", "expected shorten or omit")
        for key in ("head_chars", "tail_chars", "max_replacement_chars"):
            if key in action:
                _validate_intish(errors, f"{path}.action.{key}", action[key], min_value=0)
        if "marker" in action:
            _validate_non_empty_string(errors, f"{path}.action.marker", action["marker"])


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
    _validate_cache_pattern_rules(policy.get("pattern_rules"), errors, base=base)
    _validate_pattern_recommendation(policy, errors, base=base, expected_section="cache")


_CACHE_PATTERN_CONDITION_KEYS = {
    "pattern_hash",
    "pattern_hashes",
    "model_pattern",
    "category",
    "category_not_in",
    "workflow_phase",
    "source_surface",
    "app_family",
    "text_bucket",
    "token_bucket",
    "has_tools",
    "stream",
    "replayability_level",
    "replayability_levels",
}
_CACHE_PATTERN_ACTION_KEYS = {
    "type",
    "allow_tool_calls",
    "safe_invalidation",
    "safe_invalidation_evidence",
    "streaming",
    "estimated_saved_cost_usd",
    "reason",
}


def _validate_cache_pattern_rules(value: Any, errors: list[dict[str, str]], *, base: str) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        _add_error(errors, f"{base}.pattern_rules", "expected list")
        return
    for index, rule in enumerate(value):
        path = f"{base}.pattern_rules[{index}]"
        if not isinstance(rule, dict):
            _add_error(errors, path, "expected object")
            continue
        if "id" in rule:
            _validate_non_empty_string(errors, f"{path}.id", rule["id"])
        if "candidate_id" in rule:
            _validate_non_empty_string(errors, f"{path}.candidate_id", rule["candidate_id"])
        if "enabled" in rule:
            _validate_boolish(errors, f"{path}.enabled", rule["enabled"])
        if "policy_source" in rule and rule["policy_source"] not in POLICY_SOURCES:
            _add_error(errors, f"{path}.policy_source", "expected known policy source")
        conditions = rule.get("conditions")
        if not isinstance(conditions, dict):
            _add_error(errors, f"{path}.conditions", "expected object")
            continue
        for key in sorted(set(conditions) - _CACHE_PATTERN_CONDITION_KEYS):
            _add_error(errors, f"{path}.conditions.{key}", "unknown cache pattern condition")
        hashes = conditions.get("pattern_hashes", conditions.get("pattern_hash"))
        if isinstance(hashes, str):
            _validate_non_empty_string(errors, f"{path}.conditions.pattern_hash", hashes)
        elif isinstance(hashes, list):
            if not hashes:
                _add_error(errors, f"{path}.conditions.pattern_hashes", "expected at least one hash")
            for hash_index, item in enumerate(hashes):
                _validate_non_empty_string(errors, f"{path}.conditions.pattern_hashes[{hash_index}]", item)
        else:
            _add_error(errors, f"{path}.conditions.pattern_hashes", "expected string or list")
        for key in ("model_pattern", "category", "workflow_phase", "source_surface", "app_family", "text_bucket", "token_bucket"):
            if key in conditions:
                _validate_non_empty_string(errors, f"{path}.conditions.{key}", conditions[key])
        for key in ("replayability_level", "replayability_levels", "category_not_in"):
            if key in conditions:
                values = conditions[key]
                if isinstance(values, str):
                    _validate_non_empty_string(errors, f"{path}.conditions.{key}", values)
                elif isinstance(values, list):
                    for value_index, item in enumerate(values):
                        _validate_non_empty_string(errors, f"{path}.conditions.{key}[{value_index}]", item)
                else:
                    _add_error(errors, f"{path}.conditions.{key}", "expected string or list")
        for key in ("has_tools", "stream"):
            if key in conditions:
                _validate_boolish(errors, f"{path}.conditions.{key}", conditions[key])
        action = rule.get("action")
        if not isinstance(action, dict):
            _add_error(errors, f"{path}.action", "expected object")
            continue
        for key in sorted(set(action) - _CACHE_PATTERN_ACTION_KEYS):
            _add_error(errors, f"{path}.action.{key}", "unknown cache pattern action")
        if "type" in action and action["type"] not in {"exact_cache", "exact_cache_pattern"}:
            _add_error(errors, f"{path}.action.type", "expected exact_cache or exact_cache_pattern")
        for key in ("allow_tool_calls", "safe_invalidation", "safe_invalidation_evidence", "streaming"):
            if key in action:
                _validate_boolish(errors, f"{path}.action.{key}", action[key])
        if "estimated_saved_cost_usd" in action:
            _validate_floatish(errors, f"{path}.action.estimated_saved_cost_usd", action["estimated_saved_cost_usd"], min_value=0.0)
        if _enabled(conditions.get("has_tools")) and _enabled(action.get("allow_tool_calls")):
            has_safe_invalidation = _enabled(action.get("safe_invalidation")) or _enabled(action.get("safe_invalidation_evidence"))
            if not has_safe_invalidation:
                _add_error(
                    errors,
                    f"{path}.action.safe_invalidation",
                    "tool-call cache pattern rules require explicit safe invalidation evidence",
                )


def _validate_pattern_candidate(
    candidate: Any,
    errors: list[dict[str, str]],
    path: str,
    *,
    expected_section: str,
) -> None:
    if not isinstance(candidate, dict):
        _add_error(errors, path, "expected pattern candidate object")
        return
    candidate_id = candidate.get("candidate_id") or candidate.get("policy_id") or candidate.get("recommendation_id")
    if candidate_id is not None:
        _validate_non_empty_string(errors, f"{path}.candidate_id", candidate_id)
    if "confidence" in candidate:
        _validate_floatish(errors, f"{path}.confidence", candidate["confidence"], min_value=0.0, max_value=1.0)
    if "sample_count" in candidate:
        _validate_intish(errors, f"{path}.sample_count", candidate["sample_count"], min_value=0)
    if "error_rate" in candidate:
        _validate_floatish(errors, f"{path}.error_rate", candidate["error_rate"], min_value=0.0, max_value=1.0)
    if "local_action_requirements" in candidate:
        requirements = candidate["local_action_requirements"]
        if not isinstance(requirements, dict):
            _add_error(errors, f"{path}.local_action_requirements", "expected object")
        elif requirements.get("expected_policy_section") not in (None, expected_section):
            _add_error(
                errors,
                f"{path}.local_action_requirements.expected_policy_section",
                f"expected {expected_section}",
            )
    for key in ("confidence_inputs", "evidence", "review_evidence", "delta", "change_summary"):
        if key in candidate and not isinstance(candidate[key], dict):
            _add_error(errors, f"{path}.{key}", "expected object")
    for key in ("omission_reasons", "warning_reasons"):
        if key in candidate:
            value = candidate[key]
            if not isinstance(value, list):
                _add_error(errors, f"{path}.{key}", "expected list")
            else:
                for index, item in enumerate(value):
                    _validate_non_empty_string(errors, f"{path}.{key}[{index}]", item)


def _validate_pattern_recommendation(
    policy: dict[str, Any],
    errors: list[dict[str, str]],
    *,
    base: str,
    expected_section: str,
) -> None:
    recommendation = policy.get("recommendation")
    if recommendation is None:
        return
    if not isinstance(recommendation, dict):
        _add_error(errors, f"{base}.recommendation", "expected object")
        return
    if "candidate_count" in recommendation:
        _validate_intish(errors, f"{base}.recommendation.candidate_count", recommendation["candidate_count"], min_value=0)
    if "review_only_candidate_count" in recommendation:
        _validate_intish(
            errors,
            f"{base}.recommendation.review_only_candidate_count",
            recommendation["review_only_candidate_count"],
            min_value=0,
        )
    if "omitted_candidate_count" in recommendation:
        _validate_intish(
            errors,
            f"{base}.recommendation.omitted_candidate_count",
            recommendation["omitted_candidate_count"],
            min_value=0,
        )
    for list_key in ("candidates", "review_only_candidates", "omitted_candidates"):
        value = recommendation.get(list_key)
        if value is None:
            continue
        if not isinstance(value, list):
            _add_error(errors, f"{base}.recommendation.{list_key}", "expected list")
            continue
        for index, candidate in enumerate(value):
            _validate_pattern_candidate(
                candidate,
                errors,
                f"{base}.recommendation.{list_key}[{index}]",
                expected_section=expected_section,
            )


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


_CODEX_APP_CONDITION_KEYS = {
    "app_family",
    "workflow_phase",
    "model_field_state",
    "input_size_bucket",
    "cache_eligible",
    "cache_status",
    "replayability_level",
    "has_action_like_params",
}
_CODEX_APP_ACTION_KEYS = {
    "recommended_model",
    "model_hint",
    "crunch_profile",
    "cache_eligible",
    "cache_eligibility_reason",
    "pass_through_reason",
    "reason",
}


def _validate_codex_app_policy(policy: dict[str, Any], errors: list[dict[str, str]]) -> None:
    base = "$.policies.codex_app"
    if policy.get("policy_source") == "managed-enforced":
        _add_error(errors, f"{base}.policy_source", "managed-enforced is not accepted for review-only Codex app policies")
    if "enabled" in policy:
        _validate_boolish(errors, f"{base}.enabled", policy["enabled"])
    if "review_only" in policy:
        _validate_boolish(errors, f"{base}.review_only", policy["review_only"])
    if "surface" in policy:
        _validate_non_empty_string(errors, f"{base}.surface", policy["surface"])
    summary_hint = _validate_object_field(policy, base, "summary_model_hint", errors)
    if "enabled" in summary_hint:
        _validate_boolish(errors, f"{base}.summary_model_hint.enabled", summary_hint["enabled"])
    if "target_model" in summary_hint:
        _validate_non_empty_string(errors, f"{base}.summary_model_hint.target_model", summary_hint["target_model"])
    exact_cache = _validate_object_field(policy, base, "exact_cache", errors)
    if "enabled" in exact_cache:
        _validate_boolish(errors, f"{base}.exact_cache.enabled", exact_cache["enabled"])
    if "namespace" in exact_cache:
        _validate_non_empty_string(errors, f"{base}.exact_cache.namespace", exact_cache["namespace"])
    crunch = _validate_object_field(policy, base, "crunch", errors)
    profiles = crunch.get("profiles")
    if profiles is not None:
        if not isinstance(profiles, list):
            _add_error(errors, f"{base}.crunch.profiles", "expected list")
        else:
            for index, profile in enumerate(profiles):
                _validate_non_empty_string(errors, f"{base}.crunch.profiles[{index}]", profile)

    rules = policy.get("rules", [])
    if not isinstance(rules, list):
        _add_error(errors, f"{base}.rules", "expected list")
        return

    for index, rule in enumerate(rules):
        rule_path = f"{base}.rules[{index}]"
        if not isinstance(rule, dict):
            _add_error(errors, rule_path, "expected rule object")
            continue

        conditions = rule.get("conditions", {})
        if not isinstance(conditions, dict):
            _add_error(errors, f"{rule_path}.conditions", "expected object")
        else:
            for key in sorted(set(conditions) - _CODEX_APP_CONDITION_KEYS):
                _add_error(errors, f"{rule_path}.conditions.{key}", "unknown Codex app condition")
            for key in ("app_family", "workflow_phase", "model_field_state", "input_size_bucket", "cache_status", "replayability_level"):
                if key in conditions:
                    _validate_non_empty_string(errors, f"{rule_path}.conditions.{key}", conditions[key])
            for key in ("cache_eligible", "has_action_like_params"):
                if key in conditions:
                    _validate_boolish(errors, f"{rule_path}.conditions.{key}", conditions[key])

        action = rule.get("action")
        if not isinstance(action, dict):
            _add_error(errors, f"{rule_path}.action", "expected object")
            continue
        if not action:
            _add_error(errors, f"{rule_path}.action", "expected at least one reviewable action")
        for key in sorted(set(action) - _CODEX_APP_ACTION_KEYS):
            _add_error(errors, f"{rule_path}.action.{key}", "unknown Codex app action")
        for key in ("recommended_model", "model_hint", "crunch_profile", "cache_eligibility_reason", "pass_through_reason", "reason"):
            if key in action:
                _validate_non_empty_string(errors, f"{rule_path}.action.{key}", action[key])
        if "cache_eligible" in action:
            _validate_boolish(errors, f"{rule_path}.action.cache_eligible", action["cache_eligible"])


def _validate_policy_section_shape(section: str, value: dict[str, Any], errors: list[dict[str, str]]) -> None:
    if section == "routing":
        _validate_routing_policy(value, errors)
    elif section == "crunch":
        _validate_crunch_policy(value, errors)
    elif section == "cache":
        _validate_cache_policy(value, errors)
    elif section == "routing_experiments":
        _validate_routing_experiment_policy(value, errors)
    elif section == "codex_app":
        _validate_codex_app_policy(value, errors)


def validate_policy_bundle(bundle: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    provenance = verify_policy_bundle_provenance(bundle)

    if not isinstance(bundle, dict):
        _add_error(errors, "$", "bundle must be a JSON object")
        return {
            "schema": POLICY_BUNDLE_VALIDATION_SCHEMA,
            "ok": False,
            "bundle_schema": None,
            "errors": errors,
            "warnings": warnings,
            "provenance": provenance,
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

    for error in provenance.get("errors", []):
        if isinstance(error, dict):
            _add_error(errors, str(error.get("path") or "$.provenance"), str(error.get("message") or "provenance verification failed"))
    for warning in provenance.get("warnings", []):
        if isinstance(warning, dict):
            _add_validation_warning(
                warnings,
                str(warning.get("path") or "$.provenance"),
                str(warning.get("message") or "policy bundle provenance was not verified"),
            )

    return {
        "schema": POLICY_BUNDLE_VALIDATION_SCHEMA,
        "ok": not errors,
        "bundle_schema": bundle_schema,
        "errors": errors,
        "warnings": warnings,
        "provenance": provenance,
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


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _round_usd(value: float) -> float:
    return round(value, 6)


def _resolve_route_to(route_to: Any) -> str:
    value = str(route_to or "").strip()
    if value == "haiku":
        return os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
    if value == "sonnet":
        return os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4-6")
    if value == "opus":
        return os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4-5")
    return value


def _routing_condition_match(features: dict[str, Any], conditions: dict[str, Any], *, ignore_tools: bool = False) -> bool:
    requested = str(features.get("requested_model") or "").lower()
    category = features.get("category")
    text_chars = _as_int(features.get("text_chars"))
    max_tokens = features.get("max_tokens")

    if "model_pattern" in conditions and str(conditions["model_pattern"]).lower() not in requested:
        return False
    if "text_chars_lt" in conditions and not (text_chars < _as_int(conditions["text_chars_lt"], -1)):
        return False
    if "text_chars_gt" in conditions and not (text_chars > _as_int(conditions["text_chars_gt"], 0)):
        return False
    if "text_chars_lte" in conditions and not (text_chars <= _as_int(conditions["text_chars_lte"], -1)):
        return False
    if "text_chars_gte" in conditions and not (text_chars >= _as_int(conditions["text_chars_gte"], 0)):
        return False
    if not ignore_tools and "has_tools" in conditions and bool(conditions["has_tools"]) != bool(features.get("has_tools")):
        return False
    if "env_flag" in conditions:
        raw = os.getenv(str(conditions["env_flag"]), "0")
        if raw.strip().lower() not in {"1", "true", "yes", "on"}:
            return False
    if "max_tokens_lte" in conditions:
        if max_tokens is None or not (_as_int(max_tokens) <= _as_int(conditions["max_tokens_lte"], -1)):
            return False
    if "category" in conditions and conditions["category"] != category:
        return False
    if "category_not_in" in conditions:
        excluded = conditions["category_not_in"]
        if isinstance(excluded, str):
            excluded = [excluded]
        if category in set(excluded):
            return False
    return True


def _call_features(row: dict[str, Any]) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    category = row.get("category") or routing.get("category")
    pattern_features = routing.get("managed_pattern_features") if isinstance(routing.get("managed_pattern_features"), dict) else {}
    text_chars = _as_int(routing.get("text_chars"))
    if text_chars <= 0:
        text_chars = max(_as_int(row.get("actual_input_tokens")), _as_int(row.get("input_tokens_est"))) * 4
    has_tools = routing.get("has_tools")
    if has_tools is None:
        has_tools = str(category or "").startswith("tool-")
    thinking = bool(_as_int(row.get("thinking_output_tokens")) > 0)
    if not thinking:
        reason = str(routing.get("reason") or "").lower()
        thinking = "thinking" in reason
    return {
        "id": row.get("id"),
        "provider": row.get("provider") or "anthropic",
        "source_surface": _source_surface_for_call(row.get("provider") or "anthropic", row.get("path")),
        "app_family": _app_family_for_call(row.get("provider") or "anthropic", row.get("requested_model"), row.get("path")),
        "granularity": "provider_request",
        "requested_model": row.get("requested_model"),
        "routed_model": row.get("routed_model") or row.get("requested_model"),
        "stream": bool(row.get("stream")),
        "status_code": _as_int(row.get("status_code")),
        "retry_count": _as_int(row.get("retry_count")),
        "latency_ms": _as_int(row.get("latency_ms")),
        "input_tokens": max(_as_int(row.get("actual_input_tokens")), _as_int(row.get("input_tokens_est")), max(text_chars, 0) // 4),
        "output_tokens": max(_as_int(row.get("actual_output_tokens")), _as_int(row.get("output_tokens_est"))),
        "cache_creation_input_tokens": _as_int(row.get("cache_creation_input_tokens")),
        "cache_read_input_tokens": _as_int(row.get("cache_read_input_tokens")),
        "cost_est_usd": _as_float(row.get("cost_est_usd")),
        "cost_baseline_usd": _as_float(row.get("cost_baseline_usd")),
        "category": category,
        "workflow_phase": pattern_features.get("workflow_phase") or routing.get("workflow_phase") or category,
        "text_bucket": pattern_features.get("text_bucket"),
        "token_bucket": pattern_features.get("token_bucket"),
        "replayability_level": pattern_features.get("replayability_level") or "metadata_only",
        "text_chars": text_chars,
        "has_tools": bool(has_tools),
        "thinking": thinking,
        "pattern_hashes": _collect_pattern_hashes(pattern_features) or _collect_pattern_hashes({
            "routing": routing,
            "crunch": crunch,
            "cache": cache,
        }),
        "pattern_types": pattern_features.get("pattern_types") if isinstance(pattern_features.get("pattern_types"), list) else [],
        "routing": routing,
        "crunch": crunch,
        "cache": cache,
    }


def _sqlite_db_path(db_path: str) -> Path | None:
    if db_path.startswith("sqlite:///"):
        return Path(db_path.removeprefix("sqlite:///")).expanduser()
    if "://" in db_path:
        return None
    return Path(db_path).expanduser()


def _load_recent_call_features(db_path: str, *, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = _sqlite_db_path(db_path)
    if path is None:
        return [], {
            "status": "unavailable",
            "reason": "unsupported-db-url",
            "message": "policy impact simulation currently reads local SQLite metadata only",
        }
    if not path.exists():
        return [], {
            "status": "unavailable",
            "reason": "db-not-found",
            "message": f"AgentFlow SQLite database not found: {path}",
        }
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return [], {
            "status": "unavailable",
            "reason": "db-open-failed",
            "message": str(exc),
        }
    try:
        rows = conn.execute(
            """
            select id, created_at, provider, requested_model, routed_model, stream, status_code,
                   path, latency_ms, input_tokens_est, output_tokens_est, actual_input_tokens,
                   actual_output_tokens, cost_est_usd, cost_baseline_usd, crunch_json,
                   routing_json, cache_json, category, cache_creation_input_tokens,
                   cache_read_input_tokens, retry_count, thinking_output_tokens
            from calls
            order by datetime(created_at) desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error as exc:
        return [], {
            "status": "unavailable",
            "reason": "db-query-failed",
            "message": str(exc),
        }
    finally:
        conn.close()
    return [_call_features(dict(row)) for row in rows], None


def _codex_event_features(row: dict[str, Any]) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    pattern_features = routing.get("managed_pattern_features") if isinstance(routing.get("managed_pattern_features"), dict) else {}
    text_chars = max(_as_int(row.get("input_text_chars")), _as_int(row.get("params_chars")), _as_int(row.get("message_chars")))
    input_tokens = max(1, text_chars // 4) if text_chars > 0 else 0
    output_tokens = max(0, _as_int(row.get("result_chars")) // 4)
    return {
        "id": row.get("id"),
        "provider": "codex-app",
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "app_family": "codex",
        "granularity": "agent_turn",
        "requested_model": routing.get("requested_model") or routing.get("requested_model_value"),
        "routed_model": routing.get("routed_model") or routing.get("target_model") or routing.get("requested_model"),
        "stream": False,
        "status_code": 500 if row.get("error_code") else 200,
        "retry_count": 0,
        "latency_ms": _as_int(row.get("latency_ms")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "category": routing.get("category") or "codex_turn",
        "workflow_phase": pattern_features.get("workflow_phase") or routing.get("workflow_phase") or "codex_turn",
        "text_bucket": pattern_features.get("text_bucket"),
        "token_bucket": pattern_features.get("token_bucket"),
        "replayability_level": pattern_features.get("replayability_level") or cache.get("replayability_level") or "metadata_only",
        "text_chars": text_chars,
        "has_tools": bool(routing.get("has_tools") or routing.get("has_action_like_params")),
        "thinking": False,
        "pattern_hashes": _collect_pattern_hashes(pattern_features) or _collect_pattern_hashes({
            "routing": routing,
            "crunch": crunch,
            "cache": cache,
        }),
        "pattern_types": pattern_features.get("pattern_types") if isinstance(pattern_features.get("pattern_types"), list) else [],
        "routing": routing,
        "crunch": crunch,
        "cache": cache,
    }


def _load_recent_codex_event_features(db_path: str, *, limit: int) -> list[dict[str, Any]]:
    path = _sqlite_db_path(db_path)
    if path is None or not path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select id, created_at, direction, method, request_id, thread_id, message_chars,
                   params_chars, input_items, input_text_chars, result_chars, error_code,
                   error_message, latency_ms, session_id, routing_json, crunch_json,
                   cache_json, event_window_json, metadata_json
            from codex_app_events
            where direction = 'client_to_server'
            order by datetime(created_at) desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass
    return [_codex_event_features(dict(row)) for row in rows]


def _rollup(features: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(features)
    errors = sum(1 for item in features if item["status_code"] >= 400)
    retried = sum(1 for item in features if item["retry_count"] > 0)
    latency_values = [item["latency_ms"] for item in features if item["latency_ms"] > 0]
    return {
        "would_match_count": total,
        "historical_error_count": errors,
        "historical_error_rate": _rate(errors, total),
        "historical_retry_count": retried,
        "historical_retry_rate": _rate(retried, total),
        "avg_latency_ms": int(sum(latency_values) / len(latency_values)) if latency_values else 0,
        "current_cost_usd": _round_usd(sum(item["cost_est_usd"] for item in features)),
    }


def _impact_warning(code: str, path: str, message: str, *, severity: str = "warning") -> dict[str, str]:
    return {"code": code, "path": path, "severity": severity, "message": message}


def _routing_impact(policy: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    from agentflow_proxy.pricing import estimate_cost

    rules = policy.get("rules") if isinstance(policy.get("rules"), list) else []
    summaries: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        path = f"$.policies.routing.rules[{index}]"
        pre_safety = [item for item in calls if _routing_condition_match(item, conditions)]
        matched = [item for item in pre_safety if not item["thinking"]]
        excluded_thinking = len(pre_safety) - len(matched)
        if not pre_safety and "has_tools" in conditions:
            tool_agnostic = [item for item in calls if _routing_condition_match(item, conditions, ignore_tools=True)]
        else:
            tool_agnostic = pre_safety
        excluded_tool = sum(1 for item in tool_agnostic if item["has_tools"] and conditions.get("has_tools") is False)
        rollup = _rollup(matched)
        target_model = _resolve_route_to(action.get("route_to"))
        estimated_target_cost = 0.0
        cost_known_count = 0
        for item in matched:
            cost = estimate_cost(
                target_model or str(item["routed_model"] or item["requested_model"]),
                item["input_tokens"],
                item["output_tokens"],
                item["cache_creation_input_tokens"],
                item["cache_read_input_tokens"],
                provider=str(item["provider"] or "anthropic"),
            )
            if cost is not None:
                estimated_target_cost += cost
                cost_known_count += 1
        estimated_savings = max(0.0, rollup["current_cost_usd"] - estimated_target_cost)
        summary = {
            "path": path,
            "action": {
                "route_to": action.get("route_to"),
                "target_model": target_model,
                "reason": action.get("reason"),
            },
            "conditions": conditions,
            **rollup,
            "excluded_thinking_count": excluded_thinking,
            "excluded_tool_count": excluded_tool,
            "estimated_target_cost_usd": _round_usd(estimated_target_cost),
            "estimated_savings_usd": _round_usd(estimated_savings),
            "cost_estimate_count": cost_known_count,
        }
        if summary["historical_error_rate"] > 0.05:
            warning = _impact_warning(
                "high-error-rate-routing-match",
                path,
                f"matched historical calls had {summary['historical_error_rate']:.1%} error rate",
            )
            warnings.append(warning)
            summary.setdefault("warnings", []).append(warning)
        if summary["historical_retry_rate"] > 0.10:
            warning = _impact_warning(
                "high-retry-rate-routing-match",
                path,
                f"matched historical calls had {summary['historical_retry_rate']:.1%} retry rate",
            )
            warnings.append(warning)
            summary.setdefault("warnings", []).append(warning)
        if excluded_thinking:
            summary.setdefault("notes", []).append("thinking-history calls are excluded by local routing safety guard")
        summaries.append(summary)

    return {
        "status": "simulated",
        "policy_source": policy.get("policy_source"),
        "rule_count": len(summaries),
        "rules": summaries,
        "warnings": warnings,
    }


def _pattern_candidate_entries(policy: dict[str, Any], *, section: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if section == "crunch":
        for index, rule in enumerate(policy.get("pattern_rules") if isinstance(policy.get("pattern_rules"), list) else []):
            if not isinstance(rule, dict):
                continue
            managed = rule.get("managed_recommendation") if isinstance(rule.get("managed_recommendation"), dict) else {}
            hashes = _collect_pattern_hashes(rule.get("conditions")) or _collect_pattern_hashes(rule) or _collect_pattern_hashes(managed)
            entries.append({
                "path": f"$.policies.crunch.pattern_rules[{index}]",
                "source": "pattern_rules",
                "section": section,
                "rule_id": rule.get("id") or rule.get("rule_id"),
                "candidate_id": rule.get("candidate_id") or managed.get("candidate_id"),
                "policy_source": rule.get("policy_source") or managed.get("policy_source") or policy.get("policy_source"),
                "enabled": _enabled(rule.get("enabled", True)),
                "conditions": rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {},
                "action": rule.get("action") if isinstance(rule.get("action"), dict) else {},
                "rollout": rule.get("rollout") if isinstance(rule.get("rollout"), dict) else managed.get("rollout"),
                "pattern_hashes": hashes,
                "estimated_savings_usd": managed.get("estimated_savings_usd"),
            })

    recommendation = policy.get("recommendation") if isinstance(policy.get("recommendation"), dict) else {}
    for list_name in ("candidates", "review_only_candidates", "omitted_candidates"):
        candidates = recommendation.get(list_name) if isinstance(recommendation.get(list_name), list) else []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            hashes = _collect_pattern_hashes(candidate)
            action = {}
            if section == "cache" and isinstance(candidate.get("cache_action"), dict):
                action = candidate["cache_action"]
            elif isinstance(candidate.get("action"), dict):
                action = candidate["action"]
            dimensions = candidate.get("dimensions") if isinstance(candidate.get("dimensions"), dict) else {}
            buckets = candidate.get("buckets") if isinstance(candidate.get("buckets"), dict) else {}
            conditions = {
                key: value
                for key, value in {
                    "source_surface": dimensions.get("source_surface") or candidate.get("source_surface"),
                    "app_family": dimensions.get("app_family") or candidate.get("app_family"),
                    "category": dimensions.get("category") or candidate.get("category"),
                    "workflow_phase": dimensions.get("phase") or candidate.get("phase"),
                    "text_bucket": buckets.get("text_bucket") or candidate.get("text_bucket"),
                    "token_bucket": buckets.get("token_bucket") or candidate.get("token_bucket"),
                    "has_tools": buckets.get("has_tools") if "has_tools" in buckets else candidate.get("has_tools"),
                }.items()
                if value is not None
            }
            entries.append({
                "path": f"$.policies.{section}.recommendation.{list_name}[{index}]",
                "source": list_name,
                "section": section,
                "rule_id": candidate.get("rule_id") or candidate.get("policy_id"),
                "candidate_id": candidate.get("candidate_id") or candidate.get("policy_id") or candidate.get("recommendation_id"),
                "policy_source": candidate.get("policy_source") or recommendation.get("policy_source") or policy.get("policy_source"),
                "enabled": list_name == "candidates",
                "conditions": conditions,
                "action": action,
                "rollout": candidate.get("rollout") if isinstance(candidate.get("rollout"), dict) else action.get("rollout"),
                "pattern_hashes": hashes,
                "estimated_savings_usd": candidate.get("estimated_savings_usd"),
                "recommendation_status": list_name.removesuffix("s"),
            })
    return entries


def _condition_values_match(item: dict[str, Any], conditions: dict[str, Any], blockers: list[str]) -> bool:
    model_pattern = conditions.get("model_pattern")
    if model_pattern and str(model_pattern).lower() not in str(item.get("requested_model") or "").lower():
        blockers.append("model-pattern-mismatch")
        return False
    for key in ("source_surface", "app_family", "category", "workflow_phase", "text_bucket", "token_bucket"):
        value = conditions.get(key)
        if value is not None and str(item.get(key) or "") != str(value):
            blockers.append(f"{key.replace('_', '-')}-mismatch")
            return False
    if "category_not_in" in conditions:
        excluded = conditions["category_not_in"]
        excluded_values = {str(value) for value in (excluded if isinstance(excluded, list) else [excluded])}
        if str(item.get("category") or "") in excluded_values:
            blockers.append("category-excluded")
            return False
    if "has_tools" in conditions and bool(conditions.get("has_tools")) != bool(item.get("has_tools")):
        blockers.append("has-tools-mismatch")
        return False
    if "min_text_chars" in conditions and item["text_chars"] < _as_int(conditions.get("min_text_chars")):
        blockers.append("min-text-chars-not-met")
        return False
    if "max_text_chars" in conditions and item["text_chars"] > _as_int(conditions.get("max_text_chars")):
        blockers.append("max-text-chars-exceeded")
        return False
    return True


def _canary_selected(item: dict[str, Any], rollout: dict[str, Any]) -> bool:
    fraction = _as_float(rollout.get("canary_fraction"), 0.0)
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    salt = str(rollout.get("canary_salt") or "")
    digest = hashlib.sha256(f"{salt}:{item.get('id') or ''}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return bucket < fraction


def _pattern_traffic_rollup(items: list[dict[str, Any]]) -> dict[str, Any]:
    rollup = _rollup(items)
    return {
        **rollup,
        "estimated_chars_affected": sum(item["text_chars"] for item in items),
        "estimated_tokens_affected": sum(item["input_tokens"] for item in items),
        "estimated_cost_exposure_usd": _round_usd(sum(item["cost_est_usd"] for item in items)),
        "source_surfaces": sorted({str(item.get("source_surface") or "unknown") for item in items}),
        "app_families": sorted({str(item.get("app_family") or "unknown") for item in items}),
        "workflow_phases": sorted({str(item.get("workflow_phase") or "unknown") for item in items}),
        "categories": sorted({str(item.get("category") or "unknown") for item in items}),
        "policy_sources": sorted({
            str(
                item.get("crunch", {}).get("policy_source")
                or item.get("cache", {}).get("policy_source")
                or item.get("routing", {}).get("policy_source")
                or "unknown"
            )
            for item in items
        }),
    }


def _pattern_bundle_impact(section: str, policy: dict[str, Any], traffic: list[dict[str, Any]]) -> dict[str, Any]:
    entries = _pattern_candidate_entries(policy, section=section)
    summaries: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    for entry in entries:
        hashes = set(entry.get("pattern_hashes") or [])
        matched: list[dict[str, Any]] = []
        applied: list[dict[str, Any]] = []
        holdout = 0
        bypass = 0
        blocker_counts: dict[str, int] = {}
        for item in traffic:
            if hashes and not hashes.intersection(set(item.get("pattern_hashes") or [])):
                continue
            if not hashes and not item.get("pattern_hashes"):
                continue
            blockers: list[str] = []
            if not _condition_values_match(item, entry.get("conditions") or {}, blockers):
                for blocker in blockers:
                    blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
                continue
            matched.append(item)
            rollout = entry.get("rollout") if isinstance(entry.get("rollout"), dict) else {}
            recommendation_mode = str(rollout.get("recommendation_mode") or "").lower()
            if not entry.get("enabled"):
                blocker_counts["recommendation-not-enabled-for-local-apply"] = blocker_counts.get("recommendation-not-enabled-for-local-apply", 0) + 1
                bypass += 1
                continue
            if recommendation_mode in {"omitted", "warning-only"}:
                blocker_counts[f"recommendation-mode-{recommendation_mode}"] = blocker_counts.get(f"recommendation-mode-{recommendation_mode}", 0) + 1
                bypass += 1
                continue
            if section == "cache":
                if item.get("stream") and (entry.get("action") or {}).get("streaming") is False:
                    blocker_counts["streaming-cache-safety-block"] = blocker_counts.get("streaming-cache-safety-block", 0) + 1
                    bypass += 1
                    continue
                if item.get("replayability_level") == "features_only":
                    blocker_counts["features-only-cache-replay-block"] = blocker_counts.get("features-only-cache-replay-block", 0) + 1
                    bypass += 1
                    continue
            if isinstance(rollout, dict) and rollout.get("canary_enabled") and not _canary_selected(item, rollout):
                holdout += 1
                continue
            applied.append(item)
        safety_blockers = [
            {"reason": reason, "count": count}
            for reason, count in sorted(blocker_counts.items())
        ]
        summary = {
            **_pattern_traffic_rollup(applied),
            "path": entry["path"],
            "source": entry.get("source"),
            "candidate_id": entry.get("candidate_id"),
            "rule_id": entry.get("rule_id"),
            "policy_source": entry.get("policy_source"),
            "pattern_hashes": sorted(hashes),
            "recommendation_mode": (entry.get("rollout") or {}).get("recommendation_mode") if isinstance(entry.get("rollout"), dict) else entry.get("recommendation_status"),
            "canary": {
                "enabled": bool((entry.get("rollout") or {}).get("canary_enabled")) if isinstance(entry.get("rollout"), dict) else False,
                "fraction": (entry.get("rollout") or {}).get("canary_fraction") if isinstance(entry.get("rollout"), dict) else None,
                "salt": (entry.get("rollout") or {}).get("canary_salt") if isinstance(entry.get("rollout"), dict) else None,
            },
            "would_match_count": len(matched),
            "would_apply_count": len(applied),
            "would_holdout_count": holdout,
            "would_bypass_count": bypass,
            "safety_blocker_count": sum(blocker_counts.values()),
            "safety_blockers": safety_blockers,
            "estimated_candidate_savings_usd": entry.get("estimated_savings_usd"),
        }
        matched_dimensions = _pattern_traffic_rollup(matched)
        for key in ("source_surfaces", "app_families", "workflow_phases", "categories", "policy_sources"):
            summary[key] = matched_dimensions[key]
        if summary["historical_error_rate"] > 0.05:
            warning = _impact_warning(
                f"high-error-rate-{section}-pattern-match",
                entry["path"],
                f"pattern-matched historical rows had {summary['historical_error_rate']:.1%} error rate",
            )
            warnings.append(warning)
            summary.setdefault("warnings", []).append(warning)
        summaries.append(summary)
    return {
        "status": "simulated",
        "policy_source": policy.get("policy_source") or (policy.get("recommendation") or {}).get("policy_source"),
        "candidate_count": len(summaries),
        "would_match_count": sum(item["would_match_count"] for item in summaries),
        "would_apply_count": sum(item["would_apply_count"] for item in summaries),
        "would_holdout_count": sum(item["would_holdout_count"] for item in summaries),
        "would_bypass_count": sum(item["would_bypass_count"] for item in summaries),
        "safety_blocker_count": sum(item["safety_blocker_count"] for item in summaries),
        "estimated_chars_affected": sum(item["estimated_chars_affected"] for item in summaries),
        "estimated_tokens_affected": sum(item["estimated_tokens_affected"] for item in summaries),
        "estimated_cost_exposure_usd": _round_usd(sum(item["estimated_cost_exposure_usd"] for item in summaries)),
        "metadata_only": True,
        "raw_bodies_read": False,
        "candidates": summaries,
        "warnings": warnings,
    }


def _crunch_impact(policy: dict[str, Any], calls: list[dict[str, Any]], traffic: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    enabled = _enabled(policy.get("enabled", True))
    threshold = _as_int(policy.get("threshold_chars"), 24000)
    eligible = [item for item in calls if enabled and item["text_chars"] >= threshold]
    existing_saved_tokens = sum(_as_int(item["crunch"].get("tokens_saved_est")) for item in eligible)
    features: list[dict[str, Any]] = [{
        "name": "base_crunch",
        "would_match_count": len(eligible),
        "threshold_chars": threshold,
        "historical_tokens_saved_est": existing_saved_tokens,
        **{k: v for k, v in _rollup(eligible).items() if k != "would_match_count"},
    }]
    warnings: list[dict[str, str]] = []

    old_context = policy.get("old_context_summarization") if isinstance(policy.get("old_context_summarization"), dict) else {}
    if _enabled(old_context.get("enabled")):
        min_chars = _as_int(old_context.get("min_request_chars"), 32000)
        matched = [item for item in calls if item["text_chars"] >= min_chars]
        feature = {
            "name": "old_context_summarization",
            "would_match_count": len(matched),
            "min_request_chars": min_chars,
            **{k: v for k, v in _rollup(matched).items() if k != "would_match_count"},
        }
        warning = _impact_warning(
            "old-context-summarization-impact-risk",
            "$.policies.crunch.old_context_summarization.enabled",
            "old-context summarization changes request context; review matched error/retry rate before applying",
        )
        warnings.append(warning)
        feature["warnings"] = [warning]
        features.append(feature)

    thinking_dedup = policy.get("thinking_deduplication") if isinstance(policy.get("thinking_deduplication"), dict) else {}
    if _enabled(thinking_dedup.get("enabled")):
        min_chars = _as_int(thinking_dedup.get("min_chars"), 2000)
        matched = [item for item in calls if item["thinking"] and item["text_chars"] >= min_chars]
        features.append({
            "name": "thinking_deduplication",
            "would_match_count": len(matched),
            "min_chars": min_chars,
            **{k: v for k, v in _rollup(matched).items() if k != "would_match_count"},
        })

    pattern_impact = _pattern_bundle_impact("crunch", policy, traffic or calls)
    return {
        "status": "simulated",
        "policy_source": policy.get("policy_source"),
        "enabled": enabled,
        "features": features,
        "pattern_rules": pattern_impact,
        "warnings": [*warnings, *pattern_impact.get("warnings", [])],
    }


def _cache_impact(policy: dict[str, Any], calls: list[dict[str, Any]], traffic: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    exact = policy.get("exact_cache") if isinstance(policy.get("exact_cache"), dict) else {}
    semantic = policy.get("semantic_cache") if isinstance(policy.get("semantic_cache"), dict) else {}
    exact_enabled = _enabled(exact.get("enabled", policy.get("enabled", True)))
    cache_tool_calls = _enabled(exact.get("cache_tool_calls", False))
    non_streaming = [item for item in calls if not item["stream"]]
    eligible = [item for item in non_streaming if exact_enabled and (cache_tool_calls or not item["has_tools"])]
    excluded_streaming = sum(1 for item in calls if item["stream"])
    excluded_tool = sum(1 for item in non_streaming if item["has_tools"] and not cache_tool_calls)
    cache_hits = sum(1 for item in eligible if item["cache"].get("status") == "hit" or item["cache"].get("cache_hit"))
    warnings: list[dict[str, str]] = []
    if cache_tool_calls:
        warnings.append(_impact_warning(
            "tool-call-cache-impact-risk",
            "$.policies.cache.exact_cache.cache_tool_calls",
            "historical tool calls would become cache-eligible; stale filesystem-dependent results remain a safety risk",
        ))
    if _enabled(semantic.get("enabled")):
        warnings.append(_impact_warning(
            "semantic-cache-impact-risk",
            "$.policies.cache.semantic_cache.enabled",
            "semantic cache impact cannot be proven from metadata-only review; false-positive reuse requires quality checks",
        ))
    pattern_impact = _pattern_bundle_impact("cache", policy, traffic or calls)
    return {
        "status": "simulated",
        "policy_source": policy.get("policy_source"),
        "exact_cache": {
            "enabled": exact_enabled,
            "cache_tool_calls": cache_tool_calls,
            **_rollup(eligible),
            "excluded_streaming_count": excluded_streaming,
            "excluded_tool_count": excluded_tool,
            "historical_cache_hit_count": cache_hits,
            "estimated_savings_usd": _round_usd(sum(item["cost_est_usd"] for item in eligible if item["cache"].get("status") == "hit")),
            "estimate_note": "metadata-only review cannot infer new exact duplicate hashes without stored request bodies or hashes",
        },
        "semantic_cache": {
            "enabled": _enabled(semantic.get("enabled")),
            "threshold": semantic.get("threshold"),
        },
        "pattern_rules": pattern_impact,
        "warnings": [*warnings, *pattern_impact.get("warnings", [])],
    }


def _routing_experiment_impact(policy: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    experiment = policy.get("policy") if isinstance(policy.get("policy"), dict) else policy
    enabled = _enabled(experiment.get("enabled"))
    min_text = _as_int(experiment.get("min_text_chars"), 0)
    max_text = _as_int(experiment.get("max_text_chars"), 30000)
    categories = experiment.get("categories")
    category_set = {str(item) for item in categories} if isinstance(categories, list) else set()
    eligible = [
        item for item in calls
        if enabled
        and not item["stream"]
        and min_text <= item["text_chars"] <= max_text
        and (not category_set or item["category"] in category_set)
    ]
    sample_rate = _as_float(experiment.get("sample_rate"), 0.0)
    warnings: list[dict[str, str]] = []
    if _enabled(experiment.get("store_response_bodies")):
        warnings.append(_impact_warning(
            "routing-experiment-response-body-storage",
            "$.policies.routing_experiments.store_response_bodies",
            "storing response bodies is outside the default metadata-only posture",
        ))
    return {
        "status": "simulated",
        "policy_source": policy.get("policy_source"),
        "enabled": enabled,
        "eligible_call_count": len(eligible),
        "estimated_sample_count": round(len(eligible) * sample_rate, 2),
        "sample_rate": sample_rate,
        **{k: v for k, v in _rollup(eligible).items() if k != "would_match_count"},
        "warnings": warnings,
    }


def _codex_app_impact(policy: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    rules = policy.get("rules") if isinstance(policy.get("rules"), list) else []
    return {
        "status": "review-only",
        "policy_source": policy.get("policy_source"),
        "surface": canonical_source_surface(policy.get("surface", CODEX_APP_SOURCE_SURFACE)),
        "rule_count": len(rules),
        "applied_to_provider_routing": False,
        "applied_to_codex_app_proxy": False,
        "metadata_only": True,
        "raw_bodies_read": False,
        "note": "Codex app turn-level recommendations are reviewed in local bundles but are not applied to provider routing rules.",
    }


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _normalize_pattern_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.startswith("sha256:"):
        digest = text.removeprefix("sha256:")
    else:
        digest = text
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return f"sha256:{digest}"


def _collect_pattern_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if (
                key_l in {"pattern_hash", "normalized_pattern_hash", "crunch_pattern_hash", "cache_pattern_hash"}
                or key_l.endswith(("_pattern_hash", "_pattern_sha256"))
                or ("pattern" in key_l and key_l.endswith(("_hash", "_hashes", "_sha256")))
                or key_l in {"pattern_hashes", "hashes"}
            ):
                if isinstance(item, list):
                    hashes.extend(hash_value for nested in item if (hash_value := _normalize_pattern_hash(nested)))
                elif (hash_value := _normalize_pattern_hash(item)) is not None:
                    hashes.append(hash_value)
            hashes.extend(_collect_pattern_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(_collect_pattern_hashes(item))
    return sorted(set(hashes))


def _source_surface_for_call(provider: Any, path: Any) -> str:
    provider_l = str(provider or "").lower()
    path_l = str(path or "").lower()
    if provider_l in {"codex-app", "codex_app"}:
        return CODEX_APP_SOURCE_SURFACE
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        if "chat/completions" in path_l:
            return "openai_chat"
        return "openai_responses"
    return "unknown"


def _app_family_for_call(provider: Any, requested_model: Any, path: Any) -> str:
    provider_l = str(provider or "").lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in str(path or "").lower():
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def _codex_app_rule_candidate_id(rule: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "recommendation_id", "policy_id"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            return value
    managed = rule.get("managed_recommendation")
    if isinstance(managed, dict):
        for key in ("candidate_id", "recommendation_id", "policy_id"):
            value = managed.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return f"codex-app-rule-{index + 1}"


def codex_app_policy_review_summary(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        policy = {}
    rules = policy.get("rules") if isinstance(policy.get("rules"), list) else []
    policy_source = str(policy.get("policy_source") or "")
    review_only = policy_source.startswith("managed-") or _enabled(policy.get("review_only", True))
    supported_conditions = _as_string_list(policy.get("supported_conditions")) or sorted(_CODEX_APP_CONDITION_KEYS)
    supported_actions = _as_string_list(policy.get("supported_actions")) or sorted(_CODEX_APP_ACTION_KEYS)
    application = policy.get("application") if isinstance(policy.get("application"), dict) else {}
    rule_summaries: list[dict[str, Any]] = []
    condition_keys: set[str] = set()
    action_keys: set[str] = set()
    pass_through_reasons: list[str] = []
    omission_reasons: list[str] = []

    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        managed = rule.get("managed_recommendation") if isinstance(rule.get("managed_recommendation"), dict) else {}
        condition_keys.update(str(key) for key in conditions)
        action_keys.update(str(key) for key in action)
        for key in ("pass_through_reason", "reason", "cache_eligibility_reason"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                pass_through_reasons.append(value)
        for key in ("omission_reason", "omitted_reason", "pass_through_reason", "reason"):
            value = rule.get(key, managed.get(key))
            if isinstance(value, str) and value.strip():
                omission_reasons.append(value)
        rule_summaries.append({
            "path": f"$.policies.codex_app.rules[{index}]",
            "candidate_id": _codex_app_rule_candidate_id(rule, index),
            "conditions": conditions,
            "action": action,
            "condition_keys": sorted(str(key) for key in conditions),
            "action_keys": sorted(str(key) for key in action),
            "confidence": managed.get("confidence", rule.get("confidence")),
            "sample_count": managed.get("sample_count", rule.get("sample_count")),
            "review_only": review_only,
            "applied": not review_only,
            "application_status": "not-applied" if review_only else "applied-locally",
        })

    return {
        "status": "review-only" if review_only else "applied-locally",
        "policy_source": policy.get("policy_source"),
        "surface": canonical_source_surface(policy.get("surface", CODEX_APP_SOURCE_SURFACE)),
        "review_only": review_only,
        "enabled": _enabled(policy.get("enabled", False)),
        "rule_count": len(rule_summaries),
        "candidate_count": len(rule_summaries),
        "candidate_ids": [rule["candidate_id"] for rule in rule_summaries],
        "supported_conditions": supported_conditions,
        "supported_actions": supported_actions,
        "condition_keys_present": sorted(condition_keys),
        "action_keys_present": sorted(action_keys),
        "application": {
            "status": "not-applied" if review_only else (application.get("status") or "applied-locally"),
            "reason": application.get("reason")
            or (
                "Codex app recommendations are review-only until a future explicit local action exists."
                if review_only
                else "Safe Codex app actions are represented as local policy file writes."
            ),
            "applied_to_provider_routing": False,
            "applied_to_codex_app_proxy": not review_only,
            "writes_local_policy_files": not review_only,
        },
        "pass_through_reasons": sorted(set(pass_through_reasons)),
        "omission_reasons": sorted(set(omission_reasons)),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_params_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
        },
        "rules": rule_summaries,
    }


_PATTERN_RAW_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "command",
    "content",
    "file_content",
    "local_file",
    "message",
    "param",
    "policy_yaml",
    "prompt",
    "provider_body",
    "raw",
    "request",
    "response",
    "secret",
    "system",
    "tool_payload",
    "transcript",
)


def _safe_pattern_value(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _PATTERN_RAW_KEY_PARTS):
                continue
            cleaned = _safe_pattern_value(item)
            if cleaned is not None:
                safe[str(key)] = cleaned
        return safe
    if isinstance(value, list):
        return [item for item in (_safe_pattern_value(item) for item in value[:50]) if item is not None]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    return None


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _candidate_id(candidate: dict[str, Any], index: int, *, section: str) -> str:
    for key in ("candidate_id", "policy_id", "recommendation_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
    for key in ("candidate_id", "policy_id", "recommendation_id"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"{section}-pattern-candidate-{index + 1}"


def _bucket_sample_count(sample_count: Any) -> str:
    count = _as_int(sample_count)
    if count <= 0:
        return "zero"
    if count < 5:
        return "lt_5"
    if count < 10:
        return "5_9"
    if count < 25:
        return "10_24"
    if count < 100:
        return "25_99"
    return "gte_100"


def _bucket_savings(value: Any) -> str:
    amount = _as_float(value)
    if amount <= 0:
        return "zero"
    if amount < 0.01:
        return "lt_1_cent"
    if amount < 0.10:
        return "1_10_cents"
    if amount < 1.0:
        return "10_100_cents"
    return "gte_1_usd"


def _pattern_delta_for_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    delta = _first_mapping(candidate.get("delta"), candidate.get("change_summary"), candidate.get("health_delta"))
    confidence_inputs = candidate.get("confidence_inputs") if isinstance(candidate.get("confidence_inputs"), dict) else {}
    if "lifecycle_delta" in confidence_inputs:
        delta = {**delta, "lifecycle_delta": confidence_inputs.get("lifecycle_delta")}
    if not delta:
        status = candidate.get("delta_status") or candidate.get("change_status")
        if isinstance(status, str) and status.strip():
            delta = {"status": status}
    return _safe_pattern_value(delta) if delta else {}


def _candidate_action(candidate: dict[str, Any]) -> dict[str, Any]:
    action = _first_mapping(candidate.get("action"), candidate.get("recommended_action"))
    if action:
        return _safe_pattern_value(action) or {}
    keys = (
        "cache_eligibility",
        "cache_policy",
        "crunch_profile",
        "dedupe_rule",
        "invalidation_rule",
        "threshold_rule",
    )
    return {key: _safe_pattern_value(candidate[key]) for key in keys if key in candidate}


def _pattern_candidate_review(
    candidate: dict[str, Any],
    index: int,
    *,
    section: str,
    list_name: str,
) -> dict[str, Any]:
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else candidate
    requirements = _first_mapping(candidate.get("local_action_requirements"), evidence.get("local_action_requirements"))
    confidence_inputs = _first_mapping(candidate.get("confidence_inputs"), evidence.get("confidence_inputs"))
    review_evidence = _first_mapping(candidate.get("review_evidence"), evidence.get("review_evidence"))
    privacy_profiles = (
        candidate.get("privacy_profile_counts")
        or evidence.get("privacy_profile_counts")
        or confidence_inputs.get("privacy_profile_counts")
        or {}
    )
    sample_count = candidate.get("sample_count", evidence.get("sample_count"))
    savings = (
        candidate.get("estimated_savings_usd")
        if "estimated_savings_usd" in candidate
        else evidence.get("estimated_savings_usd", evidence.get("avg_savings_usd"))
    )
    omission_reasons = _as_string_list(candidate.get("omission_reasons")) or _as_string_list(candidate.get("reason"))
    warning_reasons = _as_string_list(candidate.get("warning_reasons"))
    actionability = requirements.get("actionability_status") or (
        "omitted" if list_name == "omitted_candidates" else "review-only-local-action"
    )
    delta = _pattern_delta_for_candidate(candidate)
    changed = bool(delta and str(delta.get("status") or "").lower() not in {"unchanged", "none", "no-change"})
    if list_name == "candidates":
        candidate_status = "candidate"
    elif list_name == "review_only_candidates":
        candidate_status = "review_only"
    else:
        candidate_status = "omitted"
    return {
        "path": f"$.policies.{section}.recommendation.{list_name}[{index}]",
        "candidate_id": _candidate_id(candidate, index, section=section),
        "policy_section": section,
        "candidate_family": candidate.get("candidate_family") or evidence.get("candidate_family") or f"{section}-policy-rule",
        "candidate_status": candidate_status,
        "policy_source": candidate.get("policy_source") or evidence.get("policy_source") or "managed-recommended",
        "confidence": candidate.get("confidence", evidence.get("confidence")),
        "confidence_inputs": _safe_pattern_value(confidence_inputs),
        "sample_count": sample_count,
        "sample_count_bucket": candidate.get("sample_count_bucket") or _bucket_sample_count(sample_count),
        "savings_bucket": candidate.get("savings_bucket") or _bucket_savings(savings),
        "estimated_savings_usd": savings,
        "error_rate": candidate.get("error_rate", evidence.get("error_rate")),
        "evidence_buckets": {
            "source_surface": candidate.get("source_surface", evidence.get("source_surface")),
            "app_family": candidate.get("app_family", evidence.get("app_family")),
            "category": candidate.get("category", evidence.get("category")),
            "phase": candidate.get("phase", evidence.get("phase")),
            "text_bucket": candidate.get("text_bucket", evidence.get("text_bucket")),
            "token_bucket": candidate.get("token_bucket", evidence.get("token_bucket")),
            "privacy_profile_counts": _safe_pattern_value(privacy_profiles),
        },
        "review_evidence": _safe_pattern_value(review_evidence),
        "action": _candidate_action(candidate),
        "local_action_requirements": _safe_pattern_value(requirements),
        "actionability_status": actionability,
        "omission_reasons": omission_reasons,
        "warning_reasons": warning_reasons,
        "delta": delta,
        "changed_since_last_review": changed,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_policy_yaml_included": False,
            "raw_provider_bodies_included": False,
            "commands_included": False,
            "credentials_included": False,
            "local_file_contents_included": False,
        },
    }


def pattern_policy_review_summary(policy: Any, *, section: str) -> dict[str, Any]:
    if not isinstance(policy, dict):
        policy = {}
    recommendation = policy.get("recommendation") if isinstance(policy.get("recommendation"), dict) else {}
    candidate_sources = (
        ("candidates", recommendation.get("candidates") if isinstance(recommendation.get("candidates"), list) else []),
        (
            "review_only_candidates",
            recommendation.get("review_only_candidates") if isinstance(recommendation.get("review_only_candidates"), list) else [],
        ),
        (
            "omitted_candidates",
            recommendation.get("omitted_candidates") if isinstance(recommendation.get("omitted_candidates"), list) else [],
        ),
    )
    candidates: list[dict[str, Any]] = []
    for list_name, items in candidate_sources:
        for index, candidate in enumerate(items):
            if isinstance(candidate, dict):
                candidates.append(_pattern_candidate_review(candidate, index, section=section, list_name=list_name))

    representable = [item for item in candidates if item["candidate_status"] == "candidate"]
    review_only = [item for item in candidates if item["actionability_status"] == "review-only-local-action"]
    omitted = [item for item in candidates if item["candidate_status"] == "omitted"]
    changed = [item for item in candidates if item["changed_since_last_review"]]
    unchanged = [
        item for item in candidates
        if item.get("delta") and not item["changed_since_last_review"]
    ]
    application_status = "review-only-not-applied" if candidates else "no-pattern-candidates"
    return {
        "schema": PATTERN_CANDIDATE_REVIEW_SCHEMA,
        "status": "review-only" if candidates else "empty",
        "policy_source": recommendation.get("policy_source") or policy.get("policy_source"),
        "policy_section": section,
        "candidate_count": len(candidates),
        "representable_candidate_count": len(representable),
        "review_only_candidate_count": len(review_only),
        "omitted_candidate_count": len(omitted),
        "changed_health_candidate_count": len(changed),
        "unchanged_candidate_count": len(unchanged),
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "application": {
            "status": application_status,
            "reason": (
                f"Managed {section} pattern candidates are reviewed locally and are not written by review commands."
                if candidates
                else f"No managed {section} pattern candidates are present."
            ),
            "expected_policy_section": section,
            "writes_local_policy_files": False,
        },
        "rationale": recommendation.get("rationale"),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_policy_yaml_included": False,
            "raw_provider_bodies_included": False,
            "commands_included": False,
            "credentials_included": False,
            "local_file_contents_included": False,
        },
        "candidates": candidates,
    }


def _review_human_summary(result: dict[str, Any]) -> list[str]:
    lines = [
        f"Policy review: {'ok' if result.get('ok') else 'invalid'}; changed sections: {', '.join(result.get('changed_sections') or []) or 'none'}.",
    ]
    section_reviews = result.get("section_reviews") if isinstance(result.get("section_reviews"), dict) else {}
    for section in ("crunch", "cache", "codex_app"):
        review = section_reviews.get(section)
        if not isinstance(review, dict):
            continue
        if section in {"crunch", "cache"}:
            lines.append(
                f"{section} pattern candidates: {review.get('candidate_count', 0)} total, "
                f"{review.get('review_only_candidate_count', 0)} review-only, "
                f"{review.get('omitted_candidate_count', 0)} omitted; application: "
                f"{(review.get('application') or {}).get('status')}."
            )
        elif review.get("candidate_count"):
            lines.append(
                f"codex_app candidates: {review.get('candidate_count', 0)} review-only; application: "
                f"{(review.get('application') or {}).get('status')}."
            )
    return lines


def _review_diff_payload(diff: dict[str, Any]) -> dict[str, Any]:
    changes = []
    for change in diff.get("changes", []):
        if not isinstance(change, dict):
            continue
        sanitized = {
            key: value
            for key, value in change.items()
            if key not in {"old", "new"}
        }
        sanitized["old"] = _safe_pattern_value(change.get("old"))
        sanitized["new"] = _safe_pattern_value(change.get("new"))
        changes.append(sanitized)
    return {
        "schema": diff.get("schema"),
        "ok": diff.get("ok"),
        "changed": diff.get("changed"),
        "changed_sections": diff.get("changed_sections", []),
        "change_count": diff.get("change_count", 0),
        "changes": changes,
    }


def simulate_policy_bundle_impact(
    proposed: Any,
    *,
    db_path: str | None = None,
    limit: int = _DEFAULT_IMPACT_LIMIT,
) -> dict[str, Any]:
    path = db_path or os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv(
        "AGENTFLOW_DB",
        str(Path.home() / ".agentflow" / "agentflow.sqlite3"),
    )
    if not isinstance(proposed, dict):
        return {
            "schema": POLICY_IMPACT_SCHEMA,
            "status": "unavailable",
            "reason": "invalid-bundle",
            "message": "proposed policy bundle is not an object",
        }
    calls, unavailable = _load_recent_call_features(str(path), limit=limit)
    if unavailable:
        return {
            "schema": POLICY_IMPACT_SCHEMA,
            "status": "unavailable",
            "ok": False,
            **unavailable,
            "db_path": str(path),
            "lookback_call_limit": limit,
            "sections": {},
            "warnings": [],
        }
    codex_events = _load_recent_codex_event_features(str(path), limit=limit)
    traffic = [*calls, *codex_events]

    policies = proposed.get("policies") if isinstance(proposed.get("policies"), dict) else {}
    sections: dict[str, Any] = {}
    if isinstance(policies.get("routing"), dict):
        sections["routing"] = _routing_impact(policies["routing"], calls)
    if isinstance(policies.get("crunch"), dict):
        sections["crunch"] = _crunch_impact(policies["crunch"], calls, traffic)
    if isinstance(policies.get("cache"), dict):
        sections["cache"] = _cache_impact(policies["cache"], calls, traffic)
    if isinstance(policies.get("routing_experiments"), dict):
        sections["routing_experiments"] = _routing_experiment_impact(policies["routing_experiments"], calls)
    if isinstance(policies.get("codex_app"), dict):
        sections["codex_app"] = _codex_app_impact(policies["codex_app"], calls)
    warnings = [
        warning
        for section in sections.values()
        if isinstance(section, dict)
        for warning in section.get("warnings", [])
    ]
    return {
        "schema": POLICY_IMPACT_SCHEMA,
        "status": "simulated",
        "ok": True,
        "metadata_only": True,
        "raw_bodies_read": False,
        "db_path": str(path),
        "lookback_call_limit": limit,
        "sampled_call_count": len(calls),
        "sampled_codex_turn_count": len(codex_events),
        "sampled_metadata_row_count": len(traffic),
        "sections": sections,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def policy_bundle_safety_warnings(bundle: Any) -> list[dict[str, str]]:
    from agentflow_proxy.recommendation_health import HEALTH_WARNING_CODES, summarize_recommendation_health

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

    recommendation_health = summarize_recommendation_health(bundle)
    for row in recommendation_health.get("rows", []):
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "health")
        code = HEALTH_WARNING_CODES.get(kind, f"managed-recommendation-{kind}")
        candidate_id = row.get("candidate_id")
        suffix = f" for {candidate_id}" if candidate_id else ""
        _add_warning(
            warnings,
            code,
            str(row.get("path") or "$.recommendation"),
            f"managed recommendation evidence health warning{suffix}: {row.get('code') or kind}",
        )

    return warnings


def review_policy_bundle(
    current: Any,
    proposed: Any,
    *,
    impact_db_path: str | None = None,
    include_impact: bool = True,
    impact_limit: int = _DEFAULT_IMPACT_LIMIT,
) -> dict[str, Any]:
    from agentflow_proxy.recommendation_health import summarize_recommendation_health

    diff = compare_policy_bundles(current, proposed)
    warnings = policy_bundle_safety_warnings(proposed)
    result = {
        "schema": POLICY_BUNDLE_REVIEW_SCHEMA,
        "ok": bool(diff["ok"]),
        "changed": bool(diff.get("changed", False)) if diff["ok"] else False,
        "changed_sections": diff.get("changed_sections", []) if diff["ok"] else [],
        "change_count": int(diff.get("change_count", 0)) if diff["ok"] else 0,
        "safety_warning_count": len(warnings),
        "safety_warnings": warnings,
        "current_validation": diff.get("before_validation"),
        "proposed_validation": diff.get("after_validation"),
        "diff": _review_diff_payload(diff),
    }
    proposed_validation = diff.get("after_validation") if isinstance(diff.get("after_validation"), dict) else {}
    result["provenance"] = proposed_validation.get("provenance")
    result["recommendation_health"] = summarize_recommendation_health(proposed)
    policies = proposed.get("policies") if isinstance(proposed, dict) and isinstance(proposed.get("policies"), dict) else {}
    section_reviews: dict[str, Any] = {}
    for section in ("crunch", "cache"):
        if isinstance(policies.get(section), dict):
            review = pattern_policy_review_summary(policies[section], section=section)
            if review.get("candidate_count"):
                section_reviews[section] = review
    if isinstance(policies.get("codex_app"), dict):
        section_reviews["codex_app"] = codex_app_policy_review_summary(policies["codex_app"])
    if section_reviews:
        result["section_reviews"] = section_reviews
    result["human_summary"] = _review_human_summary(result)
    if include_impact:
        result["impact_summary"] = simulate_policy_bundle_impact(
            proposed,
            db_path=impact_db_path,
            limit=impact_limit,
        )
    return result


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
        payload = {key: policy[key] for key in ("exact_cache", "semantic_cache", "file_watch") if isinstance(policy.get(key), dict)}
        if isinstance(policy.get("pattern_rules"), list):
            payload["pattern_rules"] = policy["pattern_rules"]
        return payload
    if section == "routing_experiments":
        experiment_policy = policy.get("policy")
        return experiment_policy if isinstance(experiment_policy, dict) else {}
    if section == "codex_app":
        payload: dict[str, Any] = {}
        if "enabled" in policy:
            payload["enabled"] = policy.get("enabled")
        if isinstance(policy.get("summary_model_hint"), dict):
            payload["summary_model_hint"] = {
                key: policy["summary_model_hint"][key]
                for key in ("enabled", "target_model")
                if key in policy["summary_model_hint"]
            }
        if isinstance(policy.get("exact_cache"), dict):
            payload["exact_cache"] = {
                key: policy["exact_cache"][key]
                for key in ("enabled", "namespace")
                if key in policy["exact_cache"]
            }
        if isinstance(policy.get("crunch"), dict):
            payload["crunch"] = {
                key: policy["crunch"][key]
                for key in ("profiles",)
                if key in policy["crunch"]
            }
        return payload
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
    requested_sections = list(sections or APPLY_POLICY_SECTIONS)
    invalid_sections = sorted(set(requested_sections) - set(APPLY_POLICY_SECTIONS))
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
        "provenance": validation.get("provenance"),
        "safety_warning_count": len(warnings),
        "safety_warnings": warnings,
        "error": None,
    }

    if invalid_sections:
        result["error"] = {
            "type": "invalid_sections",
            "message": "unknown or review-only policy section requested",
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
    for section in APPLY_POLICY_SECTIONS:
        if section not in requested_sections:
            result["skipped_sections"].append({"section": section, "reason": "not-requested"})
            continue

        policy = policies.get(section) if isinstance(policies.get(section), dict) else {}
        if section == "codex_app" and (
            str(policy.get("policy_source") or "").startswith("managed-") or bool(policy.get("review_only"))
        ):
            result["skipped_sections"].append({
                "section": "codex_app",
                "reason": "review-only-not-applied",
                "message": "Managed Codex app recommendations are visible in review output but are not written to local YAML policy files.",
            })
            continue
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
    requested_sections = list(sections or APPLY_POLICY_SECTIONS)
    invalid_sections = sorted(set(requested_sections) - set(APPLY_POLICY_SECTIONS))
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
            "message": "unknown or review-only policy section requested",
            "sections": invalid_sections,
        }
        return result

    plans: list[dict[str, Any]] = []
    missing_sections: list[str] = []
    unreadable_backups: list[dict[str, str]] = []
    for section in APPLY_POLICY_SECTIONS:
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
