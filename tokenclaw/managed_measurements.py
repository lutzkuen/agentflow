from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from tokenclaw.client_contract import CLIENT_CONTRACT_META_SCHEMA, filter_payload_by_client_contract
from tokenclaw.managed_egress import RAW_FEATURE_KEYS, assert_managed_egress_safe
from tokenclaw.managed_mode import ManagedProductMode, managed_product_mode


MANAGED_MEASUREMENT_FACTS_SCHEMA = "tokenclaw.managed_measurement_facts.v1"
MANAGED_MEASUREMENT_PRIVACY_SCHEMA = "tokenclaw.managed_measurement_privacy.v1"
VALID_MEASUREMENT_STAGES = {"preflight", "outcome"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _path_parts(path: str) -> list[str]:
    return [part for part in str(path).replace("/", ".").split(".") if part]


def _path_is_safe(path: str) -> bool:
    parts = _path_parts(path)
    return bool(parts) and not any(part.lower() in RAW_FEATURE_KEYS for part in parts)


def _flatten_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{prefix}.{key_text}" if prefix else key_text
            paths.add(child_path)
            paths.update(_flatten_paths(child, child_path))
    elif isinstance(value, list):
        for item in value:
            paths.update(_flatten_paths(item, prefix))
    return paths


def _unsafe_paths(value: Any) -> list[str]:
    blocked = []
    for path in _flatten_paths(value):
        if any(part.lower() in RAW_FEATURE_KEYS for part in _path_parts(path)):
            blocked.append(path)
    return sorted(set(blocked))


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in _path_parts(path):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _contract_from_meta(contract_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    contract = contract_meta.get("contract") if isinstance(contract_meta, dict) else None
    return contract if isinstance(contract, dict) else None


def _measurement_paths(contract: dict[str, Any] | None, stage: str) -> list[str]:
    plan = contract.get("measurement_plan") if isinstance(contract, dict) else None
    paths = plan.get(stage) if isinstance(plan, dict) else None
    if not isinstance(paths, list):
        return []
    return [str(path) for path in paths if isinstance(path, str) and path]


def _contract_hash(contract: dict[str, Any] | None, stage: str) -> str | None:
    if not isinstance(contract, dict):
        return None
    basis = {
        "schema": contract.get("schema"),
        "contract_id": contract.get("contract_id"),
        "expires_at": contract.get("expires_at"),
        "stage": stage,
        "measurement_plan": contract.get("measurement_plan"),
        "allowed_action_families": contract.get("allowed_action_families"),
    }
    return "sha256:" + hashlib.sha256(_stable_json(basis).encode("utf-8")).hexdigest()


def _privacy() -> dict[str, Any]:
    return {
        "schema": MANAGED_MEASUREMENT_PRIVACY_SCHEMA,
        "metadata_only": True,
        "raw_payload_included": False,
        "raw_prompt_included": False,
        "raw_response_included": False,
        "provider_body_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "api_key_value_included": False,
    }


def _base_result(
    *,
    stage: str,
    contract_meta: dict[str, Any] | None,
    product_mode: ManagedProductMode,
) -> dict[str, Any]:
    contract = _contract_from_meta(contract_meta)
    result: dict[str, Any] = {
        "schema": MANAGED_MEASUREMENT_FACTS_SCHEMA,
        "stage": stage,
        "status": "skipped",
        "reason": "not-evaluated",
        "enabled": bool(product_mode.server_calls_enabled),
        "active": False,
        "contract_id": contract.get("contract_id") if isinstance(contract, dict) else None,
        "contract_hash": _contract_hash(contract, stage),
        "contract_status": contract_meta.get("status") if isinstance(contract_meta, dict) else None,
        "contract_reason": contract_meta.get("reason") if isinstance(contract_meta, dict) else None,
        "contract_cache_status": contract_meta.get("cache_status") if isinstance(contract_meta, dict) else None,
        "measurement_path_count": len(_measurement_paths(contract, stage)),
        "included_field_names": [],
        "omitted_field_names": [],
        "blocked_field_names": [],
        "facts": {},
        "privacy": _privacy(),
        "product_mode": product_mode.public_meta(),
        "raw_payload_included": False,
        "fallback": "local-policy",
    }
    return {key: value for key, value in result.items() if value is not None}


def _skip_reason(contract_meta: dict[str, Any] | None, product_mode: ManagedProductMode) -> str | None:
    if product_mode.local_rules_only:
        return "local-rules-only"
    if not product_mode.server_calls_enabled:
        return product_mode.reason or "managed-disabled"
    if not isinstance(contract_meta, dict):
        return "missing-client-contract"
    if contract_meta.get("active") is True:
        return None
    schema_error = str(contract_meta.get("schema_error") or "")
    if schema_error == "expired":
        return "expired-contract"
    status = str(contract_meta.get("status") or "")
    reason = str(contract_meta.get("reason") or "")
    if status == "error" or reason in {"timeout", "unreachable", "server-error", "fetch-error"}:
        return "server-unavailable"
    if status == "invalid":
        return "invalid-contract"
    return reason or "no-active-contract"


def execute_measurement_plan(
    payload: dict[str, Any],
    contract_meta: dict[str, Any] | None,
    *,
    stage: str = "preflight",
    product_mode: ManagedProductMode | None = None,
) -> dict[str, Any]:
    """Execute a server-owned measurement plan against feature-only local facts.

    ``payload`` must already be a sanitized local feature snapshot such as a
    policy-decision preflight unit, request-facts envelope, or outcome feature
    unit. The function copies only contract-requested paths and reports missing
    or unsafe fields by path name. It never includes raw provider bodies or raw
    field values in diagnostics.
    """
    stage = stage if stage in VALID_MEASUREMENT_STAGES else "preflight"
    mode = product_mode or managed_product_mode()
    result = _base_result(stage=stage, contract_meta=contract_meta, product_mode=mode)

    reason = _skip_reason(contract_meta, mode)
    if reason is not None:
        result["reason"] = reason
        return result

    contract = _contract_from_meta(contract_meta)
    paths = _measurement_paths(contract, stage)
    safe_paths = [path for path in paths if _path_is_safe(path)]
    blocked_paths = sorted(set(path for path in paths if not _path_is_safe(path)))
    blocked_paths.extend(_unsafe_paths(payload))
    blocked_paths = sorted(set(blocked_paths))

    filtered, diagnostics = filter_payload_by_client_contract(
        payload,
        contract_meta,
        stage=stage,
    )
    assert_managed_egress_safe(filtered)

    included = [
        path
        for path in safe_paths
        if _get_path(filtered, path) is not None
    ]
    omitted = [
        path
        for path in safe_paths
        if _get_path(filtered, path) is None
    ]

    result.update({
        "status": "measured",
        "reason": "contract-measurement-executed",
        "active": True,
        "included_field_names": sorted(set(included)),
        "omitted_field_names": sorted(set(omitted)),
        "blocked_field_names": blocked_paths,
        "facts": copy.deepcopy(filtered),
        "diagnostics": {
            "schema": CLIENT_CONTRACT_META_SCHEMA,
            "stage": stage,
            "filtered": bool(diagnostics.get("filtered")),
            "allowed_field_count": diagnostics.get("allowed_field_count"),
            "copied_field_count": diagnostics.get("copied_field_count"),
            "raw_payload_included": False,
        },
    })
    assert_managed_egress_safe(result)
    return result


def execute_preflight_measurement_plan(
    payload: dict[str, Any],
    contract_meta: dict[str, Any] | None,
    *,
    product_mode: ManagedProductMode | None = None,
) -> dict[str, Any]:
    return execute_measurement_plan(
        payload,
        contract_meta,
        stage="preflight",
        product_mode=product_mode,
    )


def execute_outcome_measurement_plan(
    payload: dict[str, Any],
    contract_meta: dict[str, Any] | None,
    *,
    product_mode: ManagedProductMode | None = None,
) -> dict[str, Any]:
    return execute_measurement_plan(
        payload,
        contract_meta,
        stage="outcome",
        product_mode=product_mode,
    )
