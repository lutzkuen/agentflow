from __future__ import annotations

import unittest

from tokenclaw.client_contract import CLIENT_CONTRACT_SCHEMA
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.managed_measurements import (
    MANAGED_MEASUREMENT_FACTS_SCHEMA,
    execute_measurement_plan,
    execute_outcome_measurement_plan,
    execute_preflight_measurement_plan,
)
from tokenclaw.managed_mode import ManagedProductMode


def _mode(
    *,
    server_calls_enabled: bool = True,
    local_rules_only: bool = False,
    reason: str = "managed-live",
) -> ManagedProductMode:
    return ManagedProductMode(
        mode="live" if server_calls_enabled else "local_only",
        configured=True,
        managed_enabled=server_calls_enabled,
        local_rules_only=local_rules_only,
        server_calls_enabled=server_calls_enabled,
        local_application_enabled=server_calls_enabled,
        family_enabled={
            "routing": server_calls_enabled,
            "crunch": server_calls_enabled,
            "cache": server_calls_enabled,
        },
        reason=reason,
    )


def _contract_meta(**overrides):
    contract = {
        "schema": CLIENT_CONTRACT_SCHEMA,
        "contract_id": "contract-measurements",
        "expires_at": "2026-06-24T00:00:00+00:00",
        "provider": "openai",
        "source_surface": "openai_responses",
        "app_family": "codex",
        "measurement_plan": {
            "preflight": [
                "source_surface",
                "requested_model",
                "input_features.api_endpoint",
                "input_features.text_chars",
                "input_features.input_token_bucket",
                "tool_features.has_tools",
                "tool_features.tool_count",
                "request_facts.endpoint",
                "request_facts.text_chars",
                "grouping_identifiers.session_id_hash",
            ],
            "outcome": [
                "outcome_features.status_code",
                "outcome_features.status_class",
                "outcome_features.latency_ms",
                "action_outcome.routing.status",
                "action_outcome.routing.applied",
            ],
        },
        "allowed_action_families": ["routing", "crunch", "cache"],
        "privacy": {"metadata_only": True},
    }
    meta = {
        "active": True,
        "status": "received",
        "reason": "fetched",
        "cache_status": "stored",
        "contract_id": contract["contract_id"],
        "contract": contract,
    }
    meta.update(overrides)
    return meta


class ManagedMeasurementTests(unittest.TestCase):
    def test_preflight_plan_copies_only_contracted_feature_fields(self):
        payload = {
            "schema": "tokenclaw.policy_decision_preflight.v1",
            "source_surface": "openai_responses",
            "granularity": "provider_request",
            "app_family": "codex",
            "requested_model": "gpt-5-codex",
            "input_features": {
                "api_endpoint": "responses",
                "text_chars": 4096,
                "input_token_bucket": "1k_4k_tokens",
                "raw_prompt": "do not send",
                "cache_key": "do-not-send",
            },
            "tool_features": {"has_tools": True, "tool_count": 2},
            "request_facts": {
                "schema": "tokenclaw.request_facts.v1",
                "endpoint": "responses",
                "text_chars": 4096,
                "session_id": "local-session-value",
            },
            "grouping_identifiers": {"session_id_hash": "sha256:abc"},
            "provider_body": {"input": "raw body"},
        }

        result = execute_preflight_measurement_plan(
            payload,
            _contract_meta(),
            product_mode=_mode(),
        )

        self.assertEqual(result["schema"], MANAGED_MEASUREMENT_FACTS_SCHEMA)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["reason"], "contract-measurement-executed")
        self.assertEqual(result["contract_id"], "contract-measurements")
        self.assertIn("input_features.text_chars", result["included_field_names"])
        self.assertIn("tool_features.has_tools", result["included_field_names"])
        self.assertIn("request_facts.endpoint", result["included_field_names"])
        self.assertIn("grouping_identifiers.session_id_hash", result["included_field_names"])
        self.assertIn("input_features.raw_prompt", result["blocked_field_names"])
        self.assertIn("input_features.cache_key", result["blocked_field_names"])
        self.assertIn("provider_body", result["blocked_field_names"])

        facts = result["facts"]
        self.assertEqual(facts["input_features"]["text_chars"], 4096)
        self.assertEqual(facts["tool_features"]["tool_count"], 2)
        self.assertEqual(facts["request_facts"]["endpoint"], "responses")
        self.assertEqual(facts["grouping_identifiers"], {"session_id_hash": "sha256:abc"})
        self.assertNotIn("raw_prompt", facts["input_features"])
        self.assertNotIn("cache_key", facts["input_features"])
        self.assertNotIn("provider_body", facts)
        self.assertEqual(managed_egress_violations(result), [])
        self.assertTrue(result["privacy"]["metadata_only"])

    def test_outcome_plan_copies_action_outcome_and_omits_missing_fields(self):
        payload = {
            "schema": "tokenclaw.managed_action_outcome_features.v1",
            "outcome_features": {
                "status_code": 200,
                "status_class": "2xx",
                "latency_ms": 1234,
            },
            "action_outcome": {
                "routing": {"status": "applied", "applied": True},
            },
        }
        meta = _contract_meta()
        meta["contract"]["measurement_plan"]["outcome"].append("outcome_features.retry_count")

        result = execute_outcome_measurement_plan(
            payload,
            meta,
            product_mode=_mode(),
        )

        self.assertEqual(result["status"], "measured")
        self.assertIn("outcome_features.status_code", result["included_field_names"])
        self.assertIn("action_outcome.routing.status", result["included_field_names"])
        self.assertIn("outcome_features.retry_count", result["omitted_field_names"])
        self.assertEqual(result["facts"]["action_outcome"]["routing"]["status"], "applied")
        self.assertEqual(managed_egress_violations(result), [])

    def test_unsafe_contract_paths_are_blocked_without_values(self):
        meta = _contract_meta()
        meta["contract"]["measurement_plan"]["preflight"].extend([
            "request_facts.session_id",
            "input_features.raw_response",
            "provider_body.messages",
        ])
        payload = {
            "schema": "tokenclaw.policy_decision_preflight.v1",
            "request_facts": {"schema": "tokenclaw.request_facts.v1", "text_chars": 10},
        }

        result = execute_measurement_plan(payload, meta, stage="preflight", product_mode=_mode())

        self.assertEqual(result["status"], "measured")
        self.assertIn("request_facts.session_id", result["blocked_field_names"])
        self.assertIn("input_features.raw_response", result["blocked_field_names"])
        self.assertIn("provider_body.messages", result["blocked_field_names"])
        self.assertNotIn("session_id", result["facts"].get("request_facts", {}))
        self.assertEqual(managed_egress_violations(result), [])

    def test_managed_off_local_rules_only_and_server_unavailable_skip(self):
        payload = {"schema": "tokenclaw.policy_decision_preflight.v1"}
        disabled = execute_preflight_measurement_plan(
            payload,
            _contract_meta(),
            product_mode=_mode(server_calls_enabled=False, reason="managed-not-enabled"),
        )
        local_only = execute_preflight_measurement_plan(
            payload,
            _contract_meta(),
            product_mode=_mode(server_calls_enabled=False, local_rules_only=True, reason="local-rules-only"),
        )
        server_unavailable = execute_preflight_measurement_plan(
            payload,
            {
                "active": False,
                "status": "error",
                "reason": "unreachable",
                "cache_status": "miss",
            },
            product_mode=_mode(),
        )
        expired = execute_preflight_measurement_plan(
            payload,
            {
                "active": False,
                "status": "invalid",
                "reason": "invalid-contract",
                "schema_error": "expired",
            },
            product_mode=_mode(),
        )

        self.assertEqual(disabled["status"], "skipped")
        self.assertEqual(disabled["reason"], "managed-not-enabled")
        self.assertEqual(local_only["reason"], "local-rules-only")
        self.assertEqual(server_unavailable["reason"], "server-unavailable")
        self.assertEqual(expired["reason"], "expired-contract")
        for result in (disabled, local_only, server_unavailable, expired):
            self.assertEqual(result["facts"], {})
            self.assertEqual(result["fallback"], "local-policy")
            self.assertEqual(managed_egress_violations(result), [])


if __name__ == "__main__":
    unittest.main()
