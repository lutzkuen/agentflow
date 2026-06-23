from __future__ import annotations

import asyncio
import time
import unittest
from datetime import datetime, timedelta, timezone

from tokenclaw.client_contract import (
    CLIENT_CONTRACT_SCHEMA,
    ClientContractRequest,
    clear_client_contract_cache,
    fetch_or_get_client_contract,
    filter_payload_by_client_contract,
    normalize_client_contract,
)
from tokenclaw.managed_egress import managed_egress_violations


def _future(seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _contract(**overrides):
    body = {
        "schema": CLIENT_CONTRACT_SCHEMA,
        "contract_id": "contract-fixture",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": _future(),
        "provider": "openai",
        "source_surface": "openai_responses",
        "app_family": "codex",
        "client_version_min": "0.1.0",
        "measurement_plan": {
            "preflight": [
                "input_features.text_bucket",
                {"path": "tool_features.has_tools"},
                {"field_path": "request_facts.text_chars"},
            ],
            "outcome": ["outcome_features.status_code"],
        },
        "allowed_action_families": ["routing", "crunch"],
        "local_executor_requirements": [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts": False,
            "raw_responses": False,
            "provider_bodies": False,
            "file_paths": False,
            "cache_keys": False,
        },
        "provenance": {"issuer": "tokenclaw-server"},
    }
    body.update(overrides)
    return body


class FakeContractClient:
    def __init__(self, body=None, status_code=200):
        self.body = body if body is not None else _contract()
        self.status_code = status_code
        self.calls = 0

    async def fetch(self, request):
        self.calls += 1
        return self.status_code, self.body, 3


class ClientContractTests(unittest.TestCase):
    def setUp(self):
        clear_client_contract_cache()
        self.request = ClientContractRequest(
            provider="openai",
            source_surface="openai_responses",
            app_family="codex",
            client_version="0.1.0",
        )

    def tearDown(self):
        clear_client_contract_cache()

    def test_normalizes_metadata_only_contract(self):
        contract, error = normalize_client_contract(_contract(), self.request)

        self.assertEqual(error, "")
        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract["contract_id"], "contract-fixture")
        self.assertEqual(contract["measurement_plan"]["preflight"], [
            "input_features.text_bucket",
            "request_facts.text_chars",
            "tool_features.has_tools",
        ])
        self.assertEqual(contract["allowed_action_families"], ["crunch", "routing"])
        self.assertEqual(managed_egress_violations(contract), [])

    def test_rejects_expired_or_raw_contracts(self):
        expired, expired_error = normalize_client_contract(
            _contract(expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()),
            self.request,
        )
        raw_field, raw_error = normalize_client_contract(
            _contract(measurement_plan={"preflight": ["input_features.raw_prompt"], "outcome": []}),
            self.request,
        )
        raw_privacy, privacy_error = normalize_client_contract(
            _contract(privacy={"metadata_only": True, "raw_prompts": True}),
            self.request,
        )

        self.assertIsNone(expired)
        self.assertEqual(expired_error, "expired")
        self.assertIsNone(raw_field)
        self.assertEqual(raw_error, "unsafe-field:input_features.raw_prompt")
        self.assertIsNone(raw_privacy)
        self.assertEqual(privacy_error, "privacy-not-metadata-only")

    def test_fetches_and_caches_until_expiry(self):
        client = FakeContractClient()

        first = asyncio.run(fetch_or_get_client_contract(
            self.request,
            enabled=True,
            server_url="http://127.0.0.1:4100",
            auth_configured=True,
            auth_source="loopback-unauthenticated-dev",
            client=client,
        ))
        second = asyncio.run(fetch_or_get_client_contract(
            self.request,
            enabled=True,
            server_url="http://127.0.0.1:4100",
            auth_configured=True,
            auth_source="loopback-unauthenticated-dev",
            client=client,
        ))

        self.assertEqual(first["status"], "received")
        self.assertEqual(first["cache_status"], "stored")
        self.assertEqual(second["status"], "received")
        self.assertEqual(second["cache_status"], "hit")
        self.assertEqual(client.calls, 1)

    def test_unavailable_server_falls_back_without_active_contract(self):
        client = FakeContractClient(body={"error": "missing"}, status_code=404)

        meta = asyncio.run(fetch_or_get_client_contract(
            self.request,
            enabled=True,
            server_url="http://127.0.0.1:4100",
            auth_configured=True,
            auth_source="loopback-unauthenticated-dev",
            client=client,
        ))

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "server-error")
        self.assertFalse(meta["active"])
        self.assertEqual(meta["fallback"], "local-policy")

    def test_active_contract_filters_payload_to_allowed_measurements(self):
        contract, _ = normalize_client_contract(_contract(), self.request)
        assert contract is not None
        payload = {
            "schema": "tokenclaw.policy_decision_preflight.v1",
            "source_surface": "openai_responses",
            "granularity": "provider_request",
            "app_family": "codex",
            "requested_model": "gpt-5-codex",
            "input_features": {
                "text_bucket": "2k_8k_chars",
                "input_token_bucket": "1k_4k_tokens",
                "prompt_difficulty_features": {"downgrade_risk": "block"},
            },
            "tool_features": {"has_tools": True, "tool_count": 2},
            "request_facts": {"schema": "tokenclaw.request_facts.v1", "text_chars": 2048, "tool_count": 2},
            "grouping_identifiers": {"session_id_hash": "sha256:secret"},
        }

        filtered, diagnostics = filter_payload_by_client_contract(
            payload,
            {"active": True, "contract": contract, "status": "received", "cache_status": "stored"},
            stage="preflight",
        )

        self.assertTrue(diagnostics["filtered"])
        self.assertEqual(filtered["input_features"], {"text_bucket": "2k_8k_chars"})
        self.assertEqual(filtered["tool_features"], {"has_tools": True})
        self.assertEqual(filtered["request_facts"], {"schema": "tokenclaw.request_facts.v1", "text_chars": 2048})
        self.assertNotIn("grouping_identifiers", filtered)
        self.assertEqual(managed_egress_violations(filtered), [])


if __name__ == "__main__":
    unittest.main()
