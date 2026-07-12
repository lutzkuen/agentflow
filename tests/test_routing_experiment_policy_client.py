from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.routing_experiment_policy_client import (
    ROUTING_EXPERIMENT_POLICY_SCHEMA,
    RoutingExperimentPolicyRequest,
    clear_routing_experiment_policy_cache,
    fetch_or_get_routing_experiment_policy,
    normalize_routing_experiment_policy,
)


def _future(seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _policy(**overrides):
    body = {
        "schema": ROUTING_EXPERIMENT_POLICY_SCHEMA,
        "version": "1",
        "policy_id": "routing-experiment-policy:anthropic:anthropic-messages:claude-code:v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": _future(),
        "ttl_seconds": 86400,
        "provider": "anthropic",
        "source_surface": "anthropic_messages",
        "app_family": "claude_code",
        "controls": {
            "enabled": True,
            "kill_switch": False,
            "sample_rate": 0.1,
            "holdout_rate": 0.0,
            "daily_budget_usd": 10.0,
            "min_text_chars": 0,
            "max_text_chars": 0,
        },
        "candidates": [
            {
                "candidate_id": "routing-experiment:anthropic:opus-to-sonnet:shadow",
                "requested_model": "claude-opus-4.*",
                "routed_model": "claude-sonnet-4.*",
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "sample_rate": 0.1,
                "feature_only": True,
                "locally_executed": True,
                "managed_enforced": False,
                "provider_forwarding": False,
                "server_content_processing": False,
            }
        ],
        "safety_stops": [
            {
                "stop_id": "max-error-rate",
                "metric": "error_rate",
                "threshold": 0.05,
                "comparator": "gte",
                "action": "disable_candidate",
                "reason_code": "routing-experiment-error-rate-safety-stop",
            }
        ],
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "privacy_summary": {
            "feature_only": True,
            "metadata_only": True,
            "provider_bodies_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
        },
        "provenance": {
            "schema": "tokenclaw.routing_experiment_policy_provenance.v1",
            "issuer": "tokenclaw-server",
            "server_id": "tokenclaw-server-local",
            "key_id": "default",
            "algorithm": "hmac-sha256",
            "signature": "hmac-sha256:deadbeef",
            "signed": True,
            "signing_required": True,
        },
    }
    body.update(overrides)
    return body


class FakeRoutingExperimentPolicyClient:
    def __init__(self, body=None, status_code=200):
        self.body = body if body is not None else _policy()
        self.status_code = status_code
        self.calls = 0

    async def fetch(self, request):
        self.calls += 1
        return self.status_code, self.body, 5


class RoutingExperimentPolicyClientTests(unittest.TestCase):
    def setUp(self):
        clear_routing_experiment_policy_cache()
        self.request = RoutingExperimentPolicyRequest(
            provider="anthropic",
            source_surface="anthropic_messages",
            app_family="claude_code",
        )

    def tearDown(self):
        clear_routing_experiment_policy_cache()

    def test_normalizes_signed_metadata_only_policy(self):
        policy, error = normalize_routing_experiment_policy(_policy(), self.request)

        self.assertEqual(error, "")
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual(policy["controls"]["sample_rate"], 0.1)
        self.assertEqual(len(policy["candidates"]), 1)
        self.assertEqual(managed_egress_violations(policy), [])
        # Not cryptographically verified yet (no local verification secret configured
        # in these tests) -- the client must say so explicitly, not claim success.
        self.assertFalse(policy["signature_verification"]["verified"])

    def test_rejects_unsigned_policy(self):
        unsigned = _policy(provenance={**_policy()["provenance"], "signed": False})
        policy, error = normalize_routing_experiment_policy(unsigned, self.request)
        self.assertIsNone(policy)
        self.assertEqual(error, "policy-not-signed")

    def test_rejects_expired_policy(self):
        expired = _policy(expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
        policy, error = normalize_routing_experiment_policy(expired, self.request)
        self.assertIsNone(policy)
        self.assertEqual(error, "expired")

    def test_rejects_provider_forwarding_or_server_content_processing(self):
        forwarding = _policy(provider_forwarding=True)
        policy, error = normalize_routing_experiment_policy(forwarding, self.request)
        self.assertIsNone(policy)
        self.assertEqual(error, "server-content-or-forwarding-not-allowed")

    def test_rejects_managed_enforced_candidate(self):
        bad = _policy()
        bad["candidates"][0]["managed_enforced"] = True
        policy, error = normalize_routing_experiment_policy(bad, self.request)
        self.assertIsNone(policy)
        self.assertEqual(error, "candidate-managed-enforced-not-allowed")

    def test_rejects_sample_rate_out_of_range(self):
        bad = _policy()
        bad["controls"]["sample_rate"] = 1.5
        policy, error = normalize_routing_experiment_policy(bad, self.request)
        self.assertIsNone(policy)
        self.assertEqual(error, "controls-sample-rate-out-of-range")

    def test_fetches_and_caches_until_expiry(self):
        client = FakeRoutingExperimentPolicyClient()

        first = asyncio.run(fetch_or_get_routing_experiment_policy(
            self.request,
            enabled=True,
            server_url="http://127.0.0.1:4100",
            auth_configured=True,
            auth_source="loopback-unauthenticated-dev",
            client=client,
        ))
        second = asyncio.run(fetch_or_get_routing_experiment_policy(
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

    def test_disabled_short_circuits_without_network_call(self):
        client = FakeRoutingExperimentPolicyClient()
        result = asyncio.run(fetch_or_get_routing_experiment_policy(
            self.request,
            enabled=False,
            server_url="http://127.0.0.1:4100",
            auth_configured=True,
            auth_source="loopback-unauthenticated-dev",
            client=client,
        ))
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "disabled")
        self.assertEqual(client.calls, 0)

    def test_server_error_status_reported_and_not_cached(self):
        client = FakeRoutingExperimentPolicyClient(body={"detail": "nope"}, status_code=503)
        result = asyncio.run(fetch_or_get_routing_experiment_policy(
            self.request,
            enabled=True,
            server_url="http://127.0.0.1:4100",
            auth_configured=True,
            auth_source="loopback-unauthenticated-dev",
            client=client,
        ))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "server-error")
        self.assertEqual(result["status_code"], 503)


if __name__ == "__main__":
    unittest.main()
