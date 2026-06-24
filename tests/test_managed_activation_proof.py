import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tokenclaw import cli, recommendations
from tokenclaw.client_contract import clear_client_contract_cache
from tokenclaw.managed_activation_proof import (
    MANAGED_ACTIVATION_PROOF_SCHEMA,
    managed_activation_proof_cli,
)
from tokenclaw.managed_egress import managed_egress_violations


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class FakeManagedServerClient:
    calls = []

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, params=None, headers=None):
        self.__class__.calls.append({
            "method": "GET",
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
        })
        self.assert_endpoint(url, "/v1/client-contract")
        return FakeResponse(body={
            "schema": "tokenclaw.client_contract.v1",
            "contract_id": "proof-contract",
            "generated_at": "2026-06-24T00:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "provider_bodies_included": False,
                "file_paths_included": False,
                "cache_keys_included": False,
            },
            "provider_forwarding": False,
            "server_content_processing": False,
            "measurement_plan": {
                "preflight": [
                    "input_features.api_endpoint",
                    "input_features.category",
                    "input_features.text_chars",
                    "input_features.requested_local_actions",
                    "tool_features.has_tools",
                ],
                "outcome": ["outcome_features.recent_status_bucket"],
            },
            "allowed_action_families": ["crunch"],
        })

    async def post(self, url, json=None, headers=None):
        self.__class__.calls.append({
            "method": "POST",
            "url": url,
            "json": json,
            "headers": dict(headers or {}),
        })
        if url.endswith("/v1/policy-decision"):
            return FakeResponse(body={
                "schema": "tokenclaw.policy_decision.v1",
                "decision_id": "proof-decision-1",
                "policy_id": "proof-policy",
                "confidence": 0.92,
                "provider_forwarding": False,
                "server_content_processing": False,
                "privacy_summary": {
                    "metadata_only": True,
                    "raw_payload_included": False,
                },
                "routing": {
                    "status": "recommended",
                    "target_model": "claude-sonnet-4-5-20240620",
                    "confidence": 0.92,
                    "reason_codes": ["activation-proof"],
                },
                "crunch": {
                    "status": "recommended",
                    "profile": "managed",
                    "candidate_id": "thinking-tail-compaction",
                    "traffic_treatment": "canary",
                    "canary_fraction": 0.05,
                    "thinking_tail_readiness": {
                        "schema": "agentflow.thinking_tail_readiness_summary.v1",
                        "source": "policy-decision",
                        "candidate_id": "thinking-tail-compaction",
                        "ready": True,
                        "widening_schedule": {
                            "schema": "agentflow.thinking_tail_widening_schedule.v1",
                            "candidate_id": "thinking-tail-compaction",
                            "traffic_treatment": "canary",
                            "next_fraction_cap": 0.05,
                            "holdout_fraction": 0.1,
                            "expires_at": "2099-01-01T00:00:00+00:00",
                        },
                    },
                },
            })
        if url.endswith("/v1/policy-events"):
            return FakeResponse(body={"ok": True})
        return FakeResponse(status_code=404, body={"error": "not found"}, text="not found")

    async def patch(self, url, json=None, headers=None):
        return await self.post(url, json=json, headers=headers)

    @staticmethod
    def assert_endpoint(url, suffix):
        if not url.endswith(suffix):
            raise AssertionError(f"unexpected endpoint {url!r}")


class ManagedActivationProofTests(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
        "TOKENCLAW_LOCAL_RULES_ONLY",
        "TOKENCLAW_MANAGED_ROUTING",
        "TOKENCLAW_MANAGED_CRUNCH",
        "TOKENCLAW_MANAGED_CACHE",
        "TOKENCLAW_RECOMMENDATIONS_ENABLED",
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_POLICY_DECISIONS_ENABLED",
        "TOKENCLAW_POLICY_DECISION_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_RECOMMENDATION_TIMEOUT_SECONDS",
        "TOKENCLAW_MANAGED_API_KEY",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        clear_client_contract_cache()
        FakeManagedServerClient.calls = []

    def tearDown(self):
        clear_client_contract_cache()
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_cli_emits_metadata_only_activation_proof_with_fake_managed_server(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            stdout = io.StringIO()
            with patch.object(recommendations.httpx, "AsyncClient", FakeManagedServerClient):
                code = managed_activation_proof_cli(
                    [
                        "--family", "crunch",
                        "--candidate", "thinking-tail-compaction",
                        "--mode", "dry_run",
                        "--server-url", "http://127.0.0.1:4100",
                        "--db", tmp.name,
                    ],
                    stdout=stdout,
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], MANAGED_ACTIVATION_PROOF_SCHEMA)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(set(payload["stages"]), {"mode", "drain", "contract", "decision", "local_action", "feedback"})
        self.assertEqual(payload["stages"]["mode"]["status"], "ready")
        self.assertEqual(payload["stages"]["drain"]["status"], "no-due-feedback")
        self.assertEqual(payload["stages"]["contract"]["status"], "received")
        self.assertTrue(payload["stages"]["contract"]["active"])
        self.assertEqual(payload["stages"]["decision"]["status"], "received")
        self.assertEqual(payload["stages"]["decision"]["candidate_id"], "thinking-tail-compaction")
        self.assertEqual(payload["stages"]["local_action"]["status"], "held")
        self.assertEqual(payload["stages"]["feedback"]["status"], "sent")
        self.assertTrue(payload["privacy_summary"]["metadata_only"])
        self.assertFalse(payload["privacy_summary"]["prod_port_touched"])
        self.assertEqual(managed_egress_violations(payload), [])

        urls = [call["url"] for call in FakeManagedServerClient.calls]
        self.assertIn("http://127.0.0.1:4100/v1/client-contract", urls)
        self.assertIn("http://127.0.0.1:4100/v1/policy-decision", urls)
        self.assertIn("http://127.0.0.1:4100/v1/policy-events", urls)
        policy_call = next(call for call in FakeManagedServerClient.calls if call["url"].endswith("/v1/policy-decision"))
        sent = policy_call["json"]
        self.assertEqual(sent["schema"], "tokenclaw.policy_decision_preflight.v1")
        self.assertEqual(sent["replayability_level"], "features_only")
        self.assertNotIn("body", json.dumps(sent).lower())
        self.assertNotIn("prompt", json.dumps(sent).lower())
        self.assertNotIn("file_path", json.dumps(sent).lower())

    def test_cli_fails_closed_without_managed_mode_or_server_url(self):
        stdout = io.StringIO()

        code = managed_activation_proof_cli(["--family", "crunch"], stdout=stdout)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(payload["schema"], MANAGED_ACTIVATION_PROOF_SCHEMA)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "managed-mode-or-server-url-absent")
        self.assertEqual(payload["stages"]["mode"]["status"], "blocked")
        self.assertEqual(managed_egress_violations(payload), [])

    def test_internal_cli_lists_managed_activation_proof_command(self):
        stdout = io.StringIO()

        code = cli.tokenclaw_cli(["internal", "--list"], stdout=stdout)

        self.assertEqual(code, 0)
        self.assertIn("managed-activation-proof", stdout.getvalue().splitlines())


if __name__ == "__main__":
    unittest.main()
