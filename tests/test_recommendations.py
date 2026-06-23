import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from tokenclaw import recommendations
from tokenclaw.crunch import crunch_body
from tokenclaw.optimization import openai_features
from tokenclaw.prompt_features import prompt_difficulty_features_from_text


class FakeResponse:
    def __init__(self, status_code=200, body=None, text="", json_error=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self._body


class FakeAsyncClient:
    response = FakeResponse()
    error = None
    last_url = None
    last_json = None
    last_timeout = None
    last_headers = None

    def __init__(self, timeout):
        self.__class__.last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json, headers=None):
        self.__class__.last_url = url
        self.__class__.last_json = json
        self.__class__.last_headers = headers
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.response

    async def patch(self, url, json, headers=None):
        self.__class__.last_url = url
        self.__class__.last_json = json
        self.__class__.last_headers = headers
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.response


class _NoQueueStore:
    pass


class RecommendationTest(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_RECOMMENDATIONS_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_RECOMMENDATION_TIMEOUT_SECONDS",
        "TOKENCLAW_RECOMMENDATION_FAILURE_MODE",
        "TOKENCLAW_MANAGED_API_KEY",
        "TOKENCLAW_POLICY_DECISION_ENABLED",
        "TOKENCLAW_POLICY_DECISIONS_ENABLED",
        "TOKENCLAW_POLICY_DECISION_MIN_CONFIDENCE",
        "TOKENCLAW_POLICY_DECISION_CANARY_FRACTION",
        "TOKENCLAW_POLICY_DECISION_CANARY_SALT",
        "TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS",
        "TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        FakeAsyncClient.response = FakeResponse()
        FakeAsyncClient.error = None
        FakeAsyncClient.last_url = None
        FakeAsyncClient.last_json = None
        FakeAsyncClient.last_timeout = None
        FakeAsyncClient.last_headers = None

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _keys_in(self, value):
        keys = set()
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(str(key).lower())
                keys.update(self._keys_in(item))
        elif isinstance(value, list):
            for item in value:
                keys.update(self._keys_in(item))
        return keys

    def _assert_no_sensitive_strings(self, value):
        text = str(value)
        for secret in (
            "raw provider body",
            "raw provider response",
            "raw request body",
            "raw command text",
            "raw transcript text",
            "raw pattern text",
            "tenant-secret",
            "api-key-secret",
            "session-secret",
            "thread-secret",
            "request-secret",
            "arbitrary payload string",
        ):
            self.assertNotIn(secret, text)

    def test_disabled_path_skips_server_and_falls_back_to_local_policy(self):
        unit = {"requested_model": "claude-sonnet-4-6"}

        meta = asyncio.run(recommendations.fetch_recommendation(unit))

        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "disabled")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertIsNone(FakeAsyncClient.last_url)

    def test_request_facts_envelope_is_thin_feature_only_metadata(self):
        envelope = recommendations.build_request_facts_envelope(
            provider="unknown-provider",
            path=None,
            body={
                "messages": [{"role": "user", "content": "raw prompt should stay local"}],
                "metadata": {"session_id": "session-secret", "api_key": "api-key-secret"},
                "stream": True,
            },
            requested_model=None,
            session_id="session-secret",
            thread_id="thread-secret",
            request_id="request-secret",
        )

        facts = envelope["request_facts"]
        self.assertEqual(envelope["schema"], recommendations.REQUEST_FACTS_ENVELOPE_SCHEMA)
        self.assertEqual(facts["schema"], recommendations.REQUEST_FACTS_SCHEMA)
        self.assertEqual(facts["provider_family"], "unknown")
        self.assertEqual(facts["endpoint"], "unknown")
        self.assertIsNone(facts["requested_model"])
        self.assertFalse(facts["requested_model_present"])
        self.assertTrue(facts["stream"])
        self.assertEqual(facts["message_item_count"], 1)
        self.assertIn("session_id_hash", facts["grouping_identifiers"])
        self.assertIn("thread_id_hash", facts["grouping_identifiers"])
        self.assertIn("request_id_hash", facts["grouping_identifiers"])
        self.assertTrue(envelope["privacy_summary"]["metadata_only"])
        self.assertNotIn("candidate_target_model", self._keys_in(envelope))
        self.assertNotIn("source_surface", self._keys_in(envelope))
        self.assertNotIn("app_family", self._keys_in(envelope))
        self.assertNotIn("messages", self._keys_in(envelope))
        self.assertNotIn("content", self._keys_in(envelope))
        self.assertNotIn("request_body", self._keys_in(envelope))
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(envelope)))
        self._assert_no_sensitive_strings(envelope)
        recommendations.assert_managed_egress_safe(envelope)

    def test_policy_decision_fetch_can_use_request_facts_instead_of_rich_unit(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.policy_decision.v1",
            "policy_id": "facts-policy",
            "confidence": 0.88,
            "provider_forwarding": False,
            "server_content_processing": False,
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            "routing": {
                "status": "recommended",
                "target_model": "claude-haiku-4-5-20251001",
                "confidence": 0.88,
                "reason_codes": ["request-facts-route"],
            },
        })
        envelope = recommendations.build_request_facts_envelope(
            provider="anthropic",
            path="/v1/messages",
            body={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "raw prompt should stay local"}],
                "tools": [{"name": "lookup", "description": "raw tool description"}],
            },
            requested_model="claude-sonnet-4-6",
            stream=False,
            input_tokens_est=120,
            session_id="session-secret",
        )
        rich_unit = {
            "source_surface": "server-side-rich-unit-would-be-ignored",
            "app_family": "server-side-app",
            "candidate_target_model": "claude-haiku-4-5-20251001",
            "input_features": {
                "local_routing_reason": "local interpretation should not be posted",
                "pattern_hash": "sha256:local-rich-pattern",
                "messages": "raw prompt should not be posted",
            },
        }

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_policy_decision(rich_unit, request_facts=envelope))

        sent = FakeAsyncClient.last_json
        self.assertEqual(meta["status"], "received")
        self.assertEqual(sent["schema"], recommendations.POLICY_DECISION_PREFLIGHT_SCHEMA)
        self.assertEqual(sent["feature_schema_version"], recommendations.REQUEST_FACTS_SCHEMA)
        self.assertEqual(sent["source_surface"], "anthropic_messages")
        self.assertEqual(sent["app_family"], "claude_code")
        self.assertIsNone(sent["candidate_target_model"])
        self.assertEqual(sent["request_facts"]["schema"], recommendations.REQUEST_FACTS_SCHEMA)
        self.assertTrue(sent["input_features"]["thin_request_facts"])
        self.assertEqual(sent["input_features"]["api_endpoint"], "messages")
        self.assertEqual(sent["tool_features"]["tool_count"], 1)
        self.assertIn("session_id_hash", sent["grouping_identifiers"])
        self.assertNotIn("local_routing_reason", self._keys_in(sent))
        self.assertNotIn("pattern_hash", self._keys_in(sent))
        self.assertNotIn("messages", self._keys_in(sent))
        self.assertNotIn("content", self._keys_in(sent))
        self._assert_no_sensitive_strings(sent)
        recommendations.assert_managed_egress_safe(sent)

    def test_policy_decision_fetch_filters_payload_with_active_client_contract(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.policy_decision.v1",
            "policy_id": "contract-filtered-policy",
            "confidence": 0.86,
            "provider_forwarding": False,
            "server_content_processing": False,
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            "routing": {
                "status": "recommended",
                "target_model": "gpt-5-mini",
                "confidence": 0.86,
                "reason_codes": ["contract-filtered"],
            },
        })
        unit = openai_features.build_openai_preflight_feature_unit(
            body={
                "model": "gpt-5-codex",
                "input": "raw openai prompt should stay local",
                "stream": False,
            },
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routing_meta={
                "text_chars": 4096,
                "has_tools": False,
                "category": "chat",
                "workflow_phase": "chat",
            },
            category="chat",
            stream=False,
            input_tokens_est=1024,
        )
        contract_meta = {
            "status": "received",
            "reason": "fetched",
            "cache_status": "stored",
            "active": True,
            "contract": {
                "schema": "tokenclaw.client_contract.v1",
                "contract_id": "contract-filter",
                "expires_at": "2999-01-01T00:00:00+00:00",
                "provider": "openai",
                "source_surface": "openai_responses",
                "app_family": "codex",
                "measurement_plan": {
                    "preflight": [
                        "input_features.text_bucket",
                        "input_features.input_token_bucket",
                        "tool_features.has_tools",
                    ],
                    "outcome": [],
                },
                "allowed_action_families": ["routing"],
                "privacy": {"metadata_only": True},
            },
        }

        async def fake_contract(_payload):
            return contract_meta

        with patch.object(recommendations, "_client_contract_for_payload", fake_contract):
            with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
                meta = asyncio.run(recommendations.fetch_policy_decision(unit))

        sent = FakeAsyncClient.last_json
        self.assertEqual(meta["status"], "received")
        self.assertEqual(meta["client_contract"]["status"], "active")
        self.assertTrue(meta["client_contract"]["filtered"])
        self.assertEqual(sent["source_surface"], "openai_responses")
        self.assertEqual(
            sent["input_features"],
            {"text_bucket": "2k_8k_chars", "input_token_bucket": "1k_4k_tokens"},
        )
        self.assertEqual(sent["tool_features"], {"has_tools": False})
        self.assertNotIn("prompt_difficulty_features", self._keys_in(sent))
        self.assertNotIn("terminal_log_features", self._keys_in(sent))
        self.assertNotIn("pattern_hash", self._keys_in(sent))
        self._assert_no_sensitive_strings(sent)
        recommendations.assert_managed_egress_safe(sent)

    def test_success_path_posts_feature_unit_with_auth_and_applies_target_model(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        os.environ["TOKENCLAW_RECOMMENDATION_TIMEOUT_SECONDS"] = "0.25"
        os.environ["TOKENCLAW_MANAGED_API_KEY"] = "managed-secret"
        FakeAsyncClient.response = FakeResponse(body={
            "target_model": "claude-haiku-4-5-20251001",
            "replacement_prompt": None,
            "confidence": 0.82,
            "policy_id": "policy-1",
            "reason": "route cheaper",
            "optimization_unit_id": 42,
        })
        routing_meta = {
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-sonnet-4-6",
            "reason": "keep requested model",
            "text_chars": 1200,
            "has_tools": False,
            "category": "chat",
            "workflow_phase": "verification",
            "workflow_phase_reason": "verification-intent-text",
            "workflow_phase_confidence": "medium",
            "policy_source": "local-default",
            "command": "raw command text",
            "tenant_id": "tenant-secret",
            "prompt_difficulty_features": prompt_difficulty_features_from_text(
                "Find current outstanding vouchers in the database."
            ),
        }
        unit = recommendations.build_optimization_unit(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            routing_meta=routing_meta,
            crunch_meta={"changed": False, "pattern_text": "raw pattern text"},
            cache_meta={"status": "miss", "reason": "exact-miss", "request_body": "raw request body"},
            category="chat",
            stream=False,
            input_tokens_est=300,
            session_id="session-secret",
        )

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation(unit))
        body = {"model": "claude-sonnet-4-6"}
        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=meta,
        )

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/recommendation")
        self.assertEqual(FakeAsyncClient.last_timeout, 0.25)
        self.assertEqual(FakeAsyncClient.last_headers["authorization"], "Bearer managed-secret")
        self.assertEqual(FakeAsyncClient.last_json["replayability_level"], "features_only")
        self.assertEqual(
            FakeAsyncClient.last_json["feature_schema_version"],
            recommendations.FEATURE_SCHEMA_VERSION,
        )
        self.assertEqual(FakeAsyncClient.last_json["candidate_target_model"], "claude-sonnet-4-6")
        pattern_features = FakeAsyncClient.last_json["input_features"]["pattern_features"]
        self.assertEqual(FakeAsyncClient.last_json["pattern_features"], pattern_features)
        self.assertEqual(pattern_features["schema"], "tokenclaw.pattern_features.v1")
        self.assertEqual(pattern_features["hash_basis"], "normalized-structure-and-size-buckets")
        self.assertEqual(pattern_features["text_bucket"], "lt_2k_chars")
        self.assertEqual(pattern_features["token_bucket"], "lt_1k_tokens")
        self.assertEqual(pattern_features["workflow_phase"], "verification")
        self.assertEqual(FakeAsyncClient.last_json["input_features"]["workflow_phase"], "verification")
        self.assertEqual(FakeAsyncClient.last_json["outcome_features"]["routing_outcome_label"], "unknown")
        self.assertEqual(
            FakeAsyncClient.last_json["input_features"]["prompt_difficulty_features"]["downgrade_risk"],
            "block",
        )
        self.assertEqual(
            FakeAsyncClient.last_json["input_features"]["prompt_difficulty_features"]["external_source_dependency"],
            "database",
        )
        self.assertEqual(
            FakeAsyncClient.last_json["input_features"]["workflow_phase_reason"],
            "verification-intent-text",
        )
        self.assertEqual(FakeAsyncClient.last_json["tool_features"]["workflow_phase"], "verification")
        self.assertEqual(pattern_features["local_decision_status"], "routing:skipped|crunch:skipped|cache:miss")
        self.assertTrue(pattern_features["pattern_hash"].startswith("sha256:"))
        self.assertEqual(pattern_features["pattern_hash"], pattern_features["normalized_pattern_hash"])
        self.assertTrue(pattern_features["crunch_pattern_hash"].startswith("sha256:"))
        self.assertTrue(pattern_features["cache_pattern_hash"].startswith("sha256:"))
        self.assertEqual(FakeAsyncClient.last_json["input_features"]["pattern_hash"], pattern_features["pattern_hash"])
        self.assertEqual(
            FakeAsyncClient.last_json["input_features"]["crunch_pattern_hash"],
            pattern_features["crunch_pattern_hash"],
        )
        self.assertEqual(
            FakeAsyncClient.last_json["input_features"]["cache_pattern_hash"],
            pattern_features["cache_pattern_hash"],
        )
        diagnostics = recommendations.pattern_feature_diagnostics(FakeAsyncClient.last_json)
        self.assertTrue(diagnostics["present"])
        self.assertEqual(diagnostics["pattern_hash_count"], 3)
        self.assertFalse(diagnostics["raw_pattern_strings_included"])
        self.assertIn("session_id_hash", FakeAsyncClient.last_json["grouping_identifiers"])
        self.assertTrue(FakeAsyncClient.last_json["grouping_identifiers"]["session_id_hash"].startswith("sha256:"))
        self.assertNotIn("session-secret", str(FakeAsyncClient.last_json))
        self._assert_no_sensitive_strings(FakeAsyncClient.last_json)
        self.assertTrue(FakeAsyncClient.last_json["privacy_summary"]["metadata_only"])
        self.assertFalse(FakeAsyncClient.last_json["privacy_summary"]["raw_body_storage"])
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(FakeAsyncClient.last_json)))
        self.assertEqual(meta["status"], "received")
        self.assertTrue(meta["auth_configured"])
        self.assertFalse(meta["api_key_value_included"])
        self.assertEqual(meta["policy_source"], "managed-recommended")
        self.assertEqual(meta["optimization_unit_id"], 42)
        self.assertFalse(meta["replacement_prompt_present"])
        self.assertTrue(applied["applied"])
        self.assertTrue(applied["changed_model"])
        self.assertEqual(body["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(routing_meta["final_policy_source"], "managed-recommended")
        self.assertEqual(routing_meta["managed_policy_id"], "policy-1")

    def test_optimization_unit_can_include_finalized_routing_outcome_label(self):
        unit = recommendations.build_optimization_unit(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            routing_meta={
                "reason": "test",
                "text_chars": 1200,
                "has_tools": False,
                "category": "chat",
                "routing_outcome_label": "safe",
            },
            crunch_meta={"changed": False},
            cache_meta={"status": "miss", "reason": "exact-miss"},
            category="chat",
            stream=False,
            input_tokens_est=300,
            session_id="session-secret",
        )

        self.assertEqual(unit["outcome_features"]["routing_outcome_label"], "safe")
        self._assert_no_sensitive_strings(unit)

    def test_anthropic_local_pattern_diagnostics_do_not_require_managed_server(self):
        raw_secret = "raw anthropic tool payload secret"
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": f"tool results:\nstdout: ok\nstdout: ok\n{raw_secret}",
                        }
                    ],
                }
            ],
        }
        crunched, crunch_meta = crunch_body(body)
        routing_meta = {
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-sonnet-4-6",
            "reason": "keep requested model",
            "text_chars": len(str(crunched)),
            "has_tools": True,
            "category": "tool-result",
            "workflow_phase": "tool-result",
            "policy_source": "local-default",
        }

        unit = recommendations.build_optimization_unit(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta={"status": "skipped", "reason": "tools-disabled"},
            category="tool-result",
            stream=False,
            input_tokens_est=120,
            session_id="session-secret",
        )
        diagnostics = recommendations.pattern_feature_diagnostics(unit)

        self.assertTrue(diagnostics["present"])
        self.assertGreaterEqual(diagnostics["pattern_hash_count"], 3)
        self.assertIn("tool_results", diagnostics["local_pattern_module_families"])
        self.assertFalse(diagnostics["raw_pattern_strings_included"])
        self.assertTrue(diagnostics["pattern_hash"].startswith("sha256:"))
        self.assertTrue(diagnostics["crunch_pattern_hash"].startswith("sha256:"))
        self.assertTrue(diagnostics["cache_pattern_hash"].startswith("sha256:"))
        rendered = json.dumps(diagnostics, sort_keys=True)
        self.assertNotIn(raw_secret, rendered)
        self.assertNotIn("session-secret", json.dumps(unit, sort_keys=True))

    def test_codex_turn_feature_unit_emits_pattern_features_without_raw_payloads(self):
        unit = recommendations.build_codex_turn_optimization_unit(
            method="turn/start",
            request_id_present=True,
            thread_id_present=True,
            params_chars=2048,
            input_items=2,
            input_text_chars=12000,
            routing_meta={
                "status": "skipped",
                "reason": "summary",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "command": "raw command text",
                "api_key": "api-key-secret",
            },
            crunch_meta={
                "status": "applied",
                "changed": True,
                "saved_chars": 4096,
                "codex_patterns": [
                    {
                        "type": "repeated_input_section",
                        "count": 3,
                        "saved_chars_est": 4096,
                        "hashes": ["raw-content-hash-should-not-define-pattern"],
                    }
                ],
                "transcript": "raw transcript text",
            },
            cache_meta={
                "status": "skipped",
                "reason": "codex-app-cache-disabled",
                "eligible": True,
                "replayability_level": "local-exact-response",
                "payload": "arbitrary payload string",
            },
            request_id="request-secret",
            thread_id="thread-secret",
            prompt_difficulty_features=prompt_difficulty_features_from_text(
                "Thank you for checking the vouchers."
            ),
        )

        pattern_features = unit["input_features"]["pattern_features"]

        self.assertEqual(pattern_features["workflow_phase"], "summary")
        self.assertEqual(pattern_features["text_bucket"], "8k_32k_chars")
        self.assertEqual(pattern_features["token_bucket"], "1k_4k_tokens")
        self.assertEqual(pattern_features["pattern_types"], ["repeated_input_section"])
        self.assertEqual(pattern_features["local_decision_status"], "routing:skipped|crunch:applied|cache:skipped")
        self.assertEqual(
            unit["input_features"]["prompt_difficulty_features"]["task_intent"],
            "acknowledgement",
        )
        self.assertEqual(unit["input_features"]["prompt_difficulty_features"]["downgrade_risk"], "safe")
        self.assertTrue(pattern_features["pattern_hash"].startswith("sha256:"))
        self.assertTrue(pattern_features["crunch_pattern_hash"].startswith("sha256:"))
        self.assertTrue(pattern_features["cache_pattern_hash"].startswith("sha256:"))
        self.assertIn("request_id_hash", unit["grouping_identifiers"])
        self.assertIn("thread_id_hash", unit["grouping_identifiers"])
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(unit)))
        self._assert_no_sensitive_strings(unit)
        self.assertNotIn("raw-content-hash-should-not-define-pattern", str(unit))

    def test_valid_noop_recommendation_records_received_without_changing_model(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(body={
            "target_model": "claude-sonnet-4-6",
            "replacement_prompt": None,
            "confidence": 0.9,
            "policy_id": "baseline-pass-through",
            "reason": "local model is already safest",
            "optimization_unit_id": 123,
        })
        routing_meta = {
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-sonnet-4-6",
            "reason": "keep requested model",
            "policy_source": "local-default",
        }

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))
        body = {"model": "claude-sonnet-4-6"}
        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=meta,
        )

        self.assertEqual(meta["status"], "received")
        self.assertTrue(applied["applied"])
        self.assertFalse(applied["changed_model"])
        self.assertEqual(body["model"], "claude-sonnet-4-6")

    def test_empty_server_url_is_not_configured_and_does_not_call_network(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = ""

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "server-url-not-configured")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertIsNone(FakeAsyncClient.last_url)

    def test_recommendation_egress_guard_blocks_raw_payload_before_network(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        unsafe_unit = {
            "feature_schema_version": recommendations.FEATURE_SCHEMA_VERSION,
            "requested_model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "raw prompt must stay local"}],
            "input": "raw OpenAI Responses input must stay local",
            "tool_payload": {"arguments": "raw tool args must stay local"},
            "file_path": "/home/lutz/private/project/app.py",
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation(unsafe_unit))

        self.assertIsNone(FakeAsyncClient.last_url)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "unsafe-egress-payload")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertFalse(meta["applied"])
        self.assertTrue(meta["egress_guard"]["blocked"])
        self.assertFalse(meta["egress_guard"]["raw_values_logged"])
        self.assertIn("messages", meta["egress_guard"]["blocked_keys"])
        self.assertIn("input", meta["egress_guard"]["blocked_keys"])
        self.assertIn("tool_payload", meta["egress_guard"]["blocked_keys"])
        self.assertIn("file_path", meta["egress_guard"]["blocked_keys"])
        self.assertNotIn("raw prompt must stay local", str(meta))
        self.assertNotIn("/home/lutz/private/project/app.py", str(meta))

    def test_timeout_records_bounded_fallback_metadata(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        os.environ["TOKENCLAW_RECOMMENDATION_TIMEOUT_SECONDS"] = "0.1"
        FakeAsyncClient.error = httpx.TimeoutException("too slow")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(FakeAsyncClient.last_timeout, 0.1)
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "timeout")
        self.assertEqual(meta["fallback"], "local-policy")

    def test_unreachable_records_fallback_metadata(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.error = httpx.ConnectError("connection refused")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "unreachable")
        self.assertEqual(meta["fallback"], "local-policy")

    def test_invalid_json_and_schema_record_fallback_metadata(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(json_error=ValueError("not json"))

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            invalid_json = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        FakeAsyncClient.response = FakeResponse(body={"target_model": "claude-haiku-4-5-20251001"})
        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            invalid_schema = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(invalid_json["status"], "invalid")
        self.assertEqual(invalid_json["reason"], "invalid-json")
        self.assertEqual(invalid_json["fallback"], "local-policy")
        self.assertEqual(invalid_schema["status"], "invalid")
        self.assertEqual(invalid_schema["reason"], "invalid-schema")
        self.assertIn("schema_error", invalid_schema)
        self.assertEqual(invalid_schema["fallback"], "local-policy")

    def test_policy_decision_fetch_posts_strict_feature_snapshot_and_preserves_predictor_fields(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.policy_decision.v1",
            "optimization_unit_id": 99,
            "policy_id": "feature-policy-decision:anthropic:99",
            "confidence": 0.91,
            "provider_forwarding": False,
            "server_content_processing": False,
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            "routing": {
                "status": "recommended",
                "target_model": "claude-haiku-4-5-20251001",
                "policy_id": "routing-predictor-policy",
                "confidence": 0.91,
                "route_down_probability": 0.93,
                "recommended_mode": "shadow",
                "model_artifact_version": "routing-predictor-v1-abcd",
                "model_evidence_hash": "sha256:evidence",
                "predictor_rule_id": "routing-evidence:anthropic:sonnet->haiku",
                "required_local_gates": ["local-shadow-gate"],
                "reason_codes": ["active-routing-predictor-model"],
            },
        })
        unit = {
            "schema": "tokenclaw.openai_preflight_feature_unit.v1",
            "feature_schema_version": recommendations.FEATURE_SCHEMA_VERSION,
            "source_surface": "anthropic_messages",
            "granularity": "provider_request",
            "app_family": "claude_code",
            "requested_model": "claude-sonnet-4-6",
            "candidate_target_model": "claude-haiku-4-5-20251001",
            "input_features": {
                "path": "/v1/messages",
                "category": "tool-result",
                "text_chars": 6200,
                "input_tokens_est": 1550,
            },
            "tool_features": {"has_tools": True},
            "grouping_identifiers": {"session_id_hash": "sha256:session"},
            "privacy_summary": {"metadata_only": True},
            "pattern_features": {"raw_pattern_strings_included": False},
            "replayability_level": "features_only",
        }

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_policy_decision(unit))

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/policy-decision")
        self.assertNotIn("authorization", FakeAsyncClient.last_headers)
        sent = FakeAsyncClient.last_json
        self.assertEqual(sent["schema"], "tokenclaw.policy_decision_preflight.v1")
        self.assertEqual(sent["replayability_level"], "features_only")
        self.assertEqual(sent["input_features"]["api_endpoint"], "v1_messages")
        self.assertNotIn("path", sent["input_features"])
        self.assertNotIn("privacy_summary", sent)
        self.assertNotIn("pattern_features", sent)
        self.assertEqual(meta["status"], "received")
        self.assertTrue(meta["enabled"])
        self.assertTrue(meta["auth_configured"])
        self.assertEqual(meta["auth_source"], "loopback-unauthenticated-dev")
        self.assertTrue(meta["loopback_unauthenticated_allowed"])
        self.assertEqual(meta["policy_decision_schema"], "tokenclaw.policy_decision.v1")
        self.assertEqual(meta["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(meta["recommended_mode"], "shadow")
        self.assertEqual(meta["route_down_probability"], 0.93)
        self.assertEqual(meta["model_artifact_version"], "routing-predictor-v1-abcd")
        self.assertEqual(meta["predictor_rule_id"], "routing-evidence:anthropic:sonnet->haiku")

    def test_policy_decision_fetch_omits_openai_non_feature_input_diagnostics(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.policy_decision.v1",
            "policy_id": "feature-policy-decision:openai:1",
            "confidence": 0.91,
            "provider_forwarding": False,
            "server_content_processing": False,
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            "routing": {
                "status": "recommended",
                "target_model": "gpt-5.4-mini",
                "confidence": 0.91,
                "route_down_probability": 0.91,
                "recommended_mode": "shadow",
                "model_artifact_version": "routing-predictor-v1-openai",
                "reason_codes": ["active-routing-predictor-model"],
            },
        })
        unit = openai_features.build_openai_preflight_feature_unit(
            body={
                "model": "gpt-5.4",
                "input": [
                    {"type": "message", "content": "older raw OpenAI context"},
                    {"type": "message", "content": "older raw OpenAI context two"},
                    {"type": "message", "content": "older raw OpenAI context three"},
                    {"type": "message", "content": "older raw OpenAI context four"},
                    {"type": "message", "content": "current request"},
                ],
                "stream": False,
            },
            path="/v1/responses",
            requested_model="gpt-5.4",
            routing_meta={
                "text_chars": 4096,
                "has_tools": False,
                "category": "chat",
                "workflow_phase": "chat",
            },
            category="chat",
            stream=False,
            input_tokens_est=1024,
        )
        self.assertIn("old_context", unit["input_features"])

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_policy_decision(unit))

        sent = FakeAsyncClient.last_json
        self.assertEqual(meta["status"], "received")
        self.assertEqual(sent["source_surface"], "openai_responses")
        self.assertEqual(sent["input_features"]["api_endpoint"], "responses")
        self.assertNotIn("old_context", self._keys_in(sent["input_features"]))
        self.assertNotIn("input", self._keys_in(sent["input_features"]))
        self.assertNotIn("messages", self._keys_in(sent["input_features"]))
        self.assertNotIn("prompt", self._keys_in(sent["input_features"]))
        self.assertIn("terminal_log_features", sent["input_features"])
        self.assertIn("prompt_difficulty_features", sent["input_features"])

    def test_policy_decision_fetch_fails_closed_on_timeout_and_schema_mismatch(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.error = httpx.TimeoutException("too slow")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            timeout_meta = asyncio.run(recommendations.fetch_policy_decision({
                "source_surface": "anthropic_messages",
                "granularity": "provider_request",
                "app_family": "claude_code",
                "requested_model": "claude-sonnet-4-6",
                "input_features": {},
                "tool_features": {},
                "replayability_level": "features_only",
            }))

        FakeAsyncClient.error = None
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.unexpected.v1",
            "routing": {"status": "recommended", "target_model": "claude-haiku-4-5-20251001"},
        })
        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            schema_meta = asyncio.run(recommendations.fetch_policy_decision({
                "source_surface": "anthropic_messages",
                "granularity": "provider_request",
                "app_family": "claude_code",
                "requested_model": "claude-sonnet-4-6",
                "input_features": {},
                "tool_features": {},
                "replayability_level": "features_only",
            }))

        self.assertEqual(timeout_meta["status"], "error")
        self.assertEqual(timeout_meta["reason"], "timeout")
        self.assertEqual(timeout_meta["fallback"], "local-policy")
        self.assertEqual(schema_meta["status"], "invalid")
        self.assertEqual(schema_meta["reason"], "invalid-schema")
        self.assertEqual(schema_meta["schema_error"], "schema-mismatch")
        self.assertEqual(schema_meta["fallback"], "local-policy")

    def test_policy_decision_enabled_without_server_url_fails_closed_locally(self):
        os.environ["TOKENCLAW_RECOMMENDATIONS_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISIONS_ENABLED"] = "1"

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_policy_decision({
                "source_surface": "anthropic_messages",
                "granularity": "provider_request",
                "app_family": "claude_code",
                "requested_model": "claude-sonnet-4-6",
                "input_features": {},
                "tool_features": {},
                "replayability_level": "features_only",
            }))

        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "server-url-not-configured")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertFalse(meta["applied"])
        self.assertIsNone(FakeAsyncClient.last_url)

    def test_policy_decision_normalizes_route_to_and_applies_after_local_gate(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.policy_decision.v1",
            "policy_id": "feature-policy-decision:anthropic:route-to",
            "confidence": 0.94,
            "route_to": "claude-haiku-4-5-20251001",
            "provider_forwarding": False,
            "server_content_processing": False,
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            "routing": {
                "status": "recommended",
                "confidence": 0.94,
                "route_down_probability": 0.91,
                "recommended_mode": "shadow",
                "model_artifact_version": "routing-predictor-v1-route-to",
                "reason_codes": ["active-routing-predictor-model"],
            },
        })

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_policy_decision({
                "source_surface": "anthropic_messages",
                "granularity": "provider_request",
                "app_family": "claude_code",
                "requested_model": "claude-sonnet-4-6",
                "input_features": {},
                "tool_features": {},
                "replayability_level": "features_only",
            }))

        body = {"model": "claude-sonnet-4-6"}
        routing_meta = {"routed_model": "claude-sonnet-4-6", "reason": "local route"}
        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=meta,
        )

        self.assertEqual(meta["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(meta["route_to"], "claude-haiku-4-5-20251001")
        self.assertTrue(meta["route_to_present"])
        self.assertTrue(applied["applied"])
        self.assertTrue(applied["changed_model"])
        self.assertEqual(applied["apply_reason"], "route-to-local-safety-gate-passed")
        self.assertEqual(applied["local_action_taken"], "route_to")
        self.assertEqual(body["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(routing_meta["managed_route_recommended_mode"], "route_to")
        self.assertEqual(routing_meta["final_policy_source"], "managed-recommended")

    def test_policy_decision_application_observes_shadow_and_canary_gates_before_mutating(self):
        base_meta = {
            "status": "received",
            "policy_decision_schema": "tokenclaw.policy_decision.v1",
            "routing_status": "recommended",
            "target_model": "claude-haiku-4-5-20251001",
            "confidence": 0.92,
            "route_down_probability": 0.9,
            "model_artifact_version": "routing-predictor-v1-abcd",
            "recommended_mode": "shadow",
            "policy_id": "routing-policy",
        }
        body = {"model": "claude-sonnet-4-6"}
        routing_meta = {"routed_model": "claude-sonnet-4-6", "reason": "local route"}

        observed = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=base_meta,
        )

        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertFalse(observed["applied"])
        self.assertEqual(observed["local_action_taken"], "shadow")
        self.assertEqual(observed["would_route_model"], "claude-haiku-4-5-20251001")

        missing_version = dict(base_meta, recommended_mode="canary", model_artifact_version=None)
        skipped = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=missing_version,
        )
        self.assertEqual(skipped["apply_reason"], "routing-predictor-model-version-missing")
        self.assertEqual(body["model"], "claude-sonnet-4-6")

        os.environ["TOKENCLAW_POLICY_DECISION_CANARY_FRACTION"] = "1"
        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=dict(base_meta, recommended_mode="canary"),
        )

        self.assertTrue(applied["applied"])
        self.assertEqual(applied["local_action_taken"], "canary_applied")
        self.assertEqual(body["model"], "claude-haiku-4-5-20251001")

    def test_policy_decision_plural_env_and_routing_action_apply_live_route(self):
        os.environ["TOKENCLAW_RECOMMENDATIONS_ENABLED"] = "1"
        os.environ["TOKENCLAW_POLICY_DECISIONS_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        os.environ["TOKENCLAW_POLICY_DECISION_MIN_CONFIDENCE"] = "0.8"
        FakeAsyncClient.response = FakeResponse(body={
            "schema": "tokenclaw.policy_decision.v1",
            "policy_id": "managed-route-1",
            "provider_forwarding": False,
            "server_content_processing": False,
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            "routing_action": {
                "recommended": True,
                "target_model": "claude-haiku-4-5-20251001",
                "confidence": 0.91,
                "reason": "lossless downgrade predictor",
            },
        })

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_policy_decision({
                "source_surface": "anthropic_messages",
                "granularity": "provider_request",
                "app_family": "claude_code",
                "requested_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "input_features": {"category": "tool-result"},
                "tool_features": {"has_tools": True},
                "replayability_level": "features_only",
            }))

        body = {"model": "claude-sonnet-4-6"}
        routing_meta = {"routed_model": "claude-sonnet-4-6", "reason": "local route"}
        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=meta,
        )

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/policy-decision")
        self.assertEqual(meta["status"], "received")
        self.assertTrue(meta["explicit_routing_action"])
        self.assertEqual(meta["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(meta["confidence"], 0.91)
        self.assertTrue(applied["applied"])
        self.assertEqual(applied["local_action_taken"], "route_to")
        self.assertEqual(applied["apply_reason"], "route-to-local-safety-gate-passed")
        self.assertEqual(body["model"], "claude-haiku-4-5-20251001")
        self.assertTrue(routing_meta["managed_routing_applied"])
        self.assertEqual(routing_meta["managed_routing_confidence"], 0.91)
        self.assertEqual(routing_meta["managed_policy_id"], "managed-route-1")

    def test_unsafe_replacement_prompt_is_not_applied_and_raw_text_is_not_stored(self):
        routing_meta = {"routed_model": "claude-sonnet-4-6", "reason": "keep requested model"}
        body = {"model": "claude-sonnet-4-6"}
        recommendation_meta = {
            "status": "received",
            "target_model": "claude-haiku-4-5-20251001",
            "confidence": 0.8,
            "policy_id": "unsafe-prompt",
            "reason": "replace prompt",
            "replacement_prompt_present": True,
            "replacement_prompt_sha256": "hash-only",
        }

        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=recommendation_meta,
        )

        self.assertFalse(applied["applied"])
        self.assertEqual(applied["apply_reason"], "unsafe-replacement-prompt")
        self.assertEqual(applied["fallback"], "local-policy")
        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertNotIn("raw replacement", str(applied))

    def test_server_failure_records_metadata_and_keeps_local_model(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.error = RuntimeError("server unavailable")
        routing_meta = {"routed_model": "claude-sonnet-4-6"}

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))
        body = {"model": "claude-sonnet-4-6"}
        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=meta,
        )

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "request-failed")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertFalse(applied["applied"])
        self.assertEqual(body["model"], "claude-sonnet-4-6")

    def test_unsupported_target_model_is_not_applied(self):
        routing_meta = {"routed_model": "claude-sonnet-4-6", "reason": "keep requested model"}
        body = {"model": "claude-sonnet-4-6"}

        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta={
                "status": "received",
                "target_model": "claude-mystery-20260608",
                "policy_id": "bad-policy",
                "reason": "unknown model",
            },
        )

        self.assertFalse(applied["applied"])
        self.assertEqual(applied["apply_reason"], "unsupported-target-model")
        self.assertEqual(body["model"], "claude-sonnet-4-6")

    def test_outcome_feedback_posts_sanitized_metadata_to_unit_endpoint(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        recommendation_meta = {"optimization_unit_id": 42}
        outcome = recommendations.build_outcome_feedback(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            status_code=200,
            latency_ms=123,
            retry_count=1,
            input_tokens_est=20,
            output_tokens_est=3,
            actual_input_tokens=18,
            actual_output_tokens=2,
            cache_creation_input_tokens=4,
            cache_read_input_tokens=5,
            thinking_output_tokens=None,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            cache_meta={"status": "miss", "reason": "exact-miss"},
            crunch_meta={
                "saved_chars": 16,
                "pattern_rules": {
                    "configured_count": 1,
                    "before_chars": 3000,
                    "after_chars": 2000,
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "reviewed-pattern-rule",
                            "candidate_id": "candidate-123",
                            "policy_source": "managed-recommended",
                            "action": "shorten",
                            "matched_hashes": ["sha256:" + "a" * 64],
                            "applied_count": 1,
                            "saved_chars": 1000,
                            "skip_reasons": [],
                        }
                    ],
                },
                "raw_request": "must not leave local machine",
                "pattern_text": "raw pattern text must stay local",
            },
            routing_meta={
                "reason": "small request",
                "policy_source": "local-manual",
                "managed_recommendation": {
                    "optimization_unit_id": 42,
                    "policy_id": "policy-1",
                    "target_model": "claude-haiku-4-5-20251001",
                    "applied": True,
                },
                "prompt_difficulty_features": prompt_difficulty_features_from_text(
                    "Check the current outstanding vouchers before answering."
                ),
                "routing_experiment": {
                    "optimization_feedback": {
                        "schema": "tokenclaw.routing_experiment_feedback.v1",
                        "experiment_id": "exp-1",
                        "sampled": True,
                        "primary_model": "claude-haiku-4-5-20251001",
                        "shadow_model": "claude-sonnet-4-6",
                        "output_similarity": 0.95,
                        "primary_output_sha256": "primary-hash",
                        "shadow_output_sha256": "shadow-hash",
                        "primary_output_chars": 12,
                        "shadow_output_chars": 13,
                        "raw_response": "must be stripped",
                    }
                },
                "messages": ["must be stripped"],
            },
            category="chat",
            session_id="session-secret",
            error='{"error":{"type":"invalid_request_error","message":"bad raw body"}}',
            provider_adoption_windows=[
                {
                    "status": "abandoned",
                    "reason": "ttl-expired-without-tool-result",
                    "age_bucket": "1_6h",
                    "relationship": "emitted_tool_use",
                    "tool_use_count": 1,
                    "tool_result_count": 0,
                    "correlation_digest": "sha256:secret-tool-digest",
                    "tool_id": "tool-secret",
                }
            ],
        )

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.send_outcome_feedback(recommendation_meta, outcome))

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/optimization-units/42/outcome")
        self.assertEqual(meta["status"], "sent")
        self.assertEqual(FakeAsyncClient.last_json["status_code"], 200)
        self.assertEqual(FakeAsyncClient.last_json["routing_decision"]["reason"], "small request")
        self.assertEqual(FakeAsyncClient.last_json["managed_recommendation"]["optimization_unit_id"], 42)
        self.assertEqual(FakeAsyncClient.last_json["prompt_difficulty_features"]["downgrade_risk"], "block")
        self.assertEqual(
            FakeAsyncClient.last_json["prompt_difficulty_features"]["answerability_from_prompt_only"],
            "unlikely",
        )
        self.assertEqual(FakeAsyncClient.last_json["routing_experiment"]["output_similarity"], 0.95)
        self.assertEqual(FakeAsyncClient.last_json["routing_experiment"]["primary_output_sha256"], "primary-hash")
        self.assertEqual(FakeAsyncClient.last_json["quality_signals"]["risk_level"], "warning")
        self.assertIn("tool-use-abandoned", FakeAsyncClient.last_json["quality_signals"]["signal_codes"])
        self.assertIn("optimized-adoption-risk", FakeAsyncClient.last_json["quality_signals"]["signal_codes"])
        self.assertEqual(
            FakeAsyncClient.last_json["quality_signals"]["provider_adoption"]["status_counts"]["abandoned"],
            1,
        )
        pattern_decisions = FakeAsyncClient.last_json["pattern_decisions"]
        crunch_decision = next(item for item in pattern_decisions if item["decision_type"] == "crunch")
        cache_decision = next(item for item in pattern_decisions if item["decision_type"] == "cache")
        self.assertEqual(crunch_decision["rule_id"], "reviewed-pattern-rule")
        self.assertEqual(crunch_decision["candidate_id"], "candidate-123")
        self.assertEqual(crunch_decision["pattern_hash"], "sha256:" + "a" * 64)
        self.assertEqual(crunch_decision["outcome"], "applied")
        self.assertEqual(crunch_decision["saved_chars"], 1000)
        self.assertEqual(crunch_decision["tokens_saved_est"], 250)
        self.assertEqual(cache_decision["status"], "miss")
        self.assertEqual(cache_decision["outcome"], "skipped")
        self.assertIn("pattern_hash", cache_decision)
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(FakeAsyncClient.last_json)))
        self.assertNotIn("session-secret", str(FakeAsyncClient.last_json))
        self.assertNotIn("must not leave", str(FakeAsyncClient.last_json))
        self.assertNotIn("must be stripped", str(FakeAsyncClient.last_json))
        self.assertNotIn("raw pattern text", str(FakeAsyncClient.last_json))
        self.assertNotIn("secret-tool-digest", str(FakeAsyncClient.last_json))
        self.assertNotIn("tool-secret", str(FakeAsyncClient.last_json))

    def test_outcome_feedback_includes_old_context_summary_metadata_only(self):
        outcome = recommendations.build_outcome_feedback(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            status_code=200,
            latency_ms=2200,
            retry_count=0,
            input_tokens_est=20000,
            output_tokens_est=500,
            actual_input_tokens=18000,
            actual_output_tokens=450,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=0.03,
            cost_baseline_usd=0.05,
            cache_meta={"status": "miss", "reason": "exact-miss"},
            crunch_meta={
                "old_context_summarization": {
                    "enabled": True,
                    "status": "applied",
                    "reason": "summary-created",
                    "rule_id": "summary-rule-1",
                    "candidate_id": "summary-candidate-1",
                    "policy_source": "managed-recommended",
                    "category": "chat",
                    "before_chars": 100000,
                    "eligible_turns": 4,
                    "eligible_chars": 60000,
                    "saved_chars": 50000,
                    "tokens_saved_est": 12500,
                    "summary_cost_est_usd": 0.002,
                    "estimated_net_savings_usd": 0.02,
                    "summary_cache_hit": False,
                    "summary": "must not leave local machine",
                    "summary_request": {"messages": [{"content": "secret old turn text"}]},
                    "cache_key": "secret-cache-key",
                    "source_hash": "secret-source-hash",
                    "canary": {
                        "enabled": True,
                        "selected": True,
                        "status": "applied",
                        "cohort": "canary_applied",
                        "fraction": 0.5,
                        "unit": "source_hash",
                    },
                }
            },
            routing_meta={"reason": "small request", "policy_source": "local-manual"},
            category="chat",
            session_id="session-secret",
            error=None,
        )

        summary = outcome["old_context_summarization"]
        self.assertEqual(summary["schema"], "tokenclaw.old_context_summary_outcome_feedback.v1")
        self.assertEqual(summary["outcome"], "applied")
        self.assertEqual(summary["summary_policy_id"], "summary-rule-1")
        self.assertEqual(summary["canary_cohort"], "canary_applied")
        self.assertEqual(summary["eligible_chars_bucket"], "32k_128k_chars")
        self.assertEqual(summary["saved_tokens_bucket"], "4k_16k_tokens")
        self.assertEqual(summary["latency_bucket"], "2s_10s")
        self.assertTrue(summary["privacy"]["metadata_only"])
        rendered = str(outcome)
        self.assertNotIn("must not leave", rendered)
        self.assertNotIn("secret old turn text", rendered)
        self.assertNotIn("secret-cache-key", rendered)
        self.assertNotIn("secret-source-hash", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(outcome)))

    def test_old_context_summary_outcome_event_posts_to_policy_events(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        outcome = {
            "schema": "tokenclaw.old_context_summary_outcome_feedback.v1",
            "source_surface": "anthropic_messages",
            "category": "chat",
            "outcome": "holdout",
            "rule_id": "summary-rule-1",
            "candidate_id": "summary-candidate-1",
            "canary_cohort": "canary_holdout",
            "privacy": {"metadata_only": True},
        }
        event = recommendations.build_old_context_summary_outcome_event(outcome)

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.queue_policy_event_feedback(
                _NoQueueStore(),
                event,
                source_surface=recommendations.OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
            ))

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/policy-events")
        self.assertEqual(meta["status"], "sent")
        self.assertEqual(FakeAsyncClient.last_json["event_type"], "outcome")
        self.assertEqual(FakeAsyncClient.last_json["policy_sections"], ["crunch"])
        self.assertEqual(
            FakeAsyncClient.last_json["metadata"]["outcome"]["canary_cohort"],
            "canary_holdout",
        )
        self.assertNotIn("payload_json", str(FakeAsyncClient.last_json))

    def test_phase_routing_outcome_event_is_metadata_only_and_queue_safe(self):
        from tokenclaw.store import Store

        outcome = recommendations.build_phase_routing_outcome_feedback(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            status_code=200,
            latency_ms=1450,
            retry_count=0,
            input_tokens_est=900,
            output_tokens_est=120,
            actual_input_tokens=880,
            actual_output_tokens=110,
            thinking_output_tokens=None,
            cost_est_usd=0.001,
            cost_baseline_usd=0.006,
            cache_meta={"status": "miss", "reason": "exact-miss", "raw_request": "must stay local"},
            crunch_meta={"changed": False, "raw_response": "raw provider response"},
            routing_meta={
                "workflow_phase": "tool-execution",
                "workflow_phase_confidence": "high",
                "category": "tool-result",
                "has_tools": True,
                "policy_source": "local-manual",
                "messages": [{"content": "raw prompt text"}],
                "phase_canary": {
                    "enabled": True,
                    "policy_id": "phase-canary-1",
                    "status": "applied",
                    "cohort": "applied",
                    "reason": "selected-canary",
                    "target_model": "claude-haiku-4-5-20251001",
                    "policy_source": "local-manual",
                    "workflow_phase": "tool-execution",
                    "workflow_phase_confidence": "high",
                    "category": "tool-result",
                    "text_bucket": "2k-8k",
                    "has_tools": True,
                    "canary_fraction": 0.25,
                    "holdout_fraction": 0.25,
                    "cohort_hash": "sha256:" + "1" * 64,
                    "cohort_features": {"content": "raw canary content"},
                },
            },
            category="tool-result",
            error=None,
        )

        self.assertEqual(outcome["schema"], "tokenclaw.phase_routing_outcome_feedback.v1")
        self.assertEqual(outcome["outcome"], "applied")
        self.assertEqual(outcome["policy_id"], "phase-canary-1")
        self.assertEqual(outcome["cohort"], "applied")
        self.assertEqual(outcome["quality_signals"]["status"], "success")
        self.assertTrue(outcome["privacy"]["metadata_only"])
        rendered = str(outcome)
        self.assertNotIn("raw prompt text", rendered)
        self.assertNotIn("raw canary content", rendered)
        self.assertNotIn("must stay local", rendered)
        self.assertNotIn("raw provider response", rendered)
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(outcome)))

        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        os.environ["TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS"] = "3"
        FakeAsyncClient.error = RuntimeError("managed unavailable")
        event = recommendations.build_phase_routing_outcome_event(outcome)

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
                    meta = asyncio.run(
                        recommendations.queue_policy_event_feedback(
                            store,
                            event,
                            source_surface=recommendations.PHASE_ROUTING_OUTCOME_SOURCE_SURFACE,
                        )
                    )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, attempts, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "retryable-error")
        self.assertEqual(row["source_surface"], recommendations.PHASE_ROUTING_OUTCOME_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["attempts"], 1)
        self.assertNotIn("raw prompt text", row["payload_json"])
        self.assertNotIn("raw canary content", row["payload_json"])
        self.assertIn("phase-canary-1", row["payload_json"])

    def test_pattern_decision_summaries_cover_bypassed_and_errored_outcomes(self):
        bypassed = recommendations.pattern_decision_summaries(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            cache_meta={
                "status": "skipped",
                "reason": "cache-disabled",
                "policy_source": "local-default",
                "exact_enabled": False,
            },
            crunch_meta={},
            routing_meta={"category": "chat"},
            category="chat",
        )
        self.assertEqual(bypassed[0]["decision_type"], "cache")
        self.assertEqual(bypassed[0]["outcome"], "bypassed")

        errored = recommendations.pattern_decision_summaries(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            status_code=400,
            cost_est_usd=None,
            cost_baseline_usd=None,
            cache_meta={"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            crunch_meta={
                "pattern_rules": {
                    "configured_count": 1,
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "errored-rule",
                            "policy_source": "managed-recommended",
                            "matched_hashes": ["sha256:" + "b" * 64],
                            "applied_count": 1,
                            "saved_chars": 400,
                        }
                    ],
                }
            },
            routing_meta={"category": "tool-result"},
            category="tool-result",
        )
        outcomes = {(item["decision_type"], item["outcome"]) for item in errored}
        self.assertIn(("crunch", "errored"), outcomes)
        self.assertIn(("cache", "errored"), outcomes)

    def test_pattern_decision_summaries_distinguish_canary_cohorts(self):
        pattern_hash = "sha256:" + "f" * 64
        summaries = recommendations.pattern_decision_summaries(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            cache_meta={
                "status": "skipped",
                "reason": "tools-disabled",
                "policy_source": "local-manual",
                "exact_enabled": False,
                "pattern_rules": {
                    "configured_count": 1,
                    "matched_count": 0,
                    "skip_reasons": [
                        {
                            "rule_id": "cache-canary-rule",
                            "candidate_id": "cache-canary-candidate",
                            "policy_source": "managed-recommended",
                            "reason": "canary_holdout",
                            "matched_hashes": [pattern_hash],
                            "canary": {
                                "schema": "tokenclaw.pattern_canary_decision.v1",
                                "enabled": True,
                                "selected": False,
                                "status": "holdout",
                                "cohort": "canary_holdout",
                                "fraction": 0.1,
                                "unit": "request_fingerprint",
                                "cohort_key_hash": "sha256:" + "1" * 64,
                            },
                        }
                    ],
                },
            },
            crunch_meta={
                "pattern_rules": {
                    "configured_count": 1,
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "crunch-canary-rule",
                            "candidate_id": "crunch-canary-candidate",
                            "policy_source": "managed-recommended",
                            "matched_hashes": [pattern_hash],
                            "applied_count": 1,
                            "saved_chars": 400,
                            "canary": {
                                "schema": "tokenclaw.pattern_canary_decision.v1",
                                "enabled": True,
                                "selected": True,
                                "status": "applied",
                                "cohort": "canary_applied",
                                "fraction": 0.1,
                                "unit": "request_fingerprint",
                                "cohort_key_hash": "sha256:" + "2" * 64,
                            },
                        }
                    ],
                }
            },
            routing_meta={"category": "tool-result"},
            category="tool-result",
        )

        cohorts = {
            (item["decision_type"], item.get("candidate_id")): item.get("cohort")
            for item in summaries
        }
        self.assertEqual(cohorts[("crunch", "crunch-canary-candidate")], "canary_applied")
        self.assertEqual(cohorts[("cache", "cache-canary-candidate")], "canary_holdout")
        cache_holdout = next(item for item in summaries if item.get("candidate_id") == "cache-canary-candidate")
        self.assertEqual(cache_holdout["status"], "holdout")
        self.assertEqual(cache_holdout["outcome"], "holdout")
        self.assertEqual(cache_holdout["pattern_hash"], pattern_hash)

    def test_pattern_policy_evidence_queues_metadata_only_outcomes(self):
        from tokenclaw.store import Store, stable_json

        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        FakeAsyncClient.error = RuntimeError("feedback unavailable")
        pattern_hash = "sha256:" + "c" * 64
        routing_meta = {
            "category": "tool-result",
            "managed_recommendation": {"optimization_unit_id": 7, "policy_id": "routing-candidate"},
            "managed_pattern_features": {
                "schema": "tokenclaw.managed_pattern_feature_diagnostics.v1",
                "present": True,
                "pattern_hash": pattern_hash,
                "normalized_pattern_hash": pattern_hash,
                "pattern_hashes": [pattern_hash],
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "tool-result",
                "workflow_phase": "tool-result",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "1k_4k_tokens",
                "pattern_types": ["tool_results"],
                "local_pattern_module_families": ["tool_results"],
                "local_pattern_module_count": 1,
                "raw_pattern_strings_included": False,
            },
        }
        outcome = recommendations.build_outcome_feedback(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            status_code=400,
            latency_ms=2200,
            retry_count=2,
            input_tokens_est=1200,
            output_tokens_est=120,
            actual_input_tokens=1100,
            actual_output_tokens=80,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=0.002,
            cost_baseline_usd=0.006,
            cache_meta={
                "status": "skipped",
                "reason": "tools-disabled",
                "policy_source": "local-default",
                "pattern_rules": {
                    "configured_count": 1,
                    "skip_reasons": [
                        {
                            "rule_id": "cache-holdout-rule",
                            "candidate_id": "cache-holdout-candidate",
                            "policy_source": "managed-recommended",
                            "reason": "canary_holdout",
                            "matched_hashes": [pattern_hash],
                            "canary": {"enabled": True, "status": "holdout", "cohort": "canary_holdout"},
                        }
                    ],
                },
            },
            crunch_meta={
                "pattern_rules": {
                    "configured_count": 2,
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "crunch-applied-rule",
                            "candidate_id": "crunch-applied-candidate",
                            "policy_source": "managed-recommended",
                            "matched_hashes": [pattern_hash],
                            "applied_count": 1,
                            "saved_chars": 800,
                            "canary": {"enabled": True, "status": "applied", "cohort": "canary_applied"},
                        },
                        {
                            "rule_id": "crunch-bypass-rule",
                            "candidate_id": "crunch-bypass-candidate",
                            "policy_source": "managed-recommended",
                            "matched_hashes": [pattern_hash],
                            "applied_count": 0,
                            "skip_reasons": [{"reason": "local-canary-safety-stop", "count": 1, "pattern_hash": pattern_hash}],
                        },
                    ],
                },
                "raw_request": "must stay local",
                "pattern_text": "raw pattern text must stay local",
            },
            routing_meta=routing_meta,
            category="tool-result",
            session_id="session-secret",
            error="raw upstream error body must stay local enough to be classed only",
        )

        recommendations.assert_managed_egress_safe(outcome)
        evidence = outcome["pattern_policy_evidence"]
        cohorts = {item["cohort"] for item in evidence}
        actions = {item["action_family"] for item in evidence}
        outcomes = {item["outcome"] for item in evidence}
        self.assertIn("canary_applied", cohorts)
        self.assertIn("canary_holdout", cohorts)
        self.assertIn("bypassed", cohorts)
        self.assertIn("routing", actions)
        self.assertIn("crunch", actions)
        self.assertIn("cache", actions)
        self.assertIn("failed", outcomes)
        self.assertTrue(all(item["status_code_bucket"] == "4xx" for item in evidence))

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
                    meta = asyncio.run(
                        recommendations.queue_outcome_feedback(
                            store,
                            {"optimization_unit_id": 7},
                            outcome,
                            source_surface="anthropic_messages",
                        )
                    )
                row = store.conn.execute("select payload_json from managed_outcome_feedback_queue").fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "retryable-error")
        payload = json.loads(row["payload_json"])
        recommendations.assert_managed_egress_safe(payload)
        self.assertIn("pattern_policy_evidence", payload)
        rendered = stable_json(payload)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("must stay local", rendered)
        self.assertNotIn("raw pattern text", rendered)

    def test_outcome_feedback_failure_is_non_fatal_metadata(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        FakeAsyncClient.error = RuntimeError("feedback unavailable")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.send_outcome_feedback({"optimization_unit_id": 7}, {"status_code": 200}))

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "request-failed")
        self.assertIn("feedback unavailable", meta["error"])

    def test_outcome_feedback_egress_guard_blocks_raw_payload_before_network(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(
                recommendations.send_outcome_feedback(
                    {"optimization_unit_id": 7},
                    {
                        "source_surface": "openai_responses",
                        "status_code": 200,
                        "raw_response": "raw provider response must stay local",
                        "cache_key": "cache-key-secret",
                        "local_path": "/home/lutz/private/project/app.py",
                    },
                )
            )

        self.assertIsNone(FakeAsyncClient.last_url)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "unsafe-egress-payload")
        self.assertEqual(meta["endpoint"], "/v1/optimization-units/7/outcome")
        self.assertIn("raw_response", meta["egress_guard"]["blocked_keys"])
        self.assertIn("cache_key", meta["egress_guard"]["blocked_keys"])
        self.assertIn("local_path", meta["egress_guard"]["blocked_keys"])
        self.assertNotIn("raw provider response must stay local", str(meta))
        self.assertNotIn("cache-key-secret", str(meta))

    def test_queued_provider_outcome_feedback_disabled_does_not_enqueue(self):
        from tokenclaw.store import Store

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                meta = asyncio.run(
                    recommendations.queue_outcome_feedback(
                        store,
                        {"optimization_unit_id": 7},
                        {"source_surface": "anthropic_messages", "status_code": 200},
                    )
                )
                row = store.conn.execute("select count(*) as c from managed_outcome_feedback_queue").fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "disabled")
        self.assertEqual(row["c"], 0)

    def test_queued_provider_outcome_feedback_records_retryable_sanitized_payload(self):
        from tokenclaw.store import Store

        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        os.environ["TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS"] = "3"
        FakeAsyncClient.error = RuntimeError("feedback unavailable")

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
                    meta = asyncio.run(
                        recommendations.queue_outcome_feedback(
                            store,
                            {"optimization_unit_id": 7},
                            {
                                "source_surface": "anthropic_messages",
                                "status_code": 200,
                                "quality_signals": {"status": "success"},
                                "pattern_decisions": [
                                    {
                                        "schema": "tokenclaw.pattern_decision_summary.v1",
                                        "decision_type": "routing",
                                        "status": "applied",
                                        "policy_source": "managed-recommended",
                                    }
                                ],
                            },
                        )
                    )
                row = store.conn.execute(
                    "select source_surface, status, attempts, last_error, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "retryable-error")
        self.assertEqual(meta["reason"], "request-failed")
        self.assertEqual(row["source_surface"], "anthropic_messages")
        self.assertEqual(row["status"], "retryable-error")
        self.assertEqual(row["attempts"], 1)
        self.assertIn("feedback unavailable", row["last_error"])
        self.assertNotIn("raw_request", row["payload_json"])
        self.assertNotIn("raw_response", row["payload_json"])
        self.assertIn("quality_signals", row["payload_json"])
        self.assertIn("pattern_decisions", row["payload_json"])

    def test_queued_provider_outcome_feedback_egress_guard_does_not_enqueue_raw_payload(self):
        from tokenclaw.store import Store

        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
                    meta = asyncio.run(
                        recommendations.queue_outcome_feedback(
                            store,
                            {"optimization_unit_id": 7},
                            {
                                "source_surface": "openai_chat",
                                "status_code": 200,
                                "messages": [{"role": "user", "content": "raw chat text"}],
                                "tool_output": "raw tool output",
                                "request_id": "req-secret",
                            },
                        )
                    )
                row = store.conn.execute("select count(*) as c from managed_outcome_feedback_queue").fetchone()
            finally:
                store.conn.close()

        self.assertIsNone(FakeAsyncClient.last_url)
        self.assertEqual(row["c"], 0)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "unsafe-egress-payload")
        self.assertIn("messages", meta["egress_guard"]["blocked_keys"])
        self.assertIn("tool_output", meta["egress_guard"]["blocked_keys"])
        self.assertIn("request_id", meta["egress_guard"]["blocked_keys"])
        self.assertNotIn("raw chat text", str(meta))
        self.assertNotIn("req-secret", str(meta))

    def test_policy_event_egress_guard_blocks_raw_payload_before_network(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        event = {
            "event_type": "applied",
            "recommendation_id": "policy-1",
            "policy_sections": ["routing"],
            "metadata": {
                "schema": "tokenclaw.policy_lifecycle_metadata.v1",
                "messages": [{"content": "raw prompt must stay local"}],
                "summary_text": "raw summary must stay local",
                "workspace_path": "/home/lutz/private/project",
            },
        }

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.queue_policy_event_feedback(_NoQueueStore(), event))

        self.assertIsNone(FakeAsyncClient.last_url)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "unsafe-egress-payload")
        self.assertEqual(meta["endpoint"], "/v1/policy-events")
        self.assertIn("messages", meta["egress_guard"]["blocked_keys"])
        self.assertIn("summary_text", meta["egress_guard"]["blocked_keys"])
        self.assertIn("workspace_path", meta["egress_guard"]["blocked_keys"])
        self.assertNotIn("raw prompt must stay local", str(meta))
        self.assertNotIn("/home/lutz/private/project", str(meta))

    def test_policy_event_lifecycle_command_enum_is_allowed(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        FakeAsyncClient.response = FakeResponse(body={"accepted": True})
        event = {
            "event_type": "dry_run",
            "recommendation_id": "phase-routing:test",
            "policy_sections": ["routing"],
            "metadata": {
                "schema": "tokenclaw.phase_routing_lifecycle_metadata.v1",
                "command": "phase-routing-dry-run",
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "request_ids_included": False,
                },
            },
        }

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.queue_policy_event_feedback(_NoQueueStore(), event))

        self.assertEqual(FakeAsyncClient.last_url, "http://managed.test/v1/policy-events")
        self.assertEqual(meta["status"], "sent")
        self.assertEqual(FakeAsyncClient.last_json["metadata"]["command"], "phase-routing-dry-run")

    def test_provider_mismatch_recommendation_is_not_applied(self):
        routing_meta = {"routed_model": "claude-sonnet-4-6", "reason": "keep requested model"}
        body = {"model": "claude-sonnet-4-6"}

        applied = recommendations.apply_recommendation_to_body(
            provider="anthropic",
            body=body,
            routing_meta=routing_meta,
            recommendation_meta={
                "status": "received",
                "target_model": "gpt-5-mini",
                "policy_id": "bad-policy",
                "reason": "wrong provider",
            },
        )

        self.assertFalse(applied["applied"])
        self.assertEqual(applied["apply_reason"], "provider-mismatch")
        self.assertEqual(body["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
