import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agentflow_proxy import recommendations


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


class RecommendationTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_RECOMMENDATION_ENABLED",
        "AGENTFLOW_RECOMMENDATION_SERVER_URL",
        "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS",
        "AGENTFLOW_RECOMMENDATION_FAILURE_MODE",
        "AGENTFLOW_MANAGED_API_KEY",
        "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS",
        "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS",
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

    def test_success_path_posts_feature_unit_with_auth_and_applies_target_model(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        os.environ["AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS"] = "0.25"
        os.environ["AGENTFLOW_MANAGED_API_KEY"] = "managed-secret"
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
            "policy_source": "local-default",
            "command": "raw command text",
            "tenant_id": "tenant-secret",
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
        self.assertEqual(pattern_features["schema"], "agentflow.pattern_features.v1")
        self.assertEqual(pattern_features["hash_basis"], "normalized-structure-and-size-buckets")
        self.assertEqual(pattern_features["text_bucket"], "lt_2k_chars")
        self.assertEqual(pattern_features["token_bucket"], "lt_1k_tokens")
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
        )

        pattern_features = unit["input_features"]["pattern_features"]

        self.assertEqual(pattern_features["workflow_phase"], "summary")
        self.assertEqual(pattern_features["text_bucket"], "8k_32k_chars")
        self.assertEqual(pattern_features["token_bucket"], "1k_4k_tokens")
        self.assertEqual(pattern_features["pattern_types"], ["repeated_input_section"])
        self.assertEqual(pattern_features["local_decision_status"], "routing:skipped|crunch:applied|cache:skipped")
        self.assertTrue(pattern_features["pattern_hash"].startswith("sha256:"))
        self.assertTrue(pattern_features["crunch_pattern_hash"].startswith("sha256:"))
        self.assertTrue(pattern_features["cache_pattern_hash"].startswith("sha256:"))
        self.assertIn("request_id_hash", unit["grouping_identifiers"])
        self.assertIn("thread_id_hash", unit["grouping_identifiers"])
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(unit)))
        self._assert_no_sensitive_strings(unit)
        self.assertNotIn("raw-content-hash-should-not-define-pattern", str(unit))

    def test_valid_noop_recommendation_records_received_without_changing_model(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
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
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = ""

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "server-url-not-configured")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertIsNone(FakeAsyncClient.last_url)

    def test_timeout_records_bounded_fallback_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS"] = "0.1"
        FakeAsyncClient.error = httpx.TimeoutException("too slow")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(FakeAsyncClient.last_timeout, 0.1)
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "timeout")
        self.assertEqual(meta["fallback"], "local-policy")

    def test_unreachable_records_fallback_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        FakeAsyncClient.error = httpx.ConnectError("connection refused")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.fetch_recommendation({"requested_model": "claude-sonnet-4-6"}))

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "unreachable")
        self.assertEqual(meta["fallback"], "local-policy")

    def test_invalid_json_and_schema_record_fallback_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
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
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
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
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
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
                "routing_experiment": {
                    "optimization_feedback": {
                        "schema": "agentflow.routing_experiment_feedback.v1",
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
        )

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.send_outcome_feedback(recommendation_meta, outcome))

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/optimization-units/42/outcome")
        self.assertEqual(meta["status"], "sent")
        self.assertEqual(FakeAsyncClient.last_json["status_code"], 200)
        self.assertEqual(FakeAsyncClient.last_json["routing_decision"]["reason"], "small request")
        self.assertEqual(FakeAsyncClient.last_json["managed_recommendation"]["optimization_unit_id"], 42)
        self.assertEqual(FakeAsyncClient.last_json["routing_experiment"]["output_similarity"], 0.95)
        self.assertEqual(FakeAsyncClient.last_json["routing_experiment"]["primary_output_sha256"], "primary-hash")
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
                                "schema": "agentflow.pattern_canary_decision.v1",
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
                                "schema": "agentflow.pattern_canary_decision.v1",
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

    def test_outcome_feedback_failure_is_non_fatal_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        FakeAsyncClient.error = RuntimeError("feedback unavailable")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.send_outcome_feedback({"optimization_unit_id": 7}, {"status_code": 200}))

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "request-failed")
        self.assertIn("feedback unavailable", meta["error"])

    def test_queued_provider_outcome_feedback_disabled_does_not_enqueue(self):
        from agentflow_proxy.store import Store

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
        from agentflow_proxy.store import Store

        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        os.environ["AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS"] = "3"
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
                                "raw_request": "must stay local",
                                "raw_response": "raw provider response",
                                "quality_signals": {"status": "success"},
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
        self.assertNotIn("must stay local", row["payload_json"])
        self.assertNotIn("raw provider response", row["payload_json"])

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
