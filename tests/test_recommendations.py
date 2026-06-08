import asyncio
import os
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
        }
        unit = recommendations.build_optimization_unit(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            routing_meta=routing_meta,
            crunch_meta={"changed": False},
            cache_meta={"status": "miss", "reason": "exact-miss"},
            category="chat",
            stream=False,
            input_tokens_est=300,
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
                "raw_request": "must not leave local machine",
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
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(FakeAsyncClient.last_json)))
        self.assertNotIn("session-secret", str(FakeAsyncClient.last_json))
        self.assertNotIn("must not leave", str(FakeAsyncClient.last_json))
        self.assertNotIn("must be stripped", str(FakeAsyncClient.last_json))

    def test_outcome_feedback_failure_is_non_fatal_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        FakeAsyncClient.error = RuntimeError("feedback unavailable")

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(recommendations.send_outcome_feedback({"optimization_unit_id": 7}, {"status_code": 200}))

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "request-failed")
        self.assertIn("feedback unavailable", meta["error"])

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
