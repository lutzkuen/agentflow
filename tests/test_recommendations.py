import asyncio
import os
import unittest
from unittest.mock import patch

from agentflow_proxy import recommendations


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class FakeAsyncClient:
    response = FakeResponse()
    error = None
    last_url = None
    last_json = None
    last_timeout = None

    def __init__(self, timeout):
        self.__class__.last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json):
        self.__class__.last_url = url
        self.__class__.last_json = json
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.response


class RecommendationTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_RECOMMENDATION_ENABLED",
        "AGENTFLOW_RECOMMENDATION_SERVER_URL",
        "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS",
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

    def test_success_path_posts_feature_unit_and_applies_target_model(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"
        os.environ["AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS"] = "0.25"
        FakeAsyncClient.response = FakeResponse(body={
            "target_model": "claude-haiku-4-5-20251001",
            "replacement_prompt": "raw replacement text should not be stored",
            "confidence": 0.82,
            "policy_id": "policy-1",
            "reason": "route cheaper",
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
        self.assertEqual(FakeAsyncClient.last_json["replayability_level"], "features_only")
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(self._keys_in(FakeAsyncClient.last_json)))
        self.assertEqual(meta["status"], "received")
        self.assertEqual(meta["policy_source"], "managed-recommended")
        self.assertTrue(meta["replacement_prompt_present"])
        self.assertIn("replacement_prompt_sha256", meta)
        self.assertNotIn("raw replacement text", str(meta))
        self.assertTrue(applied["applied"])
        self.assertTrue(applied["changed_model"])
        self.assertFalse(applied["replacement_prompt_applied"])
        self.assertEqual(body["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(routing_meta["final_policy_source"], "managed-recommended")
        self.assertEqual(routing_meta["managed_policy_id"], "policy-1")

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
