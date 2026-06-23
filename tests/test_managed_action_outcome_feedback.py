from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from tokenclaw import recommendations
from tokenclaw.action_executor import ActionExecutor
from tokenclaw.managed_action_outcome_feedback import (
    MANAGED_ACTION_OUTCOME_FEEDBACK_SCHEMA,
    build_managed_action_feedback,
    record_managed_action_feedback,
)
from tokenclaw.store import Store


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class FakeAsyncClient:
    response = FakeResponse()
    error = None
    last_url = None
    last_json = None

    def __init__(self, timeout):
        self.__class__.last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json, headers=None):
        self.__class__.last_url = url
        self.__class__.last_json = json
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.response

    async def patch(self, url, json, headers=None):
        self.__class__.last_url = url
        self.__class__.last_json = json
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.response


def _keys_in(value):
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_keys_in(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_keys_in(item))
    return keys


class BuildManagedActionFeedbackTests(unittest.TestCase):
    managed_live_env = {"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "live"}

    def _execute(self, *, provider, body, decision, application_enabled=True, source_surface=None):
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            return ActionExecutor(provider=provider).execute(
                body=body,
                routing_meta={"routed_model": body.get("model"), "source_surface": source_surface or ""},
                decision=decision,
                application_enabled=application_enabled,
                source_surface=source_surface,
            )

    def test_applied_decision_produces_feedback_with_family_status(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        result = self._execute(
            provider="openai",
            body=body,
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "route-policy",
                "decision_id": "decision-1",
                "provider": "openai",
                "source_surface": "openai_responses",
                "target_model": "gpt-5-mini",
                "routing": {"target_model": "gpt-5-mini"},
            },
            source_surface="openai_responses",
        )

        feedback = build_managed_action_feedback(
            result,
            source_surface="openai_responses",
            app_family="codex",
            contract_id="contract-7",
        )

        self.assertEqual(feedback["schema"], MANAGED_ACTION_OUTCOME_FEEDBACK_SCHEMA)
        self.assertEqual(feedback["event_type"], "managed_action_outcome")
        self.assertEqual(feedback["provider"], "openai")
        self.assertEqual(feedback["source_surface"], "openai_responses")
        self.assertEqual(feedback["app_family"], "codex")
        self.assertEqual(feedback["contract_id"], "contract-7")
        self.assertEqual(feedback["decision_id"], "decision-1")
        self.assertEqual(feedback["policy_id"], "route-policy")
        self.assertEqual(feedback["local_result"], "applied")
        self.assertTrue(feedback["applied"])
        self.assertIn("routing", feedback["applied_families"])
        routing_record = next(a for a in feedback["actions"] if a["family"] == "routing")
        self.assertEqual(routing_record["status"], "applied")
        self.assertTrue(routing_record["applied"])
        self.assertTrue(feedback["privacy_summary"]["metadata_only"])

    def test_every_decision_produces_feedback_even_when_nothing_applied(self):
        # observe_only -> held, no application
        body = {"model": "claude-sonnet-4-6"}
        with patch.dict(
            os.environ,
            {"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "observe_only"},
            clear=False,
        ):
            result = ActionExecutor(provider="anthropic").execute(
                body=body,
                routing_meta={"routed_model": "claude-sonnet-4-6"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "p-held",
                    "decision_id": "d-held",
                    "provider": "anthropic",
                    "target_model": "claude-haiku-4-5-20251001",
                    "routing": {"target_model": "claude-haiku-4-5-20251001"},
                },
                application_enabled=True,
                source_surface="anthropic_messages",
            )

        feedback = build_managed_action_feedback(result, source_surface="anthropic_messages")

        self.assertEqual(feedback["local_result"], "held")
        self.assertFalse(feedback["applied"])
        self.assertEqual(feedback["applied_families"], [])
        # Every family is represented in the actions list.
        self.assertEqual(
            sorted(a["family"] for a in feedback["actions"]),
            ["cache", "crunch", "routing"],
        )

    def test_locally_vetoed_actions_are_reported_as_vetoes(self):
        body = {"model": "gpt-5-codex", "input": "hi"}
        result = self._execute(
            provider="openai",
            body=body,
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "veto-policy",
                "decision_id": "veto-1",
                "provider": "openai",
                "source_surface": "openai_responses",
                # claude target on an openai provider -> provider-mismatch veto
                "target_model": "claude-haiku-4-5-20251001",
                "routing": {"target_model": "claude-haiku-4-5-20251001"},
            },
            source_surface="openai_responses",
        )

        feedback = build_managed_action_feedback(result, source_surface="openai_responses")

        self.assertEqual(feedback["local_result"], "vetoed")
        self.assertIn("routing", feedback["vetoed_families"])
        self.assertIn("provider-mismatch", feedback["veto_reason_codes"])
        routing_record = next(a for a in feedback["actions"] if a["family"] == "routing")
        self.assertEqual(routing_record["status"], "vetoed")

    def test_unsupported_actions_surface_capability_reason_codes(self):
        body = {"model": "gpt-5-codex", "input": "hi"}
        result = self._execute(
            provider="openai",
            body=body,
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "cap-policy",
                "decision_id": "cap-1",
                "provider": "openai",
                "source_surface": "openai_responses",
                "actions": [{"type": "rewrite_replacement_prompt"}],
            },
            source_surface="openai_responses",
        )

        feedback = build_managed_action_feedback(result, source_surface="openai_responses")

        self.assertEqual(feedback["local_result"], "vetoed")
        self.assertGreaterEqual(feedback["unsupported_action_count"], 1)
        self.assertIn("unsupported-action-type", feedback["capability_reason_codes"])

    def test_outcome_metrics_are_numeric_whitelist_only(self):
        result = {
            "provider": "openai",
            "status": "applied",
            "applied": True,
            "decision_id": "m-1",
            "policy_id": "m-policy",
            "routing": {"status": "applied", "applied": True},
        }
        feedback = build_managed_action_feedback(
            result,
            source_surface="openai_responses",
            outcome_metrics={
                "status_code": 200,
                "latency_ms": 1450.0,
                "retry_count": 1,
                "fallback_count": 0,
                "input_tokens": 880,
                "output_tokens": 110,
                "total_tokens": 990,
                "estimated_cost_usd": 0.001,
                "estimated_baseline_usd": 0.006,
                # raw-like keys must be dropped, not forwarded
                "prompt": "raw prompt text",
                "messages": [{"content": "secret"}],
                "cache_key": "secret-cache-key",
            },
        )

        metrics = feedback["outcome_metrics"]
        self.assertEqual(metrics["status_code"], 200)
        self.assertEqual(metrics["latency_ms"], 1450.0)
        self.assertEqual(metrics["retry_count"], 1)
        self.assertEqual(metrics["input_tokens"], 880)
        self.assertEqual(metrics["total_tokens"], 990)
        self.assertEqual(metrics["estimated_cost_usd"], 0.001)
        self.assertEqual(metrics["estimated_baseline_usd"], 0.006)
        self.assertNotIn("prompt", metrics)
        self.assertNotIn("messages", metrics)
        self.assertNotIn("cache_key", metrics)

    def test_feedback_excludes_raw_payload_keys_and_secrets(self):
        body = {"model": "gpt-5-codex", "input": "secret prompt body"}
        result = self._execute(
            provider="openai",
            body=body,
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "safe-policy",
                "decision_id": "safe-1",
                "provider": "openai",
                "source_surface": "openai_responses",
                "target_model": "gpt-5-mini",
                "routing": {"target_model": "gpt-5-mini"},
                # these should never propagate into feedback
                "messages": [{"content": "raw prompt should stay local"}],
                "api_key": "api-key-secret",
            },
            source_surface="openai_responses",
        )
        feedback = build_managed_action_feedback(
            result,
            source_surface="openai_responses",
            outcome_metrics={"prompt": "raw prompt should stay local", "status_code": 200},
        )

        keys = _keys_in(feedback)
        self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(keys))
        rendered = str(feedback)
        self.assertNotIn("raw prompt should stay local", rendered)
        self.assertNotIn("api-key-secret", rendered)
        self.assertNotIn("secret prompt body", rendered)


class RecordManagedActionFeedbackTests(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
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

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _sample_result(self):
        return {
            "provider": "openai",
            "status": "applied",
            "applied": True,
            "decision_id": "d-send-1",
            "policy_id": "p-send-1",
            "applied_families": ["routing"],
            "routing": {"status": "applied", "applied": True, "apply_reason": "provider-body-model-rewrite"},
            "crunch": {"status": "not-present", "applied": False},
            "cache": {"status": "not-present", "applied": False},
            "product_mode": {"mode": "live"},
        }

    def test_sends_feedback_to_policy_events_endpoint_when_enabled(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"

        with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
            meta = asyncio.run(
                record_managed_action_feedback(
                    _NoQueueStore(),  # no queue methods -> direct send
                    self._sample_result(),
                    source_surface="openai_responses",
                    app_family="codex",
                    contract_id="contract-9",
                    outcome_metrics={"status_code": 200, "latency_ms": 12},
                )
            )

        self.assertEqual(FakeAsyncClient.last_url, "http://127.0.0.1:4100/v1/policy-events")
        self.assertEqual(meta["status"], "sent")
        self.assertEqual(meta["local_result"], "applied")
        self.assertFalse(meta["payload_included"])
        # The posted body is metadata-only.
        self.assertNotIn("payload_json", str(FakeAsyncClient.last_json))
        self.assertEqual(FakeAsyncClient.last_json["schema"], MANAGED_ACTION_OUTCOME_FEEDBACK_SCHEMA)
        self.assertEqual(FakeAsyncClient.last_json["provider"], "openai")

    def test_send_failure_queues_locally_with_bounded_retry(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        os.environ["TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS"] = "3"
        FakeAsyncClient.error = RuntimeError("managed unavailable")

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch.object(recommendations.httpx, "AsyncClient", FakeAsyncClient):
                    meta = asyncio.run(
                        record_managed_action_feedback(
                            store,
                            self._sample_result(),
                            source_surface="openai_responses",
                            app_family="codex",
                        )
                    )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, attempts, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "retryable-error")
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["source_surface"], "openai_responses")
        self.assertEqual(row["attempts"], 1)
        self.assertIn("p-send-1", row["payload_json"])

    def test_disabled_managed_mode_still_queues_feedback(self):
        # No recommendation server / managed disabled: feedback must still be
        # captured locally rather than dropped.
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                meta = asyncio.run(
                    record_managed_action_feedback(
                        store,
                        self._sample_result(),
                        source_surface="openai_responses",
                    )
                )
                row = store.conn.execute(
                    "select status, payload_json from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertFalse(meta["enabled"])
        self.assertIsNotNone(row)
        self.assertIn("p-send-1", row["payload_json"])

    def test_record_never_raises_on_queue_error(self):
        os.environ["TOKENCLAW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["TOKENCLAW_RECOMMENDATION_SERVER_URL"] = "http://127.0.0.1:4100"

        class _ExplodingStore:
            def enqueue_managed_outcome_feedback(self, **kwargs):
                raise RuntimeError("db down")

        meta = asyncio.run(
            record_managed_action_feedback(
                _ExplodingStore(),
                self._sample_result(),
                source_surface="openai_responses",
            )
        )

        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["reason"], "queue-failed")
        self.assertFalse(meta["payload_included"])


class _NoQueueStore:
    pass


if __name__ == "__main__":
    unittest.main()
