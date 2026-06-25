from __future__ import annotations

import copy
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tokenclaw.anthropic_proxy import _record_routing_rate_limit_fallback
from tokenclaw.managed_session_tier import apply_session_tier_to_body, clear_session_tier_cache


SONNET = "claude-sonnet-4-5-20240620"
HAIKU = "claude-haiku-4-5-20251001"


class ManagedSessionTierCanaryTests(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
        "TOKENCLAW_MANAGED_ROUTING",
        "TOKENCLAW_LOCAL_RULES_ONLY",
        "TOKENCLAW_SESSION_TIER_CANARY_SALT",
        "TOKENCLAW_ACTION_EXECUTOR_REQUIRE_SIGNATURE",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        clear_session_tier_cache()

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_session_tier_cache()

    def _managed_env(self):
        return patch.dict(
            os.environ,
            {
                "TOKENCLAW_MANAGED": "1",
                "TOKENCLAW_MANAGED_MODE": "canary",
                "TOKENCLAW_MANAGED_ROUTING": "1",
                "TOKENCLAW_SESSION_TIER_CANARY_SALT": "test-salt",
            },
            clear=False,
        )

    def _meta(self, **canary_overrides):
        canary = {
            "schema": "agentflow.session_tier_routing_canary.v1",
            "policy_id": "managed-session-tier-routing-canary-v1",
            "status": "recommended",
            "cohort_bucket": "streaming-agentic:safe-tool-result",
            "target_tier": "haiku",
            "target_model_family": "claude-haiku",
            "canary_fraction": 1.0,
            "holdout_fraction": 0.0,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "feature_only": True,
            "metadata_only": True,
        }
        canary.update(canary_overrides)
        return {
            "schema": "tokenclaw.local_session_tier_decision_meta.v1",
            "status": "received",
            "enabled": True,
            "session_tier_source": "managed",
            "policy_source": "managed-recommended",
            "tier": "sonnet",
            "target_model": SONNET,
            "confidence": 0.9,
            "reason_codes": ["coding-agent-file-ops"],
            "cache_status": "stored",
            "session_tier_canary": canary,
            "metadata_only": True,
        }

    def _apply(self, meta, *, stream=True, routing_meta=None, body=None):
        request_body = body or {"model": SONNET, "max_tokens": 1024}
        routing = routing_meta or {
            "requested_model": SONNET,
            "routed_model": SONNET,
            "category": "tool-result",
            "has_tools": True,
        }
        result = apply_session_tier_to_body(
            request_body,
            routing,
            meta,
            session_id="session-tier-canary-test",
            stream=stream,
        )
        return result, request_body, routing, meta

    def test_managed_disabled_holds_canary_without_mutating_model(self):
        result, body, routing, meta = self._apply(self._meta())

        self.assertIsNone(result)
        self.assertEqual(body["model"], SONNET)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["local_result"], "held")
        self.assertEqual(routing["local_result"], "held")
        self.assertEqual(meta["action_executor"]["apply_reason"], "local-application-disabled")

    def test_expired_canary_is_vetoed_without_mutating_model(self):
        meta = self._meta(expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())
        with self._managed_env():
            result, body, routing, meta = self._apply(meta)

        self.assertIsNone(result)
        self.assertEqual(body["model"], SONNET)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["local_result"], "vetoed")
        self.assertEqual(meta["apply_reason"], "expired-session-tier-canary")
        self.assertEqual(routing["phase_canary"]["status"], "safety_stopped")

    def test_holdout_assignment_preserves_requested_model(self):
        meta = self._meta(canary_fraction=0.0, holdout_fraction=1.0)
        with self._managed_env():
            result, body, routing, meta = self._apply(meta)

        self.assertIsNone(result)
        self.assertEqual(body["model"], SONNET)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["local_result"], "heldout")
        self.assertEqual(routing["routing_outcome_label"], "heldout")
        self.assertEqual(routing["phase_canary"]["status"], "holdout")

    def test_valid_opted_in_canary_mutates_streaming_model(self):
        meta = self._meta(canary_fraction=1.0, holdout_fraction=0.0)
        with self._managed_env():
            result, body, routing, meta = self._apply(meta)

        self.assertEqual(result, HAIKU)
        self.assertEqual(body["model"], HAIKU)
        self.assertTrue(meta["applied"])
        self.assertTrue(meta["changed_model"])
        self.assertEqual(meta["local_result"], "applied")
        self.assertEqual(routing["routing_outcome_label"], "applied")
        self.assertEqual(routing["phase_canary"]["status"], "applied")
        self.assertFalse(str(meta).find("session-tier-canary-test") >= 0)

    def test_local_thinking_safety_veto_preserves_requested_model(self):
        meta = self._meta(canary_fraction=1.0, holdout_fraction=0.0)
        routing_meta = {
            "requested_model": SONNET,
            "routed_model": SONNET,
            "category": "tool-result",
            "has_tools": True,
            "thinking_gate": {"status": "blocked"},
        }
        with self._managed_env():
            result, body, routing, meta = self._apply(meta, routing_meta=routing_meta)

        self.assertIsNone(result)
        self.assertEqual(body["model"], SONNET)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["local_result"], "vetoed")
        self.assertEqual(meta["apply_reason"], "local-thinking-safety-guard")
        self.assertEqual(routing["routing_outcome_label"], "vetoed")

    def test_executor_veto_records_unsupported_target_without_mutating_model(self):
        meta = self._meta(target_model="gpt-5-mini", target_model_family="gpt-mini", target_tier="mini")
        with self._managed_env():
            result, body, routing, meta = self._apply(meta)

        self.assertIsNone(result)
        self.assertEqual(body["model"], SONNET)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["local_result"], "vetoed")
        self.assertEqual(meta["action_executor"]["routing"]["veto_reason"], "provider-mismatch")

    def test_unsigned_canary_is_vetoed_when_signature_gate_is_required(self):
        meta = self._meta(canary_fraction=1.0, holdout_fraction=0.0)
        with self._managed_env(), patch.dict(os.environ, {"TOKENCLAW_ACTION_EXECUTOR_REQUIRE_SIGNATURE": "1"}, clear=False):
            result, body, routing, meta = self._apply(meta)

        self.assertIsNone(result)
        self.assertEqual(body["model"], SONNET)
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["local_result"], "vetoed")
        self.assertEqual(meta["action_executor"]["apply_reason"], "unsigned-policy")

    def test_rate_limit_fallback_marks_session_tier_feedback(self):
        routing = {
            "managed_session_tier": {
                "applied": True,
                "local_result": "applied",
                "target_model": HAIKU,
            },
            "phase_canary": {"enabled": True, "status": "applied"},
        }

        _record_routing_rate_limit_fallback(
            routing,
            requested_model=SONNET,
            from_model=HAIKU,
        )

        self.assertEqual(routing["routing_outcome_label"], "fallback")
        self.assertEqual(routing["managed_session_tier"]["local_result"], "fallback")
        self.assertEqual(routing["managed_session_tier"]["actual_forwarded_model"], SONNET)
        self.assertEqual(routing["phase_canary"]["fallback_reason"], "rate_limited")

    def test_canary_assignment_is_cached_for_session(self):
        meta = self._meta(canary_fraction=1.0, holdout_fraction=0.0)
        with self._managed_env():
            _, _, _, first = self._apply(meta)
            second_meta = copy.deepcopy(first)
            second_meta["cache_status"] = "hit"
            _, _, _, second = self._apply(second_meta)

        self.assertEqual(first["local_canary_assignment"]["cohort_key_hash"], second["local_canary_assignment"]["cohort_key_hash"])


if __name__ == "__main__":
    unittest.main()
