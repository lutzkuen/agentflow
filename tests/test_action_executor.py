from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tokenclaw.action_executor import ActionExecutor


class ActionExecutorTests(unittest.TestCase):
    def test_applies_only_provider_body_model_rewrite_for_routing(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        result = ActionExecutor(provider="openai").execute(
            body=body,
            routing_meta=routing_meta,
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "route-policy",
                "provider": "openai",
                "source_surface": "openai_responses",
                "target_model": "gpt-5-mini",
                "routing": {"target_model": "gpt-5-mini"},
                "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            },
            application_enabled=True,
            source_surface="openai_responses",
        )

        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertEqual(routing_meta["routed_model"], "gpt-5-mini")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["routing"]["status"], "applied")
        self.assertEqual(result["routing"]["apply_reason"], "provider-body-model-rewrite")
        self.assertEqual(result["outcome_feedback"]["status"], "applied")

    def test_unsupported_actions_are_vetoed_with_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        result = ActionExecutor(provider="openai").execute(
            body=body,
            routing_meta={"routed_model": "gpt-5-codex"},
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "unsafe-policy",
                "provider": "openai",
                "target_model": "gpt-5-mini",
                "routing": {"target_model": "gpt-5-mini"},
                "actions": [{"type": "replacement_prompt"}],
                "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            },
            application_enabled=True,
        )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["apply_reason"], "unsupported-action-type")
        self.assertEqual(result["unsupported_actions"][0]["reason"], "unsupported-action-type")
        self.assertEqual(result["outcome_feedback"]["status"], "vetoed")

    def test_locally_disabled_action_family_is_vetoed(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        result = ActionExecutor(provider="openai", supported_action_families=("crunch", "cache")).execute(
            body=body,
            routing_meta={"routed_model": "gpt-5-codex"},
            decision={
                "enabled": True,
                "status": "received",
                "policy_id": "routing-disabled",
                "provider": "openai",
                "target_model": "gpt-5-mini",
                "routing": {"target_model": "gpt-5-mini"},
                "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
            },
            application_enabled=True,
        )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["routing"]["status"], "vetoed")
        self.assertIn("routing", result["outcome_feedback"]["vetoed_families"])

    def test_executor_disabled_holds_without_changing_forwarding_body(self):
        body = {"model": "claude-sonnet-4-6"}
        with patch.dict(os.environ, {"TOKENCLAW_ACTION_EXECUTOR_ENABLED": "0"}):
            result = ActionExecutor(provider="anthropic").execute(
                body=body,
                routing_meta={"routed_model": "claude-sonnet-4-6"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "held-policy",
                    "provider": "anthropic",
                    "target_model": "claude-haiku-4-5-20251001",
                    "routing": {"target_model": "claude-haiku-4-5-20251001"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "action-executor-disabled")
        self.assertEqual(result["fallback"], "local-policy")


if __name__ == "__main__":
    unittest.main()
