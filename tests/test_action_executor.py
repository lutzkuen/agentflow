from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from tokenclaw.action_executor import ActionExecutor
from tokenclaw.managed_action_outcome_feedback import build_managed_action_feedback


class ActionExecutorTests(unittest.TestCase):
    managed_live_env = {"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "live"}

    def _routing_decision(self, **overrides):
        decision = {
            "enabled": True,
            "status": "received",
            "policy_id": "managed-routing-policy",
            "decision_id": "decision-1",
            "provider": "openai",
            "source_surface": "openai_responses",
            "routing": {
                "target_model": "gpt-5-mini",
                "route_proposal": {
                    "target_model": "gpt-5-mini",
                    "traffic_treatment": "canary",
                    "route_selected": True,
                    "canary_fraction": 0.20,
                    "holdout_fraction": 0.10,
                    "server_selected_canary_membership": True,
                },
            },
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }
        for key, value in overrides.items():
            if key == "route_proposal":
                decision["routing"]["route_proposal"].update(value)
            elif key == "routing":
                decision["routing"].update(value)
            else:
                decision[key] = value
        return decision

    def test_applies_only_provider_body_model_rewrite_for_routing(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
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
        self.assertEqual(result["product_mode"]["mode"], "live")

    def test_canary_selected_by_server_applies_route_and_records_treatment(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta=routing_meta,
                decision=self._routing_decision(),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertEqual(routing_meta["routed_model"], "gpt-5-mini")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["server_traffic_treatment"], "canary")
        self.assertTrue(result["server_route_selected"])
        self.assertEqual(result["canary_fraction"], 0.20)
        self.assertEqual(result["holdout_fraction"], 0.10)
        self.assertEqual(result["routing"]["status"], "applied")
        self.assertIn("routing", result["outcome_feedback"]["applied_families"])

    def test_canary_holdout_preserves_body_and_records_heldout_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta=routing_meta,
                decision=self._routing_decision(route_proposal={
                    "traffic_treatment": "holdout",
                    "route_selected": False,
                    "server_selected_canary_membership": False,
                }),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(routing_meta["routed_model"], "gpt-5-codex")
        self.assertEqual(result["status"], "heldout")
        self.assertEqual(result["apply_reason"], "server-canary-holdout")
        self.assertEqual(result["routing"]["status"], "heldout")
        self.assertEqual(result["routing"]["target_model"], "gpt-5-mini")
        self.assertEqual(result["outcome_feedback"]["status"], "heldout")
        self.assertIn("routing", result["outcome_feedback"]["heldout_families"])

        feedback = build_managed_action_feedback(result, source_surface="openai_responses")
        self.assertEqual(feedback["local_result"], "heldout")
        self.assertEqual(feedback["server_traffic_treatment"], "holdout")
        self.assertFalse(feedback["server_route_selected"])
        self.assertIn("routing", feedback["heldout_families"])

    def test_observe_treatment_preserves_body_even_when_live_mode_allows_application(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex", "source_surface": "openai_responses"},
                decision=self._routing_decision(route_proposal={
                    "traffic_treatment": "observe",
                    "route_selected": False,
                }),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "observe-only")
        self.assertEqual(result["routing"]["status"], "held")

    def test_expired_decision_is_vetoed_before_application(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai", now=datetime(2026, 1, 2, tzinfo=timezone.utc)).execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex", "source_surface": "openai_responses"},
                decision=self._routing_decision(expires_at="2026-01-01T00:00:00+00:00"),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["apply_reason"], "expired-policy")
        self.assertEqual(result["outcome_feedback"]["status"], "vetoed")

    def test_unsupported_target_model_is_vetoed(self):
        body = {"model": "claude-sonnet-4-6", "messages": []}
        decision = {
            "enabled": True,
            "status": "received",
            "policy_id": "unsupported-target",
            "decision_id": "unsupported-1",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "routing": {
                "target_model": "claude-future-unknown",
                "route_proposal": {
                    "target_model": "claude-future-unknown",
                    "traffic_treatment": "live",
                    "route_selected": True,
                },
            },
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="anthropic").execute(
                body=body,
                routing_meta={"routed_model": "claude-sonnet-4-6", "source_surface": "anthropic_messages"},
                decision=decision,
                application_enabled=True,
                source_surface="anthropic_messages",
            )

        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["routing"]["veto_reason"], "unsupported-target-model")
        self.assertIn("routing", result["outcome_feedback"]["vetoed_families"])

    def test_unsupported_actions_are_vetoed_with_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
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
        with patch.dict(os.environ, self.managed_live_env, clear=False):
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

    def test_product_family_opt_out_vetoes_routing_with_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, {**self.managed_live_env, "TOKENCLAW_MANAGED_ROUTING": "0"}, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "routing-product-disabled",
                    "provider": "openai",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["routing"]["veto_reason"], "local-action-family-disabled")
        self.assertIn("routing", result["outcome_feedback"]["vetoed_families"])
        self.assertFalse(result["outcome_feedback"]["raw_payload_included"])

    def test_observe_only_holds_without_changing_body(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, {"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "observe_only"}, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "observe-policy",
                    "provider": "openai",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "managed-mode-observe_only")
        self.assertEqual(result["product_mode"]["mode"], "observe_only")

    def test_local_rules_only_holds_without_server_application(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(
            os.environ,
            {**self.managed_live_env, "TOKENCLAW_LOCAL_RULES_ONLY": "1"},
            clear=False,
        ):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex", "source_surface": "openai_responses"},
                decision=self._routing_decision(),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "local-rules-only")
        self.assertEqual(result["product_mode"]["mode"], "local_only")
        self.assertFalse(result["outcome_feedback"]["raw_payload_included"])

    def test_executor_disabled_holds_without_changing_forwarding_body(self):
        body = {"model": "claude-sonnet-4-6"}
        with patch.dict(os.environ, {**self.managed_live_env, "TOKENCLAW_ACTION_EXECUTOR_ENABLED": "0"}):
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
