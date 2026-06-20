from __future__ import annotations

import unittest
from unittest.mock import patch

from tokenclaw.openai_optimization_governor import (
    attach_openai_optimization_governor,
    selected_openai_governor_family,
)
from tokenclaw.optimization_coordinator_enforcement import (
    SUPPRESSION_REASON,
    enforce_optimization_coordinator,
)


class OptimizationCoordinatorEnforcementTests(unittest.TestCase):
    def test_conflicting_managed_cache_replay_and_crunch_applies_only_selected_family(self) -> None:
        routing_meta = {
            "provider": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "category": "tool-result",
            "text_chars": 12000,
        }
        crunch_meta = {
            "policy_source": "managed-recommended",
            "terminal_output_compaction": {
                "status": "applied",
                "applied": True,
                "changed": True,
                "saved_chars": 2400,
                "policy_source": "managed-recommended",
                "candidate_id": "terminal-candidate",
            },
        }
        cache_meta = {
            "status": "hit",
            "reason": "dependency-stable",
            "policy_source": "managed-recommended",
            "pattern_rule": {
                "rule_id": "cache-rule",
                "candidate_id": "cache-candidate",
                "policy_source": "managed-recommended",
            },
            "cache_replay_canary": {
                "status": "applied",
                "reason": "dependency-stable",
                "policy_source": "managed-recommended",
            },
            "safe_invalidation_evidence": True,
        }

        enforcement = enforce_optimization_coordinator(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            input_tokens_est=3000,
            category="tool-result",
            enabled=True,
        )

        self.assertEqual(enforcement["status"], "applied")
        self.assertEqual(enforcement["selected_family"], "cache_replay")
        self.assertEqual(cache_meta["status"], "hit")
        terminal_meta = crunch_meta["terminal_output_compaction"]
        self.assertEqual(terminal_meta["status"], "suppressed")
        self.assertFalse(terminal_meta["applied"])
        self.assertIn(SUPPRESSION_REASON, terminal_meta["reason_codes"])
        self.assertIn("terminal_output_compaction", enforcement["suppressed_managed_families"])
        self.assertEqual(routing_meta["optimization_coordinator"]["selected_family"], "cache_replay")
        self.assertFalse(routing_meta["optimization_coordinator"]["privacy"]["provider_bodies_included"])

    def test_suppressed_managed_cache_replay_is_disabled_before_lookup(self) -> None:
        routing_meta = {
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "reason": "managed route applied",
            "policy_source": "managed-recommended",
            "final_policy_source": "managed-recommended",
            "managed_recommendation": {
                "applied": True,
                "changed_model": True,
                "policy_source": "managed-recommended",
            },
        }
        crunch_meta = {}
        cache_meta = {
            "status": "hit",
            "reason": "dependency-stable",
            "policy_source": "managed-recommended",
            "pattern_rule": {"policy_source": "managed-recommended"},
            "cache_replay_canary": {
                "status": "applied",
                "reason": "dependency-stable",
                "policy_source": "managed-recommended",
            },
            "safe_invalidation_evidence": True,
        }
        provider_body = {"model": "gpt-5.4-mini", "input": "hello"}

        enforcement = enforce_optimization_coordinator(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            input_tokens_est=10,
            provider_body=provider_body,
            local_routed_model="gpt-5.4-mini",
            enabled=True,
        )

        self.assertEqual(enforcement["selected_family"], "cache_replay")
        self.assertNotIn("cache_replay", enforcement["suppressed_managed_families"])
        self.assertIn("routing", enforcement["suppressed_managed_families"])
        self.assertFalse(routing_meta["managed_recommendation"]["applied"])
        self.assertEqual(provider_body["model"], "gpt-5.4-mini")

    def test_holdout_suppresses_managed_actions_without_applying_family(self) -> None:
        routing_meta = {
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "policy_source": "managed-recommended",
            "final_policy_source": "managed-recommended",
            "managed_recommendation": {
                "applied": True,
                "changed_model": True,
                "policy_source": "managed-recommended",
            },
        }
        crunch_meta = {}
        cache_meta = {"status": "miss", "reason": "exact-miss"}

        with patch.dict(
            "os.environ",
            {
                "AGENTFLOW_OPTIMIZATION_COORDINATOR_HOLDOUT_FRACTION": "1",
                "AGENTFLOW_OPTIMIZATION_COORDINATOR_CANARY_FRACTION": "0",
            },
        ):
            enforcement = enforce_optimization_coordinator(
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                provider="openai",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                input_tokens_est=10,
                enabled=True,
            )

        self.assertEqual(enforcement["selected_family"], "none")
        self.assertEqual(routing_meta["optimization_coordinator"]["reason_codes"], ["coordinator-holdout"])
        self.assertEqual(
            routing_meta["managed_recommendation"]["apply_reason"],
            SUPPRESSION_REASON,
        )

    def test_error_is_noop_and_records_metadata(self) -> None:
        routing_meta = {}
        crunch_meta = {}
        cache_meta = {}
        with patch(
            "tokenclaw.optimization_coordinator_enforcement.build_optimization_coordinator",
            side_effect=RuntimeError("boom"),
        ):
            enforcement = enforce_optimization_coordinator(
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="/v1/messages",
                enabled=True,
            )

        self.assertEqual(enforcement["status"], "error")
        self.assertEqual(enforcement["reason"], "coordinator-error")
        self.assertEqual(routing_meta["optimization_coordinator_enforcement"]["status"], "error")
        self.assertNotIn("optimization_coordinator", routing_meta)

    def test_openai_governor_bridge_prefers_active_coordinator_without_losing_governor(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "text_chars": 1200,
            "policy_source": "local-manual",
            "openai_canary": {
                "enabled": True,
                "status": "applied",
                "cohort": "canary_applied",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
                "policy_source": "local-manual",
            },
        }
        crunch_meta = {}
        cache_meta = {
            "status": "hit",
            "reason": "dependency-stable",
            "policy_source": "managed-recommended",
            "pattern_rule": {"policy_source": "managed-recommended"},
            "cache_replay_canary": {
                "status": "applied",
                "reason": "dependency-stable",
                "policy_source": "managed-recommended",
            },
            "safe_invalidation_evidence": True,
        }
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            requested_model="gpt-5.4",
            category="chat",
        )

        self.assertEqual(routing_meta["openai_optimization_governor"]["selected_action_family"], "routing")

        enforce_optimization_coordinator(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            input_tokens_est=300,
            enabled=True,
        )

        self.assertIn("openai_optimization_governor", routing_meta)
        self.assertIn("optimization_coordinator", routing_meta)
        self.assertEqual(routing_meta["optimization_coordinator"]["selected_family"], "routing")
        self.assertEqual(selected_openai_governor_family(routing_meta), "routing")
        suppressed = {
            item["family"]: item["reason_codes"]
            for item in routing_meta["openai_optimization_governor"]["suppressed_families"]
        }
        self.assertIn("conflicts-with-selected-family", suppressed["cache_replay"])


if __name__ == "__main__":
    unittest.main()
