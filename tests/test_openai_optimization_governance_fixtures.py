from __future__ import annotations

import json
import unittest

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.openai_optimization_governor import (
    attach_openai_optimization_governor,
    build_openai_optimization_governor,
    build_openai_optimization_lifecycle_event,
)


FORBIDDEN_VALUES = (
    "raw cross-family prompt secret",
    "raw cross-family response secret",
    "raw cross-family tool payload secret",
    "raw-cross-family-session",
    "req_cross_family_secret",
    "cache-key-cross-family-secret",
    "/home/lutz/private/cross_family.py",
    "managed raw recommendation secret",
)

FORBIDDEN_KEYS = (
    '"api_key"',
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages"',
    '"prompt"',
    '"provider_body"',
    '"raw_request"',
    '"raw_response"',
    '"request_id"',
    '"session_id"',
    '"tool_payload"',
)


def assert_governance_fixture_private(testcase: unittest.TestCase, payload: object) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for value in FORBIDDEN_VALUES:
        testcase.assertNotIn(value, rendered)
    for key in FORBIDDEN_KEYS:
        testcase.assertNotIn(key, rendered)
    testcase.assertEqual(managed_egress_violations(payload), [])


class OpenAIOptimizationGovernanceFixtureTests(unittest.TestCase):
    def test_managed_metadata_without_evidence_fails_closed_and_strips_raw_fields(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "reason": "managed route requested without local canary evidence",
            "text_chars": 12_000,
            "has_tools": True,
            "stream": False,
            "category": "tool-light",
            "policy_source": "managed-recommended",
            "managed_policy_id": "managed raw recommendation secret /home/lutz/private/cross_family.py",
            "request_id": "req_cross_family_secret",
            "raw_request": {"prompt": "raw cross-family prompt secret"},
            "provider_body": {"input": "raw cross-family prompt secret"},
        }
        summary_meta = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": True,
            "status": "applied",
            "applied": True,
            "policy_source": "managed-recommended",
            "candidate_id": "managed raw recommendation secret with session raw-cross-family-session",
            "reason_codes": ["applied"],
            "raw_response": {"content": "raw cross-family response secret"},
        }
        crunch_meta = {
            "changed": True,
            "old_context_summarization": summary_meta,
            "messages": [{"content": "raw cross-family prompt secret"}],
        }
        cache_meta = {
            "status": "hit",
            "reason": "exact-match",
            "policy_source": "managed-recommended",
            "cache_key": "cache-key-cross-family-secret",
            "file_path": "/home/lutz/private/cross_family.py",
        }

        governor = attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="tool-light",
            stream=False,
            session_id="raw-cross-family-session",
        )

        self.assertEqual(governor["selected_action_family"], "none")
        self.assertEqual(governor["selected_action_families"], [])
        for family in ("routing", "old_context_summary", "cache_replay"):
            self.assertFalse(governor["family_status"][family]["selected"])
        suppressed = {item["family"]: item["reason_codes"] for item in governor["suppressed_families"]}
        self.assertIn("missing-canary-evidence", suppressed["routing"])
        self.assertIn("missing-canary-evidence", suppressed["old_context_summary"])
        self.assertIn("missing-canary-evidence", suppressed["cache_replay"])
        assert_governance_fixture_private(self, governor)

        event = build_openai_optimization_lifecycle_event(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            status_code=200,
            latency_ms=300,
            retry_count=0,
            cost_est_usd=0.004,
            cost_baseline_usd=0.006,
            category="tool-light",
            call_id="req_cross_family_secret",
        )

        self.assertIsNotNone(event)
        assert event is not None
        by_family = {item["action_family"]: item for item in event["family_events"]}
        self.assertEqual(by_family["routing"]["cohort"], "suppressed")
        self.assertIn("missing-canary-evidence", by_family["routing"]["reason_codes"])
        self.assertTrue(by_family["routing"]["policy_id"].startswith("sha256:"))
        self.assertTrue(by_family["old_context_summary"]["candidate_id"].startswith("sha256:"))
        self.assertTrue(event["local_call_hash"].startswith("sha256:"))
        assert_governance_fixture_private(self, event)

    def test_malformed_managed_sections_cannot_force_multiple_conflicting_actions(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "text_chars": 2400,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "managed_recommendation": {
                "selected_action_families": ["routing", "old_context_summary", "cache_replay"],
                "raw_request": {"prompt": "raw cross-family prompt secret"},
            },
            "openai_canary": {
                "enabled": True,
                "status": "applied",
                "cohort": "canary_applied",
                "reason": "selected-canary",
                "policy_source": "managed-recommended",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
                "candidate_id": "openai-routing-governance-candidate",
            },
        }
        summary_meta = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": True,
            "status": "applied",
            "applied": True,
            "policy_source": "managed-recommended",
            "candidate_id": "openai-summary-governance-candidate",
            "reason_codes": ["applied"],
            "canary": {"enabled": True, "cohort": "canary_applied"},
            "managed_recommendation": {
                "force_apply": True,
                "tool_payload": {"arguments": "raw cross-family tool payload secret"},
            },
        }
        crunch_meta = {"changed": True, "old_context_summarization": summary_meta}
        cache_meta = {
            "status": "hit",
            "reason": "exact-match",
            "policy_source": "managed-recommended",
            "pattern_rule": {
                "rule_id": "openai-cache-governance-rule",
                "candidate_id": "openai-cache-governance-candidate",
                "policy_source": "managed-recommended",
            },
            "cache_replay_canary": {
                "status": "applied",
                "reason": "dependency-stable",
                "canary_cohort": "canary_applied",
                "policy_source": "managed-recommended",
            },
            "managed_recommendation": {
                "selected_action_families": ["cache_replay", "routing"],
                "cache_key": "cache-key-cross-family-secret",
            },
        }

        governor = build_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            session_id="raw-cross-family-session",
        )

        self.assertEqual(governor["selected_action_family"], "routing")
        self.assertEqual(governor["selected_action_families"], ["routing"])
        suppressed = {item["family"]: item["reason_codes"] for item in governor["suppressed_families"]}
        self.assertIn("conflicts-with-selected-family", suppressed["old_context_summary"])
        self.assertIn("conflicts-with-selected-family", suppressed["cache_replay"])
        assert_governance_fixture_private(self, governor)

    def test_stale_holdout_and_streaming_tool_cache_replay_suppressions_are_explicit(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "text_chars": 48_000,
            "has_tools": True,
            "stream": True,
            "category": "tool-result",
            "openai_canary": {
                "enabled": True,
                "status": "holdout",
                "cohort": "canary_holdout",
                "reason": "selected-holdout",
                "policy_source": "managed-recommended",
            },
        }
        summary_meta = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": True,
            "status": "safety_stopped",
            "applied": False,
            "policy_source": "managed-recommended",
            "candidate_id": "summary-stale-governance-candidate",
            "reason_codes": ["stale-evidence"],
            "canary": {"enabled": True, "cohort": "canary_applied"},
        }
        crunch_meta = {"old_context_summarization": summary_meta}
        cache_meta = {
            "status": "bypassed",
            "reason": "streaming",
            "policy_source": "managed-recommended",
            "pattern_rule": {
                "rule_id": "cache-streaming-tool-governance-rule",
                "candidate_id": "cache-streaming-tool-governance-candidate",
                "policy_source": "managed-recommended",
                "action": {"type": "exact_cache_pattern", "allow_tool_calls": False},
            },
            "cache_replay_canary": {
                "status": "safety_stopped",
                "reason": "stale-evidence",
                "canary_cohort": "canary_applied",
                "policy_source": "managed-recommended",
            },
        }

        governor = attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="tool-result",
            stream=True,
            session_id="raw-cross-family-session",
        )

        self.assertEqual(governor["selected_action_family"], "none")
        suppressed = {item["family"]: item["reason_codes"] for item in governor["suppressed_families"]}
        self.assertIn("missing-holdout", suppressed["routing"])
        self.assertIn("stale-evidence", suppressed["old_context_summary"])
        self.assertIn("streaming-unsupported", suppressed["cache_replay"])
        self.assertIn("stale-evidence", suppressed["cache_replay"])

        event = build_openai_optimization_lifecycle_event(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            latency_ms=700,
            retry_count=0,
            cost_est_usd=0.01,
            cost_baseline_usd=0.01,
            category="tool-result",
            stream=True,
            call_id="req_cross_family_secret",
        )

        self.assertIsNotNone(event)
        assert event is not None
        by_family = {item["action_family"]: item for item in event["family_events"]}
        self.assertEqual(by_family["routing"]["cohort"], "holdout")
        self.assertEqual(by_family["old_context_summary"]["cohort"], "safety_stop")
        self.assertEqual(by_family["cache_replay"]["cohort"], "safety_stop")
        self.assertIn("streaming-unsupported", by_family["cache_replay"]["reason_codes"])
        self.assertIn("stale-evidence", by_family["cache_replay"]["reason_codes"])
        assert_governance_fixture_private(self, event)


if __name__ == "__main__":
    unittest.main()
