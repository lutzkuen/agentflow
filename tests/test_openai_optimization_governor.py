from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from tokenclaw.openai_optimization_governor import (
    attach_openai_optimization_governor,
    build_openai_optimization_governor,
    build_openai_optimization_lifecycle_event,
    selected_openai_governor_family,
)
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.optimization.openai_outcomes import record_managed_outcome_feedback


FORBIDDEN_VALUES = (
    "raw-governor-prompt-secret",
    "raw-governor-response-secret",
    "req_governor_secret",
    "cache-key-governor-secret",
    "/home/lutz/private/governor_secret.py",
    "raw-governor-session",
)

FORBIDDEN_KEYS = (
    '"api_key"',
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages"',
    '"prompt"',
    '"raw_request"',
    '"request_id"',
    '"session_id"',
    '"tool_payload"',
)


def _assert_governor_privacy_clean(testcase: unittest.TestCase, payload: object) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for value in FORBIDDEN_VALUES:
        testcase.assertNotIn(value, rendered)
    for key in FORBIDDEN_KEYS:
        testcase.assertNotIn(key, rendered)


class FakeLifecycleStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.updated: list[tuple[str, dict[str, object]]] = []

    def enqueue_managed_outcome_feedback(self, **kwargs: object) -> None:
        self.rows.append(dict(kwargs))

    def get_managed_outcome_feedback(self, queue_id: str) -> dict[str, object] | None:
        for row in self.rows:
            if row.get("id") == queue_id:
                return row
        return None

    def update_call_routing_json(self, call_id: str, routing_json: str) -> None:
        self.updated.append((call_id, json.loads(routing_json)))


class OpenAIOptimizationGovernorTests(unittest.TestCase):
    def test_multiple_eligible_families_selects_exactly_one_and_suppresses_conflicts(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "reason": "OpenAI canary selected local route",
            "text_chars": 2200,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "policy_source": "local-manual",
            "openai_canary": {
                "enabled": True,
                "status": "applied",
                "cohort": "canary_applied",
                "reason": "selected-canary",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini",
                "policy_source": "local-manual",
            },
            "request_id": "req_governor_secret",
            "prompt": "raw-governor-prompt-secret",
        }
        summary_meta = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": True,
            "status": "applied",
            "applied": True,
            "reason_codes": ["applied"],
            "policy_source": "local-manual",
            "raw_request": {"messages": [{"content": "raw-governor-prompt-secret"}]},
        }
        crunch_meta = {
            "changed": True,
            "old_context_summarization": summary_meta,
            "content": "raw-governor-prompt-secret",
        }
        cache_meta = {
            "status": "hit",
            "reason": "exact-match",
            "policy_source": "local-default",
            "cache_key": "cache-key-governor-secret",
            "cache_replay_canary": {
                "status": "applied",
                "reason": "dependency-stable",
                "canary_cohort": "canary_applied",
                "policy_source": "local-manual",
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
            session_id="raw-governor-session",
        )

        self.assertEqual(
            set(governor["eligible_action_families"]),
            {"routing", "old_context_summary", "cache_replay"},
        )
        self.assertEqual(governor["selected_action_family"], "routing")
        self.assertEqual(governor["selected_action_families"], ["routing"])
        self.assertEqual(len(governor["selected_action_families"]), 1)
        suppressed = {item["family"]: item["reason_codes"] for item in governor["suppressed_families"]}
        self.assertIn("conflicts-with-selected-family", suppressed["old_context_summary"])
        self.assertIn("conflicts-with-selected-family", suppressed["cache_replay"])
        self.assertEqual(governor["canary"]["cohort"], "canary_applied")
        self.assertFalse(governor["privacy"]["raw_prompt_included"])
        self.assertFalse(governor["privacy"]["provider_body_included"])
        _assert_governor_privacy_clean(self, governor)

    def test_noop_reports_holdout_provider_and_cache_replay_suppression_reasons(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "text_chars": 12000,
            "has_tools": True,
            "stream": True,
            "category": "tool-light",
            "openai_canary": {
                "enabled": True,
                "status": "holdout",
                "cohort": "canary_holdout",
                "reason": "selected-holdout",
                "policy_source": "local-manual",
            },
        }
        summary_meta = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": True,
            "status": "skipped",
            "applied": False,
            "reason_codes": ["summary_fetch_error"],
            "policy_source": "local-manual",
        }
        crunch_meta = {"old_context_summarization": summary_meta}
        cache_meta = {
            "status": "bypassed",
            "reason": "file-dependency-missing",
            "pattern_rule": {
                "rule_id": "cache-replay-rule",
                "candidate_id": "cache-replay-candidate",
                "policy_source": "local-manual",
            },
            "cache_replay_canary": {
                "status": "bypassed",
                "reason": "file-dependency-missing",
                "canary_cohort": "canary_applied",
            },
        }

        governor = build_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/chat/completions",
            requested_model="gpt-5.4",
            category="tool-light",
            stream=True,
            session_id="raw-governor-session",
        )

        self.assertEqual(governor["selected_action_family"], "none")
        self.assertEqual(governor["selected_action_families"], [])
        suppressed = {item["family"]: item["reason_codes"] for item in governor["suppressed_families"]}
        self.assertIn("missing-holdout", suppressed["routing"])
        self.assertIn("summary-provider-unavailable", suppressed["old_context_summary"])
        self.assertIn("cache-replay-invalidation-missing", suppressed["cache_replay"])
        self.assertEqual(governor["family_status"]["cache_replay"]["eligible"], True)
        _assert_governor_privacy_clean(self, governor)

    def test_shared_governor_holdout_blocks_otherwise_selected_actions(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "text_chars": 900,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "openai_canary": {
                "enabled": True,
                "status": "applied",
                "cohort": "canary_applied",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
            },
        }
        with patch.dict(
            "os.environ",
            {
                "TOKENCLAW_OPENAI_OPTIMIZATION_GOVERNOR_CANARY_FRACTION": "0",
                "TOKENCLAW_OPENAI_OPTIMIZATION_GOVERNOR_HOLDOUT_FRACTION": "1",
            },
        ):
            governor = build_openai_optimization_governor(
                routing_meta=routing_meta,
                crunch_meta={},
                cache_meta={"status": "miss"},
                path="/v1/responses",
                requested_model="gpt-5.4",
                category="chat",
                stream=False,
                session_id="raw-governor-session",
            )

        self.assertEqual(governor["canary"]["cohort"], "governor_holdout")
        self.assertEqual(governor["selected_action_family"], "none")
        self.assertIn("missing-holdout", governor["suppressed_families"][0]["reason_codes"])
        _assert_governor_privacy_clean(self, governor)

    def test_attach_writes_same_governor_block_to_routing_crunch_and_cache_metadata(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "text_chars": 900,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "openai_canary": {
                "enabled": True,
                "status": "applied",
                "cohort": "canary_applied",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
            },
        }
        crunch_meta = {"changed": False}
        cache_meta = {"status": "miss", "reason": "exact-miss"}

        governor = attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            session_id="raw-governor-session",
        )

        self.assertEqual(selected_openai_governor_family(routing_meta), "routing")
        self.assertEqual(routing_meta["openai_optimization_governor"], governor)
        self.assertEqual(crunch_meta["openai_optimization_governor"], governor)
        self.assertEqual(cache_meta["openai_optimization_governor"], governor)
        _assert_governor_privacy_clean(self, routing_meta)

    def test_unified_lifecycle_event_covers_routing_summary_and_cache_replay(self) -> None:
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "reason": "OpenAI canary selected local route",
            "text_chars": 2200,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "openai_canary": {
                "enabled": True,
                "policy_id": "openai-routing-policy",
                "rule_id": "openai-routing-policy",
                "candidate_id": "openai-routing-candidate",
                "status": "applied",
                "cohort": "canary_applied",
                "reason": "selected-canary",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
            },
            "request_id": "req_governor_secret",
        }
        summary_meta = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": True,
            "status": "applied",
            "applied": True,
            "rule_id": "openai-summary-rule",
            "candidate_id": "openai-summary-candidate",
            "reason_codes": ["applied"],
            "summary": "raw-governor-prompt-secret",
        }
        crunch_meta = {
            "changed": True,
            "old_context_summarization": summary_meta,
            "messages": [{"content": "raw-governor-prompt-secret"}],
        }
        cache_meta = {
            "status": "invalidated",
            "reason": "dependency-changed",
            "cache_key": "cache-key-governor-secret",
            "cache_replay_canary": {
                "rule_id": "openai-cache-rule",
                "candidate_id": "openai-cache-candidate",
                "status": "invalidated",
                "reason": "dependency-changed",
                "canary_cohort": "canary_applied",
            },
        }
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            session_id="raw-governor-session",
        )

        event = build_openai_optimization_lifecycle_event(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            status_code=200,
            latency_ms=1250,
            retry_count=1,
            cost_est_usd=0.001,
            cost_baseline_usd=0.004,
            category="chat",
            call_id="req_governor_secret",
        )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["schema"], "tokenclaw.openai_optimization_lifecycle_feedback.v1")
        by_family = {item["action_family"]: item for item in event["family_events"]}
        self.assertEqual(by_family["routing"]["cohort"], "applied")
        self.assertEqual(by_family["old_context_summary"]["cohort"], "suppressed")
        self.assertEqual(by_family["cache_replay"]["cohort"], "invalidated")
        self.assertEqual(by_family["routing"]["candidate_id"], "openai-routing-candidate")
        self.assertEqual(by_family["old_context_summary"]["rule_id"], "openai-summary-rule")
        self.assertEqual(by_family["cache_replay"]["candidate_id"], "openai-cache-candidate")
        self.assertEqual(event["status_bucket"], "2xx")
        self.assertEqual(event["retry_bucket"], "one")
        self.assertAlmostEqual(event["observed_savings_estimate_usd"], 0.003)
        self.assertEqual(managed_egress_violations(event), [])
        _assert_governor_privacy_clean(self, event)

    def test_record_feedback_queues_unified_lifecycle_metadata_when_managed_disabled(self) -> None:
        store = FakeLifecycleStore()
        routing_meta = {
            "provider": "openai",
            "enabled": True,
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4-mini",
            "text_chars": 900,
            "has_tools": False,
            "stream": False,
            "category": "chat",
            "openai_canary": {
                "enabled": True,
                "policy_id": "openai-routing-policy",
                "candidate_id": "openai-routing-candidate",
                "status": "applied",
                "cohort": "canary_applied",
                "requested_model": "gpt-5.4",
                "actual_forwarded_model": "gpt-5.4-mini",
                "request_id": "req_governor_secret",
            },
        }
        crunch_meta = {"changed": False, "content": "raw-governor-prompt-secret"}
        cache_meta = {"status": "miss", "reason": "exact-miss", "cache_key": "cache-key-governor-secret"}
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            session_id="raw-governor-session",
        )

        with patch.dict("os.environ", {"TOKENCLAW_RECOMMENDATION_ENABLED": "0"}):
            asyncio.run(
                record_managed_outcome_feedback(
                    store=store,
                    call_id="req_governor_secret",
                    path="/v1/responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4-mini",
                    status_code=200,
                    latency_ms=400,
                    retry_count=0,
                    input_tokens_est=100,
                    output_tokens_est=10,
                    actual_input_tokens=100,
                    actual_output_tokens=10,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    thinking_output_tokens=None,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    cache_meta=cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category="chat",
                    session_id="raw-governor-session",
                )
            )

        self.assertEqual(len(store.rows), 1)
        row = store.rows[0]
        self.assertEqual(row["source_surface"], "openai_optimization_lifecycle")
        payload = json.loads(str(row["payload_json"]))
        self.assertEqual(payload["schema"], "tokenclaw.openai_optimization_lifecycle_feedback.v1")
        self.assertEqual(payload["family_events"][0]["cohort"], "applied")
        self.assertEqual(store.updated[0][1]["openai_optimization_lifecycle_feedback"]["status"], "queued")
        self.assertFalse(store.updated[0][1]["openai_optimization_lifecycle_feedback"]["payload_included"])
        self.assertEqual(managed_egress_violations(payload), [])
        _assert_governor_privacy_clean(self, payload)


if __name__ == "__main__":
    unittest.main()
