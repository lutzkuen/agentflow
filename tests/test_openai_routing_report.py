from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.openai_routing_report import build_openai_routing_promotion_decision_report, build_openai_routing_report
from agentflow_proxy.stats import stats_openai_routing_report
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class OpenAIRoutingReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_openai_call(
        self,
        *,
        requested_model: str = "gpt-5.4-mini",
        routed_model: str | None = None,
        requested_model_family: str | None = None,
        category: str = "chat",
        text_chars: int = 1200,
        stream: int = 0,
        has_tools: bool = False,
        status_code: int = 200,
        retry_count: int = 0,
        session_id: str = "secret-openai-session",
        request_json: str | None = None,
        openai_canary: dict[str, object] | None = None,
    ) -> None:
        routed_model = routed_model or requested_model
        actual_input_tokens = max(1, text_chars // 4)
        actual_output_tokens = 40
        routing_json = {
            "enabled": bool(openai_canary),
            "provider": "openai",
            "requested_model": requested_model,
            "routed_model": routed_model,
            "reason": "openai routing disabled",
            "text_chars": text_chars,
            "has_tools": has_tools,
            "category": category,
            "policy_source": "local-default",
        }
        if openai_canary is not None:
            routing_json["openai_canary"] = openai_canary
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model=requested_model,
            routed_model=routed_model,
            stream=stream,
            cache_hit=0,
            status_code=status_code,
            latency_ms=125,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=actual_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            cost_est_usd=0.001 if routed_model != requested_model else 0.002,
            cost_baseline_usd=0.002,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json(routing_json),
            cache_json=stable_json(
                {
                    "status": "skipped" if has_tools else "miss",
                    "reason": "tools-disabled" if has_tools else "exact-miss",
                    "policy_source": "local-default",
                }
            ),
            error=None,
            request_json=request_json,
            response_json=None,
            session_id=session_id,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family=requested_model_family or ("other" if requested_model.startswith("unknown") else "gpt-5"),
            routed_model_family="gpt-5",
        )

    def test_report_surfaces_disabled_openai_routing_candidates_without_raw_fields(self) -> None:
        for _ in range(6):
            self._log_openai_call(category="chat", text_chars=1200, request_json='{"input":"secret raw prompt"}')
        for _ in range(5):
            self._log_openai_call(category="short-completion", text_chars=700)
        self._log_openai_call(category="tool-light", text_chars=1100, has_tools=True)
        self._log_openai_call(category="chat", text_chars=900, stream=1)
        self._log_openai_call(requested_model="unknown-openai-model", category="chat", text_chars=900)

        result = build_openai_routing_report(self.store, limit=50)

        self.assertEqual(result["schema"], "agentflow.openai_routing_opportunity.v1")
        self.assertEqual(result["summary"]["openai_call_count"], 14)
        self.assertGreaterEqual(result["summary"]["candidate_count"], 2)
        self.assertEqual(result["summary"]["current_routed_count"], 0)
        self.assertGreater(result["summary"]["matched_count"], 0)
        self.assertGreater(result["summary"]["projected_savings_usd"], 0)
        self.assertEqual(result["summary"]["suggested_canary_fraction"], 0.05)

        chat_candidate = next(
            row for row in result["candidates"] if row["category"] == "chat" and row["blocked_count"] == 0
        )
        short_candidate = next(row for row in result["candidates"] if row["category"] == "short-completion")
        self.assertEqual(chat_candidate["matched_count"], 6)
        self.assertEqual(chat_candidate["target_model"], "gpt-5-mini")
        self.assertEqual(short_candidate["target_model"], "gpt-5-nano")

        blockers = {row["value"]: row["count"] for row in result["blocker_reason_breakdown"]}
        self.assertIn("tools-disabled", blockers)
        self.assertIn("unknown-model-family", blockers)
        self.assertIn("stream-only-evidence", blockers)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertNotIn("secret raw prompt", rendered)
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_messages_included"])
        self.assertFalse(result["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(result["privacy"]["tool_payloads_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["secrets_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])

    def test_report_preserves_gpt54_large_to_gpt54_mini_pass_through_candidate(self) -> None:
        for _ in range(6):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="chat",
                text_chars=1200,
            )

        result = build_openai_routing_report(self.store, limit=20)

        candidate = result["candidates"][0]
        self.assertEqual(candidate["requested_model"], "gpt-5.4")
        self.assertEqual(candidate["target_model"], "gpt-5.4-mini")
        self.assertEqual(candidate["current_routed_count"], 0)
        self.assertEqual(candidate["blocked_count"], 0)
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)
        readiness = candidate["promotion_readiness"]
        self.assertEqual(readiness["decision"], "keep-staged")
        self.assertIn("missing-canary-lifecycle-evidence", readiness["reason_codes"])
        self.assertEqual(readiness["evidence"]["applied_count"], 0)
        self.assertEqual(readiness["evidence"]["holdout_count"], 0)
        self.assertEqual(readiness["routing_rule_metadata"]["target_local_rule_file"], "routing_rules.yaml")

    def test_report_names_gpt54_canary_lifecycle_coverage_and_blockers(self) -> None:
        def canary(cohort: str, **extra: object) -> dict[str, object]:
            status = "applied" if cohort == "canary_applied" else "holdout"
            if cohort == "safety_stopped":
                status = "safety_stopped"
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-gpt54-mini",
                "target_candidate_id": "openai-route:responses:gpt-5:chat:no-tools:nonstream:lt-1_5k:lt-1k:to-gpt-5-4-mini",
                "status": status,
                "cohort": cohort,
                "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if cohort == "canary_applied" else "gpt-5.4",
                "category": "chat",
                "projected_input_savings_usd": 0.001,
                "canary_fraction": 0.5,
                "holdout_fraction": 0.25,
                **extra,
            }

        for _ in range(2):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="chat",
                text_chars=1200,
                openai_canary=canary("canary_applied"),
            )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            category="chat",
            text_chars=1200,
            status_code=500,
            retry_count=1,
            openai_canary=canary("canary_applied", fallback_reason="rate_limited"),
        )
        for _ in range(2):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="chat",
                text_chars=1200,
                openai_canary=canary("canary_holdout"),
            )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            category="chat",
            text_chars=1200,
            openai_canary=canary(
                "safety_stopped",
                reason="safety-stop-tripped",
                safety_stop={"tripped": True, "reason_codes": ["error-rate"]},
            ),
        )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            category="chat",
            text_chars=1200,
            openai_canary=canary("skipped", status="not_selected", reason="outside-canary-fraction"),
        )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            category="chat",
            text_chars=1200,
            openai_canary=canary("bypassed_or_disabled", status="disabled", reason="disabled"),
        )

        result = build_openai_routing_report(self.store, limit=20)

        candidate = result["candidates"][0]
        self.assertEqual(candidate["requested_model"], "gpt-5.4")
        self.assertEqual(candidate["target_model"], "gpt-5.4-mini")
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)
        lifecycle = candidate["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["schema"], "agentflow.openai_routing_canary_lifecycle_evidence.v1")
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 3)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 2)
        self.assertEqual(lifecycle["cohort_counts"]["safety_stopped"], 1)
        self.assertEqual(lifecycle["error_count"], 1)
        self.assertEqual(lifecycle["retry_count"], 1)
        self.assertEqual(lifecycle["fallback_count"], 1)
        self.assertEqual(lifecycle["cohort_counts"]["skipped"], 1)
        self.assertEqual(lifecycle["cohort_counts"]["bypassed_or_disabled"], 1)
        self.assertEqual(lifecycle["coverage"]["matched_count"], 8)
        readiness = candidate["promotion_readiness"]
        self.assertEqual(readiness["decision"], "blocked")
        self.assertEqual(readiness["evidence"]["applied_count"], 3)
        self.assertEqual(readiness["evidence"]["holdout_count"], 2)
        self.assertEqual(readiness["evidence"]["observed_count"], 8)
        self.assertEqual(readiness["evidence"]["safety_stop_count"], 1)
        self.assertEqual(readiness["evidence"]["error_count"], 1)
        self.assertEqual(readiness["evidence"]["fallback_count"], 1)
        self.assertEqual(readiness["evidence"]["retry_count"], 1)
        self.assertIn("safety-stop-observed", readiness["reason_codes"])
        self.assertIn("error-observed", readiness["reason_codes"])
        self.assertIn("fallback-observed", readiness["reason_codes"])
        self.assertIn("retry-observed", readiness["reason_codes"])
        self.assertIn("error-observed", lifecycle["blocker_codes"])
        self.assertIn("retry-observed", lifecycle["blocker_codes"])
        self.assertIn("fallback-observed", lifecycle["blocker_codes"])
        self.assertIn("safety-stop-observed", lifecycle["blocker_codes"])
        self.assertEqual(result["summary"]["openai_canary_applied_count"], 3)
        self.assertEqual(result["summary"]["openai_canary_holdout_count"], 2)
        self.assertEqual(result["summary"]["openai_canary_safety_stopped_count"], 1)
        self.assertEqual(result["summary"]["openai_canary_skipped_count"], 1)
        self.assertEqual(result["summary"]["openai_canary_bypassed_or_disabled_count"], 1)
        self.assertEqual(result["summary"]["openai_canary_stale_evidence_count"], 0)
        self.assertGreater(result["summary"]["estimated_savings_per_1000_calls_usd"], 0)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertNotIn("secret raw prompt", rendered)
        self.assertFalse(lifecycle["privacy"]["raw_prompts_included"])
        self.assertFalse(lifecycle["privacy"]["request_ids_included"])
        self.assertFalse(lifecycle["privacy"]["session_ids_included"])
        self.assertFalse(lifecycle["privacy"]["cache_keys_included"])
        self.assertFalse(readiness["privacy"]["raw_prompts_included"])
        self.assertFalse(readiness["privacy"]["provider_bodies_included"])
        self.assertFalse(readiness["privacy"]["request_ids_included"])
        self.assertFalse(readiness["privacy"]["session_ids_included"])
        self.assertFalse(readiness["privacy"]["tool_payloads_included"])

    def test_gpt54_tool_light_canary_classifies_skipped_and_unknown_before_promotion(self) -> None:
        def canary(canary_cohort: str, **extra: object) -> dict[str, object]:
            status = "applied" if canary_cohort == "canary_applied" else "holdout"
            if canary_cohort == "skipped":
                status = "not_selected"
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:1_5k-6k:1k-4k:to-gpt-5-4-mini",
                "status": status,
                "cohort": canary_cohort,
                "reason": "outside-canary-fraction" if canary_cohort == "skipped" else "selected-canary",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if canary_cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "canary_fraction": 0.15,
                "holdout_fraction": 0.10,
                **extra,
            }

        for _ in range(6):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
            )
        for _ in range(7):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_holdout", reason="selected-holdout"),
            )
        for _ in range(3):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("skipped"),
            )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            category="tool-light",
            text_chars=4000,
            has_tools=True,
            openai_canary=canary("unknown", status="mystery", cohort="mystery", reason="missing-status"),
        )

        result = build_openai_routing_report(self.store, limit=30)

        candidate = result["candidates"][0]
        lifecycle = candidate["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 6)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 7)
        self.assertEqual(lifecycle["cohort_counts"]["skipped"], 3)
        self.assertEqual(lifecycle["cohort_counts"]["unknown"], 1)
        self.assertEqual(lifecycle["skipped_reason_breakdown"], [{"value": "outside-canary-fraction", "count": 3}])
        self.assertEqual(lifecycle["unknown_reason_breakdown"], [{"value": "missing-status", "count": 1}])
        classification = lifecycle["skipped_unknown_classification"]
        self.assertEqual(classification["safe_bypass_count"], 3)
        self.assertEqual(classification["unclassified_count"], 1)
        self.assertTrue(classification["requires_operator_review"])

        readiness = candidate["promotion_readiness"]
        self.assertEqual(readiness["decision"], "keep-staged")
        self.assertFalse(readiness["promotion_ready"])
        self.assertEqual(readiness["next_action"], "classify-openai-routing-canary-skipped-unknown")
        self.assertIn("unknown-canary-lifecycle-rows", readiness["reason_codes"])
        self.assertEqual(readiness["evidence"]["applied_count"], 6)
        self.assertEqual(readiness["evidence"]["holdout_count"], 7)
        self.assertEqual(readiness["evidence"]["skipped_count"], 3)
        self.assertEqual(readiness["evidence"]["unknown_count"], 1)
        self.assertTrue(readiness["quality_gates"]["requires_classified_skipped_unknown_rows"])
        self.assertEqual(result["summary"]["promotion_ready_count"], 0)
        self.assertEqual(result["summary"]["keep_staged_count"], 1)

    def test_gpt54_tool_light_canary_narrows_unsupported_skipped_rows_before_promotion(self) -> None:
        def canary(canary_cohort: str, **extra: object) -> dict[str, object]:
            status = "applied" if canary_cohort == "canary_applied" else "holdout"
            if canary_cohort == "skipped":
                status = "skipped"
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:1_5k-6k:1k-4k:to-gpt-5-4-mini",
                "status": status,
                "cohort": canary_cohort,
                "reason": "request-too-large" if canary_cohort == "skipped" else "selected-canary",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if canary_cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "canary_fraction": 0.15,
                "holdout_fraction": 0.10,
                **extra,
            }

        for _ in range(6):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
            )
        for _ in range(7):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_holdout", reason="selected-holdout"),
            )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            category="tool-light",
            text_chars=4000,
            has_tools=True,
            openai_canary=canary("skipped"),
        )

        result = build_openai_routing_report(self.store, limit=30)

        candidate = result["candidates"][0]
        lifecycle = candidate["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["skipped_reason_breakdown"], [{"value": "request-too-large", "count": 1}])
        classification = lifecycle["skipped_unknown_classification"]
        self.assertEqual(classification["unsupported_shape_count"], 1)
        readiness = candidate["promotion_readiness"]
        self.assertEqual(readiness["decision"], "narrow")
        self.assertEqual(readiness["next_action"], "narrow-openai-routing-canary-shape")
        self.assertIn("skipped-canary-unsupported-shape", readiness["reason_codes"])
        self.assertEqual(result["summary"]["promotion_ready_count"], 0)

    def test_gpt54_tool_light_canary_promotion_readiness_promotes_healthy_evidence(self) -> None:
        def canary(cohort: str) -> dict[str, object]:
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:1_5k-6k:1k-4k:to-gpt-5-4-mini",
                "status": "applied" if cohort == "canary_applied" else "holdout",
                "cohort": cohort,
                "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "canary_fraction": 0.15,
                "holdout_fraction": 0.10,
            }

        for _ in range(6):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
            )
        for _ in range(7):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_holdout"),
            )

        result = build_openai_routing_report(self.store, limit=20)

        candidate = result["candidates"][0]
        self.assertEqual(candidate["requested_model"], "gpt-5.4")
        self.assertEqual(candidate["target_model"], "gpt-5.4-mini")
        self.assertEqual(candidate["category"], "tool-light")
        self.assertEqual(candidate["blocked_count"], 0)
        self.assertEqual(candidate["blockers"], [])
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)

        lifecycle = candidate["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 6)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 7)
        self.assertEqual(lifecycle["cohort_counts"]["safety_stopped"], 0)
        self.assertEqual(lifecycle["error_count"], 0)
        self.assertEqual(lifecycle["fallback_count"], 0)
        self.assertEqual(lifecycle["retry_count"], 0)
        self.assertEqual(lifecycle["blocker_codes"], [])

        readiness = candidate["promotion_readiness"]
        self.assertEqual(readiness["decision"], "promote")
        self.assertTrue(readiness["promotion_ready"])
        self.assertEqual(readiness["next_action"], "promote-openai-routing-rule-draft")
        self.assertEqual(readiness["reason"], "promotion-ready")
        self.assertEqual(readiness["reason_codes"], [])
        self.assertEqual(readiness["evidence"]["applied_count"], 6)
        self.assertEqual(readiness["evidence"]["holdout_count"], 7)
        self.assertEqual(readiness["evidence"]["observed_count"], 13)
        self.assertEqual(readiness["evidence"]["safety_stop_count"], 0)
        self.assertEqual(readiness["evidence"]["error_count"], 0)
        self.assertEqual(readiness["evidence"]["fallback_count"], 0)
        self.assertEqual(readiness["evidence"]["retry_count"], 0)
        self.assertGreater(readiness["evidence"]["estimated_savings_per_1000_calls_usd"], 0)
        rule = readiness["routing_rule_metadata"]["rule_preview"]
        self.assertEqual(rule["conditions"]["model_pattern"], "gpt-5.4")
        self.assertEqual(rule["conditions"]["category"], "tool-light")
        self.assertTrue(rule["conditions"]["has_tools"])
        self.assertEqual(rule["action"]["route_to"], "gpt-5.4-mini")
        self.assertEqual(result["summary"]["promotion_ready_count"], 1)
        self.assertEqual(result["summary"]["keep_staged_count"], 0)
        self.assertEqual(result["summary"]["keep_blocked_count"], 0)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertFalse(readiness["privacy"]["raw_prompts_included"])
        self.assertFalse(readiness["privacy"]["provider_bodies_included"])
        self.assertFalse(readiness["privacy"]["request_ids_included"])
        self.assertFalse(readiness["privacy"]["session_ids_included"])
        self.assertFalse(readiness["privacy"]["tool_payloads_included"])
        self.assertFalse(readiness["privacy"]["cache_keys_included"])

    def test_targeted_gpt54_tool_light_promotion_decision_aggregates_lifecycle_buckets(self) -> None:
        def canary(cohort: str) -> dict[str, object]:
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:to-gpt-5-4-mini",
                "status": "applied" if cohort == "canary_applied" else "holdout",
                "cohort": cohort,
                "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "canary_fraction": 0.15,
                "holdout_fraction": 0.10,
            }

        for _ in range(4):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
            )
        for _ in range(5):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_holdout"),
            )
        for _ in range(3):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=9000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
            )
        for _ in range(2):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=9000,
                has_tools=True,
                openai_canary=canary("canary_holdout"),
            )

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        self.assertEqual(result["schema"], "agentflow.openai_routing_promotion_decision_report.v1")
        self.assertEqual(result["decision"], "promote")
        self.assertEqual(result["summary"]["decision_count"], 1)
        self.assertEqual(len(result["decisions"]), 1)
        decision = result["promotion_decision"]
        self.assertEqual(decision["schema"], "agentflow.openai_routing_promotion_decision.v1")
        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["target"]["source_surface"], "openai_responses")
        self.assertEqual(decision["target"]["endpoint"], "responses")
        self.assertEqual(decision["target"]["category"], "tool-light")
        self.assertEqual(decision["target"]["requested_model"], "gpt-5.4")
        self.assertEqual(decision["target"]["target_model"], "gpt-5.4-mini")
        self.assertEqual(decision["target"]["target_local_policy_section"], "routing.rules")
        self.assertEqual(decision["target"]["target_local_rule_file"], "routing_rules.yaml")
        self.assertGreaterEqual(decision["candidate_count"], 2)
        self.assertEqual(decision["matched_count"], 14)
        self.assertEqual(decision["lifecycle"]["applied_count"], 7)
        self.assertEqual(decision["lifecycle"]["holdout_count"], 7)
        self.assertEqual(decision["lifecycle"]["safety_stop_count"], 0)
        self.assertEqual(decision["lifecycle"]["error_count"], 0)
        self.assertEqual(decision["lifecycle"]["fallback_count"], 0)
        self.assertEqual(decision["lifecycle"]["retry_count"], 0)
        self.assertGreater(decision["savings_per_1000_calls_usd"], 0)
        self.assertEqual(decision["reason_codes"], [])
        self.assertEqual(decision["routing_rule_metadata"]["target_local_policy_section"], "routing.rules")
        self.assertFalse(decision["privacy"]["provider_calls_made"])
        self.assertFalse(decision["privacy"]["managed_server_calls_made"])
        self.assertFalse(decision["privacy"]["policy_files_written"])
        self.assertNotIn("secret-openai-session", json.dumps(result, sort_keys=True))

        output = io.StringIO()
        exit_code = cli.openai_routing_report_cli(
            ["--db", self.db_path, "--limit", "20", "--promotion-decision"],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.openai_routing_promotion_decision_report.v1")
        self.assertEqual(payload["decision"], "promote")
        self.assertEqual(payload["summary"]["applied_count"], 7)
        self.assertEqual(payload["summary"]["holdout_count"], 7)

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        for _ in range(5):
            self._log_openai_call(category="chat", text_chars=1200)

        result = asyncio.run(stats_openai_routing_report(self.store, limit=10))
        self.assertEqual(result["schema"], "agentflow.openai_routing_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.openai_routing_report_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.openai_routing_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 5)
        self.assertNotIn("secret-openai-session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
