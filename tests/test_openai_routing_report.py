from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tokenclaw import cli
from tokenclaw.openai_routing_report import build_openai_routing_promotion_decision_report, build_openai_routing_report
from tokenclaw.stats import stats_openai_routing_report
from tokenclaw.store import SQLiteStore, stable_json, utc_now


class OpenAIRoutingReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
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
        latency_ms: int = 125,
        session_id: str = "secret-openai-session",
        request_json: str | None = None,
        openai_canary: dict[str, object] | None = None,
        crunch_tokens_saved_est: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
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
            latency_ms=latency_ms,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=actual_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            cost_est_usd=0.001 if routed_model != requested_model else 0.002,
            cost_baseline_usd=0.002,
            crunch_json=stable_json({"changed": bool(crunch_tokens_saved_est), "tokens_saved_est": crunch_tokens_saved_est}),
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
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family=requested_model_family or ("other" if requested_model.startswith("unknown") else "gpt-5"),
            routed_model_family="gpt-5",
        )

    def _log_openai_shadow_experiment(
        self,
        *,
        category: str = "tool-light",
        similarity: float = 0.95,
        passed: bool = True,
        requested_model: str = "gpt-5.4",
        shadow_model: str = "gpt-5.4-mini",
    ) -> None:
        self.store.log_routing_experiment(
            id=str(uuid.uuid4()),
            call_id=str(uuid.uuid4()),
            created_at=utc_now(),
            provider="openai",
            source_surface="openai_responses",
            stream=0,
            requested_model=requested_model,
            routed_model=shadow_model,
            primary_model=requested_model,
            shadow_model=shadow_model,
            category=category,
            routing_reason="shadow semantic quality fixture",
            input_tokens_est=1000,
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=100,
            shadow_latency_ms=100,
            primary_output_chars=1000,
            shadow_output_chars=980,
            primary_output_sha256="primary",
            shadow_output_sha256="shadow",
            output_similarity=similarity,
            passed_threshold=1 if passed else 0,
            primary_cost_est_usd=0.002,
            shadow_cost_est_usd=0.001,
            error=None,
            experiment_json=stable_json({"endpoint": "responses"}),
        )

    def _active_openai_tool_light_rule(self) -> dict[str, object]:
        return {
            "id": "test-promoted-openai-gpt54-tool-light-mini",
            "enabled": True,
            "policy_source": "local-promoted",
            "conditions": {
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "model_pattern": "gpt-5.4",
                "category": "tool-light",
                "has_tools": True,
                "stream": False,
            },
            "action": {
                "route_to": "gpt-5.4-mini",
                "reason": "test promoted OpenAI gpt-5.4 tool-light routing rule",
            },
            "metadata": {
                "source": "openai_routing_promotion_decision",
                "promoted_from_canary": True,
                "promotion_decision": {
                    "quality_gates": {
                        "requires_semantic_quality_pass": True,
                    },
                    "semantic_quality": {
                        "gate_passed": True,
                        "clean_comparison_count": 20,
                        "pass_count": 20,
                        "pass_rate": 1.0,
                    },
                },
            },
        }

    def _install_active_openai_tool_light_rule(self) -> None:
        from tokenclaw import router as router_module

        previous = list(getattr(router_module, "ROUTING_RULES", []))
        router_module.ROUTING_RULES = [self._active_openai_tool_light_rule()]
        self.addCleanup(setattr, router_module, "ROUTING_RULES", previous)

    def _install_disabled_openai_tool_light_rule(self, reason: str = "semantic-quality-regression-observed") -> None:
        from tokenclaw import router as router_module

        previous = list(getattr(router_module, "ROUTING_RULES", []))
        rule = self._active_openai_tool_light_rule()
        rule["enabled"] = False
        rule["disabled_reason"] = reason
        router_module.ROUTING_RULES = [rule]
        self.addCleanup(setattr, router_module, "ROUTING_RULES", previous)

    def test_report_surfaces_disabled_openai_routing_candidates_without_raw_fields(self) -> None:
        for _ in range(6):
            self._log_openai_call(category="chat", text_chars=1200, request_json='{"input":"secret raw prompt"}')
        for _ in range(5):
            self._log_openai_call(category="short-completion", text_chars=700)
        self._log_openai_call(category="tool-light", text_chars=1100, has_tools=True)
        self._log_openai_call(category="chat", text_chars=900, stream=1)
        self._log_openai_call(requested_model="unknown-openai-model", category="chat", text_chars=900)

        result = build_openai_routing_report(self.store, limit=50)

        self.assertEqual(result["schema"], "tokenclaw.openai_routing_opportunity.v1")
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
        # 2026-07 ladder: deprecated gpt-5.4 proposals target gpt-5.6-luna.
        self.assertEqual(chat_candidate["target_model"], "gpt-5.6-luna")
        self.assertEqual(short_candidate["target_model"], "gpt-5.6-luna")

        blockers = {row["value"]: row["count"] for row in result["blocker_reason_breakdown"]}
        self.assertIn("tools-disabled", blockers)
        self.assertIn("stream-only-evidence", blockers)
        # Models outside the routing ladder no longer get blocked proposals;
        # they stay unmatched.
        unmatched = {row["value"]: row["count"] for row in result["unmatched_reason_breakdown"]}
        self.assertIn("no-local-routing-shape-match", unmatched)

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
        self.assertEqual(candidate["target_model"], "gpt-5.6-luna")
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
        self.assertEqual(lifecycle["schema"], "tokenclaw.openai_routing_canary_lifecycle_evidence.v1")
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
        self.assertEqual(lifecycle["unknown_reason_breakdown"], [{"value": "canary-lifecycle-logging-gap", "count": 1}])
        classification = lifecycle["skipped_unknown_classification"]
        self.assertEqual(classification["safe_bypass_count"], 3)
        self.assertEqual(classification["promotion_blocker_count"], 1)
        self.assertEqual(classification["unclassified_count"], 0)
        self.assertTrue(classification["requires_operator_review"])

        readiness = candidate["promotion_readiness"]
        self.assertEqual(readiness["decision"], "blocked")
        self.assertFalse(readiness["promotion_ready"])
        self.assertEqual(readiness["next_action"], "review-openai-routing-canary-blockers")
        self.assertIn("skipped-canary-promotion-blocker", readiness["reason_codes"])
        self.assertNotIn("unknown-canary-lifecycle-rows", readiness["reason_codes"])
        self.assertEqual(readiness["evidence"]["applied_count"], 6)
        self.assertEqual(readiness["evidence"]["holdout_count"], 7)
        self.assertEqual(readiness["evidence"]["skipped_count"], 3)
        self.assertEqual(readiness["evidence"]["unknown_count"], 1)
        self.assertTrue(readiness["quality_gates"]["requires_classified_skipped_unknown_rows"])
        self.assertEqual(result["summary"]["promotion_ready_count"], 0)
        self.assertEqual(result["summary"]["keep_blocked_count"], 1)

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
        for _ in range(20):
            self._log_openai_shadow_experiment(category="tool-light", similarity=0.95, passed=True)

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
        self.assertTrue(readiness["evidence"]["semantic_quality"]["gate_passed"])
        self.assertEqual(readiness["evidence"]["semantic_quality"]["clean_comparison_count"], 20)
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
        self._install_active_openai_tool_light_rule()

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
        for _ in range(20):
            self._log_openai_shadow_experiment(category="tool-light", similarity=0.95, passed=True)

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        self.assertEqual(result["schema"], "tokenclaw.openai_routing_promotion_decision_report.v1")
        self.assertEqual(result["decision"], "active-local-policy")
        self.assertFalse(result["promotion_ready"])
        self.assertEqual(result["summary"]["decision_count"], 1)
        self.assertEqual(len(result["decisions"]), 1)
        decision = result["promotion_decision"]
        self.assertEqual(decision["schema"], "tokenclaw.openai_routing_promotion_decision.v1")
        self.assertEqual(decision["decision"], "active-local-policy")
        self.assertEqual(decision["promotion_verdict"], "active-local-policy")
        self.assertEqual(
            decision["promotion_verdict_options"],
            ["promotion-ready", "active-local-policy", "keep-staged", "keep-blocked", "rollback-required"],
        )
        self.assertEqual(decision["next_action"], "measure-openai-routing-rule-outcomes")
        self.assertEqual(decision["reason"], "matching-openai-routing-rule-active-in-local-policy")
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
        self.assertEqual(decision["lifecycle"]["skipped_count"], 0)
        self.assertEqual(decision["lifecycle"]["unknown_count"], 0)
        self.assertEqual(decision["lifecycle"]["safety_stop_count"], 0)
        self.assertEqual(decision["lifecycle"]["error_count"], 0)
        self.assertEqual(decision["lifecycle"]["fallback_count"], 0)
        self.assertEqual(decision["lifecycle"]["retry_count"], 0)
        self.assertGreater(decision["savings_per_1000_calls_usd"], 0)
        self.assertEqual(decision["reason_codes"], [])
        self.assertNotIn("candidate_ids", decision)
        self.assertFalse(decision["candidate_ids_included"])
        self.assertFalse(decision["candidate_set"]["candidate_ids_included"])
        self.assertFalse(decision["candidate_set"]["individual_candidate_ids_included"])
        self.assertEqual(decision["routing_rule_metadata"]["target_local_policy_section"], "routing.rules")
        self.assertNotIn("openai-route:", decision["routing_rule_metadata"]["rule_preview"]["id"])
        self.assertNotIn("openai-route:", decision["routing_rule_metadata"]["rule_preview"]["action"]["reason"])
        self.assertIsNone(decision["local_policy_patch"])
        active = decision["active_local_policy_rule"]
        self.assertEqual(active["status"], "active-local-policy")
        self.assertEqual(active["reason"], "matching-openai-routing-rule-active-in-local-policy")
        self.assertEqual(active["policy_source"], "local-promoted")
        self.assertEqual(active["target_local_policy_section"], "routing.rules")
        self.assertEqual(active["target_local_rule_file"], "routing_rules.yaml")
        rollback = decision["rollback_metadata"]
        self.assertEqual(rollback["schema"], "tokenclaw.openai_routing_promotion_rollback_metadata.v1")
        self.assertEqual(rollback["rollback_action_type"], "disable_openai_routing_rule")
        self.assertTrue(rollback["required_for_promotion"])
        self.assertEqual(rollback["target_local_rule_file"], "routing_rules.yaml")
        self.assertFalse(rollback["policy_files_written"])
        suppression = decision["duplicate_suppression"]
        self.assertEqual(suppression["schema"], "tokenclaw.openai_routing_promotion_duplicate_suppression.v1")
        self.assertTrue(suppression["suppresses_generic_routing_activation_issue"])
        self.assertTrue(suppression["suppresses_new_openai_routing_promotion_issue"])
        self.assertFalse(decision["privacy"]["provider_calls_made"])
        self.assertFalse(decision["privacy"]["managed_server_calls_made"])
        self.assertFalse(decision["privacy"]["policy_files_written"])
        self.assertFalse(decision["privacy"]["individual_candidate_ids_included"])

        outcome = decision["active_local_policy_outcome"]
        self.assertEqual(outcome["schema"], "tokenclaw.openai_routing_active_local_policy_outcome.v1")
        self.assertEqual(outcome["status"], "active-local-policy")
        self.assertEqual(outcome["state"], "active-local-policy")
        self.assertEqual(outcome["current_status"], "applied")
        self.assertEqual(outcome["measurement_next_action"], "measure-openai-routing-rule-outcomes")
        self.assertEqual(outcome["outcome_decision"], "keep-active")
        self.assertEqual(outcome["next_action"], "keep-active")
        self.assertEqual(outcome["deterministic_next_action"], "keep-active")
        self.assertTrue(outcome["gate_passed"])
        self.assertEqual(outcome["reason_codes"], [])
        self.assertEqual(outcome["target_local_policy_section"], "routing.rules")
        self.assertEqual(outcome["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(outcome["matched_count"], 14)
        self.assertEqual(outcome["applied_count"], 7)
        self.assertEqual(outcome["holdout_count"], 7)
        self.assertEqual(outcome["safety_stop_count"], 0)
        self.assertEqual(outcome["error_count"], 0)
        self.assertEqual(outcome["fallback_count"], 0)
        self.assertEqual(outcome["retry_count"], 0)
        self.assertGreater(outcome["savings_per_1000_calls_usd"], 0)
        self.assertEqual(outcome["savings_deltas"]["schema"], "tokenclaw.openai_routing_active_local_policy_savings_deltas.v1")
        self.assertEqual(outcome["savings_deltas"]["applied_count"], 7)
        self.assertEqual(outcome["savings_deltas"]["holdout_count"], 7)
        self.assertGreater(outcome["applied_realized_savings_usd"], 0)
        self.assertEqual(outcome["holdout_realized_savings_usd"], 0)
        self.assertGreater(outcome["applied_minus_holdout_realized_savings_avg_usd"], 0)
        self.assertIsNotNone(outcome["evidence_age_hours"])
        self.assertEqual(outcome["cohort_costs"]["canary_applied"]["count"], 7)
        self.assertEqual(outcome["cohort_costs"]["canary_holdout"]["count"], 7)
        self.assertEqual(outcome["regression_counters"]["error_count"], 0)
        self.assertEqual(outcome["regression_counters"]["fallback_count"], 0)
        self.assertEqual(outcome["regression_counters"]["retry_count"], 0)
        self.assertEqual(outcome["regression_counters"]["safety_stop_count"], 0)
        gate = outcome["outcome_gate"]
        self.assertEqual(gate["schema"], "tokenclaw.openai_routing_active_local_policy_outcome_gate.v1")
        self.assertEqual(gate["state"], "keep-active")
        self.assertEqual(gate["deterministic_next_action"], "keep-active")
        self.assertEqual(gate["decision_options"], ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"])
        self.assertEqual(gate["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(gate["target_local_policy_section"], "routing.rules")
        self.assertGreater(gate["savings_per_1000_calls_usd"], 0)
        self.assertFalse(gate["regression_counters"]["stale_evidence"]["stale"])
        canary_outcome = outcome["routing_canary_outcome"]
        self.assertEqual(canary_outcome["schema"], "tokenclaw.openai_routing_canary_outcome.v1")
        self.assertEqual(canary_outcome["next_action"], "widen")
        self.assertEqual(canary_outcome["decision_options"], ["widen", "hold", "rollback", "collect_more"])
        self.assertEqual(canary_outcome["reason_codes"], ["no-regression-routing-canary"])
        self.assertGreater(canary_outcome["routing_savings"]["routing_only_savings_usd"], 0)
        self.assertGreater(canary_outcome["routing_savings"]["routing_only_savings_per_1000_calls_usd"], 0)
        self.assertEqual(canary_outcome["non_routing_savings"]["crunch_savings_usd"], 0)
        self.assertEqual(canary_outcome["non_routing_savings"]["provider_prompt_cache_discount_usd"], 0)
        self.assertFalse(canary_outcome["quality_performance_regression"]["frontend_visible_regression_reported"])
        self.assertFalse(canary_outcome["privacy"]["raw_prompts_included"])
        self.assertFalse(canary_outcome["privacy"]["provider_bodies_included"])
        rollback_outcome = outcome["rollback_metadata"]
        self.assertEqual(rollback_outcome["schema"], "tokenclaw.openai_routing_active_local_policy_rollback_metadata.v1")
        self.assertEqual(rollback_outcome["rollback_action_type"], "disable_openai_routing_rule")
        self.assertEqual(rollback_outcome["target_rule_id"], "[REDACTED_ID]")
        self.assertFalse(rollback_outcome["rule_id_included"])
        self.assertFalse(rollback_outcome["policy_files_written"])
        self.assertFalse(outcome["candidate_ids_included"])
        self.assertFalse(outcome["candidate_set"]["individual_candidate_ids_included"])
        self.assertFalse(outcome["privacy"]["raw_prompts_included"])
        self.assertFalse(outcome["privacy"]["provider_bodies_included"])
        self.assertFalse(outcome["privacy"]["request_ids_included"])
        self.assertFalse(outcome["privacy"]["session_ids_included"])
        self.assertFalse(outcome["privacy"]["cache_keys_included"])
        self.assertFalse(outcome["privacy"]["file_paths_included"])
        self.assertFalse(outcome["privacy"]["absolute_paths_included"])
        self.assertFalse(outcome["privacy"]["individual_candidate_ids_included"])
        self.assertEqual(result["summary"]["active_local_policy_outcome_count"], 1)
        self.assertFalse(result["summary"]["candidate_ids_included"])
        self.assertEqual(len(result["active_local_policy_outcomes"]), 1)
        self.assertFalse(result["privacy"]["individual_candidate_ids_included"])

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertNotIn("openai-route:", rendered)

        output = io.StringIO()
        exit_code = cli.openai_routing_report_cli(
            ["--db", self.db_path, "--limit", "20", "--promotion-decision", "--target-model", "gpt-5.4-mini"],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.openai_routing_promotion_decision_report.v1")
        self.assertEqual(payload["decision"], "active-local-policy")
        self.assertEqual(payload["promotion_verdict"], "active-local-policy")
        self.assertEqual(payload["summary"]["applied_count"], 7)
        self.assertEqual(payload["summary"]["holdout_count"], 7)
        self.assertEqual(payload["summary"]["next_action"], "measure-openai-routing-rule-outcomes")
        self.assertEqual(payload["summary"]["active_local_policy_outcome_count"], 1)
        self.assertEqual(payload["summary"]["active_local_policy_outcome_decision"], "keep-active")
        self.assertEqual(payload["summary"]["active_local_policy_next_action"], "keep-active")
        self.assertTrue(payload["summary"]["active_local_policy_gate_passed"])
        self.assertEqual(payload["summary"]["active_local_policy_reason_codes"], [])
        self.assertEqual(payload["summary"]["routing_canary_outcome_count"], 1)
        self.assertEqual(payload["summary"]["routing_canary_next_action"], "widen")
        self.assertGreater(payload["summary"]["routing_only_savings_usd"], 0)
        self.assertGreater(payload["summary"]["routing_only_savings_per_1000_calls_usd"], 0)
        self.assertEqual(payload["summary"]["non_routing_crunch_savings_usd"], 0)
        self.assertEqual(payload["summary"]["non_routing_provider_prompt_cache_discount_usd"], 0)
        self.assertFalse(payload["summary"]["frontend_visible_regression_reported"])
        self.assertEqual(payload["summary"]["active_local_policy_rollback_action_type"], "disable_openai_routing_rule")
        self.assertGreater(payload["summary"]["active_local_policy_realized_savings_usd"], 0)
        self.assertGreater(payload["summary"]["active_local_policy_applied_minus_holdout_realized_savings_avg_usd"], 0)
        self.assertIsNotNone(payload["summary"]["active_local_policy_evidence_age_hours"])
        self.assertEqual(payload["active_local_policy_outcomes"][0]["schema"], "tokenclaw.openai_routing_active_local_policy_outcome.v1")
        self.assertEqual(payload["routing_canary_outcomes"][0]["schema"], "tokenclaw.openai_routing_canary_outcome.v1")
        self.assertFalse(payload["privacy"]["individual_candidate_ids_included"])

    def test_targeted_report_emits_semantic_regression_lifecycle_review(self) -> None:
        self._install_disabled_openai_tool_light_rule()

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

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        review = result["routing_lifecycle_review"]
        self.assertEqual(review["schema"], "tokenclaw.openai_routing_lifecycle_review.v1")
        self.assertEqual(review["next_action"], "rollback-required")
        self.assertEqual(review["deterministic_next_action"], "rollback-required")
        self.assertEqual(review["target_local_policy_section"], "routing.rules")
        self.assertEqual(review["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(review["applied_count"], 6)
        self.assertEqual(review["holdout_count"], 7)
        self.assertEqual(review["skipped_count"], 0)
        self.assertEqual(review["unknown_count"], 0)
        self.assertEqual(review["safety_stop_count"], 0)
        self.assertEqual(review["error_count"], 0)
        self.assertEqual(review["fallback_count"], 0)
        self.assertEqual(review["retry_count"], 0)
        self.assertGreater(review["savings_per_1000_calls_usd"], 0)
        self.assertIn("semantic-quality-regression-observed", review["blocker_codes"])
        semantic = review["semantic_quality_regression"]
        self.assertTrue(semantic["observed"])
        self.assertTrue(semantic["disabled_local_policy_rule_present"])
        self.assertEqual(semantic["disabled_local_policy_reason"], "semantic-quality-regression-observed")
        action = review["semantic_regression_action"]
        self.assertEqual(action["schema"], "tokenclaw.openai_routing_semantic_regression_action.v1")
        self.assertTrue(action["observed"])
        self.assertEqual(action["status"], "classified")
        self.assertEqual(action["action_classification"], "rollback-required")
        self.assertEqual(action["deterministic_next_action"], "draft-openai-routing-rollback")
        self.assertEqual(action["target_local_policy_section"], "routing.rules")
        self.assertEqual(action["target_local_rule_file"], "routing_rules.yaml")
        self.assertTrue(action["fingerprint"].startswith("openai-routing-semantic-regression:"))
        self.assertIn("semantic-quality-regression-observed", action["reason_codes"])
        self.assertEqual(action["counters"]["matched_count"], 13)
        self.assertEqual(action["counters"]["applied_count"], 6)
        self.assertEqual(action["counters"]["holdout_count"], 7)
        self.assertEqual(action["counters"]["error_count"], 0)
        self.assertEqual(action["counters"]["retry_count"], 0)
        self.assertEqual(action["counters"]["fallback_count"], 0)
        self.assertEqual(action["counters"]["safety_stop_count"], 0)
        self.assertEqual(action["review_draft"]["draft_action"], "rollback-local-routing-rule")
        self.assertFalse(action["review_draft"]["policy_files_written"])
        self.assertEqual(action["review_draft"]["target_local_rule_file"], "routing_rules.yaml")
        self.assertTrue(action["duplicate_suppression"]["suppresses_generic_semantic_regression_issue"])
        self.assertFalse(action["privacy"]["raw_prompts_included"])
        self.assertFalse(action["privacy"]["provider_bodies_included"])
        self.assertFalse(action["privacy"]["request_ids_included"])
        self.assertFalse(action["privacy"]["session_ids_included"])
        self.assertFalse(action["privacy"]["cache_keys_included"])
        self.assertFalse(action["privacy"]["file_paths_included"])
        self.assertEqual(result["promotion_decision"]["semantic_regression_action"]["action_classification"], "rollback-required")
        self.assertEqual(result["semantic_regression_action"]["action_classification"], "rollback-required")
        self.assertEqual(result["summary"]["routing_lifecycle_next_action"], "rollback-required")
        self.assertEqual(result["summary"]["semantic_regression_action_count"], 1)
        self.assertEqual(result["summary"]["semantic_regression_action"], "rollback-required")
        self.assertEqual(result["summary"]["semantic_regression_next_action"], "draft-openai-routing-rollback")
        self.assertTrue(result["summary"]["semantic_regression_fingerprint"].startswith("openai-routing-semantic-regression:"))
        self.assertIn("semantic-quality-regression-observed", result["summary"]["routing_lifecycle_blocker_codes"])
        disabled = result["promotion_decision"]["disabled_local_policy_rule"]
        self.assertEqual(disabled["schema"], "tokenclaw.openai_routing_disabled_local_policy_rule.v1")
        self.assertEqual(disabled["reason"], "semantic-quality-regression-observed")
        self.assertFalse(disabled["rule_id_included"])
        self.assertEqual(disabled["target_rule_id"], "[REDACTED_ID]")

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertNotIn("openai-route:", rendered)
        self.assertFalse(review["privacy"]["raw_prompts_included"])
        self.assertFalse(review["privacy"]["provider_bodies_included"])
        self.assertFalse(review["privacy"]["request_ids_included"])
        self.assertFalse(review["privacy"]["session_ids_included"])
        self.assertFalse(review["privacy"]["cache_keys_included"])
        self.assertFalse(review["privacy"]["file_paths_included"])
        self.assertFalse(review["privacy"]["individual_candidate_ids_included"])

    def test_targeted_report_delegates_semantic_action_when_promotion_ready(self) -> None:
        from tokenclaw import router as router_module

        previous = list(getattr(router_module, "ROUTING_RULES", []))
        router_module.ROUTING_RULES = []
        self.addCleanup(setattr, router_module, "ROUTING_RULES", previous)

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
        for _ in range(20):
            self._log_openai_shadow_experiment(category="tool-light", similarity=0.95, passed=True)

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        self.assertEqual(result["decision"], "promote")
        self.assertTrue(result["promotion_ready"])
        self.assertEqual(result["summary"]["semantic_regression_action_count"], 0)
        action = result["semantic_regression_action"]
        self.assertFalse(action["observed"])
        self.assertEqual(action["status"], "not-applicable")
        self.assertEqual(action["action_classification"], "delegate-existing-promotion-path")
        self.assertEqual(action["deterministic_next_action"], "use-existing-openai-routing-promotion-decision")
        self.assertIsNone(action["review_draft"])
        self.assertFalse(action["privacy"]["raw_prompts_included"])
        self.assertFalse(action["privacy"]["provider_bodies_included"])
        self.assertFalse(action["privacy"]["request_ids_included"])
        self.assertNotIn("secret-openai-session", json.dumps(result, sort_keys=True))

    def test_active_openai_routing_rule_outcome_reviews_skipped_unknown_coverage(self) -> None:
        self._install_active_openai_tool_light_rule()

        def canary(cohort: str, **extra: object) -> dict[str, object]:
            status = "applied" if cohort == "canary_applied" else "holdout"
            if cohort == "skipped":
                status = "not_selected"
            if cohort == "unknown":
                status = "mystery"
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:to-gpt-5-4-mini",
                "status": status,
                "cohort": cohort,
                "reason": "outside-canary-fraction" if cohort == "skipped" else "selected-canary",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                **extra,
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
        for _ in range(4):
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
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            category="tool-light",
            text_chars=4000,
            has_tools=True,
            openai_canary=canary("unknown", reason="missing-status"),
        )
        for _ in range(20):
            self._log_openai_shadow_experiment(category="tool-light", similarity=0.95, passed=True)

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        outcome = result["active_local_policy_outcomes"][0]
        self.assertEqual(outcome["outcome_decision"], "keep-blocked")
        self.assertEqual(outcome["deterministic_next_action"], "keep-blocked")
        self.assertFalse(outcome["gate_passed"])
        self.assertIn("unknown-coverage-observed", outcome["reason_codes"])
        self.assertEqual(outcome["regression_counters"]["skipped_count"], 1)
        self.assertEqual(outcome["regression_counters"]["unknown_count"], 1)
        self.assertEqual(result["summary"]["active_local_policy_outcome_decision"], "keep-blocked")
        self.assertEqual(result["summary"]["active_local_policy_next_action"], "keep-blocked")

    def test_active_openai_routing_rule_outcome_requires_rollback_on_regression(self) -> None:
        self._install_active_openai_tool_light_rule()

        def canary(cohort: str, **extra: object) -> dict[str, object]:
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
                **extra,
            }

        for _ in range(3):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
            )
        self._log_openai_call(
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            category="tool-light",
            text_chars=4000,
            has_tools=True,
            status_code=500,
            retry_count=1,
            openai_canary=canary("canary_applied", fallback_reason="rate_limited"),
        )
        for _ in range(4):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_holdout"),
            )
        for _ in range(20):
            self._log_openai_shadow_experiment(category="tool-light", similarity=0.95, passed=True)

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        outcome = result["active_local_policy_outcomes"][0]
        self.assertEqual(outcome["outcome_decision"], "rollback-required")
        self.assertEqual(outcome["deterministic_next_action"], "rollback-required")
        self.assertFalse(outcome["gate_passed"])
        self.assertEqual(outcome["routing_canary_outcome"]["next_action"], "rollback")
        self.assertIn("error-observed", outcome["routing_canary_outcome"]["reason_codes"])
        self.assertGreater(outcome["routing_canary_outcome"]["routing_savings"]["routing_only_savings_usd"], 0)
        self.assertIn("error-observed", outcome["reason_codes"])
        self.assertIn("fallback-observed", outcome["reason_codes"])
        self.assertIn("retry-observed", outcome["reason_codes"])
        self.assertEqual(outcome["rollback_metadata"]["rollback_action_type"], "disable_openai_routing_rule")
        self.assertEqual(result["summary"]["active_local_policy_outcome_decision"], "rollback-required")
        self.assertEqual(result["summary"]["active_local_policy_next_action"], "rollback-required")
        self.assertEqual(result["summary"]["routing_canary_next_action"], "rollback")
        self.assertEqual(result["summary"]["active_local_policy_rollback_action_type"], "disable_openai_routing_rule")

    def test_active_openai_routing_rule_outcome_collects_more_when_sample_too_small(self) -> None:
        self._install_active_openai_tool_light_rule()

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
            }

        for _ in range(2):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4-mini",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_applied"),
                crunch_tokens_saved_est=120,
                cache_read_input_tokens=100,
            )
        for _ in range(3):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("canary_holdout"),
            )
        for _ in range(20):
            self._log_openai_shadow_experiment(category="tool-light", similarity=0.95, passed=True)

        result = build_openai_routing_promotion_decision_report(self.store, limit=20)

        outcome = result["active_local_policy_outcomes"][0]
        canary_outcome = outcome["routing_canary_outcome"]
        self.assertEqual(canary_outcome["next_action"], "collect_more")
        self.assertIn("insufficient-applied-samples", canary_outcome["reason_codes"])
        self.assertIn("insufficient-holdout-samples", canary_outcome["reason_codes"])
        self.assertGreater(canary_outcome["routing_savings"]["routing_only_savings_usd"], 0)
        self.assertGreater(canary_outcome["non_routing_savings"]["crunch_savings_usd"], 0)
        self.assertGreater(canary_outcome["non_routing_savings"]["provider_prompt_cache_discount_usd"], 0)
        self.assertEqual(result["summary"]["routing_canary_next_action"], "collect_more")

    def test_targeted_promotion_decision_keeps_skipped_unknown_coverage_out_of_promotion(self) -> None:
        def canary(canary_cohort: str, **extra: object) -> dict[str, object]:
            status = "applied" if canary_cohort == "canary_applied" else "holdout"
            if canary_cohort == "skipped":
                status = "not_selected"
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:to-gpt-5-4-mini",
                "status": status,
                "cohort": canary_cohort,
                "reason": "outside-canary-fraction" if canary_cohort == "skipped" else "selected-canary",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if canary_cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "source_surface": "openai_responses",
                "endpoint": "responses",
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

        result = build_openai_routing_promotion_decision_report(self.store, limit=30)

        self.assertEqual(result["decision"], "keep-blocked")
        self.assertFalse(result["promotion_ready"])
        self.assertEqual(result["summary"]["promote_count"], 0)
        self.assertEqual(result["summary"]["keep_staged_count"], 0)
        self.assertEqual(result["summary"]["keep_blocked_count"], 1)
        self.assertEqual(result["summary"]["applied_count"], 6)
        self.assertEqual(result["summary"]["holdout_count"], 7)
        self.assertEqual(result["summary"]["skipped_count"], 3)
        self.assertEqual(result["summary"]["unknown_count"], 1)
        self.assertEqual(result["summary"]["next_action"], "review-openai-routing-canary-blockers")
        self.assertIn("skipped-canary-promotion-blocker", result["summary"]["reason_codes"])
        self.assertNotIn("unknown-canary-lifecycle-rows", result["summary"]["reason_codes"])
        self.assertEqual(result["summary"]["skipped_reason_breakdown"], [{"value": "outside-canary-fraction", "count": 3}])
        self.assertEqual(result["summary"]["unknown_reason_breakdown"], [{"value": "canary-lifecycle-logging-gap", "count": 1}])
        decision = result["promotion_decision"]
        self.assertEqual(decision["lifecycle"]["skipped_count"], 3)
        self.assertEqual(decision["lifecycle"]["unknown_count"], 1)
        self.assertEqual(decision["lifecycle"]["skipped_unknown_classification"]["promotion_blocker_count"], 1)
        self.assertTrue(decision["quality_gates"]["requires_classified_skipped_unknown_rows"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertFalse(decision["privacy"]["raw_prompts_included"])
        self.assertFalse(decision["privacy"]["tool_payloads_included"])

    def test_targeted_promotion_decision_narrows_unsupported_skipped_coverage(self) -> None:
        def canary(cohort: str, **extra: object) -> dict[str, object]:
            status = "applied" if cohort == "canary_applied" else "holdout"
            if cohort == "skipped":
                status = "skipped"
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:to-gpt-5-4-mini",
                "status": status,
                "cohort": cohort,
                "reason": "request-too-large" if cohort == "skipped" else "selected-canary",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "source_surface": "openai_responses",
                "endpoint": "responses",
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
        for _ in range(2):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=4000,
                has_tools=True,
                openai_canary=canary("skipped"),
            )

        result = build_openai_routing_promotion_decision_report(self.store, limit=30)

        self.assertEqual(result["decision"], "narrow")
        self.assertFalse(result["promotion_ready"])
        self.assertEqual(result["summary"]["narrow_count"], 1)
        self.assertEqual(result["summary"]["promote_count"], 0)
        self.assertEqual(result["summary"]["skipped_count"], 2)
        self.assertEqual(result["summary"]["unknown_count"], 0)
        self.assertEqual(result["summary"]["next_action"], "narrow-openai-routing-canary-shape")
        self.assertEqual(result["summary"]["reason"], "skipped-canary-unsupported-shape")
        self.assertEqual(
            result["summary"]["blocker_reason_breakdown"],
            [
                {"value": "skipped-canary-unsupported-shape", "count": 2},
                {"value": "insufficient-semantic-quality-passes", "count": 1},
                {"value": "missing-semantic-quality-evidence", "count": 1},
            ],
        )
        decision = result["promotion_decision"]
        self.assertEqual(decision["decision"], "narrow")
        self.assertEqual(decision["promotion_verdict"], "keep-staged")
        self.assertEqual(decision["lifecycle"]["skipped_unknown_classification"]["unsupported_shape_count"], 2)
        self.assertEqual(
            decision["reason_codes"],
            [
                "insufficient-semantic-quality-passes",
                "missing-semantic-quality-evidence",
                "skipped-canary-unsupported-shape",
            ],
        )
        self.assertEqual(decision["target"]["target_local_rule_file"], "routing_rules.yaml")
        self.assertIsNone(decision["local_policy_patch"])
        self.assertEqual(decision["rollback_metadata"]["target_local_policy_section"], "routing.rules")
        self.assertEqual(
            decision["duplicate_suppression"]["reason"],
            "skipped-canary-unsupported-shape",
        )

    def test_targeted_promotion_decision_collects_ineligible_lifecycle_rows_outside_simulated_shape(self) -> None:
        def canary(canary_cohort: str, **extra: object) -> dict[str, object]:
            return {
                "enabled": True,
                "policy_id": "local-openai-routing-canary-v1",
                "target_candidate_id": "openai-route:responses:gpt-5:tool-light:tools:nonstream:to-gpt-5-4-mini",
                "status": "applied" if canary_cohort == "canary_applied" else "holdout",
                "cohort": canary_cohort,
                "reason": "selected-canary" if canary_cohort == "canary_applied" else "selected-holdout",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "actual_forwarded_model": "gpt-5.4-mini" if canary_cohort == "canary_applied" else "gpt-5.4",
                "category": "tool-light",
                "source_surface": "openai_responses",
                "endpoint": "responses",
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
                openai_canary=canary("canary_holdout"),
            )
        for _ in range(2):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="tool-light",
                text_chars=20000,
                has_tools=True,
                openai_canary=canary("none", status="ineligible", cohort="none", reason="request-too-large"),
            )

        result = build_openai_routing_promotion_decision_report(self.store, limit=30)

        self.assertEqual(result["decision"], "narrow")
        self.assertEqual(result["summary"]["matched_count"], 15)
        self.assertEqual(result["summary"]["applied_count"], 6)
        self.assertEqual(result["summary"]["holdout_count"], 7)
        self.assertEqual(result["summary"]["skipped_count"], 2)
        self.assertEqual(result["summary"]["unknown_count"], 0)
        self.assertEqual(result["summary"]["skipped_reason_breakdown"], [{"value": "request-too-large", "count": 2}])
        self.assertEqual(
            result["summary"]["blocker_reason_breakdown"],
            [
                {"value": "skipped-canary-unsupported-shape", "count": 2},
                {"value": "insufficient-semantic-quality-passes", "count": 1},
                {"value": "missing-semantic-quality-evidence", "count": 1},
            ],
        )
        decision = result["promotion_decision"]
        classification = decision["lifecycle"]["skipped_unknown_classification"]
        self.assertEqual(decision["promotion_verdict"], "keep-staged")
        self.assertIsNone(decision["local_policy_patch"])
        self.assertEqual(classification["unsupported_shape_count"], 2)
        self.assertEqual(classification["unsupported_shape_reason_breakdown"], [{"value": "request-too-large", "count": 2}])
        self.assertEqual(
            decision["reason_codes"],
            [
                "insufficient-semantic-quality-passes",
                "missing-semantic-quality-evidence",
                "skipped-canary-unsupported-shape",
            ],
        )
        self.assertFalse(decision["privacy"]["raw_prompts_included"])
        self.assertFalse(decision["privacy"]["request_ids_included"])
        self.assertNotIn("secret-openai-session", json.dumps(result, sort_keys=True))

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        for _ in range(5):
            self._log_openai_call(category="chat", text_chars=1200)

        result = asyncio.run(stats_openai_routing_report(self.store, limit=10))
        self.assertEqual(result["schema"], "tokenclaw.openai_routing_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.openai_routing_report_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.openai_routing_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 5)
        self.assertNotIn("secret-openai-session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
