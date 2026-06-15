from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import yaml

from agentflow_proxy import cli
from agentflow_proxy.request_shape_rollups import (
    apply_request_shape_cache_replay_canary_action,
    apply_request_shape_crunch_canary_action,
    build_request_shape_cache_replay_canary_stage_report,
    build_request_shape_cache_replay_evidence_report,
    build_request_shape_cache_replay_policy_decision_report,
    build_request_shape_crunch_canary_impact_report,
    build_request_shape_crunch_policy_decision_ledger,
    build_request_shape_crunch_policy_decision_report,
    build_request_shape_crunch_canary_stage_report,
    build_request_shape_rollups_report,
    record_request_shape_crunch_policy_decision_ledger,
    request_shape_crunch_canary_lifecycle,
)
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class RequestShapeRollupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_call(
        self,
        *,
        provider: str = "anthropic",
        path: str = "/v1/messages",
        source_surface: str = "anthropic_messages",
        endpoint: str = "messages",
        requested_model: str = "claude-sonnet-4-6",
        routed_model: str = "claude-sonnet-4-6",
        requested_model_family: str = "claude-sonnet",
        routed_model_family: str = "claude-sonnet",
        category: str = "tool-result",
        workflow_phase: str = "tool-execution",
        stream: int = 1,
        has_tools: bool = True,
        cache_status: str = "skipped",
        cache_reason: str = "streaming",
        cache_hit: int = 0,
        text_chars: int = 24_000,
        cost: float = 0.02,
        baseline: float = 0.03,
        status_code: int = 200,
        latency_ms: int = 125,
        retry_count: int = 0,
        routing_reason: str = "keep requested model",
        routing_extra: dict[str, object] | None = None,
        cache_extra: dict[str, object] | None = None,
        crunch_extra: dict[str, object] | None = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        routing_json: dict[str, object] = {
            "provider": provider,
            "requested_model": requested_model,
            "routed_model": routed_model,
            "text_chars": text_chars,
            "has_tools": has_tools,
            "category": category,
            "workflow_phase": workflow_phase,
            "reason": routing_reason,
        }
        if routing_extra:
            routing_json.update(routing_extra)
        cache_json: dict[str, object] = {
            "status": cache_status,
            "reason": cache_reason,
            "policy_source": "local-default",
            "request_fingerprint": "raw-request-fingerprint-must-not-leak",
            "cache_key": "raw-cache-key-must-not-leak",
        }
        if cache_extra:
            cache_json.update(cache_extra)
        input_tokens = max(1, text_chars // 4)
        crunch_json: dict[str, object] = {"changed": False, "tokens_saved_est": 0}
        if crunch_extra:
            crunch_json.update(crunch_extra)
        self.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path=path,
            requested_model=requested_model,
            routed_model=routed_model,
            stream=stream,
            cache_hit=cache_hit,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=input_tokens,
            output_tokens_est=50,
            actual_input_tokens=input_tokens,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=baseline,
            crunch_json=stable_json(crunch_json),
            routing_json=stable_json(routing_json),
            cache_json=stable_json(cache_json),
            error="request-id-secret must not leak" if status_code >= 400 else None,
            request_json=stable_json(
                {
                    "prompt": "raw prompt must not leak",
                    "messages": [{"content": "provider body must not leak"}],
                    "path": "/tmp/private/source.py",
                }
            ),
            response_json=stable_json({"content": "raw response must not leak"}),
            session_id="raw-session-id-must-not-leak",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family=requested_model_family,
            routed_model_family=routed_model_family,
        )
        return call_id

    def test_repeated_shapes_collapse_and_persist_without_raw_fields(self) -> None:
        for cost in (0.02, 0.03, 0.04):
            self._log_call(cost=cost, baseline=cost + 0.01)
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=1_200,
            cost=0.004,
            baseline=0.004,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=True, run_id="test-rollup")

        self.assertEqual(report["schema"], "agentflow.request_shape_rollups.v1")
        self.assertTrue(report["persisted"])
        self.assertEqual(report["persisted_count"], 2)
        self.assertEqual(report["summary"]["rows_considered"], 4)
        self.assertEqual(report["summary"]["rollup_count"], 2)
        self.assertEqual(report["summary"]["collapsed_rows"], 2)
        self.assertEqual(report["summary"]["follow_up_candidate_count"], 2)
        self.assertEqual(report["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        follow_up = report["follow_up_candidates"]
        self.assertEqual(follow_up["schema"], "agentflow.request_shape_follow_up_candidates.v1")
        self.assertEqual(follow_up["status"], "candidates-ranked")
        self.assertEqual(follow_up["summary"]["ranked_candidate_count"], 2)
        self.assertEqual(follow_up["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(follow_up["summary"]["top_readiness_state"], "activation-ready")
        self.assertEqual(follow_up["summary"]["activation_ready_count"], 1)
        self.assertEqual(follow_up["top_candidate"]["local_action_family"], "crunch")
        self.assertEqual(follow_up["top_candidate"]["readiness_state"], "activation-ready")
        self.assertEqual(follow_up["top_candidate"]["sample_count"], 3)
        self.assertEqual(follow_up["top_candidate"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertGreater(follow_up["top_candidate"]["projected_saved_tokens"], 0)
        self.assertGreater(follow_up["top_candidate"]["projected_savings_usd"], 0)
        self.assertEqual(follow_up["blocker_cohorts"][0]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertIn("repeated_context", follow_up["top_candidate"]["candidate_work_classes"])
        self.assertIn("replayability", follow_up["top_candidate"]["candidate_work_classes"])
        self.assertTrue(follow_up["privacy"]["metadata_only"])
        self.assertTrue(follow_up["privacy"]["aggregate_only"])
        self.assertFalse(follow_up["privacy"]["individual_candidate_ids_included"])
        rendered_follow_up = json.dumps(follow_up, sort_keys=True)
        self.assertNotIn('"candidate_id"', rendered_follow_up)
        self.assertNotIn('"cohort_id"', rendered_follow_up)
        self.assertNotIn('"policy_id"', rendered_follow_up)
        repeated = next(row for row in report["rollups"] if row["row_count"] == 3)
        self.assertEqual(repeated["provider_family"], "anthropic")
        self.assertEqual(repeated["text_bucket"], "8k_32k_chars")
        self.assertIn("cache_replay", repeated["candidate_families"])
        self.assertIn("cache_blocker", repeated["candidate_families"])
        self.assertIn("repeated_context", repeated["candidate_work_classes"])
        self.assertIn("replayability", repeated["candidate_work_classes"])
        self.assertIn("crunch", repeated["candidate_work_classes"])
        self.assertIn("unsupported-streaming-shape", repeated["blocker_codes"])
        class_breakdown = {item["value"]: item["count"] for item in repeated["metadata"]["candidate_class_breakdown"]}
        self.assertEqual(class_breakdown["repeated_context"], 3)
        self.assertEqual(
            {item["value"]: item["count"] for item in repeated["metadata"]["cost_bucket_breakdown"]},
            {"0_01_0_05_usd": 3},
        )

        rows = self.store.request_shape_rollup_rows(run_id="test-rollup")
        self.assertEqual(len(rows), 2)
        persisted = json.dumps(rows, sort_keys=True)
        rendered = json.dumps(report, sort_keys=True) + persisted
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "request-id-secret",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_fingerprints_included"])

    def test_follow_up_candidates_rank_concrete_blocker_cohorts(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                text_chars=132_000,
                cost=0.09,
                baseline=0.09,
            )
        for _ in range(2):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=4_000,
                cost=0.01,
                baseline=0.01,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="tool-light",
            workflow_phase="tool-light",
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            routing_reason="keep requested model",
            text_chars=3_000,
            cost=0.02,
            baseline=0.02,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="blocker-cohorts")
        follow_up = report["follow_up_candidates"]
        top = follow_up["top_blocker_cohort"]

        self.assertEqual(top["schema"], "agentflow.request_shape_blocker_cohort.v1")
        self.assertEqual(top["rank"], 1)
        self.assertEqual(top["readiness_state"], "activation-ready")
        self.assertEqual(top["local_action_family"], "crunch")
        self.assertEqual(top["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(top["sample_count"], 3)
        self.assertIn("thinking-routing-guard", top["blocker_codes"])
        self.assertIn("tool-call-cache-disabled", top["blocker_codes"])
        self.assertIn("unsupported-streaming-shape", top["blocker_codes"])
        self.assertGreater(top["projected_saved_tokens"], 0)
        self.assertGreater(top["projected_savings_usd"], 0)
        self.assertTrue(top["privacy"]["metadata_only"])
        self.assertTrue(top["privacy"]["aggregate_only"])

        replay_ready = next(item for item in follow_up["blocker_cohorts"] if item["next_action"] == "stage-cache-replay-canary")
        self.assertEqual(replay_ready["readiness_state"], "activation-ready")
        self.assertEqual(replay_ready["local_action_family"], "cache")
        self.assertEqual(replay_ready["projected_hits"], 1)

        tool_blocked = next(item for item in follow_up["blocker_cohorts"] if item["next_action"] == "collect-tool-call-cache-invalidation-evidence")
        self.assertEqual(tool_blocked["readiness_state"], "blocked")
        self.assertEqual(tool_blocked["local_action_family"], "cache")
        rendered = json.dumps(follow_up, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_prompt_like_labels_are_rejected_to_unknown(self) -> None:
        self._log_call(
            category="raw prompt must not leak /tmp/category-secret.py",
            workflow_phase="tenant-id-secret planning",
            cache_reason="cache-key-secret request-id-secret prompt payload",
            routing_reason="session-id-secret should not leak",
            routing_extra={"workflow_phase": "tenant-id-secret planning"},
        )

        report = build_request_shape_rollups_report(self.store, limit=10, persist=False, run_id="redaction")
        rendered = json.dumps(report, sort_keys=True)

        self.assertEqual(report["rollups"][0]["category"], "unknown")
        self.assertEqual(report["rollups"][0]["workflow_phase"], "unknown")
        self.assertIn('"cache_reason_breakdown": [{"count": 1, "value": "unknown"}]', rendered)
        for forbidden in (
            "raw prompt must not leak",
            "/tmp/category-secret.py",
            "tenant-id-secret",
            "cache-key-secret",
            "request-id-secret",
            "session-id-secret",
            "prompt payload",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replayability_dry_run_ranks_ready_and_skipped_blockers(self) -> None:
        for cost in (0.01, 0.03):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=1_200,
                cost=cost,
                baseline=cost,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="summary",
            workflow_phase="summary",
            stream=1,
            has_tools=False,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=12_000,
            cost=0.02,
            baseline=0.02,
        )
        self._log_call(
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            text_chars=24_000,
            cost=0.04,
            baseline=0.04,
        )
        self._log_call(
            path="/v1/unknown",
            source_surface="unknown_surface",
            endpoint="unknown_endpoint",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=1_200,
            cost=0.01,
            baseline=0.01,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="replay-dry-run")
        dry_run = report["cache_replayability_dry_run"]
        classification = report["cache_replay_blocker_classification"]

        self.assertEqual(dry_run["schema"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(dry_run["summary"]["replay_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["skipped_cohort_count"], 3)
        self.assertEqual(dry_run["summary"]["projected_hits"], 1)
        self.assertAlmostEqual(dry_run["summary"]["projected_savings_usd"], 0.02)
        ready = next(row for row in dry_run["cohorts"] if row["readiness"] == "replay-ready")
        self.assertEqual(ready["reason"], "replay-ready-exact-non-tool-shape")
        self.assertEqual(ready["projected_hits"], 1)
        reasons = {item["value"]: item["count"] for item in dry_run["skipped_reason_breakdown"]}
        self.assertEqual(reasons["streaming-replay-not-supported"], 1)
        self.assertEqual(reasons["invalidation-evidence-missing"], 1)
        self.assertEqual(reasons["unsupported-endpoint"], 1)
        blockers = {item["value"]: item["count"] for item in dry_run["blocker_breakdown"]}
        self.assertIn("streaming-replay-not-supported", blockers)
        self.assertIn("tools-present", blockers)
        self.assertIn("invalidation-evidence-missing", blockers)
        self.assertIn("unsafe-tool-calls-without-invalidation", blockers)
        self.assertIn("unsupported-endpoint", blockers)
        rendered = json.dumps(dry_run, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(dry_run["privacy"]["raw_request_bodies_included"])
        self.assertFalse(dry_run["privacy"]["cache_keys_included"])
        self.assertFalse(dry_run["privacy"]["request_fingerprints_included"])
        self.assertFalse(dry_run["privacy"]["individual_candidate_ids_included"])

        self.assertEqual(classification["schema"], "agentflow.request_shape_cache_replay_blocker_classification.v1")
        self.assertEqual(classification["status"], "classified")
        self.assertEqual(classification["summary"]["skipped_cohort_count"], 3)
        classes = {item["value"]: item["count"] for item in classification["class_breakdown"]}
        self.assertGreater(classes["collect-invalidation-evidence"], 0)
        self.assertGreater(classes["keep-tool-cache-disabled"], 0)
        self.assertGreater(classes["streaming-replay-support-needed"], 0)
        self.assertGreater(classes["insufficient-repeat-evidence"], 0)
        self.assertGreater(classes["unsupported-safety-shape"], 0)
        next_actions = {item["value"] for item in classification["next_action_breakdown"]}
        self.assertIn("collect-cache-invalidation-evidence", next_actions)
        self.assertIn("keep-tool-cache-disabled", next_actions)
        self.assertIn("design-streaming-cache-replay-support", next_actions)
        self.assertIn("collect-more-repeat-evidence", next_actions)
        self.assertIn("keep-cache-replay-noop", next_actions)
        self.assertEqual(classification["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(classification["summary"]["unsafe_cache_apply_action_count"], 0)
        self.assertTrue(classification["acceptance"]["has_tool_blocker_class"])
        self.assertTrue(classification["acceptance"]["has_invalidation_evidence_class"])
        self.assertTrue(classification["acceptance"]["has_streaming_support_class"])
        self.assertTrue(classification["acceptance"]["has_insufficient_repeat_class"])
        self.assertTrue(classification["acceptance"]["has_unsupported_safety_shape_class"])
        self.assertTrue(classification["acceptance"]["no_cache_apply_without_invalidation_safety_evidence"])
        self.assertTrue(classification["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(classification["privacy"]["metadata_only"])
        self.assertTrue(classification["privacy"]["aggregate_only"])
        self.assertFalse(classification["privacy"]["cache_keys_included"])
        self.assertFalse(classification["privacy"]["request_fingerprints_included"])
        self.assertFalse(classification["privacy"]["individual_candidate_ids_included"])
        rendered_classification = json.dumps(classification, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered_classification)

    def test_cache_replay_canary_stage_report_targets_openai_responses_replay_ready_cohort(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="tool-light",
            workflow_phase="tool-light",
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            text_chars=6_000,
            cost=0.02,
            baseline=0.02,
        )
        for cost in (0.015, 0.025):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=1,
                has_tools=False,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.04, 0.03):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="short-completion",
                workflow_phase="summary",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.012, 0.018):
            self._log_call(
                provider="openai",
                path="/v1/embeddings",
                source_surface="openai_embeddings",
                endpoint="embeddings",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )

        self.assertEqual(report["schema"], "agentflow.request_shape_cache_replay_canary_stage.v1")
        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertTrue(report["acceptance"]["has_replay_ready_openai_responses_cohort"])
        self.assertTrue(report["acceptance"]["has_projected_hits"])
        self.assertTrue(report["acceptance"]["has_projected_savings"])
        self.assertTrue(report["acceptance"]["writes_no_provider_bodies"])
        self.assertTrue(report["acceptance"]["writes_no_cache_entries"])
        self.assertTrue(report["acceptance"]["has_holdout_metadata"])
        self.assertTrue(report["acceptance"]["has_lifecycle_metadata"])
        self.assertTrue(report["acceptance"]["has_applied_and_holdout_eligibility"])
        self.assertTrue(report["acceptance"]["records_hit_miss_bypass_invalidation_and_stale_risk"])
        self.assertTrue(report["acceptance"]["preserves_tool_and_streaming_guards"])
        self.assertTrue(report["acceptance"]["stages_only_openai_responses_chat"])
        self.assertTrue(report["acceptance"]["tool_streaming_and_invalidation_missing_cohorts_skipped"])

        action = report["top_stage_action"]
        self.assertEqual(action["schema"], "agentflow.request_shape_cache_replay_canary_action.v1")
        self.assertEqual(action["action_type"], "stage-local-openai-cache-replay-canary")
        self.assertEqual(action["target_local_policy"], "cache_canary_policy")
        self.assertEqual(action["conditions"]["provider_family"], "openai")
        self.assertEqual(action["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(action["conditions"]["endpoint"], "responses")
        self.assertEqual(action["conditions"]["category"], "chat")
        self.assertEqual(action["conditions"]["workflow_phase"], "chat")
        self.assertEqual(action["conditions"]["text_bucket"], "2k_8k_chars")
        self.assertFalse(action["conditions"]["stream"])
        self.assertFalse(action["conditions"]["has_tools"])
        self.assertEqual(action["ttl_seconds"], 3600)
        self.assertEqual(action["invalidation"]["strategy"], "session-scoped-exact-non-tool")
        self.assertEqual(action["invalidation"]["scope"], "session")
        self.assertFalse(action["invalidation"]["safe_invalidation"])
        self.assertFalse(action["invalidation"]["tool_call_cache_enabled"])
        self.assertFalse(action["invalidation"]["streaming_replay_enabled"])
        self.assertIn("ttl-limited-replay-window", action["invalidation"]["assumptions"])
        self.assertIn("invalidation-evidence-missing", action["invalidation"]["bypass_reasons"])
        self.assertGreater(action["projected_hits"], 0)
        self.assertGreater(action["projected_savings_usd"], 0)
        self.assertEqual(action["rollout_fraction"], 0.05)
        self.assertEqual(action["holdout_fraction"], 0.2)
        self.assertTrue(action["canary_applied_eligible"])
        self.assertTrue(action["canary_holdout_eligible"])
        self.assertEqual(action["projected_lifecycle"]["schema"], "agentflow.request_shape_cache_replay_canary_projected_lifecycle.v1")
        self.assertGreater(action["projected_lifecycle"]["projected_canary_applied_count"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_canary_holdout_count"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_applied_hits"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_holdout_hits"], 0)
        self.assertTrue(action["safety_gates"]["metadata_only"])
        self.assertTrue(action["safety_gates"]["aggregate_only"])
        self.assertTrue(action["safety_gates"]["exact_non_tool_only"])
        self.assertFalse(action["safety_gates"]["tool_call_cache_enabled"])
        self.assertFalse(action["safety_gates"]["streaming_replay_enabled"])
        self.assertTrue(action["cache_decision_metadata"]["records_applied"])
        self.assertTrue(action["cache_decision_metadata"]["records_holdout"])
        self.assertTrue(action["cache_decision_metadata"]["records_skipped"])
        self.assertTrue(action["cache_decision_metadata"]["records_bypass"])
        self.assertTrue(action["cache_decision_metadata"]["records_bypassed"])
        self.assertTrue(action["cache_decision_metadata"]["records_invalidated"])
        self.assertTrue(action["cache_decision_metadata"]["records_invalidation_blocked"])
        self.assertTrue(action["cache_decision_metadata"]["records_stale_risk"])
        self.assertTrue(action["cache_decision_metadata"]["records_cache_hit"])
        self.assertTrue(action["cache_decision_metadata"]["records_cache_miss"])
        self.assertTrue(action["lifecycle_metadata"]["emits_applied"])
        self.assertTrue(action["lifecycle_metadata"]["emits_holdout"])
        self.assertTrue(action["lifecycle_metadata"]["emits_skipped"])
        self.assertTrue(action["lifecycle_metadata"]["emits_bypass"])
        self.assertTrue(action["lifecycle_metadata"]["emits_bypassed"])
        self.assertTrue(action["lifecycle_metadata"]["emits_invalidated"])
        self.assertTrue(action["lifecycle_metadata"]["emits_invalidation_blocked"])
        self.assertTrue(action["lifecycle_metadata"]["emits_stale_risk"])
        self.assertTrue(action["lifecycle_metadata"]["canary_applied_eligible"])
        self.assertTrue(action["lifecycle_metadata"]["canary_holdout_eligible"])
        self.assertGreater(action["lifecycle_metadata"]["projected_canary_applied_count"], 0)
        self.assertGreater(action["lifecycle_metadata"]["projected_canary_holdout_count"], 0)
        self.assertEqual(action["lifecycle_metadata"]["impact_report"], "agentflow.openai_cache_replay_impact.v1")
        skipped_guards = report["skipped_cohort_guards"]
        self.assertEqual(skipped_guards["schema"], "agentflow.request_shape_cache_replay_canary_skipped_guards.v1")
        self.assertGreaterEqual(skipped_guards["tool_cohort_count"], 1)
        self.assertGreaterEqual(skipped_guards["streaming_cohort_count"], 1)
        self.assertGreaterEqual(skipped_guards["invalidation_missing_cohort_count"], 1)
        self.assertTrue(skipped_guards["tool_streaming_and_invalidation_missing_remain_skipped"])
        blocker_values = {item["value"] for item in skipped_guards["blocker_breakdown"]}
        self.assertIn("tools-present", blocker_values)
        self.assertIn("invalidation-evidence-missing", blocker_values)
        self.assertIn("streaming-replay-not-supported", blocker_values)
        self.assertIn("unsupported-endpoint", blocker_values)
        self.assertEqual(report["source_report"]["cache_replayability_summary"]["replay_ready_cohort_count"], 2)
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_canary_stage_cli_emits_direct_stage_payload(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-cache-stage",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.request_shape_cache_replay_canary_stage.v1")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertEqual(payload["top_stage_action"]["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(payload["top_stage_action"]["conditions"]["endpoint"], "responses")
        self.assertFalse(payload["top_stage_action"]["conditions"]["has_tools"])
        self.assertFalse(payload["top_stage_action"]["conditions"]["stream"])
        self.assertGreater(payload["top_stage_action"]["projected_hits"], 0)
        self.assertGreater(payload["top_stage_action"]["projected_savings_usd"], 0)
        self.assertEqual(payload["top_stage_action"]["holdout_fraction"], 0.2)
        self.assertTrue(payload["top_stage_action"]["lifecycle_metadata"]["emits_invalidation_blocked"])
        self.assertTrue(payload["privacy"]["aggregate_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_cache_replay_canary_stage_apply_writes_local_policy_overlay(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-stage-apply",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        result = apply_request_shape_cache_replay_canary_action(
            report["top_stage_action"],
            rules_path=policy_path,
        )

        self.assertEqual(result["schema"], "agentflow.request_shape_cache_replay_canary_apply.v1")
        self.assertTrue(result["ok"])
        self.assertFalse(result["dry_run"])
        self.assertTrue(result["wrote_policy_files"])
        self.assertFalse(result["cache_entries_written"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["rules_path_included"])
        self.assertEqual(result["target_local_policy"], "cache_canary_policy")
        self.assertEqual(result["canary_fraction"], 0.05)
        self.assertEqual(result["holdout_fraction"], 0.2)
        self.assertEqual(result["ttl_seconds"], 3600)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertTrue(result["privacy"]["aggregate_only"])

        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(written["schema"], "agentflow.openai_cache_replay_canary_policy.v1")
        self.assertEqual(written["policy_source"], "local-manual")
        self.assertEqual(len(written["pattern_rules"]), 1)
        rule = written["pattern_rules"][0]
        self.assertEqual(rule["id"], result["policy_id"])
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["candidate_id"], result["cohort_id"])
        self.assertEqual(rule["conditions"]["pattern_hashes"], ["sha256:*"])
        self.assertEqual(rule["conditions"]["provider_family"], "openai")
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["ttl_seconds"], 3600)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.05)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.2)
        self.assertEqual(rule["graduation"]["source_schema"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(rule["graduation"]["projected_hits"], 2)
        self.assertTrue(rule["graduation"]["aggregate_only"])
        self.assertEqual(rule["invalidation"]["strategy"], "session-scoped-exact-non-tool")

        rendered = json.dumps(written, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_canary_stage_cli_apply_writes_policy_overlay(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        config_dir = Path(self.tmpdir.name) / "overlay"
        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-cache-stage-apply",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
                "--config-dir",
                str(config_dir),
                "--apply",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["dry_run"])
        self.assertFalse(payload["read_only"])
        self.assertTrue(payload["wrote_policy_files"])
        self.assertFalse(payload["cache_entries_written"])
        self.assertTrue(payload["apply_result"]["ok"])
        self.assertEqual(payload["apply_result"]["schema"], "agentflow.request_shape_cache_replay_canary_apply.v1")
        policy_path = config_dir / "cache_canary_policy.yaml"
        self.assertTrue(policy_path.exists())
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(written["pattern_rules"][0]["policy_source"], "local-manual")
        self.assertEqual(written["pattern_rules"][0]["conditions"]["source_surface"], "openai_responses")
        self.assertFalse(written["pattern_rules"][0]["conditions"]["has_tools"])
        self.assertFalse(written["pattern_rules"][0]["conditions"]["stream"])
        self.assertNotIn("raw prompt must not leak", policy_path.read_text(encoding="utf-8"))

    def test_cache_replay_evidence_reports_empty_no_canary_without_leaking_paths(self) -> None:
        policy_path = Path(self.tmpdir.name) / "missing" / "cache_canary_policy.yaml"

        report = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )

        self.assertEqual(report["schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(report["status"], "no-canary-policy")
        self.assertFalse(report["ok"])
        self.assertFalse(report["source"]["policy_file_present"])
        self.assertFalse(report["source"]["policy_path_included"])
        self.assertEqual(report["staged_canary_count"], 0)
        self.assertEqual(report["summary"]["applied_count"], 0)
        self.assertEqual(report["summary"]["holdout_count"], 0)
        self.assertEqual(report["summary"]["projected_hits"], 0)
        self.assertEqual(report["summary"]["observed_hits"], 0)
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["cache_canary_policy_path_included"])
        self.assertFalse(report["privacy"]["policy_ids_included"])
        self.assertFalse(report["privacy"]["rule_ids_included"])
        self.assertFalse(report["privacy"]["cohort_ids_included"])
        self.assertNotIn(str(policy_path), json.dumps(report, sort_keys=True))

    def test_cache_replay_evidence_reports_staged_but_no_traffic(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-evidence-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(report["top_stage_action"], rules_path=policy_path)
        empty_store = SQLiteStore(str(Path(self.tmpdir.name) / "empty.sqlite3"))
        try:
            evidence = build_request_shape_cache_replay_evidence_report(
                empty_store,
                rules_path=policy_path,
                limit=20,
            )
        finally:
            empty_store.conn.close()

        self.assertEqual(evidence["schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(evidence["status"], "staged-no-traffic")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["staged_canary_count"], 1)
        self.assertEqual(evidence["summary"]["projected_hits"], 2)
        self.assertEqual(evidence["summary"]["observed_hits"], 0)
        self.assertEqual(evidence["summary"]["observed_row_count"], 0)
        self.assertEqual(evidence["summary"]["applied_count"], 0)
        self.assertEqual(evidence["summary"]["holdout_count"], 0)
        self.assertTrue(evidence["acceptance"]["has_staged_canary_metadata"])
        self.assertTrue(evidence["acceptance"]["reports_projected_vs_observed_hits"])
        self.assertTrue(evidence["acceptance"]["reports_stale_evidence_metadata"])
        self.assertFalse(evidence["source"]["policy_path_included"])
        rendered = json.dumps(evidence, sort_keys=True)
        self.assertNotIn(str(policy_path), rendered)
        self.assertNotIn("local-openai-cache-replay-canary", rendered)
        self.assertNotIn("request-shape-cache-replay:", rendered)

    def test_cache_replay_evidence_reports_observed_applied_holdout_and_blockers(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-evidence-observed",
            rollout_fraction=0.50,
            holdout_fraction=0.25,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_result = apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        rule = written["pattern_rules"][0]
        public_pattern_rule = {
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": rule["rollout"],
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "graduation": rule["graduation"],
        }
        applied_canary = {
            "schema": "agentflow.cache_replay_canary_decision.v1",
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "canary_cohort": "canary_applied",
            "status": "applied",
            "reason": "no-dependency-required",
            "projection": rule["graduation"],
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="hit",
            cache_reason="exact-pattern-hit",
            cache_hit=1,
            text_chars=6_000,
            cost=0.001,
            baseline=0.031,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": applied_canary,
                "estimated_saved_cost_usd": 0.03,
            },
        )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-pattern-miss",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": applied_canary,
            },
        )
        holdout_canary = {
            **applied_canary,
            "canary": {"enabled": True, "selected": False, "cohort": "canary_holdout"},
            "canary_cohort": "canary_holdout",
            "status": "holdout",
            "reason": "canary_holdout",
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="canary_holdout",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": {**public_pattern_rule, "canary": holdout_canary["canary"]},
                "cache_replay_canary": holdout_canary,
            },
        )
        bypass_canary = {**applied_canary, "status": "bypassed", "reason": "session-scope-missing"}
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="skipped",
            cache_reason="session-scope-missing",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": bypass_canary,
            },
        )
        invalidated_canary = {**applied_canary, "status": "invalidated", "reason": "dependency-changed"}
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="skipped",
            cache_reason="dependency-changed",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": invalidated_canary,
            },
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )

        self.assertTrue(apply_result["ok"])
        self.assertEqual(evidence["schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(evidence["status"], "observed")
        self.assertEqual(evidence["summary"]["applied_count"], 2)
        self.assertEqual(evidence["summary"]["holdout_count"], 1)
        self.assertEqual(evidence["summary"]["exact_hit_count"], 1)
        self.assertEqual(evidence["summary"]["miss_count"], 1)
        self.assertEqual(evidence["summary"]["bypass_count"], 1)
        self.assertEqual(evidence["summary"]["invalidation_skipped_count"], 1)
        self.assertEqual(evidence["summary"]["observed_hits"], 1)
        self.assertEqual(evidence["summary"]["observed_savings_usd"], 0.03)
        self.assertEqual(evidence["summary"]["projected_hits"], 2)
        self.assertFalse(evidence["stale_evidence"]["stale"])
        blockers = {item["value"]: item["count"] for item in evidence["blocker_breakdown"]}
        self.assertEqual(blockers["session-scope-missing"], 1)
        self.assertEqual(blockers["dependency-changed"], 1)
        canary_statuses = {item["value"]: item["count"] for item in evidence["canary_status_breakdown"]}
        self.assertEqual(canary_statuses["applied"], 2)
        self.assertEqual(canary_statuses["holdout"], 1)
        self.assertTrue(evidence["acceptance"]["reports_applied_and_holdout_counts"])
        self.assertTrue(evidence["acceptance"]["reports_projected_vs_observed_hits"])
        self.assertTrue(evidence["acceptance"]["reports_observed_savings_estimate"])
        self.assertTrue(evidence["acceptance"]["reports_blocker_breakdown"])

        rendered = json.dumps(evidence, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
            rule["id"],
            rule["candidate_id"],
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_evidence_cli_reads_policy_overlay(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(self.store, limit=20)
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)

        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_evidence_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--rules-path",
                str(policy_path),
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(payload["status"], "staged-no-traffic")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertTrue(payload["privacy"]["aggregate_only"])
        self.assertNotIn(str(policy_path), stdout.getvalue())

    def test_cache_replay_policy_decision_promotes_observed_hit_with_holdout(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-policy-decision",
            rollout_fraction=0.50,
            holdout_fraction=0.25,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        rule = written["pattern_rules"][0]
        public_pattern_rule = {
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": rule["rollout"],
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "graduation": rule["graduation"],
        }
        applied_canary = {
            "schema": "agentflow.cache_replay_canary_decision.v1",
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "canary_cohort": "canary_applied",
            "status": "applied",
            "reason": "no-dependency-required",
            "projection": rule["graduation"],
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="hit",
            cache_reason="exact-pattern-hit",
            cache_hit=1,
            text_chars=6_000,
            cost=0.001,
            baseline=0.031,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": applied_canary,
                "estimated_saved_cost_usd": 0.03,
            },
        )
        holdout_canary = {
            **applied_canary,
            "canary": {"enabled": True, "selected": False, "cohort": "canary_holdout"},
            "canary_cohort": "canary_holdout",
            "status": "holdout",
            "reason": "canary_holdout",
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="canary_holdout",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": {**public_pattern_rule, "canary": holdout_canary["canary"]},
                "cache_replay_canary": holdout_canary,
            },
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )
        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(decision["schema"], "agentflow.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(decision["decision"], "promote")
        self.assertTrue(decision["summary"]["promotion_allowed"])
        self.assertFalse(decision["summary"]["policy_files_written"])
        self.assertFalse(decision["summary"]["cache_entries_written"])
        self.assertEqual(decision["summary"]["applied_count"], 1)
        self.assertEqual(decision["summary"]["holdout_count"], 1)
        self.assertEqual(decision["summary"]["observed_hits"], 1)
        self.assertEqual(decision["summary"]["observed_savings_usd"], 0.03)
        top = decision["top_decision"]
        self.assertEqual(top["local_policy_patch"]["patch_type"], "promote_openai_exact_cache_replay_canary")
        self.assertEqual(top["local_policy_patch"]["target_local_rule_file"], "cache_rules.yaml")
        promoted_rule = top["local_policy_patch"]["pattern_rules"][0]
        self.assertEqual(promoted_rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(promoted_rule["conditions"]["endpoint"], "responses")
        self.assertEqual(promoted_rule["conditions"]["category"], "chat")
        self.assertFalse(promoted_rule["conditions"]["has_tools"])
        self.assertFalse(promoted_rule["conditions"]["stream"])
        self.assertFalse(promoted_rule["action"]["allow_tool_calls"])
        self.assertFalse(promoted_rule["action"]["streaming"])
        self.assertEqual(promoted_rule["rollout"]["recommendation_mode"], "active")
        self.assertEqual(top["rollback_metadata"]["rollback_action_type"], "disable_openai_exact_cache_replay_policy")
        self.assertTrue(top["coverage"]["has_applied_coverage"])
        self.assertTrue(top["coverage"]["has_holdout_coverage"])
        self.assertTrue(top["coverage"]["has_observed_hits"])
        self.assertTrue(decision["acceptance"]["targets_file_backed_cache_policy"])
        self.assertTrue(decision["acceptance"]["keeps_tool_and_streaming_replay_blocked"])
        rendered = json.dumps(decision, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
            rule["id"],
            rule["candidate_id"],
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_policy_decision_rolls_back_stale_evidence(self) -> None:
        evidence = {
            "schema": "agentflow.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "ttl_seconds": 3600,
                }
            ],
            "summary": {
                "observed_row_count": 2,
                "applied_count": 1,
                "holdout_count": 1,
                "exact_hit_count": 1,
                "observed_hits": 1,
                "projected_hits": 2,
                "observed_savings_usd": 0.03,
                "projected_savings_usd": 0.06,
                "invalidation_skipped_count": 0,
            },
            "stale_evidence": {"stale": True, "age_hours": 96.0},
            "blocker_breakdown": [],
        }

        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(decision["decision"], "rollback")
        self.assertTrue(decision["summary"]["rollback_required"])
        self.assertFalse(decision["summary"]["promotion_allowed"])
        self.assertEqual(decision["top_decision"]["reason"], "stale-cache-replay-evidence")
        self.assertEqual(decision["top_decision"]["local_policy_patch"]["patch_type"], "rollback_openai_exact_cache_replay_policy")
        self.assertIn("stale-cache-replay-evidence", decision["top_decision"]["reason_codes"])

    def test_cache_replay_policy_decision_cli_keeps_blocked_without_canary_policy(self) -> None:
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
        )
        missing_policy_path = Path(self.tmpdir.name) / "missing" / "cache_canary_policy.yaml"

        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_policy_decision_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--rules-path",
                str(missing_policy_path),
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(payload["decision"], "keep-blocked")
        self.assertTrue(payload["summary"]["keep_blocked"])
        self.assertEqual(payload["summary"]["staged_canary_count"], 0)
        self.assertIn("missing-cache-replay-canary-policy", payload["reason_codes"])
        self.assertTrue(payload["acceptance"]["drafts_local_policy_patch_or_blocker"])
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertNotIn(str(missing_policy_path), stdout.getvalue())
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_managed_recommendation_handoff_cli_ranks_local_policy_handoffs(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        stdout = io.StringIO()
        code = cli.managed_recommendation_handoff_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--request-shape-limit",
                "20",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.managed_recommendation_handoff_health.v1")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertEqual(payload["managed_dependency"], "optional")

        by_family = {row["local_action_family"]: row for row in payload["omissions"]}
        for family, rule_file in {
            "crunch": "crunch_rules.yaml",
            "cache": "cache_rules.yaml",
        }.items():
            self.assertIn(family, by_family)
            row = by_family[family]
            self.assertEqual(row["follow_up_owner"], "local-policy")
            self.assertTrue(row["local_file_backed_representation"]["exists"])
            self.assertEqual(row["local_file_backed_representation"]["rule_file"], rule_file)
            self.assertIn(rule_file, row["local_handoff_reason"])
        self.assertEqual(by_family["crunch"]["next_action"], "stage-repeated-context-crunch-canary")

        rendered = stdout.getvalue()
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertTrue(payload["privacy"]["aggregate_only"])

    def test_crunch_opportunity_dry_run_projects_repeated_context_savings(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=1_200,
            cost=0.004,
            baseline=0.004,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-dry-run")
        dry_run = report["crunch_opportunity_dry_run"]

        self.assertEqual(dry_run["schema"], "agentflow.request_shape_crunch_opportunity_dry_run.v1")
        self.assertEqual(dry_run["status"], "ranked")
        self.assertEqual(dry_run["summary"]["measurement_ready_cohort_count"], 1)
        self.assertGreater(dry_run["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(dry_run["summary"]["projected_saved_usd"], 0)
        self.assertEqual(dry_run["summary"]["activation_state"], "activation-ready")
        self.assertEqual(dry_run["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        follow_up = dry_run["activation_follow_up"]
        self.assertEqual(follow_up["schema"], "agentflow.request_shape_crunch_activation_follow_up.v1")
        self.assertEqual(follow_up["activation_state"], "activation-ready")
        self.assertEqual(follow_up["activation_mode"], "canary-candidate")
        self.assertEqual(follow_up["savings_status"], "projected-savings-ranked")
        self.assertEqual(follow_up["report_key"], "request_shape_crunch_opportunity")
        self.assertEqual(follow_up["evidence_schema"], "agentflow.request_shape_crunch_opportunity_dry_run.v1")
        self.assertEqual(follow_up["candidate_count"], dry_run["summary"]["candidate_count"])
        self.assertEqual(follow_up["matched_count"], dry_run["summary"]["matched_count"])
        self.assertEqual(follow_up["rows_considered"], dry_run["summary"]["rows_considered"])
        self.assertEqual(follow_up["projected_saved_chars"], dry_run["summary"]["projected_saved_chars"])
        self.assertEqual(follow_up["projected_saved_tokens"], dry_run["summary"]["projected_saved_tokens"])
        self.assertEqual(follow_up["projected_saved_usd"], dry_run["summary"]["projected_saved_usd"])
        self.assertEqual(follow_up["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(follow_up["target_local_policy"], "crunch_rules")
        self.assertEqual(follow_up["recommended_action_count"], 1)
        self.assertFalse(follow_up["canary_already_staged"])
        self.assertFalse(follow_up["canary_already_applied"])
        self.assertIsNone(follow_up["no_op_reason"])
        self.assertFalse(follow_up["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertEqual(follow_up["missing_measurements"], [])
        self.assertTrue(follow_up["privacy"]["metadata_only"])
        self.assertTrue(follow_up["privacy"]["aggregate_only"])
        top = dry_run["cohorts"][0]
        self.assertEqual(top["readiness"], "measurement-ready")
        self.assertEqual(top["reason"], "repeated-context-crunch-opportunity")
        self.assertIn("repeated_context", top["work_classes"])
        self.assertIn("crunch", top["work_classes"])
        self.assertEqual(top["candidate_rule"], "repeated-context-conservative-dry-run")
        self.assertEqual(dry_run["summary"]["recommended_action_count"], 1)
        action = dry_run["recommended_actions"][0]
        self.assertEqual(action["schema"], "agentflow.request_shape_crunch_canary_action.v1")
        self.assertEqual(action["target_local_policy"], "crunch_rules")
        self.assertEqual(action["rollout_fraction"], 0.1)
        self.assertEqual(action["holdout_fraction"], 0.1)
        self.assertEqual(action["candidate_count"], dry_run["summary"]["candidate_count"])
        self.assertEqual(action["projected_saved_chars"], top["projected_saved_chars"])
        self.assertEqual(action["projected_saved_tokens"], top["projected_saved_tokens"])
        self.assertEqual(action["projected_saved_usd"], top["projected_saved_usd"])
        self.assertTrue(action["safety_gates"]["metadata_only"])
        self.assertTrue(action["safety_gates"]["holdout_required"])
        class_counts = {item["value"]: item["count"] for item in dry_run["work_class_breakdown"]}
        self.assertEqual(class_counts["repeated_context"], 3)
        rendered = json.dumps(dry_run, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(dry_run["privacy"]["metadata_only"])
        self.assertTrue(dry_run["privacy"]["aggregate_only"])
        self.assertFalse(dry_run["privacy"]["raw_request_bodies_included"])
        self.assertFalse(dry_run["privacy"]["provider_bodies_included"])
        self.assertFalse(dry_run["privacy"]["session_ids_included"])
        self.assertFalse(dry_run["privacy"]["cache_keys_included"])

    def test_crunch_canary_stage_report_targets_anthropic_thinking_tool_result_cohort(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-crunch-canary-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )

        self.assertEqual(report["schema"], "agentflow.request_shape_repeated_context_crunch_canary_stage.v1")
        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertTrue(report["acceptance"]["stages_one_repeated_context_crunch_canary"])
        self.assertTrue(report["acceptance"]["has_projected_tokens"])
        self.assertTrue(report["acceptance"]["has_projected_savings"])
        self.assertTrue(report["acceptance"]["has_holdout_metadata"])
        self.assertTrue(report["acceptance"]["has_projected_lifecycle_split"])
        self.assertTrue(report["acceptance"]["has_safety_stop_metadata"])
        self.assertTrue(report["acceptance"]["unsafe_or_stale_cohorts_remain_skipped"])

        action = report["top_stage_action"]
        self.assertEqual(action["schema"], "agentflow.request_shape_crunch_canary_action.v1")
        self.assertEqual(action["action_type"], "stage-local-repeated-context-crunch-canary")
        self.assertEqual(action["target_local_policy"], "crunch_rules")
        self.assertEqual(action["conditions"]["provider_family"], "anthropic")
        self.assertEqual(action["conditions"]["source_surface"], "anthropic_messages")
        self.assertEqual(action["conditions"]["endpoint"], "messages")
        self.assertEqual(action["conditions"]["category"], "tool-result")
        self.assertEqual(action["conditions"]["workflow_phase"], "thinking")
        self.assertEqual(action["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertTrue(action["conditions"]["stream"])
        self.assertTrue(action["conditions"]["has_tools"])
        self.assertEqual(action["rollout_fraction"], 0.05)
        self.assertEqual(action["holdout_fraction"], 0.2)
        self.assertGreater(action["projected_saved_tokens"], 0)
        self.assertGreater(action["projected_saved_usd"], 0)
        self.assertEqual(action["projected_lifecycle"]["schema"], "agentflow.request_shape_crunch_canary_projected_lifecycle.v1")
        self.assertEqual(action["projected_lifecycle"]["matched_count"], 3)
        self.assertEqual(action["projected_lifecycle"]["projected_canary_applied_count"], 1)
        self.assertEqual(action["projected_lifecycle"]["projected_canary_holdout_count"], 1)
        self.assertEqual(action["projected_lifecycle"]["projected_skipped_count"], 1)
        self.assertGreater(action["projected_lifecycle"]["projected_applied_saved_tokens"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_applied_saved_usd"], 0)
        self.assertEqual(action["source_evidence_schema"], "agentflow.request_shape_rollup_row.v1")
        self.assertEqual(
            action["source_evidence_schemas"],
            [
                "agentflow.request_shape_follow_up_candidates.v1",
                "agentflow.request_shape_crunch_opportunity_dry_run.v1",
            ],
        )
        self.assertEqual(action["local_only_reason"], "file-backed-local-policy-no-managed-dependency")
        self.assertIn("thinking-routing-guard", action["evidence_blocker_codes"])
        self.assertIn("tool-call-cache-disabled", action["evidence_blocker_codes"])
        self.assertIn("unsupported-streaming-shape", action["evidence_blocker_codes"])
        self.assertEqual(
            action["projected_lifecycle"]["evidence_blocker_codes"],
            action["evidence_blocker_codes"],
        )
        self.assertTrue(action["projected_lifecycle"]["privacy"]["metadata_only"])
        self.assertTrue(action["projected_lifecycle"]["privacy"]["aggregate_only"])
        self.assertTrue(action["safety_gates"]["metadata_only"])
        self.assertTrue(action["safety_gates"]["aggregate_only"])
        self.assertTrue(action["safety_gates"]["local_only"])
        self.assertFalse(action["safety_gates"]["tool_call_cache_enabled"])
        self.assertFalse(action["safety_gates"]["tool_call_cache_enablement_allowed"])
        self.assertFalse(action["safety_gates"]["request_ids_included"])
        self.assertFalse(action["safety_gates"]["session_ids_included"])
        self.assertFalse(action["safety_gates"]["cache_keys_included"])
        self.assertFalse(action["safety_gates"]["file_paths_included"])
        self.assertFalse(action["safety_gates"]["tool_payloads_included"])
        self.assertTrue(action["safety_gates"]["holdout_required"])
        self.assertTrue(action["safety_gates"]["records_applied_holdout_skipped_safety_stopped_fallback_rollback"])
        self.assertTrue(action["lifecycle_metadata"]["emits_applied"])
        self.assertTrue(action["lifecycle_metadata"]["emits_holdout"])
        self.assertTrue(action["lifecycle_metadata"]["emits_skipped"])
        self.assertTrue(action["lifecycle_metadata"]["emits_safety_stopped"])
        self.assertEqual(action["lifecycle_metadata"]["evidence_blocker_codes"], action["evidence_blocker_codes"])
        self.assertEqual(action["lifecycle_metadata"]["projected_canary_applied_count"], 1)
        self.assertEqual(action["lifecycle_metadata"]["projected_canary_holdout_count"], 1)
        self.assertEqual(action["lifecycle_metadata"]["projected_skipped_count"], 1)
        self.assertEqual(action["lifecycle_metadata"]["impact_report"], "agentflow.request_shape_crunch_canary_impact.v1")
        projection = report["stage_lifecycle_projection"]
        self.assertEqual(projection["schema"], "agentflow.request_shape_crunch_canary_stage_lifecycle_projection.v1")
        self.assertEqual(projection["matched_count"], 3)
        self.assertEqual(projection["projected_canary_applied_count"], 1)
        self.assertEqual(projection["projected_canary_holdout_count"], 1)
        self.assertEqual(projection["projected_skipped_count"], 1)
        self.assertGreater(projection["projected_applied_saved_tokens"], 0)
        self.assertGreater(projection["projected_applied_saved_usd"], 0)
        self.assertTrue(projection["privacy"]["metadata_only"])
        self.assertTrue(projection["privacy"]["aggregate_only"])
        self.assertEqual(report["source_report"]["activation_follow_up"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_cli_emits_direct_stage_payload(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=cost,
                baseline=cost,
            )

        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-stage",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.request_shape_repeated_context_crunch_canary_stage.v1")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertEqual(payload["top_stage_action"]["conditions"]["workflow_phase"], "thinking")
        self.assertEqual(payload["top_stage_action"]["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertGreater(payload["top_stage_action"]["projected_saved_tokens"], 0)
        self.assertEqual(payload["top_stage_action"]["holdout_fraction"], 0.2)
        self.assertEqual(payload["top_stage_action"]["projected_lifecycle"]["projected_canary_applied_count"], 1)
        self.assertEqual(payload["top_stage_action"]["projected_lifecycle"]["projected_canary_holdout_count"], 1)
        self.assertTrue(payload["top_stage_action"]["lifecycle_metadata"]["emits_safety_stopped"])
        self.assertEqual(payload["stage_lifecycle_projection"]["projected_canary_applied_count"], 1)
        self.assertEqual(payload["stage_lifecycle_projection"]["projected_canary_holdout_count"], 1)
        self.assertTrue(payload["privacy"]["aggregate_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_crunch_canary_stage_cli_apply_writes_file_backed_local_rule(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=cost,
                baseline=cost,
            )

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-stage-apply",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
                "--apply",
                "--rules-path",
                str(rules_path),
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "staged-and-applied")
        self.assertFalse(payload["dry_run"])
        self.assertFalse(payload["read_only"])
        self.assertEqual(payload["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertTrue(payload["acceptance"]["stages_one_repeated_context_crunch_canary"])
        self.assertTrue(payload["acceptance"]["has_projected_lifecycle_split"])
        apply_result = payload["apply_result"]
        self.assertTrue(apply_result["ok"])
        self.assertTrue(apply_result["wrote_policy_files"])
        self.assertFalse(apply_result["rules_path_included"])
        self.assertEqual(apply_result["target_local_policy"], "crunch_rules")
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        canary = rules["request_shape_repeated_context_canaries"]["rules"][0]
        self.assertEqual(canary["id"], payload["top_stage_action"]["policy_id"])
        self.assertEqual(canary["cohort_id"], payload["top_stage_action"]["cohort_id"])
        self.assertEqual(canary["conditions"]["provider_family"], "anthropic")
        self.assertEqual(canary["conditions"]["category"], "tool-result")
        self.assertEqual(canary["conditions"]["workflow_phase"], "thinking")
        self.assertEqual(canary["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertEqual(canary["rollout"]["canary_fraction"], 0.05)
        self.assertEqual(canary["rollout"]["holdout_fraction"], 0.2)
        self.assertEqual(canary["source_evidence_schema"], "agentflow.request_shape_rollup_row.v1")
        self.assertIn("agentflow.request_shape_follow_up_candidates.v1", canary["source_evidence_schemas"])
        self.assertIn("agentflow.request_shape_crunch_opportunity_dry_run.v1", canary["source_evidence_schemas"])
        self.assertEqual(canary["local_only_reason"], "file-backed-local-policy-no-managed-dependency")
        self.assertIn("thinking-routing-guard", canary["evidence_blocker_codes"])
        self.assertIn("tool-call-cache-disabled", canary["evidence_blocker_codes"])
        self.assertIn("unsupported-streaming-shape", canary["evidence_blocker_codes"])
        self.assertEqual(canary["projected_saved_tokens"], payload["top_stage_action"]["projected_saved_tokens"])
        self.assertEqual(canary["projected_saved_usd"], payload["top_stage_action"]["projected_saved_usd"])
        self.assertTrue(canary["safety_gates"]["local_file_backed"])
        self.assertTrue(canary["safety_gates"]["local_only"])
        self.assertFalse(canary["safety_gates"]["tool_call_cache_enabled"])
        self.assertFalse(canary["safety_gates"]["tool_call_cache_enablement_allowed"])
        self.assertFalse(canary["safety_gates"]["request_ids_included"])
        self.assertFalse(canary["safety_gates"]["session_ids_included"])
        self.assertFalse(canary["safety_gates"]["cache_keys_included"])
        self.assertFalse(canary["safety_gates"]["file_paths_included"])
        self.assertFalse(canary["safety_gates"]["tool_payloads_included"])
        self.assertTrue(canary["privacy"]["metadata_only"])
        self.assertTrue(canary["privacy"]["aggregate_only"])
        self.assertFalse(canary["privacy"]["raw_prompts_included"])
        self.assertFalse(canary["privacy"]["provider_bodies_included"])
        self.assertFalse(canary["privacy"]["request_ids_included"])
        self.assertFalse(canary["privacy"]["session_ids_included"])
        self.assertFalse(canary["privacy"]["cache_keys_included"])
        self.assertFalse(canary["privacy"]["file_paths_included"])
        self.assertFalse(canary["privacy"]["tool_payloads_included"])

        rendered = stdout.getvalue()
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            str(rules_path),
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_keeps_safety_stopped_cohort_out_of_applied_holdout_split(self) -> None:
        safety_lifecycle = {
            "schema": "agentflow.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "local-repeated-context-crunch-canary-safety",
            "cohort_id": "request-shape-crunch:safety",
            "status": "safety-stopped",
            "cohort": "safety_stopped",
            "reason": "error-rate-regression",
            "safety_stop": True,
            "metadata_only": True,
        }
        for _ in range(2):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=0.08,
                baseline=0.08,
                crunch_extra={"request_shape_repeated_context_canary": safety_lifecycle},
            )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-crunch-canary-safety-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )

        self.assertEqual(report["status"], "no-stageable-cohort")
        self.assertFalse(report["ok"])
        self.assertEqual(report["staged_canary_count"], 0)
        self.assertTrue(report["acceptance"]["unsafe_or_stale_cohorts_remain_skipped"])
        projection = report["stage_lifecycle_projection"]
        self.assertEqual(projection["projected_canary_applied_count"], 0)
        self.assertEqual(projection["projected_canary_holdout_count"], 0)
        self.assertEqual(projection["projected_safety_stopped_count"], 2)
        self.assertEqual(projection["skipped_or_safety_reasons"][0]["value"], "repeated-context-crunch-canary-safety-stopped")
        self.assertEqual(report["cohort_lifecycle_projections"][0]["status"], "safety-stopped")
        self.assertEqual(report["cohort_lifecycle_projections"][0]["reason"], "repeated-context-crunch-canary-safety-stopped")
        self.assertTrue(report["cohort_lifecycle_projections"][0]["privacy"]["metadata_only"])
        self.assertTrue(report["cohort_lifecycle_projections"][0]["privacy"]["aggregate_only"])

    def test_request_shape_crunch_canary_apply_writes_local_rule_and_lifecycle_metadata(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-canary-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        rules_path.parent.mkdir()
        rules_path.write_text(
            yaml.safe_dump({"enabled": True, "unrelated_section": {"keep": True}}),
            encoding="utf-8",
        )

        dry_apply = apply_request_shape_crunch_canary_action(action, rules_path=rules_path, dry_run=True)
        self.assertTrue(dry_apply["ok"])
        self.assertFalse(dry_apply["wrote_policy_files"])
        self.assertNotIn("request_shape_repeated_context_canaries", yaml.safe_load(rules_path.read_text(encoding="utf-8")))

        applied = apply_request_shape_crunch_canary_action(action, rules_path=rules_path)
        self.assertTrue(applied["ok"])
        self.assertTrue(applied["wrote_policy_files"])
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.assertEqual(rules["unrelated_section"], {"keep": True})
        canaries = rules["request_shape_repeated_context_canaries"]
        self.assertTrue(canaries["enabled"])
        self.assertEqual(canaries["rules"][0]["id"], action["policy_id"])
        self.assertEqual(canaries["rules"][0]["rollout"]["canary_fraction"], 0.1)
        self.assertEqual(canaries["rules"][0]["rollout"]["holdout_fraction"], 0.1)

        selected: dict[str, dict[str, object]] = {}
        features = dict(action["conditions"])
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"sample-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)
        self.assertEqual(selected["applied"]["cohort"], "canary_applied")
        self.assertEqual(selected["holdout"]["cohort"], "canary_holdout")

        unrelated = request_shape_crunch_canary_lifecycle(action, {**features, "category": "chat", "cohort_sample_id": "unrelated"})
        self.assertEqual(unrelated["status"], "skipped")
        self.assertEqual(unrelated["reason"], "cohort-mismatch")

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            crunch_extra={"request_shape_repeated_context_canary": selected["applied"]},
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.07,
            baseline=0.07,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )
        lifecycle_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-canary-lifecycle")
        dry_run = lifecycle_report["crunch_opportunity_dry_run"]
        self.assertEqual(dry_run["status"], "canary-staged")
        self.assertEqual(dry_run["summary"]["canary_staged_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["canary_applied_rows"], 1)
        self.assertEqual(dry_run["summary"]["canary_holdout_rows"], 1)
        self.assertEqual(dry_run["summary"]["activation_state"], "measurement-required")
        self.assertEqual(dry_run["summary"]["top_next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(dry_run["activation_follow_up"]["activation_state"], "measurement-required")
        self.assertEqual(dry_run["activation_follow_up"]["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(
            dry_run["activation_follow_up"]["no_op_reason"],
            "matching-repeated-context-crunch-canary-already-staged",
        )
        self.assertTrue(dry_run["activation_follow_up"]["canary_already_staged"])
        self.assertTrue(dry_run["activation_follow_up"]["canary_already_applied"])
        self.assertTrue(dry_run["activation_follow_up"]["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertEqual(
            dry_run["activation_follow_up"]["duplicate_suppression"]["matching_local_policy"],
            "crunch_rules",
        )
        self.assertIn("missing-crunch-canary-impact-measurement", dry_run["activation_follow_up"]["missing_measurements"])
        self.assertEqual(dry_run["cohorts"][0]["readiness"], "canary-staged")
        self.assertEqual(dry_run["cohorts"][0]["reason"], "repeated-context-crunch-canary-applied-and-holdout")
        self.assertEqual(dry_run["recommended_actions"], [])

    def test_crunch_canary_impact_reports_positive_applied_against_holdout(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-impact-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]
        features = dict(action["conditions"])
        selected: dict[str, dict[str, object]] = {}
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"sample-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.06,
            baseline=0.08,
            crunch_extra={
                "changed": True,
                "before_chars": 80_000,
                "after_chars": 72_000,
                "saved_chars": 8_000,
                "tokens_saved_est": 2_000,
                "request_shape_repeated_context_canary": selected["applied"],
            },
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )

        impact_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact_report["schema"], "agentflow.request_shape_crunch_canary_impact.v1")
        self.assertEqual(impact_report["status"], "widen-ready")
        self.assertEqual(impact_report["next_action"], "widen")
        self.assertEqual(impact_report["summary"]["applied_count"], 1)
        self.assertEqual(impact_report["summary"]["holdout_count"], 1)
        self.assertEqual(impact_report["summary"]["saved_chars"], 8_000)
        self.assertEqual(impact_report["summary"]["saved_tokens"], 2_000)
        self.assertEqual(impact_report["summary"]["estimated_saved_tokens"], 2_000)
        self.assertGreater(impact_report["summary"]["saved_usd"], 0)
        self.assertEqual(impact_report["summary"]["error_rate_delta"], 0.0)
        self.assertEqual(impact_report["summary"]["retry_rate_delta"], 0.0)
        self.assertEqual(impact_report["summary"]["latency_avg_delta_ms"], 0.0)
        self.assertEqual(impact_report["summary"]["fallback_count"], 0)
        self.assertEqual(impact_report["summary"]["safety_stop_count"], 0)
        self.assertEqual(impact_report["summary"]["top_impact_recommendation"], "promotion-ready")
        self.assertEqual(impact_report["summary"]["promotion_ready_count"], 1)
        self.assertEqual(impact_report["summary"]["next_action"], "widen")
        self.assertEqual(impact_report["summary"]["top_next_action"], "widen")
        self.assertEqual(impact_report["summary"]["recommended_next_action"], "widen-repeated-context-crunch-canary")
        self.assertTrue(impact_report["summary"]["applied_vs_holdout_coverage"]["has_applied_coverage"])
        self.assertTrue(impact_report["summary"]["applied_vs_holdout_coverage"]["has_holdout_coverage"])
        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "widen-ready")
        self.assertEqual(candidate["impact_recommendation"], "promotion-ready")
        self.assertEqual(candidate["promotion_recommendation"], "promotion-ready")
        self.assertEqual(candidate["recommended_next_action"], "widen-repeated-context-crunch-canary")
        self.assertEqual(candidate["next_action"], "widen")
        self.assertIsNone(candidate["top_blocker"])
        self.assertEqual(candidate["cohorts"]["canary_applied"]["saved_chars"], 8_000)
        self.assertEqual(candidate["cohorts"]["canary_holdout"]["saved_chars"], 0)
        self.assertEqual(candidate["cohorts"]["canary_applied"]["latency_avg_ms"], 125.0)
        self.assertEqual(candidate["latency_avg_delta_ms"], 0.0)
        self.assertEqual(candidate["estimated_saved_tokens"], 2_000)
        self.assertEqual(candidate["coverage"]["applied_count"], 1)
        self.assertEqual(candidate["coverage"]["holdout_count"], 1)
        self.assertEqual(candidate["promotion_metadata"]["impact_recommendation"], "promotion-ready")
        self.assertEqual(candidate["promotion_metadata"]["next_action"], "widen")
        self.assertEqual(candidate["promotion_metadata"]["observed_saved_tokens"], 2_000)
        feedback = impact_report["activation_lifecycle_feedback"]
        self.assertEqual(feedback["schema"], "agentflow.activation_staged_lifecycle_feedback_summary.v1")
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["action_family"], "crunch")
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["applied_count"], 1)
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["holdout_count"], 1)

        rendered = json.dumps(impact_report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(impact_report["privacy"]["metadata_only"])
        self.assertTrue(impact_report["privacy"]["aggregate_only"])
        self.assertFalse(impact_report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(impact_report["privacy"]["provider_bodies_included"])
        self.assertFalse(impact_report["privacy"]["request_ids_included"])
        self.assertFalse(impact_report["privacy"]["session_ids_included"])
        self.assertFalse(impact_report["privacy"]["individual_candidate_ids_included"])

    def test_crunch_canary_impact_reports_no_applied_coverage_without_failing(self) -> None:
        report = build_request_shape_crunch_canary_impact_report([])

        self.assertEqual(report["schema"], "agentflow.request_shape_crunch_canary_impact.v1")
        self.assertTrue(report["ok"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["status"], "no-applied-coverage")
        self.assertEqual(report["next_action"], "stage-canary-first")
        self.assertEqual(report["summary"]["next_action"], "stage-canary-first")
        self.assertEqual(report["summary"]["candidate_count"], 0)
        self.assertEqual(report["summary"]["applied_count"], 0)
        self.assertEqual(report["summary"]["holdout_count"], 0)
        self.assertEqual(report["summary"]["estimated_saved_tokens"], 0)
        self.assertFalse(report["summary"]["applied_vs_holdout_coverage"]["has_applied_coverage"])
        self.assertIn("missing-applied-or-holdout-coverage", report["missing_measurements"])
        self.assertIn("applied-crunch-canary-coverage", report["missing_measurements"])
        self.assertIn("crunch-canary-lifecycle-metadata", report["missing_measurements"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])

    def test_crunch_canary_impact_cli_returns_no_applied_coverage_status(self) -> None:
        self._log_call()

        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_impact_cli(
            ["--db", self.db_path, "--limit", "10"],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.request_shape_crunch_canary_impact.v1")
        self.assertEqual(payload["status"], "no-applied-coverage")
        self.assertEqual(payload["next_action"], "stage-canary-first")
        self.assertEqual(payload["summary"]["applied_count"], 0)
        self.assertEqual(payload["summary"]["holdout_count"], 0)
        self.assertTrue(payload["source_report"]["privacy"]["metadata_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_crunch_canary_impact_blocks_widening_on_safety_stop(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-safety-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]
        safety_lifecycle = {
            "schema": "agentflow.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": action["policy_id"],
            "cohort_id": action["cohort_id"],
            "status": "safety-stopped",
            "cohort": "safety_stopped",
            "reason": "error-rate-regression",
            "safety_stop": True,
            "metadata_only": True,
        }

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            status_code=500,
            retry_count=2,
            crunch_extra={"request_shape_repeated_context_canary": safety_lifecycle},
        )

        impact_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-safety-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact_report["status"], "no-widen")
        self.assertEqual(impact_report["summary"]["candidate_count"], 1)
        self.assertEqual(impact_report["summary"]["applied_count"], 0)
        self.assertEqual(impact_report["summary"]["holdout_count"], 0)
        self.assertEqual(impact_report["summary"]["safety_stop_count"], 1)
        self.assertEqual(impact_report["summary"]["top_blocker_code"], "canary-safety-stopped")
        self.assertEqual(impact_report["summary"]["top_impact_recommendation"], "rollback")
        self.assertEqual(impact_report["summary"]["rollback_recommended_count"], 1)
        self.assertEqual(impact_report["summary"]["next_action"], "rollback")
        self.assertEqual(impact_report["summary"]["recommended_next_action"], "rollback-repeated-context-crunch-canary")
        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "no-widen")
        self.assertEqual(candidate["impact_recommendation"], "rollback")
        self.assertEqual(candidate["recommended_next_action"], "rollback-repeated-context-crunch-canary")
        self.assertEqual(candidate["next_action"], "rollback")
        self.assertIn("canary-safety-stopped", candidate["reason_codes"])
        self.assertIn("missing-applied-coverage", candidate["reason_codes"])
        self.assertIn("missing-holdout-coverage", candidate["reason_codes"])
        self.assertEqual(candidate["top_blocker"], "canary-safety-stopped")
        self.assertEqual(candidate["promotion_metadata"]["impact_recommendation"], "rollback")
        self.assertEqual(candidate["promotion_metadata"]["next_action"], "rollback")
        feedback = impact_report["activation_lifecycle_feedback"]
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["safety_stop_count"], 1)
        self.assertIn("canary-safety-stopped", feedback["cohort_lifecycle_metadata"][0]["reason_codes"])

    def test_crunch_canary_impact_recommends_rollback_on_regressed_applied_cohort(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-regression-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]
        features = dict(action["conditions"])
        selected: dict[str, dict[str, object]] = {}
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"sample-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            status_code=500,
            latency_ms=900,
            retry_count=2,
            crunch_extra={
                "changed": True,
                "before_chars": 80_000,
                "after_chars": 72_000,
                "saved_chars": 8_000,
                "tokens_saved_est": 2_000,
                "request_shape_repeated_context_canary": selected["applied"],
            },
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            latency_ms=100,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )

        impact_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-regression-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact_report["status"], "no-widen")
        self.assertEqual(impact_report["summary"]["top_impact_recommendation"], "rollback")
        self.assertEqual(impact_report["summary"]["rollback_recommended_count"], 1)
        self.assertEqual(impact_report["summary"]["next_action"], "rollback")
        self.assertEqual(impact_report["summary"]["error_rate_delta"], 1.0)
        self.assertEqual(impact_report["summary"]["retry_rate_delta"], 2.0)
        self.assertEqual(impact_report["summary"]["latency_avg_delta_ms"], 800.0)

        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "no-widen")
        self.assertEqual(candidate["impact_recommendation"], "rollback")
        self.assertEqual(candidate["recommended_next_action"], "rollback-repeated-context-crunch-canary")
        self.assertEqual(candidate["next_action"], "rollback")
        self.assertEqual(candidate["latency_avg_delta_ms"], 800.0)
        self.assertIn("error-rate-regression", candidate["reason_codes"])
        self.assertIn("retry-rate-regression", candidate["reason_codes"])
        self.assertEqual(candidate["promotion_metadata"]["impact_recommendation"], "rollback")
        self.assertEqual(candidate["promotion_metadata"]["next_action"], "rollback")
        self.assertEqual(candidate["promotion_metadata"]["error_rate_delta"], 1.0)
        self.assertEqual(candidate["promotion_metadata"]["retry_rate_delta"], 2.0)
        self.assertEqual(candidate["promotion_metadata"]["latency_avg_delta_ms"], 800.0)

    def test_crunch_canary_impact_emits_collect_and_keep_blocked_recommendations(self) -> None:
        def impact_row(
            *,
            policy_id: str,
            cohort_id: str,
            status: str,
            saved_tokens: int = 0,
            saved_chars: int = 0,
        ) -> dict[str, object]:
            lifecycle = {
                "schema": "agentflow.request_shape_crunch_canary_lifecycle.v1",
                "policy_id": policy_id,
                "cohort_id": cohort_id,
                "status": status,
                "cohort": "canary_applied" if status == "applied" else "canary_holdout",
                "reason": status,
                "metadata_only": True,
            }
            return {
                "created_at": utc_now(),
                "provider_family": "anthropic",
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "workflow_phase": "thinking",
                "stream": True,
                "has_tools": True,
                "text_bucket": "gte_128k_chars",
                "token_bucket": "lt_500_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "actual_input_tokens": 20_000,
                "input_tokens_est": 20_000,
                "text_chars": 80_000,
                "cost_est_usd": 0.08,
                "status_code": 200,
                "retry_count": 0,
                "latency_ms": 125,
                "crunch_json": stable_json(
                    {
                        "changed": saved_tokens > 0,
                        "before_chars": 80_000,
                        "after_chars": max(0, 80_000 - saved_chars),
                        "saved_chars": saved_chars,
                        "tokens_saved_est": saved_tokens,
                        "request_shape_repeated_context_canary": lifecycle,
                    }
                ),
            }

        report = build_request_shape_crunch_canary_impact_report(
            [
                impact_row(policy_id="collect-policy", cohort_id="collect-cohort", status="applied", saved_tokens=2_000, saved_chars=8_000),
                impact_row(policy_id="blocked-policy", cohort_id="blocked-cohort", status="applied"),
                impact_row(policy_id="blocked-policy", cohort_id="blocked-cohort", status="holdout"),
            ]
        )

        by_policy = {candidate["policy_id"]: candidate for candidate in report["candidates"]}
        self.assertEqual(by_policy["collect-policy"]["impact_recommendation"], "collect-more-evidence")
        self.assertEqual(
            by_policy["collect-policy"]["recommended_next_action"],
            "collect-repeated-context-crunch-canary-impact-evidence",
        )
        self.assertEqual(by_policy["collect-policy"]["next_action"], "keep-observing")
        self.assertIn("missing-holdout-coverage", by_policy["collect-policy"]["reason_codes"])

        self.assertEqual(by_policy["blocked-policy"]["impact_recommendation"], "keep-blocked")
        self.assertEqual(
            by_policy["blocked-policy"]["recommended_next_action"],
            "keep-repeated-context-crunch-canary-blocked",
        )
        self.assertEqual(by_policy["blocked-policy"]["next_action"], "keep-observing")
        self.assertIn("no-applied-savings", by_policy["blocked-policy"]["reason_codes"])
        self.assertEqual(report["summary"]["collect_more_evidence_count"], 1)
        self.assertEqual(report["summary"]["keep_blocked_count"], 1)
        self.assertFalse(report["privacy"]["provider_bodies_included"])

    def test_crunch_policy_decision_promotes_positive_canary_with_rollback_metadata(self) -> None:
        def impact_row(status: str, *, saved_tokens: int = 0, saved_chars: int = 0, saved_usd: float = 0.0) -> dict[str, object]:
            lifecycle = {
                "schema": "agentflow.request_shape_crunch_canary_lifecycle.v1",
                "policy_id": "policy-promote",
                "cohort_id": "cohort-promote",
                "status": status,
                "cohort": "canary_applied" if status == "applied" else "canary_holdout",
                "reason": status,
                "policy_source": "local-manual",
                "metadata_only": True,
            }
            crunch_json = {
                "changed": saved_tokens > 0,
                "before_chars": 80_000,
                "after_chars": max(0, 80_000 - saved_chars),
                "saved_chars": saved_chars,
                "tokens_saved_est": saved_tokens,
                "request_shape_repeated_context_canary": lifecycle,
            }
            if saved_usd:
                crunch_json["savings_usd"] = saved_usd
            return {
                "created_at": utc_now(),
                "provider_family": "anthropic",
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "workflow_phase": "thinking",
                "stream": True,
                "has_tools": True,
                "text_bucket": "gte_128k_chars",
                "token_bucket": "lt_500_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "actual_input_tokens": 20_000,
                "input_tokens_est": 20_000,
                "text_chars": 80_000,
                "cost_est_usd": 0.08,
                "status_code": 200,
                "retry_count": 0,
                "latency_ms": 125,
                "crunch_json": stable_json(crunch_json),
            }

        impact_report = build_request_shape_crunch_canary_impact_report(
            [
                impact_row("applied", saved_tokens=2_000, saved_chars=8_000, saved_usd=0.0125),
                impact_row("holdout"),
            ]
        )
        decision = build_request_shape_crunch_policy_decision_report(impact_report)

        self.assertEqual(decision["schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["graduation_decision"], "widen")
        self.assertEqual(decision["summary"]["decision"], "promote")
        self.assertEqual(decision["summary"]["graduation_decision"], "widen")
        self.assertTrue(decision["summary"]["promotion_allowed"])
        self.assertEqual(decision["summary"]["applied_count"], 1)
        self.assertEqual(decision["summary"]["holdout_count"], 1)
        self.assertEqual(decision["summary"]["observed_saved_tokens"], 2_000)
        self.assertEqual(decision["summary"]["observed_saved_usd"], 0.0125)
        self.assertEqual(decision["summary"]["error_rate_delta"], 0.0)
        self.assertEqual(decision["summary"]["retry_rate_delta"], 0.0)
        self.assertEqual(decision["summary"]["fallback_rate_delta"], 0.0)
        self.assertEqual(decision["summary"]["policy_source"], "local-manual")
        self.assertFalse(decision["summary"]["policy_files_written"])
        top = decision["top_decision"]
        self.assertEqual(top["local_policy_patch"]["patch_type"], "promote_repeated_context_crunch_canary")
        self.assertEqual(top["local_policy_patch"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(top["rollback_metadata"]["rollback_action_type"], "disable_repeated_context_crunch_canary")
        self.assertTrue(top["rollback_metadata"]["required_for_promotion"])
        self.assertTrue(top["coverage"]["has_applied_coverage"])
        self.assertTrue(top["coverage"]["has_holdout_coverage"])
        ledger = build_request_shape_crunch_policy_decision_ledger(
            decision,
            recorded_at="2026-06-15T18:00:00+00:00",
        )
        self.assertEqual(ledger["schema"], "agentflow.request_shape_crunch_policy_decision_ledger.v1")
        self.assertEqual(ledger["status"], "recordable")
        self.assertEqual(ledger["entries"][0]["status"], "positive")
        self.assertEqual(ledger["entries"][0]["recommendation"], "widen")
        self.assertEqual(ledger["entries"][0]["applied_count"], 1)
        self.assertEqual(ledger["entries"][0]["holdout_count"], 1)
        first = record_request_shape_crunch_policy_decision_ledger(
            decision,
            store_obj=self.store,
            recorded_at="2026-06-15T18:00:01+00:00",
        )
        rows = self.store.promotion_outcome_feedback_rows(action_family="crunch", limit=10)
        self.assertTrue(first["wrote_store"])
        self.assertEqual(first["summary"]["rows_written"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_evidence_schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(rows[0]["status"], "positive")
        self.assertEqual(rows[0]["recommendation"], "widen")
        self.assertEqual(rows[0]["applied_count"], 1)
        self.assertEqual(rows[0]["holdout_count"], 1)
        self.assertTrue(decision["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])
        self.assertFalse(decision["privacy"]["raw_prompts_included"])
        self.assertFalse(decision["privacy"]["provider_bodies_included"])
        self.assertFalse(decision["privacy"]["request_ids_included"])
        self.assertFalse(decision["privacy"]["session_ids_included"])

    def test_crunch_policy_decision_rolls_back_on_safety_stop(self) -> None:
        lifecycle = {
            "schema": "agentflow.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "policy-safety",
            "cohort_id": "cohort-safety",
            "status": "safety-stopped",
            "cohort": "safety_stopped",
            "reason": "error-rate-regression",
            "safety_stop": True,
            "metadata_only": True,
        }
        impact_report = build_request_shape_crunch_canary_impact_report(
            [
                {
                    "created_at": utc_now(),
                    "provider_family": "anthropic",
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "actual_input_tokens": 20_000,
                    "input_tokens_est": 20_000,
                    "text_chars": 80_000,
                    "cost_est_usd": 0.08,
                    "status_code": 500,
                    "retry_count": 2,
                    "latency_ms": 125,
                    "crunch_json": stable_json({"request_shape_repeated_context_canary": lifecycle}),
                }
            ]
        )
        decision = build_request_shape_crunch_policy_decision_report(impact_report)

        self.assertEqual(decision["decision"], "rollback")
        self.assertEqual(decision["graduation_decision"], "rollback")
        self.assertTrue(decision["summary"]["rollback_required"])
        self.assertEqual(decision["summary"]["safety_stop_state"], "observed")
        self.assertEqual(decision["top_decision"]["reason"], "canary-safety-stopped")
        self.assertFalse(decision["top_decision"]["promotion_allowed"])
        self.assertEqual(decision["top_decision"]["local_policy_patch"]["patch_type"], "rollback_repeated_context_crunch_canary")
        self.assertIn("canary-safety-stopped", decision["top_decision"]["reason_codes"])
        self.assertEqual(decision["top_decision"]["metrics"]["safety_stop_count"], 1)

    def test_crunch_policy_decision_cli_keeps_blocked_without_canary_metadata(self) -> None:
        self._log_call()

        stdout = io.StringIO()
        code = cli.request_shape_crunch_policy_decision_cli(
            ["--db", self.db_path, "--limit", "10"],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(payload["decision"], "keep-blocked")
        self.assertEqual(payload["graduation_decision"], "blocked")
        self.assertTrue(payload["summary"]["keep_blocked"])
        self.assertEqual(payload["summary"]["applied_count"], 0)
        self.assertEqual(payload["summary"]["holdout_count"], 0)
        self.assertEqual(payload["top_decision"]["reason"], "missing-applied-or-holdout-coverage")
        self.assertEqual(payload["ledger_update"]["schema"], "agentflow.request_shape_crunch_policy_decision_ledger.v1")
        self.assertEqual(payload["ledger_update"]["status"], "recorded")
        self.assertEqual(payload["ledger_update"]["summary"]["rows_written"], 1)
        rows = self.store.promotion_outcome_feedback_rows(action_family="crunch", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "needs-more-samples")
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertTrue(payload["source_rollups"]["privacy"]["metadata_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())
        self.assertNotIn("raw-session-id-must-not-leak", stdout.getvalue())

    def test_cli_persists_rollups_by_default_and_dry_run_skips_write(self) -> None:
        self._log_call()
        self._log_call()

        stdout = io.StringIO()
        code = cli.request_shape_rollups_cli(
            ["--db", self.db_path, "--limit", "10", "--run-id", "cli-rollup"],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["persisted"])
        self.assertEqual(payload["persisted_count"], 1)
        self.assertEqual(len(self.store.request_shape_rollup_rows(run_id="cli-rollup")), 1)

        stdout = io.StringIO()
        code = cli.request_shape_rollups_cli(
            ["--db", self.db_path, "--limit", "10", "--run-id", "cli-dry", "--dry-run"],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["persisted"])
        self.assertEqual(payload["persisted_count"], 0)
        self.assertEqual(self.store.request_shape_rollup_rows(run_id="cli-dry"), [])
