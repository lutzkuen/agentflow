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
    apply_request_shape_crunch_canary_action,
    build_request_shape_rollups_report,
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
            latency_ms=125,
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
        self.assertEqual(follow_up["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(follow_up["target_local_policy"], "crunch_rules")
        self.assertEqual(follow_up["recommended_action_count"], 1)
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
        self.assertEqual(impact_report["summary"]["applied_count"], 1)
        self.assertEqual(impact_report["summary"]["holdout_count"], 1)
        self.assertEqual(impact_report["summary"]["saved_chars"], 8_000)
        self.assertEqual(impact_report["summary"]["saved_tokens"], 2_000)
        self.assertGreater(impact_report["summary"]["saved_usd"], 0)
        self.assertEqual(impact_report["summary"]["error_rate_delta"], 0.0)
        self.assertEqual(impact_report["summary"]["retry_rate_delta"], 0.0)
        self.assertEqual(impact_report["summary"]["fallback_count"], 0)
        self.assertEqual(impact_report["summary"]["safety_stop_count"], 0)
        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "widen-ready")
        self.assertIsNone(candidate["top_blocker"])
        self.assertEqual(candidate["cohorts"]["canary_applied"]["saved_chars"], 8_000)
        self.assertEqual(candidate["cohorts"]["canary_holdout"]["saved_chars"], 0)
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
        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "no-widen")
        self.assertIn("canary-safety-stopped", candidate["reason_codes"])
        self.assertIn("missing-applied-coverage", candidate["reason_codes"])
        self.assertIn("missing-holdout-coverage", candidate["reason_codes"])
        self.assertEqual(candidate["top_blocker"], "canary-safety-stopped")
        feedback = impact_report["activation_lifecycle_feedback"]
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["safety_stop_count"], 1)
        self.assertIn("canary-safety-stopped", feedback["cohort_lifecycle_metadata"][0]["reason_codes"])

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
