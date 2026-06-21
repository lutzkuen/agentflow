from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from tokenclaw import cli
from tokenclaw.cache_smoke import build_cache_smoke_diagnostic
from tokenclaw.dashboard_app import create_dashboard_app
from tokenclaw.openai_cache_replay_apply import build_openai_cache_replay_apply_plan
from tokenclaw.openai_cache_replay_blocker_outcomes import build_openai_cache_replay_blocker_outcomes_report
from tokenclaw.openai_cache_replay_dry_run import build_openai_cache_replay_dry_run
from tokenclaw.openai_cache_replay_impact import build_openai_cache_replay_impact_report
from tokenclaw.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report
from tokenclaw.openai_cache_replay_report import build_openai_cache_replay_report
from tokenclaw.provider_adoption import capture_provider_tool_adoption
from tokenclaw.request_shape_rollups import (
    build_request_shape_cache_replay_evidence_report,
    build_request_shape_cache_replay_policy_decision_report,
)
from tokenclaw.stats import (
    stats_openai_cache_replay_impact,
    stats_openai_cache_replay_readiness,
    stats_openai_cache_replay_report,
    stats_openai_tool_cache_invalidation_burndown,
)
from tokenclaw.store import SQLiteStore, stable_json, utc_now


def _reload_cache_module_for_test():
    from tokenclaw import anthropic_proxy
    from tokenclaw import cache as cache_module

    reloaded = importlib.reload(cache_module)
    anthropic_proxy.cache_lookup_meta = reloaded.cache_lookup_meta
    anthropic_proxy.cache_replay_canary_decision = reloaded.cache_replay_canary_decision
    anthropic_proxy.streaming_cache_lookup_meta = reloaded.streaming_cache_lookup_meta
    return reloaded


class OpenAICacheReplayReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _audit(self, *, reason: str | None = None, safe: bool = False) -> dict[str, object]:
        return {
            "schema": "tokenclaw.cache_file_dependency_audit.v1",
            "file_watch_enabled": True,
            "snapshot_root_policy": "stored-local-paths",
            "root_path_included": False,
            "snapshot_count": 1 if safe else 0,
            "snapshot_count_bucket": "1" if safe else "0",
            "candidate_path_count_bucket": "1" if safe else "0",
            "max_paths": None,
            "cap_exceeded": False,
            "present_path_count": 1 if safe else 0,
            "missing_path_count": 0,
            "changed_path_count": 1 if reason == "dependency-changed" else 0,
            "deleted_path_count": 0,
            "created_path_count": 0,
            "invalidation_reason": reason,
            "safe_invalidation_evidence": safe,
            "file_dependency_evidence_available": safe,
            "paths_included": False,
        }

    def _log_openai_call(
        self,
        *,
        endpoint: str = "responses",
        category: str = "chat",
        cache_status: str = "miss",
        cache_reason: str = "exact-miss",
        cache_hit: int = 0,
        stream: int = 0,
        has_tools: bool = False,
        request_fingerprint: str | None = None,
        pattern_hashes: list[str] | None = None,
        file_dependency_audit: dict[str, object] | None = None,
        cost: float = 0.01,
        cost_baseline: float | None = None,
        status_code: int = 200,
        latency_ms: int = 125,
        retry_count: int = 0,
        cache_extra: dict[str, object] | None = None,
        session_id: str = "raw-openai-session-must-not-leak",
        created_at: str | None = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        path = "/v1/responses" if endpoint == "responses" else "/v1/chat/completions"
        text_chars = 2400
        input_tokens = text_chars // 4
        cache_json: dict[str, object] = {
            "status": cache_status,
            "reason": cache_reason,
            "policy_source": "local-default",
            "replayability_level": "local-exact-response",
        }
        if request_fingerprint:
            cache_json["pattern_features"] = {"request_fingerprint": request_fingerprint}
        if pattern_hashes:
            features = cache_json.setdefault("pattern_features", {})
            if isinstance(features, dict):
                features["pattern_hashes"] = pattern_hashes
        if file_dependency_audit is not None:
            cache_json["file_dependency_audit"] = file_dependency_audit
            cache_json["file_dependency_evidence_available"] = bool(
                file_dependency_audit.get("file_dependency_evidence_available")
            )
            cache_json["safe_invalidation_evidence"] = bool(file_dependency_audit.get("safe_invalidation_evidence"))
        if cache_extra:
            cache_json.update(cache_extra)

        self.store.log_call(
            id=call_id,
            created_at=created_at or utc_now(),
            path=path,
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            stream=stream,
            cache_hit=cache_hit,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=input_tokens,
            output_tokens_est=50,
            actual_input_tokens=input_tokens,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=cost if cost_baseline is None else cost_baseline,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json(
                {
                    "enabled": False,
                    "provider": "openai",
                    "requested_model": "gpt-5.4-mini",
                    "routed_model": "gpt-5.4-mini",
                    "text_chars": text_chars,
                    "has_tools": has_tools,
                    "category": category,
                    "openai_feature_unit": {
                        "source_surface": f"openai_{endpoint}",
                        "endpoint": endpoint,
                        "category": category,
                        "workflow_phase": category,
                        "requested_model_family": "gpt-5",
                    },
                }
            ),
            cache_json=stable_json(cache_json),
            error=None,
            request_json=stable_json({"input": "raw prompt must not leak", "cache_key": "raw-cache-key-secret"}),
            response_json=stable_json({"output_text": "raw response must not leak"}),
            session_id=session_id,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface=f"openai_{endpoint}",
            endpoint=endpoint,
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )
        return call_id

    def _cache_replay_meta(
        self,
        *,
        candidate_id: str = "openai-cache-readiness-candidate",
        rule_id: str = "openai-cache-readiness-rule",
        canary_status: str,
        cohort: str,
        reason: str,
        projected: float = 0.03,
    ) -> dict[str, object]:
        rule: dict[str, object] = {
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {
                "enabled": True,
                "selected": cohort == "canary_applied",
                "cohort": cohort,
                "fraction": 0.5,
                "unit": "session",
                "status": canary_status,
                "pattern_hashes": ["sha256:" + "f" * 64],
            },
            "graduation": {
                "source_schema": "tokenclaw.openai_cache_replay_opportunity.v1",
                "projected_hits": 10,
                "projected_savings_usd": projected,
                "sample_count": 10,
            },
        }
        return {
            "pattern_rule": rule,
            "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
            "cache_replay_canary": {
                "schema": "tokenclaw.cache_replay_canary_decision.v1",
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "policy_source": "local-manual",
                "status": canary_status,
                "reason": reason,
                "canary": rule["canary"],
                "canary_cohort": cohort,
                "projected_input_savings_usd": projected,
            },
            "estimated_saved_cost_usd": projected,
        }

    def test_openai_cache_replay_impact_recommends_stage_when_canary_evidence_is_missing(self) -> None:
        report = build_openai_cache_replay_impact_report(self.store, limit=20)

        self.assertEqual(report["schema"], "tokenclaw.openai_cache_replay_impact.v1")
        self.assertEqual(report["status"], "no-openai-cache-replay-metadata")
        evidence = report["local_promotion_evidence"]
        self.assertEqual(evidence["schema"], "tokenclaw.openai_cache_replay_local_promotion_evidence.v1")
        self.assertEqual(evidence["status"], "missing-canary-evidence")
        self.assertEqual(evidence["recommended_local_action"]["action"], "stage-cache-replay-canary")
        self.assertEqual(evidence["top_blocker"], "missing-cache-replay-canary-lifecycle-evidence")
        self.assertEqual(evidence["coverage"]["observed_replay_metadata_rows"], 0)
        self.assertEqual(evidence["outcomes"]["observed_hits"], 0)
        self.assertEqual(evidence["savings"]["projected_saved_usd"], 0.0)
        self.assertEqual(report["summary"]["recommended_local_action"], "stage-cache-replay-canary")
        self.assertFalse(evidence["privacy"]["raw_request_bodies_included"])
        self.assertFalse(evidence["privacy"]["cache_keys_included"])
        self.assertFalse(evidence["provider_calls_made"])

    def test_openai_cache_replay_readiness_keeps_staged_for_holdout_only_canary(self) -> None:
        for _ in range(2):
            self._log_openai_call(
                cache_status="bypassed",
                cache_reason="canary_holdout",
                cost=0.03,
                cost_baseline=0.03,
                cache_extra=self._cache_replay_meta(
                    canary_status="holdout",
                    cohort="canary_holdout",
                    reason="canary_holdout",
                ),
            )

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        decision = report["promotion_decision"]

        self.assertEqual(decision["schema"], "tokenclaw.openai_cache_replay_promotion_decision.v1")
        self.assertEqual(decision["decision"], "keep-staged")
        self.assertTrue(decision["keep_staged"])
        self.assertFalse(decision["promotion_allowed"])
        self.assertEqual(decision["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(decision["coverage"]["applied_count"], 0)
        self.assertEqual(decision["coverage"]["holdout_count"], 2)
        self.assertFalse(decision["coverage"]["has_applied_coverage"])
        self.assertTrue(decision["coverage"]["has_holdout_coverage"])
        self.assertIn("insufficient-applied-coverage", decision["reason_codes"])
        self.assertEqual(report["summary"]["promotion_decision"], "keep-staged")
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("sha256:" + "f" * 64, rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)
        self.assertFalse(decision["privacy"]["cache_keys_included"])
        self.assertFalse(decision["privacy"]["request_ids_included"])

    def test_openai_cache_replay_readiness_keeps_staged_for_warmup_misses_with_holdout(self) -> None:
        for _ in range(24):
            self._log_openai_call(
                cache_status="miss",
                cache_reason="cache-warmup-miss",
                cost=0.03,
                cost_baseline=0.03,
                cache_extra=self._cache_replay_meta(
                    canary_status="applied",
                    cohort="canary_applied",
                    reason="cache-warmup-miss",
                    projected=0.0021535,
                ),
            )
        for _ in range(16):
            self._log_openai_call(
                cache_status="bypassed",
                cache_reason="canary_holdout",
                cost=0.03,
                cost_baseline=0.03,
                cache_extra=self._cache_replay_meta(
                    canary_status="holdout",
                    cohort="canary_holdout",
                    reason="canary_holdout",
                    projected=0.0021535,
                ),
            )

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=100, impact_limit=100)
        decision = report["promotion_decision"]
        blockers = {row["value"]: row["count"] for row in decision["applied_miss_blocker_breakdown"]}

        self.assertEqual(decision["schema"], "tokenclaw.openai_cache_replay_promotion_decision.v1")
        self.assertEqual(decision["decision"], "keep-staged")
        self.assertEqual(decision["reason"], "cache-warmup-miss")
        self.assertEqual(decision["recommended_next_action"], "keep-openai-exact-cache-replay-canary-staged")
        self.assertTrue(decision["keep_staged"])
        self.assertFalse(decision["promotion_allowed"])
        self.assertEqual(decision["coverage"]["applied_count"], 24)
        self.assertEqual(decision["coverage"]["holdout_count"], 16)
        self.assertEqual(decision["coverage"]["miss_count"], 24)
        self.assertEqual(decision["coverage"]["observed_hits"], 0)
        self.assertEqual(decision["summary"]["top_applied_miss_blocker"], "cache-warmup-miss")
        self.assertEqual(blockers["cache-warmup-miss"], 24)
        self.assertEqual(decision["hit_recovery"]["schema"], "tokenclaw.openai_cache_replay_hit_recovery.v1")
        self.assertEqual(decision["hit_recovery"]["status"], "awaiting-live-hit")
        self.assertEqual(decision["hit_recovery"]["applied_count"], 24)
        self.assertEqual(decision["hit_recovery"]["holdout_count"], 16)
        self.assertEqual(decision["hit_recovery"]["observed_hits"], 0)
        self.assertEqual(
            decision["invalidation_safety"]["schema"],
            "tokenclaw.openai_cache_replay_invalidation_safety.v1",
        )
        self.assertEqual(decision["invalidation_safety"]["status"], "passed")
        self.assertTrue(decision["invalidation_safety"]["safe_for_promotion"])
        self.assertEqual(decision["coverage"]["invalidation_skipped_count"], 0)
        self.assertTrue(decision["coverage"]["has_clean_invalidation_safety"])
        self.assertIn("cache-warmup-miss", decision["reason_codes"])
        self.assertIn("applied-miss:cache-warmup-miss", decision["reason_codes"])
        self.assertEqual(report["summary"]["promotion_decision"], "keep-staged")
        self.assertEqual(report["summary"]["promotion_blocker"], "cache-warmup-miss")
        self.assertTrue(decision["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw-cache-key-secret", rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)
        self.assertFalse(decision["privacy"]["cache_keys_included"])
        self.assertFalse(decision["privacy"]["request_ids_included"])
        self.assertFalse(decision["privacy"]["session_ids_included"])

    def test_openai_cache_replay_impact_classifies_applied_miss_reasons_without_raw_metadata(self) -> None:
        cases = [
            (
                "exact-pattern-miss",
                {
                    "cache_replay_store": {
                        "status": "stored",
                        "reason": "compatible-success-response",
                        "cache_key_included": False,
                    }
                },
                "cache-warmup-miss",
            ),
            (
                "exact-pattern-miss",
                {
                    "cache_replay_store": {
                        "status": "skipped",
                        "reason": "responses-output-missing",
                        "cache_key_included": False,
                    }
                },
                "cache-write-absence",
            ),
            ("ttl-expired-without-tool-result", {}, "ttl-expiry"),
            (
                "exact-pattern-miss",
                {"pattern_rules": {"skip_reasons": [{"reason": "pattern-hash-mismatch"}]}},
                "fingerprint-drift",
            ),
            ("normalization-drift", {}, "normalization-mismatch"),
            (
                "exact-pattern-miss",
                {"file_dependency_audit": self._audit(reason="file-dependency-missing", safe=False)},
                "invalidation-evidence-missing",
            ),
            ("exact-cache-disabled", {"status": "disabled"}, "cache-policy-disabled"),
            ("holdout-bypass", {}, "holdout-bypass"),
        ]
        for cache_reason, extra, _expected in cases:
            meta = self._cache_replay_meta(
                canary_status="applied",
                cohort="canary_applied",
                reason=cache_reason,
                projected=0.0021535,
            )
            meta.update(extra)
            self._log_openai_call(
                cache_status=str(extra.get("status") or "miss"),
                cache_reason=cache_reason,
                cost=0.03,
                cost_baseline=0.03,
                cache_extra=meta,
            )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            cache_extra=self._cache_replay_meta(
                canary_status="holdout",
                cohort="canary_holdout",
                reason="canary_holdout",
                projected=0.0021535,
            ),
        )

        impact = build_openai_cache_replay_impact_report(self.store, limit=50)
        readiness = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=50, impact_limit=50)
        impact_reasons = {row["value"]: row["count"] for row in impact["miss_reason_breakdown"]}
        decision_reasons = {row["value"]: row["count"] for row in readiness["promotion_decision"]["miss_reason_breakdown"]}

        self.assertEqual(impact["summary"]["applied_count"], len(cases))
        self.assertEqual(impact["summary"]["holdout_count"], 1)
        self.assertEqual(impact["summary"]["miss_count"], len(cases) - 1)
        for _cache_reason, _extra, expected in cases:
            self.assertGreaterEqual(impact_reasons[expected], 1)
            self.assertGreaterEqual(decision_reasons[expected], 1)
        self.assertEqual(impact["local_promotion_evidence"]["outcomes"]["miss_reason_breakdown"], impact["miss_reason_breakdown"])
        self.assertEqual(
            readiness["promotion_decision"]["applied_miss_blocker_breakdown"],
            readiness["promotion_decision"]["miss_reason_breakdown"],
        )
        self.assertTrue(readiness["promotion_decision"]["privacy"]["metadata_only"])
        self.assertTrue(readiness["promotion_decision"]["privacy"]["aggregate_only"])
        rendered = json.dumps({"impact": impact, "readiness": readiness}, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw-cache-key-secret", rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)
        self.assertFalse(impact["privacy"]["cache_keys_included"])
        self.assertFalse(readiness["promotion_decision"]["privacy"]["request_ids_included"])
        self.assertFalse(readiness["promotion_decision"]["privacy"]["session_ids_included"])

    def test_openai_cache_replay_readiness_widens_positive_applied_and_holdout_canary(self) -> None:
        for index, baseline in enumerate((0.03, 0.04)):
            call_id = self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=baseline,
                cache_extra=self._cache_replay_meta(
                    canary_status="applied",
                    cohort="canary_applied",
                    reason="dependency-stable",
                    projected=baseline,
                ),
            )
            self._capture_openai_provider_adoption(call_id, tool_id=f"readiness_cache_apply_{index}")
        holdout_call_id = self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            cache_extra=self._cache_replay_meta(
                canary_status="holdout",
                cohort="canary_holdout",
                reason="canary_holdout",
            ),
        )
        self._capture_openai_provider_adoption(holdout_call_id, tool_id="readiness_cache_holdout")

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        decision = report["promotion_decision"]

        self.assertEqual(decision["decision"], "widen")
        self.assertTrue(decision["promotion_allowed"])
        self.assertFalse(decision["rollback_required"])
        self.assertEqual(decision["recommended_next_action"], "widen-openai-exact-cache-replay-policy")
        self.assertEqual(decision["coverage"]["applied_count"], 2)
        self.assertEqual(decision["coverage"]["holdout_count"], 1)
        self.assertEqual(decision["coverage"]["observed_hits"], 2)
        self.assertGreater(decision["outcomes"]["observed_savings_usd"], 0)
        self.assertEqual(decision["hit_recovery"]["status"], "hit-recovered")
        self.assertEqual(decision["hit_recovery"]["observed_hits"], 2)
        self.assertEqual(decision["invalidation_safety"]["status"], "passed")
        self.assertTrue(decision["invalidation_safety"]["safe_for_promotion"])
        self.assertEqual(decision["coverage"]["invalidation_skipped_count"], 0)
        self.assertIn("target-savings-met", decision["reason_codes"])
        self.assertEqual(report["summary"]["promotion_allowed"], True)

    def test_openai_cache_replay_readiness_rolls_back_failed_invalidation_safety(self) -> None:
        for index, baseline in enumerate((0.03, 0.04)):
            call_id = self._log_openai_call(
                category="tool-light",
                has_tools=True,
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=baseline,
                cache_extra=self._cache_replay_meta(
                    candidate_id="openai-tool-cache-invalidation",
                    rule_id="openai-tool-cache-invalidation-rule",
                    canary_status="applied",
                    cohort="canary_applied",
                    reason="dependency-stable",
                    projected=baseline,
                ),
            )
            self._capture_openai_provider_adoption(call_id, tool_id=f"tool_cache_apply_{index}")
        holdout_call_id = self._log_openai_call(
            category="tool-light",
            has_tools=True,
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            cache_extra=self._cache_replay_meta(
                candidate_id="openai-tool-cache-invalidation",
                rule_id="openai-tool-cache-invalidation-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                reason="canary_holdout",
            ),
        )
        self._capture_openai_provider_adoption(holdout_call_id, tool_id="tool_cache_holdout")
        self._log_openai_call(
            category="tool-light",
            has_tools=True,
            cache_status="invalidated",
            cache_reason="dependency-changed",
            cost=0.03,
            cost_baseline=0.03,
            cache_extra={
                **self._cache_replay_meta(
                    candidate_id="openai-tool-cache-invalidation",
                    rule_id="openai-tool-cache-invalidation-rule",
                    canary_status="invalidated",
                    cohort="canary_applied",
                    reason="dependency-changed",
                ),
                "invalidated": True,
                "invalidation_reason": "dependency-changed",
                "file_dependency_audit": self._audit(reason="dependency-changed", safe=False),
            },
        )

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        decision = report["promotion_decision"]

        self.assertEqual(decision["decision"], "rollback")
        self.assertTrue(decision["rollback_required"])
        self.assertFalse(decision["promotion_allowed"])
        self.assertEqual(decision["recommended_next_action"], "disable-openai-exact-cache-replay-canary")
        self.assertEqual(decision["hit_recovery"]["status"], "hit-recovered")
        self.assertEqual(decision["hit_recovery"]["observed_hits"], 2)
        self.assertEqual(decision["invalidation_safety"]["status"], "failed")
        self.assertFalse(decision["invalidation_safety"]["safe_for_promotion"])
        self.assertEqual(decision["invalidation_safety"]["invalidated_count"], 1)
        self.assertEqual(decision["coverage"]["invalidation_skipped_count"], 1)
        self.assertFalse(decision["coverage"]["has_clean_invalidation_safety"])
        self.assertIn("invalidation-safety-failed", decision["reason_codes"])
        self.assertIn("invalidation-skipped-observed", decision["reason_codes"])
        self.assertEqual(report["summary"]["rollback_required"], True)

    def test_openai_cache_replay_readiness_rolls_back_stale_canary_evidence(self) -> None:
        for index, baseline in enumerate((0.03, 0.04)):
            self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=baseline,
                created_at=f"2026-01-01T00:0{index}:00+00:00",
                cache_extra=self._cache_replay_meta(
                    canary_status="applied",
                    cohort="canary_applied",
                    reason="dependency-stable",
                    projected=baseline,
                ),
            )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            created_at="2026-01-01T00:02:00+00:00",
            cache_extra=self._cache_replay_meta(
                canary_status="holdout",
                cohort="canary_holdout",
                reason="canary_holdout",
            ),
        )

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        decision = report["promotion_decision"]

        self.assertEqual(decision["decision"], "rollback")
        self.assertTrue(decision["rollback_required"])
        self.assertFalse(decision["promotion_allowed"])
        self.assertEqual(decision["recommended_next_action"], "disable-openai-exact-cache-replay-canary")
        self.assertTrue(decision["stale_evidence"]["stale"])
        self.assertIn("stale-cache-replay-evidence", decision["reason_codes"])
        self.assertEqual(report["summary"]["rollback_required"], True)

    def _capture_openai_provider_adoption(self, fulfilled_call_id: str, *, tool_id: str) -> None:
        capture_provider_tool_adoption(
            self.store,
            provider="openai",
            path="/v1/responses",
            call_id=f"tool-use-{fulfilled_call_id}",
            session_id="raw-openai-session-must-not-leak",
            request_body={"model": "gpt-5.4-mini", "input": [{"role": "user", "content": "hi"}]},
            response_body={
                "output": [
                    {
                        "type": "function_call",
                        "call_id": tool_id,
                        "name": "lookup",
                        "arguments": '{"path":"/tmp/private.py"}',
                    }
                ]
            },
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            status_code=200,
            category="tool-light",
            routing_meta={"policy_source": "managed-recommended", "phase": "tool-execution"},
        )
        capture_provider_tool_adoption(
            self.store,
            provider="openai",
            path="/v1/responses",
            call_id=fulfilled_call_id,
            session_id="raw-openai-session-must-not-leak",
            request_body={
                "model": "gpt-5.4-mini",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": tool_id,
                        "output": "secret tool payload",
                    }
                ],
            },
            response_body={"output": []},
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            status_code=200,
            category="tool-result",
            routing_meta={"policy_source": "managed-recommended", "phase": "tool-execution"},
        )

    def _capture_openai_orphan_provider_adoption(self, call_id: str, *, tool_id: str) -> None:
        capture_provider_tool_adoption(
            self.store,
            provider="openai",
            path="/v1/responses",
            call_id=call_id,
            session_id="raw-openai-session-must-not-leak",
            request_body={
                "model": "gpt-5.4-mini",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": tool_id,
                        "output": "orphan secret tool payload",
                    }
                ],
            },
            response_body={"output": []},
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            status_code=200,
            category="tool-result",
            routing_meta={"policy_source": "managed-recommended", "phase": "tool-execution"},
        )

    def test_report_groups_openai_replay_candidates_and_blockers_without_raw_fields(self) -> None:
        for cost in (0.01, 0.02, 0.03):
            self._log_openai_call(request_fingerprint="raw-request-fingerprint-must-not-leak", cost=cost)
        self._log_openai_call(
            endpoint="chat_completions",
            category="tool-light",
            cache_status="skipped",
            cache_reason="tools-disabled",
            has_tools=True,
            file_dependency_audit=self._audit(reason="file-dependency-missing", safe=False),
            cost=0.04,
        )
        self._log_openai_call(
            category="tool-light",
            cache_status="skipped",
            cache_reason="tools-disabled",
            has_tools=True,
            file_dependency_audit=self._audit(reason="dependency-changed", safe=False),
            cost=0.05,
        )
        self._log_openai_call(cache_status="skipped", cache_reason="streaming", stream=1, cost=0.015)
        self._log_openai_call(cache_status="hit", cache_reason="exact-match", cache_hit=1, cost=0.005)

        report = build_openai_cache_replay_report(self.store, limit=20)

        self.assertEqual(report["schema"], "tokenclaw.openai_cache_replay_opportunity.v1")
        self.assertEqual(report["summary"]["openai_call_count"], 7)
        self.assertEqual(report["summary"]["request_fingerprint_rows"], 3)
        self.assertEqual(report["summary"]["request_body_rows_present_but_not_read"], 7)
        self.assertGreater(report["summary"]["projected_savings_usd"], 0)

        blockers = {row["value"]: row["count"] for row in report["blocker_reason_breakdown"]}
        self.assertEqual(blockers["exact-miss"], 3)
        self.assertEqual(blockers["replay-rule-required"], 3)
        self.assertEqual(blockers["tool-call-cache-disabled"], 2)
        self.assertEqual(blockers["file-dependency-missing"], 1)
        self.assertEqual(blockers["file-dependency-invalidated"], 1)
        self.assertEqual(blockers["unsupported-streaming-shape"], 1)
        self.assertEqual(blockers["already-cache-hit"], 1)

        replay_candidate = next(row for row in report["candidates"] if row["request_fingerprint_available"])
        self.assertEqual(replay_candidate["matched_count"], 3)
        self.assertEqual(replay_candidate["duplicate_fingerprint_groups"], 1)
        self.assertEqual(replay_candidate["duplicate_fingerprint_rows"], 3)
        self.assertGreater(replay_candidate["projected_savings_usd"], 0)
        self.assertFalse(replay_candidate["privacy"]["request_fingerprint_included"])

        dependency_candidate = next(row for row in report["candidates"] if row["file_dependency_status"] == "invalidated")
        self.assertFalse(dependency_candidate["file_dependency_audit"]["paths_included"])
        self.assertFalse(dependency_candidate["file_dependency_audit"]["root_path_included"])

        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-cache-key-secret", rendered)
        self.assertNotIn("raw-request-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(report["privacy"]["file_paths_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["request_fingerprints_included"])
        self.assertFalse(report["privacy"]["provider_calls_made"])

    def test_cache_replay_diagnostics_sanitize_raw_like_metadata_labels(self) -> None:
        self._log_openai_call(
            category="raw prompt must not leak /tmp/openai-category-secret.py",
            cache_status="skipped",
            cache_reason="cache-key-secret req-secret prompt body",
            request_fingerprint="raw-openai-diagnostic-fingerprint-secret",
            cost=0.02,
        )

        opportunity = build_openai_cache_replay_report(self.store, limit=20)
        readiness = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        smoke = build_cache_smoke_diagnostic(self.store, limit=20, scan_limit=20)
        stats_result = asyncio.run(stats_openai_cache_replay_report(self.store, limit=20))

        rendered = json.dumps(
            {
                "opportunity": opportunity,
                "readiness": readiness,
                "smoke": smoke,
                "stats": stats_result,
            },
            sort_keys=True,
        )
        for forbidden in (
            "raw prompt must not leak",
            "/tmp/openai-category-secret.py",
            "cache-key-secret",
            "req-secret",
            "prompt body",
            "raw-openai-diagnostic-fingerprint-secret",
            "raw prompt must not leak",
            "raw response must not leak",
            "raw-cache-key-secret",
            "raw-openai-session-must-not-leak",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIn('"category": "unknown"', rendered)
        self.assertIn('"cache_reason": "unknown"', rendered)
        self.assertFalse(opportunity["privacy"]["raw_request_bodies_included"])
        self.assertFalse(readiness["privacy"]["file_paths_included"])
        self.assertFalse(smoke["privacy"]["cache_keys_included"])

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        self._log_openai_call(request_fingerprint="raw-cli-request-fingerprint")
        self._log_openai_call(request_fingerprint="raw-cli-request-fingerprint")

        result = asyncio.run(stats_openai_cache_replay_report(self.store, limit=10))
        self.assertEqual(result["schema"], "tokenclaw.openai_cache_replay_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.openai_cache_replay_report_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.openai_cache_replay_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 2)
        self.assertNotIn("raw-cli-request-fingerprint", output.getvalue())

    def test_openai_dependency_evidence_distinguishes_stable_files_from_missing_files(self) -> None:
        from tokenclaw import cache as cache_module

        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "stable.py").write_text("VALUE = 1\n", encoding="utf-8")
            os.chdir(tmp_path)
            try:
                stable_body = {
                    "input": [
                        {
                            "role": "user",
                            "content": (
                                "Use src/stable.py while ignoring prose such as emotional/contextual "
                                "and preferences/facts."
                            ),
                        }
                    ]
                }
                missing_body = {"input": [{"role": "user", "content": "Use src/deleted.py"}]}
                stable_audit = cache_module.cache_file_dependency_audit(stable_body)
                missing_audit = cache_module.cache_file_dependency_audit(missing_body)
            finally:
                os.chdir(old_cwd)

        self.assertTrue(stable_audit["safe_invalidation_evidence"])
        self.assertTrue(stable_audit["file_dependency_evidence_available"])
        self.assertIsNone(stable_audit["invalidation_reason"])
        self.assertEqual(stable_audit["snapshot_count"], 1)
        self.assertEqual(stable_audit["raw_candidate_path_count_bucket"], "2_5")
        self.assertFalse(missing_audit["safe_invalidation_evidence"])
        self.assertEqual(missing_audit["invalidation_reason"], "dependency-missing")

        self._log_openai_call(
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            request_fingerprint="raw-stable-openai-dependency-fingerprint",
            file_dependency_audit=stable_audit,
            cache_extra={"cache_replay_blocker_reasons": ["tool-call-cache-disabled"]},
            cost=0.04,
        )
        self._log_openai_call(
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            request_fingerprint="raw-missing-openai-dependency-fingerprint",
            file_dependency_audit=missing_audit,
            cache_extra={"cache_replay_blocker_reasons": ["dependency-missing", "tool-call-cache-disabled"]},
            cost=0.05,
        )

        opportunity = build_openai_cache_replay_report(self.store, limit=20)
        readiness = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        smoke = build_cache_smoke_diagnostic(self.store, scan_limit=20)

        blocker_counts = {row["value"]: row["count"] for row in opportunity["blocker_reason_breakdown"]}
        dependency_status = {row["value"]: row["count"] for row in opportunity["file_dependency_status_breakdown"]}
        self.assertEqual(blocker_counts["file-dependency-missing"], 1)
        self.assertEqual(dependency_status["stable"], 1)
        self.assertEqual(dependency_status["missing"], 1)
        self.assertEqual(readiness["summary"]["top_blockers"][0]["value"], "tool-call-cache-disabled")
        self.assertEqual(
            {row["value"]: row["count"] for row in readiness["blocker_reason_breakdown"]}["file-dependency-missing"],
            1,
        )
        self.assertEqual(smoke["summary"]["file_dependency_blocked_count"], 1)

        rendered = json.dumps([opportunity, readiness, smoke], sort_keys=True)
        self.assertNotIn("raw-stable-openai-dependency-fingerprint", rendered)
        self.assertNotIn("raw-missing-openai-dependency-fingerprint", rendered)
        self.assertNotIn("src/stable.py", rendered)
        self.assertNotIn("src/deleted.py", rendered)
        self.assertNotIn("emotional/contextual", rendered)

    def test_openai_cache_replay_impact_quality_gates_and_lifecycle_are_metadata_only(self) -> None:
        def replay_meta(
            *,
            candidate_id: str,
            rule_id: str,
            canary_status: str,
            cohort: str,
            projected: float,
            reason: str,
            safety_stop: bool = False,
            invalidated: bool = False,
        ) -> dict[str, object]:
            rule: dict[str, object] = {
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "policy_source": "managed-recommended",
                "scope": "session",
                "canary": {
                    "enabled": True,
                    "selected": cohort == "canary_applied",
                    "cohort": cohort,
                    "fraction": 0.5,
                    "unit": "session",
                    "status": "applied",
                    "pattern_hashes": ["sha256:" + "a" * 64],
                },
            }
            if safety_stop:
                rule["safety_stop"] = {
                    "reason": "error-rate-regression",
                    "decision": "rollback",
                    "sample_count": 3,
                    "error_rate": 0.5,
                }
            return {
                "pattern_rule": rule,
                "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": rule_id,
                    "candidate_id": candidate_id,
                    "policy_source": "managed-recommended",
                    "status": canary_status,
                    "reason": reason,
                    "canary": rule["canary"],
                    "projected_input_savings_usd": projected,
                },
                "estimated_saved_cost_usd": projected,
                "invalidated": invalidated,
                "invalidation_reason": "dependency-changed" if invalidated else None,
            }

        for index, baseline in enumerate((0.03, 0.04)):
            call_id = self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=baseline,
                latency_ms=120 + index,
                cache_extra=replay_meta(
                    candidate_id="openai-cache-promote",
                    rule_id="openai-cache-promote-rule",
                    canary_status="applied",
                    cohort="canary_applied",
                    projected=baseline,
                    reason="dependency-stable",
                ),
            )
            self._capture_openai_provider_adoption(call_id, tool_id=f"call_cache_widen_{index}")
        call_id = self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            latency_ms=130,
            cache_extra=replay_meta(
                candidate_id="openai-cache-promote",
                rule_id="openai-cache-promote-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                projected=0.03,
                reason="canary_holdout",
            ),
        )
        self._capture_openai_provider_adoption(call_id, tool_id="call_cache_widen_holdout")
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="session-scope-missing",
            cost=0.02,
            cost_baseline=0.02,
            cache_extra=replay_meta(
                candidate_id="openai-cache-promote",
                rule_id="openai-cache-promote-rule",
                canary_status="bypassed",
                cohort="canary_applied",
                projected=0.02,
                reason="session-scope-missing",
            ),
        )

        for index, retry_count in enumerate((2, 2)):
            self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=0.02,
                retry_count=retry_count,
                latency_ms=160 + index,
                cache_extra=replay_meta(
                    candidate_id="openai-cache-retry-regression",
                    rule_id="openai-cache-retry-regression-rule",
                    canary_status="applied",
                    cohort="canary_applied",
                    projected=0.02,
                    reason="dependency-stable",
                ),
            )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.02,
            cost_baseline=0.02,
            retry_count=0,
            latency_ms=160,
            cache_extra=replay_meta(
                candidate_id="openai-cache-retry-regression",
                rule_id="openai-cache-retry-regression-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                projected=0.02,
                reason="canary_holdout",
            ),
        )

        for index in range(2):
            call_id = self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=0.02,
                latency_ms=170 + index,
                cache_extra=replay_meta(
                    candidate_id="openai-cache-provider-adoption-risk",
                    rule_id="openai-cache-provider-adoption-risk-rule",
                    canary_status="applied",
                    cohort="canary_applied",
                    projected=0.02,
                    reason="dependency-stable",
                ),
            )
            self._capture_openai_orphan_provider_adoption(call_id, tool_id=f"orphan_call_cache_{index}")
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.02,
            cost_baseline=0.02,
            latency_ms=170,
            cache_extra=replay_meta(
                candidate_id="openai-cache-provider-adoption-risk",
                rule_id="openai-cache-provider-adoption-risk-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                projected=0.02,
                reason="canary_holdout",
            ),
        )

        for index in range(2):
            self._log_openai_call(
                cache_status="hit",
                cache_reason="stale-risk-blockers",
                cache_hit=1,
                cost=0.0,
                cost_baseline=0.02,
                latency_ms=180 + index,
                cache_extra={
                    **replay_meta(
                        candidate_id="openai-cache-stale-dependency",
                        rule_id="openai-cache-stale-dependency-rule",
                        canary_status="applied",
                        cohort="canary_applied",
                        projected=0.02,
                        reason="stale-risk-blockers",
                    ),
                    "cache_replay_blocker_reasons": ["stale-risk-blockers"],
                },
            )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.02,
            cost_baseline=0.02,
            latency_ms=180,
            cache_extra=replay_meta(
                candidate_id="openai-cache-stale-dependency",
                rule_id="openai-cache-stale-dependency-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                projected=0.02,
                reason="canary_holdout",
            ),
        )

        raw_candidate = "raw-cache-key / request_id session secret"
        for index, status_code in enumerate((200, 500)):
            self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0 if status_code == 200 else 0.05,
                cost_baseline=0.02,
                status_code=status_code,
                retry_count=1 if status_code == 500 else 0,
                latency_ms=1000 + (index * 3000),
                cache_extra=replay_meta(
                    candidate_id=raw_candidate,
                    rule_id="raw-rule-id / cache key",
                    canary_status="applied",
                    cohort="canary_applied",
                    projected=0.02,
                    reason="dependency-stable",
                ),
            )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.02,
            cost_baseline=0.02,
            latency_ms=100,
            cache_extra=replay_meta(
                candidate_id=raw_candidate,
                rule_id="raw-rule-id / cache key",
                canary_status="holdout",
                cohort="canary_holdout",
                projected=0.02,
                reason="canary_holdout",
            ),
        )
        self._log_openai_call(
            cache_status="invalidated",
            cache_reason="dependency-changed",
            cost=0.02,
            cost_baseline=0.02,
            cache_extra=replay_meta(
                candidate_id=raw_candidate,
                rule_id="raw-rule-id / cache key",
                canary_status="invalidated",
                cohort="canary_applied",
                projected=0.02,
                reason="dependency-changed",
                invalidated=True,
            ),
        )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="local-canary-safety-stop",
            cost=0.02,
            cost_baseline=0.02,
            cache_extra=replay_meta(
                candidate_id=raw_candidate,
                rule_id="raw-rule-id / cache key",
                canary_status="bypassed",
                cohort="canary_applied",
                projected=0.02,
                reason="local-canary-safety-stop",
                safety_stop=True,
            ),
        )

        report = build_openai_cache_replay_impact_report(
            self.store,
            limit=20,
            min_applied_samples=2,
            min_holdout_samples=1,
            min_cache_hit_rate=0.01,
        )

        self.assertEqual(report["schema"], "tokenclaw.openai_cache_replay_impact.v1")
        self.assertEqual(report["quality_gate"]["schema"], "tokenclaw.openai_cache_replay_quality_gate.v1")
        self.assertEqual(report["summary"]["applied_count"], 10)
        self.assertEqual(report["summary"]["holdout_count"], 5)
        self.assertEqual(report["summary"]["blocked_count"], 1)
        self.assertEqual(report["summary"]["invalidated_count"], 1)
        self.assertEqual(report["summary"]["safety_stop_count"], 1)
        by_verdict = {row["verdict"]: row for row in report["candidates"]}
        self.assertEqual(by_verdict["widen"]["candidate_id"], "openai-cache-promote")
        self.assertEqual(by_verdict["widen"]["cohort_metrics"]["applied"]["cache_hit_rate"], 1.0)
        self.assertAlmostEqual(by_verdict["widen"]["observed_savings_usd"], 0.07)
        self.assertEqual(by_verdict["widen"]["provider_adoption_gate"]["status"], "passed")
        promotion = report["local_promotion_evidence"]
        self.assertEqual(promotion["schema"], "tokenclaw.openai_cache_replay_local_promotion_evidence.v1")
        self.assertEqual(promotion["coverage"]["applied_count"], 10)
        self.assertEqual(promotion["coverage"]["holdout_count"], 5)
        self.assertEqual(promotion["outcomes"]["observed_hits"], 10)
        self.assertAlmostEqual(promotion["savings"]["observed_saved_usd"], 0.18)
        self.assertEqual(promotion["recommended_local_action"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(promotion["recommended_local_action"]["action"], "rollback-or-disable-openai-cache-replay-rule")
        candidate_actions = {
            row["candidate_id"]: row["recommended_local_action"]
            for row in promotion["candidate_evidence"]
        }
        self.assertEqual(candidate_actions["openai-cache-promote"], "promote-openai-cache-replay-rule-draft")
        self.assertEqual(report["summary"]["local_promotion_status"], "rollback-required")
        self.assertEqual(report["summary"]["recommended_local_action"], "rollback-or-disable-openai-cache-replay-rule")
        self.assertIn("hold", by_verdict)
        self.assertTrue(
            any("stale-dependency-blocker" in row["reason_codes"] for row in report["candidates"]),
            report["candidates"],
        )
        self.assertTrue(
            any("retry-rate-regression" in row["reason_codes"] for row in report["candidates"]),
            report["candidates"],
        )
        self.assertTrue(
            any("provider-adoption-regression" in row["reason_codes"] for row in report["candidates"]),
            report["candidates"],
        )
        self.assertIn("rollback-error-rate", by_verdict["rollback"]["reason_codes"])
        self.assertIn("negative-observed-savings", by_verdict["rollback"]["reason_codes"])
        self.assertIn("safety-stop-observed", by_verdict["rollback"]["reason_codes"])
        self.assertIn("invalidation-rate-above-threshold", by_verdict["rollback"]["reason_codes"])
        self.assertTrue(by_verdict["rollback"]["candidate_id"].startswith("candidate-id:"))
        self.assertFalse(by_verdict["widen"]["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["raw_responses_included"])
        self.assertFalse(report["privacy"]["tool_payloads_included"])
        self.assertFalse(report["privacy"]["file_paths_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])

        stats_result = asyncio.run(stats_openai_cache_replay_impact(self.store, limit=20))
        self.assertEqual(stats_result["schema"], "tokenclaw.openai_cache_replay_impact.v1")

        sent: dict[str, object] = {}

        async def fake_queue(_store, payload, **_kwargs):
            sent["payload"] = payload
            return {
                "enabled": True,
                "endpoint": "/v1/policy-events",
                "status": "sent",
                "status_code": 202,
                "latency_ms": 3,
            }

        output = io.StringIO()
        with patch("tokenclaw.recommendations.recommendations_enabled", return_value=True), patch(
            "tokenclaw.recommendations.queue_policy_event_feedback",
            fake_queue,
        ):
            exit_code = cli.openai_cache_replay_impact_cli(["--db", self.db_path, "--limit", "20"], stdout=output)

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "sent")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        self.assertEqual(sent["payload"]["schema"], "tokenclaw.openai_cache_replay_lifecycle_feedback.v1")
        self.assertEqual(sent["payload"]["lifecycle_kind"], "openai_cache_replay")
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-cache-key-secret", rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)
        self.assertNotIn("secret tool payload", rendered)
        self.assertNotIn("orphan secret tool payload", rendered)
        self.assertNotIn("/tmp/private.py", rendered)
        self.assertNotIn("call_cache_widen_", rendered)
        self.assertNotIn("orphan_call_cache_", rendered)
        self.assertNotIn(raw_candidate, rendered)
        self.assertNotIn("raw-rule-id / cache key", rendered)
        self.assertNotIn("sha256:" + "a" * 64, rendered)
        queued_rendered = json.dumps(sent["payload"], sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "raw response must not leak",
            "raw-cache-key-secret",
            "raw-openai-session-must-not-leak",
            raw_candidate,
            "raw-rule-id / cache key",
            "sha256:" + "a" * 64,
        ):
            self.assertNotIn(forbidden, queued_rendered)

    def test_openai_cache_replay_impact_measures_projected_hits_against_actual_cohorts(self) -> None:
        def replay_meta(
            *,
            candidate_id: str,
            rule_id: str,
            canary_status: str,
            cohort: str,
            projected_hits: int,
            projected_savings: float,
            reason: str,
        ) -> dict[str, object]:
            graduation = {
                "schema": "tokenclaw.openai_cache_replay_shape_activation.v1",
                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                "source_reason": "replay-ready-exact-non-tool-shape",
                "projected_hits": projected_hits,
                "projected_savings_usd": projected_savings,
                "sample_count": 4,
                "aggregate_only": True,
                "raw_prompt": "raw graduation prompt must not leak",
            }
            rule: dict[str, object] = {
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "policy_source": "managed-recommended",
                "scope": "session",
                "graduation": graduation,
                "canary": {
                    "enabled": True,
                    "selected": cohort == "canary_applied",
                    "cohort": cohort,
                    "fraction": 0.5,
                    "unit": "request_fingerprint",
                    "status": canary_status,
                    "pattern_hashes": ["sha256:" + "b" * 64],
                },
            }
            return {
                "pattern_rule": rule,
                "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": rule_id,
                    "candidate_id": candidate_id,
                    "policy_source": "managed-recommended",
                    "status": canary_status,
                    "reason": reason,
                    "canary": rule["canary"],
                    "projected_hits": projected_hits,
                    "projected_savings_usd": projected_savings,
                },
            }

        self._log_openai_call(
            cache_status="hit",
            cache_reason="exact-match",
            cache_hit=1,
            cost=0.0,
            cost_baseline=0.03,
            cache_extra=replay_meta(
                candidate_id="openai-cache-projected-cohort",
                rule_id="openai-cache-projected-rule",
                canary_status="applied",
                cohort="canary_applied",
                projected_hits=3,
                projected_savings=0.09,
                reason="dependency-stable",
            ),
        )
        self._log_openai_call(
            cache_status="miss",
            cache_reason="exact-miss",
            cache_hit=0,
            cost=0.03,
            cost_baseline=0.03,
            cache_extra=replay_meta(
                candidate_id="openai-cache-projected-cohort",
                rule_id="openai-cache-projected-rule",
                canary_status="applied",
                cohort="canary_applied",
                projected_hits=3,
                projected_savings=0.09,
                reason="dependency-stable",
            ),
        )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cache_hit=0,
            cost=0.03,
            cost_baseline=0.03,
            cache_extra=replay_meta(
                candidate_id="openai-cache-projected-cohort",
                rule_id="openai-cache-projected-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                projected_hits=3,
                projected_savings=0.09,
                reason="canary_holdout",
            ),
        )
        self._log_openai_call(
            cache_status="bypassed",
            cache_reason="session-scope-missing",
            cache_hit=0,
            cost=0.02,
            cost_baseline=0.02,
            cache_extra=replay_meta(
                candidate_id="openai-cache-skipped-cohort",
                rule_id="openai-cache-skipped-rule",
                canary_status="bypassed",
                cohort="canary_applied",
                projected_hits=1,
                projected_savings=0.02,
                reason="session-scope-missing",
            ),
        )

        report = build_openai_cache_replay_impact_report(
            self.store,
            limit=20,
            min_applied_samples=1,
            min_holdout_samples=1,
            min_savings_realization_ratio=0.0,
        )

        self.assertEqual(report["summary"]["applied_count"], 2)
        self.assertEqual(report["summary"]["holdout_count"], 1)
        self.assertEqual(report["summary"]["projected_hits"], 4)
        self.assertAlmostEqual(report["summary"]["projected_saved_usd"], 0.11)
        self.assertAlmostEqual(report["summary"]["dry_run_projected_savings_usd"], 0.11)
        self.assertEqual(report["summary"]["actual_hits"], 1)
        self.assertEqual(report["summary"]["miss_count"], 1)
        self.assertEqual(report["summary"]["bypass_skipped_count"], 2)
        self.assertEqual(report["summary"]["replay_ready_cohort_count"], 1)
        self.assertEqual(report["summary"]["skipped_cohort_count"], 1)
        self.assertAlmostEqual(report["summary"]["actual_saved_cost_usd"], 0.03)
        aggregate_measurement = report["summary"]["canary_hit_measurement"]
        self.assertEqual(aggregate_measurement["schema"], "tokenclaw.openai_cache_replay_canary_hit_measurement.v1")
        self.assertEqual(report["summary"]["first_real_hit_status"], "observed-hit")
        self.assertTrue(report["summary"]["first_real_hit_observed"])
        self.assertEqual(report["summary"]["first_real_hit_candidate_count"], 1)
        self.assertEqual(aggregate_measurement["first_real_hit_status"], "observed-hit")
        self.assertTrue(aggregate_measurement["first_real_hit_observed"])
        self.assertEqual(aggregate_measurement["first_real_hit_candidate_count"], 1)
        self.assertEqual(aggregate_measurement["applied_count"], 2)
        self.assertEqual(aggregate_measurement["holdout_count"], 1)
        self.assertEqual(aggregate_measurement["projected_hits"], 4)
        self.assertAlmostEqual(aggregate_measurement["projected_saved_usd"], 0.11)
        self.assertEqual(aggregate_measurement["observed_hits"], 1)
        self.assertAlmostEqual(aggregate_measurement["observed_saved_usd"], 0.03)
        self.assertAlmostEqual(aggregate_measurement["hit_realization_rate"], 0.25)
        self.assertAlmostEqual(aggregate_measurement["savings_realization_rate"], 0.272727)
        self.assertEqual(aggregate_measurement["holdout_forwarded_count"], 1)

        by_id = {row["candidate_id"]: row for row in report["candidates"]}
        replay_ready = by_id["openai-cache-projected-cohort"]
        self.assertEqual(replay_ready["readiness"], "replay-ready")
        self.assertEqual(replay_ready["replay_source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(replay_ready["projected_hits"], 3)
        self.assertAlmostEqual(replay_ready["projected_saved_usd"], 0.09)
        self.assertAlmostEqual(replay_ready["dry_run_projected_savings_usd"], 0.09)
        self.assertEqual(replay_ready["first_real_hit_status"], "observed-hit")
        self.assertTrue(replay_ready["first_real_hit_observed"])
        self.assertEqual(replay_ready["actual_hits"], 1)
        self.assertEqual(replay_ready["miss_count"], 1)
        self.assertEqual(replay_ready["bypass_skipped_count"], 1)
        self.assertAlmostEqual(replay_ready["actual_saved_cost_usd"], 0.03)
        measurement = replay_ready["canary_hit_measurement"]
        self.assertEqual(measurement["first_real_hit_status"], "observed-hit")
        self.assertTrue(measurement["first_real_hit_observed"])
        self.assertEqual(measurement["applied_count"], 2)
        self.assertEqual(measurement["holdout_count"], 1)
        self.assertEqual(measurement["applied_hit_count"], 1)
        self.assertEqual(measurement["applied_miss_count"], 1)
        self.assertEqual(measurement["holdout_cache_hit_count"], 0)
        self.assertEqual(measurement["holdout_forwarded_count"], 1)
        self.assertAlmostEqual(measurement["hit_realization_rate"], 0.333333)
        self.assertAlmostEqual(measurement["savings_realization_rate"], 0.333333)
        self.assertFalse(measurement["privacy"]["raw_request_bodies_included"])
        blockers = {row["value"]: row["count"] for row in replay_ready["remaining_blocker_breakdown"]}
        self.assertEqual(blockers["canary-holdout"], 1)
        self.assertEqual(blockers["exact-miss"], 1)

        skipped = by_id["openai-cache-skipped-cohort"]
        self.assertEqual(skipped["readiness"], "skipped")
        self.assertEqual(skipped["top_remaining_blocker"], "session-scope-missing")

        lifecycle = report["quality_gate"]["candidate_results"]
        lifecycle_ready = next(row for row in lifecycle if row["candidate_id"] == "openai-cache-projected-cohort")
        self.assertEqual(lifecycle_ready["projected_hits"], 3)
        self.assertAlmostEqual(lifecycle_ready["projected_saved_usd"], 0.09)
        self.assertEqual(lifecycle_ready["actual_hits"], 1)
        self.assertEqual(lifecycle_ready["miss_count"], 1)
        self.assertEqual(lifecycle_ready["first_real_hit_status"], "observed-hit")
        self.assertTrue(lifecycle_ready["first_real_hit_observed"])
        self.assertEqual(lifecycle_ready["canary_hit_measurement"]["observed_hits"], 1)

        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw graduation prompt must not leak", rendered)
        self.assertNotIn("sha256:" + "b" * 64, rendered)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])

    def test_openai_cache_replay_readiness_dashboard_api_cli_privacy_fixtures(self) -> None:
        def replay_meta(
            *,
            candidate_id: str,
            rule_id: str | None,
            canary_status: str,
            cohort: str,
            projected: float,
            reason: str,
            invalidated: bool = False,
        ) -> dict[str, object]:
            rule: dict[str, object] = {
                "candidate_id": candidate_id,
                "policy_source": "managed-recommended",
                "scope": "session",
                "canary": {
                    "enabled": True,
                    "selected": cohort == "canary_applied",
                    "cohort": cohort,
                    "fraction": 0.5,
                    "unit": "session",
                    "status": canary_status,
                    "pattern_hashes": ["sha256:" + "c" * 64],
                },
                "cache_key": "raw-cache-key-secret",
                "raw_request_id": "req-secret-must-not-leak",
                "session_id": "session-secret-must-not-leak",
            }
            if rule_id is not None:
                rule["rule_id"] = rule_id
            return {
                "pattern_rule": rule,
                "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": rule_id,
                    "candidate_id": candidate_id,
                    "policy_source": "managed-recommended",
                    "status": canary_status,
                    "reason": reason,
                    "canary": rule["canary"],
                    "projected_input_savings_usd": projected,
                    "raw_tool_payload": {"path": "/tmp/openai-secret.py", "args": "tool payload must not leak"},
                },
                "estimated_saved_cost_usd": projected,
                "invalidated": invalidated,
                "invalidation_reason": "dependency-changed" if invalidated else None,
                "cached_response_shape": "malformed-provider-payload",
                "cached_response_preview": "raw response must not leak",
                "endpoint_shape_mismatch": True,
                "raw_cache_metadata": "raw-cache-key-secret req-secret-must-not-leak",
            }

        for index, baseline in enumerate((0.03, 0.04)):
            call_id = self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=baseline,
                cache_extra=replay_meta(
                    candidate_id="openai-cache-readiness-candidate",
                    rule_id="openai-cache-readiness-rule",
                    canary_status="applied",
                    cohort="canary_applied",
                    projected=baseline,
                    reason="dependency-stable",
                ),
            )
            self._capture_openai_provider_adoption(call_id, tool_id=f"call_cache_apply_{index}")
        call_id = self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            cache_extra=replay_meta(
                candidate_id="openai-cache-readiness-candidate",
                rule_id="openai-cache-readiness-rule",
                canary_status="holdout",
                cohort="canary_holdout",
                projected=0.03,
                reason="canary_holdout",
            ),
        )
        self._log_openai_call(
            endpoint="chat_completions",
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            file_dependency_audit={
                **self._audit(reason="dependency-changed", safe=False),
                "paths": ["/tmp/openai-secret.py"],
                "root_path": "/tmp",
            },
            cost=0.05,
            cost_baseline=0.05,
            cache_extra=replay_meta(
                candidate_id="raw-cache-key / request_id session secret",
                rule_id=None,
                canary_status="invalidated",
                cohort="canary_applied",
                projected=0.05,
                reason="dependency-changed",
                invalidated=True,
            ),
        )
        self._log_openai_call(
            endpoint="responses",
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            file_dependency_audit={
                **self._audit(safe=False),
                "paths": ["/tmp/openai-secret.py"],
                "root_path": "/tmp",
            },
            cost=0.02,
            cost_baseline=0.02,
            cache_extra=replay_meta(
                candidate_id="openai-cache-missing-invalidation",
                rule_id="openai-cache-missing-invalidation-rule",
                canary_status="blocked",
                cohort="blocked",
                projected=0.02,
                reason="invalidation-evidence-missing",
            ),
        )
        self._log_openai_call(
            endpoint="responses",
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="unsafe-tool-calls-without-invalidation",
            file_dependency_audit={
                **self._audit(safe=False),
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": False,
                "paths": ["/tmp/openai-secret-unsafe.py"],
                "root_path": "/tmp",
            },
            cost=0.025,
            cost_baseline=0.025,
        )
        self._log_openai_call(
            endpoint="responses",
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            file_dependency_audit={
                **self._audit(safe=False),
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": False,
                "invalidation_reason": None,
                "paths": ["/tmp/openai-secret-unknown.py"],
                "root_path": "/tmp",
            },
            cache_extra={"file_dependency_evidence_available": True},
            cost=0.026,
            cost_baseline=0.026,
        )

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        self.assertEqual(report["schema"], "tokenclaw.openai_cache_replay_readiness.v1")
        self.assertEqual(report["state"], "saving")
        self.assertEqual(report["summary"]["applied_count"], 2)
        self.assertEqual(report["summary"]["holdout_count"], 2)
        self.assertEqual(report["summary"]["invalidated_count"], 1)
        self.assertGreater(report["summary"]["observed_savings_usd"], 0)
        self.assertTrue(report["candidates"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["tool_payloads_included"])
        self.assertFalse(report["privacy"]["file_paths_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])

        stats_result = asyncio.run(stats_openai_cache_replay_readiness(self.store, opportunity_limit=20, impact_limit=20))
        self.assertEqual(stats_result["schema"], "tokenclaw.openai_cache_replay_readiness.v1")
        burndown = asyncio.run(
            stats_openai_tool_cache_invalidation_burndown(
                self.store,
                opportunity_limit=20,
                impact_limit=20,
                row_limit=10,
            )
        )
        self.assertEqual(burndown["schema"], "tokenclaw.openai_tool_cache_invalidation_burndown.v1")
        self.assertGreaterEqual(burndown["summary"]["missing_dependency_evidence_count"], 1)
        self.assertGreaterEqual(burndown["summary"]["safe_dependency_evidence_count"], 1)
        self.assertGreaterEqual(burndown["summary"]["stale_dependency_count"], 1)
        self.assertGreaterEqual(burndown["summary"]["unsafe_dependency_count"], 1)
        self.assertGreaterEqual(burndown["summary"]["unknown_dependency_count"], 1)
        self.assertEqual(burndown["summary"]["applied_count"], 2)
        self.assertEqual(burndown["summary"]["holdout_count"], 2)
        self.assertEqual(burndown["summary"]["exact_hit_count"], 2)
        self.assertFalse(burndown["summary"]["tool_cache_replay_enabled"])
        self.assertFalse(burndown["summary"]["streaming_replay_enabled"])
        self.assertEqual(burndown["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(burndown["summary"]["cache_entries_written"], 0)
        self.assertFalse(burndown["summary"]["policy_files_written"])
        self.assertIn("unknown-dependency-evidence", burndown["summary"]["dependency_evidence_classes"])
        self.assertFalse(burndown["privacy"]["raw_request_bodies_included"])
        self.assertFalse(burndown["privacy"]["file_paths_included"])
        self.assertFalse(burndown["privacy"]["cache_keys_included"])
        self.assertFalse(burndown["privacy"]["request_ids_included"])
        self.assertFalse(burndown["privacy"]["session_ids_included"])
        self.assertTrue(burndown["blockers"])
        self.assertTrue(
            any(
                row["outcome"] == "unsafe-dependency"
                for row in burndown["outcome_breakdown"]
            )
        )

        app = create_dashboard_app(
            store_obj=lambda: self.store,
            default_db=self.db_path,
            upstream="https://openai.test",
            limiter_status=lambda: [],
            limiter_config={
                "min_request_interval_ms": 0,
                "max_tier_backoff_wait_s": 30,
                "max_concurrent_per_tier": 2,
            },
        )
        with TestClient(app) as client:
            api_response = client.get("/tokenclaw/stats/openai-cache-replay-readiness?opportunity_limit=20&impact_limit=20")
            burndown_response = client.get(
                "/tokenclaw/stats/openai-tool-cache-invalidation-burndown?opportunity_limit=20&impact_limit=20&row_limit=10"
            )
            dashboard = client.get("/tokenclaw/dashboard")
        self.assertEqual(api_response.status_code, 200)
        self.assertEqual(api_response.json()["state"], "saving")
        self.assertEqual(burndown_response.status_code, 200)
        self.assertEqual(
            burndown_response.json()["schema"],
            "tokenclaw.openai_tool_cache_invalidation_burndown.v1",
        )
        self.assertGreaterEqual(burndown_response.json()["summary"]["missing_dependency_evidence_count"], 1)
        self.assertGreaterEqual(burndown_response.json()["summary"]["unsafe_dependency_count"], 1)
        self.assertGreaterEqual(burndown_response.json()["summary"]["unknown_dependency_count"], 1)
        self.assertEqual(burndown_response.json()["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("/tokenclaw/stats/openai-cache-replay-readiness", dashboard.text)
        self.assertIn("/tokenclaw/stats/openai-tool-cache-invalidation-burndown", dashboard.text)
        self.assertIn("OpenAI cache replay readiness", dashboard.text)
        self.assertIn("openai-cache-replay-readiness-tbody", dashboard.text)
        self.assertIn("OpenAI tool-cache invalidation burndown", dashboard.text)
        self.assertIn("openai-tool-cache-invalidation-burndown-tbody", dashboard.text)
        self.assertIn("openai-tool-cache-invalidation-blockers-tbody", dashboard.text)
        self.assertIn("OpenAI cache replay impact gates", dashboard.text)
        self.assertIn("openai-cache-replay-impact-gates-tbody", dashboard.text)

        output = io.StringIO()
        exit_code = cli.openai_cache_replay_readiness_cli(
            ["--db", self.db_path, "--opportunity-limit", "20", "--impact-limit", "20"],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["schema"], "tokenclaw.openai_cache_replay_readiness.v1")

        rendered_outputs = [
            json.dumps(report, sort_keys=True),
            json.dumps(api_response.json(), sort_keys=True),
            json.dumps(burndown, sort_keys=True),
            json.dumps(burndown_response.json(), sort_keys=True),
            output.getvalue(),
            dashboard.text,
        ]
        for rendered in rendered_outputs:
            for forbidden in (
                "raw prompt must not leak",
                "raw response must not leak",
                "raw-cache-key-secret",
                "raw-openai-session-must-not-leak",
                "req-secret-must-not-leak",
                "session-secret-must-not-leak",
                "/tmp/openai-secret.py",
                "/tmp/openai-secret-unknown.py",
                "tool payload must not leak",
                "raw-cache-key / request_id session secret",
                "sha256:" + "c" * 64,
            ):
                self.assertNotIn(forbidden, rendered)

    def test_openai_cache_replay_readiness_explains_missing_staged_canary_policy(self) -> None:
        self._log_openai_call(request_fingerprint="raw-readiness-request-fingerprint")
        self._log_openai_call(request_fingerprint="raw-readiness-request-fingerprint")

        with patch.dict(os.environ, {"TOKENCLAW_CACHE_CANARY_POLICY": str(Path(self.tmpdir.name) / "missing.yaml")}):
            report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)

        self.assertEqual(report["state"], "blocked")
        self.assertEqual(report["state_reason"], "staged-policy-missing")
        diagnostics = report["lifecycle_diagnostics"]["staged_canary_policy"]
        self.assertEqual(diagnostics["status"], "staged-policy-missing")
        self.assertIn("staged-canary-policy-missing", diagnostics["blockers"])
        self.assertEqual(report["summary"]["staged_canary_policy_status"], "staged-policy-missing")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw-readiness-request-fingerprint", encoded)
        self.assertFalse(diagnostics["privacy"]["request_fingerprints_included"])
        self.assertFalse(diagnostics["privacy"]["raw_request_bodies_included"])
        self.assertFalse(diagnostics["provider_calls_made"])

    def test_openai_cache_replay_blocker_outcomes_aggregate_dependency_checks_and_noops(self) -> None:
        def replay_meta(
            *,
            candidate_id: str,
            rule_id: str,
            canary_status: str,
            cohort: str,
            reason: str,
            projected: float = 0.02,
        ) -> dict[str, object]:
            rule: dict[str, object] = {
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "policy_source": "managed-recommended",
                "scope": "session",
                "canary": {
                    "enabled": True,
                    "selected": cohort == "canary_applied",
                    "cohort": cohort,
                    "fraction": 0.5,
                    "unit": "session",
                    "status": "applied",
                    "pattern_hashes": ["sha256:" + "d" * 64],
                },
            }
            return {
                "pattern_rule": rule,
                "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": rule_id,
                    "candidate_id": candidate_id,
                    "policy_source": "managed-recommended",
                    "status": canary_status,
                    "reason": reason,
                    "canary": rule["canary"],
                    "projected_input_savings_usd": projected,
                },
                "estimated_saved_cost_usd": projected,
                "cache_replay_blocker_reasons": [reason] if reason == "stale-risk-blockers" else [],
            }

        self._log_openai_call(request_fingerprint="raw-ready-cache-replay-fingerprint", cost=0.01)
        self._log_openai_call(request_fingerprint="raw-ready-cache-replay-fingerprint", cost=0.03)
        self._log_openai_call(
            endpoint="chat_completions",
            category="tool-light",
            cache_status="skipped",
            cache_reason="tools-disabled",
            has_tools=True,
            file_dependency_audit={
                **self._audit(reason="file-dependency-missing", safe=False),
                "paths": ["/tmp/private-openai-file.py"],
                "root_path": "/tmp",
            },
            cost=0.04,
        )
        self._log_openai_call(
            endpoint="responses",
            category="tool-light",
            cache_status="skipped",
            cache_reason="tools-disabled",
            has_tools=True,
            request_fingerprint="raw-safe-tool-cache-replay-fingerprint",
            file_dependency_audit=self._audit(safe=True),
            cost=0.06,
        )
        self._log_openai_call(
            endpoint="responses",
            category="tool-light",
            cache_status="skipped",
            cache_reason="unsafe-tool-calls-without-invalidation",
            has_tools=True,
            file_dependency_audit={
                **self._audit(safe=False),
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": False,
                "paths": ["/tmp/private-openai-unsafe-file.py"],
                "root_path": "/tmp",
            },
            cost=0.07,
        )
        self._log_openai_call(
            cache_status="skipped",
            cache_reason="streaming",
            stream=1,
            cost=0.05,
        )
        self._log_openai_call(
            cache_status="hit",
            cache_reason="stale-risk-blockers",
            cache_hit=1,
            cost=0.0,
            cost_baseline=0.02,
            cache_extra=replay_meta(
                candidate_id="raw-cache-key / request_id session secret",
                rule_id="raw-rule-id / cache key",
                canary_status="applied",
                cohort="canary_applied",
                reason="stale-risk-blockers",
            ),
        )

        with patch.dict(os.environ, {"TOKENCLAW_CACHE_CANARY_POLICY": str(Path(self.tmpdir.name) / "missing.yaml")}):
            report = build_openai_cache_replay_blocker_outcomes_report(
                self.store,
                opportunity_limit=20,
                impact_limit=20,
            )

        self.assertEqual(report["schema"], "tokenclaw.openai_cache_replay_blocker_outcomes.v1")
        self.assertEqual(report["top_next_action"], "stage-local-cache-replay-canary")
        self.assertGreaterEqual(report["summary"]["replay_ready_count"], 2)
        self.assertGreaterEqual(report["summary"]["stale_dependency_count"], 1)
        self.assertGreaterEqual(report["summary"]["unsafe_dependency_count"], 1)
        self.assertGreaterEqual(report["summary"]["missing_invalidation_count"], 1)
        self.assertGreaterEqual(report["summary"]["noop_count"], 1)
        self.assertGreaterEqual(report["summary"]["ranked_cohort_count"], 4)
        outcomes = {row["outcome"]: row["count"] for row in report["outcome_breakdown"]}
        self.assertIn("replay-ready", outcomes)
        self.assertIn("stale-dependency", outcomes)
        self.assertIn("unsafe-dependency", outcomes)
        self.assertIn("missing-invalidation", outcomes)
        self.assertIn("noop", outcomes)
        self.assertTrue(report["acceptance"]["emits_ranked_replay_ready_stale_and_missing_cohorts"])
        self.assertTrue(report["acceptance"]["emits_ranked_dependency_evidence_classes"])
        cohorts = {(row["outcome"], row["reason"]): row for row in report["cohorts"]}
        safe = cohorts[("replay-ready", "safe-invalidation-evidence-present")]
        stale = next(row for row in report["cohorts"] if row["outcome"] == "stale-dependency")
        unsafe = next(row for row in report["cohorts"] if row["outcome"] == "unsafe-dependency")
        missing = next(row for row in report["cohorts"] if row["outcome"] == "missing-invalidation")
        self.assertEqual(safe["next_action"], "stage-local-cache-replay-canary")
        self.assertTrue(safe["safe_invalidation_evidence"])
        self.assertFalse(safe["tool_cache_replay_enabled"])
        self.assertFalse(safe["policy_files_written"])
        self.assertEqual(stale["next_action"], "refresh-cache-replay-dependency-evidence")
        self.assertEqual(unsafe["next_action"], "collect-safe-invalidation-evidence")
        self.assertFalse(unsafe["tool_cache_replay_enabled"])
        self.assertFalse(unsafe["streaming_replay_enabled"])
        self.assertFalse(unsafe["emits_cache_apply_action"])
        self.assertEqual(missing["next_action"], "collect-safe-invalidation-evidence")
        reasons = {row["value"]: row["count"] for row in report["reason_breakdown"]}
        self.assertIn("stale-risk-blockers", reasons)
        self.assertIn("unsafe-tool-calls-without-invalidation", reasons)
        self.assertIn("invalidation-evidence-missing", reasons)
        self.assertIn("unsupported-streaming-shape", reasons)
        self.assertFalse(report["source_reports"]["individual_candidate_ids_included"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["absolute_paths_included"])
        self.assertFalse(report["privacy"]["individual_candidate_ids_included"])

        output = io.StringIO()
        exit_code = cli.openai_cache_replay_blocker_outcomes_cli(
            ["--db", self.db_path, "--opportunity-limit", "20", "--impact-limit", "20"],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads(output.getvalue())["schema"],
            "tokenclaw.openai_cache_replay_blocker_outcomes.v1",
        )

        rendered = json.dumps(report, sort_keys=True) + output.getvalue()
        for forbidden in (
            "raw prompt must not leak",
            "raw response must not leak",
            "raw-cache-key-secret",
            "raw-openai-session-must-not-leak",
            "raw-ready-cache-replay-fingerprint",
            "raw-safe-tool-cache-replay-fingerprint",
            "raw-cache-key / request_id session secret",
            "raw-rule-id / cache key",
            "/tmp/private-openai-file.py",
            "sha256:" + "d" * 64,
        ):
            self.assertNotIn(forbidden, rendered)

    def test_openai_cache_replay_readiness_diagnoses_staged_policy_can_run(self) -> None:
        pattern_hash = "sha256:" + "d" * 64
        policy = {
            "schema": "tokenclaw.openai_cache_replay_canary_policy.v1",
            "policy_source": "managed-recommended",
            "pattern_rules": [
                {
                    "id": "openai-cache-readiness-staged-rule",
                    "candidate_id": "openai-cache-readiness-staged-candidate",
                    "conditions": {
                        "pattern_hashes": [pattern_hash],
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "has_tools": False,
                        "stream": False,
                        "replayability_levels": ["local-exact-response"],
                    },
                    "action": {
                        "type": "exact_cache_pattern",
                        "scope": "session",
                    },
                    "rollout": {
                        "canary_enabled": True,
                        "canary_fraction": 1.0,
                        "canary_salt": "openai-readiness-staged-test",
                        "canary_unit": "request_fingerprint",
                    },
                }
            ],
        }
        policy_path = Path(self.tmpdir.name) / "cache_canary_policy.yaml"
        policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
        for index, cost in enumerate((0.01, 0.03)):
            self._log_openai_call(
                request_fingerprint="raw-staged-readiness-request-fingerprint",
                pattern_hashes=[pattern_hash],
                cost=cost,
                created_at=f"2026-06-11T08:0{index}:00+00:00",
            )

        from tokenclaw import cache as cache_module

        try:
            with patch.dict(os.environ, {"TOKENCLAW_CACHE_CANARY_POLICY": str(policy_path)}):
                _reload_cache_module_for_test()
                report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        finally:
            _reload_cache_module_for_test()

        diagnostics = report["lifecycle_diagnostics"]["staged_canary_policy"]
        self.assertEqual(diagnostics["status"], "staged-policy-can-run")
        self.assertTrue(diagnostics["runtime_loaded"])
        self.assertIsNone(diagnostics["configured_policy_path"])
        self.assertIsNone(diagnostics["runtime_loaded_policy_path"])
        self.assertTrue(diagnostics["configured_policy_path_state"]["configured"])
        self.assertFalse(diagnostics["configured_policy_path_state"]["path_included"])
        self.assertTrue(diagnostics["runtime_loaded_policy_path_state"]["configured"])
        self.assertFalse(diagnostics["runtime_loaded_policy_path_state"]["path_included"])
        self.assertEqual(diagnostics["policy_rule_count"], 1)
        self.assertEqual(diagnostics["dry_run_summary"]["projected_applied_rows"], 2)
        self.assertEqual(diagnostics["dry_run_summary"]["projected_hits"], 1)
        self.assertFalse(diagnostics["dry_run_summary"]["cache_table_mutated"])
        self.assertFalse(diagnostics["provider_calls_made"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(str(policy_path), encoded)
        self.assertNotIn(pattern_hash, encoded)
        self.assertNotIn("raw-staged-readiness-request-fingerprint", encoded)
        self.assertFalse(diagnostics["privacy"]["pattern_hashes_included"])
        self.assertFalse(diagnostics["privacy"]["request_fingerprints_included"])

    def test_openai_cache_replay_apply_writes_local_canary_overlay(self) -> None:
        pattern_hash = "sha256:" + "e" * 64
        candidate_id = "openai-cache-apply-candidate"
        rule_id = "openai-cache-apply-rule"

        def replay_meta(*, cohort: str, status: str, projected: float, reason: str) -> dict[str, object]:
            rule: dict[str, object] = {
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "policy_source": "managed-recommended",
                "matched_hashes": [pattern_hash],
                "allow_tool_calls": False,
                "safe_invalidation": True,
                "scope": "session",
                "canary": {
                    "enabled": True,
                    "selected": cohort == "canary_applied",
                    "cohort": cohort,
                    "fraction": 0.5,
                    "unit": "request_fingerprint",
                    "status": status,
                    "pattern_hashes": [pattern_hash],
                },
            }
            return {
                "pattern_rule": rule,
                "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": rule_id,
                    "candidate_id": candidate_id,
                    "policy_source": "managed-recommended",
                    "status": status,
                    "reason": reason,
                    "canary": rule["canary"],
                    "projected_input_savings_usd": projected,
                    "raw_request_id": "req-apply-secret",
                },
                "estimated_saved_cost_usd": projected,
            }

        for index, baseline in enumerate((0.03, 0.04)):
            call_id = self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=baseline,
                request_fingerprint="raw-apply-request-fingerprint",
                pattern_hashes=[pattern_hash],
                cache_extra=replay_meta(
                    cohort="canary_applied",
                    status="applied",
                    projected=baseline,
                    reason="dependency-stable",
                ),
            )
            self._capture_openai_provider_adoption(call_id, tool_id=f"call_cache_apply_{index}")
        call_id = self._log_openai_call(
            cache_status="bypassed",
            cache_reason="canary_holdout",
            cost=0.03,
            cost_baseline=0.03,
            request_fingerprint="raw-apply-request-fingerprint",
            pattern_hashes=[pattern_hash],
            cache_extra=replay_meta(
                cohort="canary_holdout",
                status="holdout",
                projected=0.03,
                reason="canary_holdout",
            ),
        )
        self._capture_openai_provider_adoption(call_id, tool_id="call_cache_apply_holdout")

        plan = build_openai_cache_replay_apply_plan(
            self.store,
            opportunity_limit=20,
            impact_limit=20,
            min_observed_savings_usd=0.001,
            holdout_fraction=0.2,
        )
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["summary"]["accepted_candidate_count"], 1)
        self.assertEqual(plan["accepted_candidates"][0]["candidate_id"], candidate_id)
        self.assertEqual(plan["accepted_candidates"][0]["canary_fraction"], 0.8)
        self.assertFalse(plan["accepted_candidates"][0]["pattern_hashes_included"])

        output = io.StringIO()
        config_dir = Path(self.tmpdir.name) / "config"
        code = cli.openai_cache_replay_apply_cli(
            [
                "--db",
                self.db_path,
                "--config-dir",
                str(config_dir),
                "--impact-limit",
                "20",
                "--opportunity-limit",
                "20",
                "--min-observed-savings-usd",
                "0.001",
                "--holdout-fraction",
                "0.2",
            ],
            stdout=output,
        )
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.openai_cache_replay_apply.v1")
        self.assertTrue(payload["wrote_policy_files"])
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["pattern_hashes_included"])
        rendered = output.getvalue()
        self.assertNotIn(pattern_hash, rendered)
        self.assertNotIn("raw-apply-request-fingerprint", rendered)
        self.assertNotIn("req-apply-secret", rendered)

        policy_path = config_dir / "cache_canary_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(policy["schema"], "tokenclaw.openai_cache_replay_canary_policy.v1")
        self.assertEqual(policy["pattern_rules"][0]["id"], rule_id)
        self.assertEqual(policy["pattern_rules"][0]["candidate_id"], candidate_id)
        self.assertEqual(policy["pattern_rules"][0]["conditions"]["pattern_hashes"], [pattern_hash])
        self.assertEqual(policy["pattern_rules"][0]["rollout"]["canary_fraction"], 0.8)
        self.assertEqual(policy["pattern_rules"][0]["action"]["scope"], "session")

        from tokenclaw import cache as cache_module

        try:
            with patch.dict(os.environ, {"TOKENCLAW_CACHE_CANARY_POLICY": str(policy_path)}):
                reloaded = _reload_cache_module_for_test()
                loaded = [rule for rule in reloaded.CACHE_PATTERN_RULES if rule.get("candidate_id") == candidate_id]
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["id"], rule_id)
                self.assertEqual(loaded[0]["conditions"]["pattern_hashes"], [pattern_hash])
                self.assertEqual(loaded[0]["policy_source"], "managed-recommended")
                self.assertEqual(str(reloaded.CACHE_CANARY_RULES_PATH), str(policy_path))
        finally:
            _reload_cache_module_for_test()

    def test_openai_cache_replay_apply_rolls_back_stale_policy_decision_to_cache_rules(self) -> None:
        config_dir = Path(self.tmpdir.name) / "rollback-config"
        config_dir.mkdir()
        canary_path = config_dir / "cache_canary_policy.yaml"
        shape = {
            "provider_family": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "category": "chat",
            "workflow_phase": "chat",
            "stream": False,
            "has_tools": False,
            "text_bucket": "2k_8k_chars",
            "token_bucket": "500_2k_tokens",
        }
        canary_path.write_text(
            yaml.safe_dump(
                {
                    "schema": "tokenclaw.openai_cache_replay_canary_policy.v1",
                    "policy_source": "local-manual",
                    "pattern_rules": [
                        {
                            "id": "local-openai-cache-replay-canary-stale",
                            "enabled": True,
                            "policy_source": "local-manual",
                            "target_cache_policy": {
                                "schema": "tokenclaw.request_shape_cache_replay_target_policy.v1",
                                "policy_section": "cache.pattern_rules",
                                "target_local_policy": "cache_rules",
                                "target_local_rule_file": "cache_rules.yaml",
                                "policy_source": "local-manual",
                                "local_file_backed": True,
                                "managed_dependency": "optional",
                                "rules_path_included": False,
                                "metadata_only": True,
                                "aggregate_only": True,
                            },
                            "conditions": {
                                "pattern_hashes": ["sha256:*"],
                                **shape,
                                "replayability_levels": ["features_only", "local-exact-response"],
                            },
                            "action": {
                                "type": "exact_cache_pattern",
                                "allow_tool_calls": False,
                                "safe_invalidation": False,
                                "streaming": False,
                                "scope": "session",
                                "ttl_seconds": 3600,
                            },
                            "rollout": {
                                "schema": "tokenclaw.pattern_policy_rollout.v1",
                                "recommendation_mode": "openai-cache-replay-request-shape-canary",
                                "canary_enabled": True,
                                "canary_fraction": 0.1,
                                "holdout_fraction": 0.1,
                                "canary_salt": "local-openai-cache-replay-canary-stale",
                                "canary_unit": "request_fingerprint",
                            },
                            "graduation": {
                                "schema": "tokenclaw.request_shape_cache_replay_shape_activation.v1",
                                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                                "source_reason": "replay-ready-exact-non-tool-shape",
                                **shape,
                                "projected_hits": 35,
                                "projected_savings_usd": 0.075373,
                                "sample_count": 36,
                                "aggregate_only": True,
                                "staged_at": "2026-06-15T00:00:00+00:00",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=canary_path,
            limit=20,
        )
        decision = build_request_shape_cache_replay_policy_decision_report(evidence)
        self.assertEqual(decision["decision"], "rollback")
        patch = decision["top_decision"]["local_policy_patch"]
        rollback_rule_id = patch["pattern_rules"][0]["id"]
        self.assertEqual(rollback_rule_id, "local-openai-cache-replay-canary-stale")
        self.assertEqual(patch["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(patch["source_canary_policy_file"], "cache_canary_policy.yaml")
        self.assertEqual(decision["top_decision"]["target_local_policy_section"], "cache.pattern_rules")
        self.assertEqual(decision["top_decision"]["reason"], "stale-no-canary-traffic")
        self.assertEqual(evidence["stale_zero_traffic_rule_count"], 1)
        self.assertTrue(decision["top_decision"]["duplicate_suppression"]["suppresses_generic_replay_ready_issue"])
        self.assertTrue(decision["top_decision"]["duplicate_suppression"]["suppresses_generic_cache_replay_activation_issue"])

        cache_rules_path = config_dir / "cache_rules.yaml"
        cache_rules_path.write_text(
            yaml.safe_dump(
                {
                    "exact_cache": {"enabled": True, "cache_tool_calls": False},
                    "semantic_cache": {"enabled": False, "threshold": 0.95},
                    "pattern_rules": [
                        {
                            "id": rollback_rule_id,
                            "enabled": True,
                            "policy_source": "local-manual",
                            "description": "Promoted OpenAI Responses exact-cache replay rule.",
                            "conditions": {
                                "pattern_hashes": ["sha256:*"],
                                **shape,
                                "replayability_levels": ["features_only", "local-exact-response"],
                            },
                            "action": {
                                "type": "exact_cache_pattern",
                                "allow_tool_calls": False,
                                "safe_invalidation": False,
                                "streaming": False,
                                "scope": "session",
                                "ttl_seconds": 3600,
                            },
                            "graduation": {
                                "schema": "tokenclaw.request_shape_cache_replay_policy_graduation.v1",
                                "source_schema": "tokenclaw.request_shape_cache_replay_evidence.v1",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        before = cache_rules_path.read_text(encoding="utf-8")

        dry_run_out = io.StringIO()
        dry_run_code = cli.openai_cache_replay_apply_cli(
            [
                "--db",
                self.db_path,
                "--config-dir",
                str(config_dir),
                "--opportunity-limit",
                "20",
                "--impact-limit",
                "20",
                "--dry-run",
            ],
            stdout=dry_run_out,
        )
        self.assertEqual(dry_run_code, 0)
        dry_run = json.loads(dry_run_out.getvalue())
        self.assertTrue(dry_run["dry_run"])
        self.assertFalse(dry_run["wrote_policy_files"])
        self.assertEqual(cache_rules_path.read_text(encoding="utf-8"), before)
        self.assertEqual(dry_run["summary"]["rollback_action_count"], 1)
        self.assertEqual(dry_run["summary"]["rollback_patch_count"], 1)
        rollback_action = dry_run["rollback_actions"][0]
        self.assertEqual(rollback_action["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(rollback_action["target_local_policy_section"], "cache.pattern_rules")
        self.assertEqual(rollback_action["local_policy_patch"]["pattern_rules"][0]["id"], rollback_rule_id)
        self.assertEqual(
            rollback_action["local_policy_patch"]["pattern_rules"][0]["disabled_reason"],
            "stale-no-canary-traffic",
        )
        self.assertGreater(rollback_action["local_policy_patch"]["pattern_rules"][0]["evidence_age_hours"], 72.0)
        self.assertTrue(rollback_action["duplicate_suppression"]["suppresses_generic_replay_ready_issue"])
        self.assertTrue(rollback_action["duplicate_suppression"]["suppresses_generic_cache_replay_activation_issue"])

        apply_out = io.StringIO()
        apply_code = cli.openai_cache_replay_apply_cli(
            [
                "--db",
                self.db_path,
                "--config-dir",
                str(config_dir),
                "--opportunity-limit",
                "20",
                "--impact-limit",
                "20",
            ],
            stdout=apply_out,
        )
        self.assertEqual(apply_code, 0)
        applied = json.loads(apply_out.getvalue())
        self.assertTrue(applied["wrote_policy_files"])
        self.assertEqual(applied["rollback_applied_rules"][0]["id"], rollback_rule_id)
        self.assertEqual(applied["rollback_applied_rules"][0]["disabled_reason"], "stale-no-canary-traffic")

        cache_rules = yaml.safe_load(cache_rules_path.read_text(encoding="utf-8"))
        rule = cache_rules["pattern_rules"][0]
        self.assertEqual(rule["id"], rollback_rule_id)
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["disabled_reason"], "stale-no-canary-traffic")
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertIn("graduation", rule)
        rendered = apply_out.getvalue()
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw-cache-key-secret", rendered)
        self.assertTrue(applied["privacy"]["metadata_only"])

    def test_openai_cache_replay_apply_rolls_back_stale_zero_traffic_rule_from_cache_rules(self) -> None:
        config_dir = Path(self.tmpdir.name) / "rollback-cache-rules-config"
        config_dir.mkdir()
        cache_rules_path = config_dir / "cache_rules.yaml"
        event_log = config_dir / "policy_events.jsonl"
        rule_id = "local-openai-cache-replay-canary-stale-direct"
        candidate_id = "request-shape-cache-replay:responses:chat:stale-direct"
        shape = {
            "provider_family": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "category": "chat",
            "workflow_phase": "chat",
            "stream": False,
            "has_tools": False,
            "text_bucket": "2k_8k_chars",
            "token_bucket": "500_2k_tokens",
        }
        cache_rules_path.write_text(
            yaml.safe_dump(
                {
                    "exact_cache": {"enabled": True, "cache_tool_calls": False},
                    "semantic_cache": {"enabled": False, "threshold": 0.95},
                    "pattern_rules": [
                        {
                            "id": rule_id,
                            "enabled": True,
                            "policy_source": "local-manual",
                            "candidate_id": candidate_id,
                            "target_cache_policy": {
                                "schema": "tokenclaw.request_shape_cache_replay_target_policy.v1",
                                "policy_section": "cache.pattern_rules",
                                "target_local_policy": "cache_rules",
                                "target_local_rule_file": "cache_rules.yaml",
                                "policy_source": "local-manual",
                                "local_file_backed": True,
                                "rules_path_included": False,
                                "metadata_only": True,
                                "aggregate_only": True,
                            },
                            "conditions": {
                                "pattern_hashes": ["sha256:*"],
                                **shape,
                                "replayability_levels": ["features_only", "local-exact-response"],
                            },
                            "action": {
                                "type": "exact_cache_pattern",
                                "allow_tool_calls": False,
                                "safe_invalidation": False,
                                "streaming": False,
                                "scope": "session",
                                "ttl_seconds": 3600,
                            },
                            "rollout": {
                                "schema": "tokenclaw.pattern_policy_rollout.v1",
                                "recommendation_mode": "openai-cache-replay-request-shape-canary",
                                "canary_enabled": True,
                                "canary_fraction": 0.1,
                                "holdout_fraction": 0.1,
                                "canary_salt": rule_id,
                                "canary_unit": "request_fingerprint",
                            },
                            "graduation": {
                                "schema": "tokenclaw.request_shape_cache_replay_shape_activation.v1",
                                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                                "source_reason": "replay-ready-exact-non-tool-shape",
                                **shape,
                                "projected_hits": 35,
                                "projected_savings_usd": 0.075373,
                                "sample_count": 36,
                                "aggregate_only": True,
                                "staged_at": "2026-06-15T00:00:00+00:00",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"TOKENCLAW_POLICY_EVENTS_LOG": str(event_log)}, clear=False):
            apply_out = io.StringIO()
            apply_code = cli.openai_cache_replay_apply_cli(
                [
                    "--db",
                    self.db_path,
                    "--config-dir",
                    str(config_dir),
                    "--opportunity-limit",
                    "20",
                    "--impact-limit",
                    "20",
                ],
                stdout=apply_out,
            )

            self.assertEqual(apply_code, 0)
            applied = json.loads(apply_out.getvalue())
            self.assertTrue(applied["wrote_policy_files"])
            self.assertEqual(applied["summary"]["rollback_action_count"], 1)
            self.assertEqual(applied["summary"]["rollback_patch_count"], 1)
            self.assertEqual(applied["rollback_actions"][0]["reason"], "stale-no-canary-traffic")
            patch_rule = applied["rollback_actions"][0]["local_policy_patch"]["pattern_rules"][0]
            self.assertEqual(patch_rule["id"], rule_id)
            self.assertEqual(patch_rule["disabled_reason"], "stale-no-canary-traffic")
            self.assertGreater(patch_rule["evidence_age_hours"], 72.0)

            cache_rules = yaml.safe_load(cache_rules_path.read_text(encoding="utf-8"))
            rule = cache_rules["pattern_rules"][0]
            self.assertEqual(rule["id"], rule_id)
            self.assertFalse(rule["enabled"])
            self.assertEqual(rule["disabled_reason"], "stale-no-canary-traffic")

            from tokenclaw.policy_events import recent_policy_events

            events = recent_policy_events(limit=1)["events"]
            self.assertEqual(events[0]["action"], "openai-cache-replay-apply")
            details = events[0]["details"]
            self.assertEqual(details["rollback_reasons"], ["stale-no-canary-traffic"])
            self.assertEqual(details["rollback_applied_rules"][0]["id"], rule_id)
            self.assertGreater(details["rollback_evidence_age_hours"][0], 72.0)

    def test_request_shape_cache_replay_recent_canary_feedback_blocks_stale_zero_traffic_rollback(self) -> None:
        config_dir = Path(self.tmpdir.name) / "fresh-feedback-config"
        config_dir.mkdir()
        cache_rules_path = config_dir / "cache_rules.yaml"
        rule_id = "local-openai-cache-replay-canary-fresh-hit"
        candidate_id = "request-shape-cache-replay:responses:chat:fresh-hit"
        shape = {
            "provider_family": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "category": "chat",
            "workflow_phase": "chat",
            "stream": False,
            "has_tools": False,
            "text_bucket": "2k_8k_chars",
            "token_bucket": "500_2k_tokens",
        }
        cache_rules_path.write_text(
            yaml.safe_dump(
                {
                    "pattern_rules": [
                        {
                            "id": rule_id,
                            "enabled": True,
                            "policy_source": "local-manual",
                            "candidate_id": candidate_id,
                            "target_cache_policy": {"target_local_rule_file": "cache_rules.yaml"},
                            "conditions": {
                                "pattern_hashes": ["sha256:*"],
                                **shape,
                                "replayability_levels": ["features_only", "local-exact-response"],
                            },
                            "action": {
                                "type": "exact_cache_pattern",
                                "allow_tool_calls": False,
                                "safe_invalidation": False,
                                "streaming": False,
                                "scope": "session",
                                "ttl_seconds": 3600,
                            },
                            "rollout": {
                                "schema": "tokenclaw.pattern_policy_rollout.v1",
                                "recommendation_mode": "openai-cache-replay-request-shape-canary",
                                "canary_enabled": True,
                                "canary_fraction": 0.1,
                                "holdout_fraction": 0.1,
                                "canary_salt": rule_id,
                                "canary_unit": "request_fingerprint",
                            },
                            "graduation": {
                                "schema": "tokenclaw.request_shape_cache_replay_shape_activation.v1",
                                "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                                "source_reason": "replay-ready-exact-non-tool-shape",
                                **shape,
                                "projected_hits": 35,
                                "projected_savings_usd": 0.075373,
                                "sample_count": 36,
                                "aggregate_only": True,
                                "staged_at": "2026-06-15T00:00:00+00:00",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self._log_openai_call(
            cache_status="hit",
            cache_reason="exact-hit",
            cache_hit=1,
            cost=0.0,
            cost_baseline=0.04,
            cache_extra=self._cache_replay_meta(
                candidate_id=candidate_id,
                rule_id=rule_id,
                canary_status="hit",
                cohort="canary_applied",
                reason="exact-hit",
            ),
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=cache_rules_path,
            limit=20,
        )
        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(evidence["stale_zero_traffic_rule_count"], 0)
        self.assertFalse(evidence["stale_evidence"]["stale"])
        self.assertNotEqual(decision["decision"], "rollback")
        self.assertFalse(decision["summary"]["rollback_required"])
        self.assertNotEqual(decision["next_action"], "rollback-cache-replay-rule")

    def test_openai_cache_replay_apply_stages_request_shape_canaries(self) -> None:
        pattern_hash = "sha256:" + "f" * 64
        for group in range(24):
            for index, cost in enumerate((0.01, 0.03)):
                self._log_openai_call(
                    category="chat",
                    cache_status="miss",
                    cache_reason="exact-miss",
                    request_fingerprint=f"raw-shape-fingerprint-{group}",
                    cost=cost,
                    session_id=f"raw-shape-session-{group}",
                    created_at=f"2026-06-11T09:{group:02d}:{index:02d}+00:00",
                )

        plan = build_openai_cache_replay_apply_plan(
            self.store,
            opportunity_limit=100,
            impact_limit=20,
            holdout_fraction=0.5,
            max_candidates=3,
        )

        self.assertTrue(plan["ok"])
        accepted = [
            row
            for row in plan["accepted_candidates"]
            if row.get("source_schema") == "tokenclaw.request_shape_cache_replayability_dry_run.v1"
        ]
        self.assertEqual(len(accepted), 1)
        self.assertGreaterEqual(accepted[0]["projected_hits"], 1)
        self.assertGreater(accepted[0]["projected_savings_usd"], 0)
        self.assertEqual(
            accepted[0]["cohort_bucket"],
            "openai_responses/responses/chat/chat/2k_8k_chars/500_2k_tokens",
        )
        self.assertEqual(plan["summary"]["projected_hits"], accepted[0]["projected_hits"])
        self.assertEqual(plan["summary"]["projected_savings_usd"], accepted[0]["projected_savings_usd"])
        self.assertGreater(plan["summary"]["applied_count"], 0)
        self.assertGreater(plan["summary"]["holdout_count"], 0)
        self.assertIn("skipped_count", plan["summary"])
        self.assertEqual(plan["summary"]["safety_stop_count"], 0)

        activation_summary = plan["activation_dry_run"]["summary"]
        self.assertGreater(activation_summary["projected_hits"], 0)
        self.assertGreater(activation_summary["projected_savings_usd"], 0)
        statuses = {row["value"]: row["count"] for row in plan["activation_dry_run"]["status_breakdown"]}
        self.assertGreater(statuses["projected-applied"], 0)
        self.assertGreater(statuses["holdout"], 0)
        activation_rows = plan["activation_dry_run"]["rows"]
        staged_rows = [
            row
            for row in activation_rows
            if row.get("cohort_bucket") == accepted[0]["cohort_bucket"]
        ]
        self.assertGreaterEqual(len(staged_rows), 2)
        self.assertEqual({row["status"] for row in staged_rows}, {"projected-applied", "holdout"})
        for row in staged_rows:
            self.assertEqual(row["matched_pattern_hash_count"], 0)
            self.assertFalse(row["matched_pattern_hashes_included"])
            self.assertEqual(row["projection"]["projected_hits"], accepted[0]["projected_hits"])
            self.assertEqual(row["projection"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
            self.assertFalse(row["projection"]["raw_request_bodies_included"])
            self.assertFalse(row["projection"]["cache_keys_included"])

        policy = plan["policy"]
        rule = policy["pattern_rules"][0]
        self.assertEqual(rule["conditions"]["pattern_hashes"], ["sha256:*"])
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.5)
        self.assertEqual(rule["graduation"]["cohort_bucket"], accepted[0]["cohort_bucket"])
        self.assertEqual(rule["graduation"]["projected_hits"], accepted[0]["projected_hits"])

        rendered = json.dumps(plan, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "raw response must not leak",
            "raw-cache-key-secret",
            "raw-shape-fingerprint-",
            "raw-shape-session-",
            pattern_hash,
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(plan["privacy"]["raw_request_bodies_included"])
        self.assertFalse(plan["privacy"]["request_fingerprints_included"])
        self.assertFalse(plan["privacy"]["session_ids_included"])
        self.assertFalse(plan["privacy"]["cache_keys_included"])
        self.assertFalse(plan["privacy"]["pattern_hashes_included"])
        self.assertFalse(plan["request_shape_evidence"]["cache_replayability_dry_run"]["privacy"]["individual_candidate_ids_included"])
        self.assertFalse(plan["activation_dry_run"]["privacy"]["pattern_hashes_included"])

    def test_openai_cache_replay_apply_stages_single_safe_tool_light_canary(self) -> None:
        for index, cost in enumerate((0.01, 0.03, 0.04)):
            self._log_openai_call(
                category="tool-light",
                has_tools=True,
                cache_status="miss",
                cache_reason="exact-miss",
                request_fingerprint="raw-safe-tool-cache-fingerprint",
                file_dependency_audit=self._audit(safe=True),
                cost=cost,
                created_at=f"2026-06-11T08:0{index}:00+00:00",
            )
        self._log_openai_call(
            category="tool-light",
            has_tools=True,
            cache_status="miss",
            cache_reason="exact-miss",
            request_fingerprint="raw-stale-tool-cache-fingerprint",
            file_dependency_audit=self._audit(reason="dependency-changed", safe=False),
            cost=0.05,
            created_at="2026-06-11T08:10:00+00:00",
        )
        self._log_openai_call(
            category="tool-light",
            has_tools=True,
            cache_status="skipped",
            cache_reason="tool-cache-disabled",
            request_fingerprint="raw-missing-tool-cache-fingerprint",
            file_dependency_audit=self._audit(reason="file-dependency-missing", safe=False),
            cost=0.02,
            created_at="2026-06-11T08:11:00+00:00",
        )

        plan = build_openai_cache_replay_apply_plan(
            self.store,
            opportunity_limit=20,
            impact_limit=20,
            holdout_fraction=0.25,
            max_candidates=10,
        )

        self.assertTrue(plan["ok"])
        accepted = [
            row
            for row in plan["accepted_candidates"]
            if row.get("source_schema") == "tokenclaw.openai_cache_replay_blocker_outcomes.v1"
        ]
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["allow_tool_calls"])
        self.assertTrue(accepted[0]["safe_invalidation"])
        self.assertTrue(accepted[0]["safe_invalidation_evidence"])
        self.assertEqual(accepted[0]["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(accepted[0]["target_local_policy_section"], "cache.pattern_rules")
        self.assertEqual(accepted[0]["canary_fraction"], 0.75)
        self.assertEqual(accepted[0]["holdout_fraction"], 0.25)
        self.assertEqual(
            accepted[0]["rollback_metadata"]["rollback_action_type"],
            "disable_openai_tool_cache_replay_canary",
        )

        skipped_reasons = {
            reason
            for row in plan["skipped_candidates"]
            if row.get("source_schema") == "tokenclaw.openai_cache_replay_blocker_outcomes.v1"
            for reason in row.get("reason_codes") or []
        }
        self.assertIn("stale-dependency", skipped_reasons)
        self.assertIn("missing-invalidation", skipped_reasons)
        self.assertTrue(all(
            not row.get("emits_cache_apply_action")
            for row in plan["skipped_candidates"]
            if row.get("source_schema") == "tokenclaw.openai_cache_replay_blocker_outcomes.v1"
        ))

        policy = plan["policy"]
        tool_rules = [
            rule
            for rule in policy["pattern_rules"]
            if (rule.get("graduation") or {}).get("source_schema") == "tokenclaw.openai_cache_replay_blocker_outcomes.v1"
        ]
        self.assertEqual(len(tool_rules), 1)
        rule = tool_rules[0]
        self.assertEqual(rule["target_cache_policy"]["policy_section"], "cache.pattern_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(rule["conditions"]["pattern_hashes"], ["sha256:*"])
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "tool-light")
        self.assertTrue(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertTrue(rule["action"]["allow_tool_calls"])
        self.assertTrue(rule["action"]["safe_invalidation"])
        self.assertTrue(rule["action"]["invalidation"]["safe_invalidation_evidence"])
        self.assertFalse(rule["action"]["invalidation"]["file_dependency_audit"]["paths_included"])
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.75)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.25)
        self.assertEqual(rule["rollback_metadata"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(rule["promotion"]["rollback_metadata"]["disable_patch"]["pattern_rules"][0]["id"], rule["id"])

        self.assertEqual(plan["blocker_outcome_evidence"]["schema"], "tokenclaw.openai_cache_replay_blocker_outcomes.v1")
        rendered = json.dumps(plan, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "raw response must not leak",
            "raw-cache-key-secret",
            "raw-safe-tool-cache-fingerprint",
            "raw-stale-tool-cache-fingerprint",
            "raw-missing-tool-cache-fingerprint",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(plan["privacy"]["raw_request_bodies_included"])
        self.assertFalse(plan["privacy"]["request_fingerprints_included"])
        self.assertFalse(plan["privacy"]["session_ids_included"])
        self.assertFalse(plan["privacy"]["cache_keys_included"])

        output = io.StringIO()
        exit_code = cli.openai_cache_replay_apply_cli(
            [
                "--db",
                self.db_path,
                "--config-dir",
                str(Path(self.tmpdir.name) / "config"),
                "--opportunity-limit",
                "20",
                "--impact-limit",
                "20",
                "--holdout-fraction",
                "0.25",
                "--dry-run",
            ],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.openai_cache_replay_apply.v1")
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["wrote_policy_files"])
        cli_accepted = [
            row
            for row in payload["accepted_candidates"]
            if row.get("source_schema") == "tokenclaw.openai_cache_replay_blocker_outcomes.v1"
        ]
        self.assertEqual(len(cli_accepted), 1)
        self.assertTrue(cli_accepted[0]["allow_tool_calls"])
        self.assertTrue(cli_accepted[0]["safe_invalidation"])
        self.assertEqual(cli_accepted[0]["target_local_rule_file"], "cache_rules.yaml")
        self.assertNotIn("raw-safe-tool-cache-fingerprint", output.getvalue())

    def test_openai_dry_run_projects_session_scoped_replay_and_dependency_blockers(self) -> None:
        pattern_hash = "sha256:" + "b" * 64
        policy = {
            "policies": {
                "cache": {
                    "pattern_rules": [
                        {
                            "id": "openai-session-cache-rule",
                            "candidate_id": "openai-session-cache-candidate",
                            "conditions": {
                                "pattern_hashes": [pattern_hash],
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "tool-light",
                                "has_tools": True,
                                "stream": False,
                                "replayability_levels": ["local-exact-response"],
                            },
                            "action": {
                                "type": "exact_cache_pattern",
                                "allow_tool_calls": True,
                                "safe_invalidation": True,
                                "scope": "session",
                            },
                            "rollout": {
                                "canary_enabled": True,
                                "canary_fraction": 1.0,
                                "canary_salt": "openai-dry-run-test",
                                "canary_unit": "session",
                            },
                        }
                    ]
                }
            }
        }
        for index, cost in enumerate((0.01, 0.03)):
            self._log_openai_call(
                category="tool-light",
                has_tools=True,
                request_fingerprint="raw-openai-request-fingerprint-must-not-leak",
                pattern_hashes=[pattern_hash],
                file_dependency_audit=self._audit(safe=True),
                cost=cost,
                created_at=f"2026-06-11T07:0{index}:00+00:00",
            )
        self._log_openai_call(
            category="tool-light",
            has_tools=True,
            request_fingerprint="raw-openai-request-fingerprint-must-not-leak",
            pattern_hashes=[pattern_hash],
            file_dependency_audit=self._audit(reason="dependency-changed", safe=False),
            cost=0.05,
            created_at="2026-06-11T07:02:00+00:00",
        )

        result = build_openai_cache_replay_dry_run(self.store, policy, limit=20)

        self.assertEqual(result["schema"], "tokenclaw.openai_cache_replay_dry_run.v1")
        self.assertEqual(result["summary"]["openai_rows_considered"], 3)
        self.assertEqual(result["summary"]["projected_applied_rows"], 2)
        self.assertEqual(result["summary"]["invalidation_required_rows"], 1)
        self.assertEqual(result["summary"]["projected_hits"], 1)
        self.assertAlmostEqual(result["summary"]["projected_savings_usd"], 0.02)
        self.assertFalse(result["summary"]["cache_table_mutated"])
        applied = next(row for row in result["rows"] if row["status"] == "projected-applied")
        self.assertEqual(applied["rule_id"], "openai-session-cache-rule")
        self.assertEqual(applied["candidate_id"], "openai-session-cache-candidate")
        self.assertTrue(applied["session_scoped_key_available"])
        self.assertTrue(applied["session_scoped_key_fingerprint"].startswith("sha256:"))
        self.assertEqual(applied["projected_hits"], 1)
        self.assertFalse(applied["matched_pattern_hashes_included"])
        self.assertEqual(applied["canary"]["cohort"], "canary_applied")
        blocked = next(row for row in result["rows"] if row["status"] == "invalidation-required")
        self.assertIn("dependency-changed", blocked["blockers"])
        endpoints = {row["value"]: row["count"] for row in result["endpoint_breakdown"]}
        self.assertEqual(endpoints["responses"], 3)
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", encoded)
        self.assertNotIn("raw response must not leak", encoded)
        self.assertNotIn("raw-openai-request-fingerprint-must-not-leak", encoded)
        self.assertNotIn("raw-openai-session-must-not-leak", encoded)
        self.assertNotIn(pattern_hash, encoded)
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["raw_responses_included"])
        self.assertFalse(result["privacy"]["raw_session_ids_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["request_fingerprints_included"])
        self.assertFalse(result["privacy"]["pattern_hashes_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])

        holdout_policy = json.loads(json.dumps(policy))
        holdout_policy["policies"]["cache"]["pattern_rules"][0]["rollout"]["canary_fraction"] = 0.0
        holdout = build_openai_cache_replay_dry_run(self.store, holdout_policy, limit=20)
        self.assertEqual(holdout["summary"]["projected_applied_rows"], 0)
        self.assertEqual(holdout["summary"]["holdout_rows"], 2)
        self.assertEqual(holdout["summary"]["invalidation_required_rows"], 1)
        holdout_row = next(row for row in holdout["rows"] if row["status"] == "holdout")
        self.assertEqual(holdout_row["canary"]["cohort"], "canary_holdout")
        self.assertTrue(holdout_row["session_scoped_key_available"])

    def test_openai_dry_run_cli_reads_policy_without_mutating_cache(self) -> None:
        pattern_hash = "sha256:" + "c" * 64
        policy = {
            "policies": {
                "cache": {
                    "pattern_rules": [
                        {
                            "id": "openai-cli-cache-rule",
                            "candidate_id": "openai-cli-cache-candidate",
                            "conditions": {
                                "pattern_hashes": [pattern_hash],
                                "source_surface": "openai_chat_completions",
                                "endpoint": "chat_completions",
                                "category": "chat",
                                "has_tools": False,
                                "stream": False,
                            },
                            "action": {"type": "exact_cache_pattern", "scope": "session"},
                        }
                    ]
                }
            }
        }
        policy_path = Path(self.tmpdir.name) / "openai-cache-policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        self.store.set_cache("existing-openai-cli-cache-key", "gpt-5.4-mini", 10, {"output_text": "cached"})
        self._log_openai_call(
            endpoint="chat_completions",
            category="chat",
            request_fingerprint="raw-openai-cli-fingerprint",
            pattern_hashes=[pattern_hash],
            cost=0.01,
        )
        self._log_openai_call(
            endpoint="chat_completions",
            category="chat",
            request_fingerprint="raw-openai-cli-fingerprint",
            pattern_hashes=[pattern_hash],
            cost=0.04,
        )

        stdout = io.StringIO()
        code = cli.openai_cache_replay_dry_run_cli([str(policy_path), "--db", self.db_path, "--limit", "20"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "tokenclaw.openai_cache_replay_dry_run.v1")
        self.assertEqual(payload["summary"]["cache_rows_before"], 1)
        self.assertEqual(payload["summary"]["cache_rows_after"], 1)
        self.assertFalse(payload["summary"]["cache_table_mutated"])
        self.assertEqual(payload["summary"]["projected_hits"], 1)
        self.assertNotIn("existing-openai-cli-cache-key", stdout.getvalue())
        self.assertNotIn("raw-openai-cli-fingerprint", stdout.getvalue())
        self.assertNotIn(pattern_hash, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
