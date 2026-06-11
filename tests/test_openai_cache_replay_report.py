from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agentflow_proxy import cli
from agentflow_proxy.dashboard_app import create_dashboard_app
from agentflow_proxy.openai_cache_replay_dry_run import build_openai_cache_replay_dry_run
from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_impact_report
from agentflow_proxy.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report
from agentflow_proxy.openai_cache_replay_report import build_openai_cache_replay_report
from agentflow_proxy.stats import (
    stats_openai_cache_replay_impact,
    stats_openai_cache_replay_readiness,
    stats_openai_cache_replay_report,
)
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class OpenAICacheReplayReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _audit(self, *, reason: str | None = None, safe: bool = False) -> dict[str, object]:
        return {
            "schema": "agentflow.cache_file_dependency_audit.v1",
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
    ) -> None:
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
            id=str(uuid.uuid4()),
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

        self.assertEqual(report["schema"], "agentflow.openai_cache_replay_opportunity.v1")
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

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        self._log_openai_call(request_fingerprint="raw-cli-request-fingerprint")
        self._log_openai_call(request_fingerprint="raw-cli-request-fingerprint")

        result = asyncio.run(stats_openai_cache_replay_report(self.store, limit=10))
        self.assertEqual(result["schema"], "agentflow.openai_cache_replay_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.openai_cache_replay_report_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.openai_cache_replay_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 2)
        self.assertNotIn("raw-cli-request-fingerprint", output.getvalue())

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
                    "schema": "agentflow.cache_replay_canary_decision.v1",
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
            self._log_openai_call(
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
        self._log_openai_call(
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

        raw_candidate = "raw-cache-key / request_id session secret"
        for index, status_code in enumerate((200, 500)):
            self._log_openai_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0 if status_code == 200 else 0.02,
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

        self.assertEqual(report["schema"], "agentflow.openai_cache_replay_impact.v1")
        self.assertEqual(report["quality_gate"]["schema"], "agentflow.openai_cache_replay_quality_gate.v1")
        self.assertEqual(report["summary"]["applied_count"], 4)
        self.assertEqual(report["summary"]["holdout_count"], 2)
        self.assertEqual(report["summary"]["blocked_count"], 1)
        self.assertEqual(report["summary"]["invalidated_count"], 1)
        self.assertEqual(report["summary"]["safety_stop_count"], 1)
        by_verdict = {row["verdict"]: row for row in report["candidates"]}
        self.assertEqual(by_verdict["promote"]["candidate_id"], "openai-cache-promote")
        self.assertEqual(by_verdict["promote"]["cohort_metrics"]["applied"]["cache_hit_rate"], 1.0)
        self.assertAlmostEqual(by_verdict["promote"]["observed_savings_usd"], 0.07)
        self.assertIn("rollback-error-rate", by_verdict["rollback"]["reason_codes"])
        self.assertIn("safety-stop-observed", by_verdict["rollback"]["reason_codes"])
        self.assertIn("invalidation-rate-above-threshold", by_verdict["rollback"]["reason_codes"])
        self.assertTrue(by_verdict["rollback"]["candidate_id"].startswith("candidate-id:"))
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["raw_responses_included"])
        self.assertFalse(report["privacy"]["tool_payloads_included"])
        self.assertFalse(report["privacy"]["file_paths_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])

        stats_result = asyncio.run(stats_openai_cache_replay_impact(self.store, limit=20))
        self.assertEqual(stats_result["schema"], "agentflow.openai_cache_replay_impact.v1")

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
        with patch("agentflow_proxy.recommendations.recommendations_enabled", return_value=True), patch(
            "agentflow_proxy.recommendations.queue_policy_event_feedback",
            fake_queue,
        ):
            exit_code = cli.openai_cache_replay_impact_cli(["--db", self.db_path, "--limit", "20"], stdout=output)

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "sent")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        self.assertEqual(sent["payload"]["schema"], "agentflow.openai_cache_replay_lifecycle_feedback.v1")
        self.assertEqual(sent["payload"]["lifecycle_kind"], "openai_cache_replay")
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-cache-key-secret", rendered)
        self.assertNotIn("raw-openai-session-must-not-leak", rendered)
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
                    "schema": "agentflow.cache_replay_canary_decision.v1",
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

        for baseline in (0.03, 0.04):
            self._log_openai_call(
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
        self._log_openai_call(
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

        report = build_openai_cache_replay_readiness_report(self.store, opportunity_limit=20, impact_limit=20)
        self.assertEqual(report["schema"], "agentflow.openai_cache_replay_readiness.v1")
        self.assertEqual(report["state"], "saving")
        self.assertEqual(report["summary"]["applied_count"], 2)
        self.assertEqual(report["summary"]["holdout_count"], 1)
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
        self.assertEqual(stats_result["schema"], "agentflow.openai_cache_replay_readiness.v1")

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
            api_response = client.get("/agentflow/stats/openai-cache-replay-readiness?opportunity_limit=20&impact_limit=20")
            dashboard = client.get("/agentflow/dashboard")
        self.assertEqual(api_response.status_code, 200)
        self.assertEqual(api_response.json()["state"], "saving")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("/agentflow/stats/openai-cache-replay-readiness", dashboard.text)
        self.assertIn("OpenAI cache replay readiness", dashboard.text)
        self.assertIn("openai-cache-replay-readiness-tbody", dashboard.text)
        self.assertIn("OpenAI cache replay impact gates", dashboard.text)
        self.assertIn("openai-cache-replay-impact-gates-tbody", dashboard.text)

        output = io.StringIO()
        exit_code = cli.openai_cache_replay_readiness_cli(
            ["--db", self.db_path, "--opportunity-limit", "20", "--impact-limit", "20"],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["schema"], "agentflow.openai_cache_replay_readiness.v1")

        rendered_outputs = [
            json.dumps(report, sort_keys=True),
            json.dumps(api_response.json(), sort_keys=True),
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
                "tool payload must not leak",
                "raw-cache-key / request_id session secret",
                "sha256:" + "c" * 64,
            ):
                self.assertNotIn(forbidden, rendered)

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

        self.assertEqual(result["schema"], "agentflow.openai_cache_replay_dry_run.v1")
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
        self.assertEqual(payload["schema"], "agentflow.openai_cache_replay_dry_run.v1")
        self.assertEqual(payload["summary"]["cache_rows_before"], 1)
        self.assertEqual(payload["summary"]["cache_rows_after"], 1)
        self.assertFalse(payload["summary"]["cache_table_mutated"])
        self.assertEqual(payload["summary"]["projected_hits"], 1)
        self.assertNotIn("existing-openai-cli-cache-key", stdout.getvalue())
        self.assertNotIn("raw-openai-cli-fingerprint", stdout.getvalue())
        self.assertNotIn(pattern_hash, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
