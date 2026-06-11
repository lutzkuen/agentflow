from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.openai_cache_replay_dry_run import build_openai_cache_replay_dry_run
from agentflow_proxy.openai_cache_replay_report import build_openai_cache_replay_report
from agentflow_proxy.stats import stats_openai_cache_replay_report
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

        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=created_at or utc_now(),
            path=path,
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            stream=stream,
            cache_hit=cache_hit,
            status_code=200,
            latency_ms=125,
            input_tokens_est=input_tokens,
            output_tokens_est=50,
            actual_input_tokens=input_tokens,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=cost,
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
            retry_count=0,
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
