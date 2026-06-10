import asyncio
import io
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import yaml

from agentflow_proxy import cli


class ManagedFeedbackFlushClient:
    calls = []
    status_code = 200
    text = '{"ok":true}'

    def __init__(self, *, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def patch(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        return httpx.Response(self.status_code, text=self.text)

    async def post(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        return httpx.Response(self.status_code, text=self.text)


class PolicyReloadCliTests(unittest.TestCase):
    def setUp(self):
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 200
        ManagedFeedbackFlushClient.text = '{"ok":true}'
        self.tmp = TemporaryDirectory()
        self.old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")
        self._old_provenance_env = {
            key: os.environ.get(key)
            for key in (
                "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET",
                "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS",
                "AGENTFLOW_MANAGED_POLICY_HMAC_SECRET",
            )
        }
        for key in self._old_provenance_env:
            os.environ[key] = ""

    def tearDown(self):
        if self.old_event_log is None:
            os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = self.old_event_log
        for key, value in self._old_provenance_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_default_policy_reload_url_uses_agentflow_port(self):
        with patch.dict(os.environ, {"AGENTFLOW_PORT": "4001"}, clear=False):
            self.assertEqual(
                cli._default_policy_reload_url(),
                "http://127.0.0.1:4001/agentflow/admin/reload-policies",
            )

    def test_loopback_url_validation(self):
        self.assertTrue(cli._is_loopback_url("http://127.0.0.1:4000/agentflow/admin/reload-policies"))
        self.assertTrue(cli._is_loopback_url("http://localhost:4000/agentflow/admin/reload-policies"))
        self.assertTrue(cli._is_loopback_url("http://[::1]:4000/agentflow/admin/reload-policies"))
        self.assertFalse(cli._is_loopback_url("http://192.168.1.20:4000/agentflow/admin/reload-policies"))
        self.assertFalse(cli._is_loopback_url("file:///tmp/reload"))

    def test_policy_reload_cli_prints_reload_json_on_success(self):
        payload = {
            "ok": True,
            "schema": "agentflow.policy_reload.v1",
            "policies": {"schema": "agentflow.policy_state.v1"},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli.httpx.post") as post:
            post.return_value = httpx.Response(200, json=payload)
            code = cli.policy_reload_cli(["--url", "http://127.0.0.1:4001/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue()), payload)
        post.assert_called_once_with("http://127.0.0.1:4001/agentflow/admin/reload-policies", timeout=10.0)

    def test_policy_reload_cli_rejects_non_loopback_url(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli.httpx.post") as post:
            code = cli.policy_reload_cli(["--url", "http://192.168.1.20:4000/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"]["type"], "unsafe_url")
        post.assert_not_called()

    def test_policy_reload_cli_reports_non_success_response(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli.httpx.post") as post:
            post.return_value = httpx.Response(403, json={"ok": False, "error": {"type": "forbidden"}})
            code = cli.policy_reload_cli(["--url", "http://127.0.0.1:4000/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status_code"], 403)

    def test_policy_export_cli_prints_policy_bundle_json(self):
        stdout = io.StringIO()

        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CODEX_APP_OPTIMIZE": "1",
                "AGENTFLOW_CODEX_APP_CACHE": "0",
            },
            clear=False,
        ):
            code = cli.policy_export_cli([], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.policy_bundle.v1")
        self.assertEqual(payload["generator"]["mode"], "local-offline")
        self.assertFalse(payload["managed_optimizer"]["enabled"])
        self.assertEqual(payload["policies"]["schema"], "agentflow.policy_state.v1")
        self.assertIn("routing", payload["policies"])
        self.assertIn("codex_app", payload["policies"])
        self.assertFalse(payload["policies"]["codex_app"]["review_only"])
        surface = payload["policies"]["source_surfaces"]["codex_turn"]
        self.assertTrue(surface["optimization"]["enabled"])
        self.assertFalse(surface["cache"]["enabled"])
        self.assertFalse(surface["managed_optimizer_required"])

    def test_codex_diagnose_cli_reads_local_metadata_only(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_codex_app_event(
                    id="start-cli",
                    created_at="2026-06-08T10:00:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-cli",
                    thread_id="thread-cli",
                    message_chars=120,
                    params_chars=80,
                    input_items=1,
                    input_text_chars=64,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="session-cli",
                    routing_json=stable_json({
                        "status": "not-applicable",
                        "reason": "codex-turn-start-model-field-absent",
                        "applied": False,
                        "policy_source": "local-default",
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hash_count": 3,
                            "hash_basis": "normalized-structure-and-size-buckets",
                            "text_bucket": "lt_2k_chars",
                            "token_bucket": "lt_1k_tokens",
                            "pattern_types": ["repeated_input_section"],
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    crunch_json=stable_json({
                        "status": "applied",
                        "reason": "codex-repeated-scaffolding-crunched",
                        "applied": True,
                        "saved_chars": 48,
                        "tokens_saved_est": 12,
                        "codex_repeated_scaffolding": {
                            "status": "applied",
                            "saved_chars": 48,
                            "patterns": [
                                {"type": "repeated_input_section", "count": 1, "saved_chars_est": 48},
                            ],
                        },
                        "codex_patterns": [
                            {"type": "repeated_input_section", "count": 1, "saved_chars_est": 48},
                        ],
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True}),
                )
                store.log_codex_app_event(
                    id="plan-cli",
                    created_at="2026-06-08T10:00:01+00:00",
                    direction="server_to_client",
                    method="turn/plan/updated",
                    request_id=None,
                    thread_id="thread-cli",
                    message_chars=40,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="session-cli",
                )
                store.log_codex_app_event(
                    id="end-cli",
                    created_at="2026-06-08T10:00:02+00:00",
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="req-cli",
                    thread_id="thread-cli",
                    message_chars=90,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=40,
                    error_code=None,
                    error_message=None,
                    latency_ms=25,
                    session_id="session-cli",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.codex_diagnose_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.codex_app_effectiveness.v1")
        self.assertEqual(payload["summary"]["turn_start_rows"], 1)
        self.assertEqual(payload["summary"]["model_field_absent"], 1)
        self.assertEqual(payload["summary"]["codex_repeated_scaffolding_saved_chars"], 48)
        self.assertEqual(payload["summary"]["managed_pattern_fingerprint_rows"], 1)
        self.assertEqual(payload["managed_pattern_fingerprints"]["pattern_hash_count"], 3)
        self.assertFalse(payload["managed_pattern_fingerprints"]["raw_pattern_strings_included"])
        self.assertTrue(payload["recent_samples"][0]["managed_pattern_features"]["present"])
        self.assertEqual(payload["workflow_phase_breakdown"][0]["phase"], "planning")
        self.assertEqual(payload["crunch_pattern_breakdown"][0]["type"], "repeated_input_section")
        self.assertFalse(payload["privacy"]["raw_params_included"])

    def test_cache_replayability_report_cli_reads_local_metadata_only(self):
        from agentflow_proxy.store import Store, stable_json

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for index, cost in enumerate((0.01, 0.02)):
                    store.log_call(
                        id=f"cli-cache-{index}",
                        created_at=f"2026-06-10T01:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=1,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=100,
                        output_tokens_est=10,
                        actual_input_tokens=100,
                        actual_output_tokens=10,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost,
                        crunch_json=stable_json({
                            "pattern_modules": {
                                "server_features": {
                                    "features": [
                                        {
                                            "family": "cacheability",
                                            "features": {
                                                "cacheability_bucket": "high",
                                                "static_information_hint": True,
                                                "exact_cache_candidate_hint": True,
                                            },
                                        }
                                    ],
                                },
                            },
                        }),
                        routing_json=stable_json({"category": "chat", "has_tools": False, "text_chars": 1200}),
                        cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
                        request_json=stable_json({"messages": [{"content": "private cli cache prompt"}]}),
                        session_id="cli-cache-session-secret",
                        category="chat",
                        retry_count=0,
                        provider="anthropic",
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.cache_replayability_report_cli(["--db", db_path, "--limit", "5"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.cache_replayability.v1")
        self.assertEqual(payload["summary"]["repeated_shape_groups"], 1)
        self.assertAlmostEqual(payload["summary"]["projected_repeated_call_cost_usd"], 0.015)
        self.assertEqual(payload["groups"][0]["replay_candidate_class"], "streaming-non-tool-exact-candidate")
        self.assertEqual(payload["groups"][0]["cacheability_bucket"], "high")
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("private cli cache prompt", encoded)
        self.assertNotIn("cli-cache-session-secret", encoded)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])

    def test_cache_replay_dry_run_cli_reads_policy_without_mutating_cache(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "a" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            policy_path = Path(tmp) / "proposed-cache-policy.json"
            policy_path.write_text(
                json.dumps({
                    "policies": {
                        "cache": {
                            "pattern_rules": [
                                {
                                    "id": "cli-cache-dry-run-rule",
                                    "candidate_id": "cli-cache-candidate",
                                    "conditions": {
                                        "pattern_hashes": [pattern_hash],
                                        "source_surface": "anthropic_messages",
                                        "category": "chat",
                                        "has_tools": False,
                                        "stream": False,
                                    },
                                    "action": {"type": "exact_cache_pattern"},
                                }
                            ],
                        }
                    }
                }),
                encoding="utf-8",
            )
            store = Store(db_path)
            try:
                store.set_cache("existing-cli-cache-key", "claude-sonnet-4-6", 10, {"content": "cached"})
                for index, cost in enumerate((0.01, 0.03)):
                    store.log_call(
                        id=f"cli-cache-dry-{index}",
                        created_at=f"2026-06-10T02:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=100,
                        output_tokens_est=10,
                        actual_input_tokens=100,
                        actual_output_tokens=10,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({
                            "text_chars": 1200,
                            "category": "chat",
                            "has_tools": False,
                            "managed_pattern_features": {
                                "pattern_hashes": [pattern_hash],
                                "source_surface": "anthropic_messages",
                                "app_family": "claude_code",
                                "category": "chat",
                                "workflow_phase": "chat",
                                "text_bucket": "lt_2k_chars",
                                "token_bucket": "lt_1k_tokens",
                                "raw_pattern_strings_included": False,
                            },
                        }),
                        cache_json=stable_json({
                            "status": "miss",
                            "reason": "exact-miss",
                            "policy_source": "local-default",
                            "replayability_level": "local-exact-response",
                        }),
                        request_json=stable_json({"messages": [{"content": "raw cli dry run prompt must not leak"}]}),
                        session_id="cli-dry-run-session-secret",
                        category="chat",
                        retry_count=0,
                        provider="anthropic",
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.cache_replay_dry_run_cli(
                [str(policy_path), "--db", db_path, "--scan-limit", "10", "--limit", "10"],
                stdout=stdout,
            )

            with sqlite3.connect(db_path) as conn:
                cache_rows = conn.execute("select count(*) from cache").fetchone()[0]

        self.assertEqual(code, 0)
        self.assertEqual(cache_rows, 1)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.cache_replay_dry_run.v1")
        self.assertEqual(payload["summary"]["cache_rows_before"], 1)
        self.assertEqual(payload["summary"]["cache_rows_after"], 1)
        self.assertFalse(payload["summary"]["cache_table_mutated"])
        self.assertEqual(payload["summary"]["projected_exact_hits"], 1)
        self.assertEqual(payload["summary"]["projected_streaming_hits"], 0)
        self.assertAlmostEqual(payload["summary"]["estimated_saved_cost_usd"], 0.02)
        self.assertEqual(payload["rows"][0]["rule_id"], "cli-cache-dry-run-rule")
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw cli dry run prompt must not leak", encoded)
        self.assertNotIn("cli-dry-run-session-secret", encoded)
        self.assertNotIn(pattern_hash, encoded)
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertFalse(payload["privacy"]["pattern_hashes_included"])

    def test_managed_pattern_rollups_cli_exports_metadata_only_cohorts(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "9" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="cli-pattern-call",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.006,
                    crunch_json=stable_json({
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "rules": [
                                {
                                    "rule_id": "cli-crunch-rule",
                                    "candidate_id": "cli-crunch-candidate",
                                    "policy_source": "managed-recommended",
                                    "matched_hashes": [pattern_hash],
                                    "applied_count": 1,
                                    "saved_chars": 800,
                                    "canary": {
                                        "enabled": True,
                                        "selected": True,
                                        "status": "applied",
                                        "cohort": "canary_applied",
                                    },
                                }
                            ],
                        }
                    }),
                    routing_json=stable_json({"category": "chat"}),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    request_json=stable_json({"prompt": "raw cli prompt must stay local"}),
                    session_id="cli-session-secret",
                    category="chat",
                    retry_count=0,
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_pattern_rollups_cli(["--db", db_path, "--limit", "5", "--min-samples", "1"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.managed_pattern_canary_cohort_rollups.v1")
        self.assertEqual(payload["summary"]["cohort_bucket_count"], 1)
        crunch = next(row for row in payload["cohorts"] if row["candidate_id"] == "cli-crunch-candidate")
        self.assertEqual(crunch["canary_cohort"], "canary_applied")
        self.assertTrue(crunch["minimum_sample_readiness"]["ready"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw cli prompt must stay local", encoded)
        self.assertNotIn("cli-session-secret", encoded)

    def test_managed_pattern_rollups_cli_exports_local_fingerprint_evidence(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "7" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="cli-local-pattern-call",
                    created_at="2026-06-10T01:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.006,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({
                        "category": "tool-result",
                        "workflow_phase": "tool-result",
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hash": pattern_hash,
                            "normalized_pattern_hash": pattern_hash,
                            "pattern_hashes": [pattern_hash],
                            "pattern_hash_count": 1,
                            "hash_basis": "normalized-structure-and-size-buckets",
                            "workflow_phase": "tool-result",
                            "category": "tool-result",
                            "source_surface": "anthropic_messages",
                            "app_family": "claude_code",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "pattern_types": ["tool_results"],
                            "local_pattern_module_families": ["tool_results"],
                            "local_pattern_module_count": 1,
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    request_json=stable_json({"prompt": "raw local fingerprint prompt must stay local"}),
                    session_id="local-fingerprint-session-secret",
                    category="tool-result",
                    retry_count=0,
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_pattern_rollups_cli(["--db", db_path, "--limit", "5", "--min-samples", "1"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        evidence = next(row for row in payload["cohorts"] if row["pattern_hash"] == pattern_hash)
        self.assertEqual(evidence["policy_section"], "local_pattern_fingerprint")
        self.assertTrue(evidence["evidence_only"])
        self.assertEqual(evidence["sample_count"], 1)
        self.assertEqual(evidence["pattern_family"], "general")
        self.assertEqual(evidence["local_pattern_module_families"], ["tool_results"])
        self.assertTrue(evidence["minimum_sample_readiness"]["ready"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw local fingerprint prompt must stay local", encoded)
        self.assertNotIn("local-fingerprint-session-secret", encoded)

    def test_optimization_eval_plan_cli_exports_family_agnostic_metadata_only_plans(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "8" * 64
        cache_pattern_hash = "sha256:" + "6" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="eval-routing-call",
                    created_at="2026-06-10T01:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=100,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    cost_baseline_usd=0.0045,
                    crunch_json=stable_json({"tokens_saved_est": 0}),
                    routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-execution", "has_tools": True, "text_chars": 4000}),
                    cache_json=stable_json({"status": "skipped", "reason": "tool-cache-disabled", "policy_source": "local-default"}),
                    request_json=stable_json({"messages": [{"content": "raw eval routing prompt must stay local"}]}),
                    session_id="eval-routing-session-secret",
                    category="tool-result",
                    retry_count=0,
                    provider="anthropic",
                )
                for index in range(2):
                    store.log_call(
                        id=f"eval-cache-call-{index}",
                        created_at=f"2026-06-10T01:1{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=1000,
                        output_tokens_est=100,
                        actual_input_tokens=1000,
                        actual_output_tokens=100,
                        cost_est_usd=0.01,
                        cost_baseline_usd=0.01,
                        crunch_json=stable_json({
                            "pattern_modules": {
                                "server_features": {
                                    "features": [
                                        {
                                            "family": "cacheability",
                                            "features": {
                                                "cacheability_bucket": "high",
                                                "static_information_hint": True,
                                                "exact_cache_candidate_hint": True,
                                            },
                                        }
                                    ],
                                },
                            },
                        }),
                        routing_json=stable_json({
                            "category": "chat",
                            "workflow_phase": "summary",
                            "has_tools": False,
                            "text_chars": 1200,
                            "managed_pattern_features": {
                                "present": True,
                                "pattern_hash": cache_pattern_hash,
                                "pattern_hashes": [cache_pattern_hash],
                                "source_surface": "anthropic_messages",
                                "app_family": "claude_code",
                                "category": "chat",
                                "workflow_phase": "summary",
                                "text_bucket": "lt_2k_chars",
                                "token_bucket": "lt_1k_tokens",
                                "raw_pattern_strings_included": False,
                            },
                        }),
                        cache_json=stable_json({
                            "status": "miss",
                            "reason": "exact-miss",
                            "policy_source": "local-default",
                            "replayability_level": "local-exact-response",
                        }),
                        request_json=stable_json({"prompt": "raw eval cache prompt must stay local"}),
                        session_id="eval-cache-session-secret",
                        category="chat",
                        retry_count=0,
                        provider="anthropic",
                    )
                store.log_call(
                    id="eval-old-context-call",
                    created_at="2026-06-10T01:30:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=100,
                    input_tokens_est=4000,
                    output_tokens_est=100,
                    actual_input_tokens=4000,
                    actual_output_tokens=100,
                    cost_est_usd=0.02,
                    cost_baseline_usd=0.03,
                    crunch_json=stable_json({
                        "old_context_summarization": {
                            "status": "applied",
                            "enabled": True,
                            "reason": "summary-created",
                            "candidate_id": "eval-old-context-candidate",
                            "rule_id": "eval-old-context-rule",
                            "policy_source": "managed-recommended",
                            "model": "claude-haiku-4-5-20251001",
                            "tokens_saved_est": 1200,
                            "summary_input_tokens": 500,
                            "summary_output_tokens": 100,
                            "summary_cost_est_usd": 0.0002,
                            "estimated_net_savings_usd": 0.009,
                            "canary": {
                                "enabled": True,
                                "fraction": 0.5,
                                "unit": "session",
                                "cohort": "canary_applied",
                            },
                        },
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "rules": [
                                {
                                    "rule_id": "eval-crunch-rule",
                                    "candidate_id": "eval-crunch-candidate",
                                    "policy_source": "managed-recommended",
                                    "matched_hashes": [pattern_hash],
                                    "applied_count": 1,
                                    "saved_chars": 800,
                                    "canary": {
                                        "enabled": True,
                                        "selected": True,
                                        "status": "applied",
                                        "cohort": "canary_applied",
                                    },
                                }
                            ],
                        },
                    }),
                    routing_json=stable_json({"category": "chat", "workflow_phase": "summary"}),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    request_json=stable_json({"prompt": "raw eval old context prompt must stay local"}),
                    session_id="eval-old-context-session-secret",
                    category="chat",
                    retry_count=0,
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.optimization_eval_plan_cli(["--db", db_path, "--limit", "20", "--min-samples", "1"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_eval_plan.v1")
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertGreaterEqual(payload["summary"]["candidate_count"], 4)
        family_counts = {row["value"]: row["count"] for row in payload["summary"]["family_counts"]}
        self.assertIn("phase_routing", family_counts)
        self.assertIn("cache_replayability", family_counts)
        self.assertIn("old_context_summarization", family_counts)
        self.assertIn("managed_pattern_candidate", family_counts)
        self.assertIn("eval-old-context-candidate", {row["candidate_id"] for row in payload["plans"]})
        self.assertIn("eval-crunch-candidate", {row["candidate_id"] for row in payload["plans"]})
        self.assertTrue(any(row["blocker_reason_codes"] for row in payload["plans"]))
        self.assertTrue({row["recommended_eval_mode"] for row in payload["plans"]})
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw eval routing prompt must stay local",
            "raw eval cache prompt must stay local",
            "raw eval old context prompt must stay local",
            "eval-routing-session-secret",
            "eval-cache-session-secret",
            "eval-old-context-session-secret",
            pattern_hash,
            cache_pattern_hash,
        ):
            self.assertNotIn(forbidden, encoded)

    def test_optimization_shadow_eval_cli_scores_fixture_plan_metadata_only(self):
        from agentflow_proxy.store import Store

        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "candidate_id": "shadow-pass",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "projected_savings_usd": 0.02,
                    "shadow_eval_fixture": {
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.98,
                        "baseline_cost_usd": 0.03,
                        "candidate_cost_usd": 0.01,
                        "output": "raw passing output must stay local",
                    },
                    "request_json": {"messages": [{"content": "raw passing prompt must stay local"}]},
                    "session_id": "shadow-pass-session-secret",
                },
                {
                    "candidate_id": "shadow-fail",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "offline_eval": {
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.42,
                    },
                },
                {
                    "candidate_id": "shadow-blocked",
                    "optimization_family": "managed_pattern_candidate",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "features_only",
                    "blocker_reason_codes": ["tool-call-disabled"],
                },
                {
                    "candidate_id": "shadow-unknown",
                    "optimization_family": "old_context_summarization",
                    "action_family": "crunch",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                },
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            jsonl_path = Path(tmp) / "results.jsonl"
            stdout = io.StringIO()

            code = cli.optimization_shadow_eval_cli(
                ["--db", db_path, "--results-jsonl", str(jsonl_path), str(plan_path)],
                stdout=stdout,
            )

            conn = sqlite3.connect(db_path)
            try:
                stored_count = conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
            finally:
                conn.close()
            jsonl_text = jsonl_path.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_shadow_eval.v1")
        self.assertEqual(payload["mode"], "plan-only")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertTrue(payload["wrote_result_records"])
        status_by_candidate = {row["candidate_id"]: row["status_class"] for row in payload["results"]}
        self.assertEqual(status_by_candidate["shadow-pass"], "pass")
        self.assertEqual(status_by_candidate["shadow-fail"], "fail")
        self.assertEqual(status_by_candidate["shadow-blocked"], "blocked")
        self.assertEqual(status_by_candidate["shadow-unknown"], "unknown")
        self.assertEqual(stored_count, 4)
        self.assertEqual(len(jsonl_text.splitlines()), 4)
        rendered = stdout.getvalue() + jsonl_text
        for forbidden in (
            "raw passing prompt must stay local",
            "raw passing output must stay local",
            "shadow-pass-session-secret",
            '"request_json"',
            '"session_id"',
            '"messages"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_optimization_shadow_eval_cli_requires_budget_for_execute(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.optimization_shadow_eval_cli(
            ["--execute", "-"],
            stdin=io.StringIO(json.dumps({"schema": "agentflow.optimization_eval_plan.v1", "plans": []})),
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error"]["type"], "missing_budget_cap")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])

    def test_optimization_shadow_eval_cli_enforces_budget_cap_without_policy_mutation(self):
        from agentflow_proxy.store import Store

        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "candidate_id": "shadow-budget-blocked",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "evidence": {"estimated_eval_cost_usd": 0.25},
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            policy_path = Path(tmp) / "routing_rules.yaml"
            original_policy = "enabled: true\nrules: []\n"
            policy_path.write_text(original_policy, encoding="utf-8")
            stdout = io.StringIO()

            code = cli.optimization_shadow_eval_cli(
                ["--db", db_path, "--execute", "--budget-usd", "0.01", str(plan_path)],
                stdout=stdout,
            )

            policy_after = policy_path.read_text(encoding="utf-8")
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "select status_class, reason_codes_json, result_json from optimization_eval_results"
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(policy_after, original_policy)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "execute")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertEqual(payload["summary"]["budget_exhausted_count"], 1)
        self.assertEqual(payload["results"][0]["status_class"], "blocked")
        self.assertIn("budget-cap-exceeded", payload["results"][0]["reason_codes"])
        self.assertEqual(row[0], "blocked")
        self.assertIn("budget-cap-exceeded", json.loads(row[1]))
        stored_result = json.loads(row[2])
        self.assertFalse(stored_result["provider_call_made"])
        self.assertNotIn("request_json", json.dumps(stored_result))

    def test_policy_validate_cli_accepts_exported_bundle_from_stdin(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["-"], stdin=io.StringIO(exported.getvalue()), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertEqual(payload["errors"], [])

    def test_policy_validate_cli_rejects_invalid_json_with_structured_errors(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["-"], stdin=io.StringIO("{"), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertIn("invalid JSON", payload["errors"][0]["message"])

    def test_policy_validate_cli_rejects_malformed_bundle(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(
            ["-"],
            stdin=io.StringIO(json.dumps({"schema": "wrong"})),
            stdout=stdout,
        )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("$.schema", {error["path"] for error in payload["errors"]})

    def test_policy_diff_cli_reports_file_changes(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())
        after["policies"]["routing"]["enabled"] = not before["policies"]["routing"]["enabled"]

        with TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli([str(before_path), str(after_path)], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["changed_sections"], ["routing"])
        self.assertEqual(payload["changes"][0]["path"], "$.policies.routing.enabled")

    def test_policy_diff_cli_accepts_one_stdin_input(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())

        with TemporaryDirectory() as tmp:
            after_path = Path(tmp) / "after.json"
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli(["-", str(after_path)], stdin=io.StringIO(json.dumps(before)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["changed"])

    def test_policy_diff_cli_rejects_invalid_json_with_structured_errors(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)

        with TemporaryDirectory() as tmp:
            after_path = Path(tmp) / "after.json"
            after_path.write_text(exported.getvalue(), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli(["-", str(after_path)], stdin=io.StringIO("{"), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_diff.v1")
        self.assertIn("invalid JSON", payload["before_validation"]["errors"][0]["message"])

    def test_policy_review_cli_reports_current_to_proposed_changes(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["enabled"] = not proposed["policies"]["routing"]["enabled"]
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_review.v1")
        self.assertEqual(payload["changed_sections"], ["routing"])
        self.assertEqual(payload["change_count"], 1)
        self.assertEqual(payload["safety_warning_count"], 0)
        self.assertIn("impact_summary", payload)

    def test_policy_review_cli_simulates_routing_impact_from_test_db(self):
        from agentflow_proxy.store import Store, stable_json

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"] = [{
            "conditions": {"model_pattern": "sonnet", "category": "chat"},
            "action": {"route_to": "haiku", "reason": "test chat downgrade"},
        }]

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="review-match-ok",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                )
                store.log_call(
                    id="review-match-thinking",
                    created_at="2026-06-08T10:01:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1200,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                    thinking_output_tokens=40,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", db_path],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        impact = payload["impact_summary"]
        self.assertEqual(impact["status"], "simulated")
        self.assertTrue(impact["metadata_only"])
        self.assertFalse(impact["raw_bodies_read"])
        rule = impact["sections"]["routing"]["rules"][0]
        self.assertEqual(rule["would_match_count"], 1)
        self.assertEqual(rule["excluded_thinking_count"], 1)
        self.assertGreater(rule["estimated_savings_usd"], 0)

    def test_policy_review_cli_reports_missing_db_impact_unavailable(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())

        with TemporaryDirectory() as tmp:
            missing_db = str(Path(tmp) / "missing.sqlite3")
            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", missing_db],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        impact = json.loads(stdout.getvalue())["impact_summary"]
        self.assertEqual(impact["status"], "unavailable")
        self.assertEqual(impact["reason"], "db-not-found")

    def test_policy_review_cli_generates_high_risk_impact_warning(self):
        from agentflow_proxy.store import Store, stable_json

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"] = [{
            "conditions": {"model_pattern": "sonnet", "category": "chat"},
            "action": {"route_to": "haiku", "reason": "test risky downgrade"},
        }]

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for index, status in enumerate((200, 500)):
                    store.log_call(
                        id=f"review-risk-{index}",
                        created_at=f"2026-06-08T10:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=status,
                        latency_ms=1000,
                        input_tokens_est=1000,
                        output_tokens_est=100,
                        actual_input_tokens=1000,
                        actual_output_tokens=100,
                        cost_est_usd=0.0045,
                        routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                        cache_json=stable_json({"status": "miss"}),
                        retry_count=1 if status >= 400 else 0,
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", db_path],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        impact = json.loads(stdout.getvalue())["impact_summary"]
        warning_codes = {warning["code"] for warning in impact["warnings"]}
        self.assertIn("high-error-rate-routing-match", warning_codes)

    def _old_context_summary_bundle(self) -> dict:
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["crunch"]["old_context_summarization"].update({
            "enabled": True,
            "rule_id": "test-old-context-dry-run",
            "model": "claude-haiku-4-5-20251001",
            "min_request_chars": 1,
            "min_summarized_chars": 10,
            "max_turns": 3,
            "keep_recent_turns": 1,
            "max_summary_chars": 80,
            "max_source_chars": 10000,
            "max_summary_cost_usd": 1.0,
            "excluded_categories": [],
            "block_tool_protocol": True,
            "block_thinking": True,
        })
        return proposed

    def _write_old_context_summary_dry_run_rows(self, db_path: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        old_text = "raw-secret-old-context " + ("durable fact " * 260)
        eligible_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": old_text},
                {"role": "assistant", "content": "decision: keep src/app.py behavior stable " * 120},
                {"role": "user", "content": "recent request stays verbatim"},
            ],
        }
        tool_blocked_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-secret", "name": "Read", "input": {"file": "secret.py"}}]},
                {"role": "user", "content": "recent request"},
            ],
        }
        store = Store(db_path)
        try:
            store.log_call(
                id="old-summary-eligible",
                created_at="2026-06-08T10:00:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=2000,
                output_tokens_est=100,
                actual_input_tokens=2000,
                actual_output_tokens=100,
                cost_est_usd=0.007,
                routing_json=stable_json({"category": "chat", "text_chars": len(stable_json(eligible_body)), "has_tools": False}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                request_json=stable_json(eligible_body),
                session_id="session-secret-should-not-leak",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
            store.log_call(
                id="old-summary-tool-blocked",
                created_at="2026-06-08T10:01:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=1000,
                output_tokens_est=100,
                actual_input_tokens=1000,
                actual_output_tokens=100,
                cost_est_usd=0.004,
                routing_json=stable_json({"category": "chat", "text_chars": len(stable_json(tool_blocked_body)), "has_tools": True}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "miss"}),
                request_json=stable_json(tool_blocked_body),
                session_id="blocked-session-secret",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
            store.log_call(
                id="old-summary-no-body",
                created_at="2026-06-08T10:02:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=1000,
                output_tokens_est=100,
                actual_input_tokens=1000,
                actual_output_tokens=100,
                cost_est_usd=0.004,
                routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "miss"}),
                request_json=None,
                session_id="missing-body-session-secret",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
        finally:
            store.conn.close()

    def test_old_context_summary_dry_run_cli_reports_metadata_only_projection(self):
        proposed = self._old_context_summary_bundle()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()

            code = cli.old_context_summary_dry_run_cli(
                ["-", "--db", db_path, "--limit", "10"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.old_context_summary_dry_run.v1")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["policy"]["rule_id"], "test-old-context-dry-run")
        self.assertEqual(payload["summary"]["sampled_call_count"], 3)
        self.assertEqual(payload["summary"]["request_body_available_count"], 2)
        self.assertEqual(payload["summary"]["eligible_call_count"], 1)
        self.assertGreater(payload["summary"]["eligible_chars"], 10)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["estimated_summary_cost_usd"], 0)
        self.assertGreater(payload["summary"]["projected_gross_savings_usd"], 0)
        skip_reasons = {row["reason"] for row in payload["summary"]["skip_reasons"]}
        self.assertIn("tool-protocol-context-blocked", skip_reasons)
        self.assertIn("request-body-unavailable", skip_reasons)
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-secret-old-context", encoded)
        self.assertNotIn("session-secret-should-not-leak", encoded)
        self.assertNotIn("tool-secret", encoded)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "disabled")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])

    def test_old_context_summary_dry_run_sends_metadata_only_lifecycle_feedback(self):
        proposed = self._old_context_summary_bundle()
        ManagedFeedbackFlushClient.calls = []
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_dry_run_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(proposed)),
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "dry-run")
        self.assertEqual(sent_payload["policy_sections"], ["crunch"])
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["lifecycle_kind"], "old_context_summarization")
        self.assertEqual(metadata["rule_id"], "test-old-context-dry-run")
        self.assertEqual(metadata["eligible_call_count"], 1)
        self.assertFalse(metadata["privacy"]["cache_keys_included"])
        rendered_payload = json.dumps(sent_payload, sort_keys=True)
        self.assertNotIn("raw-secret-old-context", rendered_payload)
        self.assertNotIn("session-secret-should-not-leak", rendered_payload)
        self.assertNotIn("tool-secret", rendered_payload)
        self.assertNotIn("secret.py", rendered_payload)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertEqual(output["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])

    def _old_context_summary_impact_dry_run(self) -> dict:
        return {
            "schema": "agentflow.old_context_summary_dry_run.v1",
            "ok": True,
            "dry_run": True,
            "read_only": True,
            "generated_at": "2026-06-08T09:00:00+00:00",
            "policy": {
                "policy_source": "managed-recommended",
                "rule_id": "test-old-context-impact",
                "candidate_id": "candidate-old-context-impact",
                "model": "claude-haiku-4-5-20251001",
                "canary": {"enabled": True, "fraction": 0.5, "unit": "source_hash"},
                "safety_stop": {"enabled": True},
            },
            "summary": {
                "eligible_call_count": 4,
                "projected_saved_chars": 16000,
                "projected_saved_tokens": 4000,
                "estimated_summary_cost_usd": 0.001,
                "projected_gross_savings_usd": 0.012,
                "projected_net_savings_usd": 0.011,
            },
            "groups": [
                {
                    "source_surface": "anthropic_messages",
                    "category": "chat",
                    "model_tier": "sonnet",
                    "stream": True,
                    "blocker": "eligible",
                    "call_count": 4,
                    "eligible_call_count": 4,
                }
            ],
            "privacy": {
                "metadata_only_output": True,
                "raw_prompts_included": False,
                "raw_request_bodies_included": False,
                "raw_session_ids_included": False,
                "cache_keys_included": False,
            },
        }

    def _write_old_context_summary_impact_rows(self, db_path: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        store = Store(db_path)
        try:
            rows = [
                (
                    "impact-applied",
                    "2026-06-08T10:00:00+00:00",
                    200,
                    2200,
                    0,
                    {
                        "enabled": True,
                        "status": "applied",
                        "reason": "summary-created",
                        "rule_id": "test-old-context-impact",
                        "candidate_id": "candidate-old-context-impact",
                        "policy_source": "managed-recommended",
                        "category": "chat",
                        "before_chars": 40000,
                        "eligible_chars": 30000,
                        "eligible_turns": 3,
                        "saved_chars": 8000,
                        "tokens_saved_est": 2000,
                        "estimated_gross_savings_usd": 0.006,
                        "summary_cost_est_usd": 0.0004,
                        "estimated_net_savings_usd": 0.0056,
                        "summary_cache_hit": False,
                        "summary_status_code": 200,
                        "canary": {"enabled": True, "cohort": "canary_applied", "selected": True, "fraction": 0.5, "unit": "source_hash"},
                    },
                ),
                (
                    "impact-holdout",
                    "2026-06-08T10:01:00+00:00",
                    200,
                    1100,
                    0,
                    {
                        "enabled": True,
                        "status": "skipped",
                        "reason": "canary_holdout",
                        "rule_id": "test-old-context-impact",
                        "candidate_id": "candidate-old-context-impact",
                        "policy_source": "managed-recommended",
                        "category": "chat",
                        "eligible_chars": 31000,
                        "eligible_turns": 3,
                        "canary": {"enabled": True, "cohort": "canary_holdout", "selected": False, "fraction": 0.5, "unit": "source_hash"},
                    },
                ),
                (
                    "impact-bypass",
                    "2026-06-08T10:02:00+00:00",
                    500,
                    900,
                    2,
                    {
                        "enabled": True,
                        "status": "bypass",
                        "reason": "local-canary-safety-stop",
                        "rule_id": "test-old-context-impact",
                        "candidate_id": "candidate-old-context-impact",
                        "policy_source": "managed-recommended",
                        "category": "chat",
                        "summary_status_code": 500,
                        "summary_error": "redacted upstream error bucket only",
                        "safety_stop_state": "stopped",
                        "canary": {"enabled": True, "cohort": "bypassed", "selected": False, "fraction": 0.5, "unit": "source_hash"},
                    },
                ),
                (
                    "impact-other",
                    "2026-06-08T10:03:00+00:00",
                    200,
                    800,
                    0,
                    {
                        "enabled": True,
                        "status": "applied",
                        "reason": "summary-created",
                        "rule_id": "different-rule",
                        "candidate_id": "different-candidate",
                        "category": "chat",
                        "canary": {"enabled": True, "cohort": "canary_applied"},
                    },
                ),
            ]
            for call_id, created_at, status_code, latency_ms, retry_count, meta in rows:
                store.log_call(
                    id=call_id,
                    created_at=created_at,
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    input_tokens_est=10000,
                    output_tokens_est=200,
                    actual_input_tokens=9000,
                    actual_output_tokens=180,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.04,
                    routing_json=stable_json({"category": "chat", "text_chars": 40000, "has_tools": False}),
                    crunch_json=stable_json({
                        "changed": meta.get("status") == "applied",
                        "old_context_summarization": meta,
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                    request_json=stable_json({"messages": [{"content": "raw impact secret must not leak"}]}),
                    response_json=stable_json({"content": "generated summary secret must not leak"}),
                    session_id="impact-session-secret",
                    category="chat",
                    retry_count=retry_count,
                    provider="anthropic",
                    error="raw error secret must not leak" if status_code >= 400 else None,
                )
        finally:
            store.conn.close()

    def _old_context_summary_quality_gate_dry_run(self) -> dict:
        dry_run = self._old_context_summary_impact_dry_run()
        dry_run["policy"]["rule_id"] = "test-old-context-quality-gate"
        dry_run["policy"]["candidate_id"] = "candidate-old-context-quality-gate"
        dry_run["policy"]["safety_gates"] = {
            "min_outcome_samples": 4,
            "min_canary_applied_samples": 2,
            "min_canary_holdout_samples": 2,
            "min_net_savings_usd": 0.0,
            "min_payback_ratio": 1.0,
            "min_projection_realization_ratio": 0.5,
            "max_error_rate": 0.05,
            "max_error_rate_delta": 0.0,
            "max_retry_rate": 0.10,
            "max_retry_rate_delta": 0.10,
            "max_summary_failure_rate": 0.02,
            "max_bypass_or_disabled_rate": 0.10,
            "max_safety_stop_count": 0,
            "max_latency_regression_ms": 2000,
            "rollback_error_rate": 0.40,
            "rollback_summary_failure_rate": 0.20,
            "rollback_safety_stop_count": 1,
            "rollback_negative_net_savings_usd": 0.0,
        }
        dry_run["summary"]["eligible_call_count"] = 4
        dry_run["summary"]["projected_saved_tokens"] = 2000
        dry_run["summary"]["projected_net_savings_usd"] = 0.004
        return dry_run

    def _write_old_context_summary_quality_gate_rows(self, db_path: str, *, scenario: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        def meta_for(cohort: str, *, status_code: int = 200, retry_count: int = 0) -> dict:
            applied = cohort == "canary_applied"
            meta = {
                "enabled": True,
                "status": "applied" if applied else "skipped",
                "reason": "summary-created" if applied else "canary_holdout",
                "rule_id": "test-old-context-quality-gate",
                "candidate_id": "candidate-old-context-quality-gate",
                "policy_source": "managed-recommended",
                "category": "chat",
                "eligible_chars": 32000,
                "eligible_turns": 3,
                "canary": {
                    "enabled": True,
                    "cohort": cohort,
                    "selected": applied,
                    "fraction": 0.5,
                    "unit": "source_hash",
                },
            }
            if applied:
                meta.update({
                    "before_chars": 40000,
                    "saved_chars": 4000,
                    "tokens_saved_est": 1000,
                    "estimated_gross_savings_usd": 0.003,
                    "summary_cost_est_usd": 0.001,
                    "estimated_net_savings_usd": 0.002,
                    "summary_status_code": 200 if status_code < 400 else status_code,
                    "summary_cache_hit": False,
                })
            if status_code >= 400:
                meta["summary_error"] = "redacted summary failure bucket"
            if retry_count:
                meta["retry_bucket"] = "retried"
            return meta

        if scenario == "insufficient-evidence":
            rows = [("applied-0", "canary_applied", 200, 0, 1000)]
        elif scenario == "hold":
            rows = [
                ("applied-0", "canary_applied", 200, 1, 1000),
                ("applied-1", "canary_applied", 200, 1, 1100),
                ("holdout-0", "canary_holdout", 200, 0, 1000),
                ("holdout-1", "canary_holdout", 200, 0, 1100),
            ]
        elif scenario == "rollback":
            rows = [
                ("applied-0", "canary_applied", 500, 0, 1000),
                ("applied-1", "canary_applied", 200, 0, 1100),
                ("holdout-0", "canary_holdout", 200, 0, 1000),
                ("holdout-1", "canary_holdout", 200, 0, 1100),
            ]
        else:
            rows = [
                ("applied-0", "canary_applied", 200, 0, 1000),
                ("applied-1", "canary_applied", 200, 0, 1100),
                ("holdout-0", "canary_holdout", 200, 0, 1000),
                ("holdout-1", "canary_holdout", 200, 0, 1100),
            ]

        store = Store(db_path)
        try:
            for suffix, cohort, status_code, retry_count, latency_ms in rows:
                store.log_call(
                    id=f"quality-gate-{scenario}-{suffix}",
                    created_at=f"2026-06-08T11:00:0{len(suffix)}+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    input_tokens_est=10000,
                    output_tokens_est=200,
                    actual_input_tokens=9000,
                    actual_output_tokens=180,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.04,
                    routing_json=stable_json({"category": "chat", "text_chars": 40000, "has_tools": False}),
                    crunch_json=stable_json({
                        "changed": cohort == "canary_applied",
                        "old_context_summarization": meta_for(cohort, status_code=status_code, retry_count=retry_count),
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                    request_json=stable_json({"messages": [{"content": "raw quality secret must not leak"}]}),
                    response_json=stable_json({"content": "generated quality summary must not leak"}),
                    session_id="quality-gate-session-secret",
                    category="chat",
                    retry_count=retry_count,
                    provider="anthropic",
                    error="raw quality error must not leak" if status_code >= 400 else None,
                )
        finally:
            store.conn.close()

    def test_old_context_summary_impact_cli_reports_metadata_only_post_apply_evidence(self):
        dry_run = self._old_context_summary_impact_dry_run()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_impact_rows(db_path)
            stdout = io.StringIO()

            code = cli.old_context_summary_impact_cli(
                ["-", "--db", db_path, "--limit", "10"],
                stdin=io.StringIO(json.dumps(dry_run)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.old_context_summary_impact.v1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["projected_affected_metadata_row_count"], 4)
        self.assertEqual(payload["summary"]["actual_matched_metadata_row_count"], 3)
        self.assertEqual(payload["summary"]["actual_canary_applied_count"], 1)
        self.assertEqual(payload["summary"]["actual_canary_holdout_count"], 1)
        self.assertEqual(payload["summary"]["actual_bypassed_or_disabled_count"], 1)
        self.assertEqual(payload["summary"]["summary_failure_count"], 1)
        self.assertEqual(payload["summary"]["actual_tokens_saved_est"], 2000)
        self.assertGreater(payload["summary"]["actual_net_savings_usd"], 0)
        self.assertEqual(payload["actual"]["latency"]["applied_minus_holdout_avg_ms"], 1100.0)
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw impact secret", encoded)
        self.assertNotIn("generated summary secret", encoded)
        self.assertNotIn("impact-session-secret", encoded)
        self.assertNotIn("raw error secret", encoded)
        self.assertFalse(payload["privacy"]["raw_old_context_included"])
        self.assertFalse(payload["privacy"]["generated_summaries_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["tenant_ids_included"])
        self.assertFalse(payload["privacy"]["local_session_ids_included"])
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "disabled")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])

    def test_old_context_summary_quality_gate_verdicts_are_metadata_only(self):
        for scenario, expected_verdict, expected_reason in (
            ("promote", "promote", "quality-gate-passed"),
            ("hold", "hold", "applied-retry-rate-above-threshold"),
            ("rollback", "rollback", "rollback-error-rate"),
            ("insufficient-evidence", "insufficient-evidence", "insufficient-matched-samples"),
        ):
            dry_run = self._old_context_summary_quality_gate_dry_run()
            with self.subTest(scenario=scenario):
                with TemporaryDirectory() as tmp:
                    db_path = str(Path(tmp) / "agentflow.sqlite3")
                    self._write_old_context_summary_quality_gate_rows(db_path, scenario=scenario)
                    stdout = io.StringIO()

                    code = cli.old_context_summary_impact_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                gate = payload["quality_gate"]
                self.assertEqual(gate["schema"], "agentflow.old_context_summary_quality_gate.v1")
                self.assertEqual(gate["verdict"], expected_verdict)
                self.assertIn(expected_reason, gate["reason_codes"])
                self.assertEqual(payload["summary"]["quality_gate_verdict"], expected_verdict)
                self.assertEqual(gate["thresholds"]["min_matched_samples"], 4)
                self.assertEqual(gate["thresholds"]["max_error_rate"], 0.05)
                self.assertEqual(gate["cohorts"]["canary_applied"]["count"], 1 if scenario == "insufficient-evidence" else 2)
                self.assertFalse(gate["privacy"]["raw_old_context_included"])
                self.assertFalse(gate["privacy"]["generated_summaries_included"])
                encoded = json.dumps(payload, sort_keys=True)
                self.assertNotIn("raw quality secret", encoded)
                self.assertNotIn("generated quality summary", encoded)
                self.assertNotIn("quality-gate-session-secret", encoded)
                self.assertNotIn("raw quality error", encoded)

    def test_old_context_summary_impact_cli_exits_nonzero_without_matches(self):
        dry_run = self._old_context_summary_impact_dry_run()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stdout = io.StringIO()

            code = cli.old_context_summary_impact_cli(
                ["-", "--db", db_path, "--limit", "10"],
                stdin=io.StringIO(json.dumps(dry_run)),
                stdout=stdout,
            )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "no-post-apply-matches")
        self.assertEqual(payload["error"]["type"], "no_post_apply_matches")
        self.assertEqual(payload["quality_gate"]["verdict"], "insufficient-evidence")
        self.assertIn("insufficient-matched-samples", payload["quality_gate"]["reason_codes"])

    def test_old_context_summary_impact_sends_metadata_only_lifecycle_feedback(self):
        dry_run = self._old_context_summary_impact_dry_run()
        ManagedFeedbackFlushClient.calls = []
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_impact_rows(db_path)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_impact_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "impact")
        self.assertEqual(sent_payload["policy_sections"], ["crunch"])
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["lifecycle_kind"], "old_context_summarization")
        self.assertEqual(metadata["command"], "old-context-summary-impact")
        self.assertEqual(metadata["rule_id"], "test-old-context-impact")
        self.assertEqual(metadata["candidate_id"], "candidate-old-context-impact")
        self.assertEqual(metadata["actual_matched_metadata_row_count"], 3)
        self.assertFalse(metadata["privacy"]["cache_keys_included"])
        rendered_payload = json.dumps(sent_payload, sort_keys=True)
        self.assertNotIn("raw impact secret", rendered_payload)
        self.assertNotIn("generated summary secret", rendered_payload)
        self.assertNotIn("impact-session-secret", rendered_payload)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertEqual(output["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])
        self.assertTrue(output["managed_server_calls_made"])

    def test_old_context_summary_quality_gate_feedback_queues_and_flushes_metadata_only_verdict(self):
        from agentflow_proxy import recommendations
        from agentflow_proxy.store import Store

        dry_run = self._old_context_summary_quality_gate_dry_run()
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable with raw quality secret"
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_quality_gate_rows(db_path, scenario="promote")
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_impact_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

                    output = json.loads(stdout.getvalue())
                    self.assertEqual(code, 0)
                    self.assertEqual(output["quality_gate"]["verdict"], "promote")
                    self.assertEqual(output["managed_lifecycle_feedback"]["status"], "retryable-error")
                    queue_id = output["managed_lifecycle_feedback"]["queue_id"]

                    store = Store(db_path)
                    try:
                        row = store.get_managed_outcome_feedback(queue_id)
                        self.assertIsNotNone(row)
                        queued_payload = json.loads(row["payload_json"])
                    finally:
                        store.conn.close()

                    metadata = queued_payload["metadata"]
                    gate = metadata["old_context_summary_quality_gate"]
                    self.assertEqual(gate["schema"], "agentflow.old_context_summary_quality_gate_feedback.v1")
                    self.assertEqual(gate["quality_gate_schema"], "agentflow.old_context_summary_quality_gate.v1")
                    self.assertEqual(gate["candidate_id"], "candidate-old-context-quality-gate")
                    self.assertEqual(gate["rule_id"], "test-old-context-quality-gate")
                    self.assertEqual(gate["policy_source"], "managed-recommended")
                    self.assertEqual(gate["verdict"], "promote")
                    self.assertEqual(gate["reason_codes"], ["quality-gate-passed"])
                    self.assertEqual(gate["cohort_counts"]["matched"], 4)
                    self.assertEqual(gate["cohort_counts"]["canary_applied"], 2)
                    self.assertEqual(gate["cohort_counts"]["canary_holdout"], 2)
                    self.assertEqual(gate["safety"]["summary_failure_count"], 0)
                    self.assertEqual(gate["safety"]["safety_stop_count"], 0)
                    self.assertEqual(gate["aggregate_deltas"]["applied_minus_holdout_error_rate"], 0.0)
                    self.assertEqual(gate["aggregate_deltas"]["applied_minus_holdout_retry_rate"], 0.0)
                    self.assertEqual(gate["aggregate_deltas"]["applied_minus_holdout_latency_avg_ms"], 0.0)
                    self.assertGreater(gate["savings"]["net_savings_usd"], 0)
                    self.assertGreater(gate["savings"]["payback_ratio"], 1.0)
                    self.assertFalse(gate["privacy"]["raw_old_context_included"])
                    self.assertFalse(gate["privacy"]["generated_summaries_included"])
                    self.assertFalse(gate["privacy"]["summary_prompts_included"])
                    self.assertFalse(gate["privacy"]["file_contents_included"])
                    self.assertFalse(gate["privacy"]["request_ids_included"])
                    self.assertFalse(gate["privacy"]["tenant_ids_included"])
                    self.assertFalse(gate["privacy"]["local_session_ids_included"])
                    self.assertFalse(gate["privacy"]["cache_keys_included"])
                    def keys_in(value):
                        if isinstance(value, dict):
                            keys = set(value)
                            for child in value.values():
                                keys.update(keys_in(child))
                            return keys
                        if isinstance(value, list):
                            keys = set()
                            for child in value:
                                keys.update(keys_in(child))
                            return keys
                        return set()

                    self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(keys_in(queued_payload) - {"command"}))
                    rendered = json.dumps(queued_payload, sort_keys=True)
                    self.assertNotIn("raw quality secret", rendered)
                    self.assertNotIn("generated quality summary", rendered)
                    self.assertNotIn("quality-gate-session-secret", rendered)
                    self.assertNotIn("raw quality error", rendered)

                    ManagedFeedbackFlushClient.status_code = 200
                    ManagedFeedbackFlushClient.text = '{"ok":true}'
                    flush_stdout = io.StringIO()
                    flush_code = cli.managed_feedback_flush_cli(
                        [
                            "--db",
                            db_path,
                            "--source-surface",
                            recommendations.OLD_CONTEXT_SUMMARY_LIFECYCLE_SOURCE_SURFACE,
                        ],
                        stdout=flush_stdout,
                    )

            self.assertEqual(flush_code, 0)
            flush_payload = json.loads(flush_stdout.getvalue())
            self.assertEqual(flush_payload["flush"]["sent"], 1)
            self.assertEqual(ManagedFeedbackFlushClient.calls[-1]["url"], "http://managed.test/v1/policy-events")
            self.assertEqual(
                ManagedFeedbackFlushClient.calls[-1]["json"]["metadata"]["old_context_summary_quality_gate"]["verdict"],
                "promote",
            )
            store = Store(db_path)
            try:
                flushed = store.get_managed_outcome_feedback(queue_id)
                self.assertEqual(flushed["status"], "sent")
            finally:
                store.conn.close()

    def _summary_rollout_action_bundle(self, *, action_type: str = "widen", raw_like: bool = False, verdict: str = "promote") -> dict:
        action = {
            "schema": "agentflow.old_context_summary_rollout_action.v1",
            "action_type": action_type,
            "target_candidate_id": "candidate-old-context-rollout",
            "target_rule_id": "managed-old-context-summary-rollout",
            "policy_section": "crunch",
            "current_fraction": 0.1,
            "recommended_fraction": 0.35 if action_type == "widen" else 0.0,
            "confidence": 0.92,
            "rationale": "Quality-gated metadata shows positive old-context summary canary outcomes.",
            "blockers": [] if action_type == "widen" else ["quality-gate-rollback"],
            "quality_gate": {
                "schema": "agentflow.old_context_summary_quality_gate.v1",
                "verdict": verdict,
                "reason_codes": ["quality-gate-passed"] if verdict == "promote" else ["applied-retry-rate-above-threshold"],
                "warning_codes": [],
            },
            "required_local_review": True,
            "managed_enforced": False,
            "privacy_summary": {
                "metadata_only": True,
                "raw_context_included": False,
                "generated_summaries_included": False,
            },
        }
        if raw_like:
            action["evidence"] = {"prompt": "raw old context must be rejected"}
        return {
            "schema": "agentflow.old_context_summary_rollout_actions.v1",
            "generated_at": "2026-06-09T04:10:00+00:00",
            "summary": {"action_count": 1, "managed_enforced": False, "required_local_review": True},
            "actions": [action],
            "privacy_summary": {
                "metadata_only": True,
                "raw_context_included": False,
                "generated_summaries_included": False,
            },
        }

    def _write_summary_rollout_rule(self, tmp: str, *, policy_source: str = "managed-recommended"):
        path = Path(tmp) / "crunch_rules.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "threshold_chars": 24000,
                    "old_context_summarization": {
                        "enabled": True,
                        "rule_id": "managed-old-context-summary-rollout",
                        "candidate_id": "candidate-old-context-rollout",
                        "policy_source": policy_source,
                        "model": "claude-haiku-4-5-20251001",
                        "placement": "system",
                        "min_request_chars": 32000,
                        "min_summarized_chars": 12000,
                        "max_turns": 6,
                        "keep_recent_turns": 4,
                        "max_summary_chars": 4000,
                        "max_source_chars": 80000,
                        "max_summary_cost_usd": 0.02,
                        "excluded_categories": ["tool-heavy", "tool-result"],
                        "block_tool_protocol": True,
                        "block_thinking": True,
                        "canary": {
                            "enabled": True,
                            "fraction": 0.1,
                            "salt": "summary-rollout-test",
                            "unit": "source_hash",
                        },
                        "safety_stop": {
                            "enabled": True,
                            "min_outcome_samples": 5,
                            "window": 500,
                            "max_error_rate": 0.1,
                            "max_retry_rate": 0.25,
                            "max_negative_net_savings_rate": 0.5,
                            "max_summary_failure_rate": 0.1,
                            "max_error_rate_delta": 0.05,
                        },
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _log_summary_rollout_call(
        self,
        store,
        *,
        cohort: str,
        created_at: str = "2026-06-09T04:20:00+00:00",
        id_suffix: str = "",
        status_code: int = 200,
    ):
        from agentflow_proxy.store import stable_json

        applied = cohort == "canary_applied"
        meta = {
            "enabled": True,
            "status": "applied" if applied else "skipped",
            "reason": "summary-created" if applied else ("canary_holdout" if cohort == "canary_holdout" else "local-canary-safety-stop"),
            "rule_id": "managed-old-context-summary-rollout",
            "candidate_id": "candidate-old-context-rollout",
            "policy_source": "managed-recommended",
            "category": "chat",
            "eligible_chars": 32000,
            "eligible_turns": 3,
            "saved_chars": 4000 if applied else 0,
            "tokens_saved_est": 1000 if applied else 0,
            "estimated_net_savings_usd": 0.002 if applied else 0.0,
            "summary_status_code": status_code,
            "canary": {
                "enabled": True,
                "cohort": cohort,
                "selected": applied,
                "fraction": 0.1,
                "unit": "source_hash",
            },
        }
        if cohort == "bypassed":
            meta["status"] = "bypass"
            meta["safety_stop_state"] = "stopped"
        store.log_call(
            id=f"summary-rollout-{cohort}{id_suffix}",
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=1000,
            input_tokens_est=10000,
            output_tokens_est=200,
            actual_input_tokens=9000,
            actual_output_tokens=180,
            cost_est_usd=0.03,
            cost_baseline_usd=0.04,
            routing_json=stable_json({"category": "chat"}),
            crunch_json=stable_json({"changed": applied, "old_context_summarization": meta}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            request_json=stable_json({"messages": [{"content": "raw rollout secret must not leak"}]}),
            response_json=stable_json({"content": "generated rollout summary must not leak"}),
            session_id="summary-rollout-session-secret",
            category="chat",
            retry_count=0,
            provider="anthropic",
        )

    def test_old_context_summary_rollout_actions_review_requires_promote_gate(self):
        bundle = self._summary_rollout_action_bundle(verdict="hold")
        with TemporaryDirectory() as tmp:
            self._write_summary_rollout_rule(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            code = cli.old_context_summary_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(code, 1)
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("widen requires local old-context summary quality_gate verdict promote", [error["message"] for error in payload["validation"]["errors"]])

    def test_old_context_summary_rollout_actions_review_reports_fraction_edit(self):
        bundle = self._summary_rollout_action_bundle(action_type="widen")
        with TemporaryDirectory() as tmp:
            self._write_summary_rollout_rule(tmp)
            stdout = io.StringIO()
            code = cli.old_context_summary_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=io.StringIO(),
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        edit = payload["actions"][0]["proposed_edit"]
        self.assertTrue(edit["changed"])
        self.assertEqual(edit["current_fraction"], 0.1)
        self.assertEqual(edit["recommended_fraction"], 0.35)
        self.assertEqual(edit["canary"]["fraction"], 0.35)
        self.assertEqual(payload["actions"][0]["quality_gate"]["verdict"], "promote")

    def test_signed_old_context_summary_rollout_actions_apply_rolls_back_and_creates_backup(self):
        from agentflow_proxy.old_context_summary_rollout_actions import attach_summary_rollout_action_provenance

        secret = "summary-rollout-secret"
        bundle = attach_summary_rollout_action_provenance(
            self._summary_rollout_action_bundle(action_type="rollback", verdict="rollback"),
            secret=secret,
            issuer="agentflow-server",
            server_id="local-dev",
            key_id="summary-rollout-key",
        )
        with TemporaryDirectory() as tmp:
            path = self._write_summary_rollout_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS": json.dumps({"summary-rollout-key": secret})}, clear=False):
                code = cli.old_context_summary_rollout_actions_apply_cli(
                    ["--config-dir", tmp, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
            written = yaml.safe_load(path.read_text(encoding="utf-8"))
            backups = list(Path(tmp).glob("crunch_rules.yaml.bak-*"))

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["provenance"]["status"], "verified")
        self.assertEqual(len(backups), 1)
        summary = written["old_context_summarization"]
        self.assertFalse(summary["enabled"])
        self.assertFalse(summary["canary"]["enabled"])
        self.assertEqual(summary["canary"]["fraction"], 0.0)
        self.assertEqual(summary["rollout_action"]["action_type"], "rollback")
        from agentflow_proxy.policy_events import recent_policy_events

        event = recent_policy_events(limit=1)["events"][0]
        self.assertEqual(event["action"], "old-context-summary-rollout-actions-apply")
        event_text = json.dumps(event)
        self.assertNotIn("raw rollout secret", event_text)
        self.assertNotIn("generated rollout summary", event_text)

    def test_old_context_summary_rollout_actions_reject_raw_and_unknown_before_writing(self):
        with TemporaryDirectory() as tmp:
            path = self._write_summary_rollout_rule(tmp)
            before = path.read_text(encoding="utf-8")
            raw_like = self._summary_rollout_action_bundle(raw_like=True)
            stdout = io.StringIO()
            code = cli.old_context_summary_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(raw_like)),
                stdout=stdout,
            )
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            self.assertIn("raw or prompt-like old-context summary rollout action payloads are not accepted", [error["message"] for error in json.loads(stdout.getvalue())["validation"]["errors"]])

            unknown = self._summary_rollout_action_bundle()
            unknown["actions"][0]["target_rule_id"] = "missing-rule"
            stdout = io.StringIO()
            code = cli.old_context_summary_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(unknown)),
                stdout=stdout,
            )
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(json.loads(stdout.getvalue())["review"]["actions"][0]["reason"], "unknown-rule")

    def test_old_context_summary_rollout_actions_dry_run_and_impact_are_metadata_only(self):
        from agentflow_proxy.store import Store

        bundle = self._summary_rollout_action_bundle(action_type="widen")
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_summary_rollout_rule(tmp)
                self._log_summary_rollout_call(store, cohort="canary_applied")
                self._log_summary_rollout_call(store, cohort="canary_holdout")
                self._log_summary_rollout_call(store, cohort="bypassed")
                dry_stdout = io.StringIO()
                code = cli.old_context_summary_rollout_actions_dry_run_cli(
                    ["--config-dir", tmp, "--db", db_path, "--limit", "10", "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=dry_stdout,
                )
                dry_run = json.loads(dry_stdout.getvalue())
                self._log_summary_rollout_call(store, cohort="canary_applied", created_at="2026-06-09T05:00:01+00:00", id_suffix="-post")
            finally:
                store.conn.close()

            impact_stdout = io.StringIO()
            impact_code = cli.old_context_summary_rollout_actions_impact_cli(
                ["--db", db_path, "--limit", "10", "--since", "2026-06-09T05:00:00+00:00", "-"],
                stdin=io.StringIO(json.dumps(dry_run)),
                stdout=impact_stdout,
            )

        self.assertEqual(code, 0)
        self.assertTrue(dry_run["ok"])
        self.assertTrue(dry_run["read_only"])
        self.assertFalse(dry_run["wrote_policy_files"])
        self.assertEqual(dry_run["summary"]["affected_metadata_row_count"], 3)
        self.assertEqual(dry_run["actions"][0]["current_canary_applied_count"], 1)
        self.assertEqual(dry_run["actions"][0]["current_canary_holdout_count"], 1)
        self.assertEqual(dry_run["actions"][0]["current_bypassed_or_disabled_count"], 1)
        self.assertEqual(dry_run["actions"][0]["projected_fraction"], 0.35)
        self.assertFalse(dry_run["privacy"]["raw_old_context_included"])
        rendered = json.dumps(dry_run)
        self.assertNotIn("summary-rollout-session-secret", rendered)
        self.assertNotIn("raw rollout secret", rendered)

        self.assertEqual(impact_code, 0)
        impact = json.loads(impact_stdout.getvalue())
        self.assertEqual(impact["schema"], "agentflow.old_context_summary_rollout_actions_impact.v1")
        self.assertTrue(impact["ok"])
        self.assertEqual(impact["summary"]["actual_matched_metadata_row_count"], 1)
        self.assertEqual(impact["actions"][0]["actual"]["actual_canary_applied_count"], 1)
        self.assertFalse(impact["privacy"]["generated_summaries_included"])

    def test_policy_review_cli_includes_old_context_summary_dry_run(self):
        proposed = self._old_context_summary_bundle()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()

            code = cli.policy_review_cli(
                ["-", "--db", db_path, "--impact-limit", "10"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        dry_run = payload["impact_summary"]["sections"]["crunch"]["old_context_summary_dry_run"]
        self.assertEqual(dry_run["schema"], "agentflow.old_context_summary_dry_run.v1")
        self.assertEqual(dry_run["summary"]["eligible_call_count"], 1)
        encoded = json.dumps(dry_run, sort_keys=True)
        self.assertNotIn("raw-secret-old-context", encoded)

    def test_policy_review_cli_surfaces_managed_old_context_summary_candidate(self):
        proposed = self._managed_policy_bundle()
        proposed["recommendation"]["candidate_ids"].append("candidate-old-context-summary")
        proposed["recommendation"]["candidate_count"] = 2
        proposed["policies"]["crunch"]["policy_source"] = "managed-recommended"
        proposed["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "candidate-old-context-summary",
                    "candidate_family": "old-context-summarization-policy-rule",
                    "policy_source": "managed-recommended",
                    "confidence": 0.84,
                    "sample_count": 42,
                    "source_model": "claude-sonnet-4-6",
                    "summary_model": "claude-haiku-4-5-20251001",
                    "conditions": {
                        "min_request_chars": 1,
                        "min_summarized_chars": 10,
                        "max_source_chars": 10000,
                    },
                    "action": {
                        "max_turns": 3,
                        "keep_recent_turns": 1,
                        "max_summary_chars": 80,
                        "max_summary_cost_usd": 1.0,
                    },
                    "canary": {
                        "enabled": True,
                        "fraction": 0.25,
                        "salt": "summary-canary-test",
                        "unit": "source_hash",
                        "holdout_sample_count": 9,
                        "widening_threshold": 0.8,
                        "rollback_threshold": 0.2,
                    },
                    "safety_gates": {
                        "max_error_rate": 0.05,
                        "max_summary_failure_rate": 0.02,
                    },
                    "net_savings_evidence": {
                        "projected_net_savings_usd": 0.42,
                        "projected_saved_tokens": 1200,
                        "estimated_summary_cost_usd": 0.01,
                    },
                    "quality_evidence": {
                        "holdout_success_count": 9,
                        "eval_pass_rate": 0.98,
                    },
                    "blocker_reason_codes": ["quality-gate-passed"],
                    "local_action_requirements": {
                        "expected_policy_section": "crunch",
                        "actionability_status": "review-only-local-action",
                    },
                    "privacy_summary": {
                        "metadata_only": True,
                        "raw_body_storage": False,
                        "aggregate_only": False,
                    },
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", db_path, "--impact-limit", "10"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        old_context = payload["section_reviews"]["crunch"]["old_context_summarization"]
        self.assertEqual(old_context["schema"], "agentflow.old_context_summary_policy_review.v1")
        self.assertEqual(old_context["candidate_ids"], ["candidate-old-context-summary"])
        candidate = old_context["candidates"][0]
        self.assertEqual(candidate["candidate_family"], "old-context-summarization-policy-rule")
        self.assertEqual(candidate["source_model"], "claude-sonnet-4-6")
        self.assertEqual(candidate["summary_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(candidate["canary"]["fraction"], 0.25)
        self.assertEqual(candidate["safety_gates"]["max_error_rate"], 0.05)
        self.assertEqual(candidate["net_savings_evidence"]["projected_net_savings_usd"], 0.42)
        self.assertEqual(candidate["quality_evidence"]["holdout_success_count"], 9)
        self.assertEqual(candidate["blocker_reason_codes"], ["quality-gate-passed"])
        self.assertFalse(old_context["application"]["writes_local_policy_files"])
        self.assertFalse(old_context["application"]["provider_calls_made"])
        dry_run = payload["impact_summary"]["sections"]["crunch"]["old_context_summary_dry_run"]
        self.assertEqual(dry_run["policy"]["candidate_id"], "candidate-old-context-summary")
        self.assertEqual(dry_run["summary"]["eligible_call_count"], 1)
        rendered = stdout.getvalue()
        self.assertNotIn("raw-secret-old-context", rendered)
        self.assertIn("old-context summarization candidates: 1", " ".join(payload["human_summary"]))

    def test_policy_review_cli_rejects_raw_like_old_context_summary_fields(self):
        proposed = self._managed_policy_bundle()
        proposed["policies"]["crunch"]["old_context_summarization"]["raw_prompt"] = "managed raw prompt must not be accepted"
        proposed["policies"]["crunch"]["old_context_summarization"]["candidate_id"] = "candidate-old-context-summary"
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        errors = payload["proposed_validation"]["errors"]
        self.assertIn("$.policies.crunch.old_context_summarization.raw_prompt", {error["path"] for error in errors})
        rendered = stdout.getvalue()
        self.assertNotIn("managed raw prompt must not be accepted", rendered)

    def test_policy_impact_simulates_managed_pattern_bundle_without_mutation(self):
        from agentflow_proxy.policy_bundle import simulate_policy_bundle_impact
        from agentflow_proxy.store import Store, stable_json

        crunch_hash = "sha256:" + ("a" * 64)
        cache_hash = "sha256:" + ("b" * 64)
        blocked_cache_hash = "sha256:" + ("c" * 64)
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["crunch"]["policy_source"] = "managed-recommended"
        proposed["policies"]["crunch"]["pattern_rules"] = [{
            "id": "managed-crunch-pattern-test",
            "enabled": True,
            "policy_source": "managed-recommended",
            "candidate_id": "crunch-candidate-test",
            "conditions": {
                "pattern_hashes": [crunch_hash],
                "model_pattern": "sonnet",
                "category": "tool-result",
                "workflow_phase": "tool-result",
                "min_text_chars": 1000,
            },
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "full-review",
                "canary_enabled": False,
                "canary_fraction": 1.0,
                "canary_salt": "sha256:" + ("1" * 64),
                "canary_unit": "request_fingerprint",
            },
            "action": {"type": "shorten", "head_chars": 100, "tail_chars": 100},
            "managed_recommendation": {"estimated_savings_usd": 0.12},
        }]
        proposed["policies"]["cache"]["policy_source"] = "managed-recommended"
        proposed["policies"]["cache"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 2,
            "candidates": [
                {
                    "candidate_id": "cache-canary-test",
                    "policy_source": "managed-recommended",
                    "pattern_hash": cache_hash,
                    "estimated_savings_usd": 0.25,
                    "dimensions": {"source_surface": "anthropic_messages", "app_family": "claude_code", "category": "chat", "phase": "chat"},
                    "buckets": {"has_tools": False},
                    "rollout": {
                        "schema": "agentflow.pattern_policy_rollout.v1",
                        "recommendation_mode": "canary-only",
                        "canary_enabled": True,
                        "canary_fraction": 0.0,
                        "canary_salt": "sha256:" + ("2" * 64),
                        "canary_unit": "request_fingerprint",
                    },
                    "cache_action": {"type": "exact_cache_pattern_review", "streaming": False, "pattern_hashes": [cache_hash]},
                },
                {
                    "candidate_id": "cache-features-only-test",
                    "policy_source": "managed-recommended",
                    "pattern_hash": blocked_cache_hash,
                    "estimated_savings_usd": 0.05,
                    "dimensions": {"source_surface": "codex_turn", "app_family": "codex", "category": "summary", "phase": "summary"},
                    "buckets": {"has_tools": False},
                    "rollout": {
                        "schema": "agentflow.pattern_policy_rollout.v1",
                        "recommendation_mode": "full-review",
                        "canary_enabled": False,
                        "canary_fraction": 1.0,
                        "canary_salt": "sha256:" + ("3" * 64),
                        "canary_unit": "request_fingerprint",
                    },
                    "cache_action": {"type": "exact_cache_pattern_review", "streaming": False, "pattern_hashes": [blocked_cache_hash]},
                },
            ],
            "review_only_candidate_count": 0,
            "omitted_candidate_count": 0,
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            config_path = Path(tmp) / "crunch_rules.yaml"
            config_path.write_text("enabled: true\n", encoding="utf-8")
            store = Store(db_path)
            try:
                store.set_cache("cache-key", "claude-sonnet-4-6", 10, {"content": "cached"})
                store.log_call(
                    id="pattern-crunch-call",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=2000,
                    output_tokens_est=200,
                    actual_input_tokens=2000,
                    actual_output_tokens=200,
                    cost_est_usd=0.006,
                    routing_json=stable_json({
                        "category": "tool-result",
                        "text_chars": 8000,
                        "has_tools": False,
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hashes": [crunch_hash],
                            "pattern_hash_count": 1,
                            "workflow_phase": "tool-result",
                            "category": "tool-result",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                )
                store.log_call(
                    id="pattern-cache-call",
                    created_at="2026-06-08T10:01:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=900,
                    input_tokens_est=1200,
                    output_tokens_est=120,
                    actual_input_tokens=1200,
                    actual_output_tokens=120,
                    cost_est_usd=0.004,
                    routing_json=stable_json({
                        "category": "chat",
                        "text_chars": 4800,
                        "has_tools": False,
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hashes": [cache_hash],
                            "pattern_hash_count": 1,
                            "workflow_phase": "chat",
                            "category": "chat",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                )
                store.log_codex_app_event(
                    id="pattern-codex-turn",
                    created_at="2026-06-08T10:02:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-pattern",
                    thread_id="thread-pattern",
                    message_chars=160,
                    params_chars=5200,
                    input_items=1,
                    input_text_chars=5000,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="session-pattern",
                    routing_json=stable_json({
                        "category": "summary",
                        "workflow_phase": "summary",
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hashes": [blocked_cache_hash],
                            "pattern_hash_count": 1,
                            "workflow_phase": "summary",
                            "category": "summary",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "replayability_level": "features_only",
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "features-only"}),
                )

                before_cache = store.get_cache("cache-key")
                event_log = Path(os.environ["AGENTFLOW_POLICY_EVENTS_LOG"])
                before_event_text = event_log.read_text(encoding="utf-8") if event_log.exists() else ""

                impact = simulate_policy_bundle_impact(proposed, db_path=db_path, limit=10)

                after_cache = store.get_cache("cache-key")
                after_event_text = event_log.read_text(encoding="utf-8") if event_log.exists() else ""
            finally:
                store.conn.close()

            self.assertEqual(config_path.read_text(encoding="utf-8"), "enabled: true\n")

        self.assertEqual(before_cache, {"content": "cached"})
        self.assertEqual(after_cache, {"content": "cached"})
        self.assertEqual(before_event_text, after_event_text)
        self.assertEqual(impact["status"], "simulated")
        self.assertEqual(impact["sampled_call_count"], 2)
        self.assertEqual(impact["sampled_codex_turn_count"], 1)
        crunch_patterns = impact["sections"]["crunch"]["pattern_rules"]
        self.assertEqual(crunch_patterns["would_match_count"], 1)
        self.assertEqual(crunch_patterns["would_apply_count"], 1)
        crunch_candidate = crunch_patterns["candidates"][0]
        self.assertEqual(crunch_candidate["candidate_id"], "crunch-candidate-test")
        self.assertEqual(crunch_candidate["estimated_tokens_affected"], 2000)
        cache_patterns = impact["sections"]["cache"]["pattern_rules"]
        self.assertEqual(cache_patterns["would_match_count"], 2)
        self.assertEqual(cache_patterns["would_apply_count"], 0)
        self.assertEqual(cache_patterns["would_holdout_count"], 1)
        self.assertEqual(cache_patterns["would_bypass_count"], 1)
        blocker_reasons = {
            blocker["reason"]
            for candidate in cache_patterns["candidates"]
            for blocker in candidate["safety_blockers"]
        }
        self.assertIn("features-only-cache-replay-block", blocker_reasons)
        self.assertIn("codex_turn", cache_patterns["candidates"][1]["source_surfaces"])

    def test_policy_review_cli_rejects_invalid_bundle(self):
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps({"schema": "wrong"})), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_review.v1")
        self.assertIn("$.schema", {error["path"] for error in payload["proposed_validation"]["errors"]})
        self.assertFalse(payload["changed"])

    def test_policy_review_cli_surfaces_risky_policy_warnings(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True
        proposed["policies"]["cache"]["semantic_cache"]["enabled"] = True
        proposed["policies"]["crunch"]["old_context_summarization"]["enabled"] = True
        proposed["policies"]["routing"]["policy_source"] = "managed-enforced"
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        warning_codes = {warning["code"] for warning in payload["safety_warnings"]}
        self.assertIn("tool-call-cache-enabled", warning_codes)
        self.assertIn("semantic-cache-enabled", warning_codes)
        self.assertIn("old-context-summarization-enabled", warning_codes)
        self.assertIn("managed-enforced-policy-source", warning_codes)
        self.assertEqual(payload["safety_warning_count"], 4)

    def test_policy_review_cli_records_compact_event(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True
        stdout = io.StringIO()

        cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "review")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["changed_sections"], ["cache"])
        self.assertEqual(events[0]["details"]["safety_warning_count"], 1)

    def _managed_policy_bundle(self, *, invalid: bool = False):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        if invalid:
            return {"schema": "wrong"}
        bundle = json.loads(exported.getvalue())
        bundle["recommendation"] = {
            "schema": "agentflow.policy_bundle_recommendation.v1",
            "policy_source": "managed-recommended",
            "candidate_ids": ["candidate-route-chat"],
            "candidate_count": 1,
            "routing_rule_count": 1,
            "omitted_candidate_count": 0,
            "filters": {"min_samples": 3},
        }
        bundle["managed_optimizer"] = {
            "enabled": False,
            "policy_source": "managed-recommended",
            "note": "Review-only managed recommendation.",
        }
        bundle["policies"]["routing"]["policy_source"] = "managed-recommended"
        bundle["policies"]["routing"].setdefault("rules", []).append({
            "conditions": {
                "model_pattern": "sonnet",
                "category": "chat",
                "has_tools": False,
            },
            "action": {
                "route_to": "claude-haiku-4-5-20251001",
                "reason": "managed candidate for local review",
            },
            "managed_recommendation": {
                "policy_source": "managed-recommended",
                "candidate_id": "candidate-route-chat",
                "confidence": 0.82,
                "sample_count": 24,
                "success_count": 23,
                "error_count": 1,
                "error_rate": 0.041,
                "estimated_savings_usd": 1.23,
                "requested_model": "claude-sonnet-4-6",
                "recommended_target_model": "claude-haiku-4-5-20251001",
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "chat",
            },
        })
        bundle["policies"]["codex_app"] = {
            **bundle["policies"]["codex_app"],
            "policy_source": "managed-recommended",
            "review_only": True,
            "application": {
                "status": "not-applied",
                "reason": "Codex app candidates are review-only in local policy tooling.",
            },
            "rules": [
                {
                    "candidate_id": "candidate-codex-summary",
                    "conditions": {
                        "app_family": "codex",
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                        "input_size_bucket": "small",
                        "cache_eligible": False,
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "crunch_profile": "pass-through",
                        "pass_through_reason": "review-only Codex app recommendation",
                    },
                    "managed_recommendation": {
                        "policy_source": "managed-recommended",
                        "candidate_id": "candidate-codex-summary",
                        "confidence": 0.76,
                        "sample_count": 18,
                    },
                }
            ],
        }
        return bundle

    def _managed_old_context_summary_bundle(self, *, raw_like: bool = False):
        bundle = self._managed_policy_bundle()
        bundle["recommendation"]["candidate_ids"].append("candidate-old-context-summary")
        bundle["recommendation"]["candidate_count"] = 2
        bundle["policies"]["crunch"]["policy_source"] = "managed-recommended"
        candidate = {
            "candidate_id": "candidate-old-context-summary",
            "candidate_family": "old-context-summarization-policy-rule",
            "policy_source": "managed-recommended",
            "confidence": 0.84,
            "sample_count": 42,
            "source_model": "claude-sonnet-4-6",
            "summary_model": "claude-haiku-4-5-20251001",
            "conditions": {
                "min_request_chars": 1,
                "min_summarized_chars": 10,
                "max_source_chars": 10000,
            },
            "action": {
                "max_turns": 3,
                "keep_recent_turns": 1,
                "max_summary_chars": 80,
                "max_summary_cost_usd": 1.0,
            },
            "canary": {
                "enabled": True,
                "fraction": 0.25,
                "salt": "summary-canary-test",
                "unit": "source_hash",
                "holdout_sample_count": 9,
                "widening_threshold": 0.8,
                "rollback_threshold": 0.2,
            },
            "safety_gates": {
                "min_outcome_samples": 5,
                "max_error_rate": 0.05,
                "max_summary_failure_rate": 0.02,
            },
            "net_savings_evidence": {
                "projected_net_savings_usd": 0.42,
                "projected_saved_tokens": 1200,
                "estimated_summary_cost_usd": 0.01,
            },
            "quality_evidence": {
                "holdout_success_count": 9,
                "eval_pass_rate": 0.98,
            },
            "blocker_reason_codes": ["quality-gate-passed"],
            "local_action_requirements": {
                "expected_policy_section": "crunch",
                "actionability_status": "review-only-local-action",
            },
            "privacy_summary": {
                "metadata_only": True,
                "raw_body_storage": False,
                "aggregate_only": False,
            },
            "rationale": "raw-secret-old-context should not be copied to events",
        }
        if raw_like:
            candidate["prompt"] = "raw-secret-old-context must stay out of policy events"
        bundle["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [candidate],
        }
        return bundle

    def _pattern_hash(self):
        return "sha256:" + ("a" * 64)

    def _pattern_hash_char(self, char: str):
        return "sha256:" + (char * 64)

    def _rollout_action_for_rule(
        self,
        *,
        section: str,
        rule_id: str,
        candidate_id: str,
        pattern_hash: str,
        module_family: str,
        recommended_fraction: float = 0.4,
        policy_profile: str | None = "conservative",
    ) -> dict:
        action = {
            "schema": "agentflow.pattern_rollout_action.v1",
            "action_type": "widen",
            "target_candidate_id": candidate_id,
            "target_rule_id": rule_id,
            "policy_section": section,
            "module_family": module_family,
            "pattern_hash": pattern_hash,
            "current_fraction": 0.1,
            "recommended_fraction": recommended_fraction,
            "confidence": 0.91,
            "rationale": "Family-specific canary outcomes are positive.",
            "blockers": [],
            "required_local_review": True,
            "managed_enforced": False,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
        }
        if policy_profile is not None:
            action["policy_profile"] = policy_profile
        return action

    def _rollout_bundle_for_actions(self, actions: list[dict]) -> dict:
        return {
            "schema": "agentflow.pattern_rollout_actions.v1",
            "generated_at": "2026-06-09T03:10:00+00:00",
            "tenant_scope": "local-dev",
            "filters": {"min_samples": 3},
            "thresholds": {"min_samples": 3, "max_error_rate": 0.05},
            "summary": {
                "candidate_count": len(actions),
                "action_count": len(actions),
                "action_counts": {"widen": len(actions)},
                "managed_enforced": False,
                "required_local_review": True,
            },
            "actions": actions,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
        }

    def _write_rollout_pattern_files(
        self,
        tmp: str,
        *,
        crunch_rules: list[dict] | None = None,
        cache_rules: list[dict] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        if crunch_rules is not None:
            path = Path(tmp) / "crunch_rules.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "enabled": True,
                        "threshold_chars": 24000,
                        "pattern_rules": crunch_rules,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            paths["crunch"] = path
        if cache_rules is not None:
            path = Path(tmp) / "cache_rules.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "exact_cache": {"enabled": True, "cache_tool_calls": False},
                        "semantic_cache": {"enabled": False, "threshold": 0.95},
                        "file_watch": {"enabled": True, "root": ".", "max_paths": 128},
                        "pattern_rules": cache_rules,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            paths["cache"] = path
        return paths

    def _family_pattern_rule(
        self,
        *,
        section: str,
        module_family: str,
        rule_id: str,
        candidate_id: str,
        pattern_hash: str,
        action: dict,
        policy_profile: str = "conservative",
        replayability_levels: list[str] | None = None,
    ) -> dict:
        conditions = {
            "pattern_hashes": [pattern_hash],
            "module_family": module_family,
        }
        if section == "crunch":
            conditions.update({
                "min_repeated_count": 2,
                "keep_recent_matches": 1,
            })
        if replayability_levels is not None:
            conditions["replayability_levels"] = replayability_levels
        return {
            "id": rule_id,
            "enabled": True,
            "policy_source": "managed-recommended",
            "candidate_id": candidate_id,
            "module_family": module_family,
            "policy_profile": policy_profile,
            "conditions": conditions,
            "action": action,
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "canary",
                "canary_enabled": True,
                "canary_fraction": 0.1,
                "canary_salt": "local-dev",
                "canary_unit": "request_fingerprint",
                "rollback_threshold": 0.2,
                "widening_threshold": 0.8,
                "min_outcome_samples": 5,
            },
        }

    def _rollout_action_bundle(self, *, action_type: str = "widen", raw_like: bool = False):
        action = {
            "schema": "agentflow.pattern_rollout_action.v1",
            "action_type": action_type,
            "target_candidate_id": "candidate-crunch-rollout",
            "target_rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
            "policy_section": "crunch",
            "pattern_hash": self._pattern_hash(),
            "current_fraction": 0.1,
            "recommended_fraction": 0.25 if action_type == "widen" else 0.0,
            "confidence": 0.91,
            "rationale": "Canary-applied outcomes show positive section savings without error regression.",
            "blockers": [] if action_type == "widen" else ["pattern-applied-cohort-errors"],
            "required_local_review": True,
            "managed_enforced": False,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
            "evidence": {
                "pattern_cohorts": {
                    "outcome_counts": {"applied": 12, "bypassed": 8},
                },
            },
        }
        if raw_like:
            action["evidence"]["raw_request"] = "raw prompt body must be rejected"
        return {
            "schema": "agentflow.pattern_rollout_actions.v1",
            "generated_at": "2026-06-09T03:10:00+00:00",
            "tenant_scope": "local-dev",
            "filters": {"min_samples": 3},
            "thresholds": {"min_samples": 3, "max_error_rate": 0.05},
            "summary": {
                "candidate_count": 1,
                "action_count": 1,
                "action_counts": {action_type: 1},
                "managed_enforced": False,
                "required_local_review": True,
            },
            "actions": [action],
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
        }

    def _write_rollout_crunch_rule(self, tmp: str, *, policy_source: str = "managed-recommended"):
        path = Path(tmp) / "crunch_rules.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "threshold_chars": 24000,
                    "pattern_rules": [
                        {
                            "id": "managed-crunch-pattern-candidate-crunch-rollout",
                            "enabled": True,
                            "policy_source": policy_source,
                            "candidate_id": "candidate-crunch-rollout",
                            "conditions": {
                                "pattern_hashes": [self._pattern_hash()],
                                "min_repeated_count": 2,
                                "keep_recent_matches": 1,
                            },
                            "action": {
                                "type": "shorten",
                                "head_chars": 800,
                                "tail_chars": 600,
                                "max_replacement_chars": 1800,
                            },
                            "rollout": {
                                "schema": "agentflow.pattern_policy_rollout.v1",
                                "recommendation_mode": "canary",
                                "canary_enabled": True,
                                "canary_fraction": 0.1,
                                "canary_salt": "local-dev",
                                "canary_unit": "request_fingerprint",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_managed_rollout_actions_review_cli_reports_local_fraction_edit(self):
        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=io.StringIO(),
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["planned_action_count"], 1)
        edit = payload["actions"][0]["proposed_edit"]
        self.assertTrue(edit["changed"])
        self.assertFalse(edit["disable"])
        self.assertEqual(edit["current_fraction"], 0.1)
        self.assertEqual(edit["recommended_fraction"], 0.25)
        self.assertEqual(edit["rollout"]["canary_fraction"], 0.25)
        self.assertEqual(payload["provenance"]["status"], "not-configured")

    def test_managed_rollout_actions_review_sends_metadata_only_lifecycle_feedback(self):
        bundle = self._rollout_action_bundle(action_type="widen", raw_like=False)
        ManagedFeedbackFlushClient.calls = []

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_review_cli(
                        ["--config-dir", tmp, "--db", db_path, "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                        stderr=io.StringIO(),
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "reviewed")
        self.assertEqual(sent_payload["policy_sections"], ["crunch"])
        self.assertTrue(str(sent_payload["bundle_hash"]).startswith("sha256:"))
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["action_type_counts"]["widen"], 1)
        self.assertEqual(metadata["policy_section_counts"]["crunch"], 1)
        self.assertFalse(metadata["privacy"]["file_paths_included"])
        rendered_payload = json.dumps(sent_payload)
        self.assertNotIn("crunch_rules.yaml", rendered_payload)
        self.assertNotIn("raw_request", rendered_payload)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])

    def test_managed_rollout_actions_apply_dry_run_reports_without_writing(self):
        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "--dry-run", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            after = path.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["applied_sections"], ["crunch"])
        self.assertTrue(payload["files"][0]["changed"])
        self.assertIsNone(payload["files"][0]["backup_path"])
        self.assertEqual(before, after)
        self.assertEqual(payload["actions"][0]["proposed_edit"]["rollout"]["canary_fraction"], 0.25)

    def test_managed_rollout_actions_accept_family_specific_safe_actions(self):
        specs = [
            ("crunch", "tool_results", "b", {"type": "shorten", "head_chars": 900, "tail_chars": 700, "max_replacement_chars": 2200, "preserve_tool_protocol": True}),
            ("crunch", "diffs", "c", {"type": "shorten", "head_chars": 1200, "tail_chars": 800, "max_replacement_chars": 2600, "marker": "[AgentFlow: diff headers and hunks preserved]", "preserve_diff_headers": True, "preserve_hunk_boundaries": True}),
            ("crunch", "generated_artifacts", "d", {"type": "shorten", "head_chars": 800, "tail_chars": 800, "max_replacement_chars": 2200, "marker": "[AgentFlow: generated artifact marker preserved]", "exactness_preserving_marker": True}),
            ("crunch", "tabular_data", "e", {"type": "shorten", "head_chars": 800, "tail_chars": 800, "max_replacement_chars": 2200, "marker": "[AgentFlow: tabular sample preserved]"}),
            ("crunch", "terminal_logs", "f", {"type": "shorten", "head_chars": 800, "tail_chars": 800, "max_replacement_chars": 2200, "marker": "[AgentFlow: terminal log errors preserved]"}),
            ("cache", "cacheability", "1", {"type": "exact_cache", "allow_tool_calls": False, "safe_invalidation": False}),
        ]
        actions = []
        crunch_rules = []
        cache_rules = []
        for section, family, hash_char, local_action in specs:
            pattern_hash = self._pattern_hash_char(hash_char)
            rule_id = f"managed-{section}-{family}"
            candidate_id = f"candidate-{section}-{family}"
            actions.append(
                self._rollout_action_for_rule(
                    section=section,
                    rule_id=rule_id,
                    candidate_id=candidate_id,
                    pattern_hash=pattern_hash,
                    module_family=family,
                    recommended_fraction=0.4,
                )
            )
            rule = self._family_pattern_rule(
                section=section,
                module_family=family,
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hash=pattern_hash,
                action=local_action,
                replayability_levels=["static_information"] if section == "cache" else None,
            )
            if section == "cache":
                cache_rules.append(rule)
            else:
                crunch_rules.append(rule)
        bundle = self._rollout_bundle_for_actions(actions)

        with TemporaryDirectory() as tmp:
            paths = self._write_rollout_pattern_files(tmp, crunch_rules=crunch_rules, cache_rules=cache_rules)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            crunch_written = yaml.safe_load(paths["crunch"].read_text(encoding="utf-8"))
            cache_written = yaml.safe_load(paths["cache"].read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["review"]["planned_action_count"], 6)
        self.assertEqual(payload["applied_sections"], ["cache", "crunch"])
        self.assertEqual({action["family_validation"]["status"] for action in payload["actions"]}, {"accepted"})
        self.assertEqual({action["family_validation"]["family"] for action in payload["actions"]}, {
            "cacheability",
            "diffs",
            "generated_artifacts",
            "tabular_data",
            "terminal_logs",
            "tool_results",
        })
        generated_rule = next(rule for rule in crunch_written["pattern_rules"] if rule["module_family"] == "generated_artifacts")
        self.assertEqual(generated_rule["rollout"]["canary_fraction"], 0.4)
        self.assertEqual(generated_rule["rollout"]["canary_unit"], "request_fingerprint")
        self.assertEqual(generated_rule["rollout_action"]["family_validation"]["family"], "generated_artifacts")
        cache_rule = cache_written["pattern_rules"][0]
        self.assertEqual(cache_rule["rollout"]["canary_fraction"], 0.4)
        self.assertEqual(cache_rule["rollout_action"]["family_validation"]["family"], "cacheability")

    def test_managed_rollout_actions_reject_family_specific_unsafe_actions_before_writing(self):
        ManagedFeedbackFlushClient.calls = []
        unsafe_specs = [
            (
                "cache",
                "tool_results",
                "2",
                {"type": "semantic_cache", "allow_tool_calls": True, "safe_invalidation": False},
                "semantic",
                ["current_state"],
            ),
            (
                "crunch",
                "diffs",
                "3",
                {"type": "omit", "head_chars": 0, "tail_chars": 0, "max_replacement_chars": 200},
                "lossy",
                None,
            ),
            (
                "crunch",
                "generated_artifacts",
                "4",
                {"type": "shorten", "head_chars": 100, "tail_chars": 100, "max_replacement_chars": 500},
                "conservative",
                None,
            ),
            (
                "crunch",
                "tabular_data",
                "5",
                {"type": "shorten", "head_chars": 100, "tail_chars": 100, "max_replacement_chars": 500, "marker": "[AgentFlow: table sample]"},
                "aggressive",
                None,
            ),
            (
                "crunch",
                "cacheability",
                "6",
                {"type": "shorten", "head_chars": 100, "tail_chars": 100, "max_replacement_chars": 500, "marker": "[AgentFlow: cacheability]"},
                "conservative",
                None,
            ),
        ]
        actions = []
        crunch_rules = []
        cache_rules = []
        for section, family, hash_char, local_action, profile, replayability in unsafe_specs:
            pattern_hash = self._pattern_hash_char(hash_char)
            rule_id = f"managed-unsafe-{section}-{family}"
            candidate_id = f"candidate-unsafe-{section}-{family}"
            actions.append(
                self._rollout_action_for_rule(
                    section=section,
                    rule_id=rule_id,
                    candidate_id=candidate_id,
                    pattern_hash=pattern_hash,
                    module_family=family,
                    policy_profile=profile,
                )
            )
            rule = self._family_pattern_rule(
                section=section,
                module_family=family,
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hash=pattern_hash,
                action=local_action,
                policy_profile=profile,
                replayability_levels=replayability,
            )
            if section == "cache":
                cache_rules.append(rule)
            else:
                crunch_rules.append(rule)
        bundle = self._rollout_bundle_for_actions(actions)

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            paths = self._write_rollout_pattern_files(tmp, crunch_rules=crunch_rules, cache_rules=cache_rules)
            before = {section: path.read_text(encoding="utf-8") for section, path in paths.items()}
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )
            after = {section: path.read_text(encoding="utf-8") for section, path in paths.items()}

        self.assertEqual(code, 1)
        self.assertEqual(after, before)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "validation_failed")
        self.assertEqual({action["status"] for action in payload["actions"]}, {"rejected"})
        self.assertEqual({action["reason"] for action in payload["actions"]}, {"family-specific-validation-failed"})
        messages = {error["message"] for error in payload["review"]["errors"]}
        self.assertIn("semantic cache profiles are not accepted for managed pattern YAML rollout", messages)
        self.assertIn("diff crunching may not omit matched content", messages)
        self.assertIn("generated artifact crunching requires an exactness-preserving generated-artifact marker", messages)
        self.assertIn("cacheability pattern actions can only write cache rules", messages)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "rejected")
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["local_status_counts"]["rejected"], 5)
        self.assertEqual(metadata["rejection_reason_counts"]["family-specific-validation-failed"], 5)
        self.assertEqual(metadata["family_validation_status_counts"]["rejected"], 5)
        self.assertFalse(metadata["privacy"]["file_paths_included"])
        rendered = json.dumps(sent_payload)
        self.assertNotIn("crunch_rules.yaml", rendered)
        self.assertNotIn("cache_rules.yaml", rendered)
        self.assertNotIn("raw_request", rendered)

    def test_managed_rollout_actions_apply_dry_run_queues_lifecycle_feedback_when_server_unavailable(self):
        bundle = self._rollout_action_bundle(action_type="widen")
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable"

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )
            status_stdout = io.StringIO()
            cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "rollout_action_lifecycle"],
                stdout=status_stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "retryable-error")
        self.assertEqual(payload["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        status_payload = json.loads(status_stdout.getvalue())
        self.assertEqual(status_payload["summary"]["retryable_error"], 1)
        self.assertEqual(status_payload["oldest_pending"]["source_surface"], "rollout_action_lifecycle")
        self.assertIsNone(status_payload["oldest_pending"]["optimization_unit_id"])
        rendered_status = status_stdout.getvalue()
        self.assertNotIn("crunch_rules.yaml", rendered_status)
        self.assertNotIn("raw_request", rendered_status)
        self.assertFalse(status_payload["privacy"]["payload_json_included"])

    def test_signed_managed_rollout_actions_apply_disables_rule_and_creates_backup(self):
        from agentflow_proxy.rollout_actions import attach_rollout_action_provenance

        secret = "rollout-secret"
        bundle = attach_rollout_action_provenance(
            self._rollout_action_bundle(action_type="rollback"),
            secret=secret,
            issuer="agentflow-server",
            server_id="local-dev",
            key_id="rollout-key",
        )

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS": json.dumps({"rollout-key": secret})}, clear=False):
                code = cli.managed_rollout_actions_apply_cli(
                    ["--config-dir", tmp, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
            written = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["provenance"]["status"], "verified")
            self.assertEqual(payload["applied_sections"], ["crunch"])
            self.assertTrue(payload["files"][0]["changed"])
            self.assertIsNotNone(payload["files"][0]["backup_path"])
            self.assertEqual(len(list(Path(tmp).glob("crunch_rules.yaml.bak-*"))), 1)
            rule = written["pattern_rules"][0]
            self.assertFalse(rule["enabled"])
            self.assertEqual(rule["rollout"]["canary_fraction"], 0.0)
            self.assertFalse(rule["rollout"]["canary_enabled"])
            self.assertEqual(rule["rollout_action"]["action_type"], "rollback")
            self.assertEqual(rule["rollout_action"]["pattern_hash"], self._pattern_hash())

        from agentflow_proxy.policy_events import recent_policy_events

        event = recent_policy_events(limit=1)["events"][0]
        self.assertEqual(event["action"], "rollout-actions-apply")
        self.assertTrue(event["ok"])

    def test_managed_rollout_actions_reject_raw_like_payload_before_writing(self):
        bundle = self._rollout_action_bundle(action_type="widen", raw_like=True)

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            messages = [error["message"] for error in payload["validation"]["errors"]]
            self.assertIn("raw or prompt-like rollout action payloads are not accepted", messages)

    def test_managed_rollout_actions_reject_unknown_rule_before_writing(self):
        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["pattern_rules"][0]["candidate_id"] = "different-candidate"
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["review"]["actions"][0]["reason"], "unknown-rule")

    def _log_rollout_pattern_call(
        self,
        store,
        *,
        pattern_hash: str,
        cohort: str,
        status_code: int = 200,
        created_at: str = "2026-06-09T03:20:00+00:00",
        id_suffix: str = "",
        error: str | None = None,
    ):
        from agentflow_proxy.store import stable_json

        applied = cohort == "canary_applied"
        rule = {
            "rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
            "candidate_id": "candidate-crunch-rollout",
            "policy_source": "managed-recommended",
            "applied_count": 1 if applied else 0,
            "saved_chars": 800 if applied else 0,
            "matched_hashes": [pattern_hash],
            "canary": {
                "enabled": True,
                "status": "holdout" if cohort == "canary_holdout" else "applied",
                "cohort": cohort,
                "fraction": 0.1,
            },
        }
        if cohort == "bypassed":
            rule["skip_reasons"] = [{"reason": "local-canary-safety-stop", "count": 1, "pattern_hash": pattern_hash}]
            rule["canary"]["status"] = "applied"

        store.log_call(
            id=f"call-{cohort}-{status_code}{id_suffix}",
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=1234 if status_code < 400 else 12000,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({
                "changed": applied,
                "policy_source": "managed-recommended",
                "pattern_rules": {
                    "configured_count": 1,
                    "policy_source": "managed-recommended",
                    "category": "tool-result",
                    "rules": [rule],
                },
            }),
            routing_json=stable_json({"category": "tool-result"}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=error,
            request_json=None,
            response_json=None,
            session_id="session-hidden",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )

    def test_managed_rollout_actions_dry_run_reports_recent_traffic_impact(self):
        from agentflow_proxy.store import Store, stable_json

        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="canary_applied")
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="canary_holdout")
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="bypassed")
                store.log_codex_app_event(
                    id="codex-start-rollout",
                    created_at="2026-06-09T03:21:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-rollout",
                    thread_id="thread-rollout",
                    message_chars=100,
                    params_chars=10,
                    input_items=1,
                    input_text_chars=100,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="codex-session-hidden",
                    routing_json=stable_json({"category": "codex_turn"}),
                    crunch_json=stable_json({
                        "changed": True,
                        "policy_source": "managed-recommended",
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "category": "codex_turn",
                            "rules": [
                                {
                                    "rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
                                    "candidate_id": "candidate-crunch-rollout",
                                    "policy_source": "managed-recommended",
                                    "applied_count": 1,
                                    "saved_chars": 400,
                                    "matched_hashes": [self._pattern_hash()],
                                    "canary": {"enabled": True, "status": "applied", "cohort": "canary_applied", "fraction": 0.1},
                                }
                            ],
                        },
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled"}),
                )
                store.log_codex_app_event(
                    id="codex-end-rollout",
                    created_at="2026-06-09T03:21:01+00:00",
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="req-rollout",
                    thread_id="thread-rollout",
                    message_chars=20,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=20,
                    error_code=None,
                    error_message=None,
                    latency_ms=100,
                    session_id="codex-session-hidden",
                )
            finally:
                store.conn.close()

            before_rule_text = (Path(tmp) / "crunch_rules.yaml").read_text(encoding="utf-8")
            before_db_size = Path(db_path).stat().st_size
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "--db", db_path, "--limit", "20", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            after_rule_text = (Path(tmp) / "crunch_rules.yaml").read_text(encoding="utf-8")
            after_db_size = Path(db_path).stat().st_size

        self.assertEqual(code, 0)
        self.assertEqual(before_rule_text, after_rule_text)
        self.assertEqual(before_db_size, after_db_size)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.pattern_rollout_actions_dry_run.v1")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["wrote_policy_files"])
        self.assertFalse(payload["wrote_store"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertEqual(payload["summary"]["sampled_provider_calls"], 3)
        self.assertEqual(payload["summary"]["sampled_codex_turns"], 1)
        self.assertEqual(payload["summary"]["affected_metadata_row_count"], 4)
        action = payload["actions"][0]
        self.assertEqual(action["affected_provider_call_count"], 3)
        self.assertEqual(action["affected_codex_turn_count"], 1)
        self.assertEqual(action["current_canary_applied_count"], 2)
        self.assertEqual(action["current_canary_holdout_count"], 1)
        self.assertEqual(action["current_bypassed_or_disabled_count"], 1)
        self.assertEqual(action["current_fraction"], 0.1)
        self.assertEqual(action["projected_fraction"], 0.25)
        self.assertEqual(action["projected_canary_applied_count"], 2)
        self.assertGreater(action["historical_tokens_saved_est"], 0)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        rendered = json.dumps(payload)
        self.assertNotIn("session-hidden", rendered)
        self.assertNotIn("codex-session-hidden", rendered)

    def test_managed_rollout_actions_impact_reports_post_apply_projection_deltas(self):
        from agentflow_proxy.policy_events import recent_policy_events
        from agentflow_proxy.store import Store, stable_json

        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_applied",
                    created_at="2026-06-09T03:20:00+00:00",
                )
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_holdout",
                    created_at="2026-06-09T03:20:01+00:00",
                )
                dry_stdout = io.StringIO()
                self.assertEqual(
                    cli.managed_rollout_actions_dry_run_cli(
                        ["--config-dir", tmp, "--db", db_path, "--limit", "20", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=dry_stdout,
                    ),
                    0,
                )
                dry_run_report = json.loads(dry_stdout.getvalue())
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_applied",
                    created_at="2026-06-09T04:00:01+00:00",
                    id_suffix="-post",
                )
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_holdout",
                    status_code=500,
                    created_at="2026-06-09T04:00:02+00:00",
                    id_suffix="-post",
                    error='{"error":{"type":"overloaded_error"}}',
                )
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="bypassed",
                    created_at="2026-06-09T04:00:03+00:00",
                    id_suffix="-post",
                )
                store.log_codex_app_event(
                    id="codex-start-rollout-post",
                    created_at="2026-06-09T04:00:04+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-rollout-post",
                    thread_id="thread-rollout-post",
                    message_chars=100,
                    params_chars=10,
                    input_items=1,
                    input_text_chars=100,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="codex-session-hidden-post",
                    routing_json=stable_json({"category": "codex_turn"}),
                    crunch_json=stable_json({
                        "changed": True,
                        "policy_source": "managed-recommended",
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "category": "codex_turn",
                            "rules": [
                                {
                                    "rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
                                    "candidate_id": "candidate-crunch-rollout",
                                    "policy_source": "managed-recommended",
                                    "applied_count": 1,
                                    "saved_chars": 400,
                                    "matched_hashes": [self._pattern_hash()],
                                    "canary": {"enabled": True, "status": "applied", "cohort": "canary_applied", "fraction": 0.25},
                                }
                            ],
                        },
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled"}),
                )
                store.log_codex_app_event(
                    id="codex-end-rollout-post",
                    created_at="2026-06-09T04:00:05+00:00",
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="req-rollout-post",
                    thread_id="thread-rollout-post",
                    message_chars=20,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=20,
                    error_code=None,
                    error_message=None,
                    latency_ms=100,
                    session_id="codex-session-hidden-post",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_rollout_actions_impact_cli(
                ["--db", db_path, "--limit", "20", "--since", "2026-06-09T04:00:00+00:00", "-"],
                stdin=io.StringIO(json.dumps(dry_run_report)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.pattern_rollout_actions_impact.v1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "matched")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["wrote_policy_files"])
        self.assertFalse(payload["wrote_store"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertEqual(payload["summary"]["projected_affected_metadata_row_count"], 2)
        self.assertEqual(payload["summary"]["actual_matched_metadata_row_count"], 4)
        self.assertEqual(payload["summary"]["actual_matched_provider_call_count"], 3)
        self.assertEqual(payload["summary"]["actual_matched_codex_turn_count"], 1)
        action = payload["actions"][0]
        self.assertTrue(action["action_id"].startswith("rollout-action:"))
        self.assertEqual(action["projection"]["affected_metadata_row_count"], 2)
        self.assertEqual(action["actual"]["matched_metadata_row_count"], 4)
        self.assertEqual(action["actual"]["actual_canary_applied_count"], 2)
        self.assertEqual(action["actual"]["actual_canary_holdout_count"], 1)
        self.assertEqual(action["actual"]["actual_bypassed_or_disabled_count"], 1)
        self.assertEqual(action["delta"]["matched_vs_projected_affected_delta"], 2)
        self.assertIn("overloaded_error", {row["value"] for row in action["actual"]["error_buckets"]})
        self.assertIn("gte_10s", {row["value"] for row in action["actual"]["latency_buckets"]})
        self.assertIn("changed", {row["value"] for row in action["actual"]["crunch_decision_status_buckets"]})
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        rendered = json.dumps(payload)
        self.assertNotIn("session-hidden", rendered)
        self.assertNotIn("req-rollout-post", rendered)
        self.assertNotIn("raw_request", rendered)
        self.assertEqual(recent_policy_events(limit=1)["events"][0]["action"], "rollout-actions-impact")

    def test_managed_rollout_actions_impact_is_useful_before_post_apply_matches(self):
        from agentflow_proxy.store import Store

        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_applied",
                    created_at="2026-06-09T03:20:00+00:00",
                )
                dry_stdout = io.StringIO()
                self.assertEqual(
                    cli.managed_rollout_actions_dry_run_cli(
                        ["--config-dir", tmp, "--db", db_path, "--limit", "20", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=dry_stdout,
                    ),
                    0,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_rollout_actions_impact_cli(
                ["--db", db_path, "--limit", "20", "--since", "2026-06-09T05:00:00+00:00", "-"],
                stdin=io.StringIO(dry_stdout.getvalue()),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "no-post-apply-matches")
        self.assertEqual(payload["summary"]["actual_matched_metadata_row_count"], 0)
        self.assertEqual(payload["summary"]["actions_without_post_apply_matches"], 1)
        self.assertEqual(payload["actions"][0]["status"], "no-post-apply-matches")
        self.assertEqual(payload["actions"][0]["projection"]["affected_metadata_row_count"], 1)
        self.assertEqual(payload["actions"][0]["actual"]["matched_metadata_row_count"], 0)
        self.assertEqual(payload["actions"][0]["actual"]["status_risk_buckets"], [])

    def test_managed_rollout_actions_dry_run_covers_hold_rollback_unknown_and_raw_rejection(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="canary_applied")
            finally:
                store.conn.close()

            hold_bundle = self._rollout_action_bundle(action_type="hold")
            hold_bundle["actions"][0]["recommended_fraction"] = 0.1
            rollback_bundle = self._rollout_action_bundle(action_type="rollback")
            for bundle, expected_fraction, expected_disabled in (
                (hold_bundle, 0.1, 0),
                (rollback_bundle, 0.0, 1),
            ):
                stdout = io.StringIO()
                code = cli.managed_rollout_actions_dry_run_cli(
                    ["--config-dir", tmp, "--db", db_path, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
                self.assertEqual(code, 0)
                action = json.loads(stdout.getvalue())["actions"][0]
                self.assertEqual(action["projected_fraction"], expected_fraction)
                self.assertEqual(action["projected_local_bypass_or_disable_count"], expected_disabled)

            unknown = self._rollout_action_bundle(action_type="widen")
            unknown["actions"][0]["target_rule_id"] = "missing-rule"
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "--db", db_path, "-"],
                stdin=io.StringIO(json.dumps(unknown)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["actions"][0]["reason"], "unknown-rule")
            self.assertEqual(payload["error"]["type"], "review_failed")

            raw_like = self._rollout_action_bundle(action_type="widen", raw_like=True)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "--db", db_path, "-"],
                stdin=io.StringIO(json.dumps(raw_like)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            messages = [error["message"] for error in payload["validation"]["errors"]]
            self.assertIn("raw or prompt-like rollout action payloads are not accepted", messages)

    def test_policy_fetch_review_cli_without_config_skips_network(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(os.environ, {cli.POLICY_BUNDLE_RECOMMENDATION_URL_ENV: "", cli.MANAGED_POLICY_API_KEY_ENV: ""}, clear=False):
            with patch("agentflow_proxy.cli.httpx.get") as get:
                code = cli.policy_fetch_review_cli([], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "missing_url")
        self.assertFalse(payload["applied"])
        self.assertFalse(payload["wrote_local_files"])
        get.assert_not_called()

    def test_policy_fetch_review_cli_fetches_reviews_and_does_not_write_rules(self):
        bundle = self._managed_policy_bundle()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_CONFIG_DIR": tmp, cli.MANAGED_POLICY_API_KEY_ENV: ""}, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--min-samples",
                            "3",
                            "--limit",
                            "7",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["validation"]["ok"])
        self.assertTrue(payload["review"]["ok"])
        self.assertEqual(payload["provenance"]["status"], "not-configured")
        self.assertEqual(payload["review"]["provenance"]["status"], "not-configured")
        self.assertFalse(payload["applied"])
        self.assertFalse(payload["wrote_local_files"])
        self.assertEqual(payload["recommendation"]["candidate_ids"], ["candidate-route-chat"])
        self.assertEqual(payload["recommendation"]["candidates"][0]["confidence"], 0.82)
        self.assertEqual(payload["recommendation"]["codex_app_candidate_ids"], ["candidate-codex-summary"])
        self.assertEqual(payload["recommendation"]["codex_app_application_status"], "not-applied")
        self.assertTrue(payload["recommendation"]["codex_app_review_only"])
        codex_review = payload["review"]["section_reviews"]["codex_app"]
        self.assertEqual(codex_review["status"], "review-only")
        self.assertEqual(codex_review["application"]["status"], "not-applied")
        self.assertFalse(codex_review["application"]["writes_local_policy_files"])
        self.assertEqual(payload["bundle"]["schema"], "agentflow.policy_bundle.v1")
        self.assertIn("agentflow-policy-apply", payload["next_manual_command"])
        call = get.call_args
        self.assertEqual(call.kwargs["headers"], {})
        self.assertEqual(call.kwargs["params"]["min_samples"], 3)
        self.assertEqual(call.kwargs["params"]["limit"], 7)

    def test_policy_fetch_review_cli_surfaces_pattern_candidates_without_raw_leakage(self):
        bundle = self._managed_policy_bundle()
        bundle["recommendation"]["candidate_ids"].extend([
            "pattern-crunch-representable",
            "pattern-cache-health-changed",
            "pattern-cache-omitted",
            "pattern-cache-unchanged",
        ])
        bundle["recommendation"]["candidate_count"] = 5
        bundle["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "pattern-crunch-representable",
                    "candidate_family": "crunch-policy-rule",
                    "confidence": 0.81,
                    "sample_count": 64,
                    "estimated_savings_usd": 2.5,
                    "action": {
                        "crunch_profile": "repeated-section-dedupe",
                        "command": "raw command must not print",
                    },
                    "local_action_requirements": {
                        "expected_policy_section": "crunch",
                        "actionability_status": "review-only-local-action",
                    },
                    "confidence_inputs": {
                        "score_family": "crunch-policy-rule",
                        "privacy_profile_counts": {"metadata-only": 64},
                    },
                    "review_evidence": {
                        "crunch": {"saved_tokens_est": 4200},
                        "raw_policy_yaml": "raw yaml must not print",
                    },
                }
            ],
        }
        bundle["policies"]["cache"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 2,
            "omitted_candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "pattern-cache-health-changed",
                    "candidate_family": "cache-policy-rule",
                    "confidence": 0.66,
                    "sample_count": 18,
                    "delta": {"status": "changed-health"},
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                    "evidence": {"api_key": "secret must not print"},
                },
                {
                    "candidate_id": "pattern-cache-unchanged",
                    "candidate_family": "cache-policy-rule",
                    "confidence": 0.58,
                    "sample_count": 14,
                    "delta": {"status": "unchanged"},
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                },
            ],
            "omitted_candidates": [
                {
                    "candidate_id": "pattern-cache-omitted",
                    "candidate_family": "cache-policy-rule",
                    "reason": "cache-policy-rule-not-representable-in-local-bundle-schema-yet",
                    "sample_count": 7,
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                    "evidence": {"raw_response": "raw provider body must not print"},
                }
            ],
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            env = {
                "AGENTFLOW_POLICY_CONFIG_DIR": tmp,
                "AGENTFLOW_POLICY_EVENTS_LOG": str(Path(tmp) / "policy_events.jsonl"),
                cli.MANAGED_POLICY_API_KEY_ENV: "",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--pretty",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

        payload = json.loads(stdout.getvalue())
        rendered = stdout.getvalue() + stderr.getvalue()

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["recommendation"]["pattern_candidate_count"], 4)
        self.assertEqual(payload["recommendation"]["crunch_pattern_candidate_count"], 1)
        self.assertEqual(payload["recommendation"]["cache_pattern_candidate_count"], 3)
        self.assertEqual(payload["review"]["section_reviews"]["crunch"]["candidate_count"], 1)
        self.assertEqual(payload["review"]["section_reviews"]["cache"]["changed_health_candidate_count"], 1)
        self.assertEqual(payload["review"]["section_reviews"]["cache"]["unchanged_candidate_count"], 1)
        self.assertIn("crunch pattern candidates: 1 total", " ".join(payload["review"]["human_summary"]))
        self.assertNotIn("raw command must not print", rendered)
        self.assertNotIn("raw yaml must not print", rendered)
        self.assertNotIn("secret must not print", rendered)
        self.assertNotIn("raw provider body must not print", rendered)

    def test_policy_fetch_review_cli_surfaces_managed_health_without_raw_leakage(self):
        bundle = self._managed_policy_bundle()
        bundle["recommendation"]["health"] = {
            "generated_at": "2026-06-08T12:00:00+00:00",
            "privacy_summary": {
                "telemetry_profile": "metadata-only",
                "raw_body_storage": False,
                "raw_prompts_included": False,
            },
            "stale_evidence": [
                {
                    "candidate_id": "candidate-route-chat",
                    "last_seen_at": "2026-06-01T12:00:00+00:00",
                    "raw_prompt": "raw prompt secret must not print",
                }
            ],
            "insufficient_samples": [
                {
                    "candidate_id": "candidate-route-chat",
                    "sample_count": 2,
                    "min_samples": 10,
                    "body": "raw request body must not print",
                }
            ],
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            env = {
                "AGENTFLOW_POLICY_CONFIG_DIR": tmp,
                "AGENTFLOW_POLICY_EVENTS_LOG": str(Path(tmp) / "policy_events.jsonl"),
                cli.MANAGED_POLICY_API_KEY_ENV: "",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--min-samples",
                            "10",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )

                from agentflow_proxy.policy_events import recent_policy_events

                events = recent_policy_events(limit=5)["events"]

        rendered = stdout.getvalue() + stderr.getvalue()
        payload = json.loads(stdout.getvalue())
        warning_codes = {warning["code"] for warning in payload["review"]["safety_warnings"]}

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["validation"]["ok"])
        self.assertTrue(payload["review"]["ok"])
        self.assertEqual(payload["recommendation"]["health"]["status"], "warning")
        self.assertEqual(payload["recommendation"]["health"]["counts"]["stale_evidence"], 1)
        self.assertEqual(payload["recommendation"]["health"]["counts"]["insufficient_samples"], 1)
        self.assertIn("managed-recommendation-stale-evidence", warning_codes)
        self.assertIn("managed-recommendation-insufficient-samples", warning_codes)
        self.assertNotIn("raw prompt secret", rendered)
        self.assertNotIn("raw request body", rendered)
        self.assertNotIn('"body"', rendered)
        self.assertEqual(events[0]["details"]["recommendation_health"]["warning_count"], 2)

    def test_policy_fetch_review_cli_rejects_invalid_bundle(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_CONFIG_DIR": tmp}, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=self._managed_policy_bundle(invalid=True))
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertEqual(list(Path(tmp).glob("*.yaml")), [])

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "validation_failed")
        self.assertIn("$.schema", {error["path"] for error in payload["validation"]["errors"]})
        self.assertFalse(payload["wrote_local_files"])

    def test_policy_fetch_review_cli_sends_auth_without_secret_leakage(self):
        secret = "super-secret-managed-key"
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(os.environ, {cli.MANAGED_POLICY_API_KEY_ENV: secret}, clear=False):
            with patch("agentflow_proxy.cli.httpx.get") as get:
                get.return_value = httpx.Response(200, json=self._managed_policy_bundle())
                code = cli.policy_fetch_review_cli(
                    [
                        "--url",
                        "http://managed.test/v1/policy-bundle-recommendation",
                        "--tenant",
                        "tenant-a",
                    ],
                    stdout=stdout,
                    stderr=stderr,
                )

        self.assertEqual(code, 0)
        self.assertEqual(get.call_args.kwargs["headers"]["authorization"], f"Bearer {secret}")
        self.assertEqual(get.call_args.kwargs["headers"]["x-agentflow-tenant"], "tenant-a")
        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(secret, rendered)

        from agentflow_proxy.policy_events import recent_policy_events

        event_text = json.dumps(recent_policy_events(limit=5)["events"])
        self.assertNotIn(secret, event_text)
        self.assertIn("env:AGENTFLOW_MANAGED_API_KEY", event_text)

    def test_policy_apply_cli_dry_run_reports_files_without_writing(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 0.91

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--dry-run", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(set(payload["applied_sections"]), {"routing", "crunch", "cache", "routing_experiments", "codex_app"})
            self.assertFalse(payload["skipped_sections"])
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())
            self.assertTrue(any(file["section"] == "cache" and file["changed"] for file in payload["files"]))
            self.assertTrue(any(file["section"] == "codex_app" for file in payload["files"]))

    def test_policy_apply_cli_dry_run_shows_old_context_summary_canary_yaml_diff(self):
        proposed = self._managed_old_context_summary_bundle()

        with TemporaryDirectory() as tmp:
            crunch_path = Path(tmp) / "crunch_rules.yaml"
            crunch_path.write_text("enabled: true\nthreshold_chars: 24000\n", encoding="utf-8")
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "--dry-run", "--pretty", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["applied_sections"], ["crunch"])
            self.assertEqual(payload["old_context_summarization"]["status"], "dry-run")
            self.assertEqual(payload["old_context_summarization"]["selected_candidate_id"], "candidate-old-context-summary")
            self.assertEqual(payload["old_context_summarization"]["canary_fraction"], 0.25)
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), "enabled: true\nthreshold_chars: 24000\n")
            self.assertEqual(len(payload["files"]), 1)
            self.assertEqual(payload["files"][0]["section"], "crunch")
            diff = payload["files"][0]["diff"]
            self.assertIn("+  candidate_id: candidate-old-context-summary", diff)
            self.assertIn("+    fraction: 0.25", diff)
            self.assertIn("+  policy_source: managed-recommended", diff)
            self.assertNotIn("recommendation:", diff)
            self.assertNotIn("raw-secret-old-context", stdout.getvalue())

    def test_policy_apply_cli_writes_managed_old_context_summary_canary_and_rollback_finds_backup(self):
        proposed = self._managed_old_context_summary_bundle()

        with TemporaryDirectory() as tmp:
            crunch_path = Path(tmp) / "crunch_rules.yaml"
            previous = "enabled: true\nthreshold_chars: 24000\n"
            crunch_path.write_text(previous, encoding="utf-8")
            apply_stdout = io.StringIO()
            apply_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=apply_stdout,
            )
            apply_payload = json.loads(apply_stdout.getvalue())

            self.assertEqual(apply_code, 0)
            self.assertEqual(apply_payload["old_context_summarization"]["status"], "applied")
            applied = yaml.safe_load(crunch_path.read_text(encoding="utf-8"))
            rule = applied["old_context_summarization"]
            self.assertTrue(rule["enabled"])
            self.assertEqual(rule["candidate_id"], "candidate-old-context-summary")
            self.assertEqual(rule["policy_source"], "managed-recommended")
            self.assertEqual(rule["model"], "claude-haiku-4-5-20251001")
            self.assertEqual(rule["source_model"], "claude-sonnet-4-6")
            self.assertEqual(rule["min_summarized_chars"], 10)
            self.assertEqual(rule["max_turns"], 3)
            self.assertTrue(rule["canary"]["enabled"])
            self.assertEqual(rule["canary"]["fraction"], 0.25)
            self.assertEqual(rule["canary"]["salt"], "summary-canary-test")
            self.assertEqual(rule["canary"]["widening_threshold"], 0.8)
            self.assertEqual(rule["canary"]["rollback_threshold"], 0.2)
            self.assertTrue(rule["safety_stop"]["enabled"])
            self.assertEqual(rule["safety_stop"]["max_error_rate"], 0.05)
            self.assertEqual(rule["safety_stop"]["max_summary_failure_rate"], 0.02)
            backups = list(Path(tmp).glob("crunch_rules.yaml.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), previous)

            rollback_stdout = io.StringIO()
            rollback_code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "crunch", "--dry-run"],
                stdout=rollback_stdout,
            )
            rollback_payload = json.loads(rollback_stdout.getvalue())

            self.assertEqual(rollback_code, 0)
            self.assertEqual(rollback_payload["old_context_summarization"]["status"], "rollback-dry-run")
            self.assertEqual(rollback_payload["restored_sections"], ["crunch"])
            self.assertEqual(rollback_payload["files"][0]["restored_from"], str(backups[0]))
            self.assertNotEqual(crunch_path.read_text(encoding="utf-8"), previous)

    def test_policy_apply_cli_normalizes_managed_enforced_old_context_summary_to_recommended_canary(self):
        proposed = self._managed_policy_bundle()
        proposed["policies"]["crunch"]["policy_source"] = "managed-recommended"
        proposed["policies"]["crunch"]["old_context_summarization"] = {
            "enabled": True,
            "rule_id": "managed-enforced-summary-rule",
            "candidate_id": "candidate-old-context-enforced",
            "policy_source": "managed-enforced",
            "model": "claude-haiku-4-5-20251001",
            "min_request_chars": 100,
            "min_summarized_chars": 50,
            "canary": {
                "enabled": True,
                "fraction": 0.1,
                "salt": "direct-canary",
                "unit": "source_hash",
            },
            "safety_stop": {
                "enabled": True,
                "max_error_rate": 0.05,
            },
        }

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "--allow-risky", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            applied = yaml.safe_load((Path(tmp) / "crunch_rules.yaml").read_text(encoding="utf-8"))
            rule = applied["old_context_summarization"]
            self.assertEqual(rule["candidate_id"], "candidate-old-context-enforced")
            self.assertEqual(rule["policy_source"], "managed-recommended")

    def test_policy_apply_cli_records_old_context_summary_events_without_raw_payloads(self):
        proposed = self._managed_old_context_summary_bundle()

        with TemporaryDirectory() as tmp:
            (Path(tmp) / "crunch_rules.yaml").write_text("enabled: true\n", encoding="utf-8")
            cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=io.StringIO(),
            )
            cli.policy_rollback_cli(["--config-dir", tmp, "--section", "crunch"], stdout=io.StringIO())

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        event_text = json.dumps(events)
        self.assertEqual(events[0]["action"], "rollback")
        self.assertEqual(events[0]["details"]["old_context_summarization"]["status"], "rolled-back")
        self.assertEqual(events[1]["action"], "apply")
        self.assertEqual(events[1]["details"]["old_context_summarization"]["status"], "applied")
        self.assertEqual(events[1]["details"]["old_context_summarization"]["selected_candidate_id"], "candidate-old-context-summary")
        self.assertNotIn("raw-secret-old-context", event_text)
        for forbidden in ("prompt", "summary_text", "messages", "transcript", "provider_body", "cache_key"):
            self.assertNotIn(forbidden, event_text)

    def test_policy_apply_cli_rejected_old_context_summary_event_omits_raw_payloads(self):
        proposed = self._managed_old_context_summary_bundle(raw_like=True)

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["old_context_summarization"]["status"], "rejected")

        from agentflow_proxy.policy_events import recent_policy_events

        event_text = json.dumps(recent_policy_events(limit=5)["events"])
        self.assertIn("candidate-old-context-summary", event_text)
        self.assertNotIn("raw-secret-old-context", event_text)

    def test_policy_apply_cli_writes_selected_section_and_creates_backup(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 0.91

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: false\n", encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["applied_sections"], ["cache"])
            self.assertEqual(payload["skipped_sections"][0]["reason"], "not-requested")
            applied = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(applied["semantic_cache"]["threshold"], 0.91)
            backups = list(Path(tmp).glob("cache_rules.yaml.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("enabled: false", backups[0].read_text(encoding="utf-8"))

            second_stdout = io.StringIO()
            second_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=second_stdout,
            )
            second_payload = json.loads(second_stdout.getvalue())
            self.assertEqual(second_code, 0)
            self.assertFalse(second_payload["files"][0]["changed"])
            self.assertEqual(len(list(Path(tmp).glob("cache_rules.yaml.bak-*"))), 1)

    def test_policy_apply_cli_refuses_risky_bundle_unless_allowed(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "risky_policy")
            self.assertEqual(payload["safety_warning_count"], 1)
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

            allowed_stdout = io.StringIO()
            allowed_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "--allow-risky", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=allowed_stdout,
            )
            allowed_payload = json.loads(allowed_stdout.getvalue())
            self.assertEqual(allowed_code, 0)
            self.assertTrue(allowed_payload["ok"])
            self.assertTrue(yaml.safe_load((Path(tmp) / "cache_rules.yaml").read_text(encoding="utf-8"))["exact_cache"]["cache_tool_calls"])

    def test_policy_apply_cli_rejects_malformed_section_schema_before_writing(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"][0]["conditions"]["text_chars_lt"] = "small"
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 2

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            paths = {error["path"] for error in payload["validation"]["errors"]}
            self.assertIn("$.policies.routing.rules[0].conditions.text_chars_lt", paths)
            self.assertIn("$.policies.cache.semantic_cache.threshold", paths)
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

    def test_policy_rollback_cli_dry_run_reports_latest_backup_without_writing(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "cache", "--dry-run"],
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["restored_sections"], ["cache"])
            self.assertTrue(payload["files"][0]["changed"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")
            self.assertEqual(len(list(Path(tmp).glob("cache_rules.yaml.bak-*"))), 1)

    def test_policy_rollback_cli_restores_selected_section_and_backs_up_current_file(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            latest = Path(tmp) / "cache_rules.yaml.bak-20260102T000000000000Z"
            latest.write_text("exact_cache:\n  enabled: newest\n", encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_rollback_cli(["--config-dir", tmp, "--section", "cache"], stdout=stdout)

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "agentflow.policy_bundle_rollback.v1")
            self.assertEqual(payload["restored_sections"], ["cache"])
            self.assertEqual(payload["files"][0]["restored_from"], str(latest))
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: newest\n")
            current_backups = [
                path
                for path in Path(tmp).glob("cache_rules.yaml.bak-*")
                if path.name != "cache_rules.yaml.bak-20260101T000000000000Z"
                and path.name != "cache_rules.yaml.bak-20260102T000000000000Z"
            ]
            self.assertEqual(len(current_backups), 1)
            self.assertEqual(current_backups[0].read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")

    def test_policy_rollback_cli_missing_backup_fails_without_partial_writes(self):
        with TemporaryDirectory() as tmp:
            routing_path = Path(tmp) / "routing_rules.yaml"
            cache_path = Path(tmp) / "cache_rules.yaml"
            routing_path.write_text("rules: []\n", encoding="utf-8")
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "routing", "--section", "cache"],
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "missing_backups")
            self.assertEqual(payload["error"]["sections"], ["routing"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")
            self.assertEqual(len(list(Path(tmp).glob("cache_rules.yaml.bak-*"))), 1)

    def test_policy_rollback_cli_records_compact_event(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            cli.policy_rollback_cli(["--config-dir", tmp, "--section", "cache"], stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "rollback")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["restored_sections"], ["cache"])
        self.assertEqual(events[0]["details"]["exit_code"], 0)

    def test_policy_cli_records_compact_local_events(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())
        after["policies"]["routing"]["enabled"] = not before["policies"]["routing"]["enabled"]

        with TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            cli.policy_diff_cli([str(before_path), str(after_path)], stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "diff")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["changed_sections"], ["routing"])
        self.assertEqual(events[0]["details"]["change_count"], 1)
        self.assertEqual(events[1]["action"], "export")
        self.assertIn("routing", events[1]["details"]["policies"])


class ManagedFeedbackCliTests(unittest.TestCase):
    def setUp(self):
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 200
        ManagedFeedbackFlushClient.text = '{"ok":true}'

    def _enqueue_feedback(self, store, *, status="queued", attempts=0):
        from agentflow_proxy.store import stable_json

        store.enqueue_managed_outcome_feedback(
            id=f"queue-{status}-{attempts}",
            created_at="2026-06-08T10:00:00+00:00",
            updated_at="2026-06-08T10:00:00+00:00",
            source_surface="codex_turn",
            endpoint="/v1/optimization-units/77/outcome",
            optimization_unit_id=77,
            payload_json=stable_json({
                "status": "success",
                "quality_signals": {"status": "success"},
                "pattern_decisions": [
                    {
                        "schema": "agentflow.pattern_decision_summary.v1",
                        "decision_type": "routing",
                        "status": "applied",
                        "policy_source": "managed-recommended",
                    }
                ],
                "pattern_policy_evidence": [
                    {
                        "schema": "agentflow.managed_pattern_policy_evidence.v1",
                        "source_surface": "codex_turn",
                        "app_family": "codex",
                        "action_family": "routing",
                        "pattern_family": "routing",
                        "pattern_hash": "sha256:" + "4" * 64,
                        "pattern_hashes": ["sha256:" + "4" * 64],
                        "candidate_id": "candidate-routing",
                        "rule_id": "rule-routing",
                        "policy_source": "managed-recommended",
                        "cohort": "canary_applied",
                        "outcome": "applied",
                        "status_code_bucket": "2xx",
                        "retry_bucket": "none",
                        "latency_bucket": "lt_500ms",
                        "savings_bucket": "lt_0_001_usd",
                        "raw_pattern_strings_included": False,
                        "raw_payload_included": False,
                    }
                ],
            }),
            status=status,
            attempts=attempts,
            next_attempt_at="2026-06-08T10:00:00+00:00",
        )

    def test_managed_feedback_status_cli_reports_metadata_only_queue_counts(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "codex_turn"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.managed_feedback_status.v1")
        self.assertEqual(payload["summary"]["queued"], 1)
        self.assertEqual(payload["summary"]["due"], 1)
        self.assertEqual(payload["pattern_evidence"]["queue_rows"], 1)
        self.assertEqual(payload["pattern_evidence"]["evidence_items"], 1)
        self.assertEqual(
            payload["pattern_evidence"]["endpoint_status_breakdown"][0],
            {
                "endpoint": "/v1/optimization-units/77/outcome",
                "status": "queued",
                "queue_rows": 1,
                "evidence_items": 1,
            },
        )
        self.assertFalse(payload["pattern_evidence"]["payload_json_included"])
        self.assertFalse(payload["due_samples"][0]["payload_included"])
        rendered = stdout.getvalue()
        self.assertNotIn("must stay local", rendered)
        self.assertNotIn("raw codex response secret", rendered)

    def test_managed_feedback_flush_dry_run_does_not_claim_or_send(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                code = cli.managed_feedback_flush_cli(
                    ["--db", db_path, "--source-surface", "codex_turn", "--dry-run"],
                    stdout=stdout,
                )

            store = Store(db_path)
            try:
                row = store.get_managed_outcome_feedback("queue-queued-0")
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls, [])
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["flush"]["would_attempt"], 1)
        self.assertEqual(payload["results"][0]["status"], "would-send")
        self.assertFalse(payload["results"][0]["payload_included"])

    def test_managed_feedback_flush_sends_sanitized_payload_and_updates_queue(self):
        from agentflow_proxy import stats as stats_views
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_feedback_flush_cli(
                        ["--db", db_path, "--source-surface", "codex_turn", "--limit", "1"],
                        stdout=stdout,
                    )

            store = Store(db_path)
            try:
                row = store.get_managed_outcome_feedback("queue-queued-0")
                codex_stats = asyncio.run(stats_views.stats_codex_effectiveness(store, limit=10))
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(row["status"], "sent")
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/optimization-units/77/outcome")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["status"], "success")
        self.assertNotIn("raw_request", sent_payload)
        self.assertNotIn("raw_response", sent_payload)
        rendered = stdout.getvalue()
        self.assertNotIn("must stay local", rendered)
        self.assertNotIn("raw codex response secret", rendered)
        payload = json.loads(rendered)
        self.assertEqual(payload["flush"]["sent"], 1)
        self.assertEqual(payload["after"]["sent"], 1)
        self.assertEqual(codex_stats["summary"]["managed_feedback_queue_sent"], payload["after"]["sent"])

    def test_managed_feedback_flush_records_retryable_error(self):
        from agentflow_proxy.store import Store

        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable"

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS": "3",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_feedback_flush_cli(["--db", db_path], stdout=stdout)

            store = Store(db_path)
            try:
                row = store.get_managed_outcome_feedback("queue-queued-0")
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(row["status"], "retryable-error")
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(row["last_status_code"], 503)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["flush"]["retryable_error"], 1)


if __name__ == "__main__":
    unittest.main()
