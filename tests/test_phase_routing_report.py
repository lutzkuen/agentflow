import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenclaw import cli
from tokenclaw.phase_routing_report import build_phase_routing_dry_run, build_phase_routing_report
from tokenclaw.store import Store, stable_json


class FailingManagedFeedbackClient:
    calls = []

    def __init__(self, *, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        raise RuntimeError("managed unavailable with raw dry-run secret")

    async def patch(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        raise RuntimeError("managed unavailable with raw dry-run secret")


def _log_call(store, suffix, *, requested_model, routed_model, category, routing=None, **overrides):
    routing_json = {
        "category": category,
        "text_chars": overrides.pop("text_chars", 4_000),
        "has_tools": category.startswith("tool"),
        "reason": "keep requested model",
    }
    if routing:
        routing_json.update(routing)
    store.log_call(
        id=f"call-{suffix}",
        created_at=f"2026-06-09T10:00:0{suffix}+00:00",
        path="/v1/messages",
        requested_model=requested_model,
        routed_model=routed_model,
        stream=overrides.pop("stream", 1),
        cache_hit=0,
        status_code=overrides.pop("status_code", 200),
        latency_ms=1200,
        input_tokens_est=overrides.pop("input_tokens_est", 1_000),
        output_tokens_est=overrides.pop("output_tokens_est", 100),
        actual_input_tokens=overrides.pop("actual_input_tokens", 1_000),
        actual_output_tokens=overrides.pop("actual_output_tokens", 100),
        cache_creation_input_tokens=overrides.pop("cache_creation_input_tokens", 0),
        cache_read_input_tokens=overrides.pop("cache_read_input_tokens", 0),
        cost_est_usd=overrides.pop("cost_est_usd", 0.0045),
        cost_baseline_usd=overrides.pop("cost_baseline_usd", 0.0045),
        crunch_json=stable_json({"tokens_saved_est": overrides.pop("tokens_saved_est", 0)}),
        routing_json=stable_json(routing_json),
        cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
        error=overrides.pop("error", None),
        session_id=overrides.pop("session_id", "secret-session-1"),
        category=category,
        retry_count=overrides.pop("retry_count", 0),
        thinking_output_tokens=overrides.pop("thinking_output_tokens", 0),
        provider=overrides.pop("provider", "anthropic"),
    )


class PhaseRoutingReportTests(unittest.TestCase):
    def test_report_groups_phase_opportunity_blockers_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "1",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    tokens_saved_est=12,
                )
                _log_call(
                    store,
                    "2",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                )
                _log_call(
                    store,
                    "3",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    routing={"reason": "keep requested model for thinking request"},
                    thinking_output_tokens=250,
                    session_id="secret-session-2",
                )
                _log_call(
                    store,
                    "4",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="short-completion",
                    routing={"workflow_phase": "summary", "fallback_reason": "rate_limited"},
                    status_code=429,
                    retry_count=1,
                    error="secret prompt text should not be emitted",
                    session_id=None,
                )
                _log_call(
                    store,
                    "5",
                    requested_model="gpt-5-codex",
                    routed_model="gpt-5-codex",
                    category="chat",
                    provider="openai",
                )

                result = build_phase_routing_report(store, limit=50)
            finally:
                store.conn.close()

        self.assertEqual(result["schema"], "agentflow.phase_routing_opportunity.v1")
        self.assertEqual(result["sampled_call_count"], 4)
        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["current_routed_count"], 1)
        self.assertGreater(result["summary"]["projected_savings_usd"], 0)
        self.assertEqual(result["summary"]["unique_session_count"], 2)
        self.assertEqual(result["summary"]["unknown_session_count"], 1)

        opportunities = {
            (row["phase"], row["model_pair"]): row
            for row in result["opportunities"]
        }
        tool = opportunities[("tool-execution", "sonnet_to_haiku")]
        self.assertEqual(tool["sample_count"], 2)
        self.assertEqual(tool["current_routed_count"], 1)
        self.assertEqual(tool["projected_candidate_count"], 1)
        self.assertIn(
            {"value": "already_routed", "count": 1},
            tool["blocked_count_by_reason"],
        )
        self.assertIn(
            {"value": "category", "count": 2},
            tool["phase_source_breakdown"],
        )

        thinking = opportunities[("thinking", "sonnet_to_haiku")]
        self.assertIn({"value": "thinking", "count": 1}, thinking["risk_exclusions"])

        summary = opportunities[("summary", "sonnet_to_haiku")]
        self.assertIn({"value": "historical_error", "count": 1}, summary["risk_exclusions"])
        self.assertIn({"value": "retried", "count": 1}, summary["risk_exclusions"])
        self.assertIn({"value": "fallback", "count": 1}, summary["risk_exclusions"])

        payload = json.dumps(result)
        self.assertNotIn("secret-session", payload)
        self.assertNotIn("secret prompt", payload)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["error_samples_included"])

    def test_cli_reads_seeded_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "1",
                    requested_model="claude-opus-4-5",
                    routed_model="claude-opus-4-5",
                    category="short-completion",
                    routing={"workflow_phase": "summary"},
                    cost_est_usd=0.0075,
                    cost_baseline_usd=0.0075,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.phase_routing_report_cli(["--db", db_path, "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["sampled_call_count"], 1)
        [row] = result["opportunities"]
        self.assertEqual(row["phase"], "summary")
        self.assertEqual(row["model_pair"], "opus_to_sonnet")
        self.assertEqual(row["target_model"], "claude-sonnet-4-6")

    def test_dry_run_simulates_local_yaml_rule_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "routing_rules.yaml"
            policy_path.write_text(
                """
rules:
  - id: local-tool-result-dry-run
    conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: dry-run tool-result phase route
""",
                encoding="utf-8",
            )
            before_policy_text = policy_path.read_text(encoding="utf-8")
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "1",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    stream=0,
                )
                _log_call(
                    store,
                    "2",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    routing={"reason": "keep requested model for thinking request"},
                    thinking_output_tokens=64,
                    stream=0,
                )
                _log_call(
                    store,
                    "3",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    stream=0,
                )
                _log_call(
                    store,
                    "4",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    status_code=500,
                    stream=0,
                )

                stdout = io.StringIO()
                code = cli.phase_routing_report_cli(
                    [
                        "--db",
                        db_path,
                        "--dry-run-policy",
                        str(policy_path),
                        "--pretty",
                        "--stale-hours",
                        "999999",
                    ],
                    stdout=stdout,
                )
            finally:
                store.conn.close()

            self.assertEqual(code, 0)
            self.assertEqual(policy_path.read_text(encoding="utf-8"), before_policy_text)
            payload = json.loads(stdout.getvalue())

        self.assertEqual(payload["schema"], "agentflow.phase_routing_dry_run.v1")
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["wrote_local_files"])
        self.assertFalse(payload["altered_provider_routing"])
        self.assertEqual(payload["policy_source"], "local-manual")
        self.assertEqual(payload["summary"]["matched_count"], 4)
        self.assertEqual(payload["summary"]["projected_candidate_count"], 1)
        self.assertGreater(payload["summary"]["projected_savings_usd"], 0)
        self.assertIn("local-tool-result-dry-run", payload["summary"]["candidate_rule_ids"])
        rule = payload["rules"][0]
        self.assertEqual(rule["matched_count"], 4)
        self.assertEqual(rule["projected_candidate_count"], 1)
        exclusions = {item["reason"]: item["count"] for item in rule["excluded_count_by_reason"]}
        self.assertEqual(exclusions["thinking"], 1)
        self.assertEqual(exclusions["already_routed"], 1)
        self.assertEqual(exclusions["high_error_rate"], 1)
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertFalse(payload["privacy"]["raw_body_columns_read"])
        self.assertNotIn("secret-session", json.dumps(payload))

    def test_dry_run_accepts_managed_bundle_candidate_and_shadow_exclusion(self):
        managed_bundle = {
            "schema": "agentflow.policy_bundle.v1",
            "policies": {
                "routing": {
                    "policy_source": "managed-recommended",
                    "rules": [
                        {
                            "rule_id": "managed-summary-rule",
                            "conditions": {
                                "model_pattern": "sonnet",
                                "workflow_phase": "summary",
                                "text_bucket": "2k_8k_chars",
                            },
                            "action": {"route_to": "haiku", "reason": "managed summary phase route"},
                            "managed_recommendation": {
                                "candidate_id": "phase-candidate-summary",
                                "policy_source": "managed-recommended",
                            },
                        }
                    ],
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "1",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="short-completion",
                    routing={"workflow_phase": "summary", "text_chars": 4000},
                    stream=1,
                )
                _log_call(
                    store,
                    "2",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="short-completion",
                    routing={"workflow_phase": "summary", "text_chars": 4000},
                    stream=0,
                )

                result = build_phase_routing_dry_run(
                    store,
                    managed_bundle,
                    limit=10,
                    stale_hours=999999,
                    require_shadow_support=True,
                )
            finally:
                store.conn.close()

        self.assertEqual(result["schema"], "agentflow.phase_routing_dry_run.v1")
        self.assertEqual(result["policy_source"], "managed-recommended")
        self.assertEqual(result["summary"]["candidate_rule_ids"], ["phase-candidate-summary"])
        self.assertEqual(result["summary"]["matched_count"], 2)
        self.assertEqual(result["summary"]["projected_candidate_count"], 1)
        rule = result["rules"][0]
        self.assertEqual(rule["candidate_id"], "phase-candidate-summary")
        self.assertEqual(rule["policy_source"], "managed-recommended")
        exclusions = {item["reason"]: item["count"] for item in rule["excluded_count_by_reason"]}
        self.assertEqual(exclusions["streaming_shadow_unsupported"], 1)

    def test_dry_run_queues_phase_routing_lifecycle_feedback_metadata_only(self):
        policy_text = """
rules:
  - id: phase-dry-run-rule
    conditions:
      model_pattern: sonnet
      workflow_phase: tool-execution
    action:
      route_to: haiku
      reason: phase dry-run route
"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            policy_path = Path(tmp) / "routing.yaml"
            policy_path.write_text(policy_text, encoding="utf-8")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "1",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    routing={"workflow_phase": "tool-execution", "text_chars": 4000},
                    stream=0,
                    session_id="raw-dry-run-session-secret",
                    cost_est_usd=0.01,
                    cost_baseline_usd=0.01,
                )
            finally:
                store.conn.close()

            FailingManagedFeedbackClient.calls = []
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
                with patch("tokenclaw.recommendations.httpx.AsyncClient", FailingManagedFeedbackClient):
                    code = cli.phase_routing_report_cli(
                        [
                            "--db",
                            db_path,
                            "--dry-run-policy",
                            str(policy_path),
                            "--stale-hours",
                            "999999",
                            "--pretty",
                        ],
                        stdout=stdout,
                    )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "agentflow.phase_routing_dry_run.v1")
            self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "retryable-error")
            self.assertEqual(payload["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
            self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
            self.assertTrue(payload["managed_server_calls_made"])
            self.assertEqual(FailingManagedFeedbackClient.calls[0]["url"], "http://managed.test/v1/policy-events")
            sent = FailingManagedFeedbackClient.calls[0]["json"]
            self.assertEqual(sent["metadata"]["schema"], "agentflow.phase_routing_lifecycle_metadata.v1")
            self.assertEqual(sent["metadata"]["lifecycle_kind"], "phase_routing")
            self.assertEqual(sent["metadata"]["candidate_rule_ids"], ["phase-dry-run-rule"])
            self.assertFalse(sent["metadata"]["privacy"]["raw_prompts_included"])
            rendered = stdout.getvalue() + json.dumps(sent)
            self.assertNotIn("raw-dry-run-session-secret", rendered)
            self.assertNotIn("raw dry-run secret", rendered)

            status_out = io.StringIO()
            cli.managed_feedback_status_cli(
                [
                    "--db",
                    db_path,
                    "--source-surface",
                    "phase_routing_lifecycle",
                    "--pretty",
                ],
                stdout=status_out,
            )
            status = json.loads(status_out.getvalue())
            self.assertEqual(status["source_surface"], "phase_routing_lifecycle")
            self.assertEqual(status["summary"]["retryable_error"], 1)
            self.assertEqual(status["summary"]["due"], 0)
            self.assertFalse(status["privacy"]["payload_json_included"])
