import io
import json
import tempfile
import unittest
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.phase_routing_report import build_phase_routing_report
from agentflow_proxy.store import Store, stable_json


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
        stream=1,
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
