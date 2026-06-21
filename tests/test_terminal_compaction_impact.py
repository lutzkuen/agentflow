import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenclaw import cli
from tokenclaw.store import Store, stable_json
from tokenclaw.terminal_compaction_impact import build_terminal_output_compaction_impact_report


def _terminal_meta(
    *,
    cohort: str,
    rule_id: str = "local-terminal-output-compaction-canary",
    candidate_id: str = "terminal-output-compaction-candidate",
    tokens_saved: int = 1000,
    planned_tokens: int = 1000,
    reason: str | None = None,
) -> dict:
    applied = cohort == "canary_applied"
    return {
        "schema": "tokenclaw.terminal_output_compaction_decision.v1",
        "enabled": True,
        "status": "applied" if applied else "holdout",
        "reason": reason or ("terminal-output-compaction-applied" if applied else "canary_holdout"),
        "changed": applied,
        "applied": applied,
        "policy_source": "local-manual",
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "category": "tool-result",
        "canary": {
            "schema": "tokenclaw.terminal_output_compaction_canary_decision.v1",
            "enabled": True,
            "selected": applied,
            "status": "applied" if applied else "holdout",
            "cohort": cohort,
        },
        "planned_saved_tokens": planned_tokens,
        "tokens_saved_est": tokens_saved if applied else 0,
        "compaction_cost_usd": 0.0,
        "raw_terminal_text_included": False,
        "raw_request_body_included": False,
        "raw_tool_ids_included": False,
        "raw_session_ids_included": False,
    }


def _log_terminal_call(
    store: Store,
    call_id: str,
    *,
    cohort: str,
    status_code: int = 200,
    retry_count: int = 0,
    latency_ms: int = 1000,
    tokens_saved: int = 1000,
    planned_tokens: int = 1000,
    rule_id: str = "local-terminal-output-compaction-canary",
    candidate_id: str = "terminal-output-compaction-candidate",
    created_at: str = "2026-06-12T10:00:00+00:00",
) -> None:
    meta = _terminal_meta(
        cohort=cohort,
        rule_id=rule_id,
        candidate_id=candidate_id,
        tokens_saved=tokens_saved,
        planned_tokens=planned_tokens,
    )
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
        input_tokens_est=10_000,
        output_tokens_est=100,
        actual_input_tokens=10_000,
        actual_output_tokens=100,
        cost_est_usd=0.03,
        cost_baseline_usd=0.033,
        crunch_json=stable_json({
            "changed": cohort == "canary_applied",
            "terminal_output_compaction": meta,
        }),
        routing_json=stable_json({
            "category": "tool-result",
            "workflow_phase": "tool-execution",
            "text_chars": 40_000,
        }),
        cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
        error="raw upstream error must not leak" if status_code >= 400 else None,
        request_json=stable_json({"messages": [{"content": "raw terminal secret must not leak"}]}),
        response_json=stable_json({"text": "raw response secret must not leak"}),
        session_id="raw-session-id-must-not-leak",
        category="tool-result",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        retry_count=retry_count,
        thinking_output_tokens=0,
        provider="anthropic",
        source_surface="anthropic_messages",
        endpoint="messages",
        requested_model_family="sonnet",
        routed_model_family="sonnet",
    )


class TerminalOutputCompactionImpactTests(unittest.TestCase):
    def _with_store(self):
        tmp = TemporaryDirectory()
        store = Store(str(Path(tmp.name) / "tokenclaw.sqlite3"))
        return tmp, store

    def test_impact_report_promotes_positive_applied_vs_holdout_metadata(self):
        tmp, store = self._with_store()
        try:
            _log_terminal_call(store, "promote-a1", cohort="canary_applied", created_at="2026-06-12T10:00:00+00:00")
            _log_terminal_call(store, "promote-a2", cohort="canary_applied", created_at="2026-06-12T10:01:00+00:00")
            _log_terminal_call(store, "promote-h1", cohort="canary_holdout", latency_ms=1200, created_at="2026-06-12T10:02:00+00:00")

            report = build_terminal_output_compaction_impact_report(store, limit=10)
        finally:
            store.conn.close()
            tmp.cleanup()

        self.assertEqual(report["schema"], "tokenclaw.terminal_output_compaction_impact.v1")
        self.assertEqual(report["summary"]["applied_count"], 2)
        self.assertEqual(report["summary"]["holdout_count"], 1)
        self.assertGreater(report["summary"]["net_savings_usd"], 0)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "promote")
        self.assertGreater(candidate["net_savings_usd"], 0)
        self.assertLess(candidate["deltas"]["latency_avg_ms_delta"], 0)
        self.assertEqual(candidate["deltas"]["error_rate_delta"], 0.0)
        self.assertEqual(candidate["deltas"]["retry_rate_delta"], 0.0)

    def test_impact_report_holds_on_latency_or_minimum_savings_gate(self):
        tmp, store = self._with_store()
        try:
            _log_terminal_call(store, "hold-a1", cohort="canary_applied", latency_ms=4000, created_at="2026-06-12T10:00:00+00:00")
            _log_terminal_call(store, "hold-a2", cohort="canary_applied", latency_ms=4200, created_at="2026-06-12T10:01:00+00:00")
            _log_terminal_call(store, "hold-h1", cohort="canary_holdout", latency_ms=500, created_at="2026-06-12T10:02:00+00:00")

            report = build_terminal_output_compaction_impact_report(
                store,
                limit=10,
                max_latency_regression_ms=1000,
                min_net_savings_usd=0.0,
            )
        finally:
            store.conn.close()
            tmp.cleanup()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "hold")
        self.assertIn("hold-latency-regression", candidate["reason_codes"])
        self.assertGreater(candidate["deltas"]["latency_avg_ms_delta"], 1000)

    def test_impact_report_rolls_back_on_error_or_retry_thresholds_with_review_actions(self):
        cases = [
            ("rollback-error-rate-delta", {"status_code": 500, "retry_count": 0}),
            ("rollback-retry-rate-delta", {"status_code": 200, "retry_count": 2}),
        ]
        for expected_reason, applied_kwargs in cases:
            tmp, store = self._with_store()
            try:
                suffix = expected_reason.replace("-", "_")
                _log_terminal_call(store, f"{suffix}-a1", cohort="canary_applied", created_at="2026-06-12T10:00:00+00:00", **applied_kwargs)
                _log_terminal_call(store, f"{suffix}-a2", cohort="canary_applied", created_at="2026-06-12T10:01:00+00:00", **applied_kwargs)
                _log_terminal_call(store, f"{suffix}-h1", cohort="canary_holdout", created_at="2026-06-12T10:02:00+00:00")

                report = build_terminal_output_compaction_impact_report(store, limit=10)
            finally:
                store.conn.close()
                tmp.cleanup()

            candidate = report["candidates"][0]
            self.assertEqual(candidate["verdict"], "rollback")
            self.assertIn(expected_reason, candidate["reason_codes"])
            self.assertEqual(report["summary"]["rollback_action_count"], 1)
            action = report["rollback_actions"][0]
            self.assertTrue(action["review_only"])
            self.assertFalse(action["wrote_local_files"])
            self.assertEqual(
                action["recommended_local_policy_patch"]["terminal_output_compaction"]["canary"]["canary_fraction"],
                0.0,
            )

    def test_impact_report_marks_insufficient_evidence_without_holdout(self):
        tmp, store = self._with_store()
        try:
            _log_terminal_call(store, "insufficient-a1", cohort="canary_applied")

            report = build_terminal_output_compaction_impact_report(store, limit=10)
        finally:
            store.conn.close()
            tmp.cleanup()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "insufficient-evidence")
        self.assertIn("insufficient-applied-samples", candidate["reason_codes"])
        self.assertIn("insufficient-holdout-samples", candidate["reason_codes"])

    def test_impact_report_and_cli_are_content_free(self):
        tmp, store = self._with_store()
        db_path = str(Path(tmp.name) / "tokenclaw.sqlite3")
        try:
            _log_terminal_call(
                store,
                "privacy-a1",
                cohort="canary_applied",
                rule_id="raw-rule-id-must-not-leak",
                candidate_id="raw-candidate-id-must-not-leak",
                created_at="2026-06-12T10:00:00+00:00",
            )
            _log_terminal_call(
                store,
                "privacy-a2",
                cohort="canary_applied",
                rule_id="raw-rule-id-must-not-leak",
                candidate_id="raw-candidate-id-must-not-leak",
                created_at="2026-06-12T10:01:00+00:00",
            )
            _log_terminal_call(
                store,
                "privacy-h1",
                cohort="canary_holdout",
                rule_id="raw-rule-id-must-not-leak",
                candidate_id="raw-candidate-id-must-not-leak",
                created_at="2026-06-12T10:02:00+00:00",
            )
            report = build_terminal_output_compaction_impact_report(store, limit=10)
        finally:
            store.conn.close()

        stdout = io.StringIO()
        code = cli.terminal_output_compaction_impact_cli(["--db", db_path, "--limit", "10"], stdout=stdout)
        tmp.cleanup()

        self.assertEqual(code, 0)
        cli_payload = json.loads(stdout.getvalue())
        self.assertEqual(cli_payload["schema"], "tokenclaw.terminal_output_compaction_impact.v1")
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["raw_terminal_text_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        rendered = json.dumps({"report": report, "cli": cli_payload}, sort_keys=True)
        for forbidden in (
            "raw terminal secret",
            "raw response secret",
            "raw upstream error",
            "raw-session-id-must-not-leak",
            "raw-rule-id-must-not-leak",
            "raw-candidate-id-must-not-leak",
        ):
            self.assertNotIn(forbidden, rendered)
