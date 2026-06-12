import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import agentflow_proxy.crunch as crunch_module
from agentflow_proxy.store import Store, stable_json
from agentflow_proxy.terminal_compaction_dry_run import (
    apply_terminal_output_compaction_plan,
    build_terminal_output_compaction_dry_run,
    plan_terminal_output_compaction,
)
from agentflow_proxy.terminal_compaction_impact import build_terminal_output_compaction_impact_report
from agentflow_proxy.terminal_compaction_report import build_terminal_output_compaction_opportunity_report


RAW_STRINGS = (
    "RAW_TERMINAL_SECRET_FIXTURE",
    "RAW_PROMPT_SECRET_FIXTURE",
    "RAW_RESPONSE_SECRET_FIXTURE",
    "RAW_TOOL_PAYLOAD_SECRET_FIXTURE",
    "raw-tool-use-id-fixture",
    "raw-request-id-fixture",
    "raw-session-id-fixture",
    "raw-cache-key-fixture",
    "raw-policy-secret-fixture",
    "tests/test_fixture_private.py",
    "/workspace/private",
)


def _terminal_text(*, secret: str = "RAW_TERMINAL_SECRET_FIXTURE", include_exit: bool = True) -> str:
    important = [
        "$ pytest tests/test_fixture_private.py",
        "============================= FAILURES =============================",
        f"FAILED tests/test_fixture_private.py::test_hidden - AssertionError: {secret}",
        "Traceback (most recent call last):",
        '  File "/workspace/private/tests/test_fixture_private.py", line 42, in test_hidden',
        "AssertionError: expected ok",
        "modified /workspace/private/app.py",
    ]
    if include_exit:
        important.append("exit code: 1")
    noisy = [
        f"2026-06-12T10:00:{second:02d}Z INFO pid=1234 shard={second} secret={secret}"
        for second in range(70)
    ]
    return "\n".join((important + noisy) * 8)


def _tool_result_body(*, include_exit: bool = True, thinking: bool = False, stream: object = True) -> dict:
    assistant_content = [
        {
            "type": "tool_use",
            "id": "raw-tool-use-id-fixture",
            "name": "Bash",
            "input": {"command": "pytest", "secret": "RAW_TOOL_PAYLOAD_SECRET_FIXTURE"},
        }
    ]
    if thinking:
        assistant_content.insert(0, {"type": "thinking", "thinking": "RAW_PROMPT_SECRET_FIXTURE reasoning"})
    body = {
        "model": "claude-sonnet-4-6",
        "stream": stream,
        "system": "RAW_PROMPT_SECRET_FIXTURE system prompt",
        "messages": [
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "raw-tool-use-id-fixture",
                        "content": [{"type": "text", "text": _terminal_text(include_exit=include_exit)}],
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "recent assistant turn"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "recent-raw-tool-use-id-fixture",
                        "content": [{"type": "text", "text": "RECENT_TERMINAL_OUTPUT_MUST_STAY"}],
                    }
                ],
            },
        ],
    }
    return body


def _write_canary_rules(config: Path, *, fraction: float = 1.0, min_samples: int = 2) -> None:
    (config / "crunch_rules.yaml").write_text(
        f"""
enabled: true
terminal_output_compaction:
  enabled: true
  rule_id: local-terminal-output-compaction-canary
  keep_recent_turns: 2
  min_block_chars: 500
  min_saved_chars: 100
  canary:
    enabled: true
    canary_fraction: {fraction}
    holdout_fraction: 1.0
    canary_salt: terminal-fixture
    canary_unit: request_fingerprint
  safety_stop:
    enabled: true
    min_outcome_samples: {min_samples}
    window: 50
    max_error_rate: 0.5
    max_retry_rate: 1.0
    max_negative_savings_rate: 1.0
""",
        encoding="utf-8",
    )


def _log_call(
    store: Store,
    call_id: str,
    *,
    created_at: str = "2026-06-12T10:00:00+00:00",
    request_json: dict | None = None,
    crunch_meta: dict | None = None,
    status_code: int = 200,
    retry_count: int = 0,
    stream: object = 1,
    text_chars: int = 48_000,
    error: str | None = None,
) -> None:
    store.log_call(
        id=call_id,
        created_at=created_at,
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=stream,
        cache_hit=0,
        status_code=status_code,
        latency_ms=100,
        input_tokens_est=text_chars // 4,
        output_tokens_est=100,
        actual_input_tokens=text_chars // 4,
        actual_output_tokens=100,
        cost_est_usd=0.05,
        cost_baseline_usd=0.05,
        crunch_json=stable_json(crunch_meta or {"changed": False, "tokens_saved_est": 0}),
        routing_json=stable_json({
            "category": "tool-result",
            "workflow_phase": "tool-execution",
            "text_chars": text_chars,
            "has_tools": True,
            "request_id": "raw-request-id-fixture",
            "terminal_log_features": {
                "schema": "agentflow.terminal_log_features.v1",
                "terminal_output_char_fraction_bucket": "gte_75pct",
                "privacy": {"metadata_only": True, "raw_terminal_text_included": False},
            },
        }),
        cache_json=stable_json({"status": "skipped", "cache_key": "raw-cache-key-fixture"}),
        error=error,
        request_json=stable_json(request_json) if request_json is not None else None,
        response_json=stable_json({"text": "RAW_RESPONSE_SECRET_FIXTURE"}),
        session_id="raw-session-id-fixture",
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


def _terminal_meta(*, cohort: str, status: str | None = None, reason: str | None = None) -> dict:
    applied = cohort == "canary_applied"
    return {
        "schema": "agentflow.terminal_output_compaction_decision.v1",
        "enabled": True,
        "status": status or ("applied" if applied else "holdout"),
        "reason": reason or ("terminal-output-compaction-applied" if applied else "canary_holdout"),
        "changed": applied,
        "applied": applied,
        "policy_source": "local-manual",
        "rule_id": "local-terminal-output-compaction-canary",
        "candidate_id": "terminal-output-compaction-candidate",
        "category": "tool-result",
        "canary": {
            "schema": "agentflow.terminal_output_compaction_canary_decision.v1",
            "enabled": True,
            "selected": applied,
            "status": "applied" if applied else "holdout",
            "cohort": cohort,
        },
        "planned_saved_tokens": 1200,
        "tokens_saved_est": 1200 if applied else 0,
        "compaction_cost_usd": 0.0,
        "raw_terminal_text_included": False,
        "raw_request_body_included": False,
        "raw_tool_ids_included": False,
        "raw_session_ids_included": False,
    }


class TerminalOutputCompactionPrivacyFixtureTests(unittest.TestCase):
    ENV_KEYS = ("AGENTFLOW_CRUNCH_RULES", "AGENTFLOW_POLICY_EVENTS_LOG", "HOME")

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(crunch_module)

    def assert_content_free(self, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in RAW_STRINGS:
            self.assertNotIn(forbidden, rendered)

    def test_planner_preserves_protocol_failure_stack_trace_and_missing_exit_metadata(self):
        body = _tool_result_body(include_exit=True)
        plan, meta = plan_terminal_output_compaction(body, keep_recent_turns=2, min_block_chars=500)
        self.assertEqual(meta["status"], "planned")
        self.assertTrue(plan["preservation_flags"]["tool_protocol_ids_preserved"])
        self.assertTrue(plan["preservation_flags"]["failure_lines_preserved"])
        self.assertTrue(plan["preservation_flags"]["stack_traces_preserved"])
        self.assertTrue(plan["preservation_flags"]["exit_status_preserved"])
        planned_body = apply_terminal_output_compaction_plan(body, plan)
        rendered = stable_json(planned_body)
        self.assertIn("raw-tool-use-id-fixture", rendered)
        self.assertIn("FAILED tests/test_fixture_private.py::test_hidden", rendered)
        self.assertIn("Traceback (most recent call last):", rendered)
        self.assertIn("exit code: 1", rendered)

        no_exit_plan, _ = plan_terminal_output_compaction(
            _tool_result_body(include_exit=False),
            keep_recent_turns=2,
            min_block_chars=500,
        )
        target = no_exit_plan["targets"][0]
        self.assertEqual(target["source_evidence_counts"]["exit_status"], 0)
        self.assertEqual(target["preserved_evidence_counts"]["exit_status"], 0)
        self.assertTrue(no_exit_plan["preservation_flags"]["exit_status_preserved"])

    def test_crunch_canary_covers_protocol_mismatch_holdout_thinking_and_safety_stop_privately(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_canary_rules(config, fraction=1.0, min_samples=2)
            os.chdir(tmp_path)
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(tmp_path / "policy_events.jsonl")
            manual = importlib.reload(crunch_module)
            body = _tool_result_body()
            plan, plan_meta = plan_terminal_output_compaction(body, keep_recent_turns=2, min_block_chars=500)
            mismatched_body = dict(plan_meta["planned_body"])
            mismatched_body["stream"] = False

            with patch.object(manual, "plan_terminal_output_compaction", return_value=(plan, {"planned_body": mismatched_body})):
                unchanged, mismatch_meta = manual.crunch_body(body)

            self.assertEqual(unchanged, body)
            self.assertEqual(mismatch_meta["terminal_output_compaction"]["status"], "bypass")
            self.assertEqual(mismatch_meta["terminal_output_compaction"]["reason"], "streaming-protocol-mismatch")

            _write_canary_rules(config, fraction=0.0, min_samples=2)
            manual = importlib.reload(crunch_module)
            holdout, holdout_meta = manual.crunch_body(body)
            self.assertEqual(holdout, body)
            self.assertEqual(holdout_meta["terminal_output_compaction"]["status"], "holdout")
            self.assertEqual(holdout_meta["terminal_output_compaction"]["canary"]["cohort"], "canary_holdout")

            _write_canary_rules(config, fraction=1.0, min_samples=2)
            manual = importlib.reload(crunch_module)
            thinking, thinking_meta = manual.crunch_body(_tool_result_body(thinking=True))
            self.assertEqual(thinking, _tool_result_body(thinking=True))
            self.assertEqual(thinking_meta["terminal_output_compaction"]["reason"], "active-thinking-blocked")

            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                for index in range(2):
                    _log_call(
                        store,
                        f"failed-canary-{index}",
                        created_at=f"2026-06-12T10:0{index}:00+00:00",
                        crunch_meta={"changed": True, "terminal_output_compaction": _terminal_meta(cohort="canary_applied")},
                        status_code=500,
                        error="RAW_TERMINAL_SECRET_FIXTURE upstream failed",
                    )
                stopped, stopped_meta = manual.crunch_body(body, store_obj=store)
            finally:
                store.conn.close()

            self.assertEqual(stopped, body)
            terminal_meta = stopped_meta["terminal_output_compaction"]
            self.assertEqual(terminal_meta["status"], "bypass")
            self.assertEqual(terminal_meta["reason"], "local-canary-safety-stop")
            self.assertEqual(terminal_meta["safety_stop"]["error_count"], 2)
            event_log = Path(os.environ["AGENTFLOW_POLICY_EVENTS_LOG"])
            self.assertTrue(event_log.exists())
            self.assert_content_free({"mismatch": mismatch_meta, "holdout": holdout_meta, "thinking": thinking_meta, "stopped": stopped_meta})
            self.assert_content_free(event_log.read_text(encoding="utf-8"))

    def test_reports_use_metadata_only_projection_and_exclude_raw_fixture_values(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(store, "metadata-only-1", created_at="2026-06-12T10:00:00+00:00", request_json=None, text_chars=48_000)
                _log_call(store, "metadata-only-2", created_at="2026-06-12T10:01:00+00:00", request_json=None, text_chars=48_200)
                _log_call(store, "dry-run-1", created_at="2026-06-12T10:02:00+00:00", request_json=_tool_result_body(), text_chars=48_100)
                _log_call(store, "dry-run-2", created_at="2026-06-12T10:03:00+00:00", request_json=_tool_result_body(), stream="bad-sse-meta", text_chars=48_300)
                for idx, cohort in enumerate(("canary_applied", "canary_applied", "canary_holdout")):
                    _log_call(
                        store,
                        f"impact-{idx}",
                        created_at=f"2026-06-12T10:1{idx}:00+00:00",
                        request_json=_tool_result_body(),
                        crunch_meta={"changed": cohort == "canary_applied", "terminal_output_compaction": _terminal_meta(cohort=cohort)},
                        status_code=200,
                    )

                opportunity = build_terminal_output_compaction_opportunity_report(store, limit=20)
                dry_run = build_terminal_output_compaction_dry_run(store, limit=20, min_block_chars=500)
                impact = build_terminal_output_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        self.assertGreater(opportunity["summary"]["projected_saved_tokens"], 0)
        metadata_only = [row for row in opportunity["candidates"] if row["metadata_only_rows"] > 0]
        self.assertTrue(metadata_only)
        self.assertGreater(dry_run["summary"]["planned_call_count"], 0)
        self.assertFalse(dry_run["privacy"]["raw_request_bodies_included"])
        self.assertFalse(dry_run["privacy"]["raw_responses_included"])
        self.assertFalse(dry_run["privacy"]["cache_keys_included"])
        self.assertEqual(impact["summary"]["applied_count"], 2)
        self.assertEqual(impact["summary"]["holdout_count"], 1)
        self.assertIn(impact["candidates"][0]["verdict"], {"promote", "hold"})
        self.assert_content_free({"opportunity": opportunity, "dry_run": dry_run, "impact": impact})


if __name__ == "__main__":
    unittest.main()
