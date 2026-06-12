import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agentflow_proxy.crunch as crunch_module
from agentflow_proxy.store import Store, stable_json


def _terminal_text(secret: str) -> str:
    important = [
        "$ pytest tests/test_private_terminal.py",
        "============================= FAILURES =============================",
        f"FAILED tests/test_private_terminal.py::test_hidden - AssertionError: {secret}",
        "Traceback (most recent call last):",
        '  File "/workspace/private/tests/test_private_terminal.py", line 42, in test_hidden',
        "AssertionError: expected ok",
        "exit code: 1",
        "modified agentflow_proxy/terminal_compaction_dry_run.py",
    ]
    noisy = [
        f"2026-06-12T10:00:{second:02d}Z INFO pid=1234 compiling shard={second} secret={secret}"
        for second in range(70)
    ]
    return "\n".join((important + noisy) * 8)


def _tool_result_body(secret: str, *, thinking: bool = False) -> dict:
    assistant_content = [{"type": "tool_use", "id": "toolu_raw_id_must_not_leak", "name": "Bash", "input": {"command": "pytest"}}]
    if thinking:
        assistant_content.insert(0, {"type": "thinking", "thinking": "private reasoning"})
    return {
        "model": "claude-sonnet-4-6",
        "stream": True,
        "messages": [
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_raw_id_must_not_leak",
                        "content": [{"type": "text", "text": _terminal_text(secret)}],
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Recent assistant turn stays intact."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_recent_raw_id",
                        "content": [{"type": "text", "text": "RECENT_TERMINAL_OUTPUT_MUST_STAY"}],
                    }
                ],
            },
        ],
    }


def _write_rules(config: Path, *, fraction: float, min_samples: int = 5, max_error_rate: float = 0.1) -> None:
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
    canary_salt: terminal-canary-test
    canary_unit: request_fingerprint
  safety_stop:
    enabled: true
    min_outcome_samples: {min_samples}
    window: 50
    max_error_rate: {max_error_rate}
    max_retry_rate: 1.0
    max_negative_savings_rate: 1.0
""",
        encoding="utf-8",
    )


def _write_conditional_managed_rules(config: Path, *, fraction: float, stream: str = "true") -> None:
    (config / "crunch_rules.yaml").write_text(
        f"""
enabled: true
terminal_output_compaction:
  enabled: true
  rules:
    - id: managed-terminal-nonmatching-chat
      enabled: true
      policy_source: managed-recommended
      candidate_id: terminal-compaction-nonmatching
      conditions:
        category: chat
      canary:
        enabled: true
        canary_fraction: 1.0
        holdout_fraction: 0.0
        canary_salt: terminal-managed-test
        canary_unit: request_fingerprint
    - id: managed-terminal-output-rule
      enabled: true
      policy_source: managed-recommended
      candidate_id: terminal-compaction-candidate-123
      action_id: terminal-compaction-rollout-action-123
      provenance:
        schema: agentflow.policy_decision_provenance.v1
        issuer: agentflow-server
        server_id: managed-prod
        key_id: managed-key-2026-06
        decision_hash: sha256:manageddecision
        signature: hmac-sha256:managedsignature
        algorithm: hmac-sha256
        verified: true
      conditions:
        source_surface: anthropic_messages
        category: tool-result
        model_pattern: sonnet
        min_text_chars: 1000
        min_saved_tokens: 100
        has_tools: true
        stream: {stream}
      action:
        type: compact_terminal_output
        keep_recent_turns: 2
        min_block_chars: 500
        head_lines: 12
        tail_lines: 16
        max_evidence_lines: 80
        min_saved_chars: 100
      canary:
        enabled: true
        canary_fraction: {fraction}
        holdout_fraction: 1.0
        canary_salt: terminal-managed-test
        canary_unit: request_fingerprint
      safety_stop:
        enabled: true
        min_outcome_samples: 5
        window: 50
        max_error_rate: 0.5
        max_retry_rate: 1.0
        max_negative_savings_rate: 1.0
""",
        encoding="utf-8",
    )


def _write_conditional_policy_report_rules(config: Path) -> None:
    (config / "crunch_rules.yaml").write_text(
        """
enabled: true
terminal_output_compaction:
  enabled: true
  rules:
    - id: managed-terminal-policy-report
      enabled: true
      policy_source: managed-recommended
      candidate_id: terminal-compaction-candidate-report
      action_id: terminal-compaction-action-report
      conditions:
        source_surface: anthropic_messages
        app_family: claude_code
        phase: tool-result
        category: tool-result
        text_bucket: 32k_128k_chars
        labels:
          - terminal-output
          - plateau-session
        expected_saved_tokens_bucket: 1k_2k_tokens
        model_pattern: sonnet
        has_tools: true
        stream: true
        uses_thinking: false
      action:
        type: compact_terminal_output
        keep_recent_turns: 3
        min_block_chars: 700
        head_lines: 9
        tail_lines: 11
        max_evidence_lines: 55
        min_saved_chars: 250
        preserve_diagnostics: true
        preserve_tool_protocol: true
      canary:
        enabled: true
        canary_fraction: 0.25
        holdout_fraction: 0.50
        canary_salt: private-local-salt
        canary_unit: request_fingerprint
      safety_stop:
        enabled: true
        min_outcome_samples: 7
        window: 77
        max_error_rate: 0.2
        max_retry_rate: 0.3
        max_negative_savings_rate: 0.4
        max_error_rate_delta: 0.05
""",
        encoding="utf-8",
    )


class TerminalOutputCompactionCanaryTests(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CRUNCH",
        "AGENTFLOW_CRUNCH_RULES",
        "AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT",
        "AGENTFLOW_POLICY_EVENTS_LOG",
        "HOME",
    )

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

    def test_default_terminal_output_compaction_is_disabled_and_forwards_unchanged(self):
        manual = importlib.reload(crunch_module)
        body = _tool_result_body("RAW_DEFAULT_SECRET")

        crunched, meta = manual.crunch_body(body)

        self.assertEqual(crunched, body)
        terminal_meta = meta["terminal_output_compaction"]
        self.assertFalse(terminal_meta["enabled"])
        self.assertEqual(terminal_meta["status"], "skipped")
        self.assertEqual(terminal_meta["reason"], "disabled")
        policy = manual.terminal_output_compaction_effective_policy()
        self.assertEqual(policy["rule_count"], 0)
        self.assertEqual(policy["rules"], [])

    def test_conditional_terminal_output_compaction_policy_reports_sanitized_rules(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_conditional_policy_report_rules(config)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            policy = manual.terminal_output_compaction_effective_policy()

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["rule_count"], 1)
        rule = policy["rules"][0]
        self.assertEqual(rule["rule_id"], "managed-terminal-policy-report")
        self.assertEqual(rule["candidate_id"], "terminal-compaction-candidate-report")
        self.assertEqual(rule["action_id"], "terminal-compaction-action-report")
        self.assertEqual(rule["policy_source"], "managed-recommended")
        self.assertEqual(rule["conditions"]["workflow_phase"], "tool-result")
        self.assertEqual(rule["conditions"]["labels"], ["terminal-output", "plateau-session"])
        self.assertEqual(rule["conditions"]["expected_saved_token_bucket"], "1k_2k_tokens")
        self.assertEqual(rule["action"]["keep_recent_turns"], 3)
        self.assertEqual(rule["action"]["min_block_chars"], 700)
        self.assertEqual(rule["action"]["head_lines"], 9)
        self.assertEqual(rule["action"]["tail_lines"], 11)
        self.assertEqual(rule["action"]["max_evidence_lines"], 55)
        self.assertEqual(rule["action"]["min_saved_chars"], 250)
        self.assertTrue(rule["action"]["preserve_diagnostics"])
        self.assertEqual(rule["canary"]["fraction"], 0.25)
        self.assertEqual(rule["canary"]["holdout_fraction"], 0.5)
        self.assertEqual(rule["safety_stop"]["min_outcome_samples"], 7)
        self.assertEqual(rule["safety_stop"]["window"], 77)
        self.assertEqual(rule["safety_stop"]["max_error_rate"], 0.2)

        rendered = json.dumps(policy, sort_keys=True)
        self.assertNotIn("RAW_", rendered)
        self.assertNotIn("toolu_", rendered)
        self.assertNotIn("private-local-salt", rendered)
        self.assertTrue(rule["canary"]["salt_configured"])
        self.assertFalse(policy["raw_terminal_text_included"])
        self.assertFalse(policy["policy_file_contents_included"])

    def test_yaml_canary_fraction_applies_terminal_output_compaction_with_private_metadata(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=1.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body("RAW_APPLIED_SECRET")

            crunched, meta = manual.crunch_body(body)

        terminal_meta = meta["terminal_output_compaction"]
        self.assertTrue(meta["changed"])
        self.assertTrue(terminal_meta["applied"])
        self.assertEqual(terminal_meta["status"], "applied")
        self.assertEqual(terminal_meta["canary"]["cohort"], "canary_applied")
        self.assertGreater(terminal_meta["tokens_saved_est"], 0)
        self.assertIn("toolu_raw_id_must_not_leak", stable_json(crunched))
        self.assertIn("RECENT_TERMINAL_OUTPUT_MUST_STAY", stable_json(crunched))
        self.assertEqual(crunched.get("stream"), body.get("stream"))

        rendered_meta = json.dumps(terminal_meta, sort_keys=True)
        for forbidden in (
            "RAW_APPLIED_SECRET",
            "toolu_raw_id_must_not_leak",
            "toolu_recent_raw_id",
            "tests/test_private_terminal.py",
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered_meta)
        self.assertFalse(terminal_meta["raw_terminal_text_included"])
        self.assertFalse(terminal_meta["raw_tool_ids_included"])

    def test_managed_conditional_rule_applies_with_canary_and_provenance(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_conditional_managed_rules(config, fraction=1.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body("RAW_MANAGED_APPLIED_SECRET")

            crunched, meta = manual.crunch_body(body)

        terminal_meta = meta["terminal_output_compaction"]
        self.assertTrue(meta["changed"])
        self.assertEqual(crunched.get("stream"), body.get("stream"))
        self.assertEqual(terminal_meta["status"], "applied")
        self.assertEqual(terminal_meta["rule_id"], "managed-terminal-output-rule")
        self.assertEqual(terminal_meta["candidate_id"], "terminal-compaction-candidate-123")
        self.assertEqual(terminal_meta["action_id"], "terminal-compaction-rollout-action-123")
        self.assertEqual(terminal_meta["policy_source"], "managed-recommended")
        self.assertEqual(terminal_meta["canary"]["cohort"], "canary_applied")
        self.assertEqual(terminal_meta["provenance"]["issuer"], "agentflow-server")
        self.assertEqual(terminal_meta["provenance"]["decision_hash"], "sha256:manageddecision")
        self.assertEqual(terminal_meta["configured_rule_count"], 2)
        self.assertEqual(terminal_meta["evaluated_rules"][0]["status"], "skipped")
        self.assertEqual(terminal_meta["evaluated_rules"][1]["status"], "matched")
        self.assertGreater(terminal_meta["tokens_saved_est"], 0)

        rendered_meta = json.dumps(terminal_meta, sort_keys=True)
        for forbidden in (
            "RAW_MANAGED_APPLIED_SECRET",
            "toolu_raw_id_must_not_leak",
            "toolu_recent_raw_id",
            "tests/test_private_terminal.py",
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered_meta)
        self.assertFalse(terminal_meta["raw_terminal_text_included"])
        self.assertFalse(terminal_meta["raw_tool_ids_included"])

    def test_managed_conditional_rule_holdout_and_nonmatching_requests_forward_unchanged(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_conditional_managed_rules(config, fraction=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body("RAW_MANAGED_HOLDOUT_SECRET")

            holdout, holdout_meta = manual.crunch_body(body)
            second_holdout, second_meta = manual.crunch_body(body)

            _write_conditional_managed_rules(config, fraction=1.0, stream="false")
            manual = importlib.reload(crunch_module)
            nonmatching, nonmatching_meta = manual.crunch_body(body)

        self.assertEqual(holdout, body)
        self.assertEqual(second_holdout, body)
        terminal_meta = holdout_meta["terminal_output_compaction"]
        self.assertEqual(terminal_meta["status"], "holdout")
        self.assertEqual(terminal_meta["rule_id"], "managed-terminal-output-rule")
        self.assertEqual(terminal_meta["policy_source"], "managed-recommended")
        self.assertEqual(terminal_meta["canary"]["cohort"], "canary_holdout")
        self.assertEqual(terminal_meta["canary"], second_meta["terminal_output_compaction"]["canary"])
        self.assertGreater(terminal_meta["planned_saved_tokens"], 0)

        self.assertEqual(nonmatching, body)
        nonmatching_terminal = nonmatching_meta["terminal_output_compaction"]
        self.assertEqual(nonmatching_terminal["status"], "skipped")
        self.assertEqual(nonmatching_terminal["reason"], "no-conditional-rule-matched")
        self.assertEqual(nonmatching_terminal["configured_rule_count"], 2)

    def test_yaml_holdout_cohort_records_metadata_without_mutating_request(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body("RAW_HOLDOUT_SECRET")

            crunched_1, meta_1 = manual.crunch_body(body)
            crunched_2, meta_2 = manual.crunch_body(body)

        self.assertEqual(crunched_1, body)
        self.assertEqual(crunched_2, body)
        terminal_meta = meta_1["terminal_output_compaction"]
        self.assertEqual(terminal_meta["status"], "holdout")
        self.assertEqual(terminal_meta["reason"], "canary_holdout")
        self.assertEqual(terminal_meta["canary"]["cohort"], "canary_holdout")
        self.assertEqual(terminal_meta["canary"], meta_2["terminal_output_compaction"]["canary"])
        self.assertGreater(terminal_meta["planned_saved_tokens"], 0)

    def test_thinking_blocker_forwards_unchanged(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=1.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body("RAW_THINKING_SECRET", thinking=True)

            crunched, meta = manual.crunch_body(body)

        self.assertEqual(crunched, body)
        terminal_meta = meta["terminal_output_compaction"]
        self.assertEqual(terminal_meta["status"], "skipped")
        self.assertEqual(terminal_meta["reason"], "active-thinking-blocked")

    def test_safety_stop_disables_further_application_after_failed_canary_samples(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=1.0, min_samples=2, max_error_rate=0.5)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            body = _tool_result_body("RAW_SAFETY_SECRET")

            healthy, healthy_meta = manual.crunch_body(body, store_obj=store)
            self.assertNotEqual(healthy, body)
            self.assertEqual(healthy_meta["terminal_output_compaction"]["status"], "applied")

            for index in range(2):
                store.log_call(
                    id=f"terminal-failed-canary-{index}",
                    created_at=f"2026-06-12T10:0{index}:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=500,
                    latency_ms=100,
                    input_tokens_est=1000,
                    output_tokens_est=0,
                    actual_input_tokens=1000,
                    actual_output_tokens=0,
                    cost_est_usd=0.0,
                    cost_baseline_usd=0.0,
                    crunch_json=stable_json({
                        "changed": True,
                        "terminal_output_compaction": {
                            "schema": "agentflow.terminal_output_compaction_decision.v1",
                            "rule_id": "local-terminal-output-compaction-canary",
                            "status": "applied",
                            "applied": True,
                            "tokens_saved_est": 1200,
                            "canary": {
                                "schema": "agentflow.terminal_output_compaction_canary_decision.v1",
                                "enabled": True,
                                "selected": True,
                                "status": "applied",
                                "cohort": "canary_applied",
                            },
                        },
                    }),
                    routing_json=stable_json({"category": "tool-result"}),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                    error="upstream failed",
                    request_json=None,
                    response_json=None,
                    session_id="terminal-safety-stop",
                    category="tool-result",
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    retry_count=0,
                    provider="anthropic",
                )

            stopped, stopped_meta = manual.crunch_body(body, store_obj=store)
            store.conn.close()

        terminal_meta = stopped_meta["terminal_output_compaction"]
        self.assertEqual(stopped, body)
        self.assertFalse(stopped_meta["changed"])
        self.assertEqual(terminal_meta["status"], "bypass")
        self.assertEqual(terminal_meta["reason"], "local-canary-safety-stop")
        self.assertEqual(terminal_meta["safety_stop"]["sample_count"], 2)
        self.assertEqual(terminal_meta["safety_stop"]["error_count"], 2)
