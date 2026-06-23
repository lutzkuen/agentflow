from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from tokenclaw import cli
import tokenclaw.crunch as crunch_module
from tokenclaw.local_compaction_canary_ramp import (
    build_local_compaction_canary_ramp,
)
from tokenclaw.store import Store, stable_json


RAW_SECRET = "raw-local-ramp-secret"


def _rules_yaml(*, fraction: float = 0.0, holdout: float = 1.0) -> str:
    return f"""
enabled: true
old_context_summarization:
  enabled: true
  rule_id: local-old-context-summarization
  candidate_id: local-old-context-summarization
  canary:
    enabled: true
    fraction: 0.0
    salt: old-context-ramp-test
anthropic_thinking_history_compaction:
  enabled: {str(fraction > 0).lower()}
  policy_source: local-manual
  rule_id: local-anthropic-thinking-history-compaction-canary
  min_text_chars: 8000
  min_block_chars: 2000
  similarity_threshold: 0.95
  canary:
    enabled: true
    canary_fraction: {fraction}
    holdout_fraction: {holdout}
    canary_salt: thinking-ramp-test
    canary_unit: thinking_block_local_fingerprint
  safety_stop:
    enabled: true
    min_outcome_samples: 1
    window: 50
    max_error_rate: 0.1
    max_retry_rate: 0.25
    max_negative_savings_rate: 0.25
    max_missing_usage_rate: 1.0
    max_error_rate_delta: 0.05
  rules:
    - id: local-repeated-context-thinking-tool-result-canary
      enabled: {str(fraction > 0).lower()}
      policy_source: local-manual
      candidate_id: repeated-context-thinking-tool-result-gte-128k
      conditions:
        source_surface: anthropic_messages
        category: tool-result
        workflow_phase: tool-result
        text_bucket: gte_128k_chars
        model_pattern: sonnet
        has_tools: true
        stream: true
        min_text_chars: 128000
      action:
        type: compact_thinking_history_block
        min_text_chars: 128000
        min_block_chars: 2000
        similarity_threshold: 0.95
        preserve_tool_protocol: true
        preserve_assistant_text_fallback: true
      block_top_level_thinking: true
      canary:
        enabled: true
        canary_fraction: {fraction}
        holdout_fraction: {holdout}
        canary_salt: thinking-ramp-test
        canary_unit: thinking_block_local_fingerprint
      safety_stop:
        enabled: true
        min_outcome_samples: 1
        window: 50
        max_error_rate: 0.1
        max_retry_rate: 0.25
        max_negative_savings_rate: 0.25
        max_missing_usage_rate: 1.0
        max_error_rate_delta: 0.05
"""


def _thinking_text(label: str, lines: int = 1400) -> str:
    return "\n".join(
        f"private reasoning {RAW_SECRET} {label} repeated-token-{index % 23} analysis step {index}"
        for index in range(lines)
    )


def _large_tool_result_body() -> dict:
    duplicate = _thinking_text("duplicate")
    return {
        "model": "claude-sonnet-4-6",
        "stream": True,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": duplicate},
                    {"type": "text", "text": "assistant fallback 0"},
                ],
            },
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": duplicate},
                    {"type": "text", "text": "assistant fallback 1"},
                ],
            },
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "using tool"},
                    {"type": "tool_use", "id": "toolu_ramp_test", "name": "Read", "input": {"file_path": "/private/ramp.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_ramp_test", "content": "private tool payload"},
                ],
            },
        ],
    }


def _thinking_crunch_meta(
    *,
    applied: bool,
    cohort: str,
    tokens_saved: int = 1000,
    output_similarity: float | None = None,
) -> dict:
    status = "applied" if applied else "holdout"
    meta: dict = {
        "enabled": True,
        "status": status,
        "reason": "thinking-history-compaction-applied" if applied else "canary_holdout",
        "changed": applied,
        "applied": applied,
        "policy_source": "local-manual",
        "rule_id": "local-repeated-context-thinking-tool-result-canary",
        "candidate_id": "repeated-context-thinking-tool-result-gte-128k",
        "category": "tool-result",
        "before_chars": 200000,
        "saved_chars": tokens_saved * 4 if applied else 0,
        "planned_saved_chars": tokens_saved * 4,
        "tokens_saved_est": tokens_saved if applied else 0,
        "planned_saved_tokens": tokens_saved,
        "canary": {"cohort": cohort, "selected": applied},
        "lifecycle_feedback": {"status": status, "cohort": cohort, "metadata_only": True},
    }
    if output_similarity is not None:
        meta["quality"] = {"output_similarity": output_similarity}
    return {
        "enabled": True,
        "changed": applied,
        "tokens_saved_est": tokens_saved if applied else 0,
        "realized_savings": {
            "schema": "tokenclaw.realized_crunch_savings.v1",
            "realized_crunch_savings_usd": 0.003 if applied else 0.0,
        },
        "anthropic_thinking_history_compaction": meta,
    }


def _old_context_crunch_meta(*, applied: bool, cohort: str, net_savings: float = 0.002) -> dict:
    status = "applied" if applied else "holdout"
    return {
        "enabled": True,
        "changed": applied,
        "tokens_saved_est": 700 if applied else 0,
        "realized_savings": {
            "schema": "tokenclaw.realized_crunch_savings.v1",
            "realized_crunch_savings_usd": net_savings if applied else 0.0,
        },
        "old_context_summarization": {
            "enabled": True,
            "status": status,
            "reason": "old-context-summary-applied" if applied else "canary_holdout",
            "rule_id": "local-old-context-summarization",
            "candidate_id": "local-old-context-summarization",
            "tokens_saved_est": 700 if applied else 0,
            "estimated_net_savings_usd": net_savings if applied else 0.003,
            "canary": {"cohort": cohort, "selected": applied},
        },
    }


def _log_call(
    store: Store,
    call_id: str,
    *,
    crunch_json: dict,
    status_code: int = 200,
    retry_count: int = 0,
    cost_est: float = 0.01,
) -> None:
    store.log_call(
        id=call_id,
        created_at=f"2026-06-23T10:0{len(call_id) % 9}:00+00:00",
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=1000,
        input_tokens_est=6000,
        output_tokens_est=300,
        actual_input_tokens=5000,
        actual_output_tokens=300,
        cost_est_usd=cost_est,
        cost_baseline_usd=cost_est + 0.01,
        crunch_json=stable_json(crunch_json),
        routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-result", "text_chars": 200000}),
        cache_json=stable_json({"status": "skipped"}),
        retry_count=retry_count,
        provider="anthropic",
        source_surface="anthropic_messages",
        endpoint="messages",
        category="tool-result",
        requested_model_family="sonnet",
        routed_model_family="sonnet",
    )


class LocalCompactionCanaryRampTests(unittest.TestCase):
    ENV_KEYS = ("TOKENCLAW_CRUNCH", "TOKENCLAW_CRUNCH_RULES", "HOME")

    def setUp(self) -> None:
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(crunch_module)

    def test_positive_realized_delta_ramps_fractions_across_cycles(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.05, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _log_call(store, "thinking-applied-a", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied"), cost_est=0.010)
                _log_call(store, "thinking-applied-b", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied"), cost_est=0.012)
                _log_call(store, "thinking-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout"), cost_est=0.030)
                _log_call(store, "summary-applied-a", crunch_json=_old_context_crunch_meta(applied=True, cohort="canary_applied"), cost_est=0.008)
                _log_call(store, "summary-applied-b", crunch_json=_old_context_crunch_meta(applied=True, cohort="canary_applied"), cost_est=0.009)
                _log_call(store, "summary-holdout", crunch_json=_old_context_crunch_meta(applied=False, cohort="canary_holdout"), cost_est=0.020)

                first = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    ramp_step=0.05,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now="2026-06-23T10:10:00+00:00",
                )
                second = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    ramp_step=0.05,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now="2026-06-23T10:20:00+00:00",
                )
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        self.assertEqual(first["status"], "applied")
        self.assertEqual(second["status"], "applied")
        thinking_rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertAlmostEqual(thinking_rule["canary"]["canary_fraction"], 0.15)
        self.assertTrue(data["anthropic_thinking_history_compaction"]["enabled"])
        self.assertEqual(thinking_rule["ramp_controller"]["decision"], "widen")
        self.assertIn("realized-canary-advantage", thinking_rule["ramp_controller"]["reason_codes"])
        self.assertAlmostEqual(data["old_context_summarization"]["canary"]["fraction"], 0.10)
        rendered = json.dumps(second, sort_keys=True)
        self.assertNotIn(RAW_SECRET, rendered)
        self.assertFalse(second["privacy"]["raw_request_bodies_included"])
        self.assertFalse(second["provider_calls_made"])
        self.assertFalse(second["managed_server_calls_made"])

    def test_regression_safety_stop_ramps_to_zero_and_records_reason(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.10, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _log_call(store, "thinking-failed", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied"), status_code=500)
                _log_call(store, "thinking-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout"), status_code=200)
                result = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    min_applied_samples=1,
                    min_holdout_samples=1,
                    now="2026-06-23T11:00:00+00:00",
                )
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "stop")
        self.assertEqual(decision["recommended_fraction"], 0.0)
        section = data["anthropic_thinking_history_compaction"]
        rule = section["rules"][0]
        self.assertFalse(section["enabled"])
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["canary"]["canary_fraction"], 0.0)
        self.assertIn("applied-error-rate-above-threshold", rule["ramp_controller"]["reason_codes"])
        self.assertEqual(rule["safety_stop"]["last_ramp_stop_reason"], "applied-error-rate-above-threshold")

    def test_similarity_floor_breach_halts_ramp(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.10, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-low-similarity",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", output_similarity=0.91),
                    cost_est=0.010,
                )
                _log_call(store, "thinking-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout"), cost_est=0.030)
                result = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    min_applied_samples=1,
                    min_holdout_samples=1,
                    similarity_floor=0.98,
                    now="2026-06-23T11:30:00+00:00",
                )
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "stop")
        self.assertIn("output-similarity-floor-breach", decision["reason_codes"])
        self.assertEqual(decision["recommended_fraction"], 0.0)

    def test_controller_enabled_gte_128k_rule_can_apply_to_live_crunch_json(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.0, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _log_call(store, "thinking-applied", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied"), cost_est=0.010)
                _log_call(store, "thinking-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout"), cost_est=0.030)
                ramp = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    initial_fraction=1.0,
                    ramp_step=1.0,
                    max_fraction=1.0,
                    min_applied_samples=1,
                    min_holdout_samples=1,
                    now="2026-06-23T12:00:00+00:00",
                )
            finally:
                store.conn.close()

            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _large_tool_result_body()
            original = copy.deepcopy(body)
            crunched, meta = manual.crunch_body(
                body,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
            )

        self.assertEqual(ramp["status"], "applied")
        compaction = meta["anthropic_thinking_history_compaction"]
        self.assertEqual(compaction["candidate_id"], "repeated-context-thinking-tool-result-gte-128k")
        self.assertEqual(compaction["status"], "applied")
        self.assertTrue(compaction["applied"])
        self.assertEqual(compaction["canary"]["cohort"], "canary_applied")
        self.assertGreater(compaction["tokens_saved_est"], 0)
        self.assertLess(len(stable_json(crunched)), len(stable_json(original)))
        rendered = json.dumps(compaction, sort_keys=True)
        self.assertNotIn(RAW_SECRET, rendered)
        self.assertNotIn("toolu_ramp_test", rendered)

    def test_internal_cli_exposes_local_compaction_canary_ramp(self) -> None:
        class Buffer:
            def __init__(self) -> None:
                self.value = ""

            def write(self, text: str) -> None:
                self.value += text

        stdout = Buffer()
        code = cli.internal_cli(["--list"], stdout=stdout, stderr=Buffer())

        self.assertEqual(code, 0)
        self.assertIn("local-compaction-canary-ramp", stdout.value.splitlines())


if __name__ == "__main__":
    unittest.main()
