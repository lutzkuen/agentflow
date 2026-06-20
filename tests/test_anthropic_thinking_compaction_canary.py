from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import tokenclaw.crunch as crunch_module
from tokenclaw.optimization_action_ledger import build_optimization_action_ledger
from tokenclaw.store import Store, stable_json


def _thinking(secret: str, suffix: str = "") -> str:
    return "\n".join(
        f"private reasoning {secret} repeated-token-{index % 19} analysis step {index} {suffix}"
        for index in range(520)
    )


def _tool_result_body(*thinking_texts: str, top_level_thinking: bool = False, redacted: bool = False) -> dict:
    messages: list[dict] = []
    for index, text in enumerate(thinking_texts):
        block = {"type": "redacted_thinking", "data": text} if redacted and index == 0 else {"type": "thinking", "thinking": text}
        messages.append({
            "role": "assistant",
            "content": [
                block,
                {"type": "text", "text": f"assistant fallback {index}"},
            ],
        })
        messages.append({"role": "user", "content": "continue"})
    messages.extend([
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "using tool"},
                {"type": "tool_use", "id": "toolu_private_id", "name": "Read", "input": {"file_path": "/private/file.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_private_id", "content": "private tool payload"},
            ],
        },
    ])
    body = {"model": "claude-sonnet-4-6", "stream": True, "messages": messages}
    if top_level_thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    return body


def _write_rules(
    config: Path,
    *,
    fraction: float,
    holdout: float,
    min_samples: int = 5,
    max_error_rate: float = 0.1,
) -> None:
    (config / "crunch_rules.yaml").write_text(
        f"""
enabled: true
thinking_deduplication:
  enabled: false
anthropic_thinking_history_compaction:
  enabled: true
  rule_id: local-anthropic-thinking-history-compaction-canary
  min_text_chars: 1000
  min_block_chars: 1000
  similarity_threshold: 0.95
  canary:
    enabled: true
    canary_fraction: {fraction}
    holdout_fraction: {holdout}
    canary_salt: thinking-canary-test
    canary_unit: thinking_block_local_fingerprint
  safety_stop:
    enabled: true
    min_outcome_samples: {min_samples}
    window: 50
    max_error_rate: {max_error_rate}
    max_retry_rate: 1.0
    max_negative_savings_rate: 1.0
    max_missing_usage_rate: 1.0
""",
        encoding="utf-8",
    )


def _write_invalid_canary_rules(config: Path) -> None:
    (config / "crunch_rules.yaml").write_text(
        """
enabled: true
thinking_deduplication:
  enabled: false
anthropic_thinking_history_compaction:
  enabled: true
  rule_id: raw-invalid-rule-id
  min_text_chars: 1000
  min_block_chars: 1000
  canary:
    enabled: true
    canary_fraction: raw-invalid-canary-fraction-secret
    holdout_fraction: 0.0
    canary_salt: raw-invalid-canary-salt-secret
""",
        encoding="utf-8",
    )


def _log_prior_canary(
    store: Store,
    *,
    call_id: str = "prior-canary",
    status_code: int = 500,
    retry_count: int = 0,
    applied: bool = True,
) -> None:
    cohort = "canary_applied" if applied else "canary_holdout"
    status = "applied" if applied else "holdout"
    meta = {
        "anthropic_thinking_history_compaction": {
            "schema": "agentflow.anthropic_thinking_history_compaction_decision.v1",
            "enabled": True,
            "status": status,
            "reason": "thinking-history-compaction-applied" if applied else "canary_holdout",
            "applied": applied,
            "rule_id": "local-anthropic-thinking-history-compaction-canary",
            "candidate_id": "anthropic-thinking-compaction:test",
            "tokens_saved_est": 250 if applied else 0,
            "canary": {"cohort": cohort, "selected": applied},
        }
    }
    store.log_call(
        id=call_id,
        created_at="2026-06-13T00:00:00+00:00",
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=100,
        input_tokens_est=1000,
        output_tokens_est=100,
        actual_input_tokens=1000,
        actual_output_tokens=100,
        cost_est_usd=0.01,
        cost_baseline_usd=0.02,
        crunch_json=stable_json(meta),
        routing_json=stable_json({"category": "tool-result", "text_chars": 4000}),
        cache_json=stable_json({"status": "skipped"}),
        retry_count=retry_count,
        provider="anthropic",
        source_surface="anthropic_messages",
        endpoint="messages",
        category="tool-result",
    )


class AnthropicThinkingCompactionCanaryTests(unittest.TestCase):
    ENV_KEYS = ("AGENTFLOW_CRUNCH", "AGENTFLOW_CRUNCH_RULES", "HOME")

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

    def test_default_thinking_compaction_is_disabled_and_forwards_unchanged(self) -> None:
        manual = importlib.reload(crunch_module)
        body = _tool_result_body(_thinking("default"))

        crunched, meta = manual.crunch_body(body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")

        policy = manual.anthropic_thinking_compaction_effective_policy()
        staged = policy["rules"][0]
        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["rule_count"], 1)
        self.assertEqual(staged["rule_id"], "local-repeated-context-thinking-tool-result-canary")
        self.assertEqual(staged["candidate_id"], "repeated-context-thinking-tool-result-gte-128k")
        self.assertEqual(staged["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertEqual(staged["canary"]["fraction"], 0.0)
        self.assertEqual(staged["canary"]["holdout_fraction"], 1.0)
        self.assertFalse(staged["canary"]["salt_included"])
        compaction = meta["anthropic_thinking_history_compaction"]
        self.assertFalse(compaction["enabled"])
        self.assertEqual(compaction["status"], "skipped")
        self.assertEqual(compaction["reason"], "disabled")
        self.assertEqual(crunched, body)

    def test_canary_applies_only_eligible_older_duplicate_thinking_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=1.0, holdout=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            duplicate = _thinking("apply")
            body = _tool_result_body(duplicate, duplicate)

            crunched, meta = manual.crunch_body(body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")

        compaction = meta["anthropic_thinking_history_compaction"]
        self.assertEqual(compaction["status"], "applied")
        self.assertTrue(compaction["applied"])
        self.assertEqual(compaction["canary"]["cohort"], "canary_applied")
        self.assertEqual(compaction["lifecycle_feedback"]["status"], "applied")
        self.assertGreater(compaction["tokens_saved_est"], 0)
        self.assertEqual(compaction["target_count"], 1)
        self.assertEqual(crunched["messages"][0]["content"], [{"type": "text", "text": "assistant fallback 0"}])
        self.assertEqual(crunched["messages"][2]["content"][0]["type"], "thinking")
        self.assertEqual(crunched["messages"][-2]["content"][1]["id"], "toolu_private_id")
        rendered = json.dumps(compaction, sort_keys=True)
        self.assertNotIn("private reasoning", rendered)
        self.assertNotIn("toolu_private_id", rendered)
        self.assertNotIn("/private/file.py", rendered)
        self.assertNotIn("thinking-canary-test", rendered)
        self.assertFalse(compaction["canary"]["salt_included"])
        self.assertNotIn("salt", compaction["canary"])

    def test_holdout_assignment_is_deterministic_and_forwards_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=0.0, holdout=1.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body(_thinking("holdout"), _thinking("holdout"))

            first, first_meta = manual.crunch_body(body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")
            second, second_meta = manual.crunch_body(body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")

        first_compaction = first_meta["anthropic_thinking_history_compaction"]
        second_compaction = second_meta["anthropic_thinking_history_compaction"]
        self.assertEqual(first, body)
        self.assertEqual(second, body)
        self.assertEqual(first_compaction["status"], "holdout")
        self.assertEqual(first_compaction["canary"]["cohort"], "canary_holdout")
        self.assertEqual(first_compaction["canary"], second_compaction["canary"])
        self.assertEqual(first_compaction["lifecycle_feedback"]["status"], "holdout")

    def test_unsafe_active_or_redacted_thinking_passes_through(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=1.0, holdout=0.0)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            duplicate = _thinking("unsafe")
            active_body = _tool_result_body(duplicate, duplicate, top_level_thinking=True)
            redacted_body = _tool_result_body(duplicate, duplicate, redacted=True)

            active, active_meta = manual.crunch_body(active_body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")
            redacted, redacted_meta = manual.crunch_body(redacted_body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")

        self.assertEqual(active, active_body)
        self.assertEqual(active_meta["anthropic_thinking_history_compaction"]["reason"], "active-top-level-thinking-request")
        self.assertEqual(redacted, redacted_body)
        self.assertIn(
            redacted_meta["anthropic_thinking_history_compaction"]["reason"],
            {"redacted-thinking-block", "no-eligible-thinking-history-blocks"},
        )

    def test_invalid_canary_policy_blocks_without_leaking_policy_values(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_invalid_canary_rules(config)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = _tool_result_body(_thinking("invalid-policy"), _thinking("invalid-policy"))

            crunched, meta = manual.crunch_body(body, provider="anthropic", source_surface="anthropic_messages", endpoint="messages")

        compaction = meta["anthropic_thinking_history_compaction"]
        self.assertEqual(crunched, body)
        self.assertEqual(compaction["status"], "bypass")
        self.assertEqual(compaction["reason"], "policy-validation-error")
        self.assertIn("invalid-canary-canary-fraction", compaction["validation_errors"])
        self.assertEqual(compaction["lifecycle_feedback"]["status"], "policy_validation_error")
        rendered = json.dumps(compaction, sort_keys=True)
        self.assertNotIn("raw-invalid-canary-fraction-secret", rendered)
        self.assertNotIn("raw-invalid-canary-salt-secret", rendered)
        self.assertNotIn("private reasoning", rendered)

    def test_safety_stop_disables_further_application_after_failed_canary_sample(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            _write_rules(config, fraction=1.0, holdout=0.0, min_samples=1, max_error_rate=0.1)
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                _log_prior_canary(store, status_code=500)
                body = _tool_result_body(_thinking("stop"), _thinking("stop"))
                stopped, stopped_meta = manual.crunch_body(
                    body,
                    store_obj=store,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                )
            finally:
                store.conn.close()

        compaction = stopped_meta["anthropic_thinking_history_compaction"]
        self.assertEqual(stopped, body)
        self.assertEqual(compaction["status"], "bypass")
        self.assertEqual(compaction["reason"], "local-canary-safety-stop")
        self.assertEqual(compaction["safety_stop_state"], "stopped")
        self.assertEqual(compaction["lifecycle_feedback"]["status"], "safety_stop")

    def test_action_ledger_reports_thinking_compaction_family(self) -> None:
        crunch_meta = {
            "anthropic_thinking_history_compaction": {
                "status": "applied",
                "reason": "thinking-history-compaction-applied",
                "applied": True,
                "policy_source": "local-manual",
                "rule_id": "local-anthropic-thinking-history-compaction-canary",
                "candidate_id": "anthropic-thinking-compaction:test",
                "canary": {"cohort": "canary_applied", "selected": True},
            }
        }

        ledger = build_optimization_action_ledger(
            row={
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "requested_model": "claude-sonnet-4-6",
                "actual_input_tokens": 1000,
            },
            routing_meta={"category": "tool-result", "text_chars": 4000},
            crunch_meta=crunch_meta,
            cache_meta={},
        )

        entries = {entry["family"]: entry for entry in ledger["entries"]}
        self.assertIn("anthropic_thinking_history_compaction", entries)
        self.assertEqual(entries["anthropic_thinking_history_compaction"]["status"], "applied")
        self.assertEqual(entries["anthropic_thinking_history_compaction"]["policy_source"], "local-manual")


if __name__ == "__main__":
    unittest.main()
