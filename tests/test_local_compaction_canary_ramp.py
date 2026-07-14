from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import yaml

from tokenclaw import cli
import tokenclaw.crunch as crunch_module
from tokenclaw.local_compaction_canary_ramp import (
    build_local_compaction_canary_ramp,
)
from tokenclaw.store import Store, stable_json


RAW_SECRET = "raw-local-ramp-secret"


def _rules_yaml(*, fraction: float = 0.0, holdout: float = 1.0, manual_disabled: bool = False) -> str:
    enabled = fraction > 0 and not manual_disabled
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
  enabled: {str(enabled).lower()}
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
      enabled: {str(enabled).lower()}
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
    realized_savings: float = 0.003,
    output_similarity: float | None = None,
    fallback: bool = False,
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
        "realized_crunch_savings_usd": realized_savings if applied else 0.0,
        "canary": {"cohort": cohort, "selected": applied},
        "lifecycle_feedback": {"status": status, "cohort": cohort, "metadata_only": True},
    }
    if fallback:
        meta["fallback_reason"] = "rate_limited"
    if output_similarity is not None:
        meta["quality"] = {"output_similarity": output_similarity}
    return {
        "enabled": True,
        "changed": applied,
        "tokens_saved_est": tokens_saved if applied else 0,
        "realized_savings": {
            "schema": "tokenclaw.realized_crunch_savings.v1",
            "realized_crunch_savings_usd": realized_savings if applied else 0.0,
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
    actual_input_tokens: int = 5000,
    actual_output_tokens: int = 300,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
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
        actual_input_tokens=actual_input_tokens,
        actual_output_tokens=actual_output_tokens,
        cost_est_usd=cost_est,
        cost_baseline_usd=cost_est + 0.01,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
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


def _thinking_feedback_payload() -> dict:
    return {
        "schema": "tokenclaw.local_compaction_rollback_feedback.v1",
        "event_type": "crunch_rollback",
        "source_surface": "anthropic_messages",
        "local_action_family": "crunch",
        "candidate_id": "thinking-tail-compaction",
        "rule_id": "local-repeated-context-thinking-tool-result-canary",
        "reason_codes": ["test-feedback"],
        "privacy": {
            "metadata_only": True,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_thinking_text_included": False,
        },
    }


def _enqueue_thinking_feedback(store: Store, *, queue_id: str = "thinking-feedback-queued") -> None:
    store.enqueue_managed_outcome_feedback(
        id=queue_id,
        created_at="2026-06-23T09:55:00+00:00",
        updated_at="2026-06-23T09:55:00+00:00",
        source_surface="anthropic_thinking_compaction_rollback",
        endpoint="/v1/policy-events",
        optimization_unit_id=0,
        payload_json=stable_json(_thinking_feedback_payload()),
        status="queued",
        attempts=0,
        next_attempt_at="2026-06-23T09:55:00+00:00",
    )


class ManagedFeedbackRampResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class ManagedFeedbackRampClient:
    calls: list[dict[str, object]] = []
    status_code = 200
    text = '{"ok":true}'

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, **kwargs):
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return ManagedFeedbackRampResponse(self.status_code, self.text)

    async def patch(self, url: str, **kwargs):
        self.calls.append({"method": "PATCH", "url": url, **kwargs})
        return ManagedFeedbackRampResponse(self.status_code, self.text)


class LocalCompactionCanaryRampTests(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_CRUNCH",
        "TOKENCLAW_CRUNCH_RULES",
        "HOME",
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
        "TOKENCLAW_MANAGED_CRUNCH",
        "TOKENCLAW_LOCAL_RULES_ONLY",
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS",
    )

    def setUp(self) -> None:
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        ManagedFeedbackRampClient.calls = []
        ManagedFeedbackRampClient.status_code = 200
        ManagedFeedbackRampClient.text = '{"ok":true}'
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

    def test_positive_net_after_prompt_cache_churn_widens_thinking_tail(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.05, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-positive-applied-a",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050),
                    cache_creation_input_tokens=1000,
                    cache_read_input_tokens=0,
                )
                _log_call(
                    store,
                    "thinking-positive-applied-b",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050),
                    cache_creation_input_tokens=1000,
                    cache_read_input_tokens=0,
                )
                _log_call(
                    store,
                    "thinking-positive-holdout",
                    crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout", tokens_saved=2000),
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=100,
                )
                result = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    ramp_step=0.05,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now="2026-06-23T10:30:00+00:00",
                )
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "widen")
        self.assertIn("realized-canary-advantage", decision["reason_codes"])
        self.assertGreater(decision["evidence"]["net_savings_after_prompt_cache_churn_usd"], 0.0)
        self.assertGreater(decision["evidence"]["prompt_cache_churn_usd"], 0.0)
        self.assertAlmostEqual(data["anthropic_thinking_history_compaction"]["rules"][0]["canary"]["canary_fraction"], 0.10)

    def test_pre_widen_drains_queued_thinking_tail_feedback_and_exposes_freshness(self) -> None:
        from tokenclaw import stats as stats_views

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.05, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _enqueue_thinking_feedback(store)
                _log_call(
                    store,
                    "thinking-drain-applied-a",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050),
                    cache_creation_input_tokens=1000,
                )
                _log_call(
                    store,
                    "thinking-drain-applied-b",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050),
                    cache_creation_input_tokens=1000,
                )
                _log_call(
                    store,
                    "thinking-drain-holdout",
                    crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout", tokens_saved=2000),
                    cache_read_input_tokens=100,
                )
                with patch.dict(
                    os.environ,
                    {
                        "TOKENCLAW_MANAGED": "1",
                        "TOKENCLAW_MANAGED_MODE": "live",
                        "TOKENCLAW_MANAGED_CRUNCH": "1",
                        "TOKENCLAW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    },
                    clear=False,
                ), patch("tokenclaw.http_client.httpx.AsyncClient", ManagedFeedbackRampClient):
                    result = build_local_compaction_canary_ramp(
                        store,
                        config_dir=config,
                        apply=True,
                        ramp_step=0.05,
                        min_applied_samples=2,
                        min_holdout_samples=1,
                        now="2026-06-23T10:30:00+00:00",
                    )
                row = store.get_managed_outcome_feedback("thinking-feedback-queued")
                freshness = stats_views.stats_managed_feedback_queue_freshness(store)
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "widen")
        self.assertEqual(decision["managed_feedback_pre_widen_gate"]["status"], "passed")
        self.assertEqual(decision["managed_feedback_pre_widen_gate"]["before"]["pending_count"], 1)
        self.assertEqual(decision["managed_feedback_pre_widen_gate"]["after"]["pending_count"], 0)
        self.assertTrue(result["managed_server_calls_made"])
        self.assertEqual(row["status"], "sent")
        self.assertIsNotNone(row["sent_at"])
        self.assertEqual(ManagedFeedbackRampClient.calls[0]["method"], "POST")
        self.assertEqual(ManagedFeedbackRampClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        self.assertAlmostEqual(data["anthropic_thinking_history_compaction"]["rules"][0]["canary"]["canary_fraction"], 0.10)
        crunch_sent = [
            group
            for group in freshness["groups"]
            if group["action_family"] == "crunch" and group["status"] == "sent"
        ]
        self.assertTrue(crunch_sent)
        self.assertFalse(crunch_sent[0]["payload_json_included"])

    def test_pre_widen_preserves_permanent_feedback_failure_and_blocks_widening(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.05, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _enqueue_thinking_feedback(store)
                _log_call(store, "thinking-permanent-applied-a", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050))
                _log_call(store, "thinking-permanent-applied-b", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050))
                _log_call(store, "thinking-permanent-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout", tokens_saved=2000))
                ManagedFeedbackRampClient.status_code = 422
                ManagedFeedbackRampClient.text = "invalid thinking-tail metadata"
                with patch.dict(
                    os.environ,
                    {
                        "TOKENCLAW_MANAGED": "1",
                        "TOKENCLAW_MANAGED_MODE": "live",
                        "TOKENCLAW_MANAGED_CRUNCH": "1",
                        "TOKENCLAW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                        "TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS": "3",
                    },
                    clear=False,
                ), patch("tokenclaw.http_client.httpx.AsyncClient", ManagedFeedbackRampClient):
                    result = build_local_compaction_canary_ramp(
                        store,
                        config_dir=config,
                        apply=True,
                        ramp_step=0.05,
                        min_applied_samples=2,
                        min_holdout_samples=1,
                        now="2026-06-23T10:30:00+00:00",
                    )
                row = store.get_managed_outcome_feedback("thinking-feedback-queued")
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "hold")
        self.assertEqual(decision["blocked_action"], "widen")
        self.assertIn("managed-feedback-not-fresh", decision["reason_codes"])
        self.assertIn("managed-feedback-failed", decision["reason_codes"])
        self.assertEqual(row["status"], "dropped-after-limit")
        self.assertEqual(row["last_status_code"], 422)
        self.assertEqual(row["last_error"], "invalid thinking-tail metadata")
        drain_result = decision["managed_feedback_pre_widen_gate"]["drain"]["results"][0]
        self.assertEqual(drain_result["status"], "dropped-after-limit")
        self.assertEqual(drain_result["reason"], "permanent-client-error")
        self.assertAlmostEqual(data["anthropic_thinking_history_compaction"]["rules"][0]["canary"]["canary_fraction"], 0.05)

    def test_pre_widen_local_rules_only_leaves_feedback_queued_without_network(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.05, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _enqueue_thinking_feedback(store)
                _log_call(store, "thinking-disabled-applied-a", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050))
                _log_call(store, "thinking-disabled-applied-b", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=2000, realized_savings=0.050))
                _log_call(store, "thinking-disabled-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout", tokens_saved=2000))
                with patch.dict(
                    os.environ,
                    {
                        "TOKENCLAW_LOCAL_RULES_ONLY": "1",
                        "TOKENCLAW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    },
                    clear=False,
                ), patch("tokenclaw.http_client.httpx.AsyncClient", ManagedFeedbackRampClient):
                    result = build_local_compaction_canary_ramp(
                        store,
                        config_dir=config,
                        apply=True,
                        ramp_step=0.05,
                        min_applied_samples=2,
                        min_holdout_samples=1,
                        now="2026-06-23T10:30:00+00:00",
                    )
                row = store.get_managed_outcome_feedback("thinking-feedback-queued")
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "hold")
        self.assertEqual(decision["managed_feedback_pre_widen_gate"]["drain"]["status"], "disabled")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        self.assertEqual(ManagedFeedbackRampClient.calls, [])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertAlmostEqual(data["anthropic_thinking_history_compaction"]["rules"][0]["canary"]["canary_fraction"], 0.05)

    def test_prompt_cache_churn_regression_stops_thinking_tail_even_with_saved_tokens(self) -> None:
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
                    "thinking-churn-applied-a",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=5000, realized_savings=0.003),
                    cache_creation_input_tokens=20000,
                    cache_read_input_tokens=0,
                )
                _log_call(
                    store,
                    "thinking-churn-applied-b",
                    crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=5000, realized_savings=0.003),
                    cache_creation_input_tokens=20000,
                    cache_read_input_tokens=0,
                )
                _log_call(
                    store,
                    "thinking-churn-holdout",
                    crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout", tokens_saved=5000),
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=10000,
                )
                result = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now="2026-06-23T10:40:00+00:00",
                )
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
                queued = store.conn.execute("select payload_json from managed_outcome_feedback_queue").fetchall()
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "stop")
        self.assertEqual(decision["recommended_fraction"], 0.0)
        self.assertIn("prompt-cache-churn-regression", decision["reason_codes"])
        self.assertIn("cache-read-savings-regression", decision["reason_codes"])
        self.assertIn("non-positive-net-realized-savings", decision["reason_codes"])
        self.assertGreater(decision["evidence"]["applied_realized_savings_usd"], 0.0)
        self.assertLessEqual(decision["evidence"]["net_savings_after_prompt_cache_churn_usd"], 0.0)
        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["canary"]["canary_fraction"], 0.0)
        self.assertEqual(rule["safety_stop"]["last_ramp_stop_reason"], "non-positive-net-realized-savings")
        self.assertEqual(len(queued), 1)
        payload = json.loads(queued[0]["payload_json"])
        self.assertIn("prompt-cache-churn-regression", payload["reason_codes"])
        self.assertLessEqual(payload["evidence"]["net_savings_after_prompt_cache_churn_usd"], 0.0)

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
                queued = store.conn.execute("select source_surface, endpoint, payload_json from managed_outcome_feedback_queue").fetchall()
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
        self.assertEqual(result["feedback_events"][0]["status"], "queued")
        self.assertEqual(result["feedback_events"][0]["payload"]["traffic_treatment"], "rollback")
        self.assertFalse(result["feedback_events"][0]["payload"]["privacy"]["raw_request_bodies_included"])
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["source_surface"], "anthropic_thinking_compaction_rollback")
        self.assertEqual(queued[0]["endpoint"], "/v1/policy-events")
        payload = json.loads(queued[0]["payload_json"])
        self.assertEqual(payload["schema"], "tokenclaw.local_compaction_rollback_feedback.v1")
        self.assertNotIn(RAW_SECRET, queued[0]["payload_json"])

    def test_non_positive_savings_missing_usage_and_fallback_regression_stop_canary(self) -> None:
        cases = [
            ("zero-savings", _thinking_crunch_meta(applied=True, cohort="canary_applied", tokens_saved=0, realized_savings=0.0), {}, "non-positive-realized-savings"),
            ("missing-usage", _thinking_crunch_meta(applied=True, cohort="canary_applied"), {"actual_input_tokens": 0}, "missing-usage"),
            ("fallback", _thinking_crunch_meta(applied=True, cohort="canary_applied", fallback=True), {}, "applied-fallback-rate-regression"),
        ]
        for label, applied_meta, call_kwargs, reason in cases:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                config = tmp_path / "config"
                config.mkdir()
                rules_path = config / "crunch_rules.yaml"
                rules_path.write_text(_rules_yaml(fraction=0.10, holdout=0.10), encoding="utf-8")
                store = Store(str(tmp_path / "tokenclaw.sqlite3"))
                try:
                    _log_call(store, f"{label}-applied", crunch_json=applied_meta, **call_kwargs)
                    _log_call(store, f"{label}-holdout", crunch_json=_thinking_crunch_meta(applied=False, cohort="canary_holdout"))
                    result = build_local_compaction_canary_ramp(
                        store,
                        config_dir=config,
                        apply=True,
                        min_applied_samples=1,
                        min_holdout_samples=1,
                        max_fallback_rate_delta=0.0,
                        now="2026-06-23T11:10:00+00:00",
                    )
                    data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
                    queued = store.conn.execute("select count(*) as c from managed_outcome_feedback_queue").fetchone()["c"]
                finally:
                    store.conn.close()

            decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
            self.assertEqual(decision["action"], "stop")
            self.assertIn(reason, decision["reason_codes"])
            self.assertFalse(data["anthropic_thinking_history_compaction"]["rules"][0]["enabled"])
            self.assertEqual(queued, 1)

    def test_insufficient_or_positive_evidence_does_not_rollback(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            rules_path = config / "crunch_rules.yaml"
            rules_path.write_text(_rules_yaml(fraction=0.10, holdout=0.10), encoding="utf-8")
            store = Store(str(tmp_path / "tokenclaw.sqlite3"))
            try:
                _log_call(store, "thinking-applied", crunch_json=_thinking_crunch_meta(applied=True, cohort="canary_applied"), cost_est=0.010)
                result = build_local_compaction_canary_ramp(
                    store,
                    config_dir=config,
                    apply=True,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now="2026-06-23T11:20:00+00:00",
                )
                queued = store.conn.execute("select count(*) as c from managed_outcome_feedback_queue").fetchone()["c"]
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            finally:
                store.conn.close()

        decision = next(item for item in result["decisions"] if item["family"] == "anthropic_thinking_history_compaction")
        self.assertEqual(decision["action"], "hold")
        self.assertEqual(queued, 0)
        self.assertTrue(data["anthropic_thinking_history_compaction"]["rules"][0]["enabled"])

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
