import asyncio
import importlib
import json
import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentflow_proxy import cli
from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.recommendations import queue_policy_event_feedback
from agentflow_proxy.store import Store, stable_json, utc_now
from agentflow_proxy import stats
import agentflow_proxy.routing_experiments as experiments


class RoutingExperimentPolicyTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_ROUTING_EXPERIMENTS",
        "AGENTFLOW_ROUTING_EXPERIMENTS_ENABLED",
        "AGENTFLOW_ROUTING_EXPERIMENT_SAMPLE_RATE",
        "AGENTFLOW_ROUTING_EXPERIMENT_DAILY_BUDGET_USD",
        "AGENTFLOW_ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD",
        "AGENTFLOW_RECOMMENDATION_ENABLED",
        "AGENTFLOW_RECOMMENDATION_SERVER_URL",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name
        os.chdir(self.home.name)
        importlib.reload(experiments)

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(experiments)

    def test_default_policy_samples_anthropic_shadow_pass_through(self):
        self.assertEqual(
            experiments.ROUTING_EXPERIMENT_POLICY["profile_id"],
            "first-safe-openai-codex-claude-shadow-pass-through-v1",
        )
        self.assertEqual(experiments.ROUTING_EXPERIMENT_POLICY["mode"], "shadow_candidate_pass_through")
        self.assertIn("anthropic", experiments.ROUTING_EXPERIMENT_POLICY["providers"])
        self.assertIn("openai", experiments.ROUTING_EXPERIMENT_POLICY["providers"])
        self.assertIn("anthropic_messages", experiments.ROUTING_EXPERIMENT_POLICY["source_surfaces"])
        self.assertIn("openai_responses", experiments.ROUTING_EXPERIMENT_POLICY["source_surfaces"])
        self.assertIn("codex_turn", experiments.ROUTING_EXPERIMENT_POLICY["source_surfaces"])

        meta = experiments.routing_experiment_decision(
            {"model": "claude-sonnet-4-6"},
            {
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "category": "short-completion",
                "text_chars": 1000,
            },
            stream=False,
            provider="anthropic",
            source_surface="anthropic_messages",
            random_value=lambda: 0.0,
        )

        self.assertTrue(meta["enabled"])
        self.assertTrue(meta["sampled"])
        self.assertEqual(meta["reason"], "sampled-shadow-candidate-pass-through")
        self.assertTrue(meta["counterfactual"])
        self.assertTrue(meta["shadow_only"])
        self.assertEqual(meta["primary_model"], "claude-sonnet-4-6")
        self.assertEqual(meta["user_visible_model"], "claude-sonnet-4-6")
        self.assertEqual(meta["shadow_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(meta["routed_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(meta["source_surface"], "anthropic_messages")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_default_policy_can_sample_anthropic_streaming_shadow_pass_through(self):
        meta = experiments.routing_experiment_decision(
            {"model": "claude-sonnet-4-6", "stream": True},
            {
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "category": "short-completion",
                "text_chars": 1000,
            },
            stream=True,
            provider="anthropic",
            source_surface="anthropic_messages",
            random_value=lambda: 0.0,
        )

        self.assertTrue(meta["sampled"])
        self.assertEqual(meta["reason"], "streaming-shadow-sampled")
        self.assertTrue(meta["streaming_shadow_supported"])
        self.assertTrue(meta["stream"])
        self.assertEqual(meta["primary_model"], "claude-sonnet-4-6")
        self.assertEqual(meta["shadow_model"], "claude-haiku-4-5-20251001")

    def test_streaming_shadow_gate_skips_non_anthropic_surfaces(self):
        meta = experiments.routing_experiment_decision(
            {"model": "gpt-5.4", "stream": True},
            {
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "category": "chat",
                "text_chars": 1000,
            },
            stream=True,
            provider="openai",
            source_surface="openai_responses",
            random_value=lambda: 0.0,
        )

        self.assertFalse(meta["sampled"])
        self.assertEqual(meta["reason"], "streaming-shadow-unsupported")
        self.assertFalse(meta["streaming_shadow_supported"])

    def test_streaming_shadow_budget_exhaustion_uses_existing_budget_controls(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                store.log_routing_experiment(
                    id="spent-streaming-budget",
                    call_id="call-1",
                    created_at=utc_now(),
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-sonnet-4-6",
                    shadow_model="claude-haiku-4-5-20251001",
                    category="short-completion",
                    routing_reason="streaming-shadow-sampled",
                    input_tokens_est=100,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=10,
                    shadow_latency_ms=20,
                    primary_output_chars=1,
                    shadow_output_chars=1,
                    primary_output_sha256="a",
                    shadow_output_sha256="b",
                    output_similarity=1.0,
                    passed_threshold=1,
                    primary_cost_est_usd=0.001,
                    shadow_cost_est_usd=10.0,
                    routing_json=stable_json({}),
                    experiment_json=stable_json({"sampled": True, "stream": True}),
                )
                meta = experiments.routing_experiment_decision(
                    {"model": "claude-sonnet-4-6", "stream": True},
                    {
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "short-completion",
                        "text_chars": 1000,
                    },
                    stream=True,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    store_obj=store,
                    random_value=lambda: 0.0,
                )
            finally:
                store.conn.close()

        self.assertFalse(meta["sampled"])
        self.assertEqual(meta["reason"], "daily-budget-exhausted")
        self.assertTrue(meta["budget_exhausted"])

    def test_scoped_caps_allow_large_anthropic_streaming_tool_result_without_widening_openai(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
min_text_chars: 0
max_text_chars: 8000
providers:
  - anthropic
  - openai
source_surfaces:
  - anthropic_messages
  - openai_responses
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
  - requested_model: gpt-5.4
    routed_model: gpt-5.4-mini
categories:
  - chat
  - tool-result
eligibility_overrides:
  - scope: provider
    provider: anthropic
    stream: true
    max_text_chars: 32000
  - scope: category
    provider: anthropic
    source_surface: anthropic_messages
    category: tool-result
    stream: true
    max_text_chars: 128000
    sample_rate: 1.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            claude = manual.routing_experiment_decision(
                {"model": "claude-sonnet-4-6", "stream": True},
                {
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "category": "tool-result",
                    "text_chars": 90000,
                },
                stream=True,
                provider="anthropic",
                source_surface="anthropic_messages",
                random_value=lambda: 0.0,
            )
            openai = manual.routing_experiment_decision(
                {"model": "gpt-5.4"},
                {
                    "requested_model": "gpt-5.4",
                    "routed_model": "gpt-5.4",
                    "category": "chat",
                    "text_chars": 9000,
                },
                stream=False,
                provider="openai",
                source_surface="openai_responses",
                random_value=lambda: 0.0,
            )

        self.assertTrue(claude["sampled"])
        self.assertEqual(claude["reason"], "streaming-shadow-sampled")
        self.assertEqual(claude["max_text_chars"], 128000)
        self.assertEqual(claude["max_text_chars_scope"], "category")
        self.assertEqual(claude["sample_rate_scope"], "category")
        self.assertFalse(openai["sampled"])
        self.assertEqual(openai["reason"], "request-too-large")
        self.assertEqual(openai["skip_diagnostic"], "global-max-text-chars-exceeded")
        self.assertEqual(openai["max_text_chars_scope"], "global")

    def test_scoped_cap_diagnostics_distinguish_provider_and_source_surface_caps(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
max_text_chars: 8000
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - chat
eligibility_overrides:
  - scope: provider
    provider: anthropic
    stream: true
    max_text_chars: 48000
  - scope: source-surface
    provider: anthropic
    source_surface: anthropic_messages
    stream: true
    max_text_chars: 24000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            source_surface_capped = manual.routing_experiment_decision(
                {"model": "claude-sonnet-4-6", "stream": True},
                {
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "category": "chat",
                    "text_chars": 30000,
                },
                stream=True,
                provider="anthropic",
                source_surface="anthropic_messages",
                random_value=lambda: 0.0,
            )

            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
max_text_chars: 8000
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - chat
eligibility_overrides:
  - scope: provider
    provider: anthropic
    stream: true
    max_text_chars: 24000
""",
                encoding="utf-8",
            )
            manual = importlib.reload(experiments)
            provider_capped = manual.routing_experiment_decision(
                {"model": "claude-sonnet-4-6", "stream": True},
                {
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "category": "chat",
                    "text_chars": 30000,
                },
                stream=True,
                provider="anthropic",
                source_surface="anthropic_messages",
                random_value=lambda: 0.0,
            )

        self.assertFalse(source_surface_capped["sampled"])
        self.assertEqual(source_surface_capped["reason"], "source-surface-max-text-chars-exceeded")
        self.assertEqual(source_surface_capped["max_text_chars_scope"], "source-surface")
        self.assertFalse(provider_capped["sampled"])
        self.assertEqual(provider_capped["reason"], "provider-max-text-chars-exceeded")
        self.assertEqual(provider_capped["max_text_chars_scope"], "provider")

    def test_scoped_category_budget_exhaustion_is_reported_without_raw_payloads(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
max_text_chars: 8000
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - tool-result
eligibility_overrides:
  - scope: category
    provider: anthropic
    source_surface: anthropic_messages
    category: tool-result
    stream: true
    max_text_chars: 128000
    daily_budget_usd: 0.01
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                store.log_routing_experiment(
                    id="spent-category-budget",
                    call_id="call-1",
                    created_at=utc_now(),
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-sonnet-4-6",
                    shadow_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    routing_reason="streaming-shadow-sampled",
                    input_tokens_est=100,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=10,
                    shadow_latency_ms=20,
                    primary_output_chars=1,
                    shadow_output_chars=1,
                    primary_output_sha256="a",
                    shadow_output_sha256="b",
                    output_similarity=1.0,
                    passed_threshold=1,
                    primary_cost_est_usd=0.001,
                    shadow_cost_est_usd=0.01,
                    routing_json=stable_json({}),
                    experiment_json=stable_json({"sampled": True, "stream": True}),
                )
                meta = manual.routing_experiment_decision(
                    {"model": "claude-sonnet-4-6", "stream": True},
                    {
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "text_chars": 90000,
                    },
                    stream=True,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    store_obj=store,
                    random_value=lambda: 0.0,
                )
            finally:
                store.conn.close()

        self.assertFalse(meta["sampled"])
        self.assertEqual(meta["reason"], "daily-budget-exhausted")
        self.assertEqual(meta["budget_cap_scope"], "category")
        self.assertTrue(meta["budget_exhausted"])
        rendered = stable_json(meta)
        self.assertNotIn("raw prompt", rendered)
        self.assertNotIn("req-secret", rendered)
        self.assertNotIn("session-secret", rendered)

    def test_default_policy_samples_codex_turn_shadow_pass_through(self):
        self.assertIn("codex-turn", experiments.ROUTING_EXPERIMENT_POLICY["categories"])

        meta = experiments.routing_experiment_decision(
            {"model": "gpt-5-codex", "input": "summarize the run"},
            {
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "text_chars": 17,
            },
            stream=False,
            provider="openai",
            source_surface="codex_turn",
            random_value=lambda: 0.0,
        )

        self.assertTrue(meta["sampled"])
        self.assertEqual(meta["mode"], "shadow_candidate_pass_through")
        self.assertTrue(meta["counterfactual"])
        self.assertTrue(meta["shadow_only"])
        self.assertEqual(meta["primary_model"], "gpt-5-codex")
        self.assertEqual(meta["user_visible_model"], "gpt-5-codex")
        self.assertEqual(meta["shadow_model"], "gpt-5-mini")
        self.assertEqual(meta["routed_model"], "gpt-5-mini")
        self.assertEqual(meta["source_surface"], "codex_turn")
        self.assertEqual(meta["reason"], "sampled-shadow-candidate-pass-through")

    def test_config_policy_samples_routed_down_non_streaming_call(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: applied_routed_down
sample_rate: 1.0
daily_budget_usd: 0.05
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - tool-result
max_text_chars: 5000
similarity_threshold: 0.9
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            meta = manual.routing_experiment_decision(
                {"model": "claude-haiku-4-5-20251001"},
                {
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-haiku-4-5-20251001",
                    "category": "tool-result",
                    "text_chars": 1000,
                },
                stream=False,
                random_value=lambda: 0.99,
            )

            self.assertTrue(meta["enabled"])
            self.assertTrue(meta["sampled"])
            self.assertEqual(meta["status"], "selected")
            self.assertEqual(meta["reason"], "sampled-routed-down-call")
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["rule_path"], str(config / "routing_experiments.yaml"))
            self.assertEqual(meta["similarity_threshold"], 0.9)
            self.assertEqual(meta["daily_budget_usd"], 0.05)
            self.assertFalse(meta["budget_exhausted"])

    def test_enabled_zero_budget_records_budget_reason_without_sampling(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: applied_routed_down
sample_rate: 1.0
daily_budget_usd: 0.0
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
workflow_phases:
  - tool-execution
categories:
  - tool-result
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            meta = manual.routing_experiment_decision(
                {"model": "claude-haiku-4-5-20251001"},
                {
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-haiku-4-5-20251001",
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "text_chars": 1000,
                },
                stream=False,
                provider="anthropic",
                source_surface="anthropic_messages",
                random_value=lambda: 0.0,
            )

            self.assertTrue(meta["enabled"])
            self.assertFalse(meta["sampled"])
            self.assertEqual(meta["reason"], "daily-budget-zero")
            self.assertTrue(meta["budget_exhausted"])
            self.assertEqual(meta["privacy"]["metadata_only"], True)

    def test_first_safe_openai_profile_can_sample_codex_pair(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
profile_id: first-safe-openai-codex-ab-v1
mode: applied_routed_down
enabled: true
sample_rate: 1.0
daily_budget_usd: 0.05
providers:
  - openai
source_surfaces:
  - openai_responses
  - openai_chat
  - codex_turn
model_pairs:
  - requested_model: gpt-5-codex
    routed_model: gpt-5-mini
categories:
  - chat
  - short-completion
max_text_chars: 8000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            meta = manual.routing_experiment_decision(
                {"model": "gpt-5-mini", "input": "short prompt"},
                {
                    "requested_model": "gpt-5-codex",
                    "routed_model": "gpt-5-mini",
                    "category": "short-completion",
                    "text_chars": 12,
                },
                stream=False,
                provider="openai",
                source_surface="openai_responses",
                random_value=lambda: 0.0,
            )

            self.assertTrue(meta["sampled"])
            self.assertEqual(meta["profile_id"], "first-safe-openai-codex-ab-v1")
            self.assertEqual(meta["shadow_model"], "gpt-5-codex")
            self.assertEqual(meta["reason"], "sampled-routed-down-call")

    def test_shadow_candidate_pass_through_mode_samples_unrouted_openai_call(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
profile_id: first-safe-openai-codex-ab-v1
mode: shadow_candidate_pass_through
enabled: true
sample_rate: 1.0
daily_budget_usd: 10.0
providers:
  - openai
source_surfaces:
  - openai_responses
model_pairs:
  - requested_model: gpt-5.4
    routed_model: gpt-5.4-mini
categories:
  - chat
max_text_chars: 8000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            meta = manual.routing_experiment_decision(
                {"model": "gpt-5.4", "input": "short prompt"},
                {
                    "requested_model": "gpt-5.4",
                    "routed_model": "gpt-5.4",
                    "category": "chat",
                    "text_chars": 12,
                },
                stream=False,
                provider="openai",
                source_surface="openai_responses",
                random_value=lambda: 0.0,
            )

        self.assertTrue(meta["sampled"])
        self.assertEqual(meta["mode"], "shadow_candidate_pass_through")
        self.assertTrue(meta["counterfactual"])
        self.assertTrue(meta["shadow_only"])
        self.assertEqual(meta["reason"], "sampled-shadow-candidate-pass-through")
        self.assertEqual(meta["requested_model"], "gpt-5.4")
        self.assertEqual(meta["routed_model"], "gpt-5.4-mini")
        self.assertEqual(meta["primary_model"], "gpt-5.4")
        self.assertEqual(meta["user_visible_model"], "gpt-5.4")
        self.assertEqual(meta["shadow_model"], "gpt-5.4-mini")

    def test_shadow_candidate_pass_through_skips_unconfigured_anthropic_pair(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
mode: shadow_candidate_pass_through
enabled: true
sample_rate: 1.0
daily_budget_usd: 10.0
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - short-completion
max_text_chars: 8000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)

            meta = manual.routing_experiment_decision(
                {"model": "claude-opus-4-5"},
                {
                    "requested_model": "claude-opus-4-5",
                    "routed_model": "claude-opus-4-5",
                    "category": "short-completion",
                    "text_chars": 1000,
                },
                stream=False,
                provider="anthropic",
                source_surface="anthropic_messages",
                random_value=lambda: 0.0,
            )

        self.assertFalse(meta["sampled"])
        self.assertEqual(meta["reason"], "model-pair-not-enabled")

    def test_budget_spend_blocks_after_cap_is_exhausted(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: applied_routed_down
sample_rate: 1.0
daily_budget_usd: 0.01
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - tool-result
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                store.log_routing_experiment(
                    id="spent-budget",
                    call_id="call-1",
                    created_at=utc_now(),
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-haiku-4-5-20251001",
                    shadow_model="claude-sonnet-4-6",
                    category="tool-result",
                    routing_reason="fixture",
                    input_tokens_est=100,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=10,
                    shadow_latency_ms=20,
                    primary_output_chars=1,
                    shadow_output_chars=1,
                    primary_output_sha256="a",
                    shadow_output_sha256="b",
                    output_similarity=1.0,
                    passed_threshold=1,
                    primary_cost_est_usd=0.001,
                    shadow_cost_est_usd=0.01,
                    routing_json=stable_json({}),
                    experiment_json=stable_json({"sampled": True}),
                )

                meta = manual.routing_experiment_decision(
                    {"model": "claude-haiku-4-5-20251001"},
                    {
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-haiku-4-5-20251001",
                        "category": "tool-result",
                        "text_chars": 1000,
                    },
                    stream=False,
                    store_obj=store,
                    random_value=lambda: 0.0,
                )
            finally:
                store.conn.close()

            self.assertFalse(meta["sampled"])
            self.assertEqual(meta["reason"], "daily-budget-exhausted")
            self.assertEqual(meta["budget_spent_usd"], 0.01)

    def test_response_comparison_produces_similarity_and_hashes(self):
        primary = {"content": [{"type": "text", "text": "summarize the build output"}]}
        shadow = {"content": [{"type": "text", "text": "summarize the build output"}]}

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertEqual(result["primary_output_chars"], 26)
        self.assertEqual(result["shadow_output_chars"], 26)
        self.assertEqual(result["primary_output_sha256"], result["shadow_output_sha256"])
        self.assertEqual(result["output_similarity"], 1.0)
        self.assertTrue(result["passed_threshold"])

    def test_response_comparison_extracts_openai_output_text_without_raw_storage(self):
        primary = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok from primary"}],
                }
            ]
        }
        shadow = {"choices": [{"message": {"content": "ok from primary"}}]}

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertEqual(result["primary_output_chars"], 15)
        self.assertEqual(result["shadow_output_chars"], 15)
        self.assertEqual(result["primary_output_sha256"], result["shadow_output_sha256"])
        self.assertEqual(result["output_similarity"], 1.0)

    def test_report_explains_unqualified_traffic_with_decision_reasons(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                store.log_call(
                    id="call-streaming-skip",
                    created_at=utc_now(),
                    path="/v1/responses",
                    provider="openai",
                    source_surface="openai_responses",
                    requested_model="gpt-5-codex",
                    routed_model="gpt-5-codex",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=10,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "streaming",
                            "mode": "shadow_candidate_pass_through",
                            "provider": "openai",
                            "source_surface": "openai_responses",
                        }
                    }),
                    category="short-completion",
                )
                report = experiments.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        self.assertEqual(report["policy"]["profile_id"], "first-safe-openai-codex-claude-shadow-pass-through-v1")
        self.assertEqual(report["summary"]["sample_count"], 0)
        self.assertEqual(report["policy"]["mode"], experiments.ROUTING_EXPERIMENT_MODE)
        self.assertEqual(report["decision_reasons"][0]["provider"], "openai")
        self.assertEqual(report["decision_reasons"][0]["source_surface"], "openai_responses")
        self.assertEqual(report["decision_reasons"][0]["reason"], "streaming")
        self.assertEqual(report["decision_reasons"][0]["count"], 1)

    def test_report_projects_claude_streaming_requests_newly_eligible_under_scoped_caps(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
min_text_chars: 0
max_text_chars: 8000
providers:
  - anthropic
  - openai
source_surfaces:
  - anthropic_messages
  - openai_responses
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
  - requested_model: gpt-5.4
    routed_model: gpt-5.4-mini
categories:
  - chat
  - tool-result
eligibility_overrides:
  - scope: category
    provider: anthropic
    source_surface: anthropic_messages
    category: tool-result
    stream: true
    max_text_chars: 128000
    sample_rate: 0.25
    daily_budget_usd: 2.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                store.log_call(
                    id="claude-large-stream-secret",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=100,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "category": "tool-result",
                        "text_chars": 90000,
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "request-too-large",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "text_chars": 90000,
                        },
                        "raw_prompt": "raw projection secret",
                        "file_path": "/tmp/projection-secret.py",
                    }),
                    category="tool-result",
                )
                store.log_call(
                    id="openai-large-secret",
                    created_at=utc_now(),
                    path="/v1/responses",
                    provider="openai",
                    source_surface="openai_responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=100,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "category": "chat",
                        "text_chars": 9000,
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "request-too-large",
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "category": "chat",
                            "text_chars": 9000,
                        },
                    }),
                    category="chat",
                )
                report = manual.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        projection = report["eligibility_projection"]
        [claude_row] = projection["claude_streaming"]
        self.assertEqual(claude_row["category"], "tool-result")
        self.assertEqual(claude_row["observed_call_count"], 1)
        self.assertEqual(claude_row["global_cap_eligible_count"], 0)
        self.assertEqual(claude_row["effective_cap_eligible_count"], 1)
        self.assertEqual(claude_row["newly_eligible_call_count"], 1)
        self.assertEqual(claude_row["effective_max_text_chars_scope"], "category")
        self.assertEqual(claude_row["effective_sample_rate"], 0.25)
        openai_row = next(row for row in projection["rows"] if row["provider"] == "openai")
        self.assertEqual(openai_row["effective_max_text_chars_scope"], "global")
        self.assertEqual(openai_row["newly_eligible_call_count"], 0)
        rendered = stable_json(report)
        self.assertNotIn("raw projection secret", rendered)
        self.assertNotIn("/tmp/projection-secret.py", rendered)
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(projection["privacy"]["file_paths_included"])

    def test_report_explains_claude_shadow_yield_for_streaming_tool_result_and_blocked_row(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: 10.0
min_text_chars: 0
max_text_chars: 8000
providers:
  - anthropic
source_surfaces:
  - anthropic_messages
streaming_shadow_source_surfaces:
  - anthropic_messages
model_pairs:
  - requested_model: claude-sonnet-4-6
    routed_model: claude-haiku-4-5-20251001
categories:
  - chat
  - tool-result
eligibility_overrides:
  - scope: category
    provider: anthropic
    source_surface: anthropic_messages
    category: tool-result
    stream: true
    max_text_chars: 128000
    sample_rate: 1.0
    daily_budget_usd: 10.0
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(experiments)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                store.log_call(
                    id="claude-stream-tool-result-secret",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=20,
                    input_tokens_est=1000,
                    output_tokens_est=20,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.03,
                    routing_json=stable_json({
                        "category": "tool-result",
                        "text_chars": 90000,
                        "routing_experiment": {
                            "schema": "agentflow.routing_experiment_decision.v1",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "status": "selected",
                            "sampled": True,
                            "reason": "streaming-shadow-sampled",
                            "requested_model": "claude-sonnet-4-6",
                            "routed_model": "claude-haiku-4-5-20251001",
                            "shadow_model": "claude-haiku-4-5-20251001",
                            "category": "tool-result",
                            "stream": True,
                            "text_chars": 90000,
                        },
                        "raw_prompt": "raw claude yield secret",
                        "file_path": "/tmp/claude-yield-secret.py",
                    }),
                    category="tool-result",
                )
                store.log_call(
                    id="claude-stream-chat-blocked-secret",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=20,
                    input_tokens_est=1000,
                    output_tokens_est=20,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.03,
                    routing_json=stable_json({
                        "category": "chat",
                        "text_chars": 90000,
                        "routing_experiment": {
                            "schema": "agentflow.routing_experiment_decision.v1",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "status": "skipped",
                            "sampled": False,
                            "reason": "request-too-large",
                            "skip_diagnostic": "global-max-text-chars-exceeded",
                            "requested_model": "claude-sonnet-4-6",
                            "routed_model": "claude-haiku-4-5-20251001",
                            "shadow_model": "claude-haiku-4-5-20251001",
                            "category": "chat",
                            "stream": True,
                            "text_chars": 90000,
                        },
                    }),
                    category="chat",
                )
                store.log_routing_experiment(
                    id="claude-sampled-stream-tool-result",
                    call_id="claude-stream-tool-result-secret",
                    created_at=utc_now(),
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-sonnet-4-6",
                    shadow_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    routing_reason="streaming-shadow-sampled",
                    input_tokens_est=1000,
                    primary_status_code=200,
                    shadow_status_code=200,
                    primary_latency_ms=100,
                    shadow_latency_ms=80,
                    primary_output_chars=20,
                    shadow_output_chars=18,
                    primary_output_sha256="primary",
                    shadow_output_sha256="shadow",
                    output_similarity=0.94,
                    passed_threshold=1,
                    primary_cost_est_usd=0.03,
                    shadow_cost_est_usd=0.005,
                    routing_json=stable_json({}),
                    experiment_json=stable_json({"mode": "shadow_candidate_pass_through"}),
                )
                report = manual.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        yield_report = report["claude_shadow_yield"]
        self.assertEqual(yield_report["schema"], "agentflow.claude_shadow_routing_yield.v1")
        self.assertEqual(yield_report["summary"]["observed_call_count"], 2)
        self.assertEqual(yield_report["summary"]["selected_count"], 1)
        self.assertEqual(yield_report["summary"]["skipped_count"], 1)
        self.assertEqual(yield_report["summary"]["sampled_count"], 1)
        self.assertEqual(yield_report["summary"]["compared_count"], 1)
        self.assertIn({"reason": "request-too-large", "count": 1}, yield_report["skipped_reason_counts"])
        self.assertIn({"reason": "request-too-large", "count": 1}, yield_report["cap_block_reason_counts"])
        tool_row = next(row for row in yield_report["observed"] if row["category"] == "tool-result")
        self.assertTrue(tool_row["stream"])
        self.assertEqual(tool_row["candidate_target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(tool_row["effective_max_text_chars"], 128000)
        self.assertEqual(tool_row["effective_max_text_chars_scope"], "category")
        self.assertEqual(tool_row["selected_count"], 1)
        chat_row = next(row for row in yield_report["observed"] if row["category"] == "chat")
        self.assertEqual(chat_row["ineligible_count"], 1)
        self.assertEqual(chat_row["decision_reasons"], [{"reason": "request-too-large", "count": 1}])
        self.assertEqual(yield_report["projection"]["projected_samples_current_sample_rate"], 1.0)
        self.assertEqual(yield_report["projection"]["projected_samples_sample_rate_100pct"], 1.0)
        rendered_yield = stable_json(yield_report)
        self.assertNotIn("raw claude yield secret", rendered_yield)
        self.assertNotIn("/tmp/claude-yield-secret.py", rendered_yield)
        self.assertFalse(yield_report["privacy"]["filesystem_paths_included"])
        self.assertFalse(yield_report["privacy"]["request_ids_included"])

    def test_report_counts_codex_app_event_decision_reasons_without_sample_rows(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                store.log_codex_app_event(
                    id="codex-start-skip",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    request_id="turn-secret",
                    thread_id="thread-secret",
                    message_chars=100,
                    params_chars=80,
                    input_items=1,
                    input_text_chars=42,
                    session_id="session-secret",
                    routing_json=stable_json({
                        "status": "skipped",
                        "reason": "codex-turn-start-model-field-absent",
                        "routing_experiment": {
                            "schema": "agentflow.routing_experiment_decision.v1",
                            "provider": "openai",
                            "source_surface": "codex_turn",
                            "status": "skipped",
                            "reason": "missing-requested-model",
                            "sampled": False,
                        },
                    }),
                )
                report = experiments.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        self.assertEqual(report["summary"]["sample_count"], 0)
        self.assertEqual(report["summary"]["decision_count"], 1)
        self.assertEqual(report["summary"]["decision_status_counts"], {"skipped": 1})
        self.assertEqual(report["decision_surfaces"], [
            {"provider": "openai", "source_surface": "codex_turn", "status": "skipped", "count": 1}
        ])
        self.assertEqual(report["decision_reasons"][0]["provider"], "openai")
        self.assertEqual(report["decision_reasons"][0]["source_surface"], "codex_turn")
        self.assertEqual(report["decision_reasons"][0]["reason"], "missing-requested-model")
        rendered = stable_json(report)
        self.assertNotIn("turn-secret", rendered)
        self.assertNotIn("thread-secret", rendered)
        self.assertNotIn("session-secret", rendered)

    def test_store_attaches_shadow_coverage_metadata_to_uncovered_proxy_calls(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                store.log_call(
                    id="uncovered-openai-call",
                    created_at=utc_now(),
                    path="/v1/chat/completions",
                    provider="openai",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=10,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "chat",
                        "raw_prompt": "raw secret prompt should not enter experiment metadata",
                    }),
                    category="chat",
                )
                row = store.conn.execute(
                    "select provider, source_surface, routing_json from calls where id = ?",
                    ("uncovered-openai-call",),
                ).fetchone()
                report = experiments.build_routing_experiment_report(store, limit=5)
            finally:
                store.conn.close()

        routing = json.loads(row["routing_json"])
        experiment = routing["routing_experiment"]
        self.assertEqual(row["source_surface"], "openai_chat")
        self.assertEqual(experiment["schema"], "agentflow.routing_experiment_decision.v1")
        self.assertEqual(experiment["status"], "skipped")
        self.assertEqual(experiment["reason"], "routing-experiment-decision-missing")
        self.assertEqual(experiment["coverage_class"], "blocked")
        self.assertEqual(experiment["source_surface"], "openai_chat")
        rendered_experiment = stable_json(experiment)
        self.assertNotIn("raw secret prompt", rendered_experiment)
        self.assertFalse(experiment["privacy"]["raw_prompts_included"])
        self.assertEqual(report["summary"]["routing_experiment_denominator_count"], 1)
        self.assertEqual(report["summary"]["routing_experiment_metadata_coverage_rate"], 1.0)
        self.assertEqual(report["summary"]["eligible_count"], 1)
        self.assertEqual(report["summary"]["blocked_count"], 1)
        self.assertEqual(report["summary"]["not_sampled_count"], 0)
        self.assertEqual(report["summary"]["out_of_scope_count"], 0)
        self.assertEqual(report["summary"]["metadata_missing_count"], 0)

    def test_report_denominator_includes_codex_turns_without_sample_rows(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                store.log_codex_app_event(
                    id="codex-uncovered-turn",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    request_id="turn-secret",
                    thread_id="thread-secret",
                    message_chars=50,
                    params_chars=10,
                    input_items=1,
                    input_text_chars=50,
                    session_id="session-secret",
                    routing_json=stable_json({
                        "status": "observed",
                        "raw_prompt": "raw codex secret should not enter experiment metadata",
                    }),
                )
                report = experiments.build_routing_experiment_report(store, limit=5)
                row = store.conn.execute(
                    "select routing_json from codex_app_events where id = ?",
                    ("codex-uncovered-turn",),
                ).fetchone()
            finally:
                store.conn.close()

        routing = json.loads(row["routing_json"])
        experiment = routing["routing_experiment"]
        self.assertEqual(experiment["provider"], "openai")
        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["reason"], "routing-experiment-decision-missing")
        self.assertEqual(report["summary"]["routing_experiment_denominator_count"], 1)
        self.assertEqual(report["summary"]["decision_coverage_counts"]["blocked"], 1)
        self.assertEqual(report["decision_surfaces"], [
            {"provider": "openai", "source_surface": "codex_turn", "status": "skipped", "count": 1}
        ])
        rendered = stable_json(report)
        self.assertNotIn("turn-secret", rendered)
        self.assertNotIn("thread-secret", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("raw codex secret", stable_json(experiment))

    def test_report_groups_anthropic_openai_and_codex_samples_separately(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for idx, (provider, surface, requested, routed, shadow) in enumerate([
                    ("anthropic", "anthropic_messages", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
                    ("openai", "openai_responses", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-mini"),
                    ("openai", "codex_turn", "gpt-5-codex", "gpt-5-mini", "gpt-5-mini"),
                ]):
                    store.log_routing_experiment(
                        id=f"sample-{idx}",
                        call_id=f"call-{idx}",
                        created_at=utc_now(),
                        provider=provider,
                        source_surface=surface,
                        requested_model=requested,
                        routed_model=routed,
                        primary_model=requested,
                        shadow_model=shadow,
                        category="short-completion" if surface != "codex_turn" else "codex-turn",
                        routing_reason="sampled-shadow-candidate-pass-through",
                        input_tokens_est=100,
                        primary_status_code=200,
                        shadow_status_code=200,
                        primary_latency_ms=100,
                        shadow_latency_ms=90,
                        primary_output_chars=2,
                        shadow_output_chars=2,
                        primary_output_sha256=f"primary-{idx}",
                        shadow_output_sha256=f"shadow-{idx}",
                        output_similarity=1.0,
                        passed_threshold=1,
                        primary_cost_est_usd=0.003,
                        shadow_cost_est_usd=0.001,
                        routing_json=stable_json({}),
                        experiment_json=stable_json({"mode": "shadow_candidate_pass_through"}),
                    )
                report = experiments.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        surfaces = {
            (item["provider"], item["source_surface"], item["requested_model"], item["routed_model"])
            for item in report["candidates"]
        }
        self.assertIn(("anthropic", "anthropic_messages", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"), surfaces)
        self.assertIn(("openai", "openai_responses", "gpt-5.4", "gpt-5.4-mini"), surfaces)
        self.assertIn(("openai", "codex_turn", "gpt-5-codex", "gpt-5-mini"), surfaces)

    def test_feedback_features_are_metadata_only(self):
        comparison = {
            "primary_output_chars": 17,
            "shadow_output_chars": 19,
            "primary_output_sha256": "primary-hash",
            "shadow_output_sha256": "shadow-hash",
            "output_similarity": 0.91,
            "passed_threshold": True,
        }

        result = experiments.routing_experiment_feedback_features(
            experiment_id="exp-1",
            experiment_meta={
                "sampled": True,
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "similarity_threshold": 0.86,
                "text_chars": 7000,
                "mode": "shadow_candidate_pass_through",
                "counterfactual": True,
                "shadow_only": True,
            },
            routing_meta={
                "category": "tool-result",
                "reason": "tool-result processing turn routed to Haiku",
            },
            comparison=comparison,
            primary_model="claude-haiku-4-5-20251001",
            shadow_model="claude-sonnet-4-6",
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=50,
            shadow_latency_ms=90,
            primary_cost_est_usd=0.001,
            shadow_cost_est_usd=0.004,
        )

        self.assertEqual(result["schema"], "agentflow.routing_experiment_feedback.v1")
        self.assertEqual(result["mode"], "shadow_candidate_pass_through")
        self.assertTrue(result["counterfactual"])
        self.assertTrue(result["shadow_only"])
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["candidate_bucket"], "tool-result:claude-sonnet-4-6->claude-haiku-4-5-20251001")
        self.assertEqual(result["text_chars_bucket"], "2k-8k")
        self.assertEqual(result["output_similarity"], 0.91)
        self.assertEqual(result["primary_output_sha256"], "primary-hash")
        self.assertNotIn("text", result)
        self.assertEqual(result["reason_codes"], ["passed"])
        self.assertTrue(result["privacy"]["metadata_only"])

    def test_outcome_event_uses_metadata_only_aggregate_fields(self):
        feedback = experiments.routing_experiment_feedback_features(
            experiment_id="local-exp-secret",
            experiment_meta={
                "sampled": True,
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "similarity_threshold": 0.86,
                "text_chars": 7000,
                "mode": "shadow_candidate_pass_through",
                "counterfactual": True,
                "shadow_only": True,
            },
            routing_meta={
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "reason": "tool-result processing turn routed to Haiku",
            },
            comparison={
                "primary_output_chars": 17,
                "shadow_output_chars": 19,
                "primary_output_sha256": "sha256:primary",
                "shadow_output_sha256": "sha256:shadow",
                "output_similarity": 0.91,
                "passed_threshold": True,
            },
            primary_model="claude-haiku-4-5-20251001",
            shadow_model="claude-sonnet-4-6",
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=50,
            shadow_latency_ms=90,
            primary_cost_est_usd=0.001,
            shadow_cost_est_usd=0.004,
        )

        event = experiments.routing_experiment_outcome_event(feedback)

        assert_managed_egress_safe(event)
        self.assertEqual(event["schema"], "agentflow.routing_experiment_outcome_event.v1")
        self.assertEqual(event["candidate"]["mode"], "shadow_candidate_pass_through")
        self.assertTrue(event["candidate"]["counterfactual"])
        self.assertTrue(event["candidate"]["shadow_only"])
        self.assertEqual(event["outcome"]["mode"], "shadow_candidate_pass_through")
        self.assertEqual(event["source_surface"], "anthropic_messages")
        self.assertEqual(event["app_family"], "claude_code")
        self.assertEqual(event["workflow_phase"], "tool-execution")
        self.assertEqual(event["candidate"]["candidate_bucket"], "tool-result:sonnet->haiku")
        self.assertEqual(event["candidate"]["requested_model_family"], "sonnet")
        self.assertEqual(event["candidate"]["routed_model_family"], "haiku")
        self.assertEqual(event["candidate"]["shadow_model_family"], "sonnet")
        self.assertEqual(event["outcome"]["primary_status_class"], "2xx")
        self.assertEqual(event["outcome"]["shadow_status_class"], "2xx")
        self.assertEqual(event["outcome"]["output_similarity"], 0.91)
        self.assertEqual(event["outcome"]["primary_output_sha256"], "sha256:primary")
        self.assertTrue(event["privacy"]["metadata_only"])
        rendered = json.dumps(event, sort_keys=True)
        self.assertNotIn("claude-sonnet-4-6", rendered)
        self.assertNotIn("claude-haiku-4-5-20251001", rendered)
        self.assertNotIn("local-exp-secret", rendered)
        for forbidden_key in (
            '"messages"',
            '"provider_body"',
            '"raw_request"',
            '"raw_response"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
            '"tenant_id"',
        ):
            self.assertNotIn(forbidden_key, rendered)

    def test_disabled_managed_mode_queues_routing_experiment_policy_event(self):
        feedback = experiments.routing_experiment_feedback_features(
            experiment_id="exp-1",
            experiment_meta={
                "sampled": True,
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "similarity_threshold": 0.86,
                "text_chars": 7000,
            },
            routing_meta={"category": "tool-result", "workflow_phase": "tool-execution"},
            comparison={
                "primary_output_chars": 17,
                "shadow_output_chars": 19,
                "primary_output_sha256": "sha256:primary",
                "shadow_output_sha256": "sha256:shadow",
                "output_similarity": 0.91,
                "passed_threshold": True,
            },
            primary_model="claude-haiku-4-5-20251001",
            shadow_model="claude-sonnet-4-6",
            primary_status_code=200,
            shadow_status_code=200,
            primary_latency_ms=50,
            shadow_latency_ms=90,
            primary_cost_est_usd=0.001,
            shadow_cost_est_usd=0.004,
        )
        event = experiments.routing_experiment_outcome_event(feedback)

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                meta = asyncio.run(
                    queue_policy_event_feedback(
                        store,
                        event,
                        source_surface=experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
                        queue_when_disabled=True,
                    )
                )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, attempts, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        self.assertEqual(row["source_surface"], experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["schema"], "agentflow.routing_experiment_outcome_event.v1")
        assert_managed_egress_safe(payload)
        self.assertNotIn("claude-sonnet-4-6", row["payload_json"])
        self.assertNotIn("claude-haiku-4-5-20251001", row["payload_json"])

    def test_enabled_managed_mode_can_queue_routing_experiment_for_explicit_flush(self):
        event = experiments.routing_experiment_outcome_event({
            "schema": "agentflow.routing_experiment_feedback.v1",
            "experiment_id": "exp-1",
            "sampled": True,
            "status": "compared",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-haiku-4-5-20251001",
            "shadow_model": "claude-sonnet-4-6",
            "category": "tool-result",
            "workflow_phase": "tool-execution",
            "text_chars_bucket": "2k-8k",
            "primary_status_code": 200,
            "shadow_status_code": 200,
            "output_similarity": 0.91,
            "similarity_threshold": 0.86,
            "passed_threshold": True,
            "reason_codes": ["passed"],
        })

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
                os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
                meta = asyncio.run(
                    queue_policy_event_feedback(
                        store,
                        event,
                        source_surface=experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
                        queue_when_disabled=True,
                        flush_immediately=False,
                    )
                )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, attempts, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued")
        self.assertEqual(row["source_surface"], experiments.ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)

    def _reload_with_promotion_fixture_policy(self, tmp_path: Path, *, daily_budget_usd: float = 10.0):
        config = tmp_path / "config"
        config.mkdir(exist_ok=True)
        (config / "routing_experiments.yaml").write_text(
            f"""
enabled: true
mode: shadow_candidate_pass_through
sample_rate: 1.0
daily_budget_usd: {daily_budget_usd}
providers:
  - openai
source_surfaces:
  - codex_turn
model_pairs:
  - requested_model: gpt-5-codex
    routed_model: gpt-5-mini
categories:
  - codex-turn
min_samples_for_confidence: 3
similarity_threshold: 0.86
""",
            encoding="utf-8",
        )
        os.chdir(tmp_path)
        return importlib.reload(experiments)

    def _log_shadow_promotion_sample(
        self,
        store: Store,
        *,
        idx: int,
        created_at: str | None = None,
        mode: str = "shadow_candidate_pass_through",
        passed: bool = True,
        shadow_status_code: int | None = 200,
        error: str | None = None,
        workflow_phase: str = "summary",
        fallback: bool = False,
        stream: bool = False,
    ) -> None:
        created_at = created_at or utc_now()
        similarity = 0.94 if passed else 0.7
        experiment_json = {
            "mode": mode,
            "counterfactual": mode == "shadow_candidate_pass_through",
            "shadow_only": mode == "shadow_candidate_pass_through",
            "workflow_phase": workflow_phase,
            "raw_prompt": "raw promotion secret",
            "request_id": "req-secret",
            "session_id": "session-secret",
            "file_path": "/tmp/secret.py",
            "cache_key": "cache-secret",
        }
        routing_json = {"workflow_phase": workflow_phase}
        if fallback:
            routing_json["fallback_reason"] = "rate_limited"
        store.log_routing_experiment(
            id=f"promotion-sample-{mode}-{idx}",
            call_id=f"promotion-call-{idx}",
            created_at=created_at,
            provider="openai",
            source_surface="codex_turn",
            stream=1 if stream else 0,
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            primary_model="gpt-5-codex",
            shadow_model="gpt-5-mini",
            category="codex-turn",
            routing_reason="sampled-shadow-candidate-pass-through",
            input_tokens_est=100,
            primary_status_code=200,
            shadow_status_code=shadow_status_code,
            primary_latency_ms=120,
            shadow_latency_ms=80,
            primary_output_chars=20,
            shadow_output_chars=18,
            primary_output_sha256=f"primary-{idx}",
            shadow_output_sha256=f"shadow-{idx}",
            output_similarity=similarity if shadow_status_code and shadow_status_code < 400 else None,
            passed_threshold=1 if passed else 0,
            primary_cost_est_usd=0.003,
            shadow_cost_est_usd=0.001,
            routing_json=stable_json(routing_json),
            experiment_json=stable_json(experiment_json),
            error=error,
        )

    def _promotion_report_for_samples(self, *, sample_count: int = 3, **sample_kwargs):
        tmp = TemporaryDirectory()
        tmp_path = Path(tmp.name)
        manual = self._reload_with_promotion_fixture_policy(
            tmp_path,
            daily_budget_usd=sample_kwargs.pop("daily_budget_usd", 10.0),
        )
        store = Store(str(tmp_path / "agentflow.sqlite3"))
        try:
            for idx in range(sample_count):
                self._log_shadow_promotion_sample(store, idx=idx, **sample_kwargs)
            return manual.build_routing_experiment_report(store, limit=10)
        finally:
            store.conn.close()
            tmp.cleanup()

    def test_shadow_promotion_report_marks_healthy_candidate_promote_without_raw_fields(self):
        report = self._promotion_report_for_samples()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "promote")
        self.assertEqual(candidate["promotion"]["evidence_kind"], "shadow_pass_through")
        self.assertEqual(candidate["promotion"]["promotion_scope"], "stage_local_canary_from_shadow")
        self.assertFalse(candidate["promotion"]["canary_evidence"])
        self.assertIn("promotion-thresholds-met", candidate["promotion_reason_codes"])
        self.assertEqual(report["summary"]["promotion_ready_candidates"], 1)
        self.assertEqual(report["summary"]["promotion_verdict_counts"]["promote"], 1)
        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw promotion secret",
            "req-secret",
            "session-secret",
            "/tmp/secret.py",
            "cache-secret",
            '"raw_prompt"',
            '"request_id"',
            '"session_id"',
            '"file_path"',
            '"cache_key"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_shadow_promotion_report_needs_more_samples_for_low_count(self):
        report = self._promotion_report_for_samples(sample_count=2)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "needs_more_samples")
        self.assertIn("insufficient-samples", candidate["promotion_reason_codes"])
        self.assertIn("insufficient-compared-samples", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_holds_stale_evidence(self):
        stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        report = self._promotion_report_for_samples(created_at=stale)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "hold")
        self.assertIn("stale-evidence", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_rejects_low_similarity_pass_rate(self):
        report = self._promotion_report_for_samples(passed=False)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "reject")
        self.assertIn("below-similarity-pass-rate", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_rejects_high_shadow_error_rate(self):
        report = self._promotion_report_for_samples(shadow_status_code=500, error="shadow failed")

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "reject")
        self.assertIn("shadow-error-rate-high", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_holds_budget_exhaustion(self):
        report = self._promotion_report_for_samples(daily_budget_usd=0.0)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["promotion_verdict"], "hold")
        self.assertIn("daily-budget-exhausted", candidate["promotion_reason_codes"])

    def test_shadow_promotion_report_separates_shadow_and_applied_modes(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manual = self._reload_with_promotion_fixture_policy(tmp_path)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                for idx in range(3):
                    self._log_shadow_promotion_sample(store, idx=idx, mode="shadow_candidate_pass_through")
                    self._log_shadow_promotion_sample(store, idx=idx + 10, mode="applied_routed_down")
                report = manual.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        modes = {candidate["mode"]: candidate for candidate in report["candidates"]}
        self.assertEqual(set(modes), {"shadow_candidate_pass_through", "applied_routed_down"})
        self.assertFalse(modes["shadow_candidate_pass_through"]["promotion"]["canary_evidence"])
        self.assertTrue(modes["applied_routed_down"]["promotion"]["canary_evidence"])

    def test_shadow_promotion_report_separates_streaming_scope(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manual = self._reload_with_promotion_fixture_policy(tmp_path)
            store = Store(str(tmp_path / "agentflow.sqlite3"))
            try:
                for idx in range(3):
                    self._log_shadow_promotion_sample(store, idx=idx, stream=True)
                    self._log_shadow_promotion_sample(store, idx=idx + 10, stream=False)
                report = manual.build_routing_experiment_report(store, limit=10)
            finally:
                store.conn.close()

        streams = {candidate["stream"] for candidate in report["candidates"]}
        self.assertEqual(streams, {True, False})
        for candidate in report["candidates"]:
            self.assertEqual(candidate["promotion_verdict"], "promote")

    def test_routing_experiment_report_cli_and_stats_full_expose_promotion_verdicts(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._reload_with_promotion_fixture_policy(tmp_path)
            db_path = str(tmp_path / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for idx in range(3):
                    self._log_shadow_promotion_sample(store, idx=idx)
                out = StringIO()
                code = cli.routing_experiment_report_cli(["--db", db_path], stdout=out)
                payload = json.loads(out.getvalue())
                full = asyncio.run(stats.stats_full(store))
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(payload["candidates"][0]["promotion_verdict"], "promote")
        self.assertEqual(full["routing_experiment_report"]["candidates"][0]["promotion_verdict"], "promote")
        self.assertEqual(full["summary"]["routing_experiment_promotion_ready_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
