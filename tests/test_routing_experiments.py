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
