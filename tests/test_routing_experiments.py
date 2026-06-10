import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agentflow_proxy.routing_experiments as experiments


class RoutingExperimentPolicyTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_ROUTING_EXPERIMENTS",
        "AGENTFLOW_ROUTING_EXPERIMENTS_ENABLED",
        "AGENTFLOW_ROUTING_EXPERIMENT_SAMPLE_RATE",
        "AGENTFLOW_ROUTING_EXPERIMENT_DAILY_BUDGET_USD",
        "AGENTFLOW_ROUTING_EXPERIMENT_SIMILARITY_THRESHOLD",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name
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

    def test_default_policy_skips_without_sampling(self):
        meta = experiments.routing_experiment_decision(
            {"model": "claude-haiku-4-5-20251001"},
            {
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "category": "tool-result",
                "text_chars": 1000,
            },
            stream=False,
            random_value=lambda: 0.0,
        )

        self.assertFalse(meta["enabled"])
        self.assertFalse(meta["sampled"])
        self.assertEqual(meta["reason"], "disabled")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_config_policy_samples_routed_down_non_streaming_call(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
sample_rate: 1.0
daily_budget_usd: 0.05
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

    def test_budget_spend_blocks_after_cap_is_exhausted(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "routing_experiments.yaml").write_text(
                """
enabled: true
sample_rate: 1.0
daily_budget_usd: 0.01
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
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["candidate_bucket"], "tool-result:claude-sonnet-4-6->claude-haiku-4-5-20251001")
        self.assertEqual(result["text_chars_bucket"], "2k-8k")
        self.assertEqual(result["output_similarity"], 0.91)
        self.assertEqual(result["primary_output_sha256"], "primary-hash")
        self.assertNotIn("text", result)
        self.assertEqual(result["reason_codes"], ["passed"])
        self.assertTrue(result["privacy"]["metadata_only"])


if __name__ == "__main__":
    unittest.main()
