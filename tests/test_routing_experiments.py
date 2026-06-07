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

    def test_response_comparison_produces_similarity_and_hashes(self):
        primary = {"content": [{"type": "text", "text": "summarize the build output"}]}
        shadow = {"content": [{"type": "text", "text": "summarize the build output"}]}

        result = experiments.compare_response_outputs(primary, shadow)

        self.assertEqual(result["primary_output_chars"], 26)
        self.assertEqual(result["shadow_output_chars"], 26)
        self.assertEqual(result["primary_output_sha256"], result["shadow_output_sha256"])
        self.assertEqual(result["output_similarity"], 1.0)
        self.assertTrue(result["passed_threshold"])


if __name__ == "__main__":
    unittest.main()
