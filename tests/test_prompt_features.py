import json
import unittest

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.prompt_features import prompt_difficulty_features_from_text


class PromptDifficultyFeatureTests(unittest.TestCase):
    def _assert_feature_only(self, features):
        rendered = json.dumps(features, sort_keys=True)
        self.assertEqual(managed_egress_violations(features), [])
        for raw in (
            "secret voucher 12345",
            "/home/lutz/project/app.py",
            "SELECT * FROM vouchers",
            "https://example.test/private",
        ):
            self.assertNotIn(raw, rendered)

    def test_acknowledgement_with_domain_word_is_safe(self):
        features = prompt_difficulty_features_from_text("Thank you for your help with the secret voucher 12345.")

        self.assertEqual(features["task_intent"], "acknowledgement")
        self.assertEqual(features["prompt_role"], "acknowledgement")
        self.assertFalse(features["requires_current_state"])
        self.assertEqual(features["external_source_dependency"], "none")
        self.assertEqual(features["tool_or_data_dependency_likelihood"], "none")
        self.assertEqual(features["answerability_from_prompt_only"], "likely")
        self.assertEqual(features["downgrade_risk"], "safe")
        self._assert_feature_only(features)

    def test_current_outstanding_lookup_with_same_domain_word_blocks_downgrade(self):
        features = prompt_difficulty_features_from_text("Find current outstanding secret voucher 12345.")

        self.assertEqual(features["task_intent"], "data_lookup")
        self.assertTrue(features["requires_current_state"])
        self.assertEqual(features["external_source_dependency"], "unknown")
        self.assertEqual(features["tool_or_data_dependency_likelihood"], "high")
        self.assertTrue(features["verification_required"])
        self.assertEqual(features["answerability_from_prompt_only"], "unlikely")
        self.assertEqual(features["downgrade_risk"], "block")
        self._assert_feature_only(features)

    def test_simple_question_is_answerable_and_safe(self):
        features = prompt_difficulty_features_from_text("What is a voucher?")

        self.assertEqual(features["task_intent"], "question")
        self.assertEqual(features["actionability"], "informational")
        self.assertEqual(features["answerability_from_prompt_only"], "likely")
        self.assertEqual(features["downgrade_risk"], "safe")
        self._assert_feature_only(features)

    def test_debugging_and_repository_investigation_are_high_risk(self):
        features = prompt_difficulty_features_from_text(
            "Debug the failing tests in /home/lutz/project/app.py and inspect the repository diff."
        )

        self.assertEqual(features["task_intent"], "debugging")
        self.assertEqual(features["external_source_dependency"], "repository")
        self.assertEqual(features["multi_step_likelihood_bucket"], "medium")
        self.assertEqual(features["tool_or_data_dependency_likelihood"], "high")
        self.assertEqual(features["downgrade_risk"], "block")
        self._assert_feature_only(features)

    def test_docs_db_web_and_multi_step_tasks_are_conservative(self):
        cases = [
            ("Check the docs and confirm the latest API behavior.", "docs"),
            ("Run SELECT * FROM vouchers and reconcile the current rows.", "database"),
            ("Look up the latest status at https://example.test/private.", "web"),
            ("First inspect the logs, then compare the failing output and verify the fix.", "logs"),
        ]

        for text, dependency in cases:
            with self.subTest(dependency=dependency):
                features = prompt_difficulty_features_from_text(text)
                self.assertEqual(features["external_source_dependency"], dependency)
                self.assertIn(features["downgrade_risk"], {"caution", "block"})
                self.assertTrue(features["verification_required"])
                self._assert_feature_only(features)


if __name__ == "__main__":
    unittest.main()
