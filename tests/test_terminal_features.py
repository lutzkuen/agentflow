import json
import unittest

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.terminal_features import terminal_log_features_from_text


class TerminalLogFeatureTests(unittest.TestCase):
    def _assert_feature_only(self, features):
        rendered = json.dumps(features, sort_keys=True)
        self.assertEqual(managed_egress_violations(features), [])
        for raw in (
            "secret-app",
            "pytest tests/test_secret.py",
            "Traceback (most recent call last)",
            "ERROR pid=1234",
            "npm ERR!",
        ):
            self.assertNotIn(raw, rendered)

    def test_pure_prose_has_empty_terminal_buckets(self):
        features = terminal_log_features_from_text(
            "This is a normal planning note. It discusses errors conceptually, not a captured log."
        )

        self.assertEqual(features["terminal_output_char_fraction_bucket"], "none")
        self.assertEqual(features["log_line_fraction_bucket"], "none")
        self.assertFalse(features["stack_trace_present"])
        self.assertFalse(features["test_output_present"])
        self.assertFalse(features["command_transcript_present"])
        self.assertEqual(features["error_line_count_bucket"], "zero")
        self._assert_feature_only(features)

    def test_pure_logs_bucket_as_terminal_and_log_heavy(self):
        text = "\n".join([
            "2026-06-09T20:00:00Z INFO pid=1234 secret-app started",
            "2026-06-09T20:00:01Z WARN pid=1234 secret-app retrying",
            "2026-06-09T20:00:02Z ERROR pid=1234 secret-app failed",
        ])

        features = terminal_log_features_from_text(text)

        self.assertEqual(features["terminal_output_char_fraction_bucket"], "gte_75pct")
        self.assertEqual(features["log_line_fraction_bucket"], "gte_75pct")
        self.assertEqual(features["timestamp_prefix_line_fraction_bucket"], "gte_75pct")
        self.assertEqual(features["error_line_count_bucket"], "one")
        self.assertEqual(features["repeated_log_prefix_pattern_count_bucket"], "one")
        self._assert_feature_only(features)

    def test_mixed_prose_and_logs_reports_partial_fraction(self):
        text = "\n".join([
            "Please inspect this failure.",
            "2026-06-09T20:00:00Z INFO pid=1234 secret-app started",
            "2026-06-09T20:00:01Z ERROR pid=1234 secret-app failed",
            "The likely fix is in the parser.",
        ])

        features = terminal_log_features_from_text(text)

        self.assertIn(features["terminal_output_char_fraction_bucket"], {"25_50pct", "50_75pct"})
        self.assertEqual(features["log_line_fraction_bucket"], "50_75pct")
        self.assertEqual(features["error_line_count_bucket"], "one")
        self._assert_feature_only(features)

    def test_stack_trace_and_test_failure_flags_are_detected(self):
        text = "\n".join([
            "$ pytest tests/test_secret.py",
            "============================= FAILURES =============================",
            "FAILED tests/test_secret.py::test_hidden_value - AssertionError",
            "Traceback (most recent call last):",
            "  File \"/home/lutz/project/tests/test_secret.py\", line 12, in test_hidden_value",
            "AssertionError: expected ok",
        ])

        features = terminal_log_features_from_text(text)

        self.assertTrue(features["command_transcript_present"])
        self.assertTrue(features["test_output_present"])
        self.assertTrue(features["stack_trace_present"])
        self.assertEqual(features["unique_error_signature_count_bucket"], "two_three")
        self._assert_feature_only(features)

    def test_build_package_and_server_output_classes_are_detected(self):
        text = "\n".join([
            "npm ERR! code ERESOLVE",
            "src/main.ts:14:7: error TS2322: Type 'string' is not assignable",
            "INFO: 127.0.0.1:55912 - \"GET /health HTTP/1.1\" 200 OK",
        ])

        features = terminal_log_features_from_text(text)

        buckets = features["class_count_buckets"]
        self.assertEqual(buckets["package_output"], "one")
        self.assertEqual(buckets["build_output"], "one")
        self.assertEqual(buckets["server_runtime_log"], "one")
        self._assert_feature_only(features)

    def test_code_snippet_strings_resembling_logs_are_not_classified(self):
        text = "\n".join([
            "```python",
            "message = \"2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed\"",
            "print(\"npm ERR! code ERESOLVE\")",
            "```",
        ])

        features = terminal_log_features_from_text(text)

        self.assertEqual(features["terminal_output_char_fraction_bucket"], "none")
        self.assertEqual(features["error_line_count_bucket"], "zero")
        self.assertFalse(features["test_output_present"])
        self._assert_feature_only(features)


if __name__ == "__main__":
    unittest.main()
