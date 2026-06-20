import copy
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import tokenclaw.crunch as crunch_module
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.pattern_modules import (
    CacheabilityPatternModule,
    DiffPatternModule,
    GeneratedArtifactPatternModule,
    LocalPatternModule,
    PatternCrunchResult,
    PatternDetection,
    PatternModuleContext,
    PatternModuleRegistry,
    PromptRolePatternModule,
    TabularDataPatternModule,
    TerminalLogPatternModule,
    ToolResultPatternModule,
    evaluate_pattern_modules,
    registered_pattern_modules,
)


class TestLocalCrunchModule(LocalPatternModule):
    family = "test_safe_crunch"
    version = "test.1"
    feature_schema = "agentflow.test_safe_crunch_features.v1"
    supports_local_crunch = True

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        detected = "SECRET_REPEAT" in context.text
        return PatternDetection(detected=detected, reason="test-pattern-detected" if detected else "missing")

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict:
        return {
            "schema": self.feature_schema,
            "module_family": self.family,
            "module_version": self.version,
            "detected": detection.detected,
            "repeated_marker_bucket": "gte_11",
            "privacy": {"metadata_only": True, "raw_content_included": False},
        }

    def apply_local_crunch(
        self,
        body: dict,
        context: PatternModuleContext,
        detection: PatternDetection,
    ) -> PatternCrunchResult:
        new_body = copy.deepcopy(body)
        before = json.dumps(new_body, sort_keys=True)

        def visit(value):
            if isinstance(value, str):
                return value.replace("SECRET_REPEAT " * 20, "[AgentFlow: test repeated block omitted]")
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, dict):
                return {key: visit(child) for key, child in value.items()}
            return value

        new_body = visit(new_body)
        after = json.dumps(new_body, sort_keys=True)
        return PatternCrunchResult(
            body=new_body,
            changed=before != after,
            saved_chars=max(0, len(before) - len(after)),
            reason="test-local-crunch-applied",
        )


class UnsafeFeatureModule(LocalPatternModule):
    family = "unsafe_fixture"
    version = "test.1"
    feature_schema = "agentflow.unsafe_fixture_features.v1"

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        return PatternDetection(detected=True, reason="unsafe-fixture")

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict:
        return {
            "schema": self.feature_schema,
            "prompt": "SECRET raw prompt text",
        }


class PatternModuleTests(unittest.TestCase):
    def test_default_registry_exposes_independent_families(self):
        modules = registered_pattern_modules()

        families = {module["family"] for module in modules}
        self.assertIn("cacheability", families)
        self.assertIn("diffs", families)
        self.assertIn("generated_artifacts", families)
        self.assertIn("tabular_data", families)
        self.assertIn("terminal_logs", families)
        self.assertIn("prompt_role", families)
        self.assertIn("tool_results", families)
        self.assertTrue(all(module["feature_schema"] for module in modules))

    def test_registered_modules_emit_privacy_safe_bucketed_features(self):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": "\n".join([
                    "Please inspect this secret failure.",
                    "2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
                    "Find current outstanding voucher 12345.",
                ]),
            }],
        }

        crunched, meta = evaluate_pattern_modules(body, category="chat")

        rendered = json.dumps(meta["server_features"], sort_keys=True)
        self.assertEqual(crunched, body)
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertEqual(meta["features_emitted_count"], 3)
        self.assertEqual(
            {item["family"] for item in meta["server_features"]["features"]},
            {"cacheability", "prompt_role", "terminal_logs"},
        )
        self.assertNotIn("secret-app", rendered)
        self.assertNotIn("voucher 12345", rendered)

    def test_modules_can_be_disabled_independently(self):
        body = {
            "messages": [{
                "role": "user",
                "content": "2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
            }],
        }

        _, meta = evaluate_pattern_modules(
            body,
            module_settings={"terminal_logs": {"enabled": False}, "prompt_role": {"enabled": True}},
            category="chat",
        )

        terminal = next(item for item in meta["modules"] if item["family"] == "terminal_logs")
        prompt = next(item for item in meta["modules"] if item["family"] == "prompt_role")
        self.assertEqual(terminal["status"], "skipped")
        self.assertEqual(terminal["reason"], "disabled")
        self.assertEqual(prompt["status"], "skipped")
        self.assertTrue(prompt["features_emitted"])
        self.assertEqual({item["family"] for item in meta["server_features"]["features"]}, {"prompt_role"})

    def test_local_crunch_application_reports_outcome_metadata(self):
        body = {"messages": [{"role": "user", "content": "SECRET_REPEAT " * 20 + "keep"}]}
        registry = PatternModuleRegistry([TestLocalCrunchModule()])

        crunched, meta = evaluate_pattern_modules(
            body,
            registry=registry,
            module_settings={"test_safe_crunch": {"enabled": True, "local_crunch_enabled": True}},
            category="chat",
        )

        module_meta = meta["modules"][0]
        self.assertNotEqual(crunched, body)
        self.assertIn("test repeated block omitted", crunched["messages"][0]["content"])
        self.assertEqual(module_meta["status"], "applied")
        self.assertEqual(module_meta["reason"], "test-local-crunch-applied")
        self.assertGreater(module_meta["saved_chars"], 0)
        self.assertEqual(meta["applied_count"], 1)
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])

    def test_privacy_guard_rejects_raw_like_feature_keys(self):
        body = {"messages": [{"role": "user", "content": "SECRET raw prompt text"}]}
        registry = PatternModuleRegistry([UnsafeFeatureModule()])

        crunched, meta = evaluate_pattern_modules(body, registry=registry, category="chat")

        module_meta = meta["modules"][0]
        self.assertEqual(crunched, body)
        self.assertEqual(module_meta["status"], "bypass")
        self.assertEqual(module_meta["reason"], "privacy-guard-rejected")
        self.assertFalse(module_meta["features_emitted"])
        self.assertEqual(module_meta["privacy_guard"]["blocked_keys"], ["prompt"])
        self.assertEqual(meta["features_emitted_count"], 0)

    def test_crunch_body_reports_pattern_module_features_and_honors_config_disable(self):
        saved_env = os.environ.get("AGENTFLOW_CRUNCH_RULES")
        try:
            with TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                rules_path = tmp_path / "crunch_rules.yaml"
                rules_path.write_text(
                    """
enabled: true
pattern_modules:
  terminal_logs:
    enabled: false
  prompt_role:
    enabled: true
""",
                    encoding="utf-8",
                )
                os.environ["AGENTFLOW_CRUNCH_RULES"] = str(rules_path)
                manual = importlib.reload(crunch_module)
                body = {
                    "messages": [{
                        "role": "user",
                        "content": "2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
                    }],
                }

                _, meta = manual.crunch_body(body)

                modules = meta["pattern_modules"]["modules"]
                terminal = next(item for item in modules if item["family"] == "terminal_logs")
                self.assertEqual(terminal["reason"], "disabled")
                self.assertEqual(meta["pattern_modules"]["features_emitted_count"], 1)
                self.assertEqual(
                    {item["family"] for item in meta["pattern_modules"]["server_features"]["features"]},
                    {"prompt_role"},
                )
                self.assertEqual(managed_egress_violations(meta["pattern_modules"]["server_features"]), [])
        finally:
            if saved_env is None:
                os.environ.pop("AGENTFLOW_CRUNCH_RULES", None)
            else:
                os.environ["AGENTFLOW_CRUNCH_RULES"] = saved_env
            importlib.reload(crunch_module)

    def _tool_result_feature(self, body, *, local_crunch_enabled=False):
        crunched, meta = evaluate_pattern_modules(
            body,
            registry=PatternModuleRegistry([ToolResultPatternModule()]),
            module_settings={"tool_results": {"enabled": True, "local_crunch_enabled": local_crunch_enabled}},
            category="tool-result",
        )
        feature = meta["server_features"]["features"][0]
        return crunched, meta, feature["features"]

    def _diff_feature(self, body, *, local_crunch_enabled=False):
        crunched, meta = evaluate_pattern_modules(
            body,
            registry=PatternModuleRegistry([DiffPatternModule()]),
            module_settings={"diffs": {"enabled": True, "local_crunch_enabled": local_crunch_enabled}},
            category="code-gen",
        )
        feature = meta["server_features"]["features"][0]
        return crunched, meta, feature["features"]

    def _artifact_feature(self, body, *, local_crunch_enabled=False):
        crunched, meta = evaluate_pattern_modules(
            body,
            registry=PatternModuleRegistry([GeneratedArtifactPatternModule()]),
            module_settings={
                "generated_artifacts": {
                    "enabled": True,
                    "local_crunch_enabled": local_crunch_enabled,
                }
            },
            category="long-context",
        )
        feature = meta["server_features"]["features"][0]
        return crunched, meta, feature["features"]

    def _tabular_feature(self, body):
        crunched, meta = evaluate_pattern_modules(
            body,
            registry=PatternModuleRegistry([TabularDataPatternModule()]),
            module_settings={"tabular_data": {"enabled": True, "local_crunch_enabled": True}},
            category="long-context",
        )
        feature = meta["server_features"]["features"][0]
        return crunched, meta, feature["features"]

    def _cacheability_feature(self, body, *, category="chat"):
        crunched, meta = evaluate_pattern_modules(
            body,
            registry=PatternModuleRegistry([CacheabilityPatternModule()]),
            module_settings={"cacheability": {"enabled": True}},
            category=category,
        )
        feature = meta["server_features"]["features"][0]
        return crunched, meta, feature["features"]

    def test_tool_result_module_emits_grep_features_without_raw_content(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_grep",
                    "content": "src/secret_app.py:42:SECRET_MATCH should stay local\nsrc/other.py:7:handler()",
                }],
            }],
        }

        _, meta, features = self._tool_result_feature(body)
        rendered = json.dumps(meta["server_features"], sort_keys=True)

        self.assertEqual(features["primary_result_shape"], "search_grep_results")
        self.assertTrue(features["exactness_required_hint"])
        self.assertTrue(features["current_state_evidence_hint"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("SECRET_MATCH", rendered)
        self.assertNotIn("src/secret_app.py", rendered)

    def test_tool_result_module_classifies_sql_issue_and_test_shapes(self):
        cases = [
            (
                "| id | status |\n| 1 | pending |\n| 2 | done |",
                "sql_table_rows",
            ),
            (
                "#154 Add local tool-result pattern module\nAF-221 Follow up",
                "github_ticket_list",
            ),
            (
                "FAILED tests/test_app.py::test_secret\nAssertionError: SECRET_VALUE\n2 failed, 4 passed",
                "test_output",
            ),
        ]

        for result_text, expected_shape in cases:
            body = {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_case", "content": result_text}],
                }],
            }
            with self.subTest(expected_shape=expected_shape):
                _, meta, features = self._tool_result_feature(body)
                self.assertEqual(features["primary_result_shape"], expected_shape)
                self.assertTrue(features["exactness_required_hint"])
                self.assertEqual(managed_egress_violations(meta["server_features"]), [])
                self.assertNotIn("SECRET_VALUE", json.dumps(meta["server_features"], sort_keys=True))

    def test_tool_result_command_failure_content_is_preserved_by_default(self):
        failure_text = "\n".join([
            "stderr: Traceback (most recent call last):",
            "File \"/tmp/private/app.py\", line 9, in <module>",
            "RuntimeError: SECRET_FAILURE",
            "exit code: 1",
        ])
        body = {
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_cmd", "content": failure_text}],
            }],
        }

        crunched, meta, features = self._tool_result_feature(body, local_crunch_enabled=True)

        self.assertEqual(crunched, body)
        self.assertEqual(meta["modules"][0]["status"], "skipped")
        self.assertEqual(meta["modules"][0]["reason"], "no-safe-repeated-framing")
        self.assertEqual(features["primary_result_shape"], "command_stdout_stderr")
        self.assertTrue(features["error_presence"])

    def test_tool_result_module_compacts_only_repeated_framing_lines(self):
        result_text = "\n".join([
            "Tool result:",
            "alpha=1",
            "Tool result:",
            "beta=2",
            "Tool result:",
            "/tmp/private/evidence.txt:13:SECRET_EVIDENCE",
        ])
        body = {
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_frame", "content": result_text}],
            }],
        }

        crunched, meta, features = self._tool_result_feature(body, local_crunch_enabled=True)
        crunched_text = crunched["messages"][0]["content"][0]["content"]

        self.assertNotEqual(crunched, body)
        self.assertEqual(meta["modules"][0]["status"], "applied")
        self.assertEqual(meta["modules"][0]["reason"], "safe-repeated-framing-compacted")
        self.assertGreater(meta["modules"][0]["saved_chars"], 0)
        self.assertEqual(crunched_text.count("Tool result:"), 1)
        self.assertIn("alpha=1", crunched_text)
        self.assertIn("beta=2", crunched_text)
        self.assertIn("SECRET_EVIDENCE", crunched_text)
        self.assertTrue(features["safe_local_crunch_hint"])
        self.assertNotIn("SECRET_EVIDENCE", json.dumps(meta["server_features"], sort_keys=True))

    def test_tool_result_module_detects_mixed_prompt_tool_result_inputs(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Use the current grep evidence, but keep secrets local."},
                    {"type": "tool_result", "tool_use_id": "toolu_mixed", "content": "app.py:10:SECRET_MIXED"},
                ],
            }],
        }

        _, meta, features = self._tool_result_feature(body)

        self.assertTrue(features["mixed_prompt_tool_result"])
        self.assertEqual(features["result_count_bucket"], "one")
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("SECRET_MIXED", json.dumps(meta["server_features"], sort_keys=True))

    def test_diff_module_emits_small_diff_features_without_raw_content(self):
        diff_text = "\n".join([
            "diff --git a/src/private.py b/src/private.py",
            "--- a/src/private.py",
            "+++ b/src/private.py",
            "@@ -1,3 +1,3 @@",
            " keep context",
            "-SECRET_OLD = 1",
            "+SECRET_NEW = 2",
        ])
        body = {"messages": [{"role": "user", "content": f"Review this patch:\n{diff_text}"}]}

        crunched, meta, features = self._diff_feature(body, local_crunch_enabled=True)
        rendered = json.dumps(meta["server_features"], sort_keys=True)

        self.assertEqual(crunched, body)
        self.assertTrue(features["diff_present"])
        self.assertFalse(features["multi_file_diff"])
        self.assertEqual(features["added_removed_line_count_bucket"], "two_three")
        self.assertTrue(features["changed_hunks_preserved_by_default"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("SECRET_NEW", rendered)
        self.assertNotIn("src/private.py", rendered)

    def test_diff_module_compacts_repeated_context_in_large_multifile_diff(self):
        repeated_context = " repeated unchanged generated context line"
        diff_text = "\n".join([
            "diff --git a/a.py b/a.py",
            "--- a/a.py",
            "+++ b/a.py",
            "@@ -1,4 +1,4 @@",
            repeated_context,
            "-old_a = 'SECRET_A'",
            "+new_a = 'SECRET_A'",
            repeated_context,
            "diff --git a/b.py b/b.py",
            "--- a/b.py",
            "+++ b/b.py",
            "@@ -10,4 +10,4 @@",
            repeated_context,
            "-old_b = 'SECRET_B'",
            "+new_b = 'SECRET_B'",
            repeated_context,
        ])
        body = {"messages": [{"role": "user", "content": diff_text}]}

        crunched, meta, features = self._diff_feature(body, local_crunch_enabled=True)
        crunched_text = crunched["messages"][0]["content"]

        self.assertNotEqual(crunched, body)
        self.assertTrue(features["multi_file_diff"])
        self.assertEqual(meta["modules"][0]["status"], "applied")
        self.assertEqual(meta["modules"][0]["reason"], "safe-repeated-diff-context-compacted")
        self.assertIn("+new_a = 'SECRET_A'", crunched_text)
        self.assertIn("-old_b = 'SECRET_B'", crunched_text)
        self.assertLess(crunched_text.count(repeated_context), diff_text.count(repeated_context))
        self.assertNotIn("SECRET_A", json.dumps(meta["server_features"], sort_keys=True))

    def test_diff_module_preserves_exact_request(self):
        diff_text = "\n".join([
            "Please keep the exact full file diff.",
            "diff --git a/app.py b/app.py",
            "--- a/app.py",
            "+++ b/app.py",
            "@@ -1,4 +1,4 @@",
            " repeated unchanged context line",
            "-before()",
            "+after()",
            " repeated unchanged context line",
        ])
        body = {"messages": [{"role": "user", "content": diff_text}]}

        crunched, meta, features = self._diff_feature(body, local_crunch_enabled=True)

        self.assertEqual(crunched, body)
        self.assertTrue(features["exact_artifact_requested_hint"])
        self.assertEqual(meta["modules"][0]["reason"], "exact-diff-request-preserved")

    def test_generated_artifact_module_preserves_dependency_lockfile_prompt(self):
        lock_text = "\n".join([
            "Resolve this dependency conflict in the exact package-lock.json.",
            '"lockfileVersion": 3,',
            '"node_modules/private-package": {"version": "1.2.3"}',
        ])
        body = {"messages": [{"role": "user", "content": lock_text}]}

        crunched, meta, features = self._artifact_feature(body, local_crunch_enabled=True)
        rendered = json.dumps(meta["server_features"], sort_keys=True)

        self.assertEqual(crunched, body)
        self.assertTrue(features["lockfile_present"])
        self.assertTrue(features["dependency_resolution_task_hint"])
        self.assertTrue(features["exact_artifact_preserved_by_default"])
        self.assertEqual(meta["modules"][0]["reason"], "exact-artifact-request-preserved")
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("private-package", rendered)

    def test_generated_artifact_module_compacts_generated_minified_bulk(self):
        generated_line = "// Code generated by protoc-gen-go. DO NOT EDIT."
        minified = "function x(){" + ("var_a=1;" * 180) + "return a;}"
        artifact_text = "\n".join([
            generated_line,
            "package privatepb",
            generated_line,
            minified,
        ])
        body = {"messages": [{"role": "user", "content": artifact_text}]}

        crunched, meta, features = self._artifact_feature(body, local_crunch_enabled=True)
        crunched_text = crunched["messages"][0]["content"]

        self.assertNotEqual(crunched, body)
        self.assertEqual(features["generated_file_likelihood_bucket"], "high")
        self.assertEqual(features["minified_content_fraction_bucket"], "25_50pct")
        self.assertEqual(meta["modules"][0]["reason"], "safe-generated-artifact-bulk-compacted")
        self.assertEqual(crunched_text.count(generated_line), 1)
        self.assertIn("generated/minified bulk omitted", crunched_text)
        self.assertNotIn("privatepb", json.dumps(meta["server_features"], sort_keys=True))

    def test_tabular_module_detects_markdown_tables_without_raw_rows(self):
        table = "\n".join([
            "Summarize the total by status.",
            "| customer | status | amount |",
            "| --- | --- | --- |",
            "| SecretCo | pending | 10 |",
            "| PrivateLLC | done | 20 |",
            "| HiddenInc | pending | 30 |",
        ])
        body = {"messages": [{"role": "user", "content": table}]}

        crunched, meta, features = self._tabular_feature(body)
        rendered = json.dumps(meta["server_features"], sort_keys=True)

        self.assertEqual(crunched, body)
        self.assertTrue(features["table_present"])
        self.assertEqual(features["table_shape"], "markdown_table")
        self.assertEqual(features["row_count_bucket"], "two_three")
        self.assertEqual(features["column_count_bucket"], "two_three")
        self.assertEqual(features["schema_repetition_bucket"], "gte_75pct")
        self.assertTrue(features["aggregation_requested_hint"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("SecretCo", rendered)
        self.assertNotIn("PrivateLLC", rendered)

    def test_tabular_module_detects_csv_blocks(self):
        csv_text = "\n".join([
            "Please summarize this export.",
            "name,status,amount",
            "SecretCo,pending,10",
            "PrivateLLC,done,20",
            "HiddenInc,pending,30",
        ])
        body = {"messages": [{"role": "user", "content": csv_text}]}

        _, meta, features = self._tabular_feature(body)

        self.assertEqual(features["table_shape"], "csv_block")
        self.assertEqual(features["row_count_bucket"], "two_three")
        self.assertEqual(features["column_count_bucket"], "two_three")
        self.assertTrue(features["aggregation_requested_hint"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("SecretCo", json.dumps(meta["server_features"], sort_keys=True))

    def test_tabular_module_detects_sql_result_rows(self):
        sql_rows = "\n".join([
            "SELECT id, status FROM private_orders;",
            "+----+---------+",
            "| id | status  |",
            "+----+---------+",
            "| 41 | pending |",
            "| 42 | done    |",
            "+----+---------+",
        ])
        body = {"messages": [{"role": "user", "content": sql_rows}]}

        _, meta, features = self._tabular_feature(body)

        self.assertEqual(features["table_shape"], "sql_result_rows")
        self.assertTrue(features["time_sensitive_hint"])
        self.assertTrue(features["exact_rows_preserved_by_default"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("private_orders", json.dumps(meta["server_features"], sort_keys=True))

    def test_tabular_module_detects_json_arrays_by_shape_only(self):
        json_rows = (
            "Count statuses in this JSON array: "
            '[{"id": 1, "name": "SecretCo", "status": "pending"}, '
            '{"id": 2, "name": "PrivateLLC", "status": "done"}, '
            '{"id": 3, "name": "HiddenInc", "status": "pending"}]'
        )
        body = {"messages": [{"role": "user", "content": json_rows}]}

        _, meta, features = self._tabular_feature(body)

        self.assertEqual(features["table_shape"], "json_object_array")
        self.assertEqual(features["row_count_bucket"], "two_three")
        self.assertEqual(features["column_count_bucket"], "two_three")
        self.assertEqual(features["schema_repetition_bucket"], "gte_75pct")
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertNotIn("SecretCo", json.dumps(meta["server_features"], sort_keys=True))

    def test_tabular_module_preserves_exact_lookup_and_reconciliation(self):
        exact_lookup = "\n".join([
            "Find the exact row for invoice #4242.",
            "invoice,status,total",
            "4241,paid,12",
            "4242,open,99",
            "4243,paid,44",
        ])
        reconciliation = "\n".join([
            "Reconcile these rows and identify mismatches.",
            "| id | ledger | bank |",
            "| --- | --- | --- |",
            "| 1 | 10 | 10 |",
            "| 2 | 15 | 14 |",
        ])

        for text, expected_hint in ((exact_lookup, "exact_lookup_requested_hint"), (reconciliation, "reconciliation_requested_hint")):
            with self.subTest(expected_hint=expected_hint):
                crunched, meta, features = self._tabular_feature({"messages": [{"role": "user", "content": text}]})
                self.assertEqual(crunched["messages"][0]["content"], text)
                self.assertTrue(features[expected_hint])
                self.assertTrue(features["exact_rows_preserved_by_default"])
                self.assertEqual(features["sample_sufficient_likelihood_bucket"], "low")
                self.assertEqual(meta["modules"][0]["status"], "skipped")
                self.assertEqual(managed_egress_violations(meta["server_features"]), [])

    def test_cacheability_module_marks_static_qa_high_cacheability(self):
        body = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}

        _, meta, features = self._cacheability_feature(body)

        self.assertEqual(features["cacheability_bucket"], "high")
        self.assertEqual(features["deterministic_answer_likelihood_bucket"], "high")
        self.assertTrue(features["static_information_hint"])
        self.assertTrue(features["exact_cache_candidate_hint"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])

    def test_cacheability_module_marks_current_state_low_cacheability(self):
        body = {"messages": [{"role": "user", "content": "What are the latest open issues today?"}]}

        _, meta, features = self._cacheability_feature(body)

        self.assertEqual(features["cacheability_bucket"], "low")
        self.assertEqual(features["deterministic_answer_likelihood_bucket"], "low")
        self.assertTrue(features["time_sensitive_hint"])
        self.assertFalse(features["exact_cache_candidate_hint"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])

    def test_cacheability_module_marks_user_specific_low_cacheability(self):
        body = {"messages": [{"role": "user", "content": "Explain my account's current billing status."}]}

        _, meta, features = self._cacheability_feature(body)

        self.assertEqual(features["cacheability_bucket"], "low")
        self.assertTrue(features["user_specific_hint"])
        self.assertTrue(features["time_sensitive_hint"])
        self.assertFalse(features["exact_cache_candidate_hint"])
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])


if __name__ == "__main__":
    unittest.main()
