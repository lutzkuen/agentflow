import asyncio
import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import agentflow_proxy.crunch as crunch_module


class CrunchRulesTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CRUNCH",
        "AGENTFLOW_CRUNCH_THRESHOLD_CHARS",
        "AGENTFLOW_PROMPT_CACHE",
        "AGENTFLOW_PROMPT_CACHE_MIN_CHARS",
        "AGENTFLOW_CRUNCH_RULES",
        "AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT",
        "AGENTFLOW_HAIKU_SUMMARY_MODEL",
        "AGENTFLOW_HAIKU_SUMMARY_MIN_REQUEST_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MIN_SUMMARIZED_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_TURNS",
        "AGENTFLOW_HAIKU_SUMMARY_KEEP_RECENT_TURNS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_SUMMARY_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_SOURCE_CHARS",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(crunch_module)

    def test_default_crunch_policy_reports_bundled_local_default_source(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CRUNCH": "1",
                "AGENTFLOW_CRUNCH_THRESHOLD_CHARS": "24000",
                "AGENTFLOW_PROMPT_CACHE": "1",
                "AGENTFLOW_PROMPT_CACHE_MIN_CHARS": "4096",
                "AGENTFLOW_CRUNCH_RULES": "",
            },
        ):
            manual = importlib.reload(crunch_module)

            _, meta = manual.crunch_body({"model": "claude-sonnet-4-6", "messages": []})

            self.assertTrue(meta["enabled"])
            self.assertEqual(meta["policy_source"], "local-default")
            self.assertTrue(meta["rule_path"].endswith("agentflow_proxy/crunch_rules.yaml"))

    def test_config_crunch_rules_can_disable_crunch_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text("enabled: false\n", encoding="utf-8")
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            body = {"messages": [{"role": "user", "content": "hello"}]}
            crunched, meta = manual.crunch_body(body)

            self.assertIs(crunched, body)
            self.assertFalse(meta["enabled"])
            self.assertFalse(meta["changed"])
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["rule_path"], str(config / "crunch_rules.yaml"))

    def test_config_crunch_rules_can_change_shortening_threshold_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 10
prompt_cache:
  enabled: true
  min_chars: 4096
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            long_text = "alpha " * 1600
            body = {
                "messages": [
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": "one"},
                    {"role": "user", "content": "two"},
                    {"role": "assistant", "content": "three"},
                    {"role": "user", "content": "four"},
                ]
            }
            crunched, meta = manual.crunch_body(body)

            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["threshold_chars"], 10)
            self.assertEqual(meta["long_blocks_shortened"], 1)
            self.assertIn("middle of long older text block omitted", crunched["messages"][0]["content"])

    def test_thinking_near_duplicate_dedup_removes_older_assistant_block(self):
        manual = importlib.reload(crunch_module)
        base_words = [f"token{i}" for i in range(520)]
        newer_words = list(base_words)
        newer_words[200] = "updated-token"
        older_thinking = " ".join(base_words)
        newer_thinking = " ".join(newer_words)
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": older_thinking, "signature": "older-signature"},
                        {"type": "tool_use", "id": "tool-1", "name": "read", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": newer_thinking, "signature": "newer-signature"},
                        {"type": "tool_use", "id": "tool-2", "name": "read", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-2", "content": "ok"}]},
            ]
        }

        crunched, meta = manual.crunch_body(body)

        self.assertTrue(meta["changed"])
        self.assertGreater(meta["saved_chars"], 2000)
        self.assertEqual(meta["thinking_near_duplicate_blocks_removed"], 1)
        self.assertEqual(crunched["messages"][0]["content"], [
            {"type": "tool_use", "id": "tool-1", "name": "read", "input": {}},
        ])
        self.assertEqual(crunched["messages"][2]["content"][0]["thinking"], newer_thinking)
        self.assertEqual(crunched["messages"][2]["content"][0]["signature"], "newer-signature")
        self.assertEqual(crunched["messages"][2]["content"][1]["type"], "tool_use")

    def test_thinking_near_duplicate_dedup_preserves_latest_assistant_block(self):
        manual = importlib.reload(crunch_module)
        thinking = " ".join(f"token{i}" for i in range(520))
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": thinking, "signature": "older-signature"},
                        {"type": "text", "text": "older done"},
                    ],
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": thinking, "signature": "latest-signature"},
                        {"type": "tool_use", "id": "tool-1", "name": "read", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}]},
            ]
        }

        crunched, meta = manual.crunch_body(body)

        latest = crunched["messages"][2]["content"][0]
        self.assertEqual(meta["thinking_near_duplicate_blocks_removed"], 1)
        self.assertEqual(latest["type"], "thinking")
        self.assertEqual(latest["thinking"], thinking)
        self.assertEqual(latest["signature"], "latest-signature")
        self.assertTrue(meta["thinking_deduplication"]["skip_latest_assistant"])

    def test_config_crunch_rules_can_disable_thinking_dedup_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
thinking_deduplication:
  enabled: false
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            thinking = " ".join(f"token{i}" for i in range(520))
            body = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": thinking},
                            {"type": "text", "text": "older"},
                        ],
                    },
                    {"role": "user", "content": "continue"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": thinking},
                            {"type": "text", "text": "latest"},
                        ],
                    },
                ]
            }

            crunched, meta = manual.crunch_body(body)

            self.assertFalse(meta["thinking_deduplication"]["enabled"])
            self.assertEqual(meta["thinking_near_duplicate_blocks_removed"], 0)
            self.assertEqual(crunched["messages"][0]["content"][0]["type"], "thinking")

    def test_old_context_summarization_is_disabled_by_default(self):
        manual = importlib.reload(crunch_module)
        plan, meta = manual.old_context_summary_plan(
            {"messages": [{"role": "user", "content": "old text " * 10000}]},
            exact_cache_enabled=True,
        )

        self.assertIsNone(plan)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "disabled")

    def test_old_context_summarization_requires_exact_cache(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            plan, meta = manual.old_context_summary_plan(
                {
                    "messages": [
                        {"role": "user", "content": "old text " * 20},
                        {"role": "assistant", "content": "recent"},
                    ]
                },
                exact_cache_enabled=False,
            )

            self.assertIsNone(plan)
            self.assertTrue(meta["enabled"])
            self.assertEqual(meta["reason"], "exact-cache-required")

    def test_old_context_summarization_plans_only_old_non_tool_turns(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 3
  keep_recent_turns: 2
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "user", "content": "alpha " * 20},
                    {"role": "assistant", "content": [{"type": "text", "text": "beta " * 20}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file"}]},
                    {"role": "user", "content": "gamma " * 20},
                    {"role": "assistant", "content": "recent assistant"},
                    {"role": "user", "content": "recent user"},
                ]
            }

            plan, meta = manual.old_context_summary_plan(body, exact_cache_enabled=True)

            self.assertIsNotNone(plan)
            self.assertEqual(plan["candidate_indexes"], [0, 1, 3])
            self.assertEqual(meta["status"], "planned")
            self.assertEqual(meta["eligible_turns"], 3)
            self.assertIn("claude-haiku-4-5-20251001", plan["summary_request"]["model"])

    def test_maybe_summarize_old_context_uses_cached_summary_without_fetch(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "user", "content": "alpha " * 40},
                    {"role": "assistant", "content": "beta " * 40},
                    {"role": "user", "content": "recent"},
                ]
            }
            plan, _ = manual.old_context_summary_plan(body, exact_cache_enabled=True)
            cache = {plan["cache_key"]: {"summary": "durable facts only"}}

            async def fail_fetch(_request):
                raise AssertionError("fetch should not run on summary cache hit")

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=True,
                get_cached_summary=cache.get,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fail_fetch,
            ))

            self.assertTrue(meta["changed"])
            self.assertEqual(meta["status"], "applied")
            self.assertEqual(meta["reason"], "summary-cache-hit")
            self.assertTrue(meta["summary_cache_hit"])
            self.assertIn("durable facts only", summarized["messages"][0]["content"][0]["text"])
            self.assertEqual(summarized["messages"][-1]["content"], "recent")

    def test_maybe_summarize_old_context_fetches_and_caches_summary(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 24000
old_context_summarization:
  enabled: true
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: 2
  keep_recent_turns: 1
  max_summary_chars: 1000
  max_source_chars: 20000
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)
            body = {
                "messages": [
                    {"role": "user", "content": "alpha " * 40},
                    {"role": "assistant", "content": "beta " * 40},
                    {"role": "user", "content": "recent"},
                ]
            }
            cache = {}
            fetch_requests = []

            async def fetch(request):
                fetch_requests.append(request)
                return {
                    "summary": "fetched compact summary",
                    "summary_input_tokens": 100,
                    "summary_output_tokens": 20,
                    "summary_cost_est_usd": 0.0008,
                    "summary_status_code": 200,
                }

            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=True,
                get_cached_summary=cache.get,
                set_cached_summary=lambda key, value: cache.__setitem__(key, value),
                fetch_summary=fetch,
            ))

            self.assertEqual(len(fetch_requests), 1)
            self.assertEqual(len(cache), 1)
            self.assertEqual(meta["reason"], "summary-created")
            self.assertEqual(meta["summary_input_tokens"], 100)
            self.assertAlmostEqual(meta["summary_cost_est_usd"], 0.0008)
            self.assertIn("fetched compact summary", summarized["messages"][0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
