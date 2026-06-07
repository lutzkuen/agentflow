import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import agentflow_proxy.router as router_module
from agentflow_proxy.router import HAIKU_DEFAULT, SONNET_DEFAULT, route_model


class RouterTest(unittest.TestCase):
    def test_tool_result_sonnet_routes_to_haiku_without_thinking(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                }
            ],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, HAIKU_DEFAULT)
        self.assertEqual(meta["category"], "tool-result")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_thinking_tool_result_keeps_requested_model(self):
        body = {
            "model": SONNET_DEFAULT,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                }
            ],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "tool-result")
        self.assertEqual(meta["reason"], "keep requested model for thinking request")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_tool_result_with_assistant_thinking_history_keeps_requested_model(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reasoning"},
                        {"type": "text", "text": "done"},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            ],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "tool-result")
        self.assertEqual(meta["reason"], "keep requested model for thinking request")

    def test_manual_routing_source_reported_for_thinking_keep(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: manual test rule
""",
                encoding="utf-8",
            )
            try:
                with patch.dict(os.environ, {"AGENTFLOW_ROUTING_RULES": str(rules_path)}):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "thinking": {"type": "enabled", "budget_tokens": 4096},
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["reason"], "keep requested model for thinking request")
                    self.assertEqual(meta["policy_source"], "local-manual")
            finally:
                importlib.reload(router_module)

    def test_max_tokens_lte_rule_requires_explicit_max_tokens(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
      max_tokens_lte: 64
    action:
      route_to: haiku
      reason: bounded small response
""",
                encoding="utf-8",
            )
            try:
                with patch.dict(os.environ, {"AGENTFLOW_ROUTING_RULES": str(rules_path)}):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [{"role": "user", "content": "Say ok."}],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["reason"], "keep requested model")
            finally:
                importlib.reload(router_module)

    def test_max_tokens_lte_rule_matches_when_explicitly_bounded(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
      max_tokens_lte: 64
    action:
      route_to: haiku
      reason: bounded small response
""",
                encoding="utf-8",
            )
            try:
                with patch.dict(os.environ, {"AGENTFLOW_ROUTING_RULES": str(rules_path)}):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": "Say ok."}],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    self.assertEqual(meta["reason"], "bounded small response")
            finally:
                importlib.reload(router_module)

    def test_disabled_thinking_with_assistant_thinking_history_keeps_requested_model(self):
        body = {
            "model": SONNET_DEFAULT,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reasoning"},
                        {"type": "text", "text": "done"},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            ],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "tool-result")
        self.assertEqual(meta["reason"], "keep requested model for thinking request")

    def test_openai_routing_is_disabled_by_default(self):
        routed, meta = router_module.route_openai_model({
            "model": "gpt-5-codex",
            "input": "small task",
        })

        self.assertEqual(routed, "gpt-5-codex")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["provider"], "openai")

    def test_openai_routing_stays_inside_openai_models(self):
        try:
            with patch.dict(os.environ, {"AGENTFLOW_OPENAI_ROUTING": "1"}):
                manual_router = importlib.reload(router_module)

                routed, meta = manual_router.route_openai_model({
                    "model": manual_router.OPENAI_LARGE_DEFAULT,
                    "input": "small task",
                })

                self.assertEqual(routed, manual_router.OPENAI_SMALL_DEFAULT)
                self.assertTrue(meta["enabled"])
                self.assertEqual(meta["provider"], "openai")
                self.assertNotIn("claude", routed)
        finally:
            importlib.reload(router_module)

    def test_small_non_tool_sonnet_routes_to_haiku_under_10000_chars(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "```python\n" + ("print('x')\n" * 860) + "```"}],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, HAIKU_DEFAULT)
        self.assertEqual(meta["reason"], "small non-tool Sonnet request routed to Haiku")
        self.assertEqual(meta["category"], "code-gen")
        self.assertFalse(meta["has_tools"])
        self.assertLess(meta["text_chars"], 10000)

    def test_small_non_tool_sonnet_threshold_excludes_10000_chars(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "a" * 10000}],
        }

        with patch.dict(os.environ, {}, clear=True):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["reason"], "keep requested model")
        self.assertEqual(meta["text_chars"], 10000)

    def test_midsize_non_tool_sonnet_routes_to_haiku_when_enabled(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "summarize this report\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {"AGENTFLOW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, HAIKU_DEFAULT)
        self.assertEqual(meta["reason"], "midsize non-tool Sonnet request routed to Haiku")
        self.assertEqual(meta["category"], "chat")
        self.assertFalse(meta["has_tools"])

    def test_midsize_non_tool_sonnet_stays_requested_by_default(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "summarize this report\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {}, clear=True):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["reason"], "keep requested model")

    def test_midsize_code_gen_sonnet_does_not_route_to_haiku(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "```python\n" + ("print('x')\n" * 1200) + "```"}],
        }

        with patch.dict(os.environ, {"AGENTFLOW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "code-gen")
        self.assertEqual(meta["reason"], "keep requested model")

    def test_large_non_tool_sonnet_above_midsize_window_stays_requested(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "summarize all of this\n" + ("a" * 31000)}],
        }

        with patch.dict(os.environ, {"AGENTFLOW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "chat")
        self.assertEqual(meta["reason"], "keep requested model")

    def test_midsize_tool_request_sonnet_does_not_route_as_non_tool(self):
        body = {
            "model": SONNET_DEFAULT,
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "read the files\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {"AGENTFLOW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, HAIKU_DEFAULT)
        self.assertEqual(meta["category"], "tool-light")
        self.assertEqual(meta["reason"], "tool-light Sonnet request routed to Haiku")

    def test_midsize_thinking_request_keeps_requested_model_when_enabled(self):
        body = {
            "model": SONNET_DEFAULT,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [{"role": "user", "content": "think about this\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {"AGENTFLOW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["reason"], "keep requested model for thinking request")


if __name__ == "__main__":
    unittest.main()
