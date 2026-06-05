import unittest

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


if __name__ == "__main__":
    unittest.main()
