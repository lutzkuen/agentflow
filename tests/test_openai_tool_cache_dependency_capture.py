from __future__ import annotations

import unittest

from agentflow_proxy.openai_proxy import _should_collect_openai_file_dependency_evidence


class OpenAIToolCacheDependencyCaptureTests(unittest.TestCase):
    def test_collects_dependency_evidence_when_routing_wins_before_cache(self) -> None:
        self.assertTrue(
            _should_collect_openai_file_dependency_evidence(
                can_cache=False,
                has_tool_blocks=True,
                selected_before_cache="routing",
                category="tool-light",
                stream=False,
            )
        )

    def test_collects_dependency_evidence_for_tool_category_conflicts(self) -> None:
        self.assertTrue(
            _should_collect_openai_file_dependency_evidence(
                can_cache=False,
                has_tool_blocks=False,
                selected_before_cache="old_context_summary",
                category="tool-result",
                stream=False,
            )
        )

    def test_collects_dependency_evidence_for_streaming_tool_shapes(self) -> None:
        self.assertTrue(
            _should_collect_openai_file_dependency_evidence(
                can_cache=False,
                has_tool_blocks=False,
                selected_before_cache=None,
                category="tool-heavy",
                stream=True,
            )
        )

    def test_plain_non_cacheable_chat_does_not_scan_dependencies(self) -> None:
        self.assertFalse(
            _should_collect_openai_file_dependency_evidence(
                can_cache=False,
                has_tool_blocks=False,
                selected_before_cache="routing",
                category="chat",
                stream=False,
            )
        )

    def test_cacheable_non_tool_shape_still_collects_dependency_evidence(self) -> None:
        self.assertTrue(
            _should_collect_openai_file_dependency_evidence(
                can_cache=True,
                has_tool_blocks=False,
                selected_before_cache=None,
                category="chat",
                stream=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
