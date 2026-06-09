import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import agentflow_proxy.router as router_module
from agentflow_proxy.store import Store, stable_json, utc_now
from agentflow_proxy.router import HAIKU_DEFAULT, SONNET_DEFAULT, classify_workflow_phase, route_model


class RouterTest(unittest.TestCase):
    def _reload_with_routing_yaml(self, tmp: str, yaml_text: str, extra_env: dict[str, str] | None = None):
        rules_path = Path(tmp) / "routing_rules.yaml"
        rules_path.write_text(yaml_text, encoding="utf-8")
        env = {"AGENTFLOW_ROUTING_RULES": str(rules_path)}
        env.update(extra_env or {})
        return patch.dict(os.environ, env)

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
        self.assertEqual(meta["workflow_phase"], "tool-execution")
        self.assertEqual(meta["workflow_phase_reason"], "last-user-tool-result")
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
        self.assertEqual(meta["workflow_phase"], "thinking")
        self.assertEqual(meta["workflow_phase_reason"], "thinking-flag-or-history")
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
        self.assertEqual(meta["workflow_phase"], "thinking")
        self.assertEqual(meta["reason"], "keep requested model for thinking request")

    def test_workflow_phase_classifier_identifies_planning_turn(self):
        body = {
            "model": SONNET_DEFAULT,
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "Inspect the project and plan the fix."}],
        }

        phase = classify_workflow_phase(body, "tool-light")

        self.assertEqual(phase["workflow_phase"], "planning")
        self.assertEqual(phase["workflow_phase_reason"], "early-tool-capable-user-turn")
        self.assertEqual(phase["workflow_phase_confidence"], "medium")

    def test_workflow_phase_classifier_identifies_verification_followup(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [
                {"role": "user", "content": "Implement the parser."},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Run the tests and verify the failure is fixed."},
            ],
        }

        phase = classify_workflow_phase(body, "chat")

        self.assertEqual(phase["workflow_phase"], "verification")
        self.assertEqual(phase["workflow_phase_reason"], "verification-intent-text")

    def test_workflow_phase_classifier_identifies_short_summary(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [
                {"role": "user", "content": "Make the change."},
                {"role": "assistant", "content": "Changed the files."},
                {"role": "user", "content": "Summarize the result."},
            ],
        }

        phase = classify_workflow_phase(body, "short-completion")

        self.assertEqual(phase["workflow_phase"], "summary")
        self.assertEqual(phase["workflow_phase_reason"], "summary-intent-text")

    def test_workflow_phase_classifier_keeps_simple_chat_distinct(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "What does this setting do?"}],
        }

        phase = classify_workflow_phase(body, "chat")

        self.assertEqual(phase["workflow_phase"], "chat")
        self.assertEqual(phase["workflow_phase_reason"], "non-tool-chat-category")

    def test_workflow_phase_classifier_returns_unknown_without_messages(self):
        phase = classify_workflow_phase({"model": SONNET_DEFAULT}, None)

        self.assertEqual(phase["workflow_phase"], "unknown")
        self.assertEqual(phase["workflow_phase_reason"], "missing-messages")

    def test_phase_metadata_does_not_change_existing_routing_decision(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "a" * 10000}],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["reason"], "keep requested model")
        self.assertEqual(meta["workflow_phase"], "chat")

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

    def test_phase_canary_applies_seeded_eligible_tool_execution(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  canary_fraction: 1.0
  holdout_fraction: 0.0
  excluded_categories: []
  safety_stop:
    enabled: false
rules: []
""",
                ):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    self.assertEqual(meta["reason"], "phase canary selected Sonnet-to-Haiku route")
                    self.assertEqual(meta["phase_canary"]["status"], "applied")
                    self.assertEqual(meta["phase_canary"]["cohort"], "applied")
                    self.assertEqual(meta["phase_canary"]["policy_id"], "test-phase-canary")
                    self.assertEqual(meta["phase_canary"]["workflow_phase"], "tool-execution")
                    self.assertIn("cohort_hash", meta["phase_canary"])
                    self.assertNotIn("content", stable_json(meta["phase_canary"]["cohort_features"]))
            finally:
                importlib.reload(router_module)

    def test_phase_canary_holdout_keeps_requested_model(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  canary_fraction: 0.0
  holdout_fraction: 1.0
  excluded_categories: []
  safety_stop:
    enabled: false
rules: []
""",
                ):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["reason"], "phase canary holdout; keep requested model")
                    self.assertEqual(meta["phase_canary"]["status"], "holdout")
                    self.assertEqual(meta["phase_canary"]["cohort"], "holdout")
            finally:
                importlib.reload(router_module)

    def test_phase_canary_ineligible_planning_passes_through_before_broad_tool_rule(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  canary_fraction: 1.0
  holdout_fraction: 0.0
  excluded_categories: []
  safety_stop:
    enabled: false
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-light
    action:
      route_to: haiku
      reason: broad tool-light route
""",
                ):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "user", "content": "Inspect the repo and plan the work."}],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["workflow_phase"], "planning")
                    self.assertEqual(meta["phase_canary"]["status"], "ineligible")
                    self.assertEqual(meta["phase_canary"]["reason"], "workflow-phase-not-enabled")
            finally:
                importlib.reload(router_module)

    def test_phase_canary_safety_stop_prevents_downgrade_after_errors(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            for index in range(2):
                store.log_call(
                    id=f"call-{index}",
                    created_at=utc_now(),
                    path="/v1/messages",
                    requested_model=SONNET_DEFAULT,
                    routed_model=HAIKU_DEFAULT,
                    stream=0,
                    cache_hit=0,
                    status_code=500,
                    latency_ms=100,
                    input_tokens_est=10,
                    output_tokens_est=10,
                    routing_json=stable_json({
                        "phase_canary": {
                            "policy_id": "test-phase-canary",
                            "status": "applied",
                        }
                    }),
                    retry_count=0,
                    provider="anthropic",
                )
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  canary_fraction: 1.0
  holdout_fraction: 0.0
  excluded_categories: []
  safety_stop:
    enabled: true
    window_hours: 24
    min_samples: 2
    max_error_rate: 0.0
    max_retry_rate: 1.0
    max_fallback_rate: 1.0
rules: []
""",
                    {"AGENTFLOW_DB": db_path},
                ):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["reason"], "phase canary safety stop; keep requested model")
                    self.assertEqual(meta["phase_canary"]["status"], "safety_stopped")
                    self.assertTrue(meta["phase_canary"]["safety_stop"]["tripped"])
                    self.assertIn("error-rate", meta["phase_canary"]["safety_stop"]["reason_codes"])
            finally:
                importlib.reload(router_module)

    def test_routing_rule_can_match_workflow_phase(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      workflow_phase: summary
    action:
      route_to: haiku
      reason: summary phase routed to Haiku
""",
                ):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {"role": "user", "content": "Make the change."},
                            {"role": "assistant", "content": "Done."},
                            {"role": "user", "content": "Summarize the result."},
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    self.assertEqual(meta["workflow_phase"], "summary")
                    self.assertEqual(meta["reason"], "summary phase routed to Haiku")
            finally:
                importlib.reload(router_module)


if __name__ == "__main__":
    unittest.main()
