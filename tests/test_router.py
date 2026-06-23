import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import tokenclaw.router as router_module
from tokenclaw.anthropic_proxy import _record_routing_rate_limit_fallback
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.store import Store, stable_json, utc_now
from tokenclaw.router import HAIKU_DEFAULT, SONNET_DEFAULT, classify_workflow_phase, route_model


def _log_memory_call(
    store: Store,
    suffix: int,
    *,
    session_id: str = "secret-session-router",
    phase: str = "tool-execution",
    category: str = "tool-result",
    status_code: int = 200,
    retry_count: int = 0,
    fallback: bool = False,
    requested_model: str = SONNET_DEFAULT,
    routed_model: str = SONNET_DEFAULT,
    adversarial_raw_fields: bool = False,
) -> None:
    routing = {
        "workflow_phase": phase,
        "category": category,
        "text_chars": 12000,
        "has_tools": category.startswith("tool"),
    }
    if fallback:
        routing["fallback_reason"] = "rate_limited"
    raw_fields = {}
    if adversarial_raw_fields:
        raw_fields = {
            "prompt": "SECRET_ROUTER_MEMORY_PROMPT",
            "messages": [{"role": "user", "content": "SECRET_ROUTER_MEMORY_MESSAGE"}],
            "content": "SECRET_ROUTER_MEMORY_CONTENT",
            "tool_payload": {"arguments": "SECRET_ROUTER_MEMORY_TOOL_PAYLOAD"},
            "request_id": "req_router_memory_secret",
            "cache_key": "cache-key-router-memory-secret",
            "file_path": "/tmp/router-memory-secret.py",
            "session_id": "secret-session-router-raw-field",
            "raw_request": {"messages": [{"content": "SECRET_ROUTER_MEMORY_PROMPT"}]},
        }
        routing.update({
            "workflow_phase": "SECRET_ROUTER_MEMORY_PROMPT",
            "category": "SECRET_ROUTER_MEMORY_MESSAGE",
            **raw_fields,
        })
    stored_category = "SECRET_ROUTER_MEMORY_MESSAGE" if adversarial_raw_fields else category
    store.log_call(
        id=f"memory-call-{suffix}",
        created_at=f"2026-06-10T10:00:{suffix:02d}+00:00",
        path="/v1/messages",
        requested_model=requested_model,
        routed_model=routed_model,
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=1000,
        input_tokens_est=3000,
        output_tokens_est=100,
        actual_input_tokens=3000,
        actual_output_tokens=100,
        cost_est_usd=0.01,
        cost_baseline_usd=0.02,
        routing_json=stable_json(routing),
        crunch_json=stable_json({"changed": False, "tokens_saved_est": 0, **raw_fields}),
        cache_json=stable_json({"status": "SECRET_ROUTER_MEMORY_CONTENT", "reason": "cache-key-router-memory-secret", **raw_fields} if adversarial_raw_fields else {"status": "skipped", "reason": "streaming"}),
        request_json=stable_json({"messages": [{"content": "SECRET_ROUTER_MEMORY_PROMPT"}], **raw_fields}) if adversarial_raw_fields else None,
        response_json=stable_json({"content": [{"text": "SECRET_ROUTER_MEMORY_RESPONSE"}]}) if adversarial_raw_fields else None,
        error="SECRET_ROUTER_MEMORY_ERROR" if adversarial_raw_fields else None,
        session_id=session_id,
        category=stored_category,
        retry_count=retry_count,
        provider="anthropic",
    )


class RouterTest(unittest.TestCase):
    def setUp(self):
        for name in ("TOKENCLAW_ROUTING_RULES", "TOKENCLAW_DATABASE_URL", "TOKENCLAW_DB"):
            os.environ.pop(name, None)
        importlib.reload(router_module)

    def _reload_with_routing_yaml(self, tmp: str, yaml_text: str, extra_env: dict[str, str] | None = None):
        rules_path = Path(tmp) / "routing_rules.yaml"
        rules_path.write_text(yaml_text, encoding="utf-8")
        env = {"TOKENCLAW_ROUTING_RULES": str(rules_path)}
        env.update(extra_env or {})
        return patch.dict(os.environ, env)

    def test_default_tool_result_sonnet_routing_is_off_without_backing(self):
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

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        self.assertEqual(meta["category"], "tool-result")
        self.assertEqual(meta["workflow_phase"], "tool-execution")
        self.assertEqual(meta["workflow_phase_reason"], "last-user-tool-result")
        self.assertEqual(meta["policy_source"], "local-default")
        self.assertFalse(meta["routing_backing"]["backed"])

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
        self.assertEqual(meta["workflow_phase_reason"], "thinking-current-request")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        self.assertEqual(meta["thinking_gate"]["status"], "blocked")
        self.assertEqual(meta["thinking_gate"]["reason"], "current-thinking-request")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_tool_result_with_assistant_thinking_history_stays_on_sonnet_without_current_thinking(self):
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
        self.assertEqual(meta["workflow_phase"], "tool-execution")
        self.assertEqual(meta["workflow_phase_reason"], "last-user-tool-result")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        self.assertEqual(meta["thinking_gate"]["status"], "blocked")
        self.assertEqual(meta["thinking_gate"]["reason"], "assistant-thinking-history")

    def test_redacted_thinking_history_is_gated_and_stripped_with_thinking_history(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reasoning"},
                        {"type": "redacted_thinking", "data": "redacted"},
                        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            ],
        }

        routed, meta = route_model(body)
        stripped, stripped_count = router_module.strip_thinking_history_blocks(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["thinking_gate"]["status"], "blocked")
        self.assertEqual(meta["thinking_gate"]["reason"], "assistant-thinking-history")
        self.assertEqual(stripped_count, 2)
        self.assertEqual([block["type"] for block in stripped["messages"][0]["content"]], ["tool_use"])

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
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
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
                with patch.dict(os.environ, {"TOKENCLAW_ROUTING_RULES": str(rules_path)}):
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
                with patch.dict(os.environ, {"TOKENCLAW_ROUTING_RULES": str(rules_path)}):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [{"role": "user", "content": "Say ok."}],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertFalse(meta["enabled"])
                    self.assertEqual(meta["reason"], "routing off: no matching manual hard rule")
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
                with patch.dict(os.environ, {"TOKENCLAW_ROUTING_RULES": str(rules_path)}):
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

    def test_manual_hard_rule_routes_only_matching_shapes(self):
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
      reason: manual tool-result rule
""",
                encoding="utf-8",
            )
            try:
                with patch.dict(os.environ, {"TOKENCLAW_ROUTING_RULES": str(rules_path)}):
                    manual_router = importlib.reload(router_module)
                    matching = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }
                    nonmatching = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [{"role": "user", "content": "Please explain this setting."}],
                    }

                    routed, meta = manual_router.route_model(matching)
                    kept, kept_meta = manual_router.route_model(nonmatching)

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    self.assertTrue(meta["enabled"])
                    self.assertEqual(meta["reason"], "manual tool-result rule")
                    self.assertEqual(meta["policy_source"], "local-manual")
                    self.assertEqual(kept, manual_router.SONNET_DEFAULT)
                    self.assertFalse(kept_meta["enabled"])
                    self.assertEqual(kept_meta["reason"], "routing off: no matching manual hard rule")
                    self.assertEqual(kept_meta["routing_backing"]["manual_hard_rule_count"], 1)
            finally:
                importlib.reload(router_module)

    def test_disabled_thinking_with_assistant_thinking_history_keeps_tool_result_on_sonnet(self):
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
        self.assertEqual(meta["workflow_phase"], "tool-execution")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        self.assertEqual(meta["thinking_gate"]["status"], "blocked")
        self.assertEqual(meta["thinking_gate"]["reason"], "assistant-thinking-history")

    def test_openai_default_canary_is_off_without_backing(self):
        routed, meta = router_module.route_openai_model({
            "model": "gpt-5.4",
            "input": "Inspect the small tool result and decide whether another lookup is needed.",
            "tools": [{"type": "function", "name": "lookup_file"}],
        })

        self.assertEqual(routed, "gpt-5.4")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["provider"], "openai")
        self.assertEqual(meta["category"], "tool-light")
        self.assertTrue(meta["has_tools"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        self.assertEqual(meta["routed_model"], "gpt-5.4")

    def test_openai_default_chat_canary_is_off_without_backing(self):
        routed, meta = router_module.route_openai_model({
            "model": "gpt-5.4",
            "input": "Summarize the recent result.\n" + ("context " * 260),
        })

        self.assertEqual(routed, "gpt-5.4")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["provider"], "openai")
        self.assertEqual(meta["category"], "chat")
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_openai_tool_light_canary_requires_manual_policy_file(self):
        routed, meta = router_module.route_openai_model({
            "model": "gpt-5.4",
            "input": "Inspect this tool-light payload.\n" + ("x" * 12000),
            "tools": [{"type": "function", "name": "lookup_file"}],
        })

        self.assertEqual(routed, "gpt-5.4")
        self.assertEqual(meta["category"], "tool-light")
        self.assertFalse(meta["enabled"])
        self.assertNotIn("openai_canary", meta)

        _, heavy_meta = router_module.route_openai_model({
            "model": "gpt-5.4",
            "input": "Inspect this heavier payload.\n" + ("x" * 17000),
            "tools": [{"type": "function", "name": "lookup_file"}],
        })
        self.assertEqual(heavy_meta["category"], "tool-heavy")
        self.assertFalse(heavy_meta["enabled"])

    def test_openai_canary_ignores_non_target_models_by_default(self):
        routed, meta = router_module.route_openai_model({
            "model": "gpt-5-codex",
            "input": "small task",
        })

        self.assertEqual(routed, "gpt-5-codex")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["provider"], "openai")
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_openai_env_routing_alone_does_not_back_default_heuristics(self):
        try:
            with patch.dict(os.environ, {"TOKENCLAW_OPENAI_ROUTING": "1"}):
                manual_router = importlib.reload(router_module)

                routed, meta = manual_router.route_openai_model({
                    "model": manual_router.OPENAI_LARGE_DEFAULT,
                    "input": "small task",
                })

                self.assertEqual(routed, manual_router.OPENAI_LARGE_DEFAULT)
                self.assertFalse(meta["enabled"])
                self.assertEqual(meta["provider"], "openai")
                self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        finally:
            importlib.reload(router_module)

    def test_openai_canary_policy_file_is_ignored_by_runtime_router(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
openai_canary:
  enabled: true
  policy_id: retired-local-openai-canary
  model_pattern: gpt-5
  target_model: gpt-5.4-mini
  canary_fraction: 1.0
  holdout_fraction: 0.0
rules: []
""",
                encoding="utf-8",
            )
            try:
                with patch.dict(os.environ, {"TOKENCLAW_ROUTING_RULES": str(rules_path)}, clear=False):
                    manual_router = importlib.reload(router_module)

                    routed, meta = manual_router.route_openai_model({
                        "model": "gpt-5",
                        "input": "Summarize the outcome without tools.",
                    })

                    self.assertEqual(routed, "gpt-5")
                    self.assertFalse(meta["enabled"])
                    self.assertEqual(meta["policy_source"], "local-manual")
                    self.assertEqual(meta["routing_source"], "local-rules")
                    self.assertNotIn("openai_canary", meta)
                    self.assertFalse(manual_router.ROUTING_OPENAI_CANARY["enabled"])
                    self.assertEqual(manual_router.ROUTING_OPENAI_CANARY["router_runtime_status"], "retired")
            finally:
                importlib.reload(router_module)

    def test_openai_explicit_local_rule_still_routes(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      provider: openai
      model_pattern: gpt-5.5
      category: short-completion
    action:
      route_to: gpt-5.4
      reason: explicit OpenAI chat route
""",
                encoding="utf-8",
            )
            try:
                with patch.dict(os.environ, {"TOKENCLAW_ROUTING_RULES": str(rules_path)}, clear=False):
                    manual_router = importlib.reload(router_module)

                    routed, meta = manual_router.route_openai_model({
                        "model": "gpt-5.5",
                        "input": "Summarize the result.",
                    })

                    self.assertEqual(routed, "gpt-5.4")
                    self.assertTrue(meta["enabled"])
                    self.assertEqual(meta["routing_source"], "explicit-local-rule")
                    self.assertEqual(meta["reason"], "explicit OpenAI chat route")
            finally:
                importlib.reload(router_module)

    def test_small_non_tool_sonnet_default_routing_is_off_without_backing(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "```python\n" + ("print('x')\n" * 860) + "```"}],
        }

        routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
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
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
        self.assertEqual(meta["text_chars"], 10000)

    def test_midsize_env_flag_alone_does_not_back_default_routing(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "summarize this report\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {"TOKENCLAW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")
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
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_midsize_code_gen_sonnet_does_not_route_to_haiku(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "```python\n" + ("print('x')\n" * 1200) + "```"}],
        }

        with patch.dict(os.environ, {"TOKENCLAW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "code-gen")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_large_non_tool_sonnet_above_midsize_window_stays_requested(self):
        body = {
            "model": SONNET_DEFAULT,
            "messages": [{"role": "user", "content": "summarize all of this\n" + ("a" * 31000)}],
        }

        with patch.dict(os.environ, {"TOKENCLAW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertEqual(meta["category"], "chat")
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_midsize_tool_request_sonnet_does_not_route_as_non_tool(self):
        body = {
            "model": SONNET_DEFAULT,
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "read the files\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {"TOKENCLAW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["category"], "tool-light")
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_midsize_thinking_request_keeps_requested_model_when_enabled(self):
        body = {
            "model": SONNET_DEFAULT,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [{"role": "user", "content": "think about this\n" + ("a" * 11900)}],
        }

        with patch.dict(os.environ, {"TOKENCLAW_ROUTE_MIDSIZE": "1"}):
            routed, meta = route_model(body)

        self.assertEqual(routed, SONNET_DEFAULT)
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "routing off: no managed server or manual hard rules")

    def test_phase_canary_policy_file_is_ignored_by_runtime_router(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: retired-local-phase-canary
  canary_fraction: 1.0
  holdout_fraction: 0.0
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
                    self.assertFalse(meta["enabled"])
                    self.assertEqual(meta["routing_source"], "local-rules")
                    self.assertNotIn("phase_canary", meta)
                    self.assertFalse(manual_router.ROUTING_PHASE_CANARY["enabled"])
                    self.assertEqual(manual_router.ROUTING_PHASE_CANARY["router_runtime_status"], "retired")
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
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

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["reason"], "phase canary selected shadow route; keep requested model")
                    self.assertEqual(meta["phase_canary"]["status"], "applied")
                    self.assertEqual(meta["phase_canary"]["cohort"], "canary_applied")
                    self.assertEqual(meta["phase_canary"]["policy_id"], "test-phase-canary")
                    self.assertEqual(meta["phase_canary"]["workflow_phase"], "tool-execution")
                    self.assertEqual(meta["phase_canary"]["shadow_model"], manual_router.HAIKU_DEFAULT)
                    self.assertEqual(meta["phase_canary"]["actual_forwarded_model"], manual_router.SONNET_DEFAULT)
                    self.assertTrue(meta["phase_canary"]["shadow_only"])
                    self.assertIn("cohort_hash", meta["phase_canary"])
                    self.assertNotIn("content", stable_json(meta["phase_canary"]["cohort_features"]))
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
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
                    self.assertEqual(meta["phase_canary"]["cohort"], "canary_holdout")
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_disabled_default_keeps_existing_rules(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: false
  canary_fraction: 1.0
  holdout_fraction: 0.0
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: existing tool-result rule
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
                    self.assertEqual(meta["reason"], "existing tool-result rule")
                    self.assertNotIn("phase_canary", meta)
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_rejects_non_anthropic_source_surface(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  source_surface: openai_responses
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

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["phase_canary"]["status"], "ineligible")
                    self.assertEqual(meta["phase_canary"]["reason"], "source-surface-not-supported")
                    self.assertEqual(meta["reason"], "phase canary ineligible: source-surface-not-supported")
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_rejects_non_anthropic_provider(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  provider: openai
  source_surface: anthropic_messages
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

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["phase_canary"]["status"], "ineligible")
                    self.assertEqual(meta["phase_canary"]["reason"], "provider-not-supported")
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_enforces_stream_scope(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  provider: anthropic
  source_surface: anthropic_messages
  stream: true
  canary_fraction: 1.0
  holdout_fraction: 0.0
  excluded_categories: []
  safety_stop:
    enabled: false
rules: []
""",
                ):
                    manual_router = importlib.reload(router_module)
                    base = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }

                    non_stream_routed, non_stream_meta = manual_router.route_model({**base, "stream": False})
                    stream_routed, stream_meta = manual_router.route_model({**base, "stream": True})

                    self.assertEqual(non_stream_routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(non_stream_meta["phase_canary"]["reason"], "stream-scope-not-enabled")
                    self.assertEqual(stream_routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(stream_meta["phase_canary"]["status"], "applied")
                    self.assertTrue(stream_meta["phase_canary"]["stream"])
                    self.assertEqual(stream_meta["phase_canary"]["shadow_model"], manual_router.HAIKU_DEFAULT)
                    self.assertFalse(stream_meta["phase_canary"]["requires_shadow_comparison"])
                    self.assertNotIn("content", stable_json(stream_meta["phase_canary"]["cohort_features"]))
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_thinking_history_stays_gated(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  provider: anthropic
  source_surface: anthropic_messages
  stream: true
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
                        "stream": True,
                        "messages": [
                            {
                                "role": "assistant",
                                "content": [{"type": "thinking", "thinking": "internal reasoning"}],
                            },
                            {
                                "role": "user",
                                "content": "Continue the plan.",
                            },
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["workflow_phase"], "thinking")
                    self.assertEqual(meta["workflow_phase_reason"], "thinking-history")
                    self.assertEqual(meta["phase_canary"]["status"], "safety_stopped")
                    self.assertEqual(meta["phase_canary"]["reason"], "thinking-safety-gate")
                    self.assertEqual(meta["phase_canary"]["cohort"], "safety_stopped")
                    self.assertIn("thinking-history-blocked", meta["phase_canary"]["safety_stop"]["reason_codes"])
                    self.assertTrue(meta["phase_canary"]["safety_gates"]["block_thinking_history"])
                    self.assertEqual(meta["thinking_gate"]["status"], "blocked")
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_safety_stops_tool_result_with_thinking_history(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  provider: anthropic
  source_surface: anthropic_messages
  stream: true
  canary_fraction: 1.0
  holdout_fraction: 0.0
  excluded_categories: []
rules: []
""",
                    {"TOKENCLAW_DB": str(Path(tmp) / "missing.sqlite3")},
                ):
                    manual_router = importlib.reload(router_module)
                    body = {
                        "model": manual_router.SONNET_DEFAULT,
                        "stream": True,
                        "messages": [
                            {
                                "role": "assistant",
                                "content": [{"type": "thinking", "thinking": "internal reasoning"}],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            },
                        ],
                    }

                    routed, meta = manual_router.route_model(body)

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["workflow_phase"], "tool-execution")
                    self.assertEqual(meta["reason"], "phase canary safety stop; keep requested model")
                    self.assertEqual(meta["thinking_gate"]["status"], "blocked")
                    self.assertEqual(meta["phase_canary"]["status"], "safety_stopped")
                    self.assertEqual(meta["phase_canary"]["cohort"], "safety_stopped")
                    self.assertEqual(meta["phase_canary"]["reason"], "thinking-safety-gate")
                    self.assertEqual(meta["phase_canary"]["workflow_phase"], "tool-execution")
                    self.assertTrue(meta["phase_canary"]["safety_stop"]["enabled"])
                    self.assertEqual(meta["phase_canary"]["safety_stop"]["status"], "tripped")
                    self.assertEqual(meta["phase_canary"]["safety_stop"]["reason_codes"], ["thinking-history-blocked"])
            finally:
                importlib.reload(router_module)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_session_cohort_uses_hashed_session_only(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
phase_canary:
  enabled: true
  policy_id: test-phase-canary
  target_candidate_id: promoted-shadow-candidate
  canary_fraction: 1.0
  holdout_fraction: 0.0
  excluded_categories: []
  cohort_unit: session
  safety_stop:
    enabled: false
rules: []
""",
                ):
                    manual_router = importlib.reload(router_module)
                    first = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                            }
                        ],
                    }
                    second = {
                        "model": manual_router.SONNET_DEFAULT,
                        "messages": [
                            {"role": "user", "content": "Make the change."},
                            {"role": "assistant", "content": "Done."},
                            {"role": "user", "content": "Summarize the result."},
                        ],
                    }

                    first_routed, first_meta = manual_router.route_model(first, session_id="raw-secret-session")
                    second_routed, second_meta = manual_router.route_model(second, session_id="raw-secret-session")

                    self.assertEqual(first_routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(second_routed, manual_router.SONNET_DEFAULT)
                    first_canary = first_meta["phase_canary"]
                    second_canary = second_meta["phase_canary"]
                    self.assertEqual(first_canary["candidate_id"], "promoted-shadow-candidate")
                    self.assertEqual(first_canary["shadow_model"], manual_router.HAIKU_DEFAULT)
                    self.assertTrue(first_canary["shadow_only"])
                    self.assertEqual(first_canary["cohort_key_hash"], second_canary["cohort_key_hash"])
                    self.assertEqual(first_canary["cohort_features"]["cohort_unit"], "session")
                    self.assertIn("session_id_hash", first_canary["cohort_features"])
                    rendered = stable_json([first_canary, second_canary])
                    self.assertNotIn("raw-secret-session", rendered)
                    self.assertNotIn("content", rendered)
                    self.assertEqual(managed_egress_violations(first_canary), [])
            finally:
                importlib.reload(router_module)

    def test_phase_canary_fallback_metadata_updates_canary_block(self):
        routing_meta = {
            "phase_canary": {
                "status": "applied",
                "cohort": "canary_applied",
                "target_model": HAIKU_DEFAULT,
                "actual_forwarded_model": HAIKU_DEFAULT,
            }
        }

        _record_routing_rate_limit_fallback(
            routing_meta,
            requested_model=SONNET_DEFAULT,
            from_model=HAIKU_DEFAULT,
        )

        self.assertEqual(routing_meta["fallback_reason"], "rate_limited")
        self.assertEqual(routing_meta["phase_canary"]["fallback_reason"], "rate_limited")
        self.assertEqual(routing_meta["phase_canary"]["fallback_from_model"], HAIKU_DEFAULT)
        self.assertEqual(routing_meta["phase_canary"]["actual_forwarded_model"], SONNET_DEFAULT)

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
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

    @unittest.skip("local phase canary routing retired; managed policy decisions own cohorts")
    def test_phase_canary_safety_stop_prevents_downgrade_after_errors(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
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
                    {"TOKENCLAW_DB": db_path},
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

    def test_session_memory_condition_is_retired_and_fails_closed(self):
        with TemporaryDirectory() as tmp:
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
      session_memory:
        enabled: true
        min_call_count: 3
    action:
      route_to: haiku
      reason: retired session memory route
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

                    routed, meta = manual_router.route_model(body, session_id="secret-session-router")

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertFalse(meta["enabled"])
                    self.assertEqual(meta["routing_backing"]["reason"], "no-matching-manual-hard-rule")
                    self.assertNotIn("session_phase_memory", meta)
            finally:
                importlib.reload(router_module)

    @unittest.skip("local DB-backed session routing gates retired; managed policy decisions own session gates")
    def test_session_memory_rule_routes_stable_tool_execution_window(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                for index in range(1, 4):
                    _log_memory_call(store, index)
            finally:
                store.conn.close()
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
      workflow_phase: tool-execution
      session_memory:
        enabled: true
        window_size: 3
        min_call_count: 3
        dominant_phase: tool-execution
        min_dominant_phase_count: 3
        max_error_count: 0
        max_retry_count: 0
        max_fallback_count: 0
        blocked_phases: [planning, verification, thinking, unknown]
        model_family_floor: sonnet
    action:
      route_to: haiku
      reason: stable tool execution memory routed to Haiku
""",
                    {"TOKENCLAW_DB": db_path},
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

                    routed, meta = manual_router.route_model(body, session_id="secret-session-router")

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    self.assertEqual(meta["reason"], "stable tool execution memory routed to Haiku")
                    memory = meta["session_phase_memory"]
                    self.assertEqual(memory["status"], "used")
                    self.assertEqual(memory["memory"]["dominant_phase"], "tool-execution")
                    self.assertEqual(memory["memory"]["window"]["call_count"], 3)
                    self.assertFalse(memory["memory"]["raw_session_id_included"])
                    self.assertNotIn("secret-session-router", stable_json(memory))
                    self.assertEqual(managed_egress_violations(memory), [])
            finally:
                importlib.reload(router_module)

    @unittest.skip("local DB-backed session routing gates retired; managed policy decisions own session gates")
    def test_session_memory_rule_buckets_adversarial_metadata_before_route_meta(self):
        forbidden = (
            "SECRET_ROUTER_MEMORY_PROMPT",
            "SECRET_ROUTER_MEMORY_MESSAGE",
            "SECRET_ROUTER_MEMORY_CONTENT",
            "SECRET_ROUTER_MEMORY_TOOL_PAYLOAD",
            "SECRET_ROUTER_MEMORY_RESPONSE",
            "SECRET_ROUTER_MEMORY_ERROR",
            "req_router_memory_secret",
            "cache-key-router-memory-secret",
            "/tmp/router-memory-secret.py",
            "secret-session-router",
        )
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                for index in range(1, 4):
                    _log_memory_call(store, index, adversarial_raw_fields=True)
            finally:
                store.conn.close()
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
      session_memory:
        enabled: true
        window_size: 3
        min_call_count: 3
        dominant_phase: unknown
        min_dominant_phase_count: 3
        allow_blockers: [small_sample]
    action:
      route_to: haiku
      reason: stable unknown memory route
""",
                    {"TOKENCLAW_DB": db_path},
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

                    routed, meta = manual_router.route_model(body, session_id="secret-session-router")

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    memory = meta["session_phase_memory"]
                    self.assertEqual(memory["status"], "used")
                    self.assertEqual(memory["memory"]["dominant_phase"], "unknown")
                    self.assertIn({"value": "unknown", "count": 3}, memory["memory"]["category_counts"])
                    self.assertEqual(managed_egress_violations(memory), [])
                    rendered = stable_json(meta)
                    for value in forbidden:
                        self.assertNotIn(value, rendered)
            finally:
                importlib.reload(router_module)

    @unittest.skip("local DB-backed session routing gates retired; managed policy decisions own session gates")
    def test_session_memory_rule_blocks_missing_memory(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            store.conn.close()
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
      session_memory:
        enabled: true
        min_call_count: 3
        dominant_phase: tool-execution
    action:
      route_to: haiku
      reason: stable memory route
""",
                    {"TOKENCLAW_DB": db_path},
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

                    routed, meta = manual_router.route_model(body, session_id="unknown-session")

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    self.assertEqual(meta["session_phase_memory"]["status"], "blocked")
                    self.assertEqual(meta["session_phase_memory"]["reason"], "memory-missing")
                    self.assertEqual(meta["reason"], "session phase memory blocked: memory-missing")
            finally:
                importlib.reload(router_module)

    @unittest.skip("local DB-backed session routing gates retired; managed policy decisions own session gates")
    def test_session_memory_rule_blocks_recent_errors_retries_and_fallbacks(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                _log_memory_call(store, 1)
                _log_memory_call(store, 2, status_code=429, retry_count=1, fallback=True)
                _log_memory_call(store, 3)
            finally:
                store.conn.close()
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
      session_memory:
        enabled: true
        window_size: 3
        min_call_count: 3
        dominant_phase: tool-execution
        max_error_count: 0
        max_retry_count: 0
        max_fallback_count: 0
    action:
      route_to: haiku
      reason: stable memory route
""",
                    {"TOKENCLAW_DB": db_path},
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

                    routed, meta = manual_router.route_model(body, session_id="secret-session-router")

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    reasons = set(meta["session_phase_memory"]["reason_codes"])
                    self.assertIn("recent_errors", reasons)
                    self.assertIn("recent_retries", reasons)
                    self.assertIn("recent_routing_fallback", reasons)
            finally:
                importlib.reload(router_module)

    @unittest.skip("local DB-backed session routing gates retired; managed policy decisions own session gates")
    def test_session_memory_rule_blocks_thinking_and_planning_windows(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                _log_memory_call(store, 1, phase="planning", category="tool-light")
                _log_memory_call(store, 2, phase="thinking", category="tool-result")
                _log_memory_call(store, 3, phase="thinking", category="tool-result")
            finally:
                store.conn.close()
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
      session_memory:
        enabled: true
        window_size: 3
        min_call_count: 3
        dominant_phase_in: [tool-execution, summary]
        blocked_phases: [planning, verification, thinking, unknown]
        allow_thinking: false
    action:
      route_to: haiku
      reason: stable memory route
""",
                    {"TOKENCLAW_DB": db_path},
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

                    routed, meta = manual_router.route_model(body, session_id="secret-session-router")

                    self.assertEqual(routed, manual_router.SONNET_DEFAULT)
                    reasons = set(meta["session_phase_memory"]["reason_codes"])
                    self.assertIn("dominant_phase_mismatch", reasons)
                    self.assertIn("blocked_phase_present", reasons)
                    self.assertIn("thinking_present", reasons)
            finally:
                importlib.reload(router_module)

    @unittest.skip("local DB-backed session routing gates retired; managed policy decisions own session gates")
    def test_session_memory_rule_routes_stable_summary_when_per_call_rule_agrees(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                for index in range(1, 4):
                    _log_memory_call(store, index, phase="summary", category="short-completion")
            finally:
                store.conn.close()
            try:
                with self._reload_with_routing_yaml(
                    tmp,
                    """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
      workflow_phase: summary
      text_chars_lt: 10000
      session_memory:
        enabled: true
        window_size: 3
        min_call_count: 3
        dominant_phase: summary
        stable_phase: true
        model_family_floor: sonnet
    action:
      route_to: haiku
      reason: stable summary memory routed to Haiku
""",
                    {"TOKENCLAW_DB": db_path},
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

                    routed, meta = manual_router.route_model(body, session_id="secret-session-router")

                    self.assertEqual(routed, manual_router.HAIKU_DEFAULT)
                    self.assertEqual(meta["workflow_phase"], "summary")
                    self.assertEqual(meta["session_phase_memory"]["status"], "used")
            finally:
                importlib.reload(router_module)


if __name__ == "__main__":
    unittest.main()
