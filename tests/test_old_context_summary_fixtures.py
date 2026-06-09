import asyncio
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agentflow_proxy.crunch as crunch_module
from agentflow_proxy.recommendations import (
    build_old_context_summary_outcome_event,
    build_old_context_summary_outcome_feedback,
    queue_policy_event_feedback,
)
from agentflow_proxy.store import Store, stable_json


class OldContextSummaryFixtureTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CRUNCH",
        "AGENTFLOW_CRUNCH_RULES",
        "AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT",
        "AGENTFLOW_HAIKU_SUMMARY_MODEL",
        "AGENTFLOW_HAIKU_SUMMARY_MIN_REQUEST_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MIN_SUMMARIZED_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_TURNS",
        "AGENTFLOW_HAIKU_SUMMARY_KEEP_RECENT_TURNS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_SUMMARY_CHARS",
        "AGENTFLOW_HAIKU_SUMMARY_MAX_SOURCE_CHARS",
        "AGENTFLOW_MANAGED_RECOMMENDATIONS",
        "AGENTFLOW_RECOMMENDATION_SERVER_URL",
        "AGENTFLOW_MANAGED_API_KEY",
        "AGENTFLOW_OLD_CONTEXT_SUMMARY_FIXTURE_REAL_MODEL",
        "AGENTFLOW_OLD_CONTEXT_SUMMARY_FIXTURE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_VERSION",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.tmp = None
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
        if self.tmp is not None:
            self.tmp.cleanup()
        self.home.cleanup()
        importlib.reload(crunch_module)

    def _load_manual_crunch(self, rules: str):
        self.tmp = TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        config = tmp_path / "config"
        config.mkdir()
        (config / "crunch_rules.yaml").write_text(rules, encoding="utf-8")
        os.chdir(tmp_path)
        return importlib.reload(crunch_module)

    def _summary_rules(
        self,
        *,
        keep_recent_turns: int = 2,
        max_turns: int = 6,
        max_summary_chars: int = 240,
        block_tool_protocol: bool = True,
        block_thinking: bool = True,
    ) -> str:
        return f"""
enabled: true
old_context_summarization:
  enabled: true
  rule_id: offline-quality-fixture
  min_request_chars: 10
  min_summarized_chars: 10
  max_turns: {max_turns}
  keep_recent_turns: {keep_recent_turns}
  max_summary_chars: {max_summary_chars}
  max_source_chars: 40000
  excluded_categories: []
  block_tool_protocol: {str(block_tool_protocol).lower()}
  block_thinking: {str(block_thinking).lower()}
  safety_stop:
    enabled: false
"""

    def _durable_non_tool_body(self) -> dict:
        return {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "OFFLINE_FIXTURE_SECRET_A must never leave metadata. "
                        "Durable constraint: edit /repo/src/agentflow_summary.py only. "
                        "Unresolved TODO: keep the fixtures offline. "
                    )
                    * 30,
                },
                {
                    "role": "assistant",
                    "content": (
                        "Decision: preserve the path /repo/src/agentflow_summary.py "
                        "and the offline-only TODO in later turns. "
                    )
                    * 30,
                },
                {"role": "user", "content": "Recent turn must stay: RECENT_USER_TOKEN."},
                {"role": "assistant", "content": "Recent answer must stay: RECENT_ASSISTANT_TOKEN."},
            ],
        }

    async def _synthetic_fixture_fetch(self, summary_request):
        if os.getenv("AGENTFLOW_OLD_CONTEXT_SUMMARY_FIXTURE_REAL_MODEL") == "1":
            return await self._real_fixture_fetch(summary_request)
        return {
            "summary": (
                "Keep /repo/src/agentflow_summary.py. "
                "Unresolved TODO: keep the fixtures offline."
            ),
            "summary_input_tokens": 100,
            "summary_output_tokens": 20,
            "summary_cost_est_usd": 0.0008,
            "summary_status_code": 200,
        }

    async def _real_fixture_fetch(self, summary_request):
        url = os.getenv("AGENTFLOW_OLD_CONTEXT_SUMMARY_FIXTURE_URL")
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        if not url or not api_key:
            self.skipTest("real summary fixture requires AGENTFLOW_OLD_CONTEXT_SUMMARY_FIXTURE_URL and an Anthropic API key")
        import httpx

        headers = {
            "content-type": "application/json",
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "x-api-key": api_key,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=summary_request)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        parts = [
            str(block.get("text") or "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return {
            "summary": "\n".join(parts).strip() if response.status_code < 400 else None,
            "summary_input_tokens": usage.get("input_tokens"),
            "summary_output_tokens": usage.get("output_tokens"),
            "summary_cost_est_usd": 0.0,
            "summary_status_code": response.status_code,
            "summary_error": response.text[:500] if response.status_code >= 400 else None,
        }

    def test_long_non_tool_fixture_preserves_constraints_and_recent_turns(self):
        manual = self._load_manual_crunch(self._summary_rules(keep_recent_turns=2, max_summary_chars=180))
        body = self._durable_non_tool_body()

        summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
            body,
            exact_cache_enabled=False,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
            fetch_summary=self._synthetic_fixture_fetch,
        ))

        rendered = stable_json(summarized)
        self.assertEqual(meta["status"], "applied")
        self.assertEqual(meta["reason"], "summary-created")
        self.assertEqual(meta["eligible_turns"], 2)
        self.assertIn("/repo/src/agentflow_summary.py", summarized["system"][0]["text"])
        self.assertIn("Unresolved TODO: keep the fixtures offline.", summarized["system"][0]["text"])
        self.assertIn("RECENT_USER_TOKEN", rendered)
        self.assertIn("RECENT_ASSISTANT_TOKEN", rendered)
        self.assertNotIn("OFFLINE_FIXTURE_SECRET_A", rendered)
        self.assertEqual([msg["role"] for msg in summarized["messages"]], ["user", "assistant"])

    def test_mixed_tool_protocol_fixture_keeps_protocol_messages_out_of_summary(self):
        manual = self._load_manual_crunch(
            self._summary_rules(keep_recent_turns=2, max_turns=4, block_tool_protocol=False)
        )
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Old durable fact: use /repo/tasks/quality.txt. " * 12},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_fixture_1", "name": "Read", "input": {"file_path": "/repo/tasks/quality.txt"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_fixture_1", "content": "TOOL_RESULT_SECRET_CONTENT"}
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Old non-tool conclusion: TODO remains open. " * 12}]},
                {"role": "user", "content": "Recent tool follow-up must stay."},
                {"role": "assistant", "content": "Recent answer must stay."},
            ],
        }

        plan, meta = manual.old_context_summary_plan(body, exact_cache_enabled=False)
        summarized = manual.apply_old_context_summary(body, plan, "Use /repo/tasks/quality.txt; TODO remains open.")

        request_text = stable_json(plan["summary_request"])
        rendered = stable_json(summarized)
        self.assertEqual(meta["status"], "planned")
        self.assertEqual(plan["candidate_indexes"], [0, 3])
        self.assertNotIn("toolu_fixture_1", request_text)
        self.assertNotIn("TOOL_RESULT_SECRET_CONTENT", request_text)
        self.assertIn("toolu_fixture_1", rendered)
        self.assertIn("TOOL_RESULT_SECRET_CONTENT", rendered)
        self.assertIn("Recent tool follow-up must stay.", rendered)
        self.assertEqual(len(summarized["messages"]), 4)

    def test_thinking_history_fixture_blocks_old_context_summary(self):
        manual = self._load_manual_crunch(self._summary_rules(keep_recent_turns=1, block_thinking=True))
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning should not be summarized " * 20},
                        {"type": "text", "text": "Older visible answer."},
                    ],
                },
                {"role": "user", "content": "Old plain text that would otherwise qualify. " * 20},
                {"role": "user", "content": "Recent turn must remain."},
            ],
        }

        plan, meta = manual.old_context_summary_plan(body, exact_cache_enabled=False)

        self.assertIsNone(plan)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "thinking-context-blocked")

    def test_summary_text_is_bounded_when_inserted_as_system_context(self):
        manual = self._load_manual_crunch(self._summary_rules(keep_recent_turns=1, max_summary_chars=40))
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Old text with repeated file path /repo/limited.txt. " * 20},
                {"role": "assistant", "content": "Old answer with unresolved TODO. " * 20},
                {"role": "user", "content": "Recent turn must stay."},
            ],
        }
        plan, _ = manual.old_context_summary_plan(body, exact_cache_enabled=False)

        summarized = manual.apply_old_context_summary(body, plan, "X" * 80 + "SHOULD_NOT_APPEAR")
        summary_block = summarized["system"][0]["text"]
        inserted_summary = summary_block.rsplit("\n\n", 1)[-1]

        self.assertEqual(inserted_summary, "X" * 40)
        self.assertNotIn("SHOULD_NOT_APPEAR", summary_block)
        self.assertEqual(summarized["messages"], [{"role": "user", "content": "Recent turn must stay."}])

    def test_fixture_metadata_and_managed_event_do_not_include_raw_context_or_summary(self):
        manual = self._load_manual_crunch(self._summary_rules(keep_recent_turns=2, max_summary_chars=180))
        body = self._durable_non_tool_body()
        store = Store(str(Path.cwd() / "agentflow.sqlite3"))

        summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
            body,
            exact_cache_enabled=False,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
            fetch_summary=self._synthetic_fixture_fetch,
        ))
        crunch_meta = {"old_context_summarization": meta}
        feedback = build_old_context_summary_outcome_feedback(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            latency_ms=120,
            retry_count=0,
            cache_hit=False,
            crunch_meta=crunch_meta,
            category=meta.get("category"),
            error=None,
        )
        event = build_old_context_summary_outcome_event(feedback)
        queue_meta = asyncio.run(queue_policy_event_feedback(
            store,
            event,
            source_surface="old_context_summary_outcome",
        ))

        rendered_meta = json.dumps(meta, sort_keys=True)
        rendered_event = stable_json(event)
        for forbidden in (
            "OFFLINE_FIXTURE_SECRET_A",
            "Keep /repo/src/agentflow_summary.py",
            "Unresolved TODO: keep the fixtures offline.",
            stable_json(body),
            stable_json(summarized),
        ):
            self.assertNotIn(forbidden, rendered_meta)
            self.assertNotIn(forbidden, rendered_event)
        self.assertFalse(feedback["privacy"]["raw_old_turns_included"])
        self.assertFalse(feedback["privacy"]["raw_summary_included"])
        self.assertEqual(feedback["enhanced_crunch"]["state"], "applied")
        self.assertEqual(feedback["enhanced_crunch"]["mode"], "local_provider_account")
        self.assertEqual(feedback["enhanced_crunch"]["profile"], "default")
        self.assertFalse(event["metadata"]["privacy"]["summary_text_included"])
        self.assertEqual(queue_meta["status"], "disabled")
        self.assertEqual(store.managed_outcome_feedback_rows(limit=10), [])


if __name__ == "__main__":
    unittest.main()
