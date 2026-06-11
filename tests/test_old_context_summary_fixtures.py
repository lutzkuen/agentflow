import asyncio
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agentflow_proxy import cli
import agentflow_proxy.crunch as crunch_module
from agentflow_proxy.managed_egress import RAW_FEATURE_KEYS
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

    def _keys_in(self, value):
        keys = set()
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(str(key).lower())
                keys.update(self._keys_in(item))
        elif isinstance(value, list):
            for item in value:
                keys.update(self._keys_in(item))
        return keys

    def _summary_feedback_for_meta(self, meta: dict, *, status_code: int = 200):
        return build_old_context_summary_outcome_feedback(
            provider="anthropic",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=status_code,
            latency_ms=120,
            retry_count=0,
            cache_hit=False,
            crunch_meta={"old_context_summarization": meta},
            category=meta.get("category"),
            error="provider response bucket only" if status_code >= 400 else None,
        )

    def _assert_feedback_metadata_only(self, feedback: dict, forbidden: tuple[str, ...]):
        self.assertIsNotNone(feedback)
        event = build_old_context_summary_outcome_event(feedback)
        rendered_feedback = json.dumps(feedback, sort_keys=True)
        rendered_event = json.dumps(event, sort_keys=True)
        for secret in forbidden:
            self.assertNotIn(secret, rendered_feedback)
            self.assertNotIn(secret, rendered_event)
        self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(self._keys_in(feedback)))
        self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(self._keys_in(event)))
        self.assertTrue(feedback["privacy"]["metadata_only"])
        self.assertFalse(feedback["privacy"]["raw_old_turns_included"])
        self.assertFalse(feedback["privacy"]["raw_summary_included"])
        self.assertFalse(event["metadata"]["privacy"]["summary_text_included"])

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

    def test_fail_closed_summary_fixtures_forward_unchanged_and_emit_metadata_only_feedback(self):
        manual = self._load_manual_crunch(self._summary_rules(keep_recent_turns=2, max_summary_chars=180))
        body = self._durable_non_tool_body()
        raw_forbidden = (
            "OFFLINE_FIXTURE_SECRET_A",
            "/repo/src/agentflow_summary.py",
            "RAW_PROVIDER_BODY_SECRET",
            "RAW_GENERATED_SUMMARY_SECRET",
            "cache-key-old-context-secret",
            "req_old_context_secret",
            "tenant-old-context-secret",
            "tool_payload_old_context_secret",
        )

        async def provider_5xx(_summary_request):
            return {
                "summary": None,
                "summary_status_code": 503,
                "summary_error": "RAW_PROVIDER_BODY_SECRET req_old_context_secret",
                "provider_body": {"content": "RAW_PROVIDER_BODY_SECRET"},
                "request_id": "req_old_context_secret",
            }

        async def malformed_summary(_summary_request):
            return {
                "summary": "",
                "summary_status_code": 200,
                "summary_error": "RAW_GENERATED_SUMMARY_SECRET",
            }

        async def oversized_summary_cost(_summary_request):
            return {
                "summary": "RAW_GENERATED_SUMMARY_SECRET should be too expensive to apply",
                "summary_status_code": 200,
                "summary_cost_est_usd": 99.0,
            }

        cases = (
            (provider_5xx, "summary-error"),
            (malformed_summary, "summary-empty"),
            (oversized_summary_cost, "summary-cost-too-high"),
        )
        for fetch, expected_reason in cases:
            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=fetch,
            ))

            self.assertEqual(summarized, body)
            self.assertFalse(meta["applied"])
            self.assertEqual(meta["reason"], expected_reason)
            feedback = self._summary_feedback_for_meta({
                **meta,
                "rule_id": "/private/project/raw_prompt_rule",
                "candidate_id": "candidate with secret message",
                "cache_key": "cache-key-old-context-secret",
                "request_id": "req_old_context_secret",
                "tenant_id": "tenant-old-context-secret",
                "tool_payload": "tool_payload_old_context_secret",
            })
            self.assertEqual(feedback["outcome"], "skipped")
            self.assertEqual(feedback["reason"], expected_reason)
            self.assertTrue(str(feedback["rule_id"]).startswith("sha256:"))
            self.assertTrue(str(feedback["candidate_id"]).startswith("sha256:"))
            self._assert_feedback_metadata_only(feedback, raw_forbidden)

        os.chdir(self.home.name)
        default_manual = importlib.reload(crunch_module)
        managed_profile = {
            "policy_source": "managed-recommended",
            "old_context_summarization": {
                "enabled": True,
                "rule_id": "managed-summary-provider-required",
                "candidate_id": "candidate-provider-required",
            },
        }
        summarized, meta = asyncio.run(default_manual.maybe_summarize_old_context(
            body,
            exact_cache_enabled=False,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
            fetch_summary=self._synthetic_fixture_fetch,
            managed_profile=managed_profile,
        ))

        self.assertEqual(summarized, body)
        self.assertEqual(meta["reason"], "fallback-not-configured")
        self.assertFalse(meta["configured"])
        feedback = self._summary_feedback_for_meta(meta)
        self.assertEqual(feedback["outcome"], "skipped")
        self._assert_feedback_metadata_only(feedback, raw_forbidden)

    def test_tool_protocol_mismatch_fixture_fails_closed_with_metadata_only_feedback(self):
        manual = self._load_manual_crunch(
            self._summary_rules(keep_recent_turns=2, max_turns=4, block_tool_protocol=False)
        )
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Old durable fact: use /repo/tasks/protocol.txt. " * 12},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_protocol_secret",
                            "name": "Read",
                            "input": {"file_path": "/repo/tasks/protocol.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_protocol_secret",
                            "content": "TOOL_PAYLOAD_PROTOCOL_SECRET",
                        }
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Old conclusion remains open. " * 12}]},
                {"role": "user", "content": "Recent user turn must remain."},
                {"role": "assistant", "content": "Recent assistant turn must remain."},
            ],
        }

        def broken_apply(_body, _plan, _summary):
            return {"model": "claude-sonnet-4-6", "messages": []}

        with patch.object(manual, "apply_old_context_summary", broken_apply):
            summarized, meta = asyncio.run(manual.maybe_summarize_old_context(
                body,
                exact_cache_enabled=False,
                get_cached_summary=lambda _key: None,
                set_cached_summary=lambda _key, _value: None,
                fetch_summary=self._synthetic_fixture_fetch,
            ))

        self.assertEqual(summarized, body)
        self.assertEqual(meta["status"], "bypass")
        self.assertEqual(meta["reason"], "tool-protocol-reconstruction-mismatch")
        self.assertFalse(meta["applied"])
        self.assertFalse(meta["preservation_check"]["ok"])
        feedback = self._summary_feedback_for_meta(meta)
        self.assertEqual(feedback["outcome"], "bypassed")
        self.assertEqual(feedback["enhanced_crunch"]["failure_state"], "tool-protocol-reconstruction-mismatch")
        self._assert_feedback_metadata_only(
            feedback,
            (
                "/repo/tasks/protocol.txt",
                "toolu_protocol_secret",
                "TOOL_PAYLOAD_PROTOCOL_SECRET",
                "Recent user turn must remain.",
                "Recent assistant turn must remain.",
            ),
        )

    def test_old_context_lifecycle_payload_hashes_malformed_policy_ids_and_redacts_raw_fields(self):
        payload = cli._old_context_summary_lifecycle_payload(
            "dry-run",
            {
                "schema": "agentflow.old_context_summary_dry_run.v1",
                "ok": True,
                "dry_run": True,
                "read_only": True,
                "policy": {
                    "policy_source": "managed-recommended",
                    "rule_id": "/private/project/raw_prompt_rule",
                    "candidate_id": "candidate with secret message",
                    "model": "claude-haiku-4-5-20251001",
                    "placement": "system",
                    "canary": {"enabled": True, "fraction": 0.5},
                    "safety_stop": {"enabled": True},
                    "prompt": "RAW_PROMPT_SHOULD_NOT_LEAK",
                    "provider_body": {"content": "RAW_PROVIDER_BODY_SHOULD_NOT_LEAK"},
                },
                "summary": {
                    "sampled_call_count": 2,
                    "eligible_call_count": 1,
                    "projected_saved_tokens": 2000,
                    "projected_net_savings_usd": 0.004,
                    "cache_key": "cache-key-lifecycle-secret",
                    "request_id": "req_lifecycle_secret",
                    "tenant_id": "tenant-lifecycle-secret",
                    "generated_summary": "RAW_GENERATED_SUMMARY_SHOULD_NOT_LEAK",
                },
                "groups": [
                    {
                        "blocker": "eligible",
                        "call_count": 1,
                        "source_surface": "anthropic_messages",
                        "category": "chat",
                        "file_path": "/private/project/secret.txt",
                        "tool_payload": "TOOL_PAYLOAD_LIFECYCLE_SECRET",
                    }
                ],
            },
        )

        self.assertIsNotNone(payload)
        metadata = payload["metadata"]
        self.assertTrue(metadata["rule_id"].startswith("sha256:"))
        self.assertTrue(metadata["candidate_id"].startswith("sha256:"))
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "/private/project/raw_prompt_rule",
            "candidate with secret message",
            "RAW_PROMPT_SHOULD_NOT_LEAK",
            "RAW_PROVIDER_BODY_SHOULD_NOT_LEAK",
            "RAW_GENERATED_SUMMARY_SHOULD_NOT_LEAK",
            "cache-key-lifecycle-secret",
            "req_lifecycle_secret",
            "tenant-lifecycle-secret",
            "/private/project/secret.txt",
            "TOOL_PAYLOAD_LIFECYCLE_SECRET",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(self._keys_in(payload) - {"command"}))
        self.assertFalse(metadata["privacy"]["cache_keys_included"])
        self.assertFalse(metadata["privacy"]["request_ids_included"])
        self.assertFalse(metadata["privacy"]["file_paths_included"])


if __name__ == "__main__":
    unittest.main()
