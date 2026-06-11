from __future__ import annotations

import asyncio
import copy
import json
import unittest

from agentflow_proxy.openai_old_context_summary import (
    SUMMARY_MARKER,
    default_openai_old_context_summary_policy,
    maybe_apply_openai_old_context_summary,
)


class OpenAIOldContextSummaryApplyTests(unittest.TestCase):
    def _policy(self, *, canary_fraction: float = 1.0, holdout_fraction: float = 0.0) -> dict[str, object]:
        policy = default_openai_old_context_summary_policy()
        policy.update({
            "enabled": True,
            "min_request_chars": 0,
            "min_source_chars": 1,
            "max_summary_cost_usd": 1.0,
            "keep_recent_items": 2,
            "max_summary_chars": 120,
            "blocked_categories": [],
        })
        policy["canary"].update({
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "salt": "test-salt",
        })
        return policy

    def _responses_body(self) -> dict[str, object]:
        old_text = "old secret context should disappear from forwarded body. " * 80
        return {
            "model": "gpt-5.4",
            "instructions": "developer instructions stay at top level",
            "stream": True,
            "text": {"format": {"type": "json_schema", "name": "kept"}},
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": old_text + "one"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": old_text + "two"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "recent user stays"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "recent assistant stays"}]},
            ],
        }

    def _chat_body(self) -> dict[str, object]:
        old_text = "old chat secret should disappear from forwarded body. " * 80
        return {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "system instruction stays"},
                {"role": "user", "content": old_text + "one"},
                {"role": "assistant", "content": old_text + "two"},
                {"role": "user", "content": "recent user stays"},
                {"role": "assistant", "content": "recent assistant stays"},
            ],
        }

    def test_default_policy_leaves_request_unchanged(self) -> None:
        body = self._responses_body()

        async def fetch_summary(_request):
            raise AssertionError("disabled policy must not fetch a summary")

        new_body, meta = asyncio.run(maybe_apply_openai_old_context_summary(
            body=copy.deepcopy(body),
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=True,
            fetch_summary=fetch_summary,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
        ))

        self.assertEqual(new_body, body)
        self.assertEqual(meta["status"], "disabled")
        self.assertFalse(meta["applied"])

    def test_canary_applies_bounded_summary_and_preserves_recent_responses_context(self) -> None:
        body = self._responses_body()
        fetches = []
        cache = {}

        async def fetch_summary(request):
            fetches.append(request)
            return {
                "summary": "bounded continuity summary",
                "summary_status_code": 200,
                "summary_input_tokens": 100,
                "summary_output_tokens": 6,
                "summary_cost_est_usd": 0.00012,
            }

        new_body, meta = asyncio.run(maybe_apply_openai_old_context_summary(
            body=copy.deepcopy(body),
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=True,
            fetch_summary=fetch_summary,
            get_cached_summary=lambda key: cache.get(key),
            set_cached_summary=lambda key, value: cache.__setitem__(key, value),
            policy=self._policy(),
        ))

        rendered_body = json.dumps(new_body, sort_keys=True)
        rendered_meta = json.dumps(meta, sort_keys=True)
        self.assertEqual(meta["status"], "applied")
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["canary"]["cohort"], "canary_applied")
        self.assertEqual(meta["summary_cost_est_usd"], 0.00012)
        self.assertFalse(meta["summary_cache_hit"])
        self.assertEqual(len(fetches), 1)
        self.assertIn(SUMMARY_MARKER, rendered_body)
        self.assertIn("bounded continuity summary", rendered_body)
        self.assertIn("recent user stays", rendered_body)
        self.assertIn("recent assistant stays", rendered_body)
        self.assertNotIn("old secret context", rendered_body)
        self.assertNotIn("bounded continuity summary", rendered_meta)
        self.assertNotIn("old secret context", rendered_meta)
        self.assertFalse(meta["privacy"]["raw_summary_included"])
        self.assertFalse(meta["privacy"]["raw_source_included"])

    def test_holdout_forwards_original_request_without_fetching(self) -> None:
        body = self._chat_body()

        async def fetch_summary(_request):
            raise AssertionError("holdout must not fetch a summary")

        new_body, meta = asyncio.run(maybe_apply_openai_old_context_summary(
            body=copy.deepcopy(body),
            path="/v1/chat/completions",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            fetch_summary=fetch_summary,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
            policy=self._policy(canary_fraction=0.0, holdout_fraction=1.0),
        ))

        self.assertEqual(new_body, body)
        self.assertEqual(meta["status"], "holdout")
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["canary"]["cohort"], "holdout")

    def test_summary_cache_hit_applies_without_provider_fetch(self) -> None:
        body = self._chat_body()
        cache_value = {
            "summary": "cached continuity summary",
            "summary_cost_est_usd": 0.0,
            "summary_input_tokens": 100,
            "summary_output_tokens": 5,
        }

        async def fetch_summary(_request):
            raise AssertionError("cache hit must not fetch a summary")

        new_body, meta = asyncio.run(maybe_apply_openai_old_context_summary(
            body=copy.deepcopy(body),
            path="/v1/chat/completions",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            fetch_summary=fetch_summary,
            get_cached_summary=lambda _key: cache_value,
            set_cached_summary=lambda _key, _value: None,
            policy=self._policy(),
        ))

        rendered_body = json.dumps(new_body, sort_keys=True)
        self.assertEqual(meta["status"], "applied")
        self.assertTrue(meta["summary_cache_hit"])
        self.assertIn("cached continuity summary", rendered_body)
        self.assertNotIn("old chat secret", rendered_body)

    def test_summary_provider_error_fails_closed(self) -> None:
        body = self._responses_body()

        async def fetch_summary(_request):
            return {
                "summary": None,
                "summary_status_code": 500,
                "summary_error": "upstream failed with secret-free message",
            }

        new_body, meta = asyncio.run(maybe_apply_openai_old_context_summary(
            body=copy.deepcopy(body),
            path="/v1/responses",
            requested_model="gpt-5.4",
            category="chat",
            stream=False,
            fetch_summary=fetch_summary,
            get_cached_summary=lambda _key: None,
            set_cached_summary=lambda _key, _value: None,
            policy=self._policy(),
        ))

        self.assertEqual(new_body, body)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason_codes"], ["summary_empty_or_malformed"])
        self.assertFalse(meta["applied"])


if __name__ == "__main__":
    unittest.main()
