from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenclaw.provider_adoption import (
    build_provider_tool_adoption_report,
    capture_provider_tool_adoption,
    provider_tool_adoption_report_cli,
)
from tokenclaw.provider_adoption_gate import build_provider_adoption_gate
from tokenclaw.store import Store


FORBIDDEN_ADOPTION_VALUES = (
    "toolu_raw_secret_123",
    "toolu_stream_secret_789",
    "toolu_missing_stream_secret",
    "call_secret_456",
    "call_chat_secret_789",
    "call_response_secret_321",
    "raw-session-id",
    "openai-session",
    "chat-session-secret",
    "responses-session-secret",
    "secret file body",
    "stream secret payload",
    "chat secret payload",
    "responses secret payload",
    "missing-id-payload",
    "/tmp/private.py",
    "/home/lutz/private/provider_adoption_secret.py",
)


def _assert_adoption_privacy_clean(testcase: unittest.TestCase, payload: object) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_ADOPTION_VALUES:
        testcase.assertNotIn(forbidden, rendered)
    for forbidden_key in (
        '"raw_prompt"',
        '"raw_response"',
        '"request_json"',
        '"response_json"',
        '"provider_body"',
        '"tool_payload"',
        '"file_path"',
        '"request_id"',
        '"session_id"',
        '"tool_id"',
        '"cache_key"',
        '"local_policy_path"',
    ):
        testcase.assertNotIn(forbidden_key, rendered)


class ProviderToolAdoptionTests(unittest.TestCase):
    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Store(str(Path(tmp.name) / "agentflow.sqlite3"))

    def test_anthropic_tool_use_window_is_fulfilled_without_raw_ids_in_report(self):
        store = self._store()
        request = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_raw_secret_123",
                    "name": "Read",
                    "input": {"file_path": "/tmp/private.py"},
                }
            ]
        }
        capture_provider_tool_adoption(
            store,
            provider="anthropic",
            path="/v1/messages",
            call_id="local-call-1",
            session_id="raw-session-id",
            request_body=request,
            response_body=response,
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            category="tool-light",
            routing_meta={"policy_source": "local-manual", "phase": "tool-execution"},
        )
        follow_up = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_raw_secret_123",
                            "content": "secret file body",
                        }
                    ],
                }
            ],
        }
        capture_provider_tool_adoption(
            store,
            provider="anthropic",
            path="/v1/messages",
            call_id="local-call-2",
            session_id="raw-session-id",
            request_body=follow_up,
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            status_code=200,
            category="tool-result",
            routing_meta={"policy_source": "local-manual", "phase": "tool-execution"},
        )

        report = build_provider_tool_adoption_report(store)
        self.assertEqual(report["status_counts"], {"fulfilled": 1})
        window = report["recent_windows"][0]
        self.assertEqual(window["status"], "fulfilled")
        self.assertEqual(window["source_surface"], "anthropic_messages")
        self.assertEqual(window["workflow_phase"], "tool-execution")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("toolu_raw_secret_123", encoded)
        self.assertNotIn("raw-session-id", encoded)
        self.assertNotIn("secret file body", encoded)
        self.assertNotIn("/tmp/private.py", encoded)
        self.assertFalse(report["privacy"]["tool_ids_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        _assert_adoption_privacy_clean(self, report)

    def test_anthropic_streaming_tool_use_and_malformed_shape_fail_closed_metadata_only(self):
        store = self._store()
        request = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
        capture_provider_tool_adoption(
            store,
            provider="anthropic",
            path="/v1/messages",
            call_id="stream-call-1",
            session_id="raw-session-id",
            request_body=request,
            response_tool_use_ids=["toolu_stream_secret_789"],
            response_tool_use_missing_ids=1,
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            category="tool-light",
        )
        follow_up = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_stream_secret_789",
                            "content": "stream secret payload",
                        }
                    ],
                }
            ],
        }
        capture_provider_tool_adoption(
            store,
            provider="anthropic",
            path="/v1/messages",
            call_id="stream-call-2",
            session_id="raw-session-id",
            request_body=follow_up,
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            category="tool-result",
        )

        report = build_provider_tool_adoption_report(store)

        self.assertEqual(report["status_counts"], {"fulfilled": 1, "unknown": 1})
        reasons = {row["reason"] for row in report["recent_windows"]}
        self.assertIn("unsupported-tool-use-shape-missing-tool-id", reasons)
        _assert_adoption_privacy_clean(self, report)

    def test_openai_orphan_result_and_missing_id_are_metadata_only(self):
        store = self._store()
        request = {
            "model": "gpt-5.4",
            "input": [
                {"type": "function_call_output", "call_id": "call_secret_456", "output": "payload"},
                {"type": "function_call_output", "output": "missing-id-payload"},
            ],
        }
        capture_provider_tool_adoption(
            store,
            provider="openai",
            path="/v1/responses",
            call_id="local-call-3",
            session_id="openai-session",
            request_body=request,
            response_body={"output": []},
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            status_code=200,
            category="tool-result",
            routing_meta={"policy_source": "managed-recommended"},
        )

        report = build_provider_tool_adoption_report(store)
        self.assertEqual(report["status_counts"], {"orphan_result": 1, "unknown": 1})
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("call_secret_456", encoded)
        self.assertNotIn("openai-session", encoded)
        self.assertNotIn("missing-id-payload", encoded)
        self.assertIn("openai_responses", encoded)
        _assert_adoption_privacy_clean(self, report)

    def test_openai_responses_and_chat_tool_shapes_fulfill_without_payload_leaks(self):
        store = self._store()
        capture_provider_tool_adoption(
            store,
            provider="openai",
            path="/v1/responses",
            call_id="responses-call-1",
            session_id="responses-session-secret",
            request_body={"model": "gpt-5.4", "input": [{"role": "user", "content": "hi"}]},
            response_body={
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_response_secret_321",
                        "name": "lookup",
                        "arguments": '{"path":"/home/lutz/private/provider_adoption_secret.py"}',
                    }
                ]
            },
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            status_code=200,
            category="tool-light",
        )
        capture_provider_tool_adoption(
            store,
            provider="openai",
            path="/v1/responses",
            call_id="responses-call-2",
            session_id="responses-session-secret",
            request_body={
                "model": "gpt-5.4",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_response_secret_321",
                        "output": "responses secret payload",
                    }
                ],
            },
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            status_code=200,
            category="tool-result",
        )
        capture_provider_tool_adoption(
            store,
            provider="openai",
            path="/v1/chat/completions",
            call_id="chat-call-1",
            session_id="chat-session-secret",
            request_body={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
            response_body={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_chat_secret_789",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"path":"/home/lutz/private/provider_adoption_secret.py"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            category="tool-light",
        )
        capture_provider_tool_adoption(
            store,
            provider="openai",
            path="/v1/chat/completions",
            call_id="chat-call-2",
            session_id="chat-session-secret",
            request_body={
                "model": "gpt-5.4",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_chat_secret_789",
                        "content": "chat secret payload",
                    }
                ],
            },
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            category="tool-result",
        )

        report = build_provider_tool_adoption_report(store)
        surfaces = {row["source_surface"] for row in report["recent_windows"]}

        self.assertEqual(report["status_counts"], {"fulfilled": 2})
        self.assertIn("openai_responses", surfaces)
        self.assertIn("openai_chat_completions", surfaces)
        _assert_adoption_privacy_clean(self, report)

    def test_pending_window_expires_as_abandoned(self):
        store = self._store()
        capture_provider_tool_adoption(
            store,
            provider="openai",
            path="/v1/chat/completions",
            call_id="local-call-4",
            session_id="chat-session",
            request_body={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
            response_body={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"type": "function_call", "id": "call_stale", "function": {"name": "lookup"}}
                            ]
                        }
                    }
                ]
            },
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            category="tool-light",
            now="2026-06-12T09:00:00+00:00",
        )

        report = build_provider_tool_adoption_report(
            store,
            now="2026-06-12T11:00:01+00:00",
        )
        self.assertEqual(report["status_counts"], {"abandoned": 1})
        self.assertEqual(report["recent_windows"][0]["reason"], "ttl-expired-without-tool-result")
        _assert_adoption_privacy_clean(self, report)

    def test_provider_adoption_gate_counts_underscored_orphan_result_as_risk(self):
        gate = build_provider_adoption_gate(
            [
                {
                    "cohort": "canary_applied",
                    "provider_adoption_windows": [
                        {
                            "status": "orphan_result",
                            "relationship": "fulfilled_tool_result",
                            "tool_use_count": 0,
                            "tool_result_count": 1,
                            "tool_id": "call_secret_456",
                            "session_id": "openai-session",
                            "correlation_digest": "sha256:secret-tool-digest",
                        }
                    ],
                }
            ],
            thresholds={"max_applied_provider_adoption_risk_rate": 0.05},
        )

        self.assertEqual(gate["status"], "blocked")
        applied = gate["cohorts"]["applied"]
        self.assertEqual(applied["orphan_result_count"], 1)
        self.assertEqual(applied["risk_window_count"], 1)
        _assert_adoption_privacy_clean(self, gate)

    def test_cli_emits_report(self):
        store = self._store()
        capture_provider_tool_adoption(
            store,
            provider="anthropic",
            path="/v1/messages",
            call_id="local-call-5",
            session_id="cli-session",
            request_body={"model": "claude-sonnet-4-6", "messages": []},
            response_body={"content": [{"type": "tool_use", "id": "toolu_cli"}]},
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            status_code=200,
            category="tool-light",
        )
        from io import StringIO

        out = StringIO()
        code = provider_tool_adoption_report_cli(["--db", store.path], stdout=out)
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["schema"], "agentflow.provider_tool_adoption_report.v1")
        self.assertEqual(payload["status_counts"], {"pending": 1})


if __name__ == "__main__":
    unittest.main()
