from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentflow_proxy.provider_adoption import (
    build_provider_tool_adoption_report,
    capture_provider_tool_adoption,
    provider_tool_adoption_report_cli,
)
from agentflow_proxy.store import Store


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
