import io
import json
import unittest
from unittest.mock import patch

from agentflow_proxy import codex_app_proxy


class FailingStore:
    def log_codex_app_event(self, **kwargs):
        raise RuntimeError("database is locked")


class CapturingStore:
    def __init__(self):
        self.events = []

    def log_codex_app_event(self, **kwargs):
        self.events.append(kwargs)


class CodexAppProxyTelemetryTest(unittest.TestCase):
    def test_locked_telemetry_store_does_not_interrupt_relay_recording(self):
        request_started = {}
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"input": "hello"},
        }

        with patch.object(codex_app_proxy, "store", FailingStore()):
            with patch.object(codex_app_proxy.sys, "stderr", io.StringIO()) as stderr:
                codex_app_proxy._record_message(
                    json.dumps(message),
                    direction="client_to_server",
                    session_id="session-a",
                    request_started=request_started,
                )

        self.assertIn("1", request_started)
        self.assertIn("database is locked", stderr.getvalue())

    def test_non_json_and_non_turn_start_pass_through_with_not_applied_metadata(self):
        raw = "not json"
        forwarded, metadata = codex_app_proxy._optimize_client_message(raw)
        self.assertEqual(forwarded, raw)
        self.assertEqual(metadata["routing"]["status"], "not-applied")
        self.assertEqual(metadata["routing"]["reason"], "non-json-frame")

        message = '{ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"input": "hello"} }'
        forwarded, metadata = codex_app_proxy._optimize_client_message(message)
        self.assertEqual(forwarded, message)
        self.assertEqual(metadata["crunch"]["status"], "not-applied")
        self.assertEqual(metadata["crunch"]["reason"], "method-not-eligible")

    def test_action_like_turn_start_passes_through_byte_for_byte(self):
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "input": [{"type": "text", "text": "inspect this"}],
                "command": "python -m unittest",
            },
        }
        raw = json.dumps(message, indent=2)

        forwarded, metadata = codex_app_proxy._optimize_client_message(raw)

        self.assertEqual(forwarded, raw)
        self.assertEqual(metadata["routing"]["status"], "not-applied")
        self.assertEqual(metadata["routing"]["reason"], "action-like-params")

    def test_safe_turn_start_crunches_large_text_without_persisting_raw_prompt(self):
        secret = "secret raw prompt must not be persisted "
        large_text = secret + ("alpha beta gamma delta " * 400)
        message = {
            "jsonrpc": "2.0",
            "id": "turn-1",
            "method": "turn/start",
            "params": {
                "threadId": "thread-1",
                "input": [
                    {"type": "text", "text": large_text},
                    {"type": "text", "text": large_text},
                ],
            },
        }
        raw = json.dumps(message)

        forwarded, metadata = codex_app_proxy._optimize_client_message(raw)
        forwarded_obj = json.loads(forwarded)

        self.assertNotEqual(forwarded, raw)
        self.assertIn("exact duplicate text block omitted", forwarded_obj["params"]["input"][1]["text"])
        self.assertEqual(metadata["crunch"]["status"], "applied")
        self.assertGreater(metadata["crunch"]["saved_chars"], 0)
        self.assertEqual(metadata["routing"]["status"], "not-applicable")

        store = CapturingStore()
        with patch.object(codex_app_proxy, "store", store):
            codex_app_proxy._record_message(
                forwarded,
                direction="client_to_server",
                session_id="session-crunch",
                request_started={},
                optimization_metadata=metadata,
            )

        [event] = store.events
        self.assertEqual(event["method"], "turn/start")
        self.assertGreater(event["input_text_chars"], 0)
        self.assertEqual(json.loads(event["crunch_json"])["status"], "applied")
        self.assertNotIn(secret, json.dumps(event))

    def test_explicit_model_field_is_routed_only_when_policy_matches(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-route",
            "method": "turn/start",
            "params": {
                "model": "claude-sonnet-4-6",
                "input": [{"type": "text", "text": "short answer please"}],
            },
        }
        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
        forwarded_obj = json.loads(forwarded)

        self.assertEqual(forwarded_obj["params"]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(metadata["routing"]["status"], "applied")
        self.assertEqual(metadata["routing"]["model_field"], "model")

        thinking_message = {
            "jsonrpc": "2.0",
            "id": "turn-thinking",
            "method": "turn/start",
            "params": {
                "model": "claude-sonnet-4-6",
                "effort": "high",
                "input": [{"type": "text", "text": "short answer please"}],
            },
        }
        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(thinking_message))
        forwarded_obj = json.loads(forwarded)

        self.assertEqual(forwarded_obj["params"]["model"], "claude-sonnet-4-6")
        self.assertEqual(metadata["routing"]["status"], "skipped")
        self.assertIn("thinking request", metadata["routing"]["reason"])


if __name__ == "__main__":
    unittest.main()
