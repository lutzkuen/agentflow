import io
import json
import tempfile
import unittest
from unittest.mock import patch

from agentflow_proxy import codex_app_proxy
from agentflow_proxy.store import Store


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
        self.assertEqual(metadata["cache"]["status"], "skipped")
        self.assertEqual(metadata["cache"]["reason"], "action-like-params")

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

    def test_codex_repeated_scaffolding_crunch_preserves_latest_task_tail(self):
        scaffold = "## Agent context scaffold\n" + ("stable instruction header " * 80)
        log_block = "## Repeated command log\n" + ("same deterministic log line " * 80)
        older_details = "## Older context\n" + ("old file context " * 2200)
        latest_task = "## Current task\nImplement the billing-free local proxy change."
        message = {
            "jsonrpc": "2.0",
            "id": "turn-scaffold",
            "method": "turn/start",
            "params": {
                "threadId": "thread-scaffold",
                "input": [
                    {"type": "text", "text": f"{scaffold}\n\n{log_block}\n\n{older_details}"},
                    {"type": "text", "text": f"{scaffold}\n\n{log_block}\n\n{latest_task}"},
                ],
            },
        }

        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
        forwarded_obj = json.loads(forwarded)
        first_text = forwarded_obj["params"]["input"][0]["text"]
        latest_text = forwarded_obj["params"]["input"][1]["text"]
        crunch = metadata["crunch"]

        self.assertLess(len(forwarded), len(json.dumps(message)))
        self.assertIn("older Codex input block shortened", first_text)
        self.assertIn("repeated Codex input section omitted", latest_text)
        self.assertIn(latest_task, latest_text)
        self.assertEqual(crunch["status"], "applied")
        self.assertEqual(crunch["reason"], "codex-repeated-scaffolding-crunched")
        self.assertGreater(crunch["saved_chars"], 0)
        pattern_types = {pattern["type"] for pattern in crunch["codex_patterns"]}
        self.assertEqual(pattern_types, {"older_input_head_tail", "repeated_input_section"})
        self.assertGreater(crunch["repeated_codex_sections_replaced"], 0)
        self.assertGreater(crunch["older_codex_input_blocks_shortened"], 0)

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

    def test_safe_codex_turn_exact_cache_hit_when_enabled(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            message = {
                "jsonrpc": "2.0",
                "id": "turn-cache-1",
                "method": "turn/start",
                "params": {
                    "input": [{"type": "text", "text": "repeatable deterministic prompt"}],
                    "temperature": 0,
                },
            }
            response = {
                "jsonrpc": "2.0",
                "id": "turn-cache-1",
                "result": {"message": "cached answer"},
            }

            with patch.object(codex_app_proxy, "store", test_store), patch.object(codex_app_proxy, "CODEX_APP_CACHE", True):
                forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                self.assertEqual(metadata["cache"]["status"], "miss")
                self.assertTrue(metadata["cache"]["eligible"])
                pending = {
                    "turn-cache-1": {
                        "cache_key": metadata["cache"]["_agentflow_cache_key"],
                        "request_chars": len(forwarded),
                        "file_deps": [],
                    }
                }
                codex_app_proxy._maybe_store_codex_cache_response(json.dumps(response), pending_cache=pending)

                second = dict(message)
                second["id"] = "turn-cache-2"
                forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(second))

            self.assertEqual(metadata["cache"]["status"], "hit")
            self.assertEqual(metadata["cache"]["hit_type"], "exact")
            replay = json.loads(metadata["cache"]["_agentflow_replay_frame"])
            self.assertEqual(replay["id"], "turn-cache-2")
            self.assertEqual(replay["result"]["message"], "cached answer")
            self.assertEqual(json.loads(forwarded)["id"], "turn-cache-2")
            test_store.conn.close()

    def test_file_dependency_change_invalidates_codex_cache_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp, tempfile.TemporaryDirectory() as root:
            test_store = Store(tmp.name)
            path = f"{root}/notes.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write("before")
            message = {
                "jsonrpc": "2.0",
                "id": "turn-file-1",
                "method": "turn/start",
                "params": {
                    "input": [{"type": "text", "text": f"summarize {path}"}],
                    "temperature": 0,
                },
            }
            response = {
                "jsonrpc": "2.0",
                "id": "turn-file-1",
                "result": {"message": "before summary"},
            }

            with (
                patch.object(codex_app_proxy, "store", test_store),
                patch.object(codex_app_proxy, "CODEX_APP_CACHE", True),
                patch("agentflow_proxy.cache.CACHE_FILE_WATCH_ROOT", root),
            ):
                forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                pending = {
                    "turn-file-1": {
                        "cache_key": metadata["cache"]["_agentflow_cache_key"],
                        "request_chars": len(forwarded),
                        "file_deps": codex_app_proxy.cache_file_dependency_snapshots(message["params"]),
                    }
                }
                codex_app_proxy._maybe_store_codex_cache_response(json.dumps(response), pending_cache=pending)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("after")

                second = dict(message)
                second["id"] = "turn-file-2"
                _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(second))

            self.assertEqual(metadata["cache"]["status"], "miss")
            self.assertEqual(metadata["cache"]["reason"], "file-dependency-changed")
            self.assertNotIn("_agentflow_replay_frame", metadata["cache"])
            test_store.conn.close()


if __name__ == "__main__":
    unittest.main()
