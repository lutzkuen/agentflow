import io
import asyncio
import json
import os
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


class ManagedResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class ManagedCodexClient:
    calls = []

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json):
        self.__class__.calls.append({"method": "post", "url": url, "json": json})
        return ManagedResponse(body={
            "target_model": "gpt-5-codex",
            "confidence": 0.7,
            "policy_id": "codex-policy-1",
            "reason": "metadata-only Codex policy candidate",
            "optimization_unit_id": 77,
        })

    async def patch(self, url, json):
        self.__class__.calls.append({"method": "patch", "url": url, "json": json})
        return ManagedResponse(body={"ok": True})


class CodexAppProxyTelemetryTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_RECOMMENDATION_ENABLED",
        "AGENTFLOW_RECOMMENDATION_SERVER_URL",
        "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        ManagedCodexClient.calls = []

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

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

    def test_codex_managed_feedback_defaults_off_without_server_call(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-managed-disabled",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "local only"}],
            },
        }
        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))

        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedCodexClient):
            pending = asyncio.run(codex_app_proxy._attach_codex_managed_recommendation(forwarded, metadata))

        managed = metadata["routing"]["managed_recommendation"]
        self.assertEqual(managed["status"], "skipped")
        self.assertEqual(managed["reason"], "disabled")
        self.assertFalse(managed["enabled"])
        self.assertEqual(ManagedCodexClient.calls, [])
        self.assertEqual(pending["managed"], managed)

    def test_codex_managed_feedback_posts_unit_and_patches_sanitized_outcome(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        secret = "raw codex prompt secret"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-managed",
            "method": "turn/start",
            "params": {
                "threadId": "thread-managed",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": secret}],
                "temperature": 0,
                "transcript": "must not leave local machine",
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-managed",
            "result": {"summary": "ok", "raw_response": "must be stripped"},
        }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                    with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedCodexClient):
                        pending = await codex_app_proxy._attach_codex_managed_recommendation(forwarded, metadata)
                        request_started = {}
                        pending_managed = {}
                        with patch.object(codex_app_proxy, "store", test_store):
                            start_event_id = codex_app_proxy._record_message(
                                forwarded,
                                direction="client_to_server",
                                session_id="session-managed",
                                request_started=request_started,
                                optimization_metadata=metadata,
                            )
                            pending["start_event_id"] = start_event_id
                            pending_managed["turn-managed"] = pending
                            await codex_app_proxy._record_codex_managed_outcome(
                                json.dumps(response),
                                session_id="session-managed",
                                request_started=request_started,
                                pending_managed=pending_managed,
                            )
                        row = test_store.conn.execute(
                            "select routing_json from codex_app_events where id = ?",
                            (start_event_id,),
                        ).fetchone()
                        return json.loads(row["routing_json"])
                finally:
                    test_store.conn.close()

        routing = asyncio.run(run_fixture())

        self.assertEqual([call["method"] for call in ManagedCodexClient.calls], ["post", "patch"])
        self.assertEqual(ManagedCodexClient.calls[0]["url"], "http://managed.test/v1/recommendation")
        self.assertEqual(ManagedCodexClient.calls[1]["url"], "http://managed.test/v1/optimization-units/77/outcome")
        unit = ManagedCodexClient.calls[0]["json"]
        outcome = ManagedCodexClient.calls[1]["json"]
        self.assertEqual(unit["source_surface"], "codex_turn")
        self.assertEqual(unit["granularity"], "agent_turn")
        self.assertEqual(outcome["source_surface"], "codex_turn")
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["managed_recommendation"]["optimization_unit_id"], 77)
        forbidden = {"prompt", "messages", "content", "raw_request", "raw_response", "params", "transcript"}
        self.assertTrue(forbidden.isdisjoint(self._keys_in(unit)))
        self.assertTrue(forbidden.isdisjoint(self._keys_in(outcome)))
        self.assertNotIn(secret, json.dumps(unit))
        self.assertNotIn(secret, json.dumps(outcome))
        self.assertNotIn("must not leave", json.dumps(unit))
        self.assertNotIn("must be stripped", json.dumps(outcome))
        managed = routing["managed_recommendation"]
        self.assertEqual(managed["status"], "received")
        self.assertFalse(managed["applied"])
        self.assertEqual(managed["apply_reason"], "codex-app-managed-recommendation-observed-only")
        self.assertEqual(managed["outcome_feedback"]["status"], "sent")


if __name__ == "__main__":
    unittest.main()
