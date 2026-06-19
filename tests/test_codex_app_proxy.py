import io
import asyncio
import functools
import importlib
import inspect
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agentflow_proxy import codex_app_policy as codex_app_policy_module
from agentflow_proxy import codex_app_proxy
from agentflow_proxy import recommendations
from agentflow_proxy import stats as stats_views
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
    feedback_error = None
    feedback_status_code = 200

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json, headers=None):
        self.__class__.calls.append({"method": "post", "url": url, "json": json, "headers": dict(headers or {})})
        return ManagedResponse(body={
            "target_model": "gpt-5-codex",
            "confidence": 0.7,
            "policy_id": "codex-policy-1",
            "reason": "metadata-only Codex policy candidate",
            "optimization_unit_id": 77,
        })

    async def patch(self, url, json, headers=None):
        self.__class__.calls.append({"method": "patch", "url": url, "json": json, "headers": dict(headers or {})})
        if self.__class__.feedback_error is not None:
            raise self.__class__.feedback_error
        return ManagedResponse(status_code=self.__class__.feedback_status_code, body={"ok": True}, text="feedback failed")


class CodexAppProxyTelemetryTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_RECOMMENDATION_ENABLED",
        "AGENTFLOW_RECOMMENDATION_SERVER_URL",
        "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS",
        "AGENTFLOW_POLICY_DECISION_ENABLED",
        "AGENTFLOW_POLICY_DECISION_CANARY_FRACTION",
        "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS",
        "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        ManagedCodexClient.calls = []
        ManagedCodexClient.feedback_error = None
        ManagedCodexClient.feedback_status_code = 200

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

    def _record_codex_turn(
        self,
        *,
        request_id: str,
        store: Store,
        request_started: dict,
        active_turn_windows: dict,
        session_id: str = "ws-session-alert",
        thread_id: str = "thread-alert-session",
        input_text: str = "Summarize the completed work.",
        result_text: str = "done",
        optimization_metadata: dict | None = None,
    ) -> None:
        start = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "turn/start",
            "params": {
                "threadId": thread_id,
                "model": "gpt-5.3-codex",
                "input": [{"type": "text", "text": input_text}],
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"message": result_text},
        }
        with patch.object(codex_app_proxy, "store", store):
            codex_app_proxy._record_message(
                json.dumps(start),
                direction="client_to_server",
                session_id=session_id,
                request_started=request_started,
                optimization_metadata=optimization_metadata,
                active_turn_windows=active_turn_windows,
            )
            codex_app_proxy._record_message(
                json.dumps(response),
                direction="server_to_client",
                session_id=session_id,
                request_started=request_started,
                active_turn_windows=active_turn_windows,
            )

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

    def test_upstream_relay_uses_expanded_websocket_frame_limit(self):
        self.assertGreaterEqual(codex_app_proxy.CODEX_APP_WEBSOCKET_MAX_SIZE, 64 * 1024 * 1024)
        self.assertIn(
            "websockets.connect(upstream_url, max_size=CODEX_APP_WEBSOCKET_MAX_SIZE)",
            inspect.getsource(codex_app_proxy.relay),
        )

    def test_quota_and_token_usage_metadata_is_allowlisted(self):
        raw_prompt = "secret raw prompt must not be stored"
        raw_command = "secret shell command must not be stored"
        raw_transcript = "secret transcript must not be stored"
        capture = CapturingStore()
        token_message = {
            "jsonrpc": "2.0",
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-private",
                "prompt": raw_prompt,
                "transcript": raw_transcript,
                "usage": {
                    "inputTokens": 1200,
                    "cachedInputTokens": 300,
                    "outputTokens": 80,
                    "reasoningOutputTokens": 20,
                },
            },
        }
        quota_message = {
            "jsonrpc": "2.0",
            "method": "account/rateLimits/updated",
            "params": {
                "command": raw_command,
                "rateLimits": {
                    "planType": "pro",
                    "primary": {"usedPercent": 92.5, "remaining": 42, "resetAfterSeconds": 1800},
                    "secondary": {"usedPercent": 30.0, "remaining": 1200, "resetAfterSeconds": 7200},
                },
            },
        }

        with patch.object(codex_app_proxy, "store", capture):
            codex_app_proxy._record_message(
                json.dumps(token_message),
                direction="server_to_client",
                session_id="session-private",
                request_started={},
            )
            codex_app_proxy._record_message(
                json.dumps(quota_message),
                direction="server_to_client",
                session_id="session-private",
                request_started={},
            )

        rendered = json.dumps(capture.events)
        self.assertNotIn(raw_prompt, rendered)
        self.assertNotIn(raw_command, rendered)
        self.assertNotIn(raw_transcript, rendered)
        metadata = [json.loads(event["metadata_json"]) for event in capture.events if event.get("metadata_json")]
        self.assertEqual({row["kind"] for row in metadata}, {"token_usage", "rate_limits"})
        token_usage = next(row["token_usage"] for row in metadata if row["kind"] == "token_usage")
        self.assertEqual(token_usage["input_tokens"], 1200)
        self.assertEqual(token_usage["total_tokens"], 1600)
        rate_limits = next(row["rate_limits"] for row in metadata if row["kind"] == "rate_limits")
        self.assertEqual(rate_limits["plan_type"], "pro")
        self.assertEqual(rate_limits["pressure"], "high")
        self.assertEqual(rate_limits["scopes"][0]["remaining_bucket"], "10_99")
        self.assertEqual(rate_limits["scopes"][0]["reset_bucket"], "1m_1h")

    def test_nested_token_usage_metadata_is_allowlisted(self):
        raw_prompt = "nested raw prompt must not be stored"
        raw_transcript = "nested raw transcript must not be stored"
        capture = CapturingStore()
        token_message = {
            "jsonrpc": "2.0",
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-private",
                "prompt": raw_prompt,
                "details": {
                    "transcript": raw_transcript,
                    "lastTurnUsage": {
                        "inputTokens": 10,
                        "cachedInputTokens": 2,
                        "outputTokens": 3,
                    },
                    "totalTokenUsage": {
                        "inputTokens": 1200,
                        "cacheReadInputTokens": 300,
                        "outputTokens": 80,
                        "reasoningTokens": 20,
                    },
                },
            },
        }

        with patch.object(codex_app_proxy, "store", capture):
            codex_app_proxy._record_message(
                json.dumps(token_message),
                direction="server_to_client",
                session_id="session-private",
                request_started={},
            )

        rendered = json.dumps(capture.events)
        self.assertNotIn(raw_prompt, rendered)
        self.assertNotIn(raw_transcript, rendered)
        metadata = [json.loads(event["metadata_json"]) for event in capture.events if event.get("metadata_json")]
        self.assertEqual(len(metadata), 1)
        token_usage = metadata[0]["token_usage"]
        self.assertEqual(token_usage["input_tokens"], 1200)
        self.assertEqual(token_usage["cached_input_tokens"], 300)
        self.assertEqual(token_usage["output_tokens"], 80)
        self.assertEqual(token_usage["reasoning_output_tokens"], 20)
        self.assertEqual(token_usage["total_tokens"], 1600)

    def test_codex_app_session_spend_alert_warns_once_per_threshold_window(self):
        raw_prompt = "secret raw prompt must not be logged"
        raw_result = "secret raw result must not be logged"
        metadata = {
            "routing": {
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5.3-codex",
                "workflow_phase": "summary",
                "policy_source": "local-default",
            },
            "crunch": {
                "changed": True,
                "applied": True,
                "tokens_before_est": 1_000,
                "workflow_phase": "summary",
                "policy_source": "local-default",
            },
            "cache": {
                "status": "miss",
                "reason": "exact-miss",
                "workflow_phase": "summary",
                "policy_source": "local-default",
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            request_started: dict = {}
            active_turn_windows: dict = {}
            codex_app_proxy._codex_app_session_alert_windows.clear()
            try:
                with (
                    patch.object(codex_app_proxy, "CODEX_APP_SESSION_COST_ALERT_USD", 0.02),
                    patch.object(codex_app_proxy, "codex_app_model", return_value="gpt-5.3-codex"),
                    patch.object(codex_app_proxy, "codex_app_processing_mode", return_value="standard"),
                    patch.object(codex_app_proxy.logging, "warning") as warning,
                ):
                    self._record_codex_turn(
                        request_id="alert-1",
                        input_text=raw_prompt,
                        result_text="x" * 3_000,
                        optimization_metadata=metadata,
                        store=test_store,
                        request_started=request_started,
                        active_turn_windows=active_turn_windows,
                    )
                    self.assertEqual(warning.call_count, 0)

                    self._record_codex_turn(
                        request_id="alert-2",
                        input_text=raw_prompt,
                        result_text="y" * 3_000,
                        optimization_metadata=metadata,
                        store=test_store,
                        request_started=request_started,
                        active_turn_windows=active_turn_windows,
                    )
                    self.assertEqual(warning.call_count, 1)

                    self._record_codex_turn(
                        request_id="alert-3",
                        input_text=raw_prompt,
                        result_text=raw_result,
                        optimization_metadata=metadata,
                        store=test_store,
                        request_started=request_started,
                        active_turn_windows=active_turn_windows,
                    )
                    self.assertEqual(warning.call_count, 1)

                rendered = warning.call_args.args[0] % warning.call_args.args[1:]
                self.assertIn("Codex app thread_id thread-a daily estimated cost", rendered)
                self.assertIn("2 turns", rendered)
                self.assertIn("phases=summary:2", rendered)
                self.assertIn("crunched_turns=2", rendered)
                self.assertNotIn(raw_prompt, rendered)
                self.assertNotIn(raw_result, rendered)
                self.assertNotIn("params", rendered.lower())
                self.assertNotIn("response", rendered.lower())
            finally:
                codex_app_proxy._codex_app_session_alert_windows.clear()
                test_store.conn.close()

    def test_codex_app_session_spend_alert_ignores_low_cost_sessions(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            request_started: dict = {}
            active_turn_windows: dict = {}
            codex_app_proxy._codex_app_session_alert_windows.clear()
            try:
                with (
                    patch.object(codex_app_proxy, "CODEX_APP_SESSION_COST_ALERT_USD", 0.02),
                    patch.object(codex_app_proxy.logging, "warning") as warning,
                ):
                    self._record_codex_turn(
                        request_id="normal-1",
                        result_text="small result",
                        store=test_store,
                        request_started=request_started,
                        active_turn_windows=active_turn_windows,
                    )

                self.assertEqual(warning.call_count, 0)
            finally:
                codex_app_proxy._codex_app_session_alert_windows.clear()
                test_store.conn.close()

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
        diagnostics = metadata["routing"]["managed_pattern_features"]
        self.assertTrue(diagnostics["present"])
        self.assertEqual(diagnostics["source_surface"], "codex_turn")
        self.assertGreaterEqual(diagnostics["pattern_hash_count"], 3)
        self.assertTrue(diagnostics["pattern_hash"].startswith("sha256:"))
        self.assertFalse(diagnostics["raw_pattern_strings_included"])
        self.assertNotIn(secret, json.dumps(diagnostics, sort_keys=True))

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

    def test_turn_start_event_window_is_persisted_as_metadata_only_summary(self):
        secret = "do not persist raw command output"
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                active_windows = {}
                request_started = {}
                start = {
                    "jsonrpc": "2.0",
                    "id": "turn-window",
                    "method": "turn/start",
                    "params": {
                        "threadId": "thread-window",
                        "model": "gpt-5-codex",
                        "input": [{"type": "text", "text": "summarize metadata only"}],
                    },
                }
                signal = {
                    "jsonrpc": "2.0",
                    "method": "item/commandExecution/outputDelta",
                    "params": {
                        "threadId": "thread-window",
                        "chunk": secret,
                    },
                }
                completed = {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {"threadId": "thread-window"},
                }

                with patch.object(codex_app_proxy, "store", test_store):
                    start_event_id = codex_app_proxy._record_message(
                        json.dumps(start),
                        direction="client_to_server",
                        session_id="session-window",
                        request_started=request_started,
                        optimization_metadata={
                            "routing": {
                                "status": "skipped",
                                "reason": "keep requested model",
                                "model_field": "model",
                                "applied": False,
                            },
                            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
                            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
                        },
                        active_turn_windows=active_windows,
                    )
                    codex_app_proxy._record_message(
                        json.dumps(signal),
                        direction="server_to_client",
                        session_id="session-window",
                        request_started=request_started,
                        active_turn_windows=active_windows,
                    )
                    codex_app_proxy._record_message(
                        json.dumps(completed),
                        direction="server_to_client",
                        session_id="session-window",
                        request_started=request_started,
                        active_turn_windows=active_windows,
                    )

                row = test_store.conn.execute(
                    "select event_window_json from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
                window = json.loads(row["event_window_json"])
            finally:
                test_store.conn.close()

        self.assertEqual(window["schema"], "agentflow.codex_app_event_window.v1")
        self.assertEqual(window["event_count"], 3)
        self.assertEqual(window["method_counts"]["turn/start"], 1)
        self.assertEqual(window["method_counts"]["item/commandExecution/outputDelta"], 1)
        self.assertEqual(window["method_counts"]["turn/completed"], 1)
        self.assertEqual(window["direction_counts"]["server_to_client"], 2)
        self.assertEqual(window["model_field_state"], "present")
        self.assertEqual(window["workflow_phase"], "tool_execution")
        self.assertEqual(window["workflow_phase_reason"], "event-window-signal:tool_execution")
        self.assertEqual(window["workflow_phase_source"], "event_window")
        self.assertEqual(window["workflow_phase_confidence"], "high")
        self.assertEqual(window["workflow_phase_signals"], ["item/commandExecution/outputDelta"])
        self.assertGreater(window["server_message_chars"], 0)
        self.assertNotIn(secret, json.dumps(window))

    def test_turn_start_event_window_derives_session_model_state_without_raw_params(self):
        secret = "raw startup instructions must not be stored"
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                active_windows = {}
                model_states = {}
                request_started = {}
                initialize = {
                    "jsonrpc": "2.0",
                    "id": "init-model",
                    "method": "initialize",
                    "params": {
                        "model": "gpt-5-codex",
                        "instructions": secret,
                    },
                }
                start = {
                    "jsonrpc": "2.0",
                    "id": "turn-derived-model",
                    "method": "turn/start",
                    "params": {
                        "threadId": "thread-derived-model",
                        "input": [{"type": "text", "text": "derive model state"}],
                    },
                }

                with patch.object(codex_app_proxy, "store", test_store):
                    codex_app_proxy._record_message(
                        json.dumps(initialize),
                        direction="client_to_server",
                        session_id="session-derived-model",
                        request_started=request_started,
                        active_turn_windows=active_windows,
                        model_states=model_states,
                    )
                    start_event_id = codex_app_proxy._record_message(
                        json.dumps(start),
                        direction="client_to_server",
                        session_id="session-derived-model",
                        request_started=request_started,
                        optimization_metadata={
                            "routing": {
                                "status": "not-applicable",
                                "reason": "codex-turn-start-model-field-absent",
                                "applied": False,
                            },
                            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
                            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
                        },
                        active_turn_windows=active_windows,
                        model_states=model_states,
                    )

                row = test_store.conn.execute(
                    "select event_window_json from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
                window = json.loads(row["event_window_json"])
            finally:
                test_store.conn.close()

        self.assertEqual(window["model_field_state"], "derived_present")
        self.assertEqual(window["model_field"], "model")
        self.assertEqual(window["model_state"]["normalized_model"], "gpt-5-codex")
        self.assertEqual(window["model_state"]["source_method"], "initialize")
        self.assertEqual(window["model_state"]["reason"], "metadata-model-field")
        self.assertNotIn(secret, json.dumps(window))

    def test_codex_model_state_signal_ignores_prompt_payload_shapes(self):
        signal = codex_app_proxy.codex_model_state_signal(
            "turn/start",
            {
                "input": [
                    {
                        "type": "text",
                        "text": "this is user content, not config",
                        "model": "prompt-provided-model-name",
                    }
                ],
            },
        )

        self.assertIsNone(signal)

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

    def test_summary_model_hint_canary_routes_safe_summary_turn(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-summary-hint",
            "method": "turn/start",
            "params": {
                "threadId": "thread-summary-hint",
                "model": "gpt-5.3-codex",
                "input": [{"type": "text", "text": "Summarize the completed work for the run log."}],
                "temperature": 0,
            },
        }

        with (
            patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT", True),
            patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_TARGET", "gpt-5-codex"),
        ):
            forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))

        forwarded_obj = json.loads(forwarded)
        routing = metadata["routing"]
        self.assertEqual(forwarded_obj["params"]["model"], "gpt-5-codex")
        self.assertEqual(routing["status"], "applied")
        self.assertTrue(routing["applied"])
        self.assertEqual(routing["reason"], "safe-summary-model-hint-canary")
        self.assertEqual(routing["requested_model"], "gpt-5.3-codex")
        self.assertEqual(routing["routed_model"], "gpt-5-codex")
        self.assertEqual(routing["workflow_phase"], "summary")
        self.assertEqual(routing["policy_source"], "local-default")
        self.assertEqual(routing["canary"], "codex-app-summary-model-hint")
        self.assertEqual(routing["summary_model_hint"]["status"], "applied")
        self.assertTrue(routing["summary_model_hint"]["eligible"])
        self.assertEqual(routing["summary_model_hint"]["target_model"], "gpt-5-codex")
        self.assertEqual(routing["summary_model_hint"]["requested_model"], "gpt-5.3-codex")
        self.assertEqual(routing["summary_model_hint"]["model_field_state"], "present")
        self.assertEqual(routing["summary_model_hint"]["workflow_phase"], "summary")
        self.assertGreaterEqual(routing["summary_model_hint"]["estimated_cost_delta"]["delta_usd"], 0)

    def test_local_codex_app_rules_enable_summary_hint_and_cache_without_env_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "codex_app_rules.yaml")
            db_path = os.path.join(tmp, "codex.sqlite3")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(
                    """
enabled: true
summary_model_hint:
  enabled: true
  target_model: gpt-5-mini
exact_cache:
  enabled: true
  namespace: local-file-test
"""
                )
            message = {
                "jsonrpc": "2.0",
                "id": "turn-file-backed-summary",
                "method": "turn/start",
                "params": {
                    "threadId": "thread-file-backed-summary",
                    "model": "gpt-5.3-codex",
                    "input": [{"type": "text", "text": "Summarize the completed work for the run log."}],
                    "temperature": 0,
                },
            }

            try:
                with patch.dict(
                    os.environ,
                    {
                        "AGENTFLOW_CODEX_APP_RULES": rules_path,
                        "AGENTFLOW_DB": db_path,
                        "HOME": tmp,
                        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT": "0",
                        "AGENTFLOW_CODEX_APP_CACHE": "0",
                    },
                    clear=False,
                ):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)
                    forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
            finally:
                importlib.reload(codex_app_policy_module)
                importlib.reload(codex_app_proxy)

        forwarded_obj = json.loads(forwarded)
        self.assertEqual(forwarded_obj["params"]["model"], "gpt-5-mini")
        self.assertEqual(metadata["routing"]["status"], "applied")
        self.assertEqual(metadata["routing"]["policy_source"], "local-manual")
        self.assertEqual(metadata["routing"]["target_model"], "gpt-5-mini")
        self.assertEqual(metadata["cache"]["status"], "miss")
        self.assertTrue(metadata["cache"]["enabled"])
        self.assertEqual(metadata["cache"]["policy_source"], "local-manual")
        self.assertEqual(metadata["cache"]["replayability_level"], "local-exact-response")

    def test_codex_app_rules_apply_summary_actions_without_top_level_toggles(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "codex_app_rules.yaml")
            db_path = os.path.join(tmp, "codex.sqlite3")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(
                    """
enabled: true
summary_model_hint:
  enabled: false
exact_cache:
  enabled: false
rules:
  - id: managed-summary-rule
    policy_source: managed-recommended
    conditions:
      app_family: codex
      workflow_phase: summary
      model_field_state: present
      input_size_bucket: small
      cache_eligible: true
      replayability_level: local-exact-response
      has_action_like_params: false
      stale_risk: false
      supported_action_family: routing
    action:
      model_hint: gpt-5-mini
      cache_eligible: true
      cache_eligibility_reason: safe rule-level summary cache
      crunch_profile: codex-repeated-scaffolding
      reason: managed summary rule
    canary:
      fraction: 1.0
      holdout_fraction: 0.0
      salt: rule-apply-test
      unit: source_hash
"""
                )
            message = {
                "jsonrpc": "2.0",
                "id": "turn-rule-summary",
                "method": "turn/start",
                "params": {
                    "threadId": "thread-rule-summary",
                    "model": "gpt-5.3-codex",
                    "input": [{"type": "text", "text": "Summarize the completed work for the run log."}],
                    "temperature": 0,
                },
            }

            try:
                with patch.dict(
                    os.environ,
                    {
                        "AGENTFLOW_CODEX_APP_RULES": rules_path,
                        "AGENTFLOW_DB": db_path,
                        "HOME": tmp,
                        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT": "0",
                        "AGENTFLOW_CODEX_APP_CACHE": "0",
                    },
                    clear=False,
                ):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)
                    forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
            finally:
                importlib.reload(codex_app_policy_module)
                importlib.reload(codex_app_proxy)

        forwarded_obj = json.loads(forwarded)
        self.assertEqual(forwarded_obj["params"]["model"], "gpt-5-mini")
        self.assertEqual(metadata["routing"]["status"], "applied")
        self.assertEqual(metadata["routing"]["reason"], "managed summary rule")
        self.assertEqual(metadata["routing"]["policy_source"], "managed-recommended")
        self.assertEqual(metadata["routing"]["codex_app_rule"]["rule_id"], "managed-summary-rule")
        self.assertTrue(metadata["routing"]["codex_app_rule"]["matched"])
        self.assertEqual(metadata["routing"]["canary_cohort"], "canary_applied")
        self.assertEqual(metadata["crunch"]["status"], "hinted")
        self.assertEqual(metadata["crunch"]["profile_hint"], "codex-repeated-scaffolding")
        self.assertEqual(metadata["cache"]["status"], "miss")
        self.assertTrue(metadata["cache"]["eligible"])
        self.assertEqual(metadata["cache"]["policy_source"], "managed-recommended")
        self.assertEqual(metadata["cache"]["replayability_level"], "local-exact-response")

    def test_codex_app_rules_holdout_keeps_safe_summary_turn_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "codex_app_rules.yaml")
            db_path = os.path.join(tmp, "codex.sqlite3")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(
                    """
enabled: true
rules:
  - id: holdout-summary-rule
    conditions:
      app_family: codex
      workflow_phase: summary
      model_field_state: present
      cache_eligible: true
      has_action_like_params: false
      stale_risk: false
    action:
      model_hint: gpt-5-mini
    canary:
      fraction: 0.0
      holdout_fraction: 1.0
      salt: rule-holdout-test
      unit: source_hash
"""
                )
            message = {
                "jsonrpc": "2.0",
                "id": "turn-rule-holdout",
                "method": "turn/start",
                "params": {
                    "model": "gpt-5.3-codex",
                    "input": [{"type": "text", "text": "Summarize the completed work."}],
                    "temperature": 0,
                },
            }
            raw = json.dumps(message)

            try:
                with patch.dict(os.environ, {"AGENTFLOW_CODEX_APP_RULES": rules_path, "AGENTFLOW_DB": db_path, "HOME": tmp}, clear=False):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)
                    forwarded, metadata = codex_app_proxy._optimize_client_message(raw)
            finally:
                importlib.reload(codex_app_policy_module)
                importlib.reload(codex_app_proxy)

        self.assertEqual(forwarded, raw)
        self.assertEqual(metadata["routing"]["status"], "skipped")
        self.assertEqual(metadata["routing"]["reason"], "codex-app-rule-canary-holdout")
        self.assertEqual(metadata["routing"]["canary_cohort"], "canary_holdout")
        self.assertEqual(metadata["routing"]["codex_app_rule"]["rule_id"], "holdout-summary-rule")
        self.assertFalse(metadata["routing"]["applied"])

    def test_codex_app_rule_explicit_safety_stop_fails_closed_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "codex_app_rules.yaml")
            db_path = os.path.join(tmp, "codex.sqlite3")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(
                    """
enabled: true
rules:
  - id: stopped-summary-rule
    policy_source: managed-recommended
    conditions:
      app_family: codex
      workflow_phase: summary
      model_field_state: present
      cache_eligible: true
      has_action_like_params: false
      stale_risk: false
    action:
      model_hint: gpt-5-mini
      cache_eligible: true
    canary:
      fraction: 1.0
      holdout_fraction: 0.2
      salt: stopped-rule-test
      unit: source_hash
    safety_stop:
      enabled: true
      tripped: true
      reason_codes:
        - applied-error-rate-regression
"""
                )
            message = {
                "jsonrpc": "2.0",
                "id": "turn-rule-stopped",
                "method": "turn/start",
                "params": {
                    "model": "gpt-5.3-codex",
                    "input": [{"type": "text", "text": "Summarize the completed work."}],
                    "temperature": 0,
                },
            }
            raw = json.dumps(message)

            try:
                with patch.dict(os.environ, {"AGENTFLOW_CODEX_APP_RULES": rules_path, "AGENTFLOW_DB": db_path, "HOME": tmp}, clear=False):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)
                    forwarded, metadata = codex_app_proxy._optimize_client_message(raw)
                    pending = codex_app_proxy._attach_codex_canary_lifecycle_pending(forwarded, metadata)
            finally:
                importlib.reload(codex_app_policy_module)
                importlib.reload(codex_app_proxy)

        self.assertEqual(forwarded, raw)
        self.assertEqual(metadata["routing"]["status"], "safety_stopped")
        self.assertEqual(metadata["routing"]["reason"], "local-canary-safety-stop")
        self.assertFalse(metadata["routing"]["applied"])
        self.assertEqual(metadata["routing"]["canary_cohort"], "safety_stopped")
        self.assertEqual(metadata["routing"]["codex_app_rule"]["rule_id"], "stopped-summary-rule")
        self.assertEqual(metadata["routing"]["safety_stop"]["reason_codes"], ["applied-error-rate-regression"])
        self.assertEqual(metadata["routing"]["safety_stop"]["source"], "policy-file")
        self.assertEqual(metadata["cache"]["status"], "skipped")
        self.assertEqual(metadata["cache"]["reason"], "local-canary-safety-stop")
        self.assertNotIn("_agentflow_cache_key", metadata["cache"])
        self.assertEqual(pending["actions"], ["routing", "cache"])
        self.assertNotIn("turn-rule-stopped", json.dumps(metadata))

    def test_codex_app_rule_missing_holdout_evidence_safety_stop_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "codex_app_rules.yaml")
            db_path = os.path.join(tmp, "codex.sqlite3")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(
                    """
enabled: true
rules:
  - id: missing-holdout-rule
    policy_source: managed-recommended
    conditions:
      app_family: codex
      workflow_phase: summary
      model_field_state: present
      cache_eligible: true
      has_action_like_params: false
      stale_risk: false
    action:
      model_hint: gpt-5-mini
    canary:
      fraction: 0.5
      holdout_fraction: 0.5
      salt: missing-holdout-test
      unit: source_hash
    safety_stop:
      enabled: true
      min_outcome_samples: 2
      min_holdout_samples: 1
      window: 10
"""
                )
            test_store = Store(db_path)
            try:
                for idx in range(2):
                    test_store.log_codex_app_event(
                        id=f"start-missing-holdout-{idx}",
                        created_at=f"2026-06-10T10:00:0{idx}+00:00",
                        direction="client_to_server",
                        method="turn/start",
                        request_id=f"req-missing-holdout-{idx}",
                        thread_id=f"thread-missing-holdout-{idx}",
                        message_chars=100,
                        params_chars=80,
                        input_items=1,
                        input_text_chars=48,
                        session_id="session-missing-holdout",
                        routing_json=json.dumps({
                            "status": "applied",
                            "reason": "managed summary rule",
                            "canary": "codex-app-rule",
                            "canary_cohort": "canary_applied",
                            "codex_app_rule": {
                                "rule_id": "missing-holdout-rule",
                                "candidate_id": "missing-holdout-rule",
                                "policy_source": "managed-recommended",
                            },
                        }),
                        cache_json=json.dumps({
                            "status": "skipped",
                            "reason": "codex-app-rule-no-cache-action",
                            "codex_app_rule": {
                                "rule_id": "missing-holdout-rule",
                                "candidate_id": "missing-holdout-rule",
                            },
                        }),
                    )
                    test_store.log_codex_app_event(
                        id=f"end-missing-holdout-{idx}",
                        created_at=f"2026-06-10T10:00:1{idx}+00:00",
                        direction="server_to_client",
                        method=None,
                        request_id=f"req-missing-holdout-{idx}",
                        thread_id=f"thread-missing-holdout-{idx}",
                        message_chars=50,
                        result_chars=20,
                        latency_ms=20,
                        session_id="session-missing-holdout",
                    )
            finally:
                test_store.conn.close()

            message = {
                "jsonrpc": "2.0",
                "id": "turn-missing-holdout-stop",
                "method": "turn/start",
                "params": {
                    "model": "gpt-5.3-codex",
                    "input": [{"type": "text", "text": "Summarize the completed work."}],
                    "temperature": 0,
                },
            }
            raw = json.dumps(message)

            try:
                with patch.dict(os.environ, {"AGENTFLOW_CODEX_APP_RULES": rules_path, "AGENTFLOW_DB": db_path, "HOME": tmp}, clear=False):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)
                    forwarded, metadata = codex_app_proxy._optimize_client_message(raw)
            finally:
                importlib.reload(codex_app_policy_module)
                importlib.reload(codex_app_proxy)

        self.assertEqual(forwarded, raw)
        self.assertEqual(metadata["routing"]["status"], "safety_stopped")
        self.assertIn("missing-holdout-evidence", metadata["routing"]["safety_stop"]["reason_codes"])
        self.assertEqual(metadata["routing"]["safety_stop"]["source"], "recent-local-outcomes")
        self.assertEqual(metadata["routing"]["safety_stop"]["applied_sample_count"], 2)
        self.assertEqual(metadata["routing"]["safety_stop"]["holdout_sample_count"], 0)

    def test_codex_effectiveness_reports_rule_safety_stop_state(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                test_store.log_codex_app_event(
                    id="start-safety-stop-stats",
                    created_at="2026-06-10T12:00:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-safety-stop-stats",
                    thread_id="thread-safety-stop-stats",
                    message_chars=120,
                    params_chars=80,
                    input_items=1,
                    input_text_chars=64,
                    session_id="session-safety-stop-stats",
                    routing_json=json.dumps({
                        "status": "safety_stopped",
                        "reason": "local-canary-safety-stop",
                        "policy_source": "managed-recommended",
                        "canary": "codex-app-rule",
                        "canary_cohort": "safety_stopped",
                        "codex_app_rule": {
                            "rule_id": "stopped-stats-rule",
                            "candidate_id": "stopped-stats-rule",
                        },
                        "safety_stop": {
                            "tripped": True,
                            "status": "stopped",
                            "reason": "local-canary-safety-stop",
                            "reason_codes": ["unsafe-cache-envelope"],
                            "source": "recent-local-outcomes",
                            "rule_id": "stopped-stats-rule",
                            "candidate_id": "stopped-stats-rule",
                            "sample_count": 3,
                            "applied_sample_count": 3,
                            "holdout_sample_count": 1,
                        },
                    }),
                    crunch_json=json.dumps({"status": "skipped", "reason": "local-canary-safety-stop"}),
                    cache_json=json.dumps({"status": "skipped", "reason": "local-canary-safety-stop"}),
                )
                test_store.log_codex_app_event(
                    id="end-safety-stop-stats",
                    created_at="2026-06-10T12:00:01+00:00",
                    direction="server_to_client",
                    method=None,
                    request_id="req-safety-stop-stats",
                    thread_id="thread-safety-stop-stats",
                    message_chars=50,
                    result_chars=20,
                    latency_ms=20,
                    session_id="session-safety-stop-stats",
                )
                payload = asyncio.run(stats_views.stats_codex_effectiveness(test_store, limit=10))
            finally:
                test_store.conn.close()

        self.assertEqual(payload["summary"]["safety_stop_rows"], 1)
        self.assertTrue(payload["safety_stop"]["active"])
        self.assertEqual(payload["safety_stop"]["latest"]["rule_id"], "stopped-stats-rule")
        self.assertEqual(payload["safety_stop"]["latest"]["reason_codes"], ["unsafe-cache-envelope"])
        self.assertEqual(payload["safety_stop"]["reason_code_breakdown"], [{"value": "unsafe-cache-envelope", "count": 1}])
        self.assertFalse(payload["safety_stop"]["privacy"]["raw_params_included"])

    def test_codex_app_rules_pass_through_unsafe_or_unsupported_turns_with_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "codex.sqlite3")

            def run_with_rules(rule_text, message):
                rules_path = os.path.join(tmp, "codex_app_rules.yaml")
                with open(rules_path, "w", encoding="utf-8") as f:
                    f.write(rule_text)
                raw = json.dumps(message)
                try:
                    with patch.dict(os.environ, {"AGENTFLOW_CODEX_APP_RULES": rules_path, "AGENTFLOW_DB": db_path, "HOME": tmp}, clear=False):
                        importlib.reload(codex_app_policy_module)
                        importlib.reload(codex_app_proxy)
                        return raw, codex_app_proxy._optimize_client_message(raw)
                finally:
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)

            safe_rule = """
enabled: true
rules:
  - id: safe-summary-only
    conditions:
      app_family: codex
      workflow_phase: summary
      model_field_state: present
      cache_eligible: true
      has_action_like_params: false
      stale_risk: false
    action:
      model_hint: gpt-5-mini
"""
            cases = [
                (
                    {
                        "jsonrpc": "2.0",
                        "id": "turn-rule-non-summary",
                        "method": "turn/start",
                        "params": {
                            "model": "gpt-5.3-codex",
                            "input": [{"type": "text", "text": "Answer this question in detail."}],
                        },
                    },
                    safe_rule,
                    "workflow-phase-not-summary",
                ),
                (
                    {
                        "jsonrpc": "2.0",
                        "id": "turn-rule-action-like",
                        "method": "turn/start",
                        "params": {
                            "model": "gpt-5.3-codex",
                            "input": [{"type": "text", "text": "Summarize the command output."}],
                            "command": "python -m unittest",
                        },
                    },
                    safe_rule,
                    "action-like-params",
                ),
                (
                    {
                        "jsonrpc": "2.0",
                        "id": "turn-rule-missing-model",
                        "method": "turn/start",
                        "params": {
                            "input": [{"type": "text", "text": "Summarize the completed work."}],
                        },
                    },
                    safe_rule,
                    "codex-turn-start-model-field-absent",
                ),
                (
                    {
                        "jsonrpc": "2.0",
                        "id": "turn-rule-stale-risk",
                        "method": "turn/start",
                        "params": {
                            "model": "gpt-5.3-codex",
                            "input": [{"type": "text", "text": "Summarize the latest state today."}],
                        },
                    },
                    safe_rule,
                    "stale-risk-blockers",
                ),
                (
                    {
                        "jsonrpc": "2.0",
                        "id": "turn-rule-unsupported-action",
                        "method": "turn/start",
                        "params": {
                            "model": "gpt-5.3-codex",
                            "input": [{"type": "text", "text": "Summarize the completed work."}],
                        },
                    },
                    """
enabled: true
rules:
  - id: unsupported-action-rule
    conditions:
      app_family: codex
      workflow_phase: summary
      model_field_state: present
      cache_eligible: true
      has_action_like_params: false
      stale_risk: false
    action:
      provider_body_rewrite: true
""",
                    "unsupported-action:provider_body_rewrite",
                ),
            ]

            for message, rule_text, reason in cases:
                raw, (forwarded, metadata) = run_with_rules(rule_text, message)
                self.assertEqual(forwarded, raw)
                self.assertEqual(metadata["routing"]["status"], "skipped")
                self.assertEqual(metadata["routing"]["reason"], reason)
                self.assertEqual(metadata["cache"]["reason"], reason)
                self.assertFalse(metadata["routing"]["applied"])

    def test_file_backed_summary_hint_canary_produces_applied_and_holdout_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = os.path.join(tmp, "codex_app_rules.yaml")
            db_path = os.path.join(tmp, "codex.sqlite3")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(
                    """
enabled: true
summary_model_hint:
  enabled: true
  target_model: gpt-5-mini
  canary:
    fraction: 0.5
    holdout_fraction: 0.5
    salt: deterministic-test
    unit: source_hash
"""
                )
            seen = {}
            try:
                with patch.dict(
                    os.environ,
                    {
                        "AGENTFLOW_CODEX_APP_RULES": rules_path,
                        "AGENTFLOW_DB": db_path,
                        "HOME": tmp,
                        "AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT": "0",
                    },
                    clear=False,
                ):
                    importlib.reload(codex_app_policy_module)
                    importlib.reload(codex_app_proxy)
                    for idx in range(100):
                        message = {
                            "jsonrpc": "2.0",
                            "id": f"turn-file-backed-canary-{idx}",
                            "method": "turn/start",
                            "params": {
                                "threadId": f"thread-file-backed-canary-{idx}",
                                "model": "gpt-5.3-codex",
                                "input": [{"type": "text", "text": f"Summarize the completed work for run {idx}."}],
                                "temperature": 0,
                            },
                        }
                        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                        status = metadata["routing"]["summary_model_hint"]["status"]
                        seen.setdefault(status, (json.loads(forwarded), metadata))
                        if {"applied", "holdout"}.issubset(seen):
                            break
            finally:
                importlib.reload(codex_app_policy_module)
                importlib.reload(codex_app_proxy)

        self.assertIn("applied", seen)
        self.assertIn("holdout", seen)
        applied_forwarded, applied_metadata = seen["applied"]
        holdout_forwarded, holdout_metadata = seen["holdout"]
        self.assertEqual(applied_forwarded["params"]["model"], "gpt-5-mini")
        self.assertEqual(holdout_forwarded["params"]["model"], "gpt-5.3-codex")
        self.assertEqual(applied_metadata["routing"]["canary_cohort"], "canary_applied")
        self.assertEqual(holdout_metadata["routing"]["canary_cohort"], "canary_holdout")
        self.assertEqual(holdout_metadata["routing"]["reason"], "summary-model-hint-canary-holdout")
        self.assertFalse(holdout_metadata["routing"]["canary_sample"]["raw_basis_included"])
        self.assertEqual(applied_metadata["routing"]["policy_source"], "local-manual")

    def test_summary_model_hint_canary_skips_uncertain_turns(self):
        cases = [
            (
                {
                    "jsonrpc": "2.0",
                    "id": "turn-tool-execution",
                    "method": "turn/start",
                    "params": {
                        "model": "gpt-5.3-codex",
                        "input": [{"type": "structured_output", "text": "pytest output"}],
                    },
                },
                "non-text-input",
            ),
            (
                {
                    "jsonrpc": "2.0",
                    "id": "turn-unknown-param",
                    "method": "turn/start",
                    "params": {
                        "model": "gpt-5.3-codex",
                        "input": [{"type": "text", "text": "Summarize the completed work."}],
                        "unknownFutureParam": True,
                    },
                },
                "unknown-param-shape",
            ),
            (
                {
                    "jsonrpc": "2.0",
                    "id": "turn-missing-model",
                    "method": "turn/start",
                    "params": {
                        "input": [{"type": "text", "text": "Summarize the completed work."}],
                    },
                },
                "codex-turn-start-model-field-absent",
            ),
        ]

        with (
            patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT", True),
            patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_TARGET", "gpt-5-codex"),
        ):
            for message, reason in cases:
                raw = json.dumps(message)
                forwarded, metadata = codex_app_proxy._optimize_client_message(raw)
                self.assertEqual(json.loads(forwarded), message)
                self.assertEqual(metadata["routing"]["status"], "skipped")
                self.assertFalse(metadata["routing"]["applied"])
                self.assertEqual(metadata["routing"]["reason"], reason)
                self.assertTrue(metadata["routing"]["canary_enabled"])
                self.assertEqual(metadata["routing"]["summary_model_hint"]["status"], "unsafe-skipped")
                self.assertEqual(metadata["routing"]["summary_model_hint"]["skip_reason"], reason)

        self.assertEqual(metadata["routing"]["canary"], "codex-app-summary-model-hint")

    def test_summary_model_hint_canary_metadata_is_redacted_and_counted(self):
        secret = "secret summary prompt must stay out of metadata"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-summary-counted",
            "method": "turn/start",
            "params": {
                "threadId": "thread-summary-counted",
                "model": "gpt-5.3-codex",
                "input": [{"type": "text", "text": f"Summarize this result: {secret}"}],
                "temperature": 0,
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-summary-counted",
            "result": {"message": "summary complete"},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            request_started: dict = {}
            try:
                with (
                    patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT", True),
                    patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_TARGET", "gpt-5-codex"),
                    patch.object(codex_app_proxy, "store", test_store),
                ):
                    forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                    start_event_id = codex_app_proxy._record_message(
                        forwarded,
                        direction="client_to_server",
                        session_id="session-summary-counted",
                        request_started=request_started,
                        optimization_metadata=metadata,
                    )
                    codex_app_proxy._record_message(
                        json.dumps(response),
                        direction="server_to_client",
                        session_id="session-summary-counted",
                        request_started=request_started,
                    )

                row = test_store.conn.execute(
                    "select routing_json, input_text_chars from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
                stats = asyncio.run(stats_views.stats_codex_effectiveness(test_store, limit=10))
            finally:
                test_store.conn.close()

        routing = json.loads(row["routing_json"])
        self.assertEqual(routing["status"], "applied")
        self.assertEqual(routing["routed_model"], "gpt-5-codex")
        self.assertGreater(row["input_text_chars"], 0)
        self.assertEqual(stats["summary"]["routing_applied"], 1)
        forbidden = {"prompt", "messages", "content", "raw_request", "raw_response", "params", "transcript", "input"}
        self.assertTrue(forbidden.isdisjoint(self._keys_in(routing)))
        self.assertNotIn(secret, json.dumps(routing))

    def test_safe_codex_turn_exact_cache_hit_when_enabled(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            message = {
                "jsonrpc": "2.0",
                "id": "turn-cache-1",
                "method": "turn/start",
                "params": {
                    "model": "gpt-5-codex",
                    "input": [{"type": "text", "text": "Please summarize the completed work for the run log."}],
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
                self.assertEqual(metadata["cache"]["reason"], "exact-miss")
                self.assertTrue(metadata["cache"]["eligible"])
                self.assertEqual(metadata["cache"]["workflow_phase"], "summary")
                self.assertEqual(metadata["cache"]["replayability_level"], "local-exact-response")
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

    def test_codex_exact_cache_canary_produces_miss_and_holdout_cohorts(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            seen = {}
            try:
                with (
                    patch.object(codex_app_proxy, "store", test_store),
                    patch.object(codex_app_proxy, "CODEX_APP_CACHE", True),
                    patch.object(codex_app_proxy, "CODEX_APP_CACHE_CANARY", {
                        "fraction": 0.5,
                        "holdout_fraction": 0.5,
                        "salt": "cache-canary-test",
                        "unit": "source_hash",
                    }),
                ):
                    for idx in range(100):
                        message = {
                            "jsonrpc": "2.0",
                            "id": f"turn-cache-canary-{idx}",
                            "method": "turn/start",
                            "params": {
                                "threadId": f"thread-cache-canary-{idx}",
                                "model": "gpt-5-codex",
                                "input": [{"type": "text", "text": f"Summarize the completed work for run {idx}."}],
                                "temperature": 0,
                            },
                        }
                        _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                        seen.setdefault(metadata["cache"]["outcome_bucket"], metadata["cache"])
                        if {"miss", "holdout"}.issubset(seen):
                            break
            finally:
                test_store.conn.close()

        self.assertIn("miss", seen)
        self.assertIn("holdout", seen)
        self.assertEqual(seen["miss"]["status"], "miss")
        self.assertEqual(seen["miss"]["canary_cohort"], "canary_applied")
        self.assertEqual(seen["holdout"]["status"], "holdout")
        self.assertEqual(seen["holdout"]["reason"], "codex-app-cache-canary-holdout")
        self.assertEqual(seen["holdout"]["canary_cohort"], "canary_holdout")
        self.assertFalse(seen["holdout"]["canary_sample"]["raw_basis_included"])
        self.assertNotIn("_agentflow_cache_key", seen["holdout"])

    def test_unsafe_codex_cached_response_is_deleted_and_not_replayed(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            message = {
                "jsonrpc": "2.0",
                "id": "turn-unsafe-cache-1",
                "method": "turn/start",
                "params": {
                    "model": "gpt-5-codex",
                    "input": [{"type": "text", "text": "Summarize the completed work."}],
                    "temperature": 0,
                },
            }
            cache_key = codex_app_proxy._codex_cache_key_for_message(message)
            test_store.set_cache(
                cache_key,
                "codex-app",
                100,
                {
                    "agentflow_cache_type": "codex-app-jsonrpc-response",
                    "version": 1,
                    "response": {
                        "jsonrpc": "2.0",
                        "id": "turn-unsafe-cache-1",
                        "result": {"tool_call": {"name": "shell"}},
                    },
                },
            )
            try:
                with patch.object(codex_app_proxy, "store", test_store), patch.object(codex_app_proxy, "CODEX_APP_CACHE", True):
                    _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                    remaining = test_store.conn.execute(
                        "select 1 from cache where cache_key = ?",
                        (cache_key,),
                    ).fetchone()
            finally:
                test_store.conn.close()

        self.assertEqual(metadata["cache"]["status"], "unsafe-skip")
        self.assertEqual(metadata["cache"]["reason"], "unsafe-cached-envelope")
        self.assertEqual(metadata["cache"]["outcome_bucket"], "unsafe-skip")
        self.assertIsNone(remaining)
        self.assertNotIn("_agentflow_replay_frame", metadata["cache"])

    def test_codex_cache_stale_risk_skips_path_without_dependency_evidence(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp, tempfile.TemporaryDirectory() as root:
            test_store = Store(tmp.name)
            message = {
                "jsonrpc": "2.0",
                "id": "turn-stale-risk",
                "method": "turn/start",
                "params": {
                    "model": "gpt-5-codex",
                    "input": [{"type": "text", "text": "Summarize ./missing-file.txt for the run log."}],
                    "temperature": 0,
                },
            }
            try:
                with (
                    patch.object(codex_app_proxy, "store", test_store),
                    patch.object(codex_app_proxy, "CODEX_APP_CACHE", True),
                    patch("agentflow_proxy.cache.CACHE_FILE_WATCH_ROOT", root),
                ):
                    _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
            finally:
                test_store.conn.close()

        self.assertEqual(metadata["cache"]["status"], "skipped")
        self.assertEqual(metadata["cache"]["reason"], "dependency-missing")
        self.assertEqual(metadata["cache"]["outcome_bucket"], "stale-risk")
        self.assertTrue(metadata["cache"]["eligible"])
        self.assertFalse(metadata["cache"]["file_dependency_audit"]["paths_included"])
        self.assertNotIn("_agentflow_cache_key", metadata["cache"])

    def test_codex_cache_disabled_records_safe_summary_eligibility_metadata(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-cache-disabled",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Summarize the last turn in one sentence."}],
                "temperature": 0,
            },
        }

        with patch.object(codex_app_proxy, "CODEX_APP_CACHE", False):
            _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))

        self.assertEqual(metadata["cache"]["status"], "skipped")
        self.assertEqual(metadata["cache"]["reason"], "codex-app-cache-disabled")
        self.assertFalse(metadata["cache"]["enabled"])
        self.assertFalse(metadata["cache"]["exact_enabled"])
        self.assertTrue(metadata["cache"]["eligible"])
        self.assertEqual(metadata["cache"]["workflow_phase"], "summary")
        self.assertEqual(metadata["cache"]["replayability_level"], "local-exact-response")

    def test_codex_cache_skips_non_summary_and_model_unknown_turns(self):
        not_summary = {
            "jsonrpc": "2.0",
            "id": "turn-not-summary",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Answer the question in detail."}],
                "temperature": 0,
            },
        }
        no_model = {
            "jsonrpc": "2.0",
            "id": "turn-no-model",
            "method": "turn/start",
            "params": {
                "input": [{"type": "text", "text": "Summarize the result."}],
                "temperature": 0,
            },
        }

        with patch.object(codex_app_proxy, "CODEX_APP_CACHE", True):
            _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(not_summary))
            self.assertEqual(metadata["cache"]["status"], "skipped")
            self.assertEqual(metadata["cache"]["reason"], "workflow-phase-not-summary")
            self.assertFalse(metadata["cache"]["eligible"])

            _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(no_model))
            self.assertEqual(metadata["cache"]["status"], "skipped")
            self.assertEqual(metadata["cache"]["reason"], "model-field-unknown")
            self.assertFalse(metadata["cache"]["eligible"])

    def test_codex_cache_skips_terminal_and_action_like_turns(self):
        terminal_text = {
            "jsonrpc": "2.0",
            "id": "turn-terminal",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Summarize the result after pytest finishes."}],
                "temperature": 0,
            },
        }
        action_like = {
            "jsonrpc": "2.0",
            "id": "turn-action",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Summarize the command output."}],
                "command": "python -m unittest",
            },
        }

        with patch.object(codex_app_proxy, "CODEX_APP_CACHE", True):
            _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(terminal_text))
            self.assertEqual(metadata["cache"]["status"], "skipped")
            self.assertEqual(metadata["cache"]["reason"], "terminal-interaction-text")
            self.assertFalse(metadata["cache"]["eligible"])

            _forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(action_like))
            self.assertEqual(metadata["cache"]["status"], "skipped")
            self.assertEqual(metadata["cache"]["reason"], "action-like-params")
            self.assertFalse(metadata["cache"]["eligible"])

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
                    "model": "gpt-5-codex",
                    "input": [{"type": "text", "text": f"Summarize {path}"}],
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
            self.assertEqual(metadata["cache"]["reason"], "dependency-changed")
            self.assertEqual(metadata["cache"]["file_dependency_audit"]["invalidation_reason"], None)
            self.assertFalse(metadata["cache"]["file_dependency_audit"]["paths_included"])
            self.assertNotIn(path, json.dumps(metadata["cache"]))
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

    def test_codex_policy_decision_records_predicted_routing_as_observed_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        secret = "raw codex policy decision prompt"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-policy-decision",
            "method": "turn/start",
            "params": {
                "threadId": "thread-policy-decision",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": secret}],
            },
        }

        class PolicyDecisionCodexClient(ManagedCodexClient):
            async def post(self, url, json, headers=None):
                self.__class__.calls.append({"method": "post", "url": url, "json": json, "headers": dict(headers or {})})
                return ManagedResponse(body={
                    "schema": "agentflow.policy_decision.v1",
                    "optimization_unit_id": 88,
                    "policy_id": "codex-policy-decision",
                    "confidence": 0.9,
                    "provider_forwarding": False,
                    "server_content_processing": False,
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                    "routing": {
                        "status": "recommended",
                        "target_model": "gpt-5-mini",
                        "confidence": 0.9,
                        "route_down_probability": 0.92,
                        "recommended_mode": "shadow",
                        "model_artifact_version": "routing-predictor-v1-codex",
                        "model_evidence_hash": "sha256:evidence",
                        "predictor_rule_id": "routing-evidence:codex:gpt5->mini",
                        "reason_codes": ["active-routing-predictor-model"],
                    },
                })

        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", PolicyDecisionCodexClient):
            pending = asyncio.run(codex_app_proxy._attach_codex_managed_recommendation(forwarded, metadata))

        managed = metadata["routing"]["managed_recommendation"]
        self.assertEqual(PolicyDecisionCodexClient.calls[0]["url"], "http://managed.test/v1/policy-decision")
        unit = PolicyDecisionCodexClient.calls[0]["json"]
        self.assertEqual(unit["schema"], "agentflow.policy_decision_preflight.v1")
        self.assertEqual(unit["source_surface"], "codex_turn")
        self.assertNotIn(secret, json.dumps(unit))
        self.assertNotIn("thread-policy-decision", json.dumps(unit))
        self.assertEqual(managed["status"], "received")
        self.assertEqual(managed["recommended_mode"], "shadow")
        self.assertEqual(managed["local_action_taken"], "shadow")
        self.assertEqual(managed["would_route_model"], "gpt-5-mini")
        self.assertFalse(managed["applied"])
        self.assertEqual(pending["managed"], managed)

    def test_codex_policy_decision_canary_routes_selected_safe_model(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["AGENTFLOW_POLICY_DECISION_CANARY_FRACTION"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        secret = "raw codex canary prompt"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-policy-canary",
            "method": "turn/start",
            "params": {
                "threadId": "thread-policy-canary",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": secret}],
            },
        }

        class PolicyDecisionCanaryClient(ManagedCodexClient):
            async def post(self, url, json, headers=None):
                self.__class__.calls.append({"method": "post", "url": url, "json": json, "headers": dict(headers or {})})
                return ManagedResponse(body={
                    "schema": "agentflow.policy_decision.v1",
                    "optimization_unit_id": 89,
                    "policy_id": "codex-policy-canary",
                    "confidence": 0.91,
                    "provider_forwarding": False,
                    "server_content_processing": False,
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                    "routing": {
                        "status": "recommended",
                        "target_model": "gpt-5-mini",
                        "confidence": 0.91,
                        "route_down_probability": 0.9,
                        "recommended_mode": "canary",
                        "model_artifact_version": "routing-predictor-v1-codex",
                        "model_evidence_hash": "sha256:evidence",
                        "predictor_rule_id": "routing-evidence:codex:gpt5->mini",
                        "reason_codes": ["active-routing-predictor-model"],
                    },
                })

        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", PolicyDecisionCanaryClient):
            pending = asyncio.run(codex_app_proxy._attach_codex_managed_recommendation(forwarded, metadata))

        forwarded_obj = json.loads(pending["forwarded"])
        managed = metadata["routing"]["managed_recommendation"]
        self.assertEqual(forwarded_obj["params"]["model"], "gpt-5-mini")
        self.assertTrue(managed["applied"])
        self.assertTrue(managed["changed_model"])
        self.assertTrue(managed["codex_app_frame_mutated"])
        self.assertEqual(managed["local_action_taken"], "canary_applied")
        self.assertEqual(managed["local_canary"]["cohort"], "canary_applied")
        self.assertEqual(metadata["routing"]["final_policy_source"], "managed-recommended")
        self.assertEqual(metadata["routing"]["canary"], "codex-app-managed-policy-decision")
        self.assertNotIn(secret, json.dumps(PolicyDecisionCanaryClient.calls[0]["json"]))

    def test_codex_policy_decision_canary_holdout_and_low_confidence_keep_requested_model(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_POLICY_DECISION_ENABLED"] = "1"
        os.environ["AGENTFLOW_POLICY_DECISION_CANARY_FRACTION"] = "0"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-policy-holdout",
            "method": "turn/start",
            "params": {
                "threadId": "thread-policy-holdout",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "metadata only"}],
            },
        }

        class PolicyDecisionHoldoutClient(ManagedCodexClient):
            async def post(self, url, json, headers=None):
                self.__class__.calls.append({"method": "post", "url": url, "json": json, "headers": dict(headers or {})})
                return ManagedResponse(body={
                    "schema": "agentflow.policy_decision.v1",
                    "policy_id": "codex-policy-holdout",
                    "confidence": 0.91,
                    "provider_forwarding": False,
                    "server_content_processing": False,
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                    "routing": {
                        "status": "recommended",
                        "target_model": "gpt-5-mini",
                        "confidence": 0.91,
                        "route_down_probability": 0.9,
                        "recommended_mode": "canary",
                        "model_artifact_version": "routing-predictor-v1-codex",
                    },
                })

        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", PolicyDecisionHoldoutClient):
            pending = asyncio.run(codex_app_proxy._attach_codex_managed_recommendation(forwarded, metadata))

        forwarded_obj = json.loads(pending["forwarded"])
        managed = metadata["routing"]["managed_recommendation"]
        self.assertEqual(forwarded_obj["params"]["model"], "gpt-5-codex")
        self.assertFalse(managed["applied"])
        self.assertFalse(managed["codex_app_frame_mutated"])
        self.assertEqual(managed["apply_reason"], "local-canary-holdout")
        self.assertEqual(managed["local_action_taken"], "canary_holdout")

        os.environ["AGENTFLOW_POLICY_DECISION_CANARY_FRACTION"] = "1"

        class PolicyDecisionLowConfidenceClient(ManagedCodexClient):
            async def post(self, url, json, headers=None):
                self.__class__.calls.append({"method": "post", "url": url, "json": json, "headers": dict(headers or {})})
                return ManagedResponse(body={
                    "schema": "agentflow.policy_decision.v1",
                    "policy_id": "codex-policy-low-confidence",
                    "confidence": 0.4,
                    "provider_forwarding": False,
                    "server_content_processing": False,
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                    "routing": {
                        "status": "recommended",
                        "target_model": "gpt-5-mini",
                        "confidence": 0.4,
                        "route_down_probability": 0.4,
                        "recommended_mode": "canary",
                        "model_artifact_version": "routing-predictor-v1-codex",
                    },
                })

        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", PolicyDecisionLowConfidenceClient):
            pending = asyncio.run(codex_app_proxy._attach_codex_managed_recommendation(forwarded, metadata))

        forwarded_obj = json.loads(pending["forwarded"])
        managed = metadata["routing"]["managed_recommendation"]
        self.assertEqual(forwarded_obj["params"]["model"], "gpt-5-codex")
        self.assertFalse(managed["applied"])
        self.assertEqual(managed["apply_reason"], "routing-predictor-confidence-too-low")

    def test_codex_managed_feedback_posts_unit_and_patches_sanitized_outcome(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        secret = "raw codex prompt secret"
        log_secret = "2026-06-09T20:00:00Z ERROR pid=1234 codex-secret failed"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-managed",
            "method": "turn/start",
            "params": {
                "threadId": "thread-managed",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": f"{secret}\n{log_secret}"}],
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
                        queue = test_store.conn.execute(
                            "select status, attempts, payload_json from managed_outcome_feedback_queue"
                        ).fetchone()
                        return json.loads(row["routing_json"]), dict(queue)
                finally:
                    test_store.conn.close()

        routing, queue = asyncio.run(run_fixture())

        self.assertEqual([call["method"] for call in ManagedCodexClient.calls], ["post", "patch"])
        self.assertEqual(ManagedCodexClient.calls[0]["url"], "http://managed.test/v1/recommendation")
        self.assertEqual(ManagedCodexClient.calls[1]["url"], "http://managed.test/v1/optimization-units/77/outcome")
        unit = ManagedCodexClient.calls[0]["json"]
        outcome = ManagedCodexClient.calls[1]["json"]
        self.assertEqual(unit["source_surface"], "codex_turn")
        self.assertEqual(unit["granularity"], "agent_turn")
        self.assertEqual(unit["feature_schema_version"], recommendations.FEATURE_SCHEMA_VERSION)
        self.assertEqual(unit["candidate_target_model"], "gpt-5-codex")
        self.assertEqual(unit["input_features"]["terminal_log_features"]["schema"], "agentflow.terminal_log_features.v1")
        self.assertEqual(unit["input_features"]["terminal_log_features"]["error_line_count_bucket"], "one")
        self.assertEqual(outcome["terminal_log_features"]["error_line_count_bucket"], "one")
        self.assertEqual(
            sorted(unit["grouping_identifiers"]),
            ["request_id_hash", "thread_id_hash"],
        )
        self.assertTrue(unit["grouping_identifiers"]["request_id_hash"].startswith("sha256:"))
        self.assertTrue(unit["grouping_identifiers"]["thread_id_hash"].startswith("sha256:"))
        self.assertTrue(unit["privacy_summary"]["metadata_only"])
        self.assertFalse(unit["privacy_summary"]["raw_body_storage"])
        self.assertEqual(outcome["source_surface"], "codex_turn")
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["quality_signals"]["status"], "success")
        self.assertIn("success", outcome["quality_signals"]["signal_codes"])
        self.assertEqual(outcome["managed_recommendation"]["optimization_unit_id"], 77)
        forbidden = {"prompt", "messages", "content", "raw_request", "raw_response", "params", "transcript"}
        self.assertTrue(forbidden.isdisjoint(self._keys_in(unit)))
        self.assertTrue(forbidden.isdisjoint(self._keys_in(outcome)))
        self.assertNotIn(secret, json.dumps(unit))
        self.assertNotIn(log_secret, json.dumps(unit))
        self.assertNotIn("turn-managed", json.dumps(unit))
        self.assertNotIn("thread-managed", json.dumps(unit))
        self.assertNotIn(secret, json.dumps(outcome))
        self.assertNotIn(log_secret, json.dumps(outcome))
        self.assertNotIn("must not leave", json.dumps(unit))
        self.assertNotIn("must be stripped", json.dumps(outcome))
        managed = routing["managed_recommendation"]
        self.assertEqual(managed["status"], "received")
        self.assertFalse(managed["applied"])
        self.assertEqual(managed["apply_reason"], "codex-app-managed-recommendation-observed-only")
        self.assertEqual(managed["outcome_feedback"]["status"], "sent")
        self.assertEqual(queue["status"], "sent")
        self.assertEqual(queue["attempts"], 1)
        self.assertTrue(forbidden.isdisjoint(self._keys_in(json.loads(queue["payload_json"]))))
        self.assertNotIn(secret, queue["payload_json"])

    def test_codex_managed_feedback_failure_queues_sanitized_retryable_outcome_and_stats(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        ManagedCodexClient.feedback_error = RuntimeError("managed feedback down")
        secret = "raw codex queued secret"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-managed-fail",
            "method": "turn/start",
            "params": {
                "threadId": "thread-managed-fail",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": secret}],
                "raw_request": "must be stripped",
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-managed-fail",
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
                                session_id="session-managed-fail",
                                request_started=request_started,
                                optimization_metadata=metadata,
                            )
                            pending["start_event_id"] = start_event_id
                            pending_managed["turn-managed-fail"] = pending
                            await codex_app_proxy._record_codex_managed_outcome(
                                json.dumps(response),
                                session_id="session-managed-fail",
                                request_started=request_started,
                                pending_managed=pending_managed,
                            )
                    routing_row = test_store.conn.execute(
                        "select routing_json from codex_app_events where id = ?",
                        (start_event_id,),
                    ).fetchone()
                    queue_row = test_store.conn.execute(
                        "select status, attempts, payload_json, last_error from managed_outcome_feedback_queue"
                    ).fetchone()
                    stats = await stats_views.stats_codex_effectiveness(test_store, limit=10)
                    return json.loads(routing_row["routing_json"]), dict(queue_row), stats
                finally:
                    test_store.conn.close()

        routing, queue, stats = asyncio.run(run_fixture())

        self.assertEqual([call["method"] for call in ManagedCodexClient.calls], ["post", "patch"])
        managed = routing["managed_recommendation"]
        self.assertEqual(managed["outcome_feedback"]["status"], "retryable-error")
        self.assertEqual(managed["outcome_feedback"]["reason"], "request-failed")
        self.assertEqual(queue["status"], "retryable-error")
        self.assertEqual(queue["attempts"], 1)
        self.assertIn("managed feedback down", queue["last_error"])
        forbidden = {"prompt", "messages", "content", "raw_request", "raw_response", "params", "transcript"}
        payload = json.loads(queue["payload_json"])
        self.assertTrue(forbidden.isdisjoint(self._keys_in(payload)))
        self.assertNotIn(secret, queue["payload_json"])
        self.assertEqual(stats["summary"]["managed_feedback_error"], 1)
        self.assertEqual(stats["summary"]["managed_feedback_queue_error"], 1)
        queue_breakdown = {row["value"]: row["count"] for row in stats["managed_feedback_queue_breakdown"]}
        self.assertEqual(queue_breakdown["retryable-error"], 1)

    def test_codex_outcome_feedback_disabled_does_not_enqueue(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                meta = asyncio.run(
                    recommendations.queue_codex_outcome_feedback(
                        test_store,
                        {"optimization_unit_id": 77},
                        {"status": "success", "raw_response": "must be stripped"},
                    )
                )
                rows = test_store.conn.execute("select count(*) as c from managed_outcome_feedback_queue").fetchone()
            finally:
                test_store.conn.close()

        self.assertEqual(meta["status"], "disabled")
        self.assertEqual(rows["c"], 0)

    def test_codex_outcome_feedback_queue_drops_after_attempt_limit(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        os.environ["AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS"] = "1"
        ManagedCodexClient.feedback_error = RuntimeError("managed feedback still down")

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedCodexClient):
                    meta = asyncio.run(
                        recommendations.queue_codex_outcome_feedback(
                            test_store,
                            {"optimization_unit_id": 77},
                            {"status": "success", "quality_signals": {"status": "success"}},
                        )
                    )
                row = test_store.conn.execute(
                    "select status, attempts, payload_json from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                test_store.conn.close()

        self.assertEqual(meta["status"], "dropped-after-limit")
        self.assertEqual(meta["reason"], "attempt-limit-reached")
        self.assertEqual(row["status"], "dropped-after-limit")
        self.assertEqual(row["attempts"], 1)
        self.assertNotIn("must be stripped", row["payload_json"])

    def test_codex_canary_lifecycle_feedback_queues_summary_and_exact_cache_metadata(self):
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        secret = "raw codex lifecycle secret"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-lifecycle-enabled",
            "method": "turn/start",
            "params": {
                "threadId": "thread-lifecycle-enabled",
                "model": "gpt-5.3-codex",
                "input": [{"type": "text", "text": f"Summarize the completed work. {secret}"}],
                "temperature": 0,
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-lifecycle-enabled",
            "result": {"message": "summary complete"},
        }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT", True),
                        patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_TARGET", "gpt-5-codex"),
                        patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_CANARY", {
                            "fraction": 1.0,
                            "holdout_fraction": 0.0,
                            "salt": "summary-lifecycle-test",
                            "unit": "source_hash",
                        }),
                        patch.object(codex_app_proxy, "CODEX_APP_CACHE", True),
                        patch.object(codex_app_proxy, "CODEX_APP_CACHE_CANARY", {
                            "fraction": 1.0,
                            "holdout_fraction": 0.0,
                            "salt": "cache-lifecycle-test",
                            "unit": "source_hash",
                        }),
                    ):
                        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                        request_started = {}
                        start_event_id = codex_app_proxy._record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id="session-lifecycle-enabled",
                            request_started=request_started,
                            optimization_metadata=metadata,
                        )
                        pending = codex_app_proxy._attach_codex_canary_lifecycle_pending(forwarded, metadata)
                        pending["start_event_id"] = start_event_id
                        pending_lifecycle = {pending["request_id"]: pending}
                        await codex_app_proxy._record_codex_canary_lifecycle_outcome(
                            json.dumps(response),
                            request_started=request_started,
                            pending_lifecycle=pending_lifecycle,
                        )
                        rows = test_store.conn.execute(
                            "select source_surface, endpoint, status, payload_json from managed_outcome_feedback_queue order by created_at"
                        ).fetchall()
                        routing_row = test_store.conn.execute(
                            "select routing_json from codex_app_events where id = ?",
                            (start_event_id,),
                        ).fetchone()
                        return [dict(row) for row in rows], json.loads(routing_row["routing_json"])
                finally:
                    test_store.conn.close()

        rows, routing = asyncio.run(run_fixture())

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_surface"] for row in rows}, {"codex_app_canary_lifecycle"})
        self.assertEqual({row["endpoint"] for row in rows}, {"/v1/policy-events"})
        self.assertEqual({row["status"] for row in rows}, {"queued"})
        payloads = [json.loads(row["payload_json"]) for row in rows]
        self.assertEqual(
            {payload["action_family"] for payload in payloads},
            {"routing", "cache"},
        )
        for payload in payloads:
            self.assertEqual(payload["schema"], "agentflow.codex_app_canary_lifecycle_feedback.v1")
            self.assertTrue(payload["privacy"]["metadata_only"])
            self.assertFalse(payload["privacy"]["raw_params_included"])
            self.assertFalse(payload["privacy"]["request_ids_included"])
            self.assertFalse(payload["privacy"]["cache_keys_included"])
            self.assertNotIn(secret, json.dumps(payload))
            self.assertNotIn("turn-lifecycle-enabled", json.dumps(payload))
            self.assertNotIn("thread-lifecycle-enabled", json.dumps(payload))
            forbidden = {"prompt", "messages", "content", "params", "request_id", "thread_id", "cache_key"}
            self.assertTrue(forbidden.isdisjoint(self._keys_in(payload)))
        feedback_meta = routing["managed_lifecycle_feedback"]
        self.assertFalse(feedback_meta["payload_included"])
        self.assertEqual(feedback_meta["results"]["routing"]["status"], "queued")
        self.assertEqual(feedback_meta["results"]["cache"]["status"], "queued")

    def test_codex_canary_lifecycle_feedback_disabled_records_skip_without_queue(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-lifecycle-disabled",
            "method": "turn/start",
            "params": {
                "model": "gpt-5.3-codex",
                "input": [{"type": "text", "text": "Summarize the completed work."}],
                "temperature": 0,
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-lifecycle-disabled",
            "result": {"message": "summary complete"},
        }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT", True),
                        patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_TARGET", "gpt-5-codex"),
                        patch.object(codex_app_proxy, "CODEX_APP_SUMMARY_MODEL_HINT_CANARY", {
                            "fraction": 1.0,
                            "holdout_fraction": 0.0,
                            "salt": "summary-lifecycle-disabled",
                            "unit": "source_hash",
                        }),
                        patch.object(codex_app_proxy, "CODEX_APP_CACHE", True),
                        patch.object(codex_app_proxy, "CODEX_APP_CACHE_CANARY", {
                            "fraction": 1.0,
                            "holdout_fraction": 0.0,
                            "salt": "cache-lifecycle-disabled",
                            "unit": "source_hash",
                        }),
                    ):
                        forwarded, metadata = codex_app_proxy._optimize_client_message(json.dumps(message))
                        request_started = {}
                        start_event_id = codex_app_proxy._record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id="session-lifecycle-disabled",
                            request_started=request_started,
                            optimization_metadata=metadata,
                        )
                        pending = codex_app_proxy._attach_codex_canary_lifecycle_pending(forwarded, metadata)
                        pending["start_event_id"] = start_event_id
                        await codex_app_proxy._record_codex_canary_lifecycle_outcome(
                            json.dumps(response),
                            request_started=request_started,
                            pending_lifecycle={pending["request_id"]: pending},
                        )
                        count = test_store.conn.execute(
                            "select count(*) as c from managed_outcome_feedback_queue"
                        ).fetchone()["c"]
                        routing_row = test_store.conn.execute(
                            "select routing_json from codex_app_events where id = ?",
                            (start_event_id,),
                        ).fetchone()
                        return count, json.loads(routing_row["routing_json"])
                finally:
                    test_store.conn.close()

        count, routing = asyncio.run(run_fixture())

        self.assertEqual(count, 0)
        feedback_meta = routing["managed_lifecycle_feedback"]
        self.assertEqual(feedback_meta["results"]["routing"]["status"], "disabled")
        self.assertEqual(feedback_meta["results"]["routing"]["reason"], "disabled")
        self.assertEqual(feedback_meta["results"]["cache"]["status"], "disabled")
        self.assertEqual(feedback_meta["results"]["cache"]["reason"], "disabled")
        self.assertFalse(feedback_meta["payload_included"])

    def test_codex_routing_experiment_samples_turn_and_queues_metadata_only_outcome(self):
        secret_prompt = "secret codex prompt must not be queued"
        secret_result = "secret codex result must not be queued"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment",
            "method": "turn/start",
            "params": {
                "threadId": "thread-routing-experiment",
                "model": "gpt-5-codex",
                "approvalPolicy": "never",
                "clientUserMessageId": "private-client-message-id",
                "cwd": "/private/workspace/path",
                "effort": "medium",
                "runtimeWorkspaceRoots": ["/private/workspace/path"],
                "sandboxPolicy": "danger-full-access",
                "input": [{"type": "text", "text": f"Summarize the following self-contained note: {secret_prompt}"}],
                "temperature": 0,
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment",
            "result": {"message": secret_result},
        }
        optimization_metadata = {
            "routing": {
                "status": "skipped",
                "reason": "keep requested Codex model",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "model_field": "model",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        def sampled_decision(body, routing_meta, *, stream, provider, source_surface, store_obj):
            self.assertFalse(stream)
            self.assertEqual(provider, "openai")
            self.assertEqual(source_surface, "codex_turn")
            self.assertIsNotNone(store_obj)
            self.assertEqual(body["model"], "gpt-5-codex")
            self.assertEqual(routing_meta["requested_model"], "gpt-5-codex")
            self.assertEqual(routing_meta["routed_model"], "gpt-5-codex")
            return {
                "schema": "agentflow.routing_experiment_decision.v1",
                "enabled": True,
                "mode": "shadow_candidate_pass_through",
                "status": "selected",
                "sampled": True,
                "reason": "sampled-shadow-candidate-pass-through",
                "counterfactual": True,
                "shadow_only": True,
                "provider": "openai",
                "source_surface": "codex_turn",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-mini",
                "primary_model": "gpt-5-codex",
                "shadow_model": "gpt-5-mini",
                "user_visible_model": "gpt-5-codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "text_chars": len(secret_prompt),
                "daily_budget_usd": 10.0,
                "budget_spent_usd": 0.0,
                "budget_remaining_usd": 10.0,
                "similarity_threshold": 0.86,
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_responses_included": False,
                    "request_ids_included": False,
                    "session_ids_included": False,
                    "file_paths_included": False,
                },
            }

        async def fake_shadow_executor(shadow_body, *, shadow_model):
            self.assertEqual(shadow_model, "gpt-5-mini")
            self.assertEqual(shadow_body["model"], "gpt-5-mini")
            self.assertFalse(shadow_body["store"])
            self.assertNotIn("threadId", shadow_body)
            self.assertNotIn("approvalPolicy", shadow_body)
            self.assertNotIn("clientUserMessageId", shadow_body)
            self.assertNotIn("cwd", shadow_body)
            self.assertNotIn("effort", shadow_body)
            self.assertNotIn("runtimeWorkspaceRoots", shadow_body)
            self.assertNotIn("sandboxPolicy", shadow_body)
            self.assertNotIn("thread-routing-experiment", json.dumps(shadow_body))
            self.assertNotIn("/private/workspace/path", json.dumps(shadow_body))
            self.assertNotIn("private-client-message-id", json.dumps(shadow_body))
            self.assertIn(secret_prompt, shadow_body["input"])
            return {
                "status_code": 200,
                "response_body": {
                    "output_text": f"Summary: {secret_result}",
                    "usage": {"input_tokens": 11, "output_tokens": 5},
                },
                "latency_ms": 37,
                "cost_est_usd": 0.00012,
                "error": None,
            }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    request_started = {}
                    pending = {}
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "routing_experiment_decision", side_effect=sampled_decision),
                        patch.object(codex_app_proxy, "_codex_shadow_api_key", return_value="test-key"),
                        patch.object(codex_app_proxy, "_execute_codex_stateless_shadow_request", side_effect=fake_shadow_executor),
                    ):
                        start_event_id = codex_app_proxy._record_message(
                            json.dumps(message),
                            direction="client_to_server",
                            session_id="session-routing-experiment",
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        codex_app_proxy._attach_codex_routing_experiment_pending(
                            json.dumps(message),
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending,
                        )
                        self.assertIn("turn-routing-experiment", pending)
                        response_frame = json.dumps(response)
                        await codex_app_proxy._maybe_run_codex_routing_experiment(
                            response_frame,
                            request_started=request_started,
                            pending_routing_experiments=pending,
                        )

                    experiment = test_store.conn.execute(
                        "select * from routing_experiments"
                    ).fetchone()
                    queue_row = test_store.conn.execute(
                        "select source_surface, endpoint, status, payload_json from managed_outcome_feedback_queue"
                    ).fetchone()
                    routing_row = test_store.conn.execute(
                        "select routing_json from codex_app_events where id = ?",
                        (start_event_id,),
                    ).fetchone()
                    from agentflow_proxy.routing_experiments import build_routing_experiment_report

                    report = build_routing_experiment_report(test_store, limit=5)
                    return dict(experiment), dict(queue_row), json.loads(routing_row["routing_json"]), report
                finally:
                    test_store.conn.close()

        experiment, queue_row, routing, report = asyncio.run(run_fixture())

        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["requested_model"], "gpt-5-codex")
        self.assertEqual(experiment["routed_model"], "gpt-5-mini")
        self.assertEqual(experiment["primary_model"], "gpt-5-codex")
        self.assertEqual(experiment["shadow_model"], "gpt-5-mini")
        self.assertEqual(experiment["primary_status_code"], 200)
        self.assertEqual(experiment["shadow_status_code"], 200)
        self.assertIsNone(experiment["error"])
        self.assertEqual(experiment["shadow_latency_ms"], 37)
        self.assertEqual(experiment["shadow_cost_est_usd"], 0.00012)
        self.assertGreater(experiment["primary_output_chars"], 0)
        self.assertGreater(experiment["shadow_output_chars"], 0)
        self.assertIsNone(experiment["primary_response_json"])
        self.assertIsNone(experiment["shadow_response_json"])
        experiment_json = json.loads(experiment["experiment_json"])
        self.assertEqual(experiment_json["mode"], "shadow_candidate_pass_through")
        self.assertTrue(experiment_json["counterfactual"])
        self.assertTrue(experiment_json["shadow_only"])
        self.assertIsNone(experiment_json["shadow_limitation"])
        self.assertEqual(experiment_json["shadow_request_preflight"]["status"], "executable")
        self.assertEqual(experiment_json["shadow_request_preflight"]["reason"], "stateless-summary-text-only-replay")
        self.assertEqual(
            experiment_json["shadow_request_preflight"]["workflow_phase_classification"]["workflow_phase"],
            "summary",
        )
        self.assertEqual(experiment_json["request_size_gate"]["status"], "within-bounds")
        self.assertEqual(experiment_json["request_size_gate"]["max_text_chars_scope"], "global")
        self.assertIn("cwd", experiment_json["shadow_request_preflight"]["omitted_turn_fields"])
        self.assertIn("runtimeWorkspaceRoots", experiment_json["shadow_request_preflight"]["omitted_turn_fields"])
        self.assertFalse(experiment_json["shadow_request_preflight"]["privacy"]["raw_prompts_included"])
        self.assertEqual(experiment_json["status"], "compared")
        self.assertNotIn("shadow-exception", experiment_json["reason_codes"])
        self.assertEqual(experiment_json["managed_feedback"]["status"], "queued")
        self.assertFalse(experiment_json["managed_feedback"]["payload_included"])

        self.assertEqual(queue_row["source_surface"], "routing_experiment_outcome")
        self.assertEqual(queue_row["endpoint"], "/v1/policy-events")
        self.assertEqual(queue_row["status"], "queued")
        payload = json.loads(queue_row["payload_json"])
        self.assertEqual(payload["schema"], "agentflow.routing_experiment_outcome_event.v1")
        self.assertEqual(payload["source_surface"], "codex_turn")
        self.assertEqual(payload["app_family"], "codex")
        self.assertTrue(payload["candidate"]["counterfactual"])
        self.assertTrue(payload["candidate"]["shadow_only"])
        self.assertEqual(payload["candidate"]["shadow_model_family"], "gpt-5")
        self.assertEqual(payload["outcome"]["status"], "compared")
        self.assertTrue(payload["outcome"]["compared"])
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["raw_responses_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertNotIn(secret_prompt, json.dumps(payload))
        self.assertNotIn(secret_result, json.dumps(payload))
        self.assertNotIn("turn-routing-experiment", json.dumps(payload))
        self.assertNotIn("thread-routing-experiment", json.dumps(payload))
        self.assertNotIn(secret_prompt, json.dumps(experiment_json))
        self.assertNotIn(secret_result, json.dumps(experiment_json))
        self.assertNotIn("/private/workspace/path", json.dumps(experiment_json))
        self.assertNotIn("private-client-message-id", json.dumps(experiment_json))

        self.assertEqual(routing["routing_experiment"]["source_surface"], "codex_turn")
        self.assertEqual(routing["routing_experiment"]["managed_feedback"]["status"], "queued")
        [candidate] = report["candidates"]
        self.assertEqual(candidate["source_surface"], "codex_turn")
        self.assertEqual(candidate["requested_model"], "gpt-5-codex")
        self.assertEqual(candidate["routed_model"], "gpt-5-mini")
        self.assertEqual(candidate["compared_samples"], 1)
        self.assertEqual(candidate["shadow_error_samples"], 0)
        self.assertEqual(report["summary"]["sample_mode_counts"], {"shadow_candidate_pass_through": 1})

    def test_codex_routing_experiment_executes_stateless_non_summary_shadow(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-stateless-text",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Explain the tradeoff between local proxy routing and exact caching."}],
                "temperature": 0,
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-stateless-text",
            "result": {"message": "Routing changes model cost; exact caching avoids repeated calls."},
        }
        optimization_metadata = {
            "routing": {
                "status": "skipped",
                "reason": "keep requested Codex model",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "unknown",
                "model_field": "model",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        def sampled_decision(body, routing_meta, *, stream, provider, source_surface, store_obj):
            return {
                "schema": "agentflow.routing_experiment_decision.v1",
                "enabled": True,
                "mode": "shadow_candidate_pass_through",
                "status": "selected",
                "sampled": True,
                "reason": "sampled-shadow-candidate-pass-through",
                "counterfactual": True,
                "shadow_only": True,
                "provider": provider,
                "source_surface": source_surface,
                "requested_model": routing_meta["requested_model"],
                "routed_model": "gpt-5-mini",
                "primary_model": routing_meta["requested_model"],
                "shadow_model": "gpt-5-mini",
                "user_visible_model": routing_meta["requested_model"],
                "category": "codex-turn",
                "workflow_phase": routing_meta["workflow_phase"],
                "text_chars": routing_meta["text_chars"],
                "min_text_chars": 0,
                "max_text_chars": 8000,
                "min_text_chars_scope": "global",
                "max_text_chars_scope": "global",
                "daily_budget_usd": 10.0,
                "budget_spent_usd": 0.0,
                "budget_remaining_usd": 10.0,
                "similarity_threshold": 0.86,
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_responses_included": False,
                    "request_ids_included": False,
                    "session_ids_included": False,
                    "file_paths_included": False,
                },
            }

        async def fake_shadow_executor(shadow_body, *, shadow_model):
            self.assertEqual(shadow_model, "gpt-5-mini")
            self.assertEqual(shadow_body["model"], "gpt-5-mini")
            self.assertFalse(shadow_body["store"])
            self.assertEqual(shadow_body["input"], "Explain the tradeoff between local proxy routing and exact caching.")
            return {
                "status_code": 200,
                "response_body": {
                    "output_text": "Routing changes model cost; exact caching avoids repeated calls.",
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                },
                "latency_ms": 21,
                "cost_est_usd": 0.00008,
                "error": None,
            }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    request_started = {}
                    pending = {}
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "routing_experiment_decision", side_effect=sampled_decision),
                        patch.object(codex_app_proxy, "_codex_shadow_api_key", return_value="test-key"),
                        patch.object(codex_app_proxy, "_execute_codex_stateless_shadow_request", side_effect=fake_shadow_executor),
                    ):
                        start_event_id = codex_app_proxy._record_message(
                            json.dumps(message),
                            direction="client_to_server",
                            session_id="session-routing-experiment-stateless-text",
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        codex_app_proxy._attach_codex_routing_experiment_pending(
                            json.dumps(message),
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending,
                        )
                        self.assertIn("turn-routing-experiment-stateless-text", pending)
                        await codex_app_proxy._maybe_run_codex_routing_experiment(
                            json.dumps(response),
                            request_started=request_started,
                            pending_routing_experiments=pending,
                        )
                    experiment = dict(test_store.conn.execute("select * from routing_experiments").fetchone())
                    queue_row = dict(test_store.conn.execute(
                        "select source_surface, endpoint, status, payload_json from managed_outcome_feedback_queue"
                    ).fetchone())
                    routing_row = dict(test_store.conn.execute(
                        "select routing_json from codex_app_events where id = ?",
                        (start_event_id,),
                    ).fetchone())
                    return experiment, queue_row, json.loads(routing_row["routing_json"])
                finally:
                    test_store.conn.close()

        experiment, queue_row, routing = asyncio.run(run_fixture())

        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["primary_status_code"], 200)
        self.assertEqual(experiment["shadow_status_code"], 200)
        self.assertIsNone(experiment["error"])
        experiment_json = json.loads(experiment["experiment_json"])
        self.assertEqual(experiment_json["status"], "compared")
        self.assertEqual(experiment_json["shadow_request_preflight"]["status"], "executable")
        self.assertEqual(experiment_json["shadow_request_preflight"]["reason"], "stateless-text-only-replay")
        phase = experiment_json["shadow_request_preflight"]["workflow_phase_classification"]
        self.assertEqual(phase["workflow_phase"], "stateless_text")
        self.assertTrue(phase["shadow_supported"])
        self.assertEqual(experiment_json["request_size_gate"]["status"], "within-bounds")
        self.assertEqual(experiment_json["request_size_gate"]["max_text_chars"], 8000)
        self.assertEqual(queue_row["source_surface"], "routing_experiment_outcome")
        payload = json.loads(queue_row["payload_json"])
        self.assertEqual(payload["source_surface"], "codex_turn")
        self.assertEqual(payload["outcome"]["status"], "compared")
        self.assertTrue(payload["outcome"]["compared"])
        self.assertEqual(routing["routing_experiment"]["managed_feedback"]["status"], "queued")

    def test_codex_routing_experiment_stateful_turn_is_unavailable_not_shadow_error(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-stateful",
            "method": "turn/start",
            "params": {
                "threadId": "thread-routing-experiment-stateful",
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Summarize this current session."}],
                "temperature": 0,
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-stateful",
            "result": {"message": "session summary"},
        }
        optimization_metadata = {
            "routing": {
                "status": "skipped",
                "reason": "keep requested Codex model",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
                "model_field": "model",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        def sampled_decision(body, routing_meta, *, stream, provider, source_surface, store_obj):
            return {
                "schema": "agentflow.routing_experiment_decision.v1",
                "enabled": True,
                "mode": "shadow_candidate_pass_through",
                "status": "selected",
                "sampled": True,
                "reason": "sampled-shadow-candidate-pass-through",
                "counterfactual": True,
                "shadow_only": True,
                "provider": "openai",
                "source_surface": "codex_turn",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-mini",
                "primary_model": "gpt-5-codex",
                "shadow_model": "gpt-5-mini",
                "user_visible_model": "gpt-5-codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "text_chars": 31,
                "daily_budget_usd": 10.0,
                "budget_spent_usd": 0.0,
                "budget_remaining_usd": 10.0,
                "similarity_threshold": 0.86,
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_responses_included": False,
                    "request_ids_included": False,
                    "session_ids_included": False,
                    "file_paths_included": False,
                },
            }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    request_started = {}
                    pending = {}
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "routing_experiment_decision", side_effect=sampled_decision),
                        patch.object(codex_app_proxy, "_codex_shadow_api_key", return_value="test-key"),
                    ):
                        start_event_id = codex_app_proxy._record_message(
                            json.dumps(message),
                            direction="client_to_server",
                            session_id="session-routing-experiment-stateful",
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        codex_app_proxy._attach_codex_routing_experiment_pending(
                            json.dumps(message),
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending,
                        )
                        self.assertIn("turn-routing-experiment-stateful", pending)
                        await codex_app_proxy._maybe_run_codex_routing_experiment(
                            json.dumps(response),
                            request_started=request_started,
                            pending_routing_experiments=pending,
                        )
                    experiment = dict(test_store.conn.execute("select * from routing_experiments").fetchone())
                    queue_row = dict(test_store.conn.execute(
                        "select source_surface, endpoint, status, payload_json from managed_outcome_feedback_queue"
                    ).fetchone())
                    routing_row = dict(test_store.conn.execute(
                        "select routing_json from codex_app_events where id = ?",
                        (start_event_id,),
                    ).fetchone())
                    report = __import__(
                        "agentflow_proxy.routing_experiments",
                        fromlist=["build_routing_experiment_report"],
                    ).build_routing_experiment_report(test_store, limit=5)
                    return experiment, queue_row, json.loads(routing_row["routing_json"]), report
                finally:
                    test_store.conn.close()

        experiment, queue_row, routing, report = asyncio.run(run_fixture())

        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["primary_status_code"], 200)
        self.assertIsNone(experiment["shadow_status_code"])
        self.assertEqual(experiment["error"], "shadow-unavailable:stateful-context-reference")
        self.assertEqual(queue_row["source_surface"], "routing_experiment_outcome")
        payload = json.loads(queue_row["payload_json"])
        self.assertEqual(payload["source_surface"], "codex_turn")
        self.assertEqual(payload["outcome"]["status"], "shadow-unavailable")
        self.assertFalse(payload["outcome"]["compared"])
        self.assertIn("shadow-unavailable-stateful-context-reference", payload["reason_codes"])
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        experiment_json = routing["routing_experiment"]
        self.assertEqual(experiment_json["status"], "shadow-unavailable")
        self.assertTrue(experiment_json["sampled"])
        self.assertTrue(experiment_json["preflight_blocked"])
        self.assertEqual(experiment_json["shadow_request_preflight"]["status"], "unavailable")
        self.assertEqual(experiment_json["shadow_request_preflight"]["reason"], "stateful-context-reference")
        self.assertIn("shadow-unavailable-stateful-context-reference", experiment_json["reason_codes"])
        [candidate] = report["candidates"]
        self.assertEqual(candidate["source_surface"], "codex_turn")
        self.assertEqual(candidate["shadow_unavailable_samples"], 1)
        self.assertEqual(report["summary"]["sample_count"], 1)

    def test_codex_routing_experiment_unsupported_shape_logs_blocked_outcome(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-non-text",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "image", "image_url": "https://example.test/private-secret.png"}],
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-non-text",
            "result": {"message": "cannot summarize image-only input"},
        }
        optimization_metadata = {
            "routing": {
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        def sampled_decision(body, routing_meta, *, stream, provider, source_surface, store_obj):
            return {
                "schema": "agentflow.routing_experiment_decision.v1",
                "enabled": True,
                "mode": "shadow_candidate_pass_through",
                "status": "selected",
                "sampled": True,
                "reason": "sampled-shadow-candidate-pass-through",
                "counterfactual": True,
                "shadow_only": True,
                "provider": "openai",
                "source_surface": "codex_turn",
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-mini",
                "primary_model": "gpt-5-codex",
                "shadow_model": "gpt-5-mini",
                "user_visible_model": "gpt-5-codex",
                "category": "codex-turn",
                "workflow_phase": "summary",
                "text_chars": 0,
                "daily_budget_usd": 10.0,
                "budget_spent_usd": 0.0,
                "budget_remaining_usd": 10.0,
                "similarity_threshold": 0.86,
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_responses_included": False,
                    "request_ids_included": False,
                    "session_ids_included": False,
                    "file_paths_included": False,
                },
            }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    request_started = {}
                    pending = {}
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "routing_experiment_decision", side_effect=sampled_decision),
                        patch.object(codex_app_proxy, "_codex_shadow_api_key", return_value="test-key"),
                    ):
                        start_event_id = codex_app_proxy._record_message(
                            json.dumps(message),
                            direction="client_to_server",
                            session_id="session-routing-experiment-non-text",
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        codex_app_proxy._attach_codex_routing_experiment_pending(
                            json.dumps(message),
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending,
                        )
                        self.assertIn("turn-routing-experiment-non-text", pending)
                        await codex_app_proxy._maybe_run_codex_routing_experiment(
                            json.dumps(response),
                            request_started=request_started,
                            pending_routing_experiments=pending,
                        )
                    experiment = dict(test_store.conn.execute("select * from routing_experiments").fetchone())
                    queue_row = dict(test_store.conn.execute("select payload_json from managed_outcome_feedback_queue").fetchone())
                    routing_row = dict(test_store.conn.execute(
                        "select routing_json from codex_app_events where id = ?",
                        (start_event_id,),
                    ).fetchone())
                    return experiment, json.loads(queue_row["payload_json"]), json.loads(routing_row["routing_json"])
                finally:
                    test_store.conn.close()

        experiment, payload, routing = asyncio.run(run_fixture())

        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["error"], "shadow-unsupported-shape:non-text-input")
        experiment_json = json.loads(experiment["experiment_json"])
        self.assertEqual(experiment_json["status"], "shadow-unsupported-shape")
        self.assertEqual(experiment_json["shadow_request_preflight"]["status"], "unsupported")
        self.assertEqual(experiment_json["shadow_request_preflight"]["reason"], "non-text-input")
        self.assertIn("unsupported-shadow-shape-non-text-input", experiment_json["reason_codes"])
        self.assertEqual(payload["outcome"]["status"], "shadow-unsupported-shape")
        self.assertIn("unsupported-shadow-shape-non-text-input", payload["reason_codes"])
        self.assertNotIn("private-secret", json.dumps(payload))
        self.assertNotIn("private-secret", json.dumps(routing["routing_experiment"]))

    def test_codex_routing_experiment_respects_budget_and_sample_skips(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-skip",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Summarize the completed work."}],
            },
        }
        optimization_metadata = {
            "routing": {
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
            },
        }

        for reason in ("daily-budget-zero", "sample-rate-not-selected"):
            pending = {}

            def skipped_decision(body, routing_meta, *, stream, provider, source_surface, store_obj, reason=reason):
                self.assertEqual(provider, "openai")
                self.assertEqual(source_surface, "codex_turn")
                return {
                    "schema": "agentflow.routing_experiment_decision.v1",
                    "sampled": False,
                    "status": "skipped",
                    "reason": reason,
                }

            with patch.object(codex_app_proxy, "routing_experiment_decision", side_effect=skipped_decision):
                codex_app_proxy._attach_codex_routing_experiment_pending(
                    json.dumps(message),
                    optimization_metadata=optimization_metadata,
                    start_event_id="start-skip",
                    pending_routing_experiments=pending,
                )

            self.assertEqual(pending, {})

    def test_codex_routing_experiment_active_model_pair_mismatch_is_visible(self):
        secret_prompt = "secret active worker prompt must not leak"
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-mismatch",
            "method": "turn/start",
            "params": {
                "threadId": "thread-routing-experiment-mismatch",
                "model": "gpt-unconfigured",
                "input": [{"type": "text", "text": secret_prompt}],
            },
        }
        optimization_metadata = {
            "routing": {
                "status": "skipped",
                "reason": "keep requested Codex model",
                "requested_model": "gpt-unconfigured",
                "routed_model": "gpt-unconfigured",
                "workflow_phase": "summary",
                "model_field": "model",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                request_started = {}
                pending = {}
                with patch.object(codex_app_proxy, "store", test_store):
                    start_event_id = codex_app_proxy._record_message(
                        json.dumps(message),
                        direction="client_to_server",
                        session_id="session-routing-experiment-mismatch",
                        request_started=request_started,
                        optimization_metadata=optimization_metadata,
                    )
                    codex_app_proxy._attach_codex_routing_experiment_pending(
                        json.dumps(message),
                        optimization_metadata=optimization_metadata,
                        start_event_id=start_event_id,
                        pending_routing_experiments=pending,
                    )
                row = test_store.conn.execute(
                    "select routing_json from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
                from agentflow_proxy.routing_experiments import build_routing_experiment_report

                report = build_routing_experiment_report(test_store, limit=5)
            finally:
                test_store.conn.close()

        self.assertEqual(pending, {})
        routing = json.loads(row["routing_json"])
        experiment = routing["routing_experiment"]
        self.assertEqual(experiment["provider"], "openai")
        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["requested_model"], "gpt-unconfigured")
        self.assertEqual(experiment["routed_model"], "gpt-unconfigured")
        self.assertEqual(experiment["status"], "skipped")
        self.assertEqual(experiment["reason"], "model-pair-not-enabled")
        self.assertFalse(experiment["sampled"])
        self.assertTrue(experiment["privacy"]["metadata_only"])
        self.assertEqual(report["summary"]["decision_count"], 1)
        self.assertEqual(report["decision_reasons"][0]["reason"], "model-pair-not-enabled")
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn(secret_prompt, rendered)
        self.assertNotIn("turn-routing-experiment-mismatch", rendered)
        self.assertNotIn("thread-routing-experiment-mismatch", rendered)

    def test_codex_routing_experiment_sample_rate_skip_is_visible(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-rate-skip",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "Summarize the completed work."}],
            },
        }
        optimization_metadata = {
            "routing": {
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                request_started = {}
                pending = {}
                decision = functools.partial(codex_app_proxy.routing_experiment_decision, random_value=lambda: 1.0)
                with (
                    patch.object(codex_app_proxy, "store", test_store),
                    patch.object(codex_app_proxy, "routing_experiment_decision", decision),
                ):
                    start_event_id = codex_app_proxy._record_message(
                        json.dumps(message),
                        direction="client_to_server",
                        session_id="session-routing-experiment-rate-skip",
                        request_started=request_started,
                        optimization_metadata=optimization_metadata,
                    )
                    codex_app_proxy._attach_codex_routing_experiment_pending(
                        json.dumps(message),
                        optimization_metadata=optimization_metadata,
                        start_event_id=start_event_id,
                        pending_routing_experiments=pending,
                    )
                    test_store.update_codex_app_event_routing_json(
                        start_event_id,
                        json.dumps(optimization_metadata["routing"]),
                    )
                row = test_store.conn.execute(
                    "select routing_json from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
            finally:
                test_store.conn.close()

        self.assertEqual(pending, {})
        experiment = json.loads(row["routing_json"])["routing_experiment"]
        self.assertEqual(experiment["status"], "skipped")
        self.assertEqual(experiment["reason"], "sample-rate-not-selected")
        self.assertFalse(experiment["sampled"])

    def test_codex_routing_experiment_live_record_path_uses_active_model_state(self):
        secret_prompt = "secret active turn prompt must stay local"
        initialize = {
            "jsonrpc": "2.0",
            "id": "init-active-model",
            "method": "initialize",
            "params": {
                "model": "gpt-5.5",
                "instructions": "private startup instructions",
            },
        }
        start = {
            "jsonrpc": "2.0",
            "id": "turn-active-model",
            "method": "turn/start",
            "params": {
                "threadId": "thread-active-model",
                "input": [{"type": "text", "text": f"Summarize this self-contained note: {secret_prompt}"}],
            },
        }
        response = {
            "jsonrpc": "2.0",
            "id": "turn-active-model",
            "result": {"message": "completed"},
        }

        async def fake_shadow_executor(shadow_body, *, shadow_model):
            self.assertEqual(shadow_model, "gpt-5.3-codex")
            self.assertEqual(shadow_body["model"], "gpt-5.3-codex")
            self.assertNotIn("threadId", shadow_body)
            self.assertNotIn("thread-active-model", json.dumps(shadow_body))
            return {
                "status_code": 200,
                "response_body": {
                    "output_text": "completed",
                    "usage": {"input_tokens": 9, "output_tokens": 1},
                },
                "latency_ms": 12,
                "cost_est_usd": 0.00001,
                "error": None,
            }

        async def run_fixture():
            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
                test_store = Store(tmp.name)
                try:
                    active_windows = {}
                    model_states = {}
                    request_started = {}
                    pending = {}
                    decision = functools.partial(codex_app_proxy.routing_experiment_decision, random_value=lambda: 0.0)
                    with (
                        patch.object(codex_app_proxy, "store", test_store),
                        patch.object(codex_app_proxy, "routing_experiment_decision", decision),
                        patch.object(codex_app_proxy, "_codex_shadow_api_key", return_value="test-key"),
                        patch.object(codex_app_proxy, "_execute_codex_stateless_shadow_request", side_effect=fake_shadow_executor),
                    ):
                        codex_app_proxy._record_message(
                            json.dumps(initialize),
                            direction="client_to_server",
                            session_id="session-active-model",
                            request_started=request_started,
                            active_turn_windows=active_windows,
                            model_states=model_states,
                        )
                        forwarded, optimization_metadata = codex_app_proxy._optimize_client_message(json.dumps(start))
                        forwarded_obj = json.loads(forwarded)
                        self.assertNotIn("model", forwarded_obj["params"])
                        start_event_id = codex_app_proxy._record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id="session-active-model",
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                            active_turn_windows=active_windows,
                            model_states=model_states,
                        )
                        codex_app_proxy._attach_codex_routing_experiment_pending(
                            forwarded,
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending,
                            session_id="session-active-model",
                            model_states=model_states,
                        )
                        self.assertIn("turn-active-model", pending)
                        await codex_app_proxy._maybe_run_codex_routing_experiment(
                            json.dumps(response),
                            request_started=request_started,
                            pending_routing_experiments=pending,
                        )
                    routing_row = test_store.conn.execute(
                        "select routing_json from codex_app_events where id = ?",
                        (start_event_id,),
                    ).fetchone()
                    experiment_row = test_store.conn.execute(
                        "select * from routing_experiments"
                    ).fetchone()
                    return json.loads(routing_row["routing_json"]), dict(experiment_row)
                finally:
                    test_store.conn.close()

        routing, experiment_row = asyncio.run(run_fixture())

        experiment = routing["routing_experiment"]
        self.assertEqual(experiment["provider"], "openai")
        self.assertEqual(experiment["source_surface"], "codex_turn")
        self.assertEqual(experiment["mode"], "shadow_candidate_pass_through")
        self.assertEqual(experiment["requested_model"], "gpt-5.5")
        self.assertEqual(experiment["routed_model"], "gpt-5.3-codex")
        self.assertEqual(experiment["primary_model"], "gpt-5.5")
        self.assertEqual(experiment["shadow_model"], "gpt-5.3-codex")
        self.assertEqual(experiment["user_visible_model"], "gpt-5.5")
        self.assertEqual(experiment["candidate_id"], "codex-gpt55-to-gpt53-codex-unknown-phase")
        self.assertEqual(experiment["requested_model_source"], "derived-model-state")
        self.assertEqual(experiment["model_state"]["source_method"], "initialize")
        self.assertTrue(experiment["privacy"]["metadata_only"])
        self.assertFalse(experiment["privacy"]["request_ids_included"])
        self.assertNotIn(secret_prompt, json.dumps(experiment, sort_keys=True))

        self.assertEqual(experiment_row["provider"], "openai")
        self.assertEqual(experiment_row["source_surface"], "codex_turn")
        self.assertEqual(experiment_row["requested_model"], "gpt-5.5")
        self.assertEqual(experiment_row["routed_model"], "gpt-5.3-codex")
        self.assertEqual(experiment_row["primary_model"], "gpt-5.5")
        self.assertEqual(experiment_row["shadow_model"], "gpt-5.3-codex")
        self.assertEqual(json.loads(experiment_row["experiment_json"])["mode"], "shadow_candidate_pass_through")

    def test_codex_routing_experiment_request_too_large_skip_is_visible(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-large",
            "method": "turn/start",
            "params": {
                "model": "gpt-5-codex",
                "input": [{"type": "text", "text": "x" * 33000}],
            },
        }
        optimization_metadata = {
            "routing": {
                "requested_model": "gpt-5-codex",
                "routed_model": "gpt-5-codex",
                "workflow_phase": "summary",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                request_started = {}
                pending = {}
                with patch.object(codex_app_proxy, "store", test_store):
                    start_event_id = codex_app_proxy._record_message(
                        json.dumps(message),
                        direction="client_to_server",
                        session_id="session-routing-experiment-large",
                        request_started=request_started,
                        optimization_metadata=optimization_metadata,
                    )
                    codex_app_proxy._attach_codex_routing_experiment_pending(
                        json.dumps(message),
                        optimization_metadata=optimization_metadata,
                        start_event_id=start_event_id,
                        pending_routing_experiments=pending,
                    )
                row = test_store.conn.execute(
                    "select routing_json, input_text_chars from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
            finally:
                test_store.conn.close()

        self.assertEqual(pending, {})
        self.assertEqual(row["input_text_chars"], 33000)
        experiment = json.loads(row["routing_json"])["routing_experiment"]
        self.assertEqual(experiment["status"], "skipped")
        self.assertEqual(experiment["reason"], "request-too-large")
        self.assertFalse(experiment["sampled"])
        self.assertEqual(experiment["text_chars"], 33000)
        self.assertEqual(experiment["request_size_gate"]["status"], "too-large")
        self.assertEqual(experiment["request_size_gate"]["input_text_chars"], 33000)
        self.assertEqual(experiment["request_size_gate"]["max_text_chars"], 8000)
        self.assertEqual(experiment["request_size_gate"]["max_text_chars_scope"], "global")
        self.assertFalse(experiment["request_size_gate"]["privacy"]["raw_prompts_included"])

    def test_codex_routing_experiment_skip_is_visible_without_sample_rows(self):
        message = {
            "jsonrpc": "2.0",
            "id": "turn-routing-experiment-diagnostic",
            "method": "turn/start",
            "params": {
                "threadId": "thread-routing-experiment-diagnostic",
                "input": [{"type": "text", "text": "Summarize the run without exposing raw prompt text."}],
            },
        }
        optimization_metadata = {
            "routing": {
                "status": "skipped",
                "reason": "codex-turn-start-model-field-absent",
                "workflow_phase": "unknown",
            },
            "crunch": {"status": "skipped", "reason": "no-change", "applied": False},
            "cache": {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            test_store = Store(tmp.name)
            try:
                request_started = {}
                pending = {}
                with (
                    patch.object(codex_app_proxy, "store", test_store),
                    patch.object(codex_app_proxy, "codex_app_model", return_value=""),
                ):
                    start_event_id = codex_app_proxy._record_message(
                        json.dumps(message),
                        direction="client_to_server",
                        session_id="session-routing-experiment-diagnostic",
                        request_started=request_started,
                        optimization_metadata=optimization_metadata,
                    )
                    codex_app_proxy._attach_codex_routing_experiment_pending(
                        json.dumps(message),
                        optimization_metadata=optimization_metadata,
                        start_event_id=start_event_id,
                        pending_routing_experiments=pending,
                    )

                self.assertEqual(pending, {})
                self.assertEqual(
                    test_store.conn.execute("select count(*) as c from routing_experiments").fetchone()["c"],
                    0,
                )
                row = test_store.conn.execute(
                    "select routing_json from codex_app_events where id = ?",
                    (start_event_id,),
                ).fetchone()
                routing = json.loads(row["routing_json"])
                experiment = routing["routing_experiment"]
                self.assertEqual(experiment["source_surface"], "codex_turn")
                self.assertEqual(experiment["status"], "skipped")
                self.assertEqual(experiment["reason"], "missing-requested-model")
                self.assertFalse(experiment["sampled"])

                from agentflow_proxy.routing_experiments import build_routing_experiment_report

                report = build_routing_experiment_report(test_store, limit=5)
            finally:
                test_store.conn.close()

        self.assertEqual(report["summary"]["sample_count"], 0)
        self.assertEqual(report["summary"]["decision_count"], 1)
        self.assertEqual(report["summary"]["decision_status_counts"], {"skipped": 1})
        self.assertEqual(report["decision_reasons"][0]["provider"], "openai")
        self.assertEqual(report["decision_reasons"][0]["source_surface"], "codex_turn")
        self.assertEqual(report["decision_reasons"][0]["status"], "skipped")
        self.assertEqual(report["decision_reasons"][0]["reason"], "missing-requested-model")


if __name__ == "__main__":
    unittest.main()
