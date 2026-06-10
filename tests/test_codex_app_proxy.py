import io
import asyncio
import importlib
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


if __name__ == "__main__":
    unittest.main()
